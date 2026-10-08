# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Bounds-guard tests for the spec-decode state updates (packet T4-2 / roadmap GDN-1).

Upstream PR #50021 (ported in `fd6895e789`) masks the accepted-token-derived state
index (`i_t = num_accepted - 1`) to the request's own row and zero-fills that
sequence's output on the invalid path. The pre-port code bounded only the lower end
(`state_idx <= 0`), which does not reject a garbage *positive* count: the derived
index then read before/past the request's row — a fault on the SM, and where the
read lands in another request's row, silent consumption of that request's state.

These tests gate the guards on ROCm. The pre-existing model-path coverage
(`tests/kernels/mamba/test_gdn_fused_mtp.py`) is CUDA-compute-capability-gated, so
the port shipped inspected-not-run; both tests below fail if a guard is removed —
the invalid sequence's output must be exactly zero and its destination state
untouched, whereas the unguarded kernel reads a wild row and writes the result.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

DEVICE = current_platform.device_type


@pytest.mark.skipif(
    current_platform.is_cpu(), reason="the spec-decode conv update is GPU-only"
)
def test_causal_conv1d_update_invalid_accepted_count_zero_fills() -> None:
    """`num_accepted` outside [1, seqlen] must zero-fill *that* sequence only.

    Sequence 1 carries an implausible positive count and sequence 2 a zero count
    (both invalid); sequence 0 stays valid. The valid one is checked against the
    same call made without `num_accepted_tokens` — the port's "bit-identical on
    valid inputs" claim, which was previously ungated.
    """
    set_random_seed(0)
    dim, width, state_len, batch = 128, 4, 8, 3
    num_slots = 4

    x = torch.randn(batch, dim, device=DEVICE)
    weight = torch.randn(dim, width, device=DEVICE)
    bias = torch.randn(dim, device=DEVICE)
    conv_state = torch.randn(num_slots, dim, state_len, device=DEVICE)
    conv_state_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE)
    # 1 accepted token per sequence in this step: valid == 1.
    invalid = torch.tensor([1, 1 << 20, 0], dtype=torch.int32, device=DEVICE)

    def call(states: torch.Tensor, accepted: torch.Tensor | None) -> torch.Tensor:
        return causal_conv1d_update(
            x.clone(),
            states,
            weight,
            bias,
            activation=None,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=accepted,
            query_start_loc=query_start_loc,
            max_query_len=1,
            out=torch.empty_like(x),
        )

    out = call(conv_state.clone(), invalid)

    # The guard's contract: invalid sequences are zero, not a wild-row read.
    assert torch.count_nonzero(out[1]) == 0, (
        "an out-of-range accepted count must zero-fill the sequence; the guard "
        "reads state rows past this request without it"
    )
    assert torch.count_nonzero(out[2]) == 0, (
        "a zero accepted count must zero-fill the sequence (the < 1 half of the guard)"
    )
    assert torch.count_nonzero(out[0]) > 0, (
        "the valid sequence must still be processed — a test that passes because "
        "the kernel no-ops everything would prove nothing"
    )

    # Valid path unchanged: with 1 accepted token the spec-decode path is the
    # plain update.
    accepted_one = torch.ones(batch, dtype=torch.int32, device=DEVICE)
    assert torch.allclose(
        out[0],
        call(conv_state.clone(), accepted_one)[0].to(out.dtype),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skipif(
    current_platform.is_cpu(), reason="selective_state_update is GPU-only"
)
def test_selective_state_update_out_of_range_accepted_count_zero_fills() -> None:
    """The row bound must reject a garbage positive count, and do it *before* the
    state read and the destination write.

    The geometry is chosen so that removing the bound is deterministically bad
    rather than merely undefined: `state_batch_indices` is (batch, max_seq_len) and
    contiguous, so the row stride is `max_seq_len`. Sequence 0's count is one past
    the row (5 with max_seq_len 4), which puts the unbounded read exactly on
    sequence 1's first entry — a real slot id, poisoned here with a huge state. The
    unguarded kernel therefore returns sequence 1's state for sequence 0 and writes
    it into sequence 0's destination slots.
    """
    set_random_seed(0)
    dim, dstate, batch, max_seq_len = 64, 16, 2, 4
    device = DEVICE

    tokens_per_seq = torch.tensor([max_seq_len // 2] * batch, device=device)
    total_tokens = int(tokens_per_seq.sum().item())
    cu_seqlens = torch.tensor(
        [0] + torch.cumsum(tokens_per_seq, dim=0).tolist(),
        dtype=torch.int32,
        device=device,
    )

    poison_slot, valid_slot, dst_base = 7, 1, 11
    state = torch.randn(16, dim, dstate, device=device)
    state[poison_slot] = 1e30  # finite but unmistakable
    state_poison_before = state[poison_slot].clone()

    state_batch_indices = torch.full(
        (batch, max_seq_len), NULL_BLOCK_ID, dtype=torch.int32, device=device
    )
    # Sequence 0: its own (unused for the read once the bound rejects it) slot.
    state_batch_indices[0, 0] = poison_slot
    # Sequence 1: entry 0 is the slot the *unbounded* sequence-0 read would land
    # on; entry 1 is its legitimate initial state.
    state_batch_indices[1, 0] = poison_slot
    state_batch_indices[1, 1] = valid_slot

    dst = torch.arange(
        dst_base, dst_base + total_tokens, dtype=torch.int32, device=device
    ).reshape(batch, max_seq_len // 2)
    dst_state_batch_indices = torch.full(
        (batch, max_seq_len), NULL_BLOCK_ID, dtype=torch.int32, device=device
    )
    dst_state_batch_indices[:, : max_seq_len // 2] = dst
    state[dst.flatten().tolist()] = -3.0  # sentinel: must survive untouched

    # 5 > max_seq_len (invalid; unbounded read would hit row 1 entry 0), 2 (valid).
    num_accepted_tokens = torch.tensor(
        [max_seq_len + 1, 2], dtype=torch.int32, device=device
    )

    x = torch.randn(total_tokens, dim, device=device)
    out = torch.empty_like(x)
    dt = torch.randn(total_tokens, dim, device=device)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(total_tokens, dstate, device=device)
    C = torch.randn(total_tokens, dstate, device=device)
    D = torch.randn(dim, device=device)

    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=None,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
        cu_seqlens=cu_seqlens,
        state_batch_indices=state_batch_indices,
        dst_state_batch_indices=dst_state_batch_indices,
        num_accepted_tokens=num_accepted_tokens,
    )

    invalid_tokens = slice(int(cu_seqlens[0]), int(cu_seqlens[1]))
    assert torch.count_nonzero(out[invalid_tokens]) == 0, (
        "the out-of-range sequence must be zero-filled; the unbounded read would "
        "have returned the poisoned slot's state here"
    )
    assert torch.equal(state[poison_slot], state_poison_before), (
        "the guard must return before the state read: the poisoned slot was touched"
    )
    assert torch.all(state[dst[0].tolist()] == -3.0), (
        "the guard must return before the destination write: sequence 0's dst state "
        "slots were modified"
    )
    assert torch.count_nonzero(out[invalid_tokens.stop :]) > 0, (
        "the valid sequence must still be processed"
    )
