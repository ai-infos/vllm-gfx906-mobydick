# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_non_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    seq_len = 1  # seq_len is 1 for decode
    num_tokens = num_reqs * seq_len
    total_entries = num_tokens * 2

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    ssm_state = torch.rand(
        total_entries, num_v_heads, head_k_dim, head_v_dim, dtype=dtype
    )
    state_indices = torch.randperm(total_entries, dtype=torch.int32)[:num_tokens]
    cu_seqlens = torch.arange(0, num_tokens + 1, dtype=torch.int32)

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(core_attn_out, core_attn_out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        last_recurrent_state, last_recurrent_state_ref, atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    num_speculative_tokens: int,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    num_tokens = num_reqs * (num_speculative_tokens + 1)
    total_entries = num_tokens * 2

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    ssm_state = torch.rand(
        total_entries, num_v_heads, head_k_dim, head_v_dim, dtype=dtype
    )
    state_indices = torch.randperm(
        total_entries,
        dtype=torch.int32,
    )[:num_tokens].view(num_reqs, num_speculative_tokens + 1)
    num_accepted_tokens = torch.randint(
        1, num_speculative_tokens + 1, (num_reqs,), dtype=torch.int32
    )
    cu_seqlens = torch.arange(
        0, num_tokens + 1, num_speculative_tokens + 1, dtype=torch.int32
    )

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(core_attn_out, core_attn_out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        last_recurrent_state, last_recurrent_state_ref, atol=1e-2, rtol=1e-2
    )


def test_fused_sigmoid_gating_readout_finite_under_large_fp32_state() -> None:
    """Decode readout must stay finite when the SSM state is fp32 and large.

    On non-Blackwell GPUs (e.g. AMD gfx906/MI50) activations are fp16 while the
    SSM state cache is fp32 (``--mamba-ssm-cache-dtype float32``). The GDN state
    is an *unnormalized* accumulator that grows over long, low-decay context and
    legitimately exceeds fp16's 65504 max. If the readout is materialized in the
    fp16 activation dtype it overflows to +/-inf at the store, RMSNormGated turns
    inf/inf into NaN, and greedy decode collapses to a single repeated token.
    Materializing the output in the state's precision keeps it finite.

    Fails on stock (fp16 output -> inf); passes once the output honors fp32 state.
    """
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    tp_size, num_reqs = 1, 4
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = 16, 32, 128, 128
    act_dtype = torch.float16
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    num_tokens = num_reqs
    total_entries = num_tokens * 2

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=act_dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [key_dim // tp_size, key_dim // tp_size, value_dim // tp_size],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=act_dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=act_dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=act_dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=act_dtype)

    ssm_state = (
        torch.rand(
            total_entries, num_v_heads, head_k_dim, head_v_dim, dtype=torch.float32
        )
        * 1e6
    )
    state_indices = torch.randperm(total_entries, dtype=torch.int32)[:num_tokens]
    cu_seqlens = torch.arange(0, num_tokens + 1, dtype=torch.int32)

    core_attn_out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    n_bad = int((~torch.isfinite(core_attn_out)).sum().item())
    assert n_bad == 0, f"{n_bad} non-finite readout entries from fp16 overflow"
