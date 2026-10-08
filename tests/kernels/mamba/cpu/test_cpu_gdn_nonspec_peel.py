# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""CPU GDN spec-mixed non-spec peel: [decode, prefill, spec] contract test.

W1 changed the GDN metadata contract: in spec-mixed batches the
``prefill_*`` fields and the chunk indices now cover real prefills only,
and 1-token non-spec decodes are peeled to the per-seq recurrent kernel
(decode-first front slice). The CPU backend must consume that contract;
before the fix, ``_spec_aware_nonspec_subset`` fed the full non-spec token
range (decodes + prefills) to the chunked path with prefill-only
cu_seqlens, so any spec batch containing a non-spec decode crashed at the
scatter step (index size mismatch) or mis-routed the decode rows.

The leaf ops are replaced with torch references (the C++ CPU ops only
exist in CPU builds), so this test exercises the dispatch/contract logic
on any platform. Kernel-vs-reference numeric accuracy is covered by
``test_cpu_gdn_ops.py`` on the CPU platform.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.cpu import gdn_attention
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

set_random_seed(20260826)

NUM_QK_HEADS = 2
NUM_V_HEADS = 2
HEAD_DIM = 8
V_HEAD_DIM = 8
QKV_DIM = (NUM_QK_HEADS + 2 * NUM_V_HEADS) * HEAD_DIM  # 48
CONV_WIDTH = 4
STATE_LEN = 6  # width - 1 + num_spec
NUM_SLOTS = 8
DECODE_SLOT = 3
PREFILL_SLOT = 7


def ref_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-5):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def ref_gating(A_log: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
               dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-decay ``g`` and ``beta`` from raw gates. Shapes follow ``a``/``b``."""
    softplus_x = F.softplus(a.float() + dt_bias.float(), beta=1.0, threshold=20.0)
    g = -torch.exp(A_log.float()) * softplus_x
    beta = torch.sigmoid(b.float())
    return g, beta


def ref_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token gated delta rule.

    q (1, t, h, d), k (1, t, h, d), v (1, t, h, dv), g (t, h) log-decay,
    beta (t, h), state (h, d, dv).
    Returns (out (t, h, dv), new_state (h, d, dv)).
    """
    q = q[0].float()
    k = k[0].float()
    v = v[0].float()
    if use_qk_l2norm:
        q = ref_l2norm(q)
        k = ref_l2norm(k)
    q = q * (1 / (HEAD_DIM**0.5))
    S = state.float().transpose(-1, -2).clone()  # (h, dv, d)
    outs = []
    for t in range(v.shape[0]):
        g_t = g[t].exp().view(-1, 1, 1)
        beta_t = beta[t].view(-1, 1)
        S = S * g_t
        kv_mem = (S * k[t].unsqueeze(-2)).sum(-1)  # (h, dv)
        delta = (v[t] - kv_mem) * beta_t  # (h, dv)
        S = S + delta.unsqueeze(-1) * k[t].unsqueeze(-2)
        outs.append((S * q[t].unsqueeze(-2)).sum(-1))  # (h, dv)
    return torch.stack(outs), S.transpose(-1, -2)


def _split_qkv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(t, QKV_DIM) -> q (1, t, h, d), k (1, t, h, d), v (1, t, h, dv)."""
    third = QKV_DIM // 3
    return (
        x[:, :third].view(-1, NUM_QK_HEADS, HEAD_DIM).unsqueeze(0),
        x[:, third : 2 * third].view(-1, NUM_V_HEADS, HEAD_DIM).unsqueeze(0),
        x[:, 2 * third :].view(-1, NUM_V_HEADS, V_HEAD_DIM).unsqueeze(0),
    )


def _ref_conv1d(
    x: torch.Tensor,
    window: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Depthwise causal conv1d of ``x (t, dim)`` seeded by ``window (dim, w-1)``."""
    z = torch.cat([window, x.float().transpose(0, 1)], dim=-1)
    y = F.conv1d(
        z.unsqueeze(0), weight.float().unsqueeze(1), bias.float(), groups=QKV_DIM
    )[0]
    return F.silu(y).transpose(0, 1).to(x.dtype)  # (t, dim)


def _make_fake_layer() -> types.SimpleNamespace:
    conv_weight = torch.randn(QKV_DIM, CONV_WIDTH, dtype=torch.bfloat16)
    conv_bias = torch.randn(QKV_DIM, dtype=torch.bfloat16)
    return types.SimpleNamespace(
        activation="silu",
        conv1d=types.SimpleNamespace(
            weight=conv_weight,
            bias=conv_bias,
            _cpu_unpacked_conv_weight=conv_weight,
        ),
        A_log=torch.randn(NUM_V_HEADS, dtype=torch.float32),
        dt_bias=torch.randn(NUM_V_HEADS, dtype=torch.bfloat16),
        rearrange_mixed_qkv=_split_qkv,
    )


def _patch_leaves(
    monkeypatch: pytest.MonkeyPatch, layer: types.SimpleNamespace
) -> None:
    """Replace the C++ CPU leaf ops with torch references (in-place state
    semantics match the production kernels)."""
    monkeypatch.setattr(torch.cpu, "_is_amx_tile_supported", lambda: False)
    # conv cache is DS layout (num_slots, dim, state_len)
    monkeypatch.setattr(
        gdn_attention, "is_conv_state_dim_first", lambda: True
    )
    w = layer.conv1d.weight

    def causal_conv1d_update_cpu(
        x, conv_state, weight, bias, activation, conv_state_indices, **kwargs
    ):
        # 1-token decode step; conv_state is the (slots, dim, width-1) view.
        sl = CONV_WIDTH - 1
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            slot = int(conv_state_indices[i].item())
            out[i] = _ref_conv1d(
                x[i : i + 1], conv_state[slot, :, :sl].float(), w, bias
            )[0]
            conv_state[slot, :, :sl] = torch.cat(
                [conv_state[slot, :, 1:sl].float(), x[i, :].float().unsqueeze(1)],
                dim=-1,
            ).to(conv_state.dtype)
        return out

    def causal_conv1d_torch(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        **kwargs,
    ):
        # x is transposed (dim, t); only the first width-1 columns of the
        # wide buffer are read/written (production torch-ref semantics).
        sl = CONV_WIDTH - 1
        xt = x.transpose(0, 1)
        out = torch.empty_like(xt)
        start = 0
        for i in range(query_start_loc.shape[0] - 1):
            end = int(query_start_loc[i + 1].item())
            if end == start:
                continue
            slot = int(cache_indices[i].item())
            if bool(has_initial_state[i].item()):
                window = conv_states[slot, :, :sl].float()
            else:
                window = torch.zeros(QKV_DIM, sl, dtype=torch.float32)
            out[start:end] = _ref_conv1d(xt[start:end], window, w, bias)
            conv_states[slot, :, :sl] = xt[end - sl : end, :].float().transpose(
                0, 1
            ).to(conv_states.dtype)
            start = end
        return out.transpose(0, 1)

    def fused_gdn_gating_cpu(A_log, a, b, dt_bias, **kwargs):
        g, beta = ref_gating(A_log, a, b, dt_bias)
        return g.unsqueeze(0), beta.unsqueeze(0)

    def fused_sigmoid_gating_delta_rule_update_cpu(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        initial_state_source,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        **kwargs,
    ):
        # 1-token sequences; state mutated in place at the given indices.
        n = initial_state_indices.shape[0]
        out = torch.empty(1, n, NUM_V_HEADS, V_HEAD_DIM, dtype=v.dtype)
        for i in range(n):
            assert int(cu_seqlens[i + 1].item()) - int(cu_seqlens[i].item()) == 1
            slot = int(initial_state_indices[i].item())
            g, beta = ref_gating(A_log, a[i, :], b[i, :], dt_bias)
            out_i, new_state = ref_delta_rule(
                q[:, i : i + 1],
                k[:, i : i + 1],
                v[:, i : i + 1],
                g.unsqueeze(0),
                beta.unsqueeze(0),
                initial_state_source[slot],
                use_qk_l2norm_in_kernel,
            )
            out[:, i] = out_i[0].to(v.dtype)
            initial_state_source[slot] = new_state.to(initial_state_source.dtype)
        return out.transpose(0, 1)  # (n, 1, h, dv)

    def chunk_gated_delta_rule_cpu(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first,
        use_qk_l2norm_in_kernel,
        initial_state_indices,
        **kwargs,
    ):
        n = initial_state_indices.shape[0]
        out = torch.empty(
            1, query.shape[1], NUM_V_HEADS, V_HEAD_DIM, dtype=query.dtype
        )
        start = 0
        for i in range(n):
            end = int(cu_seqlens[i + 1].item())
            slot = int(initial_state_indices[i].item())
            out_i, new_state = ref_delta_rule(
                query[:, start:end],
                key[:, start:end],
                value[:, start:end],
                g[0, start:end],
                beta[0, start:end],
                initial_state[slot],
                use_qk_l2norm_in_kernel,
            )
            out[:, start:end] = out_i.unsqueeze(0).to(query.dtype)
            initial_state[slot] = new_state.to(initial_state.dtype)
            start = end
        return out, initial_state[initial_state_indices]

    monkeypatch.setattr(
        gdn_attention, "causal_conv1d_update_cpu", causal_conv1d_update_cpu
    )
    monkeypatch.setattr(gdn_attention, "causal_conv1d_torch", causal_conv1d_torch)
    monkeypatch.setattr(gdn_attention.ops, "fused_gdn_gating_cpu", fused_gdn_gating_cpu)
    monkeypatch.setattr(
        gdn_attention.ops,
        "fused_sigmoid_gating_delta_rule_update_cpu",
        fused_sigmoid_gating_delta_rule_update_cpu,
    )
    monkeypatch.setattr(
        gdn_attention.ops, "chunk_gated_delta_rule_cpu", chunk_gated_delta_rule_cpu
    )
    # The spec leg is out of scope (its own kernel is covered elsewhere);
    # stub a zero output of the right shape to exercise the scatter.
    monkeypatch.setattr(
        gdn_attention,
        "_spec_forward",
        lambda layer, meta, qkv, bb, aa, conv_buf, ssm_state, width, state_len: (
            torch.zeros(
                meta.num_spec_decode_tokens,
                NUM_V_HEADS,
                V_HEAD_DIM,
                dtype=qkv.dtype,
            )
        ),
    )


def _ref_decode(layer, token: torch.Tensor, a_row: torch.Tensor, b_row: torch.Tensor,
                conv_before: torch.Tensor, ssm_before: torch.Tensor):
    """Reference for the 1-token decode of slot ``DECODE_SLOT``."""
    sl = CONV_WIDTH - 1
    window = conv_before[DECODE_SLOT, :, :sl].float()
    qkv = _ref_conv1d(token.unsqueeze(0), window, layer.conv1d.weight,
                      layer.conv1d.bias)
    q, k, v = _split_qkv(qkv)
    g, beta = ref_gating(layer.A_log, a_row, b_row, layer.dt_bias)
    out, new_state = ref_delta_rule(q, k, v, g.unsqueeze(0), beta.unsqueeze(0),
                                    ssm_before[DECODE_SLOT])
    exp_conv = torch.cat(
        [window[:, 1:], token.float().unsqueeze(1)], dim=-1
    ).to(conv_before.dtype)
    return out.squeeze(0), new_state, exp_conv


def _ref_prefill(layer, tokens: torch.Tensor, a_rows: torch.Tensor,
                 b_rows: torch.Tensor, window: torch.Tensor,
                 ssm_init: torch.Tensor):
    """Reference for the prefill of slot ``PREFILL_SLOT``.

    ``window`` is the initial conv window (dim, width-1) and ``ssm_init``
    the initial SSM state as the dispatch sees them (zeroed for fresh
    sequences via ``has_initial_state=False``).
    """
    sl = CONV_WIDTH - 1
    qkv = _ref_conv1d(tokens, window, layer.conv1d.weight, layer.conv1d.bias)
    q, k, v = _split_qkv(qkv)
    g, beta = ref_gating(layer.A_log, a_rows, b_rows, layer.dt_bias)
    out, new_state = ref_delta_rule(q, k, v, g, beta, ssm_init)
    exp_conv = tokens[-sl:, :].float().transpose(0, 1).to(tokens.dtype)
    return out, new_state, exp_conv


@torch.inference_mode()
def test_spec_mixed_decode_prefill_spec_contract(monkeypatch: pytest.MonkeyPatch):
    """[decode, prefill, spec] batch: decodes peel to the recurrent kernel,
    the prefill tail goes to the chunked path with prefill-only metadata,
    and all outputs scatter to the right token rows."""
    n_spec_tok, n_pre = 3, 5
    n_tokens = 1 + n_spec_tok + n_pre  # batch order: [decode, spec, prefill]

    mixed_qkv = torch.randn(n_tokens, QKV_DIM, dtype=torch.bfloat16)
    b = torch.randn(n_tokens, NUM_V_HEADS, dtype=torch.bfloat16)
    a = torch.randn(n_tokens, NUM_V_HEADS, dtype=torch.bfloat16)
    conv_buf = torch.randn(NUM_SLOTS, QKV_DIM, STATE_LEN, dtype=torch.bfloat16)
    ssm_state = torch.randn(
        NUM_SLOTS, NUM_V_HEADS, HEAD_DIM, V_HEAD_DIM, dtype=torch.float32
    )

    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=n_pre,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=1,
        num_spec_decode_tokens=n_spec_tok,
        num_actual_tokens=n_tokens,
        has_initial_state=torch.tensor([True, True]),
        prefill_query_start_loc=torch.tensor([0, n_pre], dtype=torch.int32),
        prefill_state_indices=torch.tensor([PREFILL_SLOT], dtype=torch.int32),
        prefill_has_initial_state=torch.tensor([True]),
        spec_query_start_loc=torch.tensor([0, n_spec_tok], dtype=torch.int32),
        non_spec_query_start_loc=torch.tensor(
            [0, 1, 1 + n_pre], dtype=torch.int32
        ),
        spec_state_indices_tensor=torch.full(
            (1, 4), 5, dtype=torch.int32
        ),
        non_spec_state_indices_tensor=torch.tensor(
            [DECODE_SLOT, PREFILL_SLOT], dtype=torch.int32
        ),
        spec_sequence_masks=torch.tensor([False, True, False]),
        spec_token_indx=torch.tensor([1, 2, 3]),
        non_spec_token_indx=torch.tensor([0, 4, 5, 6, 7, 8]),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
    )

    layer = _make_fake_layer()
    layer.kv_cache = [conv_buf, ssm_state]
    _patch_leaves(monkeypatch, layer)
    conv_before = conv_buf.clone()
    ssm_before = ssm_state.clone()

    core_attn_out = torch.empty(n_tokens, NUM_V_HEADS, V_HEAD_DIM, dtype=torch.bfloat16)
    gdn_attention._cpu_gdn_attention_spec_aware(
        layer=layer,
        attn_metadata_i=metadata,
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
        width=CONV_WIDTH,
        state_len=STATE_LEN,
    )

    exp_dec, exp_dec_state, exp_dec_conv = _ref_decode(
        layer, mixed_qkv[0], a[0], b[0], conv_before, ssm_before
    )
    exp_pre, exp_pre_state, exp_pre_conv = _ref_prefill(
        layer,
        mixed_qkv[4:9],
        a[4:9],
        b[4:9],
        conv_before[PREFILL_SLOT, :, : CONV_WIDTH - 1].float(),
        ssm_before[PREFILL_SLOT],
    )

    # Outputs land on the right rows: decode first, then spec (stub zeros),
    # then prefill.
    torch.testing.assert_close(
        core_attn_out[0], exp_dec, atol=2e-2, rtol=2e-2, check_dtype=False
    )
    torch.testing.assert_close(
        core_attn_out[1 : 1 + n_spec_tok],
        torch.zeros(n_spec_tok, NUM_V_HEADS, V_HEAD_DIM, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        core_attn_out[4:9], exp_pre, atol=2e-2, rtol=2e-2, check_dtype=False
    )

    # SSM states advance only at the decode and prefill slots.
    torch.testing.assert_close(ssm_state[DECODE_SLOT], exp_dec_state,
                               atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(ssm_state[PREFILL_SLOT], exp_pre_state,
                               atol=2e-2, rtol=2e-2)
    for slot in range(NUM_SLOTS):
        if slot not in (DECODE_SLOT, PREFILL_SLOT):
            torch.testing.assert_close(ssm_state[slot], ssm_before[slot])

    # Conv state: decode window shifted by one token, prefill window replaced
    # by its last width-1 inputs; the spec rolling region stays untouched.
    torch.testing.assert_close(conv_buf[DECODE_SLOT, :, : CONV_WIDTH - 1],
                               exp_dec_conv)
    torch.testing.assert_close(conv_buf[PREFILL_SLOT, :, : CONV_WIDTH - 1],
                               exp_pre_conv)
    torch.testing.assert_close(conv_buf[:, :, CONV_WIDTH - 1 :],
                               conv_before[:, :, CONV_WIDTH - 1 :])


@torch.inference_mode()
def test_spec_mixed_decode_only_contract(monkeypatch: pytest.MonkeyPatch):
    """[decode, spec] batch: the chunk path must not be reached at all
    (prefill metadata is None in this shape)."""
    n_spec_tok = 3
    n_tokens = 1 + n_spec_tok

    mixed_qkv = torch.randn(n_tokens, QKV_DIM, dtype=torch.bfloat16)
    b = torch.randn(n_tokens, NUM_V_HEADS, dtype=torch.bfloat16)
    a = torch.randn(n_tokens, NUM_V_HEADS, dtype=torch.bfloat16)
    conv_buf = torch.randn(NUM_SLOTS, QKV_DIM, STATE_LEN, dtype=torch.bfloat16)
    ssm_state = torch.randn(
        NUM_SLOTS, NUM_V_HEADS, HEAD_DIM, V_HEAD_DIM, dtype=torch.float32
    )

    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=1,
        num_spec_decode_tokens=n_spec_tok,
        num_actual_tokens=n_tokens,
        has_initial_state=None,
        spec_query_start_loc=torch.tensor([0, n_spec_tok], dtype=torch.int32),
        non_spec_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        spec_state_indices_tensor=torch.full((1, 4), 5, dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor(
            [DECODE_SLOT], dtype=torch.int32
        ),
        spec_sequence_masks=torch.tensor([False, True]),
        spec_token_indx=torch.tensor([1, 2, 3]),
        non_spec_token_indx=torch.tensor([0]),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
    )

    layer = _make_fake_layer()
    layer.kv_cache = [conv_buf, ssm_state]
    _patch_leaves(monkeypatch, layer)

    def fail_chunk(**kwargs):
        raise AssertionError("chunk path must not run for decode-only non-spec")

    monkeypatch.setattr(gdn_attention.ops, "chunk_gated_delta_rule_cpu", fail_chunk)

    conv_before = conv_buf.clone()
    ssm_before = ssm_state.clone()
    core_attn_out = torch.empty(n_tokens, NUM_V_HEADS, V_HEAD_DIM, dtype=torch.bfloat16)
    gdn_attention._cpu_gdn_attention_spec_aware(
        layer=layer,
        attn_metadata_i=metadata,
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
        width=CONV_WIDTH,
        state_len=STATE_LEN,
    )

    exp_dec, exp_dec_state, exp_dec_conv = _ref_decode(
        layer, mixed_qkv[0], a[0], b[0], conv_before, ssm_before
    )
    torch.testing.assert_close(
        core_attn_out[0], exp_dec, atol=2e-2, rtol=2e-2, check_dtype=False
    )
    torch.testing.assert_close(
        core_attn_out[1:],
        torch.zeros(n_spec_tok, NUM_V_HEADS, V_HEAD_DIM, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(ssm_state[DECODE_SLOT], exp_dec_state,
                               atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(conv_buf[DECODE_SLOT, :, : CONV_WIDTH - 1],
                               exp_dec_conv)


@torch.inference_mode()
def test_spec_mixed_prefill_only_contract(monkeypatch: pytest.MonkeyPatch):
    """[prefill, spec] batch (num_decodes=0): the pre-existing working shape
    must keep its numerics after the peel refactor."""
    n_spec_tok, n_pre = 3, 5
    n_tokens = n_spec_tok + n_pre  # batch order: [spec, prefill]

    mixed_qkv = torch.randn(n_tokens, QKV_DIM, dtype=torch.bfloat16)
    b = torch.randn(n_tokens, NUM_V_HEADS, dtype=torch.bfloat16)
    a = torch.randn(n_tokens, NUM_V_HEADS, dtype=torch.bfloat16)
    conv_buf = torch.randn(NUM_SLOTS, QKV_DIM, STATE_LEN, dtype=torch.bfloat16)
    ssm_state = torch.randn(
        NUM_SLOTS, NUM_V_HEADS, HEAD_DIM, V_HEAD_DIM, dtype=torch.float32
    )

    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=n_pre,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=n_spec_tok,
        num_actual_tokens=n_tokens,
        has_initial_state=torch.tensor([False]),
        prefill_query_start_loc=torch.tensor([0, n_pre], dtype=torch.int32),
        prefill_state_indices=torch.tensor([PREFILL_SLOT], dtype=torch.int32),
        prefill_has_initial_state=torch.tensor([False]),
        spec_query_start_loc=torch.tensor([0, n_spec_tok], dtype=torch.int32),
        non_spec_query_start_loc=torch.tensor([0, n_pre], dtype=torch.int32),
        spec_state_indices_tensor=torch.full((1, 4), 5, dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor(
            [PREFILL_SLOT], dtype=torch.int32
        ),
        spec_sequence_masks=torch.tensor([True, False]),
        spec_token_indx=torch.tensor([0, 1, 2]),
        non_spec_token_indx=torch.tensor([3, 4, 5, 6, 7]),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
    )

    layer = _make_fake_layer()
    layer.kv_cache = [conv_buf, ssm_state]
    _patch_leaves(monkeypatch, layer)
    ssm_before = ssm_state.clone()

    core_attn_out = torch.empty(n_tokens, NUM_V_HEADS, V_HEAD_DIM, dtype=torch.bfloat16)
    gdn_attention._cpu_gdn_attention_spec_aware(
        layer=layer,
        attn_metadata_i=metadata,
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
        width=CONV_WIDTH,
        state_len=STATE_LEN,
    )

    # has_initial_state=False: the dispatch zeroes the pool slot and the
    # conv window before the chunk call, so the reference starts from zero.
    exp_pre, exp_pre_state, exp_pre_conv = _ref_prefill(
        layer,
        mixed_qkv[3:8],
        a[3:8],
        b[3:8],
        torch.zeros(QKV_DIM, CONV_WIDTH - 1, dtype=torch.float32),
        torch.zeros_like(ssm_before[PREFILL_SLOT]),
    )
    torch.testing.assert_close(
        core_attn_out[0:3],
        torch.zeros(3, NUM_V_HEADS, V_HEAD_DIM, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        core_attn_out[3:8], exp_pre, atol=2e-2, rtol=2e-2, check_dtype=False
    )
    torch.testing.assert_close(ssm_state[PREFILL_SLOT], exp_pre_state,
                               atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(conv_buf[PREFILL_SLOT, :, : CONV_WIDTH - 1],
                               exp_pre_conv)
