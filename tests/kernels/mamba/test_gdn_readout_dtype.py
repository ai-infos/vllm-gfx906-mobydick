# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host regression for GDN readout storage through normalization/projection.

Only device kernels are replaced. The real GPU forward, kernel wrappers,
mixed-batch merge, native RMSNormGated and torch projections run on CPU.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent,
    fused_sigmoid_gating,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

K, V = 128, 4


@pytest.fixture(autouse=True)
def _cpu_execution(monkeypatch):
    monkeypatch.setattr(gdn, "on_gfx906", lambda: False)
    with torch.device("cpu"):
        yield


class _Projection(torch.nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class _ReferenceDecodeKernel:
    def __getitem__(self, grid):
        return self.run

    @staticmethod
    def run(**kwargs):
        """Orthogonal q/k and v=0 make the update exactly half the state."""
        state, out = kwargs["h0"], kwargs["o"]
        if "q" in kwargs:
            q = kwargs["q"].reshape(-1, 1, K)
        else:
            q = kwargs["mixed_qkv"][:, :K].reshape(-1, 1, K)
        q = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
        out = out.reshape(-1, 1, V)
        out.zero_()
        for row, slot in enumerate(kwargs["ssm_state_indices"].reshape(-1)[: len(q)]):
            if slot > 0:
                updated = state[slot].float() * 0.5
                out[row] = torch.einsum("hvk,hk->hv", updated, q[row]) * K**-0.5
                state[slot] = updated


def _layer(activation_dtype, state_dtype, magnitude):
    layer = gdn.QwenGatedDeltaNetAttention.__new__(gdn.QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "readout_test"
    layer.model_config = SimpleNamespace(dtype=activation_dtype)
    layer.cache_config = SimpleNamespace(
        mamba_cache_dtype="auto", mamba_ssm_cache_dtype=str(state_dtype).split(".")[-1]
    )
    layer.tp_size = layer.num_k_heads = layer.num_v_heads = 1
    layer.head_k_dim, layer.head_v_dim = K, V
    layer.key_dim, layer.value_dim = K, V
    layer.gqa_interleaved_layout = layer.enable_fused_gdn_decode = False
    layer.disable_tp_for_ba_proj = False
    layer.activation = "silu"
    layer.in_proj_qkvz = _Projection(
        1, 2 * K + 2 * V, bias=False, dtype=activation_dtype
    )
    layer.in_proj_ba = _Projection(1, 2, bias=False, dtype=activation_dtype)
    layer.out_proj = _Projection(V, V, bias=False, dtype=activation_dtype)
    with set_current_vllm_config(VllmConfig()):
        layer.norm = RMSNormGated(V, norm_before_gate=True, dtype=activation_dtype)
    layer.norm._forward_method = layer.norm.forward_native
    layer.conv1d = torch.nn.Conv1d(2 * K + V, 2 * K + V, 1, groups=2 * K + V)
    layer.A_log = layer.dt_bias = torch.zeros(1, dtype=activation_dtype)
    state = torch.zeros(3, 1, V, K, dtype=state_dtype)
    state[1, ..., 0] = magnitude
    layer.kv_cache = (torch.zeros(3, 2 * K + V, 1), state)
    with torch.no_grad():
        layer.in_proj_qkvz.weight.zero_()
        layer.in_proj_qkvz.weight[0] = 1
        layer.in_proj_qkvz.weight[K + 1] = 1
        layer.in_proj_qkvz.weight[-V:] = 1
        layer.in_proj_ba.weight.zero_()
        layer.out_proj.weight.copy_(torch.eye(V, dtype=activation_dtype))
    return layer


@pytest.mark.parametrize(
    "path", ["packed", "generic", "spec_prefill", "decode_prefill"]
)
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize(
    "activation_dtype,state_dtype",
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float16),
    ],
)
@torch.inference_mode()
def test_readout_survives_real_gpu_forward_on_cpu(
    path, sign, activation_dtype, state_dtype, monkeypatch
):
    wide = activation_dtype == torch.float16 and state_dtype == torch.float32
    layer = _layer(activation_dtype, state_dtype, sign * (2e6 if wide else 2))
    layer.enable_packed_recurrent_decode = path == "packed"
    prefill, spec = "prefill" in path, path == "spec_prefill"
    meta = GDNAttentionMetadata(
        num_prefills=int(prefill),
        num_prefill_tokens=int(prefill),
        num_decodes=0 if spec else (1 if prefill else 2),
        num_decode_tokens=0 if spec else (1 if prefill else 2),
        num_spec_decodes=int(spec),
        num_spec_decode_tokens=int(spec),
        num_actual_tokens=2,
        has_initial_state=torch.tensor([True]),
        non_spec_query_start_loc=torch.tensor([0, 1] if spec else [0, 1, 2]),
        non_spec_state_indices_tensor=torch.tensor(
            [2] if spec else ([1, 2] if prefill else [1, 0])
        ),
        spec_query_start_loc=torch.tensor([0, 1]),
        spec_state_indices_tensor=torch.tensor([[1]]),
        spec_sequence_masks=torch.tensor([False, True]) if spec else None,
        spec_token_indx=torch.tensor([1]),
        non_spec_token_indx=torch.tensor([0]),
        num_accepted_tokens=torch.tensor([1]),
        prefill_query_start_loc=torch.tensor([0, 1]),
        prefill_state_indices=torch.tensor([2]),
        prefill_has_initial_state=torch.tensor([True]),
    )
    monkeypatch.setattr(
        gdn,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer.prefix: meta}),
    )
    monkeypatch.setattr(gdn, "causal_conv1d_update", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(gdn, "causal_conv1d_fn", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(
        gdn,
        "fused_post_conv_prep",
        lambda conv_output, **kwargs: (
            *[x.squeeze(0) for x in layer.rearrange_mixed_qkv(conv_output)],
            -torch.nn.functional.softplus(kwargs["a"].float()),
            kwargs["b"].float().sigmoid(),
        ),
    )
    layer.chunk_gated_delta_rule = lambda v, **kwargs: (
        torch.ones_like(v),
        kwargs["initial_state"],
    )
    for module, name in (
        (fused_sigmoid_gating, "fused_sigmoid_gating_delta_rule_update_kernel"),
        (fused_recurrent, "fused_recurrent_gated_delta_rule_packed_decode_kernel"),
    ):
        monkeypatch.setattr(module, name, _ReferenceDecodeKernel())
    monkeypatch.setattr(gdn, "GDN_AITER_TRITON_AVAILABLE", False)
    if wide:
        # The affected combination must bypass even an advertised AITER backend.
        monkeypatch.setattr(gdn, "GDN_AITER_TRITON_AVAILABLE", True)

    core_outputs = []

    def core(qkv, b, a, out, **kwargs):
        layer._forward_core(qkv, b, a, out)
        core_outputs.append(out.clone())

    monkeypatch.setattr(torch.ops.vllm, "qwen_gdn_attention_core", core)
    result = layer.forward_hip(torch.ones(3, 1, dtype=activation_dtype))
    readout = core_outputs[0]
    assert torch.isfinite(readout).all() and torch.isfinite(result).all()
    assert readout.dtype == (torch.float32 if wide else activation_dtype)
    assert result.dtype == activation_dtype
    decode_row = 1 if spec else 0
    if wide:
        assert readout[decode_row].abs().min() > torch.finfo(torch.float16).max
        assert not torch.isfinite(readout[decode_row].to(torch.float16)).all()
    expected_readout = sign * (1e6 if wide else 1) * K**-0.5 / (1 + 1e-6) ** 0.5
    torch.testing.assert_close(
        readout[decode_row].float(),
        torch.full((1, V), expected_readout),
        rtol=5e-3,
        atol=1e-3,
    )
    expected = torch.zeros_like(result)
    value = expected_readout / (expected_readout**2 + layer.norm.eps) ** 0.5
    expected[decode_row] = value * torch.nn.functional.silu(torch.tensor(1.0))
    if prefill:
        expected[1 - decode_row] = (
            torch.nn.functional.silu(torch.tensor(1.0)) / (1 + layer.norm.eps) ** 0.5
        )
    torch.testing.assert_close(result, expected, rtol=5e-3, atol=1e-3)
    assert torch.count_nonzero(readout[2:]) == 0
    assert torch.count_nonzero(layer.kv_cache[1][0]) == 0
