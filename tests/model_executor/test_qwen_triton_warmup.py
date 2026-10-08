# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch

from vllm.model_executor.warmup.qwen_triton_warmup import (
    _FLA_POST_CONV_WARMUP_LENGTHS,
    _QwenGDNWarmupConfig,
    _warm_causal_conv1d_fwd_kernel,
    _warm_fused_post_conv_kernel,
    _warm_gated_rms_norm_kernel,
)
from vllm.platforms import current_platform


def _gdn_config(device="cuda", dtype=torch.bfloat16) -> _QwenGDNWarmupConfig:
    h, hv, k, v = 2, 2, 16, 16
    conv_kernel_size = 4
    conv_dim = 2 * h * k + hv * v
    device = torch.device(device)
    conv_state = torch.empty(
        (8, conv_dim, conv_kernel_size - 1),
        dtype=dtype,
        device=device,
    )
    return _QwenGDNWarmupConfig(
        h=h,
        hv=hv,
        k=k,
        v=v,
        conv_kernel_size=conv_kernel_size,
        conv_state=conv_state,
        conv_dtype=conv_state.dtype,
        norm_weight_dtype=dtype,
        norm_before_gate=True,
        norm_activation="silu",
        a_log=torch.zeros(hv, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(hv, dtype=torch.float32, device=device),
        state_stride_token=hv * v * k,
        state_dtype=torch.float32,
    )


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_qwen_gdn_prefill_warmup_kernels_compile_on_gpu() -> None:
    config = _gdn_config()
    device = torch.device("cuda")
    _warm_gated_rms_norm_kernel(
        device, config, max_num_tokens=16, x_dtype=config.conv_dtype
    )
    _warm_causal_conv1d_fwd_kernel(device, config)
    _warm_fused_post_conv_kernel(device, config)
    assert _FLA_POST_CONV_WARMUP_LENGTHS == (1, 2, 16)
    torch.accelerator.synchronize(device)


@pytest.mark.parametrize(
    "activation_dtype,state_dtype,readout_dtype",
    [
        (torch.float16, torch.float32, torch.float32),
        (torch.bfloat16, torch.float32, torch.bfloat16),
        (torch.float16, torch.float16, torch.float16),
    ],
)
def test_gdn_norm_warmup_compiles_actual_readout_and_gate_dtypes(
    activation_dtype, state_dtype, readout_dtype, monkeypatch
):
    from vllm.third_party.flash_linear_attention.ops import layernorm_guard

    config = replace(_gdn_config("cpu", activation_dtype), state_dtype=state_dtype)
    compiled: list[layernorm_guard.LayerNormFwdKernel.CompileKey] = []
    monkeypatch.setattr(layernorm_guard, "calc_rows_per_block", lambda *args: 1)
    monkeypatch.setattr(
        layernorm_guard._LAYER_NORM_FWD_KERNEL, "compile", compiled.append
    )
    _warm_gated_rms_norm_kernel(
        torch.device("cpu"), config, max_num_tokens=16, x_dtype=activation_dtype
    )
    assert compiled
    assert all(key.x_dtype == key.y_dtype == readout_dtype for key in compiled)
    assert all(key.z_dtype == activation_dtype for key in compiled)
