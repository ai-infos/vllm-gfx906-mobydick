# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side capability checks for gfx906 kernels."""

import os

import torch


def batch_causal(causal: bool | torch.Tensor = True) -> bool:
    """Reject masks the custom FA kernel cannot represent."""
    if not isinstance(causal, bool):
        raise NotImplementedError(
            "GFX906_FA: per-token causality is unsupported; use "
            "--attention-backend TRITON_ATTN for tensor causality."
        )
    return causal


def pad_head_dim(head_size: int) -> int | None:
    """Return the smallest enabled custom FA kernel dimension that fits."""
    if head_size <= 0:
        return None
    dims = (64, 96, 128, 256) if pad96_enabled() else (64, 128, 256)
    return next((dim for dim in dims if head_size <= dim), None)


def pad96_enabled() -> bool:
    return os.environ.get("GFX906_FA_PAD96", "1") == "1"


def padded_head_size(head_size: int) -> int | None:
    padded = pad_head_dim(head_size)
    if os.environ.get("GFX906_FA_PAD", "1") != "1":
        return head_size if padded == head_size else None
    return padded


def fused_align_m1_supported(
    topk_ids: torch.Tensor,
    block_size_m: int,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    use_v2_model_runner: bool | None,
) -> bool:
    """Allow the fused align only on its supported shapes and explicit V1."""
    return (
        os.environ.get("VLLM_GFX906_ALIGN_M1", "1") == "1"
        and use_v2_model_runner is False
        and expert_map is None
        and topk_ids.ndim == 2
        and topk_ids.size(0) == 1
        and block_size_m == 1
        and (global_num_experts, topk_ids.size(1)) in {(256, 8), (128, 6)}
        and topk_ids.dtype == torch.int32
    )
