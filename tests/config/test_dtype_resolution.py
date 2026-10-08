# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Unit tests: auto-dtype resolution on platforms without native bf16.

gfx906 (CDNA1) has no native bfloat16 and the gfx906 kernel stack is
fp16-only, so "auto" must fall back to float16 for bf16 checkpoints
there (see docs/gfx906/DEVLOG-qwen38.md).
"""

from unittest import mock

import torch

from vllm.config.model import _resolve_auto_dtype


def _resolve(native_bf16: bool, model_type: str, config_dtype: torch.dtype):
    with mock.patch("vllm.config.model.current_platform") as plat:
        plat.supported_dtypes = [
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ]
        plat.supports_native_bf16 = native_bf16
        return _resolve_auto_dtype(
            model_type, config_dtype, is_pooling_model=False
        )


def test_auto_bf16_with_native_bf16_stays_bf16():
    assert _resolve(True, "qwen3_5_text", torch.bfloat16) == torch.bfloat16


def test_auto_bf16_without_native_bf16_falls_back_to_fp16():
    assert _resolve(False, "qwen3_5_text", torch.bfloat16) == torch.float16


def test_auto_bf16_fp16_forbidden_model_keeps_bf16():
    # gemma3 forbids fp16 (numerical instability) -> the fallback must
    # not fire there; bf16 is the only sane choice.
    assert _resolve(False, "gemma3", torch.bfloat16) == torch.bfloat16


def test_auto_fp16_unaffected():
    assert _resolve(False, "qwen3_5_text", torch.float16) == torch.float16
