# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES='' on a CPU host."""

import os

import pytest
import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" or not HAS_TRITON,
    reason="Requires Triton's CPU interpreter, not GPU compilation",
)

from vllm.model_executor.layers.quantization.utils.gfx906_fp8 import (  # noqa: E402
    decode_e4m3_to_fp16,
)


@triton.jit
def _decode_kernel(src, dst, fnuz: tl.constexpr):
    offsets = tl.arange(0, 256)
    values = decode_e4m3_to_fp16(tl.load(src + offsets), fnuz)
    tl.store(dst + offsets, values)


@pytest.mark.parametrize("fnuz", [False, True])
def test_all_e4m3_bytes_match_torch_software_decode(fnuz):
    raw = torch.arange(256, dtype=torch.uint8)
    dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    expected = raw.view(dtype).to(torch.float16)
    actual = torch.empty(256, dtype=torch.float16)
    _decode_kernel[(1,)](raw, actual, fnuz)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = ~torch.isnan(expected)
    # Includes E4M3FN negative zero, which numerical equality alone misses.
    assert torch.equal(
        actual[finite].view(torch.int16), expected[finite].view(torch.int16)
    )
