# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401


def test_gptq_shuffle_opcheck():
    weight = torch.randint(
        -2000000, 2000000, (1792, 4096), device="cuda", dtype=torch.int32
    )
    bit = 4
    opcheck(torch.ops._C.gptq_shuffle, (weight, bit))


def test_gptq_gemm_opcheck():
    a = torch.rand((240, 4096), device="cuda", dtype=torch.float16)
    weight = torch.randint(
        -2000000, 2000000, (512, 6144), device="cuda", dtype=torch.int32
    )
    zeros = torch.zeros((32, 768), device="cuda", dtype=torch.int32)
    scales = torch.rand((32, 6144), device="cuda", dtype=torch.float16)
    use_exllama = True
    bit = 4
    # Test both GPTQv1 and GPTQv2 format
    opcheck(torch.ops._C.gptq_gemm, (a, weight, zeros, scales, use_exllama, True, bit))
    opcheck(torch.ops._C.gptq_gemm, (a, weight, zeros, scales, use_exllama, False, bit))


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("maxilp", ["0", "1"])
def test_gfx906_split_k_output_is_initialized_before_atomic_updates(
    rows, maxilp, monkeypatch
):
    from vllm.platforms.rocm import on_gfx906

    if not on_gfx906():
        pytest.skip("Requires gfx906 GPTQ kernels")
    monkeypatch.setenv("VLLM_GFX906_QGEMM_M1_MAXILP", maxilp)
    k, n = 8192, 1024
    a = torch.ones(rows, k, dtype=torch.float16, device="cuda")
    # Every nibble is 8; shuffle still uses the actual operator ABI.
    weight = torch.full((k // 8, n), -2004318072, dtype=torch.int32, device="cuda")
    ops.gptq_shuffle(weight, 4)
    zeros = torch.zeros(k // 128, n // 8, dtype=torch.int32, device="cuda")
    scales = torch.full((k // 128, n), 1 / 128, dtype=torch.float16, device="cuda")
    expected = torch.full((rows, n), k / 16, dtype=torch.float16, device="cuda")
    for _ in range(20):
        result = ops.gptq_gemm(a, weight, zeros, scales, True, True, 4)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
