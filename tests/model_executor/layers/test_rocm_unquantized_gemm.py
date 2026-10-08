# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_cuda():
    pytest.skip(
        "ROCm skinny GEMM tests are not supported on CUDA.",
        allow_module_level=True,
    )

from vllm.model_executor.layers import utils
from vllm.platforms.rocm import on_gfx906


def test_rocm_unquantized_gemm_gfx1x_wvsplitk_path(monkeypatch):
    x = torch.randn(1, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_makes_skinny_activation_contiguous(monkeypatch):
    x = torch.randn(64, 4, dtype=torch.float16).t()
    weight = torch.randn(128, 64, dtype=torch.float16)
    assert x.shape == (4, 64)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    x_view = wvsplitk_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_makes_llmm1_activation_contiguous(monkeypatch):
    x = torch.randn(1, 128, dtype=torch.float16)[:, ::2]
    weight = torch.randn(4, 64, dtype=torch.float16)
    assert x.shape == (1, 64)
    assert x.stride() == (128, 2)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    llmm1_mock.assert_called_once()
    x_view = llmm1_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("noncontiguous_operand", ["weight", "bias"])
def test_rocm_unquantized_gemm_rejects_unsupported_skinny_layouts(
    monkeypatch, noncontiguous_operand
):
    # Non-contiguous weight/bias must not enter the skinny use_skinny block
    # and fall back to torch. All arch predicates are off: this fork routes
    # n in (0, 4] with m > 8 to wvSplitK on gfx9/gfx1x unconditionally
    # (outside use_skinny), so a simulated skinny arch would take that
    # branch instead of the fallback under test.
    x = torch.randn(4, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)
    bias = torch.randn(128, dtype=torch.float16)
    if noncontiguous_operand == "weight":
        weight = torch.randn(64, 128, dtype=torch.float16).t()
        assert not weight.is_contiguous()
    else:
        bias = torch.randn(256, dtype=torch.float16)[::2]
        assert not bias.is_contiguous()

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, bias)
    ref = torch.nn.functional.linear(x, weight, bias)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.skipif(
    on_gfx906(), reason="wvSplitK targets matrix cores, excluded on gfx906"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_unquantized_gemm_noncontiguous_activation_real_kernel(monkeypatch, dtype):
    x = torch.randn(64, 4, device="cuda", dtype=dtype).t()
    weight = torch.randn(128, 64, device="cuda", dtype=dtype)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    original_wvsplitk = utils.ops.wvSplitK
    wvsplitk_mock = MagicMock(side_effect=original_wvsplitk)
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_rocm_unquantized_gemm_gfx1x_n_gt_5_falls_back(monkeypatch):
    # wvSplitK skinny GEMM handles n in [1, 5] (see PR #40687); n > 5 must
    # never reach it. n=17 also clears this fork's n<=16 triton_matmul
    # tail, landing on the torch fallback the assertion checks.
    x = torch.randn(17, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_rocm_unquantized_gemm_tiny_m_llmm1_padded(monkeypatch, m):
    # m < 4 (e.g. Qwen3-Next shared_expert_gate [1, K]): LLMM1 requires
    # M % 4 == 0, so the dispatch zero-pads the weight and slices the out.
    x = torch.randn(1, 2048, dtype=torch.float16)
    weight = torch.randn(m, 2048, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    # On real gfx906 hardware m==1 takes the custom GEMV (real op, needs
    # CUDA tensors); keep this test on the LLMM1 pad path it describes.
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    llmm1_mock.assert_called_once()
    w_padded = llmm1_mock.call_args.args[0]
    assert w_padded.shape == (4, 2048)
    assert torch.equal(w_padded[:m], weight)
    assert out.shape == (1, m)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
def test_rocm_unquantized_gemm_tiny_m_real_kernel(monkeypatch):
    # Real LLMM1 path for m=1 (shared_expert_gate shape): padded weight must
    # produce the same result as a plain linear on one token.
    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    weight = torch.randn(1, 2048, device="cuda", dtype=torch.float16)

    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    assert out.shape == (1, 1)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("m,expect_gemv", [(256, True), (2048, True), (1024, False)])
def test_rocm_unquantized_gemm_tiny_m_gemv_dispatch(monkeypatch, m, expect_gemv):
    # gfx906 custom W16A16 GEMV (P3-2b) dispatch rule: K=2048 rows with
    # N==256 (router) or N>=2048 (in_proj/qkv/LM head) take the GEMV; other
    # N (e.g. 1024 gate_up) stay on LLMM1. Mock-based, arch-independent.
    monkeypatch.delenv("VLLM_GFX906_DENSE_GEMV", raising=False)
    x = torch.randn(1, 2048, dtype=torch.float16)
    weight = torch.randn(m, 2048, dtype=torch.float16)

    gemv_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    llmm1_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    monkeypatch.setattr(utils.ops, "dense_gemv_gfx906", gemv_mock)
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: True)

    out = utils._llmm1_tiny_m(weight, x)
    ref = torch.nn.functional.linear(x, weight)

    assert gemv_mock.called is expect_gemv
    assert llmm1_mock.called is (not expect_gemv)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_tiny_m_gemv_never_off_gfx906(monkeypatch):
    # Regression guard: the custom GEMV is measured only on gfx906 and must
    # never route onto other ROCm targets, even for GEMV-eligible shapes.
    monkeypatch.delenv("VLLM_GFX906_DENSE_GEMV", raising=False)
    x = torch.randn(1, 2048, dtype=torch.float16)
    weight = torch.randn(256, 2048, dtype=torch.float16)

    gemv_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    llmm1_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    monkeypatch.setattr(utils.ops, "dense_gemv_gfx906", gemv_mock)
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: False)

    out = utils._llmm1_tiny_m(weight, x)
    ref = torch.nn.functional.linear(x, weight)

    gemv_mock.assert_not_called()
    llmm1_mock.assert_called_once()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("m", [256, 2048])
def test_rocm_unquantized_gemm_dense_gemv_real_kernel(monkeypatch, m):
    # Numeric gate for the gfx906 custom GEMV on its default model path
    # (K=2048, kchunk=2048 single-pass, RPT=2): must match F.linear at
    # fp16 precision. Catches row-mapping / RPT dispatch regressions.
    from vllm.platforms.rocm import on_gfx906

    if not on_gfx906():
        pytest.skip("dense_gemv_gfx906 is measured only on gfx906")
    torch.manual_seed(0)
    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    weight = torch.randn(m, 2048, device="cuda", dtype=torch.float16)

    out = utils.ops.dense_gemv_gfx906(weight, x, 2048)
    ref = torch.nn.functional.linear(x, weight)

    assert out.shape == (1, m)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.15, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("m", [256, 2048])
def test_rocm_unquantized_gemm_dense_gemv_ksplit_real_kernel(monkeypatch, m):
    # Numeric gate for the K-split (atomic packed-CAS) epilogue: kchunk=512
    # over K=2048 → ksplit=4, RPT=4 (64-bit pk4 CAS). The model path never
    # takes this branch (kchunk is hardcoded to K); this test keeps the
    # bench-only path honest.
    from vllm.platforms.rocm import on_gfx906

    if not on_gfx906():
        pytest.skip("dense_gemv_gfx906 is measured only on gfx906")
    torch.manual_seed(1)
    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    weight = torch.randn(m, 2048, device="cuda", dtype=torch.float16)

    out = utils.ops.dense_gemv_gfx906(weight, x, 512)
    ref = torch.nn.functional.linear(x, weight)

    assert out.shape == (1, m)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.15, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("m", [2, 3, 4])
def test_rocm_unquantized_gemm_spec_gemv_m4_real_kernel(monkeypatch, m):
    # Numeric gate for the M<=4 GEMV-family kernel (spec decode L1') on its
    # model path: K=5120 with kchunk=1024 (ksplit=5, packed-CAS epilogue).
    # Catches row-mapping / M-row / CAS regressions. Tolerance matches the
    # M=1 GEMV tests (fp16 output vs F.linear).
    from vllm.platforms.rocm import on_gfx906

    if not on_gfx906():
        pytest.skip("dense_gemv_m4_gfx906 is measured only on gfx906")
    monkeypatch.delenv("VLLM_GFX906_GEMVM_RPT", raising=False)
    torch.manual_seed(2)
    x = torch.randn(m, 5120, device="cuda", dtype=torch.float16)
    weight = torch.randn(1024, 5120, device="cuda", dtype=torch.float16)

    out = utils.ops.dense_gemv_m4_gfx906(weight, x, 1024)
    ref = torch.nn.functional.linear(x, weight)

    assert out.shape == (m, 1024)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.3, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
def test_rocm_unquantized_gemm_spec_gemv_m4_long_k_real_kernel(monkeypatch):
    # The K=17408 down_proj census shape (ksplit=17 at kchunk=1024) — the
    # longest CAS-accumulation chain the dispatch can select.
    from vllm.platforms.rocm import on_gfx906

    if not on_gfx906():
        pytest.skip("dense_gemv_m4_gfx906 is measured only on gfx906")
    monkeypatch.delenv("VLLM_GFX906_GEMVM_RPT", raising=False)
    torch.manual_seed(3)
    x = torch.randn(4, 17408, device="cuda", dtype=torch.float16)
    weight = torch.randn(5120, 17408, device="cuda", dtype=torch.float16)

    out = utils.ops.dense_gemv_m4_gfx906(weight, x, 1024)
    ref = torch.nn.functional.linear(x, weight)

    assert out.shape == (4, 5120)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.5, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("m", [2, 3, 4])
def test_rocm_unquantized_gemm_spec_gemv_m4_ksplit1_real_kernel(monkeypatch, m):
    # K=1024 with kchunk=1024 -> ksplit=1: the plain-store epilogue (no
    # CAS accumulation), a different code path than the ksplit>5/17
    # cases above. Shape matches the dispatch's K=1024 selection.
    from vllm.platforms.rocm import on_gfx906

    if not on_gfx906():
        pytest.skip("dense_gemv_m4_gfx906 is measured only on gfx906")
    monkeypatch.delenv("VLLM_GFX906_GEMVM_RPT", raising=False)
    torch.manual_seed(4)
    x = torch.randn(m, 1024, device="cuda", dtype=torch.float16)
    weight = torch.randn(14336, 1024, device="cuda", dtype=torch.float16)

    out = utils.ops.dense_gemv_m4_gfx906(weight, x, 1024)
    ref = torch.nn.functional.linear(x, weight)

    assert out.shape == (m, 14336)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.15, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("m", [2, 3, 4])
def test_rocm_unquantized_gemm_spec_gemv_m4_kc512_ksplit2(monkeypatch, m):
    # Regression: kchunk=512 (WARPS=1) with ksplit=2 is the single-warp
    # CAS-gather path. The M=4 runtime-M kernel previously gathered with
    # __shfl(acc_flat[t], i) at t==0 -- a uniform expression, so every
    # (r, m) slot got index 0's sum (silent wrong numerics). The dispatch
    # never selects kchunk=512 for the 27B (all Ks are 1024-multiples),
    # so only a direct-kchunk test catches this.
    from vllm.platforms.rocm import on_gfx906

    if not on_gfx906():
        pytest.skip("dense_gemv_m4_gfx906 is measured only on gfx906")
    monkeypatch.delenv("VLLM_GFX906_GEMVM_RPT", raising=False)
    torch.manual_seed(5)
    x = torch.randn(m, 1024, device="cuda", dtype=torch.float16)
    weight = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

    out = utils.ops.dense_gemv_m4_gfx906(weight, x, 512)
    ref = torch.nn.functional.linear(x, weight)

    assert out.shape == (m, 1024)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.15, rtol=2e-2)


def test_rocm_unquantized_gemm_spec_gemv_m4_dispatch(monkeypatch):
    # M=2..4 (spec decode draft steps) routes to the GEMV-family kernel on
    # gfx906; M=1 stays on the GEMV/LLMM1 path; the tuned hipBLAS special
    # case (m==5120, 2048<=k<=2304) and the kill switch stay on triton.
    monkeypatch.delenv("VLLM_GFX906_SPEC_GEMM", raising=False)
    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: True)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 64)
    triton_mock = MagicMock(side_effect=lambda a, b: a @ b.t())
    monkeypatch.setattr(utils, "triton_matmul", triton_mock)
    m4_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    llmm1_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    monkeypatch.setattr(utils.ops, "dense_gemv_m4_gfx906", m4_mock)
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    # M=4, K=5120 (the census fp16 shape): GEMV-family kernel
    x = torch.randn(4, 5120, dtype=torch.float16)
    weight = torch.randn(1024, 5120, dtype=torch.float16)
    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight)
    m4_mock.assert_called_once()
    assert m4_mock.call_args.args[2] == 1024  # kchunk
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)

    # M=1: the GEMV/LLMM1 path, not the M<=4 kernel
    m4_mock.reset_mock()
    triton_mock.reset_mock()
    x1 = torch.randn(1, 5120, dtype=torch.float16)
    out = utils.rocm_unquantized_gemm_impl(x1, weight, None)
    m4_mock.assert_not_called()
    ref1 = torch.nn.functional.linear(x1, weight)
    assert torch.allclose(out, ref1, atol=1e-3, rtol=1e-3)

    # M=4 on the tuned hipBLAS special-case shape: untouched
    m4_mock.reset_mock()
    xs = torch.randn(4, 2048, dtype=torch.float16)
    ws = torch.randn(5120, 2048, dtype=torch.float16)
    out = utils.rocm_unquantized_gemm_impl(xs, ws, None)
    m4_mock.assert_not_called()
    assert torch.allclose(out, torch.nn.functional.linear(xs, ws), atol=1e-3, rtol=1e-3)

    # Kill switch off: back to triton
    m4_mock.reset_mock()
    triton_mock.reset_mock()
    monkeypatch.setenv("VLLM_GFX906_SPEC_GEMM", "0")
    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    m4_mock.assert_not_called()
    triton_mock.assert_called_once()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_spec_gemv_m4_never_off_gfx906(monkeypatch):
    # Regression guard: the M<=4 kernel is measured only on gfx906 and must
    # never route onto other ROCm targets.
    monkeypatch.delenv("VLLM_GFX906_SPEC_GEMM", raising=False)
    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 64)
    triton_mock = MagicMock(side_effect=lambda a, b: a @ b.t())
    monkeypatch.setattr(utils, "triton_matmul", triton_mock)
    m4_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    monkeypatch.setattr(utils.ops, "dense_gemv_m4_gfx906", m4_mock)

    x = torch.randn(4, 5120, dtype=torch.float16)
    weight = torch.randn(1024, 5120, dtype=torch.float16)
    out = utils.rocm_unquantized_gemm_impl(x, weight, None)

    m4_mock.assert_not_called()
    triton_mock.assert_called_once()
    assert torch.allclose(
        out, torch.nn.functional.linear(x, weight), atol=1e-3, rtol=1e-3
    )


def test_rocm_unquantized_gemm_gfx950_wvsplitkrc_path(monkeypatch):
    x = torch.randn(1024, 16, dtype=torch.float16).t()
    weight = torch.randn(256, 1024, dtype=torch.float16)
    assert x.stride() == (1, 16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitkrc_mock = MagicMock(side_effect=lambda x_view, w, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitKrc", wvsplitkrc_mock)
    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitkrc_mock.assert_called_once()
    wvsplitk_mock.assert_not_called()
    x_view = wvsplitkrc_mock.call_args.args[0]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_fp32_gemv_mv_dispatch(monkeypatch):
    # fp32 single-token GEMVs (e.g. force_fp32_compute router gates) take
    # the hipBLAS sgemv (torch.mv) path: neither the fp16-only GEMV family
    # (ops.LLMM1 / dense_gemv_gfx906) nor triton_matmul may run. Mock-based,
    # arch-independent; the Nemotron-H router gate shape is the case.
    monkeypatch.delenv("VLLM_GFX906_DENSE_GEMV", raising=False)
    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: True)
    x = torch.randn(1, 2688, dtype=torch.float32)
    weight = torch.randn(128, 2688, dtype=torch.float32)

    llmm1_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)
    gemv_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    monkeypatch.setattr(utils.ops, "dense_gemv_gfx906", gemv_mock)
    triton_mock = MagicMock(side_effect=lambda xv, w: xv @ w.t())
    monkeypatch.setattr(utils, "triton_matmul", triton_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight)

    gemv_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    triton_mock.assert_not_called()
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_fp32_m_not_div4_stays_triton(monkeypatch):
    # Boundary pin for the mv dispatch: the n==1 GEMV guard requires
    # weight rows % 4 == 0 (or < 4); an fp32 weight outside that (m=130)
    # must NOT take torch.mv and falls through to the triton skinny path.
    monkeypatch.delenv("VLLM_GFX906_DENSE_GEMV", raising=False)
    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: True)
    x = torch.randn(1, 2688, dtype=torch.float32)
    weight = torch.randn(130, 2688, dtype=torch.float32)

    llmm1_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)
    triton_mock = MagicMock(side_effect=lambda xv, w: xv @ w.t())
    monkeypatch.setattr(utils, "triton_matmul", triton_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight)

    llmm1_mock.assert_not_called()
    triton_mock.assert_called_once()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
