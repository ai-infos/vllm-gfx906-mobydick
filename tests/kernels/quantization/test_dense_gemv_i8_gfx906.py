# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Unit tests for the NH-2' W8A16 int8-weight GEMV CUDA kernels
(dense_gemv_i8_gfx906 / dense_gemv_i8_m4_gfx906).

Correctness is checked against a float64 reference of the exact on-device
convention (w = w_i8 * s per channel, fp32-accumulated dot — int8 x fp16
products are exact in fp32, so the only path-vs-path difference is
accumulation order), and the CUDA kernels are additionally cross-checked
against the Triton k_w8a16_gemv / k_w8a16_gemm reference. A throughput
floor test guards the HBM-bound design assumption (CUDA must not trail
Triton badly on the big K=2688 rows — the serving A/B gate is the final
word).

Shapes mirror Nemotron-3.5-Lightning's int8 dense projections:
in_proj/out_proj/qkv/lm_head at K=2688 (the shapes the fp16 GEMV dispatch
does not reach) and o_proj at K=2048.
"""

import os

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

_HAS_OPS = hasattr(__import__("vllm._custom_ops", fromlist=["x"]), "dense_gemv_i8_gfx906")
requires_cuda_ops = pytest.mark.skipif(
    not _HAS_OPS, reason="_rocm_C dense_gemv_i8_* ops not built"
)

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (  # noqa: E402
    compressed_tensors_w8a16_channel_dequant as w8ch,
)


def _ref(w_i8: torch.Tensor, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """float64 reference: out[m, n] = s[n] * sum_k w_i8[n, k] * x[m, k]."""
    wf = w_i8.to(torch.float64)
    xf = x.to(torch.float64)
    sf = s.to(torch.float64)
    return (xf @ wf.t()) * sf[None, :]


def _mks(n: int, k: int, m: int, seed: int = 0):
    torch.manual_seed(seed)
    # Signed int8 weights in the realistic quantized range.
    w_i8 = torch.randint(-128, 127, (n, k), dtype=torch.int32, device="cuda").to(torch.int8)
    s = (torch.rand(n, device=w_i8.device) * 0.05 + 0.005).half()
    x = (torch.randn(m, k, device=w_i8.device) * 0.5).half()
    return w_i8.contiguous(), s.contiguous(), x.contiguous()


def _close(a: torch.Tensor, b: torch.Tensor, atol: float = 0.5,
           rtol: float = 2e-3) -> bool:
    """fp16-accumulation-tolerant closeness (family convention).

    The CUDA family accumulates in fp16 (__ockl_fdot2), like LLMM1 and the
    fp16 dense_gemv kernels — so absolute error scales with the partial-sum
    magnitude (~sqrt(K) * term scale * 2^-11), NOT with the output magnitude.
    Outputs near zero (cancellation over K=2688 terms) carry O(0.1) absolute
    error, which a pure relative check misreports as huge; hence atol + rtol.
    A real structural bug (wrong x pairing, sign-extension, missing tail)
    produces O(10+) errors and fails this comfortably.
    """
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    return bool(((a - b).abs() <= atol + rtol * b.abs()).all())


def _check(w_i8, s, x, out, label):
    ref = _ref(w_i8, s, x)
    assert _close(out, ref), (
        f"{label}: max abs err {float((out.to(torch.float64) - ref).abs().max()):.3e} "
        f"vs atol=0.5 + rtol=2e-3*|ref|"
    )
    return ref


@requires_cuda_ops
def test_m1_k2688_ksplit2_tail_masked(dist_init):
    # lm_head geometry (N trimmed): K=2688, KC=2048 -> ksplit=2 with a
    # 640-element tail block; exercises the inb mask + CAS + scale kernel.
    w_i8, s, x = _mks(1024, 2688, 1)
    out = torch.ops._rocm_C.dense_gemv_i8_gfx906(w_i8, s, x[0].contiguous(), 2048)
    assert out.shape == (1, 1024) and out.dtype == torch.float16
    _check(w_i8, s, x, out, "M=1 K=2688 KC=2048")


@requires_cuda_ops
def test_m1_k2688_single_pass(dist_init):
    # Same shape, KC=4096 single pass (no CAS, no scale kernel).
    w_i8, s, x = _mks(1024, 2688, 1)
    out = torch.ops._rocm_C.dense_gemv_i8_gfx906(w_i8, s, x[0].contiguous(), 4096)
    _check(w_i8, s, x, out, "M=1 K=2688 KC=4096")


@requires_cuda_ops
def test_m1_k2048_single_pass(dist_init):
    # o_proj geometry: K=2048 == KC.
    w_i8, s, x = _mks(512, 2048, 1)
    out = torch.ops._rocm_C.dense_gemv_i8_gfx906(w_i8, s, x[0].contiguous(), 2048)
    _check(w_i8, s, x, out, "M=1 K=2048 KC=2048")


@requires_cuda_ops
def test_m4_in_proj_geometry(dist_init):
    # The serving spec-decode target: in_proj [10304, 2688] at M=4.
    w_i8, s, x = _mks(10304, 2688, 4)
    out = torch.ops._rocm_C.dense_gemv_i8_m4_gfx906(w_i8, s, x, 2048)
    assert out.shape == (4, 10304) and out.dtype == torch.float16
    _check(w_i8, s, x, out, "M=4 K=2688 KC=2048")


@requires_cuda_ops
def test_m1_through_m4_single_pass(dist_init):
    for m in (1, 2, 3, 4):
        w_i8, s, x = _mks(256, 2048, m)
        out = torch.ops._rocm_C.dense_gemv_i8_m4_gfx906(w_i8, s, x, 2048)
        assert out.shape == (m, 256)
        _check(w_i8, s, x, out, f"M={m} K=2048 KC=2048")


@requires_cuda_ops
def test_matches_triton_reference(dist_init):
    # CUDA vs the Triton k_w8a16_gemv / k_w8a16_gemm (same convention).
    # Tolerance is _close (atol 0.5 + rtol 2e-3), not pure relative: the two
    # paths accumulate in different precisions (CUDA fp16 fdot2, Triton fp32)
    # and K=2688 outputs near zero carry O(0.1-0.2) absolute disagreement.
    w_i8, s, x = _mks(1024, 2688, 1)
    cuda_out = torch.ops._rocm_C.dense_gemv_i8_gfx906(
        w_i8, s, x[0].contiguous(), 2048
    )
    triton_out = w8ch.w8a16_gemv(w_i8, s, x[0].contiguous()).unsqueeze(0)
    assert _close(cuda_out, triton_out), (
        f"CUDA vs Triton M=1: max abs err "
        f"{float((cuda_out.to(torch.float64) - triton_out.to(torch.float64)).abs().max()):.3e}"
    )

    w_i8, s, x = _mks(1024, 2688, 4)
    cuda_out = torch.ops._rocm_C.dense_gemv_i8_m4_gfx906(w_i8, s, x, 2048)
    triton_out = w8ch.w8a16_gemm(w_i8, s, x)
    assert _close(cuda_out, triton_out), (
        f"CUDA vs Triton M=4: max abs err "
        f"{float((cuda_out.to(torch.float64) - triton_out.to(torch.float64)).abs().max()):.3e}"
    )


@requires_cuda_ops
def test_rejects_unsupported_shapes(dist_init):
    # K % 16 != 0: aligned uint4 loads impossible.
    w_i8, s, x = _mks(64, 2700, 1)
    with pytest.raises(Exception, match="multiple of 16"):
        torch.ops._rocm_C.dense_gemv_i8_gfx906(w_i8, s, x[0].contiguous(), 2048)
    # Odd N: no packed-CAS epilogue.
    w_i8, s, x = _mks(129, 2048, 1)
    with pytest.raises(Exception, match="even"):
        torch.ops._rocm_C.dense_gemv_i8_gfx906(w_i8, s, x[0].contiguous(), 2048)
    # M out of range.
    w_i8, s, x = _mks(128, 2048, 5)
    with pytest.raises(Exception, match="M must be"):
        torch.ops._rocm_C.dense_gemv_i8_m4_gfx906(w_i8, s, x, 2048)
    # Bad kchunk.
    w_i8, s, x = _mks(128, 2048, 1)
    with pytest.raises(Exception, match="kchunk must be"):
        torch.ops._rocm_C.dense_gemv_i8_gfx906(w_i8, s, x[0].contiguous(), 512)


def test_kchunk_selection():
    # Single pass whenever a supported KC covers K (measured fp16 evidence:
    # K-split is 2.4-4.2x slower at M=1 — CAS + zero_ + tiny-block overhead).
    assert w8ch._i8_cuda_kchunk(2688) == 4096  # the NH-2' target shape
    assert w8ch._i8_cuda_kchunk(2048) == 2048
    assert w8ch._i8_cuda_kchunk(1024) == 2048
    assert w8ch._i8_cuda_kchunk(4096) == 4096
    # K > 4096: split only when divisible by a supported KC.
    assert w8ch._i8_cuda_kchunk(17408) == 1024  # 17408 = 1024 * 17
    assert w8ch._i8_cuda_kchunk(8192) == 2048   # 8192 = 2048 * 4
    # Unservable: not a multiple of 16 / no supported split.
    assert w8ch._i8_cuda_kchunk(2700) is None
    assert w8ch._i8_cuda_kchunk(5000) is None


@requires_cuda_ops
def test_dispatch_routes_to_cuda(dist_init, monkeypatch):
    # With VLLM_GFX906_W8A16_INT8_CUDA=1 the scheme's apply_weights must use
    # the CUDA op for M<=4 / even-N shapes and fall back to Triton otherwise.
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A16ChannelDequant,
    )

    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: True)
    monkeypatch.setenv("VLLM_GFX906_W8A16_INT8", "1")
    monkeypatch.setenv("VLLM_GFX906_W8A16_INT8_CUDA", "1")

    scheme = CompressedTensorsW8A16ChannelDequant(layer_name="t.i8cuda")
    import torch.nn as nn

    layer = nn.Module()
    scheme.create_weights(
        layer,
        output_size=512,
        input_size=2688,
        output_partition_sizes=[512],
        input_size_per_partition=2688,
        params_dtype=torch.float16,
        weight_loader=lambda *a, **kw: None,
    )
    torch.manual_seed(0)
    w_i8 = torch.randint(-128, 127, (512, 2688), dtype=torch.int32, device="cuda").to(torch.int8)
    s = (torch.rand(512, 1, device=w_i8.device) * 0.05 + 0.005).half()
    layer.weight_i8 = nn.Parameter(w_i8.contiguous(), requires_grad=False)
    layer.weight_scale = nn.Parameter(s, requires_grad=False)

    x1 = (torch.randn(1, 2688, device=w_i8.device) * 0.5).half()
    out_cuda = scheme.apply_weights(layer, x1, None)
    ref = _ref(w_i8, s.squeeze(-1), x1)
    assert _close(out_cuda, ref), (
        f"dispatch M=1: max abs err "
        f"{float((out_cuda.to(torch.float64) - ref).abs().max()):.3e}"
    )

    # M=6 (spec-decode serving mode): the CUDA family stops at M<=4, so the
    # Triton path must handle it.
    x6 = (torch.randn(6, 2688, device=w_i8.device) * 0.5).half()
    out_triton = scheme.apply_weights(layer, x6, None)
    ref6 = _ref(w_i8, s.squeeze(-1), x6)
    assert _close(out_triton, ref6), (
        f"dispatch M=6 (Triton): max abs err "
        f"{float((out_triton.to(torch.float64) - ref6).abs().max()):.3e}"
    )


@requires_cuda_ops
def test_throughput_floor_lm_head(dist_init):
    # The design premise: byte-loading halves weight traffic vs the fp16
    # family and must not trail Triton int8 badly on the big K=2688 row.
    # (The serving A/B gate is the final verdict; this catches gross
    # regressions like a broken vectorized load.)
    n, k = 131072, 2688
    w_i8, s, x = _mks(n, k, 1)

    def bench(fn, iters=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        return st.elapsed_time(en) / iters * 1e-3  # seconds

    t_cuda = bench(
        lambda: torch.ops._rocm_C.dense_gemv_i8_gfx906(
            w_i8, s, x[0].contiguous(), 2048
        )
    )
    t_triton = bench(lambda: w8ch.w8a16_gemv(w_i8, s, x[0].contiguous()))
    gb_cuda = n * k / t_cuda / 1e9
    gb_triton = n * k / t_triton / 1e9
    print(f"\n[lm_head i8] CUDA {gb_cuda:.0f} GB/s vs Triton {gb_triton:.0f} GB/s "
          f"({t_cuda * 1e6:.1f} us vs {t_triton * 1e6:.1f} us)")
    assert gb_cuda > 0.5 * gb_triton, (
        f"CUDA int8 GEMV trails Triton by >2x ({gb_cuda:.0f} vs {gb_triton:.0f} GB/s)"
    )
