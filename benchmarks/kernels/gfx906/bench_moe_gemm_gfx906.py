#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""P2-0 diagnostics for the gfx906 WNA16 MoE kernel.

Runs the gfx906 kernel over the full M-bucket table and reports, per bucket,
per-call time and achieved TFLOPS / GB/s vs the MI50 measured dot2 peak
(~20 TFLOPS practical, <=1 TB/s HBM2).

Env:
  BENCH_MS  csv of M values (default 1,8,32,128,512,2048)
  PROF      if set and non-empty, run one M=512 prefill-sized call exactly
            (for wrapping under rocprofv3) instead of the timing loop.

Roofline reference per review:
  w13 full table traffic = E*N*K/2 = 256*1024*2048/2 bytes = 268 MB
  w13 fp16 FLOPs = EM*K*N*2 (EM = M*TOPK)
"""
import os

import torch

torch.manual_seed(0)
dev = "cuda"
E = 256
TOPK = 8
GS = 128
N13, K13 = 1024, 2048
N2, K2 = 2048, 512
PEAK_F16 = 20e12  # MI50 measured v_dot2_f32_f16 peak (~20 TFLOPS practical)
PEAK_BW = 1e12

w13_bytes = E * N13 * K13 // 2
w2_bytes = E * N2 * K2 // 2


def _pack_nib(q):
    sh = 1 << (4 * torch.arange(8, device=dev))
    return (q.view(*q.shape[:-1], q.shape[-1] // 8, 8) * sh).sum(-1).to(torch.int32)


def make_layer():
    q13 = torch.randint(0, 16, (E, K13, N13), dtype=torch.int32, device=dev)
    q2 = torch.randint(0, 16, (E, K2, N2), dtype=torch.int32, device=dev)
    w13, w2 = _pack_nib(q13), _pack_nib(q2)
    s13 = torch.rand(E, K13 // GS, N13, device=dev, dtype=torch.float16) * 0.1 + 0.01
    s2 = torch.rand(E, K2 // GS, N2, device=dev, dtype=torch.float16) * 0.1 + 0.01
    z13 = _pack_nib(
        torch.randint(0, 16, (E, K13 // GS, N13), dtype=torch.int32, device=dev))
    z2 = _pack_nib(
        torch.randint(0, 16, (E, K2 // GS, N2), dtype=torch.int32, device=dev))
    return w13, w2, s13, s2, z13, z2


def main():
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        _repack_w4a16_gfx906_expert,
    )

    w13, w2, s13, s2, z13, z2 = make_layer()
    wq13, sc13, zp13 = _repack_w4a16_gfx906_expert(w13, s13, z13)
    wq2, sc2, zp2 = _repack_w4a16_gfx906_expert(w2, s2, z2)

    profile = bool(os.environ.get("PROF"))
    if profile:
        M = int(os.environ.get("PROF_M", "512"))
        EM = M * TOPK
        bm = 8 if EM > 512 else (4 if EM > 32 else 1)  # matches gfx906_w4a16_moe.py
        x = torch.randn(M, K13, device=dev, dtype=torch.float16)
        topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=dev)
        sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids, bm, E)
        c1 = torch.zeros(EM, N13, device=dev, dtype=torch.float16)
        print(f"PROF_READY M={M} EM={EM} bm={bm} grid=({EM // bm},1,{K13 // 256})")
        # run once (profiler storms this)
        ops.moe_gptq_gemm_gfx906(x, c1, wq13, sc13, zp13,
                                 torch.empty(0, dtype=torch.float32, device=dev),
                                 sorted_ids, expert_ids, ntp, TOPK, bm, False, 0, 0)
        torch.cuda.synchronize()
        print("PROF_DONE")
        return

    from vllm import _custom_ops as ops
    Ms = [int(v) for v in os.environ.get("BENCH_MS", "1,8,32,128,512,2048").split(",")]

    def timeit(fn, n=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / n * 1e3

    print(f"{'M':>5} {'bm':>3} | {'w13 us':>8} | {'w2 us':>8} | "
          f"{'w13 TF':>8} {'w2 TF':>8} | {'w13 bw':>7} | {'w2 bw':>7}")
    for M in Ms:
        EM = M * TOPK
        bm = 8 if EM > 512 else (4 if EM > 32 else 1)  # matches gfx906_w4a16_moe.py
        x = torch.randn(M, K13, device=dev, dtype=torch.float16)
        topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=dev)
        topk_w = torch.rand(M, TOPK, device=dev, dtype=torch.float16)
        sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids, bm, E)
        c1 = torch.zeros(EM, N13, device=dev, dtype=torch.float16)
        out = torch.zeros(M, N2, device=dev, dtype=torch.float16)
        empty_tw = torch.empty(0, dtype=torch.float32, device=dev)

        def g1():
            c1.zero_()
            ops.moe_gptq_gemm_gfx906(
                x, c1, wq13, sc13, zp13, empty_tw, sorted_ids, expert_ids,
                ntp, TOPK, bm, False, 0, 0)

        g1()
        inter = (torch.nn.functional.silu(c1[:, : N13 // 2].float()) *
                 c1[:, N13 // 2:].float()).half().contiguous()

        def g2():
            out.zero_()
            ops.moe_gptq_gemm_gfx906(
                inter, out, wq2, sc2, zp2, topk_w.view(-1).float(), sorted_ids,
                expert_ids, ntp, 1, bm, True, TOPK, 0)

        t1, t2 = timeit(g1), timeit(g2)
        # prefill bandwidth: all experts read (approx) once per M-slice.
        # rows per expert = EM/E (if EM >= E), else only active experts read.
        r13 = max(1, (EM // E + bm - 1) // bm) if EM >= E else 1
        # (r13: blocks/expert over k? not exact)
        tf1 = EM * K13 * N13 * 2 / (t1 * 1e-6) / PEAK_F16 * 100
        tf2 = EM * K2 * N2 * 2 / (t2 * 1e-6) / PEAK_F16 * 100
        bw1 = w13_bytes * r13 / (t1 * 1e-6) / 1e9
        bw2 = w2_bytes * r13 / (t2 * 1e-6) / 1e9
        # pad to help g1/g2 not be skipped
        print(f"{M:5d} {bm:3d} | {t1:8.1f} | {t2:8.1f} | "
              f"{tf1:7.1f}% {tf2:7.1f}% | {bw1:6.0f} {bw2:6.0f} GB/s", flush=True)


if __name__ == "__main__":
    main()