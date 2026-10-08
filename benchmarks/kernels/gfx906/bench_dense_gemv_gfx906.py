#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""P3-2(b): M=1 W16A16 dense GEMV micro-bench (dense_gemv_gfx906 vs LLMM1).

Per-shape correctness + timing of the new gfx906 dense GEMV kernel
(csrc/rocm/dense_gemv_gfx906.cu) against the current decode path
(ops.LLMM1 rpb=4, LLGemm1_kernel), per plan gate "micro-bench per shape
before touching the model path".

Config sweep: (kchunk, RPT) pairs — kchunk 512|2048|4096 (KSPLIT =
K/kchunk; >1 uses fp16-CAS accumulation into a pre-zeroed output, zero_
included in the timed op), RPT (rows per thread) via VLLM_GFX906_GEMV_RPT
(auto = 4 if N%4==0 else 2 if N%2==0 else 1).

Shapes = the 8 LLGemm1 rows of this model (M=weight rows, K, layers/step)
from the P3-0 Q3 probe.

Run in the gfx906 vLLM image with the repo source-mounted:
  python3 -u /bench/bench_dense_gemv_gfx906.py
"""
import os

import torch

dev = "cuda"
torch.manual_seed(0)

HBM_BW = 798e9  # P3-0 Q1: measured MI50 HBM read BW
SHAPES = [
    (12288, 2048, 30, "GDN in_proj"),
    (248320, 2048, 1, "LM head"),
    (9216, 2048, 10, "FA qkv"),
    (2048, 4096, 40, "o_proj"),
    (1024, 2048, 40, "shared gate_up"),
    (2048, 512, 40, "shared down"),
    (256, 2048, 40, "router"),
    (64, 2048, 30, "GDN small"),
]


def time_us(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters  # us/call


def configs(N, K):
    """(kchunk, rpt) sweep for this shape; rpt 'auto' = default selection."""
    cfgs = []
    if K % 4096 == 0:
        cfgs += [(4096, "auto"), (4096, 2)]
    if K % 2048 == 0:
        cfgs += [(2048, "auto"), (2048, 2)]
    if K == 512:
        cfgs += [(512, "auto"), (512, 2), (512, 1)]
    return cfgs


def main():
    from vllm import _custom_ops as ops

    print("dense_gemv_gfx906 v2 (P3-2b) vs LLMM1 rpb=4 (current), M=1")
    print(f"{'shape':>16} {'name':<15} {'x/step':>6} {'LLMM1':>8} "
          f"{'cfg':>12} {'us':>8} {'floor':>8} {'vs LLMM1':>9}")

    step_total = 0.0
    for N, K, layers, name in SHAPES:
        w = torch.randn(N, K, dtype=torch.float16, device=dev)
        x = torch.randn(1, K, dtype=torch.float16, device=dev)
        ref = (x.float() @ w.float().T)  # [1, N] fp32

        iters = 50 if N * K > 8 * 1024 * 1024 else 300
        us_llmm1 = time_us(lambda: ops.LLMM1(w, x, 4), warmup=20, iters=iters)
        floor = N * K * 2 / HBM_BW * 1e6

        best_us, best_cfg = us_llmm1, "LLMM1 rpb4"
        for kc, rpt in configs(N, K):
            if rpt == "auto":
                os.environ.pop("VLLM_GFX906_GEMV_RPT", None)
            else:
                os.environ["VLLM_GFX906_GEMV_RPT"] = str(rpt)
            out = ops.dense_gemv_gfx906(w, x, kc)  # [1, N] fp16
            md = (out.float() - ref).abs().max().item()
            atol = 0.25
            assert md < atol, f"{name} kc={kc} rpt={rpt}: maxdiff {md:.3f} >= {atol}"
            us = time_us(lambda: ops.dense_gemv_gfx906(w, x, kc),
                         warmup=20, iters=iters)
            del out
            tag = f"kc{kc}/r{rpt}"
            print(f"{N}x{K:<9} {name:<15} {layers:>6} {us_llmm1:>8.1f} "
                  f"{tag:>12} {us:>8.1f} {floor:>8.1f} "
                  f"{100*(us-us_llmm1)/us_llmm1:>+8.0f}%")
            if us < best_us:
                best_us, best_cfg = us, tag
        os.environ.pop("VLLM_GFX906_GEMV_RPT", None)
        step_total += best_us * layers
        print(f"  -> best {best_cfg}: {best_us:.1f} us "
              f"({best_us/floor:.1f}x floor, "
              f"{100*(us_llmm1-best_us)/us_llmm1:+.0f}% vs LLMM1)")
        del w, x, ref

    # Current per-step total from the Day-1 rpb sweep (rpb=4): 5604 us.
    print(f"\nbest-per-shape weighted step total: {step_total:.1f} us/step "
          f"(Day-1 LLMM1 rpb=4 baseline: 5604 us/step; "
          f"floor across rows: ~4600 us)")


if __name__ == "__main__":
    main()
