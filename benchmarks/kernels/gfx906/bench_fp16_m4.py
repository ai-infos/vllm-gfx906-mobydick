#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Microbench: dense_gemv_m4_gfx906 (M<=4) vs triton vs M=1 GEMV.

Shapes from the gemm_step_census (27B dense, fp16 family, spec
draft step M=4): (N, K, calls/step).

  (96, 5120)       ~43   GDN a/b in_proj
  (14336, 5120)    ~14   FA qkv
  (248320, 5120)    1    LM head (the whale: 2.55 GB)
  (16384, 5120)     1    layer0 in_proj_qkvz (fp16)
  (5120, 6144)      1    layer0 fused
  (34816, 5120)     1    layer0 gate+up
  (5120, 17408)     1    layer0 down

Compares, per shape: triton_matmul @ M=4 (status quo), the new
dense_gemv_m4 @ M=4 (kchunk sweep, RPT via env), and the M=1
reference (old dense_gemv or m4 @ M=1) to check weight-invariance.

RPT sweep: set VLLM_GFX906_GEMVM_RPT externally (2 default / 4).
"""
import os
import sys

import torch

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

SHAPES = [
    # (N, K, calls_per_step)
    (96, 5120, 43),
    (14336, 5120, 14),
    (248320, 5120, 1),
    (16384, 5120, 1),
    (5120, 6144, 1),
    (34816, 5120, 1),
    (5120, 17408, 1),
]

dev = "cuda"
w_cache = {}


def get_w(N, K):
    key = (N, K)
    if key not in w_cache:
        w_cache[key] = torch.randn(N, K, device=dev, dtype=torch.float16)
    return w_cache[key]


def triton_m(path, N, K, M, iters):
    from vllm.model_executor.layers.utils import triton_matmul

    w = get_w(N, K)
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    for _ in range(3):
        triton_matmul(x, w)
    torch.cuda.synchronize()
    st = torch.cuda.Event(True)
    en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        triton_matmul(x, w)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000


def m4(N, K, M, kchunk, iters, op):
    w = get_w(N, K)
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    for _ in range(3):
        op(w, x, kchunk)
    torch.cuda.synchronize()
    st = torch.cuda.Event(True)
    en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        op(w, x, kchunk)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000


def ref_m1(N, K, iters):
    """M=1 engine path: old dense_gemv where the engine routes there,
    else LLMM1 — approximate with dense_gemv_m4 @ M=1 for the K=17408
    shape and old dense_gemv where valid; for K=5120 the engine uses
    LLMM1 (measure it too when available)."""
    import vllm._custom_ops as ops

    w = get_w(N, K)
    x = torch.randn(1, K, device=dev, dtype=torch.float16)
    kchunk = 1024 if K % 1024 == 0 else 512
    try:
        fn = lambda: ops.dense_gemv_gfx906(w, x, kchunk)
        name = "gemv"
    except Exception:
        fn = lambda: ops.dense_gemv_m4_gfx906(w, x, kchunk)
        name = "m4m1"
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True)
    en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000, name


def main():
    import vllm._custom_ops as ops

    rpt = os.environ.get("VLLM_GFX906_GEMVM_RPT", "2")
    print(f"RPT={rpt}  (dense_gemv_m4_gfx906 sweep)", flush=True)
    print(f"{'N':>7} {'K':>6} {'calls':>5} | {'M=1 ref':>9} | "
          f"{'M=4 triton':>11} {'M=4 kc512':>11} {'M=4 kc1024':>12} | "
          f"{'step_ms triton':>15} {'step_ms m4':>12}", flush=True)
    tot_tr = tot_m4 = tot_m1 = 0.0
    for N, K, calls in SHAPES:
        iters = 20 if N * K > 100_000_000 else 100
        t_ref, ref_name = ref_m1(N, K, iters)
        t_tri = triton_m(None, N, K, 4, iters)
        kchunks = [kc for kc in (512, 1024) if K % kc == 0]
        best = None
        parts = []
        for kc in kchunks:
            t = m4(N, K, 4, kc, iters, ops.dense_gemv_m4_gfx906)
            parts.append(t)
            if best is None or t < best:
                best = t
        parts += [float("nan")] * (2 - len(parts))
        tot_tr += t_tri * calls
        tot_m4 += best * calls
        tot_m1 += t_ref * calls
        print(f"{N:>7} {K:>6} {calls:>5} | {t_ref:8.1f} {ref_name[0]} | "
              f"{t_tri:11.1f} {parts[0]:11.1f} {parts[1]:12.1f} | "
              f"{t_tri*calls:15.1f} {best*calls:12.1f}", flush=True)
    print("-" * 78, flush=True)
    print(f"per-step total (all shapes): M=1 ref {tot_m1:7.1f} ms | "
          f"triton M=4 {tot_tr:7.1f} ms | m4 M=4 {tot_m4:7.1f} ms  "
          f"(L1' saving: {tot_tr - tot_m4:.1f} ms)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
