#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Dense Qwen3.5-27B K=5120 M=1 W16A16 GEMV micro-bench.

Tests the dense_gemv_gfx906 kernel at the dense model's hidden size
(K=5120; KSPLIT = K/kchunk with fp16-CAS accumulation) against the
current decode path (LLMM1 rpb=4), per the handover doc item 6b#3.

Dense 27B fp16 GEMV shapes (weight [N, K], layers/step):
  LM head        248320 x 5120  x 1
  FA qkv+fused   14336  x 5120  x 16   (q 24*(1+gate)=48*256 + k+v 4*256*2)
  GDN in_proj_b  6144   x 5120  x 48
  GDN in_proj_a  2048   x 5120  x 48
  FA kv rows     1024   x 5120  x 16   (coverage; vLLM fuses into qkv)

Config sweep per shape:
  baseline  : ops.LLMM1(w, x, 4)            <- current decode path (m%4==0)
  gemv 512  : kchunk=512  (KSPLIT=10)
  gemv 1024 : kchunk=1024 (KSPLIT=5)
each with RPT auto / 2 / 4 (1 rejected by the kernel for KSPLIT>1).

Prints one JSON line per (shape, config): us/call, GB/s, % of HBM floor,
max abs diff vs fp32 reference.

Run: .venv/bin/python benchmarks/kernels/gfx906/bench_dense_gemv_k5120.py
"""
import json
import os

import torch

dev = "cuda"
torch.manual_seed(0)

HBM_BW = 798e9  # measured MI50 HBM read BW (P3-0 Q1)

SHAPES = [
    (248320, 5120, 1, "LM head"),
    (14336, 5120, 16, "FA qkv+fused"),
    (6144, 5120, 48, "GDN in_proj_b"),
    (2048, 5120, 48, "GDN in_proj_a"),
    (1024, 5120, 16, "FA kv rows"),
]

# kchunk configs for K=5120 (must divide 5120); 2048/4096 do not divide.
KCHUNKS = [512, 1024]
RPTS = ["auto", 2, 4]


def time_us(fn, warmup=10, iters=100):
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


def main():
    import importlib.util

    from vllm import _custom_ops as ops

    gemv_so = "/tmp/bench/gemv_build/_gfx906_gemv_bench.so"
    if not os.path.exists(gemv_so):
        raise SystemExit("standalone GEMV module missing — run "
                         "/tmp/bench/build_gemv_local.py first")
    spec = importlib.util.spec_from_file_location("_gfx906_gemv_bench", gemv_so)
    gemv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemv)

    rows = []
    for N, K, layers, name in SHAPES:
        w = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.05
        x = torch.randn(1, K, device=dev, dtype=torch.float16) * 0.05
        ref = (x.float() @ w.float().t())[0]
        floor_us = N * K * 2 / HBM_BW * 1e6  # single weight read

        # baseline: current decode path
        t_llmm1 = time_us(lambda w=w, x=x: ops.LLMM1(w, x, 4))
        out_llmm1 = ops.LLMM1(w, x, 4)[0]
        err_llmm1 = (out_llmm1.float() - ref).abs().max().item()
        rows.append({"shape": name, "N": N, "K": K, "layers": layers,
                     "floor_us": round(floor_us, 1),
                     "cfg": "LLMM1 rpb4", "us": round(t_llmm1, 1),
                     "gbps": round(N * K * 2 / (t_llmm1 * 1e-6) / 1e9, 1),
                     "pct_floor": round(t_llmm1 / floor_us * 100, 1),
                     "maxdiff": round(err_llmm1, 5)})

        for kc in KCHUNKS:
            if K % kc != 0:
                continue
            for rpt in RPTS:
                rpt_env = str(rpt) if rpt != "auto" else "0"
                if rpt != "auto":
                    os.environ["VLLM_GFX906_GEMV_RPT"] = rpt_env
                else:
                    os.environ.pop("VLLM_GFX906_GEMV_RPT", None)
                fn = lambda w=w, x=x, kc=kc: gemv.dense_gemv_gfx906(w, x, kc)
                t = time_us(fn)
                out = fn()[0]
                err = (out.float() - ref).abs().max().item()
                rows.append({"shape": name, "N": N, "K": K, "layers": layers,
                             "floor_us": round(floor_us, 1),
                             "cfg": f"gemv kc{kc} rpt{rpt}",
                             "us": round(t, 1),
                             "gbps": round(N * K * 2 / (t * 1e-6) / 1e9, 1),
                             "pct_floor": round(t / floor_us * 100, 1),
                             "maxdiff": round(err, 5)})
    os.environ.pop("VLLM_GFX906_GEMV_RPT", None)

    # per-step estimate: best cfg per shape x layers
    best = {}
    for r in rows:
        key = (r["shape"], r["N"])
        if key not in best or r["us"] < best[key]["us"]:
            best[key] = r
    base_step = sum(r["us"] * r["layers"] for r in rows
                    if r["cfg"] == "LLMM1 rpb4")
    best_step = sum(r["us"] * r["layers"]
                    for r in best.values() if r["cfg"] != "LLMM1 rpb4")
    print("GEMV5120: " + json.dumps(
        {"rows": rows,
         "step_us_baseline": round(base_step, 1),
         "step_us_best": round(best_step, 1),
         "step_save_us": round(base_step - best_step, 1)}))


if __name__ == "__main__":
    main()
