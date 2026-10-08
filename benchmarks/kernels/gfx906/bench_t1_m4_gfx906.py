#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""T1 Task 1 — standalone int8 W8A16 GEMV bench at our fp16-mass shapes.

Dispatcher-faithful (per mi50-kernel-time-benchmarking skill): the fp16 baseline
calls the REAL production dispatcher `rocm_unquantized_gemm_impl` (routes M=1 ->
_llmm1_tiny_m / M=2..4 -> _gfx906_spec_gemv_m4, never a hand-picked op). The int8
side is measured in two variants:

  i8-cuda   ops.dense_gemv_i8_m4_gfx906 (NH-2' CUDA family, M<=4) — sweeps the
            legal kchunk set (the launcher masks a partial tail, so kc=4096 with
            ksplit=ceil(K/4096) is legal even when K % 4096 != 0; NH-2's helper
            was conservative and only used divisors).
  i8-triton w8a16_gemv from the in-repo NH-2 scheme (M=1 only) — P2's measured
            GO path (1717 us at lm_head), kept as the fallback option.

CUDA-event deciles, hot loop, mclk>=900 gate. Correctness: int8-kernel output vs
a DEQUANTIZED-fp32 reference (isolates kernel bugs from quantization error); the
quantization error itself is reported separately (int8-dequant matmul vs fp16
matmul, both in fp32) — that one is expected ~1e-2 on random data and is gated by
Task 4's KLD/PPL on real model data, not here.

GO bar: best int8 variant >= 1.5x production-fp16 at lm_head AND gdn_in_proj_qkvz
(M=1 and M=3); no shape >5% slower than fp16.

Run (GPU idle, single card):
    source ~/env-rocm-7.14-gfx906.sh
    HIP_VISIBLE_DEVICES=0 .venv/bin/python \
        benchmarks/kernels/gfx906/bench_t1_m4_gfx906.py
    (T1_BENCH_QUICK=1: lm_head + gdn_in_proj_qkvz only)
"""

import os
import re
import subprocess
import threading
import time

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a16_channel_dequant as w8a16,
)
from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

# (name, N, K) — full-model (pre-TP) shapes. lm_head + the Qwen3_8 MTP draft
# layer are the ONLY plain-BF16 decode-read mass on the serving checkpoint
# (GDN/FA/MLP are INT4-packed); gdn/fa rows kept for cross-validation only.
SHAPES = [
    ("lm_head", 248320, 5120),          # shared target+drafter head
    ("mtp_fc", 5120, 10240),            # draft layer fc (concat embed+hidden)
    ("mtp_q_proj", 12288, 5120),        # draft full-attn q
    ("mtp_k_proj", 1024, 5120),         # draft full-attn k
    ("mtp_v_proj", 1024, 5120),         # draft full-attn v
    ("mtp_o_proj", 5120, 6144),         # draft full-attn o
    ("mtp_gate_up", 17408, 5120),       # draft MLP gate+up (each)
    ("mtp_down", 5120, 17408),          # draft MLP down
    ("gdn_in_proj_qkvz", 16384, 5120),  # cross-validation only (INT4 on ckpt)
]
if os.environ.get("T1_BENCH_QUICK") == "1":
    SHAPES = [s for s in SHAPES if s[0] in ("lm_head", "gdn_in_proj_qkvz")]

MS = (1, 3)  # decode M=1, verify M=k+1=3 (MTP depth-2)


def kchunk_options(k: int):
    """Legal kchunks for the m4 launcher. The kernel masks a partial tail
    (`inb = (k0 + t*16) < K`), so every kc in {1024,2048,4096} is legal for any
    K%16==0 — NH-2's `_i8_cuda_kchunk` helper was conservative and only returned
    divisors, but the launcher itself does not require that. Largest first (fewest
    ksplit / CAS passes); the bench measures all and keeps the fastest."""
    if k % 16 != 0:
        return []
    return [kc for kc in (4096, 2048, 1024)]


def deciles(xs):
    a = np.asarray(xs, dtype=np.float64)
    p = lambda q: float(np.percentile(a, q))
    return p(5), p(25), p(50), p(75), p(95)


class MclkSampler(threading.Thread):
    """Samples `rocm-smi --showclocks` mclk every 0.25 s while running."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._stop_evt = threading.Event()

    def run(self):
        pat = re.compile(r"mclk clock level.*?\((\d+)\s*Mhz\)")
        while not self._stop_evt.is_set():
            try:
                out = subprocess.run(
                    ["rocm-smi", "--showclocks"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                m = pat.search(out)
                if m:
                    self.samples.append(int(m.group(1)))
            except Exception:
                pass
            time.sleep(0.25)

    def stop(self):
        self._stop_evt.set()


def bench(fn, iters, warmup):
    """Hot loop, CUDA-event per-iter timing (block-of-pairs), no per-iter sync."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))  # ms
    return times


def main():
    dev = "cuda"
    x_dummy = torch.zeros(2, 8, dtype=torch.float16, device=dev)  # init ctx
    del x_dummy

    print("T1 Task-1 int8 W8A16 bench (dispatcher-faithful)")
    print(f"torch {torch.__version__} | cuda dev {torch.cuda.get_device_name(0)}")
    smp = MclkSampler()
    smp.start()

    # Global warmup to boost mclk to 1000 before any timed window.
    for name, N, K in SHAPES:
        w = torch.randn(N, K, dtype=torch.float16, device=dev)
        x = torch.randn(3, K, dtype=torch.float16, device=dev)
        rocm_unquantized_gemm_impl(x, w, None)
        del w, x
    torch.cuda.synchronize()

    results = []
    for name, N, K in SHAPES:
        if K % 16 != 0 or N % 2 != 0:
            print(f"  {name} [{N},{K}]: SKIP (kernel constraint)")
            continue
        w16 = torch.randn(N, K, dtype=torch.float16, device=dev) * 0.02
        # per-channel int8 quantization (signed), scale fp16 [N]
        amax = w16.abs().amax(dim=1).clamp(min=1e-8)
        scale = (amax / 127.0).to(torch.float16)
        q = torch.round(w16 / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)

        # --- correctness: kernel vs dequantized reference (isolates kernel bug)
        x1 = (torch.randn(1, K, dtype=torch.float16, device=dev) * 0.02).contiguous()
        w_deq = (q.to(torch.float32) * scale.to(torch.float32).unsqueeze(1))
        ref_deq = (x1.to(torch.float32) @ w_deq.t())[0]
        out_i8 = ops.dense_gemv_i8_m4_gfx906(q, scale, x1, 4096 if K > 4096 else 1024)[0]
        kern_rel = ((out_i8.to(torch.float32) - ref_deq).abs()
                    / (ref_deq.abs() + 1e-3)).mean().item()
        # quantization error itself (int8-dequant matmul vs fp16 matmul, fp32)
        ref_fp16 = (x1.to(torch.float32) @ w16.to(torch.float32).t())[0]
        quant_rel = ((ref_deq - ref_fp16).abs() / (ref_fp16.abs() + 1e-3)).mean().item()

        for m in MS:
            x = torch.randn(m, K, dtype=torch.float16, device=dev) * 0.02
            t_fp16 = bench(lambda: rocm_unquantized_gemm_impl(x, w16, None),
                           iters=40, warmup=30)
            d_fp16 = deciles(t_fp16)
            best = None
            for kc in kchunk_options(K):
                t_i8 = bench(lambda: ops.dense_gemv_i8_m4_gfx906(q, scale, x, kc),
                             iters=40, warmup=30)
                d_i8 = deciles(t_i8)
                spd = d_fp16[2] / d_i8[2]
                print(f"  {name:<18} M={m} [{N},{K}] kc={kc} | "
                      f"fp16 med {d_fp16[2]*1000:8.1f} us | i8-cuda med "
                      f"{d_i8[2]*1000:8.1f} us | x{spd:.2f}")
                if best is None or d_i8[2] < best[2][2]:
                    best = (kc, spd, d_i8)
            # Triton options: w8a16_gemv (M=1, P2's GO path) and w8a16_gemm
            # (M>1) — the in-repo NH-2 scheme kernels.
            tri_line = ""
            if m == 1:
                t_tr = bench(lambda: w8a16.w8a16_gemv(q, scale, x[0].contiguous()),
                             iters=40, warmup=30)
                d_tr = deciles(t_tr)
                spd_tr = d_fp16[2] / d_tr[2]
                tri_line = f" | i8-triton-gemv med {d_tr[2]*1000:8.1f} us | x{spd_tr:.2f}"
                if d_tr[2] < best[2][2]:
                    best = ("triton", spd_tr, d_tr)
            else:
                t_tr = bench(lambda: w8a16.w8a16_gemm(q, scale, x),
                             iters=40, warmup=30)
                d_tr = deciles(t_tr)
                spd_tr = d_fp16[2] / d_tr[2]
                tri_line = f" | i8-triton-gemm med {d_tr[2]*1000:8.1f} us | x{spd_tr:.2f}"
                if d_tr[2] < best[2][2]:
                    best = ("triton", spd_tr, d_tr)
            kc_best, spd_best, d_best = best
            bw_gbs = N * K / (d_best[2] * 1e-3) / 1e9
            results.append((name, m, N, K, d_fp16, d_best, spd_best, kern_rel,
                            quant_rel, bw_gbs))
            print(f"  {'':<18} BEST={kc_best} x{spd_best:.2f} "
                  f"(i8 BW {bw_gbs:4.0f} GB/s){tri_line}")
        print(f"  {'':<18} kernel-vs-dequant rel.err = {kern_rel:.2e} | "
              f"int8-quant rel.err = {quant_rel:.2e}")
        del w16, q, scale

    smp._stop_evt.set()
    mclk_med = float(np.median(smp.samples)) if smp.samples else 0.0
    print(f"\nmclk median over measurement phase: {mclk_med:.0f} MHz "
          f"({len(smp.samples)} samples)")
    if mclk_med < 900:
        print("!! WARNING: mclk < 900 MHz — COLD-CLOCK ARTIFACT, numbers inflated. "
              "Re-run after sustained load.")

    # GO/NO-GO verdict (best variant per shape)
    print("\n## VERDICT")
    go = True
    for name, m, N, K, d_fp16, d_best, spd, kern_rel, quant_rel, bw in results:
        if kern_rel > 5e-3:
            print(f"  {name} M={m}: KERNEL BUG rel.err {kern_rel:.2e}")
            go = False
            continue
        if name in ("lm_head", "gdn_in_proj_qkvz"):
            ok = spd >= 1.5
            tag = "GO" if ok else "FAIL"
            print(f"  {name} M={m}: x{spd:.2f} (bar 1.50) -> {tag}")
            go = go and ok
        else:
            ok = spd >= 0.95
            tag = "ok" if ok else "SLOW"
            print(f"  {name} M={m}: x{spd:.2f} (bar 0.95) -> {tag}")
            go = go and ok
    print(f"\n  => {'GO for T1 kernel gate' if go else 'NO-GO'} "
          f"(mclk {mclk_med:.0f} MHz)")


if __name__ == "__main__":
    main()
