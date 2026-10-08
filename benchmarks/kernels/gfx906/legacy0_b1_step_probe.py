#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Item #1: B=1 decode step probe — LEGACY=1 vs LEGACY=0 paths.

Times the three B=1 decode sequences the gfx906 backend can run
(in-process, eager — launch-regime evidence; the serving A/B is the
gate), per (Sk, geometry):

  A  LEGACY=1 (production default):
     gather_paged_kv_quant_persistent (paged fp16 K -> in-kernel Q8
     quantize + V gather) + fa.forward (kv_split=16 default)
  B  LEGACY=0 (today's dispatch, DIRECT_PAGED_Q8=0):
     gather_paged_kv_q8 (pre-quantized Q8 side buffer, aliased strided
     view of the fp16 K half) + fa.forward (kv_split=16 default)
  C  LEGACY=0 (M5-era / opt-in DIRECT_PAGED=1):
     fa.forward_paged_direct only (internal kv_split=8 default)

Reports gather-only / FA-only / total ns per step plus the A-vs-B and
A-vs-C deltas. Fixed seed; correctness is NOT checked here (the suite
covers it) — this is a timing decomposition.

Usage:
  HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
      .venv/bin/python benchmarks/kernels/gfx906/legacy0_b1_step_probe.py
"""
import math

WARMUP = 10
ITERS = 50
SKS = [2048, 16384, 32768]
GEOMS = [(256, 16, 2), (128, 32, 2)]   # (D, Hq, Hkv): Qwen3.8 + Muse
BLOCK = 16


def time_fn(fn, warmup=WARMUP, iters=ITERS):
    import torch

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
    return s.elapsed_time(e) * 1e3 / iters  # us/iter


def run_geometry(d, hq, hkv):
    import torch

    from vllm import _gfx906_fa_C as fa

    dev = "cuda"
    torch.manual_seed(20260829)
    scale = 1.0 / math.sqrt(d)
    bpr = (d // 32) * 34

    print(f"\n=== geometry D={d} Hq={hq} Hkv={hkv} ===", flush=True)
    for sk in SKS:
        n_blocks = sk // BLOCK
        k16 = torch.randn(n_blocks, BLOCK, hkv, d, device=dev,
                          dtype=torch.float16) * 0.5
        v16 = torch.randn(n_blocks, BLOCK, hkv, d, device=dev,
                          dtype=torch.float16) * 0.5
        # LEGACY=0 side buffer: strided uint8 view of the fp16 K half
        # (head stride 2D; only the leading bpr bytes of each row are
        # valid Q8). Fill it from a full quantization of the fp16 rows.
        full_q8 = fa.quantize_q8_0(k16)
        alias_q8 = k16.view(torch.uint8)[:, :, :, :bpr]
        alias_q8.copy_(full_q8)
        bt = torch.arange(n_blocks, dtype=torch.int32, device=dev
                          ).view(1, -1)
        sl = torch.tensor([sk], dtype=torch.int32, device=dev)
        q32 = (torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32)
               * 0.5)
        k_exact = torch.empty(1, hkv, sk, bpr, dtype=torch.uint8,
                              device=dev)
        v_exact = torch.empty(1, hkv, sk, d, dtype=torch.float16,
                              device=dev)

        ga = lambda: fa.gather_paged_kv_quant_persistent(  # noqa: E731
            k16, v16, bt, sl, sk, k_out=k_exact, v_out=v_exact)
        gb = lambda: fa.gather_paged_kv_q8(  # noqa: E731
            alias_q8, v16, bt, sl, sk, k_out=k_exact, v_out=v_exact)
        fa_fwd = lambda: fa.forward(q32, k_exact, v_exact, scale,  # noqa: E731
                                    kv_max=sl)
        fc = lambda: fa.forward_paged_direct(  # noqa: E731
            q32, alias_q8, v16, bt, sl, scale, None, None)

        a_g = time_fn(ga)
        a_f = time_fn(fa_fwd)
        b_g = time_fn(gb)
        c_t = time_fn(fc)
        a_tot, b_tot = a_g + a_f, b_g + a_f
        print(f"Sk={sk}: "
              f"A(LEG1) gather {a_g:8.1f} us + FA {a_f:8.1f} us "
              f"= {a_tot:8.1f} us | "
              f"B(LEG0) gather {b_g:8.1f} us + FA {a_f:8.1f} us "
              f"= {b_tot:8.1f} us | "
              f"C(direct) {c_t:8.1f} us | "
              f"B-A {100 * (b_tot / a_tot - 1):+6.2f} %  "
              f"C-A {100 * (c_t / a_tot - 1):+6.2f} %  "
              f"(ns/tok: A {a_tot * 1e3 / sk:5.2f} "
              f"B {b_tot * 1e3 / sk:5.2f} C {c_t * 1e3 / sk:5.2f})",
              flush=True)
        del k16, v16, full_q8, alias_q8, bt, sl, q32, k_exact, v_exact
        torch.cuda.empty_cache()


def main():
    print("B=1 decode step probe (LEGACY=1 vs LEGACY=0 paths; "
          "eager, in-process — launch-regime evidence)", flush=True)
    for d, hq, hkv in GEOMS:
        run_geometry(d, hq, hkv)


if __name__ == "__main__":
    main()
