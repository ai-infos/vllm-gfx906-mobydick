#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""FA kernel track: flash_attn_tile_q8 micro-bench at B=1 decode.

Measures gfx906_fa.forward (the LEGACY serving path's attention kernel)
in isolation at the Qwen3.8-27B full-attention-layer decode shape:
Hq=12, Hkv=2, D=256 (per-rank under TP=2; GQA ratio 6), Sq sweep.

Sq matters a lot and is the point of the BENCH_FA_SQ knob:
  * Sq=2 — greedy single-query decode (Sq_pad=2). The kernel reads the KV
    once per query tile, so it runs bandwidth-bound (~100%+ of HBM; the
    "eff GB/s" column can exceed the read floor because Q8-compressed K
    halves the bytes vs the fp16 floor model — see below).
  * Sq=8 — spec-decode VERIFY shape (k=4 -> max_seqlen_q=5 -> Sq_pad=8,
    ncols1=8). The kernel re-reads the same KV rows for every query row,
    so it becomes COMPUTE-bound at ~46% of HBM. This is the production
    k=4 shape (PROFILE-k4-step-ledger.md) and where optimization headroom
    lives.

Launcher config is set via env (parsed once per process):
  BENCH_FA_HQ / BENCH_FA_HKV / BENCH_FA_D   shape (default 12/2/256)
  BENCH_FA_SQ                               Sq_pad (default 2; use 8 for k=4 verify)
  BENCH_FA_SK                               comma list of Sk (default long-context sweep)
  BENCH_FA_CHECK_MAX                        max Sk for the torch-reference check
  GFX906_FA_NC2                             GQA head-packing (1 legacy, 8)
  GFX906_FA_KVSPLIT                         gridDim.y KV-split (1 legacy)

Bandwidth note: "floor" = bytes / HBM_BW with bytes = n_tiles * Sk *
(q8-row + 2*D). At Sq=2 the effective bandwidth can read ABOVE that floor
because the Q8 K rows are ~half the fp16 byte width — treat ">100%" as
"at/beyond the read-bound limit", not a measurement error.

Usage:
  BENCH_FA_SQ=8 python3 -u bench_gfx906_fa_decode_qwen27.py
"""
import os
import sys

import torch

dev = "cuda"
torch.manual_seed(0)

Hq = int(os.environ.get("BENCH_FA_HQ", "12"))
Hkv = int(os.environ.get("BENCH_FA_HKV", "2"))
D = int(os.environ.get("BENCH_FA_D", "256"))
SQ = int(os.environ.get("BENCH_FA_SQ", "2"))  # decode Sq_pad
BPR = (D // 32) * 34  # 272 uint8 per Q8 row
SK_LIST = [int(x) for x in os.environ.get("BENCH_FA_SK", "8192,32768,65536,98304,122880").split(",")]
HBM_BW = 798e9  # P3-0 Q1: measured MI50 HBM read BW
SCALE = 1.0 / (D ** 0.5)
NC2 = int(os.environ.get("GFX906_FA_NC2", "1"))
YSPLIT = int(os.environ.get("GFX906_FA_KVSPLIT", "1"))


def time_us(fn, warmup=10, iters=50):
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


def ref_attention(q_f32, k_f16, v_f16, kv_max):
    """fp32 torch attention reference (per-seq kv_max cutoff)."""
    B = q_f32.shape[0]
    out = torch.empty(B, Hq, SQ, D, dtype=torch.float32, device=dev)
    g = Hq // k_f16.shape[1]
    for b in range(B):
        sk = int(kv_max[b])
        for h in range(Hq):
            kh = h // g
            qq = q_f32[b, h] * SCALE
            kk = k_f16[b, kh, :sk].float()
            vv = v_f16[b, kh, :sk].float()
            sc = qq @ kk.T
            sc = sc.softmax(dim=-1)
            out[b, h] = sc @ vv
    return out


def main():
    from vllm import _gfx906_fa_C as fa

    print(f"FA decode micro-bench: Hq={Hq} Hkv={Hkv} D={D} Sq={SQ} "
          f"NC2={NC2} KVSPLIT={YSPLIT}", flush=True)

    B = 1
    max_sk = max(SK_LIST)
    k16 = torch.randn(B, Hkv, max_sk, D, dtype=torch.float16, device=dev)
    v16 = torch.randn(B, Hkv, max_sk, D, dtype=torch.float16, device=dev)
    q32 = torch.randn(B, Hq, SQ, D, dtype=torch.float32, device=dev)

    rows = []
    worst_err = 0.0
    CHECK_MAX_SK = int(os.environ.get("BENCH_FA_CHECK_MAX", "8192"))
    for sk in SK_LIST:
        k16s = k16[:, :, :sk].contiguous()
        vs = v16[:, :, :sk].contiguous()
        kq = fa.quantize_q8_0(k16s)
        sl = torch.tensor([sk], dtype=torch.int32, device=dev)
        # C returns native BSHD; the reference is BHSD.
        o = fa.forward(q32, kq, vs, SCALE, kv_max=sl).permute(0, 2, 1, 3)
        if sk <= CHECK_MAX_SK:
            ref = ref_attention(q32, k16s, vs, sl)
            err = (o - ref).abs().max().item()
            worst_err = max(worst_err, err)
        else:
            err = float("nan")  # per-head torch ref is too slow at long Sk
        us = time_us(
            lambda f=fa.forward, k=kq, v=vs, s=sl:
            f(q32, k, v, SCALE, kv_max=s),
            warmup=max(3, 10 - sk // 16384),
            iters=max(5, 50 - sk // 4096),
        )
        # Each Q tile (ncols2=NC2 heads) reads ONE kv head row-set; tiles
        # may share a kv head across GQA groups, so total requested bytes
        # = n_tiles * sk * (q8-row + 2*D) (no Hkv factor).
        n_tiles = (Hq + NC2 - 1) // NC2
        bytes_ = n_tiles * sk * (BPR + 2 * D)
        bw = bytes_ / (us * 1e-6)
        floor_us = bytes_ / HBM_BW * 1e6
        rows.append((sk, us, bw, floor_us))
        print(f"Sk={sk:6d}: {us:8.1f} us   eff {bw/1e9:7.1f} GB/s "
              f"({100*bw/HBM_BW:4.1f}% of HBM)   floor {floor_us:6.1f} us "
              f"maxerr={err:.4f}", flush=True)

    import numpy as np
    b1 = [(sk, us) for (sk, us, _, _) in rows if sk >= max(512, SK_LIST[0])]
    sks = np.array([p[0] for p in b1], dtype=np.float64)
    uss = np.array([p[1] for p in b1], dtype=np.float64)
    slope, intercept = np.polyfit(sks, uss, 1)
    at_2176 = slope * (SK_LIST[-1] if len(SK_LIST)>1 else 2176) + intercept
    print(f"linear fit (512..13312): {slope*1e3:.2f} ns/token, "
          f"intercept {intercept:.1f} us; @Sk=2176: {at_2176:.1f} us; "
          f"worst maxerr={worst_err:.4f}", flush=True)
    if worst_err > 0.05:
        print("CORRECTNESS FAILED (maxerr > 0.05)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
