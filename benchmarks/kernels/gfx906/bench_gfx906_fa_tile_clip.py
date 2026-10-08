#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""FA M2 tile-clip A/B: PREFILL scan cost, both kernels, two shapes.

Measures the gfx906_fa prefill attention kernel at the Muse-Glimmer
geometry (Hq=32, Hkv=2, D=128, max_pos 131072) for a Sq=4096 chunk:

  shape A (windowed, mid-context): W=2048, chunk at the end of a
    131072-key sequence — the 39 Muse sliding-window layers at long
    context. Without M2 every q-tile scans [kv_start, seq_len) =
    ~6143 keys (384 k-tiles); with M2 ~= W keys (128 k-tiles).
  shape B (causal cap only, first chunk): window=0, q_abs=0, L=4096
    — the FIRST chunk of a full-attention model (any model): without
    the cap every q-tile scans all 4096 keys; with it tile t scans
    ~64(t+1) (~2x total FA work reduction).

Both via gfx906_fa.forward (LEGACY gather path) and
fa.forward_paged_direct (DIRECT_PAGED). A/B arm:
GFX906_FA_TILE_CLIP 0 vs 1 (bit-identical, unit-tested).

Usage (in-process, single GPU):
  HIP_VISIBLE_DEVICES=0 .venv/bin/python \
      benchmarks/kernels/gfx906/bench_gfx906_fa_tile_clip.py
"""
import math
import os
import time

import torch

dev = "cuda"
torch.manual_seed(0)

Hq = int(os.environ.get("BENCH_TQ_HQ", "32"))
Hkv = int(os.environ.get("BENCH_TQ_HKV", "2"))
D = int(os.environ.get("BENCH_TQ_D", "128"))
SQ = int(os.environ.get("BENCH_TQ_SQ", "4096"))
BLOCK = 16
BPR = (D // 32) * 34  # 136 uint8 per Q8 row (D=128)
SCALE = 1.0 / math.sqrt(D)
N_WARM = 3
N_ITERS = 8


def time_ms(fn):
    for _ in range(N_WARM):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_ITERS * 1e3


def bench_shape(fa, name, L, W, q_abs_val, kv_start_val):
    n_blocks = L // BLOCK
    kc = torch.zeros(n_blocks, BLOCK, Hkv, BPR,
                     dtype=torch.uint8, device=dev)
    kv = torch.zeros(n_blocks, 2, BLOCK, Hkv, D,
                     dtype=torch.float16, device=dev)
    K = torch.randn(L, Hkv, D, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, Hkv, D, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, Hkv, D)[:L].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(
        1, n_blocks)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    q_abs = torch.tensor([q_abs_val], dtype=torch.int32, device=dev)
    kv_start = (torch.tensor([kv_start_val], dtype=torch.int32,
                             device=dev) if kv_start_val is not None
                else None)
    q = torch.randn(1, Hq, SQ, D, device=dev, dtype=torch.float32) * 0.5

    sk_pad = (L + 31) // 32 * 32
    k_q8, v_b = fa.gather_paged_kv_q8(kc, vc, bt, sl, sk_pad)

    def fwd():
        return fa.forward(q, k_q8, v_b, SCALE, kv_max=sl,
                          q_abs_offset=q_abs, window=W, kv_start=kv_start)

    def direct():
        return fa.forward_paged_direct(q, kc, vc, bt, sl, SCALE, None,
                                       q_abs, W, kv_start)

    print(f"{name}: L={L} W={W} q_abs={q_abs_val} kv_start={kv_start_val}")
    for kname, fn in (("fwd (gather path)", fwd), ("direct (paged)", direct)):
        row = {}
        for clip in ("0", "1"):
            os.environ["GFX906_FA_TILE_CLIP"] = clip
            row[clip] = time_ms(fn)
        del os.environ["GFX906_FA_TILE_CLIP"]
        speedup = row["0"] / row["1"]
        print(f"  {kname:20s} clip=0 {row['0']:8.3f} ms | "
              f"clip=1 {row['1']:8.3f} ms | speedup {speedup:.2f}x")
    del kc, kv, K, V, staging, vc, bt, sl, q_abs, kv_start, q, k_q8, v_b
    torch.cuda.empty_cache()


def main():
    from vllm import _gfx906_fa_C as fa
    L_A = int(os.environ.get("BENCH_TQ_L", "131072"))
    W_A = int(os.environ.get("BENCH_TQ_W", "2048"))
    bench_shape(fa, "A windowed mid-context (M2 raise+cap)",
                L_A, W_A, L_A - SQ, max(0, L_A - SQ + 1 - W_A))
    bench_shape(fa, "B causal first-chunk (cap only, any model)",
                SQ, 0, 0, None)


if __name__ == "__main__":
    main()
