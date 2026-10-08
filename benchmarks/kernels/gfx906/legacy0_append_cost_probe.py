#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Append-path cost probe: the LEGACY=0 per-step per-layer KV writes.

LEGACY=1 decode step, per full-attn layer: triton_reshape_and_cache_flash
only. LEGACY=0 adds, per layer: slot_mapping int64 cast (if needed) +
gfx906_fa.reshape_and_cache_q8 (Q8 bytes into the aliased strided view).
Times both sequences at the B=1 decode shape (1 token) for the Qwen3.8
geometry (16 full-attn layers, Hkv=4, D=256, block 16) — launch-regime
evidence for the ~1.6 ms/step serving delta (serving A/B is the gate;
see DEVLOG-fa-legacy0-b1-decode).
"""
import torch

from vllm import _gfx906_fa_C as fa
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)

N_LAYERS = 16
HKV, D, BLOCK, N_BLOCKS = 4, 256, 16, 256
BPR = (D // 32) * 34
WARMUP, ITERS = 10, 100


def time_fn(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / ITERS  # us


def main():
    dev = "cuda"
    torch.manual_seed(20260829)
    kc = torch.zeros(N_BLOCKS, 2, BLOCK, HKV, D, dtype=torch.float16,
                     device=dev)
    k_half = kc[:, 0]
    kc_q8 = k_half.view(torch.uint8)[:, :, :, :BPR]
    key = torch.randn(1, HKV, D, dtype=torch.float16, device=dev) * 0.5
    value = torch.randn(1, HKV, D, dtype=torch.float16, device=dev) * 0.5
    slot = torch.tensor([7], dtype=torch.int64, device=dev)

    def legacy1_layer():
        triton_reshape_and_cache_flash(
            key, value, k_half, kc[:, 1], slot, "auto", None, None)

    def legacy0_layer():
        legacy1_layer()
        fa.reshape_and_cache_q8(key, slot, kc_q8)

    t1 = time_fn(legacy1_layer)
    t0 = time_fn(legacy0_layer)
    # isolate the Q8 write (and the int64 cast is a no-op here: slot is
    # already int64 — the serving cast cost is a separate small kernel)
    tq8 = time_fn(lambda: fa.reshape_and_cache_q8(key, slot, kc_q8))
    print(f"APPROBE per-layer @B=1 D=256 Hkv=4: "
          f"LEGACY=1 write {t1:7.1f} us | LEGACY=0 write+q8 {t0:7.1f} us "
          f"(q8 alone {tq8:5.1f} us) | x{N_LAYERS}/step: "
          f"LEGACY=1 {t1 * N_LAYERS:6.1f} us | LEGACY=0 {t0 * N_LAYERS:6.1f} us"
          f" | delta {100 * (t0 / t1 - 1):+5.1f} %/layer, "
          f"{(t0 - t1) * N_LAYERS:5.1f} us/step eager")


if __name__ == "__main__":
    main()
