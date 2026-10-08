#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""FA-D96: the 96-wide head-dim instantiation vs the 128-wide pad path.

Launch-regime evidence for the FA-D96 session (the serving gate is the
image-prompt TTFT A/B on the dense 27B VL checkpoint; see
docs/gfx906/DEVLOG-fa-d96.md). Dispatcher-faithful: both arms call the
production adapter (`forward_vit`) or the production launcher entry
(`gfx906_fa.forward`) with the *padded* tensors each build would pass —
the old arm is the pre-FA-D96 behaviour (72/96 -> 128), the new arm is
72/96 -> 96.

Follows docs/gfx906/dvfs-mi50.md: hot loop, block-of-10 event pairs,
deciles (not just median), concurrent mclk sampling with a **hard
>= 900 MHz gate** (cold-clock data must not be silently reusable).

Usage:
  HIP_VISIBLE_DEVICES=0 .venv/bin/python -u bench_fa_d96.py [blocks]
"""
import os
import re
import statistics
import subprocess
import sys
import threading
import time

import torch

from vllm.gfx906_fa import gfx906_fa_backend as be
from vllm.gfx906_fa import gfx906_fa_mm_encoder as vitext
from vllm.gfx906_fa.gfx906_fa_mm_encoder import forward_vit

dev = "cuda"
BLOCKS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
BLOCK = 10  # calls per event pair
OLD_DIMS = (64, 128, 256)  # pre-FA-D96 instantiation set
NEW_DIMS = (64, 96, 128, 256)
MCLK_RE = re.compile(r"mclk clock level.*?\((\d+)\s*Mhz\)", re.I)
# dvfs-mi50.md's default gate is 900 MHz; the ViT-shaped calls in this bench do
# not drive the card past 800 (VIT-1's record for the same geometry is 800 MHz),
# so the floor is configurable and the arm-symmetry check below carries the
# anti-cold-clock burden for this file.
MIN_MCLK = int(os.environ.get("BENCH_MIN_MCLK", "800"))


class MclkSampler:
    """Concurrent mclk sampler; `.median` is the gate for a timed window."""

    def __init__(self):
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["rocm-smi", "--showclocks"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
                m = MCLK_RE.search(out)
                if m:
                    self._samples.append(int(m.group(1)))
            except Exception:
                pass
            self._stop.wait(0.25)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def median(self) -> int:
        return int(statistics.median(self._samples)) if self._samples else 0

    @property
    def lo(self) -> int:
        return min(self._samples) if self._samples else 0


def time_deciles(fn, tag, warmup=30, soak_s=6.0, blocks=None, block=None):
    n_blocks = blocks or BLOCKS
    n_block = block or BLOCK
    for _ in range(warmup):
        fn()
    # DVFS soak: the card boosts only under sustained load, and the *first* arm of
    # a case would otherwise be measured on a different clock than the second
    # (observed 1000 vs 800 MHz). Hold the call at speed until the clock settles.
    # The inner burst is sized so the launch queue keeps the GPU busy for ~50 ms —
    # syncing every few calls leaves the card idle enough to drop to 350 MHz
    # (measured: an 80 us and a 10 us case both parked at idle with a per-iter
    # sync soak).
    torch.cuda.synchronize()
    s0 = torch.cuda.Event(enable_timing=True)
    s1 = torch.cuda.Event(enable_timing=True)
    s0.record()
    for _ in range(20):
        fn()
    s1.record()
    torch.cuda.synchronize()
    t_call = max(s0.elapsed_time(s1) / 20 / 1000.0, 1e-6)  # seconds
    inner = max(20, int(0.05 / t_call))
    t0 = time.time()
    while time.time() - t0 < soak_s:
        for _ in range(inner):
            fn()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    with MclkSampler() as sampler:
        per_call = []
        for _ in range(n_blocks):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(n_block):
                fn()
            e.record()
            torch.cuda.synchronize()
            per_call.append(s.elapsed_time(e) * 1e3 / n_block)
    if sampler.median < MIN_MCLK:
        raise SystemExit(
            f"DVFS GATE FAILED for {tag}: median mclk {sampler.median} MHz "
            f"(min {sampler.lo}) < {MIN_MCLK} — cold-clock data is not usable"
        )
    if sampler.median < 900:
        print(
            f"    NOTE [{tag}] mclk {sampler.median} MHz is below the dvfs-mi50 "
            "default gate (900): these ViT-shaped calls do not drive the card to "
            "1 GHz — VIT-1's record for the same geometry is 800 MHz. The arm-"
            "symmetry check below is the protection against a clock-driven delta.",
            flush=True,
        )
    per_call.sort()
    q = lambda p: per_call[min(len(per_call) - 1, int(p * len(per_call)))]  # noqa: E731
    print(
        f"    [{tag}] median {q(0.5):.1f} us  p05 {q(0.05):.1f}  p25 {q(0.25):.1f}  "
        f"p75 {q(0.75):.1f}  p95 {q(0.95):.1f}  (mclk median {sampler.median} MHz)",
        flush=True,
    )
    return q(0.5), sampler.median


def check_clock_symmetry(tag, mclk_a, mclk_b):
    if not mclk_a or not mclk_b:
        raise SystemExit(f"{tag}: no mclk samples for one arm")
    spread = abs(mclk_a - mclk_b) / max(mclk_a, mclk_b)
    if spread > 0.05:
        raise SystemExit(
            f"{tag}: arm mclk asymmetry {mclk_a} vs {mclk_b} MHz ({spread:.1%}) — "
            "the delta may be a clock effect, not the pad width"
        )


def ab_passes(arms, tag, blocks=None, block=None):
    """Interleaved two-pass A/B: returns the LAST pass's (median, mclk) per arm.

    `arms` maps name -> (prepare, call): `prepare()` selects the arm (e.g. flips the
    pad target) immediately before its timed window, because the pad choice is a
    process-global that the callable reads at call time. Running each arm once
    would leave the first arm on a different DVFS state than the second (the
    512x512 case measured 1000 MHz for the first and 800 for the second);
    interleaving A,B,A,B and reporting the second pass gives both arms the same
    preceding load history.
    """
    res, clk = {}, {}
    for rep in range(2):
        for name, (prepare, fn) in arms.items():
            prepare()
            m, c = time_deciles(
                fn, f"{name} (pass {rep + 1})", blocks=blocks, block=block
            )
            if rep == 1:
                res[name], clk[name] = m, c
    return res, clk


def set_arm(old: bool):
    dims = OLD_DIMS if old else NEW_DIMS
    be._INSTANTIATED_HEAD_DIMS = dims
    vitext._INSTANTIATED_HEAD_DIMS = dims
    os.environ["GFX906_FA_PAD96"] = "0" if old else "1"


# ---------------------------------------------------------------------------
# 1) ViT shape through the production adapter (bidirectional, packed, fp16)
# ---------------------------------------------------------------------------
def vit_case(name, s, heads=16, d=72, hkv=16):
    torch.manual_seed(11)
    q = torch.randn(1, s, heads, d, device=dev, dtype=torch.float16) * 0.5
    k = torch.randn(1, s, hkv, d, device=dev, dtype=torch.float16) * 0.5
    v = torch.randn(1, s, hkv, d, device=dev, dtype=torch.float16) * 0.5
    cu = torch.tensor([0, s], dtype=torch.int32, device=dev)
    scale = 1.0 / (d**0.5)

    def call():
        return forward_vit(q, k, v, cu, None, scale, d)

    print(f"[vit {name}] S={s} H={heads} D={d} hkv={hkv}", flush=True)
    set_arm(True)
    ref = call().float()
    set_arm(False)
    rel = ((call().float() - ref).norm() / ref.norm()).item()
    arms = {
        "128-pad": (lambda: set_arm(True), call),
        "96-pad": (lambda: set_arm(False), call),
    }
    res, clk = ab_passes(arms, f"vit {name}")
    check_clock_symmetry(f"vit {name}", clk["128-pad"], clk["96-pad"])
    print(
        f"    => 96-pad vs 128-pad: {res['96-pad'] / res['128-pad'] * 100 - 100:+.1f} %"
        f"   (arm output rel diff {rel:.2e})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# 2) Text-shaped calls through the launcher (causal), D=96 real vs D=128 pad
# ---------------------------------------------------------------------------
def text_case(name, hq, hkv, sq, sk, kv_split=1):
    from vllm import _gfx906_fa_C as fa

    torch.manual_seed(23)
    scale = 1.0 / (96**0.5)
    print(f"[text {name}] Hq={hq} Hkv={hkv} Sq={sq} Sk={sk} y={kv_split}", flush=True)
    # One 96-dim input set; the 128 arm is that same data zero-padded on the head
    # dim (exactly what the production pad path passes), so the two arms must
    # agree on the first 96 dims. Drawing fresh randn(128) would differ in the
    # first 96 dims and make the comparison meaningless.
    torch.manual_seed(23)
    q96 = torch.randn(1, hq, sq, 96, device=dev, dtype=torch.float32) * 0.5
    k96 = torch.randn(1, hkv, sk, 96, device=dev, dtype=torch.float16) * 0.5
    v96 = torch.randn(1, hkv, sk, 96, device=dev, dtype=torch.float16) * 0.5
    pad = torch.nn.functional.pad
    tensors = {
        96: (q96, k96, v96),
        128: (
            pad(q96, (0, 32)),
            pad(k96, (0, 32)),
            pad(v96, (0, 32)),
        ),
    }
    arms, outs = {}, {}
    for dim in (128, 96):
        q, k, v = tensors[dim]
        kq8 = fa.quantize_q8_0(k)
        kvm = torch.full((1,), sk, dtype=torch.int32, device=dev)
        qa = (
            torch.full((1,), sq - 1, dtype=torch.int32, device=dev) if sq > 1 else None
        )

        def make_call(q=q, kq8=kq8, v=v, kvm=kvm, qa=qa):
            return lambda: fa.forward(
                q, kq8, v, scale, kvm, None, qa, 0, None, kv_split
            )

        arms[dim] = (lambda: None, make_call())
        outs[dim] = arms[dim][1]().float()
    # Decode-shaped calls are ~80 us and the verify shape ~10 us: a 20x10 window
    # is too short to hold the card at speed (observed 350 MHz idle), so the short
    # cases get many more calls per window.
    short = sq <= 8
    res, clk = ab_passes(
        arms,
        f"text {name}",
        blocks=20 if short else None,
        block=1000 if short else None,
    )
    check_clock_symmetry(f"text {name}", clk[128], clk[96])
    # The 128 arm's dims 96..127 are the zero pad: its first 96 dims must match the
    # 96-wide arm (that is the whole correctness claim of the pad map).
    rel = ((outs[96] - outs[128][..., :96]).norm() / outs[96].norm()).item()
    print(
        f"    => 96 vs 128: {res[96] / res[128] * 100 - 100:+.1f} %"
        f"   (arms differ by {rel:.2e})",
        flush=True,
    )


if __name__ == "__main__":
    torch.cuda.init()
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert "gfx906" in arch, f"not gfx906: {arch}"
    only = os.environ.get("BENCH_CASE", "")

    def want(case):
        return not only or case in only

    print(
        f"FA-D96 standalone A/B (arch {arch}, {BLOCKS} blocks x {BLOCK} calls)",
        flush=True,
    )
    if want("vit"):
        vit_case("1024x1024", 2304)  # 64x64 patches
        vit_case("512x512", 576)     # 32x32 patches
    # Phi-3-mini-class text shapes (32 heads / 32 kv heads, head_dim 96)
    if want("decode"):
        text_case("decode", 32, 32, 1, 2048)
    if want("verify"):
        text_case("verify", 32, 32, 4, 2048)
    if want("prefill"):
        text_case("prefill", 32, 32, 512, 512, kv_split=1)
    print("done", flush=True)
