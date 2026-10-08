#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""C2-BM>=2: grouped/batch-decode MoE GEMM tile sweep (gfx906 W4A16).

The C2 sweep (docs/gfx906/DEVLOG-moe-c2v.md) covered the M=1 tiles only; the
BM>=2 grouped path — what `em = M*topk in (32, 512]` selects, i.e. B=4 under
MTP k=3 — was never swept on its axes. This bench drives the production op
(`moe_gptq_gemm_gfx906`, dispatcher-faithful) at the production shapes
(Qwen3.5-35B-A3B: E=256, topk=8, w13 N=1024/K=2048, w2 N=2048/K=512) and
reports per-call us + deciles, with the DVFS protocol of dvfs-mi50.md
(hot loop, block-of-10 event pairs, concurrent mclk sampling, hard gate).

Axes: block_size_m (the Python heuristic picks 1/4/8) and VLLM_GFX906_MOE_NPT
(the per-thread column count; the env now applies to every BM).

Usage:
  HIP_VISIBLE_DEVICES=0 .venv/bin/python -u bench_moe_bm_sweep.py
Env:
  BENCH_MS     csv of M values (default 4,8,16,32,64 -> em 32,64,128,256,512)
  BENCH_BMS    csv of block_size_m values (default 1,2,4,8)
  BENCH_BLOCKS blocks of 10 calls per (M, bm) point (default 20)
"""
import glob
import os
import re
import statistics
import subprocess
import threading
import time

import torch

dev = "cuda"
E, TOPK, GS = 256, 8, 128
N13, K13 = 1024, 2048
N2, K2 = 2048, 512
PEAK_F16 = 20e12
PEAK_BW = 1e12
BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))
BLOCK = 10
MCLK_RE = re.compile(r"mclk clock level.*?\((\d+)\s*Mhz\)", re.I)
MIN_MCLK = int(os.environ.get("BENCH_MIN_MCLK", "900"))


class MclkSampler:
    """Concurrent mclk sampler, sysfs-first.

    `rocm-smi` is unreliable on this box after a GPU reset (its python init has
    failed with MemoryError while the GPUs were fine), which silently produced
    "mclk 0 MHz" windows — so read the DPM table directly and keep rocm-smi as the
    fallback. The AMD DRM card for a given HIP ordinal is not fixed (GPU0 here is
    `card1`), so sample every card and report the maximum active clock.
    """

    GLOBS = "/sys/class/drm/card*/device/pp_dpm_mclk"

    def __init__(self):
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read_sysfs() -> int | None:
        best = None
        for path in glob.glob(MclkSampler.GLOBS):
            try:
                with open(path) as fh:
                    for line in fh:
                        if line.rstrip().endswith("*"):
                            m = re.search(r"(\d+)\s*Mhz", line, re.I)
                            if m:
                                v = int(m.group(1))
                                best = v if best is None else max(best, v)
            except OSError:
                continue
        return best

    def _run(self):
        while not self._stop.is_set():
            v = self._read_sysfs()
            if v is None:
                try:
                    out = subprocess.run(
                        ["rocm-smi", "--showclocks", "-d", "0"],
                        capture_output=True, text=True, timeout=10,
                    ).stdout
                    m = MCLK_RE.search(out)
                    v = int(m.group(1)) if m else None
                except Exception:
                    v = None
            if v is not None:
                self._samples.append(v)
            self._stop.wait(0.1)

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


def time_deciles(fn, tag, warmup=10, soak_s=3.0):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s0 = torch.cuda.Event(enable_timing=True)
    s1 = torch.cuda.Event(enable_timing=True)
    s0.record()
    for _ in range(20):
        fn()
    s1.record()
    torch.cuda.synchronize()
    t_call = max(s0.elapsed_time(s1) / 20 / 1000.0, 1e-6)
    inner = max(20, int(0.05 / t_call))
    t0 = time.time()
    while time.time() - t0 < soak_s:
        for _ in range(inner):
            fn()
        torch.cuda.synchronize()
    with MclkSampler() as sampler:
        per_call = []
        for _ in range(BLOCKS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(BLOCK):
                fn()
            e.record()
            torch.cuda.synchronize()
            per_call.append(s.elapsed_time(e) * 1e3 / BLOCK)
    gate = (
        "" if sampler.median >= MIN_MCLK
        else f"  <-- MCLK GATE FAIL ({sampler.median})"
    )
    per_call.sort()
    q = lambda p: per_call[min(len(per_call) - 1, int(p * len(per_call)))]  # noqa: E731
    print(
        f"    {tag}: median {q(0.5):8.1f} us  p05 {q(0.05):8.1f}  p95 {q(0.95):8.1f}  "
        f"(mclk {sampler.median} MHz){gate}",
        flush=True,
    )
    return q(0.5), sampler.median


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
        torch.randint(0, 16, (E, K13 // GS, N13), device=dev, dtype=torch.int32)
    )
    z2 = _pack_nib(
        torch.randint(0, 16, (E, K2 // GS, N2), device=dev, dtype=torch.int32)
    )
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

    ms = [int(v) for v in os.environ.get("BENCH_MS", "4,8,16,32,64").split(",")]
    bms = [int(v) for v in os.environ.get("BENCH_BMS", "1,2,4,8").split(",")]
    npt = os.environ.get("VLLM_GFX906_MOE_NPT", "unset")
    print(
        f"MoE BM sweep: VLLM_GFX906_MOE_NPT={npt} E={E} topk={TOPK} "
        f"w13 {N13}x{K13} w2 {N2}x{K2}",
        flush=True,
    )
    for M in ms:
        em = M * TOPK
        heuristic = 8 if em > 512 else (4 if em > 32 else 1)
        print(f"  M={M} em={em} (heuristic bm={heuristic})", flush=True)
        x = torch.randn(M, K13, device=dev, dtype=torch.float16)
        topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=dev)
        topk_w = torch.rand(M, TOPK, device=dev, dtype=torch.float16)
        inter_base = None
        for bm in bms:
            sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids, bm, E)
            c1 = torch.zeros(em, N13, device=dev, dtype=torch.float16)
            out = torch.zeros(M, N2, device=dev, dtype=torch.float16)
            empty_tw = torch.empty(0, dtype=torch.float32, device=dev)

            def g1():
                c1.zero_()
                ops.moe_gptq_gemm_gfx906(
                    x, c1, wq13, sc13, zp13, empty_tw, sorted_ids, expert_ids,
                    ntp, TOPK, bm, False, 0, 0,
                )

            g1()
            inter = (
                torch.nn.functional.silu(c1[:, : N13 // 2].float())
                * c1[:, N13 // 2:].float()
            ).half().contiguous()
            # Cross-check the tiles against each other (they must agree within the
            # fp16 CAS-accumulation noise band; a mismatch means the tile change
            # altered the math, not just the schedule).
            if inter_base is None:
                inter_base = inter.clone()
                note = "ref"
            else:
                rel = ((inter - inter_base).norm() / inter_base.norm()).item()
                note = f"rel vs bm={bms[0]}: {rel:.2e}"

            def g2():
                out.zero_()
                ops.moe_gptq_gemm_gfx906(
                    inter, out, wq2, sc2, zp2, topk_w.view(-1).float(),
                    sorted_ids, expert_ids, ntp, 1, bm, True, TOPK, 0,
                )

            t1, _ = time_deciles(g1, f"bm={bm} gemm1")
            t2, _ = time_deciles(g2, f"bm={bm} gemm2")
            print(
                f"      bm={bm}: total {t1 + t2:8.1f} us  "
                f"(w13 {t1:7.1f} + w2 {t2:7.1f})  {note}",
                flush=True,
            )


if __name__ == "__main__":
    torch.cuda.init()
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert "gfx906" in arch, f"not gfx906: {arch}"
    main()
