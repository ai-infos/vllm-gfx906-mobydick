#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""M4: long-context KV-split accuracy probe (roadmap M4 / qwen review #4a).

Closes the claim that the production KV-split defaults are safe at long
context. The split-K FA kernels store per-split partials (fp32, unscaled
O + (m, l) meta) merged by the fp32 log-sum-exp combine; the M4
concern was that the fp16 P·V accumulators (v_pk_fma_f16) and the
unscaled partial magnitude (grows with keys-per-slice x |V|) degrade
accuracy as the context length grows. Production defaults under test:

  gather path   (production B=1 decode):  kv_split = 16
  direct-paged  (production B>=2 decode): kv_split = clamp(16/B, 2, 8)
                                          -> 8 at B=1

Arms (each in a fresh subprocess; the gather kv_split is a C++ static
parsed on first forward call): per (sk, geometry) —
  g16 : fa.forward,          kv_split=16 (production B=1 default)
  g1  : fa.forward,          kv_split=1  (no-split baseline)
  p8  : fa.forward_paged_direct, B=1 -> kv_split=8 (paged default)
vs an fp32 torch reference (same inputs, fixed seed across arms).

Gates (printed as M4-GATE):
  G1  rel_ref < 5e-2 for every arm (the suite tolerance family)
  G2  ||g16 - g1|| / ||g1|| <= max(2 * rel_ref(g1), 1e-2) for every
      (sk, geometry) — the split machinery adds no length-growing error
      beyond the no-split arm's own fp16-accumulation error.

Usage:
  HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
      .venv/bin/python benchmarks/kernels/gfx906/m4_splitk_accuracy_probe.py
"""
import json
import math
import os
import subprocess
import sys
import tempfile

SEED = 20260829
REL_TOL = 5e-2
GEOMS = [(256, 16, 2), (128, 32, 2)]   # (D, Hq, Hkv): existing + Muse
SKS = [16384, 32768]
BLOCK = 16


def run_arm(mode, sk, split, d, hq, hkv, out_pt=None):
    """Run one arm in THIS process; return the arm's record dict."""
    import torch

    from vllm import _gfx906_fa_C as fa

    dev = "cuda"
    torch.manual_seed(SEED)
    scale = 1.0 / math.sqrt(d)

    if mode == "gather":
        k16 = torch.randn(1, hkv, sk, d, device=dev,
                          dtype=torch.float16) * 0.5
        v16 = torch.randn(1, hkv, sk, d, device=dev,
                          dtype=torch.float16) * 0.5
        q32 = (torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32)
               * 0.5)
        kq = fa.quantize_q8_0(k16)
        sl = torch.tensor([sk], dtype=torch.int32, device=dev)
        out = fa.forward(q32, kq, v16, scale, kv_max=sl)[0, 0]  # [Hq, D]
        k, v = k16[0].float(), v16[0].float()
    else:  # paged
        n_blocks = sk // BLOCK
        kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                         dtype=torch.uint8, device=dev)
        kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                         dtype=torch.float16, device=dev)
        kf = torch.randn(sk, hkv, d, device=dev, dtype=torch.float16) * 0.5
        vf = torch.randn(sk, hkv, d, device=dev, dtype=torch.float16) * 0.5
        slot = torch.arange(sk, dtype=torch.int64, device=dev)
        fa.reshape_and_cache_q8(kf, slot, kc)
        staging = torch.zeros_like(kv[:, 1])
        staging.view(-1, hkv, d)[:sk].copy_(vf)
        kv[:, 1].copy_(staging)
        q32 = (torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32)
               * 0.5)
        bt = torch.arange(n_blocks, dtype=torch.int32, device=dev
                          ).view(1, -1)
        sl = torch.tensor([sk], dtype=torch.int32, device=dev)
        vc = kv.unbind(1)[1]  # production layout: unbind(1), non-contig
        out = fa.forward_paged_direct(
            q32, kc, vc, bt, sl, scale, None, None)[0, 0]  # [Hq, D]
        k, v = kf.float().permute(1, 0, 2), vf.float().permute(1, 0, 2)

    # fp32 reference
    g = hq // hkv
    qg = q32[0, :, 0].view(hkv, g, d)
    s = torch.einsum("gjd,gld->gjl", qg, k) * scale
    ref = torch.einsum(
        "gjl,gld->gjd", torch.softmax(s, -1), v).reshape(hq, d)
    rel = ((out - ref).norm() / ref.norm()).item()
    rec = {"mode": mode, "sk": sk, "split": split, "d": d,
           "rel_ref": rel}
    if out_pt:
        torch.save(out.cpu(), out_pt)
    return rec


def spawn_arm(mode, sk, split, d, hq, hkv, tmpdir):
    out_pt = os.path.join(tmpdir, f"{mode}_{sk}_{split}_{d}.pt") \
        if mode == "gather" else None
    args = [sys.executable, __file__, mode, str(sk), str(split),
            str(d), str(hq), str(hkv)] + ([out_pt] if out_pt else [])
    env = {**os.environ, "GFX906_FA_KVSPLIT": str(split)}
    r = subprocess.run(args, env=env, capture_output=True, text=True,
                       timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"arm {args[1:]} failed:\n{r.stderr[-2000:]}")
    rec = json.loads(r.stdout.strip().splitlines()[-1])
    if out_pt:
        rec["out_pt"] = out_pt
    return rec


def main():
    import torch

    print("M4: long-context KV-split accuracy probe "
          f"(seed {SEED}, rel_tol {REL_TOL})", flush=True)
    tmpdir = tempfile.mkdtemp(prefix="m4_splitk_")
    recs = []
    for d, hq, hkv in GEOMS:
        for sk in SKS:
            for mode, split in (("gather", 16), ("gather", 1),
                                ("paged", 8)):
                recs.append(spawn_arm(mode, sk, split, d, hq, hkv, tmpdir))
                print(f"M4: {recs[-1]}", flush=True)

    ok_g1 = all(r["rel_ref"] < REL_TOL for r in recs)
    deltas = []
    for d, hq, hkv in GEOMS:
        for sk in SKS:
            g16 = next(r for r in recs
                       if r["mode"] == "gather" and r["sk"] == sk
                       and r["split"] == 16 and r["d"] == d)
            g1 = next(r for r in recs
                      if r["mode"] == "gather" and r["sk"] == sk
                      and r["split"] == 1 and r["d"] == d)
            o16 = torch.load(g16["out_pt"])
            o1 = torch.load(g1["out_pt"])
            delta = ((o16 - o1).norm() / o1.norm()).item()
            bound = max(2 * g1["rel_ref"], 1e-2)
            deltas.append({"d": d, "sk": sk, "delta": delta,
                           "bound": bound,
                           "ok": delta <= bound})
            print(f"M4: split-delta D={d} sk={sk}: "
                  f"{delta:.2e} vs bound {bound:.2e} "
                  f"({'ok' if delta <= bound else 'EXCEEDED'})",
                  flush=True)
    ok_g2 = all(x["ok"] for x in deltas)
    print(f"M4-GATE: G1(all rel_ref < {REL_TOL})={'PASS' if ok_g1 else 'FAIL'}"
          f" G2(split delta within bound)={'PASS' if ok_g2 else 'FAIL'}"
          f" => {'PASS' if ok_g1 and ok_g2 else 'FAIL'}")
    sys.exit(0 if ok_g1 and ok_g2 else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        (mode, sk, split, d, hq, hkv) = (
            sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
            int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
        rec = run_arm(mode, sk, split, d, hq, hkv,
                      sys.argv[7] if len(sys.argv) > 7 else None)
        print(json.dumps(rec))
    else:
        main()
