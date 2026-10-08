#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""P2 — int8-weight M=1 GEMV at our fp16 weight mass (probe).

Decides roadmap item I1 of docs/gfx906/int8-investigation-qwen.md §3.1: our
Qwen checkpoints ship a large slice of their per-decode-step weight traffic
in fp16 (lm_head, FA q/k/v, GDN in_proj_qkv/z/out, shared expert, layer 0 —
~5.72 GB/step dense, ~3.8 GB/step MoE), those GEMVs are measured at 98-101 %
of the HBM floor, and int8 weights halve the bytes. This measures whether an
int8-weight GEMV actually converts that into wall time on gfx906 — with the
dequant ALU (int8 -> fp32 + per-row/per-group scale) charged, since that is
the thing that could eat the win. No vLLM import.

Modes measured per shape (M=1):
  f16-torch   torch.mv fp16            (hipBLAS, realistic baseline)
  f16-triton  Triton fp16 row-GEMV     (same codegen family as the int8 one)
  i8-row      Triton int8 + per-output-row fp16 scale
  i8-group    Triton int8 + fp16 scale per 128 along K
  i8-dot      Triton int8 x int8 via tl.dot (v_dot4), x quantized per token
              (needs M>=16 padding -> 16x the MACs; only viable because the
              loop is byte-bound). Skippable with INT8_PROBE_SKIP_DOT=1.

GO iff i8-row >= 1.7x f16-triton at [248320, 5120] and >= 1.6x at
[8192, 2048] (the two shapes that dominate the fp16 mass).

RESULT (2026-08-31, MI50): GO on the probe gate -- 1.93x at lm_head (741 GB/s
= 93% of the int8 floor; 1.81x vs the recorded 3114-3193 us production fp16
floor), 2.00-2.42x at the other floor-bound shapes, never below 0.96x, and
per-128-group scales cost +28% over per-channel (use per-channel; scale after
the reduction). The serving A/B remains the shipping gate. Record:
DEVLOG-int8-transfer.md.

Usage:
  HIP_VISIBLE_DEVICES=0 .venv/bin/python \
      benchmarks/kernels/gfx906/int8_gemv_probe.py
Env: INT8_PROBE_QUICK=1, INT8_PROBE_SKIP_DOT=1, INT8_PROBE_BW=798 (floor GB/s)
Peak-memory note: the lm_head leg allocates ~2.5 GB (fp16) + 1.27 GB (int8)
+ 1.27 GB (dot-copy) at once; run with >= 6 GiB free.
"""

import os
import sys

import torch
import triton
import triton.language as tl

try:
    from triton.testing import do_bench
except Exception:  # pragma: no cover
    from triton import do_bench  # type: ignore

FLOOR_GBS = float(os.environ.get("INT8_PROBE_BW", "798"))
GROUP = 128

# (name, N, K, per-step count dense-27B, per-step count moe-35B)
# Taken from the checkpoints themselves: modules_to_not_convert decides what
# is fp16 at all. Dense AWQ config leaves q/k/v + layers.0.* + mtp.* +
# in_proj_a/b unquantized; the MoE config leaves *all* linear_attn +
# self_attn + shared_expert + mlp.gate + layers.0.* + mtp.* unquantized
# (only layers 1+ routed experts are int4). Weights are stored BF16; same
# 2 B/weight as FP16 for a bandwidth-bound GEMV.
SHAPES = [
    # ---- dense Qwen3.5-27B-AWQ, per decode token (no spec decode) -------
    ("lm_head dense", 248320, 5120, 1, 0),
    ("fa q_proj dense", 12288, 5120, 16, 0),
    ("fa k_proj dense", 1024, 5120, 16, 0),
    ("fa v_proj dense", 1024, 5120, 16, 0),
    ("L0 gdn in_proj_qkv dense", 10240, 5120, 1, 0),
    ("L0 gdn in_proj_z dense", 6144, 5120, 1, 0),
    ("L0 gdn out_proj dense", 5120, 6144, 1, 0),
    ("L0 mlp gate/up dense", 17408, 5120, 2, 0),
    ("L0 mlp down dense", 5120, 17408, 1, 0),
    # ---- dense mtp2 draft layer (one extra pass per step when on) -------
    ("mtp q_proj dense", 12288, 5120, 1, 0),
    ("mtp o_proj dense", 5120, 6144, 1, 0),
    ("mtp fc dense", 5120, 10240, 1, 0),
    ("mtp mlp gate/up dense", 17408, 5120, 2, 0),
    ("mtp mlp down dense", 5120, 17408, 1, 0),
    # ---- MoE Qwen3.5-35B-A3B-AWQ, per decode token ---------------------
    ("lm_head moe", 248320, 2048, 0, 1),
    ("gdn in_proj_qkv moe", 8192, 2048, 0, 30),
    ("gdn in_proj_z moe", 4096, 2048, 0, 30),
    ("gdn out_proj moe", 2048, 4096, 0, 30),
    ("fa q_proj moe", 8192, 2048, 0, 10),
    ("fa k_proj moe", 512, 2048, 0, 10),
    ("fa v_proj moe", 512, 2048, 0, 10),
    ("fa o_proj moe", 2048, 8192, 0, 10),
    ("shared gate_up moe", 512, 2048, 0, 80),
    ("shared down moe", 2048, 512, 0, 40),
    # layer-0 routed experts + the mtp draft expert set ship BF16 too; at
    # top-k=8/256 that is ~8 experts x (gate,up,down) x 2 sets per step.
    ("bf16 experts gate/up moe", 512, 2048, 0, 16),
    ("bf16 experts down moe", 2048, 512, 0, 8),
]


def _log(*a):
    print(*a, flush=True)


@triton.jit
def k_gemv_f16(
    x_ptr, w_ptr, o_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr
):
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    rows = pn * BN + tl.arange(0, BN)
    rmask = rows < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    per = K // SPLIT
    for k0 in range(pk * per, (pk + 1) * per, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < (pk + 1) * per
        x = tl.load(x_ptr + ks, mask=kmask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + rows[:, None] * K + ks[None, :],
            mask=rmask[:, None] & kmask[None, :],
            other=0.0,
        )
        acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    if SPLIT == 1:
        tl.store(o_ptr + rows, acc, mask=rmask)
    else:
        tl.atomic_add(o_ptr + rows, acc, mask=rmask)


@triton.jit
def k_gemv_i8_row(
    x_ptr,
    w_ptr,
    s_ptr,
    o_ptr,
    N,
    K,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    rows = pn * BN + tl.arange(0, BN)
    rmask = rows < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    per = K // SPLIT
    for k0 in range(pk * per, (pk + 1) * per, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < (pk + 1) * per
        x = tl.load(x_ptr + ks, mask=kmask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + rows[:, None] * K + ks[None, :],
            mask=rmask[:, None] & kmask[None, :],
            other=0,
        )
        acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    s = tl.load(s_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    acc = acc * s
    if SPLIT == 1:
        tl.store(o_ptr + rows, acc, mask=rmask)
    else:
        tl.atomic_add(o_ptr + rows, acc, mask=rmask)


@triton.jit
def k_gemv_i8_group(
    x_ptr,
    w_ptr,
    s_ptr,
    o_ptr,
    N,
    K,
    GRP: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """Scale per (row, K/GRP group): the accumulate resets every group, so
    one descale per group is exact (matches AWQ's group granularity)."""
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    rows = pn * BN + tl.arange(0, BN)
    rmask = rows < N
    ng = K // GRP
    per_g = ng // SPLIT
    out = tl.zeros((BN,), dtype=tl.float32)
    cols = tl.arange(0, GRP)
    for g in range(pk * per_g, (pk + 1) * per_g):
        ks = g * GRP + cols
        x = tl.load(x_ptr + ks).to(tl.float32)
        w = tl.load(
            w_ptr + rows[:, None] * K + ks[None, :], mask=rmask[:, None], other=0
        )
        part = tl.sum(w.to(tl.float32) * x[None, :], axis=1)
        s = tl.load(s_ptr + rows * ng + g, mask=rmask, other=0.0).to(tl.float32)
        out += part * s
    if SPLIT == 1:
        tl.store(o_ptr + rows, out, mask=rmask)
    else:
        tl.atomic_add(o_ptr + rows, out, mask=rmask)


@triton.jit
def k_quant_x(x_ptr, q_ptr, s_ptr, K, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < K, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    s = tl.where(amax > 0, amax / 127.0, 1.0)
    tl.store(s_ptr, s)
    q = tl.floor(x * (1.0 / s) + 0.5)
    tl.store(
        q_ptr + offs, tl.minimum(tl.maximum(q, -127), 127).to(tl.int8), mask=offs < K
    )


@triton.jit
def k_gemv_i8_dot(
    xq_ptr,
    w_ptr,
    xs_ptr,
    s_ptr,
    o_ptr,
    N,
    K,
    GRP: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """int8 x int8 -> int32 through tl.dot (v_dot4) with M padded to 16."""
    pn = tl.program_id(0)
    rows = pn * BN + tl.arange(0, BN)
    rmask = rows < N
    xs = tl.load(xs_ptr)
    acc = tl.zeros((16, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        a = tl.load(xq_ptr + ks, mask=ks < K, other=0)
        at = tl.reshape(tl.broadcast_to(a[None, :], (16, BK)), (16, BK))
        w = tl.load(
            w_ptr + ks[:, None] * N + rows[None, :], mask=(rows[None, :] < N), other=0
        )
        p = tl.dot(at, w, out_dtype=tl.int32).to(tl.float32)
        g = k0 // GRP
        s_w = tl.load(s_ptr + rows * (K // GRP) + g, mask=rmask, other=0.0).to(
            tl.float32
        )
        acc += p * (xs * s_w[None, :])
    r = tl.sum(acc, axis=0) / 16.0  # 16 identical rows; take one
    tl.store(o_ptr + rows, r.to(tl.float32), mask=rmask)


def _pick(N, K):
    """Geometry: enough programs to fill 60 CUs, and BK must divide the
    per-split K slice exactly (a mis-aligned chunk silently skips the
    remainder — a correctness bug, not a crash)."""
    bn = 32 if N >= 4096 else 16
    split = 1
    while triton.cdiv(N, bn) < 240 and split < 8 and (K % (split * 2) == 0):
        split *= 2
    per = K // split
    bk = 512
    while bk > 16 and (bk > per or per % bk):
        bk //= 2
    assert per % bk == 0, f"K={K} split={split} bk={bk}: unaligned chunk"
    return bn, bk, split


def main():
    if not torch.cuda.is_available():
        _log("NO GPU visible (set HIP_VISIBLE_DEVICES)")
        return 2
    dev = torch.cuda.current_device()
    _log(
        f"# P2 int8-weight GEMV probe — {torch.cuda.get_device_name(dev)}, "
        f"torch {torch.__version__}, triton {triton.__version__}, "
        f"visible-dev={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}"
    )
    _log(
        f"  HBM floor reference: {FLOOR_GBS:.0f} GB/s "
        f"(DEVLOG-moe-m1-sprint K=17408 GEMV calibration)"
    )
    quick = os.environ.get("INT8_PROBE_QUICK") == "1"
    shapes = SHAPES[:4] if quick else SHAPES
    skip_dot = os.environ.get("INT8_PROBE_SKIP_DOT") == "1"
    summary = {}

    hdr = (
        f"  {'shape':22s} {'N':>7s} {'K':>5s} | {'f16 torch':>9s} "
        f"{'f16 triton':>10s} {'i8 row':>8s} {'i8 grp':>8s}"
        f"{'':1s} {'i8 dot':>8s} | {'row/f16t':>8s} {'row GB/s':>9s}"
    )
    _log("\n## per-shape times (us, M=1) and the int8/fp16 conversion factor")
    _log(hdr)
    for name, N, K, cd, cm in shapes:
        x = torch.randn(K, device=dev, dtype=torch.float16) * 0.5
        wf = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02
        w8 = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        ng = max(K // GROUP, 1)
        s_row = torch.rand(N, device=dev, dtype=torch.float32) * 0.01 + 1e-3
        s_grp = torch.rand(N * ng, device=dev, dtype=torch.float32) * 0.01 + 1e-3
        out = torch.zeros(N, device=dev, dtype=torch.float32)
        xq = torch.zeros(K, device=dev, dtype=torch.int8)
        xs = torch.zeros(1, device=dev, dtype=torch.float32)
        bn, bk, split = _pick(N, K)
        grid = (triton.cdiv(N, bn), split)
        gsplit = split
        while gsplit > 1 and ng % gsplit:
            gsplit //= 2
        ggrid = (triton.cdiv(N, bn), gsplit)
        t = {}
        try:
            t["f16t"] = do_bench(lambda: torch.mv(wf, x), rep=30) * 1e3
        except Exception as e:
            _log(f"  {name}: torch.mv failed {e}")
            t["f16t"] = float("nan")
        try:
            t["f16"] = (
                do_bench(
                    lambda: k_gemv_f16[grid](
                        x, wf, out, N, K, BN=bn, BK=bk, SPLIT=split, num_warps=8
                    ),
                    rep=30,
                )
                * 1e3
            )
        except Exception as e:
            _log(f"  {name}: f16 triton failed {type(e).__name__}: {e}")
            t["f16"] = float("nan")
        try:
            t["row"] = (
                do_bench(
                    lambda: k_gemv_i8_row[grid](
                        x, w8, s_row, out, N, K, BN=bn, BK=bk, SPLIT=split, num_warps=8
                    ),
                    rep=30,
                )
                * 1e3
            )
        except Exception as e:
            _log(f"  {name}: i8 row failed {type(e).__name__}: {e}")
            t["row"] = float("nan")
        if K % GROUP == 0 and ng % gsplit == 0:
            try:
                t["grp"] = (
                    do_bench(
                        lambda: k_gemv_i8_group[ggrid](
                            x,
                            w8,
                            s_grp,
                            out,
                            N,
                            K,
                            GRP=GROUP,
                            BN=bn,
                            SPLIT=gsplit,
                            num_warps=8,
                        ),
                        rep=30,
                    )
                    * 1e3
                )
            except Exception as e:
                _log(f"  {name}: i8 group failed {type(e).__name__}: {e}")
                t["grp"] = float("nan")
        else:
            t["grp"] = float("nan")
        if skip_dot:
            t["dot"] = float("nan")
        else:
            try:
                blk = max(1024, triton.next_power_of_2(K))
                k_quant_x[(1,)](x, xq, xs, K, BLOCK=blk)
                t["dot"] = (
                    do_bench(
                        lambda: k_gemv_i8_dot[ggrid](
                            xq,
                            w8,
                            xs,
                            s_grp,
                            out,
                            N,
                            K,
                            GRP=GROUP,
                            BN=bn,
                            BK=min(bk, 512),
                            num_warps=8,
                        ),
                        rep=30,
                    )
                    * 1e3
                )
            except Exception as e:
                _log(
                    f"  {name}: i8 dot failed {type(e).__name__}: "
                    f"{str(e).splitlines()[0][:90]}"
                )
                t["dot"] = float("nan")

        # correctness spot-check on the int8 row path (real weights)
        w8f = (w8.to(torch.float32) * s_row[:, None]).to(torch.float16)
        wf2 = w8f
        out.zero_()
        k_gemv_i8_row[grid](
            x, w8, s_row, out, N, K, BN=bn, BK=bk, SPLIT=split, num_warps=8
        )
        # reference dequantizes through fp16, so expect ~1e-3, not 0
        ref = torch.mv(wf2.float(), x.float())
        rel = ((out - ref).abs().max() / ref.abs().max()).item()

        gbs_row = N * K * 1.0 / 1e9 / (t["row"] / 1e6)
        ratio = t["f16"] / t["row"]
        summary[(name, N, K)] = (t, ratio, gbs_row, rel)
        _log(
            f"  {name:22s} {N:7d} {K:5d} | {t['f16t']:9.1f} {t['f16']:10.1f} "
            f"{t['row']:8.1f} {t['grp']:8.1f}  {t['dot']:8.1f} | "
            f"{ratio:8.2f} {gbs_row:9.0f}"
        )
        f16_gbs = N * K * 2 / 1e9 / (t["f16"] / 1e6)
        _log(
            f"      cfg BN={bn} BK={bk} SPLIT={split} | int8-row rel.err "
            f"vs fp32 ref {rel:.2e} | fp16-triton {f16_gbs:.0f} GB/s | "
            f"floors: f16 {N * K * 2 / 1e9 / FLOOR_GBS * 1e6:7.1f} us, "
            f"i8 {N * K * 1.0 / 1e9 / FLOOR_GBS * 1e6:7.1f} us"
        )
        del x, wf, w8, s_row, s_grp, out, xq, xs, w8f, wf2
        torch.cuda.empty_cache()

    # projected serving effect
    _log("\n## projected per-step effect (ideal = the measured row/f16 ratio)")
    byname = {s[0]: s for s in SHAPES}
    for label, idx in (("dense Qwen3.5-27B", 3), ("MoE Qwen3.5-35B-A3B", 4)):
        for tag, want in (("base", False), ("mtp draft", True)):
            tot = saved = 0.0
            for (name, N, K), (t, ratio, gbs, rel) in summary.items():
                cnt = byname[name][idx]
                if not cnt or ratio != ratio:
                    continue
                if ("mtp" in name or "bf16 experts" in name) != want:
                    continue
                tot += t["f16"] * cnt
                saved += (t["f16"] - t["row"]) * cnt
            if tot:
                _log(
                    f"  {label} [{tag}]: fp16-GEMV mass {tot / 1000:6.2f} "
                    f"ms/step; int8 saves {saved / 1000:5.2f} ms/step"
                )
    _log("    (in_proj_a/b, conv1d, router logits and norms are fp16 too but")
    _log("     sub-MB or launch-bound; embed_tokens is a row gather, not a GEMV)")

    key_big = summary.get(("lm_head dense", 248320, 5120))
    key_gdn = summary.get(("gdn in_proj_qkv moe", 8192, 2048))
    _log("\n## VERDICT")
    r1 = key_big[1] if key_big else float("nan")
    r2 = key_gdn[1] if key_gdn else float("nan")
    _log(
        f"  lm_head dense row/f16 = {r1:.2f} (bar 1.70); "
        f"gdn in_proj_qkv row/f16 = {r2:.2f} (bar 1.60)"
    )
    measured = r1 == r1 and r2 == r2
    go = measured and r1 >= 1.70 and r2 >= 1.60
    tag = "GO" if go else ("NO-GO" if measured else "INCONCLUSIVE")
    if not measured:
        _log("  (a gate shape is missing; run full mode)")
    _log(f"  => {tag} for T1 / roadmap I1 (int8 the fp16 weight mass).")
    if go:
        _log("  Next step is NOT a kernel: run the same comparison inside the")
        _log("  serving graph (the transfer rule in AGENTS.md), and gate on")
        _log("  PPL + a KLD probe (docs/gfx906/int8-investigation-qwen.md §3.8).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
