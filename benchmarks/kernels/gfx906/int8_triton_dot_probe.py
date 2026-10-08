#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""P1 — does an int8 (A8W8) tl.dot GEMM have a future on gfx906? (probe)

Decides roadmap item I4 (A8W8 prefill) and the act-quant leg of
docs/gfx906/int8-investigation-qwen.md §3.3/§3.4. Nothing here imports vLLM.

Background for the gate (all in docs/gfx906/): v_dot4_i32_i8 is full-rate on
gfx906 (25 877 GMAC/s measured, 4.44x fp32 FMA, 1.96x packed fp16 at
13 210), and our Triton port claims to lower int8 tl.dot to it
(third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp:48-53 +
AccelerateAMDMatmul.cpp:1665-1669, "i8 x i8 -> i32 ... if k % 4 == 0").
That is a SOURCE-level reading; this probe checks the ISA that actually
comes out, the arithmetic, and the rate.

Parts:
  A  int8 tl.dot: exactness vs an fp64 reference, + ISA scan for
     v_dot4_i32_i8 (kernel.asm['amdgcn'], cache-scan fallback).
  B  rate: int8 dot vs fp16 dot vs fp32 at the same tile (expect ~1.96x
     if dot4 is really being used and the loop is issue-bound).
  C  A8W8 blockscale GEMM (the portable part of the gfx908 fork's
     triton_w8a8_gemm_kernel, dense int8 weights instead of GPTQ-packed)
     vs a same-structure fp16 Triton GEMM vs torch fp16 matmul
     (hipBLAS = the realistic production-quality baseline), at our
     prefill shapes, WITH the per-call activation quant charged.
  D  per-token act-quant: trunc vs floor(x+0.5) vs tl.extra.hip.libdevice
     .round (availability + bit-difference + cost).

GO iff A(dot4 in ISA, exact) and C >= 1.3x vs the torch fp16 baseline at
[4096, 34816, 5120] with the quant pass charged.

RESULT (2026-08-31, MI50): NO-GO. A passes -- v_dot4_i32_i8 is emitted (16x,
exact vs fp64, and exactly half the dot instructions of the same-work fp16
kernel) -- but C lands at 0.59-0.68x of hipBLAS fp16 and only 1.10-1.18x of
same-codegen Triton fp16: Triton reaches 19% of the dot4 record, so the
compiler deficit exceeds everything the int8 edge can buy. Record:
DEVLOG-int8-transfer.md and the INT8 rows in DEAD-ENDS.md.

Usage:
  HIP_VISIBLE_DEVICES=0 .venv/bin/python \
      benchmarks/kernels/gfx906/int8_triton_dot_probe.py
Env: INT8_PROBE_QUICK=1 (fewer shapes), INT8_PROBE_SKIP_DOT=1 (skip B),
     TRITON_CACHE_DIR (ISA fallback), TRITON_ALWAYS_COMPILE=1.
"""

import os
import re
import subprocess
import sys
import time

import torch
import triton
import triton.language as tl

try:
    from triton.testing import do_bench
except Exception:  # pragma: no cover - very old triton
    from triton import do_bench  # type: ignore

LLVM = "/opt/rocm/lib/llvm/bin"
REC_DOT4 = 25877.0  # GMAC/s, dequant-instructions.md (MI50, SCEV-proof)
REC_F16 = 13210.0  # GMAC/s, packed v_pk_fma_f16 on the same probe


def _log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
@triton.jit
def k_dot_i8(
    a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    """Plain int8 x int8 -> int32 dot over the full K (correctness/rate)."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        a = tl.load(
            a_ptr + rm[:, None] * K + ks[None, :], mask=(rm[:, None] < M), other=0
        )
        b = tl.load(b_ptr + ks[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    tl.store(
        c_ptr + rm[:, None] * N + rn[None, :],
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def k_dot_f16(
    a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    """Same tiling in fp16 (fdot2 path) — the apples-to-apples comparator."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        a = tl.load(
            a_ptr + rm[:, None] * K + ks[None, :], mask=(rm[:, None] < M), other=0.0
        )
        b = tl.load(b_ptr + ks[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(
        c_ptr + rm[:, None] * N + rn[None, :],
        acc.to(tl.float16),
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def k_a8w8_group(
    a_ptr,
    b_ptr,
    as_ptr,
    bs_ptr,
    c_ptr,
    M,
    N,
    K,
    GROUP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """A8W8: per-token act scale x per-K-GROUP weight scale (BK == GROUP).

    Structurally the gfx908 fork's triton_w8a8_gemm_kernel with dense int8
    weights (no GPTQ int32 unpack — that ALU is a separate question; this
    measures the dot+descale ceiling). Exact by construction: one descale
    per tile, |sum| <= 128*127*127 < 2**31 for GROUP=128.
    """
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    ng = K // GROUP
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # a_scales are PER TOKEN (one fp32 per row: the aiter pertoken_quant /
    # pertoken_quant_rn contract); weight scales are per K-GROUP block.
    a_s0 = tl.load(as_ptr + rm, mask=rm < M, other=0.0)
    for g in range(0, ng):
        ks = g * GROUP + tl.arange(0, GROUP)
        a = tl.load(
            a_ptr + rm[:, None] * K + ks[None, :], mask=(rm[:, None] < M), other=0
        )
        b = tl.load(b_ptr + ks[:, None] * N + rn[None, :])
        p = tl.dot(a, b, out_dtype=tl.int32)
        a_s = a_s0
        b_s = tl.load(bs_ptr + g * N + rn)
        acc += p.to(tl.float32) * (a_s[:, None] * b_s[None, :])
    tl.store(
        c_ptr + rm[:, None] * N + rn[None, :],
        acc.to(tl.float16),
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def k_actq(x_ptr, q_ptr, s_ptr, K, MODE: tl.constexpr, BLOCK: tl.constexpr):
    """Per-token symmetric int8 quant. MODE 0=trunc, 1=floor(x+0.5),
    2=tl.extra.hip.libdevice.round."""
    row = tl.program_id(0).to(tl.int64)
    xp = x_ptr + row * K
    qp = q_ptr + row * K
    amax = 0.0
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        x = tl.load(xp + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=0))
    s = tl.maximum(amax, 0.0) / 127.0
    s = tl.where(s == 0.0, 1.0, s)
    tl.store(s_ptr + row, s)
    rcp = 1.0 / s
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        x = tl.load(xp + offs, mask=offs < K, other=0.0).to(tl.float32)
        v = x * rcp
        if MODE == 1:
            v = tl.floor(v + 0.5)
        elif MODE == 2:
            v = tl.extra.hip.libdevice.round(v)
        v = tl.minimum(tl.maximum(v, -127.0), 127.0)
        tl.store(qp + offs, v.to(tl.int8), mask=offs < K)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _asm_of(kernel_obj, cache_dir, t_start):
    """Return (text, how) for a launched triton kernel's ISA."""
    asm = getattr(kernel_obj, "asm", None)
    if isinstance(asm, dict):
        for k in ("amdgcn", "ptx", "hsaco"):
            v = asm.get(k)
            if isinstance(v, str):
                return v, f"kernel.asm[{k!r}]"
            if isinstance(v, (bytes, bytearray)) and k == "hsaco":
                out = _objdump_bytes(v)
                if out:
                    return out, "kernel.asm['hsaco']+llvm-objdump"
    # fallback: newest .amdgcn / .hsaco in the triton cache
    try:
        cands = []
        for root, _, files in os.walk(cache_dir):
            for f in files:
                p = os.path.join(root, f)
                if (
                    f.endswith((".amdgcn", ".hsaco"))
                    and os.path.getmtime(p) >= t_start - 1
                ):
                    cands.append(p)
        if cands:
            p = max(cands, key=os.path.getmtime)
            if p.endswith(".amdgcn"):
                return open(p).read(), f"cache {os.path.basename(p)}"
            out = subprocess.run(
                [f"{LLVM}/llvm-objdump", "-d", p],
                capture_output=True,
                text=True,
                timeout=120,
            ).stdout
            return out, f"objdump {os.path.basename(p)}"
    except Exception as e:  # pragma: no cover
        _log(f"  (ISA cache fallback failed: {e})")
    return "", "unavailable"


def _objdump_bytes(b):
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
            f.write(b)
            path = f.name
        return subprocess.run(
            [f"{LLVM}/llvm-objdump", "-d", path],
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
    except Exception:
        return ""


def _isa_counts(text):
    pats = {
        "v_dot4_i32_i8": r"\bv_dot4[_a-z0-9]*\b",
        "v_dot2_f32_f16": r"\bv_dot2[_a-z0-9]*\b",
        "v_pk_fma_f16": r"\bv_pk_fma_f16\b",
        "v_mac_f32": r"\bv_mac_f32\b",
        "v_fma_f32": r"\bv_fma_f32\b",
    }
    return {k: len(re.findall(v, text)) for k, v in pats.items()}


def _gmacs(m, n, k, ms):
    """MAC/s in GMAC/s (compare against the 25 877 / 13 210 records, which
    are MACs, not FLOPs)."""
    return (m * n * k / 1e9) / (ms / 1e3)


def main():
    if not torch.cuda.is_available():
        _log("NO GPU visible (set HIP_VISIBLE_DEVICES)")
        return 2
    dev = torch.cuda.current_device()
    _log(
        f"# P1 int8-dot probe — {torch.cuda.get_device_name(dev)}, "
        f"torch {torch.__version__}, triton {triton.__version__}, "
        f"visible-dev={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}"
    )
    quick = os.environ.get("INT8_PROBE_QUICK") == "1"
    cache_dir = os.environ.get(
        "TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache")
    )
    results = {}

    # ---------------- A: exactness + ISA ---------------------------------
    _log("\n## A. int8 tl.dot exactness + emitted ISA")
    M = N = K = 64 if not quick else 32
    a = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
    b = torch.randint(-127, 128, (K, N), device=dev, dtype=torch.int8)
    c = torch.zeros((M, N), device=dev, dtype=torch.int32)
    BM = BN = 32
    BK = 16
    t0 = time.time()
    kern = k_dot_i8[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, b, c, M, N, K, BM=BM, BN=BN, BK=BK, num_warps=4
    )
    ref = (a.to(torch.float64) @ b.to(torch.float64)).to(torch.int64)
    exact = torch.equal(c.to(torch.int64), ref)
    results["A_exact"] = exact
    _log(
        f"  exact int32 accumulate vs fp64 ref: {exact} "
        f"(maxdiff {(c.to(torch.int64) - ref).abs().max().item():.0f})"
    )
    text, how = _asm_of(kern, cache_dir, t0)
    cnt = _isa_counts(text)
    results["A_dot4_in_isa"] = cnt["v_dot4_i32_i8"] > 0
    _log(f"  ISA source: {how}; counts: {cnt}")
    if text:
        head = [
            ln.strip()
            for ln in text.splitlines()
            if re.search(r"\bv_(dot|mac|fma)\w*\b", ln)
        ][:6]
        _log("  first dot/mac/fma lines: " + " | ".join(head))

    # ---------------- B: rate -------------------------------------------
    if os.environ.get("INT8_PROBE_SKIP_DOT") != "1":
        _log("\n## B. rate: int8 dot vs fp16 dot (same tiling, K % 4 == 0)")
        M = N = K = 1024
        ai = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
        bi = torch.randint(-127, 128, (K, N), device=dev, dtype=torch.int8)
        ci = torch.zeros((M, N), device=dev, dtype=torch.int32)
        af = torch.randn(M, K, device=dev, dtype=torch.float16) * 0.1
        bf = torch.randn(K, N, device=dev, dtype=torch.float16) * 0.1
        cf = torch.zeros((M, N), device=dev, dtype=torch.float16)
        rows = []
        for BM, BN, BK, w in ((64, 64, 128, 4), (128, 128, 64, 8), (128, 64, 128, 4)):
            try:
                gi = do_bench(
                    lambda: k_dot_i8[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
                        ai, bi, ci, M, N, K, BM=BM, BN=BN, BK=BK, num_warps=w
                    ),
                    rep=40,
                )
                gf = do_bench(
                    lambda: k_dot_f16[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
                        af, bf, cf, M, N, K, BM=BM, BN=BN, BK=BK, num_warps=w
                    ),
                    rep=40,
                )
            except Exception as e:
                _log(f"  tile {BM}x{BN}x{BK}: FAILED {type(e).__name__}: {e}")
                continue
            ri, rf = _gmacs(M, N, K, gi), _gmacs(M, N, K, gf)
            rows.append((BM, BN, BK, ri, rf, ri / rf))
            _log(
                f"  tile {BM:3d}x{BN:3d}x{BK:3d} w{w}: int8 {ri:8.0f} GMAC/s "
                f"({100 * ri / REC_DOT4:5.1f}% of dot4 record) | "
                f"fp16 {rf:8.0f} ({100 * rf / REC_F16:5.1f}% of f16 record) | "
                f"ratio {ri / rf:5.2f}x"
            )
        if rows:
            best = max(rows, key=lambda r: r[5])
            results["B_best_ratio"] = best[5]
            _log(
                f"  best int8/fp16 ratio = {best[5]:.2f}x "
                f"(1.96x is the ISA-level ceiling)"
            )

    # ---------------- C: production-shape A8W8 vs fp16 -------------------
    _log("\n## C. prefill shapes: A8W8 blockscale GEMM vs fp16")
    _log(
        "   (int8 row includes the per-call act-quant pass; torch fp16 "
        "matmul is the realistic-baseline reference)"
    )
    shapes = [
        (4096, 34816, 5120),
        (4096, 5120, 17408),
        (4096, 14336, 5120),
        (1024, 34816, 5120),
        (256, 34816, 5120),
    ]
    if quick:
        shapes = shapes[:2]
    GROUP = 128
    c_rows = []
    for M, N, K in shapes:
        if K % GROUP:
            continue
        xf = torch.randn(M, K, device=dev, dtype=torch.float16) * 0.5
        xq = torch.empty((M, K), device=dev, dtype=torch.int8)
        xs = torch.empty((M,), device=dev, dtype=torch.float32)
        wq = torch.randint(-127, 128, (K, N), device=dev, dtype=torch.int8)
        ws = torch.rand(K // GROUP, N, device=dev, dtype=torch.float32) * 0.01 + 0.001
        wf = wq.to(torch.float16) * 0.004
        out = torch.empty((M, N), device=dev, dtype=torch.float16)
        BM, BN, BK, w = (128, 128, GROUP, 8) if M >= 1024 else (32, 128, GROUP, 4)

        def a8w8_with_quant():
            k_actq[(M,)](xf, xq, xs, K, MODE=1, BLOCK=2048, num_warps=8)
            k_a8w8_group[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
                xq,
                wq,
                xs,
                ws,
                out,
                M,
                N,
                K,
                GROUP=GROUP,
                BM=BM,
                BN=BN,
                BK=BK,
                num_warps=w,
            )

        t_int = do_bench(a8w8_with_quant, rep=30)
        t_int_noq = do_bench(
            lambda: k_a8w8_group[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
                xq,
                wq,
                xs,
                ws,
                out,
                M,
                N,
                K,
                GROUP=GROUP,
                BM=BM,
                BN=BN,
                BK=BK,
                num_warps=w,
            ),
            rep=30,
        )
        t_f16 = do_bench(lambda: k_f16_wrap(wf, xf, out, M, N, K), rep=30)
        t_torch = do_bench(lambda: torch.matmul(xf, wf), rep=30)
        t_q = do_bench(
            lambda: k_actq[(M,)](xf, xq, xs, K, MODE=1, BLOCK=2048, num_warps=8), rep=30
        )
        r_int = _gmacs(M, N, K, t_int_noq)
        r_torch = _gmacs(M, N, K, t_torch)
        _log(
            f"  [M={M} N={N} K={K}]  int8(+quant) {t_int:7.3f} ms | "
            f"int8(no quant) {t_int_noq:7.3f} | quant alone {t_q:6.3f} | "
            f"triton fp16 {t_f16:7.3f} | torch fp16 {t_torch:7.3f}"
        )
        _log(
            f"      -> int8/fp16(triton) {t_f16 / t_int:5.2f}x | "
            f"int8/torch {t_torch / t_int:5.2f}x | rates int8 "
            f"{r_int:5.0f} ({100 * r_int / REC_DOT4:4.1f}% of dot4 record) "
            f"vs torch {r_torch:5.0f} ({100 * r_torch / REC_F16:4.1f}%)"
        )
        c_rows.append((M, N, K, t_int, t_f16, t_torch))
        del xf, xq, xs, wq, ws, wf, out
        torch.cuda.empty_cache()
    key = None
    for row in c_rows:
        if row[1] == 34816 and row[0] == 4096:
            key = row
    if key:
        r_tri, r_torch = key[4] / key[3], key[5] / key[3]
        results["C_ratio_vs_torch"] = r_torch
        results["C_ratio_vs_triton"] = r_tri
        _log(
            f"  headline leg [4096,34816,5120]: {r_tri:.2f}x vs triton fp16, "
            f"{r_torch:.2f}x vs torch fp16"
        )

    # ---------------- D: act-quant numerics/cost -------------------------
    _log("\n## D. per-token int8 act-quant: trunc vs floor(+0.5) vs libdevice")
    M, K = 4096, 5120
    xf = torch.randn(M, K, device=dev, dtype=torch.float16)
    q0 = torch.empty((M, K), device=dev, dtype=torch.int8)
    s0 = torch.empty((M,), device=dev, dtype=torch.float32)
    outs = {}
    for mode, name in ((0, "trunc"), (1, "floor(x+0.5)"), (2, "ld.round")):
        try:
            k_actq[(M,)](xf, q0, s0, K, MODE=mode, BLOCK=2048, num_warps=8)
            outs[name] = (q0.clone(), s0.clone())
            ms = do_bench(
                lambda: k_actq[(M,)](xf, q0, s0, K, MODE=mode, BLOCK=2048, num_warps=8),
                rep=30,
            )
            gbs = (M * K * (2 + 1) + M * 4) / 1e9 / (ms / 1e3)
            _log(f"  {name:14s}: {ms * 1000:7.1f} us  ({gbs:5.0f} GB/s eff.)")
        except Exception as e:
            _log(
                f"  {name:14s}: UNAVAILABLE — {type(e).__name__}: "
                f"{str(e).splitlines()[0][:110]}"
            )
    if len(outs) == 3:
        d01 = (outs["trunc"][0] != outs["floor(x+0.5)"][0]).float().mean()
        d12 = (outs["floor(x+0.5)"][0] != outs["ld.round"][0]).float().mean()
        _log(
            f"  payload disagreement: trunc vs floor = {d01.item() * 100:.1f}% "
            f"of elements (expect ~50%); floor vs libdevice.round = "
            f"{d12.item() * 100:.3f}% (expect 0 unless RN-even differs)"
        )

    # ---------------- verdict -------------------------------------------
    _log("\n## VERDICT")
    dot4 = results.get("A_dot4_in_isa")
    exact = results.get("A_exact")
    ratio = results.get("C_ratio_vs_torch")
    _log(f"  A: dot4 in ISA={dot4}  exact={exact}")
    _log(f"  B: best int8/fp16 tile ratio={results.get('B_best_ratio')}")
    _log(
        f"  C: A8W8 vs torch fp16 at [4096,34816,5120] = "
        f"{None if ratio is None else round(ratio, 3)} (GO bar 1.30)"
    )
    go = bool(dot4 and exact and ratio is not None and ratio >= 1.30)
    _log(
        f"  => {'GO' if go else 'NO-GO'} for T3, the Triton A8W8 prefill "
        f"route (recorded 2026-08-31: NO-GO, 0.59-0.68x of hipBLAS fp16). "
        f"DEAD-ENDS.md carries it; the HIP successor (T5) is scoped "
        f"separately in docs/gfx906/int8-investigation-qwen.md §3.3/§3.10."
    )
    return 0


@triton.jit
def k_f16(
    a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        a = tl.load(
            a_ptr + rm[:, None] * K + ks[None, :], mask=(rm[:, None] < M), other=0.0
        )
        b = tl.load(b_ptr + ks[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(
        c_ptr + rm[:, None] * N + rn[None, :],
        acc.to(tl.float16),
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


def k_f16_wrap(wf, xf, out, M, N, K):
    BM, BN, BK = (128, 128, 64) if M >= 1024 else (32, 128, 64)
    k_f16[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        xf, wf, out, M, N, K, BM=BM, BN=BN, BK=BK, num_warps=8
    )


if __name__ == "__main__":
    sys.exit(main())
