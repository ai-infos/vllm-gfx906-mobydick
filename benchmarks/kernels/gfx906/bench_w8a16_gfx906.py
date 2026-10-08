# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""NH-2 — int8 in-kernel W8A16 (channel) GEMV/GEMM at Nemotron's dense shapes.

Probe (launch-regime evidence) for the NH-2 item in
docs/gfx906/ROADMAP.md: the CT W8A16 channel-dequant scheme currently
dequantizes the packed int8 to fp16 at load and serves it through the
unquantized GEMV family (LLMM1 rpb=4 at Nemotron's K=2688/4096). This
sweeps the int8 in-kernel path (Triton GEMV M=1 / GEMM M>1, signed
int8 weights pre-shifted at load, per-channel fp16 scale) against the
*actual* production dispatch (rocm_unquantized_gemm_impl) at the exact
Nemotron-3.5-Lightning-30B dense-layer shapes:

  mamba in_proj  [10304, 2688] x23/step
  mamba out_proj [ 2688, 4096] x23/step
  GQA  o_proj    [ 2688, 4096] x 6/step
  GQA  q_proj    [ 4096, 2688] x 6/step
  GQA  k/v_proj  [  256, 2688] x12/step
  lm_head        [131072, 2688] x 1/step

The shipping gate is the serving A/B + PPL (DEVLOG-nemotron-h.md); this
probe is the kernel-level gate: if int8 loses at any shape it must show
up here before any serving time is spent. It also sets the static
configs in _gemv_geometry / _gemm_geometry.

Run (GPU idle, single card):
    source ~/env-rocm-7.14-gfx906.sh
    HIP_VISIBLE_DEVICES=0 .venv/bin/python \
        benchmarks/kernels/gfx906/bench_w8a16_gfx906.py
    (W8A16_PROBE_QUICK=1: first 3 shapes; W8A16_PROBE_SKIP_M=1: no M>1)
"""

import os

import torch
import triton

try:
    from triton.testing import do_bench
except ImportError:  # newer triton
    from triton import do_bench  # type: ignore

FLOOR_GBS = float(os.environ.get("W8A16_PROBE_BW", "798"))

# (label, N, K, calls per decode step)
SHAPES = [
    ("mamba in_proj", 10304, 2688, 23),
    ("mamba out_proj", 2688, 4096, 23),
    ("GQA o_proj", 2688, 4096, 6),
    ("GQA q_proj", 4096, 2688, 6),
    ("GQA k/v_proj", 256, 2688, 12),
    ("lm_head", 131072, 2688, 1),
]


def _log(*a):
    print(*a, flush=True)


def _make_weight(n: int, k: int, dev: str):
    """Random bias-128 int8 (raw quant codes) with ~10 % of bytes pinned
    to the 0x80 (zero-weight) code, plus fp16 channel scales. Returns
    (raw uint8, pre-shifted int8 view, scale)."""
    torch.manual_seed(0)
    raw = torch.randint(0, 256, (n, k), device=dev, dtype=torch.uint8)
    mask = torch.rand(n, k, device=dev) < 0.10
    raw = torch.where(mask, torch.full_like(raw, 0x80), raw).contiguous()
    w_i8 = raw.bitwise_xor(0x80).view(torch.int8)
    scale = (torch.rand(n, 1, device=dev) * 0.05 + 0.005).half()
    return raw, w_i8, scale


def _gemv_configs(n: int, k: int):
    """Candidate (BN, BK, SPLIT). Two BK families per (bn, split): P2's
    (largest pow2 <= 512, >= 64 dividing k — masked tail handles the rest)
    and the per-split-aligned one (largest pow2 >= 32 dividing k//split).
    Skip configs that can't fill a wave."""
    cfgs = set()
    for bn in (16, 32, 64):
        if n < bn:
            continue
        for split in (1, 2, 4, 8):
            if k % split or triton.cdiv(n, bn) * split < 60:
                continue
            per = k // split
            bks = set()
            bk = 512
            while bk > 64 and k % bk:
                bk //= 2
            bks.add(bk)
            bkp = 512
            while bkp > 32 and per % bkp:
                bkp //= 2
            bks.add(bkp)
            for bk in bks:
                cfgs.add((bn, bk, split))
    return sorted(cfgs)


def _gemm_configs(m: int, k: int):
    """Candidate (BM, BN, BK, num_warps)."""
    cfgs = set()
    bms = (16,) if m <= 16 else (64,)
    for bm in bms:
        for bn in (32, 64, 128):
            for bk in (64, 128):
                if k % bk:
                    continue
                for nw in (4, 8):
                    cfgs.add((bm, bn, bk, nw))
    return sorted(cfgs)


def _run_gemv(w_i8, scale, x, bn, bk, split):
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a16_channel_dequant as w8,
    )

    n, k = w_i8.shape
    if split > 1:
        out = torch.zeros(n, dtype=scale.dtype, device=w_i8.device)
    else:
        out = torch.empty(n, dtype=scale.dtype, device=w_i8.device)

    def go():
        if split > 1:
            out.zero_()  # atomic accumulators; keeps do_bench iterations independent
        w8.k_w8a16_gemv[(triton.cdiv(n, bn), split)](
            x, w_i8, scale, out, n, k, BN=bn, BK=bk, SPLIT=split, num_warps=8
        )

    go()
    return do_bench(go, rep=30) * 1e3, out


def _run_gemm(w_i8, scale, x, bm, bn, bk, nw):
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a16_channel_dequant as w8,
    )

    n, k = w_i8.shape
    m = x.shape[0]
    out = torch.empty(m, n, dtype=scale.dtype, device=w_i8.device)

    def go():
        w8.k_w8a16_gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            x, w_i8, scale, out, m, n, k, BM=bm, BN=bn, BK=bk, num_warps=nw
        )

    go()
    return do_bench(go, rep=30) * 1e3, out


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).norm().item() / b.float().norm().item()


def main() -> int:
    if not torch.cuda.is_available():
        _log("NO GPU visible (set HIP_VISIBLE_DEVICES)")
        return 2
    dev = torch.cuda.current_device()
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    _log(
        f"# NH-2 W8A16 int8 in-kernel probe — {torch.cuda.get_device_name(dev)}, "
        f"torch {torch.__version__}, triton {triton.__version__}, "
        f"visible-dev={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}"
    )
    _log(f"  HBM floor reference: {FLOOR_GBS:.0f} GB/s")
    quick = os.environ.get("W8A16_PROBE_QUICK") == "1"
    skip_m = os.environ.get("W8A16_PROBE_SKIP_M") == "1"
    shapes = SHAPES[:3] if quick else SHAPES

    step_cur = step_i8 = 0.0
    best_cfgs = {}
    errs = []
    for name, n, k, cd in shapes:
        raw, w_i8, scale = _make_weight(n, k, dev)
        wf = ((raw.float() - 128.0) * scale.float()).half()
        x1 = (torch.randn(1, k, device=dev) * 0.5).half()
        floor1 = n * k / 1e9 / FLOOR_GBS * 1e6

        # --- M=1: cur vs config sweep
        t_cur1 = do_bench(lambda: rocm_unquantized_gemm_impl(x1, wf), rep=30) * 1e3
        best = None
        _log(f"\n## {name}  [{n}, {k}]  x{cd}/step  floor(int8 M=1) = {floor1:7.1f} us")
        _log(f"   M=1 cur (production dispatch, fp16 weights): {t_cur1:8.1f} us")
        for bn, bk, split in _gemv_configs(n, k):
            try:
                t, out = _run_gemv(w_i8, scale, x1[0], bn, bk, split)
            except Exception as e:
                _log(f"   GEMV BN={bn} BK={bk} SPLIT={split}: {type(e).__name__}: {e}")
                continue
            line = (
                f"   GEMV BN={bn:2d} BK={bk:3d} SPLIT={split}: {t:8.1f} us "
                f"({t / floor1 * 100:3.0f} % of floor, {t_cur1 / t:4.2f} x vs cur)"
            )
            _log(line)
            if best is None or t < best[0]:
                best = (t, (bn, bk, split), out)
        if best:
            best_cfgs[(name, "gemv")] = best[1]
            step_cur += cd * t_cur1
            step_i8 += cd * best[0]
            ref1 = x1[0].float() @ ((raw.float() - 128.0) * scale.float()).t()
            errs.append((name, "M=1 best", _relerr(best[2].float(), ref1)))

        if skip_m:
            continue
        x4 = (torch.randn(4, k, device=dev) * 0.5).half()
        xbig = (torch.randn(4096, k, device=dev) * 0.5).half()
        t_cur4 = do_bench(lambda: rocm_unquantized_gemm_impl(x4, wf), rep=30) * 1e3
        t_curB = do_bench(lambda: rocm_unquantized_gemm_impl(xbig, wf), rep=20) * 1e3
        _log(f"   M=4  cur: {t_cur4:8.1f} us | M=4096 cur: {t_curB:9.1f} us")
        for tag, xx, tcur in (("M=4", x4, t_cur4), ("M=4096", xbig, t_curB)):
            best = None
            for bm, bn, bk, nw in _gemm_configs(xx.shape[0], k):
                try:
                    t, out = _run_gemm(w_i8, scale, xx, bm, bn, bk, nw)
                except Exception as e:
                    _log(f"   {tag} ({bm},{bn},{bk},nw{nw}): {type(e).__name__}")
                    continue
                _log(
                    f"   {tag} BM={bm:2d} BN={bn:3d} BK={bk:3d} nw={nw}: {t:9.1f} us"
                    f" ({tcur / t:4.2f} x vs cur)"
                )
                if best is None or t < best[0]:
                    best = (t, (bm, bn, bk, nw), out)
            if best:
                best_cfgs[(name, tag.lower())] = best[1]
                ref = xx.float() @ ((raw.float() - 128.0) * scale.float()).t()
                errs.append((name, tag, _relerr(best[2].float(), ref)))
        del raw, w_i8, wf, x1, x4, xbig
        torch.cuda.empty_cache()

    _log("\n## best configs (feed _gemv_geometry / _gemm_geometry)")
    for key, cfg in sorted(best_cfgs.items()):
        _log(f"   {key[0]:15s} {key[1]:6s} -> {cfg}")

    _log("\n## correctness (rel. err vs fp32 reference; 0x80 zero-codes pinned)")
    worst = 0.0
    for name, m, e in errs:
        flag = "  OK" if e <= 5e-3 else "  ** FAIL **"
        _log(f"  {name:15s} {m:8s} {e:.3e}{flag}")
        worst = max(worst, e)

    _log(
        f"\n## decode step total (M=1, best config): cur {step_cur:7.1f} us -> "
        f"i8 {step_i8:7.1f} us ({step_cur / max(step_i8, 1e-9):.2f}x, "
        f"{step_cur - step_i8:6.1f} us saved)"
    )
    if worst > 5e-3:
        _log("\nVERDICT: CORRECTNESS FAIL — do not ship")
        return 1
    _log("\nVERDICT: see tables; shipping gate is the serving A/B + PPL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
