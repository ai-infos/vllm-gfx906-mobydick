"""QSA-FN-4 gate: the tiled indexer route, selection + speed.

Record: docs/gfx906/DEVLOG-qwen38-flash-qsa.md (4); run from the repo root.

Three arms on one shape (uniform mapping, identical inputs):
  tiled      -- public op ``qsa_mqa_paged`` with a uniform mapping (dispatch
                takes the row-tiled kernel)
  perrow_op  -- same op with a *mixed* mapping, so the gate falls back
                (dispatch evidence)
  perrow_krn -- the per-row Triton kernel launched directly on the uniform
                inputs (kernel-level evidence, same inputs as `tiled`)

fp16 must show a win and top-2048 agreement 1.00000; bf16 must show ~1.00x
against perrow_krn, i.e. the gate kept it off the emulated-dot path.

Usage: probe_fn4_indexer_route.py [rows] [seqlen]
"""

import sys
import time

import torch

from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops  # noqa: E402

DEV = "cuda"
NH, HD, CR, PAGE = 4, 128, 4, 64
BUDGET = 2048
NROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
L = int(sys.argv[2]) if len(sys.argv) > 2 else 30720
CL = L // CR
NPAGES = (CL + PAGE - 1) // PAGE
SD = float(HD**0.5)
BN, BM = 32, 16


def bench(fn, it=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / it * 1e6


def topk_agreement(a, b, k):
    ka = a.topk(k, dim=1).indices
    kb = b.topk(k, dim=1).indices
    agree = 0
    for i in range(0, a.shape[0], 256):
        n = min(256, a.shape[0] - i)
        sa = torch.zeros(n, a.shape[1], device=DEV, dtype=torch.bool)
        sb = torch.zeros_like(sa)
        sa.scatter_(1, ka[i : i + n], True)
        sb.scatter_(1, kb[i : i + n], True)
        agree += (sa & sb).sum().item()
    return agree / (a.shape[0] * k)


def run(dt):
    torch.manual_seed(0)
    q = torch.randn(NROWS, NH, HD, device=DEV, dtype=dt) * 0.5
    kc = torch.randn(NPAGES, PAGE, 1, HD, device=DEV, dtype=dt) * 0.5
    pt = torch.arange(NPAGES, device=DEV, dtype=torch.int32).view(1, NPAGES)
    qpos = torch.arange(NROWS, device=DEV, dtype=torch.int32) + (L - NROWS)
    seqlen = torch.tensor([L], device=DEV, dtype=torch.int32)
    uni = torch.zeros(NROWS, device=DEV, dtype=torch.int32)
    mixed = (torch.arange(NROWS, device=DEV, dtype=torch.int32) * 2) // NROWS

    def op(t2r):
        return qsa_ops.qsa_mqa_paged(
            q, kc, pt, t2r, qpos, seqlen, CR, num_columns=CL, score_scale=SD
        )

    def krn():
        lg = torch.empty(NROWS, CL, dtype=torch.float32, device=DEV)
        vb = torch.empty(NROWS, dtype=torch.int32, device=DEV)
        qsa_ops._qsa_mqa_paged_kernel[(NROWS, (CL + BN - 1) // BN)](
            q,
            kc,
            pt,
            uni,
            qpos,
            seqlen,
            vb,
            lg,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kc.stride(0),
            kc.stride(1),
            kc.stride(3),
            pt.stride(0),
            pt.stride(1),
            lg.stride(0),
            NROWS,
            CL,
            kc.shape[0],
            pt.shape[0],
            SD,
            PAGE_SIZE=PAGE,
            PAGE_TABLE_WIDTH=NPAGES,
            NUM_HEADS=NH,
            HEAD_DIM=HD,
            BLOCK_N=BN,
            BLOCK_D=HD,
            COMPRESS_RATIO=CR,
            num_warps=4,
        )
        return lg, vb

    lg_t, vb_t = op(uni)
    lg_p, _ = krn()
    torch.cuda.synchronize()
    fin = torch.isfinite(lg_t) & torch.isfinite(lg_p)
    nrmse = (
        torch.linalg.vector_norm((lg_t - lg_p)[fin].float())
        / torch.linalg.vector_norm(lg_p[fin].float())
    ).item()
    infin = (torch.isinf(lg_t) == torch.isinf(lg_p)).float().mean().item()
    k = min(BUDGET, CL)
    agree = topk_agreement(lg_t, lg_p, k)

    reps = 3
    totals = {"tiled": 0.0, "perrow_op": 0.0, "perrow_krn": 0.0}
    for _ in range(reps):
        totals["tiled"] += bench(lambda: op(uni))
        totals["perrow_op"] += bench(lambda: op(mixed))
        totals["perrow_krn"] += bench(krn)
    t = {n: v / reps for n, v in totals.items()}
    tag = str(dt).split(".")[-1]
    print(
        f"[{tag}] tiled={t['tiled']:.0f}us  perrow_op={t['perrow_op']:.0f}us "
        f"perrow_krn={t['perrow_krn']:.0f}us  "
        f"speedup_vs_krn={t['perrow_krn'] / t['tiled']:.2f}x "
        f"speedup_vs_op={t['perrow_op'] / t['tiled']:.2f}x  "
        f"topk_agree={agree:.5f} inf_agree={infin:.4f} logits_nrmse={nrmse:.2e}",
        flush=True,
    )


run(torch.float16)
run(torch.bfloat16)
