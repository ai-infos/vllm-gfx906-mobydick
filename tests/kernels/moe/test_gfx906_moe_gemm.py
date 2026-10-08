# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Correctness test for the gfx906 WNA16 MoE grouped-GEMM kernel.

Moved from /tmp/bench/_test_gfx906_moe.py into the repo (Common Protocol item 1
of plan-moe-phase2.md). Exercises _rocm_C.moe_gptq_gemm_gfx906 end to end:
repack via the oracle, moe_align_block_size routing, both gemm passes (w13
scatter + w2 fused topk-weight/reduce), against a naive per-expert torch
reference.

Run (on a gfx906 host / with the ROCm image):
    pytest tests/kernels/moe/test_gfx906_moe_gemm.py -v
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    _repack_w4a16_gfx906_expert,
)

# small expert count is fine: the kernel is per-expert for routing
E = 16
TOPK = 4
GS = 128
DBLL = "gfx"  # (unused; kept for readability in parametrize ids)


def _make_layer(N, K, layout, gs=GS):
    """Random W4A16 weights in the requested source layout.

    Returns (w, s, qzeros_packed, q, z_ref) where q is the per-element uint4
    codes [E, K, N] and z_ref the per-element zero points [E, G, N] (AWQ /
    GPTQ K-first) or [E, N, G] (MoeWNA16) for the dequant reference.
    Symmetric layouts pass qzeros_packed=None and use the implicit zero
    point 8.
    """
    G = K // gs
    sh = 1 << (4 * torch.arange(8, device=torch.device("cuda")))
    q = torch.randint(0, 16, (E, K, N), device=torch.device("cuda"), dtype=torch.int32)
    z = torch.randint(0, 16, (E, G, N), device=torch.device("cuda"), dtype=torch.int32)
    s = (
        torch.rand(E, G, N, device=torch.device("cuda"), dtype=torch.float16) * 0.05
        + 0.01
    )
    if layout in ("awq_kfirst", "awq_kfirst_sym"):
        # AutoAWQ K-first int32: word m holds n=8m..8m+7 (low nibble first)
        w = (q.view(E, K, N // 8, 8) * sh).sum(-1).to(torch.int32)
        if layout == "awq_kfirst":
            zw = (z.view(E, G, N // 8, 8) * sh).sum(-1).to(torch.int32)
            return w, s, zw, q, z
        return (
            w,
            s,
            None,
            q,
            torch.full((E, G, N), 8, device=q.device, dtype=torch.int32),
        )
    if layout == "gptq_kfirst_sym":
        # compressed-tensors GPTQ-style K-first int32: word r holds
        # k=8r..8r+7 (low nibble first); symmetric, no stored zero points.
        w = (q.view(E, K // 8, 8, N) * sh.view(1, 1, 8, 1)).sum(2).to(torch.int32)
        return (
            w,
            s,
            None,
            q,
            torch.full((E, G, N), 8, device=q.device, dtype=torch.int32),
        )
    if layout == "gptq_kfirst":
        # compressed-tensors asymmetric pack-quantized: weights packed
        # along K like gptq_kfirst_sym; zps arrive from the MoE loader
        # already in the kernel's layout [E, G, N/8] int32 (8 consecutive
        # n per word, low nibble first — the standard CT packing).
        w = (q.view(E, K // 8, 8, N) * sh.view(1, 1, 8, 1)).sum(2).to(torch.int32)
        zw = (z.view(E, G, N // 8, 8) * sh).sum(-1).to(torch.int32)
        return w, s, zw, q, z
    # MoeWNA16 N-first uint8: byte j holds k=2j low / 2j+1 high;
    # zp byte i holds n=2i low / 2i+1 high
    qn = q.permute(0, 2, 1).reshape(E, N, K // 2, 2)
    w = (qn[..., 0] | (qn[..., 1] << 4)).to(torch.uint8)
    sc = s.permute(0, 2, 1).contiguous()
    if layout == "wna16":
        zn = z.permute(0, 2, 1).reshape(E, N // 2, 2, G)
        zw = (zn[..., 0, :] | (zn[..., 1, :] << 4)).to(torch.uint8)
        return w, sc, zw, q, z.permute(0, 2, 1)
    # "wna16_sym": symmetric, no stored zero points
    return w, sc, None, q, torch.full((E, N, G), 8, device=q.device, dtype=torch.int32)


def _dequant_ref(w, s, z, q, gs=GS):
    """[E,N,K] fp32 dequantized weights: (q - z[g(k), n]) * s[g(k), n]."""
    E_, K, N = q.shape
    g = torch.arange(K, device=q.device) // gs  # [K]
    if s.shape[1] == N:  # MoeWNA16 scales/zp: [E, N, G]
        st = s.transpose(1, 2)[:, g, :]  # [E, K, N]
        zt = z.transpose(1, 2)[:, g, :]  # [E, K, N]
        return ((q.long() - zt.long()) * st.float()).permute(0, 2, 1)
    return ((q.long() - z[:, g].long()) * s[:, g].float()).permute(0, 2, 1)


def _run_case(M, N13, K13, N2, K2, block_m, gs=GS, layout=None):
    from vllm import _custom_ops as ops

    w13, s13, z13, q13, zz13 = _make_layer(N13, K13, layout, gs)
    w2, s2, z2, q2, zz2 = _make_layer(N2, K2, layout, gs)
    wq13, sc13, zp13 = _repack_w4a16_gfx906_expert(w13, s13, z13)
    wq2, sc2, zp2 = _repack_w4a16_gfx906_expert(w2, s2, z2)

    x = torch.randn(M, K13, dtype=torch.float16) * 0.5
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32)
    topk_w = torch.rand(M, TOPK, dtype=torch.float16)
    dev = "cuda"
    x = x.to(dev)
    topk_ids = topk_ids.to(dev)
    topk_w = topk_w.to(dev)

    sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids, block_m, E)

    # ---- gemm1 (w13): [M, K13] -> [M*TOPK, N13] scatter ----
    c1 = torch.zeros(M * TOPK, N13, device=dev, dtype=torch.float16)
    empty_tw = torch.empty(0, dtype=torch.float32, device=dev)
    ops.moe_gptq_gemm_gfx906(
        x,
        c1,
        wq13,
        sc13,
        zp13,
        empty_tw,
        sorted_ids,
        expert_ids,
        ntp,
        TOPK,
        block_m,
        False,
        0,
        0,
    )

    ref_rows = torch.zeros(M * TOPK, N13, device=dev, dtype=torch.float32)
    for e in range(E):
        sel = topk_ids == e
        if not sel.any():
            continue
        rows = sel.nonzero()
        tok = rows[:, 0]
        wdeq = _dequant_ref(w13, s13, zz13, q13, gs)[e]
        ref_rows[tok * TOPK + rows[:, 1]] = x[tok].float() @ wdeq.t()
    err1 = (c1.float() - ref_rows).abs().max().item()
    rel1 = err1 / ref_rows.abs().max().item()

    # ---- activation: silu_and_mul on c1 -> [M*TOPK, N13/2] ----
    # (identity for non-gated shapes, e.g. Nemotron-3.5-Lightning relu2
    # experts where N13 == K2; the kernel is activation-agnostic)
    if N13 == K2:
        inter = c1.contiguous()
    else:
        inter = (
            (
                torch.nn.functional.silu(c1[:, : N13 // 2].float())
                * c1[:, N13 // 2 :].float()
            )
            .half()
            .contiguous()
        )

    # ---- gemm2 (w2): [M*TOPK, K2] -> [M, N2] fused weight + reduce ----
    assert N13 // 2 == K2 or N13 == K2, "test shape mismatch"
    out = torch.zeros(M, N2, device=dev, dtype=torch.float16)
    ops.moe_gptq_gemm_gfx906(
        inter,
        out,
        wq2,
        sc2,
        zp2,
        topk_w.view(-1).float(),
        sorted_ids,
        expert_ids,
        ntp,
        1,
        block_m,
        True,
        TOPK,
        0,
    )

    wdeq2 = _dequant_ref(w2, s2, zz2, q2, gs)
    ref_out = torch.zeros(M, N2, device=dev, dtype=torch.float32)
    for e in range(E):
        sel = topk_ids == e
        if not sel.any():
            continue
        rows = sel.nonzero()
        tok = rows[:, 0]
        h = inter[rows[:, 0] * TOPK + rows[:, 1]].float() @ wdeq2[e].t()
        h *= topk_w[tok, rows[:, 1]].float().unsqueeze(1)
        for i in range(len(tok)):
            ref_out[tok[i]] += h[i]
    err2 = (out.float() - ref_out).abs().max().item()
    rel2 = err2 / ref_out.abs().max().item()

    # M=1 takes the relative-error denominator over only TOPK rows, so fp16
    # accumulation noise is ~2x noisier there; BM=16 has the largest per-cell
    # CAS fan-in (16 sequential fp16 adds) and its worst cell has been
    # observed just over 5e-2: allow 1e-1 for both edge cases.
    tol = 1e-1 if (M == 1 or block_m == 16) else 5e-2
    assert rel1 < tol, f"gemm1 too far from reference: maxrel={rel1:.2e}"
    assert rel2 < tol, f"gemm2 too far from reference: maxrel={rel2:.2e}"
    return rel1, rel2


# (M, N13, K13, N2, K2, block_m, gs): Qwen-like shapes scaled to E=16
# experts, plus the Gemma-4-26B-A4B-AWQ shapes (group-32; N=1408 is not a
# multiple of the 1024-wide N tile, exercising the partial gridY block).
_CASES = [
    (1, 1024, 2048, 1024, 512, 1, 128),
    (4, 1024, 2048, 1024, 512, 1, 128),
    (8, 1024, 2048, 1024, 512, 4, 128),
    (32, 1024, 2048, 1024, 512, 4, 128),
    (64, 1024, 2048, 1024, 512, 16, 128),
    (2, 1536, 768, 1536, 768, 1, 128),  # K not mult of 256; N=1536 (gridY partial)
    (1, 1408, 2816, 2816, 704, 1, 32),  # Gemma-4 gemm1/gemm2, decode
    (32, 1408, 2816, 2816, 704, 4, 32),  # Gemma-4 gemm1/gemm2, prefill
    (1, 1856, 2688, 2688, 1856, 1, 64),  # Nemotron-3.5-Lightning, decode
    (32, 1856, 2688, 2688, 1856, 4, 64),  # Nemotron-3.5-Lightning, prefill
]


def _ids(c):
    return f"M{c[0]}-bm{c[5]}-N{c[1]}-gs{c[6]}"


@pytest.mark.parametrize(
    "layout",
    [
        "awq_kfirst",
        "wna16",
        "awq_kfirst_sym",
        "wna16_sym",
        "gptq_kfirst_sym",
        "gptq_kfirst",
    ],
)
@pytest.mark.parametrize("case", _CASES, ids=_ids)
def test_gfx906_moe_gemm(case, layout):
    _run_case(*case, layout)


def test_gfx906_repack_gptq_kfirst_sym():
    """compressed-tensors symmetric (no-zp) GPTQ K-first repack: the
    exllama shuffle is bit-exact, scales pass through, no zp tensor is
    returned (the kernel inlines the constant zero point 8 for an empty
    zp), and the asymmetric (stored-zp) variant passes its zps through
    bit-exact while a wrong zp shape fails closed.

    Shape mirrors Gemma-4: K = 704 (gemm2 K, 88 words), N = 1408 (gemm1
    N), group-32 (22 groups). CPU-only."""
    E, K8, N, G = 2, 88, 1408, 22
    torch.manual_seed(0)
    w = torch.randint(0, 2**31, (E, K8, N), dtype=torch.int32)
    s = torch.rand(E, G, N, dtype=torch.float16)
    wq, sc, zp = _repack_w4a16_gfx906_expert(w, s, None)

    assert wq.shape == (E, K8, N)
    assert torch.equal(sc, s), "scales must pass through unchanged"
    assert zp is None, "symmetric repack returns no zp tensor"

    # Bit-exact shuffle: source nibble j (k = 8*qk + j) lands at bits
    # [0,16,4,20,8,24,12,28][j].
    shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28])
    nib = (w.unsqueeze(-1) >> (4 * torch.arange(8))) & 0xF  # [E,K8,N,8]
    got = (wq.unsqueeze(-1) >> (4 * torch.arange(8))) & 0xF
    for j in range(8):
        dest = shifts[j].item() // 4  # bit slot holding source nibble j
        assert torch.equal(got[..., dest], nib[..., j]), f"nibble {j} misplaced"

    # Asymmetric pack-quantized (stored qzeros): the MoE loader presents
    # them K-first [E, G, N/8] int32 — the kernel's native layout — and
    # they must pass through bit-exact.
    zw = torch.randint(0, 2**31, (E, G, N // 8))
    wq_asym, sc_asym, zp_asym = _repack_w4a16_gfx906_expert(w, s, zw)
    assert torch.equal(wq_asym, wq), "asymmetric shuffle must match"
    assert torch.equal(sc_asym, s), "scales must pass through unchanged"
    assert torch.equal(zp_asym, zw), "zps must pass through unchanged"

    # A zp tensor outside the [E, G, N/8] contract fails closed.
    with pytest.raises(ValueError, match="N/8"):
        _repack_w4a16_gfx906_expert(w, s, torch.randint(0, 2**31, (E, G, N)))


def _take_path():
    """Read-and-reset the M=1 gemm dispatch-path marker (see
    csrc/rocm/moe_q_gemm_gfx906.cu). Returns the tile selected by the most
    recent moe_gptq_gemm_gfx906 call: 0 = legacy <1,4> gemm1 (MOE_NPT=4),
    1 = v2 512-thread gemm2, 2 = legacy <1,4> gemm2 fallback, 3 = default
    <1,2> gemm1. Needed because the M=1 kernels are atomic-accumulated and
    therefore not bit-reproducible run-to-run."""
    from vllm import _custom_ops as ops

    return int(ops.take_moe_m1_dispatch_path())


@pytest.mark.parametrize("layout", ["awq_kfirst", "awq_kfirst_sym"])
def test_gfx906_moe_gemm_m1_v2_flag(layout):
    """VLLM_GFX906_MOE_M1 re-tiles the M=1 gemm2 (fused weight/reduce)
    path to the 512-thread lane-column kernel. It is now DEFAULT-ON (the
    C2 combined A/B promoted it), so the gates are: the default and an
    explicit =1 both take the V2 tile, the explicit opt-out (=0) takes the
    legacy <1,4> tile, and every arm stays within normal reference
    tolerance. Path selection is verified with the dispatch-path marker
    (torch.ops._rocm_C.take_moe_m1_dispatch_path): the M=1 kernels
    accumulate via packed CAS atomics, so their outputs are NOT
    bit-reproducible run-to-run and cannot prove which tile ran. The test
    shape (N2=1024, K2=512) is V2-qualifying so the shape gate passes.
    awq_kfirst_sym covers the symmetric (empty zp / inlined zero 8) path."""
    import os

    from vllm import _custom_ops as ops

    N13, K13, N2, K2 = 1024, 2048, 1024, 512
    w13, s13, z13, _, _ = _make_layer(N13, K13, layout)
    w2, s2, z2, q2, zz2 = _make_layer(N2, K2, layout)
    wq13, sc13, zp13 = _repack_w4a16_gfx906_expert(w13, s13, z13)
    wq2, sc2, zp2 = _repack_w4a16_gfx906_expert(w2, s2, z2)

    dev = "cuda"
    torch.manual_seed(0)
    x = (torch.randn(1, K13, dtype=torch.float16) * 0.5).to(dev)
    topk_ids = torch.randint(0, E, (1, TOPK), dtype=torch.int32).to(dev)
    topk_w = torch.rand(1, TOPK, dtype=torch.float16).to(dev)
    sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids, 1, E)
    empty_tw = torch.empty(0, dtype=torch.float32, device=dev)

    def gemm2():
        c1 = torch.zeros(TOPK, N13, device=dev, dtype=torch.float16)
        ops.moe_gptq_gemm_gfx906(
            x,
            c1,
            wq13,
            sc13,
            zp13,
            empty_tw,
            sorted_ids,
            expert_ids,
            ntp,
            TOPK,
            1,
            False,
            0,
            0,
        )
        inter = (
            (
                torch.nn.functional.silu(c1[:, : N13 // 2].float())
                * c1[:, N13 // 2 :].float()
            )
            .half()
            .contiguous()
        )
        out = torch.zeros(1, N2, device=dev, dtype=torch.float16)
        ops.moe_gptq_gemm_gfx906(
            inter,
            out,
            wq2,
            sc2,
            zp2,
            topk_w.view(-1).float(),
            sorted_ids,
            expert_ids,
            ntp,
            1,
            1,
            True,
            TOPK,
            0,
        )
        return out

    # gemm1 is <1,2> by default (marker 3) in every arm — the flag under test
    # only touches gemm2, so read the marker after each call pair.
    for k in ("VLLM_GFX906_MOE_M1", "VLLM_GFX906_MOE_NPT"):
        os.environ.pop(k, None)
    out_default = gemm2()
    assert _take_path() == 1, (
        "default M=1 gemm2 did not take the V2 re-tile (marker != 1)"
    )
    os.environ["VLLM_GFX906_MOE_M1"] = "0"
    try:
        out_off = gemm2()
    finally:
        os.environ.pop("VLLM_GFX906_MOE_M1")
    assert _take_path() == 2, (
        "VLLM_GFX906_MOE_M1=0 did not select the legacy <1,4> gemm2 "
        "(marker != 2)"
    )
    os.environ["VLLM_GFX906_MOE_M1"] = "1"
    try:
        out_on = gemm2()
    finally:
        os.environ.pop("VLLM_GFX906_MOE_M1")
    assert _take_path() == 1, (
        "VLLM_GFX906_MOE_M1=1 did not select the V2 tile (marker != 1)"
    )

    # fp32 reference for gemm2 (recompute inter; the gemm1 path is
    # flag-invariant across these arms)
    wdeq2 = _dequant_ref(w2, s2, zz2, q2)
    ref_out = torch.zeros(1, N2, device=dev, dtype=torch.float32)
    c1 = torch.zeros(TOPK, N13, device=dev, dtype=torch.float16)
    ops.moe_gptq_gemm_gfx906(
        x,
        c1,
        wq13,
        sc13,
        zp13,
        empty_tw,
        sorted_ids,
        expert_ids,
        ntp,
        TOPK,
        1,
        False,
        0,
        0,
    )
    inter = (
        (
            torch.nn.functional.silu(c1[:, : N13 // 2].float())
            * c1[:, N13 // 2 :].float()
        )
        .half()
        .contiguous()
    )
    ref_out.zero_()
    for e in range(E):
        for i in range(TOPK):
            if topk_ids[0, i] != e:
                continue
            h = inter[i].float() @ wdeq2[e].t()
            ref_out[0] += h * topk_w[0, i].float()
    scale = ref_out.abs().max().item()
    for name, out in (("default(v2)", out_default), ("opt-out", out_off), ("=1(v2)", out_on)):
        rel = ((out.float() - ref_out).abs().max() / scale).item()
        assert rel < 1e-1, f"v2-flag gemm2 ({name}) too far from reference: maxrel={rel:.2e}"


@pytest.mark.parametrize("layout", ["awq_kfirst", "awq_kfirst_sym"])
def test_gfx906_moe_gemm_m1_npt2_flag(layout):
    """VLLM_GFX906_MOE_NPT=2 re-tiles the M=1 **gemm1** (w13 scatter) path
    to the <1,2> kernel (64 cols/block vs <1,4>'s 128). It is now
    DEFAULT-ON for BM=1 (the C2 combined A/B promoted it), so the gates
    mirror the MOE_M1 test: explicit opt-out (=4) differs from the
    default/<1,2> arm, the default matches an explicit =2, both arms stay
    within normal reference tolerance, and the off-vs-on gap is
    fp16-atomic noise level. gemm2 must be untouched by the flag (its
    dispatch ignores NPT for BM=1 unless MOE_M1 fires) — tested below via
    the dispatch marker, not just asserted in prose."""
    import os

    from vllm import _custom_ops as ops

    N13, K13, N2, K2 = 1024, 2048, 1024, 512
    w13, s13, z13, q13, zz13 = _make_layer(N13, K13, layout)
    wq13, sc13, zp13 = _repack_w4a16_gfx906_expert(w13, s13, z13)

    dev = "cuda"
    torch.manual_seed(1)
    x = (torch.randn(1, K13, dtype=torch.float16) * 0.5).to(dev)
    topk_ids = torch.randint(0, E, (1, TOPK), dtype=torch.int32).to(dev)
    sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids, 1, E)
    empty_tw = torch.empty(0, dtype=torch.float32, device=dev)

    def gemm1():
        c1 = torch.zeros(TOPK, N13, device=dev, dtype=torch.float16)
        ops.moe_gptq_gemm_gfx906(
            x,
            c1,
            wq13,
            sc13,
            zp13,
            empty_tw,
            sorted_ids,
            expert_ids,
            ntp,
            TOPK,
            1,
            False,
            0,
            0,
        )
        return c1

    # Default (no env var) must be the <1,2> re-tile now; explicit opt-out
    # (=4) selects the legacy <1,4> tile and explicit =2 must match default.
    for k in ("VLLM_GFX906_MOE_M1", "VLLM_GFX906_MOE_NPT"):
        os.environ.pop(k, None)
    out_default = gemm1()
    assert _take_path() == 3, (
        "default M=1 gemm1 did not take the <1,2> re-tile (marker != 3)"
    )
    os.environ["VLLM_GFX906_MOE_NPT"] = "4"
    try:
        out_off = gemm1()
    finally:
        os.environ.pop("VLLM_GFX906_MOE_NPT")
    assert _take_path() == 0, (
        "VLLM_GFX906_MOE_NPT=4 did not select the legacy <1,4> gemm1 "
        "(marker != 0)"
    )
    os.environ["VLLM_GFX906_MOE_NPT"] = "2"
    try:
        out_on = gemm1()
    finally:
        os.environ.pop("VLLM_GFX906_MOE_NPT")
    assert _take_path() == 3, (
        "VLLM_GFX906_MOE_NPT=2 did not select the <1,2> tile (marker != 3)"
    )
    # fp32 per-expert reference for the w13 scatter.
    wdeq13 = _dequant_ref(w13, s13, zz13, q13)
    ref_c1 = torch.zeros(TOPK, N13, device=dev, dtype=torch.float32)
    for i in range(TOPK):
        e = int(topk_ids[0, i])
        ref_c1[i] = x[0].float() @ wdeq13[e].t()
    scale = ref_c1.abs().max().item()
    for name, out in (("default(<1,2>)", out_default), ("opt-out(<1,4>)", out_off), ("=2(<1,2>)", out_on)):
        rel = ((out.float() - ref_c1).abs().max() / scale).item()
        assert rel < 1e-1, f"NPT-flag gemm1 ({name}) too far from reference: {rel:.2e}"

    # No-leak check (the docstring's "gemm2 must be untouched" claim): the
    # NPT flag must not change gemm2 dispatch. This test shape is V2-
    # qualifying, so with MOE_M1 unset gemm2 takes the default-on v2 tile
    # (marker 1) both with and without NPT=2 set.
    w2, s2, z2, q2, zz2 = _make_layer(N2, K2, layout)
    wq2, sc2, zp2 = _repack_w4a16_gfx906_expert(w2, s2, z2)
    topk_w = torch.rand(1, TOPK, dtype=torch.float16).to(dev)

    def gemm2_only():
        c1b = torch.zeros(TOPK, N13, device=dev, dtype=torch.float16)
        ops.moe_gptq_gemm_gfx906(
            x, c1b, wq13, sc13, zp13, empty_tw, sorted_ids, expert_ids, ntp,
            TOPK, 1, False, 0, 0,
        )
        inter = (
            (
                torch.nn.functional.silu(c1b[:, : N13 // 2].float())
                * c1b[:, N13 // 2 :].float()
            )
            .half()
            .contiguous()
        )
        out2 = torch.zeros(1, N2, device=dev, dtype=torch.float16)
        ops.moe_gptq_gemm_gfx906(
            inter, out2, wq2, sc2, zp2, topk_w.view(-1).float(), sorted_ids,
            expert_ids, ntp, 1, 1, True, TOPK, 0,
        )
        return out2

    for k in ("VLLM_GFX906_MOE_M1", "VLLM_GFX906_MOE_NPT"):
        os.environ.pop(k, None)
    gemm2_only()
    assert _take_path() == 1, (
        "baseline M=1 gemm2 did not take the default-on v2 tile (marker != 1)"
    )
    os.environ["VLLM_GFX906_MOE_NPT"] = "2"
    try:
        gemm2_only()
    finally:
        os.environ.pop("VLLM_GFX906_MOE_NPT")
    assert _take_path() == 1, (
        "VLLM_GFX906_MOE_NPT=2 leaked into gemm2 dispatch (marker != 1)"
    )


@pytest.mark.parametrize("layout", ["awq_kfirst", "awq_kfirst_sym"])
def test_gfx906_moe_gemm_m1_shape_gate(layout):
    """The default-on M=1 gemm2 V2 path is SHAPE-GATED: for a non-qualifying
    shape (Nemotron-3.5-Lightning decode, K2=1856 not a multiple of 256) the
    default must fall back to the legacy <1,4> tile — verified via the
    dispatch-path marker (marker 2, same as explicit opt-out =0) and within
    normal reference tolerance — while an explicit =1 fails closed
    (TORCH_CHECK) instead of silently running a kernel that rejects the
    shape. This is what keeps default-on safe for models whose gemm2 does not
    meet the V2 tile's N%256==0 / K%256==0 / K<=2048 / groupsize%32==0
    contract."""
    import os

    from vllm import _custom_ops as ops

    # Nemotron-3.5-Lightning shapes (gs=64): gemm2 = [., 1856] x [1856, 2688]
    N13, K13, N2, K2 = 1856, 2688, 2688, 1856
    gs = 64
    w13, s13, z13, _, _ = _make_layer(N13, K13, layout, gs)
    w2, s2, z2, q2, zz2 = _make_layer(N2, K2, layout, gs)
    wq13, sc13, zp13 = _repack_w4a16_gfx906_expert(w13, s13, z13)
    wq2, sc2, zp2 = _repack_w4a16_gfx906_expert(w2, s2, z2)

    dev = "cuda"
    torch.manual_seed(2)
    x = (torch.randn(1, K13, dtype=torch.float16) * 0.5).to(dev)
    topk_ids = torch.randint(0, E, (1, TOPK), dtype=torch.int32).to(dev)
    topk_w = torch.rand(1, TOPK, dtype=torch.float16).to(dev)
    sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids, 1, E)
    empty_tw = torch.empty(0, dtype=torch.float32, device=dev)

    def gemm2():
        c1 = torch.zeros(TOPK, N13, device=dev, dtype=torch.float16)
        ops.moe_gptq_gemm_gfx906(
            x,
            c1,
            wq13,
            sc13,
            zp13,
            empty_tw,
            sorted_ids,
            expert_ids,
            ntp,
            TOPK,
            1,
            False,
            0,
            0,
        )
        inter = c1.contiguous()  # identity: Nemotron relu2 experts (N13 == K2)
        out = torch.zeros(1, N2, device=dev, dtype=torch.float16)
        ops.moe_gptq_gemm_gfx906(
            inter,
            out,
            wq2,
            sc2,
            zp2,
            topk_w.view(-1).float(),
            sorted_ids,
            expert_ids,
            ntp,
            1,
            1,
            True,
            TOPK,
            0,
        )
        return out

    for k in ("VLLM_GFX906_MOE_M1", "VLLM_GFX906_MOE_NPT"):
        os.environ.pop(k, None)
    out_default = gemm2()
    assert _take_path() == 2, (
        "default M=1 gemm2 on a non-qualifying shape did not fall back to "
        "the legacy <1,4> tile (marker != 2)"
    )
    os.environ["VLLM_GFX906_MOE_M1"] = "0"
    try:
        out_off = gemm2()
    finally:
        os.environ.pop("VLLM_GFX906_MOE_M1")
    assert _take_path() == 2, (
        "VLLM_GFX906_MOE_M1=0 did not select the legacy <1,4> gemm2 "
        "(marker != 2)"
    )

    # Explicit =1 on a non-qualifying shape fails closed (TORCH_CHECK).
    os.environ["VLLM_GFX906_MOE_M1"] = "1"
    try:
        with pytest.raises(RuntimeError, match="shape"):
            gemm2()
    finally:
        os.environ.pop("VLLM_GFX906_MOE_M1")

    # The fallback result must still be correct against the fp32 reference.
    wdeq2 = _dequant_ref(w2, s2, zz2, q2, gs)
    ref_out = torch.zeros(1, N2, device=dev, dtype=torch.float32)
    c1 = torch.zeros(TOPK, N13, device=dev, dtype=torch.float16)
    ops.moe_gptq_gemm_gfx906(
        x,
        c1,
        wq13,
        sc13,
        zp13,
        empty_tw,
        sorted_ids,
        expert_ids,
        ntp,
        TOPK,
        1,
        False,
        0,
        0,
    )
    inter = c1.contiguous()  # identity: Nemotron relu2 experts (N13 == K2)
    for e in range(E):
        for i in range(TOPK):
            if topk_ids[0, i] != e:
                continue
            h = inter[i].float() @ wdeq2[e].t()
            ref_out[0] += h * topk_w[0, i].float()
    scale = ref_out.abs().max().item()
    for name, out in (("default(fallback)", out_default), ("opt-out", out_off)):
        rel = ((out.float() - ref_out).abs().max() / scale).item()
        assert rel < 1e-1, f"shape-gate fallback gemm2 ({name}) too far: maxrel={rel:.2e}"


if __name__ == "__main__":
    print("run via pytest")
