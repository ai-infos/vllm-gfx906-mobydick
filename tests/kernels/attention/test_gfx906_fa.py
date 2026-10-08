# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""gfx906 custom FlashAttention (vllm/gfx906_fa) regression tests.

The gather/direct kernels take byte strides computed in C++. A past bug
computed them from tensor SHAPES; the backend passes value_cache from
kv_cache.unbind(1) of [num_blocks, 2, block, Hkv, D] — non-contiguous with
2x block stride — so the kernel read K-cache bytes as V. These tests mirror
that allocation path exactly.
"""

import math
import os
import sys

import pytest
import torch

from vllm import _gfx906_fa_C as fa
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx906

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx906()),
    reason="gfx906 FA extension kernels",
)

BLOCK, HKV, HQ, D = 16, 2, 16, 256
BYTES = (D // 32) * 34


def _kv_split(kv: torch.Tensor):
    """K/V views of the 0.29 fused KV layout (upstream #51718): one tensor per
    layer, ``[N, Hkv, BLOCK, 2*D]``, K/V = the two halves of the last axis.
    Mirrors the backend's own split."""
    k, v = kv.transpose(1, 2).split(D, dim=-1)
    assert not k.is_contiguous() and not v.is_contiguous()
    return k, v


def _make_fused_cache(num_blocks: int, dev: str):
    """Backend-level cache in the 0.29 fused layout (see _kv_split). The
    op-level ``_make_paged_cache`` keeps the legacy [N,2,B,Hkv,D] allocation that
    the standalone C++ entries expect."""
    kc = torch.zeros(num_blocks, BLOCK, HKV, BYTES, dtype=torch.uint8, device=dev)
    kv = torch.zeros(num_blocks, HKV, BLOCK, 2 * D, dtype=torch.float16, device=dev)
    return kc, _kv_split(kv)[1], kv


def _write_v_fused(v_view: torch.Tensor, V: torch.Tensor):
    """Write token-major V rows through a fused-layout V view."""
    staging = torch.zeros(v_view.shape, dtype=v_view.dtype, device=v_view.device)
    staging.reshape(-1, HKV, D)[: V.shape[0]].copy_(V)
    v_view.copy_(staging)


def _make_paged_cache(num_blocks: int, dev: str):
    """Mirror Gfx906FABackend: one kv_cache tensor, unbind(1) -> K, V views."""
    kc = torch.zeros(num_blocks, BLOCK, HKV, BYTES, dtype=torch.uint8, device=dev)
    kv = torch.zeros(num_blocks, 2, BLOCK, HKV, D, dtype=torch.float16, device=dev)
    key_cache_q8 = kc
    _, value_cache = kv.unbind(1)
    assert not value_cache.is_contiguous()
    return key_cache_q8, value_cache, kv


def _write_v(kv: torch.Tensor, V: torch.Tensor):
    """Write token-major V rows into the V half of the [N,2,B,H,D] cache."""
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, HKV, D)[: V.shape[0]].copy_(V)
    kv[:, 1].copy_(staging)


def _fill(kv_flat_rows: torch.Tensor, kc: torch.Tensor, slot: torch.Tensor):
    fa.reshape_and_cache_q8(kv_flat_rows, slot, kc)


def test_fused_gather_matches_torch_gather_on_unbind_cache():
    dev = "cuda"
    torch.manual_seed(1)
    num_blocks = 40
    kc, vc, kv = _make_paged_cache(num_blocks, dev)
    n_rows = num_blocks * BLOCK
    K = torch.randn(n_rows, HKV, D, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(n_rows, HKV, D, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(n_rows, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    _write_v(kv, V)

    B, seq_lens, max_len = 2, [100, 300], 300
    n_blocks = (max_len + BLOCK - 1) // BLOCK
    bt = torch.arange(0, B * n_blocks, dtype=torch.int32, device=dev)
    bt = bt.view(B, n_blocks).contiguous()
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    Sk_pad = (max_len + 31) // 32 * 32

    from vllm.gfx906_fa.gfx906_fa_paged import _gather_kv_q8

    k_ref, v_ref = _gather_kv_q8(kc, vc, bt, sl, max_len)
    k_f, v_f = fa.gather_paged_kv_q8(kc, vc, bt, sl, Sk_pad)
    for b, L in enumerate(seq_lens):
        # Only the valid region must match: the fused kernel leaves K tail
        # rows unwritten (kernel cuts them via kv_max); the torch path
        # gathers real cache rows there instead.
        assert torch.equal(k_f[b, :, :L], k_ref[b, :, :L])
        assert torch.equal(v_f[b, :, :L], v_ref[b, :, :L])
        assert bool((v_f[b, :, L:] == 0).all().item())


@pytest.mark.parametrize("B,seq_lens", [(2, [100, 300]), (1, [3328]), (1, [33])])
def test_fused_gather_quantized_bit_equal_to_gather_then_quantize(B, seq_lens):
    """Stage-2 fused gather+quantize must be bit-equal to the two-kernel
    sequence (gather_paged_kv_fp16 + quantize_q8_0): same quantization
    helper, same arithmetic. Guards against future drift between the
    fused kernel and the reference quantizer."""
    dev = "cuda"
    torch.manual_seed(3)
    max_len = max(seq_lens)
    n_blocks_needed = (max_len + BLOCK - 1) // BLOCK
    num_blocks = B * n_blocks_needed + 4
    # Same allocation as Gfx906FABackend LEGACY path: one [N,2,B,H,D]
    # tensor, unbind(1) -> non-contiguous K/V views (2x block stride).
    kv = torch.zeros(num_blocks, 2, BLOCK, HKV, D,
                     dtype=torch.float16, device=dev)
    key_cache, value_cache = kv.unbind(1)
    assert not key_cache.is_contiguous() and not value_cache.is_contiguous()
    K = torch.randn(num_blocks, BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    V = torch.randn(num_blocks, BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    kv[:, 0].copy_(K)
    kv[:, 1].copy_(V)

    n_blocks = (max_len + BLOCK - 1) // BLOCK
    bt = torch.arange(0, B * n_blocks, dtype=torch.int32, device=dev)
    bt = bt.view(B, n_blocks).contiguous()
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    Sk_pad = (max_len + 31) // 32 * 32

    k_two, v_two = fa.gather_paged_kv_fp16(
        key_cache, value_cache, bt, sl, Sk_pad)
    k_ref = fa.quantize_q8_0(k_two)
    k_one, v_one = fa.gather_paged_kv_quantized(
        key_cache, value_cache, bt, sl, Sk_pad)

    assert k_one.shape == k_ref.shape
    assert k_one.dtype == torch.uint8
    for b, L in enumerate(seq_lens):
        # Valid region: bit-exact (K quantized, V copied).
        assert torch.equal(k_one[b, :, :L], k_ref[b, :, :L])
        assert torch.equal(v_one[b, :, :L], v_two[b, :, :L])
        # Tail: V zeroed, K may be garbage (FA kernel cuts via kv_max).
        assert bool((v_one[b, :, L:] == 0).all().item())


def test_cudagraph_capture_replay_legacy_decode_path():
    """M2 gate: the LEGACY (inline-quant) decode path must be FULL-capture-safe.

    Captures the exact serving composite (`forward_paged` with
    key_cache_q8=None, i.e. fp16 K cache + inline K quant) and covers the
    sub-plan T3 landmines for this path: (a) warmup at a small max_seqlen_k
    followed by capture at capacity (buffer-realloc class); (b) multi-size
    capture (B=1 then B=2) with a B=1 replay afterwards (dangling-buffer
    class); (c) the live-metadata invariant — seq_lens is re-read at replay,
    so growing Sk and filling the new K/V rows must make the replayed output
    match eager at the new length.
    """
    dev = "cuda"
    torch.manual_seed(3)
    max_len = 512
    n_blocks = (max_len + BLOCK - 1) // BLOCK
    kc, vc, kv = _make_paged_cache(n_blocks + 4, dev)
    scale = 1.0 / math.sqrt(D)

    # LEGACY path: K lives in an fp16 cache (contiguous here; the backend's
    # unbind(1) K view has the same per-element layout the C++ strides expect),
    # V in the strided unbind view as in serving (exercised via _write_v).
    k16 = torch.zeros(n_blocks + 4, BLOCK, HKV, D, dtype=torch.float16,
                      device=dev)
    K = torch.randn(max_len, HKV, D, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(max_len, HKV, D, device=dev, dtype=torch.float16) * 0.5
    _write_v(kv, V[:100])
    k16.view(-1, HKV, D)[:100].copy_(K[:100])

    from vllm.gfx906_fa.gfx906_fa_paged import forward_paged

    # Shared q_pad buffer across both graphs, as the backend's lazy-grown
    # class buffer would be at capture capacity.
    q_pad = torch.zeros(2, HQ, 2, D, dtype=torch.float32, device=dev)

    def fwd(q, bt_, sl_, cu_, msk):
        return forward_paged(
            q, k16, vc, bt_, sl_, cu_,
            max_seqlen_q=1, max_seqlen_k=msk, scale=scale,
            key_cache_q8=None, q_pad_buf=q_pad,
        )

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, -1)
    sl = torch.tensor([100], dtype=torch.int32, device=dev)
    cu1 = torch.arange(2, dtype=torch.int32, device=dev)
    q1 = torch.randn(1, HQ, D, device=dev, dtype=torch.float32) * 0.5

    # (a) warmup at small max_seqlen_k, then capture at capacity
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            fwd(q1, bt, sl, cu1, 128)
    torch.cuda.current_stream().wait_stream(s)
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        out1 = fwd(q1, bt, sl, cu1, max_len)
    g1.replay()
    torch.cuda.synchronize()
    ref1 = fwd(q1, bt, sl, cu1, max_len)
    assert ((out1 - ref1).norm() / ref1.norm()).item() < 2e-2

    # (b) capture B=2 after B=1, then replay B=1 (dangling-buffer check)
    # Both rows share the same 32 blocks (arange(n_blocks).view(2, -1) would
    # be (2, 16) — wrong column count).
    bt2 = torch.arange(n_blocks, dtype=torch.int32,
                       device=dev).view(1, -1).expand(2, -1).contiguous()
    sl2 = torch.tensor([100, 150], dtype=torch.int32, device=dev)
    cu2 = torch.arange(3, dtype=torch.int32, device=dev)
    q2 = torch.randn(2, HQ, D, device=dev, dtype=torch.float32) * 0.5
    with torch.cuda.stream(s):
        for _ in range(2):
            fwd(q2, bt2, sl2, cu2, 256)
    torch.cuda.current_stream().wait_stream(s)
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        out2 = fwd(q2, bt2, sl2, cu2, max_len)
    g2.replay()
    torch.cuda.synchronize()
    ref2 = fwd(q2, bt2, sl2, cu2, max_len)
    assert (
        (out2[1] - ref2[1]).norm() / ref2[1].norm()
    ).item() < 2e-2  # row 1 at Sk=150 exercises the longer row
    g1.replay()
    torch.cuda.synchronize()
    assert ((out1 - ref1).norm() / ref1.norm()).item() < 2e-2

    # (c) live seq_lens: grow Sk 100 -> 200, fill K/V rows, replay g1
    k16.view(-1, HKV, D)[100:200].copy_(K[100:200])
    _write_v(kv, V[:200])
    sl.fill_(200)
    g1.replay()
    torch.cuda.synchronize()
    ref200 = fwd(q1, bt, sl, cu1, max_len)
    assert ((out1 - ref200).norm() / ref200.norm()).item() < 2e-2


def test_persistent_gather_capture_replay_large_sk():
    """N4 fix gate: the persistent fused gather+quantize
    (GFX906_FA_PERSIST) must be FULL-capture-safe at capacity Sk_pad
    (262144, above the old 65535 two-kernel boundary) and its replayed
    end-to-end FA output must match the two-kernel fallback at every
    live seq_len in the sweep — the launch dim is frozen at Sk_pad while
    seq_lens is re-read at replay, exactly like the serving FULL graph.
    Buffer contents beyond seq_len may legitimately differ (tail-write
    removal, gated by the NaN-tail test); end-to-end output must not.
    """
    dev = "cuda"
    torch.manual_seed(5)
    sk_pad = 262144
    n_blocks = sk_pad // BLOCK + 4
    _, value_cache, kv = _make_paged_cache(n_blocks, dev)
    scale = 1.0 / math.sqrt(D)

    k16 = torch.zeros(n_blocks, BLOCK, HKV, D, dtype=torch.float16,
                      device=dev)
    k16.normal_(0, 0.5)
    kv[:, 1].normal_(0, 0.5)

    from vllm.gfx906_fa import gfx906_fa_paged as fpp
    from vllm.gfx906_fa.gfx906_fa_paged import forward_paged

    q = torch.randn(1, HQ, D, device=dev, dtype=torch.float32) * 0.5
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, -1)
    sl = torch.tensor([100], dtype=torch.int32, device=dev)
    cu = torch.arange(2, dtype=torch.int32, device=dev)
    q_pad = torch.zeros(1, HQ, 1, D, dtype=torch.float32, device=dev)

    def fwd(persist, msk):
        fpp._PERSISTENT = persist
        return forward_paged(
            q, k16, value_cache, bt, sl, cu,
            max_seqlen_q=1, max_seqlen_k=msk, scale=scale,
            key_cache_q8=None, q_pad_buf=q_pad,
        )

    orig = fpp._PERSISTENT
    try:
        # warmup the persistent path at small Sk, then capture at capacity
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(2):
                fwd(True, 128)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fwd(True, sk_pad)
    finally:
        fpp._PERSISTENT = orig

    def rel_err(a, b):
        return ((a - b).norm() / b.norm()).item()

    sweep = [32, 100, 2048, 65504, 65536, 65600, 131072, sk_pad - 32,
             sk_pad]
    for v in sweep:
        sl.fill_(v)
        g.replay()
        torch.cuda.synchronize()
        ref_two = fwd(False, sk_pad)          # two-kernel fallback
        ref_persist = fwd(True, sk_pad)       # eager persistent
        e_two = rel_err(out, ref_two)
        e_persist = rel_err(out, ref_persist)
        assert e_persist == 0.0, (
            f"replay vs eager persistent not bit-exact at sl={v}: "
            f"rel={e_persist}")
        assert e_two < 2e-2, (
            f"replay vs two-kernel fallback diverges at sl={v}: rel={e_two}")


def test_persistent_dispatch_fallback_large_batch():
    """P0 regression guard (fa-masked-gather review): with PERSIST default
    ON, a batch above the kernel's 16-seq bound must fall back to the
    fused/two-kernel paths (old behavior) instead of hitting the C++
    TORCH_CHECK(num_seqs <= 16) — which would crash engine start for any
    default max_num_seqs (> 16)."""
    dev = "cuda"
    torch.manual_seed(7)
    b = 17
    max_len = 512
    n_blocks = b * (max_len // BLOCK)
    kv = torch.zeros(n_blocks, 2, BLOCK, HKV, D,
                     dtype=torch.float16, device=dev)
    key_cache, value_cache = kv.unbind(1)
    kv[:, 0].normal_(0, 0.5)
    kv[:, 1].normal_(0, 0.5)

    from vllm.gfx906_fa import gfx906_fa_paged as fpp
    from vllm.gfx906_fa.gfx906_fa_paged import forward_paged

    scale = 1.0 / math.sqrt(D)
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev)
    bt = bt.view(b, max_len // BLOCK).contiguous()
    sl = torch.tensor([33, 100, 200, 333, 512] + [400] * (b - 5),
                      dtype=torch.int32, device=dev)
    cu = torch.arange(b + 1, dtype=torch.int32, device=dev)
    q = torch.randn(b, HQ, D, device=dev, dtype=torch.float32) * 0.5
    q_pad = torch.zeros(b, HQ, 1, D, device=dev, dtype=torch.float32)

    def fwd(persist):
        fpp._PERSISTENT = persist
        return forward_paged(
            q, key_cache, value_cache, bt, sl, cu,
            max_seqlen_q=1, max_seqlen_k=max_len, scale=scale,
            key_cache_q8=None, q_pad_buf=q_pad,
        )

    orig = fpp._PERSISTENT
    try:
        out_persist = fwd(True)   # must not raise (B=17 > 16)
        out_ref = fwd(False)
    finally:
        fpp._PERSISTENT = orig
    assert torch.equal(out_persist, out_ref)


def test_persistent_gather_bit_equal_to_fused_at_batch_bound():
    """B=16 (the kernel's register-prefix bound), ragged seq_lens, small
    Sk: the persistent kernel must be bit-equal to the fused kernel
    (gather_paged_kv_quantized) in-range. The capture probe covers
    B=1..4 at full 262k live; this covers the prefix bound itself."""
    dev = "cuda"
    torch.manual_seed(11)
    b, sk = 16, 1024
    n_blocks = b * (sk // BLOCK)
    kv = torch.zeros(n_blocks, 2, BLOCK, HKV, D,
                     dtype=torch.float16, device=dev)
    key_cache, value_cache = kv.unbind(1)
    kv[:, 0].normal_(0, 0.5)
    kv[:, 1].normal_(0, 0.5)

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev)
    bt = bt.view(b, sk // BLOCK).contiguous()
    sl = torch.tensor(
        [1, 33, 100, 300, 512, 777, 1000] + [640] * (b - 7),
        dtype=torch.int32, device=dev)

    kb = torch.empty(b, HKV, sk, BYTES, dtype=torch.uint8, device=dev)
    vb = torch.empty(b, HKV, sk, D, dtype=torch.float16, device=dev)
    k_fused, v_fused = fa.gather_paged_kv_quantized(
        key_cache, value_cache, bt, sl, sk)
    k_p, v_p = fa.gather_paged_kv_quant_persistent(
        key_cache, value_cache, bt, sl, sk, k_out=kb, v_out=vb)
    for s_ in range(b):
        L = int(sl[s_])
        assert torch.equal(k_p[s_, :, :L], k_fused[s_, :, :L]), f"K s={s_}"
        assert torch.equal(v_p[s_, :, :L], v_fused[s_, :, :L]), f"V s={s_}"


def test_persistent_gather_d128_matches_fused():
    """D=128 (other advertised head size; different FA tile config):
    persistent kernel bit-equal to the fused kernel in-range. Kernel is
    D-generic (V uint4 D/8 lanes, K blocks_per_row=D/32); this pins it
    before default-ON widens past the D=256 model family
    (fa-masked-gather-code-rev-qwen P1-2)."""
    dev = "cuda"
    torch.manual_seed(12)
    d = 128
    bpr = (d // 32) * 34
    b, sk = 3, 1024
    n_blocks = b * (sk // BLOCK)
    kv = torch.zeros(n_blocks, 2, BLOCK, HKV, d,
                     dtype=torch.float16, device=dev)
    key_cache, value_cache = kv.unbind(1)
    kv[:, 0].normal_(0, 0.5)
    kv[:, 1].normal_(0, 0.5)

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev)
    bt = bt.view(b, sk // BLOCK).contiguous()
    sl = torch.tensor([1, 300, 1000], dtype=torch.int32, device=dev)

    k_fused, v_fused = fa.gather_paged_kv_quantized(
        key_cache, value_cache, bt, sl, sk)
    k_p, v_p = fa.gather_paged_kv_quant_persistent(
        key_cache, value_cache, bt, sl, sk)
    assert k_p.shape == (b, HKV, sk, bpr)
    for s_ in range(b):
        L = int(sl[s_])
        assert torch.equal(k_p[s_, :, :L], k_fused[s_, :, :L]), f"K s={s_}"
        assert torch.equal(v_p[s_, :, :L], v_fused[s_, :, :L]), f"V s={s_}"


def test_fused_fp16_gather_matches_torch_gather():
    """LEGACY-path fused gather (gather_paged_kv_fp16) must match the torch
    _gather_kv reference in the valid region; V tail zeroed; K tail
    unmasked (FA kernel cuts via kv_max). Covers B=1 (Sk not a multiple
    of 32 — Sk_pad tail handling) and B=2 (per-row lengths and disjoint
    block ranges)."""
    dev = "cuda"
    torch.manual_seed(4)
    from vllm.gfx906_fa.gfx906_fa_paged import _gather_kv

    L = 500  # not a multiple of 32: exercises Sk_pad tail handling
    n_blocks = (L + BLOCK - 1) // BLOCK
    kc, vc, kv = _make_paged_cache(n_blocks, dev)
    k16 = torch.randn(n_blocks, BLOCK, HKV, D, device=dev,
                      dtype=torch.float16) * 0.5
    V = torch.randn(L, HKV, D, device=dev, dtype=torch.float16) * 0.5
    _write_v(kv, V)

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, -1)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    Sk_pad = (L + 31) // 32 * 32

    k_ref, v_ref = _gather_kv(k16, vc, bt, sl, L)
    k_f, v_f = fa.gather_paged_kv_fp16(k16, vc, bt, sl, Sk_pad)
    assert k_f.shape == (1, HKV, Sk_pad, D) and v_f.shape == (1, HKV, Sk_pad, D)
    assert torch.equal(k_f[0, :, :L], k_ref[0, :, :L])
    assert torch.equal(v_f[0, :, :L], v_ref[0, :, :L])
    assert bool((v_f[0, :, L:] == 0).all())

    # B=2: row 1 uses physical blocks disjoint from row 0's; different
    # lengths; unbind(1) strided views as in serving.
    L1, L2 = 300, 500
    n1 = (L1 + BLOCK - 1) // BLOCK
    n2 = (L2 + BLOCK - 1) // BLOCK
    width = n1 + n2
    kv2 = torch.randn(width, 2, BLOCK, HKV, D, device=dev,
                      dtype=torch.float16) * 0.5
    k16_2, vc2 = kv2.unbind(1)
    bt2 = torch.zeros(2, width, dtype=torch.int32, device=dev)
    bt2[0, :n1] = torch.arange(n1, dtype=torch.int32, device=dev)
    bt2[1, :n2] = torch.arange(n1, n1 + n2, dtype=torch.int32, device=dev)
    sl2 = torch.tensor([L1, L2], dtype=torch.int32, device=dev)
    k_ref2, v_ref2 = _gather_kv(k16_2, vc2, bt2, sl2, L2)
    k_f2, v_f2 = fa.gather_paged_kv_fp16(k16_2, vc2, bt2, sl2, Sk_pad)
    assert torch.equal(k_f2[0, :, :L1], k_ref2[0, :, :L1])
    assert torch.equal(v_f2[0, :, :L1], v_ref2[0, :, :L1])
    assert torch.equal(k_f2[1, :, :L2], k_ref2[1, :, :L2])
    assert torch.equal(v_f2[1, :, :L2], v_ref2[1, :, :L2])
    assert bool((v_f2[0, :, L1:] == 0).all())
    assert bool((v_f2[1, :, L2:] == 0).all())


def test_q_pad_buffer_survives_capture_then_prefill_grow(monkeypatch):
    """Review F1: a captured graph bakes in the VA of the q_pad buffer
    that was current at capture time. An eager prefill with a larger
    Sq_pad afterwards grows that buffer; the old one must be retired
    (kept alive) rather than freed-then-realloc'd, and the replayed
    decode must stay numerically correct. Drives the real Gfx906FAImpl
    (not hand-fed buffers) in the hazardous production order: small
    decode → capture → large prefill → decode replay.

    The q_pad buffers are CLASS-level (shared across all layer impls —
    the per-impl version was the boot J/K first-prefill OOM, see
    DEVLOG-muse-glimmer.md round 4), so this test snapshots/restores
    the class state like the gather-buffer lifecycle tests: the grow
    sequence must start from a clean shared buffer, and nothing leaks
    into other tests.

    The q_pad buffer belongs to the LEGACY=1 read path, so the env is pinned
    here (the side-buffer default has no such buffer).
    """
    monkeypatch.setenv("GFX906_FA_LEGACY", "1")
    dev = "cuda"
    torch.manual_seed(7)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    assert impl._legacy  # this test targets the default serving path
    cls = type(impl)

    saved = (cls._q_pad_buf, cls._q_pad_decode_buf, cls._q_pad_retired,
             cls._q_pad_captured)
    cls._q_pad_buf = None
    cls._q_pad_decode_buf = None
    cls._q_pad_retired = []
    cls._q_pad_captured = False
    try:
        n_blocks = 16  # 256 tokens
        _, vc, kv = _make_fused_cache(n_blocks, dev)
        k16 = _kv_split(kv)[0]
        K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
        _write_v_fused(_kv_split(kv)[1], V)

        def meta(num_tokens, sq, sk, bt_, sl_, cu_):
            return Gfx906FAMetadata(
                num_actual_tokens=num_tokens,
                max_query_len=sq,
                max_seq_len=sk,
                query_start_loc=cu_,
                seq_lens=sl_,
                block_table=bt_,
                slot_mapping=torch.empty(0, dtype=torch.int64,
                                         device=dev),
            )

        layer = None  # impl.forward does not touch the layer object
        s = torch.cuda.Stream()

        # (1) small decode (eager): allocates the small q_pad (Sq_pad=2)
        bt_d = torch.arange((100 + BLOCK - 1) // BLOCK, dtype=torch.int32,
                            device=dev).view(1, -1)
        sl_d = torch.tensor([100], dtype=torch.int32, device=dev)
        cu_d = torch.arange(2, dtype=torch.int32, device=dev)
        q_d = torch.randn(1, HQ, D, device=dev, dtype=torch.float16) * 0.5
        out_d = torch.zeros(1, HQ, D, device=dev, dtype=torch.float16)
        m_d = meta(1, 1, 100, bt_d, sl_d, cu_d)
        with torch.cuda.stream(s):
            for _ in range(2):
                impl.forward(layer, q_d, q_d, q_d, kv, m_d, output=out_d)
        torch.cuda.current_stream().wait_stream(s)
        ref_d = out_d.clone()
        small_buf = impl._q_pad_buf
        assert small_buf.shape[2] == 2

        # (2) capture the decode graph (bakes small_buf's VA in)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            impl.forward(layer, q_d, q_d, q_d, kv, m_d, output=out_d)
        assert impl._q_pad_captured
        assert impl._q_pad_buf is small_buf

        # (3) eager prefill with larger Sq_pad → grow branch
        q_p = torch.randn(64, HQ, D, device=dev, dtype=torch.float16) * 0.5
        out_p = torch.zeros(64, HQ, D, device=dev, dtype=torch.float16)
        m_p = meta(64, 64, 64,
                   torch.arange(4, dtype=torch.int32, device=dev).view(
                       1, -1),
                   torch.tensor([64], dtype=torch.int32, device=dev),
                   torch.tensor([0, 64], dtype=torch.int32, device=dev))
        impl.forward(layer, q_p, q_p, q_p, kv, m_p, output=out_p)
        assert impl._q_pad_buf is not small_buf
        assert impl._q_pad_buf.shape[2] == 64
        assert bool(torch.isfinite(out_p.float()).all())

        # (4) the captured buffer was retired, not freed
        assert any(t is small_buf for t in impl._q_pad_retired)
        assert small_buf.data_ptr() != impl._q_pad_buf.data_ptr()

        # (5) replay: the graph writes q_pad through the retired-but-alive
        # VA
        out_d.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert ((out_d - ref_d).norm() / ref_d.norm()).item() < 2e-2
    finally:
        (cls._q_pad_buf, cls._q_pad_decode_buf, cls._q_pad_retired,
         cls._q_pad_captured) = saved


def test_gather_buffers_lifecycle_postfix(monkeypatch):
    """plan-gfx906-fa-fix.md §5 — pins the POST-FIX gather-buffer
    contract (GFX906_FA_GATHER_EXACT=0) by driving the real
    Gfx906FAImpl (class-level buffers; snapshot/restore so nothing
    leaks into other tests):
    (1) Sk shrink / ping-pong (130 -> 100 -> 130): no realloc (Sk is a
        grow-only capacity), retire set unchanged;
    (4) descending capture sweep (B=2 then B=1, as get_capture_descs
        sorts): ONE generation, smaller-B reuses the base VA as a
        leading-dim slice, end-of-capture retired <= 1;
    (2) a captured generation is retired (kept alive) when replaced —
        driven directly: capture at width 128, then grow past it to 160;
    (5) the per-generation flag is reset, not OR'd: the eager
        generation (flag False) that follows a retired captured one is
        itself FREED on replacement;
    (3) eager growth (never captured) frees the old generation —
        allocator-level evidence: the delta is the new-minus-old
        excess, not the new generation on top of the old.

    Gather buffers exist only on the LEGACY=1 fp16 gather path, so the env is
    pinned here (the side-buffer default has none).
    """
    monkeypatch.setenv("GFX906_FA_LEGACY", "1")
    dev = "cuda"
    torch.manual_seed(11)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    assert impl._legacy  # this test targets the default serving path
    cls = type(impl)

    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured,
             cls._gather_exact, cls._gather_retired_warned)
    cls._k_gather_buf = cls._v_gather_buf = None
    cls._gather_retired = {}
    cls._gather_captured = False
    cls._gather_buf_captured = False
    cls._gather_exact = False
    cls._gather_retired_warned = False
    try:
        # 96 blocks so B=8 x 12-block tables stay in range (190 tokens
        # need 12 blocks of 16).
        n_blocks = 96
        _, vc, kv = _make_fused_cache(n_blocks, dev)
        k16 = _kv_split(kv)[0]
        K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
        _write_v_fused(_kv_split(kv)[1], V)

        def meta(b, sq, sk, nblk_per_seq):
            return Gfx906FAMetadata(
                num_actual_tokens=b * sq,
                max_query_len=sq,
                max_seq_len=sk,
                query_start_loc=torch.arange(
                    0, b * sq + 1, sq, dtype=torch.int32, device=dev),
                seq_lens=torch.full((b,), sk, dtype=torch.int32,
                                    device=dev),
                block_table=torch.arange(
                    b * nblk_per_seq, dtype=torch.int32,
                    device=dev).view(b, nblk_per_seq),
                slot_mapping=torch.empty(0, dtype=torch.int64,
                                         device=dev),
            )

        layer = None
        s = torch.cuda.Stream()

        def decode(b, sk, nblk, q, out, m=None):
            impl.forward(layer, q, q, q, kv,
                         m or meta(b, 1, sk, nblk), output=out)

        q1 = torch.randn(1, HQ, D, device=dev, dtype=torch.float16) * 0.5
        q2 = torch.randn(2, HQ, D, device=dev, dtype=torch.float16) * 0.5
        q4 = torch.randn(4, HQ, D, device=dev, dtype=torch.float16) * 0.5
        q8 = torch.randn(8, HQ, D, device=dev, dtype=torch.float16) * 0.5
        o1 = torch.zeros(1, HQ, D, device=dev, dtype=torch.float16)
        o2 = torch.zeros(2, HQ, D, device=dev, dtype=torch.float16)
        o4 = torch.zeros(4, HQ, D, device=dev, dtype=torch.float16)
        o8 = torch.zeros(8, HQ, D, device=dev, dtype=torch.float16)

        with torch.cuda.stream(s):
            # (1) eager growth to Sk_pad(130)=160, then ping-pong down
            # and up: capacity reuse, zero reallocs, empty retire set.
            decode(1, 130, 9, q1, o1)
            g1 = cls._k_gather_buf
            assert g1.shape[2] == 160
            decode(1, 100, 7, q1, o1)
            # sk=100 reference: this is the input the g1g graph replays.
            ref1 = o1.clone()
            assert cls._k_gather_buf is g1 and not cls._gather_retired
            decode(1, 130, 9, q1, o1)
            assert cls._k_gather_buf is g1 and not cls._gather_retired
            # Eager B-grow: gen1 never captured -> freed, not retired.
            # Exact-need sizing for freeable generations: the width is
            # NOT carried across (that would pin the B x Sk high-water
            # product — see test_gather_freeable_generation_exact_need).
            decode(2, 100, 7, q2, o2)
            ref2 = o2.clone()
            assert cls._k_gather_buf is not g1
            assert g1.data_ptr() not in cls._gather_retired
            g2 = cls._k_gather_buf
            assert g2.shape[:3] == (2, HKV, 128)
            assert not cls._gather_buf_captured
        torch.cuda.current_stream().wait_stream(s)

        # (4) descending capture sweep on the pre-allocated buffer (the
        # engine bakes the existing VA; no capture-time allocs). B=2
        # first (largest-first), then B=1 via the leading-dim slice.
        m2_cap = meta(2, 1, 100, 7)
        g2g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2g):
            decode(2, 100, 7, q2, o2, m2_cap)
        assert cls._k_gather_buf is g2
        assert cls._gather_buf_captured  # latched in the reuse path
        k_slice = cls._ensure_gather_buffers(1, HKV, 100, D,
                                             kv.device)[0]
        assert k_slice.data_ptr() == g2.data_ptr()
        assert cls._k_gather_buf is g2
        m1_cap = meta(1, 1, 100, 7)
        g1g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1g):
            decode(1, 100, 7, q1, o1, m1_cap)
        # End-of-capture: the sweep retired nothing (largest-first).
        assert len(cls._gather_retired) <= 1

        # (2) grow past the captured width (160): the captured generation
        # is retired (kept alive); the replacement starts uncaptured.
        with torch.cuda.stream(s):
            decode(2, 190, 12, q2, o2)
        torch.cuda.current_stream().wait_stream(s)
        g3 = cls._k_gather_buf
        assert g3.shape[2] == 192 and g3 is not g2
        assert g2.data_ptr() in cls._gather_retired
        assert not cls._gather_buf_captured  # reset, not OR'd

        # Replaying the B=2 graph hits the retired-but-alive base VA.
        o2.zero_()
        g2g.replay()
        torch.cuda.synchronize()
        assert ((o2 - ref2).norm() / ref2.norm()).item() < 2e-2
        o1.zero_()
        g1g.replay()
        torch.cuda.synchronize()
        assert ((o1 - ref1).norm() / ref1.norm()).item() < 2e-2

        # (5) capture gen3 (flag latches True); a B-grow retires it.
        # The following eager gen4 (flag False) is then FREED on its own
        # B-grow — the flag did not leak across the replacement.
        m2_cap190 = meta(2, 1, 190, 12)
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            decode(2, 190, 12, q2, o2, m2_cap190)
        assert cls._gather_buf_captured and cls._k_gather_buf is g3
        with torch.cuda.stream(s):
            decode(4, 190, 12, q4, o4)  # B-grow: retires g3 (captured)
        torch.cuda.current_stream().wait_stream(s)
        g4 = cls._k_gather_buf
        assert g4.shape[0] == 4 and g3.data_ptr() in cls._gather_retired
        assert not cls._gather_buf_captured
        with torch.cuda.stream(s):
            decode(8, 190, 12, q8, o8)  # B-grow: g4 (eager) freed
        torch.cuda.current_stream().wait_stream(s)
        assert g4.data_ptr() not in cls._gather_retired
        assert g3.data_ptr() in cls._gather_retired
        kept = {g.data_ptr() for g, _ in cls._gather_retired.values()}
        assert kept == {g2.data_ptr(), g3.data_ptr()}

        # (3) eager growth frees: allocator-level evidence. Fresh state.
        cls._k_gather_buf = cls._v_gather_buf = None
        cls._gather_retired = {}
        cls._gather_captured = False
        cls._gather_buf_captured = False
        with torch.cuda.stream(s):
            decode(1, 100, 7, q1, o1)
        torch.cuda.current_stream().wait_stream(s)
        # Record the VA + size but DO NOT hold the tensors — a live
        # Python reference would keep the generation alive and defeat
        # the allocator-level check (plan §5 item 3).
        gen_a_ptr = cls._k_gather_buf.data_ptr()
        gen_a_bytes = cls._k_gather_buf.numel() + cls._v_gather_buf.numel() * 2
        torch.cuda.synchronize()
        alloc_before = torch.cuda.memory_allocated()
        with torch.cuda.stream(s):
            decode(1, 130, 9, q1, o1)  # grow 128 -> 160
        torch.cuda.synchronize()
        delta = torch.cuda.memory_allocated() - alloc_before
        # If the old generation were kept alive the delta would be the
        # full new generation; freed, it is only the 160-vs-128 excess.
        assert gen_a_ptr not in cls._gather_retired
        assert delta < gen_a_bytes, (delta, gen_a_bytes)
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured,
         cls._gather_exact, cls._gather_retired_warned) = saved


def test_persistent_gather_fa_wide_buffer_poisoned_tail():
    """plan §5 — width >> live is safe end-to-end on the persistent
    path: the gather's work is live-bounded (device-side seq_lens) and
    the FA kernel cuts each sequence at kv_max, never at Sk. The stale
    region [seq_len, width) of the class buffer is poisoned with huge
    q8 bytes / fp16 NaN (exactly the data that would leak through if
    the tail mask were ever broken); the FA output must be bit-equal
    to the exact-width reference. Mixed seq_lens [37, 1000] exercise
    the per-seq margin clamp at num_seqs > 1; a second uniform
    multi-token (MTP-draft-shaped) forward reuses the same wide buffer.
    """
    dev = "cuda"
    torch.manual_seed(13)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    cls = type(impl)
    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured)
    try:
        B, seq_lens, width = 2, [37, 1000], 4096
        nblk = (seq_lens[-1] + BLOCK - 1) // BLOCK  # 63
        n_blocks = B * nblk + 4
        _, vc, kv = _make_fused_cache(n_blocks, dev)
        k16 = _kv_split(kv)[0]
        K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
        _write_v_fused(_kv_split(kv)[1], V)

        def meta(b, sq, sk, sl_, bt_):
            return Gfx906FAMetadata(
                num_actual_tokens=b * sq,
                max_query_len=sq,
                max_seq_len=sk,
                query_start_loc=torch.arange(
                    0, b * sq + 1, sq, dtype=torch.int32, device=dev),
                seq_lens=sl_,
                block_table=bt_,
                slot_mapping=torch.empty(0, dtype=torch.int64,
                                         device=dev),
            )

        bt = torch.arange(B * nblk, dtype=torch.int32,
                          device=dev).view(B, nblk)
        layer = None
        q = torch.randn(B, HQ, D, device=dev, dtype=torch.float16) * 0.5
        out = torch.zeros(B, HQ, D, device=dev, dtype=torch.float16)

        def reset_class():
            cls._k_gather_buf = cls._v_gather_buf = None
            cls._gather_retired = {}
            cls._gather_captured = False
            cls._gather_buf_captured = False

        # Reference: exact-width class buffer (fresh allocation).
        reset_class()
        sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
        m = meta(B, 1, seq_lens[-1], sl, bt)
        impl.forward(layer, q, q, q, kv, m, output=out)
        ref = out.clone()
        assert cls._k_gather_buf.shape[2] == 1024  # Sk_pad(1000)

        # Wide buffer with the stale region poisoned.
        bpr = (D // 32) * 34
        kw = torch.empty(B, HKV, width, bpr, dtype=torch.uint8,
                         device=dev)
        vw = torch.empty(B, HKV, width, D, dtype=torch.float16,
                         device=dev)
        for s_i, L in enumerate(seq_lens):
            kw[s_i, :, L:, :].fill_(0x7F)  # huge q8 scale/data bytes
            vw[s_i, :, L:, :].fill_(float("nan"))
        cls._k_gather_buf, cls._v_gather_buf = kw, vw
        cls._gather_buf_captured = False
        out.zero_()
        impl.forward(layer, q, q, q, kv, m, output=out)
        assert torch.equal(out, ref), (
            "poisoned wide buffer changed the FA output — tail mask "
            "leak or width-bound work on the persistent path")

        # MTP-draft-shaped second forward: multi-token queries, longer
        # seq_lens, same wide buffer (capacity reuse, no realloc).
        sl2 = torch.tensor([40, 1003], dtype=torch.int32, device=dev)
        qd = torch.randn(2 * 3, HQ, D, device=dev,
                         dtype=torch.float16) * 0.5
        outd = torch.zeros(2 * 3, HQ, D, device=dev, dtype=torch.float16)
        m2 = meta(2, 3, 1003, sl2, bt)
        reset_class()
        impl.forward(layer, qd, qd, qd, kv, m2, output=outd)
        refd = outd.clone()
        assert cls._k_gather_buf.shape[2] == 1024
        for s_i, L in enumerate([40, 1003]):
            kw[s_i, :, L:, :].fill_(0x7F)
            vw[s_i, :, L:, :].fill_(float("nan"))
        cls._k_gather_buf, cls._v_gather_buf = kw, vw
        cls._gather_buf_captured = False
        outd.zero_()
        impl.forward(layer, qd, qd, qd, kv, m2, output=outd)
        assert torch.equal(outd, refd), (
            "poisoned wide buffer changed the draft-shaped FA output")
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured) = saved


def test_wide_buffer_b17_fused_quant_no_leak():
    """plan §2.2c/§5 — num_seqs > _PERSIST_MAX_SEQS with a WIDE class
    buffer: the fused-quant path keeps the exact contract, so it must
    still work (no crash, finite output, wide buffer untouched) and
    must not accumulate its per-call C++ allocations across steps
    (they are freed after each call — no reuse, but no leak either).
    """
    dev = "cuda"
    torch.manual_seed(17)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    cls = type(impl)
    B = 17
    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured)
    try:
        sk = 64
        nblk = (sk + BLOCK - 1) // BLOCK  # 4
        n_blocks = B * nblk + 4
        _, vc, kv = _make_fused_cache(n_blocks, dev)
        k16 = _kv_split(kv)[0]
        K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
        _write_v_fused(_kv_split(kv)[1], V)

        m = Gfx906FAMetadata(
            num_actual_tokens=B,
            max_query_len=1,
            max_seq_len=sk,
            query_start_loc=torch.arange(B + 1, dtype=torch.int32,
                                         device=dev),
            seq_lens=torch.full((B,), sk, dtype=torch.int32, device=dev),
            block_table=torch.arange(B * nblk, dtype=torch.int32,
                                     device=dev).view(B, nblk),
            slot_mapping=torch.empty(0, dtype=torch.int64, device=dev))
        bpr = (D // 32) * 34
        cls._k_gather_buf = torch.empty(B, HKV, 1024, bpr,
                                        dtype=torch.uint8, device=dev)
        cls._v_gather_buf = torch.empty(B, HKV, 1024, D,
                                        dtype=torch.float16, device=dev)
        cls._gather_retired = {}
        cls._gather_captured = False
        cls._gather_buf_captured = False

        q = torch.randn(B, HQ, D, device=dev, dtype=torch.float16) * 0.5
        out = torch.zeros(B, HQ, D, device=dev, dtype=torch.float16)
        impl.forward(None, q, q, q, kv, m, output=out)
        assert bool(torch.isfinite(out.float()).all())
        # The wide buffer was refused by the exact-contract site
        # (no reuse, no retire, no consumption).
        assert cls._k_gather_buf.shape[2] == 1024
        assert not cls._gather_retired

        # No accumulation across steps: per-call C++ allocs are freed.
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for _ in range(3):
            impl.forward(None, q, q, q, kv, m, output=out)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() - before < 16 * 2**20
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured) = saved


def test_gather_exact_killswitch_restores_old_policy():
    """plan §5 — GFX906_FA_GATHER_EXACT=1 (cls._gather_exact here; the
    in-service A/B sets the env var, read at import by both files)
    restores the pre-fix behavior: exact-Sk realloc and the sticky
    _gather_captured latch, so an Sk ping-pong after capture retires
    EVERY generation — the unbounded growth this fix removes. Pinned
    so the kill switch stays a true A/B arm.
    Drop this test together with the switch itself at the next
    gather-lifecycle change (plan §6 lifecycle note).
    """
    dev = "cuda"
    torch.manual_seed(19)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    cls = type(impl)
    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured,
             cls._gather_exact, cls._gather_retired_warned)
    cls._k_gather_buf = cls._v_gather_buf = None
    cls._gather_retired = {}
    cls._gather_captured = False
    cls._gather_buf_captured = False
    cls._gather_exact = True
    try:
        n_blocks = 32
        _, vc, kv = _make_fused_cache(n_blocks, dev)
        k16 = _kv_split(kv)[0]
        K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
        _write_v_fused(_kv_split(kv)[1], V)

        def m(sk):
            return Gfx906FAMetadata(
                num_actual_tokens=1, max_query_len=1, max_seq_len=sk,
                query_start_loc=torch.arange(2, dtype=torch.int32,
                                             device=dev),
                seq_lens=torch.tensor([sk], dtype=torch.int32,
                                      device=dev),
                block_table=torch.arange(9, dtype=torch.int32,
                                         device=dev).view(1, 9),
                slot_mapping=torch.empty(0, dtype=torch.int64,
                                         device=dev))

        q1 = torch.randn(1, HQ, D, device=dev, dtype=torch.float16) * 0.5
        o1 = torch.zeros(1, HQ, D, device=dev, dtype=torch.float16)
        # Eager forward: allocs gen1 [1, HKV, 128] and grows q_pad.
        impl.forward(None, q1, q1, q1, kv, m(100), output=o1)
        gen1 = cls._k_gather_buf
        assert gen1.shape[2] == 128
        # Capture reuses gen1 and latches the sticky latch. Metadata is
        # pre-built (capture-time allocations/copies are illegal).
        m100 = m(100)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            impl.forward(None, q1, q1, q1, kv, m100, output=o1)
        assert cls._gather_captured
        # Sk ping-pong: every replacement retires (sticky latch) — the
        # pre-fix unbounded behavior, kept byte-for-byte by the switch.
        cls._ensure_gather_buffers(1, HKV, 130, D, kv.device)
        assert gen1.data_ptr() in cls._gather_retired
        gen2 = cls._k_gather_buf
        cls._ensure_gather_buffers(1, HKV, 100, D, kv.device)
        assert gen2.data_ptr() in cls._gather_retired
        assert len(cls._gather_retired) == 2
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured,
         cls._gather_exact, cls._gather_retired_warned) = saved


def test_gather_freeable_generation_exact_need():
    """plan §2.4 follow-up — a never-captured (freeable) generation is
    replaced at EXACT need, not grow-only max() per axis: 32-seq
    short-context decode followed by one long prefill must NOT leave a
    [32, wide] standing buffer (the B-highwater x Sk-highwater product,
    ~13 GB/rank at 256k on the arm-B geometry). Realloc frequency is
    unchanged by construction (a replacement happens exactly when the
    current buffer no longer fits); only the new allocation's shape is
    pinned here.
    """
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(29)
    from vllm.gfx906_fa.gfx906_fa_backend import Gfx906FAImpl

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    cls = type(impl)
    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured,
             cls._gather_exact, cls._gather_retired_warned)
    cls._k_gather_buf = cls._v_gather_buf = None
    cls._gather_retired = {}
    cls._gather_captured = False
    cls._gather_buf_captured = False
    cls._gather_exact = False
    cls._gather_retired_warned = False
    bpr = (D // 32) * 34
    try:
        k, _ = cls._ensure_gather_buffers(32, HKV, 100, D, dev)
        assert k.shape == (32, HKV, 128, bpr)
        # Sk grow at B=1: freeable -> exact need, NOT [32, 1024].
        k, _ = cls._ensure_gather_buffers(1, HKV, 1000, D, dev)
        assert k.shape == (1, HKV, 1024, bpr)
        assert not cls._gather_retired
        # B grow at small Sk: likewise exact, NOT [32, 1024].
        k, _ = cls._ensure_gather_buffers(32, HKV, 100, D, dev)
        assert k.shape == (32, HKV, 128, bpr)
        assert not cls._gather_retired
        # Within-capacity requests never realloc (the fit path).
        k2, _ = cls._ensure_gather_buffers(8, HKV, 100, D, dev)
        assert k2.shape == (8, HKV, 128, bpr)
        assert k2.data_ptr() == k.data_ptr()
        assert not cls._gather_retired
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured,
         cls._gather_exact, cls._gather_retired_warned) = saved


def test_gather_multi_retire_warns(monkeypatch):
    """plan §2.2b guard — retiring MORE than one capture-baked gather
    generation warns, one-shot. The original guard was dead code: it
    required `not capturing` while checking a flag that had just been
    set to `capturing`, so it could never fire. Two capture-then-B-grow
    cycles drive two retires; the third retire must not warn again.

    The gather-retire guard only exists on the LEGACY=1 fp16 gather path.
    """
    monkeypatch.setenv("GFX906_FA_LEGACY", "1")
    import vllm.gfx906_fa.gfx906_fa_backend as backend_mod

    class _LoggerStub:
        def __init__(self):
            self.warnings = []

        def warning(self, msg, *args):
            self.warnings.append(msg % args if args else msg)

    stub = _LoggerStub()
    monkeypatch.setattr(backend_mod, "logger", stub)

    dev = "cuda"
    torch.manual_seed(31)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    cls = type(impl)
    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured,
             cls._gather_exact, cls._gather_retired_warned)
    cls._k_gather_buf = cls._v_gather_buf = None
    cls._gather_retired = {}
    cls._gather_captured = False
    cls._gather_buf_captured = False
    cls._gather_exact = False
    cls._gather_retired_warned = False
    try:
        n_blocks = 128  # 16 seqs x 7 blocks
        _, vc, kv = _make_fused_cache(n_blocks, dev)
        k16 = _kv_split(kv)[0]
        K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
        _write_v_fused(_kv_split(kv)[1], V)

        def meta(b, sk, nblk):
            return Gfx906FAMetadata(
                num_actual_tokens=b,
                max_query_len=1,
                max_seq_len=sk,
                query_start_loc=torch.arange(
                    0, b + 1, dtype=torch.int32, device=dev),
                seq_lens=torch.full((b,), sk, dtype=torch.int32,
                                    device=dev),
                block_table=torch.arange(
                    b * nblk, dtype=torch.int32, device=dev).view(b, nblk),
                slot_mapping=torch.empty(0, dtype=torch.int64,
                                         device=dev),
            )

        layer = None
        s = torch.cuda.Stream()

        def decode(b, q, out, m=None):
            impl.forward(layer, q, q, q, kv, m or meta(b, 100, 7),
                         output=out)

        qs = {b: torch.randn(b, HQ, D, device=dev,
                             dtype=torch.float16) * 0.5 for b in (2, 4, 8, 16)}
        os_ = {b: torch.zeros(b, HQ, D, device=dev,
                              dtype=torch.float16) for b in (2, 4, 8, 16)}

        # Cycle 1: eager gen1, capture bakes it, B-grow retires it.
        with torch.cuda.stream(s):
            decode(2, qs[2], os_[2])
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            decode(2, qs[2], os_[2], meta(2, 100, 7))
        assert cls._gather_buf_captured
        with torch.cuda.stream(s):
            decode(4, qs[4], os_[4])
        torch.cuda.current_stream().wait_stream(s)
        assert len(cls._gather_retired) == 1 and not stub.warnings

        # Cycle 2: second capture bakes gen2, B-grow retires it -> len 2.
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            decode(4, qs[4], os_[4], meta(4, 100, 7))
        assert cls._gather_buf_captured
        with torch.cuda.stream(s):
            decode(8, qs[8], os_[8])
        torch.cuda.current_stream().wait_stream(s)
        assert len(cls._gather_retired) == 2
        assert len(stub.warnings) == 1
        assert "capture-baked gather" in stub.warnings[0]

        # Cycle 3: one-shot — a third retire must not warn again.
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            decode(8, qs[8], os_[8], meta(8, 100, 7))
        with torch.cuda.stream(s):
            decode(16, qs[16], os_[16])
        torch.cuda.current_stream().wait_stream(s)
        assert len(cls._gather_retired) == 3
        assert len(stub.warnings) == 1
        assert bool(torch.isfinite(os_[16].float()).all())
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured,
         cls._gather_exact, cls._gather_retired_warned) = saved


def test_gather_mixed_width_buffers_not_reused():
    """plan §2.2c follow-up — the persistent branch's k/v capacity
    reuse requires EQUAL widths. A hand-set class buffer pair with
    unequal K/V widths (impossible via _ensure_gather_buffers, which
    allocates the pair at one width) must NOT be half-reused: without
    the width check, Sk = K's width would pass V by a width mismatch to
    the C++ exact-match check and silently drop V to a per-call
    allocation. The forward must fall back whole (bitwise-identical
    output, class buffers untouched).
    """
    dev = "cuda"
    torch.manual_seed(37)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    cls = type(impl)
    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured)
    try:
        B, sk = 2, 100
        nblk = 7
        n_blocks = B * nblk + 4
        _, vc, kv = _make_fused_cache(n_blocks, dev)
        k16 = _kv_split(kv)[0]
        K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
        _write_v_fused(_kv_split(kv)[1], V)
        m = Gfx906FAMetadata(
            num_actual_tokens=B,
            max_query_len=1,
            max_seq_len=sk,
            query_start_loc=torch.arange(
                0, B + 1, dtype=torch.int32, device=dev),
            seq_lens=torch.full((B,), sk, dtype=torch.int32, device=dev),
            block_table=torch.arange(
                B * nblk, dtype=torch.int32, device=dev).view(B, nblk),
            slot_mapping=torch.empty(0, dtype=torch.int64, device=dev),
        )
        q = torch.randn(B, HQ, D, device=dev, dtype=torch.float16) * 0.5
        out = torch.zeros(B, HQ, D, device=dev, dtype=torch.float16)

        # Reference: fresh exact-width class buffer.
        cls._k_gather_buf = cls._v_gather_buf = None
        cls._gather_retired = {}
        cls._gather_captured = False
        cls._gather_buf_captured = False
        impl.forward(None, q, q, q, kv, m, output=out)
        ref = out.clone()
        assert cls._k_gather_buf.shape[2] == 128  # Sk_pad(100)

        # Unequal-width pair (K wide, V exact): must be refused whole.
        bpr = (D // 32) * 34
        cls._k_gather_buf = torch.empty(B, HKV, 1024, bpr,
                                        dtype=torch.uint8, device=dev)
        cls._v_gather_buf = torch.empty(B, HKV, 128, D,
                                        dtype=torch.float16, device=dev)
        cls._gather_retired = {}
        cls._gather_buf_captured = False
        out.zero_()
        impl.forward(None, q, q, q, kv, m, output=out)
        assert torch.equal(out, ref), (
            "mixed-width class buffers changed the FA output — the "
            "persistent branch half-reused an unequal-width pair")
        # The pair was refused, not consumed or replaced.
        assert cls._k_gather_buf.shape[2] == 1024
        assert cls._v_gather_buf.shape[2] == 128
        assert not cls._gather_retired
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured) = saved


def test_forward_decode_prefill_vs_sdpa_on_unbind_cache():
    dev = "cuda"
    torch.manual_seed(2)
    L = 512
    n_blocks = L // BLOCK
    kc, vc, kv = _make_paged_cache(n_blocks, dev)
    K = torch.randn(L, HKV, D, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, HKV, D, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    _write_v(kv, V)

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(D)
    k_q8, v_b = fa.gather_paged_kv_q8(kc, vc, bt, sl, (L + 31) // 32 * 32)
    g = HQ // HKV
    k, v = K.float(), V.float()

    # decode: one query, no causal
    q = torch.randn(1, HQ, 1, D, device=dev, dtype=torch.float32) * 0.5
    out = fa.forward(q, k_q8, v_b, scale, kv_max=sl)[0, 0]  # [HQ, D] (BSHD)
    qg = q[0, :, 0].view(HKV, g, D)
    s = torch.einsum("gjd,lgd->gjl", qg, k) * scale
    ref = torch.einsum("gjl,lgd->gjd", torch.softmax(s, -1), v).reshape(HQ, D)
    assert ((out - ref).norm() / ref.norm()).item() < 5e-2

    # prefill: full causal chunk
    qf = torch.randn(1, HQ, L, D, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([0], dtype=torch.int32, device=dev)
    outf = fa.forward(qf, k_q8, v_b, scale, kv_max=sl, q_abs_offset=q_abs)[0]
    # native BSHD: [0] is already [L, HQ, D]
    assert outf.shape == (L, HQ, D)
    qtok = qf[0].permute(1, 0, 2).float()  # [L, HQ, D]
    for t in (1, 63, L - 1):
        qg = qtok[t].view(HKV, g, D)
        s = torch.einsum("gjd,lgd->gjl", qg, k[: t + 1]) * scale
        ref = torch.einsum(
            "gjl,lgd->gjd", torch.softmax(s, -1), v[: t + 1]
        ).reshape(HQ, D)
        assert ((outf[t] - ref).norm() / ref.norm()).item() < 5e-2


# The NC2/KVSPLIT knobs are parsed once per process (C++ statics on the
# first forward call), so each config runs in a fresh subprocess.
_SPLIT_CHECK_SRC = """
import math
import sys
import torch

nc2 = int(sys.argv[1])
ysplit = int(sys.argv[2])
sk = int(sys.argv[3])
kv_max = int(sys.argv[4])

torch.manual_seed(0)
dev = "cuda"
HQ, HKV, D = int(sys.argv[5]), int(sys.argv[6]), 256

from vllm import _gfx906_fa_C as fa

k16 = torch.randn(1, HKV, sk, D, device=dev, dtype=torch.float16) * 0.5
v16 = torch.randn(1, HKV, sk, D, device=dev, dtype=torch.float16) * 0.5
q32 = torch.randn(1, HQ, 1, D, device=dev, dtype=torch.float32) * 0.5
k_q8 = fa.quantize_q8_0(k16)
sl = torch.tensor([kv_max], dtype=torch.int32, device=dev)
out = fa.forward(q32, k_q8, v16, 1.0 / math.sqrt(D), kv_max=sl)[0, 0]  # BSHD

g = HQ // HKV
k, v = k16[0].float(), v16[0].float()  # [HKV, sk, D]
qg = q32[0, :, 0].view(HKV, g, D)
s = torch.einsum("gjd,gld->gjl", qg, k[:, :kv_max]) * (1.0 / math.sqrt(D))
ref = torch.einsum(
    "gjl,gld->gjd", torch.softmax(s, -1), v[:, :kv_max]
).reshape(HQ, D)
rel = ((out - ref).norm() / ref.norm()).item()
print(f"nc2={nc2} ys={ysplit} sk={sk} kv_max={kv_max} rel={rel:.2e}")
sys.exit(0 if rel < 5e-2 else 1)
"""


@pytest.mark.parametrize("nc2, ys, sk, kv_max, hq, hkv", [
    (1, 1, 512, 512, 16, 2),    # legacy path, no split (sanity in-subprocess)
    (1, 4, 512, 512, 16, 2),    # KV-split only
    (1, 4, 512, 481, 16, 2),    # KV-split with empty trailing splits (kv_max<sk)
    (8, 1, 512, 512, 16, 2),    # GQA head-packing only (no combine)
    (8, 16, 512, 512, 16, 2),   # serving config: GQA pack + KV-split
    (8, 16, 123, 123, 16, 2),   # short Sk: more splits than KV tiles
    (8, 16, 512, 481, 16, 2),   # serving config + empty trailing splits
    # heads_q=6 under default nc2=8 (e.g. Qwen3.5-27B at TP=4): the bare
    # heads_q%nc2 guard used to abort before the 8->2 downgrade; must now
    # downgrade to nc2=2 (ratio 6 % 2 == 0) and produce correct output.
    (8, 16, 512, 512, 6, 1),
    # heads_q=6 per-shard ratio with actual GQA (Hq=6/Hkv=3, ratio 2).
    (8, 16, 512, 512, 6, 3),
    # M4 (qwen #4a): long-context split accuracy — production gather
    # default (kv_split=16) and the no-split baseline at 16k vs fp32 ref.
    (1, 16, 16384, 16384, 16, 2),
    (1, 1, 16384, 16384, 16, 2),
])
def test_forward_kv_split_gqa_pack_vs_fp32_ref(nc2, ys, sk, kv_max, hq, hkv):
    import os
    import subprocess

    env = {
        **os.environ,
        "GFX906_FA_NC2": str(nc2),
        "GFX906_FA_KVSPLIT": str(ys),
    }
    r = subprocess.run(
        [sys.executable, "-c", _SPLIT_CHECK_SRC, str(nc2), str(ys),
         str(sk), str(kv_max), str(hq), str(hkv)],
        env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr[-2000:]}"


# ---------------------------------------------------------------------------
# MTP-1b-0 (kv_split byte-budget guard): multi-query (Sq>1) causal attention.
# The old hard clamp `if (seq_q > 2) kv_split = 1` meant the split-KV combine
# path was NEVER exercised at Sq>2 — exactly the spec-decode verify shapes
# (k+1 queries padded to {2,4,8}). This covers those shapes plus a prefill
# shape that stays under the byte budget, on the gather path with causal
# masking, vs an fp32 torch reference, across kv_split settings.
# ---------------------------------------------------------------------------
_SQ_SPLIT_CHECK_SRC = """
import math
import sys
import torch

nc2 = int(sys.argv[1])
ysplit = int(sys.argv[2])
sq = int(sys.argv[3])
L = int(sys.argv[4])
hq, hkv = int(sys.argv[5]), int(sys.argv[6])
D = 256
BLOCK = 16

torch.manual_seed(0)
dev = "cuda"
scale = 1.0 / math.sqrt(D)

from vllm import _gfx906_fa_C as fa

n_blocks = L // BLOCK
kc = torch.zeros(n_blocks, BLOCK, hkv, (D // 32) * 34,
                 dtype=torch.uint8, device=dev)
kv = torch.zeros(n_blocks, 2, BLOCK, hkv, D, dtype=torch.float16, device=dev)
K = torch.randn(L, hkv, D, device=dev, dtype=torch.float16) * 0.5
V = torch.randn(L, hkv, D, device=dev, dtype=torch.float16) * 0.5
slot = torch.arange(L, dtype=torch.int64, device=dev)
fa.reshape_and_cache_q8(K, slot, kc)
staging = torch.zeros_like(kv[:, 1])
staging.view(-1, hkv, D)[:L].copy_(V)
kv[:, 1].copy_(staging)
vc = kv.unbind(1)[1]
bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
sl = torch.tensor([L], dtype=torch.int32, device=dev)

# verify-style: the LAST sq rows of the sequence (q_abs_offset = L - sq),
# causal. Sq in {2,3,4} are k+1 spec-decode verify shapes; 128/256 are
# prefill shapes that stay under / exceed the kv_split byte budget.
q = torch.randn(1, hq, sq, D, device=dev, dtype=torch.float32) * 0.5
q_abs = torch.tensor([L - sq], dtype=torch.int32, device=dev)
k_q8, v_b = fa.gather_paged_kv_q8(kc, vc, bt, sl, (L + 31) // 32 * 32)
out = fa.forward(q, k_q8, v_b, scale, kv_max=sl, q_abs_offset=q_abs)[0]

# torch reference: query row i sits at abs pos L-sq+i -> causal to j <= that.
g = hq // hkv
k, v = K.float(), V.float()
rows = torch.arange(L, device=dev)
ref = torch.empty(sq, hq, D, device=dev, dtype=torch.float32)
for i in range(sq):
    qpos = L - sq + i
    per_h = []
    for h in range(hq):
        hk = h // g
        s = (q[0, h, i].float() @ k[:, hk].T * scale)
        s = torch.where(rows > qpos, torch.full_like(s, float("-inf")), s)
        per_h.append(torch.softmax(s, -1) @ v[:, hk])
    ref[i] = torch.stack(per_h, 0)

rel = ((out.float() - ref).norm() / ref.norm()).item()
print(f"nc2={nc2} ys={ysplit} sq={sq} L={L} rel={rel:.2e}")
sys.exit(0 if rel < 5e-2 else 1)
"""


@pytest.mark.parametrize("ys, sq, L, hq, hkv, budget", [
    # spec-decode verify shapes: the exact path the clamp used to kill.
    (8, 2, 512, 16, 2, None),    # k=1 verify
    (8, 3, 512, 16, 2, None),    # k=2 verify (pads to 4)
    (8, 4, 512, 16, 2, None),    # k=3 verify
    (16, 3, 512, 16, 2, None),   # serving gather default split
    # prefill shape under the 512 MiB byte budget: kv_split>1 allowed.
    (8, 128, 512, 16, 2, None),
    # above the default budget -> clamped to y=1; must still be correct.
    (16, 256, 512, 16, 2, None),   # prefill under default budget: split kept
    # same shape with the budget pinned BELOW the partial buffer
    # (1*256*16*16*256*4 = 64 MiB > 32 MiB) -> clamped to y=1; must still
    # be correct. Pins the guard without needing a multi-thousand-Sq shape.
    (16, 256, 512, 16, 2, 32 * 1024**2),
])
def test_forward_sq_multi_kv_split_vs_fp32_ref(ys, sq, L, hq, hkv, budget):
    import os
    import subprocess

    env = {
        **os.environ,
        "GFX906_FA_NC2": "8",
        "GFX906_FA_KVSPLIT": str(ys),
    }
    if budget is not None:
        env["GFX906_FA_KVSPLIT_MAX_BYTES"] = str(budget)
    r = subprocess.run(
        [sys.executable, "-c", _SQ_SPLIT_CHECK_SRC, "8", str(ys),
         str(sq), str(L), str(hq), str(hkv)],
        env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr[-2000:]}"


# ---------------------------------------------------------------------------
# Sliding-window masking (window arg, Muse Glimmer iRoPE track).
# Unmasked keys per query row r: [max(0, r - W + 1), r] (window + causal).
# q_abs_offset is required for the window to apply (the backend always
# passes it for windowed batches, decode included).
# ---------------------------------------------------------------------------

def _windowed_ref(q_row, K, V, scale, r, W):
    """Torch reference for one query row at absolute position r.

    q_row: [Hq, D]; K/V: [HKV, L, D] fp32. Window W in tokens (W=None ->
    plain causal).
    """
    lo = max(0, r - W + 1) if W else 0
    hi = r + 1
    hkv = K.shape[1]
    g = q_row.shape[0] // hkv
    qg = q_row.view(hkv, g, -1)
    s = torch.einsum("gjd,lgd->gjl", qg, K[lo:hi]) * scale
    o = torch.einsum("gjl,lgd->gjd", torch.softmax(s, -1), V[lo:hi])
    return o.reshape(q_row.shape[0], -1)


@pytest.mark.parametrize("d, hq, hkv, L, W", [
    (128, 32, 2, 512, 128),   # Muse Glimmer shape (Hq 32 / Hkv 2, D 128)
    (128, 32, 2, 512, 64),    # smaller window
    (128, 32, 2, 96, 128),    # W > L: window must be inert
    (256, 16, 2, 512, 128),   # D=256 path
])
def test_forward_sliding_window_vs_torch_ref(d, hq, hkv, L, W):
    dev = "cuda"
    torch.manual_seed(3)
    n_blocks = L // BLOCK
    kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                     dtype=torch.uint8, device=dev)
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                     dtype=torch.float16, device=dev)
    K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[:L].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(d)
    sk_pad = (L + 31) // 32 * 32
    k_q8, v_b = fa.gather_paged_kv_q8(kc, vc, bt, sl, sk_pad)
    Kf, Vf = K.float(), V.float()

    # decode (Sq=1): q_abs_offset = L-1; window clips to [L-W, L-1]
    q = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([L - 1], dtype=torch.int32, device=dev)
    out = fa.forward(q, k_q8, v_b, scale, kv_max=sl,
                     q_abs_offset=q_abs, window=W)[0, 0]
    ref = _windowed_ref(q[0, :, 0], Kf, Vf, scale, L - 1, W)
    assert ((out - ref).norm() / ref.norm()).item() < 5e-2
    if L > W:
        # window must actually bite: differs from no-window attention
        out_nw = fa.forward(q, k_q8, v_b, scale, kv_max=sl,
                            q_abs_offset=q_abs)[0, 0]
        assert (out - out_nw).norm().item() > 1e-3
    else:
        # W >= L: windowed output matches no-window (inert)
        out_nw = fa.forward(q, k_q8, v_b, scale, kv_max=sl,
                            q_abs_offset=q_abs)[0, 0]
        assert ((out - out_nw).norm() / out.norm()).item() < 5e-2

    # prefill (Sq=L): per-row causal + window
    qf = torch.randn(1, hq, L, d, device=dev, dtype=torch.float32) * 0.5
    q_abs0 = torch.tensor([0], dtype=torch.int32, device=dev)
    outf = fa.forward(qf, k_q8, v_b, scale, kv_max=sl,
                      q_abs_offset=q_abs0, window=W)[0]
    assert outf.shape == (L, hq, d)
    qtok = qf[0].permute(1, 0, 2).float()  # [L, hq, d]
    rows = sorted({t for t in (0, W - 1, W, L - W, L - 1) if 0 <= t < L})
    for t in rows:
        ref = _windowed_ref(qtok[t], Kf, Vf, scale, t, W)
        assert ((outf[t] - ref).norm() / ref.norm()).item() < 5e-2, \
            f"row {t}"


# ---------------------------------------------------------------------------
# Same window semantics, DIRECT-PAGED kernel (forward_paged_direct,
# fattn-q8-paged.cuh) — the kernel the serving gate actually runs:
# _should_use_direct_paged is "auto" (min_batch=2), so every decode batch
# >= 2 (incl. the BENCH_MAX_SEQS=4 gate config) routes here, and the
# window formula is hand-duplicated in this file. Both files must match
# the torch reference.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d, hq, hkv, L, W", [
    (128, 32, 2, 512, 128),   # Muse Glimmer shape (Hq 32 / Hkv 2, D 128)
    (128, 32, 2, 512, 64),    # smaller window
    (128, 32, 2, 96, 128),    # W > L: window must be inert
    (256, 16, 2, 512, 128),   # D=256 path
])
def test_forward_paged_direct_sliding_window_vs_torch_ref(d, hq, hkv, L, W):
    dev = "cuda"
    torch.manual_seed(4)
    n_blocks = L // BLOCK
    kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                     dtype=torch.uint8, device=dev)
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                     dtype=torch.float16, device=dev)
    K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[:L].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]  # production layout: unbind(1), non-contiguous

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(d)
    Kf, Vf = K.float(), V.float()

    # decode B=1 (Sq=1): q_abs_offset = L-1; window clips to [L-W, L-1]
    q = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([L - 1], dtype=torch.int32, device=dev)
    out = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, W)[0, 0]
    ref = _windowed_ref(q[0, :, 0], Kf, Vf, scale, L - 1, W)
    assert ((out - ref).norm() / ref.norm()).item() < 5e-2
    if L > W:
        # window must actually bite: differs from no-window attention
        out_nw = fa.forward_paged_direct(
            q, kc, vc, bt, sl, scale, None, q_abs)[0, 0]
        assert (out - out_nw).norm().item() > 1e-3
    else:
        # W >= L: windowed output matches no-window (inert)
        out_nw = fa.forward_paged_direct(
            q, kc, vc, bt, sl, scale, None, q_abs)[0, 0]
        assert ((out - out_nw).norm() / out.norm()).item() < 5e-2

    # decode B=2, different lengths (the production direct mode): per-row
    # q_abs_offset/window must hold per batch element. seq1 reuses the
    # same physical blocks with a shorter kv_max.
    L2 = L * 3 // 4
    bt2 = bt.repeat(2, 1)
    sl2 = torch.tensor([L, L2], dtype=torch.int32, device=dev)
    q_abs2 = torch.tensor([L - 1, L2 - 1], dtype=torch.int32, device=dev)
    q2 = torch.randn(2, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    out2 = fa.forward_paged_direct(
        q2, kc, vc, bt2, sl2, scale, None, q_abs2, W)
    assert out2.shape == (2, 1, hq, d)
    ref0 = _windowed_ref(q2[0, :, 0], Kf, Vf, scale, L - 1, W)
    ref1 = _windowed_ref(q2[1, :, 0], Kf, Vf, scale, L2 - 1, W)
    assert ((out2[0, 0] - ref0).norm() / ref0.norm()).item() < 5e-2
    assert ((out2[1, 0] - ref1).norm() / ref1.norm()).item() < 5e-2

    # prefill (Sq=L): per-row causal + window
    qf = torch.randn(1, hq, L, d, device=dev, dtype=torch.float32) * 0.5
    q_abs0 = torch.tensor([0], dtype=torch.int32, device=dev)
    outf = fa.forward_paged_direct(
        qf, kc, vc, bt, sl, scale, None, q_abs0, W)[0]
    assert outf.shape == (L, hq, d)
    qtok = qf[0].permute(1, 0, 2).float()  # [L, hq, d]
    rows = sorted({t for t in (0, W - 1, W, L - W, L - 1) if 0 <= t < L})
    for t in rows:
        ref = _windowed_ref(qtok[t], Kf, Vf, scale, t, W)
        assert ((outf[t] - ref).norm() / ref.norm()).item() < 5e-2, \
            f"row {t}"


@pytest.mark.parametrize("d, hq, hkv", [
    (256, 16, 2),   # Qwen3.5-family decode shape
    (128, 32, 2),   # Muse Glimmer shape (Hq 32 / Hkv 2, D 128)
])
def test_forward_paged_direct_splitk_long_context_vs_fp32_ref(d, hq, hkv):
    """M4 (qwen #4a): direct-paged B=1 decode at 16k context runs the
    internal kv_split=8 default (clamp(16/B, 2, 8)); each fp16 P·V
    accumulator spans L/8 keys. Pin the long-context accuracy vs the
    fp32 reference so the production B>=2 serving path cannot silently
    drift (the small-L suite pins exercise the same split-8 default
    but with accumulators 32x shorter)."""
    dev = "cuda"
    torch.manual_seed(20260829)
    L = 16384
    n_blocks = L // BLOCK
    kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                     dtype=torch.uint8, device=dev)
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                     dtype=torch.float16, device=dev)
    K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[:L].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]  # production layout: unbind(1), non-contiguous
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, -1)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    out = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, None)[0, 0]
    g = hq // hkv
    qg = q[0, :, 0].view(hkv, g, d)
    Kf, Vf = K.float().permute(1, 0, 2), V.float().permute(1, 0, 2)
    s = torch.einsum("gjd,gld->gjl", qg, Kf) * scale
    ref = torch.einsum(
        "gjl,gld->gjd", torch.softmax(s, -1), Vf).reshape(hq, d)
    assert ((out - ref).norm() / ref.norm()).item() < 5e-2


# ---------------------------------------------------------------------------
# LEGACY=0 serving layout: the Q8 "side buffer" is NOT a separate
# allocation — it is a strided uint8 view of the K half of the fp16
# kv cache (key_cache.view(uint8)[:, :, :, :bytes_per_row]). The head
# stride is 2D bytes, not bytes_per_row, and the write order is
# triton-fp16-K-then-Q8 on the same memory. Both consumer kernels must
# be stride-generic enough to read this layout.
# ---------------------------------------------------------------------------

def test_paged_direct_and_gather_on_q8_aliased_into_fp16_khalf():
    dev = "cuda"
    torch.manual_seed(7)
    d, hq, hkv, L, W = 128, 32, 2, 512, 128
    n_blocks = L // BLOCK
    bytes_per_row = (d // 32) * 34
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d, dtype=torch.float16, device=dev)
    key_cache, value_cache = kv.unbind(1)  # production layout
    # EXACT backend alias (gfx906_fa_backend._ensure_q8_sidebuffer):
    kc = key_cache.view(torch.uint8)[:, :, :, :bytes_per_row]
    assert kc.stride(2) == 2 * d, "alias must keep the 2D-byte head stride"

    K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    # Production write order: fp16 K write first (clobbers the aliased Q8
    # bytes), then the Q8 write to the same memory. (Staging-copy into the
    # strided K half stands in for the triton kernel's scattered write.)
    staging_k = torch.zeros_like(kv[:, 0])
    staging_k.view(-1, hkv, d)[:L].copy_(K)
    kv[:, 0].copy_(staging_k)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[:L].copy_(V)
    kv[:, 1].copy_(staging)
    vc = value_cache

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(d)
    Kf, Vf = K.float(), V.float()

    # direct-paged B=2 decode, window + Phase C clip (production B>=2 path)
    L2 = L * 3 // 4
    bt2 = bt.repeat(2, 1)
    sl2 = torch.tensor([L, L2], dtype=torch.int32, device=dev)
    q_abs2 = torch.tensor([L - 1, L2 - 1], dtype=torch.int32, device=dev)
    kv_start2 = (q_abs2 + 1 - W).clamp_(min=0).contiguous()
    q2 = torch.randn(2, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    out2 = fa.forward_paged_direct(
        q2, kc, vc, bt2, sl2, scale, None, q_abs2, W, kv_start2)
    ref0 = _windowed_ref(q2[0, :, 0], Kf, Vf, scale, L - 1, W)
    ref1 = _windowed_ref(q2[1, :, 0], Kf, Vf, scale, L2 - 1, W)
    assert ((out2[0, 0] - ref0).norm() / ref0.norm()).item() < 5e-2
    assert ((out2[1, 0] - ref1).norm() / ref1.norm()).item() < 5e-2

    # fused gather (production B=1 / prefill path)
    sk_pad = (L + 31) // 32 * 32
    k_q8, v_b = fa.gather_paged_kv_q8(kc, vc, bt, sl, sk_pad)
    from vllm.gfx906_fa.gfx906_fa_paged import _gather_kv_q8
    k_ref, v_ref = _gather_kv_q8(kc, vc, bt, sl, L)
    assert torch.equal(k_q8[:, :, :L], k_ref)
    assert torch.equal(v_b[:, :, :L], v_ref)
    q1 = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    out1 = fa.forward(q1, k_q8, v_b, scale, kv_max=sl,
                      q_abs_offset=q_abs2[:1], window=W)[0, 0]
    ref1g = _windowed_ref(q1[0, :, 0], Kf, Vf, scale, L - 1, W)
    assert ((out1 - ref1g).norm() / ref1g.norm()).item() < 5e-2


# ---------------------------------------------------------------------------
# Phase C: per-row kv_start clip (forward_paged_direct, decode). The k-loop
# walks [kv_start, L) instead of [0, L); bit-identical to the full scan
# when kv_start >= q_abs + 1 - window (prefix is window-masked anyway).
# The functional check below uses an INERT mask (window=L) plus a real
# kv_start to prove the scan itself shrinks, not just the mask.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d, hq, hkv, L, W", [
    (128, 32, 2, 512, 128),
    (128, 32, 2, 512, 64),    # unaligned clip start (448 vs nbatch_fa=128)
    (128, 32, 2, 96, 128),    # W > L: clip inert (kv_start = 0)
    (256, 16, 2, 512, 128),
])
def test_forward_paged_direct_sliding_window_clip_vs_torch_ref(d, hq, hkv, L, W):
    dev = "cuda"
    torch.manual_seed(4)
    n_blocks = L // BLOCK
    kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                     dtype=torch.uint8, device=dev)
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                     dtype=torch.float16, device=dev)
    K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[:L].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(d)
    Kf, Vf = K.float(), V.float()

    q = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([L - 1], dtype=torch.int32, device=dev)
    kv_start = torch.tensor([max(0, L - W)], dtype=torch.int32, device=dev)

    out_full = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, W)[0, 0]
    out_clip = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, W, kv_start)[0, 0]
    ref = _windowed_ref(q[0, :, 0], Kf, Vf, scale, L - 1, W)
    assert ((out_clip - ref).norm() / ref.norm()).item() < 5e-2
    # clip must be (near-)bit-identical to the masked full scan: the
    # skipped keys were exactly -INF in the softmax
    assert ((out_clip - out_full).norm() / out_full.norm()).item() < 1e-6

    # Functional: INERT window (W=L, mask does nothing) + real clip start
    # -> the scan itself must shrink to the last W2 keys.
    W2 = min(96, L - 1)
    kv_start2 = torch.tensor([L - W2], dtype=torch.int32, device=dev)
    out2 = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, L, kv_start2)[0, 0]
    ref2 = _windowed_ref(q[0, :, 0], Kf, Vf, scale, L - 1, W2)
    assert ((out2 - ref2).norm() / ref2.norm()).item() < 5e-2
    if L > W2:
        assert (out2 - out_full).norm().item() > 1e-3  # clip actually bit

    # B=2, different lengths: per-row clip
    L2 = L * 3 // 4
    bt2 = bt.repeat(2, 1)
    sl2 = torch.tensor([L, L2], dtype=torch.int32, device=dev)
    q_abs2 = torch.tensor([L - 1, L2 - 1], dtype=torch.int32, device=dev)
    kv_start2_2 = torch.tensor([max(0, L - W), max(0, L2 - W)],
                               dtype=torch.int32, device=dev)
    q2 = torch.randn(2, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    outb = fa.forward_paged_direct(
        q2, kc, vc, bt2, sl2, scale, None, q_abs2, W, kv_start2_2)
    ref0 = _windowed_ref(q2[0, :, 0], Kf, Vf, scale, L - 1, W)
    ref1 = _windowed_ref(q2[1, :, 0], Kf, Vf, scale, L2 - 1, W)
    assert ((outb[0, 0] - ref0).norm() / ref0.norm()).item() < 5e-2
    assert ((outb[1, 0] - ref1).norm() / ref1.norm()).item() < 5e-2


@pytest.mark.parametrize("d, hq, hkv, L, W",
                         [(128, 32, 2, 513, 128),
                          (128, 32, 2, 1025, 256),
                          (128, 32, 2, 4353, 2048)])
def test_forward_paged_direct_clip_unaligned_bit_identical(d, hq, hkv, L, W):
    """The clip must stay bit-identical to the masked full scan for
    UNALIGNED starts (kv_start % nbatch_fa != 0 — the normal production
    case, e.g. L=4353/W=2048 -> start=2305). An unaligned start makes the
    first tile partial, repacking which lane holds each surviving score
    and re-associating the fp16 reduction (~5e-4 relative before the
    kernel floors k0_base to the tile boundary). Pre-fix this failed at
    rel 5.2e-4/5.4e-4/7.2e-4 on the three shapes; post-fix <= 1e-7.
    L is deliberately NOT a multiple of BLOCK (paged tail + unaligned
    start at once)."""
    dev = "cuda"
    torch.manual_seed(7)
    d = 128
    n_blocks = (L + BLOCK - 1) // BLOCK
    kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                     dtype=torch.uint8, device=dev)
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                     dtype=torch.float16, device=dev)
    K = torch.randn(n_blocks * BLOCK, hkv, d, device=dev,
                    dtype=torch.float16) * 0.5
    V = torch.randn(n_blocks * BLOCK, hkv, d, device=dev,
                    dtype=torch.float16) * 0.5
    slot = torch.arange(n_blocks * BLOCK, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[: n_blocks * BLOCK].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]
    K = K[:L]
    V = V[:L]

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, -1)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([L - 1], dtype=torch.int32, device=dev)
    kv_start = torch.tensor([max(0, L - W)], dtype=torch.int32, device=dev)
    out_full = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, W)[0, 0]
    out_clip = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, W, kv_start)[0, 0]
    # The skipped prefix is fully window-masked; bit-identity holds
    # (<=1e-7 catches split-combine fp32 roundoff; the pre-fix failure
    # was ~500x larger).
    assert (out_clip - out_full).abs().max().item() < 1e-7


def test_forward_paged_level_direct_branch_window_clip_b2():
    """Review #10: one level up from the binding — forward_paged's
    dispatch -> direct branch -> Python clip math -> binding, at B=2
    (the dispatch's min_batch) with window on. Catches wiring/regression
    in the direct branch itself (which the binding-level tests bypass)
    and exercises an unaligned clip start end-to-end (L=513/W=128 ->
    385, the kernel floors it). B=2 default KVSPLIT=8, so split-K +
    window + clip are all active together."""
    from vllm.gfx906_fa.gfx906_fa_paged import forward_paged
    dev = "cuda"
    torch.manual_seed(17)
    hq, hkv, d = 32, 2, 128
    L0, L1, W = 513, 480, 128
    n0 = (L0 + BLOCK - 1) // BLOCK   # row 0: blocks [0, n0)
    n1 = (L1 + BLOCK - 1) // BLOCK   # row 1: blocks [n0, n0+n1) (disjoint)
    n_blocks = n0 + n1
    kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                     dtype=torch.uint8, device=dev)
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                     dtype=torch.float16, device=dev)
    K = torch.randn(n_blocks * BLOCK, hkv, d, device=dev,
                    dtype=torch.float16) * 0.5
    V = torch.randn(n_blocks * BLOCK, hkv, d, device=dev,
                    dtype=torch.float16) * 0.5
    slot = torch.arange(n_blocks * BLOCK, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[: n_blocks * BLOCK].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]
    Kf, Vf = K.float(), V.float()

    # block_table rows are equal-width; the tail past ceil(L/16) is never
    # indexed (the kernel walks blocks < ceil(seq_len/16)) but must hold a
    # valid block id.
    width = max(n0, n1)
    bt0 = torch.zeros(width, dtype=torch.int32, device=dev)
    bt0[:n0] = torch.arange(n0, dtype=torch.int32, device=dev)
    bt1 = torch.zeros(width, dtype=torch.int32, device=dev)
    bt1[:n1] = torch.arange(n0, n0 + n1, dtype=torch.int32, device=dev)
    bt = torch.stack([bt0, bt1]).contiguous()
    sl = torch.tensor([L0, L1], dtype=torch.int32, device=dev)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(d)

    q = (torch.randn(2, hq, d, device=dev, dtype=torch.float16) * 0.5)
    # key_cache (fp16, 4D) is unused on the direct branch; value_cache is
    # read as [num_blocks, 16, Hkv, D] (the unbound half of the 5D cache)
    out = forward_paged(
        q, kv[:, 0], vc, bt, sl, cu, 1, max(L0, L1), scale,
        key_cache_q8=kc, window=W)  # [num_tokens, Hq*D]
    # row i attends to its own block range: tokens
    # [BLOCK * (n0 if i else 0), ... + L_i)
    offs = [0, n0 * BLOCK]
    for i, L in enumerate((L0, L1)):
        row = out[i].view(hq, d).float()
        ref = _windowed_ref(q[i].float(), Kf[offs[i]:offs[i] + L],
                            Vf[offs[i]:offs[i] + L], scale, L - 1, W)
        assert ((row - ref).norm() / ref.norm()).item() < 5e-2


def test_paged_direct_fully_masked_row_no_nan():
    """A row whose scores are ALL masked (KQ_sum == 0) must not produce
    NaN on the non-split (KVSPLIT=1) store path: 1.0/0 = inf times the
    zero VKQ accumulator gives inf*0 = NaN. The row is zero; its sibling
    is unaffected. (forward_paged slices padding rows out in production,
    so this is a latent hazard, not a serving bug.)"""
    dev = "cuda"
    torch.manual_seed(11)
    hq, hkv, L, W = 32, 2, 256, 64
    d = 128
    n_blocks = L // BLOCK
    kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                     dtype=torch.uint8, device=dev)
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                     dtype=torch.float16, device=dev)
    K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[:L].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(2, -1)
    sl = torch.tensor([L, L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(d)
    Kf, Vf = K.float(), V.float()

    # row 0: normal in-window decode; row 1: q_abs far past its KV, so
    # every score is window-masked (-INF) and the clip start is past L
    # (empty scan) -> KQ_sum == 0 on that row.
    q = torch.randn(2, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([L - 1, L + 100000], dtype=torch.int32, device=dev)
    kv_start = torch.tensor([L - W, L + 100000 + 1 - W],
                            dtype=torch.int32, device=dev)

    old = os.environ.get("GFX906_FA_KVSPLIT")
    os.environ["GFX906_FA_KVSPLIT"] = "1"  # force the non-split path
    try:
        out = fa.forward_paged_direct(
            q, kc, vc, bt, sl, scale, None, q_abs, W, kv_start)
    finally:
        if old is None:
            del os.environ["GFX906_FA_KVSPLIT"]
        else:
            os.environ["GFX906_FA_KVSPLIT"] = old

    assert not torch.isnan(out).any()
    # fully-masked row attends to nothing -> exact zero
    assert out[1].abs().max().item() == 0.0
    # sibling row is the correct windowed decode
    ref0 = _windowed_ref(q[0, :, 0], Kf, Vf, scale, L - 1, W)
    assert ((out[0, 0] - ref0).norm() / ref0.norm()).item() < 5e-2


# ---------------------------------------------------------------------------
# M1: gather-path window clip (persistent gather + FA k-loop shift).
#
# With _GATHER_CLIP on, windowed forward_paged through the persistent
# (legacy-path) gather gathers only [kv_start, L) per seq and the FA kernel
# starts its k-loop at floor(kv_start, nbatch_fa) — rows [0, floor) are left
# stale in the gather buffer and must never be read. This must be
# bit-identical to _GATHER_CLIP off (full gather + full scan): the skipped
# prefix is fully window-masked in both arms (the clip start is the first
# query row's window start; later rows' windows are subsets). Direct-paged
# dispatch is forced off so the B=2 shapes exercise the GATHER path (their
# Phase C direct clip would otherwise make the A/B a no-op).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nq, B, L, W",
                         [(1, 1, 4353, 2048),   # decode, unaligned start
                          (1, 2, 4353, 2048),   # decode, gather-forced B=2
                          (6, 1, 4353, 2048),   # ngram n=5 (Sq=6)
                          (6, 2, 4353, 2048)])
def test_forward_paged_gather_window_clip_bit_identical(nq, B, L, W):
    from vllm.gfx906_fa import gfx906_fa_paged as paged
    dev = "cuda"
    torch.manual_seed(11)
    scale = 1.0 / math.sqrt(D)

    # Disjoint per-seq block ranges (B=2 must not share rows).
    nb = (L + BLOCK - 1) // BLOCK
    n_blocks = B * nb
    _, vc, kv = _make_paged_cache(n_blocks + 4, dev)
    k16 = torch.zeros(n_blocks + 4, BLOCK, HKV, D, dtype=torch.float16,
                      device=dev)
    K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    for b in range(B):
        k16[b * nb:(b + 1) * nb].view(-1, HKV, D)[:L].copy_(
            K[b * nb * BLOCK:b * nb * BLOCK + L])
        stag = torch.zeros_like(kv[b * nb:(b + 1) * nb, 1])
        stag.view(-1, HKV, D)[:L].copy_(V[b * nb * BLOCK:b * nb * BLOCK + L])
        kv[b * nb:(b + 1) * nb, 1].copy_(stag)

    bt = torch.stack([torch.arange(b * nb, (b + 1) * nb, dtype=torch.int32)
                      for b in range(B)]).contiguous().to(dev)
    sl = torch.full((B,), L, dtype=torch.int32, device=dev)
    cu = torch.arange(0, B * nq + 1, nq, dtype=torch.int32, device=dev)
    q = torch.randn(B * nq, HQ, D, device=dev, dtype=torch.float32) * 0.5

    old_clip, old_mode = paged._GATHER_CLIP, paged._DIRECT_PAGED_MODE
    try:
        paged._DIRECT_PAGED_MODE = "0"   # force the gather path
        paged._GATHER_CLIP = False
        out_off = paged.forward_paged(
            q, k16, vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W)
        paged._GATHER_CLIP = True
        out_on = paged.forward_paged(
            q, k16, vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W)
    finally:
        paged._GATHER_CLIP, paged._DIRECT_PAGED_MODE = old_clip, old_mode
    # The skipped prefix is fully window-masked; the clip must not change
    # a single bit (the direct-paged unaligned test documents the ~500x
    # pre-floor failure this would catch).
    assert (out_on - out_off).abs().max().item() < 1e-7


def test_forward_paged_gather_window_clip_short_ctx():
    """L < window: kv_start = 0 everywhere (clip inert) — the gather must
    still be a full gather and the output bit-identical to the clip-off
    arm (guards against the clip math firing on clamped-zero starts)."""
    from vllm.gfx906_fa import gfx906_fa_paged as paged
    dev = "cuda"
    torch.manual_seed(13)
    B, L, W, nq = 2, 513, 2048, 1
    scale = 1.0 / math.sqrt(D)
    nb = (L + BLOCK - 1) // BLOCK
    n_blocks = B * nb
    _, vc, kv = _make_paged_cache(n_blocks + 4, dev)
    k16 = torch.zeros(n_blocks + 4, BLOCK, HKV, D, dtype=torch.float16,
                      device=dev)
    K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    for b in range(B):
        k16[b * nb:(b + 1) * nb].view(-1, HKV, D)[:L].copy_(
            K[b * nb * BLOCK:b * nb * BLOCK + L])
        stag = torch.zeros_like(kv[b * nb:(b + 1) * nb, 1])
        stag.view(-1, HKV, D)[:L].copy_(V[b * nb * BLOCK:b * nb * BLOCK + L])
        kv[b * nb:(b + 1) * nb, 1].copy_(stag)
    bt = torch.stack([torch.arange(b * nb, (b + 1) * nb, dtype=torch.int32)
                      for b in range(B)]).contiguous().to(dev)
    sl = torch.full((B,), L, dtype=torch.int32, device=dev)
    cu = torch.arange(0, B * nq + 1, nq, dtype=torch.int32, device=dev)
    q = torch.randn(B * nq, HQ, D, device=dev, dtype=torch.float32) * 0.5
    old_clip, old_mode = paged._GATHER_CLIP, paged._DIRECT_PAGED_MODE
    try:
        paged._DIRECT_PAGED_MODE = "0"
        paged._GATHER_CLIP = False
        out_off = paged.forward_paged(
            q, k16, vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W)
        paged._GATHER_CLIP = True
        out_on = paged.forward_paged(
            q, k16, vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W)
    finally:
        paged._GATHER_CLIP, paged._DIRECT_PAGED_MODE = old_clip, old_mode
    assert (out_on - out_off).abs().max().item() < 1e-7


def test_forward_paged_direct_prefill_window_clip_bit_identical():
    """M2: the DIRECT_PAGED window clip is no longer decode-only — a
    mid-context prefill chunk (nq=128 > ncols1 tiles) with the
    conservative chunk-start clip must be bit-identical to the clip-off
    arm (kernel per-q-tile raise + causal cap, default on) and to the
    LEGACY gather path's clip (cross-path agreement). Guards the
    backend gate change (max_seqlen_q == 1 dropped)."""
    from vllm.gfx906_fa import gfx906_fa_paged as paged
    dev = "cuda"
    torch.manual_seed(41)
    B, L, W, nq = 2, 1024, 512, 128
    scale = 1.0 / math.sqrt(D)
    nb = L // BLOCK
    n_blocks = B * nb
    kc, vc, kv = _make_paged_cache(n_blocks + 4, dev)
    K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    slot = torch.arange(n_blocks * BLOCK, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc[:n_blocks])
    stag = torch.zeros_like(kv[:, 1])
    stag.view(-1, HKV, D)[:n_blocks * BLOCK].copy_(V)
    kv[:, 1].copy_(stag)
    bt = torch.stack([torch.arange(b * nb, (b + 1) * nb, dtype=torch.int32)
                      for b in range(B)]).contiguous().to(dev)
    sl = torch.full((B,), L, dtype=torch.int32, device=dev)
    cu = torch.arange(0, B * nq + 1, nq, dtype=torch.int32, device=dev)
    q = torch.randn(B * nq, HQ, D, device=dev, dtype=torch.float32) * 0.5
    old_clip, old_mode, old_gclip = (paged._WINDOW_CLIP,
                                     paged._DIRECT_PAGED_MODE,
                                     paged._GATHER_CLIP)
    try:
        paged._DIRECT_PAGED_MODE = "1"
        paged._WINDOW_CLIP = False
        out_off = paged.forward_paged(
            q, kv[:, 0], vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W,
            key_cache_q8=kc)
        paged._WINDOW_CLIP = True
        out_on = paged.forward_paged(
            q, kv[:, 0], vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W,
            key_cache_q8=kc)
        # Cross-path: the LEGACY gather path (clip on) must agree bit-for-bit.
        paged._DIRECT_PAGED_MODE = "0"
        paged._GATHER_CLIP = True
        out_leg = paged.forward_paged(
            q, kv[:, 0], vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W,
            key_cache_q8=kc)
    finally:
        paged._WINDOW_CLIP, paged._DIRECT_PAGED_MODE, paged._GATHER_CLIP = \
            old_clip, old_mode, old_gclip
    assert torch.equal(out_on, out_off), \
        ("direct prefill clip on/off differ: "
         f"{(out_on - out_off).abs().max().item():.3e}")
    assert torch.equal(out_on, out_leg), \
        ("direct-on vs legacy gather differ: "
         f"{(out_on - out_leg).abs().max().item():.3e}")


# ---------------------------------------------------------------------------
# M1 clip on the FUSED Q8 gather (LEGACY=0 read path). Same A/B as
# test_forward_paged_gather_window_clip_bit_identical, but the K half is
# the production Q8 ALIAS of the fp16 cache (gfx906_fa_backend.
# _ensure_q8_sidebuffer), so forward_paged dispatches to the fused
# gather_paged_kv_q8 kernel instead of the persistent fp16 one. The
# clipped output must be bit-identical to the clip-off arm: the skipped
# prefix is fully window-masked in both arms (clip start = first query
# row's window start; later rows' windows are subsets). Direct-paged
# dispatch is forced off so B=2 exercises the GATHER path.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nq, B, L, W",
                         [(1, 1, 4353, 2048),   # decode, unaligned start
                          (1, 2, 4353, 2048),   # decode, gather-forced B=2
                          (6, 1, 4353, 2048),   # ngram n=5 (Sq=6)
                          (6, 2, 4353, 2048)])
def test_forward_paged_fusedq8_window_clip_bit_identical(nq, B, L, W):
    from vllm.gfx906_fa import gfx906_fa_paged as paged
    dev = "cuda"
    torch.manual_seed(17)
    scale = 1.0 / math.sqrt(D)

    # Disjoint per-seq block ranges (B=2 must not share rows).
    nb = (L + BLOCK - 1) // BLOCK
    n_blocks = B * nb
    _, vc, kv = _make_paged_cache(n_blocks + 4, dev)
    key_cache = kv[:, 0]
    # EXACT production LEGACY=0 alias: first BYTES of each fp16 K row.
    kc = key_cache.view(torch.uint8)[:, :, :, :BYTES]
    K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    for b in range(B):
        s0 = b * nb * BLOCK
        # Production write order: fp16 K write first (clobbers the
        # aliased Q8 bytes), then the Q8 write to the same memory.
        stag = torch.zeros_like(kv[b * nb:(b + 1) * nb, 0])
        stag.view(-1, HKV, D)[:L].copy_(K[s0:s0 + L])
        kv[b * nb:(b + 1) * nb, 0].copy_(stag)
        fa.reshape_and_cache_q8(
            K[s0:s0 + L],
            torch.arange(s0, s0 + L, dtype=torch.int64, device=dev), kc)
        _write_v(kv[b * nb:(b + 1) * nb], V[s0:s0 + L])

    bt = torch.stack([torch.arange(b * nb, (b + 1) * nb, dtype=torch.int32)
                      for b in range(B)]).contiguous().to(dev)
    sl = torch.full((B,), L, dtype=torch.int32, device=dev)
    cu = torch.arange(0, B * nq + 1, nq, dtype=torch.int32, device=dev)
    q = torch.randn(B * nq, HQ, D, device=dev, dtype=torch.float32) * 0.5

    old_clip, old_mode = paged._GATHER_CLIP, paged._DIRECT_PAGED_MODE
    try:
        paged._DIRECT_PAGED_MODE = "0"   # force the gather path
        paged._GATHER_CLIP = False
        out_off = paged.forward_paged(
            q, key_cache, vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W,
            key_cache_q8=kc)
        paged._GATHER_CLIP = True
        out_on = paged.forward_paged(
            q, key_cache, vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W,
            key_cache_q8=kc)
    finally:
        paged._GATHER_CLIP, paged._DIRECT_PAGED_MODE = old_clip, old_mode
    # The skipped prefix is fully window-masked; the clip must not change
    # a single bit (the persistent-path twin test documents the failure
    # class this guards).
    assert (out_on - out_off).abs().max().item() < 1e-7


def test_forward_paged_fusedq8_window_clip_short_ctx():
    """L < window on the fused Q8 path: kv_start = 0 everywhere (clip
    inert) — the gather must still be a full gather and the output
    bit-identical to the clip-off arm (guards against the clip math
    firing on clamped-zero starts)."""
    from vllm.gfx906_fa import gfx906_fa_paged as paged
    dev = "cuda"
    torch.manual_seed(19)
    B, L, W, nq = 1, 513, 2048, 1
    scale = 1.0 / math.sqrt(D)
    nb = (L + BLOCK - 1) // BLOCK
    n_blocks = B * nb
    _, vc, kv = _make_paged_cache(n_blocks + 4, dev)
    key_cache = kv[:, 0]
    kc = key_cache.view(torch.uint8)[:, :, :, :BYTES]
    K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    stag = torch.zeros_like(kv[:, 0])
    stag.view(-1, HKV, D)[:L].copy_(K[:L])
    kv[:, 0].copy_(stag)
    fa.reshape_and_cache_q8(
        K[:L], torch.arange(L, dtype=torch.int64, device=dev), kc)
    _write_v(kv, V[:L])
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, -1)
    sl = torch.full((B,), L, dtype=torch.int32, device=dev)
    cu = torch.arange(0, B * nq + 1, nq, dtype=torch.int32, device=dev)
    q = torch.randn(B * nq, HQ, D, device=dev, dtype=torch.float32) * 0.5
    old_clip, old_mode = paged._GATHER_CLIP, paged._DIRECT_PAGED_MODE
    try:
        paged._DIRECT_PAGED_MODE = "0"
        paged._GATHER_CLIP = False
        out_off = paged.forward_paged(
            q, key_cache, vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W,
            key_cache_q8=kc)
        paged._GATHER_CLIP = True
        out_on = paged.forward_paged(
            q, key_cache, vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W,
            key_cache_q8=kc)
    finally:
        paged._GATHER_CLIP, paged._DIRECT_PAGED_MODE = old_clip, old_mode
    assert (out_on - out_off).abs().max().item() < 1e-7


# ---------------------------------------------------------------------------
# M6 Part B (roadmap-more-models.md M6 / plan_fa_legacy0_impr_claude.md):
# GFX906_FA_DIRECT_PAGED_Q8=0 re-routes LEGACY=0 B>=2 from direct-paged
# to the fused-Q8 gather (the LEGACY=0 B=1 path). The re-routed batch
# must (1) be correct vs the torch windowed reference (the standard 5e-2
# cross-path tolerance — the two FA variants differ in K/V read order,
# so they are NOT bit-identical), (2) actually change dispatch (direct-
# paged vs gather outputs differ), and (3) keep the M1 gather clip
# consistent at B>=2 (clip on/off A/B bit-identical under the reroute —
# the clip is dispatch-agnostic, so it must keep firing on the gather
# path, not silently drop out).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nq, B", [(1, 2), (1, 4), (6, 2)])
def test_forward_paged_q8_direct_paged_q8_off_routes_gather(nq, B):
    from vllm.gfx906_fa import gfx906_fa_paged as paged
    dev = "cuda"
    torch.manual_seed(23)
    L, W = 4353, 2048   # unaligned clip start (2305 vs nbatch boundaries)
    scale = 1.0 / math.sqrt(D)

    nb = (L + BLOCK - 1) // BLOCK
    n_blocks = B * nb
    _, vc, kv = _make_paged_cache(n_blocks + 4, dev)
    key_cache = kv[:, 0]
    kc = key_cache.view(torch.uint8)[:, :, :, :BYTES]   # LEGACY=0 alias
    K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    for b in range(B):
        s0 = b * nb * BLOCK
        stag = torch.zeros_like(kv[b * nb:(b + 1) * nb, 0])
        stag.view(-1, HKV, D)[:L].copy_(K[s0:s0 + L])
        kv[b * nb:(b + 1) * nb, 0].copy_(stag)
        fa.reshape_and_cache_q8(
            K[s0:s0 + L],
            torch.arange(s0, s0 + L, dtype=torch.int64, device=dev), kc)
        _write_v(kv[b * nb:(b + 1) * nb], V[s0:s0 + L])

    bt = torch.stack([torch.arange(b * nb, (b + 1) * nb, dtype=torch.int32)
                      for b in range(B)]).contiguous().to(dev)
    sl = torch.full((B,), L, dtype=torch.int32, device=dev)
    cu = torch.arange(0, B * nq + 1, nq, dtype=torch.int32, device=dev)
    q = torch.randn(B * nq, HQ, D, device=dev, dtype=torch.float32) * 0.5

    old = paged._DIRECT_PAGED_Q8, paged._GATHER_CLIP
    try:
        # Flag on (current behavior): direct-paged at B>=2.
        paged._DIRECT_PAGED_Q8 = True
        out_direct = paged.forward_paged(
            q, key_cache, vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W,
            key_cache_q8=kc)
        # Flag off: rerouted to the fused-Q8 gather.
        paged._DIRECT_PAGED_Q8 = False
        paged._GATHER_CLIP = False
        out_off = paged.forward_paged(
            q, key_cache, vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W,
            key_cache_q8=kc)
        paged._GATHER_CLIP = True
        out_on = paged.forward_paged(
            q, key_cache, vc, bt, sl, cu,
            max_seqlen_q=nq, max_seqlen_k=L, scale=scale, window=W,
            key_cache_q8=kc)
    finally:
        paged._DIRECT_PAGED_Q8, paged._GATHER_CLIP = old

    # (3) M1 gather clip stays consistent on the rerouted B>=2 path.
    assert (out_on - out_off).abs().max().item() < 1e-7
    # (2) The flag changes dispatch: the two FA variants are not
    # bit-identical (different K/V read order).
    assert (out_direct - out_on).abs().max().item() > 0
    # (1) Correctness vs the torch windowed reference, every query row
    # of every seq (spec rows are causal rows of the same seq).
    for b in range(B):
        s0 = b * nb * BLOCK
        for i in range(nq):
            r = L - nq + i
            out_row = out_on[b * nq + i].view(HQ, D)
            ref = _windowed_ref(q[b * nq + i],
                                K[s0:s0 + L].float(), V[s0:s0 + L].float(),
                                scale, r, W)
            err = ((out_row - ref).norm() / ref.norm()).item()
            assert err < 5e-2, f"seq {b} row {r}: {err:.4f}"


def test_gather_paged_kv_q8_clip_skips_prefix():
    """Functional (not A/B): the fused Q8 gather with kv_start must leave
    rows [0, start) UNTOUCHED (sentinel preserved) and write rows
    [start, L) bit-equal to the full gather — proving the skip is real
    and the margin rows are materialized (the A/B bit-identity test
    cannot distinguish 'clipped correctly' from 'clip silently inert').
    start = kv_start - GATHER_CLIP_MARGIN (128); L=1025/W=256 ->
    kv_start=769, start=641 (unaligned)."""
    dev = "cuda"
    torch.manual_seed(23)
    B, L, W = 1, 1025, 256
    nb = (L + BLOCK - 1) // BLOCK
    _, vc, kv = _make_paged_cache(nb + 2, dev)
    key_cache = kv[:, 0]
    kc = key_cache.view(torch.uint8)[:, :, :, :BYTES]
    K = torch.randn(nb * BLOCK, HKV, D, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(nb * BLOCK, HKV, D, device=dev, dtype=torch.float16) * 0.5
    stag = torch.zeros_like(kv[:, 0])
    stag.view(-1, HKV, D)[:L].copy_(K[:L])
    kv[:, 0].copy_(stag)
    fa.reshape_and_cache_q8(
        K[:L], torch.arange(L, dtype=torch.int64, device=dev), kc)
    _write_v(kv, V[:L])
    bt = torch.arange(nb, dtype=torch.int32, device=dev).view(1, -1)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    sk_pad = (L + 31) // 32 * 32
    kv_start = torch.tensor([L - W], dtype=torch.int32, device=dev)

    # Reference: full gather into fresh buffers.
    k_full, v_full = fa.gather_paged_kv_q8(kc, vc, bt, sl, sk_pad)

    # Sentinel-filled grow-buffers; the clipped call must keep
    # [0, start) exactly as the sentinel.
    margin = 128  # GATHER_CLIP_MARGIN (kernel/gfx906-config.h)
    start = max(0, kv_start.item() - margin)
    k_out = torch.full((B, HKV, sk_pad, BYTES), 0xAB,
                       dtype=torch.uint8, device=dev)
    v_out = torch.full((B, HKV, sk_pad, D), 1.0,
                       dtype=torch.float16, device=dev)
    k_clipped, v_clipped = fa.gather_paged_kv_q8(
        kc, vc, bt, sl, sk_pad, k_out=k_out, v_out=v_out,
        kv_start=kv_start)
    assert k_clipped.data_ptr() == k_out.data_ptr(), "buffer not reused"
    # [0, start): untouched sentinel (K and V alike — LOCKSTEP with the
    # persistent kernel, which writes nothing there either).
    assert (k_out[:, :, :start] == 0xAB).all(), "K prefix was written"
    assert (v_out[:, :, :start] == 1.0).all(), "V prefix was written"
    # [start, L): bit-equal to the full gather (margin rows materialized,
    # data rows exact).
    assert torch.equal(k_out[:, :, start:L], k_full[:, :, start:L])
    assert torch.equal(v_out[:, :, start:L], v_full[:, :, start:L])
    # [L, sk_pad): V zero tail as usual (the clip does not change it).
    assert (v_out[:, :, L:] == 0.0).all()


# ---------------------------------------------------------------------------
# M2 tile clip: per-q-tile k0_base window raise + per-q-tile causal
# k_VKQ_max cap (GFX906_FA_TILE_CLIP, default on; 0 = A/B arm). The
# skipped k-tiles are window-/causal-masked to P=0 for every row of the
# tile, so on vs off must be bit-identical; the win is skipping their
# k-iterations. Both kernels (gather-path forward + direct-paged),
# prefill geometry where the per-tile raise actually kicks in (Sq >
# ncols1 so later q-tiles' window starts move).
# ---------------------------------------------------------------------------

# The check body runs in a FRESH subprocess: GFX906_FA_KVSPLIT is read once
# per process (static local in get_fa_kv_split), so an in-process pin only
# works when no FA call has happened yet in this process — the rest of this
# file makes calls, and the bit-identity arm needs the pin from the first
# call. (The pre-split-era in-process version of this test broke exactly this
# way when the shape-aware default, dfed62f133, started splitting prefill.)
_TILE_CLIP_BITIDENT_SRC = """
import math
import os
import sys
import torch

sq = int(sys.argv[1])
mode = sys.argv[2]  # "bitident" (kv_split pinned 1) | "split" (default)
dev = "cuda"
torch.manual_seed(23)
from vllm import _gfx906_fa_C as fa

BLOCK = 16
d, hq, hkv, L, W = 128, 32, 2, 1504, 512
q_abs = L - sq                     # chunk starts mid-context
kv_start_val = max(0, q_abs + 1 - W)  # conservative chunk start
scale = 1.0 / math.sqrt(d)
n_blocks = L // BLOCK
kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                 dtype=torch.uint8, device=dev)
kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                 dtype=torch.float16, device=dev)
K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
slot = torch.arange(L, dtype=torch.int64, device=dev)
fa.reshape_and_cache_q8(K, slot, kc)
staging = torch.zeros_like(kv[:, 1])
staging.view(-1, hkv, d)[:L].copy_(V)
kv[:, 1].copy_(staging)
vc = kv.unbind(1)[1]
bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
sl = torch.tensor([L], dtype=torch.int32, device=dev)
q_abs_t = torch.tensor([q_abs], dtype=torch.int32, device=dev)
kv_start_t = torch.tensor([kv_start_val], dtype=torch.int32, device=dev)
qf = torch.randn(1, hq, sq, d, device=dev, dtype=torch.float32) * 0.5
Kf, Vf = K.float(), V.float()
sk_pad = (L + 31) // 32 * 32
k_q8, v_b = fa.gather_paged_kv_q8(kc, vc, bt, sl, sk_pad)

def ref_row(t):
    # Windowed causal reference for the row at abs pos q_abs+t
    # (same einsum pattern as _windowed_ref in the parent file).
    pos = q_abs + t
    lo = max(0, pos - W + 1)
    g = hq // hkv
    qg = qf[0, :, t].view(hkv, g, -1).float()
    kf = Kf[lo:pos + 1]
    vf = Vf[lo:pos + 1]
    s = torch.einsum("gjd,lgd->gjl", qg, kf) * scale
    o = torch.einsum("gjl,lgd->gjd", torch.softmax(s, -1), vf)
    return o.reshape(hq, -1)

def run(clip):
    os.environ["GFX906_FA_TILE_CLIP"] = clip
    out_fwd = fa.forward(qf, k_q8, v_b, scale, kv_max=sl,
                         q_abs_offset=q_abs_t, window=W,
                         kv_start=kv_start_t)[0]
    out_dir = fa.forward_paged_direct(
        qf, kc, vc, bt, sl, scale, None, q_abs_t, W, kv_start_t)[0]
    del os.environ["GFX906_FA_TILE_CLIP"]
    return out_fwd, out_dir

out_off_f, out_off_d = run("0")
out_on_f, out_on_d = run("1")
if mode == "bitident":
    assert os.environ.get("GFX906_FA_KVSPLIT") == "1"
    for name, on, off in (("fwd", out_on_f, out_off_f),
                          ("direct", out_on_d, out_off_d)):
        assert torch.equal(on, off), f"tile_clip on/off differs ({name})"
# Boundary rows (first: raise no-op; middle; last: max raise + causal cap)
# in BOTH modes — the split arm guards windowed correctness under the
# shape-aware split default (not covered by the unwindowed sq test).
worst = 0.0
for t in (0, sq // 2, sq - 1):
    ref = ref_row(t)
    for name, out in (("fwd", out_on_f), ("direct", out_on_d)):
        err = ((out[t].float() - ref).norm() / ref.norm()).item()
        worst = max(worst, err)
        assert err < 5e-2, f"{name} row {t}: {err}"
print(f"sq={sq} mode={mode} worst_ref={worst:.2e}")
"""


@pytest.mark.parametrize("Sq", [256, 64])
def test_m2_tile_clip_prefill_bit_identical(Sq):
    """Mid-context prefill chunk + conservative chunk-start clip:
    tile_clip on (default) and off are bit-identical, and both match the
    torch windowed reference on boundary rows. Sq=256 = 4 q-tiles
    (ncols1=64) so the per-tile raise moves tiles 1..3; Sq=64 = 1
    q-tile (raise no-op, causal cap only).

    Two subprocess arms:
      * bitident — kv_split pinned 1 (single pass): the bit-identity
        premise only holds there (same aligned k-tiles, fully-masked
        differences). Under the shape-aware split default (dfed62f133)
        the arms reduce in different orders — correct, not bit-equal.
      * split — the default (shape-aware, splits this shape): boundary
        rows match the windowed reference within tolerance."""
    import subprocess

    for mode, extra_env in (("bitident", {"GFX906_FA_KVSPLIT": "1"}),
                            ("split", {})):
        env = {**os.environ, **extra_env}
        env.pop("GFX906_FA_TILE_CLIP", None)
        if mode == "split":  # must be the unpinned default
            env.pop("GFX906_FA_KVSPLIT", None)
        r = subprocess.run(
            [sys.executable, "-c", _TILE_CLIP_BITIDENT_SRC, str(Sq), mode],
            env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, \
            f"{mode}: stdout: {r.stdout}\nstderr: {r.stderr[-2000:]}"


@pytest.mark.parametrize("kapi", ["fwd", "direct"])
def test_m2_causal_cap_bit_identical_window_off(kapi):
    """Causal cap in isolation, BOTH kernels: q_abs_offset set,
    window=0 (no window mask, no clip) — the cap only skips causally
    masked tail k-tiles. On vs off must be bit-identical and match the
    plain causal torch reference (the Sq=1 decode geometry is the
    no-op corner: ncols1=2 cap = q_abs+1 = seq_len). The direct arm
    covers the paged kernel's own oob-tail machinery on the unaligned
    partial last tile (Sq=200), which the windowed test only covers
    via LOCKSTEP."""
    dev = "cuda"
    torch.manual_seed(29)
    d, hq, hkv, L, Sq = 128, 32, 2, 1008, 200   # 4 q-tiles, unaligned last
    q_abs = L - Sq
    scale = 1.0 / math.sqrt(d)
    n_blocks = L // BLOCK
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                     dtype=torch.float16, device=dev)
    K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                     dtype=torch.uint8, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[:L].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    q_abs_t = torch.tensor([q_abs], dtype=torch.int32, device=dev)
    qf = torch.randn(1, hq, Sq, d, device=dev, dtype=torch.float32) * 0.5
    Kf, Vf = K.float(), V.float()
    sk_pad = (L + 31) // 32 * 32
    k_q8, v_b = fa.gather_paged_kv_q8(kc, vc, bt, sl, sk_pad)

    def _run(clip):
        os.environ["GFX906_FA_TILE_CLIP"] = clip
        if kapi == "fwd":
            return fa.forward(qf, k_q8, v_b, scale, kv_max=sl,
                              q_abs_offset=q_abs_t)[0]
        # window=0, kv_start=None: the cap is the only M2 bound in
        # play (q_abs_offset set) — the production causal-prefill shape.
        return fa.forward_paged_direct(qf, kc, vc, bt, sl, scale, None,
                                       q_abs_t, 0, None)[0]

    try:
        out_off = _run("0")
        out_on = _run("1")
        assert torch.equal(out_on, out_off), \
            f"tile_clip on/off differs ({kapi}, window=0)"
        for t in (0, Sq - 1):
            ref = _windowed_ref(qf[0, :, t], Kf, Vf, scale, q_abs + t, None)
            err = ((out_on[t] - ref).norm() / ref.norm()).item()
            assert err < 5e-2, f"{kapi} row {t}: {err}"
    finally:
        del os.environ["GFX906_FA_TILE_CLIP"]

    # Decode corner (Sq=1): the cap must be a no-op — bit-identical to
    # the kv_max-only call (no q_abs_offset at all).
    q1 = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    q_abs1 = torch.tensor([L - 1], dtype=torch.int32, device=dev)
    try:
        os.environ["GFX906_FA_TILE_CLIP"] = "1"
        out_d = fa.forward(q1, k_q8, v_b, scale, kv_max=sl,
                           q_abs_offset=q_abs1)[0, 0]
        os.environ["GFX906_FA_TILE_CLIP"] = "0"
        out_d2 = fa.forward(q1, k_q8, v_b, scale, kv_max=sl,
                            q_abs_offset=q_abs1)[0, 0]
    finally:
        del os.environ["GFX906_FA_TILE_CLIP"]
    assert torch.equal(out_d, out_d2)
    ref1 = _windowed_ref(q1[0, :, 0], Kf, Vf, scale, L - 1, None)
    assert ((out_d - ref1).norm() / ref1.norm()).item() < 5e-2

# ---------------------------------------------------------------------------
# M3 kernel hygiene batch (roadmap M3, 2026-08-28)
#
# #8  device-side k0_base clamp: a negative kv_start[sequence] must clamp
#     to 0 (the old code walked the k-loop into token-negative space —
#     illegal access / wedge, not a wrong number).
# #10 overflow-free window cutoff `q_abs_row - k_pos_abs >= window`:
#     for absurd windows (INT_MAX) the cutoff must be inert and the
#     output bit-identical to plain causal (no wrapping, no silent mask).
# Hardening: amplified-V window-boundary case (probe-B trick) — one key
# misclassified at the window edge must move the output by O(400x).
# ---------------------------------------------------------------------------

def _m3_setup(dev, d, hq, hkv, L, W, v_amplify_boundary=False):
    torch.manual_seed(11)
    n_blocks = L // BLOCK
    kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                     dtype=torch.uint8, device=dev)
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                     dtype=torch.float16, device=dev)
    K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.25
    if v_amplify_boundary and L > W:
        # The key ONE before the window edge (index L-W-1 for the last row)
        # is the one an off-by-one cutoff would wrongly include: make it
        # ~400x larger so its (mis)inclusion is unmissable in the output.
        V[L - W - 1] = (
            torch.randn(1, hkv, d, device=dev, dtype=torch.float16)
            * 100.0)
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[:L].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(d)
    sk_pad = (L + 31) // 32 * 32
    return kc, vc, bt, sl, scale, sk_pad, K, V


@pytest.mark.parametrize("d, hq, hkv, L, W", [
    (128, 32, 2, 512, 128),
    (256, 16, 2, 512, 128),
])
def test_m3_10_oversized_window_bit_identical_causal(d, hq, hkv, L, W):
    """#10: window=INT_MAX must be exactly plain causal (inert cutoff,
    no int32 wrap, no silently-disabled mask) — bit-identical outputs."""
    dev = "cuda"
    kc, vc, bt, sl, scale, sk_pad, K, V = _m3_setup(dev, d, hq, hkv, L, W)
    k_q8, v_b = fa.gather_paged_kv_q8(kc, vc, bt, sl, sk_pad)
    q = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([L - 1], dtype=torch.int32, device=dev)

    out_causal = fa.forward(q, k_q8, v_b, scale, kv_max=sl,
                            q_abs_offset=q_abs)[0, 0]
    out_wmax = fa.forward(q, k_q8, v_b, scale, kv_max=sl,
                          q_abs_offset=q_abs,
                          window=2 ** 31 - 1)[0, 0]
    assert torch.equal(out_causal, out_wmax), \
        "window=INT_MAX must be bit-identical to causal (non-paged)"

    out_causal_p = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs)[0, 0]
    out_wmax_p = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, 2 ** 31 - 1)[0, 0]
    assert torch.equal(out_causal_p, out_wmax_p), \
        "window=INT_MAX must be bit-identical to causal (paged)"

    # Prefill: same claim per-row.
    qf = torch.randn(1, hq, L, d, device=dev, dtype=torch.float32) * 0.5
    q_abs0 = torch.tensor([0], dtype=torch.int32, device=dev)
    outf_causal = fa.forward(qf, k_q8, v_b, scale, kv_max=sl,
                             q_abs_offset=q_abs0)[0]
    outf_wmax = fa.forward(qf, k_q8, v_b, scale, kv_max=sl,
                           q_abs_offset=q_abs0,
                           window=2 ** 31 - 1)[0]
    assert torch.equal(outf_causal, outf_wmax), \
        "prefill window=INT_MAX must be bit-identical to causal"


@pytest.mark.parametrize("d, hq, hkv, L, W", [
    (128, 32, 2, 512, 128),
])
def test_m3_8_negative_kv_start_clamps_to_zero(d, hq, hkv, L, W):
    """#8: a negative kv_start must clamp to 0 — the k-loop may not walk
    into token-negative space. Output must equal the kv_start=0 scan."""
    dev = "cuda"
    kc, vc, bt, sl, scale, sk_pad, K, V = _m3_setup(dev, d, hq, hkv, L, W)
    k_q8, v_b = fa.gather_paged_kv_q8(kc, vc, bt, sl, sk_pad)
    q = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([L - 1], dtype=torch.int32, device=dev)

    out0 = fa.forward(q, k_q8, v_b, scale, kv_max=sl,
                      q_abs_offset=q_abs, window=W,
                      kv_start=torch.tensor([0],
                                            dtype=torch.int32,
                                            device=dev))[0, 0]
    outneg = fa.forward(q, k_q8, v_b, scale, kv_max=sl,
                        q_abs_offset=q_abs, window=W,
                        kv_start=torch.tensor([-L],
                                              dtype=torch.int32,
                                              device=dev))[0, 0]
    assert torch.equal(out0, outneg), \
        "negative kv_start must clamp to 0 (non-paged)"

    out0_p = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, W,
        torch.tensor([0], dtype=torch.int32, device=dev))[0, 0]
    outneg_p = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, W,
        torch.tensor([-L], dtype=torch.int32, device=dev))[0, 0]
    assert torch.equal(out0_p, outneg_p), \
        "negative kv_start must clamp to 0 (paged)"


@pytest.mark.parametrize("d, hq, hkv, L, W", [
    (128, 32, 2, 512, 128),
    (128, 32, 2, 512, 64),
])
def test_m3_window_boundary_amplified_v(d, hq, hkv, L, W):
    """Hardening (probe-B trick): V at the first OUT-of-window key is
    amplified ~400x — a one-key cutoff error would move the output by
    O(400x the tolerance). Checks both kernels vs the torch reference."""
    dev = "cuda"
    kc, vc, bt, sl, scale, sk_pad, K, V = _m3_setup(
        dev, d, hq, hkv, L, W, v_amplify_boundary=True)
    k_q8, v_b = fa.gather_paged_kv_q8(kc, vc, bt, sl, sk_pad)
    q = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([L - 1], dtype=torch.int32, device=dev)
    ref = _windowed_ref(q[0, :, 0], K.float(), V.float(), scale, L - 1, W)

    out = fa.forward(q, k_q8, v_b, scale, kv_max=sl,
                     q_abs_offset=q_abs, window=W)[0, 0]
    err = ((out - ref).norm() / ref.norm()).item()
    assert err < 5e-2, f"non-paged boundary error {err}"

    out_p = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, W)[0, 0]
    err_p = ((out_p - ref).norm() / ref.norm()).item()
    assert err_p < 5e-2, f"paged boundary error {err_p}"

    # The amplified key must actually be discriminating: including it
    # (wrong cutoff) moves the output far beyond the tolerance.
    # A correct-cutoff output is < 5e-2 from the right-window reference;
    # with the boundary key amplified ~400x, a wrong cutoff lands at
    # rel-err ~1.0 (the amplified V dominates one softmax weight). 0.5
    # sits 10x above the correctness tolerance and 2x below the
    # measured wrong-cutoff error.
    ref_bad = _windowed_ref(q[0, :, 0], K.float(), V.float(), scale,
                            L - 1, W + 1)
    bad_err = ((out - ref_bad).norm() / ref_bad.norm()).item()
    assert bad_err > 0.5, \
        f"amplified boundary key not discriminating (err {bad_err})"


# Part A layout pins (docs/gfx906/plan_fa_part_A.md): the Q8 K row is
# PLANAR — [quants D bytes | scale (D/32) fp16] — in every buffer the FA
# kernels read (the LEGACY=0 alias, the gather tile, the prefill
# dense-quant output). These pins are byte-level: the forward tests assert
# FA output correctness, which cannot distinguish "wrong layout read by a
# matching wrong loader" from "right layout".
# ---------------------------------------------------------------------------

def _ref_q8_0_row(x):
    """Bit-exact Python reference for the kernel Q8_0 quantizer
    (csrc/gfx906_fa/kernel/q8_0_quantize.cuh): one fp16 row of D values
    (D % 32 == 0) -> (quants, scale). quants: uint8 [D] (int8 bit
    patterns); scale: uint8 [2*(D/32)] (per-block fp16 d, little-endian).
    d = amax/127 (fp32); id = 1/d if d > 0 else 0; qi = clamp(rintf(v*id));
    scale = fp16(d)."""
    v = x.float().reshape(-1, 32)   # [nblocks, 32]
    amax = v.abs().amax(dim=-1, keepdim=True)
    d = amax / 127.0
    id_ = torch.where(d > 0.0, 1.0 / d, 0.0)
    qi = torch.round(v * id_).clamp_(-128, 127).to(torch.int8)
    return (qi.view(torch.uint8).reshape(-1),
            d.half().view(torch.uint8).reshape(-1))


def _assert_planar_row(row_bytes, x_row):
    """Assert row_bytes (uint8, (D//32)*34) is the PLANAR Q8_0 row of
    x_row (fp16, D): quants plane = per-32-block int8 (bytes [0, D));
    scale plane = per-block fp16 d (bytes [D, D + 2*(D//32)))."""
    Dv = x_row.numel()
    quants, scales = _ref_q8_0_row(x_row)
    assert row_bytes.numel() == Dv + scales.numel()
    assert torch.equal(row_bytes[:Dv], quants), "quants plane mismatch"
    assert torch.equal(row_bytes[Dv:], scales), "scale plane mismatch"


def test_q8_0_row_layout_planar_pin_reshape_alias():
    """Writer 1: reshape_and_cache_q8 (the LEGACY=0 alias writer)."""
    dev = "cuda"
    torch.manual_seed(41)
    kc, vc, kv = _make_paged_cache(4, dev)
    n_rows = 4 * BLOCK
    K = torch.randn(n_rows, HKV, D, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(n_rows, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    for r in list(range(0, n_rows, 7)) + [n_rows - 1]:
        for h in range(HKV):
            _assert_planar_row(kc[r // BLOCK, r % BLOCK, h], K[r, h])


def test_q8_0_row_layout_planar_pin_dense_quant():
    """Writer 3: quantize_q8_0 (two-kernel prefill fallback)."""
    dev = "cuda"
    torch.manual_seed(43)
    N = 37
    x = torch.randn(N, D, device=dev, dtype=torch.float16) * 0.5
    y = fa.quantize_q8_0(x)
    assert y.shape == (N, BYTES) and y.dtype == torch.uint8
    for r in list(range(0, N, 5)) + [N - 1]:
        _assert_planar_row(y[r], x[r])


def test_q8_0_row_layout_planar_pin_fused_quant_gather():
    """Writer 2 (LEGACY=1 production decode path): the fused
    gather+quantize persistent kernel's tile rows must be planar."""
    dev = "cuda"
    torch.manual_seed(47)
    B, L = 2, 100
    nb = (L + BLOCK - 1) // BLOCK
    n_blocks = B * nb
    _, vc, kv = _make_paged_cache(n_blocks + 4, dev)
    key_cache, value_cache = kv.unbind(1)
    K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    for b in range(B):
        s0 = b * nb * BLOCK
        stag = torch.zeros_like(kv[b * nb:(b + 1) * nb, 0])
        stag.view(-1, HKV, D)[:L].copy_(K[s0:s0 + L])
        kv[b * nb:(b + 1) * nb, 0].copy_(stag)
        _write_v(kv[b * nb:(b + 1) * nb], V[s0:s0 + L])
    bt = torch.stack([torch.arange(b * nb, (b + 1) * nb, dtype=torch.int32)
                      for b in range(B)]).contiguous().to(dev)
    sl = torch.full((B,), L, dtype=torch.int32, device=dev)
    sk_pad = (L + 31) // 32 * 32
    k_out = torch.zeros(B, HKV, sk_pad, BYTES, dtype=torch.uint8, device=dev)
    v_out = torch.zeros(B, HKV, sk_pad, D, dtype=torch.float16, device=dev)
    k_tile, _ = fa.gather_paged_kv_quant_persistent(
        key_cache, value_cache, bt, sl, sk_pad,
        k_out=k_out, v_out=v_out)
    assert k_tile.data_ptr() == k_out.data_ptr(), "k_out not reused"
    for b in range(B):
        for t in list(range(0, L, 13)) + [L - 1]:
            for h in range(HKV):
                blk = bt[b, t // BLOCK].item()
                _assert_planar_row(k_tile[b, h, t],
                                   key_cache[blk, t % BLOCK, h])


def test_q8_0_row_layout_planar_pin_fused_q8_gather():
    """The LEGACY=0 fused-Q8 gather (byte copy of the alias into the
    tile): tile rows must equal the alias rows (and the reference planar
    row — the copy is layout-transparent)."""
    dev = "cuda"
    torch.manual_seed(53)
    B, L = 2, 100
    nb = (L + BLOCK - 1) // BLOCK
    n_blocks = B * nb
    _, vc, kv = _make_paged_cache(n_blocks + 4, dev)
    key_cache = kv[:, 0]
    kc = key_cache.view(torch.uint8)[:, :, :, :BYTES]   # LEGACY=0 alias
    K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    for b in range(B):
        s0 = b * nb * BLOCK
        stag = torch.zeros_like(kv[b * nb:(b + 1) * nb, 0])
        stag.view(-1, HKV, D)[:L].copy_(K[s0:s0 + L])
        kv[b * nb:(b + 1) * nb, 0].copy_(stag)
        fa.reshape_and_cache_q8(
            K[s0:s0 + L],
            torch.arange(s0, s0 + L, dtype=torch.int64, device=dev), kc)
        _write_v(kv[b * nb:(b + 1) * nb], V[s0:s0 + L])
    bt = torch.stack([torch.arange(b * nb, (b + 1) * nb, dtype=torch.int32)
                      for b in range(B)]).contiguous().to(dev)
    sl = torch.full((B,), L, dtype=torch.int32, device=dev)
    sk_pad = (L + 31) // 32 * 32
    k_tile, _ = fa.gather_paged_kv_q8(kc, vc, bt, sl, sk_pad)
    for b in range(B):
        for t in list(range(0, L, 13)) + [L - 1]:
            for h in range(HKV):
                blk = bt[b, t // BLOCK].item()
                alias_row = kc[blk, t % BLOCK, h]
                assert torch.equal(k_tile[b, h, t], alias_row), \
                    "tile row != alias row"
                _assert_planar_row(k_tile[b, h, t],
                                   K[b * nb * BLOCK + t, h])

# ---------------------------------------------------------------------------
# R3 (fa-decode-fp16-hunt-combined.md): the paged-direct path's kv_split
# default must be aligned with the gather path's. Before this, the direct
# path used clamp(16/batch, 2..8) (decode-measured) for EVERY shape, so a
# GFX906_FA_LEGACY 1->0 flip would silently move the Sq>=4 verify shapes
# from 32 (shape-aware, +~10% measured) down to <=8 — a silent FA
# regression. The fa.kv_split_default probe exposes the exact C++ decision
# functions (pure, no GPU work, env read per call — monkeypatch-safe
# in-process, unlike the launch-site get_fa_kv_split static).
# ---------------------------------------------------------------------------

def test_vit_auto_is_default_on_and_falls_back(monkeypatch):
    """VIT-1: the custom ViT path is the gfx906 default, with real fallbacks.

    Pins the 2026-09-15 flip (image-prompt TTFT -11.5 % @1024x1024, -55 s of
    fresh-boot Triton JIT) so a future change to the default is deliberate, and
    pins the two escape hatches: `GFX906_FA_VIT=0` (upstream path outright) and
    `GFX906_FA_VIT_AUTO=0` (opt out of auto-selection). Unsupported shapes/dtypes
    must still resolve to the upstream backend rather than reaching the kernel.
    """
    import torch

    from vllm.gfx906_fa.gfx906_fa_mm_encoder import (
        vit_auto_enabled,
        vit_enabled,
        vit_supported,
    )

    monkeypatch.delenv("GFX906_FA_VIT_AUTO", raising=False)
    monkeypatch.delenv("GFX906_FA_VIT", raising=False)
    assert vit_enabled() and vit_auto_enabled(), "custom ViT path must be ON by default"

    from vllm.platforms import current_platform
    from vllm.platforms.interface import AttentionBackendEnum
    from vllm.platforms.rocm import on_gfx906

    monkeypatch.setenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    backend = current_platform.get_vit_attn_backend(72, torch.float16, backend=None)
    if on_gfx906():
        # the real selection function, with no runner env set
        assert backend == AttentionBackendEnum.CUSTOM, backend

    monkeypatch.setenv("GFX906_FA_VIT_AUTO", "0")
    assert not vit_auto_enabled() and vit_enabled()

    monkeypatch.setenv("GFX906_FA_VIT", "0")
    assert not vit_enabled() and not vit_auto_enabled()
    assert not vit_supported(72, torch.float16)

    # unsupported head size / dtype never reach the kernel
    monkeypatch.delenv("GFX906_FA_VIT")
    assert vit_supported(72, torch.float16)
    assert not vit_supported(72, torch.bfloat16)
    # 160 pads up to 256 (still supported); only a head dim no instantiation
    # covers (256 is the largest) must fall back to the upstream backends
    assert vit_supported(160, torch.float16)
    assert not vit_supported(320, torch.float16)


def test_vit_fallback_is_loud_and_explains_itself(monkeypatch, caplog):
    """VIT-1: an unsupported ViT shape/dtype must not fall back *silently*.

    A silent fall-through keeps the model correct but loses the MI50-tuned ViT
    kernel (-11.5 % image-prompt TTFT @1024x1024, -55 s fresh-boot Triton JIT),
    and on gfx906 the fall-through chain can end at unfused TORCH_SDPA rather
    than flash-attn — note `on_cdna()` is a substring test that is TRUE here
    ("gfx9" in "gfx906"), so the CDNA branch is the usual upstream pick. This test
    drives the real `ROCmPlatform.get_vit_attn_backend` with the default env and
    asserts a WARNING that names the reason.
    """
    import logging

    import torch

    from vllm.gfx906_fa.gfx906_fa_mm_encoder import vit_unsupported_reason
    from vllm.platforms import current_platform
    from vllm.platforms.interface import AttentionBackendEnum
    from vllm.platforms.rocm import on_gfx906

    if not on_gfx906():
        pytest.skip("gfx906-only fallback path")

    monkeypatch.delenv("GFX906_FA_VIT_AUTO", raising=False)
    monkeypatch.delenv("GFX906_FA_VIT", raising=False)
    monkeypatch.setenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

    # supported shapes/dtypes report no reason (padded head dims are fine)
    assert vit_unsupported_reason(72, torch.float16) is None
    assert vit_unsupported_reason(160, torch.float16) is None  # pads to 256

    # each fallback reason is named
    bf16_reason = vit_unsupported_reason(72, torch.bfloat16)
    assert bf16_reason and "bfloat16" in bf16_reason, bf16_reason
    dim_reason = vit_unsupported_reason(320, torch.float16)
    assert dim_reason and "320" in dim_reason, dim_reason
    monkeypatch.setenv("GFX906_FA_VIT", "0")
    assert "GFX906_FA_VIT=0" in (vit_unsupported_reason(72, torch.float16) or "")
    monkeypatch.delenv("GFX906_FA_VIT")

    # ...and the selection path warns about it, loudly, with the reason inline
    with caplog.at_level(logging.WARNING):
        backend = current_platform.get_vit_attn_backend(320, torch.float16, backend=None)
    assert backend != AttentionBackendEnum.CUSTOM, backend
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("CUSTOM ViT attention UNAVAILABLE" in m for m in msgs), msgs
    assert any("320" in m for m in msgs), msgs


def test_vit_bidirectional_matches_sdpa():
    """VIT-1: the dense FA entry serves bidirectional ragged ViT attention.

    Qwen3.5 ViT geometry: head_dim 72 (padded to 96 in-kernel since FA-D96), bidirectional,
    cache-free, ragged batches. The second item is deliberately shorter than the
    padded length, so a wrong KV bound (kv_max) would attend padded KV rows and
    show up as a large error here.
    """
    import math
    from itertools import accumulate

    import torch
    import torch.nn.functional as F

    from vllm.gfx906_fa.gfx906_fa_mm_encoder import forward_vit, vit_supported

    if not vit_supported(72, torch.float16):
        pytest.skip("gfx906 ViT FA path disabled (GFX906_FA_VIT=0)")

    torch.manual_seed(0)
    b, s, hq, hkv, d = 2, 96, 4, 4, 72
    lens = [s, 53]
    q = torch.randn(b, s, hq, d, dtype=torch.float16, device="cuda")
    k = torch.randn(b, s, hkv, d, dtype=torch.float16, device="cuda")
    v = torch.randn(b, s, hkv, d, dtype=torch.float16, device="cuda")
    cu = torch.tensor([0, *accumulate(lens)], dtype=torch.int32)

    scale = 1.0 / math.sqrt(d)
    out = forward_vit(q, k, v, cu, None, scale, d)
    assert out.shape == q.shape and out.dtype == q.dtype

    for i, ln in enumerate(lens):
        qi = q[i, :ln].transpose(0, 1).float()
        ki = k[i, :ln].transpose(0, 1).float()
        vi = v[i, :ln].transpose(0, 1).float()
        ref = F.scaled_dot_product_attention(qi, ki, vi, scale=scale, is_causal=False)
        ref = ref.transpose(0, 1)
        rel = (out[i, :ln].float() - ref).abs().max() / ref.abs().max()
        assert rel < 5e-2, f"item {i} (len {ln}): rel err {rel:.4f}"

    # The padded KV rows must not be attended: a longer KV that is *fully
    # padded past the real length* has to leave the real rows unchanged.
    k2, v2 = k.clone(), v.clone()
    k2[1, lens[1]:] = 1e3
    v2[1, lens[1]:] = 1e3
    out2 = forward_vit(q, k2, v2, cu, None, scale, d)
    rel2 = (out2[1, : lens[1]].float() - out[1, : lens[1]].float()).abs().max()
    assert rel2 < 1e-3, f"padded KV leaked into the scan: {rel2:.4f}"


def test_vit_packed_batch_matches_sdpa_and_does_not_cross_attend():
    """VIT-1: the *production* ViT layout is a packed token stream.

    `Qwen3VLModel.forward` does `hidden_states.unsqueeze(1)` before the vision
    blocks, so attention sees `[seq_len, 1, hidden]` with
    `cu_seqlens = [0, l_0, l_0+l_1, ...]` and `cu_seqlens[-1] == seq_len` — i.e.
    **one** batch row holding every image back to back (the same reason
    flash-attn's varlen wrapper asserts `cu_seqlens_q[-1] == total_seqlen_q`).

    Two properties are pinned here, and the second is the one that matters: each
    image must match its own SDPA reference, and no image may attend another
    (the kernel's per-row `kv_max` is what enforces that, so a mapping mistake
    would silently blend images rather than crash).
    """
    import math
    from itertools import accumulate

    import torch
    import torch.nn.functional as F

    from vllm.gfx906_fa.gfx906_fa_mm_encoder import forward_vit, vit_supported

    if not vit_supported(72, torch.float16):
        pytest.skip("gfx906 ViT FA path disabled (GFX906_FA_VIT=0)")

    torch.manual_seed(0)
    h, d = 16, 72
    lens = [2304, 576, 1728]  # ragged, and not a multiple of any tile size
    n = sum(lens)
    q = torch.randn(1, n, h, d, dtype=torch.float16, device="cuda")
    k = torch.randn(1, n, h, d, dtype=torch.float16, device="cuda")
    v = torch.randn(1, n, h, d, dtype=torch.float16, device="cuda")
    cu = torch.tensor([0, *accumulate(lens)], dtype=torch.int32, device="cuda")
    scale = 1.0 / math.sqrt(d)

    out = forward_vit(q, k, v, cu, None, scale, d)
    assert out.shape == q.shape and out.dtype == q.dtype

    start = 0
    for ln in lens:
        qi = q[0, start : start + ln].transpose(0, 1).float()
        ki = k[0, start : start + ln].transpose(0, 1).float()
        vi = v[0, start : start + ln].transpose(0, 1).float()
        ref = F.scaled_dot_product_attention(qi, ki, vi, scale=scale, is_causal=False)
        got = out[0, start : start + ln].float()
        rel = (got - ref.transpose(0, 1)).abs().max() / ref.abs().max()
        assert rel < 5e-2, f"len {ln} at {start}: rel err {rel:.4f}"
        start += ln

    # No image may attend another: perturb image 0's K/V and the rest must not move.
    k2, v2 = k.clone(), v.clone()
    k2[0, : lens[0]] += 3.0
    v2[0, : lens[0]] += 3.0
    out2 = forward_vit(q, k2, v2, cu, None, scale, d)
    start = lens[0]
    for ln in lens[1:]:
        moved = (
            out2[0, start : start + ln].float() - out[0, start : start + ln].float()
        ).abs().max()
        assert moved < 1e-3, f"image at {start} attended image 0: {moved:.4f}"
        start += ln


def test_legacy_default_is_side_buffer_after_kvlayout1(monkeypatch):
    """KVLAYOUT-1 (2026-09-16): the Q8 side-buffer path is now the default.

    It was fail-closed on 0.29 until verified against the fused KV-cache content
    axis (#51718): PPL 10.5472 bit-identical to LEGACY=1 (0 top-20 misses) and
    -15.5 % / -19.1 % ms/step with MTP k=3 at 64k / 120k, acceptance unchanged.
    LEGACY=1 remains the rollback (it wins ~6 % for B=1 greedy decode).
    """
    from vllm.gfx906_fa.gfx906_fa_backend import _resolve_legacy_mode

    monkeypatch.delenv("GFX906_FA_LEGACY", raising=False)
    assert _resolve_legacy_mode() is False  # default: side-buffer read path

    monkeypatch.setenv("GFX906_FA_LEGACY", "0")
    assert _resolve_legacy_mode() is False

    monkeypatch.setenv("GFX906_FA_LEGACY", "1")
    assert _resolve_legacy_mode() is True  # rollback, validated at B=1 greedy

    # the obsolete override must not resurrect the refusal
    monkeypatch.delenv("GFX906_FA_LEGACY")
    monkeypatch.setenv("GFX906_FA_LEGACY_ALLOW_UNVERIFIED", "1")
    assert _resolve_legacy_mode() is False


def test_r3_kv_split_defaults_aligned(monkeypatch):
    monkeypatch.delenv("GFX906_FA_KVSPLIT", raising=False)
    monkeypatch.delenv("GFX906_FA_KVSPLIT_SHAPEAWARE", raising=False)
    # gather rule (unchanged): shape-aware 32 for Sq>=4, 16 otherwise.
    for sq, want in ((2, 16), (4, 32), (8, 32), (1568, 32)):
        assert fa.kv_split_default(sq, 1, False) == want, sq
    # direct rule: verify shapes (Sq>=4) match the gather rule for every
    # batch — this is the alignment R3 requires.
    for sq in (4, 8, 1568):
        for b in (1, 2, 4, 8):
            assert fa.kv_split_default(sq, b, True) == 32, (sq, b)
    # direct rule: decode shapes (Sq<4) keep the batch clamp — the measured
    # S=8/8/5/2/2 for B=1/2/3/4/8 (MI50 micro-bench, DEVLOG-muse-glimmer).
    for b, want in ((1, 8), (2, 8), (3, 5), (4, 4), (8, 2)):
        assert fa.kv_split_default(2, b, True) == want, b
    # kill switch: SHAPEAWARE=0 restores the old flat-16 gather rule on the
    # verify shapes for BOTH paths; the decode clamp is unaffected.
    monkeypatch.setenv("GFX906_FA_KVSPLIT_SHAPEAWARE", "0")
    assert fa.kv_split_default(8, 1, False) == 16
    assert fa.kv_split_default(8, 1, True) == 16
    assert fa.kv_split_default(2, 1, True) == 8

# ---------------------------------------------------------------------------
# FIX-H2 (pad-tile clamp) + M3 (host cu_seqlens) - mixed-batch coverage.
# The production mixed step is a 1024-row prefill chunk co-batched with
# per-decode rows whose contexts are far longer than the chunk's own; the
# clamp (kv_max=0 on fully-pad tiles) is exact, so these tests pin
# CORRECTNESS under the pad pattern (the -41% serving A/B is the evidence
# that the clamp also fires at scale - output equality cannot distinguish
# clamped from unclamped pad tiles by construction).
#
# Layout: seq1 = 128-row chunk (ctx 2048), seq2 = 1 decode row (ctx 4096).
# Sq_pad=128 for both; ncols1=64 -> seq2 has tile0 (row 0, straddling) and
# tile1 (rows 64..127, FULLY PAD -> must be clamped).
# ---------------------------------------------------------------------------
def test_forward_mixed_batch_pad_tile_clamp_and_host_cu():
    import math

    dev = "cuda"
    torch.manual_seed(7)
    scale = 1.0 / math.sqrt(D)
    L1, L2 = 2048, 4096
    n1, n2 = 128, 1
    g = HQ // HKV
    cu = torch.tensor([0, n1, n1 + n2], dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([L1, L2], dtype=torch.int32, device=dev)
    q_abs = torch.tensor([L1 - n1, L2 - n2], dtype=torch.int32, device=dev)
    num_tokens = n1 + n2

    # Per-seq block ranges: seq1 -> blocks [0, 128), seq2 -> [128, 384).
    # Each block stores that seq's OWN tokens, so the references below use
    # K_all[:2048] for seq1 and K_all[2048:6144] for seq2.
    b1, b2 = L1 // BLOCK, L2 // BLOCK
    n_blocks = b1 + b2
    _, vc, kv = _make_fused_cache(n_blocks, dev)
    K_all = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
    V_all = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
    _kv_split(kv)[0].copy_(K_all.view(n_blocks, BLOCK, HKV, D))
    _write_v_fused(_kv_split(kv)[1], V_all)
    K, V = K_all[:L1], V_all[:L1]          # seq1's own context
    # rectangular block table; padding slots (0) are never read because
    # kv_max = seq_len caps each row's walk at its own context.
    bt = torch.zeros(2, b2, dtype=torch.int32, device=dev)
    bt[0, :b1] = torch.arange(b1, dtype=torch.int32, device=dev)
    bt[1] = torch.arange(b1, b1 + b2, dtype=torch.int32, device=dev)

    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )
    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=scale,
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    q = torch.randn(num_tokens, HQ, D, device=dev, dtype=torch.float16) * 0.5
    out = torch.zeros(num_tokens, HQ, D, device=dev, dtype=torch.float16)

    def run(host_mode):
        host_cu = None
        if host_mode == "exact":
            host_cu = cu.cpu()
        elif host_mode == "full":
            # V2 hands over the full max_num_reqs+1 slice, not V1's
            # [:num_reqs_padded+1]. The M3 host walk iterates
            # range(num_seqs) only, so the tail must never be read: garbage
            # there is the tripwire.
            host_cu = torch.full((8,), -12345, dtype=torch.int32)
            host_cu[: cu.numel()] = cu.cpu()
        m = Gfx906FAMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=n1,
            max_seq_len=L2,
            query_start_loc=cu,
            seq_lens=seq_lens,
            block_table=bt,
            slot_mapping=torch.empty(0, dtype=torch.int64, device=dev),
            query_start_loc_cpu=host_cu,
        )
        impl.forward(None, q, q, q, kv, m, output=out)
        return out[:num_tokens].clone()

    out_none = run(None)
    out_host = run("exact")
    out_full = run("full")
    assert torch.equal(out_none, out_host), \
        "M3: host cu_seqlens must not change outputs"
    assert torch.equal(out_none, out_full), \
        "M3: a full-length (V2-style) host slice must ignore its tail"

    # fp32 per-head reference: row r of seq s sits at abs q_abs[s]+r and
    # attends causally over its own sequence's K/V.
    ref = torch.empty(num_tokens, HQ, D, device=dev, dtype=torch.float32)
    for s, (n, L, qa, off) in enumerate(((n1, L1, L1 - n1, 0),
                                         (n2, L2, L2 - n2, L1))):
        kf, vf = K_all[off:off + L].float(), V_all[off:off + L].float()
        rows = torch.arange(L, device=dev)
        for r in range(n):
            qpos = qa + r
            per_h = []
            for h in range(HQ):
                hk = h // g
                sc = (q[cu[s] + r, h].float() @ kf[:L, hk].T * scale)
                sc = torch.where(rows > qpos,
                                 torch.full_like(sc, float("-inf")), sc)
                per_h.append(torch.softmax(sc, -1) @ vf[:L, hk])
            ref[cu[s] + r] = torch.stack(per_h, 0)

    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 5e-2, f"mixed-batch rel={rel:.3e}"

# ---------------------------------------------------------------------------
# M3 follow-up (co-review F3): the headroom advisory for the unprofiled
# long-context transients (gather buffers + kv_split partial buffer).
# Pure helpers — CPU-only asserts.
# ---------------------------------------------------------------------------
def test_headroom_advisory_thresholds():
    from vllm.gfx906_fa.gfx906_fa_backend import Gfx906FAImpl
    GiB = 1024**3
    err, warn = Gfx906FAImpl._headroom_advisory(
        int(2.5 * GiB), int(2.0 * GiB))
    assert err and not warn, (err, warn)
    err, warn = Gfx906FAImpl._headroom_advisory(
        int(1.5 * GiB), int(2.0 * GiB))
    assert not err and warn, (err, warn)
    err, warn = Gfx906FAImpl._headroom_advisory(
        int(0.5 * GiB), int(2.0 * GiB))
    assert not err and not warn, (err, warn)


def test_kv_split_transient_cap_rules():
    """Mirrors the C++ rules: budget forces y=1 on big-batch prefill;
    verify shapes keep y=32 with a small transient."""
    from vllm.gfx906_fa.gfx906_fa_backend import Gfx906FAImpl
    budget = 512 * 1024 * 1024
    # B=4 prefill (Sq=1024): 4x1024x12x32x256x4 = 1.6 GB > budget -> y=1.
    assert Gfx906FAImpl._kv_split_transient_cap(
        4, 1024, 12, 256, budget) == 0
    # B=1 prefill: 403 MB < budget -> splits, capped at the budget.
    cap = Gfx906FAImpl._kv_split_transient_cap(1, 1024, 12, 256, budget)
    assert 0 < cap <= budget
    # B=8 verify (Sq=8): small transient, grows linearly with B.
    cap8 = Gfx906FAImpl._kv_split_transient_cap(8, 8, 12, 256, budget)
    assert 0 < cap8 <= 64 * 1024 * 1024
    # Empty batch.
    assert Gfx906FAImpl._kv_split_transient_cap(
        0, 1024, 12, 256, budget) == 0


# ---------------------------------------------------------------------------
# A3 (revived 2026-09-14 on the V2 bring-up branch): the fused multi-step draft
# metadata protocol. Its only consumer is V2's `_generate_fused_drafts` loop,
# which builds the draft metadata once per propose round and calls
# `update_draft_decode_metadata` between draft steps. Our implementation opts in
# under VLLM_GFX906_FUSED_DRAFT=1 and the update is a no-op — correct only while
# every step-dependent field the builder hands over stays a live view of a
# persistent buffer. These three tests pin the flag, the view contract and the
# end-to-end reuse behaviour (the last one is the corruption guard).
# ---------------------------------------------------------------------------


def _a3_builder(monkeypatch, value: str | None):
    from vllm.gfx906_fa.gfx906_fa_backend import Gfx906FAMetadataBuilder
    from vllm.v1.kv_cache_interface import AttentionSpec

    if value is None:
        monkeypatch.delenv("VLLM_GFX906_FUSED_DRAFT", raising=False)
    else:
        monkeypatch.setenv("VLLM_GFX906_FUSED_DRAFT", value)
    spec = AttentionSpec(
        block_size=BLOCK, num_kv_heads=HKV, head_size=D, dtype=torch.float16
    )
    return Gfx906FAMetadataBuilder(spec, ["l0"], None, torch.device("cuda"))


def test_a3_fused_draft_flag_env_gate_and_noop_update(monkeypatch):
    """The capability flag follows VLLM_GFX906_FUSED_DRAFT and the in-place
    update is a no-op that never raises (it runs between draft steps, possibly
    inside CUDA graph capture, so it must stay host-logic-free)."""
    from vllm.gfx906_fa.gfx906_fa_backend import Gfx906FAMetadata

    dev = torch.device("cuda")
    b0 = _a3_builder(monkeypatch, None)
    assert not b0.supports_draft_decode_metadata_update
    b1 = _a3_builder(monkeypatch, "0")
    assert not b1.supports_draft_decode_metadata_update
    b2 = _a3_builder(monkeypatch, "1")
    assert b2.supports_draft_decode_metadata_update

    md = Gfx906FAMetadata(
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=8,
        query_start_loc=torch.zeros(2, dtype=torch.int32, device=dev),
        seq_lens=torch.zeros(1, dtype=torch.int32, device=dev),
        block_table=torch.zeros(1, 1, dtype=torch.int32, device=dev),
        slot_mapping=torch.zeros(1, dtype=torch.int64, device=dev),
    )
    b2.update_draft_decode_metadata(md)  # no-op: must not raise


def test_a3_metadata_build_is_persistent_views():
    """build() must hand back the caller's own tensors (views, not copies): the
    speculator's draft loop relies on mutating the persistent buffers between
    steps and seeing the change through a metadata object built ONCE at step 1."""
    from vllm.gfx906_fa.gfx906_fa_backend import Gfx906FAMetadataBuilder
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import AttentionSpec

    dev = torch.device("cuda")
    spec = AttentionSpec(
        block_size=BLOCK, num_kv_heads=HKV, head_size=D, dtype=torch.float16
    )
    builder = Gfx906FAMetadataBuilder(spec, ["l0"], None, dev)
    B, max_seq = 2, 128
    qsl = torch.arange(B + 1, dtype=torch.int32, device=dev)
    sl = torch.tensor([100, 64], dtype=torch.int32, device=dev)
    bt = torch.zeros(B, 8, dtype=torch.int32, device=dev)
    smap = torch.zeros(B, dtype=torch.int64, device=dev)
    cam = CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.cpu(),
        seq_lens=sl,
        num_reqs=B,
        num_actual_tokens=B,
        max_query_len=1,
        max_seq_len=max_seq,
        block_table_tensor=bt,
        slot_mapping=smap,
    )
    md = builder.build(0, cam)
    assert md.query_start_loc.data_ptr() == qsl.data_ptr()
    assert md.seq_lens.data_ptr() == sl.data_ptr()
    assert md.block_table.data_ptr() == bt.data_ptr()
    assert md.slot_mapping.data_ptr() == smap.data_ptr()


def test_a3_draft_step_reuse_reads_live_seq_lens(monkeypatch):
    """The fused-loop contract end-to-end: build metadata once (step 1), advance
    seq_lens + slot + KV write in place (what the captured update_draft_inputs /
    compute_slot_mappings / do_kv_cache_update kernels do between steps), call the
    (no-op) metadata update, and the second forward through the SAME metadata
    object must match a fresh reference at the advanced lengths. If any field
    were step-baked (a copy, a host scalar read by the kernel) this second
    forward would still attend over the old lengths and mismatch.

    Pinned to LEGACY=1 (the fp16 gather path this contract is written for); the
    default side-buffer path has no gather buffers, so the env is forced here."""
    monkeypatch.setenv("GFX906_FA_LEGACY", "1")
    dev = "cuda"
    torch.manual_seed(23)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadataBuilder,
    )
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import AttentionSpec

    impl = Gfx906FAImpl(
        num_heads=HQ,
        head_size=D,
        scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="float16",
    )
    cls = type(impl)
    assert impl._legacy, "test must exercise the LEGACY=1 fp16 gather path"
    saved = (
        cls._k_gather_buf,
        cls._v_gather_buf,
        cls._gather_retired,
        cls._gather_captured,
        cls._gather_buf_captured,
    )
    try:
        cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired = None, None, {}
        cls._gather_captured = False
        cls._gather_buf_captured = False

        B, L0, max_seq = 2, [100, 64], 128
        nblk = (max_seq + BLOCK - 1) // BLOCK
        num_blocks = B * nblk + 4
        # 0.29 fused cache: one tensor per layer, K/V = strided halves of the
        # last axis; the backend splits it internally (the impl takes the fused
        # tensor, as the backend-level tests do).
        _, v_view, kv = _make_fused_cache(num_blocks, dev)
        k_view = _kv_split(kv)[0]
        # Flat row r of request s lives at slot s * nblk * BLOCK + r
        # (contiguous private blocks, like the existing paged tests).
        # Kf/Vf (contiguous) are the source of truth for the reference;
        # the cache halves are copies kept in sync by the 'KV write' steps.
        Kf = (
            torch.randn(num_blocks * BLOCK, HKV, D, device=dev, dtype=torch.float16)
            * 0.5
        )
        Vf = (
            torch.randn(num_blocks * BLOCK, HKV, D, device=dev, dtype=torch.float16)
            * 0.5
        )
        k_view.copy_(Kf.view(num_blocks, BLOCK, HKV, D))
        _write_v_fused(v_view, Vf)

        def slot_of(s: int, r: int) -> int:
            return s * nblk * BLOCK + r

        # Persistent draft buffers (the speculator's input_buffers /
        # block-table views) — metadata must be built from these ONCE.
        seq_lens = torch.tensor(L0, dtype=torch.int32, device=dev)
        qsl = torch.arange(B + 1, dtype=torch.int32, device=dev)
        bt = torch.zeros(B, nblk, dtype=torch.int32, device=dev)
        for s in range(B):
            bt[s] = torch.arange(
                s * nblk, (s + 1) * nblk, dtype=torch.int32, device=dev
            )
        smap = torch.tensor(
            [slot_of(s, L) for s, L in enumerate(L0)],
            dtype=torch.int64,
            device=dev,
        )
        cam = CommonAttentionMetadata(
            query_start_loc=qsl,
            query_start_loc_cpu=qsl.cpu(),
            seq_lens=seq_lens,
            num_reqs=B,
            num_actual_tokens=B,
            max_query_len=1,
            max_seq_len=max_seq,
            block_table_tensor=bt,
            slot_mapping=smap,
        )
        spec = AttentionSpec(
            block_size=BLOCK, num_kv_heads=HKV, head_size=D, dtype=torch.float16
        )
        builder = Gfx906FAMetadataBuilder(spec, ["l0"], None, torch.device(dev))
        md = builder.build(0, cam)  # the step-1 build — never rebuilt

        def ref_at(q: torch.Tensor, lengths):
            g = HQ // HKV
            scale = 1.0 / math.sqrt(D)
            out = torch.empty(B, HQ, D, dtype=torch.float32, device=dev)
            for s, L in enumerate(lengths):
                kf = Kf[slot_of(s, 0) : slot_of(s, L)].float()
                vf = Vf[slot_of(s, 0) : slot_of(s, L)].float()
                for h in range(HQ):
                    hk = h // g
                    sc = q[s, h].float() @ kf[:, hk].T * scale
                    out[s, h] = torch.softmax(sc, -1) @ vf[:, hk]
            return out

        def run_step(q: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(B, HQ, D, dtype=torch.float16, device=dev)
            impl.forward(None, q, q, q, kv, md, output=out)
            return out

        # Step 1 (Sq=1 per request: each draft step is one query/req).
        q1 = torch.randn(B, HQ, D, device=dev, dtype=torch.float16) * 0.5
        out1 = run_step(q1)
        rel1 = ((out1.float() - ref_at(q1, L0)).norm() / ref_at(q1, L0).norm()).item()
        assert rel1 < 5e-2, f"step 1 rel={rel1:.2e}"

        # Advance to step 2 IN PLACE, mirroring the captured kernels:
        # update_draft_inputs (+1 seq_lens), compute_slot_mappings (new slot),
        # do_kv_cache_update (new K/V row at the new slot).
        new_k = torch.randn(1, HKV, D, device=dev, dtype=torch.float16) * 0.5
        new_v = torch.randn(1, HKV, D, device=dev, dtype=torch.float16) * 0.5
        for s, L in enumerate(L0):
            sl = slot_of(s, L)
            blk, off = sl // BLOCK, sl % BLOCK
            k_view[blk, off] = new_k[0]
            v_view[blk, off] = new_v[0]
            Kf[sl] = new_k[0]
            Vf[sl] = new_v[0]
        seq_lens.add_(1)
        smap.copy_(
            torch.tensor(
                [slot_of(s, L + 1) for s, L in enumerate(L0)],
                dtype=torch.int64,
                device=dev,
            )
        )
        builder.update_draft_decode_metadata(md)  # no-op between steps

        q2 = torch.randn(B, HQ, D, device=dev, dtype=torch.float16) * 0.5
        out2 = run_step(q2)  # SAME md object, advanced persistent state
        L2 = [L + 1 for L in L0]
        rel2 = ((out2.float() - ref_at(q2, L2)).norm() / ref_at(q2, L2).norm()).item()
        assert rel2 < 5e-2, f"step 2 (reused metadata) rel={rel2:.2e}"
        # The stale-lengths reference must NOT match: guards against a vacuous
        # pass (e.g. kv_max ignored entirely).
        stale = ((out2.float() - ref_at(q2, L0)).norm() / ref_at(q2, L0).norm()).item()
        assert stale > 5e-2, "step-2 output ignored the advanced seq_lens"
    finally:
        (
            cls._k_gather_buf,
            cls._v_gather_buf,
            cls._gather_retired,
            cls._gather_captured,
            cls._gather_buf_captured,
        ) = saved


# ---------------------------------------------------------------------------
# FA-COVER-1 step 2: a padded head dim must reproduce the real-dim causal
# reference. Drives the *impl* rather than the raw op, so the padding runs end
# to end: the cache is allocated through the backend's get_kv_cache_shape (so
# the row is padded), K/V go in through do_kv_cache_update - which must zero
# the pad, and the cache is pre-filled with garbage so an unzeroed pad shows up
# as a mismatch - and the query is padded on the way in and sliced on the way
# out. Opted in per test via GFX906_FA_PAD (the default is off).
# The 96 map is the default since the FA-D96 gate, so 72/80 pad onto 96 and 112 onto
# 128; GFX906_FA_PAD96=0 is the rollback map (everything onto 128).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("real_d,padded,pad96", [
    (72, 96, True),    # default since the FA-D96 gate
    (80, 96, True),
    (112, 128, True),
    (72, 128, False),  # rollback map
])
def test_padded_head_dim_matches_torch_ref(real_d, padded, pad96, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("GFX906_FA_PAD", "1")
    monkeypatch.setenv("GFX906_FA_PAD96", "1" if pad96 else "0")
    dev = "cuda"
    L, hq, hkv = 64, 4, 2
    torch.manual_seed(17)

    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FABackend,
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    n_blocks = L // BLOCK
    # The spec keeps the logical dim; vLLM fuses it into the physical cache row.
    spec = Gfx906FABackend.get_kv_cache_shape(n_blocks, BLOCK, hkv, real_d)
    assert spec[-1] == padded
    assert Gfx906FABackend.supports_head_size(real_d)

    # Backend-level (impl) cache in the 0.29 fused layout: [N, Hkv, BLOCK, 2*D], which
    # the impl splits on the last axis after transpose(1, 2). Garbage on purpose: the
    # pad channels must be zeroed by the write path, or a non-zero q8_0 block appears.
    kv = torch.randn(n_blocks, hkv, BLOCK, 2 * padded, dtype=torch.float16, device=dev)
    impl = Gfx906FAImpl(
        num_heads=hq,
        head_size=real_d,
        scale=1.0 / math.sqrt(real_d),
        num_kv_heads=hkv,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="float16",
    )
    assert impl.padded_head_size == padded
    assert impl._head_pad == padded - real_d

    layer = SimpleNamespace(_k_scale=1.0, _v_scale=1.0)
    K = torch.randn(L, hkv, real_d, dtype=torch.float16, device=dev) * 0.5
    V = torch.randn(L, hkv, real_d, dtype=torch.float16, device=dev) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    impl.do_kv_cache_update(layer, K, V, kv, slot)

    kc, vc = kv.transpose(1, 2).split(padded, dim=-1)
    assert kc[..., real_d:].abs().max().item() == 0.0, "K pad not zeroed"
    assert vc[..., real_d:].abs().max().item() == 0.0, "V pad not zeroed"

    m = Gfx906FAMetadata(
        num_actual_tokens=L,
        max_query_len=L,
        max_seq_len=L,
        query_start_loc=torch.tensor([0, L], dtype=torch.int32, device=dev),
        seq_lens=torch.tensor([L], dtype=torch.int32, device=dev),
        block_table=torch.arange(n_blocks, dtype=torch.int32, device=dev).view(
            1, n_blocks
        ),
        slot_mapping=slot,
    )
    q = torch.randn(L, hq, real_d, dtype=torch.float32, device=dev) * 0.5
    out = torch.empty(L, hq, real_d, dtype=torch.float16, device=dev)
    got = impl.forward(layer, q, K, V, kv, m, output=out)
    got = (out if got is None else got).float().view(L, hq, real_d)

    Kf, Vf = K.float(), V.float()
    scale = 1.0 / math.sqrt(real_d)
    for t in (0, 1, L // 2, L - 2, L - 1):
        ref = _windowed_ref(q[t], Kf, Vf, scale, t, None)
        rel = ((got[t] - ref).norm() / ref.norm()).item()
        assert rel < 5e-2, f"D={real_d} row {t}: rel {rel:.4f}"


# ---------------------------------------------------------------------------
# FA-D96 (2026-09-16): 96 became an instantiated kernel head dim. Every other
# test in this file runs D=64/128/256 (or the pad path onto one of those), so
# the new instantiation needs its own pin on BOTH entry points: the gather
# entry (prefill / B=1) and the direct-paged entry (decode). Both are causal
# here; the bidirectional ViT use of D=96 is covered by
# test_vit_bidirectional_matches_sdpa below (72 pads onto 96 now).
# ---------------------------------------------------------------------------

def test_head_dim_96_instantiated_gather_and_paged_vs_fp32_ref():
    """D=96 runs the 96-wide kernel on both paths, vs an fp32 torch reference."""
    dev = "cuda"
    torch.manual_seed(20260916)
    d, hq, hkv, L = 96, 16, 2, 256
    n_blocks = L // BLOCK
    bytes_per_row = (d // 32) * 34
    kc = torch.zeros(n_blocks, BLOCK, hkv, bytes_per_row,
                     dtype=torch.uint8, device=dev)
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                     dtype=torch.float16, device=dev)
    K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[:L].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]  # production layout: unbind(1), non-contiguous
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, -1)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(d)
    Kf, Vf = K.float(), V.float()

    # direct-paged B=1 decode (Sq=1, causal)
    q = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    out = fa.forward_paged_direct(q, kc, vc, bt, sl, scale, None, None)[0, 0]
    ref = _windowed_ref(q[0, :, 0], Kf, Vf, scale, L - 1, None)
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 5e-2, f"D=96 paged decode: rel {rel:.4f}"

    # gather entry (B=1 prefill-shaped: Sq=8 rows at the end of one sequence).
    # q_abs_offset is the absolute position of the FIRST query row of the tile —
    # without it (and with mask=None) the kernel computes full bidirectional
    # attention, which is the ViT case, not this one.
    sq = 8
    sk_pad = (L + 31) // 32 * 32
    k_q8, v_b = fa.gather_paged_kv_q8(kc, vc, bt, sl, sk_pad)
    q2 = torch.randn(1, hq, sq, d, device=dev, dtype=torch.float32) * 0.5
    abs0 = L - sq
    out2 = fa.forward(
        q2, k_q8, v_b, scale, kv_max=sl,
        q_abs_offset=torch.tensor([abs0], dtype=torch.int32, device=dev),
    )[0]  # [Sq, hq, d]
    for t in (0, sq // 2, sq - 1):
        ref2 = _windowed_ref(q2[0, :, t], Kf, Vf, scale, abs0 + t, None)
        rel2 = ((out2[t] - ref2).norm() / ref2.norm()).item()
        assert rel2 < 5e-2, f"D=96 gather row {t}: rel {rel2:.4f}"


# ---------------------------------------------------------------------------
# FA-NONCAUSAL (2026-09-17): a batch marked causal=False (spec-decode drafters
# whose layers are built with causal=False -- DFlash assistants) must attend in
# BOTH directions. The kernel's causal clip is gated on q_abs_offset and its
# sliding-window mask/clips on `window`, so the impl suppresses both for a
# non-causal batch; this pins that, and pins that the flag has an effect (the
# same batch served as causal must differ where the two masks disagree).
# Stage 1: the symmetric +/-window of the reference is not implemented yet, so
# the non-causal mask is the superset (full [0, seq_len)); see
# docs/gfx906/DEVLOG-fa-noncausal.md.
# ---------------------------------------------------------------------------

def test_non_causal_batch_is_bidirectional_vs_torch_ref():
    from types import SimpleNamespace

    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    dev = "cuda"
    torch.manual_seed(20260917)
    L, sq, hq, hkv, d = 64, 8, 4, 2, 128
    n_blocks = L // BLOCK

    kv = torch.zeros(n_blocks, hkv, BLOCK, 2 * d, dtype=torch.float16, device=dev)
    K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5

    impl = Gfx906FAImpl(
        num_heads=hq,
        head_size=d,
        scale=1.0 / math.sqrt(d),
        num_kv_heads=hkv,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="float16",
    )
    layer = SimpleNamespace(_k_scale=1.0, _v_scale=1.0)
    slots = torch.arange(L, dtype=torch.int64, device=dev)
    impl.do_kv_cache_update(layer, K, V, kv, slots)

    # A query BLOCK: sq rows appended after the L cached tokens, i.e. positions
    # L-sq .. L-1. A drafter's block looks exactly like this (context in the
    # cache, block queries on top).
    q = torch.randn(sq, hq, d, device=dev, dtype=torch.float32) * 0.5
    block_slots = torch.arange(L - sq, L, dtype=torch.int64, device=dev)
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
    scale = 1.0 / math.sqrt(d)

    def run(causal: bool):
        m = Gfx906FAMetadata(
            num_actual_tokens=sq,
            max_query_len=sq,
            max_seq_len=L,
            query_start_loc=torch.tensor([0, sq], dtype=torch.int32, device=dev),
            seq_lens=torch.tensor([L], dtype=torch.int32, device=dev),
            block_table=bt,
            slot_mapping=block_slots,
            causal=causal,
        )
        out = torch.empty(sq, hq, d, dtype=torch.float16, device=dev)
        got = impl.forward(layer, q, K, V, kv, m, output=out)
        return (out if got is None else got).float().view(sq, hq, d)

    got_bidir = run(causal=False)
    got_causal = run(causal=True)

    Kf, Vf = K.float(), V.float()
    g = hq // hkv
    for j in (0, sq // 2, sq - 1):
        q_row = q[j]                       # [hq, d]
        qg = q_row.view(hkv, g, d)
        # Non-causal reference: EVERY row sees ALL L keys (both directions).
        s_all = torch.einsum("gjd,lgd->gjl", qg, Kf) * scale
        ref_bidir = torch.einsum(
            "gjl,lgd->gjd", torch.softmax(s_all, -1), Vf
        ).reshape(hq, d)
        rel = ((got_bidir[j] - ref_bidir).norm() / ref_bidir.norm()).item()
        assert rel < 5e-2, f"non-causal row {j}: rel {rel:.4f}"

        # Causal reference for the same row: keys [0, L-sq+j].
        ref_causal = _windowed_ref(q[j], Kf, Vf, scale, L - sq + j, None)
        rel_c = ((got_causal[j] - ref_causal).norm() / ref_causal.norm()).item()
        assert rel_c < 5e-2, f"causal row {j}: rel {rel_c:.4f}"

    # The two arms must differ where the masks differ (rows 0..sq-2 have future
    # keys inside the block). The future keys are amplified below so a diffuse
    # softmax cannot hide the difference -- an un-amplified random batch made this
    # control pass spuriously on this seed.
    fut = L - sq + 1
    V_amp = V.clone()
    V_amp[fut:] = (V_amp[fut:] * 12.0).clamp(-8, 8)
    impl.do_kv_cache_update(layer, K, V_amp, kv, slots)
    b_amp = run(causal=False)
    c_amp = run(causal=True)
    assert not torch.allclose(b_amp[0], c_amp[0], atol=1e-2), (
        "causal=False had no effect: the two arms agree on a batch where the "
        "causal and bidirectional masks differ (the flag is not reaching the kernel)"
    )
    s_amp = torch.einsum("gjd,lgd->gjl", q[0].view(hkv, g, d), Kf) * scale
    ref_amp = torch.einsum(
        "gjl,lgd->gjd", torch.softmax(s_amp, -1), V_amp.float()
    ).reshape(hq, d)
    rel_a = ((b_amp[0] - ref_amp).norm() / ref_amp.norm()).item()
    assert rel_a < 5e-2, f"non-causal row 0 with amplified future keys: rel {rel_a:.4f}"


def test_non_causal_causality_contract(monkeypatch):
    """FA-NONCAUSAL: the bool contract is honoured, the tensor form is not claimed.

    A bool `causal=False` means "no causal clip" -- the same contract TRITON_ATTN and
    ROCM_ATTN honour for that field, so this backend may serve it. A *tensor* causal
    (per-token masks) is not expressible in this kernel: it must keep the causal
    behaviour it had before this feature (with a warning), not silently become
    bidirectional over the whole sequence.
    """
    from types import SimpleNamespace

    from vllm.gfx906_fa.gfx906_fa_backend import Gfx906FABackend, _batch_causal

    bidir = SimpleNamespace(causal=False)
    causal = SimpleNamespace(causal=True)
    per_token = SimpleNamespace(causal=torch.ones(4, dtype=torch.bool))
    missing = SimpleNamespace()  # older metadata without the field

    assert _batch_causal(bidir) is False
    assert _batch_causal(causal) is True
    assert _batch_causal(missing) is True
    # Per-token causality keeps the causal path (status quo), it does not become a
    # full-bidirectional claim.
    assert _batch_causal(per_token) is True

    # The class-level capability and its rollback.
    monkeypatch.delenv("GFX906_FA_NO_NONCAUSAL", raising=False)
    assert Gfx906FABackend.supports_non_causal()
    monkeypatch.setenv("GFX906_FA_NO_NONCAUSAL", "1")
    assert not Gfx906FABackend.supports_non_causal()
