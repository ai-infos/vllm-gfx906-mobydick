# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Tests for the gfx906 M=1 fused align+sort kernel (C1 stage 1 + NH-5).

Covers both served (E, topk) pairs: (256, 8) Qwen3.5-35B and
(128, 6) Nemotron-3.5-Lightning.

Run `pytest tests/kernels/moe/test_moe_align_m1_gfx906.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.gfx906_w4a16_moe import (
    _moe_align_block_size_fused_m1,
    _use_fused_align_m1,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx906
else:

    def on_gfx906() -> bool:
        return False


def _has_op() -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "moe_align_block_size_m1_gfx906"
    )


SHAPES = [(256, 8), (128, 6)]  # (E, topk)


def _prod_chain(topk_ids, num_experts, dev):
    """Production two-kernel chain at the wrapper's buffer sizes."""
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        topk_ids, 1, num_experts, None
    )
    k = topk_ids.shape[1]
    assert sorted_ids.shape == (k,)
    assert expert_ids.shape == (k,)
    return sorted_ids, expert_ids, ntp


def _cases_for(seed: int, num_experts: int, topk: int, g, dev):
    """Random + tie-heavy topk_id patterns for one (E, topk) pair."""
    cases = [torch.randint(0, num_experts, (1, topk), generator=g, device=dev)]
    if seed % 4 == 0:
        # tie-heavy: all slots over 2 experts
        cases.append(
            torch.randint(0, 2, (1, topk), generator=g, device=dev).int())
    if seed % 4 == 1:
        # all one expert
        cases.append(torch.full((1, topk), seed % num_experts, device=dev,
                                dtype=torch.int32))
    if seed % 4 == 2:
        # pairs: expert i gets slots i and i+topk//2
        cases.append(torch.tensor(
            [[i % (topk // 2) for i in range(topk)]],
            device=dev, dtype=torch.int32))
    if seed % 4 == 3:
        # topk distinct experts
        cases.append(torch.arange(topk, device=dev, dtype=torch.int32)
                     .reshape(1, topk))
    return cases


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx906() and _has_op()),
    reason="gfx906 M=1 fused align kernel test (C1 stage 1 / NH-5)")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", range(24))
def test_m1_fused_align_bit_equal_to_production(seed, shape):
    """The single-CTA kernel must reproduce the two-kernel generic chain
    bit-for-bit (sorted_token_ids / expert_ids / num_tokens_post_pad) for
    random and tie-heavy topk_ids. Within-expert slot order is the
    issuing-lane order in both (single warp)."""
    num_experts, topk = shape
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    for topk_ids in _cases_for(seed, num_experts, topk, g, dev):
        topk_ids = topk_ids.int()
        ref_s, ref_e, ref_n = _prod_chain(topk_ids, num_experts, dev)
        fus_s, fus_e, fus_n = _moe_align_block_size_fused_m1(
            topk_ids, num_experts)
        assert torch.equal(fus_n, ref_n), (
            f"ntp diverge: {fus_n.tolist()} vs {ref_n.tolist()}")
        assert torch.equal(fus_s, ref_s), (
            f"sorted diverge: {fus_s.tolist()} vs {ref_s.tolist()} "
            f"topk_ids={topk_ids.tolist()}")
        assert torch.equal(fus_e, ref_e), (
            f"expert_ids diverge: {fus_e.tolist()} vs {ref_e.tolist()} "
            f"topk_ids={topk_ids.tolist()}")


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx906()),
    reason="gfx906 M=1 fused align dispatch test (C1 stage 1 / NH-5)")
def test_m1_fused_align_dispatch_gate(monkeypatch):
    """The gate must fire only for the exact decode shapes (M=1, bsm=1,
    no expert_map, int32, flag on) AND (E, topk) in the supported pair
    set: (256, 8) and (128, 6), AND the V1 runner (REL30-1: on V2 the op's
    out-params come back uninitialized)."""
    monkeypatch.setenv("VLLM_GFX906_ALIGN_M1", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    dev = "cuda"
    em = torch.zeros(4, device=dev, dtype=torch.int32)
    cases = [
        # (topk_ids, bsm, E, expert_map, expected)
        (torch.zeros(1, 8, device=dev, dtype=torch.int32), 1, 256, None,
         True),
        (torch.zeros(1, 6, device=dev, dtype=torch.int32), 1, 128, None,
         True),  # NH-5: Nemotron shape
        (torch.zeros(4, 8, device=dev, dtype=torch.int32), 1, 256, None,
         False),  # M>1
        (torch.zeros(1, 6, device=dev, dtype=torch.int32), 1, 256, None,
         False),  # (E, topk) not a served pair
        (torch.zeros(1, 8, device=dev, dtype=torch.int32), 1, 128, None,
         False),  # (E, topk) not a served pair
        (torch.zeros(1, 6, device=dev, dtype=torch.int32), 4, 128, None,
         False),  # bsm!=1
        (torch.zeros(1, 6, device=dev, dtype=torch.int32), 1, 128, em,
         False),  # expert_map
        (torch.zeros(1, 8, device=dev, dtype=torch.int64), 1, 256, None,
         False),  # dtype
    ]
    for topk_ids, bsm, ne, emap, want in cases:
        got = _use_fused_align_m1(topk_ids, bsm, ne, emap)
        assert got is want, (
            f"gate={got} want={want} for "
            f"M={topk_ids.shape[0]} topk={topk_ids.shape[1]} E={ne} "
            f"bsm={bsm} map={emap is not None} dtype={topk_ids.dtype}")
    # Default is ON (serving A/B PASS); the opt-out flag disables it.
    monkeypatch.delenv("VLLM_GFX906_ALIGN_M1")
    assert _use_fused_align_m1(
        torch.zeros(1, 8, device=dev, dtype=torch.int32), 1, 256, None)
    assert _use_fused_align_m1(
        torch.zeros(1, 6, device=dev, dtype=torch.int32), 1, 128, None)
    monkeypatch.setenv("VLLM_GFX906_ALIGN_M1", "0")
    assert not _use_fused_align_m1(
        torch.zeros(1, 8, device=dev, dtype=torch.int32), 1, 256, None)
    assert not _use_fused_align_m1(
        torch.zeros(1, 6, device=dev, dtype=torch.int32), 1, 128, None)

    # REL30-1: the fused align is V1-only. V2 (explicit) and the auto/None
    # default (which resolves to V2) must both skip it -- an uninitialized
    # align made the M=1 GEMM index the expert tables with garbage ids.
    monkeypatch.delenv("VLLM_GFX906_ALIGN_M1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    assert not _use_fused_align_m1(
        torch.zeros(1, 8, device=dev, dtype=torch.int32), 1, 256, None)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER")
    assert not _use_fused_align_m1(
        torch.zeros(1, 8, device=dev, dtype=torch.int32), 1, 256, None)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx906() and _has_op()),
    reason="gfx906 M=1 fused align graph test (C1 stage 1 / NH-5)")
@pytest.mark.parametrize("shape", SHAPES)
def test_m1_fused_align_graph_capturable(shape):
    """The kernel is capture-safe (single CTA, no D2H, fixed grid) and
    replay-stable: capturing the multi-layer chain and replaying must keep
    producing bit-correct outputs."""
    num_experts, topk = shape
    dev = "cuda"
    topk_ids = torch.randint(0, num_experts, (1, topk), device=dev).int()
    s = torch.empty(topk, device=dev, dtype=torch.int32)
    e = torch.empty(topk, device=dev, dtype=torch.int32)
    n = torch.empty(1, device=dev, dtype=torch.int32)

    def chain():
        for _ in range(23):
            torch.ops._rocm_C.moe_align_block_size_m1_gfx906(
                topk_ids, num_experts, 1, s, e, n)

    ref_s, ref_e, ref_n = _prod_chain(topk_ids, num_experts, dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        chain()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(s, ref_s)
    assert torch.equal(e, ref_e)
    assert torch.equal(n, ref_n)
