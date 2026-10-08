# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Tests for the NH-5 single-group grouped_topk fast path
(n_group == 1, topk_group == 1 — the Nemotron-H degenerate case).

The fast path must be bit-equal to the generic grouped_topk chain: it
removes only the provably no-op group topk / group-mask nodes.

Run `pytest tests/kernels/moe/test_grouped_topk_single_group.py`.
"""

import pytest
import torch

import vllm.envs as envs
from vllm.config import (
    CompilationConfig,
    VllmConfig,
    get_cached_compilation_config,
    set_current_vllm_config,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    _grouped_topk_single_group,
    grouped_topk,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _generic_reference(
    gating: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    routed_scaling_factor: float,
    bias: torch.Tensor | None,
    use_sorted: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The generic grouped_topk chain with n_group = topk_group = 1,
    copied statement-for-statement from grouped_topk (pre-NH-5)."""
    num_expert_group = 1
    topk_group = 1
    if scoring_func == "softmax":
        scores = torch.softmax(gating, dim=-1)
    else:
        scores = gating.sigmoid()
    num_token = scores.size(0)
    if bias is not None:
        original_scores = scores
        scores = scores + bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
    else:
        group_scores = scores.view(
            num_token, num_expert_group, -1
        ).max(dim=-1).values
    group_idx = torch.topk(
        group_scores, k=topk_group, dim=-1, sorted=use_sorted
    )[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    if bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(
            tmp_scores, k=topk, dim=-1, sorted=use_sorted
        )
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def _cases(seed: int, m: int, e: int, topk: int, g, dev) -> list[torch.Tensor]:
    """Random + tie-heavy gating logits for one (m, e, topk) shape."""
    cases = [torch.randn(m, e, generator=g, device=dev)]
    if seed % 3 == 0:
        # tie-heavy: logits over 4 distinct values
        cases.append(torch.randint(0, 4, (m, e), generator=g, device=dev).float())
    if seed % 3 == 1:
        # all-equal rows (total ties)
        cases.append(
            torch.full((m, e), float(seed % 5), device=dev, dtype=torch.float32))
    if seed % 3 == 2:
        # blocks of ties: expert i == expert (i % 16)
        base = torch.randint(-3, 3, (m, 16), generator=g, device=dev).float()
        cases.append(base.repeat_interleave(e // 16, dim=1)[:, :e])
    return cases


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_rocm()),
    reason="NH-5 single-group fast path test needs a GPU")
@pytest.mark.parametrize("shape", [(1, 128, 6), (4, 128, 6), (1, 256, 8)])
@pytest.mark.parametrize("seed", range(6))
def test_single_group_bit_equal_to_generic(monkeypatch, seed, shape):
    """The fast path output must be bit-equal to the generic chain for
    random and tie-heavy inputs (same aten::topk kernel on bit-identical
    input; weights path unchanged)."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    m, e, topk = shape
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    bias = torch.randn(e, generator=g, device=dev)
    for scoring_func in ("sigmoid", "softmax"):
        for renormalize in (True, False):
            for scale in (1.0, 2.5):
                for with_bias in (True, False):
                    use_sorted = True  # deterministic tie order (see above)
                    b = bias if with_bias else None
                    for gating in _cases(seed, m, e, topk, g, dev):
                        ref_w, ref_i = _generic_reference(
                            gating, topk, renormalize, scoring_func,
                            scale, b, use_sorted)
                        got_w, got_i = _grouped_topk_single_group(
                            gating, topk, renormalize, scoring_func,
                            scale, b)
                        assert torch.equal(got_w, ref_w), (
                            f"weights diverge: {scoring_func} "
                            f"renorm={renormalize} scale={scale} "
                            f"bias={with_bias}\n got={got_w.tolist()}\n"
                            f" ref={ref_w.tolist()}")
                        assert torch.equal(got_i, ref_i), (
                            f"ids diverge: {scoring_func} "
                            f"renorm={renormalize} scale={scale} "
                            f"bias={with_bias}\n got={got_i.tolist()}\n"
                            f" ref={ref_i.tolist()}\n gating={gating.tolist()}")


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_rocm()),
    reason="NH-5 single-group fast path test needs a GPU")
def test_single_group_compiled_env_toggle_bit_equal(monkeypatch):
    """End-to-end: the COMPILED grouped_topk with the NH-5 fast path on
    must be bit-equal to it with the fast path off (both branches under
    the production compile backend, Nemotron shape E=128/topk=6)."""
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["all"]))
    get_cached_compilation_config.cache_clear()
    set_random_seed(0)
    m, e, topk = 1, 128, 6
    gating = torch.randn(m, e, device="cuda")
    bias = torch.randn(e, device="cuda")
    hidden = torch.empty((m, 2688), dtype=torch.float16, device="cuda")

    with set_current_vllm_config(vllm_config), monkeypatch.context() as mp:
        mp.setattr(envs, "VLLM_BATCH_INVARIANT", True)
        mp.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")
        mp.setenv("VLLM_GFX906_TOPK_SINGLE_GROUP", "0")
        w0, i0 = grouped_topk(
            hidden, gating, topk, True, 1, 1, "sigmoid", 2.5, bias)
        mp.setenv("VLLM_GFX906_TOPK_SINGLE_GROUP", "1")
        w1, i1 = grouped_topk(
            hidden, gating, topk, True, 1, 1, "sigmoid", 2.5, bias)
    assert torch.equal(w0, w1), (
        f"compiled weights diverge:\n got={w1.tolist()}\n ref={w0.tolist()}")
    assert torch.equal(i0, i1), (
        f"compiled ids diverge:\n got={i1.tolist()}\n ref={i0.tolist()}")
