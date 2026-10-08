# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for decisions that protect gfx906 kernel execution."""

import pytest
import torch

from vllm.utils.gfx906 import (
    batch_causal,
    fused_align_m1_supported,
    padded_head_size,
)


@pytest.mark.parametrize("causal", [True, False])
def test_boolean_causality_is_preserved(causal):
    assert batch_causal(causal) is causal


@pytest.mark.parametrize("causal", [torch.tensor(True), torch.tensor([True, False])])
def test_tensor_causality_is_rejected_instead_of_changing_the_mask(causal):
    with pytest.raises(NotImplementedError, match="TRITON_ATTN"):
        batch_causal(causal)


@pytest.mark.parametrize("runner", [None, True, False])
@pytest.mark.parametrize("experts,topk", [(256, 8), (128, 6)])
def test_fused_align_requires_explicit_v1(runner, experts, topk, monkeypatch):
    monkeypatch.delenv("VLLM_GFX906_ALIGN_M1", raising=False)
    ids = torch.zeros(1, topk, dtype=torch.int32)
    assert fused_align_m1_supported(ids, 1, experts, None, runner) is (runner is False)


@pytest.mark.parametrize(
    "rows,topk,experts,block,dtype,mapped",
    [
        (2, 8, 256, 1, torch.int32, False),
        (1, 6, 256, 1, torch.int32, False),
        (1, 8, 256, 4, torch.int32, False),
        (1, 8, 256, 1, torch.int64, False),
        (1, 8, 256, 1, torch.int32, True),
    ],
)
def test_fused_align_rejects_shapes_outside_its_contract(
    rows, topk, experts, block, dtype, mapped, monkeypatch
):
    monkeypatch.setenv("VLLM_GFX906_ALIGN_M1", "1")
    ids = torch.zeros(rows, topk, dtype=dtype)
    expert_map = torch.arange(experts) if mapped else None
    assert not fused_align_m1_supported(ids, block, experts, expert_map, False)


def test_fused_align_kill_switch(monkeypatch):
    monkeypatch.setenv("VLLM_GFX906_ALIGN_M1", "0")
    assert not fused_align_m1_supported(
        torch.zeros(1, 8, dtype=torch.int32), 1, 256, None, False
    )


def test_fused_align_rejects_a_non_matrix_route(monkeypatch):
    monkeypatch.setenv("VLLM_GFX906_ALIGN_M1", "1")
    assert not fused_align_m1_supported(
        torch.zeros(8, dtype=torch.int32), 1, 256, None, False
    )


@pytest.mark.parametrize("head_size", [-1, 0, 257])
def test_invalid_head_dimensions_are_not_advertised(head_size, monkeypatch):
    monkeypatch.delenv("GFX906_FA_PAD", raising=False)
    assert padded_head_size(head_size) is None


def test_rocm_predicates_are_safe_without_a_hip_build():
    if torch.version.hip is not None:
        pytest.skip("Checks the non-HIP import path")
    from vllm.platforms.rocm import _get_gcn_arch, on_gfx906

    assert _get_gcn_arch() == ""
    assert not on_gfx906()


def test_cache_gather_honors_noncontiguous_fused_views_and_sequence_bounds():
    from vllm.gfx906_fa.gfx906_fa_paged import _gather_kv

    # [blocks, heads, slots, K+V] is the production fused layout.
    cache = torch.arange(4 * 2 * 4 * 128, dtype=torch.float32).reshape(4, 2, 4, 128)
    key, value = cache.transpose(1, 2).split(64, dim=-1)
    assert not key.is_contiguous() and not value.is_contiguous()
    table = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32)
    lengths = torch.tensor([5, 8], dtype=torch.int32)
    got_key, got_value = _gather_kv(key, value, table, lengths, 8)
    for batch, blocks in enumerate([[2, 0], [1, 3]]):
        length = int(lengths[batch])
        for source, actual in [(key, got_key), (value, got_value)]:
            expected = torch.cat([source[block] for block in blocks]).transpose(0, 1)
            torch.testing.assert_close(actual[batch, :, :length], expected[:, :length])
            assert torch.count_nonzero(actual[batch, :, length:]) == 0
