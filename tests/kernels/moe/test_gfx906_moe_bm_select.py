# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""The gfx906 grouped-MoE M-tile heuristic and its A/B pin (C2-BM>=2).

The heuristic is the shipped one (`em <= 32 -> 1`, `<= 512 -> 4`, else 8) and
`VLLM_GFX906_MOE_BM` pins it so a serving A/B can be a one-flag run. Both
directions are pinned here because the knob is what the queued BM=2/BM=1 gate
depends on: a silent heuristic change would invalidate that A/B's control arm.
"""

import pytest

from vllm.model_executor.layers.fused_moe.experts.gfx906_w4a16_moe import (
    _block_size_m_for,
)


@pytest.mark.parametrize(
    "M,topk,want",
    [
        (1, 8, 1),    # em 8: B=1 MTP k=3 (M=4 -> em 32) is the boundary below
        (4, 8, 1),    # em 32: B=1 MTP k=3
        (8, 8, 4),    # em 64: B=8 greedy -- the sweep's -14.8 % point
        (16, 8, 4),   # em 128: B=4 MTP k=3 (our serving default) -- -14.4 %
        (32, 8, 4),   # em 256 -- -10.7 %
        (64, 8, 4),   # em 512: last BM=4 bucket
        (128, 8, 8),  # em 1024: prefill bucket
    ],
)
def test_shipped_heuristic_is_unchanged(M, topk, want, monkeypatch):
    monkeypatch.delenv("VLLM_GFX906_MOE_BM", raising=False)
    assert _block_size_m_for(M, topk) == want


def test_env_pin_overrides_the_mid_bucket_only(monkeypatch):
    monkeypatch.setenv("VLLM_GFX906_MOE_BM", "2")
    assert _block_size_m_for(16, 8) == 2  # em 128, mid bucket: the pin applies
    # the low bucket keeps BM=1 (the M=1 path) and prefill keeps BM=8, so a coarse
    # pin cannot silently disable either (this is what made the first BM=2 arm read
    # -10.6 % in serving: it had moved the partial-acceptance steps off BM=1)
    assert _block_size_m_for(4, 8) == 1   # em 32
    assert _block_size_m_for(128, 8) == 8  # em 1024


def test_env_pin_rejects_a_non_template_bm(monkeypatch):
    monkeypatch.setenv("VLLM_GFX906_MOE_BM", "3")
    with pytest.raises(AssertionError, match="must be 1, 2, 4 or 8"):
        _block_size_m_for(16, 8)
