# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Head-dim padding helpers for the gfx906 FA text path (FA-COVER-1 step 2, FA-D96).

Pins the padding map and the gap: text head dims outside {64,96,128,256} are still
rejected, while the ViT path already pads them. 96 became an instantiated kernel dim
on 2026-09-16 (FA-D96), so 72/80 now pad to 96 rather than 128.
"""

import pytest
import torch

try:
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FABackend,
        _pad_head_dim,
        _padded_head_size,
    )
except ImportError as exc:  # pragma: no cover - needs the built gfx906 extension
    pytest.skip(f"gfx906 FA backend unavailable: {exc}", allow_module_level=True)


@pytest.mark.parametrize(
    "head_size,want,pad96",
    [
        (32, 64, False),
        (64, 64, False),
        (72, 96, True),   # default since the 2026-09-16 FA-D96 gate
        (80, 96, True),
        (96, 96, True),
        (112, 128, True),
        (128, 128, True),
        (160, 256, True),
        (256, 256, True),
        (288, None, True),
        (1024, None, True),
        (72, 128, False),  # GFX906_FA_PAD96=0: the rollback map
        (80, 128, False),
        (96, 128, False),
        (112, 128, False),
    ],
)
def test_pad_map(head_size, want, pad96, monkeypatch):
    monkeypatch.setenv("GFX906_FA_PAD96", "1" if pad96 else "0")
    assert _pad_head_dim(head_size) == want


def test_padding_default_is_on_after_the_fa_d96_gate(monkeypatch):
    """Default map is (64, 96, 128, 256) since the 2026-09-16 FA-D96 gate."""
    monkeypatch.delenv("GFX906_FA_PAD", raising=False)
    monkeypatch.delenv("GFX906_FA_PAD96", raising=False)
    assert _padded_head_size(128) == 128  # instantiated dims are unaffected
    assert _padded_head_size(96) == 96    # native since FA-D96
    assert _padded_head_size(72) == 96    # the ViT pads onto 96, not 128
    assert _padded_head_size(80) == 96
    monkeypatch.setenv("GFX906_FA_PAD96", "0")
    assert _padded_head_size(96) == 128   # rollback map
    assert _padded_head_size(72) == 128


def test_kill_switch_restores_exact_dims_only(monkeypatch):
    monkeypatch.setenv("GFX906_FA_PAD", "0")
    assert _padded_head_size(128) == 128
    assert _padded_head_size(96) == 96   # in the default map
    assert _padded_head_size(80) is None
    monkeypatch.setenv("GFX906_FA_PAD", "1")
    assert _padded_head_size(80) == 96
    assert _padded_head_size(288) is None
    # the rollback map drops 96 from the servable set as well (pre-FA-D96 semantics)
    monkeypatch.setenv("GFX906_FA_PAD96", "0")
    assert _padded_head_size(96) == 128
    monkeypatch.setenv("GFX906_FA_PAD", "0")
    assert _padded_head_size(96) is None


def test_supports_head_size_serves_pad_able_dims_by_default(monkeypatch):
    """Pad-able dims are servable now; dims past 256 still are not."""
    monkeypatch.delenv("GFX906_FA_PAD", raising=False)
    monkeypatch.delenv("GFX906_FA_PAD96", raising=False)
    # pad-able: instantiated, below 64 (32 -> 64), and 65..256 (80/112 -> 128)
    for supported in (32, 40, 64, 72, 80, 96, 112, 128, 160, 256):
        assert Gfx906FABackend.supports_head_size(supported), supported
    # only dims beyond the largest instantiated kernel dim are out of reach
    for unsupported in (257, 288, 512):
        assert not Gfx906FABackend.supports_head_size(unsupported), unsupported
    # kill switch: only the active map's dims are servable (96 is in the default map)
    monkeypatch.setenv("GFX906_FA_PAD", "0")
    assert Gfx906FABackend.supports_head_size(128)
    assert Gfx906FABackend.supports_head_size(96)
    assert not Gfx906FABackend.supports_head_size(80)

def test_customize_spec_widens_both_halves(monkeypatch):
    """The spec must widen BOTH halves (this was Phi-3's 224-row bug)."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    def _spec(d):
        return FullAttentionSpec(
            block_size=16, num_kv_heads=32, head_size=d, head_size_v=d,
            dtype=torch.float16, kv_quant_mode=None,
        )

    monkeypatch.setenv("GFX906_FA_PAD", "1")
    monkeypatch.delenv("GFX906_FA_PAD96", raising=False)
    # 72 pads to 96 by default: both halves must move together. Widening only
    # one leaves padded+real (96 + 72 = 168), the class of bug the Phi-3 gate hit.
    widened = Gfx906FABackend.customize_spec(_spec(72))
    assert widened.head_size == 96
    assert widened.head_size_v == 96
    # other fields untouched
    assert widened.block_size == 16 and widened.num_kv_heads == 32
    # an exact 96 is already the kernel dim: the spec is returned unchanged
    assert Gfx906FABackend.customize_spec(_spec(96)).head_size == 96
    # the rollback map widens onto 128 instead
    monkeypatch.setenv("GFX906_FA_PAD96", "0")
    widened128 = Gfx906FABackend.customize_spec(_spec(72))
    assert widened128.head_size == 128 and widened128.head_size_v == 128
    assert Gfx906FABackend.customize_spec(_spec(96)).head_size == 128
    monkeypatch.delenv("GFX906_FA_PAD96", raising=False)
    # kill switch: unchanged
    monkeypatch.setenv("GFX906_FA_PAD", "0")
    assert Gfx906FABackend.customize_spec(_spec(72)).head_size == 72
    monkeypatch.setenv("GFX906_FA_PAD", "1")
    # a dim that cannot be padded is left alone even when opted in
    big = FullAttentionSpec(
        block_size=16, num_kv_heads=32, head_size=512, dtype=torch.float16,
        kv_quant_mode=None,
    )
    assert Gfx906FABackend.customize_spec(big).head_size == 512

@pytest.mark.parametrize("real_d", [72, 80, 96, 112])
@pytest.mark.parametrize("pad96", [0, 1])
def test_spec_and_shape_agree_on_the_padded_row(real_d, pad96, monkeypatch):
    """The spec's page size and the declared shape must describe the same row.

    The invariant the Phi-3 gate broke: vLLM sizes a page from the spec
    (block * num_kv_heads * (head_size + head_size_v) * dtype) and builds the tensor
    get_kv_cache_shape, so a mismatch makes the allocator invent a third row width.
    Run for both pad targets (128, and 96 under the FA-D96 opt-in).
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    monkeypatch.setenv("GFX906_FA_PAD", "1")
    monkeypatch.setenv("GFX906_FA_PAD96", "1" if pad96 else "0")
    spec = Gfx906FABackend.customize_spec(
        FullAttentionSpec(
            block_size=16, num_kv_heads=32, head_size=real_d, head_size_v=real_d,
            dtype=torch.float16, kv_quant_mode=None,
        )
    )
    shape = Gfx906FABackend.get_kv_cache_shape(
        2, spec.block_size, spec.num_kv_heads, real_d
    )
    row_elements = shape[1] * shape[-1]  # dim 1 pairs K/V; the last is the padded row
    assert row_elements == spec.head_size + spec.head_size_v, (shape, spec)
    assert (
        spec.page_size_bytes
        == spec.block_size * spec.num_kv_heads * row_elements * 2
    )
