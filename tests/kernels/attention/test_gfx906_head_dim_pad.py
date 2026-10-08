# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Head-dim padding helpers for the gfx906 FA text path (FA-COVER-1 step 2, FA-D96).

Pins the padding map and both rollback switches. Dimensions through 256 can
pad to a supported kernel width; 72/80 use 96 unless PAD96 is disabled.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.gfx906 import (
    pad_head_dim as _pad_head_dim,
)
from vllm.utils.gfx906 import (
    padded_head_size as _padded_head_size,
)


@pytest.fixture
def backend():
    module = pytest.importorskip("vllm.gfx906_fa.gfx906_fa_backend")
    return module.Gfx906FABackend


def test_metadata_rejects_tensor_causality_before_cache_mutation(backend):
    from vllm.gfx906_fa.gfx906_fa_backend import Gfx906FAMetadataBuilder

    common = SimpleNamespace(
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=2,
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([2]),
        block_table_tensor=torch.tensor([[0]]),
        slot_mapping=torch.tensor([0, 1]),
        causal=torch.tensor([True, False]),
    )
    builder = Gfx906FAMetadataBuilder.__new__(Gfx906FAMetadataBuilder)
    with pytest.raises(NotImplementedError, match="TRITON_ATTN"):
        builder.build(0, common)
    assert not backend.forward_includes_kv_cache_update


def test_missing_extension_does_not_register_custom_backend(backend, monkeypatch):
    import vllm.gfx906_fa as package
    import vllm.platforms as platforms
    import vllm.platforms.rocm as rocm
    from vllm.gfx906_fa.gfx906_fa_backend import register
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    monkeypatch.setattr(package, "ext", None)
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_rocm=lambda: True)
    )
    monkeypatch.setattr(rocm, "on_gfx906", lambda: True)
    before = AttentionBackendEnum.CUSTOM.is_overridden()
    register()
    assert AttentionBackendEnum.CUSTOM.is_overridden() == before


@pytest.mark.parametrize(
    "head_size,want,pad96",
    [
        (32, 64, False),
        (64, 64, False),
        (72, 96, True),  # default since the 2026-09-16 FA-D96 gate
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
    assert _padded_head_size(96) == 96  # native since FA-D96
    assert _padded_head_size(72) == 96  # the ViT pads onto 96, not 128
    assert _padded_head_size(80) == 96
    monkeypatch.setenv("GFX906_FA_PAD96", "0")
    assert _padded_head_size(96) == 128  # rollback map
    assert _padded_head_size(72) == 128


def test_kill_switch_restores_exact_dims_only(monkeypatch):
    monkeypatch.setenv("GFX906_FA_PAD", "0")
    assert _padded_head_size(128) == 128
    assert _padded_head_size(96) == 96  # in the default map
    assert _padded_head_size(80) is None
    monkeypatch.setenv("GFX906_FA_PAD", "1")
    assert _padded_head_size(80) == 96
    assert _padded_head_size(288) is None
    # the rollback map drops 96 from the servable set as well (pre-FA-D96 semantics)
    monkeypatch.setenv("GFX906_FA_PAD96", "0")
    assert _padded_head_size(96) == 128
    monkeypatch.setenv("GFX906_FA_PAD", "0")
    assert _padded_head_size(96) is None


def test_supports_head_size_serves_pad_able_dims_by_default(monkeypatch, backend):
    """Pad-able dims are servable now; dims past 256 still are not."""
    monkeypatch.delenv("GFX906_FA_PAD", raising=False)
    monkeypatch.delenv("GFX906_FA_PAD96", raising=False)
    # pad-able: instantiated, below 64 (32 -> 64), and 65..256 (80/112 -> 128)
    for supported in (32, 40, 64, 72, 80, 96, 112, 128, 160, 256):
        assert backend.supports_head_size(supported), supported
    # only dims beyond the largest instantiated kernel dim are out of reach
    for unsupported in (257, 288, 512):
        assert not backend.supports_head_size(unsupported), unsupported
    # kill switch: only the active map's dims are servable (96 is in the default map)
    monkeypatch.setenv("GFX906_FA_PAD", "0")
    assert backend.supports_head_size(128)
    assert backend.supports_head_size(96)
    assert not backend.supports_head_size(80)


def test_customize_spec_widens_both_halves(monkeypatch, backend):
    """The spec must widen BOTH halves (this was Phi-3's 224-row bug)."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    def _spec(d):
        return FullAttentionSpec(
            block_size=16,
            num_kv_heads=32,
            head_size=d,
            head_size_v=d,
            dtype=torch.float16,
            kv_quant_mode=None,
        )

    monkeypatch.setenv("GFX906_FA_PAD", "1")
    monkeypatch.delenv("GFX906_FA_PAD96", raising=False)
    # 72 pads to 96 by default: both halves must move together. Widening only
    # one leaves padded+real (96 + 72 = 168), the class of bug the Phi-3 gate hit.
    widened = backend.customize_spec(_spec(72))
    assert widened.head_size == 96
    assert widened.head_size_v == 96
    # other fields untouched
    assert widened.block_size == 16 and widened.num_kv_heads == 32
    # an exact 96 is already the kernel dim: the spec is returned unchanged
    assert backend.customize_spec(_spec(96)).head_size == 96
    # the rollback map widens onto 128 instead
    monkeypatch.setenv("GFX906_FA_PAD96", "0")
    widened128 = backend.customize_spec(_spec(72))
    assert widened128.head_size == 128 and widened128.head_size_v == 128
    assert backend.customize_spec(_spec(96)).head_size == 128
    monkeypatch.delenv("GFX906_FA_PAD96", raising=False)
    # kill switch: unchanged
    monkeypatch.setenv("GFX906_FA_PAD", "0")
    assert backend.customize_spec(_spec(72)).head_size == 72
    monkeypatch.setenv("GFX906_FA_PAD", "1")
    # a dim that cannot be padded is left alone even when opted in
    big = FullAttentionSpec(
        block_size=16,
        num_kv_heads=32,
        head_size=512,
        dtype=torch.float16,
        kv_quant_mode=None,
    )
    assert backend.customize_spec(big).head_size == 512


@pytest.mark.parametrize("real_d", [72, 80, 96, 112])
@pytest.mark.parametrize("pad96", [0, 1])
def test_spec_and_shape_agree_on_the_padded_row(real_d, pad96, monkeypatch, backend):
    """The spec's page size and the declared shape must describe the same row.

    The invariant the Phi-3 gate broke: vLLM sizes a page from the spec
    (block * num_kv_heads * (head_size + head_size_v) * dtype) and builds the tensor
    get_kv_cache_shape, so a mismatch makes the allocator invent a third row width.
    Run for both pad targets (128, and 96 under the FA-D96 opt-in).
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    monkeypatch.setenv("GFX906_FA_PAD", "1")
    monkeypatch.setenv("GFX906_FA_PAD96", "1" if pad96 else "0")
    spec = backend.customize_spec(
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=32,
            head_size=real_d,
            head_size_v=real_d,
            dtype=torch.float16,
            kv_quant_mode=None,
        )
    )
    shape = backend.get_kv_cache_shape(2, spec.block_size, spec.num_kv_heads, real_d)
    row_elements = shape[1] * shape[-1]  # Logical shape includes the K/V axis.
    assert row_elements == spec.head_size + spec.head_size_v, (shape, spec)
    assert (
        spec.page_size_bytes == spec.block_size * spec.num_kv_heads * row_elements * 2
    )


@pytest.mark.parametrize("head_size", [72, 80])
@pytest.mark.parametrize("causal", [False, True])
def test_cpu_padding_preserves_unquantized_attention(head_size, causal, backend):
    """Checks padding math; it does not emulate the Q8 device kernel."""
    impl_cls = backend.get_impl_cls()
    impl = impl_cls.__new__(impl_cls)
    impl.head_size = head_size
    impl.padded_head_size = _padded_head_size(head_size)
    impl._head_pad = impl.padded_head_size - head_size
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(2, 3, 4, head_size, generator=generator, dtype=torch.float64)
    k = torch.randn(2, 3, 6, head_size, generator=generator, dtype=torch.float64)
    v = torch.randn(2, 3, 6, head_size, generator=generator, dtype=torch.float64)
    scale = head_size**-0.5
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=scale
    )
    actual = torch.nn.functional.scaled_dot_product_attention(
        impl._pad_last_dim(q),
        impl._pad_last_dim(k),
        impl._pad_last_dim(v),
        is_causal=causal,
        scale=scale,
    )
    torch.testing.assert_close(
        actual[..., :head_size], expected, rtol=1e-12, atol=1e-12
    )
    assert torch.count_nonzero(actual[..., head_size:]) == 0
