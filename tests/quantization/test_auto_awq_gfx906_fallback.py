# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""N1 behavior sanity gate: the expected AutoAWQMoEMarlin fallback is quiet
on gfx906 (one info line per process) while platforms where the fallback is
unexpected keep the per-layer warning.

Exercises ``AutoAWQConfig.get_quant_method`` for a ``RoutedExperts`` layer
with the Marlin-support check monkeypatched to False, asserting both that the
MoeWNA16 path is taken (behavior unchanged) and the log level of the fallback
message. ``MoeWNA16Config.from_config`` is stubbed so the test stays focused
on the auto_awq dispatch/logging logic (what N1 changes) instead of dragging
in the full FusedMoEConfig/WNA16-oracle machinery. Requires a GPU environment
to import auto_awq (GPU-dependent imports); skipped otherwise.
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]


@pytest.fixture(autouse=True)
def _clear_once_caches():
    """info_once/warning_once dedupe via module-level lru_cache; clear them so
    reruns of these tests (e.g. pytest --count) don't see suppressed records."""
    from vllm.logger import _print_info_once, _print_warning_once

    _print_info_once.cache_clear()
    _print_warning_once.cache_clear()
    yield
    _print_info_once.cache_clear()
    _print_warning_once.cache_clear()


def _make_config():
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

    # full_config mirrors a real quantize_config.json; the fallback path
    # forwards it to MoeWNA16Config.from_config (stubbed in the tests).
    return AutoAWQConfig(
        weight_bits=4,
        group_size=128,
        zero_point=True,
        lm_head_quantized=False,
        full_config={
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "lm_head": False,
        },
    )


def _layer():
    """A RoutedExperts instance without constructing the real (heavy) layer:
    ``get_quant_method`` only needs the isinstance check on this path."""
    from vllm.model_executor.layers.fused_moe.layer import RoutedExperts

    return Mock(spec=RoutedExperts)


def _patch_wna16(monkeypatch) -> tuple[list, object]:
    """Stub MoeWNA16Config in the moe_wna16 module (imported lazily inside
    get_quant_method). Returns (config_calls, sentinel quant method)."""
    import vllm.model_executor.layers.quantization.moe_wna16 as moe_wna16

    calls: list[dict] = []
    sentinel = object()

    class _Stub:
        @classmethod
        def from_config(cls, full_config):
            calls.append(full_config)
            return cls()

        def get_quant_method(self, layer, prefix):
            return sentinel

    monkeypatch.setattr(moe_wna16, "MoeWNA16Config", _Stub)
    return calls, sentinel


def test_gfx906_fallback_is_single_info(caplog, monkeypatch):
    """gfx906: intentional fallback -> one info line, no warning; the WNA16
    path is still taken for every layer (behavior unchanged)."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.auto_awq.on_gfx906", lambda: True
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.auto_awq.check_moe_marlin_supports_layer",
        lambda *a, **k: False,
    )
    wna16_calls, sentinel = _patch_wna16(monkeypatch)

    config = _make_config()
    layer = _layer()
    with caplog.at_level(logging.INFO, logger="vllm"):
        qm1 = config.get_quant_method(layer, "model.layers.0.mlp.experts")
        qm2 = config.get_quant_method(layer, "model.layers.1.mlp.experts")

    # Behavior: the WNA16 path is taken for every layer with the full
    # checkpoint config forwarded.
    assert qm1 is sentinel and qm2 is sentinel, (
        "expected the MoeWNA16 quant method for both layers, "
        f"got {qm1!r} / {qm2!r}"
    )
    assert wna16_calls == [config.full_config] * 2

    # Logging: exactly one fallback line, at INFO (not WARNING).
    fb = [r for r in caplog.records if "AutoAWQMoEMarlin" in r.getMessage()]
    assert len(fb) == 1, f"expected 1 fallback record, got {len(fb)}: {fb}"
    assert fb[0].levelno == logging.INFO
    assert not any(
        r.levelno >= logging.WARNING and "AutoAWQMoEMarlin" in r.getMessage()
        for r in caplog.records
    ), "gfx906 fallback must not emit a warning"


def test_non_gfx906_fallback_keeps_warning(caplog, monkeypatch):
    """Platforms where the Marlin->WNA16 fallback is unexpected keep the
    per-layer warning. ``warning_once`` dedupes by (message, prefix) — so a
    repeated call for the same layer yields one record, exactly as before N1."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.auto_awq.on_gfx906", lambda: False
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.auto_awq.check_moe_marlin_supports_layer",
        lambda *a, **k: False,
    )
    wna16_calls, sentinel = _patch_wna16(monkeypatch)

    config = _make_config()
    layer = _layer()
    prefix = "model.layers.0.mlp.experts"
    with caplog.at_level(logging.WARNING, logger="vllm"):
        qm1 = config.get_quant_method(layer, prefix)
        qm2 = config.get_quant_method(layer, prefix)

    assert qm1 is sentinel and qm2 is sentinel
    assert wna16_calls == [config.full_config] * 2

    fb = [r for r in caplog.records if "AutoAWQMoEMarlin" in r.getMessage()]
    assert len(fb) == 1, f"expected deduped warning, got {len(fb)}: {fb}"
    assert fb[0].levelno >= logging.WARNING
