# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen4Exp family predicate: every spelling, and nothing outside it.

A checkpoint's ``model_type`` is not one string: the family declares
``qwen4_exp`` (wrapper and vision), ``qwen4_exp_text`` (the text sub-config), and
``SpeculativeConfig`` rewrites it to ``qwen4_exp_mtp`` for the drafter. Anything
matching the family by exact string silently misses the other spellings; the
concrete failure that motivated this was Whittle-Qwen-3.8-35B-A3B
(``qwen4_exp_text``) bypassing the compilation guard, which left the PLE n-gram
lookup's blocking device->host copy inside a full cudagraph capture and killed
engine init with ``hipErrorStreamCaptureUnsupported``.

``uses_ngram_embedding`` is the second half of that guard: the op being split out
only exists when the text config declares ``ple_layer_ids``.
"""

from types import SimpleNamespace

import pytest

from vllm.models.qwen4_exp.config import (
    QWEN4_EXP_CONFIG_CLASSES,
    QWEN4_EXP_MODEL_TYPES,
    is_qwen4_exp_config,
    is_qwen4_exp_model_type,
    needs_ple_ngram_split,
    uses_ngram_embedding,
)

# The family's spellings, spelled out: this test is the one place that has to be
# edited when a new family config class appears, which is the point (it fails if
# the derived tuple drifts instead of silently following).
EXPECTED_FAMILY_MODEL_TYPES = {"qwen4_exp", "qwen4_exp_text", "qwen4_exp_mtp"}

# Not-a-family controls: the neighbouring generation (qwen3_5), the parents of the
# family's config classes, longer strings that share the prefix, and non-strings.
NON_FAMILY_MODEL_TYPES = [
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_next",
    "qwen3_vl_vision",
    "qwen4_exp_extra",
    "",
    "qwen4",
]


def test_family_spellings_are_exactly_the_expected_set():
    assert set(QWEN4_EXP_MODEL_TYPES) == EXPECTED_FAMILY_MODEL_TYPES


def test_every_family_config_class_declares_its_own_model_type():
    """A class's spelling must be its own, not a parent's.

    ``QWEN4_EXP_MODEL_TYPES`` is derived from the family's config classes, so a
    class that forgets to declare ``model_type`` would inherit its parent's
    (``qwen3_next`` / ``qwen3_vl_vision``) and drag an unrelated architecture into
    the family -- which would make the compilation guard fire for it.
    """
    for config_cls in QWEN4_EXP_CONFIG_CLASSES:
        declared = config_cls.__dict__.get("model_type")
        assert isinstance(declared, str), (
            f"{config_cls.__name__} must declare its own model_type; inheriting "
            f"one would insert {config_cls.model_type!r} into the family"
        )
        assert is_qwen4_exp_model_type(declared)


@pytest.mark.parametrize("model_type", sorted(EXPECTED_FAMILY_MODEL_TYPES))
def test_family_spellings_are_accepted(model_type):
    assert is_qwen4_exp_model_type(model_type)


@pytest.mark.parametrize("model_type", NON_FAMILY_MODEL_TYPES)
def test_non_family_spellings_are_rejected(model_type):
    assert not is_qwen4_exp_model_type(model_type)


@pytest.mark.parametrize("model_type", NON_FAMILY_MODEL_TYPES)
def test_config_level_check_rejects_non_family(model_type):
    config = SimpleNamespace(model_type=model_type, architectures=["LlamaForCausalLM"])
    assert not is_qwen4_exp_config(config)


def test_config_level_check_accepts_the_family_via_model_type():
    config = SimpleNamespace(model_type="qwen4_exp_text", architectures=[])
    assert is_qwen4_exp_config(config)


def test_config_level_check_falls_back_to_architectures():
    """A config that omits/re-spells ``model_type`` is still the family."""
    config = SimpleNamespace(model_type=None, architectures=["Qwen4ExpForCausalLM"])
    assert is_qwen4_exp_config(config)


@pytest.mark.parametrize(
    "architectures",
    ["Qwen4ExpForCausalLM", ("Qwen4ExpForCausalLM",), {"Qwen4ExpForCausalLM"}],
    ids=["bare-string", "tuple", "set"],
)
def test_config_level_check_accepts_architecture_containers(architectures):
    config = SimpleNamespace(model_type="", architectures=architectures)
    assert is_qwen4_exp_config(config)


@pytest.mark.parametrize(
    "architectures",
    [None, 5, {"a": "Qwen4ExpForCausalLM"}, object(), [None, 7]],
    ids=["none", "int", "dict", "object", "non-str-elements"],
)
def test_config_level_check_survives_odd_architectures(architectures):
    """This runs for every model built, so a malformed value must not raise."""
    config = SimpleNamespace(model_type="", architectures=architectures)
    assert is_qwen4_exp_config(config) is False


def test_config_level_check_tolerates_a_missing_model_type():
    assert not is_qwen4_exp_config(SimpleNamespace())
    assert not is_qwen4_exp_config(None)


@pytest.mark.parametrize(
    "ple_layer_ids,expected",
    [([2], True), ([1, 2], True), ([], False), (None, False), (0, False)],
)
def test_uses_ngram_embedding_mirrors_the_model_state_gate(ple_layer_ids, expected):
    """``bool(ple_layer_ids)`` -- the same expression the model state uses."""
    config = SimpleNamespace(ple_layer_ids=ple_layer_ids)
    assert uses_ngram_embedding(config) is expected


def test_uses_ngram_embedding_defaults_to_false_without_the_attribute():
    assert uses_ngram_embedding(SimpleNamespace()) is False
    assert uses_ngram_embedding(None) is False


def _family_with_ple():
    return SimpleNamespace(model_type="qwen4_exp_text", ple_layer_ids=[2])


def test_split_is_needed_for_the_family_with_ple_on_rocm():
    assert needs_ple_ngram_split(_family_with_ple(), is_rocm=True) is True


def test_split_is_not_needed_on_cuda():
    """CUDA's lookup is a stream prefetch, so it must keep full-graph capture."""
    assert needs_ple_ngram_split(_family_with_ple(), is_rocm=False) is False


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(model_type="qwen4_exp_text", ple_layer_ids=[]),
        SimpleNamespace(model_type="qwen4_exp_text", ple_layer_ids=None),
        SimpleNamespace(model_type="qwen4_exp_text"),
        SimpleNamespace(model_type="llama", ple_layer_ids=[2]),
        SimpleNamespace(model_type="qwen3_next", ple_layer_ids=[2]),
    ],
    ids=["no-ple", "ple-none", "ple-missing", "non-family", "parent-spelling"],
)
def test_split_is_not_needed_without_both_conditions(config):
    assert needs_ple_ngram_split(config, is_rocm=True) is False
