# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Root-packed PLE embedding tables are placed on their owning layer."""

import pytest

from vllm.models.qwen4_exp.amd.model import _remap_packed_ple_table_name

PLE_LAYER = 1


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "ngram_embedding.shard_0.weight",
            "layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
        ),
        (
            "ngram_embedding.shard_511.weight",
            "layers.1.ple.ple_embedding.ngram_embedding.shard_511.weight",
        ),
        (
            "ple_embedding.layer_multipliers",
            "layers.1.ple.ple_embedding.layer_multipliers",
        ),
    ],
)
def test_root_packed_ple_table_moves_onto_the_ple_layer(
    name: str, expected: str
) -> None:
    """Whittle-Qwen-3.8-35B-A3B serializes the table at the checkpoint root."""

    assert _remap_packed_ple_table_name(name, PLE_LAYER) == expected


@pytest.mark.parametrize(
    "name",
    [
        # The parent family already nests the table under the PLE layer.
        "layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
        "layers.1.ple.ple_embedding.layer_multipliers",
        # Unrelated weights, and a name the anchored keys must not swallow.
        "layers.1.ple.key_proj.weight",
        "embed_tokens.weight",
        "ngram_embedding_extra.weight",
    ],
)
def test_remap_leaves_nested_and_unrelated_names_untouched(name: str) -> None:
    assert _remap_packed_ple_table_name(name, PLE_LAYER) == name


def test_remap_is_inert_without_a_single_ple_layer() -> None:
    """A root-packed table needs exactly one owning layer to be placed into."""

    assert (
        _remap_packed_ple_table_name("ngram_embedding.shard_0.weight", None)
        == "ngram_embedding.shard_0.weight"
    )
