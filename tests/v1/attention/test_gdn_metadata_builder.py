# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically how
non-spec decodes are classified when spec decodes exist.

Historically non-spec decodes were reclassified as 1-token "prefills"
(the #34845 fix); since the W1 dispatch fix the metadata builder keeps
`num_decodes > 0` in spec-mixed batches and peels the decode rows to
the per-seq recurrent kernel instead (prefill-only chunk metadata is
built for the remaining real prefills). The batch is ordered
decode → ... → prefill (V1 invariant), so the non-spec group is
decode-first.
"""

from dataclasses import dataclass

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode.
    # The decode is kept as a decode (peeled to the per-seq kernel).
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=1,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — nothing to peel
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to peel
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[20, 100],
        query_lens=[3, 50],
        num_decode_draft_tokens=[2, -1],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode is peeled, prefill metadata
    # covers the 50-token prefill only
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 100, 20],
        query_lens=[1, 50, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=1,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests, all peeled (no prefill)
    "multiple_decodes_with_spec": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence (at the back) excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[65, 20, 16],
        query_lens=[1, 3, 0],
        num_decode_draft_tokens=[-1, 2, -1],
        num_speculative_tokens=2,
        expected_num_decodes=1,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=1,
    ),
    # FULL-cudagraph request padding, no spec decode (issue #35). Cudagraph
    # replay pads the request dimension up to a captured size with
    # zero-length dummy requests; they are deliberately counted as decodes so
    # that num_decodes matches the captured size that indexes the GDN state /
    # cudagraph tensors (split_decodes_and_prefills: "some requests may have a
    # query length of 0 but since they are padding its fine to treat them as
    # decodes (ensures num_decodes matches the captured size)"). Used to abort
    # the engine in the builder's decode-first ramp check.
    "non_spec_full_cg_pad_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20, 20],
        query_lens=[1, 1, 1, 0],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=4,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Padding in the middle of the decode region, followed by a real prefill:
    # the peel must start after the last decode row (including the padded one).
    "non_spec_full_cg_pad_then_prefill": GDNBuildTestCase(
        seq_lens=[40, 30, 20, 2048],
        query_lens=[1, 1, 0, 2048],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=1,
        expected_num_prefill_tokens=2048,
        expected_num_spec_decodes=0,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_prefill_metadata_peels_decodes_in_spec_mixed_batch():
    """Spec + decode + prefill: prefill chunk metadata must cover the
    real prefill only (decode rows peeled off the front)."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    # Batch order (V1 invariant): decode, prefill, spec.
    # req0: 1-token non-spec decode, context 64 -> has initial state
    # req1: 50-token prefill, context 0 (fresh) -> no initial state
    # req2: 3-token spec decode
    batch = BatchSpec(seq_lens=[65, 50, 20], query_lens=[1, 50, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, -1, 2])

    assert meta.num_decodes == 1
    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == 50

    assert meta.prefill_query_start_loc is not None
    assert meta.prefill_query_start_loc.tolist() == [0, 50]
    assert meta.prefill_has_initial_state is not None
    assert meta.prefill_has_initial_state.tolist() == [False]
    # Full non-spec has_initial_state (decode + prefill, decode-first):
    # req0 context 64 > 0 -> True; req1 context 0 -> False.
    assert meta.has_initial_state is not None
    assert meta.has_initial_state.tolist() == [True, False]
    # Prefill state indices are the non-spec ones with the decode row
    # peeled off.
    assert meta.prefill_state_indices is not None
    assert meta.prefill_state_indices.shape[0] == 1


def test_no_prefill_metadata_when_only_decodes_peeled():
    """Spec + non-spec decodes only: no prefill metadata at all, and
    the full non-spec cu_seqlens lists the decodes (decode-first)."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[40, 50, 60, 20], query_lens=[1, 1, 1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, -1, -1, 2])

    assert meta.num_decodes == 3
    assert meta.num_prefills == 0
    assert meta.prefill_query_start_loc is None
    assert meta.prefill_state_indices is None
    assert meta.has_initial_state is None
    assert meta.non_spec_query_start_loc is not None
    assert meta.non_spec_query_start_loc.tolist() == [0, 1, 2, 3]


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)
