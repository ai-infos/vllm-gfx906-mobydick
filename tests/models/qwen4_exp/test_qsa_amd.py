# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.models.qwen4_exp.amd import (
    model as _qwen4_exp_model,  # noqa: F401
)
from vllm.models.qwen4_exp.amd import ple_layer as ple_layer_module
from vllm.models.qwen4_exp.amd.indexer_qsa import (
    apply_qsa_rmsnorm,
    apply_qsa_rope,
)
from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AMD QSA requires ROCm",
)

requires_qsa_kernels = pytest.mark.skipif(
    not HAS_TRITON,
    reason="AMD QSA kernels require Triton",
)


def _stage_host_table_buffers(ngram_embedding: torch.nn.Module) -> None:
    """Provide the pinned staging buffers the host-table PLE path needs.

    ``MmapShardedNGramEmbedding`` (the host-RAM PLE implementation) drives the
    custom op through per-layer pinned ``_pinned_ngram_ids`` / ``_pinned_output``
    buffers -- the device-resident implementation does not. Adding them when the
    implementation exists lets this test exercise the op on either path; on the
    device path it is a no-op.
    """
    if not hasattr(ple_layer_module, "MmapShardedNGramEmbedding"):
        return
    ngram_embedding._pinned_ngram_ids = torch.empty(
        (4, 2), dtype=torch.long, device="cpu"
    ).pin_memory()
    ngram_embedding._pinned_output = torch.empty(
        (4, 6), dtype=torch.float32, device="cpu"
    ).pin_memory()


def test_ple_ngram_embedding_custom_op_matches_reference_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_name = "model.layers.0.ple"
    layer = ple_layer_module.Qwen4ExpPLELayer.__new__(ple_layer_module.Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = torch.nn.Module()
    layer.ple_embedding.ngram_embedding = torch.nn.Embedding(8, 3)
    _stage_host_table_buffers(layer.ple_embedding)
    context = SimpleNamespace(no_compile_layers={layer_name: layer})
    monkeypatch.setattr(ple_layer_module, "get_forward_context", lambda: context)

    ngram_ids = torch.tensor([[0, 1], [2, 3]])
    output = torch.empty(2, 6)
    ple_layer_module.qwen4_exp_amd_ple_ngram_embedding(
        ngram_ids,
        output,
        layer_name,
    )

    expected = layer.ple_embedding.ngram_embedding(ngram_ids).flatten(-2)
    torch.testing.assert_close(output, expected)


def _qsa_sparse_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    output = torch.zeros_like(q)
    repeats = q.shape[1] // k_cache.shape[2]
    page_size = k_cache.shape[1]
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[logical >= 0].long()
        if not logical.numel():
            continue
        request = token_to_req[row].long()
        pages = block_table[request, logical // page_size].long()
        offsets = logical % page_size
        keys = k_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        values = v_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float())
        probabilities = torch.softmax(scores * softmax_scale, dim=-1)
        output[row] = torch.einsum("hk,khd->hd", probabilities, values.float()).to(
            q.dtype
        )
    return output


def _qsa_mqa_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pages = page_table.index_select(0, token_to_req.long()).long()
    keys = k_cache[pages, :, 0, :].flatten(1, 2)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys.float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(q.shape[-1])
    visible = torch.minimum(
        (query_positions + 1) // compress_ratio,
        sequence_lengths.index_select(0, token_to_req.long()) // compress_ratio,
    )
    positions = torch.arange(keys.shape[1], device=q.device).unsqueeze(0)
    logits = logits.masked_fill(positions >= visible.unsqueeze(1), -torch.inf)
    return logits, visible.to(torch.int32)


def test_qsa_rope_uses_platform_dispatch() -> None:
    tensor = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    positions = torch.tensor([0, 1])
    calls = []

    def apply_rotary_emb(
        rotary_input: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        calls.append((rotary_input, cos, sin))
        return rotary_input + 1

    rotary_emb = SimpleNamespace(
        rotary_dim=2,
        apply_rotary_emb=apply_rotary_emb,
        _match_cos_sin_cache_dtype=lambda _: torch.zeros(2, 4),
    )

    output = apply_qsa_rope(rotary_emb, positions, tensor)

    assert len(calls) == 1
    torch.testing.assert_close(output[..., :2], tensor[..., :2] + 1)
    torch.testing.assert_close(output[..., 2:], tensor[..., 2:])


def test_qsa_rmsnorm_uses_portable_implementation(default_vllm_config) -> None:
    norm = GemmaRMSNorm(4, eps=1e-6)
    norm.weight.data.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    output = apply_qsa_rmsnorm(norm, tensor)

    torch.testing.assert_close(output, norm.forward_native(tensor))


def test_qsa_selection_uses_portable_topk_on_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 2
    token_topk = 8
    compress_ratio = 4
    block_topk = token_topk // compress_ratio
    blocks = torch.empty((rows, block_topk), dtype=torch.int32)
    visible_blocks = torch.tensor([4, 6], dtype=torch.int32)
    logits = torch.empty((rows, 8), dtype=torch.float32)
    selection_output = torch.empty(
        (rows, token_topk + compress_ratio - 1), dtype=torch.int32
    )
    topk_call: dict[str, Any] = {}

    monkeypatch.setattr(
        qsa_ops,
        "qsa_mqa_paged",
        lambda *args, **kwargs: (logits, visible_blocks),
    )
    monkeypatch.setattr(qsa_ops.current_platform, "is_cuda", lambda: False)

    def top_k_per_row_decode(
        topk_logits,
        next_n,
        seq_lens,
        raw_topk_indices,
        num_rows,
        stride0,
        stride1,
        topk_tokens,
    ) -> None:
        topk_call.update(
            logits=topk_logits,
            next_n=next_n,
            seq_lens=seq_lens,
            raw_topk_indices=raw_topk_indices,
            num_rows=num_rows,
            strides=(stride0, stride1),
            topk_tokens=topk_tokens,
        )
        raw_topk_indices.zero_()

    def expand_qsa_block_indices(*args) -> None:
        args[-1].fill_(-1)

    monkeypatch.setattr(qsa_ops.ops, "top_k_per_row_decode", top_k_per_row_decode)
    monkeypatch.setattr(
        qsa_ops,
        "expand_qsa_block_indices_cuda",
        expand_qsa_block_indices,
    )

    output = qsa_ops.qsa_select_paged_tokens(
        torch.empty((rows, 1, 1)),
        torch.empty((1, 4, 1, 1)),
        torch.empty((1, 2), dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.arange(rows, dtype=torch.int32),
        torch.tensor([8], dtype=torch.int32),
        token_topk,
        compress_ratio,
        selection_output,
    )

    assert output is selection_output
    assert torch.all(output == -1)
    assert topk_call["logits"] is logits
    assert topk_call["next_n"] == 1
    assert topk_call["seq_lens"] is visible_blocks
    assert topk_call["raw_topk_indices"].shape == blocks.shape
    assert topk_call["num_rows"] == rows
    assert topk_call["strides"] == (logits.stride(0), logits.stride(1))
    assert topk_call["topk_tokens"] == block_topk


@requires_qsa_kernels
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    ("num_rows", "num_query_heads", "num_kv_heads", "page_size"),
    [
        pytest.param(1, 24, 2, 1792, id="tp1_split64"),
        pytest.param(16, 12, 1, 1792, id="tp2_split32"),
        pytest.param(32, 6, 1, 1024, id="tp4_split8"),
        pytest.param(257, 6, 1, 1024, id="tp4_split4"),
        pytest.param(513, 6, 1, 1024, id="tp4_split1"),
    ],
)
def test_qsa_sparse_paged_attention_matches_reference(
    num_rows: int,
    num_query_heads: int,
    num_kv_heads: int,
    page_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(2)
    head_dim = 256
    num_requests = 2
    num_selected_pages = 64
    num_pages_per_request = num_selected_pages + 1
    num_cache_blocks = num_requests * num_pages_per_request
    indexer_budget = 2048
    indexer_compress_ratio = 4
    selection_width = indexer_budget + indexer_compress_ratio - 1
    q = torch.randn(num_rows, num_query_heads, head_dim, device="cuda", dtype=dtype)
    kv_cache = torch.randn(
        num_cache_blocks,
        page_size,
        num_kv_heads,
        2 * head_dim,
        device="cuda",
        dtype=dtype,
    )
    k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
    block_table = (
        torch.randperm(num_cache_blocks, device="cuda")
        .reshape(num_requests, num_pages_per_request)
        .to(torch.int32)
    )
    rows_per_request = math.ceil(num_rows / num_requests)
    row_indices = torch.arange(num_rows, device="cuda", dtype=torch.int32)
    token_to_req = row_indices // rows_per_request
    request_row_counts = torch.tensor(
        [rows_per_request, num_rows - rows_per_request],
        device="cuda",
        dtype=torch.int32,
    )

    context_length = num_pages_per_request * page_size - 1
    block_topk = indexer_budget // indexer_compress_ratio
    compressed_blocks_per_page = page_size // indexer_compress_ratio
    selection = torch.arange(block_topk, device="cuda")
    selected_pages = selection % num_selected_pages
    selected_offsets = selection // num_selected_pages
    row_shifts = 2 * row_indices.unsqueeze(1)
    selected_offsets = (selected_offsets + row_shifts) % compressed_blocks_per_page
    block_indices = (selected_pages * compressed_blocks_per_page + selected_offsets).to(
        torch.int32
    )
    rows_within_request = row_indices % rows_per_request
    query_positions = (
        context_length - request_row_counts[token_to_req.long()] + rows_within_request
    ).to(torch.int64)
    sequence_lengths = torch.full(
        (num_requests,), context_length, device="cuda", dtype=torch.int32
    )
    logical_indices = qsa_ops.expand_qsa_block_indices_cuda(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        indexer_compress_ratio,
        indexer_budget,
    )
    assert logical_indices.shape == (num_rows, selection_width)

    actual = qsa_ops.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
    )
    expected = _qsa_sparse_paged_attention_reference(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        q.shape[-1] ** -0.5,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _qsa_mqa_paged_case(
    dtype: torch.dtype, *, num_rows: int, uniform: bool
) -> tuple[Any, ...]:
    """One indexer-scoring case.

    ``uniform`` puts every row on request 0, which is the row-tiled route's
    precondition; otherwise the rows are split across two requests, so the
    per-row kernel must be used.
    """
    torch.manual_seed(3)
    head_dim = 128
    num_query_heads = 4
    page_size, num_pages, num_requests = 64, 20, 2
    q = torch.randn(num_rows, num_query_heads, head_dim, device="cuda", dtype=dtype)
    k_cache = torch.randn(num_pages, page_size, 1, head_dim, device="cuda", dtype=dtype)
    page_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32).reshape(
        num_requests, num_pages // num_requests
    )
    if uniform:
        token_to_req = torch.zeros(num_rows, device="cuda", dtype=torch.int32)
    else:
        token_to_req = torch.repeat_interleave(
            torch.arange(num_requests, device="cuda", dtype=torch.int32),
            num_rows // num_requests,
        )
    sequence_lengths = torch.tensor(
        [num_pages * page_size, num_pages * page_size - 37],
        device="cuda",
        dtype=torch.int32,
    )
    request_lengths = sequence_lengths.index_select(0, token_to_req.long())
    query_positions = (
        request_lengths - num_rows + torch.arange(num_rows, device="cuda")
    ).to(torch.int32)
    return q, k_cache, page_table, token_to_req, query_positions, sequence_lengths


@requires_qsa_kernels
@pytest.mark.parametrize("uniform", [False, True], ids=["per-row", "tiled"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_qsa_mqa_paged_matches_reference(dtype: torch.dtype, uniform: bool) -> None:
    """The indexer's scoring kernel reads the compressed-key cache in its own dtype.

    Both routes -- the per-row kernel and the row-tiled one that the uniform
    prefill gate selects -- must reproduce the reference.
    """
    compress_ratio = 4
    num_rows = 64
    q, k_cache, page_table, token_to_req, query_positions, sequence_lengths = (
        _qsa_mqa_paged_case(dtype, num_rows=num_rows, uniform=uniform)
    )

    actual, actual_visible = qsa_ops.qsa_mqa_paged(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
    )
    expected, expected_visible = _qsa_mqa_paged_reference(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_visible, expected_visible)


@requires_qsa_kernels
@pytest.mark.parametrize(
    ("dtype", "num_rows", "uniform", "expect_tiled"),
    [
        # fp16 is a native tl.dot everywhere; uniform prefill rows take the tiled route.
        (torch.float16, 64, True, True),
        # ... but only at prefill scale, and only when the rows share a request.
        (torch.float16, 32, True, False),
        (torch.float16, 64, False, False),
        # bf16 is a hardware dot only where the platform has native bf16 (gfx906
        # emulates it: the tiled route measured 0.42x there vs 1.39x in fp16).
        (torch.bfloat16, 64, True, False),
    ],
    ids=["fp16-uniform", "fp16-small", "fp16-mixed-req", "bf16-uniform"],
)
def test_qsa_mqa_paged_route_selection(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    num_rows: int,
    uniform: bool,
    expect_tiled: bool,
) -> None:
    """Pin the row-tiled gate: it must not be entered for bf16 on gfx906.

    The tiled kernel's win is a hardware ``tl.dot``; where that is emulated
    (bf16 on gfx906) it is 2.4x slower, so the gate -- not just correctness --
    is the regression guard.
    """
    compress_ratio = 4
    launched: list[str] = []

    def recorder(name: str, original: Any) -> Any:
        """Stand-in for a Triton kernel that records the launch and delegates."""

        class _Recorder:
            def __getitem__(self, grid: Any) -> Any:
                def launch(*args: Any, **kwargs: Any) -> Any:
                    launched.append(name)
                    return original[grid](*args, **kwargs)

                return launch

        return _Recorder()

    for name in ("_qsa_mqa_paged_tiled_kernel", "_qsa_mqa_paged_kernel"):
        monkeypatch.setattr(qsa_ops, name, recorder(name, getattr(qsa_ops, name)))

    q, k_cache, page_table, token_to_req, query_positions, sequence_lengths = (
        _qsa_mqa_paged_case(dtype, num_rows=num_rows, uniform=uniform)
    )
    qsa_ops.qsa_mqa_paged(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
    )

    if not expect_tiled and dtype == torch.bfloat16 and uniform and num_rows >= 64:
        # bf16 is expected to take the tiled route only where the dot is native.
        assert current_platform.supports_native_bf16 is False

    assert launched == (
        ["_qsa_mqa_paged_tiled_kernel"] if expect_tiled else ["_qsa_mqa_paged_kernel"]
    )


def test_reference_block_ranks_follows_the_kernel_contract() -> None:
    """The reference selection mirrors ``top_k_per_row_decode``'s output contract.

    Per row: the top ``block_topk`` visible blocks by score, -1 past the row's
    visible end. Row ends are block counts, as ``qsa_mqa_paged`` returns them.
    """

    logits = torch.tensor(
        [[0.5, 3.0, 1.0, 2.0], [4.0, 0.25, 0.0, -1.0]], dtype=torch.float32
    )
    row_ends = torch.tensor([3, 1], dtype=torch.int32)
    blocks = torch.full((2, 2), 123, dtype=torch.int32)

    qsa_ops._reference_block_ranks(logits, row_ends, blocks, 2)

    # Row 0 sees blocks 0..2 -> 1 (3.0) and 2 (1.0); row 1 sees block 0 only.
    assert blocks.tolist() == [[1, 2], [0, -1]]


def test_reference_block_ranks_clamps_a_width_wider_than_the_cache() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    row_ends = torch.tensor([3], dtype=torch.int32)
    blocks = torch.full((1, 8), 123, dtype=torch.int32)

    qsa_ops._reference_block_ranks(logits, row_ends, blocks, 8)

    assert blocks.tolist() == [[2, 1, 0, -1, -1, -1, -1, -1]]


def test_reference_block_ranks_handles_an_empty_cache() -> None:
    blocks = torch.full((2, 4), 123, dtype=torch.int32)

    qsa_ops._reference_block_ranks(
        torch.empty((2, 0), dtype=torch.float32),
        torch.zeros((2,), dtype=torch.int32),
        blocks,
        4,
    )

    assert blocks.tolist() == [[-1, -1, -1, -1], [-1, -1, -1, -1]]


def test_reference_block_ranks_pads_by_visibility_not_position() -> None:
    """Blocks past a row's visible end must never be selected.

    Regression from the prefill-scale probe: padding by output position leaked real
    block indices for blocks the row cannot see (the kernel never scans them), which
    only shows up once the width exceeds the row's visible block count.
    """

    logits = torch.tensor(
        [[5.0, 4.0, 3.0, 2.0], [0.1, 0.4, 0.3, 0.2]], dtype=torch.float32
    )
    row_ends = torch.tensor([1, 4], dtype=torch.int32)
    blocks = torch.full((2, 4), 123, dtype=torch.int32)

    qsa_ops._reference_block_ranks(logits, row_ends, blocks, 4)

    # Row 0 sees only block 0, so the other three slots are invalid, not blocks 1..3.
    assert blocks[0].tolist() == [0, -1, -1, -1]
    # Row 1 sees everything, in descending score order.
    assert blocks[1].tolist() == [1, 2, 3, 0]


def test_select_qsa_block_ranks_dispatches_by_measured_ceiling(monkeypatch) -> None:
    """Kernel up to the verified width ceiling, reference above it.

    The ceiling is measured, not assumed: on gfx906 the decode kernel agrees with
    the reference selection at 8192 and corrupts memory at 12288/16384
    (`/local/tmp/wht1/topk_case.py`). A model whose indexer budget covers its
    context selects 65536 blocks, which is past the ceiling by construction.
    """

    calls: list[int] = []

    def _fake_kernel(logits, next_n, row_ends, blocks, num_rows, s0, s1, topk):
        calls.append(topk)
        blocks.fill_(-1)

    monkeypatch.setattr(qsa_ops.ops, "top_k_per_row_decode", _fake_kernel)

    logits = torch.zeros((1, 4), dtype=torch.float32)
    row_ends = torch.tensor([4], dtype=torch.int32)
    blocks = torch.empty((1, 4), dtype=torch.int32)

    for width in (2, 512, 2048, qsa_ops._TOPK_KERNEL_MAX_WIDTH):
        qsa_ops._select_qsa_block_ranks(logits, row_ends, blocks, width)

    assert calls == [2, 512, 2048, 8192]

    # Just past the ceiling -- and the width of the model this was written for.
    for width in (8193, 65536):
        qsa_ops._select_qsa_block_ranks(logits, row_ends, blocks, width)

    # Above the ceiling the reference path answers, so the kernel is not called.
    assert calls == [2, 512, 2048, 8192]


@pytest.mark.parametrize(
    ("budget", "ratio", "max_model_len", "expected"),
    [
        # Whittle-Qwen-3.8-35B-A3B: a full-context budget on a 32k server.
        (262144, 4, 32768, 8192),
        # The same checkpoint served at its full 262144 context.
        (262144, 4, 262144, 65536),
        # A budget that already fits is untouched.
        (2048, 4, 262144, 2048),
        (8192, 4, 32768, 8192),
    ],
)
def test_addressable_token_topk_caps_a_full_context_budget(
    budget: int, ratio: int, max_model_len: int, expected: int
) -> None:
    """An indexer budget wider than the context can address buys nothing.

    Unclamped, the selection buffer is ``max_num_batched_tokens x
    (budget + ratio - 1)`` int32 *per layer*: 4 GiB per layer for Whittle at
    MBT 4096, which OOMed the loader at layer 7 of 40.
    """

    from vllm.models.qwen4_exp.amd.indexer_qsa import addressable_token_topk

    assert addressable_token_topk(budget, ratio, max_model_len) == expected


def test_addressable_token_topk_never_truncates_an_addressable_row() -> None:
    """The cap stays above every row's reachable compressed tokens.

    A row addresses at most ``ceil(max_model_len / ratio)`` compressed tokens, so
    the cap must not fall below that -- and must stay divisible by the compression
    ratio, which the index-expansion kernel requires.
    """

    from vllm.models.qwen4_exp.amd.indexer_qsa import addressable_token_topk

    for max_model_len in (1, 63, 64, 65, 4096, 32768, 32770, 131072, 262144):
        for ratio in (2, 4, 8):
            width = addressable_token_topk(10**9, ratio, max_model_len)
            assert width >= math.ceil(max_model_len / ratio)
            assert width % ratio == 0
