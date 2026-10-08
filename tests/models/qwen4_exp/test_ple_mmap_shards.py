# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the mmap-backed PLE ngram embedding.

The change under test replaces a materialized `nn.Embedding` table with
`MmapShardedNGramEmbedding`: CPU tensors that are *the mmap'd safetensors
shards themselves*, held by reference (`load_weights` does
`embedding.set_shard(i, loaded_weight.to("cpu"))` with no `.copy_()` and no
device move), so a ~100 GB table is paid once per node through the page cache
instead of once per worker process.

What is asserted, and what deliberately is not:

* row-for-row equality against a plain reference gather, across shard
  boundaries, for 1-D and N-D id tensors, with the shard-index math at the
  exact capacity boundaries;
* **no copy**: the stored shard is the same object with the same `data_ptr()`
  as the tensor handed in -- that is the entire memory claim;
* it works on a genuinely `mmap`-backed tensor, not just a heap one;
* the guards that protect a silent wrong-lookup: non-CPU shard, non-CPU ids,
  dtype drift between shards, a shard index beyond `split_ngram_parts`, an
  `embedding_dim` mismatch, a shard that was never loaded, and an id outside
  the table's row range (negative, or past the last row) -- which before the
  range check matched no shard mask and left its row holding uninitialized
  memory from `new_empty()`; and the one out-of-range id that is *legitimate*,
  the MTP drafter's `-1` padding sentinel, which must be folded onto a valid
  row rather than rejected,
* `load_weights` bookkeeping: which names it reports as loaded, and that
  `hashstats_*` / `token_lookup` leaves are skipped.

The fixed-length `for shard in range(self.num_shards)` loop is the CUDA-graph
safety property (a data-dependent loop length would run a different number of
Python steps in capture than in replay). That property is structural -- see the
comment at the loop -- so it is asserted here as the observable consequence:
batches that touch different shards give correct results and leave no state
behind between calls.
"""

import mmap

import numpy as np
import pytest
import torch

ple = pytest.importorskip(
    "vllm.models.qwen4_exp.amd.ple_layer",
    reason="qwen4_exp AMD model code is not importable in this build",
)

NUM_SHARDS = 3
CAPACITY = 8  # rows per shard
DIM = 5


def _shard(base: int, dtype=torch.float16, n: int = CAPACITY) -> torch.Tensor:
    """Rows of a recognizable pattern: row r is filled with base + r."""
    return (
        torch.arange(n, dtype=torch.float32)
        .unsqueeze(-1)
        .expand(n, DIM)
        .mul(0.5)
        .add(base)
        .to(dtype)
    )


@pytest.fixture
def emb():
    e = ple.MmapShardedNGramEmbedding(NUM_SHARDS, CAPACITY, DIM)
    for i in range(NUM_SHARDS):
        e.set_shard(i, _shard(i * 100))
    return e


def _reference(shards, ids, dim):
    """What a single dense embedding table would have returned."""
    table = torch.cat(shards, dim=0)
    flat = ids.reshape(-1).long()
    return table.index_select(0, flat).reshape(*ids.shape, dim)


# --------------------------------------------------------------------------
# correctness of the lookup
# --------------------------------------------------------------------------


def test_shards_start_empty():
    e = ple.MmapShardedNGramEmbedding(NUM_SHARDS, CAPACITY, DIM)
    assert e._shards == [None] * NUM_SHARDS
    assert e.params_dtype is None
    assert e.num_shards == NUM_SHARDS and e.shard_row_capacity == CAPACITY
    assert e.embedding_dim == DIM


def test_forward_matches_reference_gather(emb):
    ids = torch.tensor([0, 1, 7, 8, 15, 16, 23])
    out = emb(ids)
    assert torch.equal(out, _reference(emb._shards, ids, DIM))
    assert out.shape == (ids.numel(), DIM)


def test_forward_shard_boundaries(emb):
    """Row 7 is the last row of shard 0, row 8 the first of shard 1."""
    ids = torch.tensor([7, 8, 15, 16])
    out = emb(ids)
    assert out[0, 0].item() == pytest.approx(3.5, abs=1e-3)  # shard 0 row 7
    assert out[1, 0].item() == pytest.approx(100.0, abs=1e-3)  # shard 1 row 0
    assert out[2, 0].item() == pytest.approx(103.5, abs=1e-3)  # shard 1 row 7
    assert out[3, 0].item() == pytest.approx(200.0, abs=1e-3)  # shard 2 row 0


def test_forward_preserves_batch_shape(emb):
    ids = torch.tensor([[1, 9], [17, 22]])
    out = emb(ids)
    assert out.shape == (2, 2, DIM)
    assert torch.equal(out, _reference(emb._shards, ids, DIM))


def test_forward_keeping_dtype_and_device(emb):
    ids = torch.tensor([2, 10, 20])
    out = emb(ids)
    assert out.dtype == emb.params_dtype is torch.float16
    assert out.device.type == "cpu"


def test_repeated_calls_over_different_shards_are_stateless(emb):
    """The graph-safety consequence: whichever shards a batch touches must not
    change what the next batch sees."""
    a = emb(torch.tensor([0, 1]))
    b = emb(torch.tensor([0, 1]))
    emb(torch.arange(NUM_SHARDS * CAPACITY))  # touches every shard in between
    c = emb(torch.tensor([0, 1]))
    assert torch.equal(a, b) and torch.equal(a, c)
    assert [t.shape for t in emb._shards] == [(CAPACITY, DIM)] * NUM_SHARDS


def test_scalar_and_empty_id_tensors(emb):
    assert emb(torch.tensor(4)).shape == (DIM,)
    assert emb(torch.tensor([], dtype=torch.long)).shape == (0, DIM)


def test_works_on_a_real_mmap_backed_tensor(tmp_path):
    """The whole point of the class: the table is the page cache, not a copy."""
    path = tmp_path / "shard0.safetensors-like"
    rows, dim = 32, 4
    arr = np.ascontiguousarray(
        np.tile(np.arange(rows, dtype=np.float32)[:, None], (1, dim))
    )
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    with path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), arr.nbytes, access=mmap.ACCESS_READ)
    e = None
    tensor = None
    try:
        tensor = torch.from_numpy(
            np.frombuffer(mm, dtype=np.float32).reshape(rows, dim)
        )

        e = ple.MmapShardedNGramEmbedding(1, rows, dim)
        e.set_shard(0, tensor)
        assert e._shards[0] is tensor, "forward must read the mmap, not a copy"
        ids = torch.tensor([3, 17, 31])
        assert torch.equal(e(ids), tensor.index_select(0, ids))
    finally:
        # The mmap cannot be closed while torch still holds an export of it.
        if e is not None:
            e._shards = [None]
        tensor = None
        mm.close()


# --------------------------------------------------------------------------
# the guards: each of these would otherwise be a silent wrong lookup
# --------------------------------------------------------------------------


def test_set_shard_rejects_non_cpu_tensor(emb):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA/ROCm device to test the device guard")
    with pytest.raises(ValueError, match="must be loaded on CPU"):
        emb.set_shard(0, _shard(0).cuda())


def test_forward_rejects_non_cpu_ids(emb):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA/ROCm device to test the device guard")
    with pytest.raises(ValueError, match="requires CPU ids"):
        emb(_shard(0).cuda().long())


def test_set_shard_rejects_dtype_drift(emb):
    with pytest.raises(ValueError, match="does not match previously loaded"):
        emb.set_shard(1, _shard(50, dtype=torch.bfloat16))


def test_first_shard_fixes_the_dtype(emb):
    assert emb.params_dtype is torch.float16


def test_missing_shard_raises_rather_than_returning_garbage():
    e = ple.MmapShardedNGramEmbedding(NUM_SHARDS, CAPACITY, DIM)
    e.set_shard(0, _shard(0))
    e.set_shard(2, _shard(200))
    # ids in shard 1's range, and shard 1 was never loaded
    with pytest.raises(RuntimeError, match="shard 1 was never loaded"):
        e(torch.tensor([9]))
    # ids that avoid the hole still work
    assert e(torch.tensor([1, 17])).shape == (2, DIM)


def test_out_of_range_ids_are_folded_onto_row_zero(emb):
    """An id past the last row matches no shard mask, so before the range test
    its row kept whatever `new_empty()` found there: a wrong embedding with no
    error. It is now folded onto row 0 instead, so every row returned is a row
    that was actually read.

    Folding rather than raising is the second half of the lesson. The first
    version of this check raised, and `0xff80ff80ff80ff80` arriving during
    CUDA-graph capture on 2026-09-23 killed three boots -- see the same date's
    entry in docs/gfx906/degradation.md."""
    table_rows = NUM_SHARDS * CAPACITY
    for bad in (table_rows, -2, 2 * table_rows):
        out = emb(torch.tensor([bad]))
        assert torch.equal(out, _reference(emb._shards, torch.tensor([0]), DIM))
    # mixed with real ids: the real ones are untouched by the fold
    ids = torch.tensor([5, table_rows, -2, 17])
    out = emb(ids)
    assert torch.equal(out, _reference(emb._shards, torch.tensor([5, 0, 0, 17]), DIM))
    # the last valid row still works, and so does the whole valid range
    assert emb(torch.tensor([table_rows - 1])).shape == (1, DIM)


def test_capture_time_poison_pattern_does_not_kill_the_boot(emb):
    """The verbatim value from the three dead boots.

    `0xff80ff80ff80ff80` is what an as-yet-unwritten pinned host buffer reads
    back as; it reached this lookup only on the compile/graph path, which is why
    `--enforce-eager` hid it. A guard here must be unable to refuse a boot."""
    poison = -35747867511423104
    assert hex(poison & (2**64 - 1)) == "0xff80ff80ff80ff80"
    ids = torch.full((4, 3), poison, dtype=torch.int64)
    out = emb(ids)
    assert out.shape == (4, 3, DIM)
    assert torch.equal(
        out, _reference(emb._shards, torch.zeros(4, 3, dtype=torch.long), DIM)
    )


def test_folding_warns_once_per_module(emb, caplog):
    """Silent folding is how this class of bug becomes invisible: the reason the
    blanket `clamp()` was worth replacing is that it made corruption look like
    traffic. So the fold is loud -- but exactly once, because a per-step warning
    on a boot-time pattern would bury everything else in the log."""
    with caplog.at_level("WARNING"):
        emb(torch.tensor([-1, NUM_SHARDS * CAPACITY]))
        first = [r for r in caplog.records if "outside [0," in r.getMessage()]
        n_after_first = len(first)
        emb(torch.tensor([-1, NUM_SHARDS * CAPACITY]))
    assert n_after_first == 1
    assert len([r for r in caplog.records if "outside [0," in r.getMessage()]) == 1
    assert "VLLM_GFX906_PLE_STRICT=1" in first[0].getMessage()
    # in-range calls say nothing
    with caplog.at_level("WARNING"):
        emb(torch.tensor([1, 2, 3]))
    assert len([r for r in caplog.records if "outside [0," in r.getMessage()]) == 1


def test_strict_mode_raises_for_development(emb, monkeypatch):
    """The diagnostic is opt-in instead of default-on: a raise on this path is a
    boot refusal, and the inputs that trigger it legitimately occur at boot."""
    monkeypatch.setenv("VLLM_GFX906_PLE_STRICT", "1")
    with pytest.raises(ValueError, match=r"out of range.*VLLM_GFX906_PLE_STRICT"):
        emb(torch.tensor([NUM_SHARDS * CAPACITY]))
    # in-range ids are unaffected by the strict switch
    assert emb(torch.tensor([1, 2])).shape == (2, DIM)


def test_drafter_padding_sentinel_is_folded_out_rather_than_rejected(emb):
    """`-1` is not corruption: the MTP drafter pre-fills `sample_idx_mapping`
    with it (`spec_decode/dflash/speculator.py:145`) and it arrives on every
    drafter warmup and capture, so rejecting it takes the whole server down at
    boot. That is not hypothetical -- the first version of the range check above
    did exactly that and killed two boots on 2026-09-23 with
    `PLE ngram id out of range: ids span [-1, -1] but the table holds
    320001536 rows (128 shards x 2500012)`; see the same date's entry in
    docs/gfx906/degradation.md.

    The other direction is the more important assertion: folding the sentinel
    out must happen *before* the range check, not instead of it. A blanket
    `ids.clamp()` satisfies this test's happy path and silently disarms the
    check, which is how real corruption goes back to being a wrong row."""
    sentinel = ple.MmapShardedNGramEmbedding.PADDING_SENTINEL
    ids = torch.tensor([sentinel, 3, sentinel])
    out = emb(ids)
    assert out.shape == (3, DIM)
    # sentinel rows come from row 0, real rows stay exactly right
    assert torch.equal(out, _reference(emb._shards, torch.tensor([0, 3, 0]), DIM))
    # and out-of-range ids in the same batch are handled by the same fold, not
    # by a path that can refuse a boot
    mixed = torch.tensor([sentinel, NUM_SHARDS * CAPACITY, 3])
    assert torch.equal(
        emb(mixed), _reference(emb._shards, torch.tensor([0, 0, 3]), DIM)
    )


# --------------------------------------------------------------------------
# load_weights: the no-copy bookkeeping
# --------------------------------------------------------------------------


class _Stub:
    """Duck-typed `Qwen4ExpNGramEmbedding`.

    Constructing the real module means running the prime search over
    `ngram_vocab_size_base`, which is not what is under test here; `load_weights`
    only ever touches the attributes below, so it is called unbound against this.
    """

    def __init__(self, split_ngram_parts=NUM_SHARDS, dim=DIM):
        self.split_ngram_parts = split_ngram_parts
        self.layer_multipliers = torch.zeros(4, dtype=torch.int64)
        self.ngram_heads_offsets = torch.zeros(2, dtype=torch.long)
        self.ngram_heads_vocab_sizes = torch.zeros(2, dtype=torch.long)
        self.ngram_embedding = ple.MmapShardedNGramEmbedding(
            split_ngram_parts, CAPACITY, dim
        )


def _load(stub, weights):
    return ple.Qwen4ExpNGramEmbedding.load_weights(stub, weights)


def test_load_weights_reports_shard_names_and_stores_by_reference():
    stub = _Stub()
    w0, w1 = _shard(0), _shard(100)
    loaded = _load(
        stub,
        [
            ("ngram_embedding.shard_0.weight", w0),
            ("ngram_embedding.shard_1.weight", w1),
        ],
    )
    assert loaded == {"ngram_embedding.shard_0", "ngram_embedding.shard_1"}
    assert stub.ngram_embedding._shards[0] is w0, "no .copy_(): by reference"
    assert stub.ngram_embedding._shards[1] is w1
    assert stub.ngram_embedding._shards[0].data_ptr() == w0.data_ptr()
    assert stub.ngram_embedding._shards[0].device.type == "cpu"


def test_load_weights_skips_hash_and_lookup_leaves():
    stub = _Stub()
    loaded = _load(
        stub,
        [
            ("ngram_embedding.hashstats_0", _shard(0)),
            ("token_lookup", _shard(0)),
            ("ngram_embedding.shard_0.weight", _shard(0)),
        ],
    )
    assert loaded == {"ngram_embedding.shard_0"}
    assert stub.ngram_embedding._shards[1] is None


def test_load_weights_fills_persistent_buffers():
    stub = _Stub()
    vals = torch.tensor([11, 22], dtype=torch.long)
    loaded = _load(stub, [("ngram_heads_offsets", vals)])
    assert "ngram_heads_offsets" in loaded
    assert torch.equal(stub.ngram_heads_offsets, vals)


def test_load_weights_rejects_buffer_shape_mismatch():
    stub = _Stub()
    with pytest.raises(ValueError, match="Shape mismatch"):
        _load(stub, [("ngram_heads_offsets", torch.zeros(7, dtype=torch.long))])


def test_load_weights_rejects_shard_index_beyond_split():
    stub = _Stub()
    with pytest.raises(ValueError, match="exceeds split_ngram_parts"):
        _load(stub, [(f"ngram_embedding.shard_{NUM_SHARDS}.weight", _shard(0))])


def test_load_weights_rejects_embedding_dim_mismatch():
    stub = _Stub()
    wrong = torch.zeros(CAPACITY, DIM + 1, dtype=torch.float16)
    with pytest.raises(ValueError, match="Shape mismatch for PLE embedding shard"):
        _load(stub, [("ngram_embedding.shard_0.weight", wrong)])


# --------------------------------------------------------------------------
# the id -> row map comes from the checkpoint's split, not from the runtime's
# stride (issue #40: a mismatch killed the engine and mis-read most ids)
# --------------------------------------------------------------------------


def _split_shard(base: int, n: int) -> torch.Tensor:
    return _shard(base, n=n)


@pytest.fixture
def checkpoints_split():
    """A checkpoint whose shards are shorter than the configured stride.

    `CAPACITY` is the stride the runtime computes from its own vocab layout before
    any weight is loaded, and the files here hold fewer rows than that per shard --
    the shape of the real case, scaled down (see
    `test_real_checkpoint_split_numbers` for the real one).
    """
    capacity, rows, num_shards = 5, [4, 4, 4, 4, 4], 5
    e = ple.MmapShardedNGramEmbedding(num_shards, capacity, DIM)
    for index, n in enumerate(rows):
        e.set_shard(index, _split_shard(index * 100, n))
    return e, capacity, rows, num_shards


def test_uniform_stride_mis_locates_ids_when_the_split_differs(checkpoints_split):
    """The bug, reproduced: with the stride alone, in-range ids either raise or
    land on another shard's rows."""
    e, _, _, _ = checkpoints_split
    assert e.table_rows == 25  # the configured uniform split, pre-finalize
    with pytest.raises(IndexError, match="index out of range"):
        e(torch.tensor([9]))  # stride 5 -> shard 1, local 4; shard 1 holds 4 rows


def test_finalized_layout_maps_every_id_to_its_own_row(checkpoints_split):
    """After finalize the lookup equals one dense table built from the shards --
    which is what the checkpoint's split means."""
    e, _, rows, num_shards = checkpoints_split
    assert e.finalize_shard_layout(sum(rows)) is True
    assert e._shard_starts == [0, 4, 8, 12, 16]
    assert e._shard_rows == rows
    assert e.table_rows == sum(rows)

    ids = torch.arange(sum(rows))
    assert torch.equal(e(ids), _reference(e._shards, ids, DIM))
    # the ids the stride alone crashed on, one per shard boundary
    assert torch.equal(
        e(torch.tensor([4, 9, 14, 19])),
        _reference(e._shards, torch.tensor([4, 9, 14, 19]), DIM),
    )
    # and the fold still covers ids outside the table
    assert torch.equal(
        e(torch.tensor([sum(rows), -1])),
        _reference(e._shards, torch.tensor([0, 0]), DIM),
    )


def test_layout_rejects_a_split_that_does_not_add_up_to_the_id_space():
    """Serving this would mean serving embeddings from rows the checkpoint never
    meant for those ids, so it is a load-time error rather than a warning."""
    with pytest.raises(
        ValueError, match=r"hold 19 rows but the model's ngram id space is 20"
    ):
        ple.build_ple_shard_layout(
            [4, 4, 4, 3, 4],
            num_shards=5,
            shard_row_capacity=5,
            vocab_size=20,
        )


def test_layout_rejects_a_non_positive_shard():
    with pytest.raises(ValueError, match="holds 0 rows"):
        ple.build_ple_shard_layout(
            [4, 0, 4], num_shards=3, shard_row_capacity=4, vocab_size=8
        )


def test_layout_accepts_the_canonical_split_without_warning(caplog):
    """vLLM's own splitter: one uniform stride, short last shard. No grumbling."""
    with caplog.at_level("WARNING"):
        starts, rows = ple.build_ple_shard_layout(
            [8, 8, 4], num_shards=3, shard_row_capacity=8, vocab_size=20
        )
    assert (starts, rows) == ([0, 8, 16], [8, 8, 4])
    assert not [r for r in caplog.records if "canonical split" in r.getMessage()]


def test_layout_warns_on_a_non_canonical_split(caplog):
    """The split is served, but whoever produced it should see that it is not the
    layout vLLM's own splitter would have written."""
    with caplog.at_level("WARNING"):
        ple.build_ple_shard_layout(
            [4, 4, 4, 4, 4], num_shards=5, shard_row_capacity=5, vocab_size=20
        )
    warned = [r for r in caplog.records if "canonical split" in r.getMessage()]
    assert len(warned) == 1
    assert "re-split" in warned[0].getMessage()


def test_layout_warns_when_the_id_space_is_unknown(caplog):
    """A load with no layout buffers (a stub, or a partial checkpoint) still gets
    the row boundaries from the shards, but says that it could not check them."""
    with caplog.at_level("WARNING"):
        starts, rows = ple.build_ple_shard_layout(
            [4, 4], num_shards=2, shard_row_capacity=4, vocab_size=None
        )
    assert (starts, rows) == ([0, 4], [4, 4])
    assert [r for r in caplog.records if "id space is unknown" in r.getMessage()]


def test_real_checkpoint_split_numbers():
    """`Whittle-Qwen-3.8-35B-A3B` verbatim: 7,812,500 x 4 + 7,790,000 rows against
    a stride of 7,808,128 derived from the runtime's own prime-based layout, with
    the checkpoint's buffers (8 x 4,880,000) defining an id space of 39,040,000.

    This is the arithmetic behind issue #40, asserted without allocating the table:
    the id that used to crash, and the ids the stride used to mis-locate.
    """
    capacity = 7_808_128
    rows = [7_812_500, 7_812_500, 7_812_500, 7_812_500, 7_790_000]
    vocab_size = 39_040_000
    starts, out_rows = ple.build_ple_shard_layout(
        rows, num_shards=5, shard_row_capacity=capacity, vocab_size=vocab_size
    )
    assert starts == [0, 7_812_500, 15_625_000, 23_437_500, 31_250_000]
    assert out_rows == rows and sum(rows) == vocab_size

    # the first id the uniform stride could not address: shard 4, local 7,790,000,
    # one past the last row of the file
    crash_lo = 4 * capacity + rows[4]
    assert crash_lo == 39_022_512
    assert (vocab_size - crash_lo) == 17_488

    # after the fix it is simply row 7,772,512 of shard 4
    assert crash_lo - starts[4] == 7_772_512
    assert crash_lo - starts[4] < rows[4]

    # an id the stride mis-located: stride says shard 1 row 1,872, the checkpoint
    # stores it as row 7,810,000 of shard 0
    misread = 7_810_000
    assert divmod(misread, capacity) == (1, 1_872)
    assert misread - starts[0] == 7_810_000 and misread < starts[1]

    # only the last shard is short, and only the last 17,488 ids were unreachable
    assert out_rows[:4] == [capacity + 4_372] * 4


def test_load_weights_finalizes_the_layout_from_the_shards():
    """`load_weights` is where a chunked delivery ends, so that is where the
    boundaries are derived -- the stub's layout buffers stay zeroed, which is the
    'id space unknown' path: the rows still define the mapping."""
    stub = _Stub()
    for index in range(NUM_SHARDS):
        _load(stub, [(f"ngram_embedding.shard_{index}.weight", _shard(index * 100))])
    emb = stub.ngram_embedding
    assert emb._shard_starts == [0, CAPACITY, 2 * CAPACITY]
    assert emb._shard_rows == [CAPACITY] * NUM_SHARDS
    ids = torch.tensor([0, CAPACITY - 1, CAPACITY, 2 * CAPACITY - 1])
    assert torch.equal(emb(ids), _reference(emb._shards, ids, DIM))


def test_load_weights_refuses_a_split_that_does_not_match_the_id_space():
    """With the layout buffers present, a split that does not add up is refused at
    load time instead of being served."""
    stub = _Stub()
    stub.ngram_heads_vocab_sizes = torch.tensor([7], dtype=torch.long)
    stub.ngram_heads_offsets = torch.tensor([0], dtype=torch.long)
    shards = [
        (f"ngram_embedding.shard_{index}.weight", _shard(index * 100))
        for index in range(NUM_SHARDS)
    ]
    weights = shards + [("ngram_heads_offsets", torch.tensor([0], dtype=torch.long))]
    with pytest.raises(ValueError, match="id space"):
        _load(stub, weights)
