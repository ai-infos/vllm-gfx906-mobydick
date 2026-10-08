# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident Qwen4Exp position-learning enhancement layers."""

import math
import os
from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)

# from ..common.ple import PLEVocabParallelEmbedding


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.eps)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


def canonical_ple_shard_rows(
    num_shards: int, shard_row_capacity: int, vocab_size: int
) -> list[int]:
    """The row count vLLM's own splitter would give each shard: one uniform stride
    with a short last shard.

    Used to *describe* the expectation -- the mapping itself is taken from the
    shards that were actually loaded -- and to warn when a checkpoint's split
    disagrees with it.
    """
    return [
        max(0, min(shard_row_capacity, vocab_size - index * shard_row_capacity))
        for index in range(num_shards)
    ]


def build_ple_shard_layout(
    rows: Sequence[int],
    *,
    num_shards: int,
    shard_row_capacity: int,
    vocab_size: int | None,
) -> tuple[list[int], list[int]]:
    """Turn the row count of each loaded shard into id -> (shard, row) boundaries.

    The boundaries have to come from the checkpoint, not from the runtime's own
    arithmetic. `shard_row_capacity` is computed before any weight is loaded --
    from the vocab layout this module builds itself, with a per-head prime and a
    padded total -- while the model's actual id space is whatever the checkpoint's
    `ngram_heads_offsets` / `ngram_heads_vocab_sizes` say once those buffers have
    been loaded over it. When the two disagree, a uniform-stride lookup reads the
    wrong row for every id past the first boundary, and ids in the gap between
    what the stride can address and the real id space index off the end of a
    shard: `Whittle-Qwen-3.8-35B-A3B` splits 7,812,500 x 4 + 7,790,000 rows
    (39,040,000 ids, matching its own flat 8 x 4,880,000 layout) against a stride
    of 7,808,128, which crashed the engine on the last 17,488 ids and mis-read
    every id >= 7,808,128 -- see issue #40.

    `vocab_size` is the model's id space, or None when it is not known yet (a
    stub load, or a checkpoint that never delivered the layout buffers): then the
    shards still define the row -> id mapping and only the consistency check is
    skipped. A `vocab_size` that does not match the rows is a hard error rather
    than a warning, because serving it means serving wrong embeddings silently.
    """
    if num_shards <= 0:
        raise ValueError(f"num shards must be positive, got {num_shards}")
    if len(rows) != num_shards:
        raise ValueError(
            f"expected {num_shards} PLE ngram shards, got {len(rows)}: {list(rows)}"
        )
    for index, count in enumerate(rows):
        if count <= 0:
            raise ValueError(f"PLE ngram shard {index} holds {count} rows")

    total = sum(rows)
    if vocab_size:
        if total != vocab_size:
            raise ValueError(
                f"PLE ngram shards hold {total} rows but the model's ngram id space "
                f"is {vocab_size} (shards: {list(rows)}). The checkpoint's split and "
                "its vocab layout disagree, so a lookup could not be mapped to a row "
                "that the checkpoint meant to store there; refusing to serve."
            )
        canonical = canonical_ple_shard_rows(num_shards, shard_row_capacity, vocab_size)
        if list(rows) != canonical:
            logger.warning(
                "PLE ngram shards are not vLLM's canonical split (%s rows, id space "
                "%d, stride %d) but %s rows totalling %d ids. Serving the checkpoint's "
                "own boundaries, which is correct for the rows on disk; a re-split "
                "would restore the uniform layout.",
                num_shards,
                vocab_size,
                shard_row_capacity,
                list(rows),
                total,
            )
    else:
        logger.warning(
            "PLE ngram id space is unknown (layout buffers not loaded); taking the "
            "row boundaries from the %d loaded shards (%s rows) without checking them "
            "against the model's vocab layout.",
            num_shards,
            list(rows),
        )

    starts: list[int] = []
    offset = 0
    for count in rows:
        starts.append(offset)
        offset += count
    return starts, [int(count) for count in rows]


class MmapShardedNGramEmbedding(nn.Module):
    """CPU-resident PLE ngram embedding backed directly by mmap'd safetensors
    shard tensors, with no TP sharding and no copying. Every rank maps the
    same on-disk shard files; the OS page cache backs all of them with the
    same physical pages, so the full table's resident-RAM cost is paid once
    across the whole node, not once per worker process."""

    def __init__(
        self,
        num_shards: int,
        shard_row_capacity: int,
        embedding_dim: int,
        dummy_weights: bool = False,
    ) -> None:
        super().__init__()
        self.num_shards = num_shards
        self.shard_row_capacity = shard_row_capacity
        self.embedding_dim = embedding_dim
        # Dummy-weight loads (``--load-format dummy``) never deliver shard
        # tensors: the shards are read straight from the checkpoint's
        # safetensors files and are not parameters, so nothing generates them.
        # Serving zero rows for a missing shard keeps the path exercisable
        # without a checkpoint (the tiny-config harness); a real load still
        # raises.
        self._dummy_weights = dummy_weights
        self.params_dtype: torch.dtype | None = None
        self._shards: list[torch.Tensor | None] = [None] * num_shards
        self._warned_bad_ids = False
        # Set by finalize_shard_layout() from the shards actually loaded. Until
        # then (and for dummy weights) the configured uniform capacity is used.
        self._shard_starts: list[int] | None = None
        self._shard_rows: list[int] | None = None

    def set_shard(self, shard_index: int, tensor: torch.Tensor) -> None:
        if tensor.device.type != "cpu":
            raise ValueError(
                f"PLE ngram shard {shard_index} must be loaded on CPU, "
                f"got device {tensor.device}"
            )
        if self.params_dtype is None:
            self.params_dtype = tensor.dtype
        elif tensor.dtype != self.params_dtype:
            raise ValueError(
                f"PLE ngram shard {shard_index} dtype {tensor.dtype} does not "
                f"match previously loaded shards' dtype {self.params_dtype}"
            )
        self._shards[shard_index] = tensor

    @property
    def table_rows(self) -> int:
        """Number of addressable ids: the loaded shards' rows once the layout is
        known, otherwise the configured uniform split's total."""
        if self._shard_rows is not None:
            return sum(self._shard_rows)
        return self.num_shards * self.shard_row_capacity

    def finalize_shard_layout(self, vocab_size: int | None) -> bool:
        """Fix the id -> (shard, row) map from the shards that were actually loaded.

        Called at the end of every `load_weights` call, so it is also the point at
        which a chunked delivery becomes complete. It needs the checkpoint's
        `ngram_heads_offsets` / `ngram_heads_vocab_sizes` buffers to have replaced
        the runtime's own layout -- only then is the model's id space known -- and
        the shards themselves, which are the only record of where each id's row
        lives. See `build_ple_shard_layout` for what goes wrong otherwise.

        Returns True once a layout was derived from every shard. Until then the
        configured uniform split stands, and a shard that never arrives still
        raises on use rather than being papered over.
        """
        if any(tensor is None for tensor in self._shards):
            if not self._dummy_weights:
                logger.debug(
                    "PLE ngram layout not finalised yet: %d of %d shards loaded",
                    sum(t is not None for t in self._shards),
                    self.num_shards,
                )
            return False
        rows = [int(tensor.shape[0]) for tensor in self._shards if tensor is not None]
        self._shard_starts, self._shard_rows = build_ple_shard_layout(
            rows,
            num_shards=self.num_shards,
            shard_row_capacity=self.shard_row_capacity,
            vocab_size=vocab_size,
        )
        return True

    # Ids outside the table's rows have two known producers,
    # both legitimate, and neither is corruption of the table:
    #
    #   * the MTP drafter's "no sample here" marker. `sample_idx_mapping` is
    #     pre-filled with PADDING_SENTINEL (spec_decode/dflash/speculator.py:145)
    #     and reaches this lookup on every drafter warmup/capture.
    #   * a pinned host id buffer that nothing has written yet. On 2026-09-23 a
    #     graph-capture-time call arrived holding the poison pattern
    #     0xff80ff80ff80ff80 repeated, which killed three boots with
    #     "PLE ngram id out of range" -- see the 2026-09-23 entry in
    #     docs/gfx906/degradation.md.
    #
    # Both are folded onto row 0 rather than rejected or clamped: rejecting them
    # refuses a boot whose inputs are fine by the time they matter (this op is a
    # splitting op, so it runs eagerly and the captured/replayed value is read
    # later, from a buffer the producer has filled by then), and clamping the
    # whole tensor would send large positive garbage to the *last* row, which is
    # a real embedding and therefore a plausible wrong answer. Row 0 costs one
    # wrong row for slots whose result is discarded anyway.
    PADDING_SENTINEL = -1

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.device.type != "cpu":
            raise ValueError("MmapShardedNGramEmbedding requires CPU ids")
        original_shape = ids.shape
        flat_ids = ids.reshape(-1).long()
        table_rows = self.table_rows
        if flat_ids.numel() > 0:
            # One range test over the whole tensor, host-side (ids are CPU by
            # the check above), instead of one per shard inside the loop below.
            low, high = int(flat_ids.min()), int(flat_ids.max())
            if low < 0 or high >= table_rows:
                bad = (flat_ids < 0) | (flat_ids >= table_rows)
                if os.environ.get("VLLM_GFX906_PLE_STRICT", "0") == "1":
                    geometry = (
                        f"{self.num_shards} shards, rows {self._shard_rows}"
                        if self._shard_rows is not None
                        else f"{self.num_shards} shards x {self.shard_row_capacity}"
                    )
                    raise ValueError(
                        f"PLE ngram id out of range: ids span [{low}, {high}] "
                        f"but the table holds {table_rows} rows ({geometry}), "
                        f"{int(bad.sum())} of {flat_ids.numel()} ids affected "
                        "(VLLM_GFX906_PLE_STRICT=1)"
                    )
                if not self._warned_bad_ids:
                    self._warned_bad_ids = True
                    logger.warning(
                        "PLE ngram lookup: %d of %d ids were outside [0, %d) "
                        "(span [%d, %d]); folded onto row 0. A few during boot "
                        "or graph capture are expected -- the pinned id buffer "
                        "is not always written first, and the drafter passes "
                        "%d for 'no sample here' -- but a steady stream means "
                        "the id producer is broken. Set "
                        "VLLM_GFX906_PLE_STRICT=1 to raise instead of folding.",
                        int(bad.sum()),
                        flat_ids.numel(),
                        table_rows,
                        low,
                        high,
                        self.PADDING_SENTINEL,
                    )
                flat_ids = flat_ids.masked_fill(bad, 0)
        starts, rows = self._shard_starts, self._shard_rows
        if starts is None:
            # No checkpoint layout recorded (dummy weights, or a direct-construction
            # harness): the configured uniform split is all there is.
            shard_idx = torch.div(
                flat_ids, self.shard_row_capacity, rounding_mode="floor"
            )
            local_idx = flat_ids - shard_idx * self.shard_row_capacity
        out = flat_ids.new_empty(
            (flat_ids.numel(), self.embedding_dim), dtype=self.params_dtype
        )
        # The loop below can only write rows whose shard index is in
        # range(num_shards). An id outside the table matches no mask, so its row
        # would keep whatever new_empty() found at that address: a silently wrong
        # embedding instead of an error. That is why the range test above folds
        # every out-of-range id onto row 0 first -- every row this function
        # returns is a row that was actually read.
        #
        # Where each id's row lives is decided by the shard boundaries
        # (finalize_shard_layout), not by a uniform stride: a checkpoint may split
        # its rows however it likes, and the runtime's stride is computed before
        # the checkpoint's vocab layout is known. Using the stride against a
        # different split both mis-maps ids and indexes off the end of a shard.
        #
        # Fixed-length loop over every shard, every call — no data-dependent
        # iteration count. Necessary for CUDA graph capture safety: a Python
        # loop whose length depends on which shards this specific batch
        # happens to touch can execute a different number of steps during
        # warmup/capture than during replay.
        for shard in range(self.num_shards):
            if starts is None:
                mask = shard_idx == shard
                local = local_idx[mask]
            else:
                start = starts[shard]
                mask = (flat_ids >= start) & (flat_ids < start + rows[shard])
                local = flat_ids[mask] - start
            if not mask.any():
                continue
            tensor = self._shards[shard]
            if tensor is None:
                if self._dummy_weights:
                    out[mask] = 0.0
                    continue
                raise RuntimeError(f"PLE ngram shard {shard} was never loaded")
            out[mask] = tensor.index_select(0, local)
        return out.reshape(*original_shape, self.embedding_dim)


class Qwen4ExpNGramEmbedding(nn.Module):
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PLE_LAYER_PRIME = 10007

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        """Mix an integer into a deterministic unsigned 64-bit value."""
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @staticmethod
    def _is_prime_64(value: int) -> bool:
        """Return whether a 64-bit integer is prime."""
        if value < 2:
            return False
        for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if value % prime == 0:
                return value == prime
        exponent = value - 1
        shifts = 0
        while exponent % 2 == 0:
            exponent //= 2
            shifts += 1
        for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if base % value == 0:
                continue
            witness = pow(base, exponent, value)
            if witness in (1, value - 1):
                continue
            for _ in range(shifts - 1):
                witness = pow(witness, 2, value)
                if witness == value - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _nth_prime_after(cls, start: int, count: int) -> int:
        """Return the ``count``-th prime strictly greater than ``start``."""
        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not cls._is_prime_64(candidate):
                candidate += 2
            prime = candidate
        return prime

    @classmethod
    def _make_layer_multipliers(
        cls,
        *,
        ngram_size: int,
        unigram_vocab_size: int,
        seed: int,
        ple_dense_layer_id: int,
    ) -> list[int]:
        """Build deterministic hash multipliers for one PLE layer."""
        max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + cls._PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(ngram_size):
            value = base_seed + cls._SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (cls._splitmix64(value) % half_bound) + 1)
        return multipliers

    @classmethod
    def _make_vocab_layout(
        cls,
        *,
        ngram_vocab_size_base: int,
        ngram_heads: int,
        ple_dense_layer_id: int,
    ) -> tuple[list[int], list[int], int]:
        """Build per-head vocabulary sizes, offsets, and total row count."""
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(ngram_heads):
            global_head = ple_dense_layer_id * ngram_heads + local_head
            size = cls._nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        max_num_reqs: int,
        prefix: str,
        layer_name: str,
        runtime_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.runtime_dtype = runtime_dtype
        self.layer_name = layer_name
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self._pinned_ngram_ids = torch.empty(
            (max_total_tokens, self.ngram_heads), dtype=torch.long, device="cpu"
        ).pin_memory()
        self._pinned_output = torch.empty(
            (max_total_tokens, self.embedding_dim),
            dtype=self.runtime_dtype,
            device="cpu",
        ).pin_memory()

        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        sizes, offsets, total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((total_vocab_size + divisor - 1) // divisor) * divisor
        shard_row_capacity = (
            padded_vocab_size + self.split_ngram_parts - 1
        ) // self.split_ngram_parts
        self.ngram_embedding = MmapShardedNGramEmbedding(
            dummy_weights=(
                get_current_vllm_config().load_config.load_format == "dummy"
            ),
            num_shards=self.split_ngram_parts,
            shard_row_capacity=shard_row_capacity,
            embedding_dim=self.head_dim,
            # params_dtype=torch.get_default_dtype(),
        )

        self.register_buffer(
            "positions_buffer",
            torch.arange(max_total_tokens, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "padded_buffer",
            torch.full(
                (max_num_reqs, max_total_tokens),
                self.eos_token_id,
                dtype=torch.int64,
            ),
            persistent=False,
        )

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1).long()
        query_start_loc = query_start_loc.long()
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]
        if num_tokens > self.positions_buffer.numel():
            raise ValueError(
                f"PLE received {num_tokens} tokens, but its workspace supports "
                f"at most {self.positions_buffer.numel()}"
            )
        if num_reqs > self.padded_buffer.shape[0]:
            raise ValueError(
                f"PLE received {num_reqs} requests, but its workspace supports "
                f"at most {self.padded_buffer.shape[0]}"
            )

        positions = self.positions_buffer[:num_tokens]
        packed = self.padded_buffer[:num_reqs]
        packed.fill_(self.eos_token_id)
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        packed[request_indices, columns] = input_ids
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )
        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[index] * self.layer_multipliers[index]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        ngram_ids = torch.cat(id_blocks, dim=-1)
        output = ngram_ids.new_empty(
            (ngram_ids.shape[0], self.embedding_dim),
            dtype=self.runtime_dtype,
            #            dtype=self.ngram_embedding.params_dtype,
        )
        torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding(
            ngram_ids,
            output,
            self.layer_name,
        )
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""

        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                embedding = self.ngram_embedding
                if loaded_weight.shape[1] != embedding.embedding_dim:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected embedding_dim {embedding.embedding_dim}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                # Store the mmap-backed tensor by reference. No .copy_() or
                # device move: this keeps the ~100GB table page-cache-backed
                # and shared across every worker process on the node instead
                # of privately materialized once per process.
                embedding.set_shard(shard_index, loaded_weight.to("cpu"))
                loaded.add(f"ngram_embedding.shard_{shard_index}")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        # Every shard and both layout buffers are in by now, so this is the first
        # point where the checkpoint's id space and its row split can be compared.
        # A checkpoint whose split does not add up to its own id space is refused
        # here rather than mis-read at serving time.
        layout = self.ngram_heads_offsets + self.ngram_heads_vocab_sizes
        vocab_size = int(layout.max()) if layout.numel() else 0
        self.ngram_embedding.finalize_shard_layout(
            vocab_size if vocab_size > 0 else None
        )
        return loaded


class Qwen4ExpPLELayer(nn.Module, MambaBase):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        self.ple_embedding: nn.Module = Qwen4ExpNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            self.ple_dense_layer_id,
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.scheduler_config.max_num_seqs,
            f"{prefix}.ple_embedding",
            prefix,
            runtime_dtype=model_config.dtype,
        )
        self.key_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hc_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + 1,
            num_spec=self.num_spec_tokens,
        )

    def _apply_norm(
        self, norm: Qwen4ExpPLEGroupedNorm, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        shape = hidden_states.shape
        return norm(hidden_states.flatten(-2)).reshape(shape)

    def _short_conv_fallback(self, inputs: torch.Tensor) -> torch.Tensor:
        # Profiling / CUDA graph capture only; conv state is not updated.
        inputs_t = inputs.transpose(0, 1).unsqueeze(0)
        output = self.conv1d(inputs_t)[..., : inputs_t.size(-1)]
        return F.silu(output).squeeze(0).transpose(0, 1)

    def _short_conv_dilated_decode_batched(
        self,
        x_d: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        state_indices_tensor_d: torch.Tensor,
        has_initial_states_d: torch.Tensor | None,
    ) -> torch.Tensor:
        state_indices = state_indices_tensor_d.to(
            device=conv_state.device, dtype=torch.int64
        )
        # FULL cudagraph padded decode rows use NULL_BLOCK_ID. Remap them to
        # slot 0 for a safe gather, then zero output and skip write-back.
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        if has_initial_states_d is None:
            has_initial_state = valid_state
        else:
            if has_initial_states_d.numel() < state_indices_tensor_d.numel():
                raise ValueError(
                    "has_initial_states_d size mismatch: "
                    f"got {has_initial_states_d.numel()}, "
                    f"need >= {state_indices_tensor_d.numel()}."
                )
            has_initial_state = has_initial_states_d[
                : state_indices_tensor_d.numel()
            ].to(device=conv_state.device, dtype=torch.bool)
            has_initial_state = has_initial_state & valid_state

        cached_state = conv_state.index_select(0, state_indices)
        state = cached_state[..., : self.conv_state_len].to(x_d.dtype)
        if self.conv_state_len > 0:
            initial_state = torch.where(
                has_initial_state.view(-1, 1, 1),
                state,
                torch.zeros_like(state),
            )
            history = torch.cat((initial_state, x_d.unsqueeze(-1)), dim=-1)
        else:
            history = x_d.unsqueeze(-1)

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        ).squeeze(-1)
        output = F.silu(conv_output)
        output = output * valid_state.view(-1, 1).to(output.dtype)

        if self.conv_state_len > 0:
            next_state = history[..., -self.conv_state_len :]
            # Padded rows are remapped to the reserved null slot. Preserve its
            # existing value while writing the new states for valid rows.
            existing_base_state = cached_state[..., : self.conv_state_len]
            safe_next_state = torch.where(
                valid_state.view(-1, 1, 1),
                next_state.to(conv_state.dtype),
                existing_base_state,
            )
            cached_state[..., : self.conv_state_len] = safe_next_state
            conv_state.index_copy_(0, state_indices, cached_state)

        return output

    def _short_conv_dilated_prefill_batched(
        self,
        x_p: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        state_indices_tensor_p: torch.Tensor,
        num_prefills: int,
        num_decode_tokens: int,
        num_prefill_tokens: int,
    ) -> torch.Tensor:
        # ``non_spec_query_start_loc`` covers the non-spec (decode + prefill)
        # requests and equals ``query_start_loc`` when spec-decode is inactive.
        non_spec_query_start_loc = metadata.non_spec_query_start_loc
        if non_spec_query_start_loc is None:
            raise ValueError("query_start_loc is required for prefill short-conv")
        query_start_loc_p = (
            non_spec_query_start_loc[-num_prefills - 1 :] - num_decode_tokens
        )
        # The metadata builder guarantees that the prefill query offsets start
        # at 0 and end at num_prefill_tokens. Avoid reading those values here,
        # since doing so would force a device-to-host synchronization.
        has_initial_states_p = metadata.has_initial_states_p
        if has_initial_states_p is None:
            raise ValueError("has_initial_states_p is required for prefill short-conv")

        output = torch.empty_like(x_p)
        q_starts = query_start_loc_p.to(torch.int64)
        if state_indices_tensor_p.numel() < num_prefills:
            raise ValueError(
                "state_indices_tensor_p size mismatch: "
                f"got {state_indices_tensor_p.numel()}, "
                f"need >= {num_prefills}."
            )
        if has_initial_states_p.numel() < num_prefills:
            raise ValueError(
                "has_initial_states_p size mismatch: "
                f"got {has_initial_states_p.numel()}, "
                f"need >= {num_prefills}."
            )
        if num_prefills == 0 or x_p.numel() == 0:
            return output
        lengths = q_starts[1:] - q_starts[:-1]
        # Use the CPU-computed packing width from the metadata builder instead
        # of synchronizing on lengths.max().
        max_len = metadata.max_prefill_query_len
        if max_len <= 0:
            return output

        hidden_size = x_p.shape[1]
        positions = torch.arange(
            num_prefill_tokens, device=x_p.device, dtype=torch.int64
        )
        req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
        col_indices = positions - q_starts[req_indices]

        packed_tokens = x_p.new_zeros((num_prefills, max_len, hidden_size))
        packed_tokens[req_indices, col_indices] = x_p
        packed_tokens = packed_tokens.transpose(1, 2).contiguous()

        state_indices = state_indices_tensor_p[:num_prefills].to(
            device=conv_state.device, dtype=torch.int64
        )
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        has_initial = has_initial_states_p[:num_prefills].to(
            device=conv_state.device, dtype=torch.bool
        )
        if self.conv_state_len > 0:
            if conv_state.shape[0] == 0:
                state = conv_state.new_zeros(
                    (num_prefills, hidden_size, self.conv_state_len),
                    dtype=x_p.dtype,
                )
            else:
                state = conv_state.index_select(0, state_indices)[
                    ..., : self.conv_state_len
                ].to(x_p.dtype)
            use_initial_mask = (valid_state & has_initial).view(num_prefills, 1, 1)
            initial_state = torch.where(
                use_initial_mask,
                state,
                torch.zeros_like(state),
            )
            history = torch.cat((initial_state, packed_tokens), dim=-1)
        else:
            history = packed_tokens

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        )
        conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

        token_positions = torch.arange(max_len, device=x_p.device, dtype=torch.int64)
        valid_tokens = token_positions.view(1, max_len) < lengths.view(num_prefills, 1)
        valid_output_mask = valid_tokens & valid_state.to(device=x_p.device).view(
            num_prefills, 1
        )
        conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
        output.copy_(conv_output[req_indices, col_indices])

        if self.conv_state_len > 0 and conv_state.shape[0] > 0:
            state_starts = lengths.to(device=history.device, dtype=torch.int64).view(
                num_prefills, 1, 1
            )
            state_offsets = torch.arange(
                self.conv_state_len, device=history.device, dtype=torch.int64
            ).view(1, 1, self.conv_state_len)
            next_state = history.gather(
                dim=2,
                index=(state_starts + state_offsets).expand(-1, history.size(1), -1),
            )
            # Write back without a host synchronization. Valid, non-empty rows
            # receive their new state; padding and zero-length rows keep the
            # current cache value.
            existing_state = conv_state.index_select(0, state_indices)
            existing_base_state = existing_state[..., : self.conv_state_len]
            update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
            safe_next_state = torch.where(
                update_mask.view(num_prefills, 1, 1),
                next_state.to(conv_state.dtype),
                existing_base_state,
            )
            existing_state[..., : self.conv_state_len] = safe_next_state
            conv_state.index_copy_(0, state_indices, existing_state)
        return output

    def _short_conv_dilated_spec_batched(
        self,
        x_spec: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        spec_state_indices_tensor: torch.Tensor,
        spec_query_start_loc: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        spec_query_len: int,
    ) -> torch.Tensor:
        """Dilated short-conv for speculative-decode (MTP) requests.

        Each spec request feeds multiple (draft + 1) query tokens. The conv
        outputs are computed causally after rolling back the previous draft
        state by ``num_accepted_tokens - 1``. The current candidate inputs stay
        in the extended cache for the next forward, matching
        ``causal_conv1d_update``.

        ``spec_query_len`` (== num_speculative_tokens + 1) is the maximum query
        length and is a Python int, so no host synchronization is needed; this
        keeps the path safe for full CUDA-graph capture/replay where the buffers
        are padded at the request level.
        """
        num_reqs = spec_state_indices_tensor.numel()
        hidden_size = x_spec.size(-1)
        # Use a fixed packing width instead of synchronizing on lengths.max().
        max_len = spec_query_len
        # Full CUDA graphs can pad these buffers. Only the first num_reqs
        # accepted-token counts belong to actual speculative requests.
        num_accepted_tokens = num_accepted_tokens[:num_reqs]
        q_starts = spec_query_start_loc[: num_reqs + 1].to(torch.int64)
        # Keep the number of real speculative tokens on the device.
        total_real_tokens = q_starts[num_reqs]

        state_indices = spec_state_indices_tensor.to(
            device=conv_state.device, dtype=torch.int64
        )
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        positions = torch.arange(
            x_spec.size(0), device=x_spec.device, dtype=torch.int64
        )
        # Route graph-padded token rows to the discarded dummy request so that
        # they cannot overwrite real packed data.
        req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
        valid_tokens = (positions < total_real_tokens) & (req_indices < num_reqs)
        clamped_req_indices = req_indices.clamp_max(max(num_reqs - 1, 0))
        col_indices = (positions - q_starts[clamped_req_indices]).clamp_(0, max_len - 1)
        pack_req_indices = torch.where(
            valid_tokens,
            clamped_req_indices,
            torch.full_like(req_indices, num_reqs),
        )
        pack_col_indices = torch.where(
            valid_tokens, col_indices, torch.zeros_like(col_indices)
        )

        # The last request row is the dummy sink for graph padding.
        packed = x_spec.new_zeros((num_reqs + 1, max_len, hidden_size))
        packed[pack_req_indices, pack_col_indices] = x_spec
        packed = packed.transpose(1, 2).contiguous()

        if self.conv_state_len > 0:
            cached_state = conv_state.index_select(0, state_indices)
            rollback_offsets = num_accepted_tokens.to(
                device=conv_state.device, dtype=torch.int64
            ).sub(1)
            rollback_offsets = torch.where(
                valid_state,
                rollback_offsets.clamp_(0, max_len - 1),
                torch.zeros_like(rollback_offsets),
            )
            state_offsets = torch.arange(
                self.conv_state_len, device=conv_state.device, dtype=torch.int64
            ).view(1, 1, self.conv_state_len)
            rollback_indices = rollback_offsets.view(-1, 1, 1) + state_offsets
            state = cached_state.gather(
                2, rollback_indices.expand(-1, hidden_size, -1)
            ).to(x_spec.dtype)
            state = torch.where(
                valid_state.view(num_reqs, 1, 1),
                state,
                torch.zeros_like(state),
            )
            # Append a zeroed dummy-row state to match the [num_reqs + 1] pack.
            dummy_state = state.new_zeros((1, hidden_size, self.conv_state_len))
            state_full = torch.cat((state, dummy_state), dim=0)
            history = torch.cat((state_full, packed), dim=-1)
        else:
            history = packed

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        )
        conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

        output = conv_output[pack_req_indices, pack_col_indices]
        output = output * valid_tokens.view(-1, 1).to(output.dtype)

        # Keep all current candidate inputs in the extended state. On the next
        # target forward, ``num_accepted_tokens - 1`` selects the rollback
        # window before processing the newly scheduled tokens.
        if self.conv_state_len > 0:
            state_capacity = self.conv_state_len + max_len - 1
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache cannot retain speculative tokens: "
                    f"got {conv_state.size(-1)}, need {state_capacity}."
                )
            candidate_state = history[:num_reqs, :, 1 : state_capacity + 1]
            query_lengths = q_starts[1:] - q_starts[:-1]
            state_positions = torch.arange(
                state_capacity, device=history.device, dtype=torch.int64
            ).view(1, 1, state_capacity)
            update_lengths = (self.conv_state_len + query_lengths - 1).view(
                num_reqs, 1, 1
            )
            update_mask = valid_state.view(num_reqs, 1, 1) & (
                state_positions < update_lengths
            )
            existing_state = cached_state[..., :state_capacity]
            next_state = torch.where(
                update_mask,
                candidate_state.to(conv_state.dtype),
                existing_state,
            )
            cached_state[..., :state_capacity] = next_state
            conv_state.index_copy_(0, state_indices, cached_state)

        return output

    def _short_conv_dilated_dispatch(
        self,
        inputs: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        has_spec = metadata.spec_sequence_masks is not None
        x = inputs[: metadata.num_actual_tokens]

        # Split spec / non-spec tokens.
        if has_spec:
            if has_prefill or has_decode:
                assert metadata.spec_token_indx is not None
                assert metadata.non_spec_token_indx is not None
                x_spec = x.index_select(0, metadata.spec_token_indx.long())
                x_non_spec = x.index_select(0, metadata.non_spec_token_indx.long())
            else:
                x_spec = x
                x_non_spec = None
        else:
            x_spec = None
            x_non_spec = x

        spec_output = None
        # 1. Run the multi-query speculative-decode part.
        if has_spec:
            assert metadata.spec_state_indices_tensor is not None
            assert metadata.spec_query_start_loc is not None
            assert metadata.num_accepted_tokens is not None
            spec_output = self._short_conv_dilated_spec_batched(
                x_spec=x_spec,
                conv_state=conv_state,
                conv_weights=conv_weights,
                spec_state_indices_tensor=metadata.spec_state_indices_tensor[
                    : metadata.num_spec_decodes
                ],
                spec_query_start_loc=metadata.spec_query_start_loc,
                num_accepted_tokens=metadata.num_accepted_tokens,
                spec_query_len=metadata.spec_query_len,
            )

        # 2. Run regular decode and prefill requests.
        conv_out_non_spec = None
        state_indices_tensor = metadata.state_indices_tensor
        if x_non_spec is not None:
            assert state_indices_tensor is not None
            if has_prefill:
                state_indices_tensor_d, state_indices_tensor_p = torch.split(
                    state_indices_tensor,
                    [num_decodes, num_prefills],
                    dim=0,
                )
                x_d, x_p = torch.split(
                    x_non_spec,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )
                non_spec_parts: list[torch.Tensor] = []
                if has_decode:
                    non_spec_parts.append(
                        self._short_conv_dilated_decode_batched(
                            x_d=x_d,
                            conv_state=conv_state,
                            conv_weights=conv_weights,
                            state_indices_tensor_d=state_indices_tensor_d,
                            has_initial_states_d=metadata.has_initial_states_d,
                        )
                    )
                non_spec_parts.append(
                    self._short_conv_dilated_prefill_batched(
                        x_p=x_p,
                        metadata=metadata,
                        conv_state=conv_state,
                        conv_weights=conv_weights,
                        state_indices_tensor_p=state_indices_tensor_p,
                        num_prefills=num_prefills,
                        num_decode_tokens=num_decode_tokens,
                        num_prefill_tokens=num_prefill_tokens,
                    )
                )
                conv_out_non_spec = torch.vstack(non_spec_parts)
            else:
                conv_out_non_spec = self._short_conv_dilated_decode_batched(
                    x_d=x_non_spec,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices_tensor_d=state_indices_tensor[: x_non_spec.size(0)],
                    has_initial_states_d=metadata.has_initial_states_d,
                )

        # 3. Merge both parts back into the original token order.
        if has_spec and conv_out_non_spec is not None:
            assert metadata.spec_token_indx is not None
            assert metadata.non_spec_token_indx is not None
            assert spec_output is not None
            output = x.new_empty((metadata.num_actual_tokens, x.size(-1)))
            output.index_copy_(0, metadata.spec_token_indx, spec_output)
            output.index_copy_(0, metadata.non_spec_token_indx, conv_out_non_spec)
            return output
        elif has_spec:
            assert spec_output is not None
            return spec_output
        if conv_out_non_spec is None:
            return x
        return conv_out_non_spec

    def _short_conv(self, inputs: torch.Tensor) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return self._short_conv_fallback(inputs)

        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}."
            )

        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            # MRV2 omits Mamba-family metadata during profile warmup.
            return self._short_conv_fallback(inputs)
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got "
                f"{type(layer_attn_metadata).__name__}."
            )

        conv_state = self.kv_cache[0]
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.squeeze(1)

        state_capacity = self.conv_state_len + self.num_spec_tokens
        if state_capacity > 0:
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache is smaller than expected for "
                    f"dilated convolution: got {conv_state.size(-1)}, "
                    f"expect at least {state_capacity}."
                )
            conv_state = conv_state[..., -state_capacity:]
        return self._short_conv_dilated_dispatch(
            inputs,
            layer_attn_metadata,
            conv_state,
            conv_weights.to(dtype=inputs.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        embeddings = self.ple_embedding(input_ids, query_start_loc, ngram_context)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        key = key.reshape(token_count, self.hc_count, self.hidden_size)
        query = hidden_states.reshape(token_count, self.hc_count, self.hidden_size)
        key = self._apply_norm(self.norm_key, key)
        query = self._apply_norm(self.norm_query, query)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated_value = gate * value.unsqueeze(-2)
        normalized = self._apply_norm(self.norm_conv, gated_value).flatten(-2)
        conv_output = torch.zeros_like(normalized)
        torch.ops.vllm.qwen4_exp_ple_short_conv(
            normalized,
            conv_output,
            self.prefix,
        )
        return gated_value.flatten(-2) + conv_output


def qwen4_exp_amd_ple_ngram_embedding(
    ngram_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpPLELayer):
        raise TypeError(f"{layer_name} is not a Qwen4Exp PLE owner")
    ple_embedding = layer.ple_embedding
    n = ngram_ids.shape[0]
    pinned_ids = ple_embedding._pinned_ngram_ids[:n]
    # The mmap'd shard lookup below runs on the host, so the device->host copy
    # has to have landed before the host reads the buffer.  non_blocking=True
    # only makes the copy stream-ordered; it does not wait for it.  The host
    # would otherwise read ids that are stale, or -- on the first step after
    # init, since _pinned_ngram_ids is torch.empty(...).pin_memory() and is
    # never zeroed -- outright uninitialized.  Those values then index the
    # shard table outside its row range (see MmapShardedNGramEmbedding.forward),
    # which is what killed the engine at 00:26 / 00:31 with 'index out of range
    # in self'.  The H2D copy of the result at the end stays async: it is
    # stream-ordered against whatever consumes `output` and needs no host wait.
    pinned_ids.copy_(ngram_ids, non_blocking=False)
    result = ple_embedding.ngram_embedding(pinned_ids).flatten(-2)
    pinned_out = ple_embedding._pinned_output[:n]
    pinned_out.copy_(result.to(dtype=output.dtype))
    output.copy_(pinned_out, non_blocking=True)


# def qwen4_exp_amd_ple_ngram_embedding(
#     ngram_ids: torch.Tensor,
#     output: torch.Tensor,
#     layer_name: str,
# ) -> None:
#     """Run the large PLE embedding lookup outside Inductor's FX graph.

#     Keeping the embedding weight in ``static_forward_context`` prevents AOT
#     compile-time autotuning from materializing a synthetic copy of the weight.
#     """
#     layer = get_forward_context().no_compile_layers[layer_name]
#     if not isinstance(layer, Qwen4ExpPLELayer):
#         raise TypeError(f"{layer_name} is not a Qwen4Exp PLE owner")
#     result = layer.ple_embedding.ngram_embedding(ngram_ids).flatten(-2)
#     output.copy_(result)


def qwen4_exp_ple_short_conv(
    inputs: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    result = layer._short_conv(inputs)
    output[: result.shape[0]].copy_(result)


direct_register_custom_op(
    op_name="qwen4_exp_amd_ple_ngram_embedding",
    op_func=qwen4_exp_amd_ple_ngram_embedding,
    mutates_args=["output"],
)


direct_register_custom_op(
    op_name="qwen4_exp_ple_short_conv",
    op_func=qwen4_exp_ple_short_conv,
    mutates_args=["output"],
)


__all__ = [
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLEGroupedNorm",
    "Qwen4ExpPLELayer",
]
