# SPDX-License-Identifier: Apache-2.0
# Copyright (C) Nick — nick413@gmail.com
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
#
# Vendored from https://github.com/cassettesgoboom/gfx906-fa-vllm
# (FlashAttention-style custom attention backend for vLLM on AMD gfx906).
"""
vLLM v1 attention backend for gfx906 (MI50) built on the Q8
FlashAttention kernel ported from llama.cpp-gfx906.

Registered as AttentionBackendEnum.CUSTOM (selected per-model via
`attention_config={"backend": "CUSTOM"}`; the legacy
VLLM_ATTENTION_BACKEND env var no longer gates this path):

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum, register_backend,
    )
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "gfx906_fa_backend.Gfx906FABackend",
    )

The KV cache layout matches TritonAttentionBackend:
    (num_blocks, 2, block_size, num_kv_heads, head_size)
so the backend can be switched without allocator changes.

Decode (LEGACY=1, the default) gathers K/V from the paged fp16 cache
into contiguous fp16 buffers with a fused HIP gather kernel, quantizes
K to Q8 on device, and runs the Q8 FA kernel.
"""

import dataclasses
import os as _os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType

# Project modules (need to be importable from the vllm package)
from vllm.gfx906_fa.gfx906_fa_paged import (  # noqa: E402
    _pick_ncols1,
    forward_paged,
)
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
@dataclass
class Gfx906FAMetadata:
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor  # [B+1] int32
    seq_lens: torch.Tensor  # [B]   int32
    block_table: torch.Tensor  # [B, max_num_blocks] int32
    slot_mapping: torch.Tensor  # [num_tokens] int64
    query_start_loc_cpu: torch.Tensor | None = None  # [B+1] host int (M3:
    # lets forward_paged skip the per-seq int(cu[...]) D2H syncs)
    use_cascade: bool = False
    common_prefix_len: int = 0
    # Per-batch causality, taken from CommonAttentionMetadata.causal. False means a
    # bidirectional batch (a spec-decode drafter: DFlash assistants build their
    # layers with causal=False), which must not get the inline causal clip. A
    # *tensor* causal (per-token masks, hybrid models) is not expressible here and
    # is treated as causal with a one-time warning -- fail-closed to today's
    # behaviour, which is what such a model gets now.
    causal: bool = True


# Kernel head dims the launcher instantiates (see csrc/gfx906_fa/gfx906_fa_launcher.cu).
# 96 was added 2026-09-16 (FA-D96) for the ViT's 72 and the head_dim-96 text class; the
# pad MAP stays on _FALLBACK_HEAD_DIMS until that item's real-model gates pass (see
# _pad96_enabled), so this set is the kernel whitelist, not the default pad map.
_INSTANTIATED_HEAD_DIMS = (64, 96, 128, 256)
_FALLBACK_HEAD_DIMS = (64, 128, 256)


_DEBUG_SHAPES = [0]  # GFX906_FA_DEBUG_SHAPES prints the first few calls


def _batch_causal(common_attn_metadata) -> bool:
    """Per-batch causality from CommonAttentionMetadata (`causal: bool | Tensor`).

    False -> a fully bidirectional batch (no causal mask). This is the same contract
    TRITON_ATTN and ROCM_ATTN honour for the bool form -- their kernels pass it straight
    to their mask logic and clip nothing -- so serving it here is not a weaker claim than
    the reference backends make. (The *window* differs: they symmetrize a causal sliding
    window via `_maybe_symmetrize_window` while stage 1 of FA-NONCAUSAL drops it, i.e. a
    superset mask; see DEVLOG-fa-noncausal.md and the acceptance guard noted there.)

    Tensor -> per-token causality, which this kernel cannot express. Warn once and keep
    the causal behaviour: that is exactly what such a batch got before this feature
    existed (the impl never read causality), so it preserves the status quo instead of
    making a silent new claim.
    """
    causal = getattr(common_attn_metadata, "causal", True)
    if isinstance(causal, torch.Tensor):
        logger.warning_once(
            "GFX906_FA: per-token (tensor) causality is not supported by the "
            "custom FA kernel; serving this batch as causal - the same behaviour "
            "as before FA-NONCAUSAL. Use TRITON_ATTN if the model needs per-token "
            "masks."
        )
        return True
    return causal is not False


def _pad_head_dim(head_size: int) -> int | None:
    """Smallest servable kernel head dim that fits (None if none does).

    Mirrors the ViT path (`gfx906_fa_mm_encoder.py`), which serves every head dim up to
    256 by zero-padding to the next instantiated dim. Padding is exact here: padded Q
    dims add 0 to the QK dot, padded K dims quantise to zero q8_0 blocks, padded V dims
    add 0 to P*V, and the head-dim padding leaves the softmax denominator unchanged.
    The cost is real (QK/PV work grows) but it replaces a fallback to ROCM_ATTN or
    TRITON_ATTN, which loses the tuned kernel entirely.

    96 is in the pad map since the FA-D96 gate (2026-09-16, default on): an exact 96 is
    served natively and 72/80 pad onto 96 instead of 128. ``GFX906_FA_PAD96=0`` restores
    the (64, 128, 256) map. See docs/gfx906/DEVLOG-fa-d96.md.
    """
    dims = _INSTANTIATED_HEAD_DIMS if _pad96_enabled() else _FALLBACK_HEAD_DIMS
    for head_dim in dims:
        if head_size <= head_dim:
            return head_dim
    return None


def _pad96_enabled() -> bool:
    """Whether the 96-wide kernel is used by the pad map (default ON since 2026-09-16).

    Gated on two A-B-A serving runs, both same-boot and mclk-1000:

    * **ViT (72 -> 96)**: image-prompt TTFT, dense 27B VL, 1024x1024 fresh image per
      rep, encoder-cache control — pad128 **5.151** / pad96 **5.080** / pad128
      **5.155** s (6 reps/arm; order control +0.08 %), i.e. **-1.46 %**. The 6-rep
      distributions do not overlap (pad96 max 5.133 < pad128 min 5.127).
    * **head_dim-96 text (Phi-3-mini, native 96)**: pp2048/tg256, 4 samples —
      pad128 **36.164** / native96 **36.379** / pad128 **36.092** t/s, i.e. **+0.69 %**
      (order control -0.2 %).

    Both are far smaller than the -5 % the item estimated (the ViT attention is ~19 %
    of TTFT and the kernel only removes ~12 % of that call), but they are consistent
    and the class also gains a 25 % narrower KV row. Kernel outputs are bit-identical
    between arms (the removed dims are the all-zero q8_0 block) and the FA suite plus
    a dedicated D=96 test pin correctness on both entry points.
    ``GFX906_FA_PAD96=0`` is the rollback to the (64, 128, 256) map.
    """
    return _os.environ.get("GFX906_FA_PAD96", "1") == "1"


def _padded_head_size(head_size: int) -> int | None:
    """``head_size`` as a servable dim, or None when it cannot be served.

    Default ON since the Phi-3-mini gate (2026-09-16, FA-COVER-1 step 2): head_dim 96 went
    from a silent fallback to CUSTOM at 36.41 t/s vs 28.62 (+27 %), with identical top-5
    tokens and PPL within 0.11 % of the fallback arm (0 top-20 misses in both). The cost is
    KV bytes: the row grows by the pad ratio, and ``GFX906_FA_PAD=0`` restores the old
    behaviour (instantiated dims only). Instantiated dims are returned unchanged either way.

    FA-D96 (2026-09-16) added 96 to the map (`_pad96_enabled`, default on): 96 is served
    natively, 72/80 pad onto 96. ``GFX906_FA_PAD96=0`` restores (64, 128, 256), which is
    what the pre-FA-D96 builds did.
    """
    if _os.environ.get("GFX906_FA_PAD", "1") != "1":
        # No padding: only dims in the *active* map are servable. With the FA-D96 opt-in
        # off that excludes 96 — the pre-FA-D96 semantics of this switch (Phi-3 falls
        # back rather than being served by a dim the map does not carry).
        dims = _INSTANTIATED_HEAD_DIMS if _pad96_enabled() else _FALLBACK_HEAD_DIMS
        return head_size if head_size in dims else None
    return _pad_head_dim(head_size)


def _resolve_legacy_mode() -> bool:
    """Resolve ``GFX906_FA_LEGACY``: 1 = LEGACY inline-quantize read path, 0/unset = Q8 side-buffer.

    The side-buffer path was verified against 0.29's fused KV-cache content axis
    (#51718) on 2026-09-16: PPL 10.5472/10.5460 across runs vs LEGACY=1's 10.5472
    — the same to within the probe's own run-to-run spread (<= 0.0012), 0 top-20
    misses throughout — and -15.5 % / -19.1 % ms/step with MTP k=3 at 64k / 120k
    acceptance unchanged — hence the default. LEGACY=1 stays as the validated
    rollback: it is ~6 % faster for B=1 greedy decode, the one regime it wins.
    ``GFX906_FA_LEGACY_ALLOW_UNVERIFIED`` is obsolete and ignored.
    See docs/gfx906/DEVLOG-fa-legacy0-b1-decode.md (ROADMAP KVLAYOUT-1).
    """
    return _os.environ.get("GFX906_FA_LEGACY", "0") == "1"


class Gfx906FAMetadataBuilder(AttentionMetadataBuilder[Gfx906FAMetadata]):
    # P3-3a M2: the LEGACY (inline-quant) decode path is FULL-capture-safe
    # (first FULL capture runs at profile_seq_lens=max_model_len, so
    # Sk-sized buffers allocate at capacity; metadata is runner-staged and
    # re-read live at replay). Verified: serving bench 53.09 t/s + 128/128
    # greedy probe identical to the Triton-FULL reference, plus
    # test_cudagraph_capture_replay_legacy_decode_path.
    # GFX906_FA_CG=never|always still overrides for experiments.
    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec
    ) -> AttentionCGSupport:
        # LEGACY=0 (Q8 side-view path): the Q8 view aliases the fp16 K
        # half, so the old desync class is structural-impossible (page
        # copies and captured writes move both halves at once). Kept as a
        # separate env-gated mode because it is a different read path
        # (experimental status until the serving gates say otherwise).
        # The Q8 side-view path (now the default) aliases the fp16 K half, so the
        # old desync class cannot arise; only worth a debug line for provenance.
        if not _resolve_legacy_mode():
            logger.debug(
                "GFX906_FA_LEGACY=0: K is read from the Q8 side view "
                "aliased into the fp16 K half (zero extra KV memory)."
            )
        mode = _os.environ.get("GFX906_FA_CG", "decode").lower()
        if mode == "always":
            return AttentionCGSupport.ALWAYS
        if mode == "decode":
            # With spec decode every decode step has query len
            # 1 + num_speculative_tokens; declaring
            # UNIFORM_SINGLE_TOKEN_DECODE makes vLLM demote the whole
            # engine to PIECEWISE for spec runs (measured ~3x step cost
            # on gfx906, 2026-08-18). The Q8 paged kernel is q_len-
            # generic (seq_q + q_abs_offset inline causal) and the
            # LEGACY=1 inline-quant path is FULL-capture safe (P3-3a M2),
            # so UNIFORM_BATCH (the level that covers spec decodes) is
            # the correct declaration when spec tokens are configured.
            if vllm_config.num_speculative_tokens > 0:
                return AttentionCGSupport.UNIFORM_BATCH
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        # A3 (revived 2026-09-14 on the V2 bring-up branch): opt in to the
        # upstream fused-draft protocol, whose only consumer is V2's
        # `_generate_fused_drafts` loop. It builds the draft metadata ONCE per
        # propose round instead of rebuilding it on the host per draft step.
        #
        # The in-place update is a NO-OP by construction, so opting in only
        # skips host work. Audit (re-run against 0.29 + V2, 2026-09-14): every
        # step-dependent input this builder hands over is a live view of a
        # persistent buffer — seq_lens is incremented in place by
        # update_draft_inputs (its draft-step kernel) and slot_mapping is
        # rewritten by compute_slot_mappings between steps, both inside the
        # captured loop — and each layer's forward re-derives its live inputs
        # from those buffers (kv_max = seq_lens, q_abs_offset = seq_lens - n_q),
        # never from a step-baked scalar. The scalar fields are step-constant:
        # num_actual_tokens (1 query per request), max_query_len (1), max_seq_len
        # (a per-propose host bound used for buffer CAPACITY only — kernel reads
        # are cut by the live kv_max), use_cascade (False for draft builds). The
        # one field a per-step rebuild would recompute
        # (seq_lens_cpu_upper_bound, CPU) is not read by this builder. Same
        # contract as TritonAttentionMetadataBuilder.
        #
        # Env-gated (default OFF): flipping this changes which spec-decode code
        # path serves, so per the standing same-boot A/B rule it must pass a
        # serving A/B at the serving k before becoming the default.
        self.supports_draft_decode_metadata_update = (
            _os.environ.get("VLLM_GFX906_FUSED_DRAFT", "0") == "1"
        )

    def update_draft_decode_metadata(self, metadata: Gfx906FAMetadata) -> None:
        # No-op by construction — see supports_draft_decode_metadata_update.
        # May be called during CUDA graph capture (the fused loop bakes the
        # call into the graph), so it must stay host-logic-free and launch
        # nothing: there is nothing to update (every kernel input is a live
        # read of the persistent buffers inside each forward).
        return

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> Gfx906FAMetadata:
        # FULL decode capture: metadata is runner-staged at capacity
        # (profile_seq_lens) and re-read live at replay — see
        # get_cudagraph_support (M2).
        return self.build(0, common_attn_metadata)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Gfx906FAMetadata:
        return Gfx906FAMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            query_start_loc_cpu=getattr(
                common_attn_metadata, "query_start_loc_cpu", None
            ),
            use_cascade=(common_prefix_len > 0),
            common_prefix_len=common_prefix_len,
            causal=_batch_causal(common_attn_metadata),
        )


# -----------------------------------------------------------------------------
# Backend
# -----------------------------------------------------------------------------
class Gfx906FABackend(AttentionBackend):
    accept_output_buffer: bool = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
    ]
    # Only fp16 KV is supported (Q8 quantization happens on the fly
    # in the kernel; the cache itself stays fp16).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "half",
    ]

    # KV writes are a separate call (triton_reshape_and_cache_flash)
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        # Must match the name in AttentionBackendEnum so vLLM can
        # resolve it via AttentionBackendEnum[name]; registered as
        # CUSTOM, hence the name.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["Gfx906FAImpl"]:
        return Gfx906FAImpl

    @classmethod
    def customize_spec(cls, spec):
        """Widen the KV spec's head dims to the padded one when padding is opted in.

        NOTE: both halves are set to the same padded width, which is what the impl's split
        and its zero-padding assume; a model with genuinely asymmetric K/V head dims would
        need per-half padding here and in the impl.

        vLLM sizes the KV cache from this spec while the kernels are dispatched on the
        padded dim, so the two must agree. With the real dim in the spec the layer allocates
        a 2*real-byte fused row, and splitting that at the padded dim leaves a remainder
        (Phi-3-mini: a 128 chunk plus a 64 remainder, which trips the Q8 row check).
        """
        if _os.environ.get("GFX906_FA_DEBUG_SHAPES"):
            print(
                f"[gfx906_fa-spec] in: head={spec.head_size} head_v={spec.head_size_v} "
                f"block={spec.block_size} hkv={spec.num_kv_heads}",
                flush=True,
            )
        padded = _padded_head_size(spec.head_size)
        if padded is None or padded == spec.head_size:
            return spec
        # BOTH halves: the 0.29 fused row is K||V, so its width is head_size +
        # head_size_v, and our impl splits it into two padded_head_size halves. Widening
        # only head_size leaves a row of padded + real (Phi-3: 128 + 96 = 224), which is
        # what the allocator then built and what the Q8 row check rejected.
        out = dataclasses.replace(spec, head_size=padded, head_size_v=padded)
        if _os.environ.get("GFX906_FA_DEBUG_SHAPES"):
            print(f"[gfx906_fa-spec] out: head={out.head_size}", flush=True)
        return out

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # Identical to TritonAttentionBackend for instantiated head dims, so backends
        # can be swapped without re-allocating the KV cache. A pad-able dim widens
        # the row instead (opt-in via GFX906_FA_PAD; see _padded_head_size).
        return (
            num_blocks,
            2,
            block_size,
            num_kv_heads,
            _padded_head_size(head_size) or head_size,
        )

    @classmethod
    def supported_kv_cache_layouts(cls):
        # 0.29 (#51718): our kernels read head-major block interiors, i.e. the
        # physical order [L, B, H, N, C] per layer with C = 2*head_size fused
        # K||V (same words cpu_attn declares).
        from vllm.v1.kv_cache_interface import KVCacheLayout

        return (KVCacheLayout.LBHNC,)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["Gfx906FAMetadataBuilder"]:
        return Gfx906FAMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Kernel instantiated for 64/96/128/256. Dims that pad onto one of those are
        # servable too when GFX906_FA_PAD opts in (default ON since the Phi-3 gate).
        return _padded_head_size(head_size) is not None

    @classmethod
    def supports_non_causal(cls) -> bool:
        """Bidirectional batches (`causal=False`) are served since 2026-09-17.

        The kernel's causal clip is gated on `q_abs_offset` and its sliding-window
        mask/clips on `window`; the impl suppresses both for a non-causal batch, which
        yields full bidirectional attention over [0, seq_len). Stage 1: the symmetric
        +/-window of the reference (`_maybe_symmetrize_window`) is NOT implemented yet,
        so a windowed drafter gets the *superset* mask (keys beyond its trained window
        stay visible) -- see DEVLOG-fa-noncausal.md; the drafter's per-position
        acceptance is the guard for that approximation.

        The claim is the same one the reference backends make for this field (a bool
        `causal=False` means "no causal clip"); what is *not* supported is its tensor form
        (per-token causality), which `_batch_causal` warns about and serves as causal --
        the pre-FA-NONCAUSAL behaviour.

        Gate (2026-09-17): the official Muse-Glimmer DFlash assistant, TP=2, k=7, graphs
        ON -- mean acceptance length **3.12/3.18** (the same drafter through ROCM_ATTN
        with the real symmetric window and no graphs measured 2.95), decode **43.1 t/s**
        vs 30.5 eager and 27.1 non-spec. So the superset mask costs this drafter nothing
        and the 41 % is the graph capture CUSTOM buys. `GFX906_FA_NO_NONCAUSAL=1`
        restores the old behaviour (CUSTOM rejected, ROCM_ATTN fallback) for an A/B or a
        rollback.
        """
        return _os.environ.get("GFX906_FA_NO_NONCAUSAL", "0") != "1"

    @classmethod
    def supports_sliding_window(cls) -> bool:
        # Per-row window cutoff in the tile kernel (window arg, Muse
        # Glimmer iRoPE 2048). Gated on q_abs_offset, which
        # forward_paged supplies for every windowed batch (decode
        # included).
        #
        # Validated envelope (tests/kernels/attention/test_gfx906_fa.py):
        # W in {64, 128, >L} and the 2048 serving value, head_dim 128
        # (GQA 16:1) and 256 (GQA 8:1), decode B=1/B=2 and per-row
        # prefill, on BOTH kernel copies (gather fattn-q8.cuh and
        # direct-paged fattn-q8-paged.cuh). W=2048 is covered by the
        # direct-paged clip tests (L=4353/W=2048 unaligned-start
        # bit-identity + the Python-level B=2 window+clip+split-K test) and
        # the e2e gates, not by a gather-kernel unit case. The mask math
        # itself is shape-independent (absolute-position cutoff on k), but
        # an untested head_dim/window/batch combo landing on this backend
        # runs unverified — extend the tests before claiming new shapes.
        #
        # Phase C KV-scan clip scope (perf, not correctness): the clip
        # only fires on the direct-paged decode path (B>=2 dispatch), so
        # a workload mixing B=1 decode (gather path, no clip yet) with
        # B>=2 windowed decode gets the HBM savings on one class of
        # request and the full scan on the other — correct in both cases,
        # but the perf of "windowed decode" here depends on the dispatch
        # path a request happens to hit.
        return True

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # gfx906: major=9, minor=0, patch=6 -> capability.to_int() -> 906.
        # ROCm does not always fill DeviceCapability correctly for gfx906,
        # so accept everything and rely on explicit backend selection.
        return True


# -----------------------------------------------------------------------------
# Impl
# -----------------------------------------------------------------------------
class Gfx906FAImpl(AttentionImpl):
    # ------------------------------------------------------------------
    # CLASS-LEVEL shared gather buffers (K_q8, V_fp16).
    #
    # All attention layers in one worker share these buffers — the kernel
    # reads the cache once per (seq, head, tok) and writes the contiguous
    # out per forward call. Nothing persists between forwards.
    #
    # Shared -> ONE buffer pair per worker, not per layer. That saves
    # N_layers x (K_buf + V_buf) VRAM (e.g. MiniMax: 60 layers x ~24 MiB
    # = 1.4 GB per sequence). Grow logic is the same as q_pad_buf.
    # ------------------------------------------------------------------
    _k_gather_buf: ClassVar[torch.Tensor | None] = None
    _v_gather_buf: ClassVar[torch.Tensor | None] = None
    # Keep-alive for gather-buffer generations whose base VA is baked
    # into a replaying CUDA graph — freeing one is a use-after-free on
    # replay (2026-08-19 init-time Memory Fault). Post-fix
    # (GFX906_FA_GATHER_EXACT=0) only generations that were current
    # during a stream capture are retired (tracked per generation in
    # _gather_buf_captured); eager-serving generations are freed, and
    # since Sk is a grow-only capacity, replacement happens only on
    # Hkv/D/B change — the set normally holds exactly the one
    # capture-time generation. Under EXACT (pre-fix policy) the sticky
    # _gather_captured latch retires on every Sk change — the unbounded
    # growth behind the 256k Qwen3.8-27B TP=2 prefill OOM (see
    # docs/gfx906/oom-256k-prefill.md, plan-gfx906-fa-fix.md §1).
    # Keyed by data_ptr: unique for every live tensor, and a retained
    # tensor is never freed, so its VA can never be reused and no entry
    # can ever be overwritten (a shape key could collide between two
    # same-shape generations, silently freeing a captured one).
    _gather_retired: ClassVar[dict[int, tuple[torch.Tensor, ...]]] = {}
    # Pre-fix sticky latch (EXACT mode only): any generation replaced
    # after the first capture is retired, whether or not it was baked.
    _gather_captured: ClassVar[bool] = False
    # Post-fix per-generation flag: True iff THIS generation's base VA
    # was current during a capture. Reset (not OR'd) at each allocation.
    _gather_buf_captured: ClassVar[bool] = False
    # Kill switch: GFX906_FA_GATHER_EXACT=1 restores the pre-fix
    # exact-match policy at the backend, for A/B. Default: the new
    # grow-only capacity policy. Read once at import here AND as
    # _GATHER_EXACT in gfx906_fa_paged.py — keep the two in sync;
    # flipping only one site produces a split (meaningless) A/B.
    # TEMPORARY A/B arm, NOT a permanent knob (plan-gfx906-fa-fix.md
    # §6): drop at the NEXT gather-lifecycle change — both read sites,
    # the _gather_exact branches in _ensure_gather_buffers, the
    # _GATHER_EXACT branch in forward_paged, and
    # test_gather_exact_killswitch_restores_old_policy — re-gated on
    # a serving A/B. Until then every lifecycle edit must touch BOTH
    # policies and keep them divergence-free.
    _gather_exact: ClassVar[bool] = (
        _os.environ.get("GFX906_FA_GATHER_EXACT", "0") == "1"
    )
    # One-shot warning once more than one capture-baked generation has
    # been retired (capture-order coupling or repeated captures — see
    # _ensure_gather_buffers).
    _gather_retired_warned: ClassVar[bool] = False
    # ------------------------------------------------------------------
    # q_pad buffers: SHARED across all layer impls (ClassVar), one set
    # per worker. Per-impl (instance) buffers were the boot J/K
    # first-prefill OOM: v1 creates one impl per attention layer, so a
    # prefill-sized grow allocated (N_layers x 256 MiB) of duplicate
    # [B, Hq, Sq_pad, D] fp32 buffers (Muse 30B: 52 x 256 MiB = 13.3
    # GiB at bt4096; the same shape is Sq_pad-proportional, which is
    # why the OOM "scaled with the chunk" and bt2048 survived). Same
    # pattern as the gather buffers above (which had the identical bug
    # fixed earlier). See DEVLOG-muse-glimmer.md round 4.
    # ------------------------------------------------------------------
    _q_pad_buf: ClassVar[torch.Tensor | None] = None
    _q_pad_decode_buf: ClassVar[torch.Tensor | None] = None
    # Buffers referenced by captured CUDA graphs: freed-then-realloc
    # would leave the graphs pointing at freed VAs (use-after-free on
    # replay), so retired captured buffers stay alive here.
    _q_pad_retired: ClassVar[list[torch.Tensor]] = []
    _q_pad_captured: ClassVar[bool] = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("GFX906_FA: alibi_slopes unsupported")
        if logits_soft_cap not in (None, 0, 0.0):
            raise NotImplementedError("GFX906_FA: logits_soft_cap unsupported")
        if sinks is not None:
            raise NotImplementedError("GFX906_FA: sinks unsupported")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(f"GFX906_FA: attn_type={attn_type} unsupported")

        self.num_heads = num_heads
        self.head_size = head_size
        # Padding to an instantiated kernel dim (opt-in). Kernels use
        # padded_head_size; the API side (scales, slicing) stays on head_size.
        self.padded_head_size = _padded_head_size(head_size) or head_size
        self._head_pad = self.padded_head_size - head_size
        self._pad_zeroed_cache: tuple | None = None
        # Sliding-window size in tokens (0 for full-attention layers).
        # The FA kernel masks keys older than the per-row window when
        # window > 0; forward_paged passes q_abs_offset for every
        # windowed batch (decode included) so the per-row formula holds.
        # GFX906_FA_NO_WINDOW (truthy = anything except "0", matching the
        # sibling knobs) forces 0 for every layer — perf-only A/B arm that
        # separates the CUSTOM-vs-ROCM_ATTN kernel-family speedup from the
        # window-mask cost (numerically WRONG for windowed layers; warn,
        # like the GFX906_FA_LEGACY precedent).
        if _os.environ.get("GFX906_FA_NO_WINDOW", "0") != "0":
            self.sliding_window = 0
            logger.warning(
                "GFX906_FA_NO_WINDOW is set: sliding-window masking is "
                "disabled on all layers — output is WRONG for windowed "
                "layers beyond the window (perf A/B arm only)."
            )
        else:
            self.sliding_window = sliding_window or 0
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = num_heads // num_kv_heads

        # ------------------------------------------------------------------
        # Q8_0 side buffer for K (LEGACY=0 mode only).
        #
        # In LEGACY=0 mode attention reads K from a block_q8_0 view
        # instead of re-quantizing the fp16 K on every read. The Q8 row
        # ((D/32)*34 bytes) fits inside the fp16 K row (2D bytes), so the
        # "side buffer" is an ALIASED VIEW into the K half of the kv
        # cache itself — see _ensure_q8_sidebuffer for the invariants.
        # ------------------------------------------------------------------
        self._k_cache_q8: torch.Tensor | None = None
        # The fp16 K-cache tensor the alias was derived from; a different
        # tensor (e.g. the profile-run dummy cache vs the live pool) means
        # the alias must be re-derived.
        self._k_cache_q8_src: torch.Tensor | None = None
        self._legacy = _resolve_legacy_mode()

        # ------------------------------------------------------------------
        # q_pad buffers for forward_paged are CLASS-level (see the
        # ClassVar block above): ONE shared set per worker, lazily
        # grown by _ensure_forward_buffers. They must NOT be instance
        # attributes — v1 creates one impl per attention layer, and a
        # per-impl prefill-sized grow is N_layers x 256 MiB of
        # duplicates (the boot J/K first-prefill OOM).
        # ------------------------------------------------------------------
        # Level 3a: mask_buf removed — causal is inlined in the kernel.

    def fused_output_quant_supported(self, quant_key):
        return False

    # ------------------------------------------------------------------
    # KV cache write (separate step, as in the Triton backend with
    # forward_includes_kv_cache_update=False)
    # ------------------------------------------------------------------
    def _ensure_q8_sidebuffer(self, key_cache: torch.Tensor) -> None:
        """Keep the Q8 side view aliased to the live fp16 K cache.

        key_cache shape: [num_blocks, block_size, Hkv, D]  fp16.

        The Q8_0 row ((D/32)*34 bytes, e.g. 136 for D=128) fits inside
        the fp16 K row (2D bytes, e.g. 256), so the side buffer is a
        strided uint8 view of the K half itself:

          * ZERO extra KV memory — in LEGACY=0 mode the fp16 K half is
            written by do_kv_cache_update but never read, so its bytes
            double as Q8 storage. No pool-sizing headroom needed (the
            old separate allocation added ~17% of the KV bytes and OOM'd
            uncapped high-util runs; it also had to be re-derived against
            the profile-run dummy cache geometry).
          * do_kv_cache_update's write order (triton fp16 K write, then
            the Q8 write to the same bytes) leaves the Q8 bytes as the
            final state of every written slot.
          * Block-manager page copies (prefix-cache COW) copy the fp16
            page, Q8 bytes included — the view cannot desync from the
            fp16 cache (the old separate allocation missed COW'd blocks
            and any write that landed on a different cache tensor).
        """
        # Identity is on the MEMORY, not the view object: key_cache is a
        # fresh kv_cache.unbind(1) view on every call, so `is` would
        # re-alias (and clobber) 52x per step. Re-derive only when the
        # underlying storage/layout actually changed (profile-run dummy
        # cache -> live pool, layout re-slices).
        src = self._k_cache_q8_src
        if (
            self._k_cache_q8 is not None
            and src is not None
            and src.data_ptr() == key_cache.data_ptr()
            and src.shape == key_cache.shape
            and src.stride() == key_cache.stride()
        ):
            return
        num_blocks, block_size, Hkv, D = key_cache.shape
        assert D % 32 == 0, f"D={D} must be multiple of 32"
        bytes_per_row = (D // 32) * 34
        row_bytes = D * key_cache.element_size()
        assert bytes_per_row <= row_bytes, (
            f"Q8 row ({bytes_per_row} B) must fit in the K row "
            f"({row_bytes} B at {key_cache.element_size()} B/elt) — "
            "the uint8 slice [:, :, :, :bytes_per_row] would clamp to "
            f"the row width and straddle into the next row"
        )
        # key_cache is kv_cache.unbind(1) of [num_blocks, 2, bs, Hkv, D]
        # — non-contiguous, last dim stride 1, so the uint8 view and the
        # last-dim slice below are legal (verified: strides come from the
        # real tensor and every consumer kernel is stride-parameterized).
        self._k_cache_q8 = key_cache.view(torch.uint8)[:, :, :, :bytes_per_row]
        self._k_cache_q8_src = key_cache
        # NOTE: no zero-fill here, ever. The alias may be re-derived while
        # the cache holds live data (layout re-slices mid-run); zeroing
        # the Q8 region would wipe the in-use context (the Q8 bytes of
        # slots written in earlier steps). Unwritten slots are never read
        # (attention seq_lens only cover written slots), so the fill buys
        # nothing.

    @classmethod
    def _ensure_forward_buffers(
        cls,
        num_heads: int,
        head_size: int,
        num_seqs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Lazy/grow-allocate the q_pad buffers (shared across all
        layer impls — see the ClassVar block). Level 3a: mask_buf
        removed — causality is inlined in the kernel. For a 60K
        prefill this saves ~480 MB of fp16 mask."""
        ncols1 = _pick_ncols1(max_seqlen_q)
        Sq_pad = ((max_seqlen_q + ncols1 - 1) // ncols1) * ncols1

        # q_pad_buf is only READ on multi-token forwards (Sq=1 decode uses
        # q_pad_decode_buf in both the gather and direct-paged branches),
        # so decode batches must not grow its dim0: a capture at B=8 would
        # otherwise balloon every layer's prefill buffer to (8, Hq, 2048,
        # 128) fp32 = 268 MB (Muse 30B: x52 layers = 14 GiB, first-request
        # OOM on the inductor prefill buffer, 2026-08-26).
        #
        # Spec-decode note: a verify step carries num_speculative_tokens+1
        # rows per sequence, so max_seqlen_q > 1 and it DOES read
        # q_pad_buf (forward_paged's Sq=1 fast paths are gated on
        # max_seqlen_q == 1). That is correct, and its Sq_pad is bounded
        # by the spec depth (small: (n_spec+1) up to the ncols1 tile), so
        # the decode-time dim0 growth it triggers stays ~B x 128 KB/layer
        # — the 14 GiB pathology required a prefill-sized Sq_pad, which
        # only real prefill rows produce. If a spec-decode config ever
        # shows q_pad growth again, this comment is where to start.
        qpad_num_seqs = num_seqs if max_seqlen_q > 1 else 1

        # Q buffer: [B, Hq, Sq_pad, D] fp32 (the kernel takes fp32 q; the
        # caller passes dtype=torch.float32).
        # Capture-safety: a CUDA graph bakes in the VA of the buffer that
        # was current when it was captured. Growing by free-then-realloc
        # would leave captured graphs pointing at freed memory on replay,
        # so once any buffer has been current during a capture, retired
        # buffers are kept alive (they are decode-sized; prefill-sized
        # buffers only ever exist eagerly). No empty_cache() here: it is
        # illegal during capture, and the caching allocator already
        # reuses freed blocks for the next allocation.
        #
        # The capture-state poll runs only until the first capture is
        # detected (_q_pad_captured latches); steady state costs nothing.
        if cls._q_pad_buf is None:
            cls._q_pad_buf = torch.empty(
                (qpad_num_seqs, num_heads, Sq_pad, head_size),
                dtype=dtype,
                device=device,
            )
            cls._q_pad_captured = torch.cuda.is_current_stream_capturing()
        elif (
            cls._q_pad_buf.shape[0] < qpad_num_seqs
            or cls._q_pad_buf.shape[2] < Sq_pad
            or cls._q_pad_buf.dtype != dtype
        ):
            capturing = torch.cuda.is_current_stream_capturing()
            cur = cls._q_pad_buf
            new_shape = (
                max(qpad_num_seqs, cur.shape[0]),
                num_heads,
                max(Sq_pad, cur.shape[2]),
                head_size,
            )
            if cls._q_pad_captured or capturing:
                cls._q_pad_retired.append(cur)
            cls._q_pad_buf = torch.empty(new_shape, dtype=dtype, device=device)
            cls._q_pad_captured = cls._q_pad_captured or capturing
        elif not cls._q_pad_captured:
            # No grow: latch the flag the first time we serve a forward
            # during capture (the buffer VA is being baked into a graph).
            cls._q_pad_captured = torch.cuda.is_current_stream_capturing()

        # Sq=1 decode buffer [B, Hq, 2, D] fp32 (Sq_pad=2 for Sq=1).
        # Grows dim0 only; capture-safe via the shared retired list.
        if max_seqlen_q == 1:
            if cls._q_pad_decode_buf is None:
                cls._q_pad_decode_buf = torch.empty(
                    (num_seqs, num_heads, 2, head_size),
                    dtype=torch.float32,
                    device=device,
                )
                cls._q_pad_captured = (
                    cls._q_pad_captured or torch.cuda.is_current_stream_capturing()
                )
            elif cls._q_pad_decode_buf.shape[0] < num_seqs:
                capturing = torch.cuda.is_current_stream_capturing()
                if cls._q_pad_captured or capturing:
                    cls._q_pad_retired.append(cls._q_pad_decode_buf)
                cls._q_pad_decode_buf = torch.empty(
                    (num_seqs, num_heads, 2, head_size),
                    dtype=torch.float32,
                    device=device,
                )
                cls._q_pad_captured = cls._q_pad_captured or capturing

    @classmethod
    # ------------------------------------------------------------------
    # M3 follow-up (co-review F3, 2026-09-12): headroom advisory for the
    # long-context activation transients that live OUTSIDE the profiled
    # pool — the gather buffers below (linear in B x Sk_pad) and the
    # kv_split partial buffer (capped by GFX906_FA_KVSPLIT_MAX_BYTES).
    # Both are allocated at first long-context use, after the KV pool is
    # sized, from the (1 - gpu_memory_utilization) headroom.
    # ------------------------------------------------------------------
    @staticmethod
    def _kv_split_transient_cap(
        num_seqs: int,
        max_seqlen_q: int,
        num_heads: int,
        head_size: int,
        budget_bytes: int,
    ) -> int:
        """Worst-case kv_split partial-buffer transient
        [B, Sq_pad, Hq, y, D] fp32 for the coming step; 0 when the byte
        budget forces y=1 (large-batch prefill). Mirrors the C++
        fa_kv_split_default / fa_apply_kv_split_budget rules via the
        extension's kv_split_default binding (single source of truth for
        the split rule; the budget constant is mirrored here — keep in
        sync with GFX906_FA_KVSPLIT_MAX_BYTES in gfx906_fa.cpp)."""
        if num_seqs <= 0 or max_seqlen_q <= 0:
            return 0
        from vllm import _gfx906_fa_C as gfx906_fa

        try:
            y = gfx906_fa.kv_split_default(max_seqlen_q, num_seqs, False)
        except Exception:
            return 0
        if y <= 1:
            return 0
        # launcher ncols1 table (switch by Sq bucket), then pad to it
        ncols1 = (
            64
            if max_seqlen_q > 32
            else 32
            if max_seqlen_q > 16
            else 16
            if max_seqlen_q > 8
            else 8
            if max_seqlen_q > 4
            else 4
            if max_seqlen_q > 2
            else 2
        )
        sq_pad = ((max_seqlen_q + ncols1 - 1) // ncols1) * ncols1
        t = num_seqs * sq_pad * num_heads * y * head_size * 4
        # Over budget the C++ forces y=1 (fa_apply_kv_split_budget) — the
        # transient disappears entirely rather than being capped.
        return 0 if t > budget_bytes else t

    @staticmethod
    def _headroom_advisory(
        demand_bytes: int, free_bytes: int
    ) -> tuple[str | None, str | None]:
        """(error, warning) for a pending allocation against the device's
        free VRAM. error = deterministic failure; warning = >60% of the
        remaining headroom (spikes may tip it)."""
        if free_bytes <= 0:
            return (
                "GFX906_FA: no free device memory reported before a "
                f"{demand_bytes / 2**30:.2f} GiB long-context "
                "activation allocation.",
                None,
            )
        if demand_bytes > free_bytes:
            return (
                f"GFX906_FA: long-context activation demand "
                f"{demand_bytes / 2**30:.2f} GiB exceeds free device "
                f"memory {free_bytes / 2**30:.2f} GiB - the next "
                "allocation will fail. Levers: reduce "
                "gpu_memory_utilization, max_model_len or batch size; "
                "or lower GFX906_FA_KVSPLIT_MAX_BYTES to drop the "
                "KV-split transient.",
                None,
            )
        if demand_bytes > 0.6 * free_bytes:
            return (
                None,
                f"GFX906_FA: long-context activation demand "
                f"{demand_bytes / 2**30:.2f} GiB is above 60% of free "
                f"device memory {free_bytes / 2**30:.2f} GiB - "
                "prefill transients may OOM at spikes. Levers: reduce "
                "gpu_memory_utilization, max_model_len or batch size.",
            )
        return None, None

    @classmethod
    def _ensure_gather_buffers(
        cls,
        num_seqs: int,
        num_kv_heads: int,
        max_seqlen_k: int,
        head_size: int,
        device: torch.device,
        max_seqlen_q: int = 0,
        num_heads: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """K_q8 + V_fp16 gather buffers, shared by all layers in a worker.

        shape:
            K: [B, Hkv, Sk, (D/32)*34]  uint8
            V: [B, Hkv, Sk, D]          fp16

        Post-fix policy (default; GFX906_FA_GATHER_EXACT=0):
        the Sk dimension is a *capacity*, not an exact shape. Any buffer
        wide enough (>= Sk_pad) is reused. Replacement sizing depends on
        the old generation's fate: one being retired (capture-baked)
        keeps capacity per axis — max(needed, existing), the in-tree
        _q_pad_buf pattern; one being freed (never capture-baked) is
        replaced at exact need, so a freeable generation never inherits
        dead capacity (see the sizing comment at the replacement site).
        FULL capture runs at max_model_len
        (build_for_cudagraph_capture), so after the first capture the
        buffer spans every seq_len the engine can schedule and eager
        chunked prefill reuses it forever — no per-chunk realloc/retire
        churn (the unbounded _gather_retired growth that drained the
        ~2 GiB post-capture headroom in the 256k Qwen3.8-27B TP=2
        prefill; docs/gfx906/oom-256k-prefill.md,
        plan-gfx906-fa-fix.md). Width-wider-than-live is safe: the
        persistent gather's work is live-bounded (device-side seq_lens)
        and the FA kernel cuts each sequence at kv_max, never at Sk.

        A request for a smaller B than the current buffer returns a
        leading-dim slice [:B] — zero-copy, contiguous, and same base VA
        (a non-leading-dim slice would copy, which is why Hkv/D changes
        realloc instead). Because FULL-graph capture bakes the base VA of
        ONE generation into every captured batch size, this slice reuse
        is what keeps a single generation alive across the whole capture
        sweep (up to ~35 batch sizes at max_num_seqs=256).

        Capture-safety: the generation current during a stream capture is
        baked into the graph by VA. _gather_buf_captured tracks that
        *per generation* — reset to the capture state at each allocation,
        latched forward only within a generation's no-realloc path — so
        only graph-baked generations are retired into _gather_retired on
        replacement. Eager-serving generations are never baked (attention
        is a splitting op — vllm::unified_attention_with_output in the
        default splitting_ops — so eager forwards run outside any
        captured subgraph) and are simply freed; the caching allocator
        reuses the blocks. Bounded: post-fix replacement happens only on
        Hkv/D/B change, so the set normally holds exactly one generation.
        (splitting_ops=[] — attention inside piecewise graphs — is still
        safe, at one extra retained generation: such generations latch
        captured and are retired, not freed.)

        Pre-fix policy (GFX906_FA_GATHER_EXACT=1, A/B kill switch):
        exact-Sk match and the sticky _gather_captured latch — the old
        behavior byte-for-byte, unbounded retire growth included.
        """
        Sk_pad = ((max_seqlen_k + 31) // 32) * 32
        bytes_per_row = (head_size // 32) * 34
        capturing = torch.cuda.is_current_stream_capturing()

        cur = cls._k_gather_buf
        if cur is not None:
            fit = (
                cur.shape[1] == num_kv_heads
                and cur.shape[3] == bytes_per_row
                and cur.device == device
                and cur.shape[0] >= num_seqs
                and (
                    cur.shape[2] == Sk_pad
                    if cls._gather_exact
                    else cur.shape[2] >= Sk_pad
                )
            )
            if fit:
                # Reuse; latch the capture flag within THIS generation's
                # lifetime only (the buffer VA may be getting baked into
                # a graph right now).
                if cls._gather_exact:
                    if not cls._gather_captured:
                        cls._gather_captured = capturing
                else:
                    if not cls._gather_buf_captured:
                        cls._gather_buf_captured = capturing
                if cur.shape[0] == num_seqs:
                    # Exact size: no view (the hot decode path calls this
                    # per FA layer per step; a fresh TensorImpl per call
                    # is measurable on short steps).
                    return cur, cls._v_gather_buf
                # A leading-dim slice is contiguous and has the same base
                # VA and row layout as the parent, so the kernel and every
                # graph captured against this base see identical addresses.
                return cur[:num_seqs], cls._v_gather_buf[:num_seqs]

        # First allocation, or a growth/shape change: retire-or-free the
        # old generation, then allocate.
        new_b = num_seqs
        new_sk = Sk_pad
        if cur is not None:
            keep = (
                (cls._gather_captured or capturing)
                if cls._gather_exact
                else cls._gather_buf_captured
            )
            if keep:
                # Graph-baked VA: must outlive every replay.
                cls._gather_retired[cur.data_ptr()] = (cur, cls._v_gather_buf)
                if (
                    not cls._gather_exact
                    and not cls._gather_retired_warned
                    and len(cls._gather_retired) > 1
                ):
                    # Post-fix, ONE capture-baked generation is the norm
                    # (the FULL-capture sweep reuses a single base VA
                    # across every captured batch size). A second retire
                    # means either the sweep itself retired generations —
                    # capture-order coupling (plan §2.2b; capture-time OOM
                    # possible) — or repeated re-captures / Hkv-D flaps.
                    logger.warning(
                        "GFX906_FA: %d retired capture-baked gather "
                        "generations (expected <= 1; capture-order "
                        "coupling — see plan-gfx906-fa-fix.md §2.2b)",
                        len(cls._gather_retired),
                    )
                    cls._gather_retired_warned = True
                if not cls._gather_exact:
                    # The retired generation's block stays resident
                    # anyway, so keeping its capacity costs nothing: a
                    # FULL capture spans B=max_num_seqs x
                    # Sk=max_model_len and later needs never exceed it.
                    new_b = max(num_seqs, cur.shape[0])
                    new_sk = max(Sk_pad, cur.shape[2])
            # else, capacity mode + never capture-baked: the replacement
            # allocates at exact need (new_b/new_sk as initialized).
            # Grow-only max() on BOTH axes here would pin the
            # B-highwater x Sk-highwater product forever — 32-way
            # short-context decode then one 250k prefill leaves a
            # [32, 262144] standing buffer (~13 GB/rank at the arm-B
            # geometry) that no single request ever needed. Reallocation
            # frequency is unchanged (a replacement happens exactly when
            # the current buffer no longer fits, and this block is freed
            # either way); only the new allocation stops inheriting dead
            # capacity. In FULL modes this branch runs only before the
            # first capture (post-capture generations are capture-baked
            # and take the keep path), so capture-time sizing — the
            # validated arm-B path — is unaffected.
        # Headroom advisory (log once per demand signature): the fresh
        # gather buffers + the kv_split transient cap both draw from the
        # post-pool headroom; mem_get_info right here already reflects any
        # retired generation.
        import os as _os

        demand = (
            new_b * num_kv_heads * new_sk * bytes_per_row
            + new_b * num_kv_heads * new_sk * head_size * 2
            + cls._kv_split_transient_cap(
                num_seqs,
                max_seqlen_q,
                num_heads,
                head_size,
                int(
                    _os.environ.get(
                        "GFX906_FA_KVSPLIT_MAX_BYTES", str(512 * 1024 * 1024)
                    )
                ),
            )
        )
        _free = torch.cuda.mem_get_info(device)[0]
        _err, _warn = cls._headroom_advisory(demand, _free)
        if _err:
            logger.error("%s", _err)
        elif _warn:
            logger.warning("%s", _warn)
        cls._k_gather_buf = torch.empty(
            (new_b, num_kv_heads, new_sk, bytes_per_row),
            dtype=torch.uint8,
            device=device,
        )
        cls._v_gather_buf = torch.empty(
            (new_b, num_kv_heads, new_sk, head_size),
            dtype=torch.float16,
            device=device,
        )
        if cls._gather_exact:
            cls._gather_captured = cls._gather_captured or capturing
        else:
            # Per generation: the new buffer starts from its own state,
            # not the old generation's capture history.
            cls._gather_buf_captured = capturing
        b, v = cls._k_gather_buf, cls._v_gather_buf
        if b.shape[0] == num_seqs:
            return b, v
        return b[:num_seqs], v[:num_seqs]

    def _pad_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Zero-pad the head dim of [.., D] (no-op when unpadded).

        Exact for attention: padded Q adds 0 to the QK dot, padded K quantises to
        zero q8_0 blocks, padded V adds 0 to P*V, and the pad is inside the head
        dim so the softmax denominator is unchanged (as the ViT path documents).
        """
        if self._head_pad == 0:
            return tensor
        shape = (*tensor.shape[:-1], self.padded_head_size)
        out = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
        out[..., : self.head_size] = tensor
        return out

    def _zero_cache_pad(
        self, key_cache: torch.Tensor, value_cache: torch.Tensor
    ) -> None:
        """Zero a padded cache's pad channels once per cache tensor.

        triton_reshape_and_cache_flash writes only the first D channels of a
        padded row, so the pad would otherwise keep whatever the allocator left
        there - and a non-zero K pad quantises to a non-zero q8_0 block,
        corrupting scores. Nothing else writes those bytes, so one zeroing per
        cache tensor suffices (keyed on identity: the cache is re-viewed every
        call, so `is` would redo it 52x per step).
        """
        if self._head_pad == 0:
            return
        key = (key_cache.data_ptr(), tuple(key_cache.shape))
        if self._pad_zeroed_cache == key:
            return
        with torch.no_grad():
            key_cache[..., self.head_size :].zero_()
            value_cache[..., self.head_size :].zero_()
        self._pad_zeroed_cache = key

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        # 0.29 KV-cache layout standardisation (#51718): the cache is a single
        # tensor with a fused content axis, [B, H, N, 2*D] per layer, so K/V
        # are the two halves of the last axis (not a separate axis-1 pair).
        key_cache, value_cache = kv_cache.transpose(1, 2).split(
            self.padded_head_size, dim=-1
        )
        if self._head_pad:
            self._zero_cache_pad(key_cache, value_cache)
            key = self._pad_last_dim(key)
            value = self._pad_last_dim(value)

        # 1) Primary fp16 write — the vLLM-standard path for V (and for K
        #    in LEGACY mode).
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        # 2) In parallel into the Q8 side view for K (fast-path
        #    forward). Skipped in LEGACY mode — forward quantizes on the
        #    fly.
        # Invariant (LEGACY=0): the Q8 view aliases the fp16 K half, so
        # this call's write order (triton fp16 K write above, then the Q8
        # write) leaves the Q8 bytes as the final state of every written
        # slot — the same memory, nothing to keep in sync. Writes that
        # bypass do_kv_cache_update (warmup/profile forwards) go through
        # it too (verified: they reach this method), and block-manager
        # page copies (prefix-cache COW) move the Q8 bytes with the
        # page. The Q8 bytes are NEVER read for a slot whose fp16 K has
        # not been written in this same call or an earlier one (attention
        # seq_lens only cover written slots).
        if not self._legacy:
            self._ensure_q8_sidebuffer(key_cache)
            from vllm import _gfx906_fa_C as gfx906_fa

            if slot_mapping.dtype != torch.int64:
                slot_mapping = slot_mapping.to(torch.int64)
            gfx906_fa.reshape_and_cache_q8(
                key.contiguous() if not key.is_contiguous() else key,
                slot_mapping,
                self._k_cache_q8,
            )

    def fused_rope_kvcache_supported(self):
        return False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        # (already in kv_cache via do_kv_cache_update)
        value: torch.Tensor,
        kv_cache: torch.Tensor,  # [num_blocks, 2, block_size, num_kv_heads, head_size]
        attn_metadata: Gfx906FAMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("GFX906_FA: output quantization unsupported")

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert not attn_metadata.use_cascade, "GFX906_FA: cascade unsupported"

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Split the fused content axis: (B, H, N, 2*D) → (K, V) each
        # [num_blocks, block_size, Hkv, D] (0.29 layout standardisation).
        key_cache, value_cache = kv_cache.transpose(1, 2).split(
            self.padded_head_size, dim=-1
        )
        if _os.environ.get("GFX906_FA_DEBUG_SHAPES") and _DEBUG_SHAPES[0] < 3:
            _DEBUG_SHAPES[0] += 1
            print(
                f"[gfx906_fa-impl] kv_cache={tuple(kv_cache.shape)} "
                f"head={self.head_size} padded={self.padded_head_size} "
                f"K={tuple(key_cache.shape)} V={tuple(value_cache.shape)}",
                flush=True,
            )

        # query [num_tokens, Hq, D] fp16 (forward_paged casts it into the
        # fp32 q_pad buffer inside the copy_ — a standalone .float() was
        # an extra kernel per layer).
        q_actual = query[:num_actual_tokens]
        if self._head_pad:
            q_actual = self._pad_last_dim(q_actual)
        out_actual = output[:num_actual_tokens]

        # Lazy-grow the forward buffers (q_pad / mask).
        num_seqs = attn_metadata.seq_lens.shape[0]
        self._ensure_forward_buffers(
            num_heads=self.num_heads,
            head_size=self.padded_head_size,
            num_seqs=num_seqs,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            device=query.device,
            dtype=torch.float32,
        )

        # Fused-gather output buffers (class-level, shared across layers).
        # Used by both the LEGACY=0 fused Q8 gather and the LEGACY=1
        # fused gather+quantize; without reuse every attention layer
        # allocates 24-200+ MiB per step on long contexts.
        k_gather_buf, v_gather_buf = self._ensure_gather_buffers(
            num_seqs=num_seqs,
            num_kv_heads=self.num_kv_heads,
            max_seqlen_k=attn_metadata.max_seq_len,
            head_size=self.head_size,
            device=query.device,
            max_seqlen_q=attn_metadata.max_query_len,
            num_heads=self.num_heads,
        )

        # forward_paged returns [num_tokens, Hq*D] float32.
        # Fast path: pass the Q8 side-buffer when present.
        out_flat = forward_paged(
            query=q_actual,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_q_host=attn_metadata.query_start_loc_cpu,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            scale=self.scale,
            key_cache_q8=self._k_cache_q8 if not self._legacy else None,
            q_pad_buf=self._q_pad_buf,
            q_pad_decode_buf=self._q_pad_decode_buf,
            k_gather_buf=k_gather_buf,
            v_gather_buf=v_gather_buf,
            window=self.sliding_window,
            non_causal=not attn_metadata.causal,
        )  # [num_tokens, Hq*D] fp32

        if self._head_pad:
            out_flat = (
                out_flat.view(num_actual_tokens, self.num_heads, self.padded_head_size)[
                    ..., : self.head_size
                ]
                .contiguous()
                .view(num_actual_tokens, -1)
            )

        # Write the result into output in-place (it is either
        # [num_tokens, Hq, D] or [num_tokens, Hq*D], depending on the
        # caller). copy_ fuses the fp32->fp16 cast (a .to() first was
        # an extra kernel per layer).
        out_view = out_actual.view(num_actual_tokens, -1)
        out_view.copy_(out_flat)

        return output


# -----------------------------------------------------------------------------
# Auto-register as CUSTOM on import
# -----------------------------------------------------------------------------
def register() -> None:
    """Register Gfx906FABackend as AttentionBackendEnum.CUSTOM.

    Called automatically on module import (and available to user code).
    No-op off gfx906: the extension only exists in gfx906 builds, and
    registering the backend elsewhere would make a broken backend
    selectable via VLLM_ATTENTION_BACKEND=CUSTOM.
    """
    from vllm.platforms import current_platform
    from vllm.platforms.rocm import on_gfx906

    if not (current_platform.is_rocm() and on_gfx906()):
        return
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        f"{__name__}.Gfx906FABackend",
    )
    logger.info("GFX906_FA backend registered as AttentionBackendEnum.CUSTOM")


# Auto-register on module import
register()
