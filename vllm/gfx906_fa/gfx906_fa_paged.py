# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) Nick — nick413@gmail.com
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
#
# Vendored from https://github.com/cassettesgoboom/gfx906-fa-vllm
# (FlashAttention-style custom attention backend for vLLM on AMD gfx906).
#
"""Paged KV cache wrapper for gfx906_fa.

Two paths:
  * fast-path: key_cache_q8 given -> read the already-quantized K directly,
    no quantization. Used when the backend holds the Q8 side-buffer.
  * legacy-path: key_cache_q8 = None -> gather fp16 K -> quantize_q8_0
    (device). Slower (extra gather + quantize per step), but correct.

API-level wrapper: builds the inputs from the vLLM paged layout into
contiguous tensors and calls gfx906_fa.forward().

Eager-only debug hooks (host-device syncs; illegal during CUDA graph
capture and they serialize eager execution — do not enable in serving):
  * GFX906_FA_DEBUG=1        — master switch: enables ALL debug hooks
                               below at once
  * GFX906_FA_FWD_DEBUG=1    — sync + per-call log to /tmp/gfx906_fa_debug/
  * GFX906_FA_DOUBLE_CHECK=1 — cross-check the gather vs torch (.item())
  * GFX906_FA_DUMP=DIR       — dump inputs/outputs via torch.save
                               (default dir /tmp/gfx906_fa_debug)
"""
from __future__ import annotations

import math
import os as _os
import time as _time

import torch

try:
    from vllm import _gfx906_fa_C as gfx906_fa
except ImportError:
    # Non-gfx906 platform: the extension is absent. This module is safe to
    # import (register() is a no-op off gfx906), the wrappers just cannot
    # run.
    gfx906_fa = None

# Master debug switch (R12): enables every off-by-default debug hook in
# this module at once; the individual knobs remain for finer control.
_FA_DEBUG = _os.environ.get("GFX906_FA_DEBUG", "0") == "1"
_DBG   = (_os.environ.get("GFX906_FA_FWD_DEBUG", "0") == "1" or _FA_DEBUG)
# Level 1: по умолчанию используем fused gather kernel. Отключить можно
# через GFX906_FA_FUSED=0 — тогда работает старый путь через fancy-indexing
# (для A/B-замеров и как быстрый safety-fallback при регрессиях).
_FUSED = _os.environ.get("GFX906_FA_FUSED", "1") != "0"
_DEBUG_PRINTS = [0]  # GFX906_FA_DEBUG_SHAPES prints the first few calls
# M3 (2026-09-12): use the host-side cu_seqlens (query_start_loc_cpu) in the
# variable-Q branches instead of int(cu[...]) device reads. Each int() is
# a D2H sync that blocks on the queued GPU work (~10.8 syncs per layer
# call in mixed steps = the post-FIX-H2 multi-batch residual, trace
# 2026-09-12 08:23). Kill-switch for A/B.
_HOST_QSEQ = _os.environ.get("GFX906_FA_HOST_QSEQ", "1") != "0"
# Level 3c: direct-paged FA — FA kernel читает K/V напрямую из paged cache
# через block_table indirection, без промежуточного gather.
#
# Режимы (GFX906_FA_DIRECT_PAGED):
#   "0"    → всегда gather+FA (baseline, заведомо корректный путь).
#   "1"    → всегда direct-paged (для A/B-замеров и benchmarking).
#   "auto" → (default) адаптивный выбор по {batch, max_seqlen_q}:
#            1) B < GFX906_FA_DIRECT_PAGED_MIN_BATCH   → gather (single-user decode).
#            2) max_seqlen_q > GFX906_FA_DIRECT_PAGED_MAX_SQ → gather
#               (длинный prefill, ncols1=64 — direct paged spill'ит VGPR и проигрывает).
#            3) иначе                                  → direct.
#
# Обоснование auto-default (MI50 / gfx906, bench_ab2.py + bench_prefill.py):
#   Decode (Sq=1, ncols1=2):
#     * B=1: gather быстрее на ~3-6% (compact access, direct добавляет
#       block_table indirection).
#     * B=2: direct быстрее на ~4-7%.
#     * B≥3: direct быстрее на 7-35%.
#     * B=8, Sk=61K: gather → CUDA OOM (24 GiB peak); direct работает (~13 ms/step).
#       [Pre-2026-08-20 observation, before the gather buffers became
#       ClassVar-shared (one pair/worker, not one per layer) and the
#       gather-lifecycle grow-only-capacity fix (21c69a8ead). Today
#       this shape is ~0.2 GiB, one-time, shared with LEGACY=1's
#       gather (already exercised at B=8 in the N=8 concurrent-decode
#       records) — the 24 GiB figure no longer applies.]
#   Prefill (bench_prefill.py, occupancy-fix применён):
#     * B=1 Sq=16 (ncols1=16): direct WIN -5..-27% (0 spill).
#     * B=2 Sq=32 (ncols1=32): direct LOSS +13% даже при 0 spill
#       (block_table lookup latency).
#     * B=2 Sq=64 (ncols1=64): direct LOSS +34% (197 spill остался, 1 wave/EU).
#     * B=4 Sq=32/64: direct LOSS +0.5..+27%.
# → threshold {min_batch=2, max_sq=16} закрывает:
#     - регрессию B=1 (gather),
#     - регрессию ncols1=32/64 prefill (gather),
#     - включает direct для decode multi-batch (Sq=1) и короткого chunked
#       prefill (Sq≤16),
#     - спасает от OOM на длинном Sk (mode=1 явный override).
_DIRECT_PAGED_MODE = _os.environ.get("GFX906_FA_DIRECT_PAGED", "auto").lower()
_DIRECT_PAGED_MIN_BATCH = int(_os.environ.get("GFX906_FA_DIRECT_PAGED_MIN_BATCH", "2"))
_DIRECT_PAGED_MAX_SQ = int(_os.environ.get("GFX906_FA_DIRECT_PAGED_MAX_SQ", "16"))
# M6 Part B (roadmap-more-models.md M6, plan_fa_legacy0_impr_claude.md):
# the direct-paged kernel reads the LEGACY=0 Q8-aliased K as misaligned
# 136-of-256-B slices plus per-row page indirection. 0 (DEFAULT since
# the 2026-08-28 serving A/B, DEVLOG-muse-glimmer round 10) routes
# LEGACY=0 B>=2 batches to the fused-Q8 gather path (the LEGACY=0 B=1
# path; the M1 gather clip is dispatch-agnostic, so the window-clip
# benefit is retained) instead of direct-paged: B=4 @2k ngram spec
# serving went 35.7 -> 46.3 t/s (parity with the 46.7 LEGACY=1
# control) while B=1 and prefill were unchanged. =1 is the opt-in
# experiment route (it also lost the M5 bake -27…-31% at B=4 @2k;
# its in-process Sq=1 A/B was a wash — no measured advantage).
_DIRECT_PAGED_Q8 = _os.environ.get("GFX906_FA_DIRECT_PAGED_Q8", "0") != "0"
# Phase C window-clip kill switch (A/B arms only): 0 disables the per-row
# kv_start so windowed decode scans the FULL history and the window MASK
# does all the work (numerically identical, slower at long context).
_WINDOW_CLIP = _os.environ.get("GFX906_FA_WINDOW_CLIP", "1") != "0"
# M1 gather-path window clip: with the persistent (legacy-path) gather,
# gather only [start, seq_len) per seq and shift the FA kernel's k-loop
# start (fattn-q8.cuh) instead of masking the skipped history. 0 = old
# full gather + full scan (numerically identical, slower at long ctx).
# Only the persistent sub-path clips; the other gather sub-paths pass
# kv_start=None and keep the full-scan semantics.
_GATHER_CLIP = _os.environ.get("GFX906_FA_GATHER_CLIP", "1") != "0"
# Diagnostics for the LEGACY=0 corruption hunt (P3-3).
_ZERO_KTAIL = (_os.environ.get("GFX906_FA_ZERO_KTAIL", "0") == "1"
               or _FA_DEBUG)
_NO_BUF_REUSE = (_os.environ.get("GFX906_FA_NO_BUF_REUSE", "0") == "1"
                 or _FA_DEBUG)
_DOUBLE_CHECK = (_os.environ.get("GFX906_FA_DOUBLE_CHECK", "0") == "1"
                 or _FA_DEBUG)
# LEGACY-path gather: fused HIP kernel (default) vs torch fancy-index path
# (A/B switch; the torch path is 128-190 us/layer at Sk~2176-3328 vs ~40 us
# for the fused fp16 gather).
_TORCH_GATHER = (_os.environ.get("GFX906_FA_TORCH_GATHER", "0") == "1"
                 or _FA_DEBUG)
# Stage 2: fuse the legacy-path gather + quantize into one kernel
# (bit-equal to gather_paged_kv_fp16 + quantize_q8_0). Dispatch
# precedence in the legacy branch: PERSIST > FUSED_QUANT > two-kernel —
# with _PERSISTENT on (default), GFX906_FA_FUSED_QUANT=0 is IGNORED and
# the two-kernel path is only reached via GFX906_FA_PERSIST=0 or
# num_seqs > _PERSIST_MAX_SEQS.
_FUSED_QUANT = _os.environ.get("GFX906_FA_FUSED_QUANT", "1") != "0"
# Persistent grid-stride fused gather+quantize (plan_masked_fa.md §2.2):
# fixed capture-time grid, work bounded by the live seq_lens tensor —
# valid at every Sk (replaces the two-kernel > 65535 fallback and the
# per-token fused kernel). K output is bit-equal to the old paths;
# gates in DEVLOG-masked-fa.md (NaN-tail, capture/replay B=1..4, PPL,
# TP=2 serving A/B: 15.9->40.9 t/s at 262k). Default ON; GFX906_FA_PERSIST=0
# is the kill switch.
_PERSISTENT = _os.environ.get("GFX906_FA_PERSIST", "1") != "0"
# Kernel bound: the persistent gather prefix-sums per-seq counts in a
# fixed 16-entry register array (csrc launcher rejects num_seqs > 16 with
# an error, which would crash engine start for any default max_num_seqs).
# Batches above the bound fall back to the fused/two-kernel paths (old
# behavior, still Sk-bounded) instead of dispatching to the persistent
# kernel.
_PERSIST_MAX_SEQS = 16
# Post-fix gather-buffer policy (plan-gfx906-fa-fix.md §2.2c): the
# persistent branch reuses any buffer with width >= Sk_pad and passes
# the buffer's own width as Sk — safe only there (live-bounded work,
# FA cuts at kv_max). Every other call site keeps the exact contract:
# their tail zeroing is width-bound work and their launchers enforce
# Sk <= 65535, so a wide buffer must not reach them (a mismatch there
# would also silently fall back to a per-layer C++ allocation).
# GFX906_FA_GATHER_EXACT=1 restores the pre-fix exact-match policy at
# every site (A/B kill switch). Read once at import here AND as
# Gfx906FAImpl._gather_exact in gfx906_fa_backend.py — keep the two in
# sync; flipping only one site produces a split (meaningless) A/B.
# TEMPORARY A/B arm, NOT a permanent knob (plan-gfx906-fa-fix.md §6):
# drop at the NEXT gather-lifecycle change — both read sites, the
# _gather_exact branches in _ensure_gather_buffers, the _GATHER_EXACT
# branch below (plus the k_exact/v_exact derivation feeding it), and
# test_gather_exact_killswitch_restores_old_policy — re-gated on a
# serving A/B. Until then every lifecycle edit must touch BOTH
# policies and keep them divergence-free.
_GATHER_EXACT = _os.environ.get("GFX906_FA_GATHER_EXACT", "0") == "1"
# P3-4: skip the LEGACY-path q_pad zero_ on the Sq=1 decode fast path
# (pad rows are per-row-independent and discarded; the q8_0 quantization
# clamps NaN/Inf garbage). GFX906_FA_QPAD_EMPTY=0 reverts to the zero_.
_QPAD_EMPTY = _os.environ.get("GFX906_FA_QPAD_EMPTY", "1") != "0"
# Under the master switch the dump default dir is the standard debug dir.
_DUMP_DIR = _os.environ.get(
    "GFX906_FA_DUMP",
    "/tmp/gfx906_fa_debug" if _FA_DEBUG else "",
)
_dump_n = 0

def _pick_ncols1(seq_q: int) -> int:
    """ncols1 (Q-tile columns) ladder — llama.cpp's
    launch_fattn_tile_q8_switch_ncols1.

    MUST stay in sync with the C++ mirror in
    csrc/gfx906_fa/gfx906_fa.cpp (fa_pick_ncols1).
    """
    if seq_q > 32:
        return 64
    if seq_q > 16:
        return 32
    if seq_q > 8:
        return 16
    if seq_q > 4:
        return 8
    if seq_q > 2:
        return 4
    return 2

def _should_use_direct_paged(num_seqs: int, max_seqlen_q: int) -> bool:
    """Решает, использовать ли direct-paged FA для текущего batch/seq_q."""
    if _DIRECT_PAGED_MODE == "0":
        return False
    if _DIRECT_PAGED_MODE == "1":
        return True
    if num_seqs < _DIRECT_PAGED_MIN_BATCH:
        return False
    return not max_seqlen_q > _DIRECT_PAGED_MAX_SQ

def _fwdlog(msg: str) -> None:
    if not _DBG:
        return
    try:
        pid = _os.getpid()
        with open(f"/tmp/gfx906_fa_debug/fwd-{pid}.log", "a") as f:
            f.write(f"[{_time.time():.3f}] {msg}\n")
    except Exception:
        pass


def _gather_kv(
    key_cache: torch.Tensor,        # [num_blocks, block_size, Hkv, D]  fp16
    value_cache: torch.Tensor,      # [num_blocks, block_size, Hkv, D]  fp16
    block_table: torch.Tensor,      # [num_seqs, max_blocks]            int32
    seq_lens: torch.Tensor,         # [num_seqs]                        int32
    max_seqlen_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K, V from the paged cache into a contiguous layout.

    Returns:
      K: [B, Hkv, max_seqlen_k, D]  fp16
      V: [B, Hkv, max_seqlen_k, D]  fp16

    Token positions beyond seq_lens[i] are zero-filled so the FA kernel's
    kv_max cut sees them as out-of-bounds.

    This torch fancy-indexing implementation is the A/B reference; the
    default path is the fused HIP kernels (gather_paged_kv_fp16 /
    gather_paged_kv_q8).
    """
    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
    num_seqs = block_table.shape[0]

    assert key_cache.shape == value_cache.shape, "K/V shape mismatch"
    assert block_table.dtype in (torch.int32, torch.int64), \
        f"block_table must be int, got {block_table.dtype}"
    assert seq_lens.shape == (num_seqs,), \
        f"seq_lens shape {seq_lens.shape} vs num_seqs={num_seqs}"

    # Blocks needed per sequence (ceiling of seqlen / block_size)
    max_blocks_needed = (max_seqlen_k + block_size - 1) // block_size
    assert block_table.shape[1] >= max_blocks_needed, \
        f"block_table columns {block_table.shape[1]} < {max_blocks_needed}"

    bt = block_table[:, :max_blocks_needed].to(torch.long)  # [B, n_blocks]

    # Fancy indexing: key_cache[bt] -> [B, n_blocks, block_size, Hkv, D],
    # then reshape to [B, n_blocks*block_size, Hkv, D]
    k_gathered = key_cache[bt]    # fp16
    v_gathered = value_cache[bt]

    # Flatten block-dim: → [B, n_blocks*block_size, Hkv, D]
    k_gathered = k_gathered.view(num_seqs, -1, num_kv_heads, head_size)
    v_gathered = v_gathered.view(num_seqs, -1, num_kv_heads, head_size)

    # Обрезать до max_seqlen_k
    k_gathered = k_gathered[:, :max_seqlen_k].contiguous()
    v_gathered = v_gathered[:, :max_seqlen_k].contiguous()

    # Маскировать «хвост» за пределами seq_lens[i] нулями.
    # Создаём маску положения: [1, max_seqlen_k] → broadcast до [B, max_seqlen_k]
    positions = torch.arange(max_seqlen_k, device=seq_lens.device, dtype=seq_lens.dtype)
    mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)           # [B, Sk]
    mask_f = mask.view(num_seqs, max_seqlen_k, 1, 1).to(k_gathered.dtype)

    k_gathered = k_gathered * mask_f
    v_gathered = v_gathered * mask_f

    # Переставляем в [B, Hkv, Sk, D] — именно этот layout ждёт наш FA kernel
    k_bhsd = k_gathered.permute(0, 2, 1, 3).contiguous()
    v_bhsd = v_gathered.permute(0, 2, 1, 3).contiguous()

    return k_bhsd, v_bhsd


def _gather_kv_q8(
    key_cache_q8: torch.Tensor,    # [num_blocks, block_size, Hkv, (D/32)*34]  uint8
    value_cache:  torch.Tensor,    # [num_blocks, block_size, Hkv, D]           fp16
    block_table:  torch.Tensor,    # [num_seqs, max_blocks]                     int
    seq_lens:     torch.Tensor,    # [num_seqs]                                 int
    max_seqlen_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast-path gather: K уже квантован в side-buffer, V — fp16.

    Возвращает:
      K_q8 : [B, Hkv, Sk, (D/32)*34]  uint8
      V    : [B, Hkv, Sk, D]           fp16

    Позиции за seq_lens[i] в V обнуляются. K — оставляем как есть
    (kernel отсекает по KV_max, доп. маска не нужна).
    """
    num_blocks, block_size, Hkv, bytes_per_row = key_cache_q8.shape
    _, _, _, head_size = value_cache.shape
    num_seqs = block_table.shape[0]
    assert value_cache.shape[0] == num_blocks and value_cache.shape[1] == block_size, \
        "key_cache_q8 / value_cache layout mismatch"

    max_blocks_needed = (max_seqlen_k + block_size - 1) // block_size
    bt = block_table[:, :max_blocks_needed].to(torch.long)

    # Fancy indexing: [B, n_blocks, bs, Hkv, bytes]
    k_gathered = key_cache_q8[bt]              # uint8
    v_gathered = value_cache[bt]               # fp16

    # Flatten block-dim: [B, n_blocks*bs, Hkv, ...]
    k_gathered = k_gathered.view(num_seqs, -1, Hkv, bytes_per_row)
    v_gathered = v_gathered.view(num_seqs, -1, Hkv, head_size)

    k_gathered = k_gathered[:, :max_seqlen_k].contiguous()
    v_gathered = v_gathered[:, :max_seqlen_k].contiguous()

    # V-маска tail: обнулим позиции за seq_lens (для безопасности).
    positions = torch.arange(max_seqlen_k, device=seq_lens.device, dtype=seq_lens.dtype)
    mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)   # [B, Sk]
    mask_f = mask.view(num_seqs, max_seqlen_k, 1, 1).to(v_gathered.dtype)
    v_gathered = v_gathered * mask_f
    # K_q8 — хвост мусор, но kernel отсекает через KV_max_d, так что OK.

    # Permute → [B, Hkv, Sk, ...]
    k_bhsd = k_gathered.permute(0, 2, 1, 3).contiguous()
    v_bhsd = v_gathered.permute(0, 2, 1, 3).contiguous()
    return k_bhsd, v_bhsd


def _gather_clip_start(seq_lens: torch.Tensor,
                       cu_seqlens_q: torch.Tensor,
                       window: int,
                       num_seqs: int) -> torch.Tensor:
    """M1: per-seq window-clip start for the gather path.

    Conservative (safe) start: the FIRST query row's window start,
    kv_start[s] = max(0, seq_lens[s] - n_q[s] + 1 - window). Later rows'
    windows are subsets, so clipping the gather at this start never
    skips a key any row needs. Returns int32 [num_seqs].

    LOCKSTEP: single source of truth for the formula, used by both the
    fused Q8 branch and the persistent fp16 branch of forward_paged;
    the gather kernels re-derive the actual skip start with
    GATHER_CLIP_MARGIN (see gfx906_fa_gather.cu).
    """
    sl_i64 = (seq_lens.to(torch.int64)
              if seq_lens.dtype != torch.int64 else seq_lens)
    cu_i64 = (cu_seqlens_q.to(torch.int64)
              if cu_seqlens_q.dtype != torch.int64 else cu_seqlens_q)
    n_q_per_seq = cu_i64[1:num_seqs + 1] - cu_i64[:num_seqs]
    q_abs = sl_i64 - n_q_per_seq
    return ((q_abs + (1 - window)).clamp_(min=0)
            .to(torch.int32).contiguous())


def forward_paged(
    query: torch.Tensor,            # [num_tokens, Hq, D] fp16 (cast into fp32 q_pad)
    key_cache: torch.Tensor,        # [num_blocks, block_size, Hkv, D]  fp16
    value_cache: torch.Tensor,      # [num_blocks, block_size, Hkv, D]  fp16
    block_table: torch.Tensor,      # [num_seqs, max_blocks]            int32
    seq_lens: torch.Tensor,         # [num_seqs]                        int32
    cu_seqlens_q: torch.Tensor,     # [num_seqs+1]                      int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float | None = None,
    key_cache_q8: torch.Tensor | None = None,
    # fast-path: [num_blocks,bs,Hkv,(D/32)*34]
    q_pad_buf: torch.Tensor | None = None,
    q_pad_decode_buf: torch.Tensor | None = None,  # [B,Hq,2,D] fp32, Sq=1 only
    k_gather_buf: torch.Tensor | None = None,  # [B,Hkv,Sk_pad,bytes_per_row] uint8
    v_gather_buf: torch.Tensor | None = None,  # [B,Hkv,Sk_pad,D]             fp16
    window: int = 0,  # sliding-window size in tokens (0 = off)
    non_causal: bool = False,  # bidirectional batch (spec-decode drafter)
    cu_seqlens_q_host: torch.Tensor | None = None,  # [B+1] CPU int — avoids
    # the per-seq int(cu[...]) D2H syncs in the variable-Q branches (M3).
) -> torch.Tensor:
    """vLLM-совместимый paged-attention forward.

    Вход / выход в flat layout num_tokens (как в vLLM forward signature).
    Внутри собирает KV из paged layout в BHSD, квантует K → Q8_0
    и вызывает gfx906_fa.forward().

    query может быть fp16: копирование в fp32 q_pad буфер делает cast
    внутри copy_ (без отдельного .float() ядра).

    Возвращает out: [num_tokens, Hq*D] fp32 (для совместимости с vLLM).
    """
    num_tokens, Hq, D = query.shape
    if _os.environ.get("GFX906_FA_DEBUG_SHAPES") and _DEBUG_PRINTS[0] < 3:
        _DEBUG_PRINTS[0] += 1
        print(
            f"[gfx906_fa] key_cache={tuple(key_cache.shape)} "
            f"value_cache={tuple(value_cache.shape)} query={tuple(query.shape)} "
            f"k_q8={None if key_cache_q8 is None else tuple(key_cache_q8.shape)} "
            f"D={D} q_pad={None if q_pad_buf is None else tuple(q_pad_buf.shape)}",
            flush=True,
        )
    assert query.dtype in (torch.float16, torch.float32), (
        "query must be fp16/fp32 for FA-q8 path (cast into the fp32 "
        "q_pad buffer at store)")
    num_seqs = block_table.shape[0]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # -------- Non-causal (bidirectional) batches --------
    # Spec-decode drafters whose layers are built with causal=False (DFlash
    # assistants: all-sliding, bidirectional) attend in both directions. Two
    # mechanisms implement causality here and both are driven by `window` and
    # `q_abs_offset`: the inline causal clip in the kernel (q_abs_row) and the
    # sliding-window mask/clips (kv_start, _gather_clip_start). Zeroing `window`
    # and suppressing the offset below therefore turns BOTH off in every
    # dispatch branch, which yields full bidirectional attention over
    # [0, seq_len) for every query row.
    #
    # Stage 1 of DEVLOG-fa-noncausal.md: the reference semantics
    # (_maybe_symmetrize_window: causal (w,0) window -> symmetric (w,w) when
    # non-causal) is NOT implemented yet, so this mask is the *superset* of the
    # drafter's windowed mask -- keys beyond the trained window are visible. The
    # guard for that approximation is the drafter's own per-position acceptance
    # (MUSE-2 measured 2.95 through the ROCM_ATTN fallback, which does implement
    # the symmetric window); if acceptance drops, Stage 2 adds the symmetric
    # window to the kernel.
    if non_causal:
        window = 0

    # -------- Sq / Sk padding --------
    # Sq_pad должен быть кратен ncols1 — размеру tile колонки в kernel.
    # launcher выбирает ncols1 по seq_q = Sq_pad (launcher.hip switch_ncols1):
    #   Sq>32→64,  Sq>16→32,  Sq>8→16,  Sq>4→8,  Sq>2→4,  Sq≤2→2.
    # Ранее Sq_pad всегда округлялся до 64, из-за чего на decode (Sq=1)
    # kernel прогонял 64 query tile'а вместо 1 — лишняя работа.
    # Важное условие: pad только если это НЕ prefill (нет causal-маски в kernel
    # бьёт по pad-позициям некорректно при ncols1<64 на реальных данных).
    # Для prefill всегда используем ncols1=64 (как до фикса).
    ncols1 = _pick_ncols1(max_seqlen_q)
    Sq_pad = ((max_seqlen_q + ncols1 - 1) // ncols1) * ncols1
    Sk_pad = ((max_seqlen_k + 31) // 32) * 32

    # -------- Level 3c: Direct-paged FA (обходит gather полностью) --------
    # Работает только когда:
    #   * key_cache_q8 передан (есть side-buffer с Q8_0)
    #   * _DIRECT_PAGED_Q8 (M6 Part B: 0 → LEGACY=0 batches go through
    #     the fused-Q8 gather below instead)
    #   * block_size == 16 (захардкожено в kernel)
    #   * _should_use_direct_paged(num_seqs, max_seqlen_q):
    #       mode=1 или (mode=auto AND B≥min_batch AND Sq≤max_sq)
    #
    # В этом режиме НЕ делаем gather/pad для K/V; kernel сам читает paged
    # cache по block_table. Только Q padding и q_abs_offset готовятся здесь.
    if (key_cache_q8 is not None
            and _DIRECT_PAGED_Q8
            and _should_use_direct_paged(num_seqs, max_seqlen_q)
            and key_cache_q8.dim() == 4
            and key_cache_q8.shape[1] == 16):
        bt_i32 = (block_table if block_table.dtype == torch.int32
                 else block_table.to(torch.int32))
        sl_i32 = (seq_lens if seq_lens.dtype == torch.int32
                 else seq_lens.to(torch.int32))
        bt_i32 = bt_i32.contiguous()
        sl_i32 = sl_i32.contiguous()

        # forward_paged_direct requires fp32 Q (the kernel operates in
        # fp32); the q_pad buffers are always allocated fp32 by the
        # backend, and the row stores below cast an fp16 query in-place —
        # the same buffer pattern as the legacy/gather branch below
        # (Sq=1 uses the dedicated decode buffer: the grown q_pad_buf
        # slice would .contiguous() copy on every FA layer).
        if (max_seqlen_q == 1
                and q_pad_decode_buf is not None
                and q_pad_decode_buf.shape[0] >= num_seqs
                and q_pad_decode_buf.shape[1] == Hq
                and q_pad_decode_buf.shape[2] == Sq_pad
                and q_pad_decode_buf.shape[3] == D
                and q_pad_decode_buf.dtype == torch.float32):
            q_padded = q_pad_decode_buf[:num_seqs]
            q_padded.zero_()
        elif (q_pad_buf is not None
                and q_pad_buf.shape[0] >= num_seqs
                and q_pad_buf.shape[1] >= Hq
                and q_pad_buf.shape[2] >= Sq_pad
                and q_pad_buf.shape[3] == D
                and q_pad_buf.dtype == torch.float32):
            q_padded = q_pad_buf[:num_seqs, :Hq, :Sq_pad, :].contiguous()
            q_padded.zero_()
        else:
            q_padded = torch.zeros(
                (num_seqs, Hq, Sq_pad, D),
                dtype=torch.float32, device=query.device
            )

        if max_seqlen_q == 1 and num_tokens == num_seqs:
            q_padded[:, :, :1, :] = query.unsqueeze(2)
        elif num_tokens == num_seqs * max_seqlen_q:
            # Uniform multi-token batch (spec decode: n_q = 1 + num_spec).
            # Host-integer check — no D2H sync (the int(cu[...]) loop below
            # is illegal during cudagraph capture; spec-decode FULL decode
            # captures reach this branch with Sq = 1 + num_spec).
            n_q = max_seqlen_q
            q_padded[:, :, :n_q, :] = (
                query.view(num_seqs, n_q, Hq, D).permute(0, 2, 1, 3))
        else:
            if _HOST_QSEQ and cu_seqlens_q_host is not None:
                cu = cu_seqlens_q_host.to(torch.long).tolist()
            else:
                cu = cu_seqlens_q.to(torch.long).tolist()
            for s in range(num_seqs):
                n = cu[s + 1] - cu[s]
                if n > 0:
                    q_seq = query[cu[s]:cu[s] + n]
                    q_padded[s, :, :n, :] = q_seq.permute(1, 0, 2)

        # Inline causal (same as gather path). window > 0 needs the offset on
        # decode batches too (the causal check is a no-op there; the
        # per-row window cutoff is not). Suppressed for non-causal batches:
        # without the offset the kernel applies no causal clip at all.
        need_causal = (not non_causal) and (max_seqlen_q > 1 or window > 0)
        q_abs_offset_tensor = None
        if need_causal:
            sl_i64 = sl_i32.to(torch.int64)
            cu_i64 = (cu_seqlens_q.to(torch.int64)
                      if cu_seqlens_q.dtype != torch.int64 else cu_seqlens_q)
            n_q_per_seq = cu_i64[1:num_seqs + 1] - cu_i64[:num_seqs]
            q_abs_offset_tensor = (sl_i64 - n_q_per_seq).to(torch.int32).contiguous()

        # Phase C: sliding-window KV clip. A row only attends to
        # [max(0, t-W+1), t), so the kernel k-loop starts at the
        # conservative chunk start kv_start = max(0, chunk_first + 1 - W)
        # instead of scanning (and masking) the prefix. Bit-identical
        # output (the prefix keys are window-masked to -INF anyway, and
        # the kernel floors the clip start to the KV tile boundary so the
        # fp16 reduction order matches the full scan); the HBM reads drop
        # to ~W/L of the full scan.
        # M2: prefill chunks included — the kernel's per-q-tile raise
        # (GFX906_FA_TILE_CLIP, default on) moves each q-tile's scan start
        # to its own floored window start, so the conservative chunk
        # start is safe at every Sq (per-row windows [max(0, t-W+1), t]
        # are supersets of the tile start's window). Before M2 this gate
        # was decode-only (max_seqlen_q == 1) because the per-sequence
        # scan start left later q-tiles scanning their masked prefix.
        # This Phase C clip is specific to the direct-paged dispatch
        # (LEGACY=0-only, and opt-in via GFX906_FA_DIRECT_PAGED_Q8=1
        # since round 10); the gather paths — the LEGACY=1 default, and
        # LEGACY=0 B>=2 since round 10 — clip via _GATHER_CLIP /
        # _gather_clip_start below, at every batch size (they pass
        # kv_start for prefill too, so the M2 kernel raise applies
        # there as well).
        kv_start_tensor = None
        if _WINDOW_CLIP and window > 0:
            # kv_start = max(0, q_abs + 1 - window); int32 throughout
            # (q_abs is bounded by the context length).
            kv_start_tensor = (q_abs_offset_tensor + (1 - window)).clamp_(
                min=0)
            kv_start_tensor = kv_start_tensor.contiguous()

        if _DBG:
            _fwdlog(f"forward_paged DIRECT_PAGED: num_tokens={num_tokens} Hq={Hq} "
                    f"D={D} num_seqs={num_seqs} Sq_max={max_seqlen_q} "
                    f"Sk_max={max_seqlen_k} q_padded={tuple(q_padded.shape)} "
                    f"causal={'inline' if need_causal else 'none'} "
                    f"kv_clip={'on' if kv_start_tensor is not None else 'off'}")
            torch.cuda.synchronize()

        try:
            out_padded = gfx906_fa.forward_paged_direct(
                q_padded,
                key_cache_q8,   # [num_blocks, 16, Hkv, (D/32)*34] uint8
                value_cache,    # [num_blocks, 16, Hkv, D] fp16
                bt_i32, sl_i32,
                float(scale),
                None,                     # mask
                q_abs_offset_tensor,      # inline causal
                window=window,
                kv_start=kv_start_tensor,
            )
            if _DBG:
                torch.cuda.synchronize()
                _fwdlog(f"forward_paged DIRECT_PAGED OK: out={tuple(out_padded.shape)}")
        except Exception as e:
            if _DBG:
                _fwdlog(f"forward_paged DIRECT_PAGED FAILED: {e!r}")
            raise

        # C returns native BSHD [B, Sq, Hq, D]; the Sq=0 row
        # [:, 0, :, :] is a contiguous [B, Hq, D] view -> zero copies.
        if max_seqlen_q == 1 and num_tokens == num_seqs:
            return out_padded[:, 0, :, :].reshape(num_tokens, Hq * D)

        if num_tokens == num_seqs * max_seqlen_q:
            # Uniform multi-token batch (spec decode) — vectorized unpad,
            # no D2H sync (the int(cu[...]) loop below would abort
            # cudagraph capture).
            n_q = max_seqlen_q
            return out_padded[:, :n_q, :, :].permute(0, 2, 1, 3).reshape(
                num_tokens, Hq * D)

        if _HOST_QSEQ and cu_seqlens_q_host is not None:
            cu = cu_seqlens_q_host.to(torch.long).tolist()
        else:
            cu = cu_seqlens_q.to(torch.long).tolist()
        out_flat = torch.empty(
            (num_tokens, Hq * D), dtype=torch.float32, device=query.device)
        for s in range(num_seqs):
            n = cu[s + 1] - cu[s]
            if n > 0:
                # BSHD: [:n] is a contiguous [n, Hq, D] -> plain view/reshape.
                out_flat[cu[s]:cu[s] + n] = out_padded[s, :n, :, :].reshape(n, Hq * D)
        return out_flat

    # -------- gather KV + (возможно) quantize K --------
    # pre-allocated буферы (если подходят по ТОЧНОМУ shape) — zero-copy reuse.
    # Это критично на длинных контекстах: без этого каждая attention layer
    # аллоцирует 24-200+ MiB в HBM → peak VRAM spike → OOM.
    bytes_per_row_expected = (D // 32) * 34
    hkv_k = key_cache_q8.shape[2] if key_cache_q8 is not None \
        else key_cache.shape[2]
    def _buf_fit(t, dtype, nhead, brow):
        return (t is not None
                and t.dtype == dtype
                and t.dim() == 4
                and t.shape[0] == num_seqs
                and t.shape[1] == nhead
                and t.shape[3] == brow
                and t.is_contiguous())

    # Capacity (>= Sk_pad) selection — the post-fix persistent-branch
    # contract. The pre-fix exact (== Sk_pad) selection is DERIVED from
    # it below, so the two width comparisons cannot drift apart.
    k_cap = k_gather_buf if (
        not _NO_BUF_REUSE
        and _buf_fit(k_gather_buf, torch.uint8, hkv_k,
                     bytes_per_row_expected)
        and k_gather_buf.shape[2] >= Sk_pad
    ) else None
    v_cap = v_gather_buf if (
        not _NO_BUF_REUSE
        and _buf_fit(v_gather_buf, torch.float16, value_cache.shape[2], D)
        and v_gather_buf.shape[2] >= Sk_pad
    ) else None
    # Exact-Sk selection — the pre-fix contract, still the ONLY contract
    # for the non-persistent call sites (fused q8 / fused quantized /
    # fp16 two-kernel): they must keep seeing kbuf=None on a wide buffer
    # exactly as they would with no buffer at all (no reuse, no silent
    # divergence — plan §2.2c).
    k_exact = (k_cap if k_cap is not None and k_cap.shape[2] == Sk_pad
               else None)
    v_exact = (v_cap if v_cap is not None and v_cap.shape[2] == Sk_pad
               else None)
    if _GATHER_EXACT:
        # Pre-fix policy: exact match at every call site, logical Sk.
        kbuf, vbuf, Sk_arg = k_exact, v_exact, Sk_pad
    else:
        # Post-fix: capacity (>= Sk_pad) reuse on the persistent branch
        # only, with the k/v decisions coupled (a mixed state — k reused,
        # v fresh per layer — would leak per-layer allocations).
        if (k_cap is not None and v_cap is not None
                and k_cap.shape[2] == v_cap.shape[2]):
            # Sk = the buffer's own width: the persistent kernel's work
            # is live-bounded and the C++ exact-match passes trivially.
            # The k/v widths must match: _ensure_gather_buffers always
            # allocates the pair at one width, but a hand-set class
            # buffer with unequal widths would otherwise pass Sk = K's
            # width and silently drop V to a per-call C++ allocation —
            # exactly the mixed state the coupling exists to prevent.
            kbuf, vbuf, Sk_arg = k_cap, v_cap, k_cap.shape[2]
        else:
            kbuf, vbuf, Sk_arg = None, None, Sk_pad
    # M1: per-seq clip start for the gather path (None unless the
    # persistent sub-path activates the clip; passed to both the gather
    # kernel and the FA kernel below).
    kv_start_tensor = None
    if key_cache_q8 is not None and _FUSED:
        # Level 1 fused path: gather K_q8 + V_fp16 одним HIP kernel'ом.
        # Возвращает tensors с Sk=Sk_pad (хвост в V уже обнулён, K — мусор).
        bt_i32 = (block_table if block_table.dtype == torch.int32
                 else block_table.to(torch.int32))
        sl_i32 = (seq_lens if seq_lens.dtype == torch.int32
                 else seq_lens.to(torch.int32))
        bt_i32 = bt_i32.contiguous()
        sl_i32 = sl_i32.contiguous()
        if _GATHER_CLIP and window > 0:
            # M1 (LOCKSTEP with the persistent branch below): per-seq
            # clip start — the fused Q8 gather skips tokens [0, start)
            # and the FA k-loop starts at floor(start); the gather
            # margin guarantees the floored start is materialized.
            kv_start_tensor = _gather_clip_start(
                seq_lens, cu_seqlens_q, window, num_seqs)
        K_q8, V_bhsd = gfx906_fa.gather_paged_kv_q8(
            key_cache_q8, value_cache, bt_i32, sl_i32, Sk_pad,
            k_out=k_exact, v_out=v_exact,
            kv_start=kv_start_tensor,
        )
        # K_q8: [B, Hkv, Sk_pad, bytes]; V_bhsd: [B, Hkv, Sk_pad, D] — уже padded.
        if _DOUBLE_CHECK and kv_start_tensor is None:
            # Clipped gather leaves rows [0, start) stale BY DESIGN (the
            # FA k-loop never reaches them) — the full-range torch
            # comparison would false-fail (same gate as the persistent
            # branch; the partial-range check stays deferred, review F3).
            k_ref, v_ref = _gather_kv_q8(
                key_cache_q8, value_cache, block_table, seq_lens, max_seqlen_k)
            ke = torch.equal(k_ref, K_q8[:, :, :max_seqlen_k])
            ve = torch.equal(v_ref, V_bhsd[:, :, :max_seqlen_k])
            vn = bool(torch.isnan(V_bhsd.float()).any().item())
            print(f"[FA-DC] fused==torch: K={ke} V={ve} V_nan={vn} "
                  f"B={num_seqs} Sk_pad={Sk_pad} sl={seq_lens.tolist()[:4]}",
                  flush=True)
        if _ZERO_KTAIL:
            sl64 = sl_i32.to(torch.int64)
            pos = torch.arange(Sk_pad, device=sl_i32.device)
            m = (pos.unsqueeze(0) >= sl64.unsqueeze(1)).view(num_seqs, 1, Sk_pad, 1)
            K_q8 = K_q8.masked_fill(m, 0)
    elif key_cache_q8 is not None:
        # Fast-path (старый): K уже квантован в side-buffer, но gather через torch.
        K_q8, V_bhsd = _gather_kv_q8(
            key_cache_q8, value_cache, block_table, seq_lens, max_seqlen_k
        )
    else:
        # Legacy-path: gather FP16 → quantize on the fly.
        # Stage 2 (default): one fused kernel — V fp16 gather + K gathered
        # and quantized to q8_0 in-kernel (bit-equal to the two-kernel
        # sequence; same quantization helper). Stage 1 fallback:
        # gather_paged_kv_fp16 + quantize_q8_0 (GFX906_FA_FUSED_QUANT=0).
        # Both: V tail zeroed; K tail unmasked — FA kernel cuts it via
        # kv_max. GFX906_FA_TORCH_GATHER=1 reverts to the torch path.
        #
        # INVARIANT (LEGACY=0 aliasing, see gfx906_fa_backend._ensure_q8_
        # sidebuffer): this branch reads raw fp16 `key_cache` and MUST stay
        # unreachable whenever the caller is in LEGACY=0 mode. LEGACY=0's
        # Q8 buffer is a strided view aliased into the fp16 K cache's own
        # bytes (not a separate allocation) — only bytes [0, bytes_per_row)
        # of each row hold valid Q8 data; a raw fp16 read of that same
        # memory under LEGACY=0 would silently reinterpret those bytes and
        # corrupt every row's leading K values with no error raised (the
        # old prefix-caching fail-closed that would have caught a related
        # desync was removed when the alias made that specific desync
        # structurally impossible; this branch is what stayed unguarded).
        # The backend enforces this by only ever passing a non-None
        # key_cache_q8 in LEGACY=0 (do_kv_cache_update calls
        # _ensure_q8_sidebuffer, forward passes self._k_cache_q8, both
        # gated on `not self._legacy`) — if that coupling ever breaks, this
        # assert is the last line of defense.
        assert key_cache_q8 is None, (
            "forward_paged: reached the raw-fp16 K gather branch with "
            "key_cache_q8 set — this must never happen under "
            "GFX906_FA_LEGACY=0 (the Q8 side view is aliased into the "
            "fp16 K cache; a raw fp16 read here would silently corrupt "
            "the first bytes of every K row). Check the LEGACY dispatch "
            "in Gfx906FAImpl.forward/do_kv_cache_update.")
        if _TORCH_GATHER:
            K_bhsd, V_bhsd = _gather_kv(
                key_cache, value_cache, block_table, seq_lens, max_seqlen_k
            )
            K_q8 = gfx906_fa.quantize_q8_0(K_bhsd)
        else:
            bt_i32 = (block_table if block_table.dtype == torch.int32
                      else block_table.to(torch.int32)).contiguous()
            sl_i32 = (seq_lens if seq_lens.dtype == torch.int32
                      else seq_lens.to(torch.int32)).contiguous()
            if _PERSISTENT and num_seqs <= _PERSIST_MAX_SEQS:
                # One kernel at every Sk: fixed grid, live-bounded work,
                # in-kernel quantize (bit-equal K), V tail rows not written
                # (FA cuts at kv_max; margin zeros per
                # GFX906_FA_PERSIST_MARGIN). num_seqs > _PERSIST_MAX_SEQS
                # falls through to the fused/two-kernel paths below.
                if _GATHER_CLIP and window > 0:
                    # M1: conservative per-seq clip start (see
                    # _gather_clip_start; LOCKSTEP with the fused Q8
                    # branch above).
                    kv_start_tensor = _gather_clip_start(
                        seq_lens, cu_seqlens_q, window, num_seqs)
                K_q8, V_bhsd = gfx906_fa.gather_paged_kv_quant_persistent(
                    key_cache, value_cache, bt_i32, sl_i32, Sk_arg,
                    k_out=kbuf, v_out=vbuf,
                    kv_start=kv_start_tensor,
                )
                if _DOUBLE_CHECK and kv_start_tensor is None:
                    # Clipped gather leaves rows [0, start) stale BY
                    # DESIGN (the FA k-loop never reaches them) — the
                    # full-range torch comparison would false-fail.

                    # Torch reference, per-seq in-range rows only (the
                    # persistent kernel does not write rows >= seq_len,
                    # unlike the fused/two-kernel paths which zero the V
                    # tail — that difference is gated separately by the
                    # NaN-tail probe, not by this check).
                    k_ref, v_ref = _gather_kv(
                        key_cache, value_cache, block_table, seq_lens,
                        max_seqlen_k)
                    kq_ref = gfx906_fa.quantize_q8_0(k_ref)
                    bad = [
                        s for s in range(num_seqs)
                        if not (torch.equal(
                                    K_q8[s, :, :int(seq_lens[s])],
                                    kq_ref[s, :, :int(seq_lens[s])])
                                and torch.equal(
                                    V_bhsd[s, :, :int(seq_lens[s])],
                                    v_ref[s, :, :int(seq_lens[s])]))
                    ]
                    print(f"[FA-DC] persistent==torch (in-range): "
                          f"{'OK' if not bad else f'MISMATCH seqs {bad}'} "
                          f"B={num_seqs} Sk_pad={Sk_pad} "
                          f"sl={seq_lens.tolist()[:4]}", flush=True)
                    if bad:
                        raise RuntimeError(
                            f"persistent gather mismatch vs torch: {bad}")
            elif _FUSED_QUANT and Sk_pad <= 65535:
                K_q8, V_bhsd = gfx906_fa.gather_paged_kv_quantized(
                    key_cache, value_cache, bt_i32, sl_i32, Sk_pad,
                    k_out=k_exact, v_out=v_exact,
                )
            else:
                # Fallback (GFX906_FA_FUSED_QUANT=0): K output is fp16 [B,Hkv,Sk,D],
                # so the uint8 q8 kbuf does not match; only V reuses the class
                # buffer.
                K_bhsd, V_bhsd = gfx906_fa.gather_paged_kv_fp16(
                    key_cache, value_cache, bt_i32, sl_i32, Sk_pad,
                    v_out=v_exact,
                )
                K_q8 = gfx906_fa.quantize_q8_0(K_bhsd)

    # Переиспользуем buffer если подходит по размеру; иначе создаём новый.
    # Sq=1 decode fast path: pad rows are never read by consumers (the
    # kernel computes per-row independently and Python keeps row 0 only;
    # NaN/Inf garbage is clamped inside the q8_0 quantization), so the
    # zero fill is skipped (GFX906_FA_QPAD_EMPTY, P3-4). PREFILL and
    # multi-token decode keep the zero_ (their pad rows feed
    # causal-masked computation).
    _decode_fast = (
        _QPAD_EMPTY and max_seqlen_q == 1 and num_tokens == num_seqs)
    # Sq=1 decode: dedicated exact-shape buffer; [:num_seqs] is a
    # leading-dim prefix slice -> contiguous, zero copies (the grown
    # q_pad_buf slice below would copy after a prefill-sized grow).
    if (max_seqlen_q == 1
            and q_pad_decode_buf is not None
            and q_pad_decode_buf.shape[0] >= num_seqs
            and q_pad_decode_buf.shape[1] == Hq
            and q_pad_decode_buf.shape[2] == Sq_pad
            and q_pad_decode_buf.shape[3] == D
            and q_pad_decode_buf.dtype == torch.float32):
        q_padded = q_pad_decode_buf[:num_seqs]
        if not _decode_fast:
            q_padded.zero_()
    elif (q_pad_buf is not None
            and q_pad_buf.shape[0] >= num_seqs
            and q_pad_buf.shape[1] >= Hq
            and q_pad_buf.shape[2] >= Sq_pad
            and q_pad_buf.shape[3] == D
            and q_pad_buf.dtype == torch.float32):
        q_padded = q_pad_buf[:num_seqs, :Hq, :Sq_pad, :].contiguous()
        if not _decode_fast:
            q_padded.zero_()
    else:
        q_padded = (
            torch.empty(
                (num_seqs, Hq, Sq_pad, D),
                dtype=torch.float32, device=query.device
            ) if _decode_fast else torch.zeros(
                (num_seqs, Hq, Sq_pad, D),
                dtype=torch.float32, device=query.device
            )
        )

    # Сложить Q в [B, Hq, Sq_pad, D] (copy_ делает fp16->fp32 cast).
    # При Sq=1 (decode) — максимально частый случай: не гоняем Python-цикл
    # если все sequences имеют Sq=1 и num_tokens == num_seqs.
    if max_seqlen_q == 1 and num_tokens == num_seqs:
        # query: [num_seqs, Hq, D] → [num_seqs, Hq, 1, D] → паддинг по Sq_pad
        q_padded[:, :, :1, :] = query.unsqueeze(2)
    elif num_tokens == num_seqs * max_seqlen_q:
        # Uniform multi-token batch (spec decode: n_q = 1 + num_spec).
        # num_tokens == num_seqs * max_seqlen_q (host ints) implies every
        # sequence has exactly max_seqlen_q queries, so no D2H sync into
        # cu_seqlens_q is needed — the int(cu[...]) loop below is illegal
        # during cudagraph capture.
        n_q = max_seqlen_q
        q_padded[:, :, :n_q, :] = (
            query.view(num_seqs, n_q, Hq, D).permute(0, 2, 1, 3))
    else:
        if _HOST_QSEQ and cu_seqlens_q_host is not None:
            cu = cu_seqlens_q_host.to(torch.long).tolist()
        else:
            cu = cu_seqlens_q.to(torch.long).tolist()
        for s in range(num_seqs):
            n = cu[s + 1] - cu[s]
            if n > 0:
                q_seq = query[cu[s]:cu[s] + n]
                q_padded[s, :, :n, :] = q_seq.permute(1, 0, 2)

    # -------- Causal + kv_max --------
    # Level 3a: материализованная fp16 mask [B, Sq_pad, Sk_pad] заменена на
    # inline causal в kernel через q_abs_offset[B]. Для контекста 60K с Sq=4096
    # mask занимал бы ~480 MB — недопустимо на 32GB MI50.
    #
    # q_abs_offset[s] = seq_lens[s] - n_q[s] — абс. позиция query-chunk в
    # sequence. Kernel считает: k_pos > (q_abs_offset[s] + col_Q_0 + j) → -INF.
    # window > 0: kernel also masks k < (q_abs_offset[s] + col_Q_0 + j)
    # - window + 1; decode rows (n_q=1) need the offset too, hence the
    # `or window > 0` below.
    kv_max_tensor = seq_lens.to(torch.int32).contiguous()

    need_causal = (not non_causal) and (max_seqlen_q > 1 or window > 0)
    q_abs_offset_tensor = None
    if need_causal:
        sl_i64 = seq_lens.to(torch.int64) if seq_lens.dtype != torch.int64 else seq_lens
        cu_i64 = (cu_seqlens_q.to(torch.int64)
                  if cu_seqlens_q.dtype != torch.int64 else cu_seqlens_q)
        n_q_per_seq = cu_i64[1:num_seqs + 1] - cu_i64[:num_seqs]
        # shape [B]. Для padded row (j >= n_q[s]) не используется — kernel таких
        # колонок не вызывает (k_VKQ_max=seq_lens[s], + col_Q_0+j > seq_len в
        # padding всё равно обрезано через Sq_pad и bounds-check в quantize_Q).
        q_abs_offset_tensor = (sl_i64 - n_q_per_seq).to(torch.int32).contiguous()

    if _DBG:
        _path = ("FUSED" if (key_cache_q8 is not None and _FUSED)
                 else "FAST" if key_cache_q8 is not None
                 else "LEGACY")
        _fwdlog(f"forward_paged pre: path={_path} "
                f"num_tokens={num_tokens} Hq={Hq} D={D} num_seqs={num_seqs} "
                f"Sq_max={max_seqlen_q} Sk_max={max_seqlen_k} "
                f"Sq_pad={Sq_pad} Sk_pad={Sk_pad} "
                f"q_padded={tuple(q_padded.shape)} "
                f"K_q8={tuple(K_q8.shape)} V={tuple(V_bhsd.shape)} "
                f"causal={'inline' if need_causal else 'none'} scale={scale}")
        torch.cuda.synchronize()
    try:
        out_padded = gfx906_fa.forward(
            q_padded, K_q8, V_bhsd, float(scale),
            kv_max=kv_max_tensor,
            mask=None,
            q_abs_offset=q_abs_offset_tensor,
            window=window,
            kv_start=kv_start_tensor,
        )
        global _dump_n
        if _DUMP_DIR and _dump_n < 40:
            import torch as _t
            _t.save({
                "n": _dump_n, "q": q_padded.detach().clone(),
                "k_q8": K_q8.detach().clone(), "v": V_bhsd.detach().clone(),
                "kv_max": kv_max_tensor.detach().clone(),
                "q_abs": (q_abs_offset_tensor.detach().clone()
                          if q_abs_offset_tensor is not None else None),
                "scale": float(scale), "seq_lens": seq_lens.detach().clone(),
                "block_table": block_table.detach().clone(),
                "out": out_padded.detach().clone(),
            }, f"{_DUMP_DIR}/fwd_{_dump_n:04d}.pt")
            _dump_n += 1
        if _DBG:
            torch.cuda.synchronize()
            _fwdlog(f"forward_paged OK: out={tuple(out_padded.shape)}")
    except Exception as e:
        if _DBG:
            _fwdlog(f"forward_paged FAILED: {e!r}")
        raise

    # -------- распаковать обратно в flat [num_tokens, Hq*D] --------
    # C возвращает нативный BSHD [B, Sq, Hq, D]; на Sq=1 decode fast path
    # строка Sq=0 ([:, 0, :, :]) — contiguous [B, Hq, D] view, без копий.
    if max_seqlen_q == 1 and num_tokens == num_seqs:
        return out_padded[:, 0, :, :].reshape(num_tokens, Hq * D)

    if num_tokens == num_seqs * max_seqlen_q:
        # Uniform multi-token batch (spec decode) — capture-safe gather.
        n_q = max_seqlen_q
        return out_padded[:, :n_q].reshape(num_tokens, Hq * D)

    if _HOST_QSEQ and cu_seqlens_q_host is not None:
        cu = cu_seqlens_q_host.to(torch.long).tolist()
    else:
        cu = cu_seqlens_q.to(torch.long).tolist()
    out_flat = torch.empty(
        (num_tokens, Hq * D), dtype=torch.float32, device=query.device)
    for s in range(num_seqs):
        n = cu[s + 1] - cu[s]
        if n > 0:
            # BSHD: [:n] is a contiguous [n, Hq, D] -> plain view/reshape.
            out_flat[cu[s]:cu[s] + n] = out_padded[s, :n, :, :].reshape(n, Hq * D)

    return out_flat
