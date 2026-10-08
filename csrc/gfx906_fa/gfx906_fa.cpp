// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) Nick — nick413@gmail.com
// SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
//
// Vendored from https://github.com/cassettesgoboom/gfx906-fa-vllm
// (FlashAttention-style custom attention backend for vLLM on AMD gfx906).
//
// gfx906_fa.cpp — pybind11 extension module для gfx906 FlashAttention Q8_0
//
// Python-доступный API:
//   gfx906_fa_forward(q, k_q8, v_fp16, scale) -> out
//     q:      fp32  [B, Hq, Sq, D]    (contiguous)
//     k_q8:   uint8 [B, Hkv, Skv, D*34/32] — pre-quantized q8_0 flat bytes
//              (planar row — docs/gfx906/plan_fa_part_A.md)
//     v_fp16: fp16  [B, Hkv, Skv, D]  (contiguous)
//     out:    fp32  [B, Sq, Hq, D]    (native BSHD — the kernel's store
//                                       layout; no transpose copy)
//
// Квантизация K (из fp16 в q8_0, planar row — plan_fa_part_A.md) делается
// отдельной утилитой.

#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdlib>
#include <climits>

// ncols1 (Q-tile columns) ladder, llama.cpp's
// launch_fattn_tile_q8_switch_ncols1. MUST stay in sync with the mirror in
// vllm/gfx906_fa/gfx906_fa_paged.py (_pick_ncols1): the Python side pads Sq
// to a multiple of the ladder's choice, and the C++ wrappers derive grid_x
// from the same seq_q.
static inline int fa_pick_ncols1(int seq_q) {
    if (seq_q > 32) return 64;
    if (seq_q > 16) return 32;
    if (seq_q > 8) return 16;
    if (seq_q > 4) return 8;
    if (seq_q > 2) return 4;
    return 2;
}

// Launcher объявлен в gfx906_fa_launcher.cu
extern "C" hipError_t gfx906_fa_launch(
    const float *  Q_fp32,
    const void  *  K_q8,
    const __half * V_f16,
    float *        O_fp32,
    float2 *       O_meta,
    const int *    KV_max_d,
    const __half * MASK_f16,
    int32_t        mask_seq_kv_padded,
    // Level 3a: если non-null, kernel использует inline causal вместо mask
    // (mask_ptr тогда ДОЛЖЕН быть nullptr). q_abs_offset[b] = seq_len_total[b] - n_q[b].
    const int32_t * Q_ABS_OFFSET_d,
    int window,
    // M1 gather-path window clip start [B] (nullptr = full scan); the K
    // buffer must contain rows [kv_start[b], seq_len[b]) (see
    // launch_gather_paged_kv_quant_persistent).
    const int32_t * KV_START_d,
    // M2 tile clip (1 = on; GFX906_FA_TILE_CLIP; see flash_attn_tile_q8).
    int tile_clip,
    int batch, int heads_q, int heads_kv,
    int seq_q, int seq_kv, int head_dim,
    float scale,
    hipStream_t stream,
    int nc2,
    int kv_split
);

// KV-split (gridDim.y > 1) partials merge (gfx906_fa_launcher.cu).
extern "C" hipError_t gfx906_fa_split_combine(
    float     *O_part,
    float2    *meta,
    float     *O_out,
    int        rows,
    int        y,
    int        D,
    hipStream_t stream
);

// FA decode parallelism knobs (FA kernel track). Defaults: GQA head-packing
// (ncols2=8) + KV split (gridDim.y=16) — ~4.2x faster than legacy at the
// B=1 decode tile, +10% e2e (57.1 -> 62.9 t/s). Legacy behavior is the
// kill switch: GFX906_FA_NC2=1 GFX906_FA_KVSPLIT=1.
static int get_fa_nc2() {
    static int v = [] {
        const char *e = std::getenv("GFX906_FA_NC2");
        return e ? std::atoi(e) : 8;
    }();
    return v;
}
// Returns -1 when the env is unset so each call site applies its own
// default: the gather path defaults to 16 (the B=1-decode tuning), the
// direct-paged path to a batch-aware value (see gfx906_fa_forward_paged_
// direct — the optimum there moves with the grid-z block count).
// Note: 0 is NOT the same as unset — 0 passes through and the launcher
// clamps it to 1 (the unset defaults above are the way to select a
// path's default).
// KV-split default, shape-aware. The flat "16" was the B=1-decode (Sq=2)
// tuning, but Sq and y interact: at Sq=2 the kernel is already ~88% HBM-bound
// and y=32 costs +3.8% (extra split-combine traffic beats the parallelism
// gain); at Sq>=4 (spec-decode verify, Sq_pad in {4,8}) it is ALU/compute-
// bound and y=32 wins ~10% (fewer blocks each scanning a shorter KV range).
// Measured on MI50 gfx906, Qwen27 per-rank Hq=12/Hkv=2/D=256, Sk=122880:
//   Sq=2: y=16 1644 us vs y=32 1707 us  (keep 16)
//   Sq=8: y=16 3094 us vs y=32 2765 us  (use 32)
// So the default is 32 for seq_q>=4, 16 otherwise. GFX906_FA_KVSPLIT still
// overrides unconditionally; GFX906_FA_KVSPLIT_SHAPEAWARE=0 restores the old
// flat-16 default (A/B / kill switch). The byte budget below still bounds any
// value, so prefill (Sq~thousands) is protected either way.
static int fa_kv_split_default(long seq_q) {
    const char *e = std::getenv("GFX906_FA_KVSPLIT_SHAPEAWARE");
    if (e && std::atoi(e) == 0) return 16;   // kill switch: old flat default
    return seq_q >= 4 ? 32 : 16;
}
// R3 (fa-decode-fp16-hunt-combined.md): the paged-direct default, aligned
// with the gather path. Verify shapes (Sq>=4) take the shape-aware value
// (32), so flipping GFX906_FA_LEGACY 1->0 does not silently change the
// split (the old batch clamp capped at 8 — a measured ~10% regression on
// the Sq=8 verify shapes). Decode shapes (Sq<4) keep this path's own
// measured batch-aware clamp: its grid is batch*heads_q blocks at NC2=1
// (the gather path's is not), so its decode optimum is S=8/8/5/2/2 for
// B=1/2/3/4/8 (MI50 micro-bench, DEVLOG-muse-glimmer.md 2026-08-27
// follow-up), not the gather path's 16.
static int fa_paged_kv_split_default(long seq_q, long batch) {
    if (seq_q >= 4) return fa_kv_split_default(seq_q);
    const long b = std::max(1L, batch);
    return static_cast<int>(std::max(2L, std::min(8L, 16L / b)));
}
static int get_fa_kv_split() {
    static int v = [] {
        const char *e = std::getenv("GFX906_FA_KVSPLIT");
        return e ? std::atoi(e) : -1;
    }();
    return v;
}

// MTP-1b-0: KV-split partial-buffer byte budget. The old hard clamp
// `if (seq_q > 2) kv_split = 1` was an OOM guard for PREFILL, where Sq is
// thousands and the per-split partial buffer [B, Sq, Hq, y, D] fp32 reaches
// hundreds of MiB. But it also fired for spec-decode VERIFY, which presents
// k+1 query tokens as one sequence (Sq = k+1, padded to {2,4,8}): there the
// partial buffer is a few MB even at kv_split=16, and forcing y=1 stripped
// all KV-split parallelism from exactly the O(Sk) full-attention that dominates
// a long-context step (k=2 verify lost ~8-16x attention workgroups vs k=1).
// Replace the seq_q threshold with a byte budget: keep kv_split>1 whenever the
// partial buffer stays under GFX906_FA_KVSPLIT_MAX_BYTES (default 512 MiB),
// which covers Sq<=~32 at B=1/Hq=24/D=256/split=16 (~24 MB) and still forces
// y=1 for real prefill. Env override lets a test pin the budget.
static int fa_kv_split_budget_bytes() {
    static int v = [] {
        const char *e = std::getenv("GFX906_FA_KVSPLIT_MAX_BYTES");
        if (!e) return 512 * 1024 * 1024;
        const long long parsed = std::strtoll(e, nullptr, 10);
        // <=0 disables the guard (tests pin it off); clamp to int range.
        return parsed > 0 ? (int)std::min<long long>(parsed, (long long)INT32_MAX) : -1;
    }();
    return v;
}
// Returns kv_split unchanged if the partial buffer [B, Sq, Hq, y, D] fp32 fits
// the byte budget, else 1. A negative/zero budget disables the guard (always
// allow) — used by tests. kv_split <= 1 passes through as-is; NOTE: an explicit
// GFX906_FA_KVSPLIT=0/negative is NOT clamped here — it reaches the launcher,
// which independently enforces kv_split >= 1 before building the grid
// (gfx906_fa_launcher.cu, both launch sites). Do not remove that dependency.
static int fa_apply_kv_split_budget(int kv_split, long batch, long seq_q,
                                    long heads_q, long head_dim) {
    if (kv_split <= 1) return kv_split;
    const int budget = fa_kv_split_budget_bytes();
    if (budget <= 0) return kv_split;
    const long bytes = batch * seq_q * heads_q * (long)kv_split * head_dim * 4L;
    return bytes > budget ? 1 : kv_split;
}

// M2 tile clip: per-q-tile k0_base window raise + per-q-tile causal
// k_VKQ_max cap in both FA kernels (bit-identical; see the kernel
// comments). Default ON; GFX906_FA_TILE_CLIP=0 is the A/B arm / kill
// switch (it disables ONLY these two M2 bounds — the M1 sequence-level
// floor is the _WINDOW_CLIP/_GATHER_CLIP backend clips).
// Deliberately NOT memoized like the get_fa_* IIFE neighbors — the
// tests/bench flip the env in-process; a memoized copy would silently
// test a stale value. A/B scope: read per call, so eager/uncaptured
// calls see flips; under FULL capture the value is a baked kernel
// argument (flipping mid-session does not affect already-captured
// graphs — M2 is prefill-only, so that never matters today; do not
// inherit this pattern for a decode-relevant knob).
static int get_fa_tile_clip() {
    const char *e = std::getenv("GFX906_FA_TILE_CLIP");
    return e ? (std::atoi(e) != 0 ? 1 : 0) : 1;
}

// Persistent gather knobs (plan_masked_fa.md). The grid is a capture-time
// constant (frozen per graph), so these are read once at process start.
// GFX906_FA_PERSIST_GRID: fixed workgroup count (default 1024 = 17 per
// CU on the 60-CU MI50; the micro-bench matrix tunes this).
static int get_fa_persist_grid() {
    static int v = [] {
        const char *e = std::getenv("GFX906_FA_PERSIST_GRID");
        int g = e ? std::atoi(e) : 1024;
        return g > 0 ? g : 1024;
    }();
    return v;
}
// GFX906_FA_PERSIST_MARGIN: V-only zero rows written past seq_len as a
// defensive headroom for the FA tail tile (max D=256 nbatch_fa is 128).
// The NaN-tail gate PASSED (DEVLOG-masked-fa.md) — 0 is safe; 128 stays
// the conservative default (belt-and-braces, ~64 KB/head; droppable).
static int get_fa_persist_margin() {
    static int v = [] {
        const char *e = std::getenv("GFX906_FA_PERSIST_MARGIN");
        int m = e ? std::atoi(e) : 128;
        return m >= 0 ? m : 0;
    }();
    return v;
}

// Device-side quantize launchers (gfx906_fa_quant.cu)
extern "C" hipError_t launch_quantize_q8_0_dense(
    const __half * x,
    uint8_t      * y,
    int N,
    int D,
    hipStream_t stream
);

extern "C" hipError_t launch_reshape_and_cache_q8(
    const __half  * key,
    const int64_t * slot_mapping,
    uint8_t       * k_cache_q8,
    int num_tokens,
    int num_kv_heads,
    int head_size,
    int block_size,
    int64_t key_token_stride,
    int64_t key_head_stride,
    int64_t cache_block_stride,
    int64_t cache_token_stride,
    int64_t cache_head_stride,
    hipStream_t stream
);

// Level 3c: direct-paged FA launcher (no gather).
extern "C" hipError_t gfx906_fa_launch_paged(
    const float *   Q_fp32,
    const void  *   K_paged,
    const __half *  V_paged,
    const int32_t * block_table,
    const int32_t * kv_max_d,
    float *         O_fp32,
    float2 *        O_meta,
    const __half *  MASK_f16,
    int32_t         mask_seq_kv_padded,
    const int32_t * Q_ABS_OFFSET_d,
    int             window,
    const int32_t * KV_START_d,
    int             tile_clip,
    int             batch,
    int             heads_q,
    int             heads_kv,
    int             seq_q,
    int             max_seq_kv,
    int             head_dim,
    int             block_size,
    int             max_blocks_per_seq,
    int64_t         k_block_stride,
    int64_t         k_token_stride,
    int64_t         k_head_stride,
    int64_t         v_block_stride,
    int64_t         v_token_stride,
    int64_t         v_head_stride,
    float           scale,
    hipStream_t     stream,
    int             kv_split
);

// Stage 2: fused gather + inline K quantization (LEGACY decode path).
extern "C" hipError_t launch_gather_paged_kv_quant(
    const __half  * key_cache,
    const __half  * value_cache,
    const int32_t * block_table,
    const int32_t * seq_lens,
    uint8_t       * k_q8_out,
    __half        * v_out,
    int num_seqs,
    int num_kv_heads,
    int Sk,
    int D,
    int bytes_per_row,
    int block_size,
    int max_blocks_per_seq,
    int64_t k_block_stride,
    int64_t k_token_stride,
    int64_t k_head_stride,
    int64_t v_block_stride,
    int64_t v_token_stride,
    int64_t v_head_stride,
    hipStream_t stream
);

// Persistent grid-stride fused gather + quantize (LEGACY decode path):
// fixed grid, live seq_lens-bounded work, valid at every Sk.
extern "C" hipError_t launch_gather_paged_kv_quant_persistent(
    const __half  * key_cache,
    const __half  * value_cache,
    const int32_t * block_table,
    const int32_t * seq_lens,
    // M1 gather-path window clip start [B] (nullptr = gather from 0).
    // Non-null: sequence s gathers rows [start[s], seq_len[s]) into buffer
    // rows [start[s], Sk) — buffer index == absolute position, so the FA
    // kernel's inline causal/window masks (keyed on absolute position) stay
    // valid unmodified; rows [0, start[s]) are left stale (never read).
    const int32_t * kv_start,
    uint8_t       * k_q8_out,
    __half        * v_out,
    int num_seqs,
    int num_kv_heads,
    int Sk,
    int D,
    int bytes_per_row,
    int block_size,
    int max_blocks_per_seq,
    int margin_zeros,
    int grid,
    int64_t k_block_stride,
    int64_t k_token_stride,
    int64_t k_head_stride,
    int64_t v_block_stride,
    int64_t v_token_stride,
    int64_t v_head_stride,
    hipStream_t stream
);

// Level 1: fused paged gather K_q8 + V_fp16 → contiguous BHSD.
// Заменяет Python-путь _gather_kv_q8 (fancy-indexing + permute).
extern "C" hipError_t launch_gather_paged_kv_q8(
    const uint8_t * key_cache_q8,
    const __half  * value_cache,
    const int32_t * block_table,
    const int32_t * seq_lens,
    const int32_t * kv_start,   // M1 clip, may be NULL
    uint8_t       * k_out,
    __half        * v_out,
    int num_seqs,
    int num_kv_heads,
    int Sk,
    int D,
    int bytes_per_row,
    int block_size,
    int max_blocks_per_seq,
    int64_t cache_block_stride,
    int64_t cache_token_stride,
    int64_t cache_head_stride_q8,
    int64_t v_cache_block_stride,
    int64_t v_cache_token_stride,
    int64_t v_cache_head_stride,
    hipStream_t stream
);

#define TORCH_CHECK_CUDA(x) TORCH_CHECK((x).device().is_cuda(), #x " must be CUDA/HIP")
#define TORCH_CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

torch::Tensor gfx906_fa_forward(
    torch::Tensor q,       // fp32 [B, Hq, Sq, D]
    torch::Tensor k_q8,    // uint8 [B, Hkv, Skv, D*34/32]
    torch::Tensor v_fp16,  // fp16 [B, Hkv, Skv, D]
    double scale,          // softmax scale, типично 1/sqrt(D)
    c10::optional<torch::Tensor> kv_max = c10::nullopt,  // int32 [B] или [B, Sq_tiles]
    c10::optional<torch::Tensor> mask  = c10::nullopt,    // fp16 [B, Sq, Skv_pad]
    // Level 3a inline causal mask. Mutually exclusive with `mask`.
    //   q_abs_offset: int32 [B] — абсолютная позиция 0-го query-токена в sequence.
    //   При non-nullopt kernel сам выкидывает k > (q_abs_offset[b] + col_Q_0 + j).
    // Избавляет от материальной [B, Sq_pad, Sk_pad] fp16 маски, критично для
    // prefill chunks >16K (mask был бы 100+ MB на GPU и приводил к OOM).
    c10::optional<torch::Tensor> q_abs_offset = c10::nullopt,
    // Sliding-window size in tokens (0 = off). REQUIRES q_abs_offset —
    // without the absolute query position the window mask cannot be
    // evaluated (the kernel silently degrades to full attention if it is
    // missing, so we error instead).
    int64_t window = 0,
    // M1 gather-path window clip start [B] (abs position; see the paged
    // binding of the same name). The K buffer must contain rows
    // [kv_start[b], seq_len[b]); the backend pairs it with the
    // kv_start-aware persistent gather.
    c10::optional<torch::Tensor> kv_start = c10::nullopt,
    // Explicit KV-split override (VIT-1). The shape-aware default is tuned for
    // DECODE (Sq in {1,2,4,8}), where splitting the KV scan across grid.y buys
    // parallelism cheaply; for a PREFILL-shaped bidirectional call (the Qwen
    // ViT: Sq = 576..2304, Hq=16, D_pad=128) the per-split partial buffer
    // [B, Sq, Hq, y, D] fp32 grows with Sq and the split-combine dominates:
    // measured 34-42 ms at Sq=1536/1728 (y=32, inside the 512 MiB budget)
    // against 7 ms at Sq=2304 (y forced to 1 by that same budget). Callers with
    // prefill-shaped inputs pass 1; <= 0 keeps default resolution.
    c10::optional<int64_t>   kv_split_override = c10::nullopt
) {
    TORCH_CHECK(window == 0 || q_abs_offset.has_value(),
        "window > 0 requires q_abs_offset (the window mask needs absolute "
        "query positions)");
    TORCH_CHECK_CUDA(q);
    TORCH_CHECK_CUDA(k_q8);
    TORCH_CHECK_CUDA(v_fp16);
    TORCH_CHECK_CONTIG(q);
    TORCH_CHECK_CONTIG(k_q8);
    TORCH_CHECK_CONTIG(v_fp16);

    TORCH_CHECK(q.dtype() == torch::kFloat32,  "q must be fp32");
    TORCH_CHECK(k_q8.dtype() == torch::kUInt8,
                "k_q8 must be uint8 (planar q8_0 row bytes, plan_fa_part_A.md)");
    TORCH_CHECK(v_fp16.dtype() == torch::kFloat16, "v_fp16 must be fp16");

    TORCH_CHECK(q.dim() == 4,      "q must be 4D [B, Hq, Sq, D]");
    TORCH_CHECK(v_fp16.dim() == 4, "v_fp16 must be 4D [B, Hkv, Skv, D]");

    const int batch     = q.size(0);
    const int heads_q   = q.size(1);
    const int seq_q     = q.size(2);
    const int head_dim  = q.size(3);

    const int heads_kv  = v_fp16.size(1);
    const int seq_kv    = v_fp16.size(2);

    TORCH_CHECK(v_fp16.size(0) == batch,    "v batch mismatch");
    TORCH_CHECK(v_fp16.size(3) == head_dim, "v head_dim mismatch");

    // K — layout [B, Hkv, Skv, (D/QK8_0) * sizeof(block_q8_0)] уложенный
    // uint8; row = planar q8_0 (plan_fa_part_A.md): quants D bytes + scale
    // (D/32) fp16 — row byte count equals sizeof(block_q8_0) * (D/32).
    constexpr int QK8_0_SZ = 32;
    constexpr int BLOCK_SZ = 34;  // sizeof(block_q8_0) = fp16 scale + 32 int8
    TORCH_CHECK(head_dim % QK8_0_SZ == 0, "head_dim must be multiple of 32");
    const int expected_k_bytes = (head_dim / QK8_0_SZ) * BLOCK_SZ;
    TORCH_CHECK(k_q8.size(0) == batch && k_q8.size(1) == heads_kv && k_q8.size(2) == seq_kv,
                "k_q8 first 3 dims must match [B, Hkv, Skv]");
    TORCH_CHECK(k_q8.size(3) == expected_k_bytes, "k_q8 last dim = (D/32)*34 bytes");

    // Output.
    // KERNEL пишет в layout [B, Sq, Hq, D] (BSHD):
    //   j_dst_unrolled = ((seq*Sq + sq_idx)*Hq + head)*gridY + split
    // Мы поэтому аллоцируем o_bshd как BSHD, передаём его в kernel
    // и возвращаем его как есть (без transpose-копии); Python-фронт
    // вырезает строку Sq=0 (contiguous view) на decode fast path.
    auto opts_f32 = q.options().dtype(torch::kFloat32);
    torch::Tensor o_bshd = torch::empty({batch, seq_q, heads_q, head_dim}, opts_f32);

    // KV-split (GFX906_FA_KVSPLIT>1): kernel writes per-split partials
    // [B, Sq, Hq, y, D] + meta [B, Sq, Hq, y, 2] (unscaled, (m, l)); merged
    // into o_bshd by gfx906_fa_split_combine.
    const int nc2 = get_fa_nc2();
    int kv_split = get_fa_kv_split();
    // VIT-1: a per-call override wins over the env/shape-aware default. y=1 is
    // always safe (no partials, no combine), so it bypasses the byte budget.
    const bool kv_split_pinned_one =
        kv_split_override.has_value() && kv_split_override.value() == 1;
    if (kv_split_override.has_value() && kv_split_override.value() > 0) {
        kv_split = static_cast<int>(kv_split_override.value());
    }
    const int tile_clip = get_fa_tile_clip();
    if (kv_split < 0) kv_split = fa_kv_split_default(seq_q);  // shape-aware (see fn)
    // MTP-1b-0: the old hard clamp `if (seq_q > 2) kv_split = 1` (a prefill
    // OOM guard) also fired for spec-decode verify (Sq = k+1 padded to {2,4,8}),
    // stripping KV-split parallelism from k>=2 verify's O(Sk) attention. The
    // byte budget below protects against the real OOM case (prefill Sq~thousands)
    // while letting small-Sq verify keep kv_split>1. An explicit
    // GFX906_FA_KVSPLIT is still bounded by the budget, so it cannot OOM.
    if (!kv_split_pinned_one) {
        kv_split = fa_apply_kv_split_budget(
            kv_split, batch, seq_q, heads_q, head_dim);
    }
    // M3 #4b: the kernel's only dst_meta write is guarded by
    // `gridDim.y != 1` (gridDim.y == kv_split), so at kv_split==1 the
    // [B, Sq, Hq, 2] meta is never written — nor read back: it is a
    // local tensor, not returned to the caller. Pass nullptr and skip
    // the dead allocation (scales with Sq: ~300 KB/layer at
    // Sq=1568/Hq=24). At kv_split>1 the written/read meta is
    // o_meta_split (consumed by the split-combine), allocated below.
    torch::Tensor o_part, o_meta_split;
    float * o_fp32_ptr = o_bshd.data_ptr<float>();
    float2 * o_meta_ptr = nullptr;
    if (kv_split > 1) {
        o_part = torch::empty({batch, seq_q, heads_q, kv_split, head_dim}, opts_f32);
        o_meta_split = torch::empty({batch, seq_q, heads_q, kv_split, 2}, opts_f32);
        o_fp32_ptr = o_part.data_ptr<float>();
        o_meta_ptr = reinterpret_cast<float2 *>(o_meta_split.data_ptr<float>());
    }

    auto stream = c10::hip::getCurrentHIPStream().stream();

    // kv_max: опциональный int32 tensor[B] для per-sequence cutoff.
    // Kernel ждёт layout [B, gridDim.x] где gridDim.x = ceil(seq_q/ncols1).
    // ncols1 выбирается launcher'ом динамически из seq_q:
    //   seq_q > 32 → 64 ; > 16 → 32 ; > 8 → 16 ; > 4 → 8 ; > 2 → 4 ; else 2
    // Launcher сам создаёт буфер нужного размера и реплицирует kv_max[b]
    // на все gridDim.x тайлов.
    const int * kv_max_ptr = nullptr;
    torch::Tensor kv_max_expanded;
    if (kv_max.has_value()) {
        auto kvm = kv_max.value();
        TORCH_CHECK_CUDA(kvm);
        TORCH_CHECK(kvm.dtype() == torch::kInt32, "kv_max must be int32");
        TORCH_CHECK(kvm.dim() == 1, "kv_max must be 1D [B]");
        TORCH_CHECK(kvm.size(0) == batch, "kv_max[0] must equal batch");

        // Compute ncols1 same way as launcher dispatcher (keep in sync!)
        const int ncols1 = fa_pick_ncols1(seq_q);
        const int grid_x = (seq_q + ncols1 - 1) / ncols1;

        // Expand [B] → [B, grid_x] (contiguous), каждая sequence получает
        // одинаковое значение для всех Q-tiles (без causal mask в MVP).
        kv_max_expanded = kvm.unsqueeze(1).expand({batch, grid_x}).contiguous();
        // FIX (H2, 2026-09-11): pad Q-tiles — все строки за n_q[b], потому
        // -INF-маскированы с отброшенным выходом — ранее наследовали
        // kv_max = seq_len[b] и each pad tile walked the sequence's ENTIRE
        // context KV from HBM only to produce discarded results. In mixed
        // prefill+decode steps that streamed each decoding sequence's whole
        // context once per chunk-token: the O(live-context) multi-batch
        // prefill tax (~16 s/step tail at 4x120k, the 75-min "stall";
        // docs/gfx906/ttft-prefill-stall.md §13.14–13.16). Clamp pad tiles
        // (t*ncols1 >= n_q[b]) to 0 — the launcher already guards all-empty
        // rows (kv_max=0 -> weights exp(-inf)=0, output rows unused).
        // n_q[b] = seq_len[b] - q_abs_offset[b]; present exactly when the
        // inline-causal path is active — without it grid_x == 1 (pure
        // decode) and there are no pad tiles.
        if (q_abs_offset.has_value() && grid_x > 1) {
            auto tile0 = torch::arange(grid_x, kvm.options()) * ncols1;
            auto pad = tile0.unsqueeze(0) >=
                (kvm - q_abs_offset.value()).unsqueeze(1);
            kv_max_expanded.masked_fill_(pad, 0);
        }
        kv_max_ptr = kv_max_expanded.data_ptr<int32_t>();
    }

    const __half * mask_ptr = nullptr;
    int32_t mask_seq_kv_padded = 0;
    torch::Tensor mask_contig;
    if (mask.has_value()) {
        auto m = mask.value();
        TORCH_CHECK_CUDA(m);
        TORCH_CHECK(m.dtype() == torch::kFloat16, "mask must be fp16");
        TORCH_CHECK(m.dim() == 3, "mask must be 3D [B, Sq, Skv_pad]");
        TORCH_CHECK(m.size(0) == batch, "mask.size(0) must equal batch");
        TORCH_CHECK(m.size(1) == seq_q, "mask.size(1) must equal seq_q");
        TORCH_CHECK(m.size(2) >= seq_kv, "mask.size(2) must be >= seq_kv");
        mask_contig = m.contiguous();
        mask_ptr = reinterpret_cast<const __half *>(mask_contig.data_ptr<at::Half>());
        mask_seq_kv_padded = (int32_t) mask_contig.size(2);
    }

    const int32_t * q_abs_offset_ptr = nullptr;
    torch::Tensor q_abs_offset_contig;
    if (q_abs_offset.has_value()) {
        TORCH_CHECK(!mask.has_value(),
            "q_abs_offset and mask are mutually exclusive (inline causal replaces mask)");
        auto qo = q_abs_offset.value();
        TORCH_CHECK_CUDA(qo);
        TORCH_CHECK(qo.dtype() == torch::kInt32, "q_abs_offset must be int32");
        TORCH_CHECK(qo.dim() == 1 && qo.size(0) == batch,
            "q_abs_offset must be 1D [B]");
        q_abs_offset_contig = qo.contiguous();
        q_abs_offset_ptr = q_abs_offset_contig.data_ptr<int32_t>();
    }

    // M1 gather-path window clip start (mirrors the paged-direct check).
    const int32_t * kv_start_ptr = nullptr;
    torch::Tensor kv_start_c;
    if (kv_start.has_value()) {
        TORCH_CHECK(q_abs_offset.has_value() && window > 0,
            "kv_start requires q_abs_offset and window > 0");
        auto ks = kv_start.value();
        TORCH_CHECK_CUDA(ks);
        TORCH_CHECK(ks.dtype() == torch::kInt32, "kv_start must be int32");
        TORCH_CHECK(ks.dim() == 1 && ks.size(0) == batch,
            "kv_start must be [batch]");
        kv_start_c = ks.contiguous();
        kv_start_ptr = kv_start_c.data_ptr<int32_t>();
    }

    hipError_t err = gfx906_fa_launch(
        q.data_ptr<float>(),
        k_q8.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(v_fp16.data_ptr<at::Half>()),
        o_fp32_ptr,
        o_meta_ptr,
        kv_max_ptr,
        mask_ptr,
        mask_seq_kv_padded,
        q_abs_offset_ptr,
        window,
        kv_start_ptr,
        tile_clip,
        batch, heads_q, heads_kv, seq_q, seq_kv, head_dim,
        (float) scale,
        stream,
        nc2,
        kv_split
    );

    TORCH_CHECK(err == hipSuccess, "gfx906_fa_launch failed: ", hipGetErrorString(err));

    if (kv_split > 1) {
        const int rows = batch * seq_q * heads_q;
        hipError_t cerr = gfx906_fa_split_combine(
            o_part.data_ptr<float>(),
            reinterpret_cast<float2 *>(o_meta_split.data_ptr<float>()),
            o_bshd.data_ptr<float>(),
            rows, kv_split, head_dim, stream);
        TORCH_CHECK(cerr == hipSuccess, "gfx906_fa_split_combine failed: ",
            hipGetErrorString(cerr));
    }

    // Native BSHD [B, Sq, Hq, D] — the kernel's store layout, returned as
    // is (the .transpose(1,2).contiguous() here was a copy per layer).
    // The Python front-end extracts the Sq=0 row for the Sq=1 decode fast
    // path ([:, 0, :, :] is a contiguous [B, Hq, D] view, zero copies).
    return o_bshd;
}

// ============================================================================
// Q8_0 quantization utility: fp16 K-tensor → q8_0 uint8 tensor (planar row)
//
// Device-side (gfx906_fa_quant.cu). Работает с N-мерным contiguous тензором
// чей последний dim = D (кратно 32). Выход: тот же layout с заменой D → D/32*34.
// ============================================================================
torch::Tensor quantize_q8_0(torch::Tensor k_fp16) {
    TORCH_CHECK_CUDA(k_fp16);
    TORCH_CHECK_CONTIG(k_fp16);
    TORCH_CHECK(k_fp16.dtype() == torch::kFloat16, "k_fp16 must be fp16");
    TORCH_CHECK(k_fp16.dim() >= 1, "k_fp16 must have at least 1 dim");

    const int D = k_fp16.size(-1);
    TORCH_CHECK(D % 32 == 0, "last dim must be multiple of 32");
    const int blocks_per_row = D / 32;
    const int bytes_per_row  = blocks_per_row * 34;

    // N = произведение всех измерений кроме последнего.
    int64_t N = 1;
    for (int i = 0; i < k_fp16.dim() - 1; ++i) {
        N *= k_fp16.size(i);
    }

    // out shape: те же dims + (bytes_per_row) вместо D.
    std::vector<int64_t> out_shape;
    for (int i = 0; i < k_fp16.dim() - 1; ++i) {
        out_shape.push_back(k_fp16.size(i));
    }
    out_shape.push_back(bytes_per_row);
    auto out = torch::empty(out_shape, k_fp16.options().dtype(torch::kUInt8));

    auto stream = c10::hip::getCurrentHIPStream().stream();
    hipError_t err = launch_quantize_q8_0_dense(
        reinterpret_cast<const __half*>(k_fp16.data_ptr<at::Half>()),
        out.data_ptr<uint8_t>(),
        (int)N,
        D,
        stream
    );
    TORCH_CHECK(err == hipSuccess,
                "launch_quantize_q8_0_dense failed: ", hipGetErrorString(err));
    return out;
}

// ============================================================================
// reshape_and_cache_q8: квантует новые K-токены и пишет их в paged Q8 cache
// по slot_mapping. Вызывается из do_kv_cache_update в backend.
//
// Аргументы:
//   key           [num_tokens, Hkv, D]                                 fp16
//   slot_mapping  [num_tokens]                                         int64
//   k_cache_q8    [num_blocks, block_size, Hkv, (D/32)*34]             uint8
//
// slot_mapping[i] < 0 → токен пропускается (padding).
// ============================================================================
void reshape_and_cache_q8(
    torch::Tensor key,            // [num_tokens, Hkv, D] fp16
    torch::Tensor slot_mapping,   // [num_tokens]         int64
    torch::Tensor k_cache_q8      // [num_blocks, block_size, Hkv, (D/32)*34] uint8
) {
    TORCH_CHECK_CUDA(key);
    TORCH_CHECK_CUDA(slot_mapping);
    TORCH_CHECK_CUDA(k_cache_q8);
    TORCH_CHECK(key.dtype() == torch::kFloat16, "key must be fp16");
    TORCH_CHECK(slot_mapping.dtype() == torch::kInt64, "slot_mapping must be int64");
    TORCH_CHECK(k_cache_q8.dtype() == torch::kUInt8, "k_cache_q8 must be uint8");

    TORCH_CHECK(key.dim() == 3, "key must be [num_tokens, Hkv, D]");
    TORCH_CHECK(k_cache_q8.dim() == 4, "k_cache_q8 must be [B, bs, Hkv, bytes]");

    const int num_tokens   = key.size(0);
    const int num_kv_heads = key.size(1);
    const int head_size    = key.size(2);
    TORCH_CHECK(head_size % 32 == 0, "head_size must be multiple of 32");
    const int bytes_per_row = (head_size / 32) * 34;

    const int block_size = k_cache_q8.size(1);
    TORCH_CHECK(k_cache_q8.size(2) == num_kv_heads,
                "k_cache_q8 Hkv mismatch: got ", k_cache_q8.size(2),
                " vs key Hkv=", num_kv_heads);
    TORCH_CHECK(k_cache_q8.size(3) == bytes_per_row,
                "k_cache_q8 last dim mismatch: got ", k_cache_q8.size(3),
                " expected ", bytes_per_row);

    TORCH_CHECK(slot_mapping.size(0) == num_tokens,
                "slot_mapping length mismatch");

    // Strides (в элементах для key, в байтах для cache).
    const int64_t key_token_stride = key.stride(0);  // обычно Hkv*D
    const int64_t key_head_stride  = key.stride(1);  // обычно D

    // Cache strides: use the tensor's REAL strides (contiguous today, but
    // do not depend on it — same bug class as the V-cache unbind stride).
    const int64_t cache_block_stride = k_cache_q8.stride(0);
    const int64_t cache_token_stride = k_cache_q8.stride(1);
    const int64_t cache_head_stride  = k_cache_q8.stride(2);
    TORCH_CHECK(k_cache_q8.stride(3) == 1,
                "reshape_and_cache_q8: last dim must be contiguous");

    auto stream = c10::hip::getCurrentHIPStream().stream();
    hipError_t err = launch_reshape_and_cache_q8(
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
        slot_mapping.data_ptr<int64_t>(),
        k_cache_q8.data_ptr<uint8_t>(),
        num_tokens, num_kv_heads, head_size, block_size,
        key_token_stride, key_head_stride,
        cache_block_stride, cache_token_stride, cache_head_stride,
        stream
    );
    TORCH_CHECK(err == hipSuccess,
                "launch_reshape_and_cache_q8 failed: ", hipGetErrorString(err));
}

// ============================================================================
// Level 1: fused gather K_q8 + V_fp16 → contiguous BHSD.
//
// Вход:
//   key_cache_q8  [num_blocks, block_size, Hkv, bytes_per_row]  uint8
//   value_cache   [num_blocks, block_size, Hkv, D]              fp16
//   block_table   [num_seqs, max_num_blocks]                    int32
//   seq_lens      [num_seqs]                                    int32
//   Sk            — max_seqlen_k (padded до кратного 32 на host)
//
// Выход (ОДНИМ вызовом, без лишних temp buffers):
//   K_out = uint8 [num_seqs, Hkv, Sk, bytes_per_row]
//   V_out = fp16  [num_seqs, Hkv, Sk, D]
//
// Tail-токены в V_out обнуляются, в K_out оставляются мусором (kernel отсекает
// через KV_max). Это идентично семантике _gather_kv_q8 в Python.
// ============================================================================
std::vector<torch::Tensor> gather_paged_kv_q8(
    torch::Tensor key_cache_q8,  // uint8 [num_blocks, block_size, Hkv, bytes_per_row]
    torch::Tensor value_cache,   // fp16  [num_blocks, block_size, Hkv, D]
    torch::Tensor block_table,   // int32 [num_seqs, max_num_blocks]
    torch::Tensor seq_lens,      // int32 [num_seqs]
    int64_t Sk,
    c10::optional<torch::Tensor> k_out_opt = c10::nullopt,  // uint8 [B,Hkv,Sk,bytes] — grow-buffer
    c10::optional<torch::Tensor> v_out_opt = c10::nullopt,   // fp16  [B,Hkv,Sk,D]
    c10::optional<torch::Tensor> kv_start_opt = c10::nullopt // int32  [B], M1 clip
) {
    TORCH_CHECK_CUDA(key_cache_q8);
    TORCH_CHECK_CUDA(value_cache);
    TORCH_CHECK_CUDA(block_table);
    TORCH_CHECK_CUDA(seq_lens);

    TORCH_CHECK(key_cache_q8.dtype() == torch::kUInt8, "key_cache_q8 must be uint8");
    TORCH_CHECK(value_cache.dtype()  == torch::kFloat16, "value_cache must be fp16");
    TORCH_CHECK(seq_lens.dtype()     == torch::kInt32,  "seq_lens must be int32");

    TORCH_CHECK(key_cache_q8.dim() == 4, "key_cache_q8 must be 4D");
    TORCH_CHECK(value_cache.dim()  == 4, "value_cache must be 4D");

    const int num_blocks        = key_cache_q8.size(0);
    const int block_size        = key_cache_q8.size(1);
    const int num_kv_heads      = key_cache_q8.size(2);
    const int bytes_per_row     = key_cache_q8.size(3);
    const int D                 = value_cache.size(3);

    TORCH_CHECK(value_cache.size(0) == num_blocks, "value_cache blocks mismatch");
    TORCH_CHECK(value_cache.size(1) == block_size, "value_cache block_size mismatch");
    TORCH_CHECK(value_cache.size(2) == num_kv_heads, "value_cache Hkv mismatch");
    TORCH_CHECK(D % 32 == 0, "D must be multiple of 32");
    TORCH_CHECK(bytes_per_row == (D / 32) * 34,
                "bytes_per_row must equal (D/32)*34, got ", bytes_per_row,
                " vs expected ", (D / 32) * 34);

    // block_table может быть int32 или int64 — приведём к int32 на стороне вызова
    // (vLLM уже int32). Для безопасности делаем contiguous + проверяем.
    TORCH_CHECK(block_table.dtype() == torch::kInt32,
                "block_table must be int32 (caller should .to(torch.int32))");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(seq_lens.is_contiguous(),   "seq_lens must be contiguous");

    const int num_seqs           = block_table.size(0);
    const int max_blocks_per_seq = block_table.size(1);
    TORCH_CHECK(seq_lens.size(0) == num_seqs, "seq_lens vs block_table batch mismatch");

    // Sk-padding: требование FA kernel — кратно 32 (Sk_pad в Python-обёртке).
    TORCH_CHECK(Sk > 0 && (Sk % 32) == 0, "Sk must be positive multiple of 32, got ", Sk);

    // Outputs — contiguous BHSD.
    // Если caller передал pre-allocated buffer ТОЧНО нужного shape/dtype/contig —
    // используем без копии. Иначе аллоцируем новый (поведение как раньше).
    //
    // Зачем точный size: slice [B', :, Sk', :] от большего buffer [B_max, Hkv, Sk_max, D]
    // остаётся contiguous только когда B' == B_max И Sk' == Sk_max, т.е. на практике
    // всегда false. Вызывающий код сам делает grow-strategy в attention backend.
    auto use_or_alloc = [&](const c10::optional<torch::Tensor>& buf,
                            at::TensorOptions opts,
                            int64_t dim3) -> torch::Tensor {
        if (buf.has_value()) {
            const auto & t = buf.value();
            if (t.dim() == 4
                && t.size(0) == num_seqs
                && t.size(1) == num_kv_heads
                && t.size(2) == Sk
                && t.size(3) == dim3
                && t.dtype() == opts.dtype()
                && t.device() == opts.device()
                && t.is_contiguous()) {
                return t;
            }
        }
        return torch::empty({(int64_t)num_seqs, (int64_t)num_kv_heads, Sk, dim3}, opts);
    };

    auto k_out = use_or_alloc(k_out_opt, key_cache_q8.options(), (int64_t)bytes_per_row);
    auto v_out = use_or_alloc(v_out_opt, value_cache.options(),  (int64_t)D);

    // Strides.
    // Use the tensors' REAL strides: value_cache is typically
    // kv_cache.unbind(1) of [num_blocks, 2, block_size, Hkv, D], i.e.
    // non-contiguous with a 2x block stride. Deriving strides from shapes
    // (block_size*Hkv*D) reads K-cache bytes as V and poisons attention.
    const int64_t cache_block_stride    = key_cache_q8.stride(0);
    const int64_t cache_token_stride    = key_cache_q8.stride(1);
    const int64_t cache_head_stride_q8  = key_cache_q8.stride(2);

    const int64_t v_cache_block_stride  = value_cache.stride(0);
    const int64_t v_cache_token_stride  = value_cache.stride(1);
    const int64_t v_cache_head_stride   = value_cache.stride(2);

    TORCH_CHECK(key_cache_q8.stride(3) == 1 && value_cache.stride(3) == 1,
                "gather_paged_kv_q8: last dim must be contiguous");

    // M1 window clip: per-seq absolute start; tokens [0, start) are not
    // written (the FA k-loop starts at floor(kv_start), which the gather
    // margin keeps materialized — see the kernel docs in
    // gfx906_fa_gather.cu). NULL = full gather (unchanged behavior).
    const int32_t * kv_start_ptr = nullptr;
    if (kv_start_opt.has_value()) {
        const auto & ks = kv_start_opt.value();
        TORCH_CHECK(ks.dtype() == torch::kInt32, "kv_start must be int32");
        TORCH_CHECK(ks.is_contiguous(), "kv_start must be contiguous");
        TORCH_CHECK(ks.dim() == 1 && ks.size(0) == num_seqs,
                    "kv_start must be [num_seqs]");
        kv_start_ptr = ks.data_ptr<int32_t>();
    }

    auto stream = c10::hip::getCurrentHIPStream().stream();
    hipError_t err = launch_gather_paged_kv_q8(
        key_cache_q8.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        kv_start_ptr,
        k_out.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(v_out.data_ptr<at::Half>()),
        num_seqs, num_kv_heads, (int)Sk, D, bytes_per_row, block_size,
        max_blocks_per_seq,
        cache_block_stride, cache_token_stride, cache_head_stride_q8,
        v_cache_block_stride, v_cache_token_stride, v_cache_head_stride,
        stream
    );
    TORCH_CHECK(err == hipSuccess,
                "launch_gather_paged_kv_q8 failed: ", hipGetErrorString(err));

    return {k_out, v_out};
}

// LEGACY-path fused gather: fp16 K cache (no Q8 side buffer) + fp16 V.
// Reuses the byte-generic gather_paged_kv_q8 kernel with bytes_per_row = 2D.
// Output K is fp16 [B, Hkv, Sk, D] (caller quantizes, as the LEGACY path
// does today); V tail is zeroed per seq_lens, K tail unmasked (FA kernel
// cuts via kv_max — same semantics as the Q8 path).
std::vector<torch::Tensor> gather_paged_kv_fp16(
    torch::Tensor key_cache,     // fp16 [num_blocks, block_size, Hkv, D]
    torch::Tensor value_cache,   // fp16 [num_blocks, block_size, Hkv, D]
    torch::Tensor block_table,   // int32 [num_seqs, max_num_blocks]
    torch::Tensor seq_lens,      // int32 [num_seqs]
    int64_t Sk,
    c10::optional<torch::Tensor> k_out_opt = c10::nullopt,  // fp16 [B,Hkv,Sk,D]
    c10::optional<torch::Tensor> v_out_opt = c10::nullopt   // fp16 [B,Hkv,Sk,D]
) {
    TORCH_CHECK_CUDA(key_cache);
    TORCH_CHECK_CUDA(value_cache);
    TORCH_CHECK_CUDA(block_table);
    TORCH_CHECK_CUDA(seq_lens);

    TORCH_CHECK(key_cache.dtype() == torch::kFloat16, "key_cache must be fp16");
    TORCH_CHECK(value_cache.dtype() == torch::kFloat16, "value_cache must be fp16");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4, "caches must be 4D");

    const int num_blocks     = key_cache.size(0);
    const int block_size     = key_cache.size(1);
    const int num_kv_heads   = key_cache.size(2);
    const int D              = key_cache.size(3);

    TORCH_CHECK(value_cache.size(0) == num_blocks, "value_cache blocks mismatch");
    TORCH_CHECK(value_cache.size(1) == block_size, "value_cache block_size mismatch");
    TORCH_CHECK(value_cache.size(2) == num_kv_heads, "value_cache Hkv mismatch");
    TORCH_CHECK(value_cache.size(3) == D, "value_cache D mismatch");
    TORCH_CHECK(D % 32 == 0, "D must be multiple of 32");

    TORCH_CHECK(block_table.dtype() == torch::kInt32,
                "block_table must be int32");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");

    const int num_seqs           = block_table.size(0);
    const int max_blocks_per_seq = block_table.size(1);
    TORCH_CHECK(seq_lens.size(0) == num_seqs, "seq_lens vs block_table batch mismatch");
    TORCH_CHECK(Sk > 0 && (Sk % 32) == 0, "Sk must be positive multiple of 32, got ", Sk);

    const int bytes_per_row = D * 2;
    auto opts = key_cache.options();
    auto use_or_alloc = [&](const c10::optional<torch::Tensor>& buf,
                            int64_t dim3) -> torch::Tensor {
        if (buf.has_value()) {
            const auto & t = buf.value();
            if (t.dim() == 4
                && t.size(0) == num_seqs
                && t.size(1) == num_kv_heads
                && t.size(2) == Sk
                && t.size(3) == dim3
                && t.dtype() == opts.dtype()
                && t.device() == opts.device()
                && t.is_contiguous()) {
                return t;
            }
        }
        return torch::empty({(int64_t)num_seqs, (int64_t)num_kv_heads, Sk, dim3}, opts);
    };
    auto k_out = use_or_alloc(k_out_opt, D);
    auto v_out = use_or_alloc(v_out_opt, D);

    // Stride domains differ by pointer type (the gather kernel is
    // byte-generic for K but uses element strides for the __half* V):
    //   K: uint8_t*  -> byte strides (fp16 element stride x 2)
    //   V: __half*   -> element strides, as the Q8 path passes them
    const int64_t k_cache_block_stride  = key_cache.stride(0)  * 2;
    const int64_t k_cache_token_stride  = key_cache.stride(1)  * 2;
    const int64_t k_cache_head_stride   = key_cache.stride(2)  * 2;
    const int64_t v_cache_block_stride  = value_cache.stride(0);
    const int64_t v_cache_token_stride  = value_cache.stride(1);
    const int64_t v_cache_head_stride   = value_cache.stride(2);
    TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
                "gather_paged_kv_fp16: last dim must be contiguous");

    auto stream = c10::hip::getCurrentHIPStream().stream();
    hipError_t err = launch_gather_paged_kv_q8(
        reinterpret_cast<const uint8_t*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        nullptr,   // no clip: the fp16 fallback path keeps full gather
        reinterpret_cast<uint8_t*>(k_out.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v_out.data_ptr<at::Half>()),
        num_seqs, num_kv_heads, (int)Sk, D, bytes_per_row, block_size,
        max_blocks_per_seq,
        k_cache_block_stride, k_cache_token_stride, k_cache_head_stride,
        v_cache_block_stride, v_cache_token_stride, v_cache_head_stride,
        stream
    );
    TORCH_CHECK(err == hipSuccess,
                "launch_gather_paged_kv_fp16 failed: ", hipGetErrorString(err));

    return {k_out, v_out};
}

// ============================================================================
// Stage 2: fused gather + inline K quantization (LEGACY decode path).
//
// Replaces the two-kernel sequence
//   gather_paged_kv_fp16 -> quantize_q8_0
// with one kernel pass: V copied fp16 (V1 semantics, tail zeroed), K read
// from the fp16 paged cache and quantized to q8_0 in-kernel. K output is
// bit-equal to quantize_q8_0(gather_paged_kv_fp16(...)) — the kernel reuses
// the same quantization helper as quantize_q8_0_dense_kernel.
// ============================================================================
std::vector<torch::Tensor> gather_paged_kv_quantized(
    torch::Tensor key_cache,     // fp16 [num_blocks, block_size, Hkv, D]
    torch::Tensor value_cache,   // fp16 [num_blocks, block_size, Hkv, D]
    torch::Tensor block_table,   // int32 [num_seqs, max_num_blocks]
    torch::Tensor seq_lens,      // int32 [num_seqs]
    int64_t Sk,
    c10::optional<torch::Tensor> k_out_opt = c10::nullopt,  // uint8 [B,Hkv,Sk,bytes] — grow-buffer
    c10::optional<torch::Tensor> v_out_opt = c10::nullopt   // fp16  [B,Hkv,Sk,D]
) {
    TORCH_CHECK_CUDA(key_cache);
    TORCH_CHECK_CUDA(value_cache);
    TORCH_CHECK_CUDA(block_table);
    TORCH_CHECK_CUDA(seq_lens);

    TORCH_CHECK(key_cache.dtype() == torch::kFloat16, "key_cache must be fp16");
    TORCH_CHECK(value_cache.dtype() == torch::kFloat16, "value_cache must be fp16");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4, "caches must be 4D");

    const int num_blocks     = key_cache.size(0);
    const int block_size     = key_cache.size(1);
    const int num_kv_heads   = key_cache.size(2);
    const int D              = key_cache.size(3);

    TORCH_CHECK(value_cache.size(0) == num_blocks, "value_cache blocks mismatch");
    TORCH_CHECK(value_cache.size(1) == block_size, "value_cache block_size mismatch");
    TORCH_CHECK(value_cache.size(2) == num_kv_heads, "value_cache Hkv mismatch");
    TORCH_CHECK(value_cache.size(3) == D, "value_cache D mismatch");
    TORCH_CHECK(D % 32 == 0, "D must be multiple of 32");

    TORCH_CHECK(block_table.dtype() == torch::kInt32,
                "block_table must be int32");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");

    const int num_seqs           = block_table.size(0);
    const int max_blocks_per_seq = block_table.size(1);
    TORCH_CHECK(seq_lens.size(0) == num_seqs, "seq_lens vs block_table batch mismatch");
    TORCH_CHECK(Sk > 0 && (Sk % 32) == 0, "Sk must be positive multiple of 32, got ", Sk);
    TORCH_CHECK(Sk <= 65535, "Sk must fit in gridDim.z (<= 65535), got ", Sk);

    const int bytes_per_row = (D / 32) * 34;

    // Reuse a caller-provided buffer when it is the exact shape/dtype
    // (same grow-strategy as gather_paged_kv_q8: the attention backend
    // holds the class-level buffers; without this every attention layer
    // allocates 24-200+ MiB per step on long contexts).
    auto use_or_alloc = [&](const c10::optional<torch::Tensor>& buf,
                            at::TensorOptions opts,
                            int64_t dim3) -> torch::Tensor {
        if (buf.has_value()) {
            const auto & t = buf.value();
            if (t.dim() == 4
                && t.size(0) == num_seqs
                && t.size(1) == num_kv_heads
                && t.size(2) == Sk
                && t.size(3) == dim3
                && t.dtype() == opts.dtype()
                && t.device() == opts.device()
                && t.is_contiguous()) {
                return t;
            }
        }
        return torch::empty({(int64_t)num_seqs, (int64_t)num_kv_heads, Sk, dim3}, opts);
    };
    auto k_out = use_or_alloc(k_out_opt, key_cache.options().dtype(torch::kUInt8),
                              (int64_t)bytes_per_row);
    auto v_out = use_or_alloc(v_out_opt, key_cache.options(), (int64_t)D);

    // Element strides for both caches (the kernel reads K as __half*).
    const int64_t k_block_stride = key_cache.stride(0);
    const int64_t k_token_stride = key_cache.stride(1);
    const int64_t k_head_stride  = key_cache.stride(2);
    const int64_t v_block_stride = value_cache.stride(0);
    const int64_t v_token_stride = value_cache.stride(1);
    const int64_t v_head_stride  = value_cache.stride(2);
    TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
                "gather_paged_kv_quantized: last dim must be contiguous");

    auto stream = c10::hip::getCurrentHIPStream().stream();
    hipError_t err = launch_gather_paged_kv_quant(
        reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        k_out.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(v_out.data_ptr<at::Half>()),
        num_seqs, num_kv_heads, (int)Sk, D, bytes_per_row, block_size,
        max_blocks_per_seq,
        k_block_stride, k_token_stride, k_head_stride,
        v_block_stride, v_token_stride, v_head_stride,
        stream
    );
    TORCH_CHECK(err == hipSuccess,
                "launch_gather_paged_kv_quant failed: ", hipGetErrorString(err));

    return {k_out, v_out};
}

// ============================================================================
// Persistent grid-stride fused gather + quantize (LEGACY decode path).
// One kernel with a fixed (capture-time-constant) grid whose work is
// bounded by the live seq_lens tensor — replaces the two-kernel > 65535
// fallback at every Sk. K output is bit-equal to
// quantize_q8_0(gather_paged_kv_fp16(x)); V is a byte copy. Rows >=
// seq_len are not written, except `margin_zeros` V-only zero rows past
// seq_len (defensive FA tail-tile headroom, GFX906_FA_PERSIST_MARGIN).
// ============================================================================
std::vector<torch::Tensor> gather_paged_kv_quant_persistent(
    torch::Tensor key_cache,     // fp16 [num_blocks, block_size, Hkv, D]
    torch::Tensor value_cache,   // fp16 [num_blocks, block_size, Hkv, D]
    torch::Tensor block_table,   // int32 [num_seqs, max_num_blocks]
    torch::Tensor seq_lens,      // int32 [num_seqs]
    int64_t Sk,
    c10::optional<torch::Tensor> k_out_opt = c10::nullopt,  // uint8 [B,Hkv,Sk,bytes]
    c10::optional<torch::Tensor> v_out_opt = c10::nullopt,  // fp16  [B,Hkv,Sk,D]
    // M1 gather-path window clip start [B] (abs position). Non-null:
    // sequence s gathers rows [start[s], seq_len[s]) into buffer rows
    // [start[s], Sk) (buffer index == absolute position; see the launcher
    // comment). start[s] >= Sk is rejected — the clipped rows would not fit.
    c10::optional<torch::Tensor> kv_start = c10::nullopt
) {
    TORCH_CHECK_CUDA(key_cache);
    TORCH_CHECK_CUDA(value_cache);
    TORCH_CHECK_CUDA(block_table);
    TORCH_CHECK_CUDA(seq_lens);

    TORCH_CHECK(key_cache.dtype() == torch::kFloat16, "key_cache must be fp16");
    TORCH_CHECK(value_cache.dtype() == torch::kFloat16, "value_cache must be fp16");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4, "caches must be 4D");

    const int num_blocks     = key_cache.size(0);
    const int block_size     = key_cache.size(1);
    const int num_kv_heads   = key_cache.size(2);
    const int D              = key_cache.size(3);

    TORCH_CHECK(value_cache.size(0) == num_blocks, "value_cache blocks mismatch");
    TORCH_CHECK(value_cache.size(1) == block_size, "value_cache block_size mismatch");
    TORCH_CHECK(value_cache.size(2) == num_kv_heads, "value_cache Hkv mismatch");
    TORCH_CHECK(value_cache.size(3) == D, "value_cache D mismatch");
    TORCH_CHECK(D % 32 == 0, "D must be multiple of 32");

    TORCH_CHECK(block_table.dtype() == torch::kInt32,
                "block_table must be int32");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");

    const int num_seqs           = block_table.size(0);
    const int max_blocks_per_seq = block_table.size(1);
    TORCH_CHECK(seq_lens.size(0) == num_seqs, "seq_lens vs block_table batch mismatch");
    TORCH_CHECK(Sk > 0 && (Sk % 32) == 0, "Sk must be positive multiple of 32, got ", Sk);
    TORCH_CHECK(num_seqs <= 16, "num_seqs must be <= 16, got ", num_seqs);

    const int32_t * kv_start_ptr = nullptr;
    torch::Tensor kv_start_c;
    if (kv_start.has_value()) {
        auto ks = kv_start.value();
        TORCH_CHECK_CUDA(ks);
        TORCH_CHECK(ks.dtype() == torch::kInt32, "kv_start must be int32");
        TORCH_CHECK(ks.dim() == 1 && ks.size(0) == num_seqs,
            "kv_start must be [num_seqs]");
        kv_start_c = ks.contiguous();
        kv_start_ptr = kv_start_c.data_ptr<int32_t>();
    }

    const int bytes_per_row = (D / 32) * 34;

    auto use_or_alloc = [&](const c10::optional<torch::Tensor>& buf,
                            at::TensorOptions opts,
                            int64_t dim3) -> torch::Tensor {
        if (buf.has_value()) {
            const auto & t = buf.value();
            if (t.dim() == 4
                && t.size(0) == num_seqs
                && t.size(1) == num_kv_heads
                && t.size(2) == Sk
                && t.size(3) == dim3
                && t.dtype() == opts.dtype()
                && t.device() == opts.device()
                && t.is_contiguous()) {
                return t;
            }
        }
        return torch::empty({(int64_t)num_seqs, (int64_t)num_kv_heads, Sk, dim3}, opts);
    };
    auto k_out = use_or_alloc(k_out_opt, key_cache.options().dtype(torch::kUInt8),
                              (int64_t)bytes_per_row);
    auto v_out = use_or_alloc(v_out_opt, key_cache.options(), (int64_t)D);

    const int64_t k_block_stride = key_cache.stride(0);
    const int64_t k_token_stride = key_cache.stride(1);
    const int64_t k_head_stride  = key_cache.stride(2);
    const int64_t v_block_stride = value_cache.stride(0);
    const int64_t v_token_stride = value_cache.stride(1);
    const int64_t v_head_stride  = value_cache.stride(2);
    TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
                "gather_paged_kv_quant_persistent: last dim must be contiguous");

    auto stream = c10::hip::getCurrentHIPStream().stream();
    hipError_t err = launch_gather_paged_kv_quant_persistent(
        reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        kv_start_ptr,
        k_out.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(v_out.data_ptr<at::Half>()),
        num_seqs, num_kv_heads, (int)Sk, D, bytes_per_row, block_size,
        max_blocks_per_seq,
        get_fa_persist_margin(), get_fa_persist_grid(),
        k_block_stride, k_token_stride, k_head_stride,
        v_block_stride, v_token_stride, v_head_stride,
        stream
    );
    TORCH_CHECK(err == hipSuccess,
                "launch_gather_paged_kv_quant_persistent failed: ",
                hipGetErrorString(err));

    return {k_out, v_out};
}

// ============================================================================
// Level 3c: Direct-paged FlashAttention.
//
// НЕТ gather step: kernel читает K/V directly from paged cache через
// block_table indirection. Уменьшает HBM трафик в 2x (нет промежуточной копии),
// ожидаемый выигрыш +15-25% на длинных контекстах.
//
// Layout (vLLM-compatible, block_size=16):
//   q              : fp32  [B, Hq, Sq_pad, D]
//   key_cache_q8   : uint8 [num_blocks, 16, Hkv, (D/32)*34]
//   value_cache    : fp16  [num_blocks, 16, Hkv, D]
//   block_table    : int32 [num_seqs, max_blocks_per_seq]
//   seq_lens       : int32 [num_seqs]
//
// Выход: [B, Sq_pad, Hq, D] fp32 в BSHD (тот же формат что у forward).
// ============================================================================
torch::Tensor gfx906_fa_forward_paged_direct(
    torch::Tensor q,               // fp32  [B, Hq, Sq_pad, D]
    torch::Tensor key_cache_q8,    // uint8 [num_blocks, 16, Hkv, (D/32)*34]
    torch::Tensor value_cache,     // fp16  [num_blocks, 16, Hkv, D]
    torch::Tensor block_table,     // int32 [num_seqs, max_blocks_per_seq]
    torch::Tensor seq_lens,        // int32 [num_seqs]
    double scale,
    c10::optional<torch::Tensor> mask         = c10::nullopt,
    c10::optional<torch::Tensor> q_abs_offset = c10::nullopt,
    int64_t window = 0,
    c10::optional<torch::Tensor> kv_start = c10::nullopt
) {
    TORCH_CHECK(window == 0 || q_abs_offset.has_value(),
        "window > 0 requires q_abs_offset (the window mask needs absolute "
        "query positions)");
    TORCH_CHECK_CUDA(q);
    TORCH_CHECK_CUDA(key_cache_q8);
    TORCH_CHECK_CUDA(value_cache);
    TORCH_CHECK_CUDA(block_table);
    TORCH_CHECK_CUDA(seq_lens);
    TORCH_CHECK_CONTIG(q);

    TORCH_CHECK(q.dtype() == torch::kFloat32, "q must be fp32");
    TORCH_CHECK(key_cache_q8.dtype() == torch::kUInt8, "key_cache_q8 must be uint8");
    TORCH_CHECK(value_cache.dtype() == torch::kFloat16, "value_cache must be fp16");
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, Hq, Sq, D]");
    TORCH_CHECK(key_cache_q8.dim() == 4,
                "key_cache_q8 must be [num_blocks, block_size, Hkv, bytes_per_row]");
    TORCH_CHECK(value_cache.dim() == 4,
                "value_cache must be [num_blocks, block_size, Hkv, D]");

    const int batch    = q.size(0);
    const int heads_q  = q.size(1);
    const int seq_q    = q.size(2);
    const int head_dim = q.size(3);

    const int num_blocks        = key_cache_q8.size(0);
    const int block_size        = key_cache_q8.size(1);
    const int heads_kv          = key_cache_q8.size(2);
    const int bytes_per_row     = key_cache_q8.size(3);
    const int D_val             = value_cache.size(3);

    TORCH_CHECK(value_cache.size(0) == num_blocks, "value_cache blocks mismatch");
    TORCH_CHECK(value_cache.size(1) == block_size, "value_cache block_size mismatch");
    TORCH_CHECK(value_cache.size(2) == heads_kv, "value_cache Hkv mismatch");
    TORCH_CHECK(D_val == head_dim, "value_cache D mismatch q D");
    TORCH_CHECK(bytes_per_row == (head_dim/32)*34,
                "bytes_per_row must equal (D/32)*34");
    TORCH_CHECK(block_size == 16, "Only block_size=16 supported in v1, got ", block_size);

    TORCH_CHECK(block_table.dim() == 2, "block_table must be [num_seqs, max_blocks]");
    TORCH_CHECK(block_table.size(0) == batch, "block_table[0] must equal batch");
    TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == batch,
                "seq_lens must be [batch]");

    auto block_table_c = block_table.is_contiguous() ? block_table : block_table.contiguous();
    auto seq_lens_c    = seq_lens.is_contiguous() ? seq_lens : seq_lens.contiguous();

    const int max_blocks_per_seq = block_table_c.size(1);

    // Strides (bytes) for paged layout. Use the tensors' REAL strides:
    // value_cache is typically kv_cache.unbind(1) of
    // [num_blocks, 2, block_size, Hkv, D] — non-contiguous with a 2x block
    // stride. Shape-derived strides would read K-cache bytes as V.
    const int64_t k_token_stride = key_cache_q8.stride(1);
    const int64_t k_head_stride  = key_cache_q8.stride(2);
    const int64_t k_block_stride = key_cache_q8.stride(0);

    TORCH_CHECK(key_cache_q8.stride(3) == 1,
                "forward_paged_direct: k_q8 last dim must be contiguous");
    TORCH_CHECK(value_cache.stride(3) == 1,
                "forward_paged_direct: v last dim must be contiguous");
    const int64_t v_esize = value_cache.element_size();
    const int64_t v_token_stride = value_cache.stride(1) * v_esize;
    const int64_t v_head_stride  = value_cache.stride(2) * v_esize;
    const int64_t v_block_stride = value_cache.stride(0) * v_esize;

    // max_seq_kv используется kernel'ом ТОЛЬКО как fallback для ne11 когда
    // KV_max==nullptr. Мы всегда передаём kv_max_d, поэтому ne11 фактически
    // не читается. Ставим верхнюю оценку max_blocks_per_seq*block_size —
    // корректно и без D2H sync overhead.
    const int max_seq_kv = max_blocks_per_seq * block_size;

    // Output allocation — тот же BSHD как в forward.
    auto opts_f32 = q.options().dtype(torch::kFloat32);
    torch::Tensor o_bshd = torch::empty({batch, seq_q, heads_q, head_dim}, opts_f32);

    // KV-split: gridDim.y partials merged by gfx906_fa_split_combine — same
    // machinery/guards as gfx906_fa_forward (the seq_q > 2 guard is the same
    // OOM protection: the partial buffer scales with Sq).
    //
    // Default when GFX906_FA_KVSPLIT is unset: fa_paged_kv_split_default
    // (R3) — shape-aware for the Sq>=4 verify shapes (aligned with the
    // gather path), batch clamp for the Sq<4 decode shapes: batch-aware
    // because the paged grid-z is batch*heads_q blocks (NC2=1 here), so
    // the split that fills the 60-CU MI50 moves with the batch (MI50
    // micro-bench, B=4/Hq=32/D=128/L=2816, DEVLOG-muse-glimmer.md
    // 2026-08-27 follow-up: S=8/8/5/2/2 best for B=1/2/3/4/8; the formula
    // matches the measured optimum at B in {1,2,3,8} and is within 4% at
    // B=4). The gather-path default (16) is the WRONG value for those
    // decode shapes: at B=4 it costs +18% (267 vs 226 us) on combine
    // traffic. GFX906_FA_KVSPLIT=1 is the kill switch (old grid.y=1
    // behavior).
    int kv_split = get_fa_kv_split();
    const int tile_clip = get_fa_tile_clip();
    if (kv_split < 0) kv_split = fa_paged_kv_split_default(seq_q, batch);
    // MTP-1b-0: same fix as the gather path — the old `if (seq_q > 2)
    // kv_split = 1` prefill OOM guard also fired for spec-decode verify
    // (Sq = k+1 padded to {2,4,8}), stripping KV-split parallelism from k>=2
    // verify. The byte budget protects against the real OOM case (prefill)
    // while letting small-Sq verify keep kv_split>1.
    kv_split = fa_apply_kv_split_budget(
        kv_split, batch, seq_q, heads_q, head_dim);
    // M3 #4b (LOCKSTEP with gfx906_fa_forward): the dst_meta write is
    // guarded by `gridDim.y != 1` (gridDim.y == kv_split), so the [B, Sq,
    // Hq, 2] meta is dead at kv_split==1 (never written, never read back)
    // and at kv_split>1 the written/read meta is o_meta_split. Pass
    // nullptr in the first case and skip the dead allocation.
    torch::Tensor o_part, o_meta_split;
    float * o_fp32_ptr = o_bshd.data_ptr<float>();
    float2 * o_meta_ptr = nullptr;
    if (kv_split > 1) {
        o_part = torch::empty({batch, seq_q, heads_q, kv_split, head_dim}, opts_f32);
        o_meta_split = torch::empty({batch, seq_q, heads_q, kv_split, 2}, opts_f32);
        o_fp32_ptr = o_part.data_ptr<float>();
        o_meta_ptr = reinterpret_cast<float2 *>(o_meta_split.data_ptr<float>());
    }

    // KV_max expansion [B] → [B, grid_x] (тот же приём что в forward).
    const int ncols1 = fa_pick_ncols1(seq_q);
    const int grid_x = (seq_q + ncols1 - 1) / ncols1;

    auto kv_max_expanded = seq_lens_c.unsqueeze(1).expand({batch, grid_x}).contiguous();

    // Optional mask.
    const __half * mask_ptr = nullptr;
    int32_t mask_seq_kv_padded = 0;
    torch::Tensor mask_contig;
    if (mask.has_value()) {
        auto m = mask.value();
        TORCH_CHECK_CUDA(m);
        TORCH_CHECK(m.dtype() == torch::kFloat16, "mask must be fp16");
        TORCH_CHECK(m.dim() == 3, "mask must be 3D");
        mask_contig = m.contiguous();
        mask_ptr = reinterpret_cast<const __half *>(mask_contig.data_ptr<at::Half>());
        mask_seq_kv_padded = (int32_t) mask_contig.size(2);
    }

    // Optional q_abs_offset (inline causal).
    const int32_t * q_abs_offset_ptr = nullptr;
    torch::Tensor q_abs_offset_c;
    if (q_abs_offset.has_value()) {
        TORCH_CHECK(!mask.has_value(),
            "q_abs_offset and mask are mutually exclusive");
        auto qo = q_abs_offset.value();
        TORCH_CHECK_CUDA(qo);
        TORCH_CHECK(qo.dtype() == torch::kInt32, "q_abs_offset must be int32");
        TORCH_CHECK(qo.dim() == 1 && qo.size(0) == batch,
            "q_abs_offset must be [batch]");
        q_abs_offset_c = qo.contiguous();
        q_abs_offset_ptr = q_abs_offset_c.data_ptr<int32_t>();
    }

    // Optional per-row KV scan start (sliding-window clip, phase C):
    // the kernel k-loop walks [kv_start_row, seq_len_row). Only valid
    // together with q_abs_offset + window (the keys before kv_start must
    // be window-masked anyway, so the output is bit-identical to the
    // full scan); the backend passes it for windowed decode (Sq=1) only.
    const int32_t * kv_start_ptr = nullptr;
    torch::Tensor kv_start_c;
    if (kv_start.has_value()) {
        TORCH_CHECK(q_abs_offset.has_value() && window > 0,
            "kv_start requires q_abs_offset and window > 0");
        auto ks = kv_start.value();
        TORCH_CHECK_CUDA(ks);
        TORCH_CHECK(ks.dtype() == torch::kInt32, "kv_start must be int32");
        TORCH_CHECK(ks.dim() == 1 && ks.size(0) == batch,
            "kv_start must be [batch]");
        kv_start_c = ks.contiguous();
        kv_start_ptr = kv_start_c.data_ptr<int32_t>();
    }

    auto stream = c10::hip::getCurrentHIPStream().stream();
    hipError_t err = gfx906_fa_launch_paged(
        q.data_ptr<float>(),
        key_cache_q8.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(value_cache.data_ptr<at::Half>()),
        block_table_c.data_ptr<int32_t>(),
        kv_max_expanded.data_ptr<int32_t>(),
        o_fp32_ptr,
        o_meta_ptr,
        mask_ptr,
        mask_seq_kv_padded,
        q_abs_offset_ptr,
        window,
        kv_start_ptr,
        tile_clip,
        batch, heads_q, heads_kv, seq_q, max_seq_kv, head_dim,
        block_size, max_blocks_per_seq,
        k_block_stride, k_token_stride, k_head_stride,
        v_block_stride, v_token_stride, v_head_stride,
        (float) scale,
        stream,
        kv_split
    );

    TORCH_CHECK(err == hipSuccess, "gfx906_fa_launch_paged failed: ", hipGetErrorString(err));

    if (kv_split > 1) {
        const int rows = batch * seq_q * heads_q;
        hipError_t cerr = gfx906_fa_split_combine(
            o_part.data_ptr<float>(),
            reinterpret_cast<float2 *>(o_meta_split.data_ptr<float>()),
            o_bshd.data_ptr<float>(),
            rows, kv_split, head_dim, stream);
        TORCH_CHECK(cerr == hipSuccess, "gfx906_fa_split_combine failed: ",
            hipGetErrorString(cerr));
    }

    // Native BSHD [B, Sq, Hq, D] — see gfx906_fa_forward (no transpose copy).
    return o_bshd;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "gfx906 FlashAttention Q8_0 KV — ported from iacopPBK/llama.cpp-gfx906";
    m.def("forward",       &gfx906_fa_forward, "FA forward (Q fp32, K q8_0, V fp16)",
          py::arg("q"), py::arg("k_q8"), py::arg("v_fp16"), py::arg("scale"),
          py::arg("kv_max")        = c10::nullopt,
          py::arg("mask")          = c10::nullopt,
          py::arg("q_abs_offset")  = c10::nullopt,
          py::arg("window")        = 0,
          py::arg("kv_start")      = c10::nullopt,
          py::arg("kv_split")      = c10::nullopt);
    m.def("quantize_q8_0", &quantize_q8_0,
          "Quantize fp16 tensor (last dim D) → block_q8_0 uint8 (device-side)",
          py::arg("k_fp16"));
    m.def("reshape_and_cache_q8", &reshape_and_cache_q8,
          "Quantize new K tokens (fp16) and scatter into paged Q8 KV-cache "
          "by slot_mapping. In-place write to k_cache_q8.",
          py::arg("key"), py::arg("slot_mapping"), py::arg("k_cache_q8"));
    m.def("gather_paged_kv_q8", &gather_paged_kv_q8,
          "Fused gather: paged K_q8 + paged V_fp16 → contiguous BHSD outputs. "
          "Returns [k_out, v_out]. V tail zeroed per seq_lens; K tail unmasked. "
          "k_out/v_out: optional pre-allocated buffers (exact shape match) to "
          "avoid peak VRAM spikes on large Sk. kv_start (M1 clip): per-seq "
          "absolute start; tokens [0, start) are not written (LOCKSTEP with "
          "the persistent gather kernel).",
          py::arg("key_cache_q8"), py::arg("value_cache"),
          py::arg("block_table"), py::arg("seq_lens"), py::arg("Sk"),
          py::arg("k_out") = c10::nullopt,
          py::arg("v_out") = c10::nullopt,
          py::arg("kv_start") = c10::nullopt);
    m.def("gather_paged_kv_fp16", &gather_paged_kv_fp16,
          "LEGACY-path fused gather: paged fp16 K + paged fp16 V -> contiguous "
          "BHSD outputs. Returns [k_out, v_out]. V tail zeroed per seq_lens; "
          "K tail unmasked (FA kernel cuts via kv_max).",
          py::arg("key_cache"), py::arg("value_cache"),
          py::arg("block_table"), py::arg("seq_lens"), py::arg("Sk"),
          py::arg("k_out") = c10::nullopt,
          py::arg("v_out") = c10::nullopt);
    m.def("gather_paged_kv_quant_persistent", &gather_paged_kv_quant_persistent,
          "Persistent grid-stride fused gather+quantize (LEGACY path): "
          "fixed grid (GFX906_FA_PERSIST_GRID), work bounded by the live "
          "seq_lens tensor, valid at every Sk. K bit-equal to "
          "quantize_q8_0(gather_paged_kv_fp16(x)); rows >= seq_len are not "
          "written except GFX906_FA_PERSIST_MARGIN V-zero rows. kv_start "
          "(M1 clip): per-seq absolute start; rows [0, start) are skipped.",
          py::arg("key_cache"), py::arg("value_cache"),
          py::arg("block_table"), py::arg("seq_lens"), py::arg("Sk"),
          py::arg("k_out") = c10::nullopt,
          py::arg("v_out") = c10::nullopt,
          py::arg("kv_start") = c10::nullopt);
    m.def("gather_paged_kv_quantized", &gather_paged_kv_quantized,
          "Stage-2 LEGACY-path fused gather: paged fp16 K + V -> K quantized "
          "in-kernel to q8_0 (bit-equal to quantize_q8_0 of the fp16 gather) "
          "+ V fp16. Returns [k_q8 uint8 [B,Hkv,Sk,(D/32)*34], "
          "v_bhsd fp16 [B,Hkv,Sk,D]]. V tail zeroed; K tail unmasked. "
          "k_out/v_out: optional pre-allocated buffers (exact shape match).",
          py::arg("key_cache"), py::arg("value_cache"),
          py::arg("block_table"), py::arg("seq_lens"), py::arg("Sk"),
          py::arg("k_out") = c10::nullopt,
          py::arg("v_out") = c10::nullopt);
    m.def("kv_split_default",
          [](long seq_q, long batch, bool direct) {
              return direct ? fa_paged_kv_split_default(seq_q, batch)
                            : fa_kv_split_default(seq_q);
          },
          "kv_split chosen when GFX906_FA_KVSPLIT is unset, for the given "
          "shape (direct = paged-direct path rule, gather = gather rule). "
          "Test probe — no GPU work.",
          py::arg("seq_q"), py::arg("batch"), py::arg("direct"));
    m.def("forward_paged_direct", &gfx906_fa_forward_paged_direct,
          "Level 3c: Direct-paged FA (no gather). Reads K/V from paged cache "
          "via block_table indirection. q: fp32 [B, Hq, Sq_pad, D]; output: "
          "fp32 [B, Sq, Hq, D] (native BSHD).",
          py::arg("q"), py::arg("key_cache_q8"), py::arg("value_cache"),
          py::arg("block_table"), py::arg("seq_lens"), py::arg("scale"),
          py::arg("mask")         = c10::nullopt,
          py::arg("q_abs_offset") = c10::nullopt,
          py::arg("window")       = 0,
          py::arg("kv_start")     = c10::nullopt);
}
