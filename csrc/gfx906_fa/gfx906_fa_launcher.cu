// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) Nick — nick413@gmail.com
// SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
//
// Vendored from https://github.com/cassettesgoboom/gfx906-fa-vllm
// (FlashAttention-style custom attention backend for vLLM on AMD gfx906).
//
// gfx906_fa_launcher.cu — host launcher для flash_attn_tile_q8<>
//
// Запускает __global__ kernel flash_attn_tile_q8<DKQ, DV, ncols1, ncols2, use_logit_softcap>
// на паре torch::Tensor Q/K/V + metadata → Output.
//
// Ограничения:
//   - DKQ = DV = 64/96/128/256 (instantiated; 96 serves the ViT pad-72 and
//     the head_dim-96 text class — see docs/gfx906/DEVLOG-fa-d96.md)
//   - use_logit_softcap = false, без sinks
//   - Causal — inline в kernel (Q_ABS_OFFSET), KV_max — per-seq cut
//   - Direct paged: K/V читаются из paged cache через block table
//     (kernel/fattn-q8-paged.cuh); gather-путь — K q8_0 pre-quantized
//     (planar row — plan_fa_part_A.md)
//   - Q — fp32 contiguous
//   - Output — native BSHD [B, Sq, Hq, D] (transpose в host API убран)
//
// Prefill dispatcher (t6b): cols_per_block выбирается по Sq как в llama.cpp
// (fa_pick_ncols1 в gfx906_fa.cpp; зеркало в Python _pick_ncols1):
//   Sq >  32 → 64, >16 → 32, >8 → 16, >4 → 8, >2 → 4, else 2.
//
// Раскладка tensors (как ggml):
//   Q: [batch, heads_q, seq_q, head_dim]        float32, contiguous
//   K: [batch, heads_kv, seq_kv, (D/QK8_0)*34] q8_0 planar row (plan_fa_part_A.md)
//   V: [batch, heads_kv, seq_kv, head_dim]       float16, contiguous
//   O: [batch, seq_q, heads_q, head_dim]         float32 (output, BSHD)

// КРИТИЧНО: torch cpp_extension форсит -D__HIP_NO_HALF_OPERATORS__=1 и
// -D__HIP_NO_HALF_CONVERSIONS__=1 в cmdline. Эти defines ломают fattn-q8.cuh
// (там `half2 z[N] = {{0.0f, 0.0f}}`, `h2 *= h2`, implicit float→half).
// Снимаем ДО включения любых ROCm-заголовков.
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif
#ifdef __HIP_NO_HALF2_OPERATORS__
#undef __HIP_NO_HALF2_OPERATORS__
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ВАЖНО: shim ставит все defines (GGML_USE_HIP, GGML_HIP_GFX906, WARP_SIZE=64, FLASH_ATTN_AVAILABLE)
// ДО включения fattn-q8.cuh
#include "ggml_shim.cuh"
#include "fattn-q8.cuh"
#include "fattn-q8-paged.cuh"

#include <cstdio>
#include <cstdint>
#include <type_traits>

// ============================================================================
// Entry point из C++/pybind11: C-linkage удобнее для диагностики
// ============================================================================
// Templated FA launch. HD is the compile-time head dimension (64/96/128/256).
template <int HD>
static hipError_t gfx906_fa_launch_impl(
    const float *      Q_fp32,
    const void  *      K_q8,
    const __half *     V_f16,
    float *            O_fp32,
    float2 *           O_meta,
    const int *        KV_max_d,
    const __half *     MASK_f16,
    int32_t            mask_seq_kv_padded,
    const int32_t *    Q_ABS_OFFSET_d,
    int                window,
    // M1 gather-path window clip start [B] (see flash_attn_tile_q8);
    // nullptr = full scan. Mirrors the paged launcher's KV_START_d.
    const int32_t *    KV_START_d,
    // M2 tile clip (1 = on; host knob GFX906_FA_TILE_CLIP; see the kernel).
    int                tile_clip,
    int                batch,
    int                heads_q,
    int                heads_kv,
    int                seq_q,
    int                seq_kv,
    float              scale,
    hipStream_t        stream,
    int                nc2,
    int                kv_split
) {
    constexpr int DKQ = HD;
    constexpr int DV  = HD;
    constexpr bool use_logit_softcap = false;

    if (heads_q % heads_kv != 0) {
        fprintf(stderr, "[gfx906_fa] heads_q=%d must be divisible by heads_kv=%d\n", heads_q, heads_kv);
        return hipErrorInvalidValue;
    }
    // GQA head-packing (ncols2) and KV-split (gridDim.y) are host-tunable.
    // The legacy NC2=1/y=1 config launches only heads_q blocks at B=1 decode
    // (16 for Hq=16) = 64/960 wavefront slots; packing GQA heads and/or
    // splitting the KV range restores parallelism. y>1 partials are merged
    // by the caller via gfx906_fa_split_combine (O unscaled, meta=(m,l)).
    if (nc2 <= 1) {
        nc2 = 1;
    } else {
        // A packed tile shares ONE kv head (K/V base = head0 / gqa_ratio),
        // so a tile must not straddle GQA groups: gqa_ratio % nc2 == 0.
        // heads_q % heads_kv == 0 is guaranteed above, so gqa_ratio is
        // integral and gqa_ratio % nc2 == 0 already implies heads_q % nc2
        // == 0 -- the old bare `heads_q % nc2` guard was redundant and
        // aborted *before* the default downgrade could run, crashing any
        // per-shard head count not divisible by 8 (e.g. heads_q=6 for a
        // 12-head model at TP=2, or Qwen3.5-27B's 24 heads at TP=4).
        // Only NC2 in {1, 2, 8} are instantiated in
        // the dispatch below; any other value is rejected loudly (clamping
        // to a non-instantiated value would still run the NC2=8 kernel and
        // silently mispack).
        // The DEFAULT nc2=8 auto-downgrades 8 -> 2 -> 1 so GQA ratios like
        // 6 (or 1, MHA) keep a valid path; an explicit env value that is
        // GQA-invalid is an error.
        const int gqa_ratio = heads_q / heads_kv;
        if (nc2 != 1 && nc2 != 2 && nc2 != 8) {
            fprintf(stderr, "[gfx906_fa] nc2=%d unsupported (instantiated: 1, 2, 8)\n", nc2);
            return hipErrorInvalidValue;
        }
        if (gqa_ratio % nc2 != 0) {
            if (nc2 == 8) {
                static bool warned = false;
                const int down = (gqa_ratio % 2 == 0) ? 2 : 1;
                if (!warned) {
                    fprintf(stderr, "[gfx906_fa] gqa_ratio=%d not divisible by "
                            "nc2=8, using nc2=%d\n", gqa_ratio, down);
                    warned = true;
                }
                nc2 = down;
            } else {
                fprintf(stderr, "[gfx906_fa] nc2=%d invalid for gqa_ratio=%d "
                        "(need ratio %% nc2 == 0; use GFX906_FA_NC2=1 or 2)\n",
                        nc2, gqa_ratio);
                return hipErrorInvalidValue;
            }
        }
    }
    if (kv_split < 1) {
        kv_split = 1;
    }
    // NC2>1 (GQA head-packing) is only validated at the decode tile
    // (seq_q <= 2 -> ncols = 2*ncols2 <= 16); prefill (larger Sq) keeps
    // the legacy NC2=1 path (the ncols=64 NC2=8 config faults on OOB).
    if (nc2 > 1 && seq_q > 2) {
        nc2 = 1;
    }

    // nb* computed in BYTES (ggml convention)
    const int32_t nb00 = sizeof(float);
    const int32_t nb01 = nb00 * HD;
    const int32_t nb02 = nb01 * seq_q;
    const int32_t nb03 = nb02 * heads_q;

    // K is planar q8_0 (plan_fa_part_A.md): row = D quants bytes + (D/32)
    // fp16 scales = (D/32)*34 bytes — same row byte count as block_q8_0.
    const int32_t nb10 = sizeof(block_q8_0);                 // per block
    const int32_t nb11 = nb10 * (HD / QK8_0);          // per K-token row
    const int32_t nb12 = nb11 * seq_kv;                      // per head
    const int64_t nb13 = (int64_t) nb12 * heads_kv;          // per batch

    // V is fp16
    const int32_t nb20 = sizeof(__half);
    const int32_t nb21 = nb20 * HD;
    const int32_t nb22 = nb21 * seq_kv;
    const int64_t nb23 = (int64_t) nb22 * heads_kv;

    // mask layout: [batch, Sq, mask_seq_kv_padded] fp16 (one plane per batch).
    // ne31 — Sq; ne32 — 1; ne33 — batch; nb31 — stride по Sq в байтах;
    // nb32 — stride по «heads» (не используем → 0); nb33 — stride по sequence.
    const int32_t ne31 = MASK_f16 ? seq_q : 0;
    const int32_t ne32 = MASK_f16 ? 1     : 0;
    const int32_t ne33 = MASK_f16 ? batch : 1;          // %ne33 → не 0
    const int32_t nb31 = MASK_f16 ? (int32_t)(mask_seq_kv_padded * sizeof(__half)) : 0;
    const int32_t nb32 = 0;
    const int64_t nb33 = MASK_f16 ? (int64_t)seq_q * nb31 : 0;

    // ne shapes
    const int32_t ne00 = HD;
    // ВАЖНО: kernel использует только ne01.z (где ggml хранит оригинальный divisor,
    // см. init_fastdiv_values в common.cuh: uint3 = (mp, L, d)).
    // Проверено: в fattn-q8.cuh нет вызовов fastdiv/fastmodulo с ne01,
    // так что .x/.y можно оставить нулями, но .z ДОЛЖНО быть = seq_q.
    const uint3   ne01 = make_uint3(0u, 0u, (unsigned) seq_q);
    const int32_t ne02 = heads_q;
    const int32_t ne03 = batch;

    const int32_t ne10 = HD;
    const int32_t ne11 = seq_kv;
    const int32_t ne12 = heads_kv;
    const int32_t ne13 = batch;

    // ------------------------------------------------------------------
    // Kernel dispatch: выбор cols_per_block (ncols1) в зависимости от Sq
    // ------------------------------------------------------------------
    //
    // Оригинал llama.cpp (launch_fattn_tile_q8_switch_ncols1, fattn-q8.cuh:846):
    //   Sq >  32  → ncols1 = 64
    //   Sq >  16  → ncols1 = 32
    //   Sq >   8  → ncols1 = 16
    //   Sq >   4  → ncols1 =  8
    //   Sq >   2  → ncols1 =  4
    //   Sq <= 2   → ncols1 =  2
    //
    // Для DKQ=DV=128 таблица (см. fattn-q8.cuh:55-60):
    //   ncols=2  → nthreads=256
    //   ncols=4  → nthreads=128   ← обратите внимание
    //   ncols=8  → nthreads=256
    //   ncols=16 → nthreads=256
    //   ncols=32 → nthreads=256
    //   ncols=64 → nthreads=256
    //
    // NC2 = 1 всегда (mask=nullptr → GQA-packing отключён).
    //
    // Lambda-макрос для DRY — все инстанциации идентичны кроме NC1.

    auto launch = [&](auto NC1_tag, auto NC2_tag) {
        constexpr int NC1 = decltype(NC1_tag)::value;
        constexpr int NC2 = decltype(NC2_tag)::value;
        dim3 grid(
            /*x=*/ (seq_q + NC1 - 1) / NC1,
            /*y=*/ kv_split,
            /*z=*/ batch * ((heads_q + NC2 - 1) / NC2)
        );
        const int nthreads = ggml_cuda_fattn_tile_q8_get_nthreads(DKQ, DV, NC1 * NC2, /*cc=*/0);
        dim3 block(32 /* warp_size */, nthreads / 32 /* nwarps */, 1);

        flash_attn_tile_q8<DKQ, DV, NC1, NC2, use_logit_softcap><<<grid, block, 0, stream>>>(
            (const char *) Q_fp32,
            (const char *) K_q8,
            (const char *) V_f16,
            /*mask=*/   (const char *) MASK_f16,
            /*sinks=*/  (const char *) nullptr,
            /*KV_max=*/ KV_max_d,
            /*q_abs_offset=*/ Q_ABS_OFFSET_d,
            /*window=*/ window,
            /*kv_start=*/ KV_START_d,
            /*tile_clip=*/ tile_clip,
            O_fp32,
            O_meta,
            scale,
            /*max_bias=*/ 0.0f,
            /*m0=*/ 1.0f, /*m1=*/ 1.0f,
            /*n_head_log2=*/ 0u,
            /*logit_softcap=*/ 0.0f,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
            ne31, ne32, ne33,
                  nb31, nb32, nb33
        );
    };

    // std::integral_constant trick — передаём compile-time int в lambda
    using T2  = std::integral_constant<int,  2>;
    using T4  = std::integral_constant<int,  4>;
    using T8  = std::integral_constant<int,  8>;
    using T16 = std::integral_constant<int, 16>;
    using T32 = std::integral_constant<int, 32>;
    using T64 = std::integral_constant<int, 64>;
    using C1 = std::integral_constant<int, 1>;
    using C2 = std::integral_constant<int, 2>;
    using C8 = std::integral_constant<int, 8>;

    // ncols1 ladder (llama.cpp). For NC2=8 the ladder caps ncols1 at 8 so
    // ncols = ncols1*ncols2 stays <= 64, the config-table maximum (no rows
    // for 128..512); larger Sq simply loses some row-packing.
    auto dispatch1 = [&](const auto &tag) {
        if      (seq_q > 32) launch(T64{}, tag);
        else if (seq_q > 16) launch(T32{}, tag);
        else if (seq_q >  8) launch(T16{}, tag);
        else if (seq_q >  4) launch(T8{},  tag);
        else if (seq_q >  2) launch(T4{},  tag);
        else                 launch(T2{},  tag);
    };
    auto dispatch2 = [&](const auto &tag) {
        // ncols = NC1*2; cap NC1 at 32 (ncols <= 64, config-table max).
        if      (seq_q > 32) launch(T32{}, tag);
        else if (seq_q > 16) launch(T16{}, tag);
        else if (seq_q >  8) launch(T8{},  tag);
        else if (seq_q >  4) launch(T4{},  tag);
        else                 launch(T2{},  tag);
    };
    auto dispatch8 = [&](const auto &tag) {
        if      (seq_q >  8) launch(T8{},  tag);
        else if (seq_q >  4) launch(T4{},  tag);
        else if (seq_q >  2) launch(T4{},  tag);
        else                 launch(T2{},  tag);
    };
    if (nc2 == 1) dispatch1(C1{});
    else if (nc2 == 2) dispatch2(C2{});
    else                dispatch8(C8{});

    return hipGetLastError();
}

// ============================================================================
// Entry point: dispatch head_dim -> templated kernel launch.
// ============================================================================
extern "C" hipError_t gfx906_fa_launch(
    const float *      Q_fp32,
    const void  *      K_q8,
    const __half *     V_f16,
    float *            O_fp32,
    float2 *           O_meta,
    const int *        KV_max_d,
    const __half *     MASK_f16,
    int32_t            mask_seq_kv_padded,
    const int32_t *    Q_ABS_OFFSET_d,
    int                window,
    const int32_t *    KV_START_d,
    int                tile_clip,
    int                batch,
    int                heads_q,
    int                heads_kv,
    int                seq_q,
    int                seq_kv,
    int                head_dim,
    float              scale,
    hipStream_t        stream,
    int                nc2,
    int                kv_split
) {
    if      (head_dim == 128) return gfx906_fa_launch_impl<128>(Q_fp32, K_q8, V_f16, O_fp32, O_meta, KV_max_d, MASK_f16, mask_seq_kv_padded, Q_ABS_OFFSET_d, window, KV_START_d, tile_clip, batch, heads_q, heads_kv, seq_q, seq_kv, scale, stream, nc2, kv_split);
    else if (head_dim == 256) return gfx906_fa_launch_impl<256>(Q_fp32, K_q8, V_f16, O_fp32, O_meta, KV_max_d, MASK_f16, mask_seq_kv_padded, Q_ABS_OFFSET_d, window, KV_START_d, tile_clip, batch, heads_q, heads_kv, seq_q, seq_kv, scale, stream, nc2, kv_split);
    else if (head_dim == 96)  return gfx906_fa_launch_impl<96> (Q_fp32, K_q8, V_f16, O_fp32, O_meta, KV_max_d, MASK_f16, mask_seq_kv_padded, Q_ABS_OFFSET_d, window, KV_START_d, tile_clip, batch, heads_q, heads_kv, seq_q, seq_kv, scale, stream, nc2, kv_split);
    else if (head_dim == 64)  return gfx906_fa_launch_impl<64> (Q_fp32, K_q8, V_f16, O_fp32, O_meta, KV_max_d, MASK_f16, mask_seq_kv_padded, Q_ABS_OFFSET_d, window, KV_START_d, tile_clip, batch, heads_q, heads_kv, seq_q, seq_kv, scale, stream, nc2, kv_split);
    fprintf(stderr, "[gfx906_fa] Unsupported head_dim=%d (supported: 64, 96, 128, 256)\n", head_dim);
    return hipErrorInvalidValue;
}

// ============================================================================
// KV-split (gridDim.y > 1) combine. The FA kernel writes per-split partials:
//   O_part: [rows, y, D] fp32, unscaled (no 1/l division when gridDim.y>1)
//   meta:   [rows, y, 2] fp32, (m = running max, l = running sum)
// where rows = B * Sq * Hq (row-major over (sequence, sq, head)).
// Merge: m* = max_p m_p; out[d] = sum_p exp(m_p - m*) * O_p[d] /
//        sum_p exp(m_p - m*) * l_p. Empty splits (m_p = -FLT_MAX/2, l_p=0)
// contribute weight exp(-inf) = 0; all-empty rows (kv_max=0) are guarded.
// One warp per row; D/4 float4 elements, 2 float4 per thread at D=256.
// ============================================================================
__global__ void fa_split_combine_kernel(
        const float4 * __restrict__ O_part,
        const float2 * __restrict__ meta,
        float4 * __restrict__ O_out,
        const int y,
        const int D) {
    const int row = blockIdx.x;
    const int d4  = threadIdx.x;            // float4 index
    const int D4  = D / 4;

    // m* is row-wide; all warp lanes compute it redundantly (same row).
    float m_star = -FLT_MAX / 2.0f;
    for (int p = 0; p < y; ++p) {
        m_star = fmaxf(m_star, meta[row * y + p].x);
    }

    for (int i = d4; i < D4; i += blockDim.x) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        float l_star = 0.f;
        for (int p = 0; p < y; ++p) {
            const float2 mp = meta[row * y + p];
            const float w = __expf(mp.x - m_star);   // 0 for empty splits
            if (w == 0.f) continue;
            const float4 op = O_part[(row * y + p) * D4 + i];
            acc.x += w * op.x;
            acc.y += w * op.y;
            acc.z += w * op.z;
            acc.w += w * op.w;
            l_star += w * mp.y;
        }
        if (l_star == 0.f) l_star = 1.f;   // degenerate: kv_max == 0 row
        const float inv_l = 1.f / l_star;
        acc.x *= inv_l;
        acc.y *= inv_l;
        acc.z *= inv_l;
        acc.w *= inv_l;
        O_out[row * D4 + i] = acc;
    }
}

extern "C" hipError_t gfx906_fa_split_combine(
        float  *O_part,    // [rows, y, D] fp32
        float2 *meta,      // [rows, y, 2] fp32
        float  *O_out,     // [rows, D] fp32
        int     rows,
        int     y,
        int     D,
        hipStream_t stream)
{
    if (y <= 1) {
        // No split: kernel already wrote the final answer in O_part layout
        // [rows, 1, D] == [rows, D].
        if (O_part != O_out) {
            hipError_t e = hipMemcpyAsync(O_out, O_part,
                (size_t) rows * D * sizeof(float), hipMemcpyDeviceToDevice, stream);
            if (e != hipSuccess) return e;
        }
        return hipSuccess;
    }
    const int D4 = D / 4;
    const int threads = D4 < 32 ? D4 : 32;
    dim3 grid(rows), block(threads);
    fa_split_combine_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float4 *>(O_part),
        reinterpret_cast<const float2 *>(meta),
        reinterpret_cast<float4 *>(O_out), y, D);
    return hipGetLastError();
}

// Forward declaration for the templated paged launcher (defined below).
template <int HD>
static hipError_t gfx906_fa_launch_paged_impl(
    const float *      Q_fp32,
    const void  *      K_paged,
    const __half *     V_paged,
    const int32_t *    block_table,
    const int32_t *    kv_max_d,
    float *            O_fp32,
    float2 *           O_meta,
    const __half *     MASK_f16,
    int32_t            mask_seq_kv_padded,
    const int32_t *    Q_ABS_OFFSET_d,
    int                window,
    const int32_t *    KV_START_d,
    int                tile_clip,
    int                batch,
    int                heads_q,
    int                heads_kv,
    int                seq_q,
    int                max_seq_kv,
    int                block_size,
    int                max_blocks_per_seq,
    int64_t            k_block_stride,
    int64_t            k_token_stride,
    int64_t            k_head_stride,
    int64_t            v_block_stride,
    int64_t            v_token_stride,
    int64_t            v_head_stride,
    float              scale,
    hipStream_t        stream,
    int                kv_split
);

// ============================================================================
// No gather. K/V read directly from paged KV cache via block_table.
// Layout (vLLM compat, block_size=16):
//   K_paged: [num_blocks, 16, Hkv, (D/32)*34]  uint8
//   V_paged: [num_blocks, 16, Hkv,  D        ]  fp16
//   block_table: [num_seqs, max_blocks_per_seq]  int32
//
// Strides пересчитываются host'ом в bytes (как в gather path).
// ============================================================================
extern "C" hipError_t gfx906_fa_launch_paged(
    const float *      Q_fp32,
    const void  *      K_paged,           // uint8, [num_blocks, bs, Hkv, bpr]
    const __half *     V_paged,           // fp16,  [num_blocks, bs, Hkv, D]
    const int32_t *    block_table,       // [num_seqs, max_blocks_per_seq]
    const int32_t *    kv_max_d,          // [B, grid_x] already expanded by caller
    float *            O_fp32,            // [B, Sq_pad, Hq, D]  (BSHD)
    float2 *           O_meta,            // [B, Sq_pad, Hq, 2]
    const __half *     MASK_f16,
    int32_t            mask_seq_kv_padded,
    const int32_t *    Q_ABS_OFFSET_d,
    int                window,
    const int32_t *    KV_START_d,
    int                tile_clip,
    int                batch,
    int                heads_q,
    int                heads_kv,
    int                seq_q,
    int                max_seq_kv,        // max(seq_lens) — для ne11
    int                head_dim,
    int                block_size,
    int                max_blocks_per_seq,
    int64_t            k_block_stride,
    int64_t            k_token_stride,
    int64_t            k_head_stride,
    int64_t            v_block_stride,
    int64_t            v_token_stride,
    int64_t            v_head_stride,
    float              scale,
    hipStream_t        stream,
    int                kv_split
) {
    if      (head_dim == 128) return gfx906_fa_launch_paged_impl<128>(Q_fp32, K_paged, V_paged, block_table, kv_max_d, O_fp32, O_meta, MASK_f16, mask_seq_kv_padded, Q_ABS_OFFSET_d, window, KV_START_d, tile_clip, batch, heads_q, heads_kv, seq_q, max_seq_kv, block_size, max_blocks_per_seq, k_block_stride, k_token_stride, k_head_stride, v_block_stride, v_token_stride, v_head_stride, scale, stream, kv_split);
    else if (head_dim == 256) return gfx906_fa_launch_paged_impl<256>(Q_fp32, K_paged, V_paged, block_table, kv_max_d, O_fp32, O_meta, MASK_f16, mask_seq_kv_padded, Q_ABS_OFFSET_d, window, KV_START_d, tile_clip, batch, heads_q, heads_kv, seq_q, max_seq_kv, block_size, max_blocks_per_seq, k_block_stride, k_token_stride, k_head_stride, v_block_stride, v_token_stride, v_head_stride, scale, stream, kv_split);
    else if (head_dim == 96)  return gfx906_fa_launch_paged_impl<96> (Q_fp32, K_paged, V_paged, block_table, kv_max_d, O_fp32, O_meta, MASK_f16, mask_seq_kv_padded, Q_ABS_OFFSET_d, window, KV_START_d, tile_clip, batch, heads_q, heads_kv, seq_q, max_seq_kv, block_size, max_blocks_per_seq, k_block_stride, k_token_stride, k_head_stride, v_block_stride, v_token_stride, v_head_stride, scale, stream, kv_split);
    else if (head_dim == 64)  return gfx906_fa_launch_paged_impl<64> (Q_fp32, K_paged, V_paged, block_table, kv_max_d, O_fp32, O_meta, MASK_f16, mask_seq_kv_padded, Q_ABS_OFFSET_d, window, KV_START_d, tile_clip, batch, heads_q, heads_kv, seq_q, max_seq_kv, block_size, max_blocks_per_seq, k_block_stride, k_token_stride, k_head_stride, v_block_stride, v_token_stride, v_head_stride, scale, stream, kv_split);
    fprintf(stderr, "[gfx906_fa_paged] Unsupported head_dim=%d (supported: 64, 96, 128, 256)\n", head_dim);
    return hipErrorInvalidValue;
}
// Templated paged FA launch. HD is the compile-time head dimension (64/96/128/256).
template <int HD>
static hipError_t gfx906_fa_launch_paged_impl(
    const float *      Q_fp32,
    const void  *      K_paged,
    const __half *     V_paged,
    const int32_t *    block_table,
    const int32_t *    kv_max_d,
    float *            O_fp32,
    float2 *           O_meta,
    const __half *     MASK_f16,
    int32_t            mask_seq_kv_padded,
    const int32_t *    Q_ABS_OFFSET_d,
    int                window,
    const int32_t *    KV_START_d,
    int                tile_clip,
    int                batch,
    int                heads_q,
    int                heads_kv,
    int                seq_q,
    int                max_seq_kv,
    int                block_size,
    int                max_blocks_per_seq,
    int64_t            k_block_stride,
    int64_t            k_token_stride,
    int64_t            k_head_stride,
    int64_t            v_block_stride,
    int64_t            v_token_stride,
    int64_t            v_head_stride,
    float              scale,
    hipStream_t        stream,
    int                kv_split
) {
    if (heads_q % heads_kv != 0) {
        fprintf(stderr, "[gfx906_fa_paged] heads_q=%d not divisible by heads_kv=%d\n",
                heads_q, heads_kv);
        return hipErrorInvalidValue;
    }
    // KV-split (GFX906_FA_KVSPLIT, gridDim.y): same strided-split +
    // split_combine machinery as the gather path. The paged kernel's k-loop
    // already strides by gridDim.y*nbatch_fa and stores unscaled partials
    // [rows, y, D] + meta (m, l) when gridDim.y>1 — only the host plumbing
    // was missing (grid was pinned to y=1), which left every block with the
    // FULL kv k-loop as its critical path (B=4 Muse decode: 5.4x step time
    // for 4x the work, DEVLOG-muse-glimmer.md 2026-08-27).
    if (kv_split < 1) {
        kv_split = 1;
    }
    if (block_size != 16) {
        fprintf(stderr, "[gfx906_fa_paged] Only block_size=16 supported, got %d\n", block_size);
        return hipErrorInvalidValue;
    }

    constexpr int DKQ = HD;
    constexpr int DV  = HD;
    constexpr bool use_logit_softcap = false;

    // Q layout (contiguous, same as non-paged).
    const int32_t nb00 = sizeof(float);
    const int32_t nb01 = nb00 * HD;
    const int32_t nb02 = nb01 * seq_q;
    const int32_t nb03 = nb02 * heads_q;

    const int32_t ne31 = MASK_f16 ? seq_q : 0;
    const int32_t ne32 = MASK_f16 ? 1     : 0;
    const int32_t ne33 = MASK_f16 ? batch : 1;
    const int32_t nb31 = MASK_f16 ? (int32_t)(mask_seq_kv_padded * sizeof(__half)) : 0;
    const int32_t nb32 = 0;
    const int64_t nb33 = MASK_f16 ? (int64_t)seq_q * nb31 : 0;

    const int32_t ne00 = HD;
    const uint3   ne01 = make_uint3(0u, 0u, (unsigned) seq_q);
    const int32_t ne02 = heads_q;
    const int32_t ne03 = batch;

    const int32_t ne10 = HD;
    const int32_t ne11 = max_seq_kv;
    const int32_t ne12 = heads_kv;
    const int32_t ne13 = batch;

    constexpr int NC2 = 1;
    const int ntiles_z = (heads_q + NC2 - 1) / NC2;

    auto launch = [&](auto NC1_tag, int nthreads) {
        constexpr int NC1 = decltype(NC1_tag)::value;
        const int grid_x = (seq_q + NC1 - 1) / NC1;

        dim3 grid(
            /*x=*/ grid_x,
            /*y=*/ kv_split,
            /*z=*/ batch * ntiles_z
        );
        dim3 block(32, nthreads / 32, 1);

        flash_attn_tile_q8_paged<DKQ, DV, NC1, NC2, use_logit_softcap><<<grid, block, 0, stream>>>(
            (const char *) Q_fp32,
            (const char *) K_paged,
            (const char *) V_paged,
            /*mask=*/   (const char *) MASK_f16,
            /*sinks=*/  (const char *) nullptr,
            /*KV_max=*/ kv_max_d,
            Q_ABS_OFFSET_d,
            window,
            KV_START_d,
            tile_clip,
            block_table,
            O_fp32,
            O_meta,
            scale,
            /*max_bias=*/ 0.0f,
            /*m0=*/ 1.0f, /*m1=*/ 1.0f,
            /*n_head_log2=*/ 0u,
            /*logit_softcap=*/ 0.0f,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
            ne31, ne32, ne33,
                  nb31, nb32, nb33,
            max_blocks_per_seq,
            k_block_stride, k_token_stride, k_head_stride,
            v_block_stride, v_token_stride, v_head_stride
        );
    };

    using T2  = std::integral_constant<int,  2>;
    using T4  = std::integral_constant<int,  4>;
    using T8  = std::integral_constant<int,  8>;
    using T16 = std::integral_constant<int, 16>;
    using T32 = std::integral_constant<int, 32>;
    using T64 = std::integral_constant<int, 64>;

    // Block dim comes from the config-derived thread count so it matches the
    // kernel's __launch_bounds__ (which uses paged_get_nthreads). For ncols=64
    // (D=128) that is 512 -- the least-spill variant of an inherently
    // register-heavy, benchmark-only config. See fattn-q8-paged.cuh.
    using LD2  = std::integral_constant<int,  2>;
    using LD4  = std::integral_constant<int,  4>;
    using LD8  = std::integral_constant<int,  8>;
    using LD16 = std::integral_constant<int, 16>;
    using LD32 = std::integral_constant<int, 32>;
    using LD64 = std::integral_constant<int, 64>;

    if      (seq_q > 32) launch(LD64{}, ggml_cuda_fattn_tile_q8_paged_get_nthreads(DKQ, DV, 64));
    else if (seq_q > 16) launch(LD32{}, ggml_cuda_fattn_tile_q8_paged_get_nthreads(DKQ, DV, 32));
    else if (seq_q >  8) launch(LD16{}, ggml_cuda_fattn_tile_q8_paged_get_nthreads(DKQ, DV, 16));
    else if (seq_q >  4) launch(LD8{},  ggml_cuda_fattn_tile_q8_paged_get_nthreads(DKQ, DV,  8));
    else if (seq_q >  2) launch(LD4{},  ggml_cuda_fattn_tile_q8_paged_get_nthreads(DKQ, DV,  4));
    else                 launch(LD2{},  ggml_cuda_fattn_tile_q8_paged_get_nthreads(DKQ, DV,  2));

    return hipGetLastError();
}
