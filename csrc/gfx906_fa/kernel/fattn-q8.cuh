// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) Nick — nick413@gmail.com
//
// Vendored from https://github.com/cassettesgoboom/gfx906-fa-vllm
// (FlashAttention-style custom attention backend for vLLM on AMD gfx906).
//
// Flash Attention with Q8_0 quantized KV cache for GFX906 (portable, no ggml)
// Uses v_dot4_i32_i8 for INT8 dot products

#pragma once
#include "ggml_shim.cuh"
#include "gfx906-config.h"

// Uses v_dot4_i32_i8 for INT8 dot products


// nbatch_fa is capped at GFX906_FA_GATHER_CLIP_MARGIN (not just the 256
// hardware ceiling below): the M1 gather-path window clip
// (gfx906_fa_gather.cu, GATHER_CLIP_MARGIN) only materializes enough
// history for the FA kernel's tile-boundary floor (this file, k0_base -=
// k0_base % nbatch_fa) to stay in-bounds if nbatch_fa never exceeds the
// margin. Raising nbatch_fa past GFX906_FA_GATHER_CLIP_MARGIN here without
// also raising the margin would silently under-cover it at runtime (stale
// gather-buffer rows read as K); this assert turns that into a build
// failure instead.
#define GGML_CUDA_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
        static_assert((nbatch_fa)         <= GFX906_FA_GATHER_CLIP_MARGIN,                            \
            "nbatch_fa exceeds GFX906_FA_GATHER_CLIP_MARGIN (gfx906-config.h) — "                     \
            "the M1 gather-path clip's margin no longer covers this tile's "                          \
            "boundary floor; raise GFX906_FA_GATHER_CLIP_MARGIN to match");                           \
        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
        return ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
    }

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_q8_get_config_amd(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2,  64, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 128, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 1,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 64, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2, 256, 2, 128,  96)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  64,  96)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  64,  96)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 1,  64,  96)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 1,  64,  96)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 64, 256, 1,  64,  96)
    // The inherited llama.cpp row here was (nbatch_fa 32, nbatch_K 48). Both
    // sides of this Q8 kernel constrain that pair, and 48 satisfies neither
    // contract:
    //   * K is consumed in whole q8_0 blocks (blocks_per_K_row = nbatch_K/32),
    //     so nbatch_K must be a multiple of 32 — 48 scored only 64 of 96 dims
    //     (silently wrong numerics; the static_asserts now catch it);
    //   * nbatch_V = nbatch_K*nbatch_fa/DV must divide nbatch_fa (the V tile
    //     loader), which needs DV % nbatch_K == 0.
    // For DV=96 the only pairs satisfying both are (nbatch_K 96, nbatch_fa any
    // multiple of 32) and (32, nbatch_fa multiple of 96). nbatch_K=96 keeps
    // num_K_tiles=1 (one 3-block K tile) and nbatch_V=nbatch_fa, so the
    // nbatch_fa/nthreads/occupancy columns mirror the D=128 rows — a first cut
    // measured the inherited nbatch_fa=32 at +6.7 % on the ViT shape and +277 %
    // on a decode shape, i.e. the fa column had to be retuned, not just K.

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 64, 256, 2,  32,  56)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 256, 1,  64, 128)
    // gfx906 tuning: occupancy pinned to 1 for the large prefill tiles so the
    // compiler may use ~256 VGPR/lane (vs ~128 at occ=2). The paged kernel
    // already overrides these to occ=1; setting it in the base table fixes the
    // FLAT path spills (D=128 ncols=16 was 336B, ncols=32 224B, ncols=64 48B).
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 1,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 1,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 64, 256, 1,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 256, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 1,  64, 128)
    // gfx906 tuning: occupancy 1 (see D=128 note above). Eliminates flat-path
    // spills for D=256 ncols=16 (144B) / ncols=32 (496B). ncols=64 already
    // runs at natural occ=1 (242 VGPR) but we set it explicitly for clarity.
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 1,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 1,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 64, 256, 1,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 32, 512, 1, 128,  64)

    return 0;
}

static __host__ uint32_t ggml_cuda_fattn_tile_q8_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_tile_q8_get_config_amd(DKQ, DV, ncols);
}

static constexpr __device__ uint32_t ggml_cuda_fattn_tile_q8_get_config(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_tile_q8_get_config_amd(DKQ, DV, ncols);
}

static __host__ int ggml_cuda_fattn_tile_q8_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols, cc) >> 0) & ((1 << 10) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_q8_get_nthreads(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
}

static __host__ int ggml_cuda_fattn_tile_q8_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols, cc) >> 10) & ((1 << 4) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_q8_get_occupancy(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
}

static __host__ int ggml_cuda_fattn_tile_q8_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols, cc) >> 14) & ((1 << 9) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_q8_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
}

static __host__ int ggml_cuda_fattn_tile_q8_get_nbatch_K(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols, cc) >> 23) & ((1 << 9) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_q8_get_nbatch_K(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
}

template<int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_q8_q8_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int stride_KV, const int i_sup) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    auto load = [&] __device__ (const int n) {
        const int stride_j = warp_size >> n;

        if (stride_j == 0) {
            return;
        }

        const int j0_start = stride_j == warp_size ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        const int j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        const int stride_i = warp_size / stride_j;

        if (j0_start == j0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_j == warp_size ? 0 : threadIdx.x / stride_j);

            if (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    const int j = j0*cpy_ne + (stride_j == warp_size ? threadIdx.x : threadIdx.x % stride_j)*cpy_ne;

                    const half2 zero[cpy_ne] = {{0.0f, 0.0f}};
                    ggml_cuda_memcpy_1<cpy_nb>(
                        tile_KV + i*(J/2 + J_padding) + j,
                        !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);
                }
            }
        }
    };

    static_assert(J % 8 == 0, "bad J");
    static_assert((J/2) % cpy_ne == 0, "bad J");
    ggml_cuda_unroll<7>{}(load);
}

// Planar Q8 K-tile load (docs/gfx906/plan_fa_part_A.md). K_q8_row0 is a
// byte pointer to the tile's first row start; k_col_block0 is the tile's
// global column-block offset (blocks of 32). Per row:
//   [quants: stride_K_q8*32 bytes] [scale: stride_K_q8 fp16]
// Quants loads are 4 x 8 B (tile row stride (D/32)*34 is 8-aligned; only
// 16-aligned when D % 256 == 0, so int4 is not used); each scale is a 2 B
// load. Writes the same LDS planes (K_values int8, K_scales half) as the
// old interleaved loader.
template<int warp_size, int nwarps, int I, int J, int K_row_stride, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_q8_q8_load_tile_q8(
        const uint8_t * const __restrict__ K_q8_row0,
        int8_t * const __restrict__ K_values,
        half * const __restrict__ K_scales,
        const int stride_K_q8,    // blocks (of 32) per full K row
        const int k_col_block0,   // global column-block offset of this tile
        const int i_sup) {

    if constexpr (J > 0) {
        constexpr int blocks_per_row = J / 32;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int total_blocks = I * blocks_per_row;
        const int bytes_per_row = stride_K_q8 * (int)sizeof(block_q8_0);
        const int quants_bytes  = stride_K_q8 * QK8_0;

        for (int block_idx = tid; block_idx < total_blocks; block_idx += blockDim.x * blockDim.y) {
            const int row = block_idx / blocks_per_row;
            const int col_block = block_idx % blocks_per_row;

            if (oob_check && row >= i_sup) {
                break;
            }

            const uint8_t * row_base = K_q8_row0 + (int64_t)row * bytes_per_row;

            const int2 * src2 = reinterpret_cast<const int2 *>(
                row_base + (k_col_block0 + col_block) * QK8_0);
            int2 * dst2 = reinterpret_cast<int2 *>(
                K_values + row * K_row_stride + col_block * QK8_0);
            dst2[0] = src2[0];
            dst2[1] = src2[1];
            dst2[2] = src2[2];
            dst2[3] = src2[3];

            K_scales[col_block * I + row] = *reinterpret_cast<const half *>(
                row_base + quants_bytes + (k_col_block0 + col_block) * 2);
        }

        __syncthreads();
    }
}

// ============================================================================
// Level 3c: Direct-paged KV loading (no gather step).
//
// Вместо чтения из contiguous K_q8[seq, head, token, block] паттерна kernel
// делает indirection через block_table: token -> physical_block -> memory.
//
// Параллельные (не заменяющие) версии load функций и iter функций.
// Старые функции (выше) остаются нетронутыми и используются для prefill.
// ============================================================================

struct PagedCacheView {
    const uint8_t * head_base;       // base + seq_offset + head_offset (precomputed by kernel)
    const int32_t * block_table_row; // block_table + sequence * max_blocks_per_seq
    int64_t         block_stride;    // bytes per paged block (block_size * token_stride)
    int64_t         token_stride;    // bytes per token (sum over all heads of per-head row)
};

// block_size захардкожен в 16 (vLLM default, assert'ится в launcher).
// shift=4, mask=0xF. Upper path это делает vector ALU 1-op.
static __device__ __forceinline__ const uint8_t * paged_token_ptr_bs16(
        const PagedCacheView & v, int global_token_idx) {
    const int logical_blk = global_token_idx >> 4;
    const int off_in_blk  = global_token_idx & 0xF;
    const int phys_blk    = v.block_table_row[logical_blk];
    return v.head_base
         + (int64_t) phys_blk    * v.block_stride
         + (int64_t) off_in_blk  * v.token_stride;
}

// Paged twin of flash_attn_tile_q8_q8_load_tile_q8 (planar row — see the
// non-paged loader above). Per paged token row:
//   [quants: stride_K_q8*32 bytes] [scale: stride_K_q8 fp16]
template<int warp_size, int nwarps, int I, int J, int K_row_stride, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_q8_q8_load_tile_q8_paged(
        const PagedCacheView paged_K,
        const int k_VKQ_0,          // global start token index for this tile
        const int k_KQ_0_block,     // k_KQ_0 / 32 : global column-block offset (head-relative)
        int8_t * const __restrict__ K_values,
        half   * const __restrict__ K_scales,
        const int stride_K_q8,      // blocks (of 32) per full K row
        const int i_sup) {

    if constexpr (J > 0) {
        constexpr int blocks_per_row = J / 32;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int total_blocks = I * blocks_per_row;
        const int quants_bytes = stride_K_q8 * QK8_0;

        for (int block_idx = tid; block_idx < total_blocks; block_idx += blockDim.x * blockDim.y) {
            const int row       = block_idx / blocks_per_row;
            const int col_block = block_idx % blocks_per_row;

            if (oob_check && row >= i_sup) {
                break;
            }

            const int global_token_idx = k_VKQ_0 + row;
            const uint8_t * tok_ptr    = paged_token_ptr_bs16(paged_K, global_token_idx);

            const int2 * src2 = reinterpret_cast<const int2 *>(
                tok_ptr + (k_KQ_0_block + col_block) * QK8_0);
            int2 * dst2 = reinterpret_cast<int2 *>(
                K_values + row * K_row_stride + col_block * QK8_0);
            dst2[0] = src2[0];
            dst2[1] = src2[1];
            dst2[2] = src2[2];
            dst2[3] = src2[3];

            K_scales[col_block * I + row] = *reinterpret_cast<const half *>(
                tok_ptr + quants_bytes + (k_KQ_0_block + col_block) * 2);
        }

        __syncthreads();
    }
}

// V load (fp16). Параллельный аналог flash_attn_tile_q8_q8_load_tile для paged.
// Каждый row (I штук, локально в tile) читается из paged V по индексу
// (k_VKQ_0 + row). Внутри token — обычная coalesced загрузка.
template<int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_q8_q8_load_tile_paged(
        const PagedCacheView paged_V,
        const int k_VKQ_0,
        half2 * const __restrict__ tile_KV,
        const int i_sup) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    auto load = [&] __device__ (const int n) {
        const int stride_j = warp_size >> n;

        if (stride_j == 0) {
            return;
        }

        const int j0_start = stride_j == warp_size ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        const int j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        const int stride_i = warp_size / stride_j;

        if (j0_start == j0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_j == warp_size ? 0 : threadIdx.x / stride_j);

            if (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    const int j = j0*cpy_ne + (stride_j == warp_size ? threadIdx.x : threadIdx.x % stride_j)*cpy_ne;

                    const half2 zero[cpy_ne] = {{0.0f, 0.0f}};
                    const bool  in_range = !oob_check || i < i_sup;
                    const half2 * src;
                    if (in_range) {
                        const uint8_t * tok_ptr = paged_token_ptr_bs16(paged_V, k_VKQ_0 + i);
                        src = reinterpret_cast<const half2 *>(tok_ptr) + j;
                    } else {
                        src = zero;
                    }
                    ggml_cuda_memcpy_1<cpy_nb>(
                        tile_KV + i*(J/2 + J_padding) + j,
                        src);
                }
            }
        }
    };

    static_assert(J % 8 == 0, "bad J");
    static_assert((J/2) % cpy_ne == 0, "bad J");
    ggml_cuda_unroll<7>{}(load);
}

template<int nthreads, int ncols, int ncols2, int DKQ>
static __device__ __forceinline__ void flash_attn_tile_q8_quantize_Q_to_shared(
        const float * __restrict__ Q_f,
        int8_t * __restrict__ Q_values,
        half * __restrict__ Q_scales,
        const int col_Q_0,
        const int ne01,
        const int head0,
        const int ne02,
        const int32_t nb01,
        const int32_t nb02,
        const float scale) {

    constexpr int blocks_per_col = DKQ / QK8_0;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_blocks = ncols * blocks_per_col;

    for (int block_idx = tid; block_idx < total_blocks; block_idx += blockDim.x * blockDim.y) {
        const int col = block_idx / blocks_per_col;
        const int col_block = block_idx % blocks_per_col;
        const int block_start = col_block * QK8_0;

        const int jc = col;
        const int j = jc / ncols2;
        const int c = jc % ncols2;

        // Bounds check for non-power-of-2 GQA ratios
        // ncols1 = ncols / ncols2
        if ((ncols > ncols2 && col_Q_0 + j >= ne01) || (ncols2 > 1 && head0 + c >= ne02)) {
            continue;
        }

        float Q_vals[QK8_0];
        block_q8_0 Q_block;

        const int base_offset = c*(nb02/sizeof(float)) + j*(nb01/sizeof(float)) + block_start;

        // Use float4 vectorized loads for better memory bandwidth (8x float4 = 32 floats)
        // DKQ is always multiple of 32 (static_assert at line 622), so no bounds check needed
        const float4* Q_f4 = reinterpret_cast<const float4*>(Q_f + base_offset);
        float4* Q_vals4 = reinterpret_cast<float4*>(Q_vals);

        #pragma unroll
        for (int i = 0; i < QK8_0/4; i++) {
            float4 tmp = Q_f4[i];
            tmp.x *= scale;
            tmp.y *= scale;
            tmp.z *= scale;
            tmp.w *= scale;
            Q_vals4[i] = tmp;
        }

        quantize_f32_q8_0_block(Q_vals, &Q_block);

        Q_scales[col_block * ncols + jc] = Q_block.d;

        int8_t * dst = Q_values + jc * DKQ + col_block * 32;
        const int4* src_int4 = (const int4*)Q_block.qs;
        int4* dst_int4 = (int4*)dst;

        dst_int4[0] = src_int4[0];
        dst_int4[1] = src_int4[1];
    }

    __syncthreads();
}

template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int nbatch_fa, int nbatch_K,
    bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_q8_q8_iter_KQ(
        int8_t * const Q_values,
        half * const Q_scales,
        const uint8_t * const __restrict__ K_q8,
        int8_t * const K_values,
        half * const K_scales,
        const int stride_K_q8,
        const int k_VKQ_0,
        const int k_VKQ_sup,
        const int k_KQ_0,
        float * KQ_acc) {
    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1;
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1;

    constexpr int K_row_stride = nbatch_K + 16;

    const uint8_t * k_q8_row0 =
        K_q8 + int64_t(k_VKQ_0) * stride_K_q8 * (int)sizeof(block_q8_0);
    flash_attn_tile_q8_q8_load_tile_q8<warp_size, nwarps, nbatch_fa, nbatch_K, K_row_stride, oob_check>
        (k_q8_row0, K_values, K_scales, stride_K_q8, k_KQ_0/32, k_VKQ_sup);
    __syncthreads();

    static_assert(nbatch_K % 4 == 0, "nbatch_K must be multiple of 4 for sdot4");

    constexpr int blocks_per_K_row = nbatch_K / 32;

    #pragma unroll 4
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y / np)*cpw;

        #pragma unroll 4
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;
            const int idx = i_KQ_0/(np*warp_size)*cpw + jc0;

            const int8_t* K_row_base = K_values + i_KQ * K_row_stride;

            #pragma unroll 4
            for (int block_id = 0; block_id < blocks_per_K_row; block_id++) {
                const int4* K_ptr4 = (const int4*)(K_row_base + block_id * 32);
                const int4* Q_ptr4 = (const int4*)&Q_values[jc * DKQ + k_KQ_0 + block_id * 32];

                int acc_int = 0;
                {
                    const int4 K_lo = K_ptr4[0];
                    const int4 Q_lo = Q_ptr4[0];
                    acc_int = ggml_cuda_dp4a(K_lo.x, Q_lo.x, acc_int);
                    acc_int = ggml_cuda_dp4a(K_lo.y, Q_lo.y, acc_int);
                    acc_int = ggml_cuda_dp4a(K_lo.z, Q_lo.z, acc_int);
                    acc_int = ggml_cuda_dp4a(K_lo.w, Q_lo.w, acc_int);
                }
                {
                    const int4 K_hi = K_ptr4[1];
                    const int4 Q_hi = Q_ptr4[1];
                    acc_int = ggml_cuda_dp4a(K_hi.x, Q_hi.x, acc_int);
                    acc_int = ggml_cuda_dp4a(K_hi.y, Q_hi.y, acc_int);
                    acc_int = ggml_cuda_dp4a(K_hi.z, Q_hi.z, acc_int);
                    acc_int = ggml_cuda_dp4a(K_hi.w, Q_hi.w, acc_int);
                }

                const half combined_scale_h = __hmul(
                    K_scales[block_id * nbatch_fa + i_KQ],
                    Q_scales[((k_KQ_0/32) + block_id) * ncols + jc]);
                KQ_acc[idx] += __half2float(combined_scale_h) * (float)acc_int;
            }
        }
    }

    if (k_KQ_0 + nbatch_K < DKQ) {
        __syncthreads();
    }
}

// Level 3c: paged twin of flash_attn_tile_q8_q8_iter_KQ. Only difference —
// K_q8 pointer replaced by PagedCacheView + k_VKQ_0 (global) for indirection.
template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int nbatch_fa, int nbatch_K,
    bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_q8_q8_iter_KQ_paged(
        int8_t * const Q_values,
        half * const Q_scales,
        const PagedCacheView paged_K,
        int8_t * const K_values,
        half * const K_scales,
        const int k_VKQ_0,
        const int k_VKQ_sup,
        const int k_KQ_0,
        float * KQ_acc) {
    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1;
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1;

    constexpr int K_row_stride = nbatch_K + 16;

    flash_attn_tile_q8_q8_load_tile_q8_paged<warp_size, nwarps, nbatch_fa, nbatch_K, K_row_stride, oob_check>
        (paged_K, k_VKQ_0, (k_KQ_0/32), K_values, K_scales, DKQ/32, k_VKQ_sup);
    __syncthreads();

    static_assert(nbatch_K % 4 == 0, "nbatch_K must be multiple of 4 for sdot4");

    constexpr int blocks_per_K_row = nbatch_K / 32;

    #pragma unroll 4
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y / np)*cpw;

        #pragma unroll 4
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;
            const int idx = i_KQ_0/(np*warp_size)*cpw + jc0;

            const int8_t* K_row_base = K_values + i_KQ * K_row_stride;

            #pragma unroll 4
            for (int block_id = 0; block_id < blocks_per_K_row; block_id++) {
                const int4* K_ptr4 = (const int4*)(K_row_base + block_id * 32);
                const int4* Q_ptr4 = (const int4*)&Q_values[jc * DKQ + k_KQ_0 + block_id * 32];

                int acc_int = 0;
                {
                    const int4 K_lo = K_ptr4[0];
                    const int4 Q_lo = Q_ptr4[0];
                    acc_int = ggml_cuda_dp4a(K_lo.x, Q_lo.x, acc_int);
                    acc_int = ggml_cuda_dp4a(K_lo.y, Q_lo.y, acc_int);
                    acc_int = ggml_cuda_dp4a(K_lo.z, Q_lo.z, acc_int);
                    acc_int = ggml_cuda_dp4a(K_lo.w, Q_lo.w, acc_int);
                }
                {
                    const int4 K_hi = K_ptr4[1];
                    const int4 Q_hi = Q_ptr4[1];
                    acc_int = ggml_cuda_dp4a(K_hi.x, Q_hi.x, acc_int);
                    acc_int = ggml_cuda_dp4a(K_hi.y, Q_hi.y, acc_int);
                    acc_int = ggml_cuda_dp4a(K_hi.z, Q_hi.z, acc_int);
                    acc_int = ggml_cuda_dp4a(K_hi.w, Q_hi.w, acc_int);
                }

                const half combined_scale_h = __hmul(
                    K_scales[block_id * nbatch_fa + i_KQ],
                    Q_scales[((k_KQ_0/32) + block_id) * ncols + jc]);
                KQ_acc[idx] += __half2float(combined_scale_h) * (float)acc_int;
            }
        }
    }

    if (k_KQ_0 + nbatch_K < DKQ) {
        __syncthreads();
    }
}

template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int DV, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check, typename T_KQ, typename T_acc>
static __device__ __forceinline__ void flash_attn_tile_q8_q8_iter(
        int8_t * const Q_values,
        half * const Q_scales,
        const uint8_t * const __restrict__ K_q8,
        const half2 * const __restrict__ V_h2,
        const half  * const __restrict__ mask,
        const float logit_softcap,
        const float slope,
        T_KQ      * const KQ,
        int8_t * const K_values,
        half * const K_scales,
        half2 * const V_tmp,
        const int stride_K_q8,
        const int stride_V2,
        const int stride_mask,
        float * const KQ_max,
        float * const KQ_sum,
        T_acc * const VKQ,
        const int k_VKQ_0,
        const int k_VKQ_max,
        // Level 3a inline-causal extension. If non-null, every score whose
        // absolute k-position > q_abs_for_row is forced to -INF before softmax,
        // eliminating the need to materialise a [B, Sq_pad, Sk_pad] fp16 mask.
        const int32_t * const __restrict__ q_abs_offset,
        // Sliding-window extension: when > 0, also force -INF on scores whose
        // absolute k-position < q_abs_for_row - window + 1 (keys older than the
        // per-row window). Inert when q_abs_offset is null; the backend passes
        // q_abs_offset for every windowed batch (decode included), so the
        // per-row formula covers decode rows (seq_len - 1) too.
        // LOCKSTEP: the cutoff formula below is duplicated verbatim in
        // fattn-q8-paged.cuh (two call sites per file); change both
        // together (tests cover both via test_forward_sliding_window* /
        // test_forward_paged_direct_sliding_window*).
        const int window,
        const int sequence,
        const int col_Q_0_iter) {
    constexpr int cpy_ne = ggml_cuda_get_max_cpy_bytes() / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1;
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1;

    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1);

    constexpr int KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    const int k_VKQ_sup = k_VKQ_max - k_VKQ_0;

    // Precompute per-sequence absolute causal cutoff for this query-tile.
    // q_abs_for_row[j] = q_abs_base + j (0-based local row inside the tile).
    // We only need q_abs_base since per-lane k_pos is added separately below.
    const int q_abs_base = q_abs_offset ? q_abs_offset[sequence] + col_Q_0_iter : 0;

    float KQ_max_new[cpw];
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    constexpr int num_i_KQ_iters = nbatch_fa/(np*warp_size);

    // Overlay KQ_acc onto V_tmp shared memory to reduce VGPR pressure.
    // KQ_acc and V_tmp are never used simultaneously (KQ_acc during dot products, V_tmp during V loading).
    constexpr int kv_tmp_elems = nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp - DV;
    constexpr size_t kv_tmp_bytes = kv_tmp_elems * sizeof(half2);
    constexpr size_t kq_acc_bytes = nwarps * warp_size * num_i_KQ_iters * cpw * sizeof(float);
    constexpr bool use_kq_acc_overlay = (kq_acc_bytes <= kv_tmp_bytes);

    float  KQ_acc_local[use_kq_acc_overlay ? 1 : num_i_KQ_iters * cpw];
    float* KQ_acc;

    if constexpr (use_kq_acc_overlay) {
        const int tid = threadIdx.y * warp_size + threadIdx.x;
        KQ_acc = reinterpret_cast<float*>(V_tmp) + tid * (num_i_KQ_iters * cpw);
    } else {
        KQ_acc = KQ_acc_local;
    }
    #pragma unroll
    for (int i = 0; i < num_i_KQ_iters * cpw; ++i) {
        KQ_acc[i] = 0.0f;
    }

    constexpr int nbatch_K_last = DKQ % nbatch_K;
    constexpr int num_K_tiles = (DKQ - nbatch_K_last) / nbatch_K;

    #pragma unroll
    for (int tile = 0; tile < num_K_tiles; tile++) {
        const int k_KQ_0 = tile * nbatch_K;
        flash_attn_tile_q8_q8_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, oob_check>(
            Q_values, Q_scales, K_q8, K_values, K_scales, stride_K_q8, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    if constexpr (nbatch_K_last > 0) {
        constexpr int k_KQ_0 = DKQ - nbatch_K_last;
        flash_attn_tile_q8_q8_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_last, oob_check>(
            Q_values, Q_scales, K_q8, K_values, K_scales, stride_K_q8, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    if constexpr (num_i_KQ_iters == 1) {
        const int i_KQ = (threadIdx.y % np)*warp_size + threadIdx.x;
        const int k_pos_abs = k_VKQ_0 + i_KQ;

#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int j = (jc0 + (threadIdx.y / np)*cpw)/ncols2;

            if (use_logit_softcap) {
                KQ_acc[jc0] = logit_softcap * fast_tanh_f32(KQ_acc[jc0]);
            }

            if (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[jc0] += mask != nullptr ?
                    slope*__half2float(mask[j*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;

                // Level 3a: inline causal — overrides KQ_acc with -INF for
                // positions past the per-row absolute cutoff. Works lane-by-lane
                // (k_pos_abs is per-thread, q_abs_row is per-jc0). window>0
                // adds the sliding cutoff (keys older than the window).
                if (q_abs_offset) {
                    const int q_abs_row = q_abs_base + j;
                    // M3 #10: overflow-free window cutoff — equivalent to
                    // `k_pos_abs < q_abs_row - window + 1` for all int32
                    // window (both operands non-negative; the subtraction
                    // stays in range), no wrapping for absurd windows.
                    if (k_pos_abs > q_abs_row ||
                        (window > 0 && q_abs_row - k_pos_abs >= window)) {
                        KQ_acc[jc0] = -INFINITY;
                    }
                }

                KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[jc0]);
            }

            KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
        }
    } else {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int j = (jc0 + (threadIdx.y / np)*cpw)/ncols2;

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
                const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;

                if (use_logit_softcap) {
                    KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] = logit_softcap * fast_tanh_f32(KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0]);
                }

                if (!oob_check || i_KQ < k_VKQ_sup) {
                    KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] += mask != nullptr ?
                        slope*__half2float(mask[j*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;

                    if (q_abs_offset) {
                        const int q_abs_row = q_abs_base + j;
                        const int k_pos_abs = k_VKQ_0 + i_KQ;
                        // M3 #10 (LOCKSTEP with the num_i_KQ_iters==1 site
                        // above and fattn-q8-paged.cuh): overflow-free
                        // window cutoff.
                        if (k_pos_abs > q_abs_row ||
                            (window > 0 && q_abs_row - k_pos_abs >= window)) {
                            KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] = -INFINITY;
                        }
                    }

                    KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0]);
                }
            }

            KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
        }
    }

    if constexpr (np == 1) {
        __syncthreads();
    } else {
        static_assert(cpw == 1, "bad cpw");
        __shared__ float KQ_max_new_shared[nwarps];
        if (threadIdx.x == 0) {
            KQ_max_new_shared[threadIdx.y] = KQ_max_new[0];
        }
        __syncthreads();
        KQ_max_new[0] = KQ_max_new_shared[(threadIdx.y & ~(np-1)) + threadIdx.x % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    if constexpr (num_i_KQ_iters == 1) {
        const int i_KQ = (threadIdx.y % np)*warp_size + threadIdx.x;

#pragma unroll
        for (int jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
            half tmp[1][KQ_cs];

#pragma unroll
            for (int jc1 = 0; jc1 < KQ_cs; ++jc1) {
                const int jc = jc0 + jc1;

                const float KQ_max_scale = fast_exp_f32(KQ_max[jc] - KQ_max_new[jc]);
                KQ_max[jc] = KQ_max_new[jc];

                const float val = !oob_check || i_KQ < k_VKQ_sup ?
                    fast_exp_f32(KQ_acc[jc] - KQ_max[jc]) : 0.0f;
                const float KQ_sum_add = val;
                tmp[0][jc1] = val;

                KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;

                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
                for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                    VKQ[jc*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
                }
            }

            ggml_cuda_memcpy_1<sizeof(tmp[0])>(
                KQ + (jc0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs))*(nbatch_fa*KQ_cs) + i_KQ*KQ_cs,
                tmp[0]);
        }
    } else {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
            half tmp[num_i_KQ_iters][KQ_cs];

#pragma unroll
            for (int jc1 = 0; jc1 < KQ_cs; ++jc1) {
                const int jc = jc0 + jc1;

                const float KQ_max_scale = fast_exp_f32(KQ_max[jc] - KQ_max_new[jc]);
                KQ_max[jc] = KQ_max_new[jc];

                float KQ_sum_add = 0.0f;
#pragma unroll
                for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
                    const float val = !oob_check || i0 + (threadIdx.y % np)*warp_size + threadIdx.x < k_VKQ_sup ?
                        fast_exp_f32(KQ_acc[(i0/(np*warp_size))*cpw + jc] - KQ_max[jc]) : 0.0f;
                    KQ_sum_add += val;
                    tmp[i0/(np*warp_size)][jc1] = val;
                }
                KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;

                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
                for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                    VKQ[jc*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
                }
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
                const int i = i0 + (threadIdx.y % np)*warp_size + threadIdx.x;

                ggml_cuda_memcpy_1<sizeof(tmp[0])>(
                    KQ + (jc0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs))*(nbatch_fa*KQ_cs) + i*KQ_cs,
                    tmp[i0/(np*warp_size)]);
            }
        }
    }

    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr int nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV;
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");

    if constexpr (use_kq_acc_overlay) {
        // KQ_acc was overlaid onto V_tmp; ensure all threads finished reading before V_tmp reuse.
        __syncthreads();
    }

#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        flash_attn_tile_q8_q8_load_tile<warp_size, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, V_tmp, stride_V2, k_VKQ_sup - k0);
        __syncthreads();

#pragma unroll
        for (int k1 = 0; k1 < nbatch_V; k1 += np) {
            half2 V_k[(DVp/2)/warp_size];
            half2 KQ_k[cpw];

            constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;

#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/warp_size], &V_tmp[(k1 + threadIdx.y % np)*(DV/2) + i0 + threadIdx.x*cpy_ne_D]);
            }

#pragma unroll
            for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                const int jc_KQ = jc_VKQ_0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs);

                half tmp[KQ_cs];
                ggml_cuda_memcpy_1<KQ_cs*sizeof(half)>(
                    &tmp, KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + threadIdx.y % np)*KQ_cs);
#pragma unroll
                for (int jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                    KQ_k[jc_VKQ_0+jc_VKQ_1] = __half2half2(tmp[jc_VKQ_1]);
                }
            }

#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                const half2 v_val = V_k[i0/warp_size];
#pragma unroll
                for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size] += v_val * KQ_k[jc_VKQ_0];
                }
            }
        }

        __syncthreads();
    }
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(ggml_cuda_fattn_tile_q8_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_tile_q8_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_tile_q8(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        // Level 3a: inline causal masking. When non-null, replaces the materialised
        // [B, Sq, Sk_pad] fp16 causal mask with per-lane checks inside the kernel.
        // q_abs_offset[b] = absolute query position at the start of this sequence's
        // query-chunk (i.e. seq_len_total - n_q[b]); per-row cutoff is
        //   q_abs_offset[sequence] + col_Q_0 + j  (j = local row inside tile).
        // Must be nullptr if the caller passes a real mask tensor.
        const int32_t * __restrict__ q_abs_offset,
        // Sliding-window size in tokens (0 = plain causal / no window).
        // See the tile-iter doc; inert when q_abs_offset is null.
        const int window,
        // Per-sequence KV scan start (gather-path sliding-window clip,
        // M1): the k-loop walks [k0_base, k_VKQ_max) instead of
        // [0, k_VKQ_max). LOCKSTEP with flash_attn_tile_q8_paged
        // (fattn-q8-paged.cuh, phase C) — the floor block below is
        // duplicated verbatim; change both together (tests cover both
        // via test_forward_sliding_window* / test_forward_paged_direct_
        // sliding_window*). In the gather layout the contiguous buffer
        // is indexed at ABSOLUTE positions (the gather materializes
        // [start, seq_len) in-place), so k_pos_abs stays
        // k_VKQ_0 + i_KQ and the mask math is unchanged — only the
        // loop start moves. The gather kernel (gfx906_fa_gather.cu)
        // covers [start - GATHER_CLIP_MARGIN, seq_len) with
        // GATHER_CLIP_MARGIN >= the max nbatch_fa in the config table,
        // so the floored k0_base is always materialized.
        const int32_t * __restrict__ kv_start,
        // M2 tile clip (host knob GFX906_FA_TILE_CLIP, default 1): per-q-tile
        // k0_base window raise + per-q-tile causal k_VKQ_max cap. The
        // skipped tiles are window-/causal-masked to P=0 for every row of
        // the tile, so both are bit-identical to the full scan; the win is
        // skipping their k-iterations. 0 = disable (A/B arm). LOCKSTEP with
        // flash_attn_tile_q8_paged (fattn-q8-paged.cuh).
        const int tile_clip,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE

    if (use_logit_softcap && !(DV == 128 || DV == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, q_abs_offset, window, kv_start, tile_clip, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    static_assert(ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");
    static_assert(DKQ % 32 == 0, "DKQ must be multiple of 32 for Q8_0 quantization");
    // K is consumed in whole q8_0 blocks (blocks_per_K_row = nbatch_K/32): a tile
    // size that is not a multiple of 32 silently scores fewer dims than DKQ, so
    // the config table must not carry such a row for an instantiated dim. This
    // caught the inherited (96,96) row (nbatch_K 48 -> only 64 of 96 dims).

    constexpr int ncols     = ncols1*ncols2;
    constexpr int warp_size = 32;
    constexpr int nwarps    = ggml_cuda_fattn_tile_q8_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_size;
    constexpr int nbatch_fa = ggml_cuda_fattn_tile_q8_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr int nbatch_K  = ggml_cuda_fattn_tile_q8_get_nbatch_K (DKQ, DV, ncols1*ncols2);
    static_assert(nbatch_K % 32 == 0,
        "nbatch_K must be a multiple of 32 (K is read in whole q8_0 blocks) — "
        "fix the config table row, not the kernel");

    const int col_Q_0 = blockIdx.x * ncols1;

    // Use ceiling division to handle non-power-of-2 GQA ratios
    const int ntiles_z = (ne02 + ncols2 - 1) / ncols2;
    const int sequence = blockIdx.z / ntiles_z;
    const int zt = blockIdx.z - sequence * ntiles_z;
    const int head0 = zt * ncols2;
    const int gqa_ratio = ne02 / ne12;
    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02* head0              + nb01*col_Q_0);
    const uint8_t * K_q8 = (const uint8_t *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
    const half2 * V_h2 = (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio));

    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33) + nb31*col_Q_0) : nullptr;

    const int stride_K_q8 = nb11 / sizeof(block_q8_0);
    const int stride_V2   = nb21 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    float slope_tmp = 0.0f;
    if (threadIdx.x == 0) {
        slope_tmp = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;
    }
    const float slope = sgpr_broadcast_f32(slope_tmp);

    constexpr int cpy_ne = ggml_cuda_get_max_cpy_bytes() / 4;

    constexpr int cpw = ncols > nwarps ? ncols/nwarps : 1;
    constexpr int np  = nwarps > ncols ? nwarps/ncols : 1;
    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_size) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr int DVp  = (DV  + 2*warp_size - 1) & ~(2*warp_size - 1);

    __shared__ int8_t Q_values[ncols * DKQ];
    __shared__ half   Q_scales[ncols * (DKQ/32)];

    constexpr int K_row_padding = 16;
    __shared__ int8_t K_values[nbatch_fa * (nbatch_K + K_row_padding)];
    __shared__ half   K_scales[nbatch_fa * (nbatch_K/32)];

    __shared__ half2 KV_tmp[nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV];

    __shared__ half  KQ[ncols * nbatch_fa];
    half2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};

    float KQ_max[cpw];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float KQ_sum[cpw] = {0.0f};

    flash_attn_tile_q8_quantize_Q_to_shared<nwarps*warp_size, ncols, ncols2, DKQ>(
        Q_f, Q_values, Q_scales, col_Q_0, int(ne01.z), head0, ne02, nb01, nb02, scale);

    int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    // Sliding-window clip start (0 = full scan; see param comment).
    // LOCKSTEP with fattn-q8-paged.cuh phase C — duplicated verbatim;
    // change both together. Floor to the nbatch_fa tile boundary so
    // every tile visited is the same aligned tile the full scan visits
    // (bit-identity — an unaligned start makes the first tile partial,
    // repacking lanes and re-associating the fp16 reduction, ~5e-4
    // relative). Only floor when the keys gained by the floor,
    // [floor, k0_base), are window-masked (pos < q_abs + 1 - window) —
    // the production clip passes exactly k0_base = max(0, q_abs + 1 -
    // window); other kv_start callers (tests emulating a window via the
    // clip) keep their exact start.
    // M3 #8: device-side clamp — a negative kv_start would start the
    // k-loop at a negative token index (illegal access / wedge, not a
    // wrong number). The Python clip math already max(0, ·)-guards,
    // this is defense in depth.
    int k0_base = kv_start ? (int) kv_start[sequence] : 0;
    if (k0_base < 0) k0_base = 0;
    if (k0_base > 0 && window > 0 && q_abs_offset) {
        const int win_start = q_abs_offset[sequence] + 1 - window;
        if (k0_base <= (win_start > 0 ? win_start : 0)) {
            k0_base -= k0_base % nbatch_fa;
        }
    }
    // M2 tile clip (GFX906_FA_TILE_CLIP, default 1; 0 = A/B arm).
    // LOCKSTEP with fattn-q8-paged.cuh — duplicated verbatim.
    // (1) Per-q-tile window raise: this tile's first row (abs pos
    //     q_abs + col_Q_0) has the smallest window start in the tile;
    //     keys before q_abs + col_Q_0 + 1 - window are window-masked to
    //     -INF for EVERY row of the tile, so raising k0_base to the
    //     floored tile start skips them (same floor rule as above →
    //     bit-identical). Clip mode only (kv_start set): without it
    //     k0_base stays 0 by contract (the clip-vs-no-clip bit-identity
    //     tests) and the window mask does the work. In prefill the
    //     backend passes the conservative chunk-start clip
    //     (kv_start = max(0, chunk_first + 1 - window)), so this raises
    //     every q-tile after the first — the scan cost becomes ~window
    //     instead of ~chunk for sliding-window prefill.
    // (2) Per-q-tile causal cap: the last VALID row of this tile is at
    //     abs pos q_abs + min(col_Q_0 + ncols1, ne01.z) - 1; keys at or
    //     past that +1 are causally masked (k > q_abs_row) for every row
    //     of the tile, exactly as in the full scan, so capping
    //     k_VKQ_max skips those k-tiles (bit-identical; the oob tail
    //     logic handles the partial tile at the capped end). For Sq<=2
    //     (ncols1=2) the cap is q_abs + ne01.z = seq_len — decode is
    //     unchanged.
    if (tile_clip && q_abs_offset) {
        if (kv_start && window > 0) {
            const int ts = q_abs_offset[sequence] + col_Q_0 + 1 - window;
            if (ts > 0) {
                const int tsf = ts - ts % nbatch_fa;
                if (tsf > k0_base) k0_base = tsf;
            }
        }
        const int tile_end = q_abs_offset[sequence]
            + (col_Q_0 + ncols1 < (int) ne01.z ? col_Q_0 + ncols1 : (int) ne01.z);
        if (tile_end < k_VKQ_max) k_VKQ_max = tile_end;
    }
    if (ncols2 == 1) {
        int k_VKQ_0 = k0_base + blockIdx.y*nbatch_fa;
        while (k_VKQ_0 < k_VKQ_max - nbatch_fa) {
            constexpr bool oob_check = false;
            flash_attn_tile_q8_q8_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
                (Q_values, Q_scales, K_q8, V_h2, maskh, logit_softcap, slope, KQ, K_values, K_scales, KV_tmp,
                stride_K_q8, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max,
                q_abs_offset, window, sequence, col_Q_0);
            k_VKQ_0 += gridDim.y*nbatch_fa;
        }
        if (k_VKQ_0 < k_VKQ_max) {
            constexpr bool oob_check = true;
            flash_attn_tile_q8_q8_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
                (Q_values, Q_scales, K_q8, V_h2, maskh, logit_softcap, slope, KQ, K_values, K_scales, KV_tmp,
                stride_K_q8, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max,
                q_abs_offset, window, sequence, col_Q_0);
        }
    } else {
        for (int k_VKQ_0 = k0_base + blockIdx.y*nbatch_fa; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nbatch_fa) {
            if (k_VKQ_0 + nbatch_fa > k_VKQ_max) {
                // Tail strided tile crossing k_VKQ_max: mask OOB rows.
                constexpr bool oob_check = true;
                flash_attn_tile_q8_q8_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
                    (Q_values, Q_scales, K_q8, V_h2, maskh, logit_softcap, slope, KQ, K_values, K_scales, KV_tmp,
                    stride_K_q8, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max,
                    q_abs_offset, window, sequence, col_Q_0);
            } else {
                constexpr bool oob_check = false;
                flash_attn_tile_q8_q8_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
                    (Q_values, Q_scales, K_q8, V_h2, maskh, logit_softcap, slope, KQ, K_values, K_scales, KV_tmp,
                    stride_K_q8, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max,
                    q_abs_offset, window, sequence, col_Q_0);
            }
        }
    }

#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_sum[jc0] = warp_reduce_sum<warp_size>(KQ_sum[jc0]);
    }

    if constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

        half2 * VKQ_combine    = (half2 *) KV_tmp;
        float * KQ_sum_combine = (float *) Q_values;

        if (threadIdx.y % np != 0) {
            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&VKQ_combine[threadIdx.y*(DVp/2) + i0 + threadIdx.x*cpy_ne_D], &VKQ[i0/warp_size]);
            }

            if (threadIdx.x == 0) {
                KQ_sum_combine[threadIdx.y] = KQ_sum[0];
            }

            return;
        }

        __syncthreads();

#pragma unroll
        for (int ip = 1; ip < np; ++ip) {
            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                half2 tmp[cpy_ne_D];
                ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(threadIdx.y + ip)*(DVp/2) + i0 + threadIdx.x*cpy_ne_D]);
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_size + i1] += tmp[i1];
                }
            }

            KQ_sum[0] += KQ_sum_combine[threadIdx.y + ip];
        }
    }

    if (sinks && blockIdx.y == 0) {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (threadIdx.y/np)*cpw;
            const float sink = ((const float *) sinks)[head0 + jc % ncols2];

            float KQ_max_new_j = fmaxf(KQ_max[jc0], sink);
            const float KQ_max_scale = fast_exp_f32(KQ_max[jc0] - KQ_max_new_j);
            KQ_max[jc0] = KQ_max_new_j;

            const float val = fast_exp_f32(sink - KQ_max[jc0]);
            KQ_sum[jc0] = KQ_sum[jc0]*KQ_max_scale + val;

            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
            }
        }
    }

#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y/np)*cpw;

        const int j = jc / ncols2;
        const int c = jc % ncols2;

        // Bounds check for non-power-of-2 GQA ratios
        if ((ncols1 > 1 && col_Q_0 + j >= int(ne01.z)) || (ncols2 > 1 && head0 + c >= ne02)) {
            continue;
        }

        // Fully-masked rows (KQ_sum == 0) would give inf*0 = NaN; the
        // split-combine path has the analogous l_star guard.
        const float scale = gridDim.y == 1
            ? (KQ_sum[jc0] > 0.0f ? 1.0f/KQ_sum[jc0] : 0.0f)
            : 1.0f;

        const int j_dst_unrolled = ((sequence*int(ne01.z) + col_Q_0 + j)*ne02 + head0 + c)*gridDim.y + blockIdx.y;

        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
            float2 tmp[cpy_ne_D];
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = __half22float2(VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size + i1]);
                tmp[i1].x *= scale;
                tmp[i1].y *= scale;
            }
            if (i0 + warp_size*cpy_ne_D <= DV/2 || i0 + threadIdx.x*cpy_ne_D < DV/2) {
                ggml_cuda_memcpy_1<sizeof(tmp)>(&dst[j_dst_unrolled*DV + 2*i0 + threadIdx.x*(2*cpy_ne_D)], tmp);
            }
        }

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_sum[jc0]);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, q_abs_offset, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif
}

