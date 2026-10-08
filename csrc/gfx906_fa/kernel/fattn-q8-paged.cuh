// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) Nick — nick413@gmail.com
//
// Vendored from https://github.com/cassettesgoboom/gfx906-fa-vllm
// (FlashAttention-style custom attention backend for vLLM on AMD gfx906).
//
// fattn-q8-paged.cuh — Direct-paged FlashAttention Q8 kernel (Level 3c).
//
// Parallel to fattn-q8.cuh kernel but reads K/V **directly** from paged KV
// cache via block_table indirection, eliminating the gather step entirely.
//
// Layout of paged caches (vLLM compatible, block_size=16):
//   K_paged: [num_blocks, block_size, Hkv, (D/32)*34]  uint8  (planar q8_0
//            row — docs/gfx906/plan_fa_part_A.md: [quants D bytes | scale
//            (D/32) fp16]; same row byte count as block_q8_0)
//   V_paged: [num_blocks, block_size, Hkv,  D      ]  fp16
//   block_table: [num_seqs, max_blocks_per_seq]       int32
//
// Kernel computes: token_addr = base + block_table[seq, tok/16]*block_stride
//                               + (tok%16)*token_stride + head_kv*head_stride.
//
// Numerical semantics are identical to fattn-q8.cuh (same dp4a accumulation,
// same softmax, same VKQ path). Only address computation differs.

#pragma once
#include "fattn-q8.cuh"

// ----------------------------------------------------------------------------
// Paged-specific launch_bounds config.
//
// Baseline config в fattn-q8.cuh для DKQ=DV=128, ncols∈{16,32,64} выставляет
// occupancy=2 (2 blocks/CU) => maxreg=128 VGPR. Paged kernel добавляет
// PagedCacheView state + indirect block_table load, что при maxreg=128
// даёт огромный spill:
//   ncols=16: 82 VGPR spill, scratch=320B
//   ncols=32: 176 VGPR spill, scratch=528B
//   ncols=64: 381 VGPR spill, scratch=1424B   <- performance killer
// Каждый spilled VGPR превращается в store/load через scratch (L1/HBM).
//
// Решение: снизить occupancy до 1 для ncols∈{16,32,64} на DKQ=DV=128.
// 1 block/CU даёт maxreg=256 VGPR => 0 spill ожидаемо. Меньшая occupancy
// компенсируется устранением scratch traffic (каждое обращение к spill
// location стоит десятки циклов). При большем Sk (ALU-bound) выигрыш
// от 0 spill преобладает.
//
// Для ncols∈{2,4,8} оставляем исходный occupancy (spill 0-27 — приемлемо).
// ----------------------------------------------------------------------------
static constexpr __host__ __device__ int ggml_cuda_fattn_tile_q8_paged_get_occupancy(
        const int DKQ, const int DV, const int ncols) {
    // gfx906: pin every real-target config to occupancy 1 so the compiler can
    // use the full ~256 VGPR/lane budget. Paged adds PagedCacheView indirection
    // state that pushes small-ncols decode kernels past the 128-VGPR occ=2 cap,
    // so all of these spilled without the occ=1 override. (Validated: the
    // earlier "occ=1 crashes small ncols" report was a bad test -- see Gotchas.)
    if (DKQ == 128 && DV == 128 && ncols >= 8 && ncols <= 64) {
        return 1;   // ncols 8/16/32/64 (2/4 were already spill-free at occ=2)
    }
    if (DKQ == 256 && DV == 256 && ncols != 4 && ncols >= 2 && ncols <= 64) {
        return 1;   // decode(2) + prefill(8..64). ncols=4 (125 VGPR) is left at
    }               // base occ=2: it fits 8 wavefronts spill-free, so keep them.
    if (DKQ ==  64 && DV ==  64 && ncols == 32) {
        return 1;
    }
    return (ggml_cuda_fattn_tile_q8_get_config_amd(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
}

// Paged-specific nthreads/nbatch_fa/nbatch_K override.
//
// ncols=64 (D=128) is the one config where occupancy=1 alone does NOT clear
// the spill: the paged kernel genuinely needs more than 256 VGPR. Empirically
// (measured via llvm-readobj scratch):
//   nthreads=256, occ=1 -> 256 VGPR, 784B scratch   (compiler fills the budget,
//                                                     working set grows)
//   nthreads=512, occ=1 -> 128 VGPR, 512B scratch   <- least spill, kept here
// More threads tighten the per-thread VGPR cap (65536 VGPR/CU / nthreads) but
// also shrink each thread's live set, and for this kernel the 512-thread
// variant spills the fewest bytes. paged ncols=64 is benchmark-only anyway:
// the auto dispatcher routes Sq>16 to the gather/flat path, so this config is
// only reached with GFX906_FA_DIRECT_PAGED=1.
static constexpr __host__ __device__ int ggml_cuda_fattn_tile_q8_paged_get_nthreads(
        const int DKQ, const int DV, const int ncols) {
    if (DKQ == 128 && DV == 128 && ncols == 64) {
        return 512;
    }
    return (ggml_cuda_fattn_tile_q8_get_config_amd(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
}

static constexpr __host__ __device__ int ggml_cuda_fattn_tile_q8_paged_get_nbatch_fa(
        const int DKQ, const int DV, const int ncols) {
    // nbatch_fa inherited from base config (не зависит от nthreads).
    return (ggml_cuda_fattn_tile_q8_get_config_amd(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
}

static constexpr __host__ __device__ int ggml_cuda_fattn_tile_q8_paged_get_nbatch_K(
        const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_q8_get_config_amd(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
}

// ----------------------------------------------------------------------------
// flash_attn_tile_q8_q8_iter_paged — параллельная копия flash_attn_tile_q8_q8_iter
// с заменой K_q8/V_h2 на PagedCacheView и вызовом _paged load функций.
// ----------------------------------------------------------------------------
template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int DV, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check, typename T_KQ, typename T_acc>
static __device__ __forceinline__ void flash_attn_tile_q8_q8_iter_paged(
        int8_t * const Q_values,
        half * const Q_scales,
        const PagedCacheView paged_K,
        const PagedCacheView paged_V,
        const half  * const __restrict__ mask,
        const float logit_softcap,
        const float slope,
        T_KQ      * const KQ,
        int8_t * const K_values,
        half * const K_scales,
        half2 * const V_tmp,
        const int stride_mask,
        float * const KQ_max,
        float * const KQ_sum,
        T_acc * const VKQ,
        const int k_VKQ_0,
        const int k_VKQ_max,
        const int32_t * const __restrict__ q_abs_offset,
        // Sliding-window extension: when > 0, also force -INF on scores whose
        // absolute k-position < q_abs_for_row - window + 1. Inert when
        // q_abs_offset is null (same contract as the contiguous kernel).
        // LOCKSTEP: the cutoff formula below is duplicated verbatim in
        // fattn-q8.cuh (two call sites per file); change both together
        // (tests cover both via test_forward_sliding_window* /
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

    const int q_abs_base = q_abs_offset ? q_abs_offset[sequence] + col_Q_0_iter : 0;

    float KQ_max_new[cpw];
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    constexpr int num_i_KQ_iters = nbatch_fa/(np*warp_size);

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
        flash_attn_tile_q8_q8_iter_KQ_paged<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, oob_check>(
            Q_values, Q_scales, paged_K, K_values, K_scales, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    if constexpr (nbatch_K_last > 0) {
        constexpr int k_KQ_0 = DKQ - nbatch_K_last;
        flash_attn_tile_q8_q8_iter_KQ_paged<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_last, oob_check>(
            Q_values, Q_scales, paged_K, K_values, K_scales, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
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

                if (q_abs_offset) {
                    const int q_abs_row = q_abs_base + j;
                    // M3 #10 (LOCKSTEP with fattn-q8.cuh and the
                    // num_i_KQ_iters>1 site below): overflow-free window
                    // cutoff — `q_abs_row - k_pos_abs >= window` is
                    // equivalent for all int32 window and cannot wrap.
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
                        // above and fattn-q8.cuh): overflow-free window
                        // cutoff.
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
        __syncthreads();
    }

#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        // Direct-paged V load: global token index = k_VKQ_0 + k0 + i (row).
        flash_attn_tile_q8_q8_load_tile_paged<warp_size, nwarps, nbatch_V, DV, 0, oob_check>
            (paged_V, k_VKQ_0 + k0, V_tmp, k_VKQ_sup - k0);
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

// ----------------------------------------------------------------------------
// flash_attn_tile_q8_paged — параллельный __global__ kernel для direct-paged.
//
// Отличия от flash_attn_tile_q8:
//   1. K, V указывают на paged cache base pointers (не contiguous).
//   2. Новые параметры: block_table, block_size, max_blocks_per_seq,
//      strides (K/V) в bytes: block_stride, token_stride, head_stride.
//   3. Внутри kernel: вычисляем PagedCacheView для K и V с head_base,
//      пробрасывая в iter_paged.
//
// Остальное идентично (та же dispatch logic, те же output pathways).
// ----------------------------------------------------------------------------
template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(ggml_cuda_fattn_tile_q8_paged_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_tile_q8_paged_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_tile_q8_paged(
        const char * __restrict__ Q,
        const char * __restrict__ K_paged_base,
        const char * __restrict__ V_paged_base,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        const int32_t * __restrict__ q_abs_offset,
        const int window,
        // Per-row KV scan start (sliding-window clip, phase C): the k-loop
        // walks [k0_base, k_VKQ_max) instead of [0, k_VKQ_max). Bit-identical
        // to scanning [0, k0_base) when k0_base >= q_abs_row - window + 1
        // (those keys are window-masked to -INF anyway, and fully-masked
        // tiles are exact no-ops); the win is skipping the HBM reads.
        // When the keys a tile-boundary floor would add are window-masked
        // (production clip: k0_base == max(0, q_abs + 1 - window)), k0_base
        // is floored to an nbatch_fa tile boundary — an unaligned start
        // would make the first tile partial, repacking the lanes and
        // re-associating the fp16 reduction (~5e-4..2e-3 relative, not
        // bit-identical). split-K still covers the clipped range exactly:
        // slice y visits k0_base + (y+i*gridDim.y)*nbatch_fa.
        const int32_t * __restrict__ kv_start,
        // M2 tile clip (host knob GFX906_FA_TILE_CLIP, default 1): per-q-tile
        // k0_base window raise + per-q-tile causal k_VKQ_max cap. The
        // skipped tiles are window-/causal-masked to P=0 for every row of
        // the tile, so both are bit-identical to the full scan; the win is
        // skipping their k-iterations. 0 = disable (A/B arm). LOCKSTEP with
        // flash_attn_tile_q8 (fattn-q8.cuh).
        const int tile_clip,
        const int32_t * __restrict__ block_table,
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
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        // paged-specific:
        const int32_t max_blocks_per_seq,
        const int64_t k_block_stride, const int64_t k_token_stride, const int64_t k_head_stride,
        const int64_t v_block_stride, const int64_t v_token_stride, const int64_t v_head_stride) {
#ifdef FLASH_ATTN_AVAILABLE

    if (use_logit_softcap && !(DV == 128 || DV == 256)) {
        GGML_UNUSED_VARS(Q, K_paged_base, V_paged_base, mask, sinks, KV_max, q_abs_offset, window, kv_start, tile_clip, block_table,
            dst, dst_meta, scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03, nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
            ne31, ne32, ne33, nb31, nb32, nb33,
            max_blocks_per_seq,
            k_block_stride, k_token_stride, k_head_stride,
            v_block_stride, v_token_stride, v_head_stride);
        NO_DEVICE_CODE;
        return;
    }

    static_assert(ggml_cuda_fattn_tile_q8_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");
    static_assert(DKQ % 32 == 0, "DKQ must be multiple of 32 for Q8_0 quantization");

    constexpr int ncols     = ncols1*ncols2;
    constexpr int warp_size = 32;
    constexpr int nwarps    = ggml_cuda_fattn_tile_q8_paged_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_size;
    constexpr int nbatch_fa = ggml_cuda_fattn_tile_q8_paged_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr int nbatch_K  = ggml_cuda_fattn_tile_q8_paged_get_nbatch_K (DKQ, DV, ncols1*ncols2);
    static_assert(nbatch_K % 32 == 0,
        "nbatch_K must be a multiple of 32 (K is read in whole q8_0 blocks) — "
        "fix the config table row, not the kernel");

    const int col_Q_0 = blockIdx.x * ncols1;

    const int ntiles_z = (ne02 + ncols2 - 1) / ncols2;
    const int sequence = blockIdx.z / ntiles_z;
    const int zt = blockIdx.z - sequence * ntiles_z;
    const int head0 = zt * ncols2;
    const int gqa_ratio = ne02 / ne12;
    const int head_kv_idx = head0 / gqa_ratio;

    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02* head0 + nb01*col_Q_0);

    // Build paged views. head_base precomputed per (sequence, head_kv):
    //   K/V paged layout: [num_blocks, block_size, Hkv, row].
    //   block_stride (bytes) and token_stride (bytes) passed by host.
    // Note: paged cache is **shared** across sequences (num_blocks global);
    //       seq_offset is implicit via block_table indirection.
    PagedCacheView paged_K;
    paged_K.head_base       = (const uint8_t *) K_paged_base + (int64_t) head_kv_idx * k_head_stride;
    paged_K.block_table_row = block_table + (int64_t) sequence * max_blocks_per_seq;
    paged_K.block_stride    = k_block_stride;
    paged_K.token_stride    = k_token_stride;

    PagedCacheView paged_V;
    paged_V.head_base       = (const uint8_t *) V_paged_base + (int64_t) head_kv_idx * v_head_stride;
    paged_V.block_table_row = block_table + (int64_t) sequence * max_blocks_per_seq;
    paged_V.block_stride    = v_block_stride;
    paged_V.token_stride    = v_token_stride;

    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33) + nb31*col_Q_0) : nullptr;

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
    // Sliding-window clip start (0 = full scan; see param comment). Floor
    // to the nbatch_fa tile boundary so every tile visited is the same
    // aligned tile the full scan visits (bit-identity — an unaligned start
    // makes the first tile partial, repacking lanes and re-associating the
    // fp16 reduction, ~5e-4 relative). Only floor when the keys gained by
    // the floor, [floor, k0_base), are window-masked (pos < q_abs + 1 -
    // window) — the production clip passes exactly k0_base = max(0, q_abs
    // + 1 - window); other kv_start callers (tests emulating a window via
    // the clip) keep their exact start.
    // M3 #8: device-side clamp — a negative kv_start would start the
    // k-loop at a negative token index (illegal access / wedge, not a
    // wrong number). The Python clip math already max(0, ·)-guards,
    // this is defense in depth. LOCKSTEP with fattn-q8.cuh.
    int k0_base = kv_start ? (int) kv_start[sequence] : 0;
    if (k0_base < 0) k0_base = 0;
    if (k0_base > 0 && window > 0 && q_abs_offset) {
        const int win_start = q_abs_offset[sequence] + 1 - window;
        if (k0_base <= (win_start > 0 ? win_start : 0)) {
            k0_base -= k0_base % nbatch_fa;
        }
    }
    // M2 tile clip (GFX906_FA_TILE_CLIP, default 1; 0 = A/B arm).
    // LOCKSTEP with fattn-q8.cuh — duplicated verbatim.
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
            flash_attn_tile_q8_q8_iter_paged<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
                (Q_values, Q_scales, paged_K, paged_V, maskh, logit_softcap, slope, KQ, K_values, K_scales, KV_tmp,
                stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max,
                q_abs_offset, window, sequence, col_Q_0);
            k_VKQ_0 += gridDim.y*nbatch_fa;
        }
        if (k_VKQ_0 < k_VKQ_max) {
            constexpr bool oob_check = true;
            flash_attn_tile_q8_q8_iter_paged<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
                (Q_values, Q_scales, paged_K, paged_V, maskh, logit_softcap, slope, KQ, K_values, K_scales, KV_tmp,
                stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max,
                q_abs_offset, window, sequence, col_Q_0);
        }
    } else {
        for (int k_VKQ_0 = k0_base + blockIdx.y*nbatch_fa; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nbatch_fa) {
            if (k_VKQ_0 + nbatch_fa > k_VKQ_max) {
                // Tail strided tile crossing k_VKQ_max: mask OOB rows.
                constexpr bool oob_check = true;
                flash_attn_tile_q8_q8_iter_paged<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
                    (Q_values, Q_scales, paged_K, paged_V, maskh, logit_softcap, slope, KQ, K_values, K_scales, KV_tmp,
                    stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max,
                    q_abs_offset, window, sequence, col_Q_0);
            } else {
                constexpr bool oob_check = false;
                flash_attn_tile_q8_q8_iter_paged<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
                    (Q_values, Q_scales, paged_K, paged_V, maskh, logit_softcap, slope, KQ, K_values, K_scales, KV_tmp,
                    stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max,
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

        if ((ncols1 > 1 && col_Q_0 + j >= int(ne01.z)) || (ncols2 > 1 && head0 + c >= ne02)) {
            continue;
        }

        // Fully-masked rows (KQ_sum == 0, e.g. a padded row's q_abs far
        // past its KV) would give inf*0 = NaN; the split-combine path has
        // the analogous l_star guard.
        const float scale_out = gridDim.y == 1
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
                tmp[i1].x *= scale_out;
                tmp[i1].y *= scale_out;
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
    GGML_UNUSED_VARS(Q, K_paged_base, V_paged_base, mask, sinks, KV_max, q_abs_offset, window, kv_start, block_table,
        dst, dst_meta, scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
        ne31, ne32, ne33, nb31, nb32, nb33,
        max_blocks_per_seq,
        k_block_stride, k_token_stride, k_head_stride,
        v_block_stride, v_token_stride, v_head_stride);
    NO_DEVICE_CODE;
#endif
}
