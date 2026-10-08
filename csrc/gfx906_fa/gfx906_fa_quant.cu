// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) Nick — nick413@gmail.com
//
// Vendored from https://github.com/cassettesgoboom/gfx906-fa-vllm
// (FlashAttention-style custom attention backend for vLLM on AMD gfx906).
//
// gfx906_fa_quant.cu — HIP kernels для квантования FP16 → block_q8_0
// на устройстве. Заменяет старый CPU-based путь (было: k.cpu() → копия туда-сюда
// через PCIe, ~100+ ms на 8k контекст).
//
// Два kernel'а:
//   1. quantize_q8_0_dense   — для произвольного contiguous BHSD тензора.
//                              Используется в legacy-пути и в тестах.
//   2. reshape_and_cache_q8  — квантует K-токены и пишет их в paged Q8 buffer
//                              по slot_mapping. Это hot-path: вызывается
//                              ровно 1 раз на decode-step и квантует только
//                              новые токены (инкрементально).
//
// Layout block_q8_0 (идентичен ggml-common.h):
//   struct { __half d; int8_t qs[32]; } — 34 байта, QK8_0=32.
//
// Параметризация threads: 1 workgroup = 64 threads (wavefront) = 2 блока Q8
// (для D=128 у нас 4 блока в row, значит 2 wavefront/row — укладывается в
// 1 workgroup 128 threads = 2 wavefronts × 2 blocks/wave = 4 blocks = 1 row).

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
#include <cstdint>

#include "kernel/q8_0_quantize.cuh"



// ---------------------------------------------------------------------------
// Kernel 1: quantize_q8_0_dense
//
// Вход:  x_f16   [N, D]   — contiguous row-major, D кратно 32
// Выход: y_q8    [N, (D/32)*34 bytes]  — contiguous uint8, PLANAR row
//         (plan_fa_part_A.md): [quants D bytes | scale (D/32) fp16]
//
// Grid: (ceil(N / rows_per_block), 1, 1)
// Block: (64, rows_per_block, 1) — каждый wavefront = 1 row × (D/32 blocks).
// Для D=128: 4 блока/row — 2 wavefronts обрабатывают один row параллельно.
// Для простоты MVP: 1 row/workgroup, 64 threads, обрабатываем последовательно
// D/32 пар блоков (для D=128 это 2 итерации).
// ---------------------------------------------------------------------------
extern "C" __global__ void quantize_q8_0_dense_kernel(
    const __half * __restrict__ x,
    uint8_t      * __restrict__ y,
    int N,
    int D
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    const int lane    = threadIdx.x;             // 0..63
    const int half_id = lane / 32;               // 0 или 1: какая "полуволна"
    const int lane_in = lane % 32;               // 0..31

    const int blocks_per_row = D / QK8_0_SZ;
    const __half * x_row = x + (int64_t)row * D;
    uint8_t      * y_row = y + (int64_t)row * blocks_per_row * Q8_0_BYTES;

    // Каждая пара полуволн обрабатывает 2 блока за итерацию.
    // Итерируем пока не покроем все blocks_per_row.
    for (int b0 = 0; b0 < blocks_per_row; b0 += 2) {
        const int b = b0 + half_id;
        if (b < blocks_per_row) {
            quantize_block_q8_0_halfwarp_planar(
                x_row + b * QK8_0_SZ,
                y_row + b * QK8_0_SZ,
                (uint16_t *)(y_row + D) + b,
                lane_in
            );
        }
    }
}


// ---------------------------------------------------------------------------
// Kernel 2: reshape_and_cache_q8
//
// Квантует FP16 K-токены и пишет их в paged Q8 KV-cache по slot_mapping.
//
// Вход:
//   key        [num_tokens, Hkv, D]            fp16 — новые K-токены
//   slot_mapping [num_tokens]                  int64 — физ. слот в paged cache
//                                                      (или -1 для пропуска токена)
// Выход:
//   k_cache_q8 [num_blocks, block_size, Hkv, (D/32)*34]  uint8
//              layout: (slot = block_idx * block_size + block_offset)
//              → k_cache_q8[block_idx, block_offset, h, :, :]
//              PLANAR row (plan_fa_part_A.md): [quants D bytes | scale
//              (D/32) fp16] — same row byte count as block_q8_0
//
// Одна строка = один token × один KV-head → D/32 блоков × 34 байта.
// Grid: (num_tokens, Hkv)  — один workgroup = один (token, head) pair.
// Block: (64, 1, 1) — wavefront квантует D/32 блоков за ceil((D/32)/2) итераций.
// ---------------------------------------------------------------------------
extern "C" __global__ void reshape_and_cache_q8_kernel(
    const __half  * __restrict__ key,
    const int64_t * __restrict__ slot_mapping,
    uint8_t       * __restrict__ k_cache_q8,
    int num_tokens,
    int num_kv_heads,
    int head_size,
    int block_size,
    int64_t key_token_stride,      // в элементах fp16: обычно Hkv * D
    int64_t key_head_stride,       // обычно D
    int64_t cache_block_stride,    // в байтах: block_size * Hkv * (D/32) * 34
    int64_t cache_token_stride,    // внутри блока: Hkv * (D/32) * 34 (в байтах)
    int64_t cache_head_stride      // (D/32) * 34 (в байтах)
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    if (token_idx >= num_tokens || head_idx >= num_kv_heads) return;

    const int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;  // padding/skipped token

    const int64_t block_idx    = slot / block_size;
    const int64_t block_offset = slot % block_size;

    // Источник: key[token_idx, head_idx, :D]
    const __half * x_row = key
        + token_idx * key_token_stride
        + head_idx  * key_head_stride;

    // Назначение: k_cache_q8[block_idx, block_offset, head_idx, :, :]
    uint8_t * y_row = k_cache_q8
        + block_idx    * cache_block_stride
        + block_offset * cache_token_stride
        + head_idx     * cache_head_stride;

    const int lane    = threadIdx.x;
    const int half_id = lane / 32;
    const int lane_in = lane % 32;
    const int blocks_per_row = head_size / QK8_0_SZ;

    for (int b0 = 0; b0 < blocks_per_row; b0 += 2) {
        const int b = b0 + half_id;
        if (b < blocks_per_row) {
            quantize_block_q8_0_halfwarp_planar(
                x_row + b * QK8_0_SZ,
                y_row + b * QK8_0_SZ,
                (uint16_t *)(y_row + head_size) + b,
                lane_in
            );
        }
    }
}


// ---------------------------------------------------------------------------
// Host launchers (C-linkage, вызываются из gfx906_fa.cpp)
// ---------------------------------------------------------------------------
extern "C" hipError_t launch_quantize_q8_0_dense(
    const __half * x,
    uint8_t      * y,
    int N,
    int D,
    hipStream_t stream
) {
    if (D % QK8_0_SZ != 0) return hipErrorInvalidValue;

    // Один workgroup = 1 row, 64 threads.
    // На больших N группируем rows_per_block=4 — меньше launch-overhead.
    constexpr int ROWS_PER_BLOCK = 4;
    dim3 block(64, ROWS_PER_BLOCK, 1);
    dim3 grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);

    quantize_q8_0_dense_kernel<<<grid, block, 0, stream>>>(x, y, N, D);
    return hipGetLastError();
}

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
) {
    if (head_size % QK8_0_SZ != 0) return hipErrorInvalidValue;
    if (num_tokens == 0)           return hipSuccess;

    dim3 block(64, 1, 1);
    dim3 grid(num_tokens, num_kv_heads, 1);

    reshape_and_cache_q8_kernel<<<grid, block, 0, stream>>>(
        key, slot_mapping, k_cache_q8,
        num_tokens, num_kv_heads, head_size, block_size,
        key_token_stride, key_head_stride,
        cache_block_stride, cache_token_stride, cache_head_stride
    );
    return hipGetLastError();
}
