// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
//
// Fused MoE W4A16 GEMM kernel for gfx906 (Vega 20, no MFMA).
//
// Port of moe_q_gemm_rdna3.cu for the gfx906 dense AWQ fast path:
//   - fp16 activations only (AWQ/GPTQ W4A16 on gfx906 runs fp16)
//   - dot via __ockl_fdot2 (v_dot2_f32_f16), same as the proven dense
//     gptq_gemm path in csrc/libtorch_stable/quantization/gptq/q_gemm.cu
//   - zero_offset is a runtime argument: 0 for AWQ (stored zero points),
//     1 for GPTQ-v1 style zeros (kernel adds 1 to the stored value)
//
// Weight format (per expert, same as dense gfx906 W4A16):
//   - Packed int32 [E, K/8, N] with exllama shuffle
//     (even/odd interleaved: bits[3:0]=k0 [7:4]=k2 [11:8]=k4 [15:12]=k6
//      [19:16]=k1 [23:20]=k3 [27:24]=k5 [31:28]=k7 for k = 8*qk .. 8*qk+7)
//   - Scales [E, groups, N] fp16
//   - Zero points [E, groups, N/8] packed int32 (8 nibbles per word,
//     ascending n order)
//
// Design: THREADS_X=256 (4 waves on wave64), BLOCK_KN_SIZE=256, each thread
// handles N_PER_THREAD N columns (4 for BM<8, 2 for BM>=8). grid =
// (num_token_blocks, ceil(N/(256*N_PER_THREAD)), ceil(K/256)). Output is
// accumulated with packed 64-bit CAS atomic-adds into a pre-zeroed output
// tensor (no FP32 scratch buffer).

#include <atomic>
#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace moe_gptq_gfx906 {

#define BLOCK_KN_SIZE 256
#define THREADS_X 256

// 8 consecutive halves must be 16B-aligned: one ds_read_b128 instead of
// four ds_read_b32 (halves LDS instruction pressure in the K-loop).
__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr) {
  union {
    uint4 u;
    half2 h[4];
  } v;
  v.u = *(const uint4*)a_ptr;
  float result = {};
  #pragma unroll
  for (int i = 0; i < 4; i++)
    result = __ockl_fdot2(dq[i], v.h[i], result, true);
  return result;
}

// Packed 2-half atomic add via one 32-bit CAS loop (N_PER_THREAD == 2).
__forceinline__ __device__ void atomic_add_pk2_f16(half* addr, half2 v01) {
  unsigned* addr_u = reinterpret_cast<unsigned*>(addr);
  unsigned old = *addr_u;
  while (true) {
    union {
      unsigned u;
      half2 h;
    } cur, sum;
    cur.u = old;
    sum.h = __hadd2(cur.h, v01);
    unsigned prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

// Packed 4-half atomic add via one 64-bit CAS loop.
__forceinline__ __device__ void atomic_add_pk4_f16(half* addr, half2 v01,
                                                   half2 v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

// N nibbles starting at column n within a single packed uint32 word.
// Safe only when (n % 8) + N <= 8: all N nibbles live in one word.  This holds
// whenever n is aligned to N_PER_THREAD and N_PER_THREAD divides 8, which is
// guaranteed by the launcher (n = offset_n + t*N_PER_THREAD, offset_n aligned
// to BLOCK_KN_SIZE*N_PER_THREAD, both divisible by 8 for N_PER_THREAD ∈ {2,4}).
template <int N>
__forceinline__ __device__ void loadN_zeros(const uint32_t* qzeros_row, int n,
                                            int (&zeros)[N]) {
  static_assert(N == 2 || N == 4, "loadN_zeros: N must be 2 or 4");
  uint32_t d = qzeros_row[n / 8] >> ((n & 0x07) * 4);
  #pragma unroll
  for (int i = 0; i < N; ++i) zeros[i] = (int)((d >> (4 * i)) & 0xF);
}

// Precompute scale-baked dequant constants for one zero/scale pair.
//   z1z16[0] = scale * (-1024 - zero)   ("low" pairs: q + 1024)
//   z1z16[1] = scale * (-64 - zero)     ("high" pairs: q*16 + 1024)
//   y1y16[0] = scale * 1
//   y1y16[1] = scale * (1/16)
__forceinline__ __device__ void prep_zero_scale_fp16(uint32_t zero, half scale,
                                                     half2 (&z1z16)[2],
                                                     half2 (&y1y16)[2]) {
  half z1 = __float2half_rn(-1024.0f - (float)zero);
  half z16 = __hsub(__int2half_rn(-64), __int2half_rn((int)zero));

  half2 scale2 = __half2half2(scale);
  z1z16[0] = __hmul2(scale2, __half2half2(z1));
  z1z16[1] = __hmul2(scale2, __half2half2(z16));

  half y1 = __float2half_rn(1.0f);
  half y16 = __float2half_rn(1.0f / 16.0f);
  y1y16[0] = __hmul2(scale2, __half2half2(y1));
  y1y16[1] = __hmul2(scale2, __half2half2(y16));
}

// Dequantize one int32 (8 shuffled 4-bit weights) into 4 half2 pairs with the
// scale baked in: dq[j] = (q[2j], q[2j+1]) * scale - zero * scale.
__forceinline__ __device__ void dequant_4bit_8_fp16(uint32_t qa,
                                                    half2 (&dq)[4],
                                                    half2 (&z1z16)[2],
                                                    half2 (&y1y16)[2]) {
  const uint32_t c0 = 0x64006400;

  union {
    uint32_t u;
    half2 h2;
  } q0, q1, q2, q3;
  q0.u = (qa & 0x000F000F) | c0;  // half2(q[0]+1024, q[1]+1024)
  q1.u = (qa & 0x00F000F0) | c0;  // half2(q[2]*16+1024, q[3]*16+1024)
  uint32_t qa_hi = qa >> 8;
  q2.u = (qa_hi & 0x000F000F) | c0;  // half2(q[4]+1024, q[5]+1024)
  q3.u = (qa_hi & 0x00F000F0) | c0;  // half2(q[6]*16+1024, q[7]*16+1024)

  dq[0] = __hfma2(q0.h2, y1y16[0], z1z16[0]);
  dq[1] = __hfma2(q1.h2, y1y16[1], z1z16[1]);
  dq[2] = __hfma2(q2.h2, y1y16[0], z1z16[0]);
  dq[3] = __hfma2(q3.h2, y1y16[1], z1z16[1]);
}

// ---------------------------------------------------------------------------
// Fused MoE kernel.
// ---------------------------------------------------------------------------

// N_PER_THREAD: output columns per thread (4 = original layout; 2 halves
// accumulator/dequant register pressure for ~2x occupancy at large BM).
template <int BLOCK_SIZE_M, int N_PER_THREAD>
__global__ void __launch_bounds__(THREADS_X) moe_gemm_q4_kernel_gfx906(
    const half* __restrict__ a,  // [size_m, size_k] or [M*topk, K]
    half* __restrict__ c,        // [M*topk, size_n] or [M, size_n], pre-zeroed
    const uint32_t* __restrict__ b_q_weight,  // [E, K/8, N] packed
    const half* __restrict__ b_scales,        // [E, groups, N]
    const uint32_t* __restrict__ b_qzeros,    // [E, groups, N/8] packed
    const float* __restrict__ topk_weights,   // [M*topk] or nullptr
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    const int size_m,  // total tokens (original M, or M*topk for w2 pass)
    const int size_n,  // output features per expert
    const int size_k,  // input features
    const int groups,  // K / group_size
    const int top_k,   // routing top-k (1 for w2 pass)
    // Per-expert strides (in elements, not bytes)
    const int expert_weight_stride,  // (K/8) * N
    const int expert_scales_stride,  // groups * N
    const int expert_zeros_stride,   // groups * (N/8)
    const bool mul_topk_weight,
    const int output_topk,           // >0: write to row token_id/output_topk
    const int zero_offset) {
  static_assert(N_PER_THREAD == 4 || N_PER_THREAD == 2,
                "N_PER_THREAD must be 2 or 4");
  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * BLOCK_KN_SIZE * N_PER_THREAD;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * N_PER_THREAD;

  // Early exit for padding blocks or invalid experts (expert_map = -1)
  if (token_block * BLOCK_SIZE_M >= num_tokens_post_padded[0]) return;

  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  // Expert-specific pointers
  const uint32_t* expert_weights =
      b_q_weight + (int64_t)expert_id * expert_weight_stride;
  const half* expert_scales =
      b_scales + (int64_t)expert_id * expert_scales_stride;
  const uint32_t* expert_qzeros =
      b_qzeros ? b_qzeros + (int64_t)expert_id * expert_zeros_stride : nullptr;

  // LDS for activations; pad to 16-byte alignment so dot22_8_f can use uint4.
  constexpr int LDS_PAD = 8;
  static_assert((BLOCK_KN_SIZE + LDS_PAD) % 8 == 0,
                "LDS row stride must be 16-byte aligned for ds_read_b128");
  __shared__ half block_a[BLOCK_SIZE_M][BLOCK_KN_SIZE + LDS_PAD];

  static_assert(BLOCK_KN_SIZE == THREADS_X,
                "BLOCK_KN_SIZE must equal THREADS_X");

  const int offset_m_base = token_block * BLOCK_SIZE_M;

  if (offset_k + t < end_k) {
    #pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m) {
      int32_t token_id = sorted_token_ids[offset_m_base + m];
      int token_row = token_id / top_k;
      half av;
      if (token_row < size_m) {
        av = a[(int64_t)token_row * size_k + offset_k + t];
      } else {
        av = __float2half_rn(0.0f);
      }
      block_a[m][t] = av;
    }
  }
  __syncthreads();

  if (n >= size_n) return;

  // Group bookkeeping
  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  // Weight pointer for this expert
  int qk = offset_k / 8;
  const uint32_t* b_ptr = expert_weights + qk * size_n + n;

  // Per-column dequant constants (N_PER_THREAD columns per thread)
  half2 z1z16_h[N_PER_THREAD][2], y1y16_h[N_PER_THREAD][2];

  auto refresh_group = [&](int g) {
    const half* sc_row = expert_scales + g * size_n;
    int zeros[N_PER_THREAD];
    if (b_qzeros == nullptr) {
      // Symmetric quantization: constant zero point 8 (uint4 midpoint),
      // inlined so no zp tensor is materialized or streamed.
      #pragma unroll
      for (int i = 0; i < N_PER_THREAD; ++i) zeros[i] = 8;
    } else {
      const uint32_t* qz_row = expert_qzeros + g * (size_n / 8);
      loadN_zeros<N_PER_THREAD>(qz_row, n, zeros);
    }
    #pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
      half scale = sc_row[n + i];
      prep_zero_scale_fp16((uint32_t)(zeros[i] + zero_offset), scale,
                           z1z16_h[i], y1y16_h[i]);
    }
  };

  refresh_group(group);

  float block_c[BLOCK_SIZE_M][N_PER_THREAD];
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
    #pragma unroll
    for (int j = 0; j < N_PER_THREAD; ++j) block_c[m][j] = 0.0f;
  }

  // --- Main K-loop (single-stage weight prefetch) ---
  // NOTE: double-buffered prefetch was tried (both a swap-based and an
  // unrolled-by-2 ping-pong structure) but the compiler kept both chunk
  // buffers live across the whole consume phase -> 256 VGPRs + heavy spills
  // at BM=16 (5x slower). Single-stage stays; see DEVLOG P2-1.
  int k = offset_k;
  uint32_t b_w[4][N_PER_THREAD];
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh_group(group);
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      if constexpr (N_PER_THREAD == 4) {
        int4 v = *(const int4*)(b_ptr + j * size_n);
        b_w[j][0] = v.x;
        b_w[j][1] = v.y;
        b_w[j][2] = v.z;
        b_w[j][3] = v.w;
      } else {
        uint2 v = *(const uint2*)(b_ptr + j * size_n);
        b_w[j][0] = v.x;
        b_w[j][1] = v.y;
      }
    }
    b_ptr += 4 * size_n;

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;

      half2 dq[N_PER_THREAD][4];
      #pragma unroll
      for (int i = 0; i < N_PER_THREAD; ++i)
        dequant_4bit_8_fp16(b_w[j][i], dq[i], z1z16_h[i], y1y16_h[i]);

      #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        const half* a_ptr =
            reinterpret_cast<const half*>(&block_a[m][a_off]);
        #pragma unroll
        for (int i = 0; i < N_PER_THREAD; ++i)
          block_c[m][i] += dot22_8_f(dq[i], a_ptr);
      }
    }
    k += 32;
  }

  // --- Epilogue: apply topk_weight and atomic-add to output ---
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
    int32_t token_id = sorted_token_ids[offset_m_base + m];
    if (token_id / top_k >= size_m) continue;

    // Apply router weight
    if (mul_topk_weight && topk_weights != nullptr) {
      float tw = topk_weights[token_id];
      #pragma unroll
      for (int j = 0; j < N_PER_THREAD; ++j) block_c[m][j] *= tw;
    }

    // output_topk > 0: reduce by mapping token_id back to original token
    // (multiple experts write to the same row via atomics)
    int64_t out_row = (output_topk > 0) ? (int64_t)(token_id / output_topk)
                                        : (int64_t)token_id;
    half* out = c + out_row * size_n + n;
    if constexpr (N_PER_THREAD == 4) {
      half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                 __float2half_rn(block_c[m][1]));
      half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                 __float2half_rn(block_c[m][3]));
      atomic_add_pk4_f16(out, r01, r23);
    } else {
      half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                 __float2half_rn(block_c[m][1]));
      atomic_add_pk2_f16(out, r01);
    }
  }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <int BLOCK_SIZE_M, int N_PER_THREAD>
void launch_moe_gemm_q4(
    const half* a, half* c, const uint32_t* b_q_weight, const half* b_scales,
    const uint32_t* b_qzeros, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int groups, int top_k, int expert_weight_stride,
    int expert_scales_stride, int expert_zeros_stride, bool mul_topk_weight,
    int output_topk, int zero_offset, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid(num_token_blocks,
            (size_n + BLOCK_KN_SIZE * N_PER_THREAD - 1) /
                (BLOCK_KN_SIZE * N_PER_THREAD),
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  moe_gemm_q4_kernel_gfx906<BLOCK_SIZE_M, N_PER_THREAD>
      <<<grid, block, 0, stream>>>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, size_m, size_n, size_k, groups,
          top_k, expert_weight_stride, expert_scales_stride,
          expert_zeros_stride, mul_topk_weight, output_topk, zero_offset);
}

// ---------------------------------------------------------------------------
// M=1 (single-token decode) re-tile: one 512-thread CTA covers 64*NPT output
// columns with cross-wave K parallelism (wave w owns K-slice w*size_k/8).
// Columns are lane-based (all waves see the same column set), so the per-wave
// fp32 partials are reduced through LDS and only wave 0 runs the epilogue --
// direct store for gemm1, packed CAS into the pre-zeroed token row for gemm2
// (all 8 slot x-blocks share the row and would otherwise add 8x).
// Standalone A/B at M=1 (see docs/gfx906/DEVLOG-moe-m1-sprint.md):
// gemm1 K=2048 32.0 -> 27.1 us, gemm2 K=512 21.4 -> 10.8 us per launch.
// ---------------------------------------------------------------------------

template <int THREADS, int NPT, int SLICE>
__global__ void __launch_bounds__(THREADS)
    moe_gemm_q4_v2_kernel_gfx906(const half* __restrict__ a,
                                 half* __restrict__ c,
                                 const uint32_t* __restrict__ b_q_weight,
                                 const half* __restrict__ b_scales,
                                 const uint32_t* __restrict__ b_qzeros,
                                 const float* __restrict__ topk_weights,
                                 const int32_t* __restrict__ sorted_token_ids,
                                 const int32_t* __restrict__ expert_ids,
                                 const int32_t* __restrict__ num_tokens_post_padded,
                                 const int size_m, const int size_n,
                                 const int size_k, const int groups,
                                 const int top_k,
                                 const int expert_weight_stride,
                                 const int expert_scales_stride,
                                 const int expert_zeros_stride,
                                 const bool mul_topk_w, const int output_topk,
                                 const int zero_offset) {
  static_assert(NPT == 2 || NPT == 4);
  static_assert(SLICE % 32 == 0 && SLICE >= 32);
  constexpr int NWAVES = THREADS / 64;
  constexpr int BLOCK_COLS = 64 * NPT;
  constexpr int LDS_PAD = 8;
  const int t = threadIdx.x;
  const int w = t / 64;
  const int tl = t % 64;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * BLOCK_COLS;
  const int n = offset_n + tl * NPT;
  const int slice_k = size_k / NWAVES;
  const int offset_k = w * slice_k;
  const int end_k = min(offset_k + slice_k, size_k);

  if (token_block >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  const uint32_t* expert_weights =
      b_q_weight + (int64_t)expert_id * expert_weight_stride;
  const half* expert_scales =
      b_scales + (int64_t)expert_id * expert_scales_stride;
  const uint32_t* expert_qzeros =
      b_qzeros ? b_qzeros + (int64_t)expert_id * expert_zeros_stride : nullptr;

  // per-wave activation slice (16B-padded rows)
  __shared__ half block_a[NWAVES][SLICE + LDS_PAD];
  // per-wave fp32 partials: [wave][lane][NPT]
  __shared__ float partial[NWAVES][64][NPT];

  int32_t token_id = sorted_token_ids[token_block];
  int token_row = token_id / top_k;
  // 64 lanes fill the full slice_k-wide LDS row for this wave (strided)
  #pragma unroll
  for (int i = 0; i < SLICE / 64; ++i) {
    int pos = tl + i * 64;
    if (offset_k + pos < end_k) {
      half av = (token_row < size_m)
                    ? a[(int64_t)token_row * size_k + offset_k + pos]
                    : __float2half_rn(0.0f);
      block_a[w][pos] = av;
    }
  }
  __syncthreads();

  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;
  int qk = offset_k / 8;
  const uint32_t* b_ptr = expert_weights + qk * size_n + n;

  half2 z1z16_h[NPT][2], y1y16_h[NPT][2];
  auto refresh_group = [&](int g) {
    const half* sc_row = expert_scales + g * size_n;
    int zeros[NPT];
    if (b_qzeros == nullptr) {
      // Symmetric quantization: constant zero point 8, inlined.
      #pragma unroll
      for (int i = 0; i < NPT; ++i) zeros[i] = 8;
    } else {
      const uint32_t* qz_row = expert_qzeros + g * (size_n / 8);
      uint32_t d = qz_row[n / 8] >> ((n & 0x07) * 4);
      #pragma unroll
      for (int i = 0; i < NPT; ++i) zeros[i] = (int)((d >> (4 * i)) & 0xF);
    }
    #pragma unroll
    for (int i = 0; i < NPT; ++i) {
      half scale = sc_row[n + i];
      prep_zero_scale_fp16((uint32_t)(zeros[i] + zero_offset), scale,
                           z1z16_h[i], y1y16_h[i]);
    }
  };
  refresh_group(group);

  float acc[NPT];
  #pragma unroll
  for (int i = 0; i < NPT; ++i) acc[i] = 0.0f;
  int k = offset_k;
  uint32_t b_w[4][NPT];
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh_group(group);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (NPT == 4) {
        uint4 v = *(const uint4*)(b_ptr + j * size_n);
        b_w[j][0] = v.x;
        b_w[j][1] = v.y;
        b_w[j][2] = v.z;
        b_w[j][3] = v.w;
      } else {
        uint2 v = *(const uint2*)(b_ptr + j * size_n);
        b_w[j][0] = v.x;
        b_w[j][1] = v.y;
      }
    }
    b_ptr += 4 * size_n;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;
      half2 dq[NPT][4];
      #pragma unroll
      for (int i = 0; i < NPT; ++i)
        dequant_4bit_8_fp16(b_w[j][i], dq[i], z1z16_h[i], y1y16_h[i]);
      const half* a_ptr =
          reinterpret_cast<const half*>(&block_a[w][a_off]);
      #pragma unroll
      for (int i = 0; i < NPT; ++i) acc[i] += dot22_8_f(dq[i], a_ptr);
    }
    k += 32;
  }

  // cross-wave reduce through LDS (lane-based: same lane across waves)
  #pragma unroll
  for (int i = 0; i < NPT; ++i) partial[w][tl][i] = acc[i];
  __syncthreads();
  float r[NPT];
  #pragma unroll
  for (int i = 0; i < NPT; ++i) r[i] = 0.0f;
  #pragma unroll
  for (int ww = 0; ww < NWAVES; ++ww)
    #pragma unroll
    for (int i = 0; i < NPT; ++i) r[i] += partial[ww][tl][i];

  // All waves hold the same reduced value for the same columns: only wave 0
  // runs the epilogue (direct stores are idempotent, but the gemm2 CAS must
  // fire exactly once per cell).
  if (w != 0) return;
  if (token_id / top_k >= size_m) return;
  if (mul_topk_w && topk_weights != nullptr) {
    float tw = topk_weights[token_id];
    #pragma unroll
    for (int i = 0; i < NPT; ++i) r[i] *= tw;
  }
  int64_t out_row = (output_topk > 0) ? (int64_t)(token_id / output_topk)
                                      : (int64_t)token_id;
  half* out = c + out_row * size_n + offset_n + tl * NPT;
  if (output_topk > 0) {
    // gemm2: x-blocks (one per expert slot) share the token row
    if (NPT == 4) {
      atomic_add_pk4_f16(out, __halves2half2(__float2half_rn(r[0]),
                                             __float2half_rn(r[1])),
                         __halves2half2(__float2half_rn(r[2]),
                                        __float2half_rn(r[3])));
    } else {
      atomic_add_pk2_f16(out, __halves2half2(__float2half_rn(r[0]),
                                             __float2half_rn(r[1])));
    }
  } else {
    if (NPT == 4) {
      *(half2*)out = __halves2half2(__float2half_rn(r[0]),
                                    __float2half_rn(r[1]));
      *(half2*)(out + 2) = __halves2half2(__float2half_rn(r[2]),
                                          __float2half_rn(r[3]));
    } else {
      *(half2*)out = __halves2half2(__float2half_rn(r[0]),
                                    __float2half_rn(r[1]));
    }
  }
}

template <int THREADS, int NPT, int SLICE>
void launch_moe_gemm_q4_v2(
    const half* a, half* c, const uint32_t* b_q_weight, const half* b_scales,
    const uint32_t* b_qzeros, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int groups, int top_k, int expert_weight_stride,
    int expert_scales_stride, int expert_zeros_stride, bool mul_topk_weight,
    int output_topk, int zero_offset, cudaStream_t stream) {
  dim3 block(THREADS);
  dim3 grid(num_token_blocks, size_n / (64 * NPT));
  moe_gemm_q4_v2_kernel_gfx906<THREADS, NPT, SLICE>
      <<<grid, block, 0, stream>>>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, size_m, size_n, size_k, groups,
          top_k, expert_weight_stride, expert_scales_stride,
          expert_zeros_stride, mul_topk_weight, output_topk, zero_offset);
}

// N_PER_THREAD selection. BM < 8 keeps the original 4-column layout (decode
// regime is latency-bound, not occupancy-bound). For BM >= 8 the default is
// 2 columns/thread: ~half the accumulator/dequant register pressure,
// doubling occupancy at BM=16 (4 -> 8 waves/CU) for a small but consistent
// prefill speedup. VLLM_GFX906_MOE_NPT=4|2 overrides for tuning.
//
// The override now applies to every BM (it was BM>=8 only), because the
// grouped/batch-decode regime (BM=4 is what em in (32, 512] selects, i.e. B=4
// under MTP k=3) had never been swept on the NPT axis — see the C2-BM>=2 note
// in docs/gfx906/DEVLOG-moe-c2v.md. Unset = the historical defaults, so an
// unset env is bit-identical to the pre-change kernel choice.
static int select_n_per_thread(int block_size_m) {
  static int cached = [] {
    const char* e = getenv("VLLM_GFX906_MOE_NPT");
    if (e == nullptr) return 0;
    if (e[0] == '2') return 2;
    if (e[0] == '4') return 4;
    return 0;
  }();
  if (cached != 0) return cached;
  return block_size_m < 8 ? 4 : 2;
}

#define LAUNCH_MOE(BM, NPT)                                                \
  launch_moe_gemm_q4<BM, NPT>(a, c, b_q_weight, b_scales, b_qzeros,       \
                              topk_weights, sorted_token_ids, expert_ids, \
                              num_tokens_post_padded, num_token_blocks,   \
                              size_m, size_n, size_k, groups, top_k,      \
                              expert_weight_stride, expert_scales_stride, \
                              expert_zeros_stride, mul_topk_weight,       \
                              output_topk, zero_offset, stream)

// Test-only dispatch-path marker for the M=1 gemm tiles: records which kernel
// dispatch_moe_gemm_q4 selected on the most recent call (read-and-reset via
// take_moe_m1_dispatch_path_impl, wrapped at global scope by
// take_moe_m1_dispatch_path for the public binding). The kernels accumulate
// with packed CAS atomics (K-split), so their outputs are NOT bit-reproducible
// run-to-run — even two calls of the SAME kernel differ by ~1 fp16 ulp. A
// dispatch-path marker is therefore the only deterministic way for a test to
// verify which tile actually ran. Values: 0 = legacy <1,4> gemm1 (MOE_NPT=4
// opt-out), 1 = v2 512-thread gemm2, 2 = legacy <1,4> gemm2 (shape-gate
// fallback / MOE_M1=0), 3 = <1,2> gemm1 (default). Non-M=1 calls never touch
// it. Single-writer by construction (one dispatch per call).
std::atomic<int> moe_m1_dispatch_path{0};

int64_t take_moe_m1_dispatch_path_impl() {
  return moe_m1_dispatch_path.exchange(0);
}

void dispatch_moe_gemm_q4(
    const half* a, half* c, const uint32_t* b_q_weight, const half* b_scales,
    const uint32_t* b_qzeros, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int groups, int top_k, int block_size_m,
    int expert_weight_stride, int expert_scales_stride, int expert_zeros_stride,
    bool mul_topk_weight, int output_topk, int zero_offset, cudaStream_t stream) {
  const int npt = select_n_per_thread(block_size_m);
  // M=1 decode fast path (VLLM_GFX906_MOE_M1, default ON since the C2
  // combined A/B): the gemm2 fused topk-weight/CAS path is re-tiled to the
  // 512-thread lane-column kernel (26.8 -> 22.3 us/call in-model). The
  // gemm1 re-tile measured neutral here and stays on the <1,4> kernel, so
  // this path requires output_topk > 0 (gemm2). For gemm2 a = act_out
  // [EM, N2] with EM = M*topk, so size_m == output_topk identifies the
  // single-token case.
  //
  // The v2 kernel consumes 32 k-elements per iteration per wave and
  // refreshes scale/zeros only at 32-aligned group boundaries, so it is
  // CORRECT only when slice_k = size_k/8 is a multiple of 32 (===>
  // size_k % 256 == 0) and groupsize % 32 == 0. Default-on therefore
  // shape-gates: non-qualifying shapes (e.g. K > 2048, as in Nemotron-H's
  // gemm2 K=2688) silently fall back to the legacy <1,4> tile — the exact
  // path they used before this flag existed — so enabling the default can
  // never change a model's numerics or crash it. The gate is M=1-only (inert
  // at N>=2 / prefill). Set VLLM_GFX906_MOE_M1=0 to force the legacy <1,4>
  // tile; =1 forces v2 and re-arms the shape assert (a forced bad shape is a
  // programming error, not a fallback case). Combined TP=2 M=1 A/B
  // (Qwen3.5-35B-A3B, pp2048/tg256, graph): +2.72 % vs off; see
  // docs/gfx906/DEVLOG-moe-c2v.md "C2 combined TP=2 M=1 A/B".
  if (block_size_m == 1 && output_topk > 0 && size_m == output_topk) {
    const bool v2_shape_ok =
        size_n % 256 == 0 && size_k % 256 == 0 && size_k <= 2048 &&
        groups > 0 && (size_k / groups) % 32 == 0;
    // Read per call (80x/step) so the flag can be flipped at runtime in
    // tests. Note: the flag only affects launches dispatched after the
    // flip — it has no effect on an already-captured CUDA graph (the
    // replayed kernel was chosen at capture time). The getenv cost is
    // negligible next to the launch.
    const char* m1_env = getenv("VLLM_GFX906_MOE_M1");
    const bool force_v2 = m1_env != nullptr && m1_env[0] == '1';
    // Only an exact "1" forces v2 (pre-C2, any non-"0" value did and failed
    // closed on bad shapes); other values like "true" now leave the default-on
    // shape gate to decide instead of asserting. Documented behavior change.
    const bool disable_v2 = m1_env != nullptr && m1_env[0] == '0';
    if (!disable_v2 && (force_v2 || v2_shape_ok)) {
      // Only assert the shape when the user explicitly forced v2 (=1); a
      // default-on non-qualifying shape is handled by the fallback above.
      TORCH_CHECK(
          !force_v2 || v2_shape_ok,
          "VLLM_GFX906_MOE_M1=1 forces the v2 M=1 tile but the gemm2 shape "
          "does not qualify (requires size_n%256==0, size_k%256==0, "
          "size_k<=2048, groupsize%32==0); unset the flag to fall back");
      moe_m1_dispatch_path.store(1);            // path 1: v2 gemm2
      launch_moe_gemm_q4_v2<512, 4, 256>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights,
          sorted_token_ids, expert_ids, num_tokens_post_padded,
          num_token_blocks, size_m, size_n, size_k, groups, top_k,
          expert_weight_stride, expert_scales_stride, expert_zeros_stride,
          mul_topk_weight, output_topk, zero_offset, stream);
      return;
    }
  }
  // M=1 gemm1 re-tile (C2 combined A/B, default ON): the <1,2> kernel
  // (64 cols/block vs <1,4>'s 128) for gemm1 (output_topk == 0). Standalone
  // sweep: 25.13 vs ~26.9 us/call (N=1024, K=2048); the powered in-model A/B
  // measured +2.82 % TP=2 M=1 / neutral TP=1 M=1 (see
  // docs/gfx906/DEVLOG-moe-c2v.md "C2 combined TP=2 M=1 A/B"). Default-on is
  // safe: the gate below is M=1-only (inert at N>=2 / prefill) and gemm2
  // keeps <1,4> (or the MOE_M1 v2 tile when that flag is on), so the two
  // flags never touch the same kernel. Set VLLM_GFX906_MOE_NPT=4 to force
  // the legacy <1,4> M=1 gemm1 tile; any other value (incl. =2) selects the
  // default <1,2> tile — only 4 and the default are distinct here (BM>=8 NPT
  // tuning is unaffected — see select_n_per_thread). Read per call like
  // MOE_M1 (80x/step, negligible next to the launch) so tests can flip it
  // at runtime; captured graphs replay the kernel chosen at capture.
  const char* npt_env = getenv("VLLM_GFX906_MOE_NPT");
  const bool m1_npt4 = block_size_m == 1 && output_topk == 0 &&
      npt_env != nullptr && npt_env[0] == '4';
  switch (block_size_m) {
    case 1:
      if (output_topk == 0) {
        // gemm1: <1,2> is the default since the C2 combined A/B; MOE_NPT=4
        // forces the legacy <1,4> tile.
        if (m1_npt4) {
          moe_m1_dispatch_path.store(0);       // path 0: legacy <1,4> gemm1
          LAUNCH_MOE(1, 4);
        } else {
          moe_m1_dispatch_path.store(3);      // path 3: default <1,2> gemm1
          LAUNCH_MOE(1, 2);
        }
      } else {
        // gemm2 fallback (v2 not taken: MOE_M1=0 or shape-gate miss). Keep
        // the legacy <1,4> tile — the exact path that existed before any of
        // these flags. The NPT default-on must NOT leak into gemm2: the A/B
        // that promoted it only measured gemm1 on <1,2>, and the shape-gate
        // fallback contract is "bit-identical to pre-flag behavior".
        moe_m1_dispatch_path.store(2);        // path 2: legacy <1,4> gemm2
        LAUNCH_MOE(1, 4);
      }
      break;
    case 2:
      if (npt == 2) {
        LAUNCH_MOE(2, 2);
      } else {
        LAUNCH_MOE(2, 4);
      }
      break;
    case 4:
      if (npt == 2) {
        LAUNCH_MOE(4, 2);
      } else {
        LAUNCH_MOE(4, 4);
      }
      break;
    case 8:
      if (npt == 2) {
        LAUNCH_MOE(8, 2);
      } else {
        LAUNCH_MOE(8, 4);
      }
      break;
    case 16:
      if (npt == 2) {
        LAUNCH_MOE(16, 2);
      } else {
        LAUNCH_MOE(16, 4);
      }
      break;
    default:
      TORCH_CHECK(false,
                  "moe_gptq_gemm_gfx906: block_size_m must be 1, 2, 4, 8 or 16, "
                  "got ",
                  block_size_m);
  }
}

}  // namespace moe_gptq_gfx906
}  // namespace vllm

// Public (global-scope) accessor for the M=1 dispatch-path marker; wraps the
// namespaced impl so the torch binding can reach it. See the marker's doc
// comment inside vllm::moe_gptq_gfx906 above.
int64_t take_moe_m1_dispatch_path() {
  return vllm::moe_gptq_gfx906::take_moe_m1_dispatch_path_impl();
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
//
// Inputs:
//   a                      [M, K] or [M*top_k, K]  half
//   c                      [M*top_k, N] or [M, N]  half (pre-zeroed!)
//   b_q_weight             [E, K/8, N]              int32 (shuffled)
//   b_scales               [E, groups, N]           half
//   b_qzeros               [E, groups, N/8]         int32 (packed 4-bit), or
//                                                                  empty
//                                                                  (symmetric:
//                                                                  kernel
//                                                                  inlines the
//                                                                  constant 8)
//   topk_weights           [M*top_k] or empty       float32
//   sorted_token_ids       [num_blocks * block_m]   int32
//   expert_ids             [num_blocks]             int32
//   num_tokens_post_padded [1]                      int32
//   top_k                  int
//   block_size_m           int (1, 2, 4, 8 or 16)
//   mul_topk_weight        bool
//   output_topk            int (>0: reduce to row token_id/output_topk)
//   zero_offset            int (0 for AWQ, 1 for GPTQ-v1 zeros)

void moe_gptq_gemm_gfx906(torch::Tensor a, torch::Tensor c,
                          torch::Tensor b_q_weight, torch::Tensor b_scales,
                          torch::Tensor b_qzeros, torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t top_k,
                          int64_t block_size_m, bool mul_topk_weight,
                          int64_t output_topk, int64_t zero_offset) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(c.is_cuda(), "c must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D");
  TORCH_CHECK(c.dim() == 2, "c must be 2D");
  TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be 3D [E, K/8, N]");
  TORCH_CHECK(b_scales.dim() == 3, "b_scales must be 3D [E, groups, N]");
  // An empty b_qzeros (numel()==0) selects the symmetric path: the kernels
  // inline the constant zero point 8 instead of streaming a packed tensor.
  const bool has_zp = b_qzeros.numel() > 0;
  if (has_zp) {
    TORCH_CHECK(b_qzeros.dim() == 3,
                "b_qzeros must be 3D [E, groups, N/8] or empty");
  }
  TORCH_CHECK(a.scalar_type() == torch::kHalf, "a must be half");
  TORCH_CHECK(c.scalar_type() == torch::kHalf, "c must be half");
  TORCH_CHECK(b_scales.scalar_type() == torch::kHalf,
              "b_scales dtype must be half");
  // atomic_add_pk2_f16 and atomic_add_pk4_f16 both require 4-byte alignment on
  // the output pointer: NPT=2 writes 2 halves (4 bytes), NPT=4 writes 4 halves
  // (8 bytes). size_n % 4 == 0 covers both.
  TORCH_CHECK(b_q_weight.size(2) % 4 == 0,
              "moe_gptq_gemm_gfx906: size_n (", b_q_weight.size(2),
              ") must be a multiple of 4 for CAS alignment");
  // Used as a divisor inside the kernels (token_id / top_k): a zero would
  // be on-device UB, so fail on the host instead.
  TORCH_CHECK(top_k > 0, "moe_gptq_gemm_gfx906: top_k must be positive");
  TORCH_CHECK(block_size_m == 1 || block_size_m == 2 || block_size_m == 4 ||
                  block_size_m == 8 || block_size_m == 16,
              "moe_gptq_gemm_gfx906: block_size_m must be 1, 2, 4, 8 or 16");
  TORCH_CHECK(
      sorted_token_ids.dim() == 1 &&
          sorted_token_ids.size(0) % block_size_m == 0,
      "moe_gptq_gemm_gfx906: sorted token rows must contain full blocks");
  TORCH_CHECK(b_scales.size(0) == b_q_weight.size(0),
              "moe_gptq_gemm_gfx906: scale expert count must match qweight");
  if (has_zp) {
    TORCH_CHECK(
        b_qzeros.size(0) == b_q_weight.size(0) &&
            b_qzeros.size(1) == b_scales.size(1),
        "moe_gptq_gemm_gfx906: zero-point expert/group counts must match");
  }
  TORCH_CHECK(expert_ids.dim() == 1 &&
                  expert_ids.size(0) >= sorted_token_ids.size(0) / block_size_m,
              "moe_gptq_gemm_gfx906: expert ids must cover every token block");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(2);
  int groups = (int)b_scales.size(1);

  // Caller contract: the kernel derives group boundaries as
  // (size_k / groups) and packs 8 nibbles per int32, so a mismatched or
  // non-divisible shape is silent garbage, not a crash.
  TORCH_CHECK(
      size_k == (int)b_q_weight.size(1) * 8,
      "moe_gptq_gemm_gfx906: a.size(1) (", size_k, ") != qweight rows * 8 (",
      (int)b_q_weight.size(1) * 8, ")");
  TORCH_CHECK(groups > 0 && size_k % groups == 0,
              "moe_gptq_gemm_gfx906: size_k (", size_k,
              ") must be divisible by the number of groups (", groups, ")");
  TORCH_CHECK(size_n % 8 == 0, "moe_gptq_gemm_gfx906: size_n (", size_n,
              ") must be a multiple of 8 (packed 4-bit zero rows)");
  TORCH_CHECK(b_scales.size(2) == size_n,
              "moe_gptq_gemm_gfx906: scales N (", b_scales.size(2),
              ") != qweight N (", size_n, ")");
  if (has_zp) {
    TORCH_CHECK(b_qzeros.size(2) * 8 == size_n,
                "moe_gptq_gemm_gfx906: zeros N (", b_qzeros.size(2) * 8,
                ") != qweight N (", size_n, ")");
  }

  // Per-expert strides
  int expert_weight_stride = (int)(b_q_weight.size(1) * b_q_weight.size(2));
  int expert_scales_stride = (int)(b_scales.size(1) * b_scales.size(2));
  int expert_zeros_stride = has_zp ? (int)(b_qzeros.size(1) * b_qzeros.size(2))
                                   : 0;
  const uint32_t* qzeros_ptr =
      has_zp ? (const uint32_t*)b_qzeros.data_ptr<int32_t>() : nullptr;

  int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);


  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  vllm::moe_gptq_gfx906::dispatch_moe_gemm_q4(
      (const half*)a.data_ptr(), (half*)c.data_ptr(),
      (const uint32_t*)b_q_weight.data_ptr<int32_t>(),
      (const half*)b_scales.data_ptr(),
      qzeros_ptr, topk_w_ptr,
      sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
      num_tokens_post_padded.data_ptr<int32_t>(), num_token_blocks, size_m,
      size_n, size_k, groups, (int)top_k, (int)block_size_m,
      expert_weight_stride, expert_scales_stride, expert_zeros_stride,
      mul_topk_weight, (int)output_topk, (int)zero_offset, stream);
}
