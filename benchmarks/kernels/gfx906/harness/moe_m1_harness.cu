// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
//
// Standalone A/B harness for the M=1 MoE expert GEMM (S5 + C2-gemm1).
//
// Kernels under comparison (same inputs):
//   current  — moe_gemm_q4_kernel_gfx906<1,4> as shipped: 256-thread
//              blocks, grid.z = K/256 K-splits, fp16 CAS epilogue into a
//              pre-zeroed output (copied verbatim from
//              csrc/rocm/moe_q_gemm_gfx906.cu).
//   v2       — 512 threads (8 waves), NPT=2, BM=1; wave w owns K-slice
//              w*(K/8); fp32 per-wave partials reduced through LDS;
//              direct store (gemm1) or 8-way CAS (gemm2 output_topk).
//   sweep    — the current kernel templated on (THREADS_X == BLOCK_KN,
//              NPT): maps the z-split-granularity axis (C2, 2026-08-19).
//   v1       — full-K per block, direct store, no CAS/zeroing (C2's V1
//              design; REJECTED at 64 blocks: one long weight stream per
//              block is bandwidth-starved, 2.2-4.3x slower than current).
//              N-split axis (C2 finish, 2026-08-31): the same kernel at
//              128/256/512 blocks halves the per-block weight stream each
//              step (128 -> 64 -> 32 -> 16 KB) while keeping the full-K
//              loop, to test whether more independent streams recover the
//              in-flight bandwidth. Same kernel, new instantiations only.
//
// Checks: both vs an fp32 CPU reference (dequant + matmul, fp16-rounded
// weights), and v2 vs current (must agree within fp16 accumulation noise —
// the current kernel's fp16-atomic order is itself nondeterministic).
// Times: 1000-launch GPU span (launch-regime) for both.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
  exit(1); } } while (0)

// ---------------------------------------------------------------------------
// Shared device helpers (verbatim from csrc/rocm/moe_q_gemm_gfx906.cu)
// ---------------------------------------------------------------------------

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

__forceinline__ __device__ void loadN_zeros(const uint32_t* qzeros_row, int n,
                                            int (&zeros)[4]) {
  uint32_t d = qzeros_row[n / 8] >> ((n & 0x07) * 4);
  #pragma unroll
  for (int i = 0; i < 4; ++i) zeros[i] = (int)((d >> (4 * i)) & 0xF);
}

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

__forceinline__ __device__ void dequant_4bit_8_fp16(uint32_t qa, half2 (&dq)[4],
                                                    half2 (&z1z16)[2],
                                                    half2 (&y1y16)[2]) {
  const uint32_t c0 = 0x64006400;
  union {
    uint32_t u;
    half2 h2;
  } q0, q1, q2, q3;
  q0.u = (qa & 0x000F000F) | c0;
  q1.u = (qa & 0x00F000F0) | c0;
  uint32_t qa_hi = qa >> 8;
  q2.u = (qa_hi & 0x000F000F) | c0;
  q3.u = (qa_hi & 0x00F000F0) | c0;
  dq[0] = __hfma2(q0.h2, y1y16[0], z1z16[0]);
  dq[1] = __hfma2(q1.h2, y1y16[1], z1z16[1]);
  dq[2] = __hfma2(q2.h2, y1y16[0], z1z16[0]);
  dq[3] = __hfma2(q3.h2, y1y16[1], z1z16[1]);
}

// ---------------------------------------------------------------------------
// Current kernel (verbatim, <1,4>), trimmed to the harness arg list.
// ---------------------------------------------------------------------------

#define CURRENT_THREADS_X 256
#define CURRENT_BLOCK_KN 256
static constexpr int NPT = 4;

__global__ void __launch_bounds__(CURRENT_THREADS_X)
    moe_gemm_q4_current(const half* __restrict__ a, half* __restrict__ c,
                        const uint32_t* __restrict__ b_q_weight,
                        const half* __restrict__ b_scales,
                        const uint32_t* __restrict__ b_qzeros,
                        const float* __restrict__ topk_weights,
                        const int32_t* __restrict__ sorted_token_ids,
                        const int32_t* __restrict__ expert_ids,
                        const int32_t* __restrict__ num_tokens_post_padded,
                        const int size_m, const int size_n, const int size_k,
                        const int groups, const int top_k,
                        const int expert_weight_stride,
                        const int expert_scales_stride,
                        const int expert_zeros_stride, const bool mul_topk_w,
                        const int output_topk, const int zero_offset) {
  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * CURRENT_BLOCK_KN * NPT;
  const int offset_k = blockIdx.z * CURRENT_BLOCK_KN;
  const int end_k = min(offset_k + CURRENT_BLOCK_KN, size_k);
  const int n = offset_n + t * NPT;

  if (token_block >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  const uint32_t* expert_weights =
      b_q_weight + (int64_t)expert_id * expert_weight_stride;
  const half* expert_scales =
      b_scales + (int64_t)expert_id * expert_scales_stride;
  const uint32_t* expert_qzeros =
      b_qzeros + (int64_t)expert_id * expert_zeros_stride;

  constexpr int LDS_PAD = 8;
  __shared__ half block_a[1][CURRENT_BLOCK_KN + LDS_PAD];

  if (offset_k + t < end_k) {
    int32_t token_id = sorted_token_ids[token_block];
    int token_row = token_id / top_k;
    half av = (token_row < size_m)
                  ? a[(int64_t)token_row * size_k + offset_k + t]
                  : __float2half_rn(0.0f);
    block_a[0][t] = av;
  }
  __syncthreads();

  if (n >= size_n) return;

  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;
  int qk = offset_k / 8;
  const uint32_t* b_ptr = expert_weights + qk * size_n + n;

  half2 z1z16_h[NPT][2], y1y16_h[NPT][2];
  auto refresh_group = [&](int g) {
    const uint32_t* qz_row = expert_qzeros + g * (size_n / 8);
    const half* sc_row = expert_scales + g * size_n;
    int zeros[NPT];
    loadN_zeros(qz_row, n, zeros);
    #pragma unroll
    for (int i = 0; i < NPT; ++i) {
      half scale = sc_row[n + i];
      prep_zero_scale_fp16((uint32_t)(zeros[i] + zero_offset), scale,
                           z1z16_h[i], y1y16_h[i]);
    }
  };
  refresh_group(group);

  float block_c[NPT] = {0.0f, 0.0f, 0.0f, 0.0f};
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
      int4 v = *(const int4*)(b_ptr + j * size_n);
      b_w[j][0] = v.x;
      b_w[j][1] = v.y;
      b_w[j][2] = v.z;
      b_w[j][3] = v.w;
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
          reinterpret_cast<const half*>(&block_a[0][a_off]);
      #pragma unroll
      for (int i = 0; i < NPT; ++i) block_c[i] += dot22_8_f(dq[i], a_ptr);
    }
    k += 32;
  }

  int32_t token_id = sorted_token_ids[token_block];
  if (token_id / top_k >= size_m) return;
  if (mul_topk_w && topk_weights != nullptr) {
    float tw = topk_weights[token_id];
    #pragma unroll
    for (int j = 0; j < NPT; ++j) block_c[j] *= tw;
  }
  int64_t out_row = (output_topk > 0) ? (int64_t)(token_id / output_topk)
                                      : (int64_t)token_id;
  half* out = c + out_row * size_n + n;
  half2 r01 = __halves2half2(__float2half_rn(block_c[0]),
                             __float2half_rn(block_c[1]));
  half2 r23 = __halves2half2(__float2half_rn(block_c[2]),
                             __float2half_rn(block_c[3]));
  unsigned long long* addr_u =
      reinterpret_cast<unsigned long long*>(out);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], r01);
    sum.h2[1] = __hadd2(cur.h2[1], r23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

// ---------------------------------------------------------------------------
// V2-B: 512 threads = 8 waves x 64 lanes. Lane-based columns (n = tl*2 + i,
// 128 cols/block) so every wave sees the SAME 128 columns; wave w owns K
// slice w*(size_k/8). fp32 per-wave partials [wave][lane][2] reduced through
// LDS, then direct store (gemm1) or 8-way CAS (gemm2 output_topk).
// ---------------------------------------------------------------------------

template <int THREADS, int NPT, int SLICE, bool DIRECT = false>
__global__ void __launch_bounds__(THREADS)
    moe_gemm_q4_v2(const half* __restrict__ a, half* __restrict__ c,
                   const uint32_t* __restrict__ b_q_weight,
                   const half* __restrict__ b_scales,
                   const uint32_t* __restrict__ b_qzeros,
                   const float* __restrict__ topk_weights,
                   const int32_t* __restrict__ sorted_token_ids,
                   const int32_t* __restrict__ expert_ids,
                   const int32_t* __restrict__ num_tokens_post_padded,
                   const int size_m, const int size_n, const int size_k,
                   const int groups, const int top_k,
                   const int expert_weight_stride,
                   const int expert_scales_stride,
                   const int expert_zeros_stride, const bool mul_topk_w,
                   const int output_topk, const int zero_offset,
                   float* dbg) {
  constexpr int NWAVES = THREADS / 64;
  constexpr int BLOCK_COLS = 64 * NPT;
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
      b_qzeros + (int64_t)expert_id * expert_zeros_stride;

  static_assert(SLICE % 32 == 0);
  static_assert(SLICE >= 32);
  constexpr int LDS_PAD = 8;
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
    const uint32_t* qz_row = expert_qzeros + g * (size_n / 8);
    const half* sc_row = expert_scales + g * size_n;
    int zeros[NPT];
    uint32_t d = qz_row[n / 8] >> ((n & 0x07) * 4);
    #pragma unroll
    for (int i = 0; i < NPT; ++i) zeros[i] = (int)((d >> (4 * i)) & 0xF);
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
      } else if (NPT == 2) {
        uint2 v = *(const uint2*)(b_ptr + j * size_n);
        b_w[j][0] = v.x;
        b_w[j][1] = v.y;
      } else {
        b_w[j][0] = *(const uint32_t*)(b_ptr + j * size_n);
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

  if (dbg && tl == 0) {
    dbg[blockIdx.y * 8 + w] = r[0];
    dbg[64 + blockIdx.y * 8 + w] = r[1];
    dbg[128 + blockIdx.y * 8 + w] = r[2];
    dbg[192 + blockIdx.y * 8 + w] = r[3];
  }
  // All waves hold the same reduced value for the same (lane-based)
  // columns: only wave 0 performs the epilogue (direct stores are
  // idempotent, but the gemm2 CAS must fire exactly once per cell).
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
  if (output_topk > 0 && !DIRECT) {
    // gemm2: x-blocks (one per expert slot) share the token row
    if (NPT == 4) {
      atomic_add_pk4_f16(out, __halves2half2(__float2half_rn(r[0]),
                                             __float2half_rn(r[1])),
                         __halves2half2(__float2half_rn(r[2]),
                                        __float2half_rn(r[3])));
    } else if (NPT == 2) {
      atomic_add_pk2_f16(out, __halves2half2(__float2half_rn(r[0]),
                                             __float2half_rn(r[1])));
    } else {
      atomicAdd(out, __float2half_rn(r[0]));
    }
  } else {
    if (NPT >= 4) {
      *(half2*)out = __halves2half2(__float2half_rn(r[0]),
                                    __float2half_rn(r[1]));
      *(half2*)(out + 2) = __halves2half2(__float2half_rn(r[2]),
                                          __float2half_rn(r[3]));
    } else if (NPT == 2) {
      *(half2*)out = __halves2half2(__float2half_rn(r[0]),
                                    __float2half_rn(r[1]));
    } else {
      *out = __float2half_rn(r[0]);
    }
  }
}


// ---------------------------------------------------------------------------
// Sweep: the current kernel templated on (THREADS_X == BLOCK_KN, NPT), to
// map the z-split granularity axis: finer K-splits -> more independent
// weight streams (more in-flight bytes) but higher CAS fan-in per cell.
// (256, 4) is the in-tree config; it must reproduce `current`'s numbers.
// ---------------------------------------------------------------------------

template <int THREADS_X, int NPT>
__global__ void __launch_bounds__(THREADS_X)
    moe_gemm_q4_sweep(const half* __restrict__ a, half* __restrict__ c,
                      const uint32_t* __restrict__ b_q_weight,
                      const half* __restrict__ b_scales,
                      const uint32_t* __restrict__ b_qzeros,
                      const float* __restrict__ topk_weights,
                      const int32_t* __restrict__ sorted_token_ids,
                      const int32_t* __restrict__ expert_ids,
                      const int32_t* __restrict__ num_tokens_post_padded,
                      const int size_m, const int size_n, const int size_k,
                      const int groups, const int top_k,
                      const int expert_weight_stride,
                      const int expert_scales_stride,
                      const int expert_zeros_stride, const bool mul_topk_w,
                      const int output_topk, const int zero_offset) {
  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * THREADS_X * NPT;
  const int offset_k = blockIdx.z * THREADS_X;
  const int end_k = min(offset_k + THREADS_X, size_k);
  const int n = offset_n + t * NPT;

  if (token_block >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  const uint32_t* expert_weights =
      b_q_weight + (int64_t)expert_id * expert_weight_stride;
  const half* expert_scales =
      b_scales + (int64_t)expert_id * expert_scales_stride;
  const uint32_t* expert_qzeros =
      b_qzeros + (int64_t)expert_id * expert_zeros_stride;

  constexpr int LDS_PAD = 8;
  __shared__ half block_a[1][THREADS_X + LDS_PAD];

  if (offset_k + t < end_k) {
    int32_t token_id = sorted_token_ids[token_block];
    int token_row = token_id / top_k;
    half av = (token_row < size_m)
                  ? a[(int64_t)token_row * size_k + offset_k + t]
                  : __float2half_rn(0.0f);
    block_a[0][t] = av;
  }
  __syncthreads();

  if (n >= size_n) return;

  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;
  int qk = offset_k / 8;
  const uint32_t* b_ptr = expert_weights + qk * size_n + n;

  half2 z1z16_h[NPT][2], y1y16_h[NPT][2];
  auto refresh_group = [&](int g) {
    const uint32_t* qz_row = expert_qzeros + g * (size_n / 8);
    const half* sc_row = expert_scales + g * size_n;
    int zeros[4];
    loadN_zeros(qz_row, n, zeros);
    #pragma unroll
    for (int i = 0; i < NPT; ++i) {
      half scale = sc_row[n + i];
      prep_zero_scale_fp16((uint32_t)(zeros[i] + zero_offset), scale,
                           z1z16_h[i], y1y16_h[i]);
    }
  };
  refresh_group(group);

  float block_c[NPT];
  #pragma unroll
  for (int i = 0; i < NPT; ++i) block_c[i] = 0.0f;
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
      if constexpr (NPT == 4) {
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
      half2 dq[NPT][4];
      #pragma unroll
      for (int i = 0; i < NPT; ++i)
        dequant_4bit_8_fp16(b_w[j][i], dq[i], z1z16_h[i], y1y16_h[i]);
      const half* a_ptr =
          reinterpret_cast<const half*>(&block_a[0][a_off]);
      #pragma unroll
      for (int i = 0; i < NPT; ++i) block_c[i] += dot22_8_f(dq[i], a_ptr);
    }
    k += 32;
  }

  int32_t token_id = sorted_token_ids[token_block];
  if (token_id / top_k >= size_m) return;
  if (mul_topk_w && topk_weights != nullptr) {
    float tw = topk_weights[token_id];
    #pragma unroll
    for (int j = 0; j < NPT; ++j) block_c[j] *= tw;
  }
  int64_t out_row = (output_topk > 0) ? (int64_t)(token_id / output_topk)
                                      : (int64_t)token_id;
  half* out = c + out_row * size_n + n;
  if constexpr (NPT == 4) {
    half2 r01 = __halves2half2(__float2half_rn(block_c[0]),
                               __float2half_rn(block_c[1]));
    half2 r23 = __halves2half2(__float2half_rn(block_c[2]),
                               __float2half_rn(block_c[3]));
    unsigned long long* addr_u =
        reinterpret_cast<unsigned long long*>(out);
    unsigned long long old = *addr_u;
    while (true) {
      union {
        unsigned long long u;
        half2 h2[2];
      } cur, sum;
      cur.u = old;
      sum.h2[0] = __hadd2(cur.h2[0], r01);
      sum.h2[1] = __hadd2(cur.h2[1], r23);
      unsigned long long prev = atomicCAS(addr_u, old, sum.u);
      if (prev == old) break;
      old = prev;
    }
  } else {
    half2 r01 = __halves2half2(__float2half_rn(block_c[0]),
                               __float2half_rn(block_c[1]));
    atomic_add_pk2_f16(out, r01);
  }
}

// ---------------------------------------------------------------------------
// V1 (C2): full-K per block, direct store. No CAS, no zeroing, no K-split.
// NOTE: the kernel groups threads in 32s (w = t/32); on this gfx906
// warpSize = 64 (hipDeviceProp probe), so THREADS=32 is half a wavefront.
//   v1a: THREADS=32, NPT=4 — lane owns 4 columns, 128 cols/block; the
//         roadmap's "64 blocks, each a single 32-thread group looping K".
//   v1b: THREADS=128 (4 groups of 32), NPT=1 — group w owns the disjoint
//         32-col strip [w*32, w*32+32); same total thread count as current,
//         isolates the CAS/z-split structure cost.
//   v1c/v1d/v1e (C2 finish): N-split axis — 128/256/512 blocks with 64/32/16
//         cols per block; see "V1 N-split axis" in DEVLOG-moe-c2v.md.
// gemm1-only (direct store; M=1: every slot reads activation row 0).
// ---------------------------------------------------------------------------

template <int THREADS, int NPT, int K_T>
__global__ void __launch_bounds__(THREADS)
    moe_gemm_q4_v1(const half* __restrict__ a, half* __restrict__ c,
                   const uint32_t* __restrict__ b_q_weight,
                   const half* __restrict__ b_scales,
                   const uint32_t* __restrict__ b_qzeros,
                   const int32_t* __restrict__ sorted_token_ids,
                   const int32_t* __restrict__ expert_ids,
                   const int32_t* __restrict__ num_tokens_post_padded,
                   const int size_m, const int size_n, const int size_k,
                   const int groups, const int top_k,
                   const int expert_weight_stride,
                   const int expert_scales_stride,
                   const int expert_zeros_stride, const int zero_offset) {
  static_assert(K_T % 32 == 0);
  const int t = threadIdx.x;
  const int w = t / 32;
  const int tl = t % 32;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * (THREADS * NPT);
  const int n = offset_n + w * 32 + tl * NPT;

  if (token_block >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  const uint32_t* expert_weights =
      b_q_weight + (int64_t)expert_id * expert_weight_stride;
  const half* expert_scales =
      b_scales + (int64_t)expert_id * expert_scales_stride;
  const uint32_t* expert_qzeros =
      b_qzeros + (int64_t)expert_id * expert_zeros_stride;

  __shared__ half block_a[K_T + 8];

  int32_t token_id = sorted_token_ids[token_block];
  int token_row = token_id / top_k;
  constexpr int FILL_ITERS = K_T / THREADS;
  #pragma unroll
  for (int i = 0; i < FILL_ITERS; ++i) {
    int pos = i * THREADS + t;
    block_a[pos] = (token_row < size_m)
                       ? a[(int64_t)token_row * size_k + pos]
                       : __float2half_rn(0.0f);
  }
  __syncthreads();

  if (n >= size_n) return;

  const int groupsize = size_k / groups;
  int group = 0;
  int nextgroup = groupsize;
  const uint32_t* b_ptr = expert_weights + n;

  half2 z1z16_h[NPT][2], y1y16_h[NPT][2];
  auto refresh_group = [&](int g) {
    const uint32_t* qz_row = expert_qzeros + g * (size_n / 8);
    const half* sc_row = expert_scales + g * size_n;
    int zeros[NPT];
    uint32_t d = qz_row[n / 8] >> ((n & 0x07) * 4);
    #pragma unroll
    for (int i = 0; i < NPT; ++i) zeros[i] = (int)((d >> (4 * i)) & 0xF);
    #pragma unroll
    for (int i = 0; i < NPT; ++i) {
      half scale = sc_row[n + i];
      prep_zero_scale_fp16((uint32_t)(zeros[i] + zero_offset), scale,
                           z1z16_h[i], y1y16_h[i]);
    }
  };
  refresh_group(0);

  float block_c[NPT];
  #pragma unroll
  for (int i = 0; i < NPT; ++i) block_c[i] = 0.0f;
  int k = 0;
  uint32_t b_w[4][NPT];
  while (k < K_T) {
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
      } else if (NPT == 2) {
        uint2 v = *(const uint2*)(b_ptr + j * size_n);
        b_w[j][0] = v.x;
        b_w[j][1] = v.y;
      } else {
        b_w[j][0] = *(const uint32_t*)(b_ptr + j * size_n);
      }
    }
    b_ptr += 4 * size_n;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = k + 8 * j;
      half2 dq[NPT][4];
      #pragma unroll
      for (int i = 0; i < NPT; ++i)
        dequant_4bit_8_fp16(b_w[j][i], dq[i], z1z16_h[i], y1y16_h[i]);
      const half* a_ptr =
          reinterpret_cast<const half*>(&block_a[a_off]);
      #pragma unroll
      for (int i = 0; i < NPT; ++i) block_c[i] += dot22_8_f(dq[i], a_ptr);
    }
    k += 32;
  }

  half* out = c + (int64_t)token_id * size_n + n;
  if (NPT >= 4) {
    *(half2*)out = __halves2half2(__float2half_rn(block_c[0]),
                                  __float2half_rn(block_c[1]));
    *(half2*)(out + 2) = __halves2half2(__float2half_rn(block_c[2]),
                                        __float2half_rn(block_c[3]));
  } else if (NPT == 2) {
    *(half2*)out = __halves2half2(__float2half_rn(block_c[0]),
                                  __float2half_rn(block_c[1]));
  } else {
    *out = __float2half_rn(block_c[0]);
  }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------



struct L2 {
  half* d_c3;
  const uint32_t* d_w; const half* d_sc; const uint32_t* d_z;
  const float* d_topkw; const int32_t* d_sorted, *d_eids, *d_npp;
  const half* d_a;
  int M, N, K, GROUPS, TOPK, wstride, sstride, zstride;
  float* dbg = nullptr;
  void cur() {
    dim3 blk(CURRENT_THREADS_X);
    dim3 grid(M * TOPK, N / (CURRENT_BLOCK_KN * 4),
              (K + CURRENT_BLOCK_KN - 1) / CURRENT_BLOCK_KN);
    moe_gemm_q4_current<<<grid, blk>>>(
        d_a, d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, M, N, K,
        GROUPS, TOPK, wstride, sstride, zstride, true, TOPK, 0);
    CHECK(hipGetLastError());
  }
  template <int THREADS, int NPT_, int SLICE>
  void v2() {
    constexpr int BC = 64 * NPT_;
    dim3 blk(THREADS);
    dim3 grid(M * TOPK, N / BC);
    moe_gemm_q4_v2<THREADS, NPT_, SLICE><<<grid, blk>>>(
        d_a, d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, M, N, K,
        GROUPS, TOPK, wstride, sstride, zstride, true, TOPK, 0, dbg);
    CHECK(hipGetLastError());
  }
  template <int THREADS, int NPT_, int SLICE>
  void g1d(half* c_buf) {  // gemm2 args, forced direct store (A/B)
    constexpr int BC = 64 * NPT_;
    dim3 blk(THREADS);
    dim3 grid(M * TOPK, N / BC);
    moe_gemm_q4_v2<THREADS, NPT_, SLICE, true><<<grid, blk>>>(
        d_a, c_buf, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, M, N, K,
        GROUPS, TOPK, wstride, sstride, zstride, true, TOPK, 0, nullptr);
    CHECK(hipGetLastError());
  }
  template <int THREADS, int NPT_, int SLICE>
  void g1(half* c_buf) {  // gemm1 semantics: direct store, no topk weight
    constexpr int BC = 64 * NPT_;
    dim3 blk(THREADS);
    dim3 grid(M * TOPK, N / BC);
    moe_gemm_q4_v2<THREADS, NPT_, SLICE><<<grid, blk>>>(
        d_a, c_buf, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, M, N, K,
        GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0, nullptr);
    CHECK(hipGetLastError());
  }
};

struct V2L {
  half* d_a; half* d_c1;
  const uint32_t* d_w; const half* d_sc; const uint32_t* d_z;
  const float* d_topkw;
  const int32_t* d_sorted; const int32_t* d_eids; const int32_t* d_npp;
  int M, N, K, GROUPS, TOPK, wstride, sstride, zstride;
  template <int THREADS, int NPT_, int SLICE>
  void operator()() {
    constexpr int BLOCK_COLS = 64 * NPT_;
    dim3 blk(THREADS);
    dim3 grid(M * TOPK, N / BLOCK_COLS);
    moe_gemm_q4_v2<THREADS, NPT_, SLICE><<<grid, blk>>>(
        d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, M, N, K,
        GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0, nullptr);
    CHECK(hipGetLastError());
  }
};

// Shuffle-order unpack of one int32: bit offset 4*j holds k position
// [0,2,4,6,1,3,5,7][j]
static void unpack8(uint32_t qa, int (&q)[8]) {
  q[0] = qa & 0xF;          // bits  3:0
  q[1] = (qa >> 16) & 0xF;  // bits 19:16
  q[2] = (qa >> 4) & 0xF;   // bits  7:4
  q[3] = (qa >> 20) & 0xF;  // bits 23:20
  q[4] = (qa >> 8) & 0xF;   // bits 11:8
  q[5] = (qa >> 24) & 0xF;  // bits 27:24
  q[6] = (qa >> 12) & 0xF;  // bits 15:12
  q[7] = (qa >> 28) & 0xF;  // bits 31:28
}

int main(int argc, char** argv) {
  const int K = argc > 2 ? atoi(argv[2]) : 2048;
  const bool isolated = (argc > 3 && atoi(argv[3]) == 2);
  const int mode3 = (argc > 3 ? atoi(argv[3]) : 0);
  const int N = 1024, E = 8, TOPK = 8, M = 1;
  const int GROUPS = K / 128;
  const int EM = M * TOPK;
  const int niter = argc > 1 ? atoi(argv[1]) : 1000;
  printf("K=%d GROUPS=%d\n", K, GROUPS);

  // host data
  std::vector<half> h_a((size_t)M * K);
  std::vector<uint32_t> h_w((size_t)E * (K / 8) * N);
  std::vector<half> h_sc((size_t)E * GROUPS * N);
  std::vector<uint32_t> h_z((size_t)E * GROUPS * (N / 8));
  srand(7);
  auto rnd16 = [] { return (rand() % 4000 - 2000) / 2000.0f; };
  for (auto& x : h_a) x = __float2half_rn(rnd16());
  for (auto& x : h_sc) x = __float2half_rn(0.01f + (rand() % 100) / 5000.f);
  for (int e = 0; e < E; ++e)
    for (int g = 0; g < GROUPS; ++g)
      for (int nn = 0; nn < N / 8; ++nn) {
        uint32_t d = 0;
        for (int j = 0; j < 8; ++j)
          d |= (uint32_t)(rand() % 16) << (4 * j);
        h_z[(size_t)e * GROUPS * (N / 8) + g * (N / 8) + nn] = d;
      }
  for (int e = 0; e < E; ++e)
    for (int qk = 0; qk < K / 8; ++qk)
      for (int nn = 0; nn < N; ++nn) {
        uint32_t d = 0;
        for (int j = 0; j < 8; ++j)
          d |= (uint32_t)(rand() % 16) << (4 * j);
        h_w[(size_t)e * (K / 8) * N + (size_t)qk * N + nn] = d;
      }

  std::vector<int32_t> h_sorted(EM), h_eids(EM);
  for (int i = 0; i < EM; ++i) {
    h_sorted[i] = i;   // slot i -> token 0, slot i
    h_eids[i] = i % E; // expert for slot i
  }
  int32_t h_npp = EM;
  std::vector<float> h_topkw(EM, 1.0f);

  // device
  half *d_a, *d_c1, *d_c2, *d_c3;
  uint32_t *d_w, *d_z;
  half* d_sc;
  float* d_topkw;
  int32_t *d_sorted, *d_eids, *d_npp;
  CHECK(hipMalloc(&d_a, M * K * sizeof(half)));
  CHECK(hipMalloc(&d_c1, EM * N * sizeof(half)));
  CHECK(hipMalloc(&d_c2, EM * N * sizeof(half)));
  CHECK(hipMalloc(&d_c3, M * N * sizeof(half)));
  CHECK(hipMalloc(&d_w, h_w.size() * sizeof(uint32_t)));
  CHECK(hipMalloc(&d_sc, h_sc.size() * sizeof(half)));
  CHECK(hipMalloc(&d_z, h_z.size() * sizeof(uint32_t)));
  CHECK(hipMalloc(&d_topkw, EM * sizeof(float)));
  CHECK(hipMalloc(&d_sorted, EM * sizeof(int32_t)));
  CHECK(hipMalloc(&d_eids, EM * sizeof(int32_t)));
  CHECK(hipMalloc(&d_npp, sizeof(int32_t)));
  CHECK(hipMemcpy(d_a, h_a.data(), M * K * sizeof(half), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(uint32_t),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_sc, h_sc.data(), h_sc.size() * sizeof(half),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_z, h_z.data(), h_z.size() * sizeof(uint32_t),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_topkw, h_topkw.data(), EM * sizeof(float),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_sorted, h_sorted.data(), EM * sizeof(int32_t),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_eids, h_eids.data(), EM * sizeof(int32_t),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_npp, &h_npp, sizeof(int32_t), hipMemcpyHostToDevice));

  const int wstride = (K / 8) * N, sstride = GROUPS * N, zstride = GROUPS * N / 8;

  // ---- CPU reference (gemm1): out[slot][n] = sum_k a[0][k] * W[e][k][n]
  std::vector<float> ref(EM * N, 0.f);
  for (int slot = 0; slot < EM; ++slot) {
    int e = h_eids[slot];
    for (int nn = 0; nn < N; ++nn) {
      float acc = 0.f;
      for (int qk = 0; qk < K / 8; ++qk) {
        int g = (qk * 8) / (K / GROUPS);
        half scale = h_sc[(size_t)e * GROUPS * N + g * N + nn];
        uint32_t zd =
            h_z[(size_t)e * GROUPS * (N / 8) + g * (N / 8) + nn / 8];
        int zn = (zd >> ((nn & 7) * 4)) & 0xF;
        int q[8];
        unpack8(h_w[(size_t)e * (K / 8) * N + (size_t)qk * N + nn], q);
        for (int j = 0; j < 8; ++j) {
          float wv =
              ((float)q[j] - (float)zn) * __half2float(scale);
          acc += __half2float(h_a[(size_t)0 * K + qk * 8 + j]) * wv;
        }
      }
      ref[(size_t)slot * N + nn] = acc;
    }
  }

  // ---- run current (gemm1: pre-zeroed c, z=K/256)
  auto run_current = [&]() {
    CHECK(hipMemset(d_c1, 0, EM * N * sizeof(half)));
    dim3 blk(CURRENT_THREADS_X);
    dim3 grid(EM, (N + CURRENT_BLOCK_KN * NPT - 1) / (CURRENT_BLOCK_KN * NPT),
              (K + CURRENT_BLOCK_KN - 1) / CURRENT_BLOCK_KN);
    moe_gemm_q4_current<<<grid, blk>>>(
        d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, M, N, K,
        GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
    CHECK(hipGetLastError());
  };
  // ---- run v2 (gemm1: direct store, no zero)
  V2L v2{d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted,
          d_eids, d_npp, M, N, K, GROUPS, TOPK, wstride,
          sstride, zstride};

  int bad = 0;
  // current vs ref
  run_current();
  CHECK(hipDeviceSynchronize());
  std::vector<half> hc(EM * N);
  CHECK(hipMemcpy(hc.data(), d_c1, EM * N * sizeof(half),
                  hipMemcpyDeviceToHost));
  float maxerr = 0, maxrel = 0;
  for (size_t i = 0; i < hc.size(); ++i) {
    float d = fabsf(__half2float(hc[i]) - ref[i]);
    maxerr = std::max(maxerr, d);
    maxrel = std::max(maxrel, d / (fabsf(ref[i]) + 1e-3f));
  }
  printf("current vs cpu-ref: max abs err %.4f max rel %.3e\n",
         maxerr, maxrel);
  if (maxerr > 0.35f) { printf("CURRENT FAILS REF\n"); bad = 1; }

  // v2 vs ref
  v2.template operator()<512, 2, 256>();
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(hc.data(), d_c1, EM * N * sizeof(half),
                  hipMemcpyDeviceToHost));
  maxerr = 0;
  int ninf = 0, firstinf = -1;
  for (size_t i = 0; i < hc.size(); ++i) {
    float v = __half2float(hc[i]);
    if (__builtin_isinf(v) || __builtin_isnan(v)) {
      ++ninf;
      if (firstinf < 0) firstinf = (int)i;
    }
    maxerr = std::max(maxerr, fabsf(v - ref[i]));
  }
  printf("v2(512,2) vs cpu-ref: max abs err %.4f (%d inf/nan)",
         maxerr, ninf);
  if (firstinf >= 0 && firstinf + 1 < (int)hc.size())
    printf(" first at slot %d col %d (ref %.4f, next %.4f)",
           firstinf / N, firstinf % N, ref[firstinf],
           __half2float(hc[firstinf + 1]));
  printf("\n");
  if (maxerr > 0.35f || ninf) { printf("V2 FAILS REF\n"); bad = 1; }

  // v2 vs current (fp16-noise tolerance; current's atomic order varies)
  CHECK(hipMemset(d_c2, 0, EM * N * sizeof(half)));
  {
    dim3 blk(CURRENT_THREADS_X);
    dim3 grid(EM, (N + CURRENT_BLOCK_KN * NPT - 1) / (CURRENT_BLOCK_KN * NPT),
              (K + CURRENT_BLOCK_KN - 1) / CURRENT_BLOCK_KN);
    moe_gemm_q4_current<<<grid, blk>>>(
        d_a, d_c2, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, M, N, K,
        GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
    CHECK(hipDeviceSynchronize());
  }
  v2.template operator()<512, 2, 256>();
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(hc.data(), d_c1, EM * N * sizeof(half),
                  hipMemcpyDeviceToHost));
  std::vector<half> hc2(EM * N);
  CHECK(hipMemcpy(hc2.data(), d_c2, EM * N * sizeof(half),
                  hipMemcpyDeviceToHost));
  maxerr = 0;
  int nbad = 0;
  for (size_t i = 0; i < hc.size(); ++i) {
    float d = fabsf(__half2float(hc[i]) - __half2float(hc2[i]));
    maxerr = std::max(maxerr, d);
    if (d > 0.03f) ++nbad;
  }
  printf("v2(512,2) vs current: max abs err %.4f (%d cells > 0.03)\n",
         maxerr, nbad);
  if (maxerr > 0.15f) { printf("V2 DIVERGES FROM CURRENT\n"); bad = 1; }

  // ---- gemm2 path (output_topk=1): all 8 slots sum into token row 0,
  //      current CAS vs v2 CAS, vs CPU ref
  {
    // CPU ref: out[0][n] = sum_slot a[0] . W[e_slot][n]  (topk_w = 1.0)
    std::vector<float> ref2(N, 0.f);
    for (int slot = 0; slot < EM; ++slot)
      for (int nn = 0; nn < N; ++nn) ref2[nn] += ref[(size_t)slot * N + nn];

    auto run2 = [&](auto launch, bool zero) {
      if (zero) CHECK(hipMemset(d_c3, 0, M * N * sizeof(half)));
      launch(true);
      CHECK(hipDeviceSynchronize());
    };
    float ref2max = 0;
    for (int nn = 0; nn < N; ++nn)
      ref2max = std::max(ref2max, fabsf(ref2[nn]));
    auto report = [&](const char* name, bool expect_dirty = false) {
      std::vector<half> h(N);
      CHECK(hipMemcpy(h.data(), d_c3, M * N * sizeof(half),
                      hipMemcpyDeviceToHost));
      float me = 0;
      int ni = 0;
      for (int nn = 0; nn < N; ++nn) {
        float v = __half2float(h[nn]);
        if (__builtin_isinf(v) || __builtin_isnan(v)) ++ni;
        me = std::max(me, fabsf(v - ref2[nn]));
      }
      // 8-slot fp16 CAS accumulation is ~0.6 abs err on ~20-magnitude
      // random data: gate on relative error (the in-repo test suite's
      // 5e-2 relative tolerance), not absolute.
      float rel = me / (ref2max + 1e-6f);
      if (!strncmp(name, "v2-512t4col", 11))
        for (int j = 0; j < 16; ++j)
          printf("  [%2d] got %10.4f ref2 %10.4f\n", j, __half2float(h[j]),
                 ref2[j]);
      printf("gemm2 %s: max abs err %.4f (rel %.3e, %d inf/nan)\n", name, me,
             rel, ni);
      if (expect_dirty) {
        // negative control: running on the pre-zeroed buffer must NOT
        // happen; a clean result here means the control is stale
        if (rel < 0.5f) bad = 1;
      } else if (rel > 5e-2f || ni) {
        bad = 1;
      }
    };

    L2 l{d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
         M, N, K, GROUPS, TOPK, wstride, sstride, zstride};

    run2([&] (bool) { l.cur(); }, true);
    report("current(8x z-CAS)");
    run2([&] (bool) { l.v2<512, 4, 256>(); }, true);
    report("v2-512t4col(8x CAS)");
    run2([&] (bool) { l.v2<512, 2, 256>(); }, true);
    report("v2-512t2col(8x CAS)");
    run2([&] (bool) { l.v2<512, 4, 256>(); }, false);
    report("v2-nozero-dirty(8x CAS)", /*expect_dirty=*/true);
  }

  // ---- gemm1 with 4col variant (direct store, no CAS at all)
  {
    L2 l{d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
         M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
    l.g1<512, 4, 256>(d_c1);
    CHECK(hipDeviceSynchronize());
    std::vector<half> h(EM * N);
    CHECK(hipMemcpy(h.data(), d_c1, EM * N * sizeof(half),
                    hipMemcpyDeviceToHost));
    float me = 0;
    for (size_t i = 0; i < h.size(); ++i)
      me = std::max(me, fabsf(__half2float(h[i]) - ref[i]));
    printf("gemm1 v2 512t4col direct-store (vs cpu-ref): max err %.4f\n", me);
    if (me > 0.35f) bad = 1;
  }

  // ---- V1 (C2): full-K direct-store variants, gemm1 (K=2048 only)
  auto run_v1 = [&](auto launch, const char* name) {
    launch();
    CHECK(hipDeviceSynchronize());
    std::vector<half> h(EM * N);
    CHECK(hipMemcpy(h.data(), d_c1, EM * N * sizeof(half),
                    hipMemcpyDeviceToHost));
    float me = 0;
    for (size_t i = 0; i < h.size(); ++i)
      me = std::max(me, fabsf(__half2float(h[i]) - ref[i]));
    printf("gemm1 %s (vs cpu-ref): max err %.4f\n", name, me);
    if (me > 0.35f) bad = 1;
  };
  if (K == 2048) {
    run_v1([&] {
      dim3 blk(32);
      dim3 grid(EM, N / 128);
      moe_gemm_q4_v1<32, 4, 2048><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_sorted, d_eids, d_npp, M, N, K,
          GROUPS, TOPK, wstride, sstride, zstride, 0);
      CHECK(hipGetLastError());
    }, "v1a 32t 4col full-K direct");
    run_v1([&] {
      dim3 blk(128);
      dim3 grid(EM, N / 128);
      moe_gemm_q4_v1<128, 1, 2048><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_sorted, d_eids, d_npp, M, N, K,
          GROUPS, TOPK, wstride, sstride, zstride, 0);
      CHECK(hipGetLastError());
    }, "v1b 128t 1col full-K direct");
    // N-split axis (C2 finish): more blocks, shorter per-block weight
    // stream (64/32/16 KB vs v1a/v1b's 128 KB), same full-K loop.
    run_v1([&] {
      dim3 blk(32);
      dim3 grid(EM, N / 64);
      moe_gemm_q4_v1<32, 2, 2048><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_sorted, d_eids, d_npp, M, N, K,
          GROUPS, TOPK, wstride, sstride, zstride, 0);
      CHECK(hipGetLastError());
    }, "v1c 32t 2col full-K direct (128 blk)");
    run_v1([&] {
      dim3 blk(32);
      dim3 grid(EM, N / 32);
      moe_gemm_q4_v1<32, 1, 2048><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_sorted, d_eids, d_npp, M, N, K,
          GROUPS, TOPK, wstride, sstride, zstride, 0);
      CHECK(hipGetLastError());
    }, "v1d 32t 1col full-K direct (256 blk)");
    run_v1([&] {
      dim3 blk(16);
      dim3 grid(EM, N / 16);
      moe_gemm_q4_v1<16, 1, 2048><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_sorted, d_eids, d_npp, M, N, K,
          GROUPS, TOPK, wstride, sstride, zstride, 0);
      CHECK(hipGetLastError());
    }, "v1e 16t 1col full-K direct (512 blk)");
  }

  // ---- z-split sweep (gemm1 semantics: pre-zeroed, direct CAS fan-in)
  auto run_sweep = [&](auto launch, const char* name) {
    CHECK(hipMemset(d_c1, 0, EM * N * sizeof(half)));
    launch();
    CHECK(hipDeviceSynchronize());
    std::vector<half> h(EM * N);
    CHECK(hipMemcpy(h.data(), d_c1, EM * N * sizeof(half),
                    hipMemcpyDeviceToHost));
    float me = 0;
    for (size_t i = 0; i < h.size(); ++i)
      me = std::max(me, fabsf(__half2float(h[i]) - ref[i]));
    printf("sweep %s (vs cpu-ref): max err %.4f\n", name, me);
    if (me > 0.35f) bad = 1;
  };
  auto sweep_cfg = [&](int tx, int np_, auto launch) {
    if (K % tx != 0) {
      printf("sweep %dt NPT=%d skipped (K %% %d)\n", tx, np_, tx);
      return;
    }
    char name[32];
    snprintf(name, sizeof(name), "%dt NPT=%d", tx, np_);
    run_sweep(launch, name);
  };
  sweep_cfg(256, 4, [&] {
    dim3 blk(256);
    dim3 grid(EM, (N + 256 * 4 - 1) / (256 * 4), (K + 255) / 256);
    moe_gemm_q4_sweep<256, 4><<<grid, blk>>>(
        d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
        M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
    CHECK(hipGetLastError());
  });
  sweep_cfg(256, 2, [&] {
    dim3 blk(256);
    dim3 grid(EM, (N + 256 * 2 - 1) / (256 * 2), (K + 255) / 256);
    moe_gemm_q4_sweep<256, 2><<<grid, blk>>>(
        d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
        M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
    CHECK(hipGetLastError());
  });
  sweep_cfg(128, 4, [&] {
    dim3 blk(128);
    dim3 grid(EM, (N + 128 * 4 - 1) / (128 * 4), (K + 127) / 128);
    moe_gemm_q4_sweep<128, 4><<<grid, blk>>>(
        d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
        M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
    CHECK(hipGetLastError());
  });
  sweep_cfg(128, 2, [&] {
    dim3 blk(128);
    dim3 grid(EM, (N + 128 * 2 - 1) / (128 * 2), (K + 127) / 128);
    moe_gemm_q4_sweep<128, 2><<<grid, blk>>>(
        d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
        M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
    CHECK(hipGetLastError());
  });
  sweep_cfg(64, 4, [&] {
    dim3 blk(64);
    dim3 grid(EM, (N + 64 * 4 - 1) / (64 * 4), (K + 63) / 64);
    moe_gemm_q4_sweep<64, 4><<<grid, blk>>>(
        d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
        M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
    CHECK(hipGetLastError());
  });
  sweep_cfg(64, 2, [&] {
    dim3 blk(64);
    dim3 grid(EM, (N + 64 * 2 - 1) / (64 * 2), (K + 63) / 64);
    moe_gemm_q4_sweep<64, 2><<<grid, blk>>>(
        d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
        M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
    CHECK(hipGetLastError());
  });
  sweep_cfg(512, 4, [&] {
    dim3 blk(512);
    dim3 grid(EM, (N + 512 * 4 - 1) / (512 * 4), (K + 511) / 512);
    moe_gemm_q4_sweep<512, 4><<<grid, blk>>>(
        d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
        M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
    CHECK(hipGetLastError());
  });
  sweep_cfg(512, 2, [&] {
    dim3 blk(512);
    dim3 grid(EM, (N + 512 * 2 - 1) / (512 * 2), (K + 511) / 512);
    moe_gemm_q4_sweep<512, 2><<<grid, blk>>>(
        d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
        M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
    CHECK(hipGetLastError());
  });

  // ---- single-slot gemm2 (npp=1): only x-block 0 contributes
  {
    int32_t one_npp = 1;
    CHECK(hipMemcpy(d_npp, &one_npp, sizeof(int32_t), hipMemcpyHostToDevice));
    CHECK(hipMemset(d_c3, 0, M * N * sizeof(half)));
    float* d_dbg2;
    CHECK(hipMalloc(&d_dbg2, 4 * 512 * sizeof(float)));
    L2 l{d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
         M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
    l.dbg = d_dbg2;
    l.v2<512, 4, 256>();
    CHECK(hipDeviceSynchronize());
    float hd[4 * 512];
    CHECK(hipMemcpy(hd, d_dbg2, 4 * 512 * sizeof(float),
                    hipMemcpyDeviceToHost));
    CHECK(hipFree(d_dbg2));
    printf("  dbg r[0] per thread(y*64+w): ");
    for (int y = 0; y < 4; ++y)
      for (int w2 = 0; w2 < 8; ++w2) printf("%.2f ", hd[(y * 64 + w2)]);
    printf("\n");
    std::vector<half> h(N);
    CHECK(hipMemcpy(h.data(), d_c3, M * N * sizeof(half),
                    hipMemcpyDeviceToHost));
    float me = 0;
    for (int nn = 0; nn < N; ++nn)
      me = std::max(me, fabsf(__half2float(h[nn]) - ref[(size_t)0 * N + nn]));
    printf("gemm2 v2 single-slot (vs ref slot0): max err %.4f", me);
    for (int j = 0; j < 16; ++j)
      printf("\n  [%2d] got %12.4f ref %12.4f ratio %.3f", j,
             __half2float(h[j]), ref[j],
             fabsf(ref[j]) > 1e-3f ? __half2float(h[j]) / ref[j] : 0.f);
    printf("\n  bad cells (|ratio-1|>0.3) among 0..255:\n");
    for (int j = 0; j < 256; ++j) {
      float r = fabsf(ref[j]) > 1e-3f ? __half2float(h[j]) / ref[j] : 0.f;
      if (fabsf(r - 1.f) > 0.3f)
        printf("    col %3d tl=%d got %9.4f ref %9.4f ratio %7.3f\n", j,
               j / 4, __half2float(h[j]), ref[j], r);
    }
    printf("\n");
    if (me > 0.35f) bad = 1;
    CHECK(hipMemcpy(d_npp, &h_npp, sizeof(int32_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_npp, &h_npp, sizeof(int32_t), hipMemcpyHostToDevice));
  }

  // ---- timing (GPU span of niter launches, both)
  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  for (int i = 0; i < 50; ++i) run_current();
  CHECK(hipStreamSynchronize(0));
  hipEventRecord(e0, 0);
  for (int i = 0; i < niter; ++i) run_current();
  hipEventRecord(e1, 0);
  CHECK(hipStreamSynchronize(0));
  float ms;
  hipEventElapsedTime(&ms, e0, e1);
  printf("current          : %.2f us/launch (gemm1, pre-zeroed)\n",
         ms * 1000.f / niter);

  auto time_v2 = [&](auto kernel, const char* name) {
    for (int i = 0; i < 50; ++i) kernel();
    CHECK(hipStreamSynchronize(0));
    hipEventRecord(e0, 0);
    for (int i = 0; i < niter; ++i) kernel();
    hipEventRecord(e1, 0);
    CHECK(hipStreamSynchronize(0));
    float m;
    hipEventElapsedTime(&m, e0, e1);
    printf("v2 %-14s: %.2f us/launch\n", name, m * 1000.f / niter);
  };
  time_v2([&] { v2.template operator()<512, 2, 256>(); }, "512t 2col s256");
  time_v2([&] { v2.template operator()<512, 4, 256>(); }, "512t 4col s256");
  time_v2([&] { v2.template operator()<256, 2, 512>(); }, "256t 2col s512");
  time_v2([&] { v2.template operator()<128, 1, 1024>(); }, "128t 1col s1024");
  if (K == 2048) {
    // also time current kernel WITHOUT the zeroing memset (kernel only)
    for (int i = 0; i < 50; ++i) {
      dim3 blk(CURRENT_THREADS_X);
      dim3 grid(EM, (N + CURRENT_BLOCK_KN * NPT - 1) / (CURRENT_BLOCK_KN * NPT),
                (K + CURRENT_BLOCK_KN - 1) / CURRENT_BLOCK_KN);
      moe_gemm_q4_current<<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, M, N, K,
          GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
      CHECK(hipGetLastError());
    }
    CHECK(hipStreamSynchronize(0));
    hipEventRecord(e0, 0);
    for (int i = 0; i < niter; ++i) {
      dim3 blk(CURRENT_THREADS_X);
      dim3 grid(EM, (N + CURRENT_BLOCK_KN * NPT - 1) / (CURRENT_BLOCK_KN * NPT),
                (K + CURRENT_BLOCK_KN - 1) / CURRENT_BLOCK_KN);
      moe_gemm_q4_current<<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, M, N, K,
          GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
      CHECK(hipGetLastError());
    }
    hipEventRecord(e1, 0);
    CHECK(hipStreamSynchronize(0));
    hipEventElapsedTime(&ms, e0, e1);
    printf("current (no zero)  : %.2f us/launch\n", ms * 1000.f / niter);
    time_v2([&] {
      dim3 blk(32);
      dim3 grid(EM, N / 128);
      moe_gemm_q4_v1<32, 4, 2048><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_sorted, d_eids, d_npp, M, N, K,
          GROUPS, TOPK, wstride, sstride, zstride, 0);
      CHECK(hipGetLastError());
    }, "v1a 32t 4col fullK");
    time_v2([&] {
      dim3 blk(128);
      dim3 grid(EM, N / 128);
      moe_gemm_q4_v1<128, 1, 2048><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_sorted, d_eids, d_npp, M, N, K,
          GROUPS, TOPK, wstride, sstride, zstride, 0);
      CHECK(hipGetLastError());
    }, "v1b 128t 1col fullK");
    time_v2([&] {
      dim3 blk(32);
      dim3 grid(EM, N / 64);
      moe_gemm_q4_v1<32, 2, 2048><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_sorted, d_eids, d_npp, M, N, K,
          GROUPS, TOPK, wstride, sstride, zstride, 0);
      CHECK(hipGetLastError());
    }, "v1c 32t 2col fullK");
    time_v2([&] {
      dim3 blk(32);
      dim3 grid(EM, N / 32);
      moe_gemm_q4_v1<32, 1, 2048><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_sorted, d_eids, d_npp, M, N, K,
          GROUPS, TOPK, wstride, sstride, zstride, 0);
      CHECK(hipGetLastError());
    }, "v1d 32t 1col fullK");
    time_v2([&] {
      dim3 blk(16);
      dim3 grid(EM, N / 16);
      moe_gemm_q4_v1<16, 1, 2048><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_sorted, d_eids, d_npp, M, N, K,
          GROUPS, TOPK, wstride, sstride, zstride, 0);
      CHECK(hipGetLastError());
    }, "v1e 16t 1col fullK");
    auto time_sweep = [&](auto launch, const char* name) {
      for (int i = 0; i < 50; ++i) launch();
      CHECK(hipStreamSynchronize(0));
      hipEventRecord(e0, 0);
      for (int i = 0; i < niter; ++i) launch();
      hipEventRecord(e1, 0);
      CHECK(hipStreamSynchronize(0));
      float m;
      hipEventElapsedTime(&m, e0, e1);
      printf("sweep %-12s: %.2f us/launch\n", name, m * 1000.f / niter);
    };
    time_sweep([&] {
      dim3 blk(256);
      dim3 grid(EM, (N + 256 * 4 - 1) / (256 * 4), (K + 255) / 256);
      moe_gemm_q4_sweep<256, 4><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
      CHECK(hipGetLastError());
    }, "256t NPT=4");
    time_sweep([&] {
      dim3 blk(256);
      dim3 grid(EM, (N + 256 * 2 - 1) / (256 * 2), (K + 255) / 256);
      moe_gemm_q4_sweep<256, 2><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
      CHECK(hipGetLastError());
    }, "256t NPT=2");
    time_sweep([&] {
      dim3 blk(128);
      dim3 grid(EM, (N + 128 * 4 - 1) / (128 * 4), (K + 127) / 128);
      moe_gemm_q4_sweep<128, 4><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
      CHECK(hipGetLastError());
    }, "128t NPT=4");
    time_sweep([&] {
      dim3 blk(128);
      dim3 grid(EM, (N + 128 * 2 - 1) / (128 * 2), (K + 127) / 128);
      moe_gemm_q4_sweep<128, 2><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
      CHECK(hipGetLastError());
    }, "128t NPT=2");
    time_sweep([&] {
      dim3 blk(64);
      dim3 grid(EM, (N + 64 * 4 - 1) / (64 * 4), (K + 63) / 64);
      moe_gemm_q4_sweep<64, 4><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
      CHECK(hipGetLastError());
    }, "64t NPT=4");
    time_sweep([&] {
      dim3 blk(64);
      dim3 grid(EM, (N + 64 * 2 - 1) / (64 * 2), (K + 63) / 64);
      moe_gemm_q4_sweep<64, 2><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
      CHECK(hipGetLastError());
    }, "64t NPT=2");
    time_sweep([&] {
      dim3 blk(512);
      dim3 grid(EM, (N + 512 * 4 - 1) / (512 * 4), (K + 511) / 512);
      moe_gemm_q4_sweep<512, 4><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
      CHECK(hipGetLastError());
    }, "512t NPT=4");
    time_sweep([&] {
      dim3 blk(512);
      dim3 grid(EM, (N + 512 * 2 - 1) / (512 * 2), (K + 511) / 512);
      moe_gemm_q4_sweep<512, 2><<<grid, blk>>>(
          d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride, false, 0, 0);
      CHECK(hipGetLastError());
    }, "512t NPT=2");
  }

  if (mode3 == 11) {
    // production gemm2 convention: a = [EM, K] per-slot rows, top_k=1,
    // output_topk=TOPK, all 8 slots CAS into token row 0.
    std::vector<half> h_a8(EM * K);
    for (size_t i = 0; i < h_a8.size(); ++i)
      h_a8[i] = __float2half_rn((rand() % 4000 - 2000) / 2000.0f);
    half* d_a8;
    CHECK(hipMalloc(&d_a8, EM * K * sizeof(half)));
    CHECK(hipMemcpy(d_a8, h_a8.data(), EM * K * sizeof(half),
                    hipMemcpyHostToDevice));
    int32_t npp8 = EM;
    CHECK(hipMemcpy(d_npp, &npp8, sizeof(int32_t), hipMemcpyHostToDevice));
    // ref[nn] = sum_slot tw[slot] * sum_k a8[slot][k] * W[eids[slot]][k][nn]
    std::vector<float> ref0(N, 0.f);
    for (int slot = 0; slot < EM; ++slot) {
      int e = h_eids[slot];
      for (int nn = 0; nn < N; ++nn) {
        float acc = 0.f;
        for (int qk = 0; qk < K / 8; ++qk) {
          int g = (qk * 8) / (K / GROUPS);
          half scale = h_sc[(size_t)e * GROUPS * N + g * N + nn];
          uint32_t zd =
              h_z[(size_t)e * GROUPS * (N / 8) + g * (N / 8) + nn / 8];
          int zn = (zd >> ((nn & 7) * 4)) & 0xF;
          int q[8];
          unpack8(h_w[(size_t)e * (K / 8) * N + (size_t)qk * N + nn], q);
          for (int j = 0; j < 8; ++j)
            acc += __half2float(h_a8[(size_t)slot * K + qk * 8 + j]) *
                   (((float)q[j] - (float)zn) * __half2float(scale));
        }
        ref0[nn] += 1.0f * acc;  // tw = 1.0
      }
    }
    auto run_and_check = [&](const char* tag, auto launch) {
      CHECK(hipMemset(d_c2, 0, EM * N * sizeof(half)));
      launch();
      CHECK(hipDeviceSynchronize());
      std::vector<half> hc(N);
      CHECK(hipMemcpy(hc.data(), d_c2, M * N * sizeof(half),
                      hipMemcpyDeviceToHost));
      float me = 0;
      int nb = 0;
      for (int nn = 0; nn < N; ++nn) {
        float d = fabsf(__half2float(hc[nn]) - ref0[nn]);
        me = std::max(me, d);
        nb += (d > 0.75f);
      }
      printf("prod-gemm2 %s: max err %.4f (%d bad)\n", tag, me, nb);
    };
    run_and_check("current(8x z-CAS)", [&] {
      dim3 blk(CURRENT_THREADS_X);
      dim3 grid(EM, (N + CURRENT_BLOCK_KN * NPT - 1) / (CURRENT_BLOCK_KN * NPT),
                (K + CURRENT_BLOCK_KN - 1) / CURRENT_BLOCK_KN);
      moe_gemm_q4_current<<<grid, blk>>>(
          d_a8, d_c2, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          EM, N, K, GROUPS, 1, wstride, sstride, zstride, true, TOPK, 0);
      CHECK(hipGetLastError());
    });
    run_and_check("v2-512t4col(8x CAS)", [&] {
      dim3 blk(512);
      dim3 grid(EM, N / 256);
      moe_gemm_q4_v2<512, 4, 256><<<grid, blk>>>(
          d_a8, d_c2, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          EM, N, K, GROUPS, 1, wstride, sstride, zstride, true, TOPK, 0,
          nullptr);
      CHECK(hipGetLastError());
    });
    run_and_check("v2-512t2col(8x CAS)", [&] {
      dim3 blk(512);
      dim3 grid(EM, N / 128);
      moe_gemm_q4_v2<512, 2, 256><<<grid, blk>>>(
          d_a8, d_c2, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          EM, N, K, GROUPS, 1, wstride, sstride, zstride, true, TOPK, 0,
          nullptr);
      CHECK(hipGetLastError());
    });
    return 0;
  }
  if (mode3 == 10) {
    // timing sweep only (uses npp=EM)
    auto time_v2 = [&](auto kernel, const char* name) {
      for (int i = 0; i < 50; ++i) kernel();
      CHECK(hipStreamSynchronize(0));
      hipEvent_t e0, e1;
      hipEventCreate(&e0);
      hipEventCreate(&e1);
      hipEventRecord(e0, 0);
      for (int i = 0; i < 2000; ++i) kernel();
      hipEventRecord(e1, 0);
      CHECK(hipStreamSynchronize(0));
      float ms;
      hipEventElapsedTime(&ms, e0, e1);
      printf("v2 %-14s: %.2f us/launch\n", name, ms * 1000.f / 2000);
      hipEventDestroy(e0);
      hipEventDestroy(e1);
    };
    L2 lt{d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
    V2L v2t{d_a, d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
            M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
    time_v2([&] { v2t.template operator()<512, 2, 256>(); }, "512t 2col s256");
    time_v2([&] { v2t.template operator()<512, 4, 256>(); }, "512t 4col s256");
    time_v2([&] { v2t.template operator()<256, 4, 512>(); }, "256t 4col s512");
    time_v2([&] { v2t.template operator()<256, 2, 512>(); }, "256t 2col s512");
    time_v2([&] { v2t.template operator()<128, 1, 1024>(); }, "128t 1col s1024");
    return 0;
  }
  if (mode3 == 9) {
    auto dump_r = [&](float* d_dbg, const char* tag) {
      float hdg[256];
      CHECK(hipMemcpy(hdg, d_dbg, 256 * sizeof(float), hipMemcpyDeviceToHost));
      printf("%s y=0 tl=0 r[0..3] per wave (ref row0: %.4f %.4f %.4f %.4f):\n",
             tag, ref[0], ref[1], ref[2], ref[3]);
      for (int ww = 0; ww < 8; ww += 4)
        printf("    w%d: %.4f %.4f %.4f %.4f\n", ww, hdg[ww], hdg[64 + ww],
               hdg[128 + ww], hdg[192 + ww]);
    };
    float* d_dbg;
    CHECK(hipMalloc(&d_dbg, 256 * sizeof(float)));
    // gemm1 mode: npp=8, direct store, no topk weight
    {
      int32_t npp8 = EM;
      CHECK(hipMemcpy(d_npp, &npp8, sizeof(int32_t), hipMemcpyHostToDevice));
      L2 l{d_c1, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
           M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
      l.dbg = d_dbg;
      l.g1<512, 4, 256>(d_c1);
      CHECK(hipDeviceSynchronize());
      dump_r(d_dbg, "gemm1-mode (npp=8)   ");
    }
    // single-slot gemm2 mode: npp=1, CAS
    {
      int32_t one = 1;
      CHECK(hipMemcpy(d_npp, &one, sizeof(int32_t), hipMemcpyHostToDevice));
      CHECK(hipMemset(d_c3, 0, M * N * sizeof(half)));
      L2 l{d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
           M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
      l.dbg = d_dbg;
      l.v2<512, 4, 256>();
      CHECK(hipDeviceSynchronize());
      dump_r(d_dbg, "single-slot (npp=1)  ");
    }
    CHECK(hipFree(d_dbg));
    return 0;
  }
  if (mode3 == 8) {
    auto cur_ss = [&]() {
      int32_t one_npp = 1;
      CHECK(hipMemcpy(d_npp, &one_npp, sizeof(int32_t), hipMemcpyHostToDevice));
      CHECK(hipMemset(d_c2, 0, M * N * sizeof(half)));
      dim3 blk(CURRENT_THREADS_X);
      dim3 grid(EM, (N + CURRENT_BLOCK_KN * NPT - 1) / (CURRENT_BLOCK_KN * NPT),
                (K + CURRENT_BLOCK_KN - 1) / CURRENT_BLOCK_KN);
      moe_gemm_q4_current<<<grid, blk>>>(
          d_a, d_c2, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride, true, TOPK, 0);
      CHECK(hipGetLastError());
      CHECK(hipDeviceSynchronize());
      std::vector<half> hn(N);
      CHECK(hipMemcpy(hn.data(), d_c2, M * N * sizeof(half),
                      hipMemcpyDeviceToHost));
      float mev = 0;
      int nbv = 0;
      for (int nn = 0; nn < N; ++nn) {
        float d = fabsf(__half2float(hn[nn]) - ref[(size_t)0 * N + nn]);
        mev = std::max(mev, d);
        nbv += (d > 0.5f);
      }
      printf("    probe: current single-slot max err %.4f (%d bad)\n", mev,
             nbv);
    };
    // replicate mode2 it0 EXACTLY, probing after each step
    // NOTE: mode2 starts right after the MAIN FLOW ran; here we also run the
    // main-flow-equivalent state first: run the same npp=8 gemm1 into d_c1
    // like main flow did (approximation: run current gemm1)
    cur_ss();
    L2 l2{d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
    int32_t one_npp = 1;
    CHECK(hipMemcpy(d_npp, &one_npp, sizeof(int32_t), hipMemcpyHostToDevice));
    printf("  [1] npp=1 set\n");
    CHECK(hipMemset(d_c3, 0, M * N * sizeof(half)));
    printf("  [2] d_c3 zeroed\n");
    l2.v2<512, 2, 128>();
    CHECK(hipDeviceSynchronize());
    printf("  [3] v2<512,2,128> done\n");
    std::vector<half> h2(N);
    CHECK(hipMemcpy(h2.data(), d_c3, M * N * sizeof(half),
                    hipMemcpyDeviceToHost));
    printf("  [4] h2 read\n");
    CHECK(hipMemset(d_c3, 0, M * N * sizeof(half)));
    l2.v2<512, 4, 256>();
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(h2.data(), d_c3, M * N * sizeof(half),
                    hipMemcpyDeviceToHost));
    printf("  [5] v2<512,4,256> done + read\n");
    float* d_dbg3;
    CHECK(hipMalloc(&d_dbg3, 256 * sizeof(float)));
    L2 l3{d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
    l3.dbg = d_dbg3;
    CHECK(hipMemset(d_c3, 0, M * N * sizeof(half)));
    l3.g1d<512, 4, 256>(d_c3);
    CHECK(hipDeviceSynchronize());
    printf("  [6] g1d+dbg done\n");
    CHECK(hipFree(d_dbg3));
    CHECK(hipMemcpy(h2.data(), d_c3, M * N * sizeof(half),
                    hipMemcpyDeviceToHost));
    printf("  [7] final read; probes after each step:\n");
    cur_ss();
    return 0;
  }
  if (mode3 == 7) {
    auto cur_ss = [&]() {
      int32_t one_npp = 1;
      CHECK(hipMemcpy(d_npp, &one_npp, sizeof(int32_t), hipMemcpyHostToDevice));
      CHECK(hipMemset(d_c2, 0, M * N * sizeof(half)));
      dim3 blk(CURRENT_THREADS_X);
      dim3 grid(EM, (N + CURRENT_BLOCK_KN * NPT - 1) / (CURRENT_BLOCK_KN * NPT),
                (K + CURRENT_BLOCK_KN - 1) / CURRENT_BLOCK_KN);
      moe_gemm_q4_current<<<grid, blk>>>(
          d_a, d_c2, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride, true, TOPK, 0);
      CHECK(hipGetLastError());
      CHECK(hipDeviceSynchronize());
      std::vector<half> hn(N);
      CHECK(hipMemcpy(hn.data(), d_c2, M * N * sizeof(half),
                      hipMemcpyDeviceToHost));
      float mev = 0;
      int nbv = 0;
      for (int nn = 0; nn < N; ++nn) {
        float d = fabsf(__half2float(hn[nn]) - ref[(size_t)0 * N + nn]);
        mev = std::max(mev, d);
        nbv += (d > 0.5f);
      }
      printf("  current single-slot: max err %.4f (%d bad)\n", mev, nbv);
    };
    cur_ss();  // baseline
    CHECK(hipMemset(d_c3, 0, M * N * sizeof(half)));
    L2 lp{d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
    lp.v2<512, 2, 128>();
    CHECK(hipDeviceSynchronize());
    cur_ss();  // after s128 overflow
    float* d_dbg3;
    CHECK(hipMalloc(&d_dbg3, 256 * sizeof(float)));
    L2 l3{d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
    l3.dbg = d_dbg3;
    CHECK(hipMemset(d_c3, 0, M * N * sizeof(half)));
    l3.g1d<512, 4, 256>(d_c3);
    CHECK(hipDeviceSynchronize());
    cur_ss();  // after g1d+dbg
    CHECK(hipFree(d_dbg3));
    return 0;
  }
  if (mode3 == 5 || mode3 == 6) {
    auto sweep_and_cur = [&](bool with_v2) {
      for (int npp_v : {1, 2, 4, 8}) {
        CHECK(hipMemcpy(d_npp, &npp_v, sizeof(int32_t), hipMemcpyHostToDevice));
        CHECK(hipMemset(d_c2, 0, M * N * sizeof(half)));
        dim3 blk(CURRENT_THREADS_X);
        dim3 grid(EM, (N + CURRENT_BLOCK_KN * NPT - 1) / (CURRENT_BLOCK_KN * NPT),
                  (K + CURRENT_BLOCK_KN - 1) / CURRENT_BLOCK_KN);
        moe_gemm_q4_current<<<grid, blk>>>(
            d_a, d_c2, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
            M, N, K, GROUPS, TOPK, wstride, sstride, zstride, true, TOPK, 0);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
      }
      if (with_v2) {
        CHECK(hipMemset(d_c3, 0, M * N * sizeof(half)));
        L2 lp{d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
              M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
        lp.v2<512, 4, 256>();
        CHECK(hipDeviceSynchronize());
      }
      int32_t one_npp = 1;
      CHECK(hipMemcpy(d_npp, &one_npp, sizeof(int32_t), hipMemcpyHostToDevice));
      CHECK(hipMemset(d_c2, 0, M * N * sizeof(half)));
      dim3 blk(CURRENT_THREADS_X);
      dim3 grid(EM, (N + CURRENT_BLOCK_KN * NPT - 1) / (CURRENT_BLOCK_KN * NPT),
                (K + CURRENT_BLOCK_KN - 1) / CURRENT_BLOCK_KN);
      moe_gemm_q4_current<<<grid, blk>>>(
          d_a, d_c2, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
          M, N, K, GROUPS, TOPK, wstride, sstride, zstride, true, TOPK, 0);
      CHECK(hipGetLastError());
      CHECK(hipDeviceSynchronize());
      std::vector<half> hn(N);
      CHECK(hipMemcpy(hn.data(), d_c2, M * N * sizeof(half),
                      hipMemcpyDeviceToHost));
      float mev = 0;
      int nbv = 0;
      for (int nn = 0; nn < N; ++nn) {
        float d = fabsf(__half2float(hn[nn]) - ref[(size_t)0 * N + nn]);
        mev = std::max(mev, d);
        nbv += (d > 0.5f);
      }
      printf("  mode with_v2=%d: current single-slot max err %.4f (%d bad)\n",
             (int)with_v2, mev, nbv);
    };
    sweep_and_cur(mode3 == 6);
    return 0;
  }
  if (mode3 == 3 || mode3 == 4) {
    auto cur_singleslot = [&]() -> float {
      int32_t one_npp = 1;
      CHECK(hipMemcpy(d_npp, &one_npp, sizeof(int32_t), hipMemcpyHostToDevice));
      CHECK(hipMemset(d_c2, 0, M * N * sizeof(half)));
      {
        dim3 blk(CURRENT_THREADS_X);
        dim3 grid(EM, (N + CURRENT_BLOCK_KN * NPT - 1) / (CURRENT_BLOCK_KN * NPT),
                  (K + CURRENT_BLOCK_KN - 1) / CURRENT_BLOCK_KN);
        moe_gemm_q4_current<<<grid, blk>>>(
            d_a, d_c2, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
            M, N, K, GROUPS, TOPK, wstride, sstride, zstride, true, TOPK, 0);
        CHECK(hipGetLastError());
      }
      CHECK(hipDeviceSynchronize());
      std::vector<half> hn(N);
      CHECK(hipMemcpy(hn.data(), d_c2, M * N * sizeof(half),
                      hipMemcpyDeviceToHost));
      float mev = 0;
      int nbv = 0;
      for (int nn = 0; nn < N; ++nn) {
        float d = fabsf(__half2float(hn[nn]) - ref[(size_t)0 * N + nn]);
        mev = std::max(mev, d);
        nbv += (d > 0.5f);
      }
      printf("  current single-slot: max err %.4f (%d bad)\n", mev, nbv);
      return mev;
    };
    if (mode3 == 4) {
      // poison: run the LDS-overflowing config (NPT=2, SLICE=128, K=2048)
      int32_t one_npp = 1;
      CHECK(hipMemcpy(d_npp, &one_npp, sizeof(int32_t), hipMemcpyHostToDevice));
      CHECK(hipMemset(d_c3, 0, M * N * sizeof(half)));
      L2 lp{d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
            M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
      for (int i = 0; i < 3; ++i) lp.v2<512, 2, 128>();
      CHECK(hipDeviceSynchronize());
      printf("  (poison: 3x v2<512,2,128> LDS-overflow runs done)\n");
    }
    cur_singleslot();
    cur_singleslot();
    return 0;
  }
  if (isolated) {
    // npp sweep with current kernel: ref = sum of active slots
    for (int npp_v : {1, 2, 4, 8}) {
      CHECK(hipMemcpy(d_npp, &npp_v, sizeof(int32_t), hipMemcpyHostToDevice));
      CHECK(hipMemset(d_c2, 0, M * N * sizeof(half)));
      {
        dim3 blk(CURRENT_THREADS_X);
        dim3 grid(EM, (N + CURRENT_BLOCK_KN * NPT - 1) / (CURRENT_BLOCK_KN * NPT),
                  (K + CURRENT_BLOCK_KN - 1) / CURRENT_BLOCK_KN);
        moe_gemm_q4_current<<<grid, blk>>>(
            d_a, d_c2, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
            M, N, K, GROUPS, TOPK, wstride, sstride, zstride, true, TOPK, 0);
        CHECK(hipGetLastError());
      }
      CHECK(hipDeviceSynchronize());
      std::vector<half> hn(N);
      CHECK(hipMemcpy(hn.data(), d_c2, M * N * sizeof(half),
                      hipMemcpyDeviceToHost));
      float mev = 0;
      int nbv = 0;
      for (int nn = 0; nn < N; ++nn) {
        float rv = 0.f;
        for (int sl = 0; sl < npp_v; ++sl) rv += ref[(size_t)sl * N + nn];
        float d = fabsf(__half2float(hn[nn]) - rv);
        mev = std::max(mev, d);
        nbv += (d > 0.5f);
      }
      printf("npp-sweep current npp=%d: max err %.4f (%d bad)\n", npp_v, mev,
             nbv);
    }
    {
      int32_t one_npp = 1;
      CHECK(hipMemcpy(d_npp, &one_npp, sizeof(int32_t),
                      hipMemcpyHostToDevice));
    }
    for (int it = 0; it < 3; ++it) {
      {
        int32_t one_npp = 1;
        CHECK(hipMemcpy(d_npp, &one_npp, sizeof(int32_t),
                        hipMemcpyHostToDevice));
      }
      CHECK(hipMemset(d_c3, 0, M * N * sizeof(half)));
      L2 l2{d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
            M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
      l2.v2<512, 2, 128>();
      CHECK(hipDeviceSynchronize());
      std::vector<half> h2(N);
      CHECK(hipMemcpy(h2.data(), d_c3, M * N * sizeof(half),
                      hipMemcpyDeviceToHost));
      float me2 = 0;
      for (int nn = 0; nn < N; ++nn)
        me2 = std::max(me2,
                       fabsf(__half2float(h2[nn]) - ref[(size_t)0 * N + nn]));
      printf("isolated it%d NPT=2 s128 single-slot: max err %.4f\n", it, me2);
      CHECK(hipMemset(d_c3, 0, M * N * sizeof(half)));
      l2.v2<512, 4, 256>();
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(h2.data(), d_c3, M * N * sizeof(half),
                      hipMemcpyDeviceToHost));
      float me4 = 0;
      int nbad4 = 0;
      if (it == 0) {
        // A/B: same setup, forced direct store (NPT=4) into d_c3
        float* d_dbg3;
        CHECK(hipMalloc(&d_dbg3, 256 * sizeof(float)));
        L2 l3{d_c3, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp, d_a,
              M, N, K, GROUPS, TOPK, wstride, sstride, zstride};
        l3.dbg = d_dbg3;
        CHECK(hipMemset(d_c3, 0, M * N * sizeof(half)));
        l3.g1d<512, 4, 256>(d_c3);
        CHECK(hipDeviceSynchronize());
        float hdg[256];
        CHECK(hipMemcpy(hdg, d_dbg3, 256 * sizeof(float), hipMemcpyDeviceToHost));
        printf("  y=0 r[] per wave (CPU ref col0..3: %.4f %.4f %.4f %.4f):\n",
               ref[0], ref[1], ref[2], ref[3]);
        for (int ww = 0; ww < 8; ++ww)
          printf("    w%d: %.4f %.4f %.4f %.4f\n", ww, hdg[ww], hdg[64 + ww],
                 hdg[128 + ww], hdg[192 + ww]);
        CHECK(hipFree(d_dbg3));
        std::vector<half> hd(N);
        CHECK(hipMemcpy(hd.data(), d_c3, M * N * sizeof(half),
                        hipMemcpyDeviceToHost));
        float med = 0;
        int nbad = 0;
        for (int nn = 0; nn < N; ++nn) {
          float d = fabsf(__half2float(hd[nn]) - ref[(size_t)0 * N + nn]);
          med = std::max(med, d);
          nbad += (d > 0.35f);
        }
        printf("isolated NPT=4 s256 DIRECT-store single-slot: max err %.4f (%d bad)\n",
               med, nbad);
      }
      for (int nn = 0; nn < N; ++nn) {
        float d = fabsf(__half2float(h2[nn]) - ref[(size_t)0 * N + nn]);
        me4 = std::max(me4, d);
        nbad4 += (d > 0.35f);
      }
      printf("isolated it%d NPT=4 s256 single-slot: max err %.4f (%d bad)\n",
             it, me4, nbad4);
      // current-kernel single slot vs cpu-ref (A/B for ref validity)
      if (it == 0) {
        CHECK(hipMemset(d_c2, 0, M * N * sizeof(half)));
        {
          constexpr int BC = 1024;
          dim3 blk(256);
          dim3 grid(M * TOPK, N / BC);
          moe_gemm_q4_current<<<grid, blk>>>(
              d_a, d_c2, d_w, d_sc, d_z, d_topkw, d_sorted, d_eids, d_npp,
              M, N, K, GROUPS, TOPK, wstride, sstride, zstride, true, TOPK,
              0);
          CHECK(hipGetLastError());
        }
        CHECK(hipDeviceSynchronize());
        std::vector<half> hc(N);
        CHECK(hipMemcpy(hc.data(), d_c2, M * N * sizeof(half),
                        hipMemcpyDeviceToHost));
        float meC = 0;
        int nbC = 0;
        for (int nn = 0; nn < N; ++nn) {
          float d = fabsf(__half2float(hc[nn]) - ref[(size_t)0 * N + nn]);
          meC = std::max(meC, d);
          nbC += (d > 0.35f);
        }
        printf("isolated CURRENT-kernel single-slot: max err %.4f (%d bad) cols0-3: %.4f %.4f %.4f %.4f\n",
               meC, nbC, __half2float(hc[0]), __half2float(hc[1]),
               __half2float(hc[2]), __half2float(hc[3]));
      }
    }
  }
  printf("%s\n", bad ? "HARNESS FAIL" : "HARNESS PASS");
  return bad;
}
