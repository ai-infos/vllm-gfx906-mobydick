// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
//
// M=1 W16A16 dense GEMV for gfx906 (Vega 20, no MFMA). Phase 3 P3-2(b).
//
//   out[1, N] = x[1, K] @ W[N, K]^T      (all fp16, fp32 accumulation)
//
// Design (row-parallel, same structure as LLGemm1_kernel which saturates
// HBM on the big rows):
//   - Block covers RPT weight rows; threads split the K-chunk in 8-half
//     (16B) units: thread t handles k = k0 + 8*t .. k0 + 8*t + 7.
//   - Each thread loads one 16B slice of x (shared across its RPT rows) and
//     RPT 16B weight rows, dots via __ockl_fdot2 (v_dot2_f32_f16) — the
//     same 16B-load / fp32-dot pattern as moe_q_gemm_gfx906.cu.
//   - KCHUNK=512/1024/2048/4096 → 64/128/256/512 threads
//     (1/2/4/8 wavefronts; KCHUNK=4096 exceeds MI50's 256-thread
//     workgroup limit — bench-only path, never used by model dispatch).
//     KSPLIT = K / KCHUNK.
//   - KSPLIT==1: cross-warp reduce in LDS, single fp16 store (no atomics).
//   - KSPLIT>1:  grid.y spans the K-chunks; each block reduces its chunk in
//     fp32, converts to fp16, and atomic-adds (packed 32/64-bit CAS) into a
//     pre-zeroed output. Precision: one extra fp16 rounding per chunk
//     (reorder-class change, A/B-diffed at integration).
//
// Measured on gfx906 (MI50), DEVLOG "P3-2(b)": the winning configuration
// is single-pass KCHUNK=K with RPT=2 for K=2048 rows of N==256 or N>=2048
// (qkv -23%, router -17%, in_proj/LM head -6% vs LLMM1 rpb=4). The v1
// K-split hypothesis for the small rows is falsified: at M=1 the CAS +
// zero_ + tiny-block overhead makes splits 2.4-4.2x slower than LLMM1; the
// 3.6-14x-floor rows (GDN-small, shared down) are launch/latency-bound,
// not CU-occupancy-bound, and no GEMM kernel closes them. K-split is kept
// as a supported path (RPT>=2 only) but no model shape uses it.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace dense_gemv_gfx906 {

// Packed 2-half atomic add via one 32-bit CAS loop (RPT=2 epilogue).
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

// Packed 4-half atomic add via one 64-bit CAS loop (RPT=4 epilogue).
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

// 8-wide fp32 dot: 4 x v_dot2_f32_f16 over (w.h2[i], a.h2[i]).
__forceinline__ __device__ float dot8_f32(const half2 (&w)[4],
                                          const half2 (&a)[4]) {
  float r = 0.0f;
  #pragma unroll
  for (int i = 0; i < 4; i++)
    r = __ockl_fdot2(w[i], a[i], r, true);
  return r;
}

template <int RPT, int KCHUNK>
__global__ void __launch_bounds__(KCHUNK / 8)
    dense_gemv_kernel(const half* __restrict__ x,   // [K]
                      const half* __restrict__ w,   // [N, K] row-major
                      half* __restrict__ out,       // [N], pre-zeroed if KSPLIT>1
                      const int N, const int K,
                      const int ksplit) {
  static_assert(KCHUNK == 512 || KCHUNK == 1024 || KCHUNK == 2048 ||
                    KCHUNK == 4096,
                "KCHUNK must be 512, 1024, 2048 or 4096");
  static_assert(RPT == 1 || RPT == 2 || RPT == 4, "RPT must be 1, 2 or 4");
  constexpr int THREADS = KCHUNK / 8;
  constexpr int WARPS = THREADS / 64;
  const int t = threadIdx.x;
  const int row0 = blockIdx.x * RPT;
  const int k0 = blockIdx.y * KCHUNK;

  // 8-half (16B) slice of x, shared across this thread's RPT rows.
  union {
    uint4 u;
    half2 h2[4];
  } xa;
  xa.u = *(const uint4*)(x + k0 + t * 8);

  float acc[RPT];
  #pragma unroll
  for (int r = 0; r < RPT; ++r) {
    const int row = row0 + r;
    if (row >= N) {
      acc[r] = 0.0f;
      continue;
    }
    union {
      uint4 u;
      half2 h2[4];
    } wa;
    wa.u = *(const uint4*)(w + (int64_t)row * K + k0 + t * 8);
    acc[r] = dot8_f32(wa.h2, xa.h2);
  }

  // Reduce the K-split within this block (THREADS/64 wavefronts).
  #pragma unroll
  for (int mask = 32; mask >= 1; mask /= 2) {
    #pragma unroll
    for (int r = 0; r < RPT; ++r)
      acc[r] += __shfl_xor(acc[r], mask);
  }

  if constexpr (WARPS == 1) {
    // Single wavefront: lane r < RPT holds row r's full sum.
    if (t < RPT) {
      const int row = row0 + t;
      if (row >= N) return;
      if (ksplit == 1) {
        out[row] = __float2half_rn(acc[t]);
      } else if constexpr (RPT == 4) {
        // One 64-bit CAS per block covering rows row0..row0+3.
        if (t == 0) {
          half2 h01 = __halves2half2(__float2half_rn(acc[0]),
                                     __float2half_rn(acc[1]));
          half2 h23 = __halves2half2(__float2half_rn(acc[2]),
                                     __float2half_rn(acc[3]));
          atomic_add_pk4_f16(out + row0, h01, h23);
        }
      } else if constexpr (RPT == 2) {
        // One 32-bit CAS per block covering rows row0..row0+1.
        if (t == 0) {
          half2 h01 = __halves2half2(__float2half_rn(acc[0]),
                                     __float2half_rn(acc[1]));
          atomic_add_pk2_f16(out + row0, h01);
        }
      }
      // RPT==1 with ksplit>1 is rejected by the launcher.
    }
  } else {
    // Multiple wavefronts: exchange per-row partials through LDS.
    __shared__ float red_smem[RPT][8];  // WARPS <= 8 (KCHUNK <= 4096)
    const int warp = t / 64;
    const int lane = t % 64;
    if (lane < RPT) red_smem[lane][warp] = acc[lane];
    __syncthreads();
    if (warp == 0 && lane < RPT) {
      const int row = row0 + lane;
      if (row >= N) return;
      float s = 0.0f;
      #pragma unroll
      for (int wp = 0; wp < WARPS; ++wp) s += red_smem[lane][wp];
      if (ksplit == 1) {
        out[row] = __float2half_rn(s);
      } else if constexpr (RPT == 4) {
        // One 64-bit CAS per block covering rows row0..row0+3.
        if (lane == 0) {
          float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
          #pragma unroll
          for (int wp = 0; wp < WARPS; ++wp) {
            s1 += red_smem[1][wp];
            s2 += red_smem[2][wp];
            s3 += red_smem[3][wp];
          }
          half2 h01 = __halves2half2(__float2half_rn(s),
                                     __float2half_rn(s1));
          half2 h23 = __halves2half2(__float2half_rn(s2),
                                     __float2half_rn(s3));
          atomic_add_pk4_f16(out + row0, h01, h23);
        }
      } else if constexpr (RPT == 2) {
        // One 32-bit CAS per block covering rows row0..row0+1.
        if (lane == 0) {
          float s1 = 0.0f;
          #pragma unroll
          for (int wp = 0; wp < WARPS; ++wp) s1 += red_smem[1][wp];
          half2 h01 = __halves2half2(__float2half_rn(s),
                                     __float2half_rn(s1));
          atomic_add_pk2_f16(out + row0, h01);
        }
      }
      // RPT==1 with ksplit>1 is rejected by the launcher.
    }
  }
}

// ---------------------------------------------------------------------------
// M<=4 variant (spec-decode draft steps, Phase 1 L1').
//
//   out[M, N] = x[M, K] @ W[N, K]^T
//
// Same row-parallel structure as the M=1 kernel, but each thread also
// holds the 8-half x slice for every M row (registers; x re-reads hit
// L2) and accumulates RPT*M fp32 partials. Weight traffic is
// M-invariant — the whole point: at M=4 the triton_matmul fallback
// costs ~7x the HBM floor (174 us for a 31.5 MB weight), while this
// stays at the M=1 weight-read speed.
//
// K-split (kchunk < K) uses the same packed-fp16 CAS epilogue as the
// M=1 kernel: per M row, one 32-bit CAS (RPT=2) or 64-bit CAS (RPT=4)
// over the RPT adjacent output rows. RPT>=2 required (per-M-row output
// addresses are N apart, so the RPT=1 packed CAS is impossible).
// ---------------------------------------------------------------------------

template <int RPT, int KCHUNK, int M>
__global__ void __launch_bounds__(KCHUNK / 8)
    dense_gemv_m_kernel(const half* __restrict__ x,   // [M, K]
                       const half* __restrict__ w,   // [N, K]
                       half* __restrict__ out,       // [M, N], pre-zeroed
                       // if KSPLIT>1
                       const int N, const int K, const int ksplit) {
  static_assert(KCHUNK == 512 || KCHUNK == 1024 || KCHUNK == 2048 ||
                    KCHUNK == 4096,
                "KCHUNK must be 512, 1024, 2048 or 4096");
  static_assert(RPT == 2 || RPT == 4, "RPT must be 2 or 4");
  static_assert(M >= 1 && M <= 4, "M must be 1..4");
  constexpr int THREADS = KCHUNK / 8;
  constexpr int WARPS = THREADS / 64;
  constexpr int NV = RPT * M;  // values per thread (flattened r*M+m)
  const int t = threadIdx.x;
  const int row0 = blockIdx.x * RPT;
  const int k0 = blockIdx.y * KCHUNK;

  // x slices for all M rows (x is small: M*K*2B <= 40 KB, L2-resident;
  // every block re-reads its k-chunk of it). M is a template parameter so
  // the register arrays below size to the actual M (a runtime-M version
  // allocated 4x x-slices + RPT*4 accumulators unconditionally, which cut
  // occupancy ~35% on MI50 at KCHUNK=1024).
  union {
    uint4 u;
    half2 h2[4];
  } xa[M];
  #pragma unroll
  for (int m = 0; m < M; ++m)
    xa[m].u = *(const uint4*)(x + (int64_t)m * K + k0 + t * 8);

  float acc[RPT][M];
  #pragma unroll
  for (int r = 0; r < RPT; ++r) {
    const int row = row0 + r;
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
    if (row >= N) continue;
    union {
      uint4 u;
      half2 h2[4];
    } wa;
    wa.u = *(const uint4*)(w + (int64_t)row * K + k0 + t * 8);
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = dot8_f32(wa.h2, xa[m].h2);
  }

  // In-place shfl reduction of the RPT*M values (lanes < NV).
  #pragma unroll
  for (int mask = 32; mask >= 1; mask /= 2)
    #pragma unroll
    for (int r = 0; r < RPT; ++r)
      #pragma unroll
      for (int m = 0; m < M; ++m)
        acc[r][m] += __shfl_xor(acc[r][m], mask);

  if constexpr (WARPS == 1) {
    // Lane i < RPT*M holds (r=i/M, m=i%M)'s full sum.
    if (ksplit == 1) {
      if (t < NV) {
        const int r = t / M, m = t % M, row = row0 + r;
        if (row < N)
          out[(int64_t)m * N + row] = __float2half_rn(acc[r][m]);
      }
    } else {
      // CAS packs the RPT adjacent rows of each out[m]; lane 0 issues the
      // M CAS ops. The butterfly above is a full-wavefront broadcast, so
      // lane 0 already holds every (r, m) sum in its own registers. Do NOT
      // "fix" this back to __shfl(acc[...], i): on ROCm 7.14 clang the
      // divergent cross-lane __shfl lowers to ds_bpermute with an
      // offset-encoded 16-byte window read that lands on the wrong VGPRs
      // when the data registers are not 16B-aligned (silent wrong results
      // for M=2/3; see DEVLOG-spec-decode).
      if (t == 0) {
        float s[NV];
        #pragma unroll
        for (int i = 0; i < NV; ++i) s[i] = acc[i / M][i % M];
        #pragma unroll
        for (int m = 0; m < M; ++m) {
          if (row0 + RPT - 1 >= N) continue;  // ragged tail: RPT rows valid
          if constexpr (RPT == 4)
            atomic_add_pk4_f16(
                out + (int64_t)m * N + row0,
                __halves2half2(__float2half_rn(s[0 * M + m]),
                               __float2half_rn(s[1 * M + m])),
                __halves2half2(__float2half_rn(s[2 * M + m]),
                               __float2half_rn(s[3 * M + m])));
          else
            atomic_add_pk2_f16(
                out + (int64_t)m * N + row0,
                __halves2half2(__float2half_rn(s[0 * M + m]),
                               __float2half_rn(s[1 * M + m])));
        }
      }
    }
  } else {
    __shared__ float red_smem[NV][8];  // WARPS <= 8 (KCHUNK <= 4096)
    const int warp = t / 64;
    const int lane = t % 64;
    if (lane < NV) red_smem[lane][warp] = acc[lane / M][lane % M];
    __syncthreads();
    if (warp == 0) {
      if (ksplit == 1) {
        if (lane < NV) {
          const int r = lane / M, m = lane % M, row = row0 + r;
          float s = 0.0f;
          #pragma unroll
          for (int wp = 0; wp < WARPS; ++wp) s += red_smem[lane][wp];
          if (row < N) out[(int64_t)m * N + row] = __float2half_rn(s);
        }
      } else {
        if (lane == 0) {
          float s[NV];
          #pragma unroll
          for (int i = 0; i < NV; ++i) {
            s[i] = 0.0f;
            #pragma unroll
            for (int wp = 0; wp < WARPS; ++wp) s[i] += red_smem[i][wp];
          }
          #pragma unroll
          for (int m = 0; m < M; ++m) {
            if (row0 + RPT - 1 >= N) continue;  // ragged tail
            if constexpr (RPT == 4)
              atomic_add_pk4_f16(
                  out + (int64_t)m * N + row0,
                  __halves2half2(__float2half_rn(s[0 * M + m]),
                                 __float2half_rn(s[1 * M + m])),
                  __halves2half2(__float2half_rn(s[2 * M + m]),
                                 __float2half_rn(s[3 * M + m])));
            else
              atomic_add_pk2_f16(
                  out + (int64_t)m * N + row0,
                  __halves2half2(__float2half_rn(s[0 * M + m]),
                                 __float2half_rn(s[1 * M + m])));
          }
        }
      }
    }
  }
}

// Runtime-M variant (kept for M=4): the M-templated kernel above is faster
// for M=1..3 (static register arrays), but measured ~1.6x slower at M=4 on
// MI50 (507 -> 311 GB/s on the 248320x5120 LM head; cause unattributed -
// see docs/gfx906/DEVLOG-spec-decode.md). The launcher dispatches M=4 here.

template <int RPT, int KCHUNK>
__global__ void __launch_bounds__(KCHUNK / 8)
    dense_gemv_m_kernel_rt(const half* __restrict__ x,   // [M, K]
                       const half* __restrict__ w,   // [N, K]
                       half* __restrict__ out,       // [M, N], pre-zeroed
                       // if KSPLIT>1
                       const int M, const int N, const int K,
                       const int ksplit) {
  static_assert(KCHUNK == 512 || KCHUNK == 1024 || KCHUNK == 2048 ||
                    KCHUNK == 4096,
                "KCHUNK must be 512, 1024, 2048 or 4096");
  static_assert(RPT == 2 || RPT == 4, "RPT must be 2 or 4");
  constexpr int THREADS = KCHUNK / 8;
  constexpr int WARPS = THREADS / 64;
  const int t = threadIdx.x;
  const int row0 = blockIdx.x * RPT;
  const int k0 = blockIdx.y * KCHUNK;

  // x slices for all M rows (x is small: M*K*2B <= 40 KB, L2-resident;
  // every block re-reads its k-chunk of it).
  union {
    uint4 u;
    half2 h2[4];
  } xa[4];
  #pragma unroll
  for (int m = 0; m < 4; ++m)
    if (m < M) xa[m].u = *(const uint4*)(x + (int64_t)m * K + k0 + t * 8);

  float acc[RPT][4];
  #pragma unroll
  for (int r = 0; r < RPT; ++r) {
    const int row = row0 + r;
    #pragma unroll
    for (int m = 0; m < 4; ++m) acc[r][m] = 0.0f;
    if (row >= N) continue;
    union {
      uint4 u;
      half2 h2[4];
    } wa;
    wa.u = *(const uint4*)(w + (int64_t)row * K + k0 + t * 8);
    #pragma unroll
    for (int m = 0; m < 4; ++m)
      if (m < M) acc[r][m] = dot8_f32(wa.h2, xa[m].h2);
  }

  // Flatten to acc_flat[r*4+m] for the reduction (lanes < RPT*4).
  float acc_flat[RPT * 4];
  #pragma unroll
  for (int r = 0; r < RPT; ++r)
    #pragma unroll
    for (int m = 0; m < 4; ++m) acc_flat[r * 4 + m] = acc[r][m];

  #pragma unroll
  for (int mask = 32; mask >= 1; mask /= 2)
    #pragma unroll
    for (int i = 0; i < RPT * 4; ++i) acc_flat[i] += __shfl_xor(acc_flat[i], mask);

  if constexpr (WARPS == 1) {
    // Lane i < RPT*4 holds (r=i/4, m=i%4)'s full sum.
    if (ksplit == 1) {
      if (t < RPT * 4) {
        const int r = t / 4, m = t % 4, row = row0 + r;
        if (m < M && row < N)
          out[(int64_t)m * N + row] = __float2half_rn(acc_flat[t]);
      }
    } else {
      // CAS packs the RPT adjacent rows of each out[m]. After the
      // butterfly every lane holds every (r, m) sum, so lane 0 reads its
      // own registers (shfl(acc_flat[0], i) -- the earlier form -- would
      // have returned index 0's sum for all i).
      if (t == 0) {
        float s[RPT * 4];
        #pragma unroll
        for (int i = 0; i < RPT * 4; ++i) s[i] = acc_flat[i];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
          if (m >= M) continue;
          if (row0 + RPT - 1 >= N) continue;  // ragged tail: RPT rows valid
          if constexpr (RPT == 4)
            atomic_add_pk4_f16(
                out + (int64_t)m * N + row0,
                __halves2half2(__float2half_rn(s[0 * 4 + m]),
                               __float2half_rn(s[1 * 4 + m])),
                __halves2half2(__float2half_rn(s[2 * 4 + m]),
                               __float2half_rn(s[3 * 4 + m])));
          else
            atomic_add_pk2_f16(
                out + (int64_t)m * N + row0,
                __halves2half2(__float2half_rn(s[0 * 4 + m]),
                               __float2half_rn(s[1 * 4 + m])));
        }
      }
    }
  } else {
    __shared__ float red_smem[RPT * 4][8];  // WARPS <= 8 (KCHUNK <= 4096)
    const int warp = t / 64;
    const int lane = t % 64;
    if (lane < RPT * 4) red_smem[lane][warp] = acc_flat[lane];
    __syncthreads();
    if (warp == 0) {
      if (ksplit == 1) {
        if (lane < RPT * 4) {
          const int r = lane / 4, m = lane % 4, row = row0 + r;
          float s = 0.0f;
          #pragma unroll
          for (int wp = 0; wp < WARPS; ++wp) s += red_smem[lane][wp];
          if (m < M && row < N) out[(int64_t)m * N + row] = __float2half_rn(s);
        }
      } else {
        if (lane == 0) {
          float s[RPT * 4];
          #pragma unroll
          for (int i = 0; i < RPT * 4; ++i) {
            s[i] = 0.0f;
            #pragma unroll
            for (int wp = 0; wp < WARPS; ++wp) s[i] += red_smem[i][wp];
          }
          #pragma unroll
          for (int m = 0; m < 4; ++m) {
            if (m >= M) continue;
            if (row0 + RPT - 1 >= N) continue;  // ragged tail
            if constexpr (RPT == 4)
              atomic_add_pk4_f16(
                  out + (int64_t)m * N + row0,
                  __halves2half2(__float2half_rn(s[0 * 4 + m]),
                                 __float2half_rn(s[1 * 4 + m])),
                  __halves2half2(__float2half_rn(s[2 * 4 + m]),
                                 __float2half_rn(s[3 * 4 + m])));
            else
              atomic_add_pk2_f16(
                  out + (int64_t)m * N + row0,
                  __halves2half2(__float2half_rn(s[0 * 4 + m]),
                                 __float2half_rn(s[1 * 4 + m])));
          }
        }
      }
    }
  }
}

// W4 skinny M=5..16: RPT=1, exact-M-templated (the templated M=1..4 rail
// structure with one weight row per block and M compile-time dots).
// Measured (bench_fp16_skinny_m.py / /tmp/moespec/real27b): the kernel
// is x-L2-re-read bound at ~M*B/1.6 TB/s (B = N*K*2 weight bytes -
// every block re-reads all of x from L2), matching the M=4 rail at
// M=4 (61 us on 5120x2048) and scaling linearly in M; triton is
// ~B/0.2 TB/s plus a ~100 us floor on small shapes, so this wins at
// M<=7 on all shapes and above only where triton hits its small-shape
// floor (a_b 1.3 MB: 7.5x at M=8, 4.6x at M=16; fa_kv 5.2 MB: 3.7x/
// 1.8x) - the python-side gate (VLLM_GFX906_SKINNY_M16, M-dependent
// size thresholds) routes the rest to triton. ksplit>1 uses the
// compiler-lowered fp16 atomicAdd (the pk2 CAS would be misaligned
// for odd rows - RPT=1 rows are not pair-aligned; the HSA aperture
// violation on an odd 32-bit CAS was observed, 2026-08-23).
template <int KCHUNK, int M>
__global__ void __launch_bounds__(KCHUNK / 8)
    dense_gemv_m_kernel_m16(const half* __restrict__ x,   // [M, K]
                            const half* __restrict__ w,   // [N, K]
                            half* __restrict__ out,       // [M, N], pre-zeroed
                                                            // if KSPLIT>1
                            const int N, const int K,
                            const int ksplit) {
  static_assert(KCHUNK == 512 || KCHUNK == 1024 || KCHUNK == 2048 ||
                    KCHUNK == 4096,
                "KCHUNK must be 512, 1024, 2048 or 4096");
  static_assert(M >= 5 && M <= 16, "M must be 5..16");
  constexpr int THREADS = KCHUNK / 8;
  constexpr int WARPS = THREADS / 64;
  const int t = threadIdx.x;
  const int row = blockIdx.x;
  const int k0 = blockIdx.y * KCHUNK;

  union {
    uint4 u;
    half2 h2[4];
  } xa[M];
  #pragma unroll
  for (int m = 0; m < M; ++m)
    xa[m].u = *(const uint4*)(x + (int64_t)m * K + k0 + t * 8);

  float acc[M];
  union {
    uint4 u;
    half2 h2[4];
  } wa;
  wa.u = *(const uint4*)(w + (int64_t)row * K + k0 + t * 8);
  #pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = dot8_f32(wa.h2, xa[m].h2);

  // Full 64-lane wavefront butterfly (masks 32..1 - mask 32 is the
  // cross-32-lane-half exchange on gfx906; house pattern, see
  // dense_gemv_m_kernel), then cross-warp in shared memory.
  #pragma unroll
  for (int mask = 32; mask >= 1; mask /= 2)
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] += __shfl_xor(acc[m], mask);

  __shared__ float red_smem[M][WARPS];  // WARPS <= 8 (KCHUNK <= 4096)
  const int warp = t / 64;
  const int lane = t % 64;
  if (lane < M) red_smem[lane][warp] = acc[lane];
  __syncthreads();
  if (warp == 0 && lane < M) {
    float s = 0.0f;
    #pragma unroll
    for (int wp = 0; wp < WARPS; ++wp) s += red_smem[lane][wp];
    if (ksplit == 1) {
      out[(int64_t)lane * N + row] = __float2half_rn(s);
    } else {
      atomicAdd(&out[(int64_t)lane * N + row], __float2half(s));
    }
  }
}

// ---------------------------------------------------------------------------
// W8A16 int8-weight GEMV family (NH-2', 2026-08-30).
//
//   out[1, N] = s[n] * sum_k w_i8[n, k] * x[k]       M=1
//   out[M, N] = s[n] * sum_k w_i8[n, k] * x[m, k]    1 <= M <= 4
//
// Row-parallel, same structure as the fp16 kernels above, but each thread
// owns a 16-byte (16 int8) weight slice and dequantizes it in-register to
// half2 pairs (__int2half_rn; gfx906 has no integer dot product, so the MACs
// run through the existing __ockl_fdot2 chain). Weight bytes per step are
// halved vs the fp16 family: at Nemotron's mid-N dense shapes the fp16
// kernels sit at 400-790 GB/s while Triton int8 reaches only 170-600 (NH-2
// NO-GO, DEVLOG-nemotron-h.md); P2 measured i8-row at 741 GB/s on the big
// rows (int8_gemv_probe.py). The serving mode (ngram spec n=5 -> M=6/step)
// needs the M<=4 CUDA path to win at mid-N — that is the NH-2' gate. These
// are exactly the K=2688 shapes (in_proj/out_proj/qkv/lm_head) and o_proj
// (K=2048) that the fp16 GEMV dispatch does NOT reach (_llmm1_tiny_m only
// routes K in {2048,512} at M=1; _gfx906_spec_gemv_m4 requires k%512==0 and
// 2688/512 is not integral).
//
// w_i8 is the pre-shifted signed-int8 view served by
// CompressedTensorsW8A16ChannelDequant (byte ^ 0x80 at load, plain two's
// complement; w = w_i8 * s per P2's 3-op convention). s is fp16 [N], one
// scale per output row (per-channel — P2 measured per-128-group scales cost
// +28 %). Scale application:
//   M=1, ksplit==1 : in the epilogue, after the K reduction (one mul/row)
//   M=1, ksplit>1  : partials accumulate UNSCALED through the packed CAS; a
//                    dense_gemv_i8_scale kernel scales them on the same
//                    stream afterwards (no per-chunk rounding at all)
//   M<=4          : the epilogue scales its fp32 partials before the packed
//                    CAS (one extra fp16 rounding per chunk — the fp16
//                    dense_gemv_m_kernel makes the identical trade; the
//                    ksplit>1 M<=4 shapes are small)
//
// KC is BYTES of weight per thread-slice = 16 int8. THREADS = KC/16 must be
// a whole number of 64-lane wavefronts (the shfl+LDS reduction assumes it),
// so KC in {1024, 2048, 4096} -> 64 | 128 | 256 threads. K % 16 == 0 is
// required for the aligned uint4 loads. KC need NOT divide K: ksplit =
// ceil(K / KC); a thread whose slice [k0 + t*16, k0 + t*16 + 16) starts at or
// past K contributes zero via the inb mask (the slice is either fully in-
// bounds or fully out — both K and the slice edges are multiples of 16).
// Real Nemotron int8 dense Ks: 2688 (KC=2048 -> ksplit=2 with a 640-element
// tail block; KC=4096 -> single pass) and 2048 (KC=2048 single pass).

// Signed two's-complement byte -> int (bytes >= 0x80 read as -128..-1; the
// scheme stores pre-shifted signed int8, so a plain (short)(b) cast would
// sign-extend from the wrong width).
__forceinline__ __device__ int i8_byte(int b) { return b >= 0x80 ? b - 256 : b; }

__forceinline__ __device__ void dequant_i8_to_h2(const uint4 u, half2 (&h)[8]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned word = ((const unsigned*)&u)[i];
    h[2 * i] = __halves2half2(
        __int2half_rn(i8_byte((word >> 0) & 0xff)),
        __int2half_rn(i8_byte((word >> 8) & 0xff)));
    h[2 * i + 1] = __halves2half2(
        __int2half_rn(i8_byte((word >> 16) & 0xff)),
        __int2half_rn(i8_byte((word >> 24) & 0xff)));
  }
}

template <int RPT, int KC>
__global__ void __launch_bounds__(KC / 16)
    dense_gemv_i8_kernel(const half* __restrict__ x,   // [K]
                         const signed char* __restrict__ w,  // [N, K]
                         const half* __restrict__ s,   // [N]
                         half* __restrict__ out,       // [N], pre-zeroed if KSPLIT>1
                         const int N, const int K,
                         const int ksplit) {
  static_assert(KC == 1024 || KC == 2048 || KC == 4096,
                "KC must be 1024, 2048 or 4096 (whole wavefronts)");
  static_assert(RPT == 2 || RPT == 4, "RPT must be 2 or 4");
  constexpr int THREADS = KC / 16;
  static_assert(THREADS % 64 == 0, "KC/16 must be a whole number of wavefronts");
  constexpr int WARPS = THREADS / 64;
  const int t = threadIdx.x;
  const int row0 = blockIdx.x * RPT;
  const int k0 = blockIdx.y * KC;

  // This thread covers 16 int8 weights at element indices [k0 + t*16,
  // k0 + t*16 + 16), so it needs 16 halves of x = two uint4s. (An fp16
  // thread covers only 8 elements / one uint4; the int8 slice is 2x wider in
  // element count because a byte holds one weight, not half.)
  const bool inb = (k0 + t * 16) < K;  // slice fully in-bounds or fully out
  union {
    uint4 u;
    half2 h2[4];
  } xa0, xa1;
  if (inb) {
    xa0.u = *(const uint4*)(x + k0 + t * 16);      // x[k0+t*16 .. +7]
    xa1.u = *(const uint4*)(x + k0 + t * 16 + 8);  // x[k0+t*16+8 .. +15]
  }

  float acc[RPT];
#pragma unroll
  for (int r = 0; r < RPT; ++r) {
    const int row = row0 + r;
    if (row >= N || !inb) {
      acc[r] = 0.0f;
      continue;
    }
    half2 wh[8];
    dequant_i8_to_h2(*(const uint4*)(w + (int64_t)row * K + k0 + t * 16), wh);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      // wh[i] holds weights at element offsets 2i,2i+1; pair with the same
      // x offsets: h2[0..3] from xa0 (offsets 0..7), h2[4..7] from xa1 (8..15).
      const half2& xx = (i < 4) ? xa0.h2[i] : xa1.h2[i - 4];
      acc[r] = __ockl_fdot2(wh[i], xx, acc[r], true);
    }
  }

  // Reduce the K-split within this block (THREADS/64 wavefronts) — mirrors
  // the fp16 dense_gemv_kernel exactly.
#pragma unroll
  for (int mask = 32; mask >= 1; mask /= 2) {
#pragma unroll
    for (int r = 0; r < RPT; ++r)
      acc[r] += __shfl_xor(acc[r], mask);
  }

  if constexpr (WARPS == 1) {
    // Single wavefront: lane r < RPT holds row r's full sum.
    if (t < RPT) {
      const int row = row0 + t;
      if (row >= N) return;
      if (ksplit == 1) {
        // Single pass: scale in the epilogue (P2: after the reduction).
        out[row] = __float2half_rn(acc[t] * __half2float(s[row]));
      } else if constexpr (RPT == 4) {
        if (t == 0)
          atomic_add_pk4_f16(
              out + row0,
              __halves2half2(__float2half_rn(acc[0]), __float2half_rn(acc[1])),
              __halves2half2(__float2half_rn(acc[2]), __float2half_rn(acc[3])));
      } else {
        if (t == 0)
          atomic_add_pk2_f16(
              out + row0,
              __halves2half2(__float2half_rn(acc[0]), __float2half_rn(acc[1])));
      }
    }
  } else {
    __shared__ float red_smem[RPT][8];  // WARPS <= 4 (KC <= 4096)
    const int warp = t / 64;
    const int lane = t % 64;
    if (lane < RPT) red_smem[lane][warp] = acc[lane];
    __syncthreads();
    if (warp == 0 && lane < RPT) {
      const int row = row0 + lane;
      if (row >= N) return;
      float sum = 0.0f;
#pragma unroll
      for (int wp = 0; wp < WARPS; ++wp) sum += red_smem[lane][wp];
      if (ksplit == 1) {
        out[row] = __float2half_rn(sum * __half2float(s[row]));
      } else if constexpr (RPT == 4) {
        if (lane == 0) {
          float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma unroll
          for (int wp = 0; wp < WARPS; ++wp) {
            s1 += red_smem[1][wp];
            s2 += red_smem[2][wp];
            s3 += red_smem[3][wp];
          }
          atomic_add_pk4_f16(
              out + row0,
              __halves2half2(__float2half_rn(sum), __float2half_rn(s1)),
              __halves2half2(__float2half_rn(s2), __float2half_rn(s3)));
        }
      } else {
        if (lane == 0) {
          float s1 = 0.0f;
#pragma unroll
          for (int wp = 0; wp < WARPS; ++wp) s1 += red_smem[1][wp];
          atomic_add_pk2_f16(
              out + row0,
              __halves2half2(__float2half_rn(sum), __float2half_rn(s1)));
        }
      }
    }
  }
}

template <int RPT, int KC, int M>
__global__ void __launch_bounds__(KC / 16)
    dense_gemv_i8_m_kernel(const half* __restrict__ x,   // [M, K]
                           const signed char* __restrict__ w,  // [N, K]
                           const half* __restrict__ s,   // [N]
                           half* __restrict__ out,       // [M, N], pre-zeroed if KSPLIT>1
                           const int N, const int K,
                           const int ksplit) {
  static_assert(KC == 1024 || KC == 2048 || KC == 4096,
                "KC must be 1024, 2048 or 4096 (whole wavefronts)");
  static_assert(RPT == 2 || RPT == 4, "RPT must be 2 or 4");
  // M <= 8 (T-1.5, 2026-09-05): the original bound was M<=4; extending to 8
  // covers full-acceptance k=4 verify (M=k+1=5) and batched small-M decode so
  // they run int8 in-kernel instead of the cached-dequant + unquantized-GEMM
  // fallback. Register cost at M=8/RPT=2: 16 fp32 accs + 16 x-slices (x is
  // L2-resident, re-read per block) — occupancy drops vs M=4 but weight
  // traffic is M-invariant and HALVED vs the fp16 path, so it still wins.
  static_assert(M >= 1 && M <= 8, "M must be 1..8");
  constexpr int THREADS = KC / 16;
  static_assert(THREADS % 64 == 0, "KC/16 must be a whole number of wavefronts");
  constexpr int WARPS = THREADS / 64;
  constexpr int NV = RPT * M;  // values per thread (flattened r*M+m)
  const int t = threadIdx.x;
  const int row0 = blockIdx.x * RPT;
  const int k0 = blockIdx.y * KC;

  // x slices for all M rows, each 16 halfs (two uint4s). x is small
  // (M*K*2B <= 64 KB, L2-resident); every block re-reads its k-chunk.
  const bool inb = (k0 + t * 16) < K;
  union {
    uint4 u;
    half2 h2[4];
  } xa0[M], xa1[M];
  if (inb) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      xa0[m].u = *(const uint4*)(x + (int64_t)m * K + k0 + t * 16);
      xa1[m].u = *(const uint4*)(x + (int64_t)m * K + k0 + t * 16 + 8);
    }
  }

  float acc[RPT][M];
#pragma unroll
  for (int r = 0; r < RPT; ++r) {
    const int row = row0 + r;
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
    if (row >= N || !inb) continue;
    half2 wh[8];
    dequant_i8_to_h2(*(const uint4*)(w + (int64_t)row * K + k0 + t * 16), wh);
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const half2& xx = (i < 4) ? xa0[m].h2[i] : xa1[m].h2[i - 4];
        acc[r][m] = __ockl_fdot2(wh[i], xx, acc[r][m], true);
      }
  }

#pragma unroll
  for (int mask = 32; mask >= 1; mask /= 2)
    #pragma unroll
    for (int r = 0; r < RPT; ++r)
      #pragma unroll
      for (int m = 0; m < M; ++m)
        acc[r][m] += __shfl_xor(acc[r][m], mask);

  if constexpr (WARPS == 1) {
    // Lane i < RPT*M holds (r=i/M, m=i%M)'s full sum. After the butterfly
    // every lane holds every value in its own registers — read them from
    // `acc` directly, never via __shfl(acc[...], i) (ROCm 7.14 lowers a
    // divergent cross-lane shfl to a misaligned ds_bpermute window read;
    // see dense_gemv_m_kernel).
    if (ksplit == 1) {
      if (t < NV) {
        const int r = t / M, m = t % M, row = row0 + r;
        if (row < N)
          out[(int64_t)m * N + row] =
              __float2half_rn(acc[r][m] * __half2float(s[row]));
      }
    } else {
      // CAS packs the RPT adjacent rows of each out[m]; lane 0 issues the M
      // CAS ops. Scales applied to the fp32 partials before the packed-fp16
      // conversion (see the family header for the rounding trade-off).
      if (t == 0) {
        float sv[NV];
#pragma unroll
        for (int i = 0; i < NV; ++i) {
          const int r = i / M, m = i % M;
          sv[i] = acc[r][m] * __half2float(s[row0 + r]);
        }
        if (row0 + RPT - 1 < N) {  // ragged tail (launcher enforces N%RPT==0)
#pragma unroll
          for (int m = 0; m < M; ++m) {
            if constexpr (RPT == 4)
              atomic_add_pk4_f16(
                  out + (int64_t)m * N + row0,
                  __halves2half2(__float2half_rn(sv[0 * M + m]),
                                 __float2half_rn(sv[1 * M + m])),
                  __halves2half2(__float2half_rn(sv[2 * M + m]),
                                 __float2half_rn(sv[3 * M + m])));
            else
              atomic_add_pk2_f16(
                  out + (int64_t)m * N + row0,
                  __halves2half2(__float2half_rn(sv[0 * M + m]),
                                 __float2half_rn(sv[1 * M + m])));
          }
        }
      }
    }
  } else {
    __shared__ float red_smem[NV][8];  // WARPS <= 4 (KC <= 4096)
    const int warp = t / 64;
    const int lane = t % 64;
    if (lane < NV) red_smem[lane][warp] = acc[lane / M][lane % M];
    __syncthreads();
    if (warp == 0) {
      if (ksplit == 1) {
        if (lane < NV) {
          const int r = lane / M, m = lane % M, row = row0 + r;
          float sum = 0.0f;
#pragma unroll
          for (int wp = 0; wp < WARPS; ++wp) sum += red_smem[lane][wp];
          if (row < N)
            out[(int64_t)m * N + row] =
                __float2half_rn(sum * __half2float(s[row]));
        }
      } else {
        if (lane == 0) {
          float sv[NV];
#pragma unroll
          for (int i = 0; i < NV; ++i) {
            const int r = i / M;  // m = i % M recovered per-column below
            float sum = 0.0f;
#pragma unroll
            for (int wp = 0; wp < WARPS; ++wp) sum += red_smem[i][wp];
            sv[i] = sum * __half2float(s[row0 + r]);
          }
          if (row0 + RPT - 1 < N) {  // ragged tail
#pragma unroll
            for (int m = 0; m < M; ++m) {
              if constexpr (RPT == 4)
                atomic_add_pk4_f16(
                    out + (int64_t)m * N + row0,
                    __halves2half2(__float2half_rn(sv[0 * M + m]),
                                   __float2half_rn(sv[1 * M + m])),
                    __halves2half2(__float2half_rn(sv[2 * M + m]),
                                   __float2half_rn(sv[3 * M + m])));
              else
                atomic_add_pk2_f16(
                    out + (int64_t)m * N + row0,
                    __halves2half2(__float2half_rn(sv[0 * M + m]),
                                   __float2half_rn(sv[1 * M + m])));
            }
          }
        }
      }
    }
  }
}

// Post-accumulation per-channel scale for the M=1 ksplit>1 path. Runs on the
// same stream, so it observes every CAS from the GEMV launch.
__global__ void dense_gemv_i8_scale_kernel(half* __restrict__ out,
                                           const half* __restrict__ s,
                                           const int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = __float2half_rn(__half2float(out[i]) * __half2float(s[i]));
}

}  // namespace dense_gemv_gfx906
}  // namespace vllm

// ---------------------------------------------------------------------------
// Entry point
//
//   weight: [N, K] fp16, row-major, contiguous
//   x:      [1, K] (or [K]) fp16, contiguous
//   kchunk: 512, 1024, 2048 or 4096 (must divide K); kchunk >= K = single pass
//   rpt:    rows per thread override (VLLM_GFX906_GEMV_RPT env); default
//           auto: 4 if N%4==0, 2 if N%2==0, else 1. RPT=1 forbids K-split.
//
// Returns out: [1, N] fp16. Pre-zeroed internally when K > kchunk.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// M<=16 entry point (spec decode M<=4; W4 skinny M=5..16; see
// dense_gemv_m_kernel above and dense_gemv_m_kernel_m16 below).
//
//   weight: [N, K] fp16 row-major; x: [M, K] fp16, 1 <= M <= 16
//   Returns out: [M, N] fp16.
// ---------------------------------------------------------------------------
torch::Tensor dense_gemv_m4_gfx906(torch::Tensor weight, torch::Tensor x,
                                   int64_t kchunk) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda());
  TORCH_CHECK(weight.dim() == 2 && x.dim() == 2);
  TORCH_CHECK(weight.scalar_type() == torch::kHalf);
  TORCH_CHECK(x.scalar_type() == torch::kHalf);
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous());
  const int64_t M = x.size(0);
  const int64_t N = weight.size(0);
  const int64_t K = weight.size(1);
  TORCH_CHECK(M >= 1 && M <= 16, "M must be 1..16 (got ", M, ")");
  TORCH_CHECK(x.size(1) == K, "x/weight K mismatch");
  TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
  TORCH_CHECK(kchunk == 512 || kchunk == 1024 || kchunk == 2048 ||
                  kchunk == 4096,
              "kchunk must be 512, 1024, 2048 or 4096");
  TORCH_CHECK(K % kchunk == 0, "K must be divisible by kchunk");

  // RPT is 2 or 4 (the packed CAS epilogue needs adjacent rows); env
  // override for micro-bench sweeps, default 2 (the M=1 K=17408 winner).
  int rpt = 2;
  if (const char* e = getenv("VLLM_GFX906_GEMVM_RPT")) {
    const int v = atoi(e);
    if (v == 2 || v == 4) rpt = v;
  }
  TORCH_CHECK(N % rpt == 0, "N (", N, ") not divisible by RPT (", rpt, ")");

  const int ksplit = (int)(K / kchunk);
  auto out = torch::empty({M, N}, weight.options());
  if (ksplit > 1) out.zero_();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  auto stream = at::cuda::getCurrentCUDAStream();
  const half* wp = (const half*)weight.data_ptr();
  const half* xp = (const half*)x.data_ptr();
  half* op = (half*)out.data_ptr();

  #define LAUNCHM(MVAL, RPT, KC)                                          \
    {                                                                     \
      dim3 grid(N / RPT, ksplit);                                         \
      vllm::dense_gemv_gfx906::dense_gemv_m_kernel<RPT, KC, MVAL>         \
          <<<grid, KC / 8, 0, stream>>>(xp, wp, op, (int)N, (int)K,       \
                                        ksplit);                          \
    }
  #define LAUNCHM_RT(RPT, KC)                                             \
    {                                                                     \
      dim3 grid(N / RPT, ksplit);                                         \
      vllm::dense_gemv_gfx906::dense_gemv_m_kernel_rt<RPT, KC>            \
          <<<grid, KC / 8, 0, stream>>>(xp, wp, op, (int)M, (int)N,       \
                                        (int)K, ksplit);                  \
    }
  #define LAUNCHM_RT_BY_RPT(KCVAL)                                        \
    do {                                                                  \
      if (rpt == 4)                                                       \
        LAUNCHM_RT(4, KCVAL)                                              \
      else                                                                \
        LAUNCHM_RT(2, KCVAL)                                              \
    } while (0)
  #define LAUNCHM_BY_RPT(MVAL, KCVAL)                                     \
    do {                                                                  \
      if (rpt == 4)                                                       \
        LAUNCHM(MVAL, 4, KCVAL)                                           \
      else                                                                \
        LAUNCHM(MVAL, 2, KCVAL)                                           \
    } while (0)
  #define LAUNCHM_BY_KC(MVAL)                                             \
    do {                                                                  \
      if (kchunk == 4096)                                                 \
        LAUNCHM_BY_RPT(MVAL, 4096);                                       \
      else if (kchunk == 2048)                                            \
        LAUNCHM_BY_RPT(MVAL, 2048);                                       \
      else if (kchunk == 1024)                                            \
        LAUNCHM_BY_RPT(MVAL, 1024);                                       \
      else                                                                \
        LAUNCHM_BY_RPT(MVAL, 512);                                        \
    } while (0)
  // M=5..16: RPT=1 exact-M m16 kernel (W4; see the kernel above for the
  // measured win/gate rationale).
  #define LAUNCHM16(MVAL, KC)                                             \
    {                                                                     \
      dim3 grid(N, ksplit);                                               \
      vllm::dense_gemv_gfx906::dense_gemv_m_kernel_m16<KC, MVAL>          \
          <<<grid, KC / 8, 0, stream>>>(xp, wp, op, (int)N, (int)K,       \
                                        ksplit);                          \
    }
  #define LAUNCHM16_BY_KC(MAXM)                                           \
    do {                                                                  \
      if (kchunk == 4096)                                                 \
        LAUNCHM16(MAXM, 4096)                                             \
      else if (kchunk == 2048)                                            \
        LAUNCHM16(MAXM, 2048)                                             \
      else if (kchunk == 1024)                                            \
        LAUNCHM16(MAXM, 1024)                                             \
      else                                                                \
        LAUNCHM16(MAXM, 512)                                              \
    } while (0)
  if (M == 1)
    LAUNCHM_BY_KC(1);
  else if (M == 2)
    LAUNCHM_BY_KC(2);
  else if (M == 3)
    LAUNCHM_BY_KC(3);
  else if (M == 4)
    // M=4: runtime-M kernel (the templated M=4 measured slower, see above).
    do {
      if (kchunk == 4096)
        LAUNCHM_RT_BY_RPT(4096);
      else if (kchunk == 2048)
        LAUNCHM_RT_BY_RPT(2048);
      else if (kchunk == 1024)
        LAUNCHM_RT_BY_RPT(1024);
      else
        LAUNCHM_RT_BY_RPT(512);
    } while (0);
  else if (M == 5)
    LAUNCHM16_BY_KC(5);
  else if (M == 6)
    LAUNCHM16_BY_KC(6);
  else if (M == 7)
    LAUNCHM16_BY_KC(7);
  else if (M == 8)
    LAUNCHM16_BY_KC(8);
  else if (M == 9)
    LAUNCHM16_BY_KC(9);
  else if (M == 10)
    LAUNCHM16_BY_KC(10);
  else if (M == 11)
    LAUNCHM16_BY_KC(11);
  else if (M == 12)
    LAUNCHM16_BY_KC(12);
  else if (M == 13)
    LAUNCHM16_BY_KC(13);
  else if (M == 14)
    LAUNCHM16_BY_KC(14);
  else if (M == 15)
    LAUNCHM16_BY_KC(15);
  else
    LAUNCHM16_BY_KC(16);
  return out;
}

torch::Tensor dense_gemv_gfx906(torch::Tensor weight, torch::Tensor x,
                                int64_t kchunk) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda());
  TORCH_CHECK(weight.dim() == 2 && x.dim() == 2);
  TORCH_CHECK(weight.scalar_type() == torch::kHalf);
  TORCH_CHECK(x.scalar_type() == torch::kHalf);
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous());
  const int64_t N = weight.size(0);
  const int64_t K = weight.size(1);
  TORCH_CHECK(x.size(0) == 1, "x must be [1, K] (M=1 only)");
  TORCH_CHECK(x.size(1) == K, "x/weight K mismatch");
  TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
  TORCH_CHECK(kchunk == 512 || kchunk == 1024 || kchunk == 2048 ||
                  kchunk == 4096,
              "kchunk must be 512, 1024, 2048 or 4096");
  TORCH_CHECK(K % kchunk == 0, "K must be divisible by kchunk");

  // Rows per thread: env override (micro-bench sweeps), else the
  // gfx906 (MI50)-measured rule: single-pass KCHUNK=2048 with RPT=2 wins
  // for N==256 (router) and N>=2048 (in_proj/qkv/LM head); RPT=4
  // elsewhere (RPT=2 is far worse on the 1024-row shared gate_up).
  int rpt = -1;
  if (const char* e = getenv("VLLM_GFX906_GEMV_RPT")) {
    rpt = atoi(e);
    TORCH_CHECK(rpt != 0,
                "VLLM_GFX906_GEMV_RPT must be 1, 2 or 4 (got 0)");
    if (rpt != 1 && rpt != 2 && rpt != 4) {
      TORCH_WARN_ONCE(
          "VLLM_GFX906_GEMV_RPT (", rpt, ") is not one of 1/2/4; using the "
          "default rule instead.");
      rpt = -1;
    }
  }
  if (rpt < 0) {
    if (kchunk == 2048 && (N == 256 || N >= 2048))
      rpt = 2;
    else if (kchunk == 512 && N == 2048)
      // shared-expert down_proj [2048, 512] (M=1 decode): RPT=2 measured
      // 5.6-5.7 us vs 6.7-7.7 us LLMM1 rpb4 and 8.0-8.2 us for RPT=4
      // (bench benchmarks/kernels/gfx906/bench_dense_gemv_gfx906.py).
      rpt = 2;
    else if (kchunk == 1024)
      // K=17408 down_proj (N=5120, ksplit=17): RPT=2 measured at 100.2%
      // of the HBM floor vs 116% for RPT=4 (bench /
      // benchmarks/kernels/gfx906/bench_dense_gemv_k5120.py).
      rpt = (N % 2 == 0) ? 2 : 1;
    else
      rpt = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
  }
  TORCH_CHECK(N % rpt == 0, "N (", N, ") not divisible by RPT (", rpt, ")");
  if (rpt == 1)
    TORCH_CHECK(kchunk >= K, "RPT=1 requires kchunk >= K (no K-split)");

  const int ksplit = (int)(K / kchunk);
  auto out = torch::empty({1, N}, weight.options());
  if (ksplit > 1) out.zero_();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  auto stream = at::cuda::getCurrentCUDAStream();
  const half* wp = (const half*)weight.data_ptr();
  const half* xp = (const half*)x.data_ptr();
  half* op = (half*)out.data_ptr();

  #define LAUNCH(RPT, KC)                                                   \
    {                                                                       \
      dim3 grid((N + RPT - 1) / RPT, ksplit);                               \
      vllm::dense_gemv_gfx906::dense_gemv_kernel<RPT, KC>                   \
          <<<grid, KC / 8, 0, stream>>>(xp, wp, op, (int)N, (int)K, ksplit);\
    }
  #define LAUNCH_BY_RPT(KCVAL)                                              \
    do {                                                                    \
      if (rpt == 4)                                                         \
        LAUNCH(4, KCVAL)                                                    \
      else if (rpt == 2)                                                    \
        LAUNCH(2, KCVAL)                                                    \
      else                                                                  \
        LAUNCH(1, KCVAL)                                                    \
    } while (0)

  if (kchunk == 4096)
    LAUNCH_BY_RPT(4096);
  else if (kchunk == 2048)
    LAUNCH_BY_RPT(2048);
  else if (kchunk == 1024)
    LAUNCH_BY_RPT(1024);
  else
    LAUNCH_BY_RPT(512);
  return out;
}

// ---------------------------------------------------------------------------
// NH-2' entry points (W8A16 int8-weight GEMV family).
//
//   weight: [N, K] int8, row-major, contiguous — the pre-shifted signed
//           view served by CompressedTensorsW8A16ChannelDequant (byte ^
//           0x80 at load; w = weight * scale)
//   scale:  [N] or [N, 1] fp16 (per output channel)
//   x:      [K] (M=1) or [M, K] (1 <= M <= 4) fp16, contiguous
//   kchunk: BYTES of weight per thread-slice; 1024, 2048 or 4096.
//           ksplit = ceil(K / kchunk); a partial tail block (threads whose
//           slice starts at/after K) contributes zero via the inb mask.
//
// Returns out: [1, N] (M=1) or [M, N] fp16. K % 16 == 0 is required
// (aligned 16B slice loads); other shapes fall back to the Triton path.
// RPT defaults to 2 (N%2==0) or 4 (N%4==0 and ksplit>1), overridable with
// VLLM_GFX906_GEMV_I8_RPT for micro-bench sweeps.
// ---------------------------------------------------------------------------

namespace {

int i8_rpt(int N, int ksplit, int default_rpt) {
  const char* e = getenv("VLLM_GFX906_GEMV_I8_RPT");
  if (e) {
    const int v = atoi(e);
    TORCH_CHECK(v == 2 || v == 4,
                "VLLM_GFX906_GEMV_I8_RPT must be 2 or 4 (got ", v, ")");
    return v;
  }
  return default_rpt;
}

bool i8_kchunk_ok(int64_t kchunk) {
  return kchunk == 1024 || kchunk == 2048 || kchunk == 4096;
}

}  // namespace

torch::Tensor dense_gemv_i8_gfx906(torch::Tensor weight, torch::Tensor scale,
                                   torch::Tensor x, int64_t kchunk) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda());
  TORCH_CHECK(weight.dim() == 2 && x.dim() == 1);
  TORCH_CHECK(weight.scalar_type() == torch::kChar);
  TORCH_CHECK(x.scalar_type() == torch::kHalf);
  TORCH_CHECK(scale.scalar_type() == torch::kHalf);
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous());
  const int64_t N = weight.size(0);
  const int64_t K = weight.size(1);
  TORCH_CHECK(x.size(0) == K, "x/weight K mismatch");
  TORCH_CHECK(scale.numel() == N, "scale must have N elements");
  TORCH_CHECK(K % 16 == 0, "K (", K, ") must be a multiple of 16");
  TORCH_CHECK(i8_kchunk_ok(kchunk),
              "kchunk must be 1024, 2048 or 4096 (got ", kchunk, ")");

  const int ksplit = (int)((K + kchunk - 1) / kchunk);  // ceil; tail masked
  if (ksplit > 1)
    TORCH_CHECK(N % 2 == 0 || N % 4 == 0,
                "N must be even for the packed CAS epilogue");
  int rpt = (N % 4 == 0 && ksplit > 1) ? 4 : (N % 2 == 0 ? 2 : -1);
  if (rpt < 0)
    TORCH_CHECK(false, "N (", N, ") must be even for the packed CAS epilogue");
  rpt = i8_rpt((int)N, ksplit, rpt);
  TORCH_CHECK(N % rpt == 0, "N (", N, ") not divisible by RPT (", rpt, ")");

  auto out = torch::empty({1, N}, x.options());
  if (ksplit > 1) out.zero_();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  auto stream = at::cuda::getCurrentCUDAStream();
  const signed char* wp = (const signed char*)weight.data_ptr();
  const half* sp = (const half*)scale.data_ptr();
  const half* xp = (const half*)x.data_ptr();
  half* op = (half*)out.data_ptr();

#define LAUNCH_I8(RPT, KC)                                                  \
  {                                                                         \
    dim3 grid((N + RPT - 1) / RPT, ksplit);                                 \
    vllm::dense_gemv_gfx906::dense_gemv_i8_kernel<RPT, KC>                  \
        <<<grid, KC / 16, 0, stream>>>(xp, wp, sp, op, (int)N, (int)K,      \
                                       ksplit);                             \
  }

#define LAUNCH_I8_BY_RPT(KCVAL)                                             \
  do {                                                                      \
    if (rpt == 4)                                                           \
      LAUNCH_I8(4, KCVAL)                                                   \
    else                                                                    \
      LAUNCH_I8(2, KCVAL)                                                   \
  } while (0)

  if (kchunk == 4096)
    LAUNCH_I8_BY_RPT(4096);
  else if (kchunk == 2048)
    LAUNCH_I8_BY_RPT(2048);
  else
    LAUNCH_I8_BY_RPT(1024);

#undef LAUNCH_I8
#undef LAUNCH_I8_BY_RPT

  if (ksplit > 1) {
    const int threads = 256;
    const int blocks = (int)((N + threads - 1) / threads);
    vllm::dense_gemv_gfx906::dense_gemv_i8_scale_kernel<<<blocks, threads, 0, stream>>>(
        op, sp, (int)N);
  }
  return out;
}

torch::Tensor dense_gemv_i8_m4_gfx906(torch::Tensor weight, torch::Tensor scale,
                                      torch::Tensor x, int64_t kchunk) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda());
  TORCH_CHECK(weight.dim() == 2 && x.dim() == 2);
  TORCH_CHECK(weight.scalar_type() == torch::kChar);
  TORCH_CHECK(x.scalar_type() == torch::kHalf);
  TORCH_CHECK(scale.scalar_type() == torch::kHalf);
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous());
  const int64_t M = x.size(0);
  const int64_t N = weight.size(0);
  const int64_t K = weight.size(1);
  // T-1.5 (2026-09-05): M<=8 (was <=4). Covers full-acceptance k=4 verify
  // (M=k+1=5) in-kernel; the op name keeps "m4" for ABI stability.
  TORCH_CHECK(M >= 1 && M <= 8, "M must be 1..8 (got ", M, ")");
  TORCH_CHECK(x.size(1) == K, "x/weight K mismatch");
  TORCH_CHECK(scale.numel() == N, "scale must have N elements");
  TORCH_CHECK(K % 16 == 0, "K (", K, ") must be a multiple of 16");
  TORCH_CHECK(i8_kchunk_ok(kchunk),
              "kchunk must be 1024, 2048 or 4096 (got ", kchunk, ")");

  const int ksplit = (int)((K + kchunk - 1) / kchunk);  // ceil; tail masked
  int rpt = (N % 4 == 0) ? 4 : (N % 2 == 0 ? 2 : -1);
  if (rpt < 0)
    TORCH_CHECK(false, "N (", N, ") must be even for the packed CAS epilogue");
  rpt = i8_rpt((int)N, ksplit, rpt);
  TORCH_CHECK(N % rpt == 0, "N (", N, ") not divisible by RPT (", rpt, ")");

  auto out = torch::empty({M, N}, x.options());
  if (ksplit > 1) out.zero_();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  auto stream = at::cuda::getCurrentCUDAStream();
  const signed char* wp = (const signed char*)weight.data_ptr();
  const half* sp = (const half*)scale.data_ptr();
  const half* xp = (const half*)x.data_ptr();
  half* op = (half*)out.data_ptr();

#define LAUNCHM_I8(MVAL, RPT, KC)                                           \
  {                                                                         \
    dim3 grid((N + RPT - 1) / RPT, ksplit);                                 \
    vllm::dense_gemv_gfx906::dense_gemv_i8_m_kernel<RPT, KC, MVAL>          \
        <<<grid, KC / 16, 0, stream>>>(xp, wp, sp, op, (int)N, (int)K,      \
                                       ksplit);                             \
  }

#define LAUNCHM_I8_BY_RPT(MVAL, KCVAL)                                      \
  do {                                                                      \
    if (rpt == 4)                                                           \
      LAUNCHM_I8(MVAL, 4, KCVAL)                                            \
    else                                                                    \
      LAUNCHM_I8(MVAL, 2, KCVAL)                                            \
  } while (0)

#define LAUNCHM_I8_BY_KC(MVAL)                                              \
  do {                                                                      \
    if (kchunk == 4096)                                                     \
      LAUNCHM_I8_BY_RPT(MVAL, 4096);                                        \
    else if (kchunk == 2048)                                                \
      LAUNCHM_I8_BY_RPT(MVAL, 2048);                                        \
    else                                                                    \
      LAUNCHM_I8_BY_RPT(MVAL, 1024);                                        \
  } while (0)

  // T-1.5 (2026-09-05): explicit cases — the macro needs a compile-time M
  // token, and each value is its own kernel instantiation (static register
  // arrays sized to M).
  if (M == 1)
    LAUNCHM_I8_BY_KC(1);
  else if (M == 2)
    LAUNCHM_I8_BY_KC(2);
  else if (M == 3)
    LAUNCHM_I8_BY_KC(3);
  else if (M == 4)
    LAUNCHM_I8_BY_KC(4);
  else if (M == 5)
    LAUNCHM_I8_BY_KC(5);
  else if (M == 6)
    LAUNCHM_I8_BY_KC(6);
  else if (M == 7)
    LAUNCHM_I8_BY_KC(7);
  else
    LAUNCHM_I8_BY_KC(8);

#undef LAUNCHM_I8
#undef LAUNCHM_I8_BY_RPT
#undef LAUNCHM_I8_BY_KC
  return out;
}
