// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
//
// Standalone fast-iteration harness for moe_topk_gfx906.cu (S2).
// Compiles the kernel in isolation (no torch), checks ids/weights against a
// CPU softmax reference, times per-call us, and prints the ISA shuffle mix.
// The bit-equal gate vs the GPU generic topkGating lives in
// tests/kernels/moe/test_fused_topk.py (full build required).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
  exit(1); } } while (0)

static constexpr int K_TOP = 8;
static constexpr int LANES = 32;
static constexpr int VPT = 8;
static constexpr int WAVE = 64;

// Force a real wave barrier: the backend elides __syncthreads() for
// single-wave CTAs (per-wave LDS FIFO ordering makes it unnecessary,
// verified by stress test), but we don't want correctness to hinge on
// that codegen choice surviving flag changes.
#define WAVE_BARRIER() asm volatile("s_barrier\n" ::: "memory")

__shared__ float s_row[LANES];   // row-max / row-sum butterfly
__shared__ float s_mv[LANES];     // per-pass local argmax values
__shared__ int s_ex[LANES];       // per-pass local argmax expert ids
__shared__ float s_mp;            // winner's p broadcast (renormalize)

__global__ void __launch_bounds__(64)
    topk_softmax_m1_gfx906_kernel(const __half* __restrict__ gating,
                                  float* __restrict__ topk_weights,
                                  int* __restrict__ topk_ids,
                                  int* __restrict__ token_expert_ids,
                                  const bool renormalize) {
  const int t = threadIdx.x;
  const int expert_base = VPT * t;

  float p[VPT];
  if (t < LANES) {
    uint4 raw = *(const uint4*)(gating + VPT * t);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
    #pragma unroll
    for (int i = 0; i < VPT / 2; ++i) {
      float2 f = __half22float2(h2[i]);
      p[2 * i] = f.x;
      p[2 * i + 1] = f.y;
    }
  } else {
    #pragma unroll
    for (int i = 0; i < VPT; ++i) p[i] = -INFINITY;
  }

  // (1) row max: per-lane max, then LDS reduce (max is exact — any
  // tree/order gives the generic's bit-equal result).
  float row_max = p[0];
  #pragma unroll
  for (int i = 1; i < VPT; ++i) row_max = fmaxf(row_max, p[i]);
  if (t < LANES) s_row[t] = row_max;
  WAVE_BARRIER();
  if (t < LANES) {
    #pragma unroll
    for (int j = 0; j < LANES; ++j) row_max = fmaxf(row_max, s_row[j]);
  }

  // (2) softmax: expf, sequential local sum, then the generic's exact
  // xor(16,8,4,2,1) butterfly reproduced round-for-round via LDS.
  float row_sum = 0.f;
  #pragma unroll
  for (int i = 0; i < VPT; ++i) {
    p[i] = expf(p[i] - row_max);
    row_sum += p[i];
  }
  #pragma unroll
  for (int mask = LANES / 2; mask > 0; mask >>= 1) {
    if (t < LANES) s_row[t] = row_sum;
    WAVE_BARRIER();
    if (t < LANES) row_sum += s_row[t ^ mask];
  }

  const float recip = 1.f / row_sum;
  #pragma unroll
  for (int i = 0; i < VPT; ++i) p[i] *= recip;

  #pragma unroll
  for (int i = 0; i < VPT; ++i)
    if (isnan(p[i]) || isinf(p[i])) p[i] = 0.f;

  float choice[VPT];
  #pragma unroll
  for (int i = 0; i < VPT; ++i) choice[i] = p[i];

  float selected_sum = 0.f;
  #pragma unroll
  for (int k_idx = 0; k_idx < K_TOP; ++k_idx) {
    float mv = choice[0];
    float mv_p = p[0];
    int expert = expert_base;
    #pragma unroll
    for (int i = 1; i < VPT; ++i) {
      if (choice[i] > mv) {
        mv = choice[i];
        mv_p = p[i];
        expert = expert_base + i;
      }
    }
    // Cross-lane argmax via LDS: every lane scans the same 32 (value, id)
    // pairs, so all lanes agree on the winner without any shuffle. The
    // comparison is the generic's strict total order (value desc, expert
    // asc), which makes the winner tree/order independent.
    if (t < LANES) {
      s_mv[t] = mv;
      s_ex[t] = expert;
    }
    WAVE_BARRIER();
    float wmv = -INFINITY;
    int wex = 0x7fffffff;
    if (t < LANES) {
      #pragma unroll
      for (int j = 0; j < LANES; ++j) {
        float v = s_mv[j];
        int e = s_ex[j];
        if (v > wmv || (v == wmv && e < wex)) {
          wmv = v;
          wex = e;
        }
      }
    }
    const int owner = wex / VPT;  // always < 32: dummies carry ids 256+
    if (t == owner) {
      topk_weights[k_idx] = mv_p;
      if (renormalize) s_mp = mv_p;
    }
    if (renormalize) WAVE_BARRIER();
    if (t == 0) {
      topk_ids[k_idx] = wex;
      token_expert_ids[k_idx] = k_idx;
      if (renormalize) selected_sum += s_mp;
    }
    if (k_idx + 1 < K_TOP && t == owner)
      choice[wex % VPT] = -10000.f;
  }

  if (t == 0 && renormalize) {
    const float denom = selected_sum > 0.f ? selected_sum : 1.f;
    const float scale = 1.0f / denom;
    #pragma unroll
    for (int i = 0; i < K_TOP; ++i) topk_weights[i] *= scale;
  }
}

// CPU reference: ids must match; weights must match the fp32 softmax of the
// chosen experts (within a few ulps; GPU expf may differ from CPU expf).
static void cpu_ref(const float* g, float* ref_w, int* ref_i, bool renorm) {
  float m = g[0];
  for (int i = 1; i < 256; ++i) m = fmaxf(m, g[i]);
  double s = 0;
  float e[256];
  for (int i = 0; i < 256; ++i) { e[i] = expf(g[i] - m); s += e[i]; }
  for (int i = 0; i < 256; ++i) e[i] = e[i] / (float)s;
  for (int k = 0; k < K_TOP; ++k) {
    int best = -1;
    for (int i = 0; i < 256; ++i)
      if (best < 0 || e[i] > e[best]) best = i;
    ref_w[k] = e[best];
    ref_i[k] = best;
    e[best] = -1e30f;
  }
  if (renorm) {
    double ss = 0;
    for (int k = 0; k < K_TOP; ++k) ss += ref_w[k];
    float scale = 1.0f / (float)(ss > 0 ? ss : 1);
    for (int k = 0; k < K_TOP; ++k) ref_w[k] *= scale;
  }
}

int main(int argc, char** argv) {
  const int stress = argc > 1 ? atoi(argv[1]) : 0;
  float* h_g = (float*)malloc(256 * sizeof(float));
  __half* h_gh2 = (__half*)malloc(256 * sizeof(__half));
  __half* d_g2; CHECK(hipMalloc(&d_g2, 256 * sizeof(__half)));
  float* d_w2; int* d_i2; int* d_t2;
  CHECK(hipMalloc(&d_w2, 8 * sizeof(float)));
  CHECK(hipMalloc(&d_i2, 8 * sizeof(int)));
  CHECK(hipMalloc(&d_t2, 8 * sizeof(int)));
  if (stress > 0) {
    int bad = 0;
    for (int trial = 0; trial < stress; ++trial) {
      float scale = 1.f + (trial % 4) * 2.f;
      if (trial % 37 == 0) scale = 0.f;          // all-equal (ties)
      for (int i = 0; i < 256; ++i)
        h_g[i] = (rand() % 2000 - 1000) / 100.f * scale;
      if (trial % 53 == 0) {                      // sparse: one big, rest far
        for (int i = 0; i < 256; ++i) h_g[i] = -500.f + (rand() % 10);
        h_g[rand() % 256] = 300.f;
      }
      if (trial % 91 == 0) {                      // 64-way tie
        for (int i = 0; i < 256; ++i) h_g[i] = (i % 4 == 0) ? 1.f : 0.f;
      }
      for (int i = 0; i < 256; ++i) {
        h_gh2[i] = __float2half(h_g[i]);
        h_g[i] = __half2float(h_gh2[i]);          // exact fp16-rounded input
      }
      CHECK(hipMemcpy(d_g2, h_gh2, 256 * sizeof(__half), hipMemcpyHostToDevice));
      float rw[8]; int ri[8];
      cpu_ref(h_g, rw, ri, true);
      topk_softmax_m1_gfx906_kernel<<<1, 64>>>(d_g2, d_w2, d_i2, d_t2, true);
      CHECK(hipDeviceSynchronize());
      float w[8]; int ids[8];
      CHECK(hipMemcpy(w, d_w2, 8 * sizeof(float), hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(ids, d_i2, 8 * sizeof(int), hipMemcpyDeviceToHost));
      for (int k = 0; k < 8; ++k) {
        if (ids[k] != ri[k]) {
          if (bad < 5) printf("trial %d: id[%d]=%d expected %d\n",
                              trial, k, ids[k], ri[k]);
          ++bad;
        }
        if (fabsf(w[k] - rw[k]) > 8.0f * rw[k] * 1.19e-7f + 1e-7f) {
          if (bad < 5) printf("trial %d: w[%d]=%g expected %g\n",
                              trial, k, w[k], rw[k]);
          ++bad;
        }
      }
    }
    printf("stress %d trials: %s (%d mismatches)\n",
           stress, bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
  }
  srand(1234);
  for (int i = 0; i < 256; ++i) h_g[i] = (rand() % 2000 - 1000) / 100.f;

  __half* h_gh = h_gh2;
  for (int i = 0; i < 256; ++i) h_gh[i] = __float2half(h_g[i]);

  __half* d_g = d_g2;
  CHECK(hipMemcpy(d_g, h_gh, 256 * sizeof(__half), hipMemcpyHostToDevice));
  float* d_w = d_w2; int* d_i = d_i2; int* d_tei = d_t2;

  bool ok = true;
  for (int ren = 0; ren < 2; ++ren) {
    CHECK(hipMemset(d_w, 0, 8 * sizeof(float)));
    CHECK(hipMemset(d_i, 0, 8 * sizeof(int)));
    topk_softmax_m1_gfx906_kernel<<<1, 64>>>(d_g, d_w, d_i, d_tei, ren);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    float w[8]; int ids[8];
    CHECK(hipMemcpy(w, d_w, 8 * sizeof(float), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(ids, d_i, 8 * sizeof(int), hipMemcpyDeviceToHost));
    float rw[8]; int ri[8];
    // CPU ref on the exact fp16-rounded inputs
    float g16[256];
    for (int i = 0; i < 256; ++i) g16[i] = __half2float(h_gh[i]);
    cpu_ref(g16, rw, ri, ren);
    for (int k = 0; k < 8; ++k) {
      if (ids[k] != ri[k]) {
        printf("renorm=%d: id[%d]=%d expected %d  MISMATCH\n", ren, k, ids[k], ri[k]);
        ok = false;
      }
      float diff = fabsf(w[k] - rw[k]);
      float ulps = diff / (rw[k] * 1.19e-7f);
      if (ulps > 8.0f) {
        printf("renorm=%d: w[%d]=%g expected %g (%.1f ulp) MISMATCH\n",
               ren, k, w[k], rw[k], ulps);
        ok = false;
      }
    }
    printf("renorm=%d: ids=%d,%d,%d,%d,%d,%d,%d,%d  w0=%.6f  %s\n",
           ren, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7],
           w[0], ok ? "ok" : "BAD");
  }

  // timing (launch-to-launch, same regime as the microbench)
  hipStream_t s; CHECK(hipStreamCreate(&s));
  for (int i = 0; i < 50; ++i)
    topk_softmax_m1_gfx906_kernel<<<1, 64, 0, s>>>(d_g, d_w, d_i, d_tei, true);
  CHECK(hipStreamSynchronize(s));
  hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
  hipEventRecord(e0, s);
  for (int i = 0; i < 1000; ++i)
    topk_softmax_m1_gfx906_kernel<<<1, 64, 0, s>>>(d_g, d_w, d_i, d_tei, true);
  hipEventRecord(e1, s);
  CHECK(hipStreamSynchronize(s));
  float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
  printf("per-call (1000 launches): %.2f us\n", ms * 1000.f);
  printf("%s\n", ok ? "HARNESS PASS" : "HARNESS FAIL");
  return ok ? 0 : 1;
}
