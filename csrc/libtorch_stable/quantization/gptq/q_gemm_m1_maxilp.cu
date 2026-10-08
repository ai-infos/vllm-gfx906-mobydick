// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>

/*
gfx906: M=1 4-bit GPTQ GEMM compiled with the LLVM max-ilp scheduler
strategy (see the max-ilp block in CMakeLists.txt).

Why a separate translation unit: the per-file `-amdgpu-sched-strategy=
max-ilp` flag applies to every kernel instantiation in q_gemm.cu, but
measured (docs/gfx906/DEVLOG-spec-decode.md, 2026-08-24, ROCm
7.14.60850) the flag helps the M=1 q_gemm (+19..24%) while it
REGRESSES the M=2..4 shapes used by spec-decode steps (-14..-23%).
So M=1 takes this max-ilp kernel and M>=2 keeps the unflagged kernel
in q_gemm.cu; gemm_half_q_half_cuda_part routes on m_count.

The kernel is a renamed copy of gemm_half_q_half_gptq_4bit_kernel
(m_count=1 only) so the two TUs can carry the same algorithm without
linking the same symbol. The max-ilp flag is applied to this TU only
for gfx906 builds (CMakeLists.txt); the launch is additionally guarded
at RUNTIME on gcnArchName == "gfx906" in q_gemm.cu, so other arches
compile the kernel but never launch it.

Env kill-switch: VLLM_GFX906_QGEMM_M1_MAXILP=0 restores the unflagged
M=1 kernel.

================================================================================
SYNCHRONIZATION WARNING -- READ BEFORE EDITING EITHER COPY
================================================================================
The gemm_half_q_half_gptq_4bit_kernel_m1mi kernel below is a 140-line
renamed copy of gemm_half_q_half_gptq_4bit_kernel in q_gemm.cu (plus
the dot22_8_f device helper, copied as dot22_8_f_m1mi). The ONLY
intended differences are the two names and the compile flag applied to
this TU (-amdgpu-sched-strategy=max-ilp, see CMakeLists.txt).

* If you change the 4-bit kernel or dot22_8_f in q_gemm.cu (bug fix,
  perf tweak, upstream port), you MUST apply the identical change here
  (renaming the two symbols) and rebuild BOTH translation units.
* If you change it here, apply the identical change to q_gemm.cu.
* The two copies are verified identical by a normalized textual diff
  (names/params aside) -- do not let that drift silently; a one-sided
  edit changes M=1 numerics/perf without any compile or test signal.
* A zero-duplication design (shared header with macro-renamed kernel)
  was considered and rejected for this local branch; reconsider it if
  this ever gets upstreamed.
================================================================================
*/

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../../torch_utils.h"
#include <torch/csrc/stable/ops.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "compat.cuh"
#include "matrix_view.cuh"
#include "qdq_4.cuh"

namespace vllm {
namespace gptq {

#define BLOCK_KN_SIZE 256
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

// SYNC-COPY: renamed copy of dot22_8_f in q_gemm.cu -- keep in lockstep.
static __forceinline__ __device__ float dot22_8_f_m1mi(
    half2 (&dq)[4], const half* a_ptr) {
  float result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
  #pragma unroll
  for (int i = 0; i < 4; i++) result = __ockl_fdot2(dq[i], *a2_ptr++, result, true);
  return result;
}

// SYNC-COPY (1/2): this kernel must stay in lockstep with
// gemm_half_q_half_gptq_4bit_kernel in q_gemm.cu (names aside).
// See the SYNCHRONIZATION WARNING in the header of this file.
template <int m_count>
__launch_bounds__(BLOCK_KN_SIZE)
__global__ void gemm_half_q_half_gptq_4bit_kernel_m1mi(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const bool use_v2_format) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  // GPTQv2 and GPTQv1 handles zero points differently
  int zero_offset = use_v2_format ? 0 : 1;

  auto t = threadIdx.x;

  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];

      block_a_ptr[t] = a_ptr[offset_k + t];
    }
  }

  // Zero output
  if (n >= size_n) return;

  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  __syncthreads();

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / (32 / 4);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  float scales[4];
  half2 z1z16[4][2];
  half2 y1y16[4][2];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_f(scales, group, n);
  dequant_4bit_8_prep_zero(zeros[0] + zero_offset, z1z16[0], y1y16[0]);
  dequant_4bit_8_prep_zero(zeros[1] + zero_offset, z1z16[1], y1y16[1]);
  dequant_4bit_8_prep_zero(zeros[2] + zero_offset, z1z16[2], y1y16[2]);
  dequant_4bit_8_prep_zero(zeros[3] + zero_offset, z1z16[3], y1y16[3]);

  // Column result
  float block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_f(scales, group, n);
      dequant_4bit_8_prep_zero(zeros[0] + zero_offset, z1z16[0], y1y16[0]);
      dequant_4bit_8_prep_zero(zeros[1] + zero_offset, z1z16[1], y1y16[1]);
      dequant_4bit_8_prep_zero(zeros[2] + zero_offset, z1z16[2], y1y16[2]);
      dequant_4bit_8_prep_zero(zeros[3] + zero_offset, z1z16[3], y1y16[3]);
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;

      half2 dq[4][4];
      dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0], y1y16[0], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1], y1y16[1], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2], y1y16[2], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3], y1y16[3], size_n,
                          false);

#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] = fma(dot22_8_f_m1mi(dq[0], a_ptr + m * a_stride),
                            scales[0], block_c[m][0]);
        block_c[m][1] = fma(dot22_8_f_m1mi(dq[1], a_ptr + m * a_stride),
                            scales[1], block_c[m][1]);
        block_c[m][2] = fma(dot22_8_f_m1mi(dq[2], a_ptr + m * a_stride),
                            scales[2], block_c[m][2]);
        block_c[m][3] = fma(dot22_8_f_m1mi(dq[3], a_ptr + m * a_stride),
                            scales[3], block_c[m][3]);
      }

      b_ptr += size_n;
      a_ptr += 8;
    }

    k += 32;
  }

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                    __float2half_rn(block_c[m][1]));
    half2 result23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                    __float2half_rn(block_c[m][3]));
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

// M=1, 4-bit only (the measured gfx906 workload); everything else stays on
// the q_gemm.cu path. Grid math mirrors gemm_half_q_half_cuda_part with
// m_count=1.
void qgemm_m1_maxilp_launch(const half* a, const uint32_t* b_q_weight,
                            const uint32_t* b_gptq_qzeros,
                            const half* b_gptq_scales, half* c, int size_m,
                            int size_n, int size_k, int groups,
                            bool use_v2_format) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE * 4);
  gridDim.y = size_m;
  gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);

  const cudaStream_t stream = get_current_cuda_stream();
  gemm_half_q_half_gptq_4bit_kernel_m1mi<1><<<gridDim, blockDim, 0, stream>>>(
      a, b_q_weight, b_gptq_qzeros, b_gptq_scales, c, size_m, size_n, size_k,
      groups, use_v2_format);
}

}  // namespace gptq
}  // namespace vllm
