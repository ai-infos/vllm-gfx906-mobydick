// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
//
// M=1 fused moe_align_block_size + count_and_sort kernel for gfx906
// (Vega 20). Serves exactly two (E, topk) pairs:
//   (256, 8) — Qwen3.5-35B (C1 stage 1, see
//              docs/gfx906/DEVLOG-moe-c1-routing-fusion.md)
//   (128, 6) — Nemotron-3.5-Lightning-30B (NH-5, same C1-stage-1
//              design extended; see docs/gfx906/DEVLOG-nemotron-h.md)
//
// The generic moe_align_block_size binding launches TWO kernels for
// these shapes (moe_align_block_size_kernel<<<2, 1024>>> for counting +
// cumsum + expert_ids, then count_and_sort_expert_tokens_kernel for the
// atomic-based placement of sorted_token_ids). Both are over-provisioned
// for the handful of (token, expert) pairs: 1024-thread CUB BlockScan
// over E counters, a second CTA, global atomics. This kernel does the
// same job in one 128-thread CTA with LDS atomics and a shfl_down warp
// scan, so the two graph nodes collapse into one.
//
// Outputs are bit-equal to the two-kernel chain for each pair (verified
// in tests/kernels/moe/test_moe_align_m1_gfx906.py, incl. tie-heavy
// cases):
//   sorted_token_ids[i] = flat topk slot index (0..TOPK-1) at sorted i
//   expert_ids[i]       = expert at sorted position i (block_size == 1)
//   num_tokens_post_pad = TOPK
// At these shapes the production wrapper sizes both buffers to exactly
// numel == TOPK, so no sentinel fill is needed. Within-expert slot order
// follows the issuing-lane order (slots 0..TOPK-1), matching the
// production kernel's observed single-warp global-atomic order.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "../cuda_compat.h"

namespace vllm {
namespace moe_align_m1_gfx906 {

static constexpr int THREADS = 128;

template <int E, int TOPK>
struct AlignM1Config {
  static_assert((E == 256 && TOPK == 8) || (E == 128 && TOPK == 6),
                "only the (256, 8) and (128, 6) decode shapes are "
                "supported");
  static constexpr int COUNTS_PER_LANE = E / 32;
  static_assert(COUNTS_PER_LANE * 32 == E, "E must be a multiple of 32");
};

template <int E, int TOPK>
__global__ void __launch_bounds__(THREADS) moe_align_m1_gfx906_kernel(
    const int* __restrict__ topk_ids,      // [TOPK]
    int* __restrict__ sorted_token_ids,    // [TOPK]
    int* __restrict__ expert_ids,          // [TOPK]
    int* __restrict__ num_tokens_post_pad) {  // [1]
  constexpr int CPLE = AlignM1Config<E, TOPK>::COUNTS_PER_LANE;
  __shared__ int s_counts[E];
  __shared__ int s_base[E];
  __shared__ int s_rank[E];

  const int t = threadIdx.x;

  // Zero counts + rank (2 x E ints over 128 threads).
  #pragma unroll
  for (int i = 0; i < 2 * E / THREADS; ++i) {
    const int idx = t + i * THREADS;
    if (idx < E) s_counts[idx] = 0;
    else s_rank[idx - E] = 0;
  }
  __syncthreads();

  // Count tokens per expert (LDS atomics; order-independent).
  if (t < TOPK) {
    atomicAdd(&s_counts[topk_ids[t]], 1);
  }
  __syncthreads();

  // Exclusive prefix sum over the E per-expert counts: warp 0, CPLE counts
  // per lane. Hillis-Steele suffix scan via shfl_down (5 steps); the
  // exclusive prefix is total - suffix with total == TOPK slots.
  if (t < 32) {
    int lane_excl[CPLE];
    int acc = 0;
    #pragma unroll
    for (int j = 0; j < CPLE; ++j) {
      int c = s_counts[CPLE * t + j];
      lane_excl[j] = acc;
      acc += c;
    }
    int incl = acc;  // lane sum
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int n = VLLM_SHFL_DOWN_SYNC(incl, d);
      if (t + d < 32) incl += n;
    }
    // incl = sum of this lane and all higher lanes; exclusive prefix of
    // this lane = total (TOPK) - incl.
    int const excl = TOPK - incl;
    #pragma unroll
    for (int j = 0; j < CPLE; ++j) {
      s_base[CPLE * t + j] = excl + lane_excl[j];
    }
    if (t == 31) {
      num_tokens_post_pad[0] = TOPK;  // total = sum of all counts
    }
  }
  __syncthreads();

  // Place each slot into its expert's (single-slot) block; lane-atomic rank
  // reproduces the production within-expert slot order.
  if (t < TOPK) {
    const int e = topk_ids[t];
    const int pos = s_base[e] + atomicAdd(&s_rank[e], 1);
    sorted_token_ids[pos] = t;
    expert_ids[pos] = e;
  }
}

}  // namespace moe_align_m1_gfx906
}  // namespace vllm

void moe_align_block_size_m1_gfx906(torch::Tensor topk_ids,
                                    int64_t num_experts,
                                    int64_t block_size,
                                    torch::Tensor sorted_token_ids,
                                    torch::Tensor expert_ids,
                                    torch::Tensor num_tokens_post_pad) {
  TORCH_CHECK(topk_ids.is_cuda() && sorted_token_ids.is_cuda() &&
                  expert_ids.is_cuda() && num_tokens_post_pad.is_cuda(),
              "all tensors must be CUDA/HIP tensors");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt &&
                  sorted_token_ids.scalar_type() == torch::kInt &&
                  expert_ids.scalar_type() == torch::kInt &&
                  num_tokens_post_pad.scalar_type() == torch::kInt,
              "all tensors must be int32");
  TORCH_CHECK(topk_ids.size(0) == 1, "M=1 only, got M=",
              topk_ids.size(0));
  TORCH_CHECK(block_size == 1, "block_size=1 only, got ", block_size);
  TORCH_CHECK(topk_ids.is_contiguous() &&
                  sorted_token_ids.is_contiguous() &&
                  expert_ids.is_contiguous() &&
                  num_tokens_post_pad.is_contiguous(),
              "all tensors must be contiguous");

  const int topk = topk_ids.size(1);
  const int numel = topk_ids.size(1);
  TORCH_CHECK(sorted_token_ids.size(0) == numel &&
                  expert_ids.size(0) == numel,
              "buffers must be sized ", numel,
              " (wrapper convention for M=1), got ",
              sorted_token_ids.size(0), " / ", expert_ids.size(0));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(topk_ids));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int* ids = topk_ids.data_ptr<int>();
  int* s = sorted_token_ids.data_ptr<int>();
  int* e = expert_ids.data_ptr<int>();
  int* n = num_tokens_post_pad.data_ptr<int>();

  if (num_experts == 256 && topk == 8) {
    vllm::moe_align_m1_gfx906::moe_align_m1_gfx906_kernel<256, 8>
        <<<1, vllm::moe_align_m1_gfx906::THREADS, 0, stream>>>(ids, s, e, n);
  } else if (num_experts == 128 && topk == 6) {
    vllm::moe_align_m1_gfx906::moe_align_m1_gfx906_kernel<128, 6>
        <<<1, vllm::moe_align_m1_gfx906::THREADS, 0, stream>>>(ids, s, e, n);
  } else {
    TORCH_CHECK(false, "unsupported (E, topk) pair: (", num_experts, ", ",
                topk, "); supported: (256, 8), (128, 6)");
  }
}
