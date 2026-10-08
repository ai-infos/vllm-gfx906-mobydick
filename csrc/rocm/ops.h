#pragma once

#include <torch/all.h>

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

torch::Tensor wvSplitK_int4_g(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_zero_points,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size);

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              bool use_v2_format);

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales, bool use_v2_format);

void moe_gptq_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

void moe_gptq_gemm_gfx906(torch::Tensor a, torch::Tensor c,
                          torch::Tensor b_q_weight, torch::Tensor b_scales,
                          torch::Tensor b_qzeros, torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded,
                          int64_t top_k, int64_t block_size_m,
                          bool mul_topk_weight, int64_t output_topk,
                          int64_t zero_offset);

// Test-only: returns the M=1 gemm dispatch-path marker set by the most recent
// moe_gptq_gemm_gfx906 call (0 = legacy <1,4> gemm1 opt-out, 1 = v2 512-thread
// gemm2, 2 = legacy <1,4> gemm2 fallback, 3 = default <1,2> gemm1) and resets
// it to 0. Needed because the M=1 kernels are atomic-accumulated and therefore
// not bit-reproducible run-to-run — output comparison cannot prove which tile
// ran (see csrc/rocm/moe_q_gemm_gfx906.cu). Returns int64_t (schema "int").
int64_t take_moe_m1_dispatch_path();

// M<=4 W16A16 dense GEMM (GEMV-family) for gfx906 spec decode (L1').
torch::Tensor dense_gemv_m4_gfx906(torch::Tensor weight, torch::Tensor x,
                                   int64_t kchunk);

// M=1 W8A16 int8-weight dense GEMV for gfx906 (NH-2'; see
// dense_gemv_gfx906.cu). weight [N, K] row-major pre-shifted signed int8
// (CompressedTensorsW8A16ChannelDequant convention: w = weight * scale),
// scale [N] fp16 per-channel, x [K] fp16. kchunk 512|1024|2048|4096 BYTES
// of weight per thread-slice (must divide K when split; >= K means a
// single pass). K % 16 == 0 required. Returns out [1, N] fp16.
torch::Tensor dense_gemv_i8_gfx906(torch::Tensor weight, torch::Tensor scale,
                                   torch::Tensor x, int64_t kchunk);

// M<=4 W8A16 int8-weight dense GEMM (GEMV-family) for gfx906 spec decode
// (NH-2'). Same conventions as dense_gemv_i8_gfx906; x is [M, K] with
// 1 <= M <= 4. Returns out [M, N] fp16.
torch::Tensor dense_gemv_i8_m4_gfx906(torch::Tensor weight, torch::Tensor scale,
                                      torch::Tensor x, int64_t kchunk);

// M=1 W16A16 dense GEMV for gfx906 (P3-2b; see dense_gemv_gfx906.cu).
// weight [N, K] row-major fp16, x [1, K] fp16, kchunk 512|1024|2048|4096 (divides K).
// Returns out [1, N] fp16.
torch::Tensor dense_gemv_gfx906(torch::Tensor weight, torch::Tensor x,
                                int64_t kchunk);

// M=1 fused moe_align_block_size + count_and_sort for gfx906, E=256,
// topk=8, block_size=1 (C1 stage 1; see moe_align_m1_gfx906.cu).
// Buffers are the wrapper-convention sizes for this shape (8 / 8 / 1).
void moe_align_block_size_m1_gfx906(torch::Tensor topk_ids,          // [1, 8] i32
                                    int64_t num_experts,             // 256
                                    int64_t block_size,              // 1
                                    torch::Tensor sorted_token_ids,  // [8] i32
                                    torch::Tensor expert_ids,        // [8] i32
                                    torch::Tensor num_tokens_post_pad);  // [1] i32

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type);
