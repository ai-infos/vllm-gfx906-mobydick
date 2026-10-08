// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) Nick — nick413@gmail.com
// SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
//
// Vendored from https://github.com/cassettesgoboom/gfx906-fa-vllm
// (FlashAttention-style custom attention backend for vLLM on AMD gfx906).
//
// gfx906_fa_gather.cu — fused HIP gather for paged KV -> contiguous BHSD.
//
// Level 1 optimization: replaces the Python fancy-indexing path
// (_gather_kv_q8), which did 2 passes through HBM:
//   1) key_cache_q8[block_table]  -> temp [B, n_blocks, bs, Hkv, bytes]
//   2) permute + contiguous       -> [B, Hkv, Sk, bytes]
//
// Here we do the same in ONE pass: each workgroup handles one
// (seq_idx, kv_head, token_pos) triple; it reads
// block_table[seq_idx, token_pos/bs], then copies the K_q8 row and the
// V_fp16 row into the contiguous output BHSD.
//
// Additionally, V rows beyond seq_lens[seq_idx] are zeroed inline (a
// kernel requirement: the V "tail" must not contribute to softmax). The
// K tail garbage is irrelevant because the FA kernel cuts at kv_max.
//
// ---------------------------------------------------------------------------
// Parameterization:
//   Block(64, 1, 1) — one wavefront per (seq, head, tok).
//   Grid(num_seqs, Hkv, max_seqlen_k) — big, but each workgroup is light
//     (a copy of D*34/32 bytes for K + D*2 bytes for V, plus one int
//     block_table read).
//
// gfx906 optimizations (64 KB LDS/CU, 64-wide waves):
//   - byte copy via unsigned int (4 bytes per load/store) -> coalesced HBM.
//   - V copied as __half4 (8 bytes per thread) — 128-bit HBM bursts.
//   - block_table / seq_lens read by ONE thread per workgroup + broadcast
//     via shared memory.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

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
#include <cstdlib>
#include "kernel/q8_0_quantize.cuh"
#include "kernel/gfx906-config.h"

// M1 window-clip margin — defined once here (visible to every gather
// kernel in this file), shared with the FA config table via
// kernel/gfx906-config.h: fattn-q8.cuh's config macro static-asserts
// nbatch_fa <= this value, so a table edit that outgrows the margin
// fails to compile instead of silently under-covering the clip.
constexpr int GATHER_CLIP_MARGIN = GFX906_FA_GATHER_CLIP_MARGIN;

// ---------------------------------------------------------------------------
// Fused gather K_q8 + V_fp16 → contiguous BHSD.
//
// Shape contracts:
//   key_cache_q8  [num_blocks, block_size, Hkv, bytes_per_row]  uint8
//   value_cache   [num_blocks, block_size, Hkv, D]              fp16
//   block_table   [num_seqs, max_num_blocks]                    int32
//   seq_lens      [num_seqs]                                    int32
//   kv_start      [num_seqs]                                    int32 (may be NULL)
//   k_out         [num_seqs, Hkv, Sk, bytes_per_row]            uint8
//   v_out         [num_seqs, Hkv, Sk, D]                        fp16
//
// Sk = max_seqlen_k (the host rounds it up to a multiple of 32).
//
// M1 window clip (LOCKSTEP with the persistent kernel below): with
// non-null kv_start, per-seq tokens [0, start) are NOT written,
// start = max(0, kv_start[s] - GATHER_CLIP_MARGIN) clamped to
// min(seq_len, Sk). Rows are written at ABSOLUTE token indices, so the
// FA kernel's k-loop start (floor(kv_start, nbatch_fa)) indexes k_out/
// v_out directly; GATHER_CLIP_MARGIN (kernel/gfx906-config.h) guarantees
// the floored start is always materialized (its static_assert ties the
// FA config table to this margin). Tokens [0, start) are never read by
// the FA kernel — its window mask covers [0, kv_start) and the k-loop
// starts at floor(kv_start) >= start.
// ---------------------------------------------------------------------------
extern "C" __global__ void gather_paged_kv_q8_kernel(
    const uint8_t * __restrict__ key_cache_q8,
    const __half  * __restrict__ value_cache,
    const int32_t * __restrict__ block_table,
    const int32_t * __restrict__ seq_lens,
    const int32_t * __restrict__ kv_start,
    uint8_t       * __restrict__ k_out,
    __half        * __restrict__ v_out,
    int num_seqs,
    int num_kv_heads,
    int Sk,                       // max_seqlen_k (multiple of 32)
    int D,                        // head_size
    int bytes_per_row,            // (D/32) * 34
    int block_size,
    int max_blocks_per_seq,
    int64_t cache_block_stride,   // block_size * Hkv * bytes_per_row
    int64_t cache_token_stride,   // Hkv * bytes_per_row
    int64_t cache_head_stride_q8, // bytes_per_row (K)
    int64_t v_cache_block_stride, // block_size * Hkv * D
    int64_t v_cache_token_stride, // Hkv * D
    int64_t v_cache_head_stride   // D
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tok_pos  = blockIdx.z;
    if (seq_idx >= num_seqs || head_idx >= num_kv_heads || tok_pos >= Sk) return;

    const int lane = threadIdx.x;   // 0..63

    // Read seq_len ONCE per workgroup (lane 0), broadcast via __shfl.
    int seq_len = 0;
    if (lane == 0) seq_len = seq_lens[seq_idx];
    seq_len = __shfl(seq_len, 0, 64);

    // M1 window clip (LOCKSTEP with the persistent kernel below): tokens
    // below the per-seq clip start are not written (FA never reads them —
    // see the kernel doc). The stale prefix stays as the buffer's
    // previous content: finite (previous layer's rows), never visited.
    int clip_start = 0;
    if (kv_start) {
        int kv0 = 0;
        if (lane == 0) kv0 = kv_start[seq_idx];
        kv0 = __shfl(kv0, 0, 64);
        if (kv0 > 0) {
            const int sl = seq_len < Sk ? seq_len : Sk;
            clip_start = kv0 - GATHER_CLIP_MARGIN;
            if (clip_start < 0) clip_start = 0;
            if (clip_start > sl) clip_start = sl;
        }
    }
    if (tok_pos < clip_start) return;

    // Same for block_table[seq, tok_pos / block_size].
    const int block_tab_idx = tok_pos / block_size;
    const int block_offset  = tok_pos % block_size;

    int64_t v_dst_base =
        ((int64_t)seq_idx * num_kv_heads + head_idx) * (int64_t)Sk * D
        + (int64_t)tok_pos * D;
    int64_t k_dst_base =
        ((int64_t)seq_idx * num_kv_heads + head_idx) * (int64_t)Sk * bytes_per_row
        + (int64_t)tok_pos * bytes_per_row;

    // Out-of-range (or block_table shorter than Sk — guard, same as V2)
    // -> zero V (leave K alone; the FA kernel cuts it).
    if (tok_pos >= seq_len || block_tab_idx >= max_blocks_per_seq) {
        __half * vdst = v_out + v_dst_base;
        for (int i = lane; i < D; i += 64) {
            vdst[i] = __float2half(0.0f);
        }
        return;
    }

    // Valid token -> read block_table[seq, block_tab_idx].
    int phys_block = 0;
    if (lane == 0) {
        phys_block = block_table[seq_idx * max_blocks_per_seq + block_tab_idx];
    }
    phys_block = __shfl(phys_block, 0, 64);

    // ---------- K copy (uint8, bytes_per_row) ----------
    const uint8_t * k_src =
        key_cache_q8
        + (int64_t)phys_block   * cache_block_stride
        + (int64_t)block_offset * cache_token_stride
        + (int64_t)head_idx     * cache_head_stride_q8;
    uint8_t * k_dst = k_out + k_dst_base;

    // bytes_per_row = (D/32)*34. For D=128 that is 136 bytes. Copy via
    // uint32_t where possible, the tail byte by byte.
    const int n_u32 = bytes_per_row >> 2;          // whole 4-byte chunks
    const int tail_start = n_u32 << 2;             // tail, in bytes
    const uint32_t * k_src_u32 = reinterpret_cast<const uint32_t *>(k_src);
    uint32_t       * k_dst_u32 = reinterpret_cast<uint32_t       *>(k_dst);
    for (int i = lane; i < n_u32; i += 64) {
        k_dst_u32[i] = k_src_u32[i];
    }
    // tail (0..3 bytes). For D%32==0 and 34*(D/32) ->
    // bytes_per_row % 4 in {0, 2}: (D/32)*34 mod 4 = (D/32)*2 mod 4 ->
    // D=64 -> 68 -> 0; D=128 -> 136 -> 0. Fine. Handle the tail via lane 0
    // anyway, just in case.
    if (lane == 0) {
        for (int i = tail_start; i < bytes_per_row; ++i) {
            k_dst[i] = k_src[i];
        }
    }

    // ---------- V copy (fp16 × D) ----------
    const __half * v_src =
        value_cache
        + (int64_t)phys_block   * v_cache_block_stride
        + (int64_t)block_offset * v_cache_token_stride
        + (int64_t)head_idx     * v_cache_head_stride;
    __half * vdst = v_out + v_dst_base;

    // Copy via uint2 (8 bytes = 4 x fp16) when D is 4-aligned.
    // For D=128: 32 iterations of 4 fp16 -> 2 bursts per lane over 64
    // threads.
    const int n_u2 = D >> 2;                       // D/4 chunks of 8 bytes
    const uint2 * v_src_u2 = reinterpret_cast<const uint2 *>(v_src);
    uint2       * v_dst_u2 = reinterpret_cast<uint2       *>(vdst);
    for (int i = lane; i < n_u2; i += 64) {
        v_dst_u2[i] = v_src_u2[i];
    }
    // tail: D % 4. For D=128 and D=64 this is 0 -> could be skipped, but
    // keep it.
    const int v_tail = D & 3;
    if (v_tail != 0 && lane == 0) {
        for (int i = D - v_tail; i < D; ++i) vdst[i] = v_src[i];
    }
}

// ---------------------------------------------------------------------------
// V2: Paged-block-coalesced gather (1 workgroup = 1 paged block, 16 tokens).
//
// Motivation (rocprof_decode.py at 60K, batch=4):
//   The per-token kernel: grid(4, 8, 61440) = ~2M workgroups, 64 threads.
//   Effective HBM BW ~330 GB/s of the MI50's ~1 TB/s peak (33% utilization).
//   Main cause: launch overhead from 2M wavefronts + small 392 bytes/wg
//   transfers without full burst fill.
//
// V2 approach:
//   * 1 workgroup serves ONE (seq, head, paged-block) triple
//   * block_size=16 tokens at once -> 16x(136+256)=6272 bytes per wg
//   * 128 threads/wg -> ~49 bytes/thread = 2-3 uint4 each
//   * block_table is read once per block (instead of once per token)
//
// Layout contract — identical to V1 (so the env switch is safe):
//   src:  [num_blocks, block_size, Hkv, bytes_per_row | D]
//   dst:  [num_seqs, Hkv, Sk, bytes_per_row | D]
//   Sk is a multiple of 32 (host rounds it).
//
// M1 window clip (LOCKSTEP with the persistent kernel): with non-null
// kv_start, per-seq tokens [0, start) are not written, start = max(0,
// kv_start[s] - GATHER_CLIP_MARGIN) clamped to min(seq_len, Sk); the FA
// k-loop starts at floor(kv_start) >= start (absolute-position dst
// layout — see the V1 kernel doc above).
//
// bytes_per_row = (D/32)*34 for q8_0 (D=128 -> 136 bytes). Not a multiple
// of 16, so for K we do 8xuint4 (128 bytes) + a 1xuint2 (8 bytes) tail.
// For V (D*2 bytes, D%4==0) it is fully 16-aligned -> pure uint4 loads.
// ---------------------------------------------------------------------------
extern "C" __global__ void gather_paged_kv_q8_kernel_v2(
    const uint8_t * __restrict__ key_cache_q8,
    const __half  * __restrict__ value_cache,
    const int32_t * __restrict__ block_table,
    const int32_t * __restrict__ seq_lens,
    const int32_t * __restrict__ kv_start,
    uint8_t       * __restrict__ k_out,
    __half        * __restrict__ v_out,
    int num_seqs,
    int num_kv_heads,
    int Sk,
    int D,
    int bytes_per_row,
    int block_size,
    int max_blocks_per_seq,
    int64_t cache_block_stride,
    int64_t cache_token_stride,
    int64_t cache_head_stride_q8,
    int64_t v_cache_block_stride,
    int64_t v_cache_token_stride,
    int64_t v_cache_head_stride
) {
    const int seq_idx        = blockIdx.x;
    const int head_idx       = blockIdx.y;
    const int paged_block    = blockIdx.z;              // 0..ceil(Sk/block_size)-1
    const int block_start_tok = paged_block * block_size;
    if (seq_idx >= num_seqs || head_idx >= num_kv_heads || block_start_tok >= Sk) return;

    const int tid = threadIdx.x;
    const int nth = blockDim.x;   // 128

    // ---------- Read phys_block + seq_len ONCE per workgroup ----------
    __shared__ int s_phys_block;
    __shared__ int s_seq_len;
    __shared__ int s_clip_start;
    if (tid == 0) {
        s_seq_len = seq_lens[seq_idx];
        s_clip_start = kv_start ? kv_start[seq_idx] : 0;
        const int block_tab_idx = block_start_tok / block_size;
        s_phys_block = (block_tab_idx < max_blocks_per_seq)
            ? block_table[seq_idx * max_blocks_per_seq + block_tab_idx]
            : -1;
    }
    __syncthreads();
    const int phys_block = s_phys_block;
    const int seq_len    = s_seq_len;

    // M1 window clip (LOCKSTEP with the persistent kernel below): tokens
    // below the per-seq clip start are not written (FA never reads them —
    // see the V1 kernel doc).
    const int sl = seq_len < Sk ? seq_len : Sk;
    int clip_start = 0;
    if (s_clip_start > 0) {
        clip_start = s_clip_start - GATHER_CLIP_MARGIN;
        if (clip_start < 0) clip_start = 0;
        if (clip_start > sl) clip_start = sl;
    }
    // Entire paged block inside the clipped prefix: no work.
    if (block_start_tok + block_size <= clip_start) return;

    // Source base pointers (for a valid phys_block).
    const uint8_t * k_src_base_bh = (phys_block >= 0)
        ? key_cache_q8
          + (int64_t)phys_block * cache_block_stride
          + (int64_t)head_idx   * cache_head_stride_q8
        : nullptr;
    const __half  * v_src_base_bh = (phys_block >= 0)
        ? value_cache
          + (int64_t)phys_block   * v_cache_block_stride
          + (int64_t)head_idx     * v_cache_head_stride
        : nullptr;

    // dst base offsets: [seq, head, tok, 0]. Fixed per workgroup.
    const int64_t dst_K_sh_base =
        ((int64_t)seq_idx * num_kv_heads + head_idx) * (int64_t)Sk * bytes_per_row;
    const int64_t dst_V_sh_base =
        ((int64_t)seq_idx * num_kv_heads + head_idx) * (int64_t)Sk * D;

    // Precompute the uint4/uint2 boundaries for K.
    // bytes_per_row may not be a multiple of 16. For D=128: 136 = 8*16 + 8.
    const int k_n_u4  = bytes_per_row >> 4;           // 8 for D=128
    const int k_tail  = bytes_per_row & 15;           // 8 for D=128
    const int k_tail_u2 = k_tail >> 3;                // 1 (when there are 8 bytes)
    const int k_tail_byte = k_tail & 7;               // remaining 0..7 bytes (usually 0)

    // V layout: D*2 bytes per token. D=128 -> 256 bytes = 16 x uint4. Pure
    // vectorized path.
    const int v_n_u4 = (D * (int)sizeof(__half)) >> 4;  // 16 for D=128

    // If no token in the paged block is valid — just zero V, leave K alone.
    const bool full_oob = (block_start_tok >= seq_len) || (phys_block < 0);

    // ---------- FLAT ITERATION: the WG's work is spread evenly ----------
    //
    // Main difference from the first V2 version: no inner `for t` loop.
    // Instead, all uint4 chunks for K and V over ALL 16 tokens of the block
    // are distributed across the 128 threads by a single global range-for.
    // This gives:
    //   * even load on all threads (earlier 8 of 128 did the work)
    //   * consecutive threads touch consecutive addresses (coalesced HBM)
    //   * less divergence at token boundaries
    //
    // Work size per workgroup:
    //   V: block_size x v_n_u4 = 16 x 16 = 256 uint4  (4096 bytes)
    //   K: block_size x k_n_u4 = 16 x 8  = 128 uint4  (2048 bytes)
    //   K tail: block_size x k_tail_u2 = 16 x 1 = 16 uint2 (128 bytes)
    //
    // Over 128 threads that is 2 uint4/thread for V and 1 uint4/thread for
    // K — excellent utilization and a short program at the same time.

    const int v_total_u4 = block_size * v_n_u4;       // 256
    const int k_total_u4 = block_size * k_n_u4;       // 128
    const int k_total_u2 = block_size * k_tail_u2;    // 16

    // --------- V pass: copy-or-zero per index ----------
    // idx -> (t, c) layout: consecutive threads in a wave land on adjacent
    // uint4 chunks WITHIN one token (v_n_u4=16 is >= the 16 used here, even
    // though gfx906 prefers 64; a 64-wide wave covers 4 tokens, which is
    // fine).
    for (int idx = tid; idx < v_total_u4; idx += nth) {
        const int t = idx / v_n_u4;
        const int c = idx - t * v_n_u4;
        const int tok_global = block_start_tok + t;
        if (tok_global >= Sk) continue;
        if (tok_global < clip_start) continue;   // M1 clip: no write

        const bool tok_valid = !full_oob && (tok_global < seq_len);
        uint4 val;
        if (tok_valid) {
            const __half * v_src_tok = v_src_base_bh + (int64_t)t * v_cache_token_stride;
            val = reinterpret_cast<const uint4 *>(v_src_tok)[c];
        } else {
            val = make_uint4(0u, 0u, 0u, 0u);
        }
        __half * v_dst_tok = v_out + dst_V_sh_base + (int64_t)tok_global * D;
        reinterpret_cast<uint4 *>(v_dst_tok)[c] = val;
    }

    // --------- K pass (uint4 body): copy only for tok_valid ----------
    // The K tail (8 of the 136 bytes) is handled separately below.
    if (!full_oob) {
        for (int idx = tid; idx < k_total_u4; idx += nth) {
            const int t = idx / k_n_u4;
            const int c = idx - t * k_n_u4;
            const int tok_global = block_start_tok + t;
            if (tok_global >= Sk) continue;
            if (tok_global < clip_start) continue;  // M1 clip: leave stale
            if (tok_global >= seq_len) continue;  // out-of-range — leave K

            const uint8_t * k_src_tok = k_src_base_bh + (int64_t)t * cache_token_stride;
            uint8_t       * k_dst_tok = k_out + dst_K_sh_base + (int64_t)tok_global * bytes_per_row;
            reinterpret_cast<uint4 *>(k_dst_tok)[c] =
                reinterpret_cast<const uint4 *>(k_src_tok)[c];
        }

        // --------- K tail uint2 (8 bytes) — one per token for D=128 ----------
        if (k_tail_u2 > 0) {
            for (int idx = tid; idx < k_total_u2; idx += nth) {
                const int t = idx / k_tail_u2;
                const int c = idx - t * k_tail_u2;
                const int tok_global = block_start_tok + t;
                if (tok_global >= Sk) continue;
                if (tok_global < clip_start) continue;  // M1 clip: leave stale
                if (tok_global >= seq_len) continue;

                const uint8_t * k_src_tail = k_src_base_bh
                    + (int64_t)t * cache_token_stride + (int64_t)(k_n_u4 << 4);
                uint8_t * k_dst_tail = k_out + dst_K_sh_base
                    + (int64_t)tok_global * bytes_per_row + (int64_t)(k_n_u4 << 4);
                reinterpret_cast<uint2 *>(k_dst_tail)[c] =
                    reinterpret_cast<const uint2 *>(k_src_tail)[c];
            }
        }

        // --------- K byte tail (0..7 bytes; 0 for D=128) — cold path ----------
        if (k_tail_byte > 0) {
            for (int idx = tid; idx < block_size * k_tail_byte; idx += nth) {
                const int t = idx / k_tail_byte;
                const int c = idx - t * k_tail_byte;
                const int tok_global = block_start_tok + t;
                if (tok_global >= Sk) continue;
                if (tok_global < clip_start) continue;  // M1 clip: leave stale
                if (tok_global >= seq_len) continue;
                const int base = (k_n_u4 << 4) + (k_tail_u2 << 3);
                const uint8_t * k_src_tok = k_src_base_bh + (int64_t)t * cache_token_stride;
                uint8_t * k_dst_tok = k_out + dst_K_sh_base + (int64_t)tok_global * bytes_per_row;
                k_dst_tok[base + c] = k_src_tok[base + c];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused gather-and-quantize for the LEGACY (inline-quant) decode path.
//
// Motivation (P3-3a stage 2): the legacy path ran two kernels per FA
// layer at decode:
//   1) gather_paged_kv_fp16  — K fp16 + V fp16 from paged blocks
//   2) quantize_q8_0_dense   — K fp16 -> K q8_0 (second read of K)
// At B=1 both are latency/launch bound (effective HBM BW << peak), so
// fusing saves a full kernel launch per layer plus the K fp16 round
// trip. This kernel does in one pass:
//   * V: fp16 copy, identical semantics to V1 (tail zeroed inline)
//   * K: fp16 row read from the paged cache and quantized to q8_0 via
//     the SAME quantize_block_q8_0_halfwarp helper as
//     quantize_q8_0_dense_kernel — output is bit-equal to
//     quantize_q8_0(gather_paged_kv_fp16(x)) by construction.
// K tail (tok >= seq_len) is left unmasked, exactly like V1: the FA
// kernel cuts it via kv_max.
//
// Grid (num_seqs, num_kv_heads, Sk), block (64,1,1): one wavefront per
// (seq, head, tok). K quantization: halfwave 0 -> blocks 0,2,4,...;
// halfwave 1 -> blocks 1,3,5,... (stride 2), each thread loads one
// value per block (32 consecutive halfs per block -> coalesced).
//
// Constraint: Sk must fit in gridDim.z (<= 65535) — same as V1. The
// caller falls back to the two-kernel path beyond that.
// ---------------------------------------------------------------------------
extern "C" __global__ void gather_paged_kv_quant_kernel(
    const __half  * __restrict__ key_cache,   // fp16 [num_blocks, bs, Hkv, D]
    const __half  * __restrict__ value_cache, // fp16 [num_blocks, bs, Hkv, D]
    const int32_t * __restrict__ block_table, // [num_seqs, max_blocks_per_seq]
    const int32_t * __restrict__ seq_lens,    // [num_seqs]
    uint8_t       * __restrict__ k_q8_out,    // [num_seqs, Hkv, Sk, bytes_per_row]
    __half        * __restrict__ v_out,       // [num_seqs, Hkv, Sk, D]
    int num_seqs,
    int num_kv_heads,
    int Sk,
    int D,
    int bytes_per_row,            // (D/32) * 34
    int block_size,
    int max_blocks_per_seq,
    int64_t k_block_stride,       // block_size * Hkv * D (elements)
    int64_t k_token_stride,       // Hkv * D
    int64_t k_head_stride,        // D
    int64_t v_block_stride,
    int64_t v_token_stride,
    int64_t v_head_stride
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tok_pos  = blockIdx.z;
    if (seq_idx >= num_seqs || head_idx >= num_kv_heads || tok_pos >= Sk) return;

    const int lane = threadIdx.x;   // 0..63

    int seq_len = 0;
    if (lane == 0) seq_len = seq_lens[seq_idx];
    seq_len = __shfl(seq_len, 0, 64);

    const int block_tab_idx = tok_pos / block_size;
    const int block_offset  = tok_pos % block_size;

    int64_t v_dst_base =
        ((int64_t)seq_idx * num_kv_heads + head_idx) * (int64_t)Sk * D
        + (int64_t)tok_pos * D;
    int64_t k_dst_base =
        ((int64_t)seq_idx * num_kv_heads + head_idx) * (int64_t)Sk * bytes_per_row
        + (int64_t)tok_pos * bytes_per_row;

    // Tail token: zero V, leave K (FA kernel cuts it via kv_max).
    if (tok_pos >= seq_len || block_tab_idx >= max_blocks_per_seq) {
        __half * vdst = v_out + v_dst_base;
        for (int i = lane; i < D; i += 64) {
            vdst[i] = __float2half(0.0f);
        }
        return;
    }

    int phys_block = 0;
    if (lane == 0) {
        phys_block = block_table[seq_idx * max_blocks_per_seq + block_tab_idx];
    }
    phys_block = __shfl(phys_block, 0, 64);

    // ---------- K: gather + quantize to q8_0 (planar row) ----------
    // Row layout (plan_fa_part_A.md): [quants D bytes | scale (D/32) fp16].
    const __half * k_src =
        key_cache
        + (int64_t)phys_block   * k_block_stride
        + (int64_t)block_offset * k_token_stride
        + (int64_t)head_idx     * k_head_stride;
    uint8_t * k_dst = k_q8_out + k_dst_base;

    const int half_id = lane / 32;   // 0 or 1
    const int lane_in = lane % 32;   // 0..31
    const int blocks_per_row = D / QK8_0_SZ;
    for (int b0 = 0; b0 < blocks_per_row; b0 += 2) {
        const int b = b0 + half_id;
        if (b < blocks_per_row) {
            quantize_block_q8_0_halfwarp_planar(
                k_src + b * QK8_0_SZ,
                k_dst + b * QK8_0_SZ,
                (uint16_t *)(k_dst + D) + b,
                lane_in
            );
        }
    }

    // ---------- V: fp16 copy (V1 semantics) ----------
    const __half * v_src =
        value_cache
        + (int64_t)phys_block   * v_block_stride
        + (int64_t)block_offset * v_token_stride
        + (int64_t)head_idx     * v_head_stride;
    __half * vdst = v_out + v_dst_base;

    const int n_u2 = D >> 2;                       // D/4 chunks of 8 bytes
    const uint2 * v_src_u2 = reinterpret_cast<const uint2 *>(v_src);
    uint2       * v_dst_u2 = reinterpret_cast<uint2       *>(vdst);
    for (int i = lane; i < n_u2; i += 64) {
        v_dst_u2[i] = v_src_u2[i];
    }
    const int v_tail = D & 3;
    if (v_tail != 0 && lane == 0) {
        for (int i = D - v_tail; i < D; ++i) vdst[i] = v_src[i];
    }
}

// ---------------------------------------------------------------------------
// Persistent grid-stride fused gather + quantize (LEGACY decode path).
//
// Replaces the two-kernel fallback that runs at Sk > 65535
// (gather_paged_kv_q8_kernel_v2 + quantize_q8_0_dense_kernel) and — when
// selected — the per-token fused gather_paged_kv_quant_kernel. The grid is
// a fixed capture-time constant (not a function of Sk, batch, or any live
// value), so the kernel is legal under CUDA-graph capture at any
// max_model_len; the amount of work is bounded by the live seq_lens tensor
// read inside the kernel — the same live-bounding pattern the FA kernel's
// KV_max uses. Per (seq, head, token) row with token < seq_len:
//   * K: fp16 row from the paged cache quantized to q8_0 in-kernel via the
//     same quantize_block_q8_0_halfwarp helper — bit-equal to
//     quantize_q8_0(gather_paged_kv_fp16(x))
//   * V: fp16 byte copy
// Rows [seq_len, Sk) are NOT written: the FA kernel's oob_check tail tile
// zero-fills out-of-range K/V rows in LDS without a global read, so it
// never reads buffer rows at or beyond kv_max (gated by the NaN-tail
// test, plan_masked_fa.md step 2). As a defensive headroom until that gate
// lands, `margin_zeros` rows past seq_len get V-zeroed (one FA tail-tile
// width); K margin rows are left untouched as always.
//
// Work mapping: rows are grouped (seq, head, token) with token fastest;
// per-seq counts (Hkv × (seq_len + margin)) are prefix-summed in registers
// (num_seqs <= 16). Each 64-thread workgroup grid-strides the flat row
// space; all lanes compute the same (uniform) mapping per row.
//
// M1 gather-path window clip: with non-null kv_start, per-seq rows
// [0, start) are skipped, start = max(0, kv_start[s] - GATHER_CLIP_MARGIN)
// (clamped to seq_len). The FA kernel (fattn-q8.cuh) floors its clip
// start to an nbatch_fa tile boundary — GATHER_CLIP_MARGIN (defined once
// in kernel/gfx906-config.h as GFX906_FA_GATHER_CLIP_MARGIN, shared with
// fattn-q8.cuh's config-table static_assert) guarantees the floored start
// is always materialized: a config-table edit that raises nbatch_fa past
// this value fails to compile in fattn-q8.cuh instead of silently
// under-covering the margin here. Rows are written at ABSOLUTE token
// indices, so the FA k-loop start (floor(kv_start)) indexes this buffer
// directly — no compaction, no reindexing.
// ---------------------------------------------------------------------------
extern "C" __global__ void gather_paged_kv_quant_persistent_kernel(
    const __half  * __restrict__ key_cache,
    const __half  * __restrict__ value_cache,
    const int32_t * __restrict__ block_table,
    const int32_t * __restrict__ seq_lens,
    const int32_t * __restrict__ kv_start,
    uint8_t       * __restrict__ k_q8_out,
    __half        * __restrict__ v_out,
    int num_seqs,
    int num_kv_heads,
    int Sk,
    int D,
    int bytes_per_row,
    int block_size,
    int max_blocks_per_seq,
    int margin_zeros,
    int64_t k_block_stride,
    int64_t k_token_stride,
    int64_t k_head_stride,
    int64_t v_block_stride,
    int64_t v_token_stride,
    int64_t v_head_stride
) {
    const int lane    = threadIdx.x;   // 0..63
    const int half_id = lane >> 5;     // 0 or 1
    const int lane_in = lane & 31;     // 0..31

    const int B   = num_seqs;
    const int Hkv = num_kv_heads;

    // Per-seq row counts (uniform across the wavefront; B is small).
    //   gstart[s] = clip start (see kernel doc; 0 without kv_start)
    //   rph[s] = (min(seq_len[s], Sk) - gstart[s]) + margin, margin =
    //            min(margin_zeros, Sk - min(seq_len[s], Sk))
    //   cnt[s] = Hkv * rph[s]
    int rph[16];
    int gstart[16];
    int total = 0;
    #pragma unroll 4
    for (int s = 0; s < B; ++s) {
        int sl = seq_lens[s];
        if (sl > Sk) sl = Sk;
        int st = 0;
        if (kv_start && kv_start[s] > 0) {
            st = kv_start[s] - GATHER_CLIP_MARGIN;
            if (st < 0) st = 0;
            if (st > sl) st = sl;
        }
        gstart[s] = st;
        int extra = margin_zeros;
        if (extra > Sk - sl) extra = Sk - sl;
        rph[s] = (sl - st) + extra;
        total += Hkv * rph[s];
    }

    const int blocks_per_row = D / QK8_0_SZ;
    const int v_n_u4 = (D * (int)sizeof(__half)) >> 4;

    const int G = gridDim.x;
    for (int row = blockIdx.x; row < total; row += G) {
        // row -> (seq, head, token)
        int rem = row;
        int seq_idx = B - 1;
        #pragma unroll 4
        for (int s = 0; s < B; ++s) {
            if (rem < Hkv * rph[s]) { seq_idx = s; break; }
            rem -= Hkv * rph[s];
        }
        const int rows_per_head = rph[seq_idx];
        const int head_idx = rem / rows_per_head;
        const int tok      = gstart[seq_idx] + rem - head_idx * rows_per_head;

        int sl = seq_lens[seq_idx];
        if (sl > Sk) sl = Sk;

        const int64_t v_dst_base =
            ((int64_t)seq_idx * Hkv + head_idx) * (int64_t)Sk * D
            + (int64_t)tok * D;

        // Margin row (tok >= seq_len): V-zero only; K untouched (the FA
        // kernel cuts at kv_max = seq_len and never reads these).
        if (tok >= sl) {
            if (lane < v_n_u4) {
                reinterpret_cast<uint4 *>(v_out + v_dst_base)[lane] =
                    make_uint4(0u, 0u, 0u, 0u);
            }
            continue;
        }

        const int block_tab_idx = tok / block_size;
        const int block_offset  = tok - block_tab_idx * block_size;

        int phys_block = 0;
        if (lane == 0) {
            phys_block = (block_tab_idx < max_blocks_per_seq)
                ? block_table[seq_idx * max_blocks_per_seq + block_tab_idx]
                : -1;
        }
        phys_block = __shfl(phys_block, 0, 64);
        if (phys_block < 0) continue;

        const __half * k_src = key_cache
            + (int64_t)phys_block   * k_block_stride
            + (int64_t)block_offset * k_token_stride
            + (int64_t)head_idx     * k_head_stride;
        const __half * v_src = value_cache
            + (int64_t)phys_block   * v_block_stride
            + (int64_t)block_offset * v_token_stride
            + (int64_t)head_idx     * v_head_stride;
        uint8_t * k_dst = k_q8_out
            + (((int64_t)seq_idx * Hkv + head_idx) * (int64_t)Sk + tok)
              * bytes_per_row;

        // K: fp16 -> q8_0 planar row (bit-equal to the dense/fused paths;
        // each 32-value block is quantized independently, so row
        // partitioning across workgroups does not change the result).
        for (int b0 = 0; b0 < blocks_per_row; b0 += 2) {
            const int b = b0 + half_id;
            if (b < blocks_per_row) {
                quantize_block_q8_0_halfwarp_planar(
                    k_src + b * QK8_0_SZ,
                    k_dst + b * QK8_0_SZ,
                    (uint16_t *)(k_dst + D) + b,
                    lane_in
                );
            }
        }

        // V: byte copy, 16 B per lane (D=256 -> 32 lanes, one uint4 each).
        if (lane < v_n_u4) {
            reinterpret_cast<uint4 *>(v_out + v_dst_base)[lane] =
                reinterpret_cast<const uint4 *>(v_src)[lane];
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
extern "C" hipError_t launch_gather_paged_kv_quant_persistent(
    const __half  * key_cache,
    const __half  * value_cache,
    const int32_t * block_table,
    const int32_t * seq_lens,
    const int32_t * kv_start,
    uint8_t       * k_q8_out,
    __half        * v_out,
    int num_seqs,
    int num_kv_heads,
    int Sk,
    int D,
    int bytes_per_row,
    int block_size,
    int max_blocks_per_seq,
    int margin_zeros,
    int grid,
    int64_t k_block_stride,
    int64_t k_token_stride,
    int64_t k_head_stride,
    int64_t v_block_stride,
    int64_t v_token_stride,
    int64_t v_head_stride,
    hipStream_t stream
) {
    if (num_seqs == 0 || num_kv_heads == 0 || Sk == 0) return hipSuccess;
    if (D % QK8_0_SZ != 0) return hipErrorInvalidValue;
    if (num_seqs > 16) return hipErrorInvalidValue;
    if (grid <= 0) grid = 1024;

    dim3 block(64, 1, 1);
    dim3 grid_d(grid, 1, 1);
   hipLaunchKernelGGL(( gather_paged_kv_quant_persistent_kernel), dim3(grid_d), dim3(block), 0, stream, 
        key_cache, value_cache,
        block_table, seq_lens, kv_start,
        k_q8_out, v_out,
        num_seqs, num_kv_heads, Sk, D, bytes_per_row, block_size,
        max_blocks_per_seq, margin_zeros,
        k_block_stride, k_token_stride, k_head_stride,
        v_block_stride, v_token_stride, v_head_stride
    );
    return hipGetLastError();
}

extern "C" hipError_t launch_gather_paged_kv_quant(
    const __half  * key_cache,
    const __half  * value_cache,
    const int32_t * block_table,
    const int32_t * seq_lens,
    uint8_t       * k_q8_out,
    __half        * v_out,
    int num_seqs,
    int num_kv_heads,
    int Sk,
    int D,
    int bytes_per_row,
    int block_size,
    int max_blocks_per_seq,
    int64_t k_block_stride,
    int64_t k_token_stride,
    int64_t k_head_stride,
    int64_t v_block_stride,
    int64_t v_token_stride,
    int64_t v_head_stride,
    hipStream_t stream
) {
    if (num_seqs == 0 || num_kv_heads == 0 || Sk == 0) return hipSuccess;
    if (D % 32 != 0) return hipErrorInvalidValue;
    if (Sk > 65535) return hipErrorInvalidValue;   // gridDim.z cap (V1 same)

    dim3 block(64, 1, 1);
    dim3 grid(num_seqs, num_kv_heads, Sk);
    gather_paged_kv_quant_kernel<<<grid, block, 0, stream>>>(
        key_cache, value_cache,
        block_table, seq_lens,
        k_q8_out, v_out,
        num_seqs, num_kv_heads, Sk, D, bytes_per_row, block_size,
        max_blocks_per_seq,
        k_block_stride, k_token_stride, k_head_stride,
        v_block_stride, v_token_stride, v_head_stride
    );
    return hipGetLastError();
}

extern "C" hipError_t launch_gather_paged_kv_q8(
    const uint8_t * key_cache_q8,
    const __half  * value_cache,
    const int32_t * block_table,
    const int32_t * seq_lens,
    const int32_t * kv_start,   // M1 clip (LOCKSTEP), may be NULL
    uint8_t       * k_out,
    __half        * v_out,
    int num_seqs,
    int num_kv_heads,
    int Sk,
    int D,
    int bytes_per_row,
    int block_size,
    int max_blocks_per_seq,
    int64_t cache_block_stride,
    int64_t cache_token_stride,
    int64_t cache_head_stride_q8,
    int64_t v_cache_block_stride,
    int64_t v_cache_token_stride,
    int64_t v_cache_head_stride,
    hipStream_t stream
) {
    if (num_seqs == 0 || num_kv_heads == 0 || Sk == 0) return hipSuccess;
    if (D % 32 != 0) return hipErrorInvalidValue;

    // Level 3c-step-A: GFX906_FA_GATHER_V selects the kernel variant.
    // Default is V1 (per-token, grid (B, Hkv, Sk), 64 threads): in serving
    // (FULL decode graph, D=256, Sk=3328) V1 is 15% faster than V2 (56.9 vs
    // 49.6 t/s e2e) — V2 (416 WG + __syncthreads) degrades in the serving
    // context (285 us/call vs 41 us isolated). V2 stays available via
    // GFX906_FA_GATHER_V=2.
    // The env var is read once (thread-safe: amortized over all calls).
    static int cached_version = -1;
    if (cached_version < 0) {
        const char * env = getenv("GFX906_FA_GATHER_V");
        cached_version = (env && env[0] == '2') ? 2 : 1;
    }
    // HIP's gridDim.z is capped at 65535 and V1 puts Sk directly in
    // gridDim.z: for very long contexts (max_model_len ~ 65-70K) switch to
    // V2 (gridDim.z = ceil(Sk/block_size) — safe).
    if (cached_version == 1 && Sk > 65535) cached_version = 2;

    if (cached_version == 1) {
        dim3 block(64, 1, 1);
        dim3 grid(num_seqs, num_kv_heads, Sk);
        gather_paged_kv_q8_kernel<<<grid, block, 0, stream>>>(
            key_cache_q8, value_cache,
            block_table, seq_lens, kv_start,
            k_out, v_out,
            num_seqs, num_kv_heads, Sk, D, bytes_per_row, block_size,
            max_blocks_per_seq,
            cache_block_stride, cache_token_stride, cache_head_stride_q8,
            v_cache_block_stride, v_cache_token_stride, v_cache_head_stride
        );
    } else {
        // V2: 1 wg = 1 paged block, 128 threads, grid reduced by block_size.
        const int n_paged_blocks = (Sk + block_size - 1) / block_size;
        dim3 block(128, 1, 1);
        dim3 grid(num_seqs, num_kv_heads, n_paged_blocks);
        gather_paged_kv_q8_kernel_v2<<<grid, block, 0, stream>>>(
            key_cache_q8, value_cache,
            block_table, seq_lens, kv_start,
            k_out, v_out,
            num_seqs, num_kv_heads, Sk, D, bytes_per_row, block_size,
            max_blocks_per_seq,
            cache_block_stride, cache_token_stride, cache_head_stride_q8,
            v_cache_block_stride, v_cache_token_stride, v_cache_head_stride
        );
    }
    return hipGetLastError();
}
