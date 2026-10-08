// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) Nick — nick413@gmail.com
//
// Vendored from https://github.com/cassettesgoboom/gfx906-fa-vllm
// (FlashAttention-style custom attention backend for vLLM on AMD gfx906).
//
#pragma once

// GFX906 (Vega 20 / MI50) kernel configuration

#ifdef GGML_HIP_GFX906

// ============================================
// MMQ Kernel Configuration
// ============================================
#define GFX906_MMQ_NWARPS 2

// ============================================
// Q8 Cache Configuration
// ============================================
#define GFX906_KVQ_MOE_CACHE_ENABLED 0
// Layer-cycling: N cycles, slot size = TOTAL / N
#define GFX906_Q8_CACHE_TOTAL_SIZE      (128 * 1024 * 1024)  // Total cache size: 128MB
#define GFX906_Q8_CACHE_NUM_SLOTS       1                    // Number of cycles  
#define GFX906_Q8_CACHE_LAYERS_PER_SLOT 1                    // 1 layer per slot

// ============================================
// ROPE Optimization
// ============================================
#define GFX906_ROPE_ENABLED 1

// ============================================
// M1 gather-path window clip margin
// ============================================
// GATHER_CLIP_MARGIN (gfx906_fa_gather.cu): the persistent gather writes
// rows [kv_start - GATHER_CLIP_MARGIN, seq_len) so the FA kernel's
// tile-boundary floor (fattn-q8.cuh, k0_base -= k0_base % nbatch_fa) never
// floors past what was actually materialized. This is only correct if
// GATHER_CLIP_MARGIN >= every nbatch_fa the FA config table
// (GGML_CUDA_FATTN_TILE_CONFIG_CASE) can return — the floor can move the
// start left by at most nbatch_fa - 1. Defined once here (not in either
// consumer file) so a config-table edit that raises nbatch_fa past this
// value fails to compile in fattn-q8.cuh instead of silently
// under-covering the margin at runtime.
#define GFX906_FA_GATHER_CLIP_MARGIN 128

#endif // GGML_HIP_GFX906
