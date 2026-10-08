// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) Nick — nick413@gmail.com
//
// Vendored from https://github.com/cassettesgoboom/gfx906-fa-vllm
// (FlashAttention-style custom attention backend for vLLM on AMD gfx906).
//
// ggml_shim.cuh — минимальный shim для портирования gfx906 fattn-q8 kernel
// вне ggml ecosystem. Определяет все ggml-зависимости через стандартный HIP/C++.
//
// Этот файл обеспечивает чтобы fattn-q8.cuh (скопированный из llama.cpp-gfx906)
// собирался в контексте PyTorch HIP cpp_extension БЕЗ линковки с ggml.

#pragma once

// КРИТИЧНО: torch cpp_extension форсит -D__HIP_NO_HALF_OPERATORS__=1 и
// -D__HIP_NO_HALF_CONVERSIONS__=1 ПОСЛЕ наших флагов, что ломает fattn-q8.cuh
// (там используется aggregate-init `half2 z[N] = {{0.0f,0.0f}}`, оператор `*=`
// и неявные конверсии float→half). Снимаем defines перед включением hip_fp16.h.
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
#include <cstdio>
#include <cstdlib>
#include <cfloat>   // FLT_MAX
#include <cfloat>
#include <cmath>

// ============================================================================
// Platform defines
// ============================================================================
#ifndef GGML_USE_HIP
#define GGML_USE_HIP
#endif

#ifndef GGML_HIP_GFX906
#define GGML_HIP_GFX906
#endif

#ifndef FLASH_ATTN_AVAILABLE
#define FLASH_ATTN_AVAILABLE
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Architecture detection — для gfx906 ветки в ggml_cuda_dp4a
#if !defined(__gfx906__) && defined(__HIP_DEVICE_COMPILE__)
#define __gfx906__
#endif

// ============================================================================
// GGML assert/abort stubs
// ============================================================================
#define GGML_ASSERT(x) do { if (!(x)) { printf("GGML_ASSERT failed: %s at %s:%d\n", #x, __FILE__, __LINE__); abort(); } } while(0)
#define GGML_ABORT(msg) do { printf("GGML_ABORT: %s at %s:%d\n", (msg), __FILE__, __LINE__); abort(); } while(0)
#define GGML_UNUSED(x) (void)(x)
#define GGML_UNUSED_VARS(...) do { } while(0)
#define NO_DEVICE_CODE do { asm("trap;"); } while(0)

// GGML CUDA compute-capability stubs (для совместимости с проверками в fattn-q8.cuh)
#define GGML_CUDA_CC_IS_NVIDIA(cc) (false)

// ============================================================================
// ggml_half, ggml_half2 — HIP-native типы
// ============================================================================
typedef __half  ggml_half;
typedef __half2 ggml_half2;

// ============================================================================
// Q8_0 block структура (идентична ggml-common.h)
// ============================================================================
#ifndef QK8_0
#define QK8_0 32
#endif
#ifndef QK8_1
#define QK8_1 32
#endif
#ifndef QR8_0
#define QR8_0 1
#endif
#ifndef QI8_0
#define QI8_0 (QK8_0 / (4 * QR8_0))
#endif

typedef struct {
    ggml_half d;        // delta (scale)
    int8_t    qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");

// ============================================================================
// Copy granularity helper
// ============================================================================
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
    return 16; // gfx906 supports 128-bit loads
}

// ============================================================================
// ggml_cuda_memcpy_1 — unrolled coalesced copy
// ============================================================================
template<int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char  *) dst)[i] = ((const char  *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int   *) dst)[i] = ((const int   *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2  *) dst)[i] = ((const int2  *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4  *) dst)[i] = ((const int4  *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

// ============================================================================
// ggml_cuda_unroll — template recursive unroll
// ============================================================================
template <int n>
struct ggml_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_cuda_unroll<n - 1>{}(f, args...);
    }
};
template <>
struct ggml_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

// ============================================================================
// ggml_cuda_dp4a — INT8 dot-product-accumulate (gfx906: v_dot4_i32_i8)
// ============================================================================
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// ============================================================================
// get_alibi_slope — ALiBi positional bias formula
// ============================================================================
static __device__ __forceinline__ float get_alibi_slope(
    const float max_bias, const uint32_t h, const uint32_t n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;
    return powf(base, exph);
}

// ============================================================================
// quantize_f32_q8_0_block — квантизация 32 fp32 → block_q8_0
// ============================================================================
static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f;

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d  = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = __float2half(d);

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j] * id;
        y->qs[j] = (int8_t) roundf(x0);
    }
}

// ============================================================================
// make_half2 float→half overloads (ROCm стандартный make_half2 требует __half)
// ============================================================================
// Используем uppercase-namespace-хак: добавляем overload в глобальное пространство,
// ROCm's make_half2 — свободная inline функция, поэтому overload по ADL работает.
static __device__ __host__ __forceinline__ __half2 make_half2(float x, float y) {
    return __half2{__float2half(x), __float2half(y)};
}
static __device__ __host__ __forceinline__ __half2 make_half2(double x, double y) {
    return __half2{__float2half((float)x), __float2half((float)y)};
}

// ============================================================================
// Generic warp_reduce_sum / warp_reduce_max (non-gfx906 path)
// На gfx906 лучше использовать gfx906_warp_reduce_* (из gfx906-common.cuh),
// но fattn-q8.cuh вызывает безprefix версии. Делаем их алиасами.
// ============================================================================
// NB: шаблон должен быть определён ДО include "gfx906-common.cuh" если там
// используется, но в нашем случае gfx906-common определяет их как gfx906_*-prefix.
// Kernel использует безprefix warp_reduce_sum<W>, которого в ggml common.cuh
// — generic implementation. Повторяем её здесь.

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor(x, offset, width));
    }
    return x;
}

// ============================================================================
// Подключаем gfx906-common.cuh (warp reductions, DPP intrinsics)
// ============================================================================
#include "gfx906-common.cuh"
