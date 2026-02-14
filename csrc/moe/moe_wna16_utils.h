
#include <cuda_fp16.h>

template <typename scalar_t>
class ScalarType {};

template <>
class ScalarType<half> {
 public:
  using scalar_t = half;
  using scalar_t2 = half2;

  static __device__ float inline num2float(const half x) {
    return __half2float(x);
  }

  static __device__ half2 inline num2num2(const half x) {
    return __half2half2(x);
  }

  static __device__ half2 inline nums2num2(const half x1, const half x2) {
    return __halves2half2(x1, x2);
  }

  static __host__ __device__ half inline float2num(const float x) {
    return __float2half(x);
  }

  static __host__ __device__ half inline int2num(const float x) {
    return __int2half_rn(x);
  }

  static __host__ __device__ float2 inline num22float2(const half2 x) {
    return __half22float2(x);
  }

  static __host__ __device__ half2 inline float22num2(const float2 x) {
    return __float22half2_rn(x);
  }
};

// TODO: support 8
// template <int start_byte, int mask>
// __device__ inline uint32_t prmt(uint32_t a) {
//   uint32_t res;
//   asm volatile("prmt.b32 %0, %1, %2, %3;\n"
//                : "=r"(res)
//                : "r"(a), "n"(start_byte), "n"(mask));
//   return res;
// }

__device__ __forceinline__ void atomicAdd_half(half* address, half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    __half_raw hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    half tmpres = __hadd(hsum, val);
    hsum = __half_raw(tmpres);
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}

__device__ __forceinline__ uint32_t bfi(const uint32_t S0, const uint32_t S1,
                                        const uint32_t S2) {
#if defined(USE_ROCM)
  uint32_t result;
  __asm__ (
    "  v_bfi_b32  %0, %1, %2, %3  \n"
    : "=v" (result)
    : "v"(S0), "v"(S1), "v"(S2)
  );
  return result;
#else
  return (S0 & S1) | (~S0 & S2);
#endif
}

template <typename scalar_t2, int bit>
__device__ inline void dequant(int q, scalar_t2* res) {}

template <>
__device__ inline void dequant<half2, 4>(int q, half2* res) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd400d400;

  int lo0 = bfi(LO, q, EX);
  int hi0 = bfi(HI, q, EX);
  q >>= 8;
  int lo1 = bfi(LO, q, EX);
  int hi1 = bfi(HI, q, EX);

  res[0] = __hsub2(*reinterpret_cast<half2*>(&lo0),
                   *reinterpret_cast<const half2*>(&SUB));
  res[1] = __hfma2(*reinterpret_cast<half2*>(&hi0),
                   *reinterpret_cast<const half2*>(&MUL),
                   *reinterpret_cast<const half2*>(&ADD));
  res[2] = __hsub2(*reinterpret_cast<half2*>(&lo1),
                   *reinterpret_cast<const half2*>(&SUB));
  res[3] = __hfma2(*reinterpret_cast<half2*>(&hi1),
                   *reinterpret_cast<const half2*>(&MUL),
                   *reinterpret_cast<const half2*>(&ADD));
}

// TODO: support 8
// template <>
// __device__ inline void dequant<half2, 8>(int q, half2* res) {
//   static constexpr uint32_t mask_for_elt_01 = 0x5250;
//   static constexpr uint32_t mask_for_elt_23 = 0x5351;
//   static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

//   uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
//   uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

//   static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;

//   res[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
//                    *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
//   res[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
//                    *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
// }