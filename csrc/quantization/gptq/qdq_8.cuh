/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _qdq_8_cuh
#define _qdq_8_cuh

#include "qdq_util.cuh"

namespace vllm {
namespace gptq {

__forceinline__ __device__ void shuffle_8bit_4(uint32_t* q, int stride) {}

__forceinline__ __device__ void dequant_8bit_8(const uint32_t q_0,
                                               const uint32_t q_1,
                                               half2 (&dq)[4], int stride,
                                               const uint32_t zero) {
  constexpr uint32_t ext10 = 0x07010700;
  constexpr uint32_t ext32 = 0x07030702;
  constexpr uint32_t c0 = 0x64006400;

  half2 z2 = __hadd2(*reinterpret_cast<const half2*>(&c0),
                     __half2half2(__int2half_rn(zero)));

  half2_uint32 qa(__builtin_amdgcn_perm(c0, q_0, ext10));
  half2_uint32 qb(__builtin_amdgcn_perm(c0, q_0, ext32));
  half2_uint32 qc(__builtin_amdgcn_perm(c0, q_1, ext10));
  half2_uint32 qd(__builtin_amdgcn_perm(c0, q_1, ext32));

  dq[0] = __hsub2(qa.as_half2, z2);
  dq[1] = __hsub2(qb.as_half2, z2);
  dq[2] = __hsub2(qc.as_half2, z2);
  dq[3] = __hsub2(qd.as_half2, z2);
}

}  // namespace gptq
}  // namespace vllm

#endif