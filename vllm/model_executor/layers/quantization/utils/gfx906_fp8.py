# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 software decoding without native FP8 instructions."""

from vllm.triton_utils import tl, triton


@triton.jit
def decode_e4m3_to_fp16(bits, fnuz: tl.constexpr = False):
    bits = bits.to(tl.uint8, bitcast=True)
    sign = (bits & 0x80).to(tl.uint16) << 8
    exponent = ((bits & 0x78) >> 3).to(tl.uint16)
    mantissa = (bits & 0x07).to(tl.uint16)
    bias_delta: tl.constexpr = 7 if fnuz else 8
    normal_bits = sign | ((exponent + bias_delta) << 10) | (mantissa << 7)
    normal = normal_bits.to(tl.float16, bitcast=True)
    subnormal = mantissa.to(tl.float16) * (1.0 / (1024 if fnuz else 512))
    subnormal = tl.where(sign != 0, -subnormal, subnormal)
    value = tl.where(exponent == 0, subnormal, normal)
    is_nan = bits == 0x80 if fnuz else (bits & 0x7F) == 0x7F
    return tl.where(is_nan, float("nan"), value).to(tl.float16)
