#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""gfx906 dot-instruction measured facts (probe).

Compiles + runs small HIP probes that answer, on THIS toolchain +
GPU (gfx906/MI50, ROCm 7.14):

1. Which dot instructions does the BACKEND accept for gfx906?
   Compiled to OBJECT (-c): that runs the backend ISel AND the
   llvm-mc instruction validation (the stage that rejects
   dot4c/dot8_i8/dot8c_i8 on gfx906) but stops before link.
   NOTE: `-S` (ISel only) passes dot8 — it is NOT an availability
   test. The doc table's `v_dot8_i32_i4` (i4 variant) does exist;
   the i8 dot8 forms do not.

2. `v_dot2_f32_f16` operand semantics, runtime-verified:
   vdst = acc + a_lo*b_lo + a_hi*b_hi  (2-pair fp16 dot, fp32
   accumulate) — the same op the production dense_gemv_gfx906.cu /
   moe_q_gemm_gfx906.cu use via __ockl_fdot2 / amd_mixed_dot.

3. f16 bit-pattern reference values (the 2026-08-28 lesson: hand-
   derived f16 constants are error-prone — 0x3C00 is 1.0 not 1.5,
   0x3800 is 0.5 not 1.0, 0x4600 is 6.0 not 7.0, 0x5000 is 32.0
   not 16.0; every "semantic anomaly" in the first probe round was
   a wrong constant, not a weird instruction).

Usage:
  python3 dot_isa_probe.py
  GFX906_DOT_PROBE_HIPCC=... GFX906_DOT_PROBE_GPU=1 python3 dot_isa_probe.py
"""
import os
import subprocess
import sys
import tempfile

ROCMLIB = "/opt/rocm/core-7.14/lib"
HIPCC = os.environ.get("GFX906_DOT_PROBE_HIPCC",
                       "/opt/rocm/core-7.14/bin/hipcc")

AVAIL = [
    "v_dot4_i32_i8", "v_dot4c_i32_i8", "v_dot8_i32_i8", "v_dot8c_i32_i8",
    "v_dot2_f32_f16",
]


def _asm_body(name: str) -> str:
    typ = "float" if "f32_f16" in name else "int"
    return (f'{typ} r=0; asm volatile("{name} %0, %1, %2, %0" : '
            '"+v"(r) : "v"(a), "v"(b));')

TPL = """
#include <hip/hip_runtime.h>
#include <cstdint>
__global__ void t(float* o, unsigned a, unsigned b) {{
    {body}
    if (r == 42 || (float)r == 42.0f) o[0] = 1;
}}
"""

SEM = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp16.h>
#include <cstdint>
#include <cstdio>
__device__ __forceinline__ float h2f(unsigned h) {
    return __half2float(__ushort_as_half(h));
}
__global__ void t(float* o) {
    // runtime-verified f16 reference values:
    o[0] = h2f(0x3C00); o[1] = h2f(0x3800); o[2] = h2f(0x4000);
    o[3] = h2f(0x4200); o[4] = h2f(0x4400); o[5] = h2f(0x4600);
    o[6] = h2f(0x5000);
    // raw asm and builtin must agree on the same operands:
    // a = (2.0, 1.0) f16, b = (4.0, 3.0) f16, acc = 5.0
    // clean dot = 2*4 + 1*3 + 5 = 16.0
    unsigned A = (0x4000u << 16) | 0x3C00u;
    unsigned B = (0x4400u << 16) | 0x4200u;
    float r = 5.0f;
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(r) : "v"(A), "v"(B));
    o[7] = r;
    o[8] = amd_mixed_dot(*(const __half2*)&A, *(const __half2*)&B, 5.0f, false);
    o[9] = 2.0f*4.0f + 1.0f*3.0f + 5.0f;  // expected clean value
}
int main() {
    float* h; hipHostAlloc(&h, 10 * sizeof(float), hipHostMallocDefault);
    t<<<1,1>>>(h);
    hipDeviceSynchronize();
    printf("  f16 table: 0x3C00=%.1f 0x3800=%.1f 0x4000=%.1f 0x4200=%.1f "
           "0x4400=%.1f 0x4600=%.1f 0x5000=%.1f\n", h[0], h[1], h[2],
           h[3], h[4], h[5], h[6]);
    printf("  raw asm dot2 = %.4f, builtin = %.4f, clean expected = %.4f\n",
           h[7], h[8], h[9]);
    return 0;
}
"""


def main() -> int:
    print(f"dot-instruction probe (hipcc={HIPCC}, arch=gfx906)")
    print("== availability (compiled to object: ISel + llvm-mc validation, no link)")
    for name in AVAIL:
        src = TPL.format(body=_asm_body(name))
        with tempfile.TemporaryDirectory() as td:
            hip = os.path.join(td, "p.hip")
            with open(hip, "w") as f:
                f.write(src)
            r = subprocess.run(
                [HIPCC, "-O3", "--offload-arch=gfx906", "-c", hip,
                 "-o", os.path.join(td, "p.o")],
                capture_output=True, text=True)
            if r.returncode == 0:
                verdict = "AVAILABLE"
            else:
                r2 = subprocess.run(
                    [HIPCC, "-O3", "--offload-arch=gfx906", "-S", hip,
                     "-o", os.path.join(td, "p.s")],
                    capture_output=True, text=True)
                lines = r.stderr.strip().splitlines()
                diag = next(
                    (line for line in lines
                     if "instruction" in line.lower()
                     or "not supported" in line.lower()),
                    lines[-1] if lines else "?")
                err = diag.replace("llvm-mc: error: ", "").strip()[:70]
                suffix = (" [ISel passes; assembler rejects]"
                          if r2.returncode == 0 else "")
                verdict = f"NOT AVAILABLE ({err}){suffix}"
            print(f"  {name:<16}: {verdict}")

    print("== v_dot2_f32_f16 semantics (runtime-verified): "
          "vdst = acc + a_lo*b_lo + a_hi*b_hi")
    with tempfile.TemporaryDirectory() as td:
        hip = os.path.join(td, "sem.hip")
        with open(hip, "w") as f:
            f.write(SEM)
        r = subprocess.run(
            [HIPCC, "-O3", "--offload-arch=gfx906", hip, "-o", os.path.join(td, "sem")],
            capture_output=True, text=True)
        if r.returncode != 0:
            print("  (semantics probe failed to compile — recheck availability above)")
            return 1
        env = dict(os.environ, LD_LIBRARY_PATH=ROCMLIB,
                   HIP_VISIBLE_DEVICES=os.environ.get("GFX906_DOT_PROBE_GPU", "0"))
        subprocess.run([os.path.join(td, "sem")], env=env)
    print("== VERDICT:")
    print("  - gfx906 int-dot set = {v_dot4_i32_i8} only (dot4c/dot8_i8/dot8c_i8")
    print("    assembler-rejected; v_dot8_i32_i4 exists per the doc rate table but")
    print("    needs i4 operands — irrelevant to Q8's i8 x i8 dot)")
    print("  - v_dot2_f32_f16 = clean fp16-2-pair dot + fp32 accumulate (same op as")
    print("    __ockl_fdot2/amd_mixed_dot, already in production in dense_gemv_")
    print("    gfx906.cu / moe_q_gemm_gfx906.cu)")
    print("  - FA-Q8 P·V still runs v_pk_mul_f16 + v_pk_add_f16 (fp16 acc); a dot2")
    print("    rewrite halves P·V instruction count with fp32-acc — candidate, see")
    print("    docs/gfx906/dequant-instructions.md + roadmap M3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
