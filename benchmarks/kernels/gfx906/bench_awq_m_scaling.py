#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""M-scaling of the gfx906 AWQ GEMM kernel (gptq_gemm / q_gemm).

The gfx906 auto_awq path dispatches to ops.gptq_qgemm (the q_gemm
kernel, per-file max-ilp tuned for M=1). Spec-decode draft steps run
every quantized projection at M=4 instead of M=1 — this measures the
M=1 -> M=4 cost delta at the dense 27B's shapes to size the skinny-M
lever. Kernel-level (direct op call), no model load.
"""
import torch

from vllm import _custom_ops as ops

PACK = 8  # 4-bit
GROUP = 128

# (N, K, name) — dense 27B: MLP 5120<->17408, GDN in_proj_qkvz 5120->8192
SHAPES = [
    (17408, 5120, "down_proj"),
    (5120, 17408, "gate_proj"),
    (8192, 5120, "gdn_in_proj_qkvz"),
    (3072, 5120, "fa_q_proj"),
]
MS = (1, 2, 4, 8)


def bench(fn, iters=30):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    dev = "cuda"
    torch.manual_seed(0)
    print(f"{'shape':22s} " + " ".join(f"M={m:<8d}" for m in MS))
    for n, k, name in SHAPES:
        qweight = torch.randint(-2**31, 2**31, (k // PACK, n),
                                device=dev, dtype=torch.int32)
        scales = torch.rand(k // GROUP, n, device=dev,
                            dtype=torch.float16) * 0.01 + 0.001
        qzeros = torch.randint(0, 16, (k // GROUP, n // PACK),
                               device=dev, dtype=torch.int32)
        row = []
        for m in MS:
            x = torch.randn(m, k, device=dev, dtype=torch.float16)
            g_idx = torch.empty(0, device=dev)
            ms = bench(lambda: ops.gptq_gemm(
                x, qweight, qzeros, scales, g_idx, True, True, 4))
            row.append(ms)
        print(f"{name} {n}x{k:<8d} " +
              " ".join(f"{ms*1000:<10.0f}" for ms in row) + "  (us)")


if __name__ == "__main__":
    main()
