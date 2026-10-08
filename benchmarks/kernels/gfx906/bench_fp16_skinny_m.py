#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""fp16 skinny-M GEMM dispatch bench for gfx906 (spec-decode L1).

Spec draft steps run the 27B's unquantized projections
(self_attn.q/k/v, linear_attn.in_proj_a/b, layer 0) at M=4. The
`rocm_unquantized_gemm` dispatcher sends 0 < n <= 16 to the Triton
skinny `triton_matmul`, which costs ~377 us/call in-engine at M=4.
This compares Triton vs rocBLAS (F.linear) vs the M=1 GEMV kernel
per shape to set the gfx906 dispatch for 1 < M <= 16.
"""
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.utils import (
    _gfx906_spec_gemv_m4,
    triton_matmul,
)

# (m=N_out, k=K_in, name)
SHAPES = [
    (3072, 5120, "fa_q"),
    (512, 5120, "fa_kv"),
    (128, 5120, "gdn_a_b"),
    (5120, 5120, "layer0_sq"),
    (17408, 5120, "layer0_down"),
    (5120, 17408, "layer0_gate"),
]
MS = (1, 2, 4, 8, 16)


def bench(fn, iters=30):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    v = s.elapsed_time(e) / iters * 1000  # us
    return v


def bench_maybe(fn, iters=30):
    """bench() for fns that may return None (gated kernel paths)."""
    if fn() is None:
        return None
    return bench(fn, iters)


def main():
    dev = "cuda"
    torch.manual_seed(0)
    header = f"{'shape':16s} {'path':8s} " + " ".join(f"M={m:<8d}" for m in MS)
    for m, k, name in SHAPES:
        w = torch.randn(m, k, device=dev, dtype=torch.float16) * 0.02
        for path, fn in (
            ("triton", lambda x: triton_matmul(x, w)),
            ("rocmblas", lambda x: F.linear(x, w)),
            ("gemv", lambda x: _gfx906_spec_gemv_m4(w, x)),
        ):
            row = []
            for mm in MS:
                x = torch.randn(mm, k, device=dev, dtype=torch.float16)
                if path == "gemv":
                    v = bench_maybe(lambda x=x: fn(x))
                else:
                    v = bench(lambda x=x: fn(x))
                row.append(v)
            cells = " ".join(
                f"{v:<10.0f}" if v is not None else f"{'n/a':<10s}"
                for v in row)
            print(f"{name:16s} {path:8s} " + cells + "  (us)", flush=True)
        # sanity: same result? (triton vs rocmblas, and the gated gemv
        # path vs rocmblas at M=2, 8, 16 when enabled)
        x = torch.randn(4, k, device=dev, dtype=torch.float16)
        a = triton_matmul(x, w)
        b = F.linear(x, w)
        print(f"{name:16s} {'diff':8s} max|a-b|={
            (a - b).abs().max().item():.5f}", flush=True)
        for mm in (2, 8, 16):
            x = torch.randn(mm, k, device=dev, dtype=torch.float16) * 0.5
            g = _gfx906_spec_gemv_m4(w, x)
            if g is not None:
                d = (g - F.linear(x, w)).abs().max().item()
                print(f"{name:16s} {'diff':8s} gemv M={mm}: max|g-b|={d:.5f}",
                      flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
