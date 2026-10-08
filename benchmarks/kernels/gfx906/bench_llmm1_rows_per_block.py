#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""P3-2(a) probe: LLGemm1 (LLMM1) rows_per_block sweep at decode M=1 shapes.

The aiter probe (plan §4 P3-2a) is structurally a no-op on gfx906 — all
aiter gemm paths in rocm_unquantized_gemm_impl are gated behind
`not on_gfx906()`, wvSplitK is excluded (no matrix cores on Vega20), and
VLLM_ROCM_USE_AITER defaults to False. The one exposed knob on the actual
decode surface (our LLGemm1_kernel, csrc/rocm/skinny_gemms.cu) is
rows_per_block; dispatch hardcodes 4. This bench measures RPB in
{2,4,8,16} for every LLGemm1 shape in the model at M=1, to scope P3-2(b):
which rows carry slack vs the 798 GB/s floor, and whether a config change
alone moves the needle (>=20% on any shape would continue the probe).

Shapes are (weight rows M, K, layers/step, name) from the P3-0 Q3 in-proc
Linear probe (230 LLGemm1 calls/step + LM head inside the 5.83 ms row).

Run in the gfx906 vLLM image with the repo source-mounted:
  python3 -u /bench/bench_llmm1_rows_per_block.py
"""
import torch

dev = "cuda"
torch.manual_seed(0)

HBM_BW = 798e9  # P3-0 Q1: measured MI50 HBM read BW
RPB_LIST = (2, 4, 8, 16)
SHAPES = [
    (12288, 2048, 30, "GDN in_proj"),
    (248320, 2048, 1, "LM head"),
    (9216, 2048, 10, "FA qkv"),
    (2048, 4096, 40, "o_proj"),
    (1024, 2048, 40, "shared gate_up"),
    (2048, 512, 40, "shared down"),
    (256, 2048, 40, "router"),
    (64, 2048, 30, "GDN small"),
]


def time_us(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters  # us/call


def main():
    from vllm import _custom_ops as ops

    print("LLGemm1 (ops.LLMM1) M=1 sweep, rows_per_block in", RPB_LIST)
    print(f"{'shape':>16} {'name':<15} {'x/step':>6} "
          + " ".join(f"rpb={r:<6}" for r in RPB_LIST)
          + f" {'floor':>8} {'best rpb':>9}")

    step_total = {r: 0.0 for r in RPB_LIST}
    for M, K, layers, name in SHAPES:
        w = torch.randn(M, K, dtype=torch.float16, device=dev)
        x = torch.randn(1, K, dtype=torch.float16, device=dev)
        ref = (x.float() @ w.float().T).to(torch.float16)

        row = []
        for rpb in RPB_LIST:
            assert M % rpb == 0
            out = ops.LLMM1(w, x, rpb)  # [1, M]
            maxdiff = (out.float() - ref.float()).abs().max().item()
            assert maxdiff < 2e-1, f"rpb={rpb} {name}: maxdiff {maxdiff}"
            iters = 200 if M * K <= 4 * 1024 * 1024 else 50
            us = time_us(lambda: ops.LLMM1(w, x, rpb),
                         warmup=20, iters=iters)
            row.append(us)
            step_total[rpb] += us * layers
            del out
        floor = M * K * 2 / HBM_BW * 1e6
        best_rpb = RPB_LIST[int(row.index(min(row)))]
        best = min(row)
        print(f"{M}x{K:<9} {name:<15} {layers:>6} "
              + " ".join(f"{u:<9.1f}" for u in row)
              + f" {floor:>7.1f} {best_rpb:>5} "
              f"(x{best/floor:.1f} floor)")
        del w, x, ref

    print("\nweighted per-step total (sum over 230 calls + LM head):")
    for r in RPB_LIST:
        tag = "  <- current dispatch" if r == 4 else ""
        print(f"  rpb={r:<3} {step_total[r]:>8.1f} us/step{tag}")
    base = step_total[4]
    best_r = min(step_total, key=step_total.get)
    print(f"\nbest rpb={best_r}: {step_total[best_r]:.1f} us/step vs "
          f"current rpb=4 {base:.1f} us/step "
          f"({100*(base-step_total[best_r])/base:+.1f}%)")


if __name__ == "__main__":
    main()
