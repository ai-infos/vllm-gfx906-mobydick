#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""C1: routing pipeline structural probe (topk_softmax + moe_align_block_size).

Answers the roadmap's precondition — "is the ~1 ms/step routing cost
structural (graph-node floor) or kernel-work-bound" — before any
model-path work. Model shapes: E=256, topk=8 (Qwen3.5-35B-A3B).

Per M in {1, 8, 32, 128} (decode block_size_m 1/4/4/8 per the gfx906
expert's em buckets):
  eager   : per-call us for topk alone / align alone / the pair
            (200 iters, cuda events — launches pipeline).
  graph   : 40-layer chain (80 nodes: topk+align x40) captured, replay
            us per replay; plus a 80-node dummy floor graph (tiny
            add_ ops) for the per-node replay overhead.

Plus the S2 check: the dedicated M=1 topk kernel
(`torch.ops._rocm_C.moe_topk_softmax_m1_gfx906`, VLLM_GFX906_TOPK_M1,
shipped default-OFF after a NEUTRAL serving verdict) measured in the
SAME 40-node graph regime — if it hits the same per-node latency as
the generic kernel, S2's failure is structurally explained.

Interpretation: if graph(80 routing nodes) ~= graph(80 dummy nodes),
the cost is the node floor and fusion (80 -> 40 nodes) is the lever.
Usage:
  HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
      .venv/bin/python benchmarks/kernels/gfx906/c1_routing_structural_probe.py
"""
import torch
import torch.cuda as tc

from vllm import _custom_ops as ops

E, TOPK, LAYERS, ITERS = 256, 8, 40, 200
BSM = {1: 1, 8: 4, 32: 4, 128: 8}   # em=M*topk buckets in the gfx906 expert


def time_iters(fn, iters=ITERS):
    for _ in range(10):
        fn()
    tc.synchronize()
    s = tc.Event(enable_timing=True)
    e = tc.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    tc.synchronize()
    return s.elapsed_time(e) * 1e3 / iters  # us/iter


def graph_replay_us(fn, iters=ITERS):
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    s = tc.Event(enable_timing=True)
    e = tc.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


def make_inputs(m, dev):
    # production gate output is fp16 (both the generic topkGating and the
    # S2 M=1 kernel dispatch on the input dtype)
    logits = torch.randn(m, E, device=dev, dtype=torch.float16) * 0.1
    tw = torch.empty(m, TOPK, device=dev, dtype=torch.float32)
    ti = torch.empty(m, TOPK, device=dev, dtype=torch.int32)
    tei = torch.empty(m, TOPK, device=dev, dtype=torch.int32)
    pad = m * TOPK + E * BSM[m] - BSM[m]
    sti = torch.empty(pad, device=dev, dtype=torch.int32)
    eids = torch.empty(pad // BSM[m], device=dev, dtype=torch.int32)
    ntp = torch.empty(E, device=dev, dtype=torch.int32)
    return (logits, tw, ti, tei, sti, eids, ntp)


def main():
    dev = "cuda"
    torch.manual_seed(20260829)
    print("C1 routing structural probe (E=256, topk=8, "
          f"{LAYERS} layers = 80 nodes)", flush=True)
    for m in (1, 8, 32, 128):
        (logits, tw, ti, tei, sti, eids, ntp) = make_inputs(m, dev)
        bsm = BSM[m]

        def topk():
            ops.topk_softmax(tw, ti, tei, logits, False)

        def align():
            ops.moe_align_block_size(ti, E, bsm, sti, eids, ntp, None)

        def pair():
            topk()
            align()

        def chain40():
            for _ in range(LAYERS):
                pair()

        t_topk = time_iters(topk)
        t_align = time_iters(align)
        t_pair = time_iters(pair)

        x = torch.zeros(128, device=dev)

        def dummy80():
            for _ in range(2 * LAYERS):
                x.add_(0.0)

        g_chain = graph_replay_us(chain40)
        g_dummy = graph_replay_us(dummy80)
        g_topk = graph_replay_us(
            lambda: [topk() for _ in range(LAYERS)])
        g_align = graph_replay_us(
            lambda: [align() for _ in range(LAYERS)])
        g_align_m1 = None
        if m == 1 and hasattr(torch.ops._rocm_C,
                              "moe_align_block_size_m1_gfx906"):
            # wrapper-convention sizes for M=1/bsm=1: 8 / 8 / 1
            s8 = torch.empty(8, device=dev, dtype=torch.int32)
            e8 = torch.empty(8, device=dev, dtype=torch.int32)
            n1 = torch.empty(1, device=dev, dtype=torch.int32)
            g_align_m1 = graph_replay_us(
                lambda: [
                    torch.ops._rocm_C.moe_align_block_size_m1_gfx906(
                        ti, E, 1, s8, e8, n1)
                    for _ in range(LAYERS)])
        g_rf = None
        if m == 1 and hasattr(torch.ops._rocm_C,
                              "moe_routing_fused_m1_gfx906"):
            s8 = torch.empty(8, device=dev, dtype=torch.int32)
            e8 = torch.empty(8, device=dev, dtype=torch.int32)
            n1 = torch.empty(1, device=dev, dtype=torch.int32)
            tw1 = torch.empty(1, TOPK, device=dev, dtype=torch.float32)
            ti1 = torch.empty(1, TOPK, device=dev, dtype=torch.int32)
            tei1 = torch.empty(1, TOPK, device=dev, dtype=torch.int32)
            g_rf = graph_replay_us(
                lambda: [
                    torch.ops._rocm_C.moe_routing_fused_m1_gfx906(
                        logits, tw1, ti1, tei1, s8, e8, n1, True)
                    for _ in range(LAYERS)])

        am1 = (f" | align_m1 {g_align_m1:7.1f} us "
               f"({g_align_m1 / 40:.1f} us/node)"
               if g_align_m1 is not None else "")
        rf = (f" | routing_fused {g_rf:7.1f} us "
              f"({g_rf / 40:.1f} us/node)"
              if g_rf is not None else "")
        print(f"M={m:4d} bsm={bsm}: eager topk {t_topk:6.1f} | "
              f"align {t_align:6.1f} | pair {t_pair:6.1f} us "
              f"| graph40: routing {g_chain:7.1f} us "
              f"({g_chain / 80:.1f} us/node) | topk-only "
              f"{g_topk:7.1f} | align-only {g_align:7.1f} | "
              f"dummy80 {g_dummy:7.1f} us ({g_dummy / 80:.1f} us/node)"
              f"{am1}{rf}",
              flush=True)
        if m == 1 and hasattr(torch.ops._rocm_C,
                              "moe_topk_softmax_m1_gfx906"):
            s2 = torch.ops._rocm_C.moe_topk_softmax_m1_gfx906

            def topk_m1():
                s2(tw, ti, tei, logits, False)

            t_m1 = time_iters(topk_m1)
            g_m1 = graph_replay_us(
                lambda: [topk_m1() for _ in range(LAYERS)])
            print(f"      S2-check: M=1 dedicated topk eager {t_m1:6.1f} us | "
                  f"graph40 {g_m1:7.1f} us ({g_m1 / 40:.1f} us/node) "
                  f"vs generic {g_topk:7.1f} us ({g_topk / 40:.1f} us/node)",
                  flush=True)
        del logits, tw, ti, tei, sti, eids, ntp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
