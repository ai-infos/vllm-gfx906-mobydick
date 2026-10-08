#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""In-engine per-step GPU wall probe (spec decode cost model calibration).

Eager single-prompt run. Conv-anchor step segmentation (48 GDN convs
per step, single-seq only); one CUDA event at each step boundary
(conv #48 of the step) — the delta between consecutive boundaries is
the step GPU wall (each boundary sits just before the step's tail:
the last FA layer + sampler, a consistent offset that cancels in the
delta).

Arms:
  (default)     nospec, M=1
  --spec        ngram min2 k3 (draft steps M=1..4, SPEC_GEMM=1)
  --spec --gemv0  same with VLLM_GFX906_SPEC_GEMM=0 (triton at M>=2)

Per-family per-step costs come from the microbenches (bench_fp16_m4,
bench_awq_m_scaling) — this probe measures the ENGINE walls they must
sum into (launch overhead + CPU + proposer + B3 = residual).

Pitfalls: enforce_eager (graph replay skips spies), single prompt,
os._exit(0).
"""
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import collections
import time

import torch

MODEL = "/data/models/qwen/Qwen3.5-27B-AWQ"
SPEC = "--spec" in sys.argv
GEMV0 = "--gemv0" in sys.argv
if GEMV0:
    os.environ["VLLM_GFX906_SPEC_GEMM"] = "0"
elif "VLLM_GFX906_SPEC_GEMM" not in os.environ:
    os.environ["VLLM_GFX906_SPEC_GEMM"] = "1"


def main():
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as g

    conv_count = [0]
    step_events = []
    gdn_kind = collections.defaultdict(set)   # step -> {"fused"/"packed"}
    ev_pool = [torch.cuda.Event(enable_timing=True) for _ in range(512)]
    ev_idx = [0]

    def next_ev():
        e = ev_pool[ev_idx[0]]
        ev_idx[0] = (ev_idx[0] + 1) % len(ev_pool)
        e.record()
        return e

    real_conv = g.causal_conv1d_update

    def conv_spy(x, *a, **kw):
        i = conv_count[0]
        out = real_conv(x, *a, **kw)
        step = i // 48
        if i % 48 == 47:
            step_events.append((step + 1, next_ev()))
        conv_count[0] = i + 1
        return out

    real_packed = g.fused_recurrent_gated_delta_rule_packed_decode

    def packed_spy(mixed_qkv=None, **kw):
        gdn_kind[conv_count[0] // 48].add("packed")
        return real_packed(mixed_qkv, **kw)

    real_fused = g.fused_sigmoid_gating_delta_rule_update

    def fused_spy(A_log=None, a=None, b=None, dt_bias=None, q=None, k=None,
                  v=None, cu_seqlens=None, ssm_state_indices=None,
                  num_accepted_tokens=None, **kw):
        gdn_kind[conv_count[0] // 48].add("fused")
        return real_fused(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k,
                          v=v, cu_seqlens=cu_seqlens,
                          ssm_state_indices=ssm_state_indices,
                          num_accepted_tokens=num_accepted_tokens, **kw)

    g.causal_conv1d_update = conv_spy
    g.fused_recurrent_gated_delta_rule_packed_decode = packed_spy
    g.fused_sigmoid_gating_delta_rule_update = fused_spy

    from vllm import LLM, SamplingParams

    common = dict(
        model=MODEL, max_num_seqs=4, max_model_len=2816,
        gpu_memory_utilization=0.95, dtype="float16",
        enforce_eager=True)
    if SPEC:
        common["speculative_config"] = {
            "method": "ngram", "num_speculative_tokens": 3,
            "prompt_lookup_min": 2, "prompt_lookup_max": 5}
    llm = LLM(**common)
    tok = llm.get_tokenizer()
    enc = tok.apply_chat_template(
        [{"role": "user", "content":
          "Repeat the following sentence exactly 30 times, once per "
          "line, with no changes: the quick brown fox jumps over the "
          "lazy dog"}],
        add_generation_prompt=True, enable_thinking=False)
    p1 = list(enc["input_ids"]) if not isinstance(enc, str) else enc

    llm.generate([p1], SamplingParams(max_tokens=16, temperature=0))
    conv_count[0] = 0
    step_events.clear()
    gdn_kind.clear()
    t0 = time.perf_counter()
    out = llm.generate([p1], SamplingParams(max_tokens=128, temperature=0))
    wall = time.perf_counter() - t0
    n_tok = len(out[0].outputs[0].token_ids)
    arm = "nospec" if not SPEC else ("spec-triton" if GEMV0 else "spec-m4")
    print(f"ARM={arm} committed={n_tok} wall={wall:.1f}s "
          f"({n_tok/wall:.1f} t/s eager)", flush=True)

    ev_by_step = dict(step_events)
    steps_sorted = sorted(ev_by_step)
    walls = collections.defaultdict(list)
    for i in range(1, len(steps_sorted)):
        s_prev, s = steps_sorted[i - 1], steps_sorted[i]
        if s - s_prev != 1:
            continue
        ms = ev_by_step[s_prev].elapsed_time(ev_by_step[s])
        # label L sits at the end of step L-1
        stype = ("draft" if "fused" in gdn_kind.get(s - 1, set())
                 else "nodraft")
        walls[stype].append(ms)

    for stype in ("nodraft", "draft"):
        ws = sorted(walls.get(stype, []))
        if not ws:
            continue
        q = len(ws) // 4
        print(f"{stype}: n={len(ws)} min={ws[0]:7.2f} "
              f"p25={ws[q]:7.2f} mean={sum(ws)/len(ws):7.2f} ms",
              flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
