#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Per-step GEMM census for spec decode (dense 27B, gfx906).

Wraps the three GEMM families the 27B uses per decode step —
AWQ gptq_gemm, the fp16 triton_matmul (rocm_unquantized_gemm
fallback) and the M=1 GEMV ops — and, in an enforce_eager
single-prompt run, counts calls per decode step by (N, K) shape.

Output: for each step, the GEMM census; then a per-step-type
(draft vs no-draft, inferred from step token count via the GDN
spy's fused_seq/packed split) aggregated shape histogram.

This sizes the L2 (AWQ M=4) target: are the M=4 steps making the
same GEMM calls as M=1 (per-call cost up) or different calls
(more/split launches)?

Same pitfalls as gdn_step_spy.py: enforce_eager, single prompt,
os._exit(0).
"""
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import collections

import torch

MODEL = "/data/models/qwen/Qwen3.5-27B-AWQ"


def main():
    spec = "--spec" in sys.argv
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as g
    import vllm._custom_ops as ops
    import vllm.model_executor.layers.utils as lu

    recs = []          # (kind, n, k, step, capturing)
    conv_count = [0]

    def tag(kind, n, k):
        recs.append((kind, n, k, conv_count[0] // 48,
                     torch.cuda.is_current_stream_capturing()))

    # AWQ path (gfx906 auto_awq -> ops.gptq_gemm)
    real_gptq = ops.gptq_gemm

    def gptq_spy(a, b, *rest):
        tag("gptq", b.shape[1] if b.dim() == 2 else -1, a.shape[-1])
        return real_gptq(a, b, *rest)

    # fp16 path: rocm_unquantized_gemm -> triton_matmul (n>1)
    real_tm = lu.triton_matmul

    def tm_spy(a, b):
        tag("triton", b.shape[0], b.shape[1])
        return real_tm(a, b)

    # GDN conv anchor (one per GDN layer per step, single-seq)
    real_conv = g.causal_conv1d_update

    def conv_spy(x, *a, **kw):
        conv_count[0] += 1
        return real_conv(x, *a, **kw)

    # GDN path markers (draft vs no-draft step classification)
    real_packed = g.fused_recurrent_gated_delta_rule_packed_decode

    def packed_spy(mixed_qkv=None, **kw):
        tag("packed", 0, 0)
        return real_packed(mixed_qkv, **kw)

    real_fused = g.fused_sigmoid_gating_delta_rule_update

    def fused_spy(A_log=None, a=None, b=None, dt_bias=None, q=None, k=None,
                  v=None, cu_seqlens=None, ssm_state_indices=None,
                  num_accepted_tokens=None, **kw):
        tag("fused_seq", 0, 0)
        return real_fused(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k,
                          v=v, cu_seqlens=cu_seqlens,
                          ssm_state_indices=ssm_state_indices,
                          num_accepted_tokens=num_accepted_tokens, **kw)

    ops.gptq_gemm = gptq_spy
    lu.triton_matmul = tm_spy
    g.causal_conv1d_update = conv_spy
    g.fused_recurrent_gated_delta_rule_packed_decode = packed_spy
    g.fused_sigmoid_gating_delta_rule_update = fused_spy

    from vllm import LLM, SamplingParams

    common = dict(
        model=MODEL, max_num_seqs=4, max_model_len=2816,
        gpu_memory_utilization=0.95, dtype="float16",
        enforce_eager=True)
    if spec:
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
    recs.clear()
    conv_count[0] = 0
    out = llm.generate([p1], SamplingParams(max_tokens=128, temperature=0))

    # per-step census
    steps = collections.defaultdict(collections.Counter)
    fused_steps = set()
    packed_steps = set()
    for kind, n, k, step, cap in recs:
        if kind == "fused_seq":
            fused_steps.add(step)
        elif kind == "packed":
            packed_steps.add(step)
        elif kind in ("gptq", "triton"):
            steps[step][(kind, n, k)] += 1

    n_tok = len(out[0].outputs[0].token_ids)
    print(f"ARM={'spec' if spec else 'nospec'} committed={n_tok}",
          flush=True)
    # aggregate: draft steps (fused_seq seen) vs no-draft (packed)
    agg = collections.defaultdict(collections.Counter)
    n_draft = n_nodraft = 0
    for step, census in sorted(steps.items()):
        stype = "draft" if step in fused_steps else "nodraft"
        if stype == "draft":
            n_draft += 1
        else:
            n_nodraft += 1
        for shape, cnt in census.items():
            agg[stype][shape] += cnt
    for stype in ("nodraft", "draft"):
        if stype not in agg:
            continue
        n = n_nodraft if stype == "nodraft" else n_draft
        total = sum(agg[stype].values())
        print(f"== {stype} steps: {n}, total GEMM calls {total} "
              f"({total/n:.0f}/step)", flush=True)
        for (kind, nn, kk), cnt in sorted(agg[stype].items(),
                                          key=lambda kv: -kv[1]):
            print(f"   {kind:7s} N={nn:<6d} K={kk:<6d} x{cnt/n:7.1f}/step"
                  f"  (total {cnt})", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
