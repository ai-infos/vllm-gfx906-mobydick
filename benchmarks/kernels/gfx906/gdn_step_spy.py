#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Per-step GDN kernel-path attribution for spec decode (dense 27B).

Wraps the GDN leaf kernel entry points (packed decode, fused
sequential, chunk, MTP op) and the conv update, runs a single-prompt
128-token generation, and prints:
  - one row per decode step: which path each of the 48 GDN layers took
  - per-path totals

Usage: python gdn_step_spy.py [--spec]

Pitfalls baked in here (see DEVLOG-spec-decode.md):
  - MUST run with enforce_eager (set below): CUDA-graph replay skips
    Python, so under graphs the spies only see eager/capture steps.
  - Single prompt: on mixed-batch steps conv runs twice per layer
    (spec + non-spec branches), which breaks the conv//48 step
    anchor. One prompt keeps it exact (one conv per layer per step).
  - os._exit(0) at the end: the vLLM teardown path hits a known
    heap-corruption abort (exit 134) that would eat the report.
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
    from vllm import _custom_ops as ops

    recs = []          # (kind, info)
    conv_count = [0]   # 48 per decode step -> step index

    def tag(kind, info=""):
        capturing = torch.cuda.is_current_stream_capturing()
        recs.append((kind, info, conv_count[0] // 48, capturing))

    # 1. packed decode (non-spec fast path)
    real_packed = g.fused_recurrent_gated_delta_rule_packed_decode

    def packed_spy(mixed_qkv=None, **kw):
        tag("packed", f"B={mixed_qkv.shape[0]}")
        return real_packed(mixed_qkv, **kw)

    # 2. fused sequential (spec 2.1 / decode 2.2-2.3)
    real_fused = g.fused_sigmoid_gating_delta_rule_update

    def fused_spy(A_log=None, a=None, b=None, dt_bias=None, q=None, k=None,
                  v=None, cu_seqlens=None, ssm_state_indices=None,
                  num_accepted_tokens=None, **kw):
        info = f"q={tuple(q.shape)}"
        if cu_seqlens is not None and not torch.cuda.is_current_stream_capturing():
            try:
                info += " cu=" + str(cu_seqlens.tolist())
            except Exception:
                pass
        if ssm_state_indices is not None:
            info += f" idx={tuple(ssm_state_indices.shape)}"
        if num_accepted_tokens is not None:
            info += " natt=1"
        tag("fused_seq", info)
        return real_fused(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k,
                          v=v, cu_seqlens=cu_seqlens,
                          ssm_state_indices=ssm_state_indices,
                          num_accepted_tokens=num_accepted_tokens, **kw)

    # 3. chunk (prefill backend; also the B1 suspect)
    real_chunk = g.fla_chunk_gated_delta_rule

    def chunk_spy(*a, **kw):
        q = kw.get("q", None)
        if q is None and len(a) > 0:
            q = a[0]
        info = f"q={tuple(q.shape)}" if q is not None else ""
        if "cu_seqlens" in kw and kw["cu_seqlens"] is not None and not torch.cuda.is_current_stream_capturing():
            try:
                info += " cu=" + str(kw["cu_seqlens"].tolist())
            except Exception:
                pass
        tag("chunk", info)
        return real_chunk(*a, **kw)

    # 4. MTP fused op (disabled on gfx906: requires gdn_decode_kernel=cuda)
    real_mtp = ops.fused_gdn_decode_post_conv_mtp

    def mtp_spy(*a, **kw):
        tag("mtp", str(tuple(kw.get("cu_seqlens", ())).shape)
            if hasattr(kw.get("cu_seqlens", ()), "shape") else "")
        return real_mtp(*a, **kw)

    # 5. conv update (anchor: exactly one per GDN layer per decode step)
    real_conv = g.causal_conv1d_update

    def conv_spy(x, *a, **kw):
        conv_count[0] += 1
        tag("conv", f"x={tuple(x.shape)}")
        return real_conv(x, *a, **kw)

    g.fused_recurrent_gated_delta_rule_packed_decode = packed_spy
    g.fused_sigmoid_gating_delta_rule_update = fused_spy
    g.fla_chunk_gated_delta_rule = chunk_spy
    ops.fused_gdn_decode_post_conv_mtp = mtp_spy
    g.causal_conv1d_update = conv_spy

    from vllm import LLM, SamplingParams

    common = dict(
        model=MODEL, max_num_seqs=4, max_model_len=2816,
        gpu_memory_utilization=0.95, dtype="float16",
        enforce_eager=True)  # no cuda graphs: every step runs Python so
    # the spies see every step (capture-time routing == replay routing)
    if spec:
        common["speculative_config"] = {
            "method": "ngram", "num_speculative_tokens": 3,
            "prompt_lookup_min": 2, "prompt_lookup_max": 5}
    llm = LLM(**common)
    tok = llm.get_tokenizer()

    def chat_ids(content):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, enable_thinking=False)
        # this tokenizer's template is token-based: returns BatchEncoding
        if not isinstance(enc, str):
            return list(enc["input_ids"])
        return enc

    rep = ("Repeat the following sentence exactly 30 times, once per "
           "line, with no changes: the quick brown fox jumps over the "
           "lazy dog")
    p1 = chat_ids(rep)

    # Warmup (capture etc.), unmeasured
    llm.generate([p1], SamplingParams(max_tokens=16, temperature=0))
    recs.clear()
    conv_count[0] = 0
    sp = SamplingParams(max_tokens=128, temperature=0)
    # single prompt: every decode step is single-seq -> exactly 48 GDN
    # kernel records per step (one per GDN layer), so steps can be cut
    # directly on the record count.
    out = llm.generate([p1], sp)

    # Single-seq run: conv runs exactly once per GDN layer per step, so
    # conv_count//48 is the true step index for every record.
    steps = collections.defaultdict(list)
    for kind, info, step, capturing in recs:
        if kind in ("packed", "fused_seq", "chunk", "mtp"):
            steps[step].append((kind, info, capturing))
    n_tok = [len(o.outputs[0].token_ids) for o in out]
    print(f"ARM={'spec' if spec else 'nospec'} committed={n_tok}", flush=True)
    n_decode = 0
    for step in sorted(steps):
        kinds = [k for k, _, _ in steps[step]]
        c = collections.Counter(kinds)
        sample = steps[step][0]
        tag = ""
        if any(cap for _, _, cap in steps[step]):
            tag = " [capture]"
        if len(kinds) == 48:
            n_decode += 1
        print(f"step {step}: n={len(kinds)} {dict(c)}"
              f"  e.g. {sample[0]} {sample[1][:100]}{tag}", flush=True)
    print(f"uniform 48-steps: {n_decode}", flush=True)

    totals = collections.Counter()
    for kind, _, _, _ in recs:
        totals[kind] += 1
    print("TOTALS", dict(totals), flush=True)
    # skip the engine-teardown path (known heap-corruption on exit)
    os._exit(0)


if __name__ == "__main__":
    main()
