#!/usr/bin/env python3
# Copyright Kevin Read <me@kevin-read.com>
"""Shape-level GEMM spy for MTP k=2 eager runs: fp16 dispatcher + AWQ
gptq_gemm + big F.linear. Attributes the drafter's GEMMs and the
reconstruct-fallback shapes."""
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import collections

def main():
    import torch
    import vllm.model_executor.layers.utils as lu
    import vllm._custom_ops as ops

    disp = collections.Counter()
    awq = collections.Counter()
    lin = collections.Counter()
    real = lu.rocm_unquantized_gemm
    real_g = ops.gptq_gemm
    real_linear = torch.nn.functional.linear

    def spy(layer, x, weight, bias=None):
        n = x.numel() // x.size(-1)
        m, k = weight.shape
        disp[f"n={n} m={m} k={k} bias={bias is not None}"] += 1
        return real(layer, x, weight, bias)

    def g_spy(a, b_q, qz, sc, b_g, ue, uv, bit):
        M, K = a.shape
        N = b_q.shape[-1]
        awq[f"M={M} N={N} K={K}"] += 1
        return real_g(a, b_q, qz, sc, b_g, ue, uv, bit)

    def lin_spy(x, w, b=None):
        n = x.numel() // x.size(-1)
        m, k = w.shape
        lin[f"n={n} m={m} k={k}"] += 1
        return real_linear(x, w, b)

    from vllm import LLM, SamplingParams
    llm = LLM(
        model="/data/models/qwen/Qwen3.5-27B-AWQ",
        max_num_seqs=4, max_model_len=2816,
        gpu_memory_utilization=0.95, dtype="float16",
        enforce_eager=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 2})
    tok = llm.get_tokenizer()
    enc = tok.apply_chat_template(
        [{"role": "user", "content":
          "Repeat the following sentence exactly 30 times, once per "
          "line, with no changes: the quick brown fox jumps over the "
          "lazy dog"}],
        add_generation_prompt=True, enable_thinking=False)
    p1 = list(enc["input_ids"]) if not isinstance(enc, str) else enc

    lu.rocm_unquantized_gemm = spy
    ops.gptq_gemm = g_spy
    torch.nn.functional.linear = lin_spy
    llm.generate([p1], SamplingParams(max_tokens=16, temperature=0))
    for c in (disp, awq, lin):
        c.clear()
    out = llm.generate([p1], SamplingParams(max_tokens=128, temperature=0))
    n_tok = len(out[0].outputs[0].token_ids)
    print(f"OUT={n_tok}", flush=True)
    print("=== fp16 dispatcher:", flush=True)
    for key, c in sorted(disp.items(), key=lambda kv: -kv[1]):
        print(f"  {c:6d}  {key}", flush=True)
    print("=== AWQ gptq_gemm:", flush=True)
    for key, c in sorted(awq.items(), key=lambda kv: -kv[1]):
        print(f"  {c:6d}  {key}", flush=True)
    print("=== F.linear:", flush=True)
    for key, c in sorted(lin.items(), key=lambda kv: -kv[1]):
        print(f"  {c:6d}  {key}", flush=True)
    os._exit(0)

if __name__ == "__main__":
    main()
