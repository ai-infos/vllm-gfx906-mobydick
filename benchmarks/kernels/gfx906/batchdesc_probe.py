#!/usr/bin/env python3
# Copyright Kevin Read <me@kevin-read.com>
"""Batch-descriptor spy: logs num_tokens/padded sizes for every step.
Evidence for the L5 no-draft padding bug (pre-cg-small-fix)."""
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import collections, logging

def main():
    import vllm.v1.worker.gpu_model_runner as gmr
    real = gmr.GPUModelRunner._determine_batch_execution_and_padding
    descs = []
    def spy(self, num_tokens, *a, **kw):
        mode, desc = real(self, num_tokens, *a, **kw)[:2]
        descs.append((num_tokens, str(desc), str(mode)))
        return real(self, num_tokens, *a, **kw)
    gmr.GPUModelRunner._determine_batch_execution_and_padding = spy

    from vllm import LLM, SamplingParams
    llm = LLM(
        model="/data/models/qwen/Qwen3.5-27B-AWQ",
        max_num_seqs=4, max_model_len=2816,
        gpu_memory_utilization=0.95, dtype="float16",
        speculative_config={
            "method": "ngram", "num_speculative_tokens": 3,
            "prompt_lookup_min": 2, "prompt_lookup_max": 5})
    tok = llm.get_tokenizer()
    enc = tok.apply_chat_template(
        [{"role": "user", "content":
          "Repeat the following sentence exactly 30 times, once per "
          "line, with no changes: the quick brown fox jumps over the "
          "lazy dog"}],
        add_generation_prompt=True, enable_thinking=False)
    p1 = list(enc["input_ids"]) if not isinstance(enc, str) else enc
    llm.generate([p1], SamplingParams(max_tokens=16, temperature=0))
    descs.clear()
    out = llm.generate([p1], SamplingParams(max_tokens=128, temperature=0))
    n = len(out[0].outputs[0].token_ids)
    c = collections.Counter(descs)
    print(f"OUT={n}", flush=True)
    for (nt, d, m), cnt in c.most_common():
        print(f"  x{cnt:3d}  num_tokens={nt:3d}  {d}  {m}", flush=True)
    os._exit(0)

if __name__ == "__main__":
    main()
