# Copyright Kevin Read <me@kevin-read.com>
"""Profile a fixed repetitive generation with/without ngram spec decode.

Same LLM, same prompt (chat-templated, thinking disabled), 128 tokens.
Prints top CUDA ops by total device time. Diff spec-on vs spec-off to
attribute the spec-step overhead.
"""
import argparse
import os
import time

# MUST be in-process: the EngineCore runs in a child process by
# default (spawn) and the torch.profiler in this process then captures
# nothing but hipDeviceSynchronize.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="?", const="ngram3",
                    help="spec method: ngram3 (default) or mtpK")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    common = dict(
        model="/data/models/qwen/Qwen3.5-27B-AWQ",
        max_num_seqs=4,
        max_model_len=2816,
        gpu_memory_utilization=0.95,
        dtype="float16",
    )
    if args.enforce_eager:
        common["enforce_eager"] = True
    if args.spec:
        if args.spec.startswith("mtp"):
            k = int(args.spec[3:])
            common["speculative_config"] = {
                "method": "mtp", "num_speculative_tokens": k,
            }
        else:
            common["speculative_config"] = {
                "method": "ngram",
                "num_speculative_tokens": 3,
                "prompt_lookup_min": 2,
                "prompt_lookup_max": 5,
            }
    llm = LLM(**common)
    tok = llm.get_tokenizer()
    enc = tok.apply_chat_template(
        [{"role": "user", "content":
          ("Repeat the following sentence exactly 30 times, once per "
           "line, with no changes: the quick brown fox jumps over the "
           "lazy dog")}],
        add_generation_prompt=True, enable_thinking=False)
    # this tokenizer's template is token-based: returns BatchEncoding
    prompt = list(enc["input_ids"]) if not isinstance(enc, str) else enc

    # Warmup (capture etc.)
    llm.generate(prompt, SamplingParams(max_tokens=16, temperature=0))

    sp = SamplingParams(max_tokens=args.tokens, temperature=0)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = llm.generate(prompt, sp)
        dt = time.perf_counter()
    n_out = len(out[0].outputs[0].token_ids)
    print(f"OUT_TOKENS={n_out} ARM={args.spec or 'nospec'}")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=25,
        max_name_column_width=170))


if __name__ == "__main__":
    main()
