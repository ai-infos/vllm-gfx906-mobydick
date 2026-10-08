# SPDX-License-Identifier: Apache-2.0
# Copyright Kevin Read <me@kevin-read.com>
"""Gemma-4-26B-A4B-AWQ quality gate (chat-templated PPL + greedy decode).

Gemma-4 is a thinking-mode instruct model: raw continuation prompts
degenerate into confident repetition loops, so every prompt here goes
through apply_chat_template. Same protocol as ppl_probe.py (k=20
top-k prompt logprobs, actual-token lookup, residual-mass fallback),
plus a 128-token greedy decode for token-exact A/B comparison.

Usage:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      python benchmarks/kernels/gfx906/gemma4_chat_probe.py \
      [--moe-backend auto|triton] [--max-tokens 128]
"""

import argparse
import json
import math

from vllm import LLM, SamplingParams

SNAP = (
    "/local/cache/huggingface/hub/models--cyankiwi--gemma-4-26B-A4B-it-AWQ-4bit"
    "/snapshots/0ef577a5710035bd2d3a3f27e4f5cb2e86a9a9ba"
)

PROMPTS = [
    (
        "The quick brown fox jumps over the lazy dog while the farmer watches "
        "from the porch, counting the chickens that wandered into the yard."
    ),
    (
        "In distributed systems, consistency comes at the cost of availability "
        "and latency; engineers choose where on that spectrum their product "
        "must live, and they document the tradeoff carefully."
    ),
    (
        "Photosynthesis converts light energy into chemical energy stored in "
        "glucose, a process that ultimately powers nearly every food chain on "
        "the planet and shapes the composition of the atmosphere."
    ),
    (
        "The compiler first parses the source into an abstract syntax tree, "
        "then walks that tree emitting instructions while keeping the register "
        "pressure within the limits of the target architecture."
    ),
    (
        "Marble statues weather slowly as acid rain etches their surfaces, "
        "turning sharp chiselled detail into soft shapes over the course of "
        "centuries of exposure to the open air."
    ),
    (
        "A well-tended garden rewards patience: the tomato seedlings that "
        "survive their first frost bear heavy fruit in late summer, and the "
        "herbs return stronger each spring."
    ),
    (
        "Quantum computers exploit superposition and entanglement to explore "
        "many candidate solutions at once, though error correction remains the "
        "principal obstacle to practical machines."
    ),
    (
        "The lighthouse keeper climbed the spiral staircase each evening, "
        "trimmed the wick, and watched the beam sweep the dark water until "
        "the first ferry lights appeared on the horizon."
    ),
    (
        "Battery capacity fades with age because the solid electrolyte "
        "interface layer thickens on the anode, trapping lithium ions and "
        "reducing the charge the cell can deliver."
    ),
    (
        "Good tests describe intent: a name like test_overflow_refunds_when "
        "total_exceeds_budget tells the reader what behavior is protected "
        "without opening the body of the function."
    ),
    (
        "The river braids and splits around gravel islands each spring, "
        "carrying snowmelt from the high valleys down to the delta where the "
        "marsh grass bends but does not break."
    ),
    (
        "Compilers for tensor workloads must reason about memory bandwidth as "
        "carefully as arithmetic intensity, because a kernel that fits in "
        "cache runs orders of magnitude faster than it streams."
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SNAP)
    ap.add_argument("--moe-backend", default="auto")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    llm = LLM(
        model=args.model,
        dtype="float16",
        max_num_seqs=4,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        max_model_len=4096,
        moe_backend=args.moe_backend,
    )
    tokenizer = llm.get_tokenizer()
    # This tokenizer's default chat-template kwargs tokenize=True (returns a
    # BatchEncoding); force plain strings.
    templated = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in PROMPTS
    ]
    assert all(isinstance(t, str) for t in templated)

    # Multimodal wrapper: prompts must be dicts.
    prompts_mm = [{"prompt": t} for t in templated]

    # --- PPL (prompt logprobs, house protocol) ---
    # Tokenize with add_special_tokens=False to match the engine's prompt
    # tokenization (this tokenizer's encode() otherwise prepends a BOS and
    # shifts every lookup by one). Both variants are printed so a protocol
    # mismatch is visible as an implausible PPL.
    sp_ppl = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=20)
    outs = llm.generate(prompts_mm, sp_ppl)
    for bos in (False, True):
        total_lp, n_tok, n_miss = 0.0, 0, 0
        for prompt, o in zip(templated, outs):
            ids = tokenizer.encode(prompt, add_special_tokens=not bos)
            if len(ids) != len(o.prompt_logprobs):
                print(
                    f"PPL(bos={bos}) length mismatch {len(ids)} vs "
                    f"{len(o.prompt_logprobs)} - skipping"
                )
                break
            for i in range(1, len(ids) - 1):
                entry = o.prompt_logprobs[i]
                lp_obj = entry.get(ids[i])
                if lp_obj is not None:
                    total_lp += lp_obj.logprob
                else:
                    mass = sum(math.exp(v.logprob) for v in entry.values())
                    total_lp += math.log(max(1.0 - mass, 1e-12))
                    n_miss += 1
                n_tok += 1
        else:
            ppl = math.exp(-total_lp / n_tok)
            print(f"PPL(bos={bos}) {ppl:.4f}  ({n_tok} tokens, {n_miss} top-20 misses)")

    # --- greedy decode for A/B token comparison ---
    sp_gen = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outs = llm.generate(prompts_mm, sp_gen)
    for i, o in enumerate(outs):
        ids = list(o.outputs[0].token_ids)
        txt = o.outputs[0].text
        print(f"GEN[{i}] {json.dumps(ids)}")
        print(f"TEXT[{i}] {txt[:300]!r}")
    print("PROBE DONE")


if __name__ == "__main__":
    main()
