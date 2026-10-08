# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Gate for *instruction-tuned* checkpoints (Gemma-4-*-it, Muse-Glimmer, …).

Why this exists (GEMMA4-1, 2026-09-15): those checkpoints answer only inside their own
chat template. Fed raw text they emit fluent garbage that looks like a broken
model/kernel — and the in-process prompt-logprob PPL probe cannot gate them at all:
raw text scores PPL 84261 (362/363 top-20 misses) and *templating the prompt makes it
worse* (PPL 1278491), because prompt-logprob PPL asks the model to predict the **user's**
tokens, which an instruct model was never trained to model. Both numbers are prompt-format
artifacts, not defects.

What to run instead: this script. It renders each prompt through the model's own chat
template (with the generation prompt) and reports, per prompt, the greedy continuation and
the **first-token top-k logprobs** — a confident first token (logprob ≈ 0) with a sensible
completion is the sanity signal, and running it under two configurations (e.g.
V1 vs V2, or two triton builds) gives a parity gate by comparing text and logprobs.

Usage:
    BENCH_MODEL=<ckpt> HIP_VISIBLE_DEVICES=0 .venv/bin/python \
        benchmarks/kernels/gfx906/ift_chat_gate.py

Env: BENCH_MODEL (required), BENCH_TOP_K (default 5), BENCH_MAX_TOKENS (default 16),
ATTN_BACKEND (optional, forwarded to LLM(attention_backend=...)), and the usual
VLLM_* / VLLM_USE_V2_MODEL_RUNNER knobs. Prompts can be overridden with BENCH_PROMPTS
(one per line, newline-escaped with \\n as literal).
"""

from __future__ import annotations

import os

from vllm import LLM, SamplingParams

PROMPTS = [
    "What is the capital of France? Answer with one word.",
    "Write a Python function that adds two numbers.",
]

EXTRA = [p for p in os.environ.get("BENCH_PROMPTS", "").splitlines() if p.strip()]


def main() -> None:
    model = os.environ["BENCH_MODEL"]
    top_k = int(os.environ.get("BENCH_TOP_K", "5"))
    max_tokens = int(os.environ.get("BENCH_MAX_TOKENS", "16"))
    ab = os.environ.get("ATTN_BACKEND") or None

    llm = LLM(
        model=model,
        max_model_len=2048,
        max_num_seqs=2,
        gpu_memory_utilization=0.90,
        dtype=os.environ.get("BENCH_DTYPE", "float16"),
        seed=0,
        attention_backend=ab,
    )
    tok = llm.get_tokenizer()
    if not getattr(tok, "chat_template", None):
        raise SystemExit(
            "this tokenizer has no chat template — for raw-text-tolerant models use "
            "docs/gfx906/_bench_gfx906.py (speed) or benchmarks/kernels/gfx906/"
            "ppl_probe.py (numerics) instead"
        )

    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in (PROMPTS + EXTRA)]
    print(f"GATE runner={os.environ.get('VLLM_USE_V2_MODEL_RUNNER', 'default')} "
          f"n_prompts={len(prompts)} top_k={top_k}", flush=True)

    for p, rendered in zip(PROMPTS + EXTRA, prompts):
        o = llm.generate([rendered], SamplingParams(temperature=0.0, max_tokens=max_tokens,
                                                   ignore_eos=True, logprobs=top_k),
                         use_tqdm=False)
        out = o[0].outputs[0]
        first = (out.logprobs or [{}])[0] if out.logprobs else {}
        top = sorted(
            ((t, lp.logprob) for t, lp in first.items()), key=lambda kv: -kv[1]
        )[:top_k]
        print(f"  PROMPT {p[:60]!r}")
        print(f"      text : {out.text[:120]!r}")
        print(f"      top{top_k} : " + ", ".join(f"{t!r}={v:.2f}" for t, v in top), flush=True)


if __name__ == "__main__":
    main()
