#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Greedy token-identity probe for kernel dispatch A/B gates.

Generates 12 x 128 greedy tokens with the standard bench config and writes
one sha256 line per prompt plus a combined hash to OUT (default
/tmp/greedy_<TAG>.txt). Two runs (dispatch on/off) must produce identical
files for the change to be considered output-safe.

Usage:
  HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 HIP_VISIBLE_DEVICES=0 \
  .venv/bin/python benchmarks/kernels/gfx906/greedy_probe.py TAG
"""
import hashlib
import os
import sys

MODEL = os.environ.get("BENCH_MODEL", "/local/models/QuantTrio/Qwen3.5-35B-A3B-AWQ")
PROMPTS = [
    "The capital of France is",
    "In a distant galaxy, the last star began to",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "To be or not to be, that is the",
    "The quick brown fox jumps over the lazy",
    "SELECT * FROM users WHERE id =",
    "Once upon a time in a small village near the",
    "Theorem: for all n, the sum of the first n odd",
    "#!/usr/bin/env python3\nimport sys\n\ndef main():",
    "The patient was admitted with a three-day history of",
    "In quantum mechanics, the uncertainty principle states that",
    "The 2024 championship final ended in a dramatic",
]
N_TOK = 128


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "default"
    out_path = f"/tmp/greedy_{tag}.txt"

    from vllm import LLM, SamplingParams

    extra = {}
    # BENCH_MOE_BACKEND (e.g. triton) overrides the MoE backend selection
    # for A/B runs (default auto picks the gfx906 W4A16 kernel where gated).
    moe_backend = os.environ.get("BENCH_MOE_BACKEND")
    if moe_backend:
        extra["moe_backend"] = moe_backend
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=8,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        seed=0,
        **extra,
    )
    tok = llm.get_tokenizer()
    prompts = []
    for i, p in enumerate(PROMPTS):
        t = tok.encode(p)
        # deterministic padding to distinct lengths
        prompts.append(t + ([(i % 500) + 1000] * (20 + i * 7)))
    sampling = SamplingParams(temperature=0.0, max_tokens=N_TOK)
    outs = llm.generate([tok.decode(t) for t in prompts], sampling)

    lines = []
    for i, o in enumerate(outs):
        ids = o.outputs[0].token_ids
        h = hashlib.sha256(
            ",".join(str(i) for i in ids).encode()).hexdigest()[:32]
        lines.append(f"{i:2d} {len(ids):4d} {h}")
    combined = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    lines.append(f"ALL {combined}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
