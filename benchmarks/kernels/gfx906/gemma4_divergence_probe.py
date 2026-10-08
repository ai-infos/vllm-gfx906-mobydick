# SPDX-License-Identifier: Apache-2.0
# Copyright Kevin Read <me@kevin-read.com>
"""Gemma-4 arm-vs-arm token-divergence probe (MED-2 numerics record).

Re-runs the 12-prompt x 64-token greedy A/B from DEVLOG-gemma4-moe.md
(Numerical A/B) with logprobs recorded per step, plus per-position
prompt logprobs to characterize the prefill regime (the model's
prefill-time logprobs are anomalous on this stack — the "garbage
prefill" regime). One JSON per arm:

  python gemma4_divergence_probe.py --moe-backend auto  --out /tmp/g4_auto.json
  python gemma4_divergence_probe.py --moe-backend triton --out /tmp/g4_triton.json
  python gemma4_divergence_probe.py --analyze /tmp/g4_auto.json /tmp/g4_triton.json

The analyze step answers the review question: do first-diff positions
cluster in the prefill-output regime (position ~0, where the first
decode token is sampled from the anomalous prefill distribution) or
are they spread through pure decode, with ties at the diff position
indistinguishable from fp16-noise near-ties elsewhere?
"""

import argparse
import json
import math
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# Same 12 prompts as gemma4_chat_probe.py (the Numerical A/B set).
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

SNAP = (
    "/local/cache/huggingface/hub/models--cyankiwi--gemma-4-26B-A4B-it-AWQ-4bit"
    "/snapshots/0ef577a5710035bd2d3a3f27e4f5cb2e86a9a9ba"
)


def collect(moe_backend: str, out_path: str) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=SNAP,
        dtype="float16",
        max_num_seqs=4,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        max_model_len=4096,
        moe_backend=moe_backend,
    )
    tokenizer = llm.get_tokenizer()
    templated = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in PROMPTS
    ]
    prompts_mm = [{"prompt": t} for t in templated]

    # Prefill regime: per-position prompt logprobs (k=20 top-k).
    outs = llm.generate(
        prompts_mm, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=20)
    )
    # Per position: list of (logprob, token_id), sorted by lp descending.
    # Logprobs are <= 0, so top-1 is the MAX logprob.
    prompt_lp = []
    for o in outs:
        row = [
            sorted(((v.logprob, k) for k, v in e.items()), reverse=True)
            for e in o.prompt_logprobs[1:]
        ]
        prompt_lp.append(row)

    # Decode: 64 tokens with logprobs=10 for the per-step comparison.
    outs = llm.generate(
        prompts_mm, SamplingParams(temperature=0.0, max_tokens=64, logprobs=10)
    )
    gens = []
    for o in outs:
        steps = []
        for lp_list in o.outputs[0].logprobs:
            top10 = sorted(((v.logprob, k) for k, v in lp_list.items()), reverse=True)
            # Greedy (temperature=0): the sampled token is the top-1.
            lp0, tid = top10[0]
            steps.append({"id": tid, "lp": lp0, "top10": top10})
        gens.append(steps)

    with open(out_path, "w") as f:
        json.dump(
            {
                "moe_backend": moe_backend,
                "prompt_top1_lp": prompt_lp,
                "gen": gens,
            },
            f,
        )
    print(f"WROTE {out_path}")


def _percentile(vals, p):
    vals = sorted(vals)
    k = (len(vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


def analyze(auto_path: str, triton_path: str) -> None:
    with open(auto_path) as f:
        a = json.load(f)
    with open(triton_path) as f:
        t = json.load(f)

    print(f"arms: {a['moe_backend']} vs {t['moe_backend']}")
    diffs = []
    matching_dlp = []
    diff_dlp = []
    diff_tie_gap = []
    match_tie_gap = []
    for i, (ga, gt) in enumerate(zip(a["gen"], t["gen"])):
        first_diff = None
        for pos, (sa, st) in enumerate(zip(ga, gt)):
            # tie gap: sampled-token (top-1) lp minus the next-closest
            # token's lp; negative only if the top-1 is a true tie
            gap_a = sa["lp"] - sa["top10"][1][0]
            gap_t = st["lp"] - st["top10"][1][0]
            if sa["id"] != st["id"]:
                if first_diff is None:
                    first_diff = pos
                    diff_dlp.append(abs(sa["lp"] - st["lp"]))
                    diff_tie_gap.append(min(gap_a, gap_t))
            else:
                matching_dlp.append(abs(sa["lp"] - st["lp"]))
                match_tie_gap.append(min(gap_a, gap_t))
        n = len(ga)
        diffs.append(
            (
                i,
                first_diff,
                n,
                sum(1 for sa, st in zip(ga, gt) if sa["id"] == st["id"]) / n,
            )
        )

    print("\nper-prompt: idx, first-diff, len, token-match")
    for i, fd, n, m in diffs:
        print(f"  {i:2d}: first_diff={fd!s:4s} len={n} match={m:.2f}")

    fd_pos = [fd for _, fd, _, _ in diffs if fd is not None]
    divergent = [i for i, fd, _, _ in diffs if fd is not None]
    print(f"\ndivergent prompts: {len(divergent)}/{len(diffs)} {divergent}")
    if fd_pos:
        print(f"first-diff positions: {fd_pos}")
        print(
            f"  median={_percentile(fd_pos, 0.5):.0f} min={min(fd_pos)} "
            f"max={max(fd_pos)} (of 64 decode positions)"
        )
        early = sum(1 for p in fd_pos if p <= 1)
        print(f"  at position 0-1: {early}/{len(fd_pos)}")

    def stats(vals, name):
        if not vals:
            return
        print(
            f"{name}: n={len(vals)} median={_percentile(vals, 0.5):.4f} "
            f"p90={_percentile(vals, 0.9):.4f} "
            f"p99={_percentile(vals, 0.99):.4f} max={max(vals):.4f}"
        )

    print("\n|dLP| of the sampled token at matching steps:")
    stats(matching_dlp, "  matching")
    print("|dLP| at first-diff steps (arms' sampled tokens differ):")
    stats(diff_dlp, "  diff")
    print("tie gap (sampled lp - next lp) at matching vs diff steps:")
    stats(match_tie_gap, "  matching")
    stats(diff_tie_gap, "  diff")

    # Prefill-regime flatness per prompt: the model's top-1 logprob over
    # its own prompt tokens (per position). A healthy model scores its own
    # prompt near 0 (it knows what comes next); the garbage-prefill regime
    # on this model/stack shows deeply negative, flat top-1 lp.
    print(
        "\nprefill flatness (mean top-1 prompt lp; near 0 = healthy,"
        " deeply negative = garbage regime):"
    )
    for i in range(len(diffs)):
        lps = [row[0][0] for row in a["prompt_top1_lp"][i]]
        mlp = sum(lps) / len(lps)
        tag = "DIVERGENT" if i in divergent else ""
        print(f"  {i:2d}: mean_top1_lp={mlp:8.3f} {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moe-backend", choices=["auto", "triton"])
    ap.add_argument("--out")
    ap.add_argument("--analyze", nargs=2, metavar=("AUTO", "TRITON"))
    args = ap.parse_args()
    if args.analyze:
        analyze(*args.analyze)
    else:
        assert args.moe_backend and args.out
        collect(args.moe_backend, args.out)


if __name__ == "__main__":
    main()
