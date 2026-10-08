# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Prefill TTFT probe: warm, one-time-free, multi-block cache test.

Sequential max_tokens=1 requests against a running OpenAI server,
measuring TTFT and derived prefill rate per request:

  W:  warmup (not measured) -- absorbs per-server one-time costs
      (Triton JIT of the mamba-align/eagle kernels, lazy init)
  A:  795-tok prompt, fresh + x2 (sub-block: baseline block 784
      caches 784 -> ~0.14 s hits; mtp2 block 800 caches nothing)
  B:  619-tok prompt, fresh (sub-block in both arms -- no hits)
  L:  ~1631-tok prompt (PROMPTS[0] + extra messages, so it shares
      A's prefix), fresh + x2 (two full blocks: baseline hits 1568
      from rep1; mtp2 follows the Marconi admission pattern:
      fresh, fresh, then a recurring 800-tok hit)

NOTE: run the baseline arm with --gpu-memory-utilization 0.93 --
at 0.95 with a warm inductor cache the engine OOMs on the second
request (356 MiB runtime allocation inside the piecewise graph;
see docs/gfx906/DEVLOG-spec-decode.md, Prefill/TTFT section).

Usage:
  python prefill_ttft_probe.py --port 8901 --arm baseline|mtp2
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_ngram_dense import PROMPTS  # noqa: E402

LONG_EXTRA = (
    " Also write a detailed technical analysis of the following "
    "system design, covering trade-offs, failure modes, scaling "
    "behavior, and concrete recommendations. "
) * 30
LONG = PROMPTS[0] + [
    {"role": "user", "content": LONG_EXTRA},
    {"role": "assistant", "content": "Here is the analysis."},
    {"role": "user", "content": "Continue with the remaining sections."},
]


def ttft_probe(base, msgs, max_tokens=1):
    body = json.dumps({
        "model": "qwen27", "messages": msgs, "max_tokens": max_tokens,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    p_tok = c_tok = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                p_tok = chunk["usage"].get("prompt_tokens", 0)
                c_tok = chunk["usage"].get("completion_tokens", c_tok)
            for ch in chunk.get("choices", []):
                if ch.get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
    return {
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "prompt_tokens": p_tok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--arm", required=True)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    plan = [
        ("W_warmup", PROMPTS[2], False),
        ("A_fresh", PROMPTS[0], True),
        ("A_rep1", PROMPTS[0], True),
        ("A_rep2", PROMPTS[0], True),
        ("B_fresh", PROMPTS[1], True),
        ("L_fresh", LONG, True),
        ("L_rep1", LONG, True),
        ("L_rep2", LONG, True),
    ]
    recs = []
    for name, msgs, measured in plan:
        rec = {"req": name, "arm": args.arm}
        rec.update(ttft_probe(base, msgs))
        if measured and rec["ttft_s"] and rec["prompt_tokens"]:
            rec["prefill_tps"] = round(
                rec["prompt_tokens"] / rec["ttft_s"], 1)
        recs.append(rec)
        print(json.dumps(rec), flush=True)
        time.sleep(2)
    print("SUMMARY", args.arm, json.dumps(
        {k: [r.get(k) for r in recs] for k in
         ("req", "ttft_s", "prompt_tokens", "prefill_tps")}),
        flush=True)


if __name__ == "__main__":
    main()
