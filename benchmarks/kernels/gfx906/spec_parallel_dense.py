# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Staggered 2-request parallel spec-decode probe (dense 27B).

Sends request A, then request B --stagger s later; B's prefill lands
while A is in (spec) decode, so the two arms differ in exactly the
mixed-batch behavior:

* baseline: B's prefill mixes with A's M=1 decode steps;
* mtp2:     B's prefill mixes with A's M<=3 verify steps (M up to
  2*(1+k) when both sequences carry drafts).

Metrics per request: TTFT (prefill proxy), end-to-end t/s; per pair:
aggregate output t/s; plus the engine's spec counters for the pair.

Usage (server: --served-model-name qwen27, --max-num-seqs>=2):
  python spec_parallel_dense.py --port 8900 --arm baseline
  python spec_parallel_dense.py --port 8900 --arm mtp2
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_ngram_dense import PROMPTS, spec_counters  # noqa: E402

OUT_TOKENS = 512


def stream_chat(base, msgs, out_tokens):
    """Streaming chat completion; returns (ttft_s, elapsed_s, n_out,
    text_sha)."""
    body = json.dumps({
        "model": "qwen27", "messages": msgs, "max_tokens": out_tokens,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n_out = 0
    buf = []
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
                n_out = chunk["usage"].get("completion_tokens", n_out)
            for ch in chunk.get("choices", []):
                delta = ch.get("delta", {})
                piece = delta.get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    buf.append(piece)
    dt = time.perf_counter() - t0
    if ttft is None:
        ttft = dt
    text = "".join(buf)
    return ttft, dt, n_out, hashlib.sha256(text.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--outdir", default="/tmp/spec_par")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--stagger", type=float, default=2.0,
                    help="seconds between request A and B starts")
    ap.add_argument("--out-tokens", type=int, default=OUT_TOKENS)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"

    # Warmup (tokenizer/capture paths), unmeasured.
    stream_chat(base, [{"role": "user", "content": "hi"}], 8)

    results = []
    for rep in range(args.repeats):
        before = spec_counters(args.port)
        res = {}

        def worker(key, msgs):
            res[key] = stream_chat(base, msgs, args.out_tokens)

        t0 = time.perf_counter()
        ta = threading.Thread(target=worker,
                              args=("A", PROMPTS[0]))
        ta.start()
        time.sleep(args.stagger)
        tb = threading.Thread(target=worker,
                              args=("B", PROMPTS[1]))
        tb.start()
        ta.join()
        tb.join()
        wall = time.perf_counter() - t0
        after = spec_counters(args.port)

        a = res["A"]
        b = res["B"]
        tok = a[2] + b[2]
        drafts = (after.get("vllm:spec_decode_num_drafts_total", 0)
                  - before.get("vllm:spec_decode_num_drafts_total", 0))
        draft_tok = (after.get("vllm:spec_decode_num_draft_tokens_total", 0)
                     - before.get(
                         "vllm:spec_decode_num_draft_tokens_total", 0))
        acc = (after.get("vllm:spec_decode_num_accepted_tokens_total", 0)
               - before.get("vllm:spec_decode_num_accepted_tokens_total", 0))
        rec = {
            "rep": rep, "arm": args.arm,
            "A_ttft_s": round(a[0], 3), "A_elapsed_s": round(a[1], 3),
            "A_out_tokens": a[2], "A_tps": round(a[2] / a[1], 3),
            "A_sha": a[3],
            "B_ttft_s": round(b[0], 3), "B_elapsed_s": round(b[1], 3),
            "B_out_tokens": b[2], "B_tps": round(b[2] / b[1], 3),
            "B_sha": b[3],
            "pair_wall_s": round(wall, 3),
            "pair_tokens": tok,
            "agg_tps": round(tok / wall, 3),
            "drafts": drafts, "draft_tokens": draft_tok,
            "accepted_tokens": acc,
            "accept_rate_pct": (round(100.0 * acc / draft_tok, 2)
                                if draft_tok else None),
        }
        results.append(rec)
        print(json.dumps(rec), flush=True)

    def key_stats(k):
        vals = [r[k] for r in results]
        return round(statistics.mean(vals), 3), round(
            statistics.stdev(vals) if len(vals) > 1 else 0.0, 3)

    a_tps, a_sd = key_stats("A_tps")
    b_tps, b_sd = key_stats("B_tps")
    agg, agg_sd = key_stats("agg_tps")
    b_ttft, b_ttft_sd = key_stats("B_ttft_s")
    a_ttft, a_ttft_sd = key_stats("A_ttft_s")
    summary = {
        "n": len(results),
        "A_tps_mean": a_tps, "A_tps_sd": a_sd,
        "B_tps_mean": b_tps, "B_tps_sd": b_sd,
        "A_ttft_mean_s": a_ttft, "A_ttft_sd_s": a_ttft_sd,
        "B_ttft_mean_s": b_ttft, "B_ttft_sd_s": b_ttft_sd,
        "agg_tps_mean": agg, "agg_tps_sd": agg_sd,
        "mean_accept_rate_pct": (
            statistics.mean(r["accept_rate_pct"] for r in results
                             if r["accept_rate_pct"] is not None)
            if any(r["accept_rate_pct"] is not None for r in results)
            else None),
    }
    print(f"SUMMARY {args.arm} {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
