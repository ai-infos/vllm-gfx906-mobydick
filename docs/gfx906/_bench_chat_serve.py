#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Chat-templated serving client with TTFT/decode metrics (MUSE-1 and friends).

Instruction-tuned checkpoints (Muse-Glimmer, Gemma-4) answer only inside their
own chat template, so every arm of a runner/config A/B has to go through
`/v1/chat/completions`. This client reports, per request: TTFT, decode t/s
(tokens after the first / wall after the first), total t/s, `prompt_sha1`
(identical prompts across arms is a hard rule) and an output digest so the two
arms' *content* can be compared as well as their speed.

Usage:
  python _bench_chat_serve.py --url http://127.0.0.1:8200 --model muse30b \
      --prompt-file /local/tmp/muse/prompt_2k.txt --reps 3 [--concurrency 1]
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor


def one_request(url: str, model: str, prompt: str, max_tokens: int,
                timeout: float, temperature: float = 0.0):
    host = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(host.hostname, host.port, timeout=timeout)
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    t0 = time.perf_counter()
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:400]!r}")
    first = None
    t_last = t0
    text: list[str] = []
    n_chunks = 0
    usage = None
    while True:
        line = resp.readline()
        if not line:
            break
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices", []):
            delta = ch.get("delta") or {}
            # Reasoning models (Muse-Glimmer, Gemma-4 thinking) stream their output in
            # `reasoning` under a reasoning parser; counting only `content` reads zero
            # tokens and looks like an empty generation.
            piece = delta.get("content") or delta.get("reasoning") \
                or delta.get("reasoning_content")
            if piece:
                if first is None:
                    first = time.perf_counter()
                t_last = time.perf_counter()
                text.append(piece)
                n_chunks += 1
    conn.close()
    if first is None:
        raise RuntimeError("no content received")
    out = "".join(text)
    ntok = (usage or {}).get("completion_tokens") or n_chunks
    ttft = first - t0
    decode_s = max(t_last - first, 1e-6)
    return {
        "ttft_s": round(ttft, 4),
        "decode_tps": round(max(ntok - 1, 0) / decode_s, 2),
        "total_tps": round(ntok / (t_last - t0), 2),
        "n_tokens": ntok,
        "out_sha1": hashlib.sha1(out.encode()).hexdigest()[:12],
        "text_head": out[:70].replace("\n", " "),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    with open(args.prompt_file) as fh:
        prompt = fh.read()
    sha = hashlib.sha1(prompt.encode()).hexdigest()[:12]
    print(
        f"# tag={args.tag} model={args.model} url={args.url} "
        f"prompt_sha1={sha} chars={len(prompt)} concurrency={args.concurrency}",
        flush=True,
    )
    for rep in range(args.reps):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(args.concurrency) as ex:
            rows = list(
                ex.map(
                    lambda _i: one_request(
                        args.url, args.model, prompt, args.max_tokens, args.timeout
                    ),
                    range(args.concurrency),
                )
            )
        wall = time.perf_counter() - t0
        agg = {
            "tag": args.tag,
            "rep": rep,
            "prompt_sha1": sha,
            "B": args.concurrency,
            "wall_s": round(wall, 3),
            "agg_tps": round(sum(r["n_tokens"] for r in rows) / wall, 2),
            "ttft_max_s": max(r["ttft_s"] for r in rows),
            "rows": rows,
        }
        print(json.dumps(agg), flush=True)


if __name__ == "__main__":
    main()
