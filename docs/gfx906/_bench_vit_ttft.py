#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Image-prompt TTFT client for the gfx906 ViT gate (VIT-1 / FA-D96).

Sends a chat-completions request with a *fresh* random image per rep and
reports TTFT (time to the first streamed content token). The freshness
matters: the mm encoder cache is not configurable in 0.29 and a repeated
image is served from it, so the ViT never runs (that artifact made VIT-1's
first A/B read "TTFT unchanged").

Prompt text is identical in every rep and every arm — `prompt_sha1` is logged
so the caller can assert it (AGENTS.md: never put the arm label in a prompt).
The image content varies by design, so `image_sha1` is logged next to it.

Usage:
  python _bench_vit_ttft.py --url http://127.0.0.1:8123 --model qwen27vl \
      --image-size 1024 --reps 3 --tag armA [--repeat-last]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import random
import struct
import time
import urllib.parse
import zlib

PROMPT = "Describe what you see in this image in one short sentence."


def make_png(size: int, seed: int) -> bytes:
    """A random-noise PNG of `size` x `size` (no PIL dependency).

    Fresh content every call is the point: it defeats the encoder cache.
    """
    rng = random.Random(seed)
    raw = bytearray()
    for _y in range(size):
        raw.append(0)  # filter type 0
        for _x in range(size):
            raw.extend((rng.randrange(256), rng.randrange(256), rng.randrange(256)))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def ttft(url: str, model: str, png: bytes, max_tokens: int, timeout: float):
    host = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(host.hostname, host.port, timeout=timeout)
    b64 = base64.b64encode(png).decode()
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True,
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:400]!r}")
    first = None
    text = []
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
        for ch in obj.get("choices", []):
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                if first is None:
                    first = time.perf_counter()
                text.append(piece)
    total = time.perf_counter() - t0
    conn.close()
    if first is None:
        raise RuntimeError("no content chunk received")
    return first - t0, total, "".join(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8123")
    ap.add_argument("--model", required=True)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--repeat-last", action="store_true",
                    help="send the last image again (encoder-cache control)")
    args = ap.parse_args()

    print(
        f"# tag={args.tag} url={args.url} model={args.model} "
        f"image={args.image_size}x{args.image_size} prompt_sha1="
        f"{hashlib.sha1(PROMPT.encode()).hexdigest()[:12]}",
        flush=True,
    )
    last_png = None
    for rep in range(args.reps):
        png = make_png(args.image_size, args.seed0 + rep)
        t, total, text = ttft(
            args.url, args.model, png, args.max_tokens, args.timeout
        )
        last_png = png
        print(
            json.dumps(
                {
                    "tag": args.tag,
                    "rep": rep,
                    "kind": "fresh",
                    "image_sha1": hashlib.sha1(png).hexdigest()[:12],
                    "ttft_s": round(t, 3),
                    "total_s": round(total, 3),
                    "text_head": text[:60],
                }
            ),
            flush=True,
        )
    if args.repeat_last and last_png is not None:
        t, total, text = ttft(
            args.url, args.model, last_png, args.max_tokens, args.timeout
        )
        print(
            json.dumps(
                {
                    "tag": args.tag,
                    "rep": args.reps,
                    "kind": "repeat",
                    "image_sha1": hashlib.sha1(last_png).hexdigest()[:12],
                    "ttft_s": round(t, 3),
                    "total_s": round(total, 3),
                    "text_head": text[:60],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
