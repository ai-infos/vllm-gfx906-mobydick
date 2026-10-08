#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""
Serving benchmark grid for a running gfx906 TP=2 (or TP=1) server.

Model/snapshot are env-driven (Muse defaults):
  LC_MODEL=<served name> LC_SNAP=<snapshot path> \
      .venv/bin/python docs/gfx906/_bench_serve_grid_gfx906.py \
      '[[pp, tg], ...]' <samples>
Default grid is the long-context prefill sweep (see README "Long-context
performance"): [[32768,128],[65536,128],[112640,128]] x 2.

Two prompt modes:
  1. Filler (default): mirrors _bench_gfx906.py prompt semantics
     (repetitive fox filler padded to exactly pp tokens, raw
     /v1/completions, string prompt).
  2. Corpus (BENCH_CORPUS=<path>): exact-length token-id prompts from a
     replay corpus json (the mixed-v2 production payload) — header +
     body[:pp-len(header)], the same construction as sweep_client_3pt.py:
       BENCH_CORPUS=/local/tmp/mtp1/corpus.json \
       BENCH_CORPUS_KEY=mixed BENCH_CORPUS_POINT=65536 \
       BENCH_HEADER_SNAP=<qwen3.8 snapshot path>
     Bodies are truncated at the FRONT (chat content leads the body);
     any pp <= body length is a well-defined point. Each concurrent
     sequence in a batch gets a distinct body.

Grid cells may be [pp, tg] (nreqs=1) or [pp, tg, nreqs] (concurrent
batch — B=4 campaign cells; nreqs sequences launch as one
ThreadPoolExecutor batch and the rep's timing is the batch window).

BENCH_METRICS=1 adds per-rep spec-decode acceptance from /metrics
deltas (aggregate over the concurrent batch; vLLM's spec counters carry
no per-request labels): drafts (steps), draft_tokens (proposed),
accepted, mean_acc = accepted/steps, per-position acceptance.

Every rep also reports the validity screens: stop_agreement (all
sequences hit max_tokens under ignore_eos) and a cheap repetition
screen (distinct 8-gram fraction of the output; < 0.5 = degenerate
loop). For B>1 compare acceptance RATES across B, never token identity
(batched reduction order flips near-tie argmaxes).
"""
import contextlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = f"http://localhost:{os.environ.get('BENCH_PORT', '8000')}"
MODEL = os.environ.get("LC_MODEL", "Muse-Glimmer-30B")
SNAP = os.environ.get(
    "LC_SNAP",
    "/local/cache/huggingface/hub/models--cyankiwi--Muse-Glimmer-30B-"
    "AWQ-INT4/snapshots/cba01edf73e0f0f4f013615cc01281ea04e79f85")
CORPUS = os.environ.get("BENCH_CORPUS", "")
CORPUS_KEY = os.environ.get("BENCH_CORPUS_KEY", "mixed")
CORPUS_POINT = os.environ.get("BENCH_CORPUS_POINT", "65536")
HEADER_SNAP = os.environ.get(
    "BENCH_HEADER_SNAP",
    "/local/cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-"
    "INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea")
WANT_METRICS = os.environ.get("BENCH_METRICS", "0") == "1"

FILLERS = [
    "The quick brown fox jumps over the lazy dog. ",
    "A pale gold coin spins slowly in the cold museum light. ",
    "The tide pulls the driftwood logs in long silver arcs. ",
    "Every spring the river braids around gravel islands again. ",
]

_bodies = None
_h_tok = None


def corpus_bodies():
    global _bodies
    if _bodies is None:
        with open(CORPUS) as f:
            _bodies = json.load(f)[CORPUS_KEY][CORPUS_POINT]
    return _bodies


def header_tok():
    global _h_tok
    if _h_tok is None:
        from transformers import AutoTokenizer
        _h_tok = AutoTokenizer.from_pretrained(HEADER_SNAP)
    return _h_tok


# Per-socket timeout. Must exceed the LONGEST expected TTFT of the
# bench point (a streaming read gets no data until first token): the
# 120k B=4 cell's last TTFT is ~3600 s (measured 2026-09-10), which
# killed the 3600 s default mid-batch. Override with BENCH_API_TIMEOUT.
def api(path, body=None, timeout=None):
    if timeout is None:
        timeout = float(os.environ.get("BENCH_API_TIMEOUT", "10800"))
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def make_prompt(tok, pp, variant, tag):
    """Build a prompt of EXACTLY pp tokens (bulk filler, O(1) encodes).

    BOS-safe: some tokenizers (e.g. Muse's) prepend a BOS on encode and
    render it as literal text on decode — decoding a list that contains
    the BOS makes the round-trip grow by one, so strip it first.
    """
    filler = FILLERS[variant % len(FILLERS)]
    n_fill = len(tok.encode(filler))
    prompt = f"[run {tag}] "
    n = len(tok.encode(prompt))
    while n < pp:
        prompt += filler * max(1, (pp - n) // n_fill)
        n = len(tok.encode(prompt))
    toks = tok.encode(prompt)
    has_bos = bool(toks) and toks[0] == tok.bos_token_id
    body = toks[1:] if has_bos else toks
    return tok.decode(body[:pp - (1 if has_bos else 0)])


def corpus_prompt_ids(pp, tag, body_idx):
    """Token-id prompt of EXACTLY pp: unique header + body front-slice
    (same construction as sweep_client_3pt.py)."""
    h = f"RESEARCH-BRIEFING-{tag}-x7f3e9a2 "
    header_ids = header_tok()(h, add_special_tokens=False)["input_ids"]
    body_ids = corpus_bodies()[body_idx % len(corpus_bodies())]
    ids = header_ids + body_ids[:pp - len(header_ids)]
    assert len(ids) == pp, (len(ids), pp)
    return ids


def rep_screen(text):
    """Cheap degenerate-loop screen: distinct 8-gram fraction."""
    grams = [text[i:i + 8] for i in range(max(0, len(text) - 7))]
    if len(grams) < 8:
        return 1.0
    frac = len(set(grams)) / len(grams)
    return round(frac, 4)


def run_one(tok, pp, tg, variant, tag):
    if CORPUS:
        prompt = corpus_prompt_ids(pp, f"{tag}", variant)
    else:
        prompt = make_prompt(tok, pp, variant, tag)
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": tg,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    t0 = time.time()
    first, t_end, usage, out_text = None, t0, None, []
    with api("/v1/completions", body) as f:
        for raw in f:
            raw = raw.strip()
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("text"):
                if first is None:
                    first = time.time()
                t_end = time.time()
                out_text.append(ch[0]["text"])
    t_end = max(t_end, time.time())
    out = usage["completion_tokens"] if usage else 0
    ttft = (first - t0) if first is not None else (t_end - t0)
    wall = t_end - t0
    text = "".join(out_text)
    return {
        "out": out,
        "ttft_s": round(ttft, 3),
        "wall_s": round(wall, 3),
        "prefill_tps": round(pp / ttft, 1) if ttft > 0 else 0.0,
        "decode_tps": round((out - 1) / (wall - ttft), 2)
        if wall > ttft and out > 1 else 0.0,
        "wall_tps": round(out / wall, 2) if wall > 0 else 0.0,
        "rep_frac8": rep_screen(text),
    }


def get_metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        out = {}
        for line in r.read().decode().splitlines():
            if line.startswith("#") or not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            with contextlib.suppress(ValueError):
                out[parts[0]] = float(parts[1])
        return out


def _sum_prefix(d, name):
    return sum(v for k, v in d.items() if k.startswith(name))


def spec_delta(before, after):
    drafts = _sum_prefix(after, "vllm:spec_decode_num_drafts_total") \
        - _sum_prefix(before, "vllm:spec_decode_num_drafts_total")
    draft_tok = _sum_prefix(after, "vllm:spec_decode_num_draft_tokens_total") \
        - _sum_prefix(before, "vllm:spec_decode_num_draft_tokens_total")
    acc = _sum_prefix(after, "vllm:spec_decode_num_accepted_tokens_total") \
        - _sum_prefix(before, "vllm:spec_decode_num_accepted_tokens_total")
    pos = {}
    for src, sign in ((before, -1.0), (after, 1.0)):
        for k, v in src.items():
            if "spec_decode_num_accepted_tokens_per_pos" not in k \
                    or "{" not in k:
                continue
            lbl = k[k.index("{") + 1:k.index("}")]
            for pair in lbl.split(","):
                pair = pair.strip()
                if pair.startswith("position="):
                    p = pair.split("=")[1].strip('"')
                    pos[p] = pos.get(p, 0.0) + sign * v
    if drafts <= 0:
        return None
    return {
        "steps": int(drafts),
        "draft_tokens": int(draft_tok),
        "accepted": int(acc),
        "mean_acc": round((drafts + acc) / drafts, 4),
        "acc_rate": round(acc / draft_tok, 4) if draft_tok else None,
        "per_pos": {p: round(v / drafts, 4) for p, v in sorted(pos.items())},
    }


def main():
    tok = None
    if not CORPUS:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(SNAP)
    else:
        # Force first use on the main thread: with B > 1 the worker pool
        # does its first `from transformers import ...` + from_pretrained
        # concurrently, which hits a thread-safety gap in this
        # transformers build (ImportError: cannot import name
        # 'AutoTokenizer').
        corpus_bodies()
        header_tok()

    grid = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
        [32768, 128], [65536, 128], [112640, 128],
    ]
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    print("BENCH-SERVE: " + json.dumps({
        "model": MODEL, "server": BASE, "samples": samples,
        "grid": grid, "corpus": CORPUS_KEY if CORPUS else "filler",
        "corpus_point": CORPUS_POINT if CORPUS else None,
        "metrics": WANT_METRICS}), flush=True)

    for cell in grid:
        pp, tg = cell[0], cell[1]
        nreqs = cell[2] if len(cell) > 2 else 1
        for s in range(samples):
            t_wall0 = time.time()
            before = get_metrics() if WANT_METRICS else None
            if nreqs == 1:
                row = run_one(
                    tok, pp, tg, s, f"pp{pp}tg{tg}s{s}")
                rows = [row]
            else:
                def _worker(v, pp=pp, tg=tg, s=s):
                    return run_one(
                        tok, pp, tg, v, f"pp{pp}tg{tg}s{s}r{v}")

                with ThreadPoolExecutor(nreqs) as ex:
                    rows = list(ex.map(_worker, range(nreqs)))
            wall = time.time() - t_wall0
            after = get_metrics() if WANT_METRICS else None
            tot_out = sum(r["out"] for r in rows)
            agg = {
                "B": nreqs,
                "out_total": tot_out,
                "wall_s": round(wall, 3),
                "wall_tps": round(tot_out / wall, 2),
                "ttft_max_s": round(max(r["ttft_s"] for r in rows)),
                "prefill_tps_aggregate": round(nreqs * pp / wall, 1),
                "stop_agreement": all(r["out"] == tg for r in rows),
                "rep_frac8_min": min(r["rep_frac8"] for r in rows),
                "seqs": [
                    {k: r[k] for k in
                     ("out", "ttft_s", "wall_s", "decode_tps",
                      "wall_tps", "rep_frac8")}
                    for r in rows],
            }
            if after is not None:
                agg["spec"] = spec_delta(before, after)
            tag = f"pp{pp}/tg{tg}/B{nreqs} s{s}"
            print(f"BENCH-SERVE {tag}: {json.dumps(agg)}", flush=True)

    print("BENCH-SERVE-DONE", flush=True)


if __name__ == "__main__":
    main()
