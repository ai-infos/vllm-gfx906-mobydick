#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""CAT-1: build a vocab-truncated draft head for Qwen3.8-27B-AWQ-INT4 MTP.

Adapted from syv-ai/qwen38-27b-rtx3090 prepare/build_draft_vocab.py (their
checkpoint is int8-packed; ours is unquantized bf16, so slicing is a plain row
index_select and the output tensor is mtp.draft_lm_head.weight). Runtime side:
vllm/model_executor/models/qwen3_5_mtp.py (CAT-1 port, same attribution).

Phases:
  count - tokenize MODEL-GENERATED text (reasoning+response+tool_calls; see
          docs/gfx906/CAT1-draft-vocab-corpus.md). --n is an UPPER BOUND: the
          final list is top-N minus specials plus all_special_ids (+ --extra-ids),
          so len(ids) <= n. Any N works under TP=2/4: ParallelLMHead pads to a
          multiple of 64 before sharding, so no divisibility constraint applies.
          Report held-out coverage at several N. Format scaffolding tokens are
          picked up by frequency (they dominate agentic output); use --extra-ids
          (JSON list of ints) to force-include any specific ids.
  slice - slice rows from lm_head.weight into a WORKING COPY (--work-dir):
          mtp_draft_vocab_ids.pt + model_extra_tensors.safetensors + updated
          index (loader only reads files listed in the weight_map).

Usage:
  .venv/bin/python tools/build_draft_vocab.py count --snapshot S \
      --corpus /local/tmp/mtp1/corpus/*.jsonl --n 131072 \
      --out-ids /local/tmp/mtp1/cat1_ids.json
  .venv/bin/python tools/build_draft_vocab.py slice --snapshot S \
      --work-dir /local/cache/.../cat1-draft --ids /local/tmp/mtp1/cat1_ids.json
"""
import argparse
import collections
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# --- draft-vocab manifest (shared contract; the serving stack verifies the
# same file at load - see tools/draft_vocab_manifest.py and
# vllm/model_executor/models/qwen3_5_mtp.py) ---------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_vocab_manifest import (  # noqa: E402
    MANIFEST_FORMAT,
    MANIFEST_NAME,
    build_manifest,
    ids_sha1,
)


def _is_jsonl(path):
    """JSONL by content, not by extension: a corpus file renamed without
    `.jsonl` used to be read as ONE plain-text blob (and, being a single item,
    could land entirely in the held-out split, silently producing a list of
    specials only)."""
    if path.endswith((".jsonl", ".json")):
        return True
    with open(path, errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if s:
                return s.startswith("{")
    return False


def texts_from(path, limit_bytes=20_000_000):
    """Yield generated-text strings. JSONL: reasoning/response/tool_calls only
    (prompt ignored - input tokens are never drafted). .txt: everything.

    `limit_bytes` guards against a runaway file but MUST NOT be silent: it is
    reported on stderr and named in the `count` summary, because a truncated
    corpus used to look identical to a complete one."""
    n = 0
    if _is_jsonl(path):
        for line in open(path, errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            parts = []
            for k in ("reasoning", "response"):
                if isinstance(r.get(k), str):
                    parts.append(r[k])
            tc = r.get("tool_calls")
            if isinstance(tc, list):
                parts += [t for t in tc if isinstance(t, str)]
            elif isinstance(tc, str):
                parts.append(tc)
            if isinstance(r.get("messages"), list):
                for m in r["messages"]:
                    if not isinstance(m, dict) or m.get("role") != "assistant":
                        continue
                    c = m.get("content")
                    if isinstance(c, str):
                        parts.append(c)
                    elif isinstance(c, list):
                        # Block-style content (pi/Qwen): silently dropping
                        # these used to yield an empty corpus for this shape.
                        parts += [b.get("text", "") for b in c
                                  if isinstance(b, dict) and b.get("type") == "text"
                                  and isinstance(b.get("text"), str)]
            t = "\n".join(parts)
            if t:
                sid = r.get("session_id")
                yield t, (sid if isinstance(sid, str) else None)
                n += len(t)
            if n > limit_bytes:
                print(f"WARNING: {path}: text truncated at {limit_bytes} bytes "
                      f"(raise --max-text-bytes or split the file)", file=sys.stderr)
                return
    else:
        yield open(path, errors="ignore").read(), None


def forced_ids(tok, extra_file, control_tokens=True):
    """Ids forced into the list regardless of corpus frequency.

    ``all_special_ids`` alone is NOT enough: the Qwen3.8 tokenizer carries the
    tool-call/thinking markup (`<tool_call>`, `</tool_call>`, `<tool_response>`,
    `<think>`, …) as *added* tokens that are not special, and a corpus built
    from parsed request logs (reasoning/response/tool_calls) can never contain
    them — yet the model emits them at every tool boundary, so a list without
    them forces a rejection there (measured: 98.0 % of a raw tool-call
    continuation covered without them, 100 % with). Default: include the whole
    added-token family. Pass --no-control-tokens to opt out.
    """
    s = set(tok.all_special_ids)
    if control_tokens:
        added = getattr(tok, "get_added_vocab", None)
        if callable(added):
            s |= {int(i) for i in added().values()}
    if extra_file:
        s |= {int(x) for x in json.load(open(extra_file))}
    return s


def coverage(counts, ids_set):
    tot = sum(counts.values())
    if not tot:
        return 0.0
    return sum(c for t, c in counts.items() if t in ids_set) / tot


def cmd_count(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.snapshot)
    counts = collections.Counter()
    held = collections.Counter()
    total = 0
    sessions: dict[str, int] = {}
    scheme = "record"

    def is_held(sid, j):
        # Spec: every 10th SESSION held out. Records inside a session are highly
        # correlated, so a record-level split flatters the number and changes
        # with file chunking; fall back to record-level only for corpora with
        # no session_id.
        nonlocal scheme
        if args.holdout == "record" or sid is None:
            return j % 10 == 0
        scheme = "session"
        return sessions.setdefault(sid, len(sessions)) % 10 == 0

    for path in args.corpus:
        for j, (t, sid) in enumerate(texts_from(path, args.max_text_bytes)):
            ids = tok(t, add_special_tokens=False).input_ids
            (held if is_held(sid, j) else counts).update(ids)
            total += len(ids)
    print(f"corpus tokens: {total} (holdout: {scheme}-level)")

    special = forced_ids(tok, args.extra_ids,
                         control_tokens=not args.no_control_tokens)
    if not counts:
        raise SystemExit(
            "count: nothing was counted (empty corpus, or every file landed in "
            "the held-out split). Refusing to emit a list of specials only.")
    top = [t for t, _ in counts.most_common() if t not in special]
    ids = sorted(set(top[: max(0, args.n - len(special))]) | special)
    js = set(ids)
    print(f"draft vocab: {len(ids)} ids "
          f"(held-out token coverage {coverage(held, js) * 100:.2f}%)")
    for n_try in (32768, 65536, 98304, 131072):
        s = set(t for t, _ in counts.most_common(n_try)) | special
        print(f"  coverage at N={n_try}: {coverage(held, s) * 100:.2f}%")
    json.dump(ids, open(args.out_ids, "w"))
    print(f"id list written to {args.out_ids}")
    provenance = {
        "format": MANIFEST_FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_ids": len(ids),
        "ids_sha1": ids_sha1(ids),
        "n_requested": args.n,
        "holdout": scheme,
        "control_tokens": not args.no_control_tokens,
        "forced_ids": sorted(special),
        "corpus_tokens": total,
        "corpus_files": [
            {"path": p, "bytes": os.path.getsize(p)} for p in args.corpus
        ],
        "held_out_coverage": round(coverage(held, js) * 100, 3),
        "snapshot": args.snapshot,
    }
    prov_path = args.out_ids + ".provenance.json"
    with open(prov_path, "w") as fh:
        json.dump(provenance, fh, indent=2)
    print(f"provenance written to {prov_path}")


def cmd_slice(args):
    ids = sorted(set(int(x) for x in json.load(open(args.ids))))
    idx_path = args.snapshot + "/model.safetensors.index.json"
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]
    head_shard = wm["lm_head.weight"]
    with safe_open(args.snapshot + "/" + head_shard, framework="pt") as f:
        w = f.get_slice("lm_head.weight")
        full = f.get_tensor("lm_head.weight")  # [248320, 5120] bf16
    assert tuple(full.shape) == (248320, 5120), f"unexpected lm_head {tuple(full.shape)}"
    del w
    ids_t = torch.tensor(ids, dtype=torch.int64)
    sub = full.index_select(0, ids_t).contiguous()
    print(f"draft head: {tuple(sub.shape)} {sub.dtype} "
          f"{sub.numel() * 2 / 1e6:.0f} MB")

    work = args.work_dir.rstrip("/") + "/"
    os.makedirs(work, exist_ok=True)
    # working copy: symlink the shards (read-only use), real files for extras.
    # NEVER symlink a file this script overwrites: json.dump/torch.save open
    # the path with O_TRUNC and would write THROUGH the symlink into the
    # original snapshot (this actually corrupted the NFS mirror once).
    overwritten = {"model.safetensors.index.json", "mtp_draft_vocab_ids.pt",
                   "model_extra_tensors.safetensors"}
    for fn in sorted(os.listdir(args.snapshot)):
        src = args.snapshot + "/" + fn
        dst = work + fn
        if os.path.islink(dst):
            os.unlink(dst)  # a stale symlink would be written through
        if not os.path.exists(dst):
            if fn in overwritten:
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)
    tensors = {}
    extra = "model_extra_tensors.safetensors"
    if os.path.exists(work + extra):
        with safe_open(work + extra, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    tensors["mtp.draft_lm_head.weight"] = sub
    save_file(tensors, work + extra, metadata={"format": "pt"})
    wm["mtp.draft_lm_head.weight"] = extra
    json.dump(idx, open(work + "model.safetensors.index.json", "w"), indent=2)
    torch.save(ids_t, work + "mtp_draft_vocab_ids.pt")
    # Manifest: lets the server log which corpus a list came from and catch a
    # mismatched ids/head pair at load (the row order *is* the id mapping; a
    # count mismatch fails on shape, an equal-count swap would otherwise be
    # silent). Optional: a work dir without it keeps working exactly as before.
    prov_path = args.ids + ".provenance.json"
    provenance = None
    if os.path.exists(prov_path):
        with open(prov_path) as fh:
            provenance = json.load(fh)
    manifest = build_manifest(work, ids, extra, args.snapshot, provenance)
    with open(work + MANIFEST_NAME, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"manifest written to {work + MANIFEST_NAME}")
    print(f"done: {work} (index updated, ids .pt written)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("count")
    pc.add_argument("--snapshot", required=True)
    pc.add_argument("--corpus", nargs="+", required=True)
    pc.add_argument("--n", type=int, default=131072)
    pc.add_argument("--extra-ids", default=None)
    pc.add_argument("--no-control-tokens", action="store_true",
                    help="do not force the tokenizer's added-token family "
                         "(tool-call/thinking markup) into the list; only "
                         "all_special_ids are forced")
    pc.add_argument("--holdout", choices=("session", "record"), default="session",
                    help="held-out split granularity (spec: session-level)")
    pc.add_argument("--max-text-bytes", type=int, default=20_000_000,
                    help="per-file text cap; hitting it is warned about and "
                         "truncates the corpus (default 20 MB)")
    pc.add_argument("--out-ids", required=True)
    pc.set_defaults(fn=cmd_count)
    ps = sub.add_parser("slice")
    ps.add_argument("--snapshot", required=True)
    ps.add_argument("--work-dir", required=True)
    ps.add_argument("--ids", required=True)
    ps.set_defaults(fn=cmd_slice)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
