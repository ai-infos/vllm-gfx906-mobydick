#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Validate an MTP draft-vocab id list (and the work dir it was sliced into).

Checks, cheapest first:
  1. MUST_HAVE_TOKENS present (Qwen3.8 control/markup family — see
     docs/gfx906/CAT1-corpus-build.md: parsed logs cannot contain these, so
     they must be forced, or the drafter can never propose them).
  2. ids sorted + unique.
  3. work-dir consistency: mtp_draft_vocab_ids.pt == the JSON list; the index
     adds exactly one key; the draft-head rows are byte-identical to
     lm_head.weight[ids] (catches an off-by-one row misalignment).
  4. optional: coverage of a *raw* greedy continuation captured with
     `logprobs` from the serving stack (the only offline metric that sees the
     markup loss; corpus occurrence-coverage cannot).

Usage:
  .venv/bin/python tools/check_draft_vocab_list.py --ids cat1_ids.json \
      --work-dir /local/models/cat1_work --snapshot /path/to/checkpoint \
      --continuation continuation.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_vocab_manifest import describe, verify_manifest  # noqa: E402

# Qwen3.8 control/markup family (added tokens, ids 248044-248076). `all_special_ids`
# covers only 9 of these; the tool-call ones are emitted at every tool boundary.
MUST_HAVE_TOKENS = {
    248044: "<|endoftext|>",
    248045: "<|im_start|>",
    248046: "<|im_end|>",
    248058: "<tool_call>",
    248059: "</tool_call>",
    248066: "<tool_response>",
    248067: "</tool_response>",
    248068: "<think>",
    248069: "</think>",
    248053: "<|vision_start|>",
    248054: "<|vision_end|>",
    248056: "<|image_pad|>",
    248057: "<|video_pad|>",
    248070: "<|audio_start|>",
    248071: "<|audio_end|>",
    248076: "<|audio_pad|>",
}


def fail(msg):
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="ids JSON list written by `count`")
    ap.add_argument("--work-dir", default=None, help="dir written by `slice`")
    ap.add_argument(
        "--snapshot",
        default=None,
        help="source checkpoint (needed for the row-identity check)",
    )
    ap.add_argument(
        "--continuation",
        default=None,
        help="serving response JSON captured with logprobs",
    )
    ap.add_argument(
        "--tokenizer", default=None, help="tokenizer path (default: --snapshot)"
    )
    args = ap.parse_args()

    with open(args.ids) as fh:
        ids = json.load(fh)
    S = set(ids)
    print(f"list: {len(ids)} ids, min {min(ids)}, max {max(ids)}")

    missing = {i: t for i, t in MUST_HAVE_TOKENS.items() if i not in S}
    if missing:
        fail(
            "missing must-have tokens: "
            + ", ".join(f"{t} ({i})" for i, t in missing.items())
            + "  -- rebuild with the control-token family forced"
        )
    print(f"must-have tokens: all {len(MUST_HAVE_TOKENS)} present")

    if sorted(set(ids)) != list(ids):
        fail("ids are not sorted/unique")
    print("ids sorted and unique")

    if args.work_dir:
        import torch
        from safetensors import safe_open

        manifest, problems = verify_manifest(args.work_dir, ids)
        if manifest is None:
            print("manifest: absent (older work dir - no provenance/pairing check)")
        elif problems:
            fail("manifest mismatch: " + "; ".join(problems))
        else:
            print("manifest: OK - " + describe(manifest))
        pt_path = os.path.join(args.work_dir, "mtp_draft_vocab_ids.pt")
        pt = torch.load(pt_path, map_location="cpu")
        if {int(x) for x in pt.tolist()} != S:
            fail("work-dir .pt != --ids list")
        print("work-dir .pt matches the list")
        index_path = os.path.join(args.work_dir, "model.safetensors.index.json")
        with open(index_path) as fh:
            idx = json.load(fh)["weight_map"]
        for name in ("mtp.draft_lm_head.weight",):
            if name not in idx:
                fail(f"index does not map {name}")
        if args.snapshot:
            snap_index = os.path.join(args.snapshot, "model.safetensors.index.json")
            with open(snap_index) as fh:
                snap_idx = json.load(fh)["weight_map"]
            added = {k: v for k, v in idx.items() if k not in snap_idx}
            changed = {k for k in snap_idx if k in idx and snap_idx[k] != idx[k]}
            if changed or set(snap_idx) - set(idx):
                fail(f"index not a pure addition (changed={len(changed)})")
            print(f"index adds exactly: {added}")
            with safe_open(
                os.path.join(args.snapshot, snap_idx["lm_head.weight"]), framework="pt"
            ) as f:
                full = f.get_tensor("lm_head.weight")
            extra = os.path.join(args.work_dir, "model_extra_tensors.safetensors")
            with safe_open(extra, framework="pt") as f:
                sub = f.get_tensor("mtp.draft_lm_head.weight")
            ref = full.index_select(0, torch.tensor(ids, dtype=torch.int64))
            if sub.shape != ref.shape or not torch.equal(sub, ref):
                fail("draft-head rows are NOT byte-identical to lm_head[ids]")
            print(f"draft-head rows == lm_head[ids]  ({tuple(sub.shape)} {sub.dtype})")

    if args.continuation:
        from transformers import AutoTokenizer

        tok_path = args.tokenizer or args.snapshot
        if not tok_path:
            fail("--continuation needs --tokenizer (or --snapshot) to detokenize")
        tok = AutoTokenizer.from_pretrained(tok_path)
        with open(args.continuation) as fh:
            d = json.load(fh)
        ch = d["response"]["choices"][0] if "response" in d else d["choices"][0]
        lp = ch.get("logprobs") or {}
        toks = lp.get("tokens") or []
        text = tok.convert_tokens_to_string(toks) if toks else ch["text"]
        pos = tok(text, add_special_tokens=False).input_ids
        miss = [t for t in pos if t not in S]
        print(
            f"raw-continuation coverage: {100 * (1 - len(miss) / len(pos)):.2f}% "
            f"({len(miss)}/{len(pos)} positions outside the list)"
        )
        for t in sorted(set(miss))[:10]:
            print(f"    missed: {tok.decode([t])!r} ({t})")
    print("OK")


if __name__ == "__main__":
    main()
