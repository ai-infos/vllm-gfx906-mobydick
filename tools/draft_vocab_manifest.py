#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Draft-vocab manifest: provenance + ids<->head pairing contract.

`build_draft_vocab.py count` writes `<ids>.provenance.json`;
`build_draft_vocab.py slice` folds it into `<work dir>/cat1_manifest.json`
alongside the artifacts, and `check_draft_vocab_list.py` re-verifies it here.
The serving stack verifies the same manifest at load time (see
`vllm/model_executor/models/qwen3_5_mtp.py::_check_draft_vocab_manifest`), so the
hash definitions below are a contract: change them only in lockstep, in both
files. A mismatch is loud, never silent.

Why: the draft head's *row order is the id mapping*, so an ids file swapped for
another list of the same length would otherwise serve wrong draft logits
silently — and nothing recorded which corpus a list came from.
"""

import hashlib
import json
import os

MANIFEST_NAME = "cat1_manifest.json"
MANIFEST_FORMAT = 1


def ids_sha1(ids) -> str:
    """Canonical hash of an id list (order-insensitive, dtype-independent)."""
    return hashlib.sha1(",".join(str(int(i)) for i in sorted(ids)).encode()).hexdigest()


def file_sha1(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    work_dir: str, ids, head_file: str, snapshot: str, provenance: dict | None
) -> dict:
    """Manifest written into the work dir by `slice`."""
    manifest = {
        "format": MANIFEST_FORMAT,
        "n_ids": len(ids),
        "ids_sha1": ids_sha1(ids),
        "head_file": head_file,
        "head_sha1": file_sha1(os.path.join(work_dir, head_file)),
        "snapshot": snapshot,
    }
    if provenance:
        manifest["provenance"] = provenance
    return manifest


def verify_manifest(work_dir: str, ids, check_head_file: bool = True):
    """Return (manifest, problems). manifest is None when there is no manifest.

    Checks the ids hash always (cheap) and the head-file hash when asked
    (reads the sliced head, ~0.3 s for 35 k rows).
    """
    path = os.path.join(work_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        return None, []
    with open(path) as fh:
        manifest = json.load(fh)
    problems = []
    want = manifest.get("ids_sha1")
    got = ids_sha1(ids)
    if want and got != want:
        problems.append(
            f"ids sha1 {got} != manifest {want}: the ids file and the sliced "
            f"head are not a matched pair"
        )
    head_file = manifest.get("head_file")
    if check_head_file and head_file and manifest.get("head_sha1"):
        got_h = file_sha1(os.path.join(work_dir, head_file))
        if got_h != manifest["head_sha1"]:
            problems.append(
                f"{head_file} sha1 {got_h} != manifest "
                f"{manifest['head_sha1']}: the sliced head is not the one this "
                f"ids list was built with"
            )
    return manifest, problems


def describe(manifest: dict) -> str:
    """One-line provenance summary for logs/reports."""
    prov = manifest.get("provenance") or {}
    files = (
        ", ".join(
            os.path.basename(f.get("path", "?"))
            for f in prov.get("corpus_files", [])[:3]
        )
        or "?"
    )
    return (
        f"{manifest.get('n_ids', '?')} ids, sha1 "
        f"{str(manifest.get('ids_sha1'))[:12]}, built "
        f"{prov.get('created', manifest.get('created', '?'))}, corpus {files} "
        f"({prov.get('corpus_tokens', 0) / 1e6:.1f}M tokens, holdout "
        f"{prov.get('holdout', '?')}, control tokens "
        f"{prov.get('control_tokens', '?')})"
    )
