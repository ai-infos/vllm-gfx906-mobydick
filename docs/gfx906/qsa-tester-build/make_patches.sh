#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright Kevin Read <me@kevin-read.com>
#
# Build the QSA-FN-8 tester bundle (patch set + harness + README) from the
# gfx906/qsa-fn branch. Reproducible: everything is derived from committed
# state, nothing is hand-copied.
#
#   docs/gfx906/qsa-tester-build/make_patches.sh [outdir]
#
# Default outdir: /local/tmp/qsa-tester-build (persists across reboots).
# The patch set applies, in order, to the branch base (see README).
set -eu
REPO=/local/git/vllm-gfx906-mobydick
BRANCH=${BRANCH:-gfx906/qsa-fn}
BASE=${BASE:-gfx906/v0.29.0}
OUT=${1:-/local/tmp/qsa-tester-build}
SRC=$(cd "$(dirname "$0")" && pwd)
cd "$REPO"

MERGE_BASE=$(git merge-base "$BASE" "$BRANCH")
mkdir -p "$OUT/patches"

# 1. fp16 QSA enablement (QSA-FN-1): the reported failure.
git diff "$MERGE_BASE" d3e14e3edf -- \
  vllm/models/qwen4_exp tests/models/qwen4_exp \
  > "$OUT/patches/0001-qsa-fp16-enablement.patch"

# 2. V2-MAMBA-1: the align-mode seed fix (needed for prefix caching).
git diff d3e14e3edf 213c306987 -- \
  vllm/v1/worker/gpu/model_states/mamba_hybrid.py \
  tests/v1/worker/test_mamba_hybrid_model_state.py \
  tests/kernels/mamba \
  > "$OUT/patches/0002-v2-mamba-align-seed.patch"

# 3. Tiled QSA indexer, fp16-gated (QSA-FN-4).
git diff 213c306987 01ebad98c0 -- \
  vllm/models/qwen4_exp tests/models/qwen4_exp \
  benchmarks/kernels/gfx906/probe_fn4_indexer_route.py \
  > "$OUT/patches/0003-qsa-tiled-indexer.patch"

# 4. Harness: tiny model + both serve recipes, at their final state.
git diff "$MERGE_BASE" "$BRANCH" -- \
  docs/gfx906/_qsa_tiny_model.py docs/gfx906/_serve_qsa_tiny_gfx906.sh \
  docs/gfx906/_serve_qsa_flash_gfx906.sh \
  > "$OUT/patches/0004-qsa-harness.patch"

cp "$SRC/README.md" "$SRC/HANDOVER.md" "$OUT/"

# Offline tokenizer + the tiny config, so the smoke rig needs no HF access.
mkdir -p "$OUT/tiny-tokenizer"
cp /local/models/tiny-qwen38-flash/{tokenizer_config.json,vocab.json,merges.txt,\
chat_template.jinja,generation_config.json,preprocessor_config.json,config.json} \
  "$OUT/tiny-tokenizer/" 2>/dev/null || true
{
  echo "branch:      $BRANCH"
  echo "head:        $(git rev-parse "$BRANCH")"
  echo "base:        $BASE ($(git rev-parse "$BASE"))"
  echo "merge-base:  $MERGE_BASE"
  echo "generated:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "patches:"
  for p in "$OUT"/patches/*.patch; do
    printf '  %-40s %s lines\n' "$(basename "$p")" "$(wc -l < "$p")"
  done
} > "$OUT/BUILD-INFO.txt"

tar -C "$(dirname "$OUT")" -czf "$OUT.tgz" "$(basename "$OUT")"
cat "$OUT/BUILD-INFO.txt"
echo "bundle: $OUT  ($(du -sh "$OUT" | cut -f1)), tarball: $OUT.tgz"
