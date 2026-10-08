#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
# rocprofv3 sqlite3 test (2026-09-11): per-kernel attribution of a post-FIX
# 8k prefill (ttft-prefill-stall.md §13.1 follow-up — the known-good rocprof
# pattern: sqlite output, small focused trace). In-process LLM, TP=2, mtp3,
# pp=1024, one 8192-token prefill (~8 steps). rocprofv3 follows worker child
# processes via inherited env.
set -uo pipefail
source ~/env-rocm-7.14-gfx906.sh
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HF_HUB_OFFLINE=1
export DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
cd /local/git/vllm-gfx906-mobydick

OUT=/local/tmp/b4/rp3
mkdir -p "$OUT" && cd "$OUT"
rm -f *.db *.db-* 2>/dev/null

PROF_PP="${PROF_PP:-1024}"
KB_PREFILL="${KB_PREFILL:-8192}"

rocprofv3 -d rp3out --output-format rocpd -- \
  env PROF_PP="$PROF_PP" KB_PREFILL="$KB_PREFILL" \
  env KB_FLUSH_WAIT=20 /local/git/vllm-gfx906-mobydick/.venv/bin/python /local/tmp/b4/kb_probe.py \
  > rp3_run.log 2>&1

rc=$?
echo "rocprofv3 rc=$rc"
ls -la "$OUT"
echo "---- tail of run log ----"
tail -20 rp3_run.log
