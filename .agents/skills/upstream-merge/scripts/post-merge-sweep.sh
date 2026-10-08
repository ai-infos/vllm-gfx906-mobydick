#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
#
# Post-merge static sweep for the gfx906 fork. Run from anywhere inside the repo.
#
#   .agents/skills/upstream-merge/scripts/post-merge-sweep.sh [--imports]
#
# Checks (cheap -> slower):
#   1. no conflict markers anywhere in tracked files
#   2. tree-wide ruff F821 (undefined names) + F811 (redefinitions)
#   3. in-memory syntax compile of vllm/ and tools/ (no .pyc writes, so it does
#      not trip over root-owned __pycache__ dirs)
#   4. duplicate top-level def/class scan over the files git flagged as conflicted
#   5. optional: import a set of merge-sensitive modules by name
#
# Exit non-zero on: conflict markers, any F821, a compile error, or a failed
# --imports import. F811 and duplicate defs are reported but do not fail (they
# are sometimes pre-existing upstream; check attribution before deleting).
set -u

REPO=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "not inside a git repo" >&2
  exit 2
}
cd "$REPO" || exit 2
PY="$REPO/.venv/bin/python"
RUFF=("$PY" -m ruff)
fail=0

echo "== 1. conflict markers"
if git grep -nE '^(<<<<<<<|>>>>>>>)' -- . >/tmp/pm-conflict-markers.txt 2>/dev/null; then
  echo "FAIL: conflict markers present:"; cat /tmp/pm-conflict-markers.txt; fail=1
else
  echo "ok"
fi

echo "== 2. ruff F821 (undefined names)"
if ! "${RUFF[@]}" check --select F821 --output-format concise vllm/ tests/ \
      >/tmp/pm-f821.txt 2>&1; then
  echo "FAIL: undefined names:"; cat /tmp/pm-f821.txt; fail=1
else
  echo "ok"
fi

echo "== 3. ruff F811 (redefinitions)"
if ! "${RUFF[@]}" check --select F811 --output-format concise vllm/ tests/ \
      >/tmp/pm-f811.txt 2>&1; then
  echo "WARN (check attribution; pre-existing upstream is possible):"
  cat /tmp/pm-f811.txt
else
  echo "ok"
fi

echo "== 4. syntax compile vllm/ tools/ (in memory)"
if ! "$PY" - >/tmp/pm-compile.txt 2>&1 <<'PYEOF'
import pathlib
import sys

bad = []
for root in ("vllm", "tools"):
    for p in sorted(pathlib.Path(root).rglob("*.py")):
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
        except SyntaxError as e:
            bad.append(f"{p}:{e.lineno}: {e.msg}")
        except (UnicodeDecodeError, OSError):
            pass
print("\n".join(bad) if bad else "ok")
sys.exit(1 if bad else 0)
PYEOF
then
  echo "FAIL: syntax errors:"; cat /tmp/pm-compile.txt; fail=1
else
  echo "ok"
fi

echo "== 5. duplicate top-level defs in conflicted files"
# /tmp/conflicts.txt is produced by the skill's step-0 command; fall back to the
# files git currently reports as unmerged if it is absent.
CONFLICTS=/tmp/conflicts.txt
if [ ! -s "$CONFLICTS" ]; then
  git diff --name-only --diff-filter=U > "$CONFLICTS" 2>/dev/null || true
fi
if [ -s "$CONFLICTS" ]; then
  while read -r f; do
    [ -f "$f" ] || continue
    dup=$(grep -oE '^(def|class) [A-Za-z_][A-Za-z0-9_]*' "$f" 2>/dev/null | sort | uniq -d)
    [ -n "$dup" ] && echo "WARN $f: $dup"
  done < "$CONFLICTS"
  echo "(done)"
else
  echo "no conflict list (run the skill's step-0 command to populate $CONFLICTS)"
fi

if [ "${1:-}" = "--imports" ]; then
  echo "== 6. import smoke (merge-sensitive modules)"
  # shellcheck disable=SC2016
  if ! HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE VLLM_PLUGINS=gfx906_fa \
      "$PY" - <<'PYEOF'
import vllm  # noqa: F401
import vllm.config.attention  # noqa: F401
import vllm.config.vllm  # noqa: F401
import vllm.envs  # noqa: F401
import vllm.model_executor.layers.fused_moe.oracle.int_wna16  # noqa: F401
import vllm.model_executor.layers.utils  # noqa: F401
import vllm.model_executor.kernels.linear.mixed_precision  # noqa: F401
import vllm.model_executor.layers.quantization.auto_awq  # noqa: F401
import vllm.model_executor.layers.quantization.auto_gptq  # noqa: F401
import vllm.models.minimax_m3.amd.ops.index_topk  # noqa: F401
import vllm.models.minimax_m3.common.ops.index_topk  # noqa: F401
import vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse  # noqa: F401
import vllm.v1.sample.ops.topk_topp_sampler  # noqa: F401
print("imports ok")
PYEOF
  then
    echo "FAIL: import smoke"; fail=1
  fi
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "post-merge sweep: PASS"
else
  echo "post-merge sweep: FAIL (see above)"
fi
exit "$fail"
