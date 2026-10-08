#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright Kevin Read <me@kevin-read.com>
#
# FN-3 harness — serve the TINY Qwen3.8-Flash-Next (`qwen4_exp`) config on one
# MI50 so the QSA + PLE + hyperconnection + MoE path is runnable locally.
#
# Weights are random (`--load-format dummy`): this validates that the model
# *executes* in fp16 (every layer, both cache types, graphs, spec decode) and
# that arms can be A/B'd — it is NOT a quality gate. See
# docs/gfx906/_qsa_tiny_model.py and docs/gfx906/DEVLOG-qwen38-flash-qsa.md.
#
# Run from /local/git/vllm-gfx906-mobydick; logs to /local/tmp (persists).
#
#   _serve_qsa_tiny_gfx906.sh model          # fetch tokenizer + write config.json
#   _serve_qsa_tiny_gfx906.sh start [tag]    # serve; tag names the log file
#   _serve_qsa_tiny_gfx906.sh wait [tag]     # /health poll, 10 min
#   _serve_qsa_tiny_gfx906.sh stop           # SIGTERM + VRAM check
#
# Environment overrides:
#   PORT=8341             server port
#   DTYPE=                empty => "auto" (bf16 checkpoint -> fp16 on gfx906);
#                         set DTYPE=float16 to force, DTYPE=bfloat16 to A/B the
#                         bf16 QSA arm
#   MAXLEN=4096           --max-model-len
#   MAXSEQS=4             --max-num-seqs (and the capture-size ladder)
#   MBT=4096              --max-num-batched-tokens
#   UTIL=0.35             --gpu-memory-utilization
#   SPEC=                 e.g. '{"method":"mtp","num_speculative_tokens":1}'
#   EAGER=0               1 => --enforce-eager (no graph capture)
#   RUNNER=v2             v2 (default, upstream) | v1. **This model needs v2**:
#                         V1 passes query_start_loc/ngram_context as None and the
#                         PLE layer raises "PLE inputs were not prepared" (the
#                         plumbing lives in v1/worker/gpu/model_states).
#   PLUGINS=off           off => VLLM_PLUGINS="" (default). This venv has five
#                         stale vLLM general-plugin profilers installed
#                         (agdn/mtp1/pfk4/syv9/t1 phase, armed by files under
#                         /local/tmp/mtp1); vLLM loads every discovered plugin
#                         when VLLM_PLUGINS is unset. on => upstream behaviour.
#   EXTRA_ARGS=           appended verbatim
#
# Prefix caching is ON by default here, which puts the model in mamba
# 'align' mode -- the configuration that used to fault in
# precopy_mamba_align_fused_kernel (V2-MAMBA-1, fixed 2026-09-17). It is
# therefore also the regression rig for that fix: start the server, then send
# three prompts of 1344 / 2016 / 4031 tokens that share a prefix (each one a
# prefix-cache hit on the last):
#   .venv/bin/python /local/tmp/v2mamba/v2mamba_repro.py 8341 qsa-tiny 1343 2015 4030
# compare against EXTRA_ARGS="--no-enable-prefix-caching" (logprobs identical).
set -u
# Resolve the repo root from this script's location (docs/gfx906/...), so the
# bundle works from any checkout.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_DIR=/local/models/tiny-qwen38-flash
NAME=qsa-tiny
PORT="${PORT:-8341}"
DTYPE="${DTYPE:-}"
MAXLEN="${MAXLEN:-4096}"
MAXSEQS="${MAXSEQS:-4}"
MBT="${MBT:-4096}"
UTIL="${UTIL:-0.35}"
SPEC="${SPEC:-}"
EAGER="${EAGER:-0}"
RUNNER="${RUNNER:-v2}"
PLUGINS="${PLUGINS:-off}"
TAG="${2:-${TAG:-base}}"
LOG="/local/tmp/qsatiny-${TAG}.log"
PIDFILE="/local/tmp/qsatiny-${TAG}.pid"

case "$1" in
model)
  if [ ! -f "$MODEL_DIR/config.json" ]; then
    for f in tokenizer_config.json vocab.json merges.txt chat_template.jinja \
             generation_config.json preprocessor_config.json; do
      [ -f "$MODEL_DIR/$f" ] || hf download Qwen/Qwen3.8-Flash-Next "$f" \
          --local-dir "$MODEL_DIR" >/dev/null || exit 1
    done
  fi
  .venv/bin/python docs/gfx906/_qsa_tiny_model.py "$MODEL_DIR"
  ;;

start)
  [ -f "$MODEL_DIR/config.json" ] || { echo "run '$0 model' first"; exit 1; }
  if pgrep -f "vllm serve $MODEL_DIR" >/dev/null; then
    echo "already serving $MODEL_DIR"; exit 1
  fi
  dtype_args=(--dtype "${DTYPE:-auto}")
  [ "$EAGER" = 1 ] && dtype_args+=(--enforce-eager)
  spec_args=(); [ -n "$SPEC" ] && spec_args=(--speculative-config "$SPEC")
  runner_env=()
  case "$RUNNER" in
    v1) runner_env=(VLLM_USE_V2_MODEL_RUNNER=0) ;;
    v2) runner_env=(VLLM_USE_V2_MODEL_RUNNER=1) ;;
    default) runner_env=() ;;
  esac
  plugin_env=()
  [ "$PLUGINS" = off ] && plugin_env=(VLLM_PLUGINS=)
  env HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
      HF_HUB_OFFLINE=1 "${runner_env[@]}" "${plugin_env[@]}" \
    setsid nohup .venv/bin/vllm serve "$MODEL_DIR" \
      --load-format dummy --served-model-name "$NAME" --port "$PORT" \
      --tensor-parallel-size 1 \
      --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
      --max-num-batched-tokens "$MBT" --block-size 64 \
      --gpu-memory-utilization "$UTIL" \
      --compilation-config "{\"cudagraph_capture_sizes\":[$(seq -s, 1 "$MAXSEQS")]}" \
      "${dtype_args[@]}" "${spec_args[@]}" \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml \
      --reasoning-parser qwen3 \
      ${EXTRA_ARGS:-} \
      > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  echo "started tag=$TAG pid=$(cat "$PIDFILE") port=$PORT log=$LOG"
  ;;

wait)
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "healthy after ${i}x5s"; exit 0
    fi
    if ! kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
      echo "server died; tail of $LOG:"; tail -25 "$LOG"; exit 1
    fi
    sleep 5
  done
  echo "not healthy after 10 min; tail of $LOG:"; tail -25 "$LOG"; exit 1
  ;;

stop)
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null
    for i in $(seq 1 30); do
      kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break
      sleep 2
    done
    rm -f "$PIDFILE"
  fi
  pkill -f "vllm serve $MODEL_DIR" 2>/dev/null
  sleep 3
  echo "VRAM: $(rocm-smi --showmeminfo vram 2>/dev/null | grep -c Used) lines; \
used=$(rocm-smi --showmeminfo vram 2>/dev/null | grep 'GPU\[0\]' -A1 | tail -1)"
  rocm-smi --showmeminfo vram 2>/dev/null | grep Used
  ;;

*)
  sed -n '3,30p' "$0"; exit 2
  ;;
esac
