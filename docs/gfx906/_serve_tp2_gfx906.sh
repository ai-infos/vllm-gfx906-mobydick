#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright Kevin Read <me@kevin-read.com>
#
# gfx906 TP=2 serve helper for bench sessions (long-context sweeps etc.).
# Runs from /local/git/vllm-gfx906-mobydick; logs to /local/tmp (persists
# across reboots).
#
#   _serve_tp2_gfx906.sh start <tag> <snap> <served-name> <max-len> <tool> <reason>
#   _serve_tp2_gfx906.sh wait <tag>                 # /health poll, ~25 min
#   _serve_tp2_gfx906.sh stop <tag>                 # SIGTERM + VRAM release
#
# Flags (the 2026-08-29 long-context recipe): TP=2, float16,
# max-num-seqs 4, max-num-batched-tokens 4096, trimmed capture [1,2,3,4],
# prefix caching OFF, generation-config auto. KV budget via KVBYTES
# (default 6 GiB; Qwen3.8-27B at 256k max-len needs >= 8.09 GiB — use
# 10737418240). No speculative config: prefill/TTFT benchmarks do not
# need it (spec only shapes decode).
#
# EXTRA_SERVE_ENV: optional space-separated VAR=val pairs prepended to the
# server env (e.g. EXTRA_SERVE_ENV="GFX906_FA_LEGACY=0" for FA A/B bakes).
set -u
cd /local/git/vllm-gfx906-mobydick

case "$1" in
start)
  tag="$2"; snap="$3"; name="$4"; maxlen="$5"; tool="$6"; reason="$7"
  # 0.29.0: upstream defaults to Model Runner V2 for ALL models; V2 is still
  # unvalidated on gfx906 (it wedged at init on the Y16 smoke), so pin V1 until
  # DFL2-2 brings V2 up. Every fork recipe must carry this.
  env HIP_VISIBLE_DEVICES=0,1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
      VLLM_USE_V2_MODEL_RUNNER=0 \
      HF_HUB_OFFLINE=1 ${EXTRA_SERVE_ENV:-} \
    setsid nohup .venv/bin/vllm serve "$snap" \
      --served-model-name "$name" \
      --tensor-parallel-size 2 --dtype float16 \
      --max-model-len "$maxlen" --max-num-seqs 4 \
      --max-num-batched-tokens 4096 \
      --kv-cache-memory-bytes "${KVBYTES:-6442450944}" \
      --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4]}' \
      --no-enable-prefix-caching \
      --enable-auto-tool-choice --tool-call-parser "$tool" \
      --reasoning-parser "$reason" \
      --generation-config auto \
      > "/local/tmp/lcbench_${tag}_server.log" 2>&1 < /dev/null &
  echo "launched $tag (pid $!) -> /local/tmp/lcbench_${tag}_server.log"
  ;;
wait)
  tag="$2"
  for i in $(seq 1 300); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
    if [ "$code" = "200" ]; then echo "$tag READY after ~$((i*5))s"; exit 0; fi
    if ! pgrep -f "vllm serve" > /dev/null; then
      echo "$tag DIED during boot:"; tail -25 "/local/tmp/lcbench_${tag}_server.log"; exit 1
    fi
    sleep 5
  done
  echo "$tag TIMEOUT waiting for /health"; tail -25 "/local/tmp/lcbench_${tag}_server.log"; exit 1
  ;;
stop)
  tag="$2"
  pkill -TERM -f "vllm serve" 2>/dev/null
  for i in $(seq 1 120); do
    pgrep -f "vllm serve" > /dev/null || break
    sleep 5
  done
  if pgrep -f "vllm serve" > /dev/null; then
    echo "$tag: still alive after 10 min SIGTERM wait — NOT escalating (TP=2 rule)"; exit 1
  fi
  for i in $(seq 1 60); do
    vram=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Total Used" | awk '{print $NF}' | sort -n | tail -1)
    [ -n "$vram" ] && [ "$vram" -lt 100000000 ] && break
    sleep 5
  done
  echo "$tag torn down; max VRAM used now: ${vram:-?} B"
  rocm-smi --showuse --showmeminfo vram 2>/dev/null | grep -E "GPU use|Total Used"
  ;;
esac
