#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright Kevin Read <me@kevin-read.com>
#
# QSA-FN-2 / QSA-FN-8 — serve recipe for the REAL Qwen3.8-Flash-Next
# (`qwen4_exp` QSA) on gfx906, for a tester with enough VRAM.
#
# This box cannot load it (~120 B params of MoE: W4A16 ≈ 60 GB, plus a PLE
# ngram table the CDNA recipe offloads 60 GB of), so **the flag set below is
# derived, not validated end-to-end here**: the V2 requirement, the
# prefix-caching workaround, the parsers and the MTP arm were validated on the
# tiny harness (`_qsa_tiny_model.py`, FN-3); the rest follows the MI210
# production recipe with the gfx906 deltas called out. Read
# docs/gfx906/DEVLOG-qwen38-flash-qsa.md and RECON-qwen38-flash-qsa.md §2.
#
#   _serve_qsa_flash_gfx906.sh start <model-dir> [tag]
#   _serve_qsa_flash_gfx906.sh wait  [tag]
#   _serve_qsa_flash_gfx906.sh stop  [tag]
#   _serve_qsa_flash_gfx906.sh report       # what to send back
#
# Deltas vs the CDNA (gfx90a) production launch, and why:
#   --dtype float16            gfx906 has no native bf16: the checkpoint is
#                              bf16 and everything downstream is fp16 (the
#                              reported failure was this fallback meeting the
#                              old bf16-only guards). Explicit beats implicit.
#   VLLM_USE_V2_MODEL_RUNNER=1 Qwen4Exp hides the PLE/ngram inputs behind the
#                              V2 model states; on V1 the PLE layer raises
#                              "PLE inputs were not prepared". (Our other model
#                              recipes pin V1 pending DFL2-2 — do NOT copy that
#                              pin here.)
#   (prefix caching ON)        the default. It forces mamba_cache_mode='align',
#                              whose V2 pre-copy kernel used to IMA on any
#                              model with a KV group finer than the mamba block
#                              size (V2-MAMBA-1, fixed 2026-09-17). On a build
#                              predating that fix, append
#                              --no-enable-prefix-caching to EXTRA_ARGS.
#   (no --mamba-cache-dtype)   the CDNA recipe pins bf16; on gfx906 leave it
#                              auto (= model dtype = fp16).
#   --max-model-len 262144     native, un-scaled RoPE. Do NOT add YaRN or go
#                              past 262144 (the CDNA notes: rope scaling
#                              degrades quality at *all* positions).
#   tools/reasoning parsers    qwen3_xml + qwen3, as in the CDNA recipe.
#
# Not in this script: `--kv-cache-memory` (CDNA-specific pinning; leave vLLM to
# size the pool) and `--trust-remote-code` (not needed for this checkpoint).
# Unvalidated knobs are env-overridable; see the variable block.
#
# Known-good config (Tester-validated, 2026-09-22, 4x MI50 32 GB, PCIe-only):
#   TP=4 GPUTIL=0.91 MAXLEN=147456 MAXSEQS=3 MBT=4096 SPEC k=3
#   46.8 t/s at B=1; without MTP 25.4 t/s. The PLE ngram table must live in host
#   RAM (their mmap offload, PR #2) -- that, not the KV dtype, is what makes the
#   model fit on 32 GB cards. Two findings from that run are in ROADMAP QSA-FN-2b:
#   the official Qwen chat template raises on an all-tool-result tail (pass a fixed
#   template with --chat-template), and enforce_eager is a no-op inside
#   --speculative-config on this build.
#
# Environment overrides:
#   TP=2            tensor-parallel size (needs >= ~64 GB across the cards)
#   GPUTIL=0.90     --gpu-memory-utilization; lower it if init OOMs (the
#                   inductor prefill buffer needs headroom - see the dense-27B
#                   note in the root AGENTS.md)
#   MAXLEN=262144   --max-model-len
#   MAXSEQS=4       --max-num-seqs
#   MBT=4096        --max-num-batched-tokens
#   SPEC='{"method":"mtp","num_speculative_tokens":3}'
#                   empty string disables speculation
#   OFFLOAD_GB=0    --cpu-offload-gb (CDNA used 60 for `ngram_embedding`)
#   OFFLOAD_PARAMS=ngram_embedding
#   PORT=8321       server port
#   EXTRA_ARGS=     appended verbatim
set -u
# Resolve the repo root from this script's location (docs/gfx906/...), so the
# bundle works from any checkout.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TP="${TP:-2}"
GPUTIL="${GPUTIL:-0.90}"
MAXLEN="${MAXLEN:-262144}"
MAXSEQS="${MAXSEQS:-4}"
MBT="${MBT:-4096}"
SPEC_DEFAULT='{"method":"mtp","num_speculative_tokens":3}'
SPEC="${SPEC-$SPEC_DEFAULT}"
OFFLOAD_GB="${OFFLOAD_GB:-0}"
OFFLOAD_PARAMS="${OFFLOAD_PARAMS:-ngram_embedding}"
PORT="${PORT:-8321}"
MODEL="${2:-}"

case "$1" in
start) TAG="${3:-${TAG:-qsa}}" ;;
*)     TAG="${2:-${TAG:-qsa}}" ;;
esac
LOG="/local/tmp/qsaflash-${TAG}.log"
PIDFILE="/local/tmp/qsaflash-${TAG}.pid"

case "$1" in
start)
  [ -n "$MODEL" ] && [ -e "$MODEL" ] || {
    echo "usage: $0 start <model-dir> [tag]"; exit 2; }
  # capture ladder: multiples of k+1 == spec tokens + 1, up to max_seqs*(k+1)
  if [ -n "$SPEC" ]; then
    k=$(printf '%s' "$SPEC" | sed -n 's/.*num_speculative_tokens"*: *\([0-9]*\).*/\1/p')
    step=$(( ${k:-3} + 1 ))
  else
    step=1
  fi
  sizes=$(seq -s, "$step" "$step" $((MAXSEQS * step)))
  offload=()
  [ "$OFFLOAD_GB" != 0 ] && offload=(--cpu-offload-gb "$OFFLOAD_GB" \
                                      --cpu-offload-params "$OFFLOAD_PARAMS")
  spec_args=(); [ -n "$SPEC" ] && spec_args=(--speculative-config "$SPEC")

  env HIP_VISIBLE_DEVICES=$(seq -s, 0 $((TP - 1))) \
      FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HF_HUB_OFFLINE=1 \
      VLLM_USE_V2_MODEL_RUNNER=1 VLLM_PLUGINS= \
    setsid nohup .venv/bin/vllm serve "$MODEL" \
      --served-model-name q38fn --port "$PORT" \
      --tensor-parallel-size "$TP" --enable-expert-parallel \
      --dtype float16 --gpu-memory-utilization "$GPUTIL" \
      --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
      --max-num-batched-tokens "$MBT" --block-size 64 \
      --compilation-config "{\"cudagraph_capture_sizes\":[$sizes]}" \
      "${spec_args[@]}" "${offload[@]}" \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml \
      --reasoning-parser qwen3 --disable-custom-all-reduce \
      ${EXTRA_ARGS:-} \
      > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  echo "started tag=$TAG pid=$(cat "$PIDFILE") port=$PORT model=$MODEL"
  echo "log=$LOG   (first load of a 60 GB checkpoint: tens of minutes)"
  ;;

wait)
  for i in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "healthy after ${i}x5s"
      grep -E "GPU KV cache size|Available KV cache memory|Resolved architecture" "$LOG" | tail -3
      exit 0
    fi
    if [ -f "$PIDFILE" ] && ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "server died; tail of $LOG:"; tail -30 "$LOG"; exit 1
    fi
    sleep 5
  done
  echo "not healthy after 30 min; tail of $LOG:"; tail -30 "$LOG"; exit 1
  ;;

stop)
  [ -f "$PIDFILE" ] && { kill "$(cat "$PIDFILE")" 2>/dev/null; rm -f "$PIDFILE"; }
  pkill -f "vllm serve $MODEL" 2>/dev/null
  sleep 5
  rocm-smi --showmeminfo vram 2>/dev/null | grep Used
  ;;

report)
  cat <<'REPORT'
Send back (the FN-8 gate — without these the recipe stays unvalidated):

1. your launch line, plus `rocm-smi` / card model / VRAM per card;
2. did it load: the `Resolved architecture`, `GPU KV cache size` and
   `Available KV cache memory` lines from the server log;
3. one short greedy completion and one ~2000-token prompt: TTFT and decode
   tok/s (state the config: pp/tg, batch size, prefix caching on/off);
4. one long-context (e.g. 100k+) request: does it complete, is the output
   coherent, and is a needle retrievable (start/middle/end);
5. which of these happened FIRST if it broke: OOM at init / a dtype or
   NotImplementedError / a kernel fault (paste the kernel name and grid) /
   garbage-but-running output;
6. if you enabled speculation: the acceptance rate and decode t/s with and
   without, same prompts, interleaved.
REPORT
  ;;

*)
  sed -n '3,50p' "$0"; exit 2
  ;;
esac
