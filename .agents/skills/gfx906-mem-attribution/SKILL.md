---
name: gfx906-mem-attribution
description: "Attribute peak GPU-memory transients (startup / first-prefill OOMs) on this repo's vLLM fork (2x MI50, ROCm 7.14). Use when an engine OOMs on a first-prefill chunk or at startup and the owner is unclear (inductor vs custom FA vs GEMM workspace vs per-impl buffers), when a transient scales with the chunk or differs eager-vs-compiled, when an OOM traceback names an innocent-looking allocation frame (it is the last straw, not the owner), or when memory_summary/snapshot/kineto come back useless on this torch. Validated 2026-08-27/28 (per-impl q_pad OOM root cause)."
---

# gfx906 memory attribution

Who owns a peak GPU-memory transient on the MI50 vLLM fork, when
allocation stack traces don't work on this torch build. Canonical
worked case: `docs/gfx906/DEVLOG-muse-glimmer.md` round 4; probe:
`docs/gfx906/_probe_mem_attribution_gfx906.py`.

## Core principle

**The OOM traceback's allocation frame is the last straw, not the
owner** — worked case: traceback pointed at inductor's `gptq_gemm`
(508 MiB) while the real owner was our FA's per-impl prefill buffers
(13.3 GiB). Never attribute an OOM from its traceback alone.

**Owner by elimination:** run the same in-process config with one
candidate removed per arm, take deltas:
- `custom` arm: gfx906 FA + inductor (default)
- `rocm` arm: `attention_config` pinned `ROCM_ATTN` (Triton FA) + inductor
- `eager` arm: `enforce_eager` + gfx906 FA (removes inductor)

Δ(custom − rocm) = FA-attributable; Δ(eager − custom) = inductor cost
(negative = inductor *saves*, e.g. memory-planner reuse).

Cap the KV cache explicitly (`kv_cache_memory_bytes`, e.g. 0.5 GiB
TP=1) so `transient = max_memory_allocated(after first prefill) −
allocated(after init)` is isolated from pool size. Pick a prompt just
over one `max_num_batched_tokens` chunk (PP=4097 with bt4096) so the
first generate executes exactly the OOM-inducing prefill.

Run: `ARM=custom|rocm|eager TP=1 PP=4097 ... .venv/bin/python
docs/gfx906/_probe_mem_attribution_gfx906.py`. Prints `PROBE: init`,
`PROBE: SURVIVED|FAILED`, `PROBE: peak_transient=...`, `PROBE: RESULT`.

## Per-layer hooks

`PROBE_PER_LAYER=1` wraps the FA backend stack and prints
`memory_allocated()` deltas per attention call: whole `forward`,
`forward_paged` wrapper, raw `ext.forward` binding, `_ensure_*`
buffer-growth calls (`PROBE-ENS`), first-3-forward snapshot diff
(`PROBE_SNAPDIFF`). Read them as:
- **monotone +N MiB per call, never freed** → per-call or per-impl
  buffer retention; multiply by layer count (52 for Muse-Glimmer).
  The worked root cause: `+256 MiB/call × 52 = 13.3 GiB` — the q_pad
  buffer was an instance attribute and v1 creates **one backend impl
  per attention layer**, so each impl grew its own prefill-sized
  buffer on first prefill.
- binding flat, wrapper +N, whole-forward +M → retention is between
  those layers (Python-side bookkeeping, not the kernel).
- `PROBE-ENS` +N on call 0 only → one-shot grow (fine); +N on every
  call → the bug.

## Bisection order (when a delta needs a name)

Outer → inner: model `forward` → backend `forward` → `forward_paged`
wrapper → `_ensure_forward_buffers` → `ext.forward`
(`csrc/gfx906_fa/gfx906_fa.cpp`). Hook each, diff
`memory_allocated()` before/after. The C++ hipMalloc-free path uses
PyTorch's caching allocator, so `memory_allocated()` sees it;
`torch.cuda.memory._snapshot` gives per-block sizes for confirming a
shape (`size/128 = cols`, `÷2 = fp16 rows`).

## In-process fresh-compile env (ALL of these, together)

This torch (≥2.10-dev) writes `async_compile.wait()` into every
generated wrapper, and its process pools fail on this box: FORK is
HSA-unsafe after the parent inits HSA; SPAWN children fail HSA init
on wedge-accumulated boots. Required for any in-process run that
compiles new shapes:

```bash
cd /local/git/vllm-gfx906-mobydick   # no env sourcing needed: /opt/rocm is default (HK-1)
env VLLM_USE_AOT_COMPILE=0 \
    TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0 \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HF_HUB_OFFLINE=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 HIP_VISIBLE_DEVICES=0 \
    .venv/bin/python <script>
```

- `VLLM_USE_AOT_COMPILE=0` — default is ON for torch ≥2.10
  (`vllm/envs.py` `use_aot_compile()`); the out-of-process AOT
  worker crashes in-process on degraded boots.
- `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` — the rblock variant-compile
  path crashes (`JITFunction.fn is None`).
- The probe also monkeypatches `AsyncCompile.process_pool` →
  `ThreadPoolExecutor(max_workers=2)` (`PROBE_THREAD_COMPILE`,
  default 1) so compilation runs in parent-process threads where HSA
  works. A non-probe script needs the same patch (see the probe's
  header) or a warm on-disk inductor/AOT cache — with a warm cache
  none of this triggers, but keep the env for safety.

**Bash env-propagation trap:** `VAR=1 cmd1 && cmd2` applies VAR only
to cmd1. Use `env VAR=1 setsid nohup ... > log 2>&1 < /dev/null &`.

## Dead ends on this build — do not retry

- `torch.cuda.memory_summary()` / `memory_snapshot()` stack capture:
  only `?:0` unwind frames, no Python/C++ user frames.
- Kineto / `torch.profiler` chrome traces: **no memory events at all**
  (categories: `None`, `cpu_op`, `Trace` only). GPU-event timestamps
  are not wall-aligned (repo AGENTS.md note).
- Re-running the same OOM to "catch" a different last-straw frame:
  the frame is where memory ran out; it moves with allocator state.

## Interpretation

- The last straw splits with TP: TP=2's 254 MiB ×2 = TP=1's 508 MiB →
  N-split GEMM buffer, shape ≈ `[4096, 32512]` fp16.
- Buffer ∝ Sq_pad ∝ chunk explains "transient scales linearly with
  the chunk"; a per-chunk OOM that survives at half the chunk is a
  sizing bug, not out-of-model.
- `attn_metadata.seq_lens.shape[0]` can be padded to `max_num_seqs`
  while the real request count is `block_table.shape[0]` — a buffer
  sized from the former is `max_num_seqs`× bigger than it looks.
- Acceptance test after a fix: re-run the same arm; survival at the
  same cap + a transient matching "core + one-shot grows" (worked
  case: 3.785 → 1.285 GiB transient, 0.00 → 4.89 GiB free).

## Housekeeping

- Probe/server runs: `setsid nohup ... &`, verify the log in the same
  shell call; kill probes with `pkill -f "probe_[m]"`-style patterns
  from a *separate* invocation.
- GPU wedge protocol (repo AGENTS.md): isolated reset → 1 retry; 2nd
  consecutive reset = burst → stop GPU work, reboot (root), record in
  `docs/gfx906/degradation.md` + `degradation_details.md`. OOM
  teardowns can trigger collateral resets.
- TP=2 teardown: SIGTERM and wait for both cards at VRAM baseline
  (~10.8 MB) before relaunch; SIGKILL mid-P2P wedges GPU1.
- Record attribution results as a falsifiable round (HYPOTHESIS /
  GATE matrix / VERDICT) in the topic's `DEVLOG-*.md`.
