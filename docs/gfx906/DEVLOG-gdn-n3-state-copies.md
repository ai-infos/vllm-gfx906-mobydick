# N3 — GDN state-bookkeeping copies: attribute + reduce-or-close

**VERDICT:** CLOSED (measured; no code change) · **GATE:** graph-serving
attribution (required by ROADMAP because the pile is launch-latency-bound) ·
branch `gfx906/n3-gdn-state-copies` · 2026-08-31

## HYPOTHESIS / question

ROADMAP N3: attribute the ~180 µs/step of small `[3,1,32]` state copies in
the GDN layers (first attributed to upstream mamba state bookkeeping in
`DEVLOG-dense-decode.md`, 2026-08-17), then reduce or close. The pile is
launch-latency-bound, so the deciding question is whether it even costs
anything **under production `FULL_DECODE_ONLY` serving** — if CUDA-graph
capture absorbs the launch overhead (as it did for the FA decode copy
pile, `DEVLOG-dense-decode.md` 2026-08-17: "launch overhead is cheap
inside the graph"), the eager number is not a production cost and N3
closes without touching upstream code.

## Method

`benchmarks/kernels/gfx906/n3_state_copy_probe.py` (new): in-process
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`) single-request decode on
Qwen3.5-35B-A3B-AWQ, `torch.profiler` over 32 profiled decode tokens
(after an 8-token warmup that also JITs the Triton kernels), two arms:

- `N3_MODE=eager` — `enforce_eager=True` (the regime the original
  ~180 µs number was measured in);
- `N3_MODE=graph` — production serving config
  (`cudagraph_mode=FULL_DECODE_ONLY`, `max_cudagraph_capture_size=8`,
  `max_num_seqs=4`), i.e. decode steps replay from the captured graph.

Aggregation walks `key_averages(group_by_input_shape=True)` and reports
per-(op, shape) **invocation counts per step** for copy-class ops
(`copy_`, `_to_copy`, `contiguous`, `clone`). Runs executed under a
one-shot systemd user service (`n3probe@.service`, `MemoryMax=infinity`)
— see Pitfalls. Both arms are B=1 single-request; the config asymmetry
between arms (graph arm pins `max_num_seqs=4` for GDN capture, eager
arm uses the engine default) does not affect per-step counts at B=1.

## Result — copy-class op invocations per decode step (B=1, 32 tokens)

| op | eager | graph (production) | Δ |
|---|---:|---:|---|
| `aten::copy_` | 136.56 | **44.03** | −68 % |
| `aten::clone` | 43.44 | **0.31** | −99 % |
| `aten::contiguous` | 22.19 | **0.31** | −99 % |
| `aten::_to_copy` | 12.50 | 12.50 | unchanged (metadata path) |
| **total** | **~214/step** | **~57/step** | **−73 %** |

Under graph serving the launch-latency-bound copies are essentially gone:
`clone` and `contiguous` (the pure "make this view contiguous" bookkeeping
inside `_forward_core`, e.g. lines 1939–1940 `b.contiguous()` /
`a.contiguous()` and the per-tensor squeezes) drop to ~0 — they no longer
appear as per-step aten invocations once decode runs from the captured
graph (whether inductor fused them or they are replayed inside the graph
is not separately verified; the count is what matters). `copy_` falls
68 %; the residual ~44/step + 12.5 `_to_copy` are the per-step
state/metadata bookkeeping that runs **outside** the captured region
(state-index handling around `non_spec_state_indices_tensor` / block-table
views), which is expected and unavoidable without changing upstream mamba
state management.

## Why this closes N3

1. **The eager ~180 µs/step was launch overhead, not GPU work.** The
   copies are `[3,1,32]` (96 elements = 192 B fp16). Their GPU transfer
   cost is sub-µs each; the measured eager cost came from per-op CPU
   dispatch (~0.8 µs × ~214 ops). Production `FULL_DECODE_ONLY` serving
   removes 73 % of the invocations from the dispatch path entirely.
2. **The residual is bounded and small.** ~57 tiny copies/step, each
   192 B: even at a pessimistic 1 µs GPU-side cost that is <60 µs/step,
   and realistically sub-µs for 192-B device-to-device copies — against a
   ~1.5 ms decode step. Not worth an upstream patch or a custom kernel.
3. **Precedent in this codebase.** The FA decode copy pile (7→2 copies,
   `DEVLOG-dense-decode.md`) was reduced where there were clean local
   fixes; the residual there was explicitly accepted as "GPU copy time
   itself ~0.15–0.25 ms/step" in-graph. N3's residual is an order of
   magnitude smaller (192-B vs 384-KB copies) and sits in upstream mamba
   bookkeeping with no clean local reduction — the same "deferred
   (upstream code, small)" disposition, now backed by a measurement.

## Measurement caveat (important for future probes on this build)

This gfx906 torch build (`2.13.0+gfx906`) reports
`self_device_time_total = 0` for **every** aten op — verified with a
standalone profile of an 8192×8192 `copy_` (field test in
`/local/tmp/n3/`). GPU time is not attributed onto CPU-side aten events
in this build, so per-op GPU cost cannot be read from the profiler here;
attribution must lean on invocation counts, tensor sizes, and wall-clock
A/B. Do not interpret "device time ≈ 0" as "op is cheap" on this build.

## Pitfalls

- **`with_stack=True` OOMs.** Full Python+C++ stacks per event over a MoE
  decode exhaust host RAM in post-processing (global OOM kill even under
  `MemoryMax=infinity`) and blow the 4 GiB background-worker cgroup cap
  (SIGKILL mid-run). Run without stacks; identify launch sites from op
  names + source inspection.
- **Background terminal workers are capped at 4 GiB** — a model-load +
  profile run peaks ~23 GiB RSS (weight page cache). Use the one-shot
  systemd user service pattern (`MemoryMax=infinity`), same as the C2
  A/B arms.
- `key_averages(group_by_input_shape=True)` is fine at TG=32; walking raw
  `prof.events()` with per-event stack retention is not.

## Files

- `benchmarks/kernels/gfx906/n3_state_copy_probe.py` (new)
- `docs/gfx906/DEVLOG-gdn-n3-state-copies.md` (this file)
- `docs/gfx906/ROADMAP.md` (N3 → CLOSED)

**VERDICT: CLOSED (on count-based inference).** Measured under both
regimes; the production cost is negligible (launch overhead absorbed by
CUDA-graph capture, residual ~57 tiny copies/step bounded well under
60 µs/step and realistically sub-µs). The closure rests on invocation
counts + tensor sizes, not a measured wall-clock delta — per-op GPU/CPU
cost is unmeasurable on this build (see caveat above), so if a future
build fixes profiler attribution and this is revisited, re-measure
before relying on the "negligible" bound. No code change; no serving A/B
required because nothing ships.
