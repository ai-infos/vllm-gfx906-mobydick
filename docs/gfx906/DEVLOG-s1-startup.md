# DEVLOG — S1: startup (graph + inductor) time

Maintainer: Kevin Read / Hermes agent. Status: **in progress** (2026-09-04).
Plan: `~/.hermes/plans/2026-09-04_123500-s1-startup-time.md`. Artifacts: `/local/tmp/s1/`.

## TL;DR

The 27B checkpoint is **multimodal** (`Qwen3_5ForConditionalGeneration`,
`vision_config: true`, 333 `model.visual.*` tensors) even though we serve it
text-only. Every startup, `profile_run()` profiles a **max-feature-size dummy
image through the ViT** — measured at **~210 s of the ~234 s warm engine-init**
(boot U, 2026-09-04). The stock levers we planned (`--kv-cache-memory-bytes`,
`-O1`) are small potatoes next to this. `--language-model-only` (L3) removes the
vision profiling; **wired into all arms as DEFAULT ON on 2026-09-04** (Kevin's
decision, option a — "wire it into the arms"). **Measured warm result: engine
init 233.85 s → 14.54 s (16×)** with L3+KV-pin; end-to-end dev boot is now
~90–100 s, weights-load-bound (S2 territory, deprioritized).

## Measured baseline (boot U, warm AOT cache)

`s1base` instrumented stock boot (15:13–15:20, TP=2 MTP k=2, util 0.85, maxlen
131072, capture [1,2,3,4]):

| milestone | time | Δ |
|---|---|---:|
| weights TP0 done (40.97 s) | 15:15:06 | — |
| model loading done (TP1, 55.9 s total) | 15:15:21 | +15 s |
| **"Encoder cache will be initialized … profiled with 1 image items of the maximum feature size"** | 15:15:22 | +1 s |
| *(gap: vision dummy forward — see stack dumps)* | | **+213 s** |
| AOT reconstructed (compile counter 0.87 s) | 15:18:55 | |
| KV cache size determined | 15:19:11 | +16 s |
| graph capture done (4 s, 0.66 GiB) | 15:19:15 | +4 s |
| **init engine total** | 15:19:15 | **233.85 s** (compilation counter 0.95 s) |

vLLM's own suggestion for KV pinning this boot: `--kv-cache-memory=13688363316`.

### Stack-dump proof of the gap (the decisive evidence)

In-process dumps, Worker_TP0 (pid 41219), 42 consecutive 5 s samples
15:15:26 → 15:18:51, ALL pinned in:

```
vllm/v1/worker/gpu_model_runner.py:6611 profile_run
  → vllm/model_executor/models/qwen3_vl.py:2882 embed_multimodal
    → vllm/v1/attention/ops/vit_attn_wrappers.py:111 vit_flash_attn_wrapper
      → flash_attn_maxseqlen_wrapper   (GPU ViT kernels; no Python frames while running)
```

So the "pre-compile gap" is not dynamo, not NCCL, not triton JIT — it is the
vision-encoder profiling forward. The compile counter (0.95 s warm) confirms
compilation itself is a non-issue on a warm AOT cache.

### Boot-to-boot variance note

The 12:0x pilot boot logged init engine = **461 s** with compilation = **100.3 s**
(backbone 87.7 + eagle_head 12.6). Same config, same host — the difference is the
AOT cache state (cold for that config-key at 12:0x → warm by 15:13; note each
`--optimization-level`/config change gets its own cache entry). Warm-cache startup
is therefore ~234 s, not ~461 s.

## Levers (all standard upstream flags — no code port; wired into run_server.sh as env opt-ins, DEFAULT OFF)

| lever | flag | expected effect | status |
|---|---|---|---|
| L1 KV pin | `--kv-cache-memory-bytes` (env `KV_MEM_BYTES`) | skips memory-profiling accounting + CUDA-graph-mem estimate (code: `gpu_worker.py determine_available_memory` — **still runs profile_run()**) | wired; measured ~20 s on this config; NOT a global default (bench arms keep stock KV sizing) |
| L2 dev-fast | `--optimization-level 1` (env `DEV_FAST=1`) | PIECEWISE cudagraphs, smaller compile range; matters mostly on COLD AOT cache (warm = 0.95 s already) | wired; **deprioritized** — Task 5 audit: cache survives reboots, cold compile is one-time per config key |
| **L3 text-only** | `--language-model-only` (DEFAULT ON; `FULL_MM=1` opts out) | zeroes modality limits → **skips the vision dummy forward (~210 s)** + shrinks encoder-cache budget | **wired as arm default 2026-09-04; MEASURED: warm init 233.85 s → 14.54 s (with L1)**. Wedges #19/#20 were HW; final warm boot clean |

## Incidents this session

- **Sampler SIGUSR1 killed the launcher** (first probe run): bare SIGUSR1 to the
  bash wrapper pre-`exec` = default terminate. Fixed: age-gate (>12 s observed) +
  comm exclusion for shells + `.ARMED` marker concept; see `lever_probe.sh`.
- **Wedge #19** (15:57:36, GPU1): TP=2 init with L3 → `hipErrorLaunchFailure`,
  BACO reset recovered. HW family per Kevin's ruling; documented in
  `degradation.md`/`degradation_details.md`. GPUs clean post-recovery (no zombie
  KFD handles).
- **Wedge #20** (17:20:31, GPU0): L3 re-run, identical signature (`unspecified
  launch failure` @ SetDevice → `Fence fallback timer expired on ring comp_1.0.0`
  → BACO reset(3) recovered). Stack dumps prove the vision path was NOT running
  (0 vision frames in all 233 dumps; "text-only mode" logged both workers) — crash
  is in the TEXT profile stage, not the removed ViT forward. Two-for-two on L3 vs
  a clean stock baseline 90 min earlier → leading hypothesis **post-reset
  degradation of GPU0** (BACO-reset at #19, wedged again 83 min later without
  reboot). Flag exonerated by construction (config-only change; a software fault
  would give an assertion/shape error, not a fence timeout).

## FINAL RESULT (measured, warm AOT cache, boot U)

| config | init engine | compile | notes |
|---|---:|---:|---|
| baseline stock (s1base) | **233.85 s** | 0.95 s (warm) | includes ~213 s vision dummy forward |
| L1+L3, COLD AOT key (s1l1) | 247.05 s | 103.42 s (one-time) | new config key → full recompile; +92 s pre-compile text profile_run; 143 s post-compile = CPU-side eagle-head compile + KV sizing (GPU idle per csv) |
| **L1+L3, WARM (s1l1w)** | **14.54 s** | 0.92 s (warm) | **16× faster than baseline**; coherent sample; KV 365,661 tokens (pin exact); capture 4 s / 0.57 GiB |

So the steady-state dev boot with `--language-model-only` + KV pin is **~15 s of engine init**
(vs ~234 s stock). End-to-end process-start→healthy ≈ 90–100 s (weights load dominates now:
TP0 36.6 s + TP1 14.6 s, warm page cache — that's the S2 territory, deprioritized).

**L3 stability:** the warm L3 boot ran clean (healthy at 19:00:23, coherent sample, clean
teardown) — wedges #19/#20 remain unexplained HW events; no further L3 correlation.

### t/s sanity check on the L3-default arm (post-reboot, 2026-09-04 ~19:50–20:10)

`sweep_client.py mtp s9 65536` n=3 against the default-arm server (L3 ON, TP=2 MTP k=2):

| rep | t/s @64k |
|---|---:|
| 0 | 38.80 |
| 1 | 38.79 |
| 2 | 38.78 |

Median **38.79 t/s** — inside the historical arm0 @64k band (37.95–38.85) → **L3 has no
decode-path impact**, as expected (it only removes vision profiling at init). Canary this
boot: 39.2 t/s. S1 close-out complete.

## Status: **COMPLETE (2026-09-04)** — L3 wired as arm default; S1 goals met

## Next steps (all resolved 2026-09-04)

1. ~~Canary gate~~ — passed 39.1 t/s at 18:4x before the probe window.
2. ~~L3 re-run for the measured number~~ — done twice (s1l1 cold, s1l1w warm). Warm = **14.54 s init**.
3. L1+L3 combined = the numbers above (KV pin confirmed: identical 365,661-token KV both boots;
   ~20 s saved vs no-pin on this config — modest but free).
4. Task 5 cache audit — done (read-only): see below.

## Task 5 — AOT cache audit (read-only)

- `~/.cache/vllm/torch_compile_cache`: **32 GB, 133 entries**, on the durable home partition
  (`/dev/mapper/ubuntu--vg-ubuntu--lv`, NOT tmpfs). Oldest entry 2026-08-22 → **survives reboots**.
- Each distinct config (incl. `--language-model-only`, `-O` level, KV pin) gets its own cache
  key; first boot of a new key pays a one-time ~103 s full compile (backbone 89 + eagle-head 14),
  subsequent boots reconstruct in <1 s. **Cold starts are a one-time cost per config**, not per
  reboot — so L2 (`-O1`) value is limited to shaving that one-time cost; deprioritized.

## Open decisions — resolved

1. **L3 retry strategy** → (a) chosen by Kevin: wired into the arms as default (run_server.sh,
   `FULL_MM=1` opts out). Measured warm win: 234 s → ~15 s engine init.
2. **Cold-cache durability** → durable across reboots (home partition); one-time per config key.
3. **KV pin as dev default** → NOT made the global default (bench-exact arms must keep stock KV
   sizing); available via `KV_MEM_BYTES=13688363316` for dev boots. L3 IS the default; L1/L2 opt-in.

## Files

- `/local/tmp/s1/` — `phase_probe.sh`, `lever_probe.sh`, `s1_stackdump.py`
  (sitecustomize SIGUSR1 dumper), `analyze_dumps.py`, dumps + logs per tag
  (`s1base`, `s1lang`), `gpu_<tag>.csv`.
- `/local/tmp/mtp1/run_server.sh` — L1/L2/L3 env wiring (default off).
