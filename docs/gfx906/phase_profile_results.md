# MTP-1 kernel-level phase breakdown @120k — RESULTS (2026-09-02, boots R+S)

Method: CUDA-event forward hooks in a vLLM general plugin (`mtp1_phase_plugin.py`),
mode-NONE eager path, TP=2, util 0.85, 122880-token context, median aggregation.
Full process → skill `mi50-kernel-time-benchmarking`.

**Status (2026-09-04): harness RETAINED.** Official vLLM methods compared in
`dvfs-mi50.md` ("Profiling methods" section) — none superior on this host:
torch-profiler is CPU-only here, nsys not installed, cProfile is CPU-side.
This harness remains the only per-module GPU-time attribution method available.

## VALID: Greedy @120k (run p9771)
Median extrapolated total = **78.4 ms/step**, vs measured wall 78.7 ms/token
(12.74 t/s graphed; 10.82 t/s in this eager run → 92.4 ms/token) → method validated.

| category | layers | ms/step | % of step |
|---|---:|---:|---:|
| **full_attn** (O(Sk)) | 16 | **57.3** | **73%** |
| mlp (dense GEMM) | 64 | 12.6 | 16% |
| lin_attn (GDN, O(1)/step) | 48 | 8.6 | 11% |

Per-call medians: full_attn 3579 µs, mlp 197 µs, lin_attn 178 µs. Single stream per
rank; both TP ranks agree <0.1%. Eager≈graphed for greedy (10.82 vs 12.74 t/s = 0.85×).

**Interpretation:** at 120k context, **full attention dominates the decode step (73%)**.
This is exactly the O(Sk) growth that turns MTP from a win (≤32k) into a loss (≥64k).

## REJECTED: MTP @120k phase runs — deterministic EAGER-MODE ARTIFACT, not HW
Two runs, pre- and post-reboot, on clean GPUs (canary 39.2 t/s, matmul OK, no zombie
handles): **467.0 s and 468.9 s per 256 tok = 0.55 / 0.54 t/s — identical.** The earlier
"degraded GPU after wedge #5" verdict was WRONG; it reproduces across a full reboot.

Why it's invalid for absolute claims:
- MTP eager window = **0.546 t/s vs graphed sweep 9.18 t/s → 0.059×** (17× slower), while
  greedy eager is only 0.85× of graphed. The asymmetry is the tell: spec-decode's per-step
  CPU syncs + Python proposer loop are eliminated by cudagraph capture in production;
  greedy stays GPU-bound either way.
- Hooked module time = **297 ms/step vs ~1,830 ms/token wall** → ~85% of each step is NOT
  in the hooked modules (CPU-side proposer + per-step syncs).
- Window log: 78 Triton autotune events (17.8 s) + 11 JIT compiles during inference —
  zero of both in the greedy window. Autotune alone explains only ~4% of the window;
  the rest is structural eager overhead.
- `n_draft_seen: 0` — drafter hook never fired (gc found a stale instance first). FIXED in
  plugin v5.1: now hooks ALL MTP-class instances; a future run should show n_draft > 0.

Directionally (not quantitatively) consistent with the validated greedy split:
full_attn 90.7% / mlp 4.9% / lin_attn 4.4% — attention share goes UP for MTP vs greedy's
73%, as expected once drafter+verify attention joins a step that is mostly CPU overhead.

## Why MTP loses at long context (synthesis of sweep + validated greedy breakdown)
1. Base decode step @120k ≈ 78 ms (graphed), **73% full attention** (O(S·k)).
2. MTP depth-2 adds drafter forward(s) + extra verify per accepted token; the drafter
   re-runs O(Sk) attention over the long KV → cost scales with context like the target's.
3. ≤32k: attention cheap, ~1.5× extra work still nets a win (acceptance 2.0).
   ≥64k: attention-heavy drafter+verify exceeds tokens saved → loss, widening to 0.72× @120k.

## GPU0 wedge log 2026-09-02 (5×, mode-NONE + long-ctx family)
boot-Q 20:29 · 08:54 · 10:56 (prefill@120k) · ~11:07 (weight load) · 16:43 (MTP weight load).
All recovered via BACO; VRAM to ~10 MiB baseline each time; no zombie handles. Post-reboot
S: canary 39.2 t/s healthy, matmul OK both GPUs — and the MTP artifact reproduced, proving
the wedges did NOT leave GPU0 in a soft-degraded state for this workload.

## Next steps (if MTP-side split is ever needed)
- A valid MTP kernel split requires capturing under conditions where eager≈graphed. Options:
  (a) pre-warm all Triton kernels before the window (kills the autotune/JIT component but
  NOT the structural per-step CPU overhead — likely still invalid); (b) accept that mode-NONE
  distorts spec-decode ~17× and rely on the validated greedy breakdown + sweep for MTP-1b;
  (c) kernel-level tracing via a future ROCm fix (all HSA tracing paths currently dead).
- The relative finding (full attention dominates at long ctx → drives the crossover) does
  NOT depend on any MTP phase run and stands on validated data alone.

## Artifacts
- Plugin: `/local/tmp/mtp1/mtp1_phase_plugin.py` (v5.1, drafter fix) — also in venv site-packages.
- Driver `phase_driver.py` · launcher `run_phase.sh` + `mtp1phase@.service`.
- Sanity test `event_pair_test.py` · summary `phase_summary.py` · window analysis `window_autotune.py`.
- Raw JSON: `phase_greedy_p9771.json` (VALID), `phase_mtp_p2469/2470.json` (rejected, artifact).
