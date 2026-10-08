# WORKED EXAMPLE — refactored NPT=2 gemm1 dead-end (from DEVLOG-moe-gemm1-retiling §3)

> This is the verbatim demo of the `_devlog-template.md` format, applied to
> the section the user used as the archetype. Numbers unchanged; structure
> is the new convention. Delete this file once the convention is adopted.

# gemm1 retiling for NPT=2 — does the z-split best point transfer?

> Branch `gfx906/moe-gemm1-v1` off `gfx906/main` · model
> Qwen3.5-35B-A3B-AWQ · date 2026-08-19 · roadmap item C2 ("gemm1 ≈1070
> µs/step stays open; V1 full-K and activation-fusion are next").

**VERDICT:** `DEAD-END` — all retiling of the decode-size gemm1 is closed.

**GATE:** serving `_bench_gfx906.py`, graph mode, pp=2048/tg=256, 4 samples,
`BENCH_MAX_SEQS=32`, GPU util 0.95. The −186 µs/step census win and the
−3.1 µs standalone sweep win are **evidence, not the gate** — neither
transfers.

**DROPPED/REVERTED:** dispatch flag `VLLM_GFX906_MOE_G1NPT2` + 2 unit tests
removed by git-revert; rebuilt, 25/25 suite. Kept (as reference tooling):
harness `sweep` + `v1` kernels.

---

## HYPOTHESIS

NPT=2 (more blocks = more independent weight streams) is the sweep best
point at every BLOCK_KN (−3.1 µs / −11.6% vs the in-tree NPT=4). **If** a
real 186 µs/step kernel-time cut shows up in the in-model census, **then**
it must appear in serving wall-clock. Falsified below.

## What was done

- Harness template `moe_gemm_q4_sweep<THREADS_X, NPT>` — all 8 configs
  correctness-checked vs CPU reference; full sweep table in the source log.
- `case 1:` in `dispatch_moe_gemm_q4` launched `<1,2>` for gemm1
  (`output_topk == 0`) under the per-call flag, default off.
- 2 unit tests (tile taken + both paths within 1e-1 of fp32 ref; awq_kfirst
  + wna16_sym). Removed with the revert.
- Logs: `/tmp/ab_g1npt2_{off,on}.log`, `/tmp/ab_eager_{off,on}.log`,
  `/tmp/prof_{off,on}.log` (eager census), `/tmp/gprof_{off,on}.log`.

## Evidence — FOR

- **launch-regime** sweep: 256t/NPT=2 = 23.78–23.90 µs vs in-tree 26.88–27.08
  → −11.6% standalone.
- **eager census** (in-model): gemm1 `<1,4>` 29.8 → `<1,2>` 25.0 µs/call,
  −186 µs/step. gemm2 unchanged 26.9 (by design).
- Unit/PPL: 27/27 suite; PPL 16.0081 vs 16.0011 (noise — PPL is the BM≥8
  prefill path, untouched).

## Evidence — AGAINST (the gate)

- **serving graph**: off 66.53 vs on 66.61 → **+0.08 t/s, NEUTRAL** (inside
  the off arm's own 0.17 spread).
- **serving eager**: off 23.504 vs on 23.477 → **−0.03 t/s, NEUTRAL**.
- The census's 186 µs/step never appears in wall-clock in *either* regime.

## Why it failed

The serving step's critical path is insensitive to gemm1 kernel time —
proven three independent times. Per-kernel profiler rows are pipeline-state
dependent even in eager, not just under graphs. So the kernel-time win was
real but not on the critical path that wall-clock measures.

## Interactions / superseded-by

- 3rd consecutive decode-size gemm1 retiling to fail transfer (S5 V2:
  standalone 1.18×, in-model neutral; S2 topk: standalone win, graph
  replay loss).
- Closes roadmap V3/V4 without building them (negative expectation).
- Graph-regime per-kernel census proven impossible here (~7% visibility;
  `FULL_AND_PIECEWISE` equally blind, OOMs at 32 seq). Retires profiler-based
  in-model A/B under graphs as a method.

## Refrigerated residue

- C3 zeroing fold into neighbor kernels: ~234 µs/step measured, no numerics
  change — cheap lever if a future branch takes it.
- Activation-fusion (fold SiLU·mul into gemm1's epilogue) is the only
  remaining C2 substance; transfer expectation now low.

## Search keys

`HYPOTHESIS:` gemm1 NPT=2 · `VERDICT:` DEAD-END · `GATE:` serving graph.
