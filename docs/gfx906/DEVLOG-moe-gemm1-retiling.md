# C2-gemm1 retiling: V1 full-K direct store + z-split sweep (2026-08-19)

Branch `gfx906/moe-gemm1-v1` off `gfx906/main`. Model: Qwen3.5-35B-A3B-AWQ
(local NVMe). Item: `moe-decode-roadmap.md` C2 — "gemm1 (≈1070 µs/step) stays
open; the V1 full-K single-wave design and the activation-fusion idea are the
next candidates."

**VERDICT: DEAD-END — all retiling of the decode-size gemm1 is closed.** V1
rejected (2.2–4.3× slower standalone), the (BLOCK_KN, NPT) tiling surface
mapped (best point 12% off current in the launch regime), and the best
point's in-model gain **does not transfer to serving wall-clock in either
eager or graph mode**. No dispatch change shipped; the harness extensions and
the measurement findings are the deliverables.

**GATE:** serving `_bench_gfx906.py`, pp=2048/tg=256, 4 samples,
`BENCH_MAX_SEQS=32`, GPU util 0.95. Both eager and graph arms below use this
same config (eager adds `BENCH_EAGER=1`). Per-kernel census / standalone
harness results are evidence, not the gate.

## 1. V1 (full-K per block, direct store) — REJECTED

Roadmap design: 64 blocks for gemm1 (8 slots × 8 n-tiles), each a single
wavefront looping the full K=2048; kills atomics + both zeroings.
`moe_gemm_q4_v1<THREADS, NPT, K_T>` in
`benchmarks/kernels/gfx906/harness/moe_m1_harness.cu`, gemm1 shape (N=1024,
K=2048, 8 slots), launch regime, 2000 launches (µs/launch, band over all
post-fix runs):

| variant | blocks × waves | µs/launch | vs current 26.9 |
|---|---|---|---|
| current (256t, z=8 CAS, no zero) | 64 × 4 | 26.9–27.1 | — |
| v1b (128t = 2 waves, NPT=1, full-K, direct) | 64 × 2 | 60.4–60.5 | **2.2× slower** |
| v1a (32t = half wavefront, NPT=4, full-K, direct) | 64 × ½ | 117.2–117.4 | **4.3× slower** |

Both pass correctness (max err 0.2511 = current's fp16 noise; the first v1
build had an `a_off` bug — `8*j` instead of `k + 8*j`, re-dotted a[0..32)
every 32-K step; fixed before timing).

**Mechanism:** 64 blocks each streaming one long 128 KB weight sequence can't
keep enough bytes in flight on the MI50 (~128–512 KB in flight vs ~480 KB
needed at ~300 GB/s × ~600 ns HBM latency). The current 8-way K-split keeps
64 independent 16 KB streams turning over. v1a is also an inefficient
half-wavefront config; v1b is the fair test and still loses.

## 2. z-split / NPT surface sweep — NPT=2 is the launch-regime best point

Templated in-tree kernel (`moe_gemm_q4_sweep<THREADS_X, NPT>`,
`THREADS_X == BLOCK_KN`) in the same harness, all 8 configs correctness-
checked against the CPU reference:

| config | blocks (gemm1) | CAS fan-in | µs/launch (3-run medians) |
|---|---|---|---|
| **256t NPT=2** | **128** | 8 | **23.78 / 23.81 / 23.90** |
| 256t NPT=4 (in-tree) | 64 | 8 | 26.88 / 26.95 / 27.08 |
| 128t NPT=2 | 512 | 16 | 25.07 / 25.14 / 25.14 |
| 128t NPT=4 | 256 | 16 | 27.84 / 27.94 / 27.88 |
| 64t NPT=2 | 2048 | 32 | 25.13 (single run) |
| 64t NPT=4 | 1024 | 32 | 33.39 (single run) |
| 512t NPT=2 | 64 | 4 | 29.16 (single run) |
| 512t NPT=4 | 32 | 4 | 37.31 (single run) |
| v2 512t 2col (S5) | 64 | 8 | ~27.1 |
| v2 512t 4col (S5) | 64 | 8 | ~28.1 |

NPT=2 beats NPT=4 at every BLOCK_KN (more blocks = more independent streams);
256t/NPT=2 is the best point, −3.1 µs (−11.6%) vs the in-tree config.
Two monotone trends in this table alone close V3/V4 without building them:
raising the CAS count is monotone-worse (27.8→33.4 from z=16→32), so V4
(fewer CAS) can't be the win; a V3 scratch+add K-split adds launch+scratch
(~184 µs/step) on top of a best point that doesn't transfer (§3) — negative
expectation.

## 3. In-model trial of NPT=2 — the gain does not transfer

Trial (reverted after A/B): `case 1:` in `dispatch_moe_gemm_q4` launched
`<1,2>` for gemm1 (output_topk == 0) under `VLLM_GFX906_MOE_G1NPT2` (per
call, default off). Gates (configs under **GATE** above):

1. Unit test (2 new, removed with the revert): flag flips the output +
   both paths within 1e-1 of fp32 ref, awq_kfirst + wna16_sym. 27/27 with
   the change, 25/25 after. (Pre-existing flake: M8-bm4-N1024-awq_kfirst_sym
   at the 5e-2 boundary, 3/3 in isolation.)
2. PPL (fixed 12-prompt, off/on, same build): 16.0081 vs 16.0011 — noise;
   PPL is the BM≥8 prefill path, untouched by the flag.
3. In-model eager census (`kernel_prof_probe.py`, 256-step): gemm1 `<1,4>`
   29.8 → `<1,2>` 25.0 µs/call, **−186 µs/step**; gemm2 unchanged (by design).
4. **Serving graph** (GATE): off mean 66.53 vs on 66.61 → **+0.08 t/s,
   neutral** (inside the off arm's own 0.17 spread).
5. **Serving eager**: off 23.504 vs on 23.477 → **−0.03 t/s, neutral**.

The census's −186 µs/step never appears in wall-clock in *either* regime.
This is the **third consecutive** decode-size gemm1 retiling to fail
transfer (history: S5 V2 standalone 1.18×/in-model neutral, S2 topk
standalone win/graph replay loss — details in `DEVLOG-moe-m1-sprint.md`
§S2/§S5), confirming the roadmap's warning that per-kernel profiler rows are
pipeline-state dependent even in eager.

## 4. Measurement finding: the torch profiler can't see graph replay perkernel

Attempted to measure `<1,2>` vs `<1,4>` inside graph replay
(`kernel_prof_probe.py`, `enforce_eager=False`). In both `FULL_DECODE_ONLY`
(serving config) and `FULL_AND_PIECEWISE` (capture-size 8) the profiler sees
~5 of 80 gemm calls/step and ~7% GPU-busy — equally blind either way.
(Default `FULL_AND_PIECEWISE` @ max_num_seqs=32 OOMs the FA `_q_pad_buf`
during 40–64-seq capture; the bench's `FULL_DECODE_ONLY`+size-8 is what fits.)

**Result:** CUDA-graph replay is invisible to the torch profiler at per-kernel
granularity on this stack, empirically confirming **wall-clock serving A/B is
the gate** for µs-scale verdicts; per-kernel profiler in-model A/B under
graphs is retired as a method.

## 5. State and follow-ups

- **Reopened (2026-08-23, C2-V `DEVLOG-moe-c2v.md`):** the §3 "failed
  transfer" verdict was TP=1-only. The re-landed NPT=2 `<1,2>` tile
  (commit `c03bc3349e`, env-gated `VLLM_GFX906_MOE_NPT=2`, default off)
  measured **+1.23 % graph / +1.32 % eager at TP=2 M=1** (6 repeats,
  identical outputs) — the per-rank-halved gemm1 shape (N=512) is a
  tiling axis the sweep never covered. Powered TP=1 re-runs confirm
  neutral there (+0.48 % graph ≈ 0, 0.00 % eager). The C2 close
  stands for TP=1 M=1 and batch; the TP=2 M=1 scope reopens per the
  roadmap C2-V rule. Flag still default off pending that decision.
- **Reverted:** dispatch flag + unit tests (zero measured benefit; vLLM
  no-busywork rule); rebuilt, 25/25 suite. gemm1 stays on the established
  `<1,4>`; the in-tree M=1 gemm2 path (V2 tile) is untouched.
- **Kept:** harness `sweep` + `v1` kernels (the tool that produced this
  verdict) + two harness fixes found in self-review: the v2-nozero-dirty
  negative control set `bad`, and the gemm2 absolute-0.35 threshold was
  always exceeded by the 8-slot fp16 accumulation (~0.65 abs) — both fixed
  (control asserts the buffer WAS dirty / gemm2 gated at 5e-2 relative); the
  harness now reports `HARNESS PASS`.
- **Remaining C2 substance (refrigerated residue):**
  - **Activation-fusion** (fold SiLU·mul into gemm1's epilogue, kill the
    activation round-trip) is a *kernel-structure* change, not tiling — the
    sensible next idea, but transfer expectation is low (see §3).
  - **C3** (fold the two zeroings into neighbor kernels, ~234 µs/step
    measured, no numerics change) is the cheap C2-adjacent lever if a future
    branch takes it.
  - gemm2 NPT=2 at the legacy BM=1 path wasn't measured on the sweep — the
    in-tree dispatch routes M=1 gemm2 through the V2 tile; the legacy path
    isn't used in the serving records.

## Bench log locations

- standalone: `/tmp/v1_ab.log` etc. (wiped on reboot; numbers above are the
  record), binary recipe: `hipcc -O3 -o /tmp/moe_m1_harness
  benchmarks/kernels/gfx906/harness/moe_m1_harness.cu` after
  `source ~/env-rocm-7.14-gfx906.sh`.
- A/B logs (this session): `/tmp/ab_g1npt2_{off,on}.log` (graph),
  `/tmp/ab_eager_{off,on}.log` (eager), `/tmp/prof_{off,on}.log` (eager
  census), `/tmp/gprof_{off,on}.log` (graph census attempts).
