# TTFT prefill stall — TP=2 long-context prefill investigation (Qwen3.8-27B-AWQ, mtp3)

**VERDICT:** SHIPPED (root cause identified; residual mechanism OPEN) ·
**GATE:** serving walls @120k (mtp3b4/TP=2) + full-corpus step-model fit
**Branch:** gfx906/fa-decode-fp16 (2026-09-09 → 2026-09-12)
**Full detail:** `git log bdf1d814c4..6f215f7e7b -- docs/gfx906/ttft-prefill-stall.md`
(the pre-restructure file — 2,404 lines, §1–§13.25 prose preserved there)

## 2026-09-09 — HYPOTHESIS: the ~4-min TTFT stalls at 120k are a bug
(host spin, thread starvation, or shim), not raw prefill throughput

Symptom: TP=2 serving, 120k prompts, per-request TTFT ~4 min with
freeze-then-burst pacing. Theory matrix (T1 spin-wait, T5, T6 thread
cap, shim, P1/P3) + client/engine timeline + CPU census.

## What was done / verdicts (each falsified or confirmed)

- **T1 (host spin-wait owner)**: REFUTED — the "100% spin" was a
  max-cumulative sampler artifact; the main thread blocks in a KFD wait.
- **T6 (torch thread cap)**: REFUTED — OMP ratio 1.00; decode CPU at
  idle level.
- **Shim (blocking-sync .pth)**: EXONERATED — arbiter review §9; later
  stall8 shim-OFF A/B: without it workers burn ~4.0 cores vs ~1.26,
  walls identical → shim stays (S13.10, DECISIVE).
- **D3 (the ~2-core/worker burner)**: native KFD/HIP-runtime thread,
  ~3% EPYC, no t/s impact — accepted (S13.9/S13.10, CLOSED).
- **D2 (DVFS)**: mclk flicker ruled out; 78% MFU not a clock artifact.
- **D1 (live-context probe)**: a fresh 2k prefill with a 67k holder
  live costs +9.5 s → the slope tracks the SUM OF LIVE CONTEXTS.

## 2026-09-10 — VERDICT (production chunk-size A/B, §13)

~92% of per-step cost is chunk-linear GPU work + 0.14 s fixed; effective
~10.3–10.4 TFLOPS (~78% of MI50 fp16 peak) at pp=1024 and pp=256. The
"stall" is **raw W4A16 prefill throughput (GEMM-dominated) + a
34.1 µs/cumulative-token FA slope** — a throughput question, not a bug.
Full-corpus verification (§13.4): every measured TTFT on the cumulative
step model within ~5%; prefill is token-serial.

## 2026-09-11 — the FA slope root-caused and fixed

The 34.1 µs/tok slope = the FA kv_max pad-tile expansion → FIX-H2
(−41% @120k×B4, validated). See `DEVLOG-fa-multibatch-prefill.md`
(same-boot serving A/B is that chain's gate).

## 2026-09-12 — residual mechanism OPEN: batch-level mixed-admission
collapse

Counter decomposition (/metrics polling — RAM-safe method): with a 2k
probe admitted alongside a 67k mtp3 decoder, the holder's decode fell
40 → ~1.6 t/s and the probe's chunks did not complete inside its
7.098-s ttft (admission instant — not a scheduler hold). The residual
is a **batch-level collapse under mixed prefill+long-ctx-decode
admission**; the M3 sync fix did not move it (7.047 vs 7.16 s).
Mechanism candidates: scheduling priority (chunk vs verify), eager/
piecewise path for mixed shapes, draft step under mixed admission.
Caveat: /metrics gauges publish on a ~10-s cadence — step-level
refinement needs engine-stats timestamps or an offline equivalent
(torch profiler BANNED — `DEVLOG-profiling-tooling.md`).

## Related anomalies logged here pre-restructure

- **Hold anomaly** (R2 200 s → mtp3b4-s1 ~40 min → FD-1 5–7 min): a
  request admitted with zero progress, then released. Scheduler/
  tail-race class, engine-side, independent of FIX-H2. OPEN.
- **Straggler release**: 12,288-token prefill burst after a hold
  (S13.18 trace).
- Wedge/degradation entries (#18–#66) live in `degradation.md` +
  `degradation_details.md`.
