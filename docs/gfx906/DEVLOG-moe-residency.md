# MoE expert-weight residency (C8) — L2/TCC vs HBM for the active W4 set

**VERDICT:** SHIPPED (measurement complete) · **GATE:** n/a — C8 is a
measurement task; its deliverable is the number, not a serving A/B. Feeds
the roadmap C2 target-selection question (HBM floor vs latency/occupancy).
**Model/context:** Qwen3.5-35B-A3B-AWQ, M=1 decode, gfx906 (MI50), both GPUs
probed 2026-08-31.

## HYPOTHESIS
If the per-layer active W4 weight set (~12 MB = gemm1 8.3 + gemm2 4.2) stays
L2/TCC-resident, then C2's kernel target is latency/occupancy (weights hit);
if it streams from HBM every step, the target is the HBM read-bandwidth floor.

## What was done
Standalone HIP probe (`/local/tmp/c8_residency_probe.cu`, `hipcc -O3`), run on
device 0 and device 1 (identical results to ~2%). Three parts: (1) working-set
sweep of HOT repeated reads — BW vs W, knee = effective L2; (2) the real
per-layer active W4 sets in HOT mode; (3) the production gemm1 `<1,4>` kernel
(M=1) for achieved weight-read BW as a cross-check. Logs:
`/local/tmp/c8_run_gpu0.log`, `/local/tmp/c8_run_gpu1.log`.

## Evidence
- **L2/TCC = 8 MB** (both GPUs). Sweep knee near W≈4 MB (~735 GB/s); the true
  HBM streaming floor (W=256 MB cold) is **804–809 GB/s**. (Small W reads are
  latency-limited, not capacity-limited — a tiny buffer can't keep enough loads
  in flight, so sub-L2 BW < HBM floor.)
- **Combined active W4 set = 12.47 MB > 8 MB L2/TCC** ⇒ it does NOT fit; the
  per-layer expert weights partially stream from HBM each step. HOT read BW at
  that size: ~676–679 GB/s (just under the HBM floor).
- **Production gemm1 `<1,4>` at M=1:** 44.5–44.9 µs/launch, active W bytes =
  8.31 MB → **achieved weight-read BW ≈ 194–196 GB/s** = **24% of the HBM
  floor** and only **58–64% of the achievable read BW** for that working set.

## Why it matters (the C8 answer)
The active set exceeds L2, so weights are *not* fully resident — but the
production kernel is now nowhere near a memory wall: it achieves ~1/4 of HBM
floor and <2/3 of the achievable read BW for its own working set. The binding
constraint at M=1 is **latency/occupancy (memory-level parallelism), not the
HBM floor**. ⇒ C2's target should be set on closing that read-BW gap, not on
approaching HBM peak. There is ~2×+ headroom before bandwidth binds.

## Interactions / superseded-by
- Feeds roadmap **C8** (this) → informs **C2** target selection and any
  gemm1 retiling follow-up. Cross-link: `DEVLOG-moe-c2v.md` (C2 default-on),
  `DEVLOG-moe-gemm1-retiling.md` (the retiling family).

## Search keys
`HYPOTHESIS:` `VERDICT:` `C8` `residency` `L2/TCC` `W4 active set` `weight-read BW`
