# C2-V (v2) — MoE gemm re-tile verdicts under batch-decode and TP=2 regime

> Branch `gfx906/moe-c2v` off `gfx906/main` (d608aa40a5) · model
> Qwen3.5-35B-A3B-AWQ (`/local/models/QuantTrio/Qwen3.5-35B-A3B-AWQ`) ·
> 2026-08-22 · roadmap item C2 (`moe-decode-roadmap.md`).

**VERDICT:** SHIPPED (program complete 2026-08-23: reopen gate
TRIGGERED — both re-tile candidates ≥0.5 % at **TP=2 M=1** (gemm2-v2
+1.47 % graph; gemm1 NPT=2 +1.23 % graph / +1.32 % eager, 6 repeats,
identical outputs). The C2-gemm1/S5 branch reopens TP=2-scoped; the
close stands for TP=1 M=1 and batch. Follow-ups in residue below.)

**GATE:** serving A/B, pp=2048/tg=256, graph mode (eager for Stage-1
A/B arms), N=1/4/8/32 concurrent, TP=1 and TP=2, flags off/on with
interleaved engines, 3 repeats per engine. Reopen rule (roadmap
C2-V(v2)): **any positive ≥0.5% reopens C2-gemm1 and the S5 gemm1
branch.** Metric: decode-only t/s via the Δ-wall method
(`benchmarks/kernels/gfx906/moe_multireq_ab.py`: tg vs tg_short in
the same engine — prefill/overlap cancels in the difference). NOTE:
the 67.39 t/s record (band 65.9–67.0) is the *official harness*
metric (`_bench_gfx906.py`: 256 / (prefill+decode wall)) — a
different quantity, not comparable to Δ-method numbers (at equal
step time the harness reads ~22% lower). The A/B delta is the gate
(same metric on both arms); the absolute host/build anchor is the
official harness run (queued after Stage 0, expect the 65.9–67.0
band) + the mtp2 canary.

---

## HYPOTHESIS

If the three "failed transfer" verdicts (S5-V2 gemm2 M=1 tile, S2 topk
M=1, NPT-sweep gemm1 M=1 — all rendered in the single-request gate
regime) are regime artifacts, then either (a) a busier concurrent
decode step or (b) the TP=2 per-rank shape change (expert N halved:
gemm1 1024→512, gemm2 2048→1024) makes the re-tiles show ≥0.5%
wall-clock gain — in which case the C2-gemm1/S5 branch reopens.
Kevin overruled TP=1-only scoping on 2026-08-22: closing on TP=1
alone would discard a TP=2 win; TP=2 stability on this box is proven
(27B dense, official amdgpu DKMS 6.19.14 driver) and TP=2 MoE-35B is
assumed fine — the smoke run verifies.

## Scope correction (2026-08-22, pre-run dispatch-gate audit)

`dispatch_moe_gemm_q4` (`csrc/rocm/moe_q_gemm_gfx906.cu`) shows both
existing re-tile candidates are **M=1-only**:

- `VLLM_GFX906_MOE_M1` (gemm2 v2 512t tile) fires only when
  `block_size_m == 1 && output_topk > 0 && size_m == output_topk` —
  i.e. a single decode token. At N≥2, EM=N·topk > topk → **structurally
  inert** (running it at N≥2 would only re-measure the off arm).
- The NPT=2 gemm1 trial dispatch (reverted per
  `DEVLOG-moe-gemm1-retiling.md` §5, never committed) was the BM=1
  path — M=1 only.

Consequences for the (v2) matrix:

1. Roadmap (v2)(a) — "busier step transfers the M=1 savings" — is moot
   for N≥2: the batch-decode gemm path is the **BM=4 grouped GEMM**
   (em=N·8: N=4→BM=1, N=8/32→BM=4), which the (BLOCK_KN, NPT) sweep
   **never measured** (roadmap (v2)(b)). The batch axis therefore
   reduces to *characterizing* the BM=4 path; re-tiling it (if it
   shows headroom) is a Stage-2 decision, not the (v2) as written.
2. The only batch point where an M=1-gated flag can fire is **N=4**
   (em=32 ≤ 32 → BM=1): the NPT=2 gemm1 arm runs there. `MOE_M1` at
   N=4 is expected inert (EM=32 ≠ topk=8) — one confirmation run
   documents the gate.
3. **TP=2 is a genuinely new tiling axis even at M=1.** The gemm2 v2
   tile's shape gate still passes at TP=2 (N=1024 % 256 == 0,
   K=512 % 256 == 0, groupsize 128 % 32 == 0); gemm1 `<1,4>`/`<1,2>`
   at N=512 is unmeasured. → TP=2 N=1 A/B is a first-class arm.
4. TP=2 35B-MoE is a **first run on this box** (TP=2 proven on the
   27B dense only; MoE W4A16 kernels + GDN + Q8 FA under RCCL P2P
   unverified). A smoke (the first t2n1 run) gates the rest of the
   TP=2 axis; teardown per the TP=2 protocol (clean exit / SIGTERM,
   VRAM 0% verified between runs; the bench's SIGTERM handler does an
   engine-core shutdown before exit).

## Matrix

| stage | point | arms | regime | notes |
|---|---|---|---|---|
| 0.1 | canary (27B mtp2, 60 s) | — | — | `degradation_details.md` protocol; host was rebooted 4 h earlier |
| 0.2 | TP=1 N=1 | off, `MOE_M1=1` | graph | anchor: re-confirms the known +0.60 t/s S5 result on this build |
| 0.3 | TP=2 N=1 | off, `MOE_M1=1` | graph | first TP=2 35B run = smoke; the TP=2 M=1 tiling axis |
| 0.4 | TP=1 N=4/8/32 | off (N=4 + `MOE_M1=1` = inertness proof) | graph | BM=4-path characterization (the (v2)(b) measurement) |
| 1 | TP=1/TP=2 N=1, TP=1 N=4 | off, NPT=2 gemm1 on (rebuild) | graph + eager | land `<1,2>` instantiation + env-gated dispatch (default off) first |

Stage-0 runtime budget ~1 h GPU (8 engines × load+warmup+3 repeats);
Stage 1 ~1.5 h (6 engines × 2 regimes + rebuild). Driver:
`/tmp/c2v/run_stage0.sh` (per-run logs `/tmp/c2v/<label>.log`,
results TSV `/tmp/c2v/stage0_results.tsv`).

## What was done

- 2026-08-22: branch created; dispatch-gate audit (above); bench
  `benchmarks/kernels/gfx906/moe_multireq_ab.py` (Δ-wall decode-only
  t/s: prefill overlap cancels in the tg vs tg_short difference);
  Stage-0 driver; roadmap C2-V state line updated. (Commit
  3824813bcf.)
- (in progress) Stage-0 run sequence (driver PID-logged, logs
  `/tmp/c2v/`): canary → t1n1 off/on → t2n1 off/on (smoke) → t1n4
  off/on → t1n8 off → t1n32 off; official-harness 35B run queued
  after the driver (metric anchor).

### Results so far (2026-08-22, build fed585110)

- **Canary (27B mtp2, 60 s): 38.8 t/s** — below the ~40–47 healthy
  band but well above the <25 REBOOT line. Soft signal; the 35B
  official-harness run is the tie-breaker (if it lands < 65.9 the
  host is off-record and Stage-0 deltas are suspect).
- **Official harness (35B, graph, 4 samples): 65.61 / 65.623 /
  65.603 / 65.566 → 65.60 ± 0.03 t/s.** Marginally below the record
  band (65.9–67.0; record 67.39) — ~0.5 % under the band floor, i.e.
  at record level within run-to-run drift (degradation would be a
  ~3× collapse, not 0.5 %). Cross-check: 256/3.902 s = 65.6 and the
  Δ-method's 240/2.95 s = 81.2 both resolve to d ≈ 12.3 ms/step with
  prefill ≈ 0.75 s — internally consistent. **Host judged healthy for
  A/B gating; deltas, not absolutes, are the gate.**
- **t1n1_off (TP=1, N=1, graph, flag off): 81.17 t/s Δ-metric**
  (stdev 0.7, min 80.39 / max 81.72; prefill ≈ 0.9 s, step ≈ 12.3 ms).
  Not comparable to the 67.39 record (different metric — see GATE); the
  harness anchor run is pending.
- **t1n1_m1on (TP=1, N=1, graph, `MOE_M1=1`): 82.46 t/s** (stdev
  0.13, 82.38–82.61) → **+1.29 t/s = +1.59 % vs off.** This is the
  KNOWN S5 gemm2-v2 single-request effect re-confirmed on this build
  (S5 recorded +0.90 % on the harness metric — consistent once
  prefill dilution is accounted for; 48 layers × ~4.5 µs ≈ 0.2 ms
  ≈ 1.6 % of a 12.3 ms step). The anchor works as designed: the
  flag's single-request gain is real and ≥0.5 % on this build. This
  is NOT new evidence — it is the regime the C2 close already
  measured; the (v2) verdict rests on the TP=2 and batch/N=4 arms.

### TP=2 smoke failure + workaround (t2n1_off, 20:05Z)

First TP=2 35B-MoE run crashed: rank-1 worker died in
`RocmPlatform.get_device_name` (torch-compile-cache-dir query) during
`profile_run` at `moe_forward_shared` — final exception
`AMDSMI_STATUS_NOT_INIT` from the `with_amdsmi_context` wrapper's
`finally: amdsmi_shut_down()`, masking the primary error. Rank 0 hung
on the shm broadcast (GPU0 100 %) until SIGTERM. **No kernel reset**
(software crash — recorded in `degradation.md` +
`degradation_details.md` "2026-08-22 evening", incl. a transient
44 %-VRAM-on-GPU1-with-no-owner observation that turned out to be
the next config's in-flight allocation). Root fragility: **amdsmi is
broken on this boot in every run** (TP=1 included — all runs log the
protected `Failed to get total memory via amdsmi` fallback);
`get_device_name` is the one unprotected caller. Workaround:
sitecustomize shim (`/tmp/c2v/shim/sitecustomize.py`) — swallows
amdsmi init/shutdown failures and gives `get_device_name` the same
`AMD_<arch>` fallback the code already has for the 0-handles case.
Upstream fix belongs in `vllm/platforms/rocm.py` (the wrapper's
`finally` must not raise over the primary error). Stage-0b
(`/tmp/c2v/run_stage0b.sh`, waiting on the harness job) re-runs:
TP=1 batch (t1n4 off/on, t1n8, t1n32) then TP=2 N=1 off/on WITH the
shim (`*_s` arms).

### Stage 0 complete — batch + TP=2-shim results (Δ-metric t/s, graph)

| point | off | on (flag) | Δ | note |
|---|---|---|---|---|
| TP=1 N=1 | 81.17 (±0.7) | 82.46 (±0.13, MOE_M1) | +1.29 (+1.59 %) | known S5 effect re-confirmed (anchor) |
| **TP=2 N=1 (shim)** | **80.32 (±0.03)** | **81.50 (±0.08, MOE_M1)** | **+1.18 (+1.47 %)** | **reopen gate triggered (≥0.5 %)** |
| TP=1 N=4 | 184.59 (±2.1) | 183.76 (±1.3, MOE_M1) | −0.83 (−0.45 %) | inert as designed (M=1 gate) — proof |
| TP=1 N=8 | 167.4 (rep1/2; rep0 cold) | — | — | BM=4 grouped-path characterization |
| TP=1 N=32 | OOM (all configs) | — | — | single-card memory ceiling (below) |

Findings:

1. **TP=2 M=1 is a real, positive axis**: the gemm2-v2 tile transfers
   to the per-rank-halved shapes (gemm2 N=1024, K=512) at +1.47 %, on
   par with TP=1 (+1.59 %). Per the roadmap rule ("any positive ≥0.5 %
   reopens C2-gemm1 and the S5 gemm1 branch") the branch **reopens** —
   at minimum TP=2-scoped; combined with the re-confirmed TP=1 effect,
   the natural follow-up question (for Kevin) is whether `MOE_M1` goes
   default-on (it is env-gated, default off, since the C2 close).
2. **Batch flag arms are structurally inert, as predicted**: MOE_M1 at
   N=4 is within noise (−0.45 %, |Δ| < stdev) — the M=1 gate
   (`size_m == output_topk`) excludes every N≥2 point. The batch
   decode regime runs the BM=4 grouped path (N=8: em=64), which the
   (BLOCK_KN, NPT) sweep never touched; both Stage-1 flags are BM=1
   only, so the N=4 NPT=2 arm is the sole batch A/B point.
3. **Step scaling**: N=1 12.3 ms → N=4 (BM=1) 21.7 ms → N=8 (BM=4)
   47.8 ms. The BM=1→BM=4 transition costs +161 %/step for 2× the
   tokens — the grouped path is the expensive one (per-slot cost
   rises), consistent with the (v2)(b) premise. This is the real
   batch-regime lever if batch decode ever becomes the target.
4. **N=32 single-card memory ceiling (new finding)**: 32-concurrent
   35B decode OOMs on one MI50 32 GB in this build at every config
   tried (graph/eager × util 0.95/0.90 × maxlen 4096/2816/2048 ×
   pp 2048/1024). Failure is the gfx906-FA `_q_pad_buf` [B, Hq,
   Sq_pad, D] fp32 prefill Q buffer (544 MiB at B=32, maxlen-
   independent — it grows to the batch size) on top of a ~28 GiB
   non-KV footprint (17.6 GB weights + GDN state pool + inductor +
   FA buffers); KV pool left with 41–57k tokens < the 74k needed.
   32-seq 35B serving needs TP=2 (per-rank Hq/2 halves the q_pad) or
   a larger card. Not a C2-V gate point (flag inert at BM=4 anyway).
5. **Bench artifact noted**: the first repeat's tg_short run is cold
   at N=8 (wall_short 11.7 vs 6.45 s steady) and inflates the Δ
   metric (308 vs 167 t/s); steady-state = rep1/2 (agreed to 0.1 t/s).
   Future A/B arms should report rep1/2.

### Stage 1 (NPT=2 gemm1) — in progress

`csrc/rocm/moe_q_gemm_gfx906.cu`: `VLLM_GFX906_MOE_NPT=2` now also
selects the `<1,2>` kernel (64 cols/block vs `<1,4>`'s 128) for the
BM=1 **gemm1** path (`output_topk == 0`) — the reverted trial, re-
landed fresh (never committed), env-gated default off, per-call
getenv like MOE_M1. gemm2 stays `<1,4>` (or the MOE_M1 v2 tile) so
the two trial flags never touch the same kernel. Incremental
`build_ext --inplace` (ccache) running; matrix: off/npt2 × {TP1 N1,
TP2 N1 (shim), TP1 N4} graph + {TP1 N1, TP2 N1} eager — TP=2 N=1
includes the first measurement of the gemm1 tile at the halved
N=512 shape (the new tiling axis).

### Stage 1 results (complete 2026-08-23, 6 repeats, fingerprints match
all 6 off/npt2 pairs — the `<1,2>` kernel is output-correct at every
shape incl. 32-slot N=4 and the TP=2 halved shapes)

| point | off | npt2 (`MOE_NPT=2`) | Δ |
|---|---|---|---|
| TP=1 N=1 graph | 81.19 (±1.06) | 81.58 (±1.09) | +0.39 (+0.48 %) — under the 0.5 % gate, ~0.3σ, i.e. neutral (confirms the original 4-sample neutral verdict, now powered) |
| TP=1 N=1 eager | 25.90 (±0.06) | 25.90 (±0.06) | 0.00 % — neutral |
| **TP=2 N=1 graph** | **81.18 (±0.22)** | **82.18 (±0.04)** | **+1.00 (+1.23 %) — above gate, ~4.5σ of off-stdev** |
| **TP=2 N=1 eager** | **21.25 (±0.12)** | **21.53 (±0.07)** | **+0.28 (+1.32 %) — above gate, ~3σ; regime-consistent with graph** |
| TP=1 N=4 graph | 184.33 (±1.91) | 183.55 (±0.71) | −0.78 (−0.42 %) — noise; no batch effect |

Notes: TP=1 N=4's BM=1 gemm1 tile is neutral at 32 slots (the
standalone −11.6 % was an M=1 launch-regime effect). Absolute levels
drifted ~+1 % on TP=2 off-arm overnight (80.32 → 81.18) — the A/B
deltas are intra-driver (adjacent engines) and unaffected; no wedge
or reset overnight (VRAM 0/0 at completion, no degradation-table
entries).

## Combined default-on decision (C2 final, 2026-08-31)

**VERDICT:** SHIPPED · **GATE:** serving A/B, Qwen3.5-35B-A3B-AWQ,
pp=2048/tg=256, graph mode, util 0.95, N=1, 3 repeats/arm, TP=1 and
TP=2 (Δ-wall decode t/s); numerics gate = output fingerprint identical
across all arms.

### HYPOTHESIS

If the two C2-V re-tile wins (gemm2-v2 `MOE_M1`, gemm1 `<1,2>`
`MOE_NPT=2`) are real and non-interfering, then running them **together**
in one M=1 decode engine beats the off arm by more than either alone at
TP=2 M=1 (≥ +3%, well above the 0.5% reopen threshold) with bit-identical
outputs — in which case both become default-on for the M=1 path.

### What was done

8-arm combined A/B (4 arms × TP ∈ {1,2}): off / `MOE_M1=1` /
`MOE_NPT=2` / both, per-arm VLLM_CACHE_ROOT isolation. TP=2 under
systemd user services (`c2arm@.service`, MemoryMax=infinity — the
worker-cgroup OOM finding in `degradation_details.md`); TP=1 in-process.
Driver + results: `/local/tmp/c2v/` (combined_results.tsv, per-arm logs).

### Evidence FOR

| arm | TP | MOE_M1 | MOE_NPT | decode t/s (Δ-metric) | vs off |
|---|---|---|---|---|---|
| off | 1 | — | — | 82.37 ± 0.07 | — |
| m1 | 1 | 1 | — | 84.41 ± 0.06 | +2.5 % |
| npt2 | 1 | — | 2 | 84.64 ± 0.09 | +2.8 % |
| **both** | 1 | 1 | 2 | **84.74 ± 0.10** | **+2.9 %** |
| off | 2 | — | — | 81.58 ± 0.58 | — |
| m1 | 2 | 1 | — | 83.80 ± 0.03 | +2.7 % |
| npt2 | 2 | — | 2 | 83.88 ± 0.03 | +2.8 % |
| **both** | 2 | 1 | 2 | **85.65 ± 0.04** | **+5.0 %** |

- Numerics gate PASS: output fingerprint `d2e5262183c6b92f` identical
  across all 8 arms (every repeat).
- Hypothesis CONFIRMED at TP=2 M=1 (+5.0% > +3%, ~14× the off-arm
  stdev); positive at TP=1 too — no regime regresses.

### Evidence AGAINST

(none)

### Why it failed (if applicable)

—

### In-tree state after this verdict

Both flags default-on for the M=1 decode path: gemm2 takes the v2
512-thread tile when its shape gate passes (non-qualifying shapes — e.g.
Nemotron-H K2=1856 — fall back to legacy `<1,4>` silently, the exact
pre-flag path); gemm1 takes the `<1,2>` re-tile. Env overrides retained:
`VLLM_GFX906_MOE_M1=0` / `VLLM_GFX906_MOE_NPT=4` opt out; `MOE_M1=1`
still hard-asserts the shape gate (documented narrowing: only an exact
`"1"` forces v2 — pre-C2 any non-`"0"` did). Dispatch-path marker op
(`take_moe_m1_dispatch_path`, registered via CompositeExplicitAutograd
as the codebase pattern for zero-tensor ops; test-only but cheap — a
single atomic exchange, kept ungated so tests need no import hook) +
per-flag unit tests guard which tile actually ran.

Note: TP=1 deltas are larger than C2-V's powered neutral (+0.48%): the
combined A/B ran on a fresh boot (P) with a different build; the gate
decision rests on TP=2 M=1 where the signal was always largest, and
TP=1 is positive here too.

Session cost notes: one chronic weight-load wedge (08:49:34, GPU0,
isolated per house recipe) and the worker-cgroup OOM finding → systemd
migration; both recorded in `degradation_details.md`.

## V1 N-split axis + corrected-harness re-run (C2 finish, 2026-08-31)

**VERDICT:** SHIPPED (measurement complete; the V1 block-count axis is
closed — no dispatch change, per the no-busywork rule). **GATE:** n/a for
a measurement-only close — standalone launch-regime numbers are the
deliverable here because every V1 variant is ≥2.1× slower than current in
the regime where it would have to win (v1b 59.0 vs 28.7 µs), so no serving
A/B can be reached (the transfer failures of §3/DEVLOG-moe-gemm1-retiling
make a serving gate uninformative at this margin).

### HYPOTHESIS

The V1 family was rejected at 64 blocks on the stream-length mechanism.
If more N-splitting (128/256/512 blocks → 64/32/16 KB per-block streams)
recovers enough independent weight streams, then some block count beats or
approaches current — in which case a V1 variant becomes a dispatch
candidate. (Note: the new points also change wavefront config, so the data
cannot isolate stream length alone — see the confound note below.)

### What was done

- `benchmarks/kernels/gfx906/harness/moe_m1_harness.cu`: three new
  instantiations of the existing `moe_gemm_q4_v1<THREADS,NPT,K_T>` kernel
  (no kernel change — only grid/cols): v1c `<32,2>` = 128 blocks × 64 cols
  (64 KB stream), v1d `<32,1>` = 256 blocks × 32 cols (32 KB), v1e `<16,1>`
  = 512 blocks × 16 cols (16 KB). Added to both the correctness section
  (vs fp32 CPU reference) and the timing section.
- Corrected-harness PASS flow re-run: 4 full runs of `moe_m1_harness 2000`
  (boot P, host clean — canary 9.9 TFLOP/s fp16 both GPUs pre-run, no
  amdgpu resets in journal since boot; a hipDeviceProp warpSize probe was
  also run). All checks green; records in `/local/tmp/c2finish/`
  (run_main_2000.log, run_rep2.log, run_rep3.log, run_final.log — the last
  two after the comment-only edits, confirming no behavior change).

### Evidence — FOR (axis closure)

All five V1 variants correct vs CPU ref (max abs err 0.2511 each = the
fp16-accumulation noise band of v1a/v1b and v2; gate 0.35). "Per-block
weight read" = cols/block × K/2 bytes (each output column is a K/2-byte
W4 weight strip):

| variant | blocks × cols/block | per-block weight read | µs/launch (3-run median) | vs current 28.7 |
|---|---|---|---|---|
| current (gemm1, pre-zeroed) | 64 × 1024 (z=8 CAS) | — | 28.70–28.79 | — |
| v1a `<32,4>` (record 117.2–117.4) | 64 × 128 | 128 KB | 115.58–115.60 | **4.0× slower** |
| v1b `<128,1>` (record 60.4–60.5) | 64 × 128 | 128 KB | 59.00–59.04 | **2.1× slower** |
| v1c `<32,2>` (NEW) | 128 × 64 | 64 KB | 81.92–81.95 | **2.9× slower** |
| v1d `<32,1>` (NEW) | 256 × 32 | 32 KB | 74.82–74.86 | **2.6× slower** |
| v1e `<16,1>` (NEW) | 512 × 16 | 16 KB | 87.45–87.47 | **3.0× slower** |

- The best point of all five variants is v1b (64 blocks, 128 KB/block):
  59.0 µs — still **2.1× slower than current**. Every N-split variant
  (v1c/v1d/v1e = 64/32/16 KB/block) is SLOWER than v1b: +22.9 / +15.8 /
  +28.5 µs, and the axis is non-monotone within the N-split family
  (128→256→512 blocks = 81.9→74.8→87.5 µs). No variant approaches
  current's 28.7 µs at any block count.
- **Confound noted (mechanism not cleanly isolated):** the N-split variants
  shorten the per-block stream *and* drop to half/quarter-wavefront blocks
  (v1c/v1d = 32 threads, v1e = 16) at the same time, whereas v1b is 2 full
  wavefronts/block. The v1a-vs-v1b pair (both 64 blocks / 128 cols / 128 KB
  stream; only wavefront config differs) shows that config dominates — v1b's
  2 full wavefronts beat v1a's half wavefront ~2× (59.0 vs 115.6 µs). So the
  N-split loss is a mix of shorter-stream and fewer-wavefront effects, and
  this data **neither confirms nor falsifies** the original stream-length
  mechanism in isolation — but it does show that adding blocks to shorten the
  stream buys nothing on top of the best V1 point. The residual structural
  cost common to all V1 variants is the per-block K-loop length: every V1
  block walks the full 64-iteration K-loop over K=2048 (current's z-split
  blocks walk 8), capping per-block memory-level parallelism regardless of
  tiling.
- v1e `<16,1>` (512 blocks) is slower than v1d `<32,1>` (256 blocks): both
  have NPT=1, so the only difference is threads/block — 16 vs 32. On this
  gfx906 `warpSize = 64` (hipDeviceProp probe), so a 32-thread block already
  runs at half-wavefront occupancy and a 16-thread block at a quarter; v1e
  issues fewer concurrent loads per block than v1d, which is why it loses.
- Corrected-harness PASS flow: `HARNESS PASS` ×4. v1a/v1b bands match the
  08-19 records (117.2–117.4 / 60.4–60.5) within 2.5 % (115.6 / 59.0);
  current's pre-zeroed time reads 28.7 here vs the 08-19 record 26.9
  (~+7 %, boot/build-to-build drift — the A/B ratios in this entry are
  intra-run, so unaffected). Old S5 microbenchmark numbers re-validated.

### Evidence — AGAINST (a V1 dispatch candidate)

(none needed — the axis closes on the FOR side: the best of all five V1
variants is v1b at 59.0 µs vs current's 28.7, a 2.1× gap; no variant
comes within an order of magnitude, so a serving A/B is unreachable.)

### Why it failed (mechanism, refined)

The N-split axis closes on measurement, not on a clean mechanism proof:
every block count loses by ≥2.1× vs current, and the best V1 point (v1b,
59.0 µs) is untouched by adding blocks — every N-split variant is slower
than it. The v1a-vs-v1b pair isolates one clean effect (wavefront config:
2 full wavefronts beat a half wavefront ~2× at equal block count and
stream length), but the N-split variants confound shorter streams with
fewer wavefronts, so the original stream-length mechanism is neither
confirmed nor falsified by this data. What IS established for all V1
variants: a structural cost common to the family — every V1 block walks
all 64 K-iterations of K=2048 (current's z-split blocks walk 8), capping
per-block memory-level parallelism regardless of tiling. N-splitting only
moves the operating point along the V1 family (best-ever 59.0 µs at v1b) —
it cannot cross to current's 28.7 µs. The V1 family is closed at every
measured block count; the axis is exhausted without a serving gate.

### In-tree state after this verdict

Harness-only change (benchmark file, no production dispatch touched).
V1 N-split axis: CLOSED (measured-and-rejected, not untested). C2's
remaining open axis: the BM≥2 grouped-path measurement (serving A/B —
needs multi-hour serving infrastructure; left open on the roadmap).

## Verdict (final)

- **Reopen gate TRIGGERED** (roadmap rule: any positive ≥0.5 %):
  - gemm2-v2 tile (`MOE_M1`): TP=2 M=1 **+1.47 %** (Stage 0).
  - gemm1 NPT=2 (`MOE_NPT=2`, `<1,2>`): TP=2 M=1 **+1.23 % graph /
    +1.32 % eager** (Stage 1) — the C2 close's "failed transfer"
    verdict was TP=1-only; at the halved per-rank gemm1 shape (N=512)
    the tile's −11.6 % standalone gain DOES transfer.
- The close **stands for TP=1 M=1** (now measured with power: +0.48 %
  ≈ neutral, 6 samples) **and for batch** (both flags structurally
  inert at N≥2; N=4 tile trial neutral — the batch lever is the
  un-tilled BM=4 grouped path, a separate C2-scope item).
- **C2-gemm1/S5 branch reopens, TP=2-scoped.** In-tree state: both
  flags env-gated, default off (no dispatch default change without a
  Kevin decision — see below).

## Refrigerated residue (updated)

- **Follow-up for Kevin:** default-on candidates now measured: `MOE_M1`
  (gemm2-v2: +1.59 % TP=1 / +1.47 % TP=2 M=1) and `MOE_NPT=2` (gemm1:
  +1.23/+1.32 % TP=2 M=1, neutral TP=1). Combined M=1 decode would take
  both. Neither helps N≥2 (M=1 gates) — batch decode is untouched.
- **BM=4/8 grouped-path tiling** — the real batch-decode lever
  (N=8: 47.8 ms/step, per-slot cost rises at the BM=1→BM=4 jump);
  now known to be *unmeasured, not measured-and-rejected*.
- **N=32 single-card ceiling** (FA q_pad + ~28 GiB non-KV): 32-seq 35B
  serving needs TP=2 or a bigger card — relevant to any batch-decode
  target setting.
- The `with_amdsmi_context` finally-masks-primary bug
  (`vllm/platforms/rocm.py`) — upstream fix candidate; the shim
  (`/tmp/c2v/shim/sitecustomize.py`, wiped on reboot) is C2-V-only.
- The (v1) power question is answered for free: 6-sample TP=1 A/B
  resolves the NPT=2 question at ±~0.7 % — the old 4-sample "neutral"
  was indeed below detection, and the powered answer IS neutral at
  TP=1 (the gain was TP=2-specific all along).

## Evidence — FOR

(none yet)

## Evidence — AGAINST

(none yet)

## Why it failed (if applicable)

—

## Interactions / superseded-by

- `DEVLOG-moe-gemm1-retiling.md` (the C2 close this re-tests — its §5
  now carries the reopen pointer) and `DEVLOG-moe-m1-sprint.md` (S5/S2
  provenance).
- Outcome: both flags ≥0.5 % at TP=2 M=1 → the "failed transfer"
  verdicts stand for TP=1 M=1 and batch, and the branch reopens
  **TP=2-scoped** (Verdict above).

## Search keys

`HYPOTHESIS:` `VERDICT:` `C2-V` `moe-c2v` `batch decode` `TP=2`

## 2026-09-16 — C2-BM≥2: the grouped/batch-decode tile had never been swept (and the shipped BM=4 is the worst of three)

**VERDICT:** `OPEN` — enabler shipped, isolated result is a **10–15 % per-call regression
from the shipped tile** at every production `em`; the default does NOT move without a
serving A/B. · **GATE:** serving A/B at **B=4 MTP k=3 (em=128)** on the 35B MoE, ms/step
lead, same boot, interleaved arms — *not run* (the night's load budget was consumed by the
FA-D96 gate: wedge #96 + GPU resets (2)/(3) in 8 min = burst).
`VLLM_GFX906_MOE_BM` (1/2/4/8) pins the tile for that A/B; unset = the historical choice.

### HYPOTHESIS

The C2 work closed the M=1 tiles and characterized the BM=4 grouped path as expensive
("BM=1→BM=4 costs +161 %/step for 2× the tokens") but never *swept* it: `select_n_per_thread`
was BM≥8 only and `case 2`/`case 4` hard-coded NPT=4. If the grouped path is tile-limited
rather than shape-limited, then (a) the NPT axis and (b) the BM choice itself should move
the per-call time on the production `em` values (em = M·topk: 64 at B=8 greedy, 128 at B=4
MTP k=3, 256 at B=16/B=4·k=7).

### What was done

- `csrc/rocm/moe_q_gemm_gfx906.cu`: `VLLM_GFX906_MOE_NPT` now applies to **every** BM (was
  BM≥8), and `case 2`/`case 4` honour it. Unset = the historical choices → bit-identical
  dispatch to the pre-change build.
- `vllm/model_executor/layers/fused_moe/experts/gfx906_w4a16_moe.py`: the heuristic moved
  into `_block_size_m_for(M, topk)` with the new `VLLM_GFX906_MOE_BM` pin.
- New bench `benchmarks/kernels/gfx906/bench_moe_bm_sweep.py` (production shapes E=256,
  topk=8, w13 1024×2048, w2 2048×512; dispatcher-faithful; deciles; cross-tile output
  check; DVFS gate).
- Tests: `tests/kernels/moe/test_gfx906_moe_bm_select.py` (heuristic table + pin + bad-pin
  rejection) and the MoE GEMM suite re-run — 76 passed.

### Evidence — FOR (*launch-regime*, mclk 1000 MHz gated, medians of 15×10 calls)

Per-call gemm1+gemm2 totals, Qwen3.5-35B-A3B shapes, GPU0 (`c2_focus.log`):

| em (M) | production bucket | bm=1 | bm=2 | **bm=4 (shipped)** | best vs shipped |
|---|---|---|---|---|---|
| 64 (M=8, B=8 greedy) | 4 | 198.1 | **193.6** | 227.3 | **−14.8 %** (bm=2) |
| 128 (M=16, B=4 MTP k=3) | 4 | **347.6** | 350.7 | 406.2 | **−14.4 %** (bm=1) |
| 256 (M=32) | 4 | 722.1 | **604.6** | 676.8 | **−10.7 %** (bm=2) |

- The NPT axis is a wash in this regime: with `VLLM_GFX906_MOE_NPT=2` the same grid reads
  199.2/195.1/225.0, 350.9/389.6/432.3, 705.7/754.2/752.1 — i.e. BM=4 is still the worst
  and the best tile per point is the same shape (bm=2 at em 64/256, bm=1 at em 128).
- Cross-tile agreement: `rel vs bm=1` ≤ 1.2e-4 on every point (the fp16 CAS-accumulation
  noise band — the kernels are documented as not bit-reproducible run-to-run).
- The two independent sweep runs (initial ungated run, then the gated focus run) agree on
  the ordering and within 4–8 % on the magnitudes.

### Evidence — AGAINST (why the default does not move)

- **The gate has never been run.** This is the fourth tile/dispatch change in this area
  whose isolated win was hoped to transfer: S5-V2 gemm2 M=1 tile, S2 top-k specialization
  and the gemm1 re-tile all inverted or vanished in the graph-serving regime (§"Transfer
  failures"). A 10–15 % kernel-level delta is exactly the size the M=1 history says cannot
  be assumed.
- The 128-token `inter` tensor is produced by a *different* block count per arm (16 blocks
  at BM=8 vs 128 at BM=1), so the arms also differ in padding waste
  (`num_tokens_post_padded`); the bench measures the production code path but only at one
  activation distribution.
- No B=1 arm is affected: at em ≤ 32 the heuristic already selects BM=1, and the M=1
  v2/NPT tiles are gated on `size_m == output_topk`, so this item is purely the B≥4
  concurrent-decode regime.

### Why the shipped bucket may be wrong

BM=4 pads every expert's token block up to 4 rows and halves the block count vs BM=2; at
em=64–256 the extra rows are padding-only work, while the CAS K-split epilogue cost is
per-block, so fewer, fatter blocks lose on both counts. BM=8 wins only once em is large
enough that the padding is amortized (the prefill bucket, where the original BM=8-vs-BM=16
measurement came from).

### Interactions / superseded-by

- Complements, not supersedes, the M=1 work: those tiles are selected by `em ≤ 32`, this
  item is the `32 < em ≤ 512` bucket.
- The `VLLM_GFX906_MOE_NPT` widening is inert by default and enables the same A/B for any
  future BM question.

### Refrigerated residue

- **Queued serving gate (fresh boot, 1 pair of arms):** 35B MoE, `--max-num-seqs 4` +
  `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`, agentic corpus,
  `VLLM_GFX906_MOE_BM=2` vs unset (and `=1` at the 35B's em=128), ms/step lead + acceptance
  + t/s, interleaved. If a tile wins ≥0.5 % with the fingerprint intact, change
  `_block_size_m_for`'s middle bucket and keep the env as the rollback.
- `BM=2` is instantiated but the heuristic can never select it today (`case 2` existed
  unused) — the sweep is the first data on that template in the batch regime.
- Bench-methodology note for the next reader: `rocm-smi` was unusable after the night's GPU
  resets (python init MemoryError) and silently produced "mclk 0 MHz" windows; the sampler
  now reads `/sys/class/drm/card*/device/pp_dpm_mclk` first (the DPM table marks the active
  level with `*`) and takes the max across cards, because GPU0 is `card1` on this box.

## 2026-09-16 (later) — C2-BM>=2 serving gate: NEUTRAL, and the low bucket is the real win

**VERDICT:** `NEUTRAL` (item closed as an active fusion/tile item) · **GATE:** in-process
graph serving A/B (`docs/gfx906/_bench_gfx906.py`), Qwen3.5-35B-A3B-AWQ, TP=1, util 0.95,
**MTP k=3, B=4 concurrent** (em=128 -> the mid bucket), pp2048/tg192, 3 samples/arm,
mclk 1000 MHz on every window, arms back-to-back in the order A,B,C,D.

The isolated 10–15 % per-call win from the sweep above **does not transfer**: all three
mid-bucket tiles are within 0.5 % in serving.

| arm | mid-bucket BM | t/s (3 samples) | mean |
|---|---|---|---|
| A1 unset (shipped) | 4 | 85.741 / 85.188 / 85.547 | **85.49** |
| B `MOE_BM=2` | 2 | 85.803 / 85.724 / 85.598 | **85.71** |
| C `MOE_BM=1` | 1 | 85.354 / 85.215 / 85.235 | **85.27** |
| D unset (order control) | 4 | 85.752 / 85.697 / 85.606 | **85.69** |

- A vs D (order control) = **+0.2 %** — the shipped arm reproduces, so the ~0.5 % spread
  between B/C is the noise floor, not a signal.
- No dispatch change ships: `_block_size_m_for`'s mid bucket stays 4, and
  `VLLM_GFX906_MOE_BM` stays as the A/B/rollback knob (mid bucket only).

**The finding that did transfer — the *low* bucket.** The first BM=2 arm pinned **every**
em value and read **76.6 / 76.5 / 76.4 t/s = −10.6 %** vs the shipped 85.6. That arm is
invalid for the mid-bucket question (it also moved `em <= 32` off BM=1 and prefill off
BM=8), but it is a clean measurement of something else: at B=4 MTP k=3 a large share of
steps are partial-acceptance steps with `em = 4·k_accept·topk <= 32`, and those run the
**M=1 tile + the fused-align/v2-gemm2 M=1 path** — moving them to BM=2 costs ~10 %. So
the heuristic's BM=1 bucket is load-bearing in exactly the way the C2 M=1 work claimed.
This is why the knob is now **mid-bucket-scoped** (`_block_size_m_for`): a coarse pin
silently measures a different experiment.

**Why the isolated win vanished (mechanism).** The isolated bench timed the grouped GEMM
alone; in the step the same kernel is one of several (FA verify, GDN, routing, drafter,
shared expert) and the mid-bucket tile swap only re-schedules the same total work
(BM=4 pads more per expert but launches half the blocks) — with the step's other
components unchanged, a ±15 % kernel delta that is really a *schedule* delta washes out.
This is the fourth confirmation of the repo's transfer rule and the first on this axis.

**Post-state:** `VLLM_GFX906_MOE_NPT` (all-BM) + `VLLM_GFX906_MOE_BM` (mid bucket) +
`bench_moe_bm_sweep.py` + `test_gfx906_moe_bm_select.py` remain as the reusable
instrumentation for any future BM question.
