# DEVLOG-fp16-skinny (W4): weight-row-parallel skinny fp16 GEMM, M≤16

**VERDICT: SHIPPED** — 35B MoE N=8 graph **+14.5 %** (191.0 vs 166.9
t/s steady), 27B (Qwen3.8) N=8 graph **+6.1 %** (104.2 vs 98.2, microbench
prediction +5.5 %), 27B N=4 control flat (−0.6 %, flag inert). Kernel in-tree
behind `VLLM_GFX906_SKINNY_M16` (default off — flip once it soaks; see
Residue).
**Date:** 2026-08-23
**Build:** `vllm 0.28.0rc2.dev318+gfed585110` + W4 kernel (this branch)
**Branch:** `gfx906/skinny-fp16` (off `gfx906/main` d608aa40a5)

## HYPOTHESIS

The `rocm_unquantized_gemm` dispatcher sends fp16 GEMMs with
1 < M ≤ 16 to the Triton skinny `triton_matmul`, which runs ~5× off
HBM roofline on the big shapes (27B: fa_q 31.5 MB → 31 µs roofline
vs 174 µs measured at M=1-bound; rocBLAS is *slower*, 340 µs) —
M-invariant (weight-bound). A weight-row-parallel skinny fp16 GEMM
(GEMV structure, M≤16, extending the in-tree `dense_gemv_m4_gfx906`
kernel) would cut the fp16 GEMM floor for concurrent decode (5–16
seqs) and spec-verify batches that land in M=5..16. Does NOT lift M=1
(already the dense_gemv GEMV) or M=2..4 (the L1' rail, unchanged).

**Original estimate (2026-08-23 morning, pre-measurement): ~40 % off
every decode step. FALSIFIED by the microbench — see Design
iterations: the x-L2-re-read wall caps the win at ~3–5 % of the 27B/35B
fp16 GEMM floor in concurrent decode. The kernel ships on the
measured evidence, not the original headline.**

## GATE

- Kernel correctness: allclose vs CPU fp32 ref at M=5/8/16 across the
  shape set (maxdiff < 0.05) — PASS (see Results).
- Per-shape microbench: kernel beats triton at the gated shapes
  (2–7.5× on small-N projections; ties/losses outside the gate).
- Serving wall-clock A/B (the gate): 27B (Qwen3.8 local) + 35B MoE,
  N=8 graph, `VLLM_GFX906_SKINNY_M16` off/on (driver
  /tmp/moespec/run_w4.sh, harness
  benchmarks/kernels/gfx906/moe_multireq_ab.py — the C2-V
  multi-request Δ-metric harness, reused) + 27B N=4 control (flag
  must be inert). Result pending at time of writing.
- 27B token-identity gate usable (dense 27B is deterministic at
  temp=0); 35B FPs not usable (baseline non-deterministic, C2-V).
  Correction (soak): the local 27B is Qwen3.8, which is ALSO
  non-deterministic at temp=0 (the A/B off arm showed 3/3 distinct
  FPs across 3 reps without the flag; the 30-rep on soak showed 30/30
  distinct). Token-identity gates are unusable on both local models
  at this build; perf + output completeness + no-crash is the gate.
- Soak (flag ON, the ship precondition): 30 back-to-back N=8 reps
  per model (driver /tmp/moespec/run_soak.sh, same harness/configs
  as the A/B): rc=0 on both, t/s flat first-half vs second-half
  (|drift| < 0.2 %), VRAM flat (35B pool 99 % throughout - fixed
  pool rules out OOM creep), all reps complete-length. Numbers in
  Results.

## Design iterations (2026-08-23)

Standalone harnesses in /tmp/moespec (bench16.hip, k16v3.hip,
real27b.hip); all numbers below at KCHUNK=1024 unless noted.

1. **v1 (RPT=2, MAXM∈{8,16} capacity-templated)** — the first port.
   Correct only after two bug fixes; slow: 126–166 GB/s at M=8 on the
   5120×2048 21 MB shape vs 308–618 GB/s for the M≤4 rail.
   - Bug 1 (correctness): butterfly started at `mask = NV/2`. On
     gfx906 the wavefront is 64 lanes and `__shfl_xor(v, 32)` is the
     cross-32-lane-half exchange — every house kernel starts at
     mask=32. Starting lower dropped half the wavefront's partial
     sums (M=8/16 outputs off by ~2.0–2.8 abs).
   - Bug 2 (build): `LAUNCHM16_BY_KC` had `;` after `LAUNCHM16(...)`
     inside its if/else chain; LAUNCHM16 expands to a compound
     statement, so the `;` terminated the if and dangled the else
     ("expected expression"). House macros omit the `;`.
2. **v2 (x reloaded from L2 per (r,m), no x residency)** — 20 %
   *slower* than v1 everywhere ⇒ the M=8 collapse is not register
   pressure.
3. **v3 (RPT=1, exact-M-templated `<KCHUNK, M>`, M compile-time
   5..16) — the shipped kernel.** M=4 through v3 = 61 µs on
   5120×2048 = the old M=4 rail's time exactly; scales linearly in M
   (61→123→225 µs for M=4/8/16). Per-M cost is uniform ⇒ the limiter
   is the x traffic, not ALU or registers.
   - **The x-L2-re-read wall (the finding):** every block re-reads all
     of x from L2; total x L2 traffic = N·M·K·2 B, independent of
     KCHUNK (measured: KC=512/1024/2048 on K=2048: 262/173/133 µs at
     M=8 — ksplit=1 best, CAS rounds cost ~25 % at ksplit=2, ~47 % at
     4; KC=5120 for K=5120 no better). Effective x-L2 ceiling ≈
     1.55–1.6 TB/s; v3 ≈ M·B/1.6 TB/s (B = N·K·2). Triton ≈
     B/0.2 TB/s plus a ~100 µs floor on small shapes (a_b 1.3 MB:
     98 µs at every M). Crossover at M ≈ 7.4, size-independent.
     No KC choice moves the wall (x re-reads ∝ ksplit × per-block x,
     product = K).
4. **Epilogue alignment bug (v3, caught by HSA crash):** RPT=1 CAS
   addresses are `lane*N + row` — odd for odd rows — so the 32-bit
   pk2 CAS is misaligned half the time → `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`.
   (The repo's RPT=2 call sites are always even: row0 = 2·blockIdx.x,
   N even by TORCH_CHECK.) v3 uses the compiler-lowered
   `atomicAdd(__half*)` instead — verified correct by the allclose
   checks (which exercise odd rows).

## Real-shape microbench (v3 vs triton, µs)

Qwen3.5-27B shapes (standalone, /tmp/moespec/real27b.hip):

| shape (N×K) | triton M=8 | v3 M=8 | triton M=16 | v3 M=16 |
|---|---|---|---|---|
| gate_up 17408×5120 (178 MB) | 863 | 845 | 869 | 1924 |
| down 5120×17408 (178 MB) | 772 | 883 | 777 | 1952 |
| fa_q 3072×5120 (31.5 MB) | 174 | 156 | 176 | 347 |
| fa_kv 512×5120 (5.2 MB) | 112 | 30 | 123 | 67 |
| a_b 128×5120 (1.3 MB) | 98 | 13 | 101 | 22 |

⇒ wins 2–7.5× on the small-N projections at every M; ties at M=8 on
the big shapes; loses at M≥12 (x-L2 wall). Note the AWQ 27B's MLP GEMMs
(W4) do not take this path — the in-engine fp16 shapes are the FA/GDN
projections (and lm_head, which the gate excludes at M≥8 by size).

**Gate (python, `_gfx906_spec_gemv_m4`):** M≤7 → v3 all shapes;
M=8 → v3 if B ≤ 32 MB; M≥9 → v3 if B ≤ 10 MB (small-shape
triton-floor regime). Pre-existing exceptions unchanged (hipBLAS
5120×[2048,2304] special case; K%512≠0; N%rpt≠0).

## What was done

- 2026-08-23: branch + dev log; triton/rocBLAS M-curve census
  (bench_fp16_skinny_m.py, extended with the gated-gemv column +
  allclose diffs); v1 → v2 → v3 design iterations above; v3 ported to
  `csrc/rocm/dense_gemv_gfx906.cu` (kernel + 12×4 dispatch,
  M=5..16 × KCHUNK∈{512,1024,2048,4096}), python gate in
  `vllm/model_executor/layers/utils.py` behind
  `VLLM_GFX906_SKINNY_M16` (default off); C2-V harness
  `benchmarks/kernels/gfx906/moe_multireq_ab.py` brought onto the
  branch for the serving A/B (+ skinny-env echo in the summary).
- 2026-08-23: serving A/B launched (27B Qwen3.8 N=8 off/on, 27B N=4
  off/on control, 35B N=8 off/on; graph; Δ-metric decode t/s).

## Results

- Kernel allclose vs CPU fp32 ref: M=5 (3072×5120), M=8 (5120×2048),
  M=16 (512×5120) — maxdiff 0.00000 each. In-engine microbench diffs
  (bench_fp16_skinny_m.py) ≤ 0.008 (fp16 rounding at large outputs).
- v3 vs triton: table above; per-step fp16 GEMM floor, 27B-class
  (30 GDN + 10 FA layers, fp16 shapes only): M=8 ≈ 54.9 → 53.2 ms
  (−3 %) with the gate; M=16 ≈ 55.4 → 52.5 ms (−5 %). (AWQ MLP GEMMs
  excluded — not on the fp16 path.)
- **Serving A/B (2026-08-23, driver /tmp/moespec/run_w4{,b,c}.sh,
  graph, steady = reps 1-2):**

  | arm | off t/s | on t/s | Δ | FPs |
  |---|---|---|---|---|
  | 35B MoE N=8 | 166.9 | 191.0 | **+14.5 %** | differ (baseline non-deterministic, C2-V) — perf+sanity gate |
  | 27B Qwen3.8 N=8 (pp 1024, tg 160, maxlen 1280, util 0.90) | 98.2 | 104.2 | **+6.1 %** | differ within arm too (Qwen3.8 non-deterministic) |
  | 27B Qwen3.8 N=4 (control) | 79.63 | 79.14 | −0.6 % | consistent with inert (M≤4 unrouted) |

  - 35B off arm 166.9 = C2-V t1n8 steady record (167.4) → host clean,
    clean control. +14.5 % beats the 3-5 % floor estimate because the
    35B GDN a/b projections (8/16 MB × 30 layers) are all inside the
    M=8 ≤32 MB gate (triton ~14.7 ms/step at N=8 → ~4.6 ms v3).
  - 27B +6.1 % matches the microbench prediction (+5.5 %: 16 FA fa_kv
    20 MB + 48 GDN a 20 MB routed; fa_q/o/b 60 MB correctly gated out
    at M=8).
  - First-ever N=8 records for Qwen3.8-27B (off arm = 98.2 t/s at
    util 0.90).
  - 27B needed util 0.90 / maxlen 1280: Qwen3.8 FA KV is 655
    KB/token (head_dim 256, 4 kv heads, 16 FA layers) — 8×4096
    OOMs at 0.93 (356 MB inductor prefill buffer, free: 0). The
    aborted OOM arm's teardown produced a one-off
    `hipErrorLaunchFailure` in the next arm (recorded in
    degradation.md); clean re-run passed — the ksplit=5 atomicAdd
    path (27B K=5120 shapes) is graph-safe.
- **Soak (2026-08-23, flag ON, 30 reps each, steady = reps 1-30):**

  | arm | steady t/s (stdev) | first vs second half | band | FPs | VRAM |
  |---|---|---|---|---|---|
  | 35B N=8 | 189.9 (0.4, 0.2 %) | −0.15 % | 188.47–190.54 | 2 known-good (incl. the A/B + C2-V FP) | 99 % flat (fixed pool) |
  | 27B Qwen3.8 N=8 | 102.1 (0.14, 0.14 %) | −0.13 % | 101.87–102.46 | 30/30 distinct (model non-determinism, pre-existing - off arm 3/3 distinct) | 98 % flat (fixed pool) |

  rc=0 both; all reps complete-length (2048 / 1280 tokens). Soak
  steady matches the A/B steady (35B 191.0, 27B 102.4-106.0), so the
  A/B deltas are not a short-run artifact. **Soak PASSED.**

## Refrigerated residue

- **W4b (the real fix for big shapes):** the x-L2 wall breaks when x
  is block-resident (persistent blocks, x in LDS: fits for M·K·2 ≤
  64 KB — all 35B dense shapes at M≤16 (K≤2048), 27B only M≤1 at
  K=17408). Not done; the v3 gate leaves big shapes on triton. If
  pursued: the 27B Qwen3.8 fa_q/o/b 60 MB shapes (the remaining
  ~2/3 of its routed-out fp16 share) and the NFS 27B's 178 MB MLP
  shapes are the targets.
- M=8 big-shape tie (845 vs 863 on gate_up) leaves ~2 % uncollected;
  ksplit>1 CAS could be replaced by a warp-level fp32 accumulate if
  ever worth it.
- **Default-on decision (Kevin):** the A/B arms are positive on both
  models and the flag-on soak passed (30 reps × 2 models, flat), so
  `VLLM_GFX906_SKINNY_M16=1` is cleared to go default-on; flipping
  it (or merging the branch with the flag on) is Kevin's call.
  Note: the ksplit>1 epilogue is fp16 `atomicAdd` (value-
deterministic up to fp16 rounding order) - on the non-deterministic
  Qwen3.8 this is unobservable, but on a deterministic model the
  last-bit values of ksplit>1 shapes could vary run-to-run (the M≤4
  rail's pk2 CAS has the same property and has shipped since L1').
- W4 + W2 interaction: a faster no-spec baseline changes the spec A/B
  denominator — re-measure spec arms if the flag goes default-on
  (W2's 35B mtp2 numbers were set on the pre-W4 build; the 35B
  no-spec N=8 baseline moves 166.9 → 191.0, +14.5 %).

## Interactions / superseded-by

- `CHANGELOG.md` (the completed W4 entry); the open 27B L2 note in
  `spec-decode-roadmap.md`
  (triton vs rocBLAS vs GEMV per shape at M=1..16).
- `csrc/rocm/dense_gemv_gfx906.cu` (the M=1 GEMV + M=2..4 rail this
  extends); `68243a61b2` (m4 M-templating, KCHUNK set).
- `DEVLOG-moe-c2v.md` (the 35B baseline non-determinism caveat that
  makes 35B FPs unusable as an A/B gate; the moe_multireq_ab.py
  harness origin).
- `DEVLOG-moe-spec-decode.md` (W2: the in-process A/B method this
  serving A/B supersedes for W4).

## Search keys

fp16 skinny gemm, triton_matmul 174us, weight-row parallel, M<=16,
rocm_unquantized_gemm, dense_gemv_m_kernel_m16, x-L2 re-read wall,
VLLM_GFX906_SKINNY_M16, HSA aperture violation odd CAS, shfl_xor
mask 32 wavefront 64, W4, concurrent decode gemm
