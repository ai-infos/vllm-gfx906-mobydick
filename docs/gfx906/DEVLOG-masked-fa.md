# max_model_len decode tax (N4) — persistent live-bounded gather+quantize — tax removed at 131k AND 262k, 2.57× at 262k

Branch: `gfx906/fa-masked-gather` (off `gfx906/main` @ `fed5851105`) · 2026-08-21/22
Model: `cyankiwi/Qwen3.8-27B-AWQ-INT4` (snapshot `63768c10`, local HF
cache) · 64 layers (48 GDN + 16 FA), Hq=24, Hkv=4 (2/GPU at TP=2), D=256
Plan: `plan_masked_fa.md` §2.2 design · Diagnosis:
`tp_decode_investigation.md` RESOLUTION / roadmap N4 · Reviews:
`plan_masked_fa_plan_rev_claude.md`, `plan_masked_fa_rev_glm5.md`,
`plan_masked_fa_rev_qwen.md`

## 2026-08-21/22 — N4 tax fixed: persistent grid-stride gather, all gates pass

**VERDICT:** SHIPPED · **GATE:** TP=2 serving A/B (PERSIST=0 vs =1 at
max_model_len 131072/262144, 1091-token prompt, tg=256 greedy, 3 reps,
util 0.93, capture sizes [1,2,3,4]).

## HYPOTHESIS

If the S8 −25% decode tax at max_model_len 262144 is the two-kernel
fallback's O(Sk_pad) work baked into the FULL-graph launch dims
(`gather_paged_kv_q8_kernel_v2` + `quantize_q8_0_dense_kernel`, both
launched at `Sk_pad = pad32(max_model_len)` with live `seq_lens` ignored
for the tail), then a single persistent grid-stride kernel with a fixed
capture-time grid and live-`seq_lens`-bounded work removes the tax at
every max_model_len — and 131k configs improve too (their gather was
frozen at 131k-wide for ~1.5k-live decode).

## What was done

- Kernel `gather_paged_kv_quant_persistent`
  (`csrc/gfx906_fa/gfx906_fa_gather.cu`) + launcher + pybind
  (`gfx906_fa.cpp`) + dispatch (`gfx906_fa_paged.py`, env
  `GFX906_FA_PERSIST`). Fixed 1024-WG grid (`GFX906_FA_PERSIST_GRID`),
  grid-stride over (seq,head,token) rows, per-seq register prefix
  (B ≤ 16), in-kernel q8_0 via `quantize_block_q8_0_halfwarp` (bit-equal
  to the dense/fused paths), V byte-copy; rows [seq_len, Sk) NOT written
  except `GFX906_FA_PERSIST_MARGIN` (default 128 = max D=256 FA tail-tile
  width) V-zero rows. One code path at every Sk; the > 65535 two-kernel
  fallback is the PERSIST=0 rollback path (kept in tree).
- Gates run, in order: NaN-tail (precondition for tail-write removal) →
  standalone bit-exact + timing probe → capture/replay probe (B=1..4) →
  suite test → PPL A/B → serving A/B. Probes in
  `benchmarks/kernels/gfx906/fa_{nantail,persist,capture_replay}_probe.py`.
- Build note: the tree's `.hip` files are untracked hipify artifacts; the
  `.cu` files are the build sources (hipify regenerates `.hip` into the
  build dir at configure time).
- Also: TP=1 in-model tax curve (`_bench_gfx906.py`, pp2048/tg256,
  4 samples, live ~1.5k): 0.95 util: 2816 → 25.02 t/s, 16384 → 23.87
  t/s (−4.6% already at 16k); uniform 0.93 rerun: 2816 → 25.13, 16384
  → 24.83 (−1.2%). The 0.95 run OOM'd at ≥ 65504 (known dense-27B
  warm-cache inductor mode, AGENTS.md — 0.93 is the working util); the
  0.93 ≥ 65504 points were never captured (platform launch-failure
  flake burst, then a stale-VRAM hold from a crashed run shrank the KV
  budget to 3.98 GiB) — not chased further: the serving A/B (THE gate)
  already establishes the in-model tax (P0 22.36 vs 15.91) and its
  removal, so the TP=1 curve is corroborating evidence only.

## Evidence FOR

- **NaN-tail gate PASS** (precondition): FA `forward` outputs bit-equal
  with the K_q8/V tail [sl, Sk) poisoned NaN — aligned/misaligned 32-row
  tiles (oob_check path), 1-row-before-Sk worst case, full, tiny, B=2
  ragged, GQA-packed (Hq=24/Hkv=2,4 → nc2=2). The FA kernel never reads
  rows at/beyond kv_max; tail contents are irrelevant to the output.
- **Standalone per-layer, Hkv=2, live 1536** (launch-regime evidence):
  | Sk_pad | two-kernel | persistent | speedup |
  |---|---|---|---|
  | 3328 | 52.4 µs | 27.8 µs | 1.9× |
  | 65504 | 740.3 µs | 26.2 µs | 28.2× |
  | 131072 | 1136.8 µs | 27.0 µs | 42.2× |
  | 262144 | 2257.1 µs | 26.7 µs | 84.5× |
  (Hkv=4: 101.5→55.2, 1157.8→56.4, 2269.9→56.0, 4508.4→54.0 µs. K and V
  bit-equal in-range; margin V rows verified zero. Persistent also beats
  the small-Sk fused V1 kernel (~40 µs at 3328 per devlog).)
- **Capture/replay bit-exact** at Sk_pad=262144 with live sl swept:
  B=1 sl ∈ {32,1536,2177,8192,262112,262144}; B=1..4 sl ∈
  {128,1536,262112,262144} — the full capture-size range. Grid frozen,
  seq_lens re-read at replay. Suite test:
  `test_persistent_gather_capture_replay_large_sk` (passes).
- **PPL gate**: dense 27B, 12 prompts / 359 tokens / 0 top-20 misses:
  PERSIST=0 → **10.5516**, PERSIST=1 → **10.5516** — identical.
- **Serving A/B (THE gate)**, TP=2, mean of 3 greedy reps:
  | arm | t/s (reps) | vs P0 |
  |---|---|---|
  | 131k, P0 | 22.36 (22.39/22.37/22.32) | — |
  | 262k, P0 | 15.91 (15.94/15.91/15.89) | −28.8% (the S8 tax) |
  | 131k, P1 | 40.86 (40.86/40.84/40.87) | **+82.7%** |
  | 262k, P1 | 40.89 (40.89/40.87/40.90) | **+157.0%** |
  P1 residual tax (131k vs 262k): +0.07% = noise. Step-time transfer:
  131k Δmeasured 20.2 ms vs standalone (1137−27)µs×16 = 17.8 ms;
  262k Δmeasured 38.4 ms vs (2257−27)µs×16 = 35.7 ms — in-model
  ≈ standalone (×1.07–1.14; the S8 session's 0.47 coefficient was
  build/harness-specific). Reproduction note: the S8 39.9/29.9 numbers
  were the mtp2 harness; plain-greedy absolute values differ, but the
  P0 tax (−28.8% here vs −25% there) and the P1 fix transfer.

## Evidence AGAINST / caveats

- **Greedy 12×128 hashes diverge between P0 and P1 (3/12 prompts)** —
  refuted as a PERSIST effect by controls: P0-vs-P0 diverges too
  (p11), and p00 alone (batch=1, max_num_seqs=1) produces 3 distinct
  outputs across 3 P0 runs, with the batch-12 "P0-only" and "P1-only"
  values cross-appearing under both configs at batch=1 (1dd4a52d under
  P0×1 + P1×2; b10a3f65 under P0×1 + P1×1 in batch-12). First 4 tokens
  identical across all runs; PPL/prefill identical ⇒ the nondeterminism
  is in the DECODE path of this hybrid model (GDN triton decode is the
  prime suspect), not the FA gather. Bit-exact greedy is not a valid
  gate instrument for this model family; PPL + kernel-level bit-exact
  gates carry the correctness evidence. (Future: if a bit-exact greedy
  gate is wanted, run it on a pure-FA model or isolate the GDN
  nondeterminism first.)
- **Long-live-context A/B not run** (live ~1.1k only). Expected from the
  traffic math: at live 128k / Sk 262k, P1 does 128k live rows
  (≈0.5 ms/layer bandwidth-bound) vs P0's 262k rows — P1 still wins,
  less dramatically; the mechanism (work = live, not Sk) is kernel-level
  verified. Follow-up if the serving population is long-context-heavy.
- **rocprofv3 (plan step 1) remains unusable** in-model (attach thread
  not exposed; direct launch dies hipErrorLaunchFailure at device init,
  both TP=1 and TP=2). Standalone probes + serving A/B substitute; the
  tax split (gather vs FA-compute) is confirmed by the S8 eager-no-gap /
  graph-gap evidence plus this A/B closing the graph gap.

## Platform flake, same nights (context, NOT this change)

4+ `hipErrorLaunchFailure` wedges during TP=2 weight-load (S4 residual,
historical ~1/3; cold GPUs at 31°C, clean VRAM between runs; recovered
by retry; one TP=1 probe also aborted once and passed on rerun). BACO
reset / power-cycle need root, which this user lacks — if a session hits
the burst, escalate for a reset. Also: a server that boots past /health
can still half-dead (shm-broadcast timeout) and serve empty
`finish_reason: stop` streams — check request output, not just /health.

## Interactions / superseded-by · Refrigerated residue

- **Supersedes** the single-bound bounded-capture design and its
  dispatch-fallback (tp_decode_investigation.md CORRECTION section): the
  kernel live-bounding is the recommended path; no bound/fallback code
  built, none needed — one graph serves every live Sk. Multi-tier
  capture (§2.3) not needed.
- **Roadmap N4: RESOLVED** (this branch; merge to `gfx906/main` after
  review). `tp_decode_investigation.md` RESOLUTION: fix lever = this
  kernel.
- **Default flip**: `GFX906_FA_PERSIST` default ON after the A/B gate
  (kill switch `=0`); margin_zeros stays 128 (defensive, ~64 KB/head,
  removable to 0 — the NaN gate passed).
- Refrigerated: (a) delete the > 65535 two-kernel fallback + V2
  `cached_version` machinery once PERSIST=0 has no users; (b)
  `GFX906_FA_PERSIST_GRID` sweep (512/2048) — 1024 is fine at every
  probed shape; (c) LEGACY=0 direct-paged revisit (COW gap, N2) is
  unaffected and still blocked as before.

## 2026-08-22 (post-commit) — code review found one P0, fixed on-branch

Review: `fa-masked-gather-code-rev-qwen.md` (VERDICT: ship-worthy after
the P0 fix). **P0**: the `_PERSISTENT` dispatch had no `num_seqs` guard
while the C++ side throws at `num_seqs > 16` (fixed 16-entry register
prefix) — with default ON, any `max_num_seqs > 16` (vLLM default 1024;
house MoE 32) would crash at engine start (default capture sizes go up
to 512). All gated runs used `--max-num-seqs 4`, which masked it. Fixed
in `ddd2adbdeb`: dispatch clamped to B ≤ 16 with fallback to the
fused/two-kernel paths above the bound (old behavior, no throw) + two
regression tests (B=17 dispatch fallback; B=16 kernel bit-equal) —
suite 21/21. Same commit: `fa_capture_replay_probe` block-table width
was too small, so its sl=262144 point was skipping ~196k rows in every
path (bit-equal but unmaterialized) — now fully materialized, re-run
PASS. Residuals (review P1/P2): mtp2 A/B before re-baselining S8,
D=128 probe before widening default-ON, prefix-widen to B=32
follow-up, B>16 configs still pay the Sk tax (old behavior).

## 2026-08-22 (post-commit 2) — cross-review fold + gate hardening

**VERDICT:** SHIPPED · **GATE:** probe/suite re-runs (no kernel-behavior
change — C++ edit comment-only, Python debug-only — so no serving A/B
required).

Third review (`fa-masked-gather-code-rev-glm5.md`, folding the ds4 and
qwen reviews; every claim validated or rejected against the tree at
`ddd2adbdeb`):

- **qwen P0-1 / ds4 F1 (B>16 throw at engine start): VALIDATED.**
  Reproduced through `forward_paged` at B=17 pre-fix; default capture
  sizes `[1,2,4]+range(8,256,8)+...≤512` (`vllm/config/compilation.py`)
  and default `max_num_seqs=1024` (`arg_utils.py`) confirmed — engine
  start would crash at default config. `ddd2adbdeb` fix validated
  (B=17 dispatches to the fused path, no throw; suite 21/21).
- **ds4 F5 (margin knob re-read per launch): REJECTED** — both
  `GFX906_FA_PERSIST_{GRID,MARGIN}` are `static` read-once
  (`get_fa_persist_*`), identical pattern; no asymmetry.
- **qwen P2-5 residual ("now fully materialized"): PARTLY** — the
  `ddd2adbdeb` width fix removed the skip-guard path, but the last 16
  blocks/seq still aliased phys block 0 via the zero-filled table
  tail. Fixed here (full-width fill).

Hardening landed (this commit):

- `fa_nantail_probe.py`: Sq>1 inline-causal cases — mtp2 verify (sq=3),
  prefill chunk (sq=64, ncols1=64), worst-case tail sq=2, GQA-packed
  mtp2 B=2 — closing the Sq=1-only coverage gap (tail-write removal
  now gated on the spec-decode/prefill tile geometries too).
  Re-run: **11/11 PASS**.
- `fa_capture_replay_probe.py`: full-width block tables (no tail
  aliasing); dead `REPLAY_SLS` constant removed. Re-run: **16/16
  PASS** (B=1..4 × sl {128, 1536, 262112, 262144}).
- `gfx906_fa_paged.py`: `GFX906_FA_DOUBLE_CHECK=1` now cross-checks the
  persistent branch against the torch reference (per-seq in-range
  rows; raises on mismatch) — smoke-run OK at B=3 ragged.
  `GFX906_FA_FUSED_QUANT=0`-dead-under-PERSIST documented at the knob.
- `csrc/gfx906_fa/gfx906_fa.cpp`: stale margin comment fixed (NaN gate
  passed; 128 stays as the decided conservative default).
- `tp_decode_investigation.md`: fix-lever/alternatives sections
  bannered SUPERSEDED → this log (they read as still-open after the
  2026-08-21 rewrite).

Suite: 21/21. Refrigerated (unchanged): mtp2 in-model A/B before
re-baselining S8; D=128 probe before widening default-ON past the
D=256 model family; register-prefix widen to B=32 (house MoE
`max_num_seqs=32` still pays the Sk tax above 16);
small-`max_model_len` serving A/B (persistent replaced fused V1 at
Sk≤65535 with kernel-level evidence only); `MARGIN=0` optional drop.

## 2026-08-22 (post-commit 3) — review follow-up gates: D=128, small-Sk serving A/B, mtp2 A/B

Three review residuals run to ground (GPU session, branch at
`3b40ad6349`).

**1. D=128 kernel gates — SHIPPED · GATE: probe re-runs**
(`fa_persist_probe` D∈{128,256}×Hkv∈{2,4}×Sk{3328..262144},
`fa_nantail_probe` +3 D=128 cases, suite `test_persistent_gather_d128_matches_fused`).
K/V bit-equal to the two-kernel path and margin-zero at every D=128
combo; tail-read precondition holds at the D=128 FA tile config;
launch-regime: D=128 persistent 16.7–18.3 µs flat vs two-kernel
37.4 µs (3328) / 1224.6 µs (262144). Suite 22/22. Default-ON is now
kernel-gated at both shipped head sizes.

**2. Small-`max_model_len` serving A/B (F7) — SHIPPED, win not just
parity · GATE:** TP=1 GPU0, dense Qwen3.8-27B (local),
`_bench_gfx906.py` graph mode (FULL_DECODE_ONLY), util 0.93,
pp2048/tg256, 4 samples, `BENCH_MAX_SEQS=4`, arms maxlen×PERSIST:

| maxlen | P0 (fused V1) | P1 (persistent) | Δ |
|---|---|---|---|
| 8192 | 24.50 (24.61/24.60/24.24/24.55) | 24.76 (24.78/24.77/24.74/24.74) | +1.1% |
| 32768 | 22.39 (22.40/22.40/22.38/22.38) | 24.75 (24.87/24.85/24.44/24.82) | **+10.5%** |

No regression at small Sk; at 32k PERSIST removes V1's Sk-proportional
tail-zero/dispatch cost entirely (P1 flat 24.75–24.76 across 8k→32k;
P0 pays −8.6% at 32k vs its own 8k). Samples fully separated at 32k
(p0 max 22.40 < p1 min 24.44). One arm aborted at weight-load (exit
134, the known S4-residual flake) — clean rerun, numbers above.

**3. mtp2 TP=2 A/B (re-baseline gate, qwen P1-1) — tax removal
CONFIRMED under mtp2; absolute mtp2 regression discovered (OPEN) ·
GATE:** TP=2, util 0.93, max_num_seqs 4, capture [1,2,3,4],
`--speculative-config {mtp, n=2}`, 1091-token prompt, tg 256, 3 reps,
client tgs TTFT-excluded (same harness as the N4 plain-greedy gate):

| arm | mean | steady (reps 2-3) |
|---|---|---|
| 131k P0 | 15.70 | 16.50 |
| 262k P0 | 12.09 | 12.54 |
| 131k P1 | 24.64 | 24.90 |
| 262k P1 | 24.61 | 24.91 |

- **Tax removal transfers**: P0 tax 131k→262k −24.0% (reproduces S8's
  −25% under mtp2); P1 residual 24.90 vs 24.91 = **0.0%**; P1 vs P0
  +51% (131k) / +98% (262k) steady.
- **NEW, pre-existing, NOT an N4 artifact (P0 shows it too): mtp2 on
  this branch is absolutely regressed.** Steady 24.9 < plain-greedy
  P1's 40.9 (same client/harness), and ≪ the pre-merge tp2-branch
  mtp2 record 39.9 @131k (S8, v0.27.2 build; S5's 39.7 was at
  max_seq_len 4096 per `tp2-bench-final.log`). Acceptance is healthy
  (metrics: mean acceptance length 3.00) → the pathology is step time:
  ~120 ms/verify step vs ~24 ms plain decode step, i.e. the mtp2
  verify/draft path is ~5× the plain step on the current build
  (v0.28.0rc2.dev318+gfed58511). [CORRECTED 2026-08-22: not
  "post-upstream-merge" — the merge changed no step-path code; the
  slowness was host-state degradation, see the CLOSED update below.
  Also S5's 39.7 was TP=2 @131k, not maxlen 4096 — the
  `tp2-bench-final.log` run had speculative_config=None.] Suspect the spec-decode path
  lost graph coverage in the merge; needs its own investigation
  (DEVLOG-spec-decode.md territory) — **do not re-baseline S8 mtp2
  records on gfx906/main until it is root-caused.** Plain-greedy
  records are unaffected.

  **Update 2026-08-22 (later, GPU session):** root cause characterized —
  see DEVLOG-spec-decode.md "TP=2 mtp2 engine-cadence overhead" and
  `fa-masked-mtp-regression-glm5.md` (repo docs/gfx906/) (folds the ds4/qwen reviews). GPU
  work is healthy at both TP sizes; the ~95 ms/step is engine-level spec
  bookkeeping (worker input-waits + propose/reject CPU chain);
  `--async-scheduling` tested, no effect. TP=1 healthy. Clean-rebuild
  A/B of the record binary remains the confirmation experiment.

  **CLOSED 2026-08-22 (post-reboot, canary-passed clean re-run):** the
  entire OPEN item was host-state degradation (see
  `degradation_details.md`) — the characterization above described the
  degraded host, not the code. Clean 4-arm re-baseline (token-true
  usage-based client, identical output hashes `35c4e6794fa4` in all
  arms):

  | arm | t/s | ms/step |
  |---|---|---|
  | 131k P0 | 49.44 | 60.7 |
  | 262k P0 | 37.69 | 79.6 |
  | 131k P1 | **74.74** | 40.1 |
  | 262k P1 | **73.51** (steady reps 74.83) | 40.1 |

  Tax removal CONFIRMED under mtp2 on the healthy host: P0 tax −23.8 %
  (S8's −25 %), P1 residual 0.0 % steady, +51 %/+95 %. mtp2 P1 =
  **1.83× plain greedy** (40.86). S8 mtp2 re-baselines may proceed on
  this branch; the 39.9-era record is superseded (it paid the N4 tax).

Refrigerated (updated): register-prefix widen to B=32 (kernel change,
full gate protocol, needs a >16-concurrent-decode serving workload to
A/B); `MARGIN=0` drop; grid sweep 512/2048.
