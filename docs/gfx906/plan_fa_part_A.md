# Part A plan — repack the Q8 tile layout into aligned quants/scale planes (M6)

Status: **EXECUTED 2026-08-28 (round 11) — microbench hard stop-gate
FIRED: flip question DEAD-END, code change NEUTRAL; MERGED to main
2026-08-29 as loader hygiene** (`02d197189f`; same-boot B=1
A/B PASS — slope −4.3/−4.8 %, bit-identical, merged suite 74/74).
Outcome: `DEVLOG-muse-glimmer.md` round 11 + `DEAD-ENDS.md` (`MG`
row). Plan text below is the pre-execution record (rev 2).

Part B is gated and merged (round 10). This revision incorporates a
two-reviewer plan review (since deleted after folding): corrected diff
scope (the K tile is shared by BOTH LEGACY modes — this is not a
LEGACY=0-only change), stated regime tension, hard microbench
stop-rule, corrected flip-gate semantics, and a provisional mark on
the unrecorded F2 numbers.

## Inputs (what this plan must survive)

- **F1 erratum (round 10, `fa-legacy-m0-m6-code-rev.md`):** the
  B=4 loss mechanism (strided Q8-slice reads) is a *leading but
  unconfirmed* hypothesis — the in-process Sq=1 A/B on the identical
  strided-read path was a wash. Part A's benefit theory (misaligned
  reads cost measurable wall-clock) is therefore **unproven on the
  B=4 point**; the one point where LEGACY=0 demonstrably loses
  *despite a leaner byte count* is **B=1** (M5: −2.5…−3.7 % @2k/8k;
  review's secondary data point: fused-Q8 gather moves ~808 B/row
  incl. writes vs ~904 for LEGACY=1's fp16 gather, yet LEGACY=0 is
  still slower). Part A is aimed at the B=1 gap, not the
  (already-fixed) B=4 gap.
- **F2 in-process A/B (run 2026-08-28; recorded in
  `DEVLOG-muse-glimmer.md` round-10 erratum):** B=4/Sq=1, record
  recipe, pp8192/tg256, 4 samples, boot M. Arm A (LEGACY=1 fp16
  gather): 6.091/6.080/6.075/6.078 (≈6.081 — reproduces the
  round-7 record). Arm B (LEGACY=0 fused-Q8 gather, post-Part-B
  reroute): 6.097/6.094/6.087/6.086 (≈6.091). **Wash** (+0.16 %, in
  noise); against round 10's same-config direct-paged number
  (6.064) the rerouted gather is ≥ direct-paged at B=4 in-process
  too. Per the open question below (Arm B outcome): the B=4
gather-vs-gather question is closed at parity and the B=1 gap is
  the entire remaining LEGACY=0 deficit — proceed per the test
  plan.
- **Decode-path regime (DEVLOG-fa-kernel-batches.md 2026-08-28) — and the
  tension it creates:** B=1 decode is *assessed*
  gather-HBM-bound (~2.7× at D=128, NC2=8) — ALU-only fixes cannot
  surface there; a layout fix must reduce *read* cost. **The tension
  the plan must state honestly:** on the B=1 gather path the repack
  changes *no HBM traffic* — the alias bytes [0,136) of each 256-B K
  row are contiguous in both layouts, the gather kernel's cache-side
  reads are byte-identical, and the tile-side sector count is
  identical (136 B ⊂ 5 sectors either way). What remains is the FA
  loader's *instruction* stream (today the 34-B struct load at
  2-mod-4 tile offsets forces narrow loads; planar A1 ≈ fewer, wider
  loads) plus the write-side scatter. Under a strict HBM-bound
  regime, issue-count savings cannot surface — so either the regime
  input is wrong (plausible: it is an analytic roofline, and M5's B=1
  loss *despite fewer bytes moved* is itself evidence the achieved
  throughput is not HBM-limited) or the hypothesis cannot deliver.
  The microbench step (2) is the arbiter and is a **hard stop-gate**
  (see Test plan).

**Expectation arithmetic (do not skip):** at NC2=8 (ncols=8) the KQ
dot costs ~290 VALU ops/KV row while the tile loader costs
~68 VMEM instructions today (4 blocks × ~17 loads) vs ~20 planar —
i.e. the loader is ≤ ~15 % of FA-kernel issue, so the e2e ceiling is
~1–3 % even if loader cost vanishes entirely and the kernel is fully
issue-bound. That is *at* the gate threshold, not comfortably above
it. Part A's realistic upside is parity-plus-noise; its value is
mechanism resolution + Part C groundwork (see HYPOTHESIS).

## HYPOTHESIS

If the LEGACY=0 B=1 deficit (−2.5…−3.7 % vs LEGACY=1) is caused by
the interleaved `block_q8_0` layout — in the **gather tile** (34-B
block strides at 2-mod-4 offsets forcing narrow loader instructions
and scale loads scattered inside the structs) and in the **alias
write/read addressing** — then repacking the same 136 B into a
**quants plane** (D bytes) plus a **scale plane** ((D/32)×2 bytes)
recovers the B=1 gap on the M5 serving grid — *without changing the
byte budget, the row stride, the alias contract, or the numerics*.

Stated cost model (what exactly is supposed to improve, so a null
result is unambiguous): (a) FA-loader VMEM instruction count per
tile row drops (~68 → ~20 at D=128 A1; fewer still at A2),
reducing issue pressure and VMEM-queue occupancy — NOT sectors, NOT
bytes; (b) scale loads coalesce from four scattered 2-B reads to one
uint2; (c) the write path changes addressing only (same bytes).

Falsified if: the microbench stop-rule fires (step 2), or the
serving gate shows the B=1 deficit unchanged (±noise) after the
repack (then the B=1 gap is elsewhere — quantize write path, FA
Q-side, or the gather kernel's own traffic — and Part A is a
DEAD-END for the flip question).

## Layout

Same row byte count as today — `(D/32)×34 == D + (D/32)×2`:

```
today (interleaved block_q8_0 × 4, D=128):          Part A (planar, D=128):
[ d0 | qs0[32] | d1 | qs1[32] | d2 | qs2[32] |     [ qs0[32] qs1[32] qs2[32] qs3[32] | d0 d1 d2 d3 |
  d3 | qs3[32] ]  = 136 B of the 256-B fp16 row     ]  = 128 B quants + 8 B scales, same 136 B
```

- Quants plane: contiguous int8. **In-cache alignment caveat
  (review R4): the row start is 256-B aligned only at Hkv=1** —
  `cache_head_stride` = 136 B at D=128, so head h≥1 rows sit at
  odd multiples of 136 B (8-B aligned). Muse TP=2 (Hkv=1/rank) is
  fine; Qwen3.8-27B TP=2 (Hkv=2) breaks the 16-B claim for half
  its heads — affects the opt-in direct-paged path only. In the
  *gathered tile* (136-B row stride) odd rows are 8-B aligned →
  A1/A2 below.
- Scale plane: (D/32) fp16 scales packed together, 8-B aligned —
  one uint2 per row covers all blocks (today: four scattered 2-B
  reads inside the 34-B structs).
- D=256 (272 B) generalizes cleanly (272 = 16×17 → tile rows stay
  16-B aligned; the suite's D=256 default exercises it). **D=64
  caveat (review R6): a 68-B tile row stride is 4 mod 8, so A1's
  8-B uint2 chunks fail (4-B alignment only) — D=64 would need A2
  or narrower loads.** No current model uses D=64; keep the plane
  offsets a function of (quants_bytes, scale_bytes) regardless.
- The fp16 K row's bytes [136, 256) stay dead space (as today);
  COW/prefix-cache semantics unchanged (whole-row page copies move
  the same region).

**A1 (default): keep the 136-B gather-tile row stride.** Loads from
the tile are 8-B-aligned (int2/uint2 chunks; legal on CDNA1) — no
buffer growth, no copy-size change, smallest diff.
**A2 (fallback if A1's gate is marginal): pad the tile stride to
144 B** → tile rows 16-B aligned → 16-B int4 quants loads; costs
+5.9 % gather-buffer bytes and a slightly larger copy. Decided by
the launch-regime microbench (evidence only), not by theory.

## Diff scope (re-audited 2026-08-28; corrected by the plan review R1)

**CORRECTION (review R1, blocking in rev 1): the K gather tile is
shared by BOTH LEGACY modes** — one `_k_gather_buf` per worker, one
FA tile loader. Changing the tile layout therefore changes the
**production LEGACY=1 decode path**, not just LEGACY=0. There are
**three tile writers**, not one:

1. **Alias writes — `csrc/gfx906_fa/gfx906_fa_quant.cu`,
   `reshape_and_cache_q8`** (LEGACY=0): write each block's 32 quants
   at `row + b×32`, scale at `row + D + b×2` (today: `{d, qs}`
   struct at `row + b×34`).
2. **Fused gather+quantize — `csrc/gfx906_fa/gfx906_fa_gather.cu`,
   TWO call sites** (`gather_paged_kv_q8_kernel` V1 ~L540, the
   persistent V2 ~L726; both call `quantize_block_q8_0_halfwarp(
   k_src + b*32, k_dst + b*Q8_0_BYTES, …)` into `k_q8_out`): this is
   the **LEGACY=1 production decode path** (`GFX906_FA_FUSED_QUANT`
   default on). Same addressing change as (1).
3. **Two-kernel fallback — `quantize_q8_0_dense_kernel`**
   (`gfx906_fa_quant.cu` kernel 1, LEGACY=1 `GFX906_FA_FUSED=0`):
   also a tile writer, same change.

Unchanged: every stride, buffer shape, TORCH_CHECK, the Python alias
view (`gfx906_fa_backend.py:501` — layout-agnostic byte-prefix
slice, review-confirmed), the gather buffer sizes; the Q-side
`block_q8_0` (register struct); the FA loaders' *LDS output format*
(already planar: `K_values` int8 + `K_scales` half are separate
arrays — the inner Q·K loop, the dp4a dot, and P·V are untouched).

Read paths — `csrc/gfx906_fa/kernel/fattn-q8.cuh` (the two Q8 K tile
loaders, gathered-tile ~L208 and paged ~L278) and
`fattn-q8-paged.cuh` (audit at implementation time; it consumes the
same loaders via `PagedCacheView`): replace the 34-B struct load
with a quants chunk load (`row + b×32`) + scale load (`row + D +
b×2`).

`gfx906_fa_launcher.cu` (~L160): `nb10/11/12/13` are pure
row/head/batch stride arithmetic (review-confirmed stride-only
usage) — byte-invariant; update comments only. The LEGACY=0
byte-copy gathers (V1/V2 K pass) ARE layout-transparent (1:1 byte
copies) — no change there.

Unit tests (`tests/kernels/attention/test_gfx906_fa.py`):
(a) layout pin on the **alias** — known fp16 row →
`reshape_and_cache_q8` → assert quants at [0,D) + scales at [D,
D+(D/32)·2) byte-exact; (b) layout pin on the **tile** (review R7) —
one LEGACY=1 fused-quant forward (`GFX906_FA_FUSED_QUANT=1`) and one
LEGACY=0 fused-Q8-gather forward, each asserting the *tile* bytes
are planar — this is the R1-class guard: a missed tile writer fails
here, not in a downstream numeric mismatch; (c) the existing 60/60
suite must pass **unchanged** — the bit-identity proof that the
repack changed storage, not numerics.

## Test plan

1. **Unit (GPU1 or GPU0, ~1 min):** layout pins (alias + both tile
   writers) + full suite. A failure of the suite with the pins
   passing means a *reader* was missed in the audit — the audit
   working as intended; a pin failure means a writer was missed.
2. **Launch-regime microbench — HARD STOP-GATE (review R2/R5).**
   Standalone gather+FA at B=1, L=8k/W=2048 (the round-5 M1 shape),
   old vs new layout. Deliverables: (i) **ISA dump of the current
   loader** (disassemble the built kernel; record the emitted global
   load sequence for the 34-B struct load — this converts the
   "narrow loads" premise from inference to measurement, review
   C-F2) and of the new loader; (ii) loader VMEM instruction count
   per tile row, old vs new; (iii) step time, old vs new.
   **Stop rule (numeric, decided before running):** proceed to step
   3 only if BOTH (a) the ISA-verified loader instruction count
   drops ≥2× AND (b) the standalone step time moves ≥2 %. Either
   failing = the mechanism is not load-instruction-bound at
   measurable scale → record DEAD-END for the flip question, keep
   the microbench as the mechanism resolution, do not spend the
   in-process or serving slots on the perf question.
3. **In-process e2e (evidence + early gate):** M1-recipe harness,
   pp8192/tg256/BENCH_NREQS=1, 4 samples, LEGACY=0 new layout vs
   the round-5/10 LEGACY=0 B=1 records (6.042 @pp8192 clip-on) and
   vs a same-boot LEGACY=1 arm. The in-process number decides only
   whether to spend the serving slot, not the verdict.
4. **GATE (the actual verdict): the M5 B=1 serving grid** (the
   round-9 recipe verbatim: TP=2, ngram n=5, bt4096, 6 GiB KV cap,
   capture [6,12,18,24], prefix off, filler prompts): B=1 @2k and
   @8k, **4 samples each** (s0 cold-flag protocol: drop/flag the
   first sample if it deviates >3 % from s1–s3, per the round-10
   93.5-vs-97.8 incident). **Same boot** (round-10 lesson), fresh
   boot (boot M is at 3 wedge observations in ~2.5 h), canary
   first. **Three arms, not two (review R1)**:
   - **Arm 1 — LEGACY=1 regression control:** new build, LEGACY=1,
     the standard grid. The tile layout change touches the
     production path; this arm proves it did not regress the
     default. (If dev time is short: an explicit in-process
     LEGACY=1 old-vs-new-build A/B may substitute, but the serving
     arm is preferred.)
   - **Arm 2 — LEGACY=0 new layout:** the flip-candidate arm.
   - **Arm 3 — reference:** LEGACY=1 pre-change numbers (round-10
     arm-1 config re-run on the same boot, or the same-boot arm 1
     doubles as it).
   - **SHIPPED** requires: (i) arm 1 vs arm 3 within noise — no
     production regression; (ii) arm 2 reaches **parity or better**
     vs arm 1 at B=1 @2k AND @8k. **Flip-gate semantics (review
     R3, corrected): parity does NOT flip `GFX906_FA_LEGACY`** —
     under the M5 rule only a *win* justifies flipping. Part A's
     realistic role is to neutralize the B=1 deficit (see the
     expectation arithmetic) so the flip question reduces to
     whether LEGACY=0's remaining value proposition (no repeated
     inline quantize; Part C Q4 groundwork) justifies a future
     combined flip case — a separate decision with its own gate.
   - **DEAD-END** (for the flip question) if the B=1 deficit is
     unchanged (±1 % of the pre-repack LEGACY=0 numbers) — record
     the mechanism as refuted for B=1, revert the layout (or keep
     it as hygiene if the microbench showed a real loader
     improvement and the write-path cost is a wash — a separate
     `NEUTRAL` call), and the flip stays closed on B=1.
   - PPL invariance is NOT a gate here (storage-only change →
     bit-identical by construction, pinned by the unit suite),
     unlike the M3 dot2 rewrite.

## Risks / rollback

- **The production default IS modified (review R1).** The fused
  gather+quantize kernels are the LEGACY=1 decode path; the change
  is addressing-only within the same tile bytes, and the tile
  layout pin + unchanged 60/60 suite pin bit-identity — but the
  risk section must say it and the gate must measure it (arm 1).
- **No runtime kill switch.** The planar and interleaved layouts
  cannot coexist in the same 136-B region; rollback = `git revert`
  of the single implementation commit. Acceptable *because* the
  numerics are storage-invariant (bit-identity pinned at unit
  level) — the only failure modes are performance (the gate
  measures it) and a missed writer (the pins catch it).
- **Missed reader/writer:** any Q8 site not in the audit reads or
  writes garbage. Mitigated by the two-level pins (alias + tile,
  both modes) and the unchanged suite (gather V1, paged FA, clip
  paths, ngram Sq=6 shape).
- **Write-path cost:** same 136 B written; the scale-plane scatter
  is one uint2 per row. No expected regression at the KV-write
  cadence.
- **Cudagraphs:** the FA kernel signature is unchanged (pointers +
  strides); the round-8/10 capture fixes are orthogonal.

## Cost estimate

- Diff: ~200–300 lines across 4 files (three tile writers — two
  files, ~4 call sites — plus 2–3 loader functions and launcher
  comments) + ~110 lines of new unit tests (alias pin + two tile
  pins).
- Build: one full extension rebuild (~15 min).
- GPU time: unit (~1 min) + microbench with ISA dump (~30 min) +
  in-process 2 arms (~30 min, single card) + serving gate on a
  fresh boot (arm 1 + arm 2 × 2 points × 4 samples ≈ 2–2.5 h TP=2).

## Open questions

- A1 vs A2 (136 vs 144 B tile stride): decided by the microbench
  deliverables (load width actually emitted vs +5.9 % buffer cost).
- Does the LEGACY=0 B=1 gap partly come from the *quantize write
  path*? The in-process e2e arm separates this: if new-layout B=1
  in-process shows no movement, the write path is exonerated along
  with the read layout and the deficit is FA-inner (Q-side
  quantize, dot, softmax) — routing the work to M3 (P·V dot2)
  territory instead.
- **Arm B outcome — RESOLVED 2026-08-28: wash** (6.091 vs 6.081,
  F2 input above; recorded in the round-10 erratum) → the gather
  itself is exonerated at B=4 and the layout hypothesis carries the
  whole B=1 story. Proceed per the test plan (the loss branch —
  re-opening the plan review — did not fire).
- Should the planar layout later be shared with Part C's Q4-KV
  (a q4 quants plane + scale plane at 80 B/row)? The layout
  *pattern* ports directly; the write kernel is shared. Keep the
  plane offsets a function of (quants_bytes, scale_bytes), not
  hardcoded 128/8.

## External reference

`mxxm-t/mx-llama.cpp` PR #4 ("q8 repack system", merged 2026-08-18):
independent gfx906 dp4a-based Q8_0 **weight** kernels with the same
two-plane shape (contiguous `qs` plane + separate `d` plane, row
stride padded +1 sub-block when power-of-two — explicitly a
**bank-conflict** dodge, a mechanism this plan's A1/A2 decision
should also weigh; tile stride 136 B = 4×34 is incidentally never a
power of two). Validated bit-exact PPL on 3 models. **Measured split
relevant to this plan's expectations:** their single-token mat-vec
(the byte-bound analog of our B=1 decode gather) was **neutral
(−1.1…+6.4 %)** after the same class of layout change, while their
multi-token GEMM (issue/LDS-bound, LDS-staged + double-buffered — no
FA analog) won +21…+51 % — independent on-hardware confirmation of
this plan's regime tension and ~1–3 % B=1 ceiling (Inputs). Their
one decode win (dense +6.4 %) is a fused GLU epilogue, not a layout
effect — no FA analog. Scale-plane read mechanics (`uint16` plane +
mask) and the PPL-parity gate methodology are the portable parts.
Layout-pattern and methodology reference only — different problem
(write-once static weights, no paging/COW constraints, repack at
load), not a port candidate.
