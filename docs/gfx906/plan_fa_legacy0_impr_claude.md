# Plan: LEGACY=0 salvage (M6)

> Follow-up to roadmap M6 / M5. Written by Claude Sonnet 5. **Rewritten
> 2026-08-28** after a second agent's ISA rate probe + roofline analysis
> (`DEVLOG-fa-attention.md` 2026-08-28 entry, `dequant-instructions.md`
> "Measured dot-instruction rates") falsified this plan's original Part
> A and confirmed/sharpened its original Part B. See "Revision history"
> at the bottom for what changed and why. Not started — plan only.

## Why this exists

M5's gate (`docs/gfx906/roadmap-more-models.md`) closed with **keep
`GFX906_FA_LEGACY=1`**: the LEGACY=0 TP=2 bake (boot M, round 9,
`DEVLOG-muse-glimmer.md`) measured B=1 decode −2.5…−3.7% and B=4 @2k
aggregate −27…−31% against the LEGACY=1 control, prefill a wash.

M5's devlog read was "no int8 matrix path, so the Q8 FA dot is
fp32-ALU vs fp16 FMA" — **this premise is now refuted by measurement.**
An SCEV-proof rate probe (native ISA verified via `-S` dump, `/local/tmp/dotrate/dotrate2.cu`,
recorded in `dequant-instructions.md`) found:

| op | GMAC/s | vs fp32 FMA |
|---|---|---|
| fp32 FMA | 5824 | 1.00× |
| packed half2 FMA | 13210 | 2.27× |
| `v_dot4_i32_i8` (dp4a, what the KQ loop already uses) | **25877** | **4.44×** |
| `v_dot8_i32_i4` (native i4×i4, not yet used anywhere) | **49600** | **8.52×** |

`v_dot4_i32_i8` is full-rate on gfx906 — the fastest Q8 dot available,
2× the packed-fp16 rate. It is already the instruction in
`fattn-q8.cuh`'s KQ loop (`ggml_cuda_dp4a` → `__builtin_amdgcn_sdot4`).
There is no int8-compute gap to close and no faster Q8 instruction to
swap in.

A companion roofline (D=128, B=1 decode) explains why an ALU-side fix
couldn't have shown up in the M5 numbers regardless: the read path is
HBM-bound by ~2.7× even at NC2=8, so shaving ALU ops on a path that
isn't the bottleneck has no effect on wall clock.

**What the M5 deltas actually measure**, per the same analysis
(code-level, both LEGACY arms share the identical KQ kernel and dot
instruction — the only difference is the read path):

- The Q8 side view aliases 4×34 B `block_q8_0` blocks into the first
  136 B of every 256-B fp16 K row. Every consumer (V2 fused-Q8 gather,
  direct-paged) therefore reads a **136-of-256-B misaligned slice**:
  8×uint4 + 1×uint2 + a 120 B gap per token, ≤85% sector efficiency
  (416/512 = 1.23× effective lean, not the nominal 512→392 = 1.31×),
  plus per-token tail handling. This explains the B=1 loss.
- At B≥2, direct-paged reads the same misaligned slices straight from
  pages with per-row indirection — amplifying the same misalignment
  into the much larger B=4 −27…−31% loss.

This is a **read-layout problem**, not a compute problem. The plan
below reflects that.

## Explicit non-goals

- Not switching the flip decision — `GFX906_FA_LEGACY` default stays
  `1` regardless of outcome here, until a fresh serving A/B passes the
  M5 gate (B=1 **and** B=4, per the roadmap's re-open condition).
- Not touching V's dtype/layout in Parts A/B below — V stays fp16.
  (Part C, Q4-KV, is the one item that would touch V, and it's
  PPL-gated and scoped separately.)
- Not a MFMA/matrix-core path — gfx906 (CDNA1) has none; out of scope
  for this hardware entirely.
- Not fixing M2 (per-row prefill clip) or M4 (long-context split-K
  accuracy) — separate roadmap items, may share a rebuild window but
  are not part of this plan.
- Not the FA-Q8 P·V `v_dot2_f32_f16` rewrite (M3 candidate, added
  2026-08-28 by `12ac1b7e94` after a backend-verified ISA probe:
  `v_dot2_f32_f16` is runtime-clean, `vdst = acc + a_lo·b_lo +
  a_hi·b_hi`, matching the production dense-GEMV/MoE path; P·V
  currently runs `v_pk_mul_f16`+`v_pk_add_f16`, 2 instr/2 MAC fp16-acc,
  vs 1 instr/2 MAC fp32-acc for the dot2 form). Genuinely adjacent —
  same kernel family, same probe — but it speeds **both** LEGACY modes
  equally, so it cannot move the LEGACY=0-vs-1 comparison and doesn't
  belong in an M6 salvage plan. See `roadmap-more-models.md` M3 for the
  gate (kernel A/B + PPL invariance + bit-identity reference updates).
- **Superseded — do not implement:** the original Part A of this plan
  (batching the per-block fp16 rescale in the KQ loop) is a dead end.
  It targets ALU cost on a path that measurement shows is HBM-bound,
  not ALU-bound, at B=1. Kept below under "Superseded work" for the
  record per the dev-log discipline (state the revert explicitly), not
  as a candidate.

## Part A — repack the Q8 side view into aligned planes (targets the B=1 gap)

**Hypothesis:** replacing the interleaved `block_q8_0` alias (136 B
inside each 256-B fp16 row) with two separate, aligned planes —
a 16-B-aligned quants plane (128 B/row at D=128) and a separate scale
plane (8 B/row) — turns the nominal 512→392 B/row saving (1.31×) into
an actual bandwidth win instead of today's net loss from misaligned
sector reads. Gain should grow with context (8k+ decode gains the
most, since more of the step is KV-read-bound at longer sequences).

**Design sketch:**
- Two-plane layout, same shape as `mxxm-t/mx-llama.cpp` PR #4's
  weight-repack scheme (see "External reference" below) but applied to
  the KV *cache* alias rather than static weights:
  - Quants plane: contiguous int8 `qs` bytes, 16-B aligned, no
    interleaving with scales — restores clean `uint4` burst reads.
  - Scale plane: fp16 scales, one per 32-element block, packed
    together so a reader that needs several rows' scales (KQ dot,
    same as today) reads them adjacently instead of striding through
    K-row gaps.
- This is a **layout change to LEGACY=0's Q8 side-view aliasing only**
  (the write path in `reshape_and_cache_q8`-family code and the
  gather/direct-paged read paths). It does not change `block_q8_0`
  semantics for LEGACY=1's dense/fused-quant path, which quantizes
  fresh every call and has no alignment problem today (LEGACY=1's
  gather already reads 16 aligned uint4/row cleanly, per the
  DEVLOG-fa-attention 2026-08-28 entry).
- Write-path quantize and both LEGACY=0 readers (fused-Q8 gather V2,
  direct-paged) need updating together — this is not an isolated
  one-file change; audit every LEGACY=0 read site before starting.

**Test plan:**
- Bit-identity suite (the 46/46 → 51/51 → 57/57 LEGACY=0 suite
  referenced across M1/M5) extended to cover the new plane layout —
  must still produce bit-identical KQ output vs the current interleaved
  read, since this is a storage-format change, not a numerics change.
- Standalone microbench (launch-regime evidence only, not a gate per
  `docs/gfx906/AGENTS.md`): measure actual sectors-fetched/row before
  vs after, to confirm the alignment fix closes the gap between
  nominal (1.31×) and effective (1.23×) bandwidth savings before
  spending a serving A/B slot.
- **Gate (the actual verdict):** repeat the M5 B=1 grid exactly (boot
  L/M recipe: bt4096, 6 GiB KV cap, ngram n=5, capture [6,12,18,24],
  prefix cache off, filler prompts) — 2k and 8k decode, 3 samples each,
  against the LEGACY=1 control numbers on record (111.5 @2k, ~99 @8k).
  Report per the dev-log template in `DEVLOG-fa-attention.md`,
  `VERDICT:` one of `DEAD-END`/`SHIPPED`/`NEUTRAL`.

**External reference (2026-08-28):** `mxxm-t/mx-llama.cpp` PR #4
("q8 repack system", based on `sixvolts/llamacpp-gfx906-furnace`) is an
independent gfx906 `dp4a`-based Q8_0 kernel for llama.cpp's *weight*
matmul path (not attention/KV — different problem, same hardware). Its
README describes exactly this plane-separation shape, already built
and working on real gfx906 hardware: a two-plane repack (contiguous
`qs` plane + separate `d` scale plane, row stride padded to dodge
bank/sector conflicts) consumed by `dp4a`-based mat-vec/GEMM kernels.
This is outside evidence that plane-separated, aligned Q8 layouts are
a sound and buildable pattern on this hardware — it does not replace
this plan's own bit-identity tests or serving gate, and their kernel's
GEMM tiling against static weights doesn't map onto FA's per-query KV
streaming, so treat it as a layout-pattern reference only, not a port
candidate.

## Part B — route LEGACY=0 B≥2 through the fused-Q8 gather (targets the B=4 gap, faster than Part A alone)

**Hypothesis:** direct-paged's B≥2 dispatch reads the same misaligned
136-of-256-B slices as B=1, but with added per-row page indirection —
routing LEGACY=0 at B≥2 through the fused-Q8 gather (V2) instead
(a 1:1 byte copy, half the fp16 gather's read, and the same code path
already exercised at B=1) recovers most of the −27…−31% B=4 loss
**before** the Part A repack even lands, since it sidesteps
direct-paged's indirection without needing the plane-layout change.

**Design sketch:**
- This changes **dispatch**, not layout: at LEGACY=0 + B≥2, prefer the
  fused-Q8 gather path over direct-paged, mirroring what already
  happens at LEGACY=0 + B=1.
- Check interaction with Phase C window clip (`GFX906_FA_WINDOW_CLIP`)
  and the batch-aware KVSPLIT — both currently fire on the direct-paged
  B≥2 dispatch (per the README's `GFX906_FA_WINDOW_CLIP` row: "Only
  reachable via the direct-paged path"). Rerouting B≥2 to the gather
  path under LEGACY=0 may need the M1 gather-path clip (already shipped
  for LEGACY=1's default path, `d36f7d0900`) ported to run under
  LEGACY=0 too, or the gate needs to confirm clip-off B=4 numbers are
  still acceptable without it.

**Test plan:**
- Bit-identity: confirm the fused-Q8 gather path produces the same
  output at B≥2 as it already does at B=1 — this is dispatch-only, no
  new kernel arithmetic, so this should be a small diff.
- **Gate:** B=4 @2k aggregate, same recipe as M5's control (46.7 t/s
  LEGACY=1 baseline), 2+ samples for stability (M5 saw sample-to-sample
  variance: 35.79/32.08 s1/s2 under direct-paged). Also check an 8k+
  point once available — M5's B=4 grid only covered ctx ≤2k, where the
  window clip is a no-op (this gap was already flagged in the original
  M5 gate text and is still open).

## Part C — Q4-KV via native `v_dot8_i32_i4` (the remaining instruction-level upside)

**Hypothesis:** `v_dot8_i32_i4` is measured full-rate at 8.52× fp32 FMA
(2× dp4a's rate) with packed-nibble operands on both sides and zero
unpack ALU in the dot loop. A Q4-quantized K (72 B/row vs 136 B q8 /
256 B fp16) — paired with V either staying Q8 (dot4, 136 B) or also
going Q4 — drops the per-row read to 208–144 B (2.5–3.6× leaner than
LEGACY=1's 512 B) and halves the KQ/V dot ALU again versus dot4. This
is the one candidate with genuine instruction-level upside beyond
fixing the current read-layout waste; large-context models
(Qwen3.8-27B at 256k, Muse-Glimmer at 16k+) are the intended
beneficiaries.

**This is a bigger, separate undertaking than Parts A/B** — it's a
data-format change (not a read-path fix), requires:
- A new write-path quantizer producing Q4-packed K (and optionally V).
- Q must also be i4-packed to feed `v_dot8_i32_i4` — either a
  per-forward Q8→Q4 requantize step, or quantizing Q directly to Q4.
- A new layout for the Q4 side buffer/plane (can reuse Part A's
  plane-separation approach if that lands first, or be designed
  independently).
- An accuracy gate: PPL probe bands on the existing 442-token set
  (the project's standard correctness gate — see README "Correctness
  gates"). **Q4 V is explicitly flagged as the risk item** by the
  ISA-rate analysis — llama.cpp ships q8/q4 KV caches as precedent that
  it's viable, but this project's own PPL band needs to confirm it
  holds for the models actually served here (Qwen3.5, Muse-Glimmer).

**Sequencing relative to Parts A/B:** do not start Part C until at
least Part A (and ideally B) has landed and gated — Part C only pays
off once the read-layout waste that's currently eating the B=1/B=4
losses is fixed; a Q4 format change on top of a still-misaligned
layout would confound the measurement of which change did what.

## Sequencing

1. **Part B first** if a quick win is wanted — dispatch-only change,
   smallest diff, targets the largest measured loss (B=4).
2. **Part A next** — the layout repack, larger diff, targets B=1 and
   compounds with Part B's B≥2 routing once both land (Part B's gather
   route will also benefit from Part A's alignment fix, since the
   fused-Q8 gather reads the same aliased bytes).
3. **Part C last, and only after A/B are gated** — the Q4-KV format
   change is the biggest lift and depends on a clean baseline to
   measure against.
4. Neither Part A nor Part B alone reopens the M5 flip decision — both
   B=1 (Part A) and B=4 (Part B) serving A/Bs need to be green, per the
   original M5 flip rule ("only a win flips"), before `GFX906_FA_LEGACY`
   default is reconsidered. Part C, if it ships, would need its own
   fresh A/B on top of whatever A/B landed.
5. Standalone/launch-regime numbers (bit-identity suites, microbenches,
   the ISA rate probe itself) are evidence, never the gate, per
   `docs/gfx906/AGENTS.md`'s gate rules.

## Open questions for whoever picks this up

- Should Part A's plane-separation layout be shared with a future
  native-Q8-weight effort (cf. the `mxxm-t/mx-llama.cpp` PR #4
  reference), or is KV-cache aliasing different enough (write-once
  weights vs. continuously-appended KV) that a shared layout module
  isn't worth the coupling?
- Does Part B's dispatch reroute need a new env knob (e.g. forcing
  direct-paged even at LEGACY=0+B≥2 for A/B purposes), or is a clean
  cutover fine once gated?
- Part C's Q4 V risk: worth a cheap standalone PPL check on Q4 V alone
  (K still Q8) before committing to the full Q4-KV design, to isolate
  whether V or K is the more sensitive operand?

## Superseded work (kept for the record, not a candidate)

**DROPPED before implementation** (never coded, so nothing to revert
in git — recorded here per the "state the revert explicitly" dev-log
discipline): the original Part A of this plan proposed batching the
per-block fp16 rescale in `flash_attn_tile_q8_q8_iter_KQ`
(`fattn-q8.cuh:420`/`:499`) to cut scalar `__hmul`/`__half2float`/FMA
ops from 4×/row to 1×/row, on the theory that the B=1 loss was an
ALU-side compute gap. **Refuted 2026-08-28** by the ISA rate probe +
roofline (`DEVLOG-fa-attention.md`, `DEAD-ENDS.md`): `v_dot4_i32_i8` is
already full-rate (4.44× fp32 FMA), and B=1 decode is HBM-bound by
~2.7× even at NC2=8 — the rescale ALU was never on the critical path,
so batching it could not have moved wall-clock regardless of how much
smaller the diff was. Superseded by this file's Part A (the read-layout
repack), which targets the actual bottleneck identified by the same
analysis.
