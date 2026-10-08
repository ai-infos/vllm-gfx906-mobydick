# Multi-batch prefill slowdown — the O(live-context) per-step tax

**Analysis (2026-09-11, boot Y8).** Scope: why a B=4/120k prefill batch takes
~75 min/sample when a lone B=1/120k prefill takes ~9 min, and whether the
multi-batch context-growth effect can be cut. Builds on the validated step
model in `ttft-prefill-stall.md` §10 and the owner analysis §12/§13.

> **CORRECTED (2026-09-11, arbiter review):** the original draft called the
> slope "CPU-side / host-bound (T1 strengthened, §11)". That framing predated
> the §12/§13 results and contradicts them: **T1 is REFUTED** (§12.3 — the
> worker main thread blocks in a KFD wait at 0.22 cores during a stall; the
> corrected per-TID census §13.8–13.10 shows no host thread whose CPU grows
> with n) and the per-step bulk is **92 % chunk-linear GPU work** (§13.2–13.3).
> The owner is **UNRESOLVED — host pass vs GPU-side pool-wide op**, with D1c's
> cross-request taxation (§13.8) excluding all per-request own-KV
> explanations. Sections below have been rewritten to this state; MBT-1 is
> promoted to owner discriminator and runs FIRST.

## Verdict

1. **The multi-batch slowdown is a per-step cost proportional to the SUM of
   the live requests' contexts (A), not a per-request effect.** The step model
   `c(n) ≈ 2.63 s + 34.1 µs·n` (n = cumulative batch tokens, a proxy for A)
   predicts every B=4 cell within ~5 %. A B=4/120k batch accumulates 480k
   cumulative tokens → the O(n²) tail term dominates: ~5320 s (89 min) model
   vs ~568 s (9.5 min) for a lone 120k. The clean s0 measured 75.4 min — the
   tax is real and is the whole difference between B=4 and 4×B=1.
2. **The slope's owner is UNRESOLVED — host pass vs GPU-side pool-wide op.**
   What IS established constrains it hard:
   - **Cross-request (D1c, §13.8):** a decode-resident 67 k context taxes a
     concurrent 2 k prefill by ~4.8 s/step — the op walks ALL live contexts
     per prefill step. Per-request own-KV work (including §13.3's original
     "FA over the full KV" guess) is therefore EXCLUDED; the tax is overhead
     by construction, and only its location (host vs GPU) is open.
   - **Candidate class:** shared per-prefill-step metadata/pool-walk over live
     contexts — block-table / paged-KV metadata, varlen prefill metadata, GDN
     live-state pool handling — either host (a Python/C walk in the runner or
     engine core) or GPU (a kernel reading all live blocks/pages).
   - **The corrected CPU census leans GPU-side:** §12.3 + §13.8–13.10 show no
     thread whose CPU ramps with n during a stall (main blocks at 0.22 cores;
     the burner is constant ~0.99 and accepted as KFD-runtime event
     processing with no t/s impact). A 34.1 µs/tok host pass would surface as
     a thread ramping toward +0.4–0.5 cores by the end of a 64 k prefill.
     The one uncovered host spot is the engine core's per-TID detail (E3
     covers it).
3. **Consequence for optimization:** the knobs (O1 chunk size, O2 concurrency)
   act on step count and A regardless of owner and are testable today. The
   root fix is owner-conditional: a host pass → de-µs it (O3, E3 names it); a
   GPU-side pool-walk → kernel/metadata-path fix, with the parked per-kernel
   breakdown (kineto build fix, §13.1) as arbiter. **E1 below is the cheap
   owner discriminator — run it before E3.**

## The pattern (measured, clean, prefix-OFF — boot Y8 s0)

B=4, 4×122880 simultaneous, bt=1024, TP=2, util 0.93:

| request (FIFO) | ttft | live-context A during its prefill |
|---|---|---|
| 2nd (first to complete) | 8.8 min | own 0→120k (avg ~60k) |
| 3rd | 25.7 min | ~120k resident + own (avg ~180k) |
| 4th | 49.9 min | ~240k resident + own (avg ~300k) |
| 1st (last) | 74.9 min | ~360k resident + own (avg ~420k) |

Each successive prefill is slower because more requests are already live and
their contexts are added to A. Total wall 75.4 min vs ~9.1 min for a lone
120k (B=1 record 567 s). **B=4 costs ~8× a single prefill, not 4×** — the
extra 4× is the O(live-context) tax.

## The step model (validated, `ttft-prefill-stall.md` §10.3)

Per 1024-token step, FIFO prefill order, one decode token per running request:
`c(n) = 2.63 s + 3.41e-5 s/tok · n` (n = cumulative batch tokens). Total for a
batch of N tokens:
`T ≈ 2.63·(N/1024) + 34.1 µs · N²/(2·1024)`.

- The **linear term** (2.63 s/step) is the per-step constant (GDN-state CPU
  tensor op, §13 owner frame) — paid N/1024 times.
- The **quadratic term** (34.1 µs·n per step) is the multi-batch tax — it
  grows with the running cumulative context, so it is ~16× larger for a 480k
  batch than a 120k batch (480²/120² = 16). This is the "context-growth
  slowdown."

Both terms scale with **how many steps run and how big A is during them**,
which is what the knobs below move.

## The mechanism (owner unresolved — host pass vs GPU-side pool-walk)

The taxed quantity is settled: a per-prefill-step op over the SUM of live
contexts, cross-request (D1c), prefill-only (decode steps at 120 k run ~10/s —
a 34.1 µs×120 k term would be 4.1 s/step). The location is not:

- **Host-pass variant:** 34.1 µs/tok at 1 core is ~100+ host-level ops/token
  (a Python walk is ~0.1–1 µs/op; a tight C metadata loop far cheaper) — a
  per-step walk over prefix-scaled structures (block tables, slot mappings,
  paged-KV/GDN metadata) in the worker or engine core. Tension: the censuses
  (§12.3, §13.8–13.10) show no thread ramping with n — either it hides in the
  engine core's unprofiled per-TID detail, or it does not exist.
- **GPU-side variant:** a per-prefill-step kernel/ops sequence that reads all
  live pages/blocks (paged-KV metadata rebuild, varlen prefill metadata,
  pool-walk in a custom kernel). Fits the blocking main thread (the host only
  waits) and explains why the 8 k chunk A/B saw nothing (the tax is ~5 % of a
  step there).

**E1 discriminates** (at the 120 k×B4 scale, where the tax is ~75 % of the
wall): per-step-repeated tax → wall drops ~2× at bt=2048 and ~4× at bt=4096;
chunk-invariant tax → wall ~flat. E3 names a host owner if one exists; if E1
says chunk-invariant and E3 comes back empty, the GPU per-kernel breakdown
(kineto build fix, §13.1) is the arbiter.

## Optimizations (ranked by leverage)

- **O3 — remove the per-step live-context pass (the real fix,
  owner-conditional).** If E3 names a host walk: vectorize/fuse/move-to-GPU —
  a 10–100× slope cut is plausible for a Python walk. If E1/E3 point GPU-side
  (pool-walk kernel/metadata): the fix lives in that kernel/metadata path
  instead, with the parked per-kernel breakdown (§13.1) as locator. **Gated on
  E1 (owner class) + E3 (identity).** Highest leverage either way — the tax is
  ~75 % of the 120 k×B4 wall.
- **O1 — larger prefill chunk (`--max-num-batched-tokens` 1024 → 2048/4096).**
  Fewer steps → the per-step constant AND the per-step live-context pass run
  fewer times (owner-conditional: if the tax is chunk-invariant, only the
  constant/F fraction is saved — E1 decides). At B=1 this is the known +12 % (the attention is chunk-size
  invariant, N²/2); at B=4 the overhead fraction is larger, so the gain should
  be bigger — **measure it (E1).** Risk: the inductor prefill buffer OOM at
  large chunks (known, `AGENTS.md`); bt=2048 is the safe middle, 4096 needs the
  OOM check.
- **O2 — smaller concurrency (`--max-num-seqs` 4 → 2).** Process the 4 requests
  as 2×B=2: each phase's cumulative N is 240k not 480k, so the quadratic tax is
  halved (480² → 2·240², i.e. /2). Trades in-flight throughput for a shorter
  batch wall. **Measure it (E2).**
- **O4 — KV-cache quant (FP8/INT8).** Halves/quarters KV bytes → cuts the KV
  read — but a live-block-count-proportional walk is unchanged by KV dtype,
  and genuine attention is a minor term at this shape. Low priority for this
  tax either way. (gfx906 FP8/INT8 KV support is
  the gate; likely limited on MI50.)

## Experiments (→ ROADMAP `MBT-1/2/3`)

> **E2 RESULT (2026-09-11, boot Y9): A-HALVING CONFIRMED — seqs2 wall
> 3598.0 s (60.0 min) vs seqs4 75.3 min (−20%).** Staircase 525/1534/2558/
> 3569 s (seqs4: 528/1542/2994/4494) — the benefit concentrates in requests
> #3–4 (the high-A phases, −14.5%/−20.6%); req #1–2 unchanged (A small).
> prefill agg 136.6 vs 108.6 t/s (+26%). Sub-linear vs the naive ~35%
> slope-only prediction (decode-overlap keeps finished requests live in A;
> the D1c cross-request tax charges them to whoever prefills next) — the
> live-context model is the right predictor, the phase bookkeeping is
> fuzzy. Combined with E1: **the tax = per-prefill-token × SUM(live
> contexts), chunk-invariant, A-scalable — both knobs behave as the model
> says** (O1 dead, O2 real but modest). The root lever stays O3/the H2
> kernel fix. Session note: attempt 1 wedged (#59, chronic family, ~60 s,
> fresh boot's first TP=2 load — earliest phase yet); the authorized retry
> loaded clean and completed. See degradation.md #59.

> **E1 RESULT (2026-09-11, boot Y8): CHUNK-INVARIANT — the per-step-repeated
> model is REFUTED; the tax is per-PREFILL-TOKEN × live-context.** bt=2048
> clean run: wall 4868.7 s (81.1 min) vs bt=1024 75.3 min, staircase
> 511/1501/2943/4824 s ≈ identical, prefill agg 101.0 vs 108.6 t/s. A
> per-step-repeated pass would have halved the tax at bt=2048 (predicted
> ~48 min) — it did not. Consequences: **O1 is DEAD as a tax mitigation**
> (only the B=1 GEMM-M question remains, unrelated to the tax); **O2 (halve
> A) is the only live knob** (~40 min predicted for 2×B=2 at 120k, queued
> post-reboot); **O3 reframed** — the owner behaves as per-prefill-token
> full-live-KV streaming at ~HBM bandwidth (~16 ms/token at A=480k ≈ 480k×
> 64 KB/1.9 TB/s), i.e. 10–20× the attention-FLOP floor → a kernel-
> efficiency target in long-context prefill attention (FA work line, A11),
> not a host-walk cleanup. bt4096: both attempts died in the chronic
> weight-load wedge family (#57/#58, GPU1 BACO — see degradation.md) →
> burst, reboot; invariance is established by the bt1024/bt2048 pair
> regardless. Full record: `ttft-prefill-stall.md` §13.14.

The pre-registered experiment definitions follow (kept for the record;
E1's gate question is now answered):

- **E1 (MBT-1) — chunk-size A/B at B=4/120k. OWNER DISCRIMINATOR — run
  first.** bt ∈ {1024, 2048, 4096}, same 4×122880 simultaneous, prefix OFF.
  GATE: batch wall + per-request ttft, and the bt=4096-vs-1024 ratio decides
  the owner class: per-step-repeated tax → wall ≈ **38–40 min** (tax and
  per-step constant scale with step count); chunk-invariant tax (per-token
  work over live contexts) → wall ≈ **~75 min** (flat; only F·Δsteps ≈ −50 s).
  Skip 4096 if it OOMs the inductor prefill buffer (2048 is the safe middle
  and still gives the scaling exponent).
- **E2 (MBT-2) — concurrency A/B at B=4/120k.** max-num-seqs ∈ {4, 2}, same
  4×122880. GATE: batch wall. Expect ~1.5–1.6× at seqs=2 (halved quadratic
  term); confirms the tax scales with (concurrent batch tokens)².
- **E3 (MBT-3) — pin the 34.1 µs/tok owner IF host-side.** cProfile + py-spy
  during a B=4/120k prefill (engine core + workers; in-process main hosts the
  engine core), target the per-step pass. GATE: a named op + its µs/tok — or
  an EMPTY profile, which (with E1 chunk-invariant) moves the owner to a
  GPU-side pool-walk and makes the kineto per-kernel breakdown (§13.1, parked
  on the torch build) the arbiter. Run AFTER E1: E1's outcome picks the
  interpretation of an empty E3.

## Expected impact

- **O1 + O2 (knobs, no code):** IF the tax is per-step-repeated, O1 alone cuts
  the B=4/120k wall toward ~38–40 min and O2 stacks on top; IF chunk-invariant,
  O1 buys only the ~0.14 s/step fixed overhead (≈ −1 %) and O2 is the only
  real knob (≈ −35–40 % by halving A). E1/E2 decide which world we are in —
  that is their primary value, ahead of any mitigation.
- **O3 (owner-conditional, code):** a named host walk → a 10–100× slope cut
  takes the wall toward the ~15–25 min GEMM/attention floor. A GPU-side
  pool-walk → kernel/metadata fix, same order of payoff, different code path.
  Gated on E1 + E3.

## Open / honest

- The exact chunk-dependence of the tax (does a 4096 chunk cut it ~4×, or is
  it chunk-invariant?) is **not yet measured** — E1 resolves it, and at the
  120 k×B4 scale the answer IS the owner verdict (per-step overhead vs
  per-token work). The +12 % B=1 chunk-size number is NOT evidence here: at
  8 k the tax is ~5 % of a step, invisible to that A/B.
- The 34.1 µs/tok owner is **unresolved** (host pass vs GPU-side pool-walk;
  the census leans GPU-side) — E1 + E3 resolve it. Until then O3 is a
  hypothesis, not a plan, and the earlier "host-bound" framing is retracted
  (see Verdict §2).
- This analysis is for the **greedy** B=4/120k shape. The mtp3b4 arm (k=3)
  changes the per-step token count (1+k verify) and may shift the tax balance;
  re-validate on that arm if it matters.
