# M6 Part A — merge prep instructions (`feat/fa-legacy0-m6-partA` → main)

Status: OPEN (2026-08-29). Decision made: **merge, relabeled as a
loader-hygiene change** — the flip question is DEAD-END (gate fired,
`DEVLOG-muse-glimmer.md` round 11) but the code is bit-identical,
test-pinned, and carries a measured −2.4 % standalone B=1 decode-step
win that applies to the production LEGACY=1 path (shared tile loader).
This document is the work list for the main agent doing the merge.

## What is on the branch (3 commits over `7cdedf8de6`)

| commit | content |
|---|---|
| `af16ba1717` | `plan_fa_part_A.md` rev 2 (docs only; now marked EXECUTED) |
| `1cb643e42e` | mx-llama.cpp PR #4 learnings → `dequant-instructions.md` + `lds-layout.md` (pure docs, branch-independent) |
| `c8de3eacd0` | the code: planar Q8 layout in all three tile writers + both FA loaders, 4 byte-level layout pins, records |

Verified branch facts (2026-08-29): suite 64/64 on its own base;
decode-vs-SDPA 0.43 % in BOTH LEGACY modes; ISA-verified loader loads
10 → 6 per tile-row (D=128); the D=256 scale-plane offset bug found
in its review is FIXED on the branch; interleaved helper kept
(byte-identical). **The branch predates M2+M3 — its kernel changes
were never built or tested against them.** That is the merge risk.

## Pre-conditions

1. `feat/fa-m3-hygiene` (M2+M3+changelog close-out, `6af801938a`) is
   merged to main FIRST — partA's roadmap edit conflicts semantically
   with the slimmed Muse section that lands there.
2. GPU0 free (other agents' tests done); check the degradation canary
   (`docs/gfx906/degradation.md` protocol) before trusting any timing.
3. Do not edit the partA branch itself — merge it into a prep branch
   off main, verify, then fast-forward/merge main.

## Work list

### 1. Merge and resolve the single conflict

```bash
git checkout -b chore/m6-partA-merge main
git merge --no-ff feat/fa-legacy0-m6-partA
```

`git merge-tree` analysis (2026-08-29): exactly **one textual
conflict**, in `tests/kernels/attention/test_gfx906_fa.py` — an
append-append at the file tail (main's M2/M3 test sections vs the
Part A layout pins). Resolution: **keep both sides** (M2/M3 sections
first, Part A pins after). The other both-sides files
(`gfx906_fa.cpp`, `gfx906_fa_launcher.cu`, `fattn-q8.cuh`,
`fattn-q8-paged.cuh`, `roadmap-more-models.md`) auto-merge but are
NOT verified — step 2 is mandatory.

### 2. Hand-verify the semantic composition (the untested combination)

The planar loaders were never built against M2's tile clip or M3's
hygiene edits. The regions are disjoint by design but check each of
these in BOTH kernel files (LOCKSTEP pairs must stay identical):

- **fattn-q8.cuh / fattn-q8-paged.cuh**: Part A loaders
  (`flash_attn_tile_q8_q8_load_tile_q8` + the paged twin, ~L183/L278
  region) vs M2's `tile_clip` block (~L1066/`599`) and M3's `k0_base`
  clamp (~L1037/`570`) + the four `q_abs_row - k_pos_abs >= window`
  cutoff sites (~L696/`730`). Layout changes (byte offsets) and index
  changes (k-loop bounds) must not have interleaved in one hunk.
- **`gfx906_fa_launcher.cu`**: Part A's `nb10` stride-arithmetic
  changes vs the `tile_clip` parameter threading (M2) through the
  same signatures — all three head_dim instantiations compile-checked
  by build, but eyeball that the param order survived the merge.
- **`gfx906_fa.cpp`**: M3's final `o_meta` nullptr form must be the
  surviving version (Part A's base still had the old allocation when
  the branch was cut).
- **`roadmap-more-models.md`**: main's slimmed Muse section wins;
  re-add only Part A's "candidate 1 closed" one-liner if it did not
  survive (the M6 residuals section should already say Part A's gate
  fired — from the changelog close-out commit).

### 3. Build with the contamination countermeasure

The M3 NaN-scare rule: after any branch switch/merge touching `csrc/`,
wipe the extension's build state or the hipify skip-check may link a
stale `.o`:

```bash
rm -rf build/temp.*/CMakeFiles/_gfx906_fa_C.dir
rm -rf build/temp.*/csrc/gfx906_fa/*.hip build/temp.*/csrc/gfx906_fa/kernel/*.cuh
# then the normal editable-build rebuild (~15 min)
```

### 4. Suite gate — expect 74/74

70 (current main: 60 base + 5 M2 + 5 M3) + 4 Part A layout pins.
Any layout-pin failure = the semantic composition broke a writer/
loader pair — fix before proceeding; any other failure = resolve as
usual. If the count is not 74, reconcile before trusting the run.

```bash
HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
.venv/bin/python -m pytest tests/kernels/attention/test_gfx906_fa.py -q
```

### 5. The never-run serving sanity gate (closes the review gap)

Part A's perf evidence is standalone-only (the stop-rule fired before
the serving slot). There is no runtime layout knob after the merge,
so this is a **build A/B across commits, same boot**:

1. On main (pre-merge build): in-process B=1 decode bench —
   `benchmarks/kernels/gfx906/bench_gfx906_fa_decode.py`, Muse
   geometry, 2+ samples. Record.
2. Rebuild with the merged tree, same boot, same bench.
3. **Pass**: merged ≤ pre-merge within noise, or better (the −2.4 %
   standalone suggests neutral-to-better). **Fail/abort**: merged
   slower beyond noise → `git revert -m 1 <merge>` (single revert,
   the branch itself is untouched) and record the revert in
   `DEAD-ENDS.md` (flip the MG row to reverted).

Optional but cheap: one record-recipe serving spot-check at the end
(LEGACY=1 default, B=1 @2k) to confirm no e2e movement.

### 6. Records to update at merge time

- **CHANGELOG**: flip the 2026-08-27–28 Part A bullet's "merge-or-
  revert is pending" line to merged (add the date + the sanity-gate
  number); note the relabel (loader hygiene, not an M6 flip item).
- **`plan_fa_part_A.md`** header: "kept on this branch" → merged date.
- **`DEAD-ENDS.md`** MG row: append "merged as hygiene <date>, gate
  verdict unchanged".
- **README env table**: no new knobs (the layout is unconditional);
  nothing to add. `GFX906_FA_TILE_CLIP` row (M2) should already be
  there from the M3-branch close-out — if not, add it then.
- **Merge commit message**: lead with *"loader hygiene: planar Q8
  quants/scale planes (M6 Part A; flip question DEAD-END per gate —
  merged for the aligned-loader win and Part C groundwork)"*.

## Abort conditions (revert the merge)

- Any suite failure not resolved within one focused session.
- Serving sanity gate regresses beyond same-boot noise.
- The composition verification (step 2) finds an interleaved hunk that
  cannot be resolved by inspection — then prefer a clean re-apply of
  `c8de3eacd0`'s diff onto current main over hand-untangling a merge.

## Time budget

~1–1.5 h total: merge+resolve (10 min) · hand verification (20 min) ·
wipe+rebuild (15 min) · suite (2 min) · build A/B + optional serving
spot-check (30–40 min) · records (10 min).
