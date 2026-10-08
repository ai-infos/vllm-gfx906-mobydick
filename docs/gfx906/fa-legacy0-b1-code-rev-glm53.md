# Code review — branch `feat/fa-legacy0-b1-decode` (item #1: B=1 LEGACY gap closed)

Reviewed: 2026-08-29 (boot O), against `04d1b878a7..bcbe58be3c` — all four
commits (`654128a7f9` probe+audit, `03e9efebc2` skills, `996481ceed`
markdown-copyright sweep, `bcbe58be3c` closure), every changed line read.
Reviewer verification actually run on GPU0 this session (GPUs idle, post-
canary): both committed probes re-executed, dispatch audit re-derived from
`vllm/gfx906_fa/gfx906_fa_paged.py`, serving-bake logs cross-checked, and
all persistent `/local/tmp` records named by the dev log confirmed present
with matching contents. The serving A/B itself was not re-run (same-boot
gate already executed on boot O; a re-run would be a different boot and
therefore not the gate).

## VERDICT

**SHIP-WORTHY AFTER RECORD-HYGIENE FIXES (F1).** The adjudication is the
strongest this question has had: same-boot three-arm A/B with a cross-boot
anchor, sample spread ≤0.13 %, and the decisive B≈C row (two arms with
very different FA/gather subcomponents landing within 0.2 % in serving)
that localizes the gap off the kernel and onto a LEGACY=0-common per-step
cost. The dead-end bookkeeping (DEAD-ENDS row, CHANGELOG, verdict-first
dev log, refrigerated lever) is in place — but two committed records
still describe the same-boot adjudication as *never run* (F1), which is
exactly the staleness the cross-link rule exists to prevent. No production
code changes on this branch; nothing to revert. Probes and the
`EXTRA_SERVE_ENV` harness knob are committed (not just used).

## What was independently verified

- **Dispatch audit** (re-derived from source): `_DIRECT_PAGED_Q8` default
  `0` (`gfx906_fa_paged.py:104`), auto `min_batch=2` (`:91`, `:205`) —
  at B=1 the direct-paged branch is unreachable; LEGACY=0 B=1 runs
  `gather_paged_kv_q8` on the aliased side view (`:652`), LEGACY=1 runs
  `gather_paged_kv_quant_persistent` (`:735`). Arm C's config
  (`DIRECT_PAGED=1` + `DIRECT_PAGED_Q8=1`) forces the M5-era path. The
  audit's premises hold on the code as merged.
- **Step probe re-run** (`legacy0_b1_step_probe.py`, GPU0, eager): all 18
  cells reproduce the dev-log table within ~1 % (e.g. D=256/Sk=32768:
  A 964.2 vs recorded 970.7 µs, B−A −45.5 % vs −45.0 %; D=128/Sk=2048:
  C−A +12.6 % vs +10.9 %). The recorded table also matches the persisted
  `/local/tmp/b1_step_probe_run1.log` digit-for-digit.
- **Append probe re-run** (`legacy0_append_cost_probe.py`, GPU0): q8-alone
  **6.6 µs** — exact match; combined per-layer delta **3.8 µs** vs the
  recorded 5.9 µs → **+59.6 µs/step** vs the recorded +94.6 (see F2 —
  eager variance, conclusion unaffected either way).
- **Serving numbers, internal consistency**: 37.61/40.11 = −6.23 % →
  "−6.3 %" ✓; step-time delta 1.66–1.70 ms; minus the append term ≈
  1.56–1.60 ms ≈ the "~1.55 ms/step remainder" ✓; the B=4 parenthesized
  aggregates (38.24/38.14 vs 38.20/38.12 — also within 0.2 %) corroborate
  the LEGACY=0-common localization at a second batch size ✓; boot-N arm-A
  anchor and boot-O canary recorded ✓; both GPU1 wedges + the boot-N burst
  are in `degradation.md`/`_details` per protocol, including the
  4th-consecutive-boot pattern note ✓.
- **Foreign commits** (mid-session, reviewed since they ride this branch):
  `03e9efebc2` — both SKILL.md files reference real artifacts
  (`dequant-instructions.md`, `_probe_mem_attribution_gfx906.py`,
  `dot_isa_probe.py`), carry no copyright line, and their content matches
  repo lore (stale-`.hip.o` hazard, SIGTERM teardown, wall-alignment
  caveat). `996481ceed` — 37 files, every removed hunk is a copyright
  line or the AGENTS.md copyright section; no content damage.

## Findings

### F1 (P1, fix before merge) — two committed records still say the same-boot adjudication never ran

**FIXED in `a7c07526fd`.**

- **What**: `roadmap-more-models.md`, M6 residuals: "The never-run
  same-boot B=1 adjudication (107.2 boot M vs 111.5 boot L) remains
  available if the question ever reopens." And `README.md`, the
  `GFX906_FA_LEGACY` knob row, ends: "the LEGACY flip itself still needs
  the B=1 same-boot adjudication". Both are now false — this branch ran
  it (boot O: −6.3 %/−6.4 %) and closed the flip question.
- **Why it matters**: the dead-end rule requires the one-line revert note
  on the roadmap candidate precisely so a future grep doesn't re-open or
  re-run settled work. These two are the canonical pointers.
- **Fix**: one line each, cross-linking `DEVLOG-fa-legacy0-b1-decode.md`
  (e.g. "same-boot adjudication run 2026-08-29 boot O: B 37.61/37.56,
  C 37.55/37.54 vs A 40.11/40.12 (−6.3/−6.4 %) — flip closed"). Docs-only.

### F2 (P2) — append-cost number is a point estimate from an eager, launch-overlap-sensitive measurement

**FIXED in `a7c07526fd`** (band + bound recorded; dev log + CHANGELOG).

- **What**: the dev log/CHANGELOG record "+94.6 µs/step eager" as *the*
  append cost. My same-config re-run got +59.6 µs/step (q8-alone matches
  exactly at 6.6 µs; 6.6×16 = 105.6 µs is the no-overlap upper bound).
  The recorded value came from one run where the combined sequence
  overlapped less than mine did.
- **Impact**: none on the verdict — any value in the 60–105 µs band is an
  order of magnitude below the ~1.55–1.70 ms/step serving delta, and the
  log already labels the number eager/launch-regime.
- **Suggested fix**: record the band (≈ +60–105 µs/step eager, q8-alone
  ×16 as the bound) instead of the point estimate. One sentence.

### F3 (P2, nit) — inconsistent geometry labeling between the two probes

**FIXED in `a7c07526fd`** (dev-log clarification; probe .py left untouched
so the branch stays docs-only).

The step probe runs Qwen3.8 at Hq=16/**Hkv=2** (per-shard, TP=2); the
append probe's header and printout say "Qwen3.8 geometry … **Hkv=4**".
One of them mislabels full-model vs per-shard Hkv. Timing is
launch-dominated at 1 token (≤2 KB/layer difference), so the measured
number stands; fix the label for the future reader.

### F4 (P3, nit) — GATE-line typo

**FIXED in `a7c07526fd`.**

Dev-log GATE line reads "A 40.11/**40.11**"; the table, commit message,
and degradation row all say 40.11/**40.12**. Trivial.

### F5 (observation) — the unexplained remainder is honestly fenced

The ~1.55 ms/step remainder is explicitly marked unmeasured, the
graph-node count (16–32/step) is presented as hypothesis not fact, and
the trace-route is correctly blocked by the wall-alignment caveat. The
refrigerated lever (fuse the Q8 write into the triton append) is stored
with its own limitation stated (would not alone close 6 %). This is the
right epistemic shape; no action.

## Merge notes

- Branch carries the convention change (markdown no-copyright) + the two
  skills; merging it publishes those. `feat/fa-m4-splitk-accuracy` (same
  base `04d1b878a7`) adds a new dev log that still has a copyright line
  and inserts its CHANGELOG bullet at the same anchor as this branch's —
  expect one trivial CHANGELOG conflict when merging the second branch;
  keep both bullets.
- No `csrc/` or `vllm/` production-code changes → no rebuild, no
  suite impact (suite count unchanged; nothing here touches tests).

## Reviewer commands run (this session, GPU0, boot O)

    HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HF_HUB_OFFLINE=1 \
      .venv/bin/python benchmarks/kernels/gfx906/legacy0_append_cost_probe.py
    HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HF_HUB_OFFLINE=1 \
      .venv/bin/python benchmarks/kernels/gfx906/legacy0_b1_step_probe.py

Results: append → 36.6/40.4/6.6 µs, +59.6 µs/step (exit 0); step probe →
table above (exit 0). Logs: `/tmp/review_append_probe.log`,
`/tmp/review_step_probe.log`.

## Post-review addendum (2026-08-29, same session)

**F6 — no roadmap entry for the ~1.55 ms/step remainder (FOUND, FIXED
on-branch).** The unexplained LEGACY=0-common per-step cost existed only
in this dev log: a "not further decomposed" mechanism note plus a
"revisit only with a graph-node overhead measurement" precondition that
nobody tracked — the forward-looking candidate belonged in the roadmap
per the log/roadmap split. Added `roadmap-more-models.md` → "Decode-graph
node-overhead point (shared infra)" (G1: per-node replay-cost probe,
dummy-no-op-node A/B, falsifiable both ways) + the one-line cross-link
from this log's refrigerated section. Deliberately NOT fixed here: F1
(the stale "never-run same-boot adjudication" notes in the M6 paragraph
and the README knob row sit in the same file region and remain the dev
agent's/open decision from the review).
