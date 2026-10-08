# Code review — branch `feat/fa-m4-splitk-accuracy` (M4: long-context split-K accuracy)

Reviewed: 2026-08-29 (boot O), against `04d1b878a7..2006b508d5` (single
commit), every changed line read. Reviewer verification actually run on
GPU0 this session: the committed probe re-executed end-to-end (all 12
arms + both gates), the 4 new suite tests re-run (4/4 pass, "74
deselected" confirming the 74→78 count), the premise correction
re-derived from kernel source, and the persisted records
(`/local/tmp/m4_probe_run2.log`, `m4_suite.log`) confirmed present.

## VERDICT

**SHIP-WORTHY AFTER ONE TRIVIAL DOCS FIX (F1).** The M4 premise was
indeed stale and the corrected mechanism is verified in source: split
partials are **fp32** in both paths (`o_part`/`o_meta_split` allocated
`opts_f32` — `gfx906_fa.cpp:393-394` gather, `:1240-1241` paged; fp32
log-sum-exp combine via `gfx906_fa_split_combine`), and the only fp16
exposure is the in-register P·V accumulator (`half2 VKQ[...]` —
`fattn-q8.cuh:1028`, `fattn-q8-paged.cuh:545`), which splitting
*shortens*. My probe re-run reproduced every recorded number
digit-for-digit and both gates pass. Zero production-code changes
(defaults untouched, no `csrc/`/`vllm/` edits → no rebuild, no serving
impact); the deliverable is probes + suite pins + records.

## What was independently verified

- **Probe re-run** (`m4_splitk_accuracy_probe.py`, GPU0, seed 20260829):
  all 12 arms match the dev-log/CHANGELOG table exactly — gather
  split-16 D=256: 5.19e-3/6.64e-3; split-1: 1.90e-2/2.63e-2; paged
  split-8: 3.98e-3/4.97e-3; D=128: 5.64e-3/7.18e-3, 1.90e-2/2.73e-2,
  3.75e-3/5.51e-3 (16k/32k). Split-delta G2: 1.95e-2/2.70e-2 (D=256),
  1.96e-2/2.80e-2 (D=128), all within the 2×rel_ref(g1) bound.
  `M4-GATE: G1=PASS G2=PASS => PASS`, exit 0
  (`/tmp/review_m4_probe.log`).
- **Suite**: the two new 16k gather arms + both geometries of the new
  direct-paged L=16384 pin pass; 74 deselected → 78 total
  (`/tmp/review_m4_tests.log`, 14 s).
- **Defaults as claimed**: gather `kv_split=16` (`gfx906_fa.cpp:374`),
  paged `clamp(16/batch,2,8)` → 8 at B=1 (`:1218`); the `seq_q>2 → 1`
  guard is inert in all new tests (Sq=1).
- **Roadmap closure is clean**: the M4 section is removed (roadmap =
  what we might do), the record lives in CHANGELOG + dev log; no stale
  M4 references remain in README/running/spec-decode-roadmap.

## Findings

### F1 (P1, fix before merge) — new dev log still carries a copyright line

**FIXED in `0ba4d96e47`.**

`DEVLOG-fa-splitk-accuracy.md` line 3: "Copyright Kevin Read
<me@kevin-read.com>". This commit landed 13:22; the convention change
(markdown carries no copyright line) landed 15:46 on
`feat/fa-legacy0-b1-decode` (`996481ceed`), which this branch does not
contain — so the sweep will NOT fix this file at merge time. The
developer report's "all my markdown respects the new no-copyright
convention" holds only for the legacy0 branch's own files. Fix: delete
the line here (docs-only).

### F2 (P2) — stale source line refs in the dev log's mechanism section

**FIXED in `0ba4d96e47`.**

The mechanism is right but three cited locations are wrong on this
tree: "`fattn-q8-paged.cuh:261`" → the `half2 VKQ` accumulator is at
`:545`; "`fattn-q8.cuh:532`" → `:1028`; "`gfx906_fa.cpp:383`" → the
fp32 partial allocs are at `:393-394` (gather) / `:1240-1241` (paged).
(`gfx906_fa_launcher.cu:106` for the combine is correct.) Since the dev
log's whole point is one-pass lookup, fix the refs.

### F3 (P2) — single-entry log file vs the grouping rule

**FIXED in `0ba4d96e47`** (option: naming-list entries added to
`docs/gfx906/AGENTS.md`; folding into `DEVLOG-fa-attention.md` stays an
option for its next staleness pass).

The conventions say a log covering one experiment is a weak search
target and route FA work to `DEVLOG-fa-attention.md`. That file is
57.5 KB (2.3–2.9× the hard budget), so a fresh file is defensible — but
then either add `DEVLOG-fa-splitk-accuracy.md` to the naming list in
`docs/gfx906/AGENTS.md`, or fold this entry into `DEVLOG-fa-attention.md`
at its next staleness pass and delete the file. Pick one; one line.

### F4 (P3, observation) — the new 16k gather arms pin nc2=1

The two suite arms isolate the split dimension (NC2=1); the production
B=1 gather combo (default NC2 + split 16) is exercised at small L by
the existing "serving config" arms. NC2 packs q-heads per tile without
changing per-head numerics, so this is fine for the accuracy claim —
noted so nobody reads nc2=1 as the production config.

### F5 (P3, nit) — probe docstring wording

`run_arm`'s "Run one arm in THIS process" is written from the child's
perspective (it is only ever spawned by `spawn_arm`); harmless, but
"run one arm (child process of the grid runner)" would read better.

## Merge notes

- Expect one trivial CHANGELOG conflict against
  `feat/fa-legacy0-b1-decode` (same insertion anchor after the M2/M3
  block) when the second branch merges; keep both bullets.
- Merge order is otherwise free; if this branch merges after the
  convention change, F1 becomes visibly inconsistent and should still
  be fixed here (the sweep commit is historical, not a living rule).

## Reviewer commands run (this session, GPU0, boot O)

    HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HF_HUB_OFFLINE=1 \
      .venv/bin/python -m pytest tests/kernels/attention/test_gfx906_fa.py \
        -v -k "16384 or long_context"          # 4 passed, 74 deselected
    HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HF_HUB_OFFLINE=1 \
      .venv/bin/python benchmarks/kernels/gfx906/m4_splitk_accuracy_probe.py
                                               # M4-GATE: PASS (exit 0)

Logs: `/tmp/review_m4_tests.log`, `/tmp/review_m4_probe.log`.
