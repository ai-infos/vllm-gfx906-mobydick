# gfx906 main promotion — validation and recovery log

This log records the promotion of `gfx906/main` to the fork's actual
`main` branch. It does not describe an upstream vLLM merge: `upstream/main`
continues to be the source for future updates, while the fork's `main` now
points at the validated gfx906 line.

Promotion started from `gfx906/main` at `a4cb86c4aa` and finished locally at
`9687673e27`. The remote `origin/main` has not yet been updated because the
TP=2 recovery gate was interrupted by a GPU half-wedge and requires a host
reboot first.

## 2026-08-29 — M4 + LEGACY0-B1 review branches promoted (docs/tests/probes only)

## HYPOTHESIS

If the two review branches contain no production-code changes (verified
by diff), their promotion to `main` cannot regress serving or the
suite, and the only validation needed is that the merged tree is the
correct union of both branches.

## GATE

`git diff 04d1b878a7..main -- csrc/ vllm/ CMakeLists.txt cmake/` is
EMPTY (no rebuild required); suite/test verification was executed on
the exact branch trees during review (GPU0, boot O) and the merged tree
differs from them only in the other branch's doc files.

## What was done

- Review fixes applied on-branch first: M4 F1-F3 (`0ba4d96e47` —
  copyright line, stale line refs, AGENTS.md naming list); LEGACY0
  F1-F4 (`a7c07526fd` — stale "never-run adjudication" notes in
  roadmap M6 + README knob row, append-cost band, Hkv label, typo).
- `feat/fa-m4-splitk-accuracy` merged (merge commit below this entry).
- `feat/fa-legacy0-b1-decode` merged; one textual CHANGELOG conflict
  at the shared insertion anchor resolved by keeping both bullets
  (M4 first, chronological).

## Evidence FOR

- Union verified on the merged tree: skills present, G1 roadmap
  section present, M4 dev log copyright-free, AGENTS.md naming list
  updated AND copyright section removed (sweep), new split-K tests
  present, no conflict markers anywhere under `docs/gfx906/`.
- Reviewer-run validation this session (GPU0, boot O, canary-healthy):
  M4 probe 12/12 arms reproduce the recorded numbers, `M4-GATE: PASS`;
  4/4 new suite tests pass (74→78 confirmed); legacy0 step probe
  reproduces all 18 cells within ~1 %; append probe reproduces
  q8-alone exactly (6.6 µs).
- Merged in a separate worktree; the main checkout (paused agent's
  MoE C1 WIP on `feat/moe-c1-routing-fusion`) untouched.

## Evidence AGAINST

None — but note `gfx906/main` (`284ce5ff6a`) is now behind `main` and
`origin`/`kintegrated` remotes are untouched (push is a separate
decision). The FA suite was not re-run on the merged tree itself
(doc-only delta vs the tested branch trees).

VERDICT: SHIPPED

## 2026-08-26 — roadmap archive and local branch promotion

## HYPOTHESIS

If the fork's old `main` is an ancestor of `gfx906/main`, the gfx906 effort
can replace it with a normal fast-forward without rewriting history.

## GATE

`git merge --ff-only gfx906/main` must succeed with the old `main` preserved
at a named rollback reference.

## What was done

- Archived completed roadmap work in `docs/gfx906/CHANGELOG.md` and reduced
  the roadmap files to open, deferred, blocked, parked, or unmerged work.
- Removed superseded standalone review artifacts after preserving their
  findings in the topic logs/changelog.
- Updated the gfx906 README file table to list the generic `DEVLOG-*.md`
  pattern, not individual dev-log files.
- Created two commits for the documentation work:
  - `3c8f99f755` — roadmap/changelog archive
  - `c786b0fdec` — obsolete review-artifact removal
- Created rollback references before promotion:
  - `backup/main-before-gfx906` → `ff063e44e2`
  - `gfx906-main-pre-promotion` → `9687673e27`
- Verified that old local `main` and `origin/main` were both
  `ff063e44e2`, and that `main...gfx906/main` was `0 2720`.
- Fast-forwarded local `main` to `gfx906/main` with `git merge --ff-only`.

## Evidence FOR

The promotion completed without conflicts or force-pushes. Final local refs:

```text
main        = 9687673e27
gfx906/main = 9687673e27
backup      = ff063e44e2
```

## Evidence AGAINST

The remote `origin/main` is still at `ff063e44e2`; the remote promotion remains
pending the recovery and final push gate.

VERDICT: OPEN

## 2026-08-26 — benchmark default and single-card validation

## HYPOTHESIS

A 4096-token prefill batch is a more realistic gfx906 serving default than the
old 1024-token experiment, without changing the decode path or causing the
validated model configurations to fail.

## What was done

- Changed `_bench_gfx906.py` so `BENCH_BATCHED_TOKENS` is effective and
  defaults to `4096`.
- Updated `docs/gfx906/README.md` and `docs/gfx906/running.md` to use 4096.
- This is a gfx906 harness/recipe default; the global vLLM scheduler defaults
  were not changed.
- Committed as `a27b10ab2f`.

**GATE:** local editable venv, ROCm 7.14, `HIP_VISIBLE_DEVICES=0`,
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, `HF_HUB_OFFLINE=1`, graph mode,
`pp=2048`, `tg=256`, four samples unless noted, explicit
`max_num_batched_tokens=4096`.

## Evidence FOR

The harness logs show `max_num_batched_tokens=4096` in the scheduler for every
4096-token run. Results:

| model | util | max seqs | mean decode |
|---|---:|---:|---:|
| Qwen3.5-35B-A3B-AWQ | 0.95 | 32 | **66.44 t/s** |
| Qwen3.5-27B-AWQ | 0.93 | 4 | **24.99 t/s** |
| Gemma-4-26B-A4B-it-AWQ | 0.95 | 32 | **68.78 t/s** |
| Ornith-1.5-35B-A3B-AWQ-INT4 | 0.95 | 8 | **65.44 t/s**; last three samples 65.92 |
| Qwen3.8-27B-AWQ-INT4 | 0.90 | 4 | **24.88 t/s** |
| Qwen3.5-9B canary | 0.85 | 32 | **62.55 t/s** |

The 35B result is consistent with the existing 66.1 final-build restamp and
below the 67.39 record. Ornith remains in the same performance class as
Qwen3.5-35B. The 9B canary was clean at both the old effective 8192 setting
(62.59 t/s) and the new 4096 setting (62.55 t/s).

Persistent copies of all benchmark logs are in
`/local/tmp/gfx906-promotion-2026-08-26/`.

## Evidence AGAINST

These are validation snapshots, not strict restamps of every historical
record: old harness runs may have used an effective 8192-token batch cap, and
model-specific utilization, max-sequence, max-length, and prompt settings
differ. The Ornith first measured sample was also an outlier at 64.01 t/s.

VERDICT: SHIPPED

## 2026-08-26 — kernel and dispatch gates

## HYPOTHESIS

The promoted gfx906 line still passes the hardware gates that protect the
custom FA, MoE, and dense GEMV dispatches.

## GATE

Real gfx906 GPU tests in the local `.venv`, with ROCm 7.14 and
`HIP_VISIBLE_DEVICES=0`.

## Evidence FOR

- `tests/kernels/attention/test_gfx906_fa.py`: **28 passed**
- `tests/kernels/moe/test_gfx906_moe_gemm.py`: **51 passed**
- `tests/model_executor/layers/test_rocm_unquantized_gemm.py`:
  **31 passed, 2 skipped**
- `git diff --check`: passed

The pytest runs emitted existing deprecation warnings and a permission warning
when pytest attempted to write `.pytest_cache`; neither affected the results.

VERDICT: SHIPPED

## 2026-08-26 19:55–20:05Z — TP=2 recovery gate

## HYPOTHESIS

The promoted line can initialize Qwen3.8-27B in TP=2 with the validated
trimmed graph capture sizes and 4096-token batch cap, then execute a basic
request cleanly.

## GATE

Qwen3.8-27B-AWQ-INT4, `HIP_VISIBLE_DEVICES=0,1`,
`tensor_parallel_size=2`, `dtype=float16`, `gpu_memory_utilization=0.82`,
`max_model_len=4096`, `max_num_batched_tokens=4096`, `max_num_seqs=4`,
capture sizes `[1,2,3,4]`, followed by a 64-token generation and clean
shutdown.

## What was done

- The first attempt was invalid because Python was launched from stdin while
  vLLM required `spawn`; it failed before worker initialization.
- The corrected file-based attempt loaded, captured, generated 64 tokens, and
  returned coherent repeated text. Its process still exited with status 1
  during multiprocessing teardown and was not accepted as a clean gate.
- A second corrected attempt reached the five-shard weight load but failed on
  worker TP1/GPU1 at `SetDevice`/`copy_()` at **20:05:23Z**.

Kernel evidence:

```text
qcm fence wait loop timeout expired
The cp might be in an unrecoverable state due to an unsuccessful queues preemption
Failed to evict process queues
Failed to quiesce KFD
GPU reset begin!. Source: 4
BACO reset
GPU reset succeeded, trying to resume
VRAM is lost due to GPU reset!
Fence fallback timer expired on ring comp_1.0.0
GPU reset(1) succeeded
[drm] device wedged, but recovered through reset
```

This was a **HW half-wedge** on GPU1 (`0000:0e:00.0`), not a full wedge.
Both cards returned to 0% VRAM and `rocm-smi` remained responsive. All logs and
the TP=2 probe script were copied before any recovery attempt to
`/local/tmp/gfx906-promotion-2026-08-26/`.

## Evidence AGAINST

The TP=2 gate has no clean pass. Per the gfx906 recovery protocol, further
multi-GPU inference is stopped until a host reboot. Both `sudo -n systemctl
reboot` and `systemctl reboot` failed because this account lacks interactive
reboot authorization.

No full-wedge (`PSP resume failed`, return `-62`) was observed. The event is
recorded in both `degradation.md` and `degradation_details.md`.

VERDICT: OPEN

## 2026-08-29 — Nemotron-H onboarding branch promoted (Python-only)

## HYPOTHESIS

If the branch contains no C/C++ changes (verified by diff), promotion
cannot invalidate the compiled extension build, and the validation
already executed on the exact branch tree (tests, PPL, serving A/Bs,
and a main-vs-branch regression A/B on the reference dense model)
transfers unchanged to `main` and `gfx906/v0.28.0`.

## GATE

`git diff main..gfx906/nemotron-h-onboard -- csrc/ CMakeLists.txt
cmake/` is EMPTY (no rebuild required); the validation below ran on
`c00103a2cd` (branch tip) on GPU0 in one session.

## What was done

- `gfx906/nemotron-h-onboard` merged into `main` (merge commit follows
  this entry), `main` fast-forwarded into `gfx906/v0.28.0` (linear
  release-branch style).
- Remotes untouched (push is a separate decision).

## Evidence FOR

- Branch-tip validation (same session, GPU0): 94/94
  `test_mamba_ssm_ssd.py` (ssd_chunk_scan restructure), 104 MoE
  kernel/oracle tests, 33+2 `test_rocm_unquantized_gemm.py` (incl. the
  new fp32-mv dispatch tests), 3/3 dequant-scheme unit tests.
- Serving gates on the branch tree: Nemotron 70.40 tok/s (graph,
  pp2048/tg256, 4 samples, 70.34–70.44) after the five fixes; PPL band
  26.96–27.02 across all arms; greedy smoke coherent.
- Regression A/B vs `main` on Qwen3.5-27B-AWQ (reference dense model):
  16.40/16.38 (branch) vs 16.36/16.33 (main worktree) — identical; the
  gap to the 25.25 record exists on both trees (host-state suspect,
  predates this branch; see the session notes in
  `DEVLOG-nemotron-h.md`).

## Evidence AGAINST

None. `gfx906/main` remains behind `main` (unchanged by this
promotion); the Qwen record-gap host-state question is open and
tracked outside this branch.
