---
name: upstream-merge
description: "Merge a new upstream vLLM release into the gfx906 fork (2x MI50, ROCm 7.14) and validate it. Use when starting or reviewing a `gfx906/vX.Y.Z` merge train, resolving merge conflicts against `upstream/releases/*`, triaging merge-induced breakage, or when a post-merge build/PPL/serving failure could be a semantic (non-conflicting) merge bug. Derived from the v0.28.0/v0.29.0/v0.30.0 merges, 2026-08-26..2026-09-26."
---

# Merging an upstream vLLM release into the gfx906 fork

`main` is a permanent fork of `vllm-project/vllm`; each upstream release is
merged in on a `gfx906/vX.Y.Z` branch (see `docs/gfx906/AGENTS.md` merge-train
rules, and `docs/gfx906/MERGE-0.30.0-review.md` for the conflict-triage sheet).

**The one thing to internalise: a clean `grep '<<<<<<<'` proves nothing.**
Most of the real bugs found in the 0.28/0.29/0.30 merges were *semantic*
conflicts — places where git auto-merged happily but the result is wrong or
crashes, because upstream changed a signature/definition in one file while the
fork's caller sat in another. They were found by **building and running**, not
by reading conflict markers.

## When to use

- Starting the `gfx906/vX.Y.Z` merge train (branch cut + conflict resolution).
- Reviewing a completed merge before promotion.
- A post-merge failure (build, PPL probe, first serving init) that smells like
  merge fallout.

Do **not** use it for feature work on top of a merged base — that has its own
dev-log conventions (`docs/gfx906/AGENTS.md`).

## Approach

### 0. Scope first (before touching the tree)

```bash
MB=$(git merge-base main upstream/releases/vX.Y.Z)
# CONFLICT COUNT — `merge-tree --name-only` also prints "Auto-merging" lines on
# stdout, so always filter for the marker (this miscounted 31 as 148 once):
git merge-tree --write-tree main upstream/releases/vX.Y.Z 2>&1 | grep -c '^CONFLICT'
git merge-tree --write-tree main upstream/releases/vX.Y.Z 2>&1 \
  | grep '^CONFLICT' | sed 's/.*Merge conflict in //' | sort -u > /tmp/conflicts.txt
```

Classify each conflict before resolving (the review sheet's classes):
**L** our live gfx906 code (hand-merge) · **G** our code gated off by default
(check the record — *off-by-default ≠ dead*) · **C** an upstream commit we
carry (take upstream's version) · **D** bindings/CI/test glue.

Per file: our deltas, their deltas, our substantive commits.

```bash
while read -r f; do
  echo "== $f"
  git diff --numstat $MB main -- "$f"
  git diff --numstat $MB upstream/releases/vX.Y.Z -- "$f"
  git log --no-merges --format='%h %s' main --not upstream/releases/vX.Y.Z -- "$f"
done < /tmp/conflicts.txt
```

Sequencing rules learned: **do not merge first**, and **not onto an rc** —
merge a released tag (`vX.Y.Z`), or you re-merge the same files when final lands.

### 1. Cut the branch and merge

```bash
git checkout -b gfx906/vX.Y.Z main
git merge --no-commit --no-ff vX.Y.Z    # stop and resolve by hand
```

For a pure upstream-carry file, `git checkout --theirs -- <f>` is right — but it
replaces the **whole file**, discarding any of our non-conflicting hunks in it.
Only do that when the whole file is a carry.

### 2. Static sweep (before any build) — script: `scripts/post-merge-sweep.sh`

```bash
.agents/skills/upstream-merge/scripts/post-merge-sweep.sh
```

It runs: no conflict markers · tree-wide `ruff --select F821,F811` ·
in-memory syntax compile · a duplicate top-level `def` scan · "resolved modules
import". This is what caught the 0.29 merge's undefined names.

### 2b. Cross-check the release notes' breaking-changes list against the fork

```bash
gh release view vX.Y.Z --repo vllm-project/vllm | sed -n '/Breaking Changes/,/New Contributors/p'
```

For every removed symbol/env var/behavior, grep the fork for it. A removal the
merge took **without a conflict** can leave a fork guard or call site silently
dead — e.g. `CommonAttentionMetadata.seq_lens_cpu` removed in v0.30.0 (#55353)
left the fork's `#47042` chunked-continuation guard running through
`getattr(..., "seq_lens_cpu", None)`, which is now always `None`. Re-derive from
the replacement field (`seq_lens_cpu_upper_bound`) or delete the dead branch.
Also classify each item as *applies* / *not applicable* and record why.

### 3. Rebuild the C++/HIP extensions — mandatory, in-tree `.so` is tracked-out

The installed `.so` keeps the **old op schemas** after a merge that touches
`csrc/`, and the merged Python will call the new arity. Rebuild before trusting
any run (full recipe: `docs/gfx906/running.md` §0):

```bash
export PATH="$PWD/.venv/bin:$PATH"
export VLLM_VERSION_OVERRIDE=$(grep -oP "__version__ = '\K[^']+" vllm/_version.py)
export FETCHCONTENT_BASE_DIR=/tmp/vllm-deps
export TRITON_KERNELS_SRC_DIR="$PWD/.deps/triton_kernels-src/python/triton_kernels/triton_kernels"
MAX_JOBS=10 HIP_VISIBLE_DEVICES=0 .venv/bin/python setup.py build_ext --inplace
```

Then verify the schemas actually changed:

```bash
.venv/bin/python -c "import vllm, torch; print(torch.ops._C.gptq_gemm.default._schema)"
```

### 4. Runtime validation ladder (in order, cheap → expensive)

1. **Import the resolved modules by name** (`post-merge-sweep.sh`).
2. **The fork's own unit suites** for the touched areas — these encode fork
   invariants upstream's tests do not:
   `tests/quantization/test_moe_wna16.py`, `tests/test_config.py`,
   `tests/kernels/attention/test_minimax_m3.py`, `tests/models/minimax_m3/`.
3. **PPL gate probe** on **both a dense and an MoE** model (different quant
   paths) — the dense one gates `gptq_gemm`/`ExllamaLinearKernel`, the MoE one
   gates the WNA16 oracle + `moe_gptq_gemm_gfx906`:
   ```bash
   env VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 \
       FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE BENCH_DTYPE=float16 HIP_VISIBLE_DEVICES=0 \
       BENCH_MODEL=<ckpt> .venv/bin/python benchmarks/kernels/gfx906/ppl_probe.py
   ```
   Recorded bands: Qwen3.8-27B-AWQ-INT4 10.5472–10.5516 · Qwen3.5-35B-A3B-AWQ
   15.94–16.02 · Nemotron-3.5-Lightning-30B 26.96–27.02.
4. **Serving smoke / bench** (TP=2): `docs/gfx906/_serve_tp2_gfx906.sh` +
   `docs/gfx906/_bench_serve_grid_gfx906.py`, or `vllm bench serve`.

### 5. Commit what you validated, then record

- Commit the merge, then the fixes as **separate** commits, then the docs.
- **`git status` must be clean the moment you claim validation** — the 0.29
  merge validated a working tree whose fixes were never committed (`db19388a30`).
- Record in `docs/gfx906/CHANGELOG.md` (one dated entry) and append the outcome
  to the `MERGE-*review.md`; update `ROADMAP.md` UP-3.

## Lessons from past merges (each one cost a session)

1. **Rebuild before validating.** The 0.30 tree's `.so` still exposed
   `gptq_gemm(..., b_g_idx, ...)`; the merged Python passes 7 args, so the first
   AWQ/GPTQ call would have crashed. Query the op schema after every build.
2. **Undefined names from semantic conflicts.** 0.29 (`7e886d22bf`): a merged
   method kept one fork line inside upstream's rewrite → `NameError` on dummy-MM
   profiling (`qwen3_vl.py`); and the fork had deleted upstream internals that
   upstream's new indexer calls (`rocm_fp8_*` → 8× F821). Guard: tree-wide
   `ruff --select F821,F811`, not just the conflicted files.
3. **A signature change in file A breaks the fork's caller in file B, with no
   conflict.** 0.30: `_process_weights_gfx906` still returned a 14-tuple against
   the new 10-tuple API; `auto_awq.py` used the old `gptq_gemm`/`gptq_shuffle`
   arity; `envs.py` had lost an env var the fork's `arg_utils.py` still reads;
   `use_v2_model_runner`'s fork cases broke upstream's `SimpleNamespace` tests.
   Guard: import smoke + the feature's tests; a clean merge is not a correct one.
4. **Your own resolution is a prime suspect.** 0.29 produced a duplicated
   `dot22_8_f` declaration (caught by the C++ build); 0.30 produced
   `**score_kwargs, **_index_score_launch_kwargs()` — a duplicate `num_warps`
   keyword that only raises when the kernel is *executed*. Parsing is not
   running: exercise every resolved kernel launch.
5. **Fork tests are the spec for fork invariants — run them, and repair them.**
   The `AttentionConfig(indexer_kv_dtype="fp16")` test was red and exposed a
   latent dtype bug (the fork's Literal accepted `"fp16"` but nothing
   canonicalized it to `"float16"`). `test_fp32_kv_config.py` imported a helper
   that never existed, so it had *never collected*. A skip/deletion needs a
   reason naming what now covers the behaviour.
6. **Merging creates duplicates.** 0.30 had two dead `minimax_m3_index_decode`
   definitions and a byte-identical duplicate `test_dflash2_*`. Guard: the
   duplicate-`def` scan in the sweep.
7. **Upstream feature removal ≠ remove your orthogonal work.** #54809 removed
   GPTQ group/dynamic activation ordering across the whole quant stack; the fork
   adopted the removal but re-ported its **M=1 4-bit max-ilp** scheduling onto
   the new kernel skeleton (the twin in `q_gemm_m1_maxilp.cu`). Document the
   capability narrowing (gfx906 GPTQ act-order checkpoints no longer load).
8. **Keep a lockstep twin in lockstep.** `q_gemm_m1_maxilp.cu` is a renamed copy
   of the base 4-bit kernel; after the merge it must be re-synced *semantically*
   (the header's "normalized textual diff" will show cosmetic wrapping only —
   check that it is only wrapping).
9. **Re-check fork defaults after every release.** 0.29 made Model Runner V2
   upstream's default; the fork must keep `VLLM_USE_V2_MODEL_RUNNER=0` in every
   recipe. Same class: env vars upstream deletes, cudagraph capture ladders,
   registry duplicates.
10. **Off-by-default ≠ dead, and sweep stale verdicts first.** Check the record
    (ROADMAP/DEVLOG/DEAD-ENDS) before deleting a flag; fix code comments that
    assert a default or a "pending gate" the record has overtaken
    (`docs/gfx906/AGENTS.md` merge-train rule 6).
11. **Keep the build stack pinned.** 0.29: take upstream's `requirements/build/*`
    changes as a commented note, not as a silent toolchain swap.
12. **The merge must not be the first thing tested.** Land urgent upstreamable
    fixes first; a merge invalidates the base a tester is serving on.
13. **Read the release notes' breaking-changes list, then grep for each removal.**
    An upstream removal with no conflict marker leaves fork code that still
    *references* it — often behind a `getattr(..., None)`/`is_set(name)` guard that
    turns a real check into a no-op or an exception. v0.30.0: `seq_lens_cpu`
    removed killed the fork's `#47042` guard (fixed 2026-09-26); the
    `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` env removal crashed `arg_utils.py`'s
    deprecated-env read. Both were in the notes under "Breaking Changes".

## Repo hazards that bite during a merge session

- **`pkill -f "<script>"` matches your own command line** when the pattern
  appears in a heredoc you just wrote — kill by PID, or bracket the pattern
  (`vllm[ ]serve`). This killed an agent shell mid-command.
- **Launch long sessions detached** (`setsid nohup … < /dev/null &`) — tool
  aborts kill the whole process group, not just the child.
- **Disarm leftover profiling plugins** before measuring: armed
  `/local/tmp/mtp1/*_arm.cfg` (`agdn`, `pfk4`, `syv9`) install forward hooks that
  call `os.path.exists`/`open` inside the graph and break inductor AOT compile.
  Set `VLLM_PLUGINS=gfx906_fa` (the fork's own plugin) for a clean compiled boot.
- **`/tmp` is wiped on reboot** — put anything that must survive in `/local/tmp`.

## Definition of done

- [ ] No conflict markers; `post-merge-sweep.sh` clean.
- [ ] Extension rebuilt; op schemas match the merged Python.
- [ ] Fork unit suites for the touched areas green (with skips justified).
- [ ] PPL probe green on one dense + one MoE model, inside the recorded band.
- [ ] Serving smoke/bench on TP=2 (or the touched path) coherent.
- [ ] `git status` clean; merge + fixes + docs committed separately.
- [ ] `CHANGELOG.md` + `MERGE-*review.md` + `ROADMAP.md` updated; residual risks
      and untestable paths (VRAM-bound models, non-runnable architectures) named.
