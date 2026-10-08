# PR #2 review (joochung) — the first end-to-end Qwen3.8-Flash-Next result, and what to merge

> Review of `KIntegrated/vllm-gfx906-mobydick#2` (head `joochung:gfx906/qsa-fn`,
> base `main`) and its sibling `#1` (PLE mmap offload), plus the test run behind
> them. Reviewer: this session. **Verdict: merge-worthy in parts — take 6 of their
> commits with 3 required edits; drop 2 sets; ask for 2 test artifacts.**

## Headline: the FN-8 gate is met

The tester served the real checkpoint on **4× MI50 (32 GB, PCIe-only, no XGMI)**,
`TP=4`, fp16, `max-model-len=147456`, `max-num-seqs=3`, MTP k=3, piecewise
cudagraphs, and measured **46.8 t/s at B=1**. That is the end-to-end number the
whole train was missing (we cannot load it on 2 cards), and it confirms the fp16
enablement (QSA-FN-1) works on the real model rather than only in kernel probes.

Recorded findings worth keeping (generic or model-specific, not box-specific):

- fp16 QSA is the working arm; bf16 unchanged at ~1.00× — consistent with our
  probes (fp16 `v_dot2_f32_f16` vs emulated bf16).
- **Drafter CUDA graphs are worth +6 % and cost 2.3 % of the KV cache** with MTP
  k=3 on this stack. Their `enforce_eager` attempt was a no-op: that flag is read
  only by the legacy `v1/spec_decode/` proposer, never by
  `v1/worker/gpu/spec_decode/`, and is not propagated to the draft `ModelConfig`.
- **Boot segfault (MTP + graphs)**: dies inside `gc.collect()`'s traversal
  (`deduce_unreachable`), i.e. GC is the *victim* of a heap corruption whose
  culprit is unidentified (their prime suspect: a stale tvm_ffi torch-C-DLPack
  addon keyed by filename without a torch-version hash). `gc.freeze()` +
  `gc.disable()` do **not** prevent it (an explicit third-party `gc.collect()`
  still traverses); shadowing the `gc.collect` attribute for the region does.
  **Root cause OPEN; the guard is a mitigation, not a fix.**
- `tvm_ffi` installs its own `SIGSEGV` handler and *replaces* `faulthandler`'s,
  which is why their crash dumps had no Python frames.
- Their topology has **no P2P** (each GPU behind its own root port + its own AMD
  `14a1` switch DSP; `ACSCtl ReqRedir+ CmpltRedir+`): a reconstructed fullbar
  amdgpu patch was rejected — ACS redirect is the real blocker. Dead end, worth an
  `AGENTS`-level note rather than a code change.

## Read the history shape first: do not merge their branch

Their branch's merge base with ours is `f79ebf2d44` (pre-QSA), and it **re-commits
our own work with new SHAs** (they applied the tester bundle, then merged `main`),
so a branch merge would duplicate ~20 commits of our history and fight our
`gfx906/v0.29.0-final` release. Cherry-pick their unique commits instead, and ask
them to re-target the PR at **`gfx906/qsa-fn`**.

## Their unique commits, triaged

| commit | what | call |
|---|---|---|
| `d65391dc61` | PLE ngram table → pinned host RAM via mmap-shaped CPU shards (`MmapShardedNGramEmbedding`), replacing the TP-sharded on-device embedding | **TAKE, but gate it** (see edit 2) |
| `745c35e098` | PLE ngram D2H-sync fix (stale shard index) + graph-safe shard loop; V2 GC-freeze guard; `GFX906_DRAFTER_GRAPHS` | **TAKE with edit 1**; the two PLE correctness fixes are the valuable part |
| `0aae70351e` | GC freeze: shadow `gc.collect` (the part that actually stops the crash) | **TAKE, comments trimmed** |
| `4ca518b908` | GC guard exit path: park-at-exit by default instead of unfreeze+collect (`GFX906_GC_THAW` restores the fatal exit) | **TAKE** |
| `d37e9d2aae` | `style:` repo pre-commit over 19 files | **TAKE the Python hunk only** (our files are not all ruff-format clean); it is mechanical |
| `23c2115c72` | `style:` whole-file clang-format of `csrc/rocm/dense_gemv_gfx906.cu` (273+/293−) | **DROP** — keep only their +34 content lines; a 600-line whitespace rewrite of a hand-tuned fork file hides real changes and conflicts |
| 4× 1-line commits | append rows to `docs/gfx906/degradation.md` with their ops narrative (`/ai/...`, pids, `run-vllm`, `BOOT_TRIES`, cache paths) | **DROP/REWRITE** — keep the three generic findings above, without the box-specific runbook |
| `ae6492f7c2`, `ac830d0e6a`, `78279001b5` | merge of `main`; our handover re-applied; a "log PR #2" line | drop (already ours / bookkeeping) |

Also trivial and takeable: `# shellcheck disable=SC2086` pragmas in both serve
recipes, and the missing SPDX line in `_qsa_tiny_model.py`.

## Required edits before merging

1. **Invert the drafter-graph default.** As written, an unset
   `GFX906_DRAFTER_GRAPHS` gives the drafter `CUDAGraphMode.NONE`, i.e. it
   **changes default behaviour** for every spec-decode run — contradicting the
   PR's own claim that "unset env ⇒ upstream behaviour". Our house recipes serve
   MTP k=3 with graphs, so their default would cost ~6 % (the inverse of the +6 %
   they measured). Required form: default = inherit the target's mode (upstream);
   `GFX906_DRAFTER_GRAPHS=0` disables, with their measurement in the note.
2. **Gate the PLE host-table path** behind `GFX906_PLE_HOST_TABLE=1` (default
   off). It is a *structural replacement* of the TP-sharded embedding — also not
   covered by the PR's "all changes gated" claim — and our validated path must
   stay the default; a 4×32 GB / 147 k-ctx config needs it on, which the dev-log
   note should say. It also swaps `dtype=params_dtype` for a `runtime_dtype`, so
   the two paths need a shared dtype contract.
3. **Trim the code comments to house style.** Several of their additions are
   100+-line dev-log essays inside `model_runner.py` citing boot timestamps, pids
   and their log files. Keep the mechanism (a few lines) + a dev-log pointer; the
   narrative belongs in a log, and the repo style says minimise comments. Also drop
   the `# or a cheaper sanity check` aside they left in the `rocm.py` amdsmi
   fallback — that fallback itself is a **generic, takeable robustness fix**.

## Ask them for

- The **GC-guard unit harness** (they cite 18 assertions incl. nesting, exception
  path, enabled-state restore) and the PLE-path test they used. Both are referenced
  in commit messages but are not in the PR; a shadowing guard and a structural PLE
  swap are not mergeable credit-blind.
- Re-target to `gfx906/qsa-fn`, and correct the PR body's two "default-off ⇒
  byte-identical" claims (drafter graphs, PLE host table).

## Correction (2026-09-22, after measuring it on our box)

The claim in "Required edits 1" — that their default (drafter graphs `NONE` unless
`GFX906_DRAFTER_GRAPHS=1`) would cost our configs ~6 % — **is wrong, and backwards on
our config**: with TP=2, MTP k=3, 32 k context, forcing the drafter to `NONE` is
**+7.9 %** (59.78 vs 55.40 t/s, order-controlled A-B-A; `DEVLOG-spec-decode.md`).
Their +6 % for *enabling* graphs holds on their config (TP=4, 147 k, max-num-seqs
3). So the edit still stands — keep it a switch with the upstream default — but for
the opposite reason: the sign is config-dependent, so neither default should be
imposed by either measurement.

## Verification we can and cannot do here

Can: build (the `.cu` content lines), `tests/models/qwen4_exp/` +
`tests/kernels/mamba/` + `tests/kernels/moe/test_c4_layer0_quant.py`, the tiny rig
(which exercises PLE + mamba align + QSA in one server), the house 35B bench
(59.77 t/s baseline) and the PPL probe (15.9361) for non-regression.

Cannot: 4×MI50 TP=4, 147 456 ctx, the boot-segfault class (never reproduced on
this box) — those stay the tester's evidence.

## Next actions

1. Cherry-pick the 6 commits above onto `gfx906/qsa-fn`, applying edits 1–3 and
   dropping the `.cu` reformat + the degradation rows.
2. Build, run the suites + tiny rig + the two house gates; record in
   `CHANGELOG.md` / `DEVLOG-qwen38-flash-qsa.md` as **QSA-FN-9** (tester result +
   merged fixes).
3. Ask the tester for the two test artifacts before the merge lands.

## 2026-09-23 — upstream overlap and the conflict accounting (changes the plan)

**Upstream's newest is `v0.30.1rc0` (2026-09-23); there is no rc1 yet.** Checked
file-by-file against it — **none of the PR, and none of our QSA train, is upstream**:

| artifact | in `v0.30.1rc0`? |
|---|---|
| fp16 QSA guards (QSA-FN-1, the reported error) | **no** — `supported_dtypes = [torch.bfloat16]`, "Qwen4Exp QSA requires BF16 Q/K/V" verbatim |
| tiled indexer (QSA-FN-4) | **no** — `_qsa_mqa_paged_tiled_kernel` absent |
| V2-MAMBA-1 seed fix | **no** — `// self.cache_config.block_size` verbatim (line 122) |
| SKINNY_M16 / C4 free-list flips | **no** — fork-only flags and module, no upstream counterpart |
| PLE mmap host-table offload | **no** — upstream's `ple_layer.py` still builds the TP-sharded on-device embedding; no mmap/CPU/pin path |
| V2 boot GC guard | **no** — the V2 runner has bare `gc.collect()` (2 sites) and no `gc.freeze` at all |
| drafter-graph knob | **no** — `init_cudagraph_manager(cudagraph_mode)` verbatim |
| amdsmi arch sanity fallback | **no** — upstream calls `_query_gcn_arch_from_amdsmi()` directly |
| the `enforce_eager` finding | **still true upstream** — the field exists in `SpeculativeConfig` but only the *target's* value is propagated (`config/speculative.py:1301`), so a `--speculative-config {"enforce_eager": …}` is inert |

**The one real overlap is a capability, not code.** Upstream (and our fork) already
ship a **generic per-parameter CPU offload** — `vllm/config/offload.py`,
`--cpu-offload-gb` / `--cpu-offload-params` — which is exactly what the CDNA recipe
used to keep `ngram_embedding` in host RAM. The tester's
`MmapShardedNGramEmbedding` is a *parallel implementation of the same idea* inside
`ple_layer.py`, with a bespoke host-weight GEMV path in `dense_gemv_gfx906.cu` +
`utils.py`. So "upstream must have PLE offloading" is half right: it has **generic**
offloading, not a PLE-specific one.

### Conflict accounting (merge-tree, `CONFLICT` lines, same method both sides)

| base | ours | ours + PR | delta |
|---|---|---|---|
| `v0.30.1rc0` | **44** | **46** | **+2**: `vllm/platforms/rocm.py`, `vllm/v1/worker/gpu/model_runner.py` |
| `releases/v0.30.0` | 32 | 33 | +1 |

Nothing stops conflicting. So cherry-picking the PR costs **~2 extra conflict files per
future merge train**, both in *hot* upstream files (`model_runner.py` already diverges
403+/421−, `rocm.py` 248+/71−). The PR's other touched files are **fork-only** and cost
nothing: `csrc/rocm/dense_gemv_gfx906.cu`, `c4_layer0_moe.py`,
`gfx906_fa/gfx906_fa_backend.py`. `ple_layer.py` does not conflict *today* (upstream
has not touched it since our base) but it is upstream's file.

**And the merge target matters more than the PR:** 44 conflicts against rc0 vs 32
against 0.30.0 — chasing the newest upstream costs +12 files for this branch.

### Revised plan

1. **Ask the tester for one comparison** before taking the PLE work: their mmap path
   vs `--cpu-offload-params ngram_embedding` at the same config (VRAM + prefill/decode
   t/s). If the generic mechanism is comparable, drop the bespoke class — that removes
   the `ple_layer.py`/`dense_gemv.cu`/`utils.py` complex, one of the two new conflict
   sites' neighbours, and the `rocm.py` hooks with it.
2. **Take now** (small / gated / model-local): the amdsmi arch fallback, the V2 GC
   guard (gated; report the segfault class upstream with their repro — it is an
   unidentified heap corruption), the drafter-graph knob (gated; our real-payload
   sweep says neutral), the PLE correctness fixes, the shellcheck/SPDX trivia.
3. **Don't take**: the whole-file clang-format of `dense_gemv_gfx906.cu`, the ops-runbook
   doc rows.
4. **Upstream the generic bits instead of carrying them** (`UP-4`): the amdsmi fallback
   and the `enforce_eager` propagation are small, generic and independently valuable —
   and our own logs show amdsmi breaking on this stack (it was the pre-wedge symptom in
   wedge #106). Landed upstream, they arrive through the next merge rather than becoming
   fork-local divergence.
5. **Pin the merge target now.** rc0 is +12 conflict files over 0.30.0; decide whether
   the 0.30.0 train ships first (cheaper, and the QSA work is on a branch anyway) or we
   wait for 0.30.1 final and absorb the larger set.

## 2026-09-23 (later) — upstream has an *official* PLE CPU offload; #57497 brings it to ROCm

`vllm-project/vllm#57497` — "[Qwen4Exp][ROCm] PLE n-gram table CPU offload" (author
mrodden, **OPEN**, base `main`, updated 2026-09-23) — is the official version of
what the tester hand-built. Checked state:

| piece | `v0.30.0` | `v0.30.1rc0` (released) | `upstream/main` + #57497 |
|---|---|---|---|
| `vllm/config/engram.py` (PLE config + gate) | present, gate `is_cuda()` | present, gate **`is_cuda_alike()`** | same, message wording only |
| `VLLM_PLE_CPU_OFFLOAD` | present, default **on** | present, default on | same |
| pinned-host PLE classes | — | **NVIDIA only** (`qwen4_exp/nvidia/ngram_embedding.py`, "device and pinned-host storage", `is_uva_available`) | moved to `qwen4_exp/common/ngram_embedding.py`, **AMD selects the pinned class when `engram_config.cpu_offload`** |
| AMD PLE path | device-resident | device-resident (`PLEVocabParallelEmbedding`) | pinned-host, behind a new custom op `qwen4_exp_amd_ple_ngram_embedding_pinned` (`mutates_args=["output"]`, so inductor's autotuning cannot materialise the table) |
| our 0.29-line tree | **none of it** — no `engram.py`, no `VLLM_PLE_CPU_OFFLOAD`, no pinned classes | | |

So the mechanism is: **pinned host RAM + UVA view, default on, ROCm accepted since
rc0 — but the ROCm *implementation* is exactly what #57497 adds** (today the AMD path
still builds a device-resident table, so the relaxed gate alone does not offload
anything on gfx906). Their own numbers: offload ON vs OFF is equal within noise
(TTFT/decode/burst), i.e. the value is VRAM, not speed — the same conclusion as the
tester's mmap path.

### What this changes for us

1. **The bespoke PLE path is superseded in direction, not yet in time.** Upstream's
   mechanism is not in any release for ROCm, and our tree has none of the
   infrastructure, so the tester's path (or the generic `--cpu-offload-params`) is the
   only thing that works *today* on the 0.29 base. Treat theirs as an **interim
   stopgap**: do not gate it, do not polish it, and delete it when we adopt upstream's.
2. **The PLE measurement stays useful but retargeted**: the meaningful comparison is
   theirs vs **upstream's pinned-host offload** — the generic `--cpu-offload-params`
   arm is only the closest stand-in that exists on our current base.
3. **Pinned RAM vs page cache is the real open question.** Upstream pins the table in
   host RAM (26 GiB here; 51–102 GiB for the FP8/bf16 checkpoints they target), while
   the tester's path reads page-cache-backed file pages and never pins. On a 128 GB box
   that difference decides big-table practicality, so the comparison is worth running
   even though upstream will own the mechanism.
4. **Their D2H/stale-id bug class transfers.** Upstream's pinned path also stages ids
   through a pinned buffer; the tester's finding (an async D2H leaves the host reading
   stale/uninitialised ids, which then index out of range) is a hazard worth reporting
   against #57497 — a concrete, valuable contribution rather than a fork-local patch.
5. **Merge sequencing is unchanged, but the merge now has an added benefit**: it brings
   `engram.py` + the pinned classes + the relaxed gate, i.e. a capability the 0.29 line
   lacks entirely. The AMD half still needs #57497 (or a small port of it) after that.


## 2026-09-24 — integration done; the guard needed a third revision (theirs)

The cherry-picks are on `gfx906/qsa-fn` (their six commits as six commits, their
authorship, plus two integration commits authored by them as well — see the dev
log for the `dummy_weights` and splitting-op/piecewise fixes, which their tip does
not carry). Two rounds of reviewer work happened after the first comment, and both
are worth knowing before this is merged:

1. **The range guard we took from `6c26a8edd6` was boot-breaking**, and it took
   them two corrections to land the right shape (`04e142cfe`, then `a1c70dedb`):
   the drafter's `-1` `sample_idx_mapping` padding sentinel, then a third producer
   — unwritten pinned host memory (`0xff80ff80ff80ff80`) arriving on the
   compile/graph path only. Adopted as `fe3bcc9e23` (fold onto row 0 + warn once +
   `VLLM_GFX906_PLE_STRICT=1` to raise). The general lesson is in the dev log, and
   it argues against accepting a `raise` on any hot path as free hardening.
2. **Their eager-only workaround is probably unnecessary on our config**, which is
   the one piece of leverage we gained from their report: the poison requires the
   lookup to be *inside* the captured graph, and our splitting-op + piecewise
   change moves it out. Our tiny rig (MTP k=1 and k=3) logs no out-of-range id at
   all. Stated to them as a hypothesis to test on the real model, not as a result.

**The PLE comparison they were asked for still has not arrived** (neither the
numbers nor a partial), so the mmap-vs-upstream decision is still open and still
QSA-FN-11's problem. Do not let the guard work be mistaken for that evidence: it
is a correctness fix to the interim path, not a reason to keep it.

**Cleanups are done on our side** (drafter default restored to upstream, comment
trim, `degradation.md` rows re-homed as generic findings) and the tester was told
no action is needed from them there.

**Open, ours:** the `PIECEWISE` downgrade for qwen4_exp is a judgement call (it
costs this model the full-graph decode path; the alternative is warn-only plus
`--enforce-eager`); and whether to file the stale/uninitialised-id hazard against
#57497 ourselves using their finding. Both are recorded above and await a decision.

## 2026-09-27 — settling the merge: the `.cu` hunk is out, the GC guard is upstream's job

Two open items from the merge report are closed, both against the cherry-pick rather
than against the merge resolution:

1. **The PLE commit's `csrc/rocm/dense_gemv_gfx906.cu` hunk is dropped** — it does not
   compile on this toolchain (`error: call to 'atomicAdd' is ambiguous` at the
   ksplit>1 epilogue, reproduced through the real ninja build), it is unrelated to the
   PLE path, and the base's comment documents why that call site uses the
   compiler-lowered fp16 atomicAdd. So the merge no longer needs a rebuild to be
   trustworthy on the GEMV rail: the file is the base's.
2. **The GC guard needs no consolidation** — upstream #54646 (`c28feab989`) added both
   `freeze_gc_for_cudagraph_capture` and the `if gc.isenabled():` skip in
   `breakable_cudagraph.py::_capture`, i.e. exactly the protection the tester's
   shadowing was written to provide on 0.29 (where neither exists). Keep their guard on
   the 0.29 line as the off-by-default diagnostic it already is; carry it into 0.30
   unchanged, and record #54646 as its supersession rather than re-implementing anything
   in `gc_utils.py`. Their 28-case harness tests the guard, not the upstream helper, so
   it stays with the guard.

Still open for the merge, unchanged: the 8 doc conflicts (a real union, not a side
pick), the house gates on a healthy boot (mamba kernels, FA, PPL, 35B), and the PLE
implementation comparison, which only the tester's box can answer.

### The doc conflicts, itemised (for whoever lands the merge)

The merge had 8 doc conflicts. `ROADMAP.md` is **done** (the 0.30 roadmap plus the
branch's P2P and whole Qwen3.8-Flash-Next/QSA sections — they had diverged into two
different documents: 0.30 carries the release-prep/upstream queue, the branch carries the
QSA-FN journey). The other seven still hold the 0.30 side, and each needs a genuine
editorial union rather than a side pick:

| file | what the branch adds |
|---|---|
| `CHANGELOG.md` | the QSA-FN entries (172 lines) — land them when the work lands |
| `MERGE-0.30.0-review.md` | the branch's copy of the 0.30 merge review (conflict accounting, off-by-default inventory); the 0.30 line's copy is the canonical one for the release, so port only the QSA-relevant rows |
| `degradation.md`, `degradation_details.md` | observations #108/#109 and the #108 addendum (small, additive rows) |
| `DEAD-ENDS.md` | the allreduce no-op (NEUTRAL) and int8-QK (CLOSED) rows |
| `README.md` | the release-snapshot / stale V1-pin recipe corrections |
| `AGENTS.md` | 11 lines the 0.30 line added (rule 6 is already in both) |

Mechanical recipe that works for additive cases: `git diff <merge-base> gfx906/qsa-fn -- <file>`
then `git apply --3way` in the merge tree (the 3-way apply reports conflicts for
overlapping rows — `--reject` cannot be combined with it).

