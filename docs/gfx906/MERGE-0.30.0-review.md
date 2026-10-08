# 0.30.0 merge review — which conflicts are our live code, and which are dead weight

> Analysis for decisions (no code changed). Branch `gfx906/v0.29.0` →
> `upstream/releases/v0.30.0` (both fetched 2026-09-17). Related:
> [`ROADMAP.md`](ROADMAP.md) `UP-3` (the merge train), [`DEAD-ENDS.md`](DEAD-ENDS.md)
> (the one-pass dead index), `REFRIGERATOR.md`.
>
> **Correction to an earlier count:** this merge is **31 conflicted files**, not
> 148. The 148 came from `git merge-tree --write-tree --name-only | wc -l`, which
> prints `Auto-merging <path>` progress lines **on stdout** alongside the
> conflicted names (85 of the 149 lines). Trust the `CONFLICT` lines
> (`git merge-tree … 2>&1 | grep -c '^CONFLICT'` → 30) or filter
> `^Auto-merging`; every one of the 31 has both sides changed since the merge base
> (or is an add/add), while 116 of the "extra" paths exist on neither side.

**Second correction (this one matters):** my first pass put four
off-by-default switches in a "verify then delete" bucket as if they were parked
experiments. Three of them are the opposite — **gated wins whose gate already
passed and whose only open item is a default flip**: `SKINNY_M16` (measured
+14.5 % / +6.1 %), `QUANT_LAYER0_MOE` (measured +3.0 %), and the FD-1/A3 flag
(a revived mechanism with three tests and an in-tree consumer). Table 2 now
carries the record per switch. The lesson: *off by default ≠ dead* — on this box
a default flip is a deliberate, documented "Kevin's call" step, so grep the dev
logs and the roadmap for the flag before proposing a deletion.

## Method (re-runnable)

```bash
MB=$(git merge-base gfx906/v0.29.0 upstream/releases/v0.30.0)   # f5c3cc240b
git merge-tree --write-tree gfx906/v0.29.0 upstream/releases/v0.30.0 2>&1 \
  | grep '^CONFLICT' | sed 's/.*Merge conflict in //' | sort -u > conflicts.txt
# per file: our-side change / their-side change / our substantive commits
while read -r f; do
  echo "== $f"
  git diff --numstat $MB gfx906/v0.29.0 -- "$f"
  git diff --numstat $MB upstream/releases/v0.30.0 -- "$f"
  git log --no-merges --format='%h %s' gfx906/v0.29.0 --not upstream/releases/v0.30.0 -- "$f"
done < conflicts.txt
```

## Table 1 — the 31 conflicts, by class

Classes: **L** = our live gfx906 code (default-on or model-support; resolve by
hand) · **G** = our code that is **gated off by default** (archive candidate) ·
**C** = an upstream commit our fork carries (cherry-pick) or a fix upstream also
has (take theirs, drop our copy) · **D** = bindings/CI/test glue (mechanical).

| file | ours | theirs | class | our-side change (fork-only commits) → recommendation |
|---|---|---|---|---|
| `csrc/libtorch_stable/quantization/gptq/q_gemm.cu` | 178+/64− | 112+/423− | L | q_gemm M=1 max-ILP split + the 0.29-merge fixes; **kill-switch `VLLM_GFX906_QGEMM_M1_MAXILP` is default-ON** (AGENTS: build default). KEEP — highest-value hand-merge |
| `vllm/_custom_ops.py` | 241+/2− | 483+/155− | L | MoE M1/NPT default-on, dead S2-topk removal, Minimax-M3 bindings. KEEP |
| `vllm/model_executor/layers/utils.py` | 494+/60− | 50+/27− | L+G | fp32 router-gate GEMV (NH-3), `DENSE_GEMV`/`DOWN_GEMV` default-on, triton_matmul >2-D — **plus `SKINNY_M16` (default off)**. KEEP; split out the M16 hunk (see Table 2) |
| `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` | 513+/42− | 123+/152− | L | Nemotron-3.5 INT4/INT8 wna16 support + code-review fixes. KEEP |
| `tests/quantization/test_moe_wna16.py` | 460+/0− | 223+/25− | L | the wna16 tests for the above. KEEP |
| `vllm/model_executor/layers/mamba/mamba_mixer2.py` | 23+/0− | 155+/46− | **G** | **NH-4 fused grouped gated-norm, `VLLM_GFX906_MAMBA_FUSED_GROUP_NORM` default OFF** — gate **ran, neutral** (+0.4 %, Table 2). Optional strip; the merge cost either way is these 23 lines |
| `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | 76+/21− | 58+/17− | L | dense GEMV switches (default-on). KEEP |
| `vllm/models/minimax_m3/amd/model.py` | 128+/18− | 181+/14− | L | Minimax-M3-AWQ-INT4 gfx906 support (the train we are ON). KEEP |
| `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` | 292+/79− | 373+/102− | L | Minimax-M3 sparse/indexer fixes (uses upstream's `VLLM_ROCM_MLA_SPARSE_*`). KEEP |
| `vllm/v1/sample/ops/topk_topp_sampler.py` | 106+/0− | 24+/13− | L | SYV-4 port (`VLLM_GFX906_SORT_FREE_SMALL_K` default ON). KEEP |
| `vllm/model_executor/layers/quantization/compressed_tensors/.../compressed_tensors_moe_wna16.py` | 20+/2− | 14+/129− | L | asymmetric CT W4A16 MoE + wna16 review fixes. KEEP |
| `vllm/model_executor/kernels/linear/mixed_precision/exllama.py` | 59+/32− | 3+/31− | L | Minimax-M3-AWQ exllama path. KEEP (model-scoped) |
| `csrc/libtorch_stable/torch_bindings.cpp` | 13+/0− | 121+/22− | D | Minimax-M3/DSv4 kernel bindings. KEEP (additive) |
| `tests/test_config.py` | 58+/2− | 1120+/10− | L | DFlash2 spec-decode config + `VLLM_GFX906_SPEC_CG_SMALL` (default ON). KEEP, small |
| `vllm/config/attention.py` | 13+/1− | 27+/25− | D | Minimax-M3 attention config (rocm-gated). KEEP, small |
| `vllm/distributed/device_communicators/cuda_communicator.py` | 5+/4− | 126+/5− | D | force-disable custom all-reduce on this topology. KEEP, small |
| `vllm/model_executor/layers/quantization/auto_gptq.py` | 13+/2− | 7+/92− | D | rocm guard + `VLLM_ROCM_USE_SKINNY_GEMM` plumbing. KEEP, small |
| `vllm/models/kimi_k3/amd/kda.py` | 4+/2− | 44+/23− | L | kimi-k3 non-spec decode peel (model not runnable here). KEEP, small |
| `vllm/model_executor/models/deepseek_mtp.py` | 20+/6− | 16+/2− | D | MTP plumbing the Minimax train needed. KEEP, small |
| `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` | 6+/0− | 39+/0− | C | upstream "avoid sync in TRT-LLM ragged prefill" — 0.30.0 has its own version. TAKE THEIRS |
| `vllm/multimodal/utils.py` | 9+/4− | 11+/7− | C | upstream SHM prefix-covered-items fix. TAKE THEIRS |
| `vllm/v1/core/kv_cache_utils.py` | 0+/10− | 434+/38− | C | upstream "remove misleading mamba prefix-cache warning". TAKE THEIRS |
| `tests/v1/core/test_kv_cache_utils.py` | 0+/15− | 858+/33− | C | the test half of that fix. TAKE THEIRS |
| `vllm/v1/spec_decode/dflash.py` | 0+/15− | 11+/22− | C | upstream DFlash RoPE-layout fix. TAKE THEIRS |
| `vllm/_aiter_ops.py` | 2+/2− | 563+/187− | D | rocm guard/registration line. KEEP, small |
| `vllm/models/minimax_m3/amd/ops/index_topk.py` | 30+/0− | 1006+/306− | C? | our "port gfx906 minimax indexer fixes into 0.25" is a **backport**; 0.30.0 has the full implementation. TAKE THEIRS after verifying the sparse path still works |
| `tests/kernels/moe/test_zen_cpu_int8_moe.py` | 253+/0− | 251+/0− | C | add/add of the same upstream Zen test we cherry-picked. TAKE THEIRS |
| `vllm/model_executor/layers/fused_moe/experts/cpu_moe.py` | 162+/0− | 394+/341− | C | upstream "[CPU][Zen] route int8 MoE through zentorch". TAKE THEIRS |
| `.buildkite/hardware_tests/cpu.yaml` | 3+/0− | 80+/23− | C | the CI hook of that cherry-pick. TAKE THEIRS |
| `.buildkite/test_areas/misc.yaml` | 7+/1− | 189+/24− | C | CI hook of an upstream PP-send fix. TAKE THEIRS |
| `tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP2-PCP2-EP.yaml` | 1+/1− | 3+/1− | C | upstream CI-config tweak. TAKE THEIRS |

**Reading of Table 1:** only **one** of the 31 is our code that is *off by
default* (`mamba_mixer2.py`, NH-4). ~11 are our live feature code and need a real
hand-merge; ~10 are upstream commits we carry, where 0.30.0 already has its own
(or better) version and the right move is to **drop our copy** rather than
re-apply it; the rest is additive glue.

## Table 2 — off-by-default switches: **check the record before deleting**

Lesson from the first pass of this doc: *off by default ≠ dead*. Three of the six
switches I first listed as deletion candidates are **gated wins whose gate already
passed and whose only open item is a default flip** (Kevin's call). The record, per
switch:

| flag | status in our own docs | recommendation |
|---|---|---|
| `VLLM_GFX906_SKINNY_M16` | **SHIPPED, not parked** — `DEVLOG-fp16-skinny.md`: 35B MoE N=8 graph **+14.5 %** (191.0 vs 166.9 t/s), 27B (Qwen3.8) N=8 **+6.1 %**, 27B N=4 control flat (−0.6 %, flag inert); kernel correctness + per-shape 2–7.5× PASS; *"the A/B arms are positive on both models and the flag-on soak passed (30 reps × 2 models, flat), so `VLLM_GFX906_SKINNY_M16=1` is cleared to go default-on; flipping it is Kevin's call"* | **FLIP, do not delete.** It covers the M=5..16 spec-verify / 5–16-seq concurrent-decode regime; today we leave the win unclaimed. Caveat on record: the ksplit>1 epilogue is fp16 `atomicAdd` (same property as the shipped M≤4 rail) | 
| `VLLM_GFX906_QUANT_LAYER0_MOE` | **GO, not parked** — `ROADMAP.md` C4 + `DEVLOG-c4-layer0-quant.md`: unquantized layer-0 experts cost ~740 µs/call vs 182 µs for the W4A16 rail ⇒ ~558 µs/step; gates all passed — unit 8/8, PPL 15.9531 → **15.9929** (Δ +0.04, gate < 0.5), greedy fingerprint bit-identical (`d2e5262183c6b92f`), serving A/B off 84.95 → **87.51 t/s = +3.0 %** (noise floor ~1.8 %); ~1.5 GiB returned to graph capture. Open item: *"default-on decision after soak"* | **FLIP (soak is a process call, the measurements are done), do not delete** — it is a quality trade-off the checkpoint author did not make, so it is Kevin's call, but the numbers are in |
| `VLLM_GFX906_FUSED_DRAFT` | **live, not dead** — `tests/kernels/attention/test_gfx906_fa.py` "A3 (revived 2026-09-14 on the V2 bring-up branch)": the flag is the opt-in for the fused multi-step draft *metadata* protocol, whose consumer **is in-tree** (`v1/worker/gpu/spec_decode/autoregressive/speculator.py:_generate_fused_drafts`), pinned by **three tests** (flag, view contract, and an end-to-end reuse/corruption guard). `DEAD-ENDS.md`'s "has no reader in-tree now" is **stale** — the reader is the gfx906 FA builder, the attribute consumer is upstream V2 | **KEEP; fix the stale DEAD-ENDS row.** FD-1's *neutral* verdict was about the B=4/120k offline arm (stack-confounded), not about deleting the mechanism |
| `VLLM_GFX906_MAMBA_FUSED_GROUP_NORM` | NH-4: gate **ran** 2026-08-30 (A–B–A, fresh boot/arm, TP=2+EP): A 109.8 / B 110.05 / A2 109.37 t/s = +0.4 % inside noise (the A-vs-A2 drift exceeds the effect), PPL 24.9034 vs 24.8944, 0 top-20 misses; isolated 68→55 µs/layer ≈ 0.29 ms/step, hidden by a MoE-GEMV-bound step. The dev log labels it SHIPPED with the flip documented as a one-liner for a config that stops being GEMV-bound; the in-code comment still says "pending the A/B" (stale) | strip is **behaviorally safe** (default-off, measured-neutral in the served config) and saves 23 conflict lines; keeping it costs the same 23 lines at merge. Genuinely optional — my default is keep, fix the comment |
| `VLLM_GFX906_FA_STRICT` | a fail-closed *policy* switch (raise instead of warn when FA degrades quietly) | keep — not dead code |
| `VLLM_GFX906_MOE_BM`/`_M1`/`_NPT`, `GEMV_RPT`, `GEMVM_RPT`, `GEMV_I8_RPT`, `W8A16_INT8*`, `QGEMM_M1_MAXILP` | null-means-on (default **on**) | keep |

Switches already default-on (`ALIGN_M1`, `DENSE_GEMV`, `DOWN_GEMV`,
`SORT_FREE_SMALL_K`, `SPEC_CG_SMALL`, `SPEC_GEMM`, `TOPK_SINGLE_GROUP`) are live.

## Proposed plan (revised after checking the records)

1. **Flip the two gated wins** (one-line default change each, no new code):
   `VLLM_GFX906_SKINNY_M16` (measured +14.5 % / +6.1 % at N=8, soak passed) and
   `VLLM_GFX906_QUANT_LAYER0_MOE` (measured +3.0 % serving). Both flips should be
   followed by the house gates (35B `_bench_gfx906.py`, PPL probe, and the
   `tests/kernels/moe/test_c4_layer0_quant.py` / FA suites) — the measurements
   exist, the *defaults* are what is missing.
2. **Fix the stale records** rather than delete code: `DEAD-ENDS.md`'s FD-1 "no
   reader" line (the A3 opt-in was revived with three tests) and the NH-4 comment
   in `mamba_mixer2.py` ("pending the serving A/B gate" — the gate ran and was
   neutral).
3. **Optional hygiene:** strip NH-4 only (default-off, measured-neutral, 23 lines)
   and preserve it on an `archive/gfx906-dead-2` branch — do it as a separate,
   revertible commit if the tree should be lean. I would keep it: the merge costs
   the same 23 lines either way, and the dev log kept the flip deliberately.
4. **Do not** delete `SKINNY_M16`, `QUANT_LAYER0_MOE` or the FD-1 flag; all three
   have measured or test-guarded reasons to exist (Table 2).
5. **For the merge itself**, the work is: hand-merge the ~11 live-code conflicts,
   take 0.30.0's version for the ~10 upstream carries (re-checking only the three a
   served model needed: Zen CPU path, mamba prefix-cache warning, DFlash RoPE
   layout), and resolve the glue. Nothing in the conflict set is worth deleting to
   make the merge smaller — the largest single our-side conflict is 494 lines
   (`layers/utils.py`, live code), and the off-by-default items are 23 lines or
   live-elsewhere.

## Open questions for Kevin

- Flip `SKINNY_M16` and `QUANT_LAYER0_MOE` to default-on now (with the two gates
  re-run on the same boot), or keep them opt-in? Both logs say "Kevin's call".
- NH-4: keep the 23 lines and fix the comment, or strip-and-archive for a lean tree?
- The ~10 upstream carries: retire them wholesale on the merge (take 0.30.0's
  versions) or re-apply per model?

## Outcome (2026-09-24) — the merge is done

`gfx906/v0.30.0` was cut from `main` (`524ac6f2d6`, = `gfx906/v0.29.0`) and
upstream `v0.30.0` (`ced6857afa`) merged in: merge commit **`8893a50e54`**, 2017
files, **31 conflicted files** — the count this sheet scoped. A condensed
per-resolution record is in the merge commit body; the theme log entry is
`CHANGELOG.md` 2026-09-24.

What the merge confirmed, versus this sheet's predictions:

- **The conflict classes held** (L/G/C/D): the ~11 live-code files needed hand
  work, the ~10 upstream carries went to 0.30.0's version, and the rest was
  glue. Nothing was deleted to shrink the merge.
- **The one change that forced the shape was not on this sheet: #54809 (GPTQ
  group/dynamic activation ordering removed).** It rewrites `q_gemm.cu`, the
  marlin/marlin-moe-wna16 templates, CPU WNA16, `permute_cols` and the RDNA3
  kernels at once. Preserving the fork's `b_g_idx` API would have forked the
  whole quant stack, so the fork adopted the removal; the max-ilp M=1 dispatch
  and its SYNC-COPY twin were re-ported onto the new signature. This is the one
  **functional narrowing** of the merge: gfx906 GPTQ act-order checkpoints are
  no longer supported.
- **A merge can leave the tree internally inconsistent without any conflict
  marker.** The three post-merge findings (`_process_weights_gfx906`'s 14-tuple,
  `auto_awq`'s old `gptq_gemm`/`gptq_shuffle` arity, the missing
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` env registration) were all in files
  that merged *cleanly* because upstream changed one side of an API and the
  fork's caller was in a third file. Import-level and pytest validation caught
  them; a marker-only audit would not have. Worth adding to the merge-train
  checklist: after resolving, import the touched modules and run the feature's
  unit tests, not just `grep '<<<<<<<'`.
- **Upstream tests written against `SimpleNamespace` configs break on fork-only
  helpers.** `use_v2_model_runner`'s feature cases now `getattr`-guard so
  upstream's `test_models_default_to_v2_model_runner` and the DSA test pass
  unmodified.

Open follow-ups (also in `ROADMAP.md` UP-3): a model-level pass on the gfx906
GPTQ/AWQ paths after #54809, and one on the `rocm_aiter_mla_sparse` gfx906
fp16-sparse early-return + MiniMax-M3 `amd/ops` routing.
