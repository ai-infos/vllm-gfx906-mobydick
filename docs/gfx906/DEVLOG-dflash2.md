# DFLASH2 — DFlash2 speculation on gfx906 (bring-up, gate, kernel coverage)

> Branch `gfx906/v2-bringup` → `main` · model Qwen3.8-27B-AWQ-INT4 (dense) +
> `incoai/Qwen3.8-27B-DFlash2` drafter · 2026-09-15 · roadmap `DFL2-1`, `DFL2-7`.

**VERDICT:** `OPEN` — the bring-up crash is fixed and shipped; the baseline gate (DFlash2 vs
MTP k=3) has not run yet.

**GATE:** serving A/B, agentic corpus, DFlash2 at its recommended `num_speculative_tokens=7`
against MTP k=3 (with and without the CAT-1 shortlist — CAT-1 is MTP-only, verified, so a
"DFlash2 + CAT-1" cell does not exist). Reference band on the 0.29 line: MTP k=3 + CAT-1 on V2
= **35.44 / 24.61 t/s** (64k/120k, 3-rep), plain MTP k=3 = 33.62 / 23.75. Note the metric: the
two methods use different draft depths, so t/s at each method's recommended config decides,
with ms/step and acceptance as the supporting detail.

---

## HYPOTHESIS

If the DFlash2 drafter is cheaper per step than the MTP head (5 layers, block drafting, vs a
draft head), then plain DFlash2 should beat MTP k=3 on gfx906 at equal-or-better acceptance —
independently of the n-gram-chain feature that `DFL2-1` targets.

## What was done (2026-09-15)

- Drafter `incoai/Qwen3.8-27B-DFlash2` obtained (3.6 GB, one safetensors): 5 layers, hidden
  5120, 32 heads / 8 KV, head_dim 128, `is_causal: false`, `sliding_window` 2048,
  `dflash_config` `{block_size 8, selector_rank 256, mask_token_id 248070}`. `DFlash2DraftModel`
  is **already registered** in our tree, and `_is_dflash2_draft()` forces V2 (validated).
  `run_server.sh` gained a `dflash2` arm (port 8137, k=7, V2-only).
- First gfx906 bring-up **crashed twice at model warmup** — software, **0 kernel resets**, so
  not the wedge lottery: `AssertionError: Incompatible dimensions` in
  `vllm/model_executor/layers/utils.py::triton_matmul`, our gfx906 ROCm-GEMM fallback, reached
  from an inductor graph via `torch.ops.vllm.rocm_unquantized_gemm`.
- Root cause, after adding the offending shapes to the assert message:
  **`a=(2, 7, 5120)` is 3-D** (the drafter's per-block tokens) while `triton_matmul` is 2-D
  only, and the weight arrives as **`[K, N]`** (`b=(256, 5120)`, the `selector_rank` projection)
  where this kernel wants `[N, K]` — the fork inverted that convention
  (`# NOTE(gfx906): b.shape inv`).
- **Fix (shipped):** `triton_matmul` flattens leading dims and restores them on the way out,
  tolerates a `[K, N]` weight (transposes it, one-shot warning), and still fails loud on
  genuinely incompatible shapes — with the shapes now printed in the message.

## Evidence — FOR

- (launch-regime, GPU0, at the exact failing shapes) 3-D with `[N, K]` weight: out
  `(2, 7, 256)`, rel err **2.6e-4** vs `F.linear`; 3-D with `[K, N]` weight: matches `a @ b`
  (2.6e-4, warning fired once); 2-D regression check unchanged (2.0e-4).
- Engine log shows `Resolved architecture: DFlash2DraftModel`, `num_speculative_tokens': 7`,
  and the target's `Using CUSTOM (gfx906 FA) backend for ViT attention` (VIT-1 active).

## Evidence — AGAINST / still open

- The two crashed attempts consumed the first session's load budget, so **Q1 is unmeasured**:
  no DFlash2 t/s or acceptance exists yet on gfx906.
- `DFL2-1`'s chain patch cannot be exercised yet: it hard-depends on
  `dflash2-lookup-drafting` (`DFL2-3`), which our tree lacks (`from
  vllm.v1.worker.gpu.spec_decode.dflash2.lookup import …`; the patch itself warns
  `VLLM_DFLASH2_CHAIN=1 needs VLLM_DFLASH2_LOOKUP=1; disabling`). Both patches also target a
  ~480-line speculator where ours is 217 (upstream refactored), so they are adaptations rather
  than `git apply`.

## Why the crash existed (mechanism)

Our gfx906 GEMM dispatch (`rocm_unquantized_gemm`) has fast paths for exactly the shapes its
authors tuned: `n == 1` (LLMM1 / long-k GEMV) and `n == 2..4` (spec-GEMV-M4). DFlash2's draft
projections are a **new shape regime** — `n = 2 × 7 = 14` block tokens with
`m ∈ {256, 6144, 17408}`, `k = 5120` — so they fall through to the generic `triton_matmul`,
which had only ever been reached with 2-D activations and the fork's `[N, K]` weight
convention. Neither the rank nor the layout had a guard, so the mismatch surfaced as an
indictment of our own code, not of DFlash2.

## Optimization target found here (Kevin's ask)

With the crash fixed, those projections now *run*, but on the **generic fp16 `triton_matmul`**
(`waves_per_eu=1`, no max-ilp tuning) — and any `[K, N]` weight pays a per-call
`t().contiguous()` copy on top. That is several projections × 5 layers **per draft step**, i.e.
on the critical path of every DFlash2 step. Candidate work is queued as ROADMAP **DFL2-7**:
extend the gfx906 GEMV/skinny coverage to `n ≤ ~16` for the drafter's `(m, k)` pairs, and/or
add a stride-aware variant so no transpose copy is needed. **Profile before optimizing**
(`DFL2-1` subtask 1, rocprofv3) — standalone numbers on this box have repeatedly failed to
transfer.

## 2026-09-15 (late) — arm C runs: DFlash2 is *far* behind MTP k=3, and the drafter's attention is the suspect

**Arm C (DFlash2, no patch, k=7, agentic, V2, same boot as the fix) — complete, 4 rows:**

| context | t/s (2 reps) | acceptance | ms/step |
|---|---|---|---|
| 64k | 2.48 / 2.52 | 0.0451 / 0.0625 | 421.9 / 422.0 |
| 120k | 1.37 / 1.37 | **0.0 / 0.0** | 728.4 / 728.2 |

**Arm B (MTP k=3 + CAT-1, live control, same boot) — 4 rows:** 64k 35.97 / 35.44 t/s
(ms/step 77.1 / 83.7), 120k 24.58 / 24.82 (123.5 / 123.7) → **within +0.8 % / +0.4 % of the
recorded V2 band (35.44 / 24.61)**, so the historical anchor this arm was measured against is
validated on this boot, and arm C's comparison is sound.

So **plain DFlash2 is ~14x slower than MTP k=3 with acceptance collapsing to zero by 120k** — Q1
answered: no benefit, and the drafter's drafts are useless as well as slow (a *wrong* draft
context, see below). No downstream patch closes that.

**Why it is that slow — the drafter's attention is not on our FA.** The engine log for this arm:

- target (dense 27B, 20:40:58): `Found incompatible backend(s) [TURBOQUANT] with
  AttentionType.DECODER. Overriding with **CUSTOM**` — the target *does* get our FA, so the arm
  stays comparable with the MTP arms;
- drafter (3.58 GiB checkpoint, 20:41:25): `Found incompatible backend(s) [**CUSTOM**,
  TURBOQUANT] … Overriding with **ROCM_ATTN**` — the drafter's attention runs upstream ROCM_ATTN,
  whose paged kernel then reports `Cannot use ROCm custom paged attention kernel, falling back to
  Triton implementation` (1 occurrence, i.e. the drafter only). ~422 ms/step over 5 sliding
  layers is consistent with a Triton fallback attending over far more than the 2048-token window,
  and a wrong context would also explain the 0.045 acceptance.

**Why CUSTOM is rejected for the drafter — reasons are not logged.** The base validity checks
(`v1/attention/backend.py`) are head_size, dtype, kv_cache_dtype, block_size, mm_prefix, `use_mla`
vs `is_mla()`, sinks, `use_sparse` vs `is_sparse()`, per-head quant scales, compute capability and
attn_type. The drafter's config passes the ones we can verify by hand (head_dim 128 supported,
fp16, DECODER, and — checked — `supports_sliding_window()` is **True** in our backend), so the
culprit is one of the remaining ones, with **sparse** the leading hypothesis given this stack's
`sparse_attn_indexer` machinery. But the platform logs only the *names* of rejected backends, so
the first step is to make it log the *reasons* — see ROADMAP **DFL2-8**.

**Consequence for the family's priority:** the chain patch (DFL2-1) is not the path to viability —
a +7 % copy-cell feature cannot close a 13x gap. **DFL2-8 (the drafter's attention backend) is now
the item that decides whether DFlash2 is viable here at all**, and it is a correctness question as
much as a speed one (0.045 acceptance says the draft context itself is wrong). `DFL2-3`
(lookup-drafting) and the rest stay queued behind it.

**VERDICT:** `OPEN` — the baseline gate is measured and **negative** (2.48/1.37 t/s, acceptance
0.045 -> 0.0, vs MTP k=3's 35.97/24.58); the drafter's attention backend (`DFL2-8`) is the
blocking unknown and the only plausible path to viability.

## 2026-09-15 (late) — is this our stack, the drafter's attention, or the pairing? (Kevin's challenge)

Kevin's question is the right one: *can an attention-path problem really produce acceptance ~0, or
is the drafter model itself wrong?* The evidence, cheapest first:

- **Our target-side FA is excluded as the cause.** The same build/boot runs MTP k=3 + CAT-1 at
  35.97/24.58 t/s with acceptance 1.78-2.11; a corrupted target attention would sink that arm too.
  So this is *not* our FA on the decoder path.
- **Per-position acceptance says the drafter's *first* token is wrong**, not that the trajectories
  drift: pos0 = 0.041 / 0.0625 at 64k and **0.0 / 0.0 at 120k**, pos1+ = 0 everywhere (MTP's pos0 is
  ~0.85). A draft whose first token is essentially never right is mis-conditioned or mis-mapped —
  not merely badly tuned.
- **The drafter is a block-diffusion model fed by the target's hidden states.** Its config says
  `dflash_config.target_layer_ids: [5, 19, 33, 47, 61]`, `mask_token_id: 248070`, `block_size: 8`,
  `selector_rank: 256`, `selector_top_k: 16`; the card calls it a "block-diffusion drafter" that is
  "not a standalone language model" and pairs it with **`Qwen/Qwen3.8-27B` (bf16)**. Ours runs
  against an **AWQ-INT4** target.
- **Its attention is windowed-bidirectional** (all 5 layers `sliding_attention`,
  `sliding_window 2048`, `is_causal: false`) and our CUSTOM FA is *rejected* for it, so it runs
  upstream ROCM_ATTN whose paged kernel falls back to Triton.

Three candidate causes, in the order a cheap test can decide them:

1. **The drafter's attention fallback masks/conditions wrongly** (e.g. causal masking applied to a
   non-causal block drafter, and/or the 2048 window not applied — the latter would also explain the
   O(context) step cost, 422 ms at 64k vs 728 ms at 120k). Decidable by profiling the drafter's
   kernels (`DFL2-8`/subtask 1, rocprofv3) and by logging *why* CUSTOM was rejected (step 1 of
   DFL2-8: the reasons are already computed, only the names are logged).
2. **The block-diffusion machinery is incomplete/mismatched in our tree** (mask tokens, selector,
   the z/lookup schedule). The RTX3090 fork needed a dedicated DFlash2 backport, so this is a live
   possibility; decidable by running the card's pairing on a non-fork stack.
3. **Pairing**: the drafter consumes the *target's* hidden states at layers 5/19/33/47/61, and it was
   published for the bf16 `Qwen/Qwen3.8-27B`; an AWQ-INT4 target's activations differ (weight-only
   quantisation noise). Normally that costs some acceptance, not all of it — but it is cheap to rule
   out on a non-fork box by running bf16 vs AWQ.

**Cheap decisive test (Kevin can run it off-fork):** the card's own recipe — `Qwen/Qwen3.8-27B` +
`incoai/Qwen3.8-27B-DFlash2`, `--speculative-config {"method":"dflash","model":…,
"num_speculative_tokens":7}` — and report per-position acceptance and ms/step. If acceptance is
normal (~2-3) there, the checkpoint is fine and the fault is ours (pairing or the drafter's attention
backend); if it is degenerate there too, the checkpoint/mechanism is the problem. Running the same on
an AWQ target isolates candidate 3.

**VERDICT:** `OPEN` — the negative baseline stands; the cause is narrowed to (drafter attention
backend | in-tree block-diffusion support | target-quantisation pairing), with the target-side FA
excluded, and the off-fork run plus the reason-logging/profile pair are the next two cheap steps.

## 2026-09-16 — the club-3090 INT8 recipe narrows it to the *pairing* (and retracts "incomplete support")

Kevin supplied the external recipe for this exact model family: `lued/Qwen3.8-27B-INT8-W8A16-DFlash2`
(29.6 GB, `Qwen3_5ForConditionalGeneration`) + `lued/Qwen3.8-27B-DFlash2-W8` (2.2 GB,
`DFlash2DraftModel`), served with two vendored vLLM patches. Checking each against our tree:

- **vLLM#52816 ("./dflash2: local convolution + candidate selector") is ALREADY IN our tree** —
  `74a6576b9b`, merged upstream 2026-08-21, i.e. before the 0.29 tag. The recipe carries it as a
  patch only because its pinned nightly predates that merge. So "our DFlash2 support is incomplete"
  is **not** the explanation.
- **vLLM#48375 ("honor `drop_eagle_block` in MambaManager") is excluded too**: the patch's own note
  says it is *inert under the shipped prefix-off default* — the patched path only runs with prefix
  caching enabled, and **all our DFlash2 arms ran `--no-enable-prefix-caching`**.
- **club-3090's `vllm-gdn-mtp-async-spec-order`** (GDN + MTP + prefix caching + async scheduling)
  is a *cross-stream race that manifests as a CUDA illegal memory access*, not as low acceptance,
  and needs prefix caching on. Not our case.
- **The target-side FA stays excluded** by the MTP control (same build/boot: 35.97/24.58 t/s,
  acceptance 1.78-2.11).

**Leading cause: the drafter/target quantisation pairing.** The ecosystem ships drafters *matched to
the target's quantisation* — `qwen…-DFlash2` for bf16, `…-DFlash2-W8` for W8A16 — and there is no
AWQ variant. We paired the **bf16** drafter with an **AWQ-INT4** target, while the drafter consumes
the target's hidden states at layers `[5, 19, 33, 47, 61]`; `draft_vocab_size: None` (full 248 320
vocab) rules out a draft-vocab mapping mismatch. The remaining candidate is the drafter's *attention*
fallback (`DFL2-8`), which certainly explains the 422 -> 728 ms per-step growth but is no longer the
front-runner for the acceptance collapse.

**Plan (the INT8 route, which also fits 2x MI50):** pull the matched pair to NFS (started
2026-09-16 05:19, `HF_HUB_CACHE=/data/cache/huggingface/hub` — 29.6 GB does not fit `/local`'s
~27 GB), then serve it with our stack's adaptations: `--tensor-parallel-size 2`,
**`--kv-cache-dtype fp16`** (the recipe's `fp8_e4m3` does not exist on gfx906), V2 (forced by
dflash2), and a `--max-model-len` sized to the KV budget a ~30 GB model leaves. Gate: **per-position
acceptance** (the discriminating number — ours was 0.041/0.063 at 64k and 0.0 at 120k against MTP's
~0.85 at pos 0) plus ms/step. Pull rate is ~80 MB/min unauthenticated (~6 h for 29.6 GB); an
`HF_TOKEN` would cut that by an order of magnitude.

**VERDICT:** `OPEN` — causes excluded: target FA, PR52816 (already present), #48375 (inert
prefix-off), the GDN async-spec race (crash, prefix-on). Leading: drafter/target quantisation
pairing; secondary: the drafter's attention backend. The matched INT8 pair is the decisive test.

## 2026-09-16 — the INT8 route is blocked by OUR `pack-quantized` W8A16 gap, not by DFlash2

**VERDICT:** `lued/Qwen3.8-27B-INT8-W8A16-DFlash2` cannot be served on this fork today.
It is a compressed-tensors **`pack-quantized`** checkpoint (`weight_packed` /
`weight_scale` / `weight_shape`; **401 packed tensors**, including the embedding and the
GDN `linear_attn.in_proj_qkv` / `in_proj_z`), and the loader died in `Qwen3_5Model` with
`ValueError: There is no module or parameter named 'embed_tokens.weight_packed'`.

**Correction (2026-09-16, INT8-PACKED-1):** the cause was **not** missing quantisation
support. Our tree already routes these groups to `CompressedTensorsWNA16(strategy=group,
num_bits=8, group_size=128)` and already ships a pack-quantized embedding scheme
(`compressed_tensors_embedding.py`, a Triton gather that unpacks int32-packed INT weights and
dequantises in one pass). `Qwen3_5Model` simply built `embed_tokens` without
`quant_config`/`prefix`, so the quantized embedding never registered packed parameters. Fixed
in two lines on `gfx906/int8-packed`; the checkpoint then loads with zero skipped tensors and
generates coherent output (`DEVLOG-int8-packed.md`). It is also the **VL** variant
(`Qwen3_5ForConditionalGeneration`, `model.visual.*` keys).

The checkpoint itself is fine: 6 shards totalling 29.53 GB, exactly the index's
`total_size`. The earlier 109 % directory size was **duplicate `.incomplete` partials**
from two concurrent downloaders (4.1 GB, removed).

So the pairing question stays open *on our box* until packed int8/W8A16 support lands
(ROADMAP `INT8-PACKED-1`). Nothing here bears on the DFlash2 mechanism itself — the
exclusions recorded above still stand, and the cheapest decisive test remains the
*cards' own matched pair on a stock vLLM* (their nightly has both DFlash2 and
pack-quantized).

## Refrigerated residue

`rocm_unquantized_gemm`'s 3-D branches still pass `x` (not the flattened view) to
`triton_matmul` in one path — harmless now that the helper flattens and restores, but the
call sites would be clearer if they passed `x_view` explicitly.

## Search keys

`HYPOTHESIS:` DFlash2 cheaper than MTP k=3; `VERDICT:` OPEN; `GATE:` DFlash2-vs-MTP serving A/B;
`triton_matmul` 3-D / `[K, N]`; `DFL2-7` GEMV coverage.
