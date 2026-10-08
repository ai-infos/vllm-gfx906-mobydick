# RECON — T-1 (int8 W8A16 "mass") vs syv-ai upstream SYV-3

**Date:** 2026-09-05
**Upstream source:** https://github.com/syv-ai/qwen38-27b-rtx3090 (cloned at `/local/tmp/syv-recon`, HEAD `0e951951`)
**Our implementation:** `vllm/model_executor/layers/t1_int8_w8a16.py` + `csrc/rocm/dense_gemv_gfx906.cu` (branch `gfx906/t1-int8-fp16-mass`)
**Purpose:** deep comparison of our T-1 port against the upstream implementation and notes, to find missing parts, bugs, and inefficiencies. Companion to `RECON-syv-qwen38-27b-rtx3090.md` (perf/strategy recon) and the CAT-1 comparison done for draft-vocab.

## 1. What syv actually does (source-verified)

syv's "int8 drafter + int8 lm_head" (SYV-3 in our ROADMAP) is **NOT a runtime transform and NOT custom CUDA**. It is:

1. **Offline checkpoint surgery** (`prepare/quant_lm_head.py`, `prepare/quant_mtp.py`):
   - `lm_head.weight` [248k, 5120] bf16 → int8 **group-128 symmetric**, compressed-tensors pack-quantized format (`weight_packed` int32, `weight_scale` fp16 [N, K/128], `weight_shape`).
   - All 8 MTP draft linears (fc + one decoder layer's q/k/v/o/gate/up/down) → same int8 group-128 format.
   - `config.json` `quantization_config.config_groups` gets new groups targeting `re:.*lm_head$` and `re:^mtp\..*`; `ignore` list updated. Backups written next to originals.
   - Measured round-trip error: lm_head 0.64% (Frobenius), each MTP linear printed per-module.
2. **Stock vLLM runtime path** — zero custom kernels for the int8 math itself. The checkpoint is just a normal compressed-tensors WNA16 checkpoint; vLLM's `CompressedTensorsWNA16` scheme + kernel selection does the rest.
3. **A 2-line model patch per embedding** (`patches/qwen3_5-embed-quant.patch`) — see §4, that is a *separate* lever (embed_tokens), not part of SYV-3's GEMM path.

**Their measured results (docs/optimizations.md + docs/quality.md, RTX 3090):**
- int8 lm_head: **+12% aggregate throughput**, ~1.3 GB VRAM freed (per their docstring).
- int8 MTP module: "a draft now costs ~0.5–1 ms and four of them pay off" (vs ~3 ms bf16 per extra draft).
- Quality: lm_head int8 → perplexity 10.68→(baseline), GSM8K 95.5%, acceptance 107/109 vs baseline; **int8 MTP: "measured acceptance change: none"** (their docstring, quant_mtp.py).
- They also ship an int4 GPTQ variant ("fast variant": int4 lm_head + int4 MTP, GSM8K 96.5%, ~114/124 tok/s) — a further step we have not considered at all.

## 2. What we do (our T-1)

- **Runtime load-time transform** (`T1_INT8_MASS=1`, default OFF): quantize `model.lm_head` + the draft linears in-process to **per-row (channel-wise, group-K) symmetric int8**, served by our **custom m4 CUDA GEMV kernel** (`dense_gemv_gfx906`, M≤4, fp16-A) with a dequant-cache fallback for M>4.
- No checkpoint surgery; the AWQ-INT4 checkpoint stays untouched.

## 3. Side-by-side diff

| Aspect | syv SYV-3 | our T-1 |
|---|---|---|
| Quantization granularity | **group-128 symmetric** (scale per 128 K-elements) | **per-row / channel-wise** (one scale per output row, group-K = full K) |
| Where quant happens | offline, in the checkpoint file | runtime, at load (`process_weights_after_loading`) |
| Runtime kernel | stock vLLM WNA16 dispatch: Marlin (sm86) / Exllama+gptq_gemm (gfx906 ROCm list) | custom `dense_gemv_i8_m4_gfx906` CUDA op + dequant-cache GEMM fallback for M>4 |
| Weight format | compressed-tensors pack-quantized (int32-packed int8, fp16 scales [N,K/128]) | raw int8 [N,K] + fp16 scale [N] |
| VRAM per rank (lm_head) | 0.64 GB int8 (+scales ~1 MB) | 0.64 GB int8 (+scale ~0.5 MB) — **same** |
| VRAM at M>4 fallback | n/a (kernel path, no dequant cache) | **+cached bf16 copy (~0.64 GB/rank)** until T-1.5 removes it |
| k=4 drafter linears (M=3/4 step 0) | int8 kernel path directly | M≤4 → m4 kernel ✓ (same idea) |
| fc inclusion | included by default (`--keep-fc` to exclude) | **excluded by default** (dominant-element rows → 12.5% rel-L2), behind `T1_INT8_MASS_FC` |
| Env guard / optionality | n/a (it's the checkpoint) | `T1_INT8_MASS=1` default OFF ✓ |
| A/B result on our box | n/a (different HW: sm86, 3090) | **NOT PASS** — parity at k=2 (−3%), parity at k=4 (−1% after M>4 fix) |

### 3.1 Granularity difference (the main technical divergence)

syv uses **group-128**; we use **per-row**. For a [N,K] weight:
- group-128: N·(K/128) scales, finer error control → lower round-trip error per element.
- per-row: N scales, coarser — but our quality gate (0/120 argmax flips on real hidden states, KLD max 2.75e-4 nats) already proved per-row is *safe* for this model's lm_head under greedy verify. The drafter linears are speed-only by construction (draft distribution never verified). So the granularity difference is **not a correctness gap** for us — it would matter if we ever shipped int8 with *stochastic* sampling of the target head, which we do not (target stays bf16).

### 3.2 Why syv's approach nets +12% and ours nets parity

This is the crux of the recon. Three compounding reasons:

1. **Hardware/kernel asymmetry.** On sm86, vLLM's WNA16 dispatch lands on **Marlin** — a fused int8→fp16 tensor-core GEMM that reads int8 bytes directly (no fp16 weight materialization, no separate dequant pass). Our gfx906 ROCm list has **no Marlin**; the stock path would be Exllama (`gptq_gemm`, 4-bit-oriented) or Conch (~3.8 ms/M=1 GEMV on MI50 per our own Nemotron devlog — that is why we built custom kernels at all). So syv's "free" stock path is only free *on sm86*. On gfx906 the stock path is slow, which is exactly the gap our m4 kernel was built to close.
2. **We already captured the kernel win; what remains is overhead.** Our m4 kernel runs at 821 GB/s (82% of peak) — faster per-byte than anything stock offers on gfx906. The parity result means the *remaining* T-1 cost (dispatch prologue, bf16→fp16 cast on the M≤4 path, kchunk table gaps for o_proj K=3072 / down K=8704) is roughly equal to the bytes saved at our step shape. syv doesn't pay those overheads because their runtime is stock vLLM's (already-optimized dispatch, no cast — Marlin takes fp16 activations natively... on sm86; and their drafter linears go through the same stock path).
3. **Their +12% is an aggregate/batch number** (64 concurrent, int8 tensor-core GEMMs for the batched verify). Our A/B is single-stream decode at long context where the lm_head GEMV is a smaller fraction of the step. Not directly comparable, but it means even their own headline overstates what the *same* lever is worth in our serving regime (consistent with round-4/5 gap analysis: drafter/target-head levers are diluted by our heavier target-verify-forward).

**Conclusion:** T-1 is not a buggy port — it implements the same principle (int8 weight-only on the mass matrices) with a *better* kernel for our hardware, and correctly lands at parity once overheads are accounted. The upstream approach would NOT reproduce its +12% on gfx906 because the stock kernel path that makes it free there does not exist here. **No missing part to port from SYV-3 itself.**

## 4. Missing parts found in the recon (actionable)

### 4.1 `embed_tokens` is still full bf16 — syv requantizes it, we do not (GAP)

Both our checkpoint and syv's leave **two** untied 2.5 GB bf16 matrices: `lm_head` and `model.language_model.embed_tokens` (verified: `tie_word_embeddings: false`, both [248320, 5120] BF16 in the AWQ-INT4 snapshot). syv requantizes **both** (`quant_lm_head.py` + `quant_embed.py`) — "2.6 GB back" — and wires vLLM's existing quantized-embedding kernel via a **2-line patch** (`patches/qwen3_5-embed-quant.patch`: pass `quant_config` + `prefix` to `VocabParallelEmbedding` in both `qwen3_5.py` and `qwen3_5_mtp.py`).

Our model code (`vllm/model_executor/models/qwen3_5.py:244`) does **not** pass `quant_config` to the embedding — same upstream gap syv patched. T-1 only touches `lm_head`, so we still carry the full 2.5 GB bf16 embed table (1.27 GB/rank under TP=2).

**Value on our box:**
- VRAM: ~1.27 GB/rank freed → more KV cache headroom (real, measurable at boot).
- Speed: modest — embedding gather is 10 KB/row (one row per token), not the full-matrix read of lm_head. Expect <1% e2e; the win is memory, which matters on this box.
- **Caveat for our checkpoint:** the AWQ-INT4 checkpoint has no compressed-tensors `quantization_config` covering embeddings, so syv's exact 2-line patch cannot be applied as-is — it needs either (a) an offline requant of `embed_tokens` to int8 group-128 CT format + config surgery (syv's recipe, ported), or (b) a runtime transform in the T-1 style (extend `t1_int8_w8a16.py` to also transform the embedding + route through vLLM's dequant-on-gather kernel). Option (a) is closer to upstream and reuses their scripts; option (b) keeps the checkpoint untouched (our convention so far).

### 4.2 fc default: syv includes it, we exclude it (documented divergence, not a bug)

syv quantizes `mtp.fc` by default (`--keep-fc` opts out); we exclude it by default because our quality probe found dominant-element rows → 12.5% rel-L2 on this model's fc. Keep as-is; revisit only if T-1 ships and the fc slice is measured to matter (it is ~0.1 GB/rank, small).

### 4.3 int4 GPTQ variant (not ported, out of scope for now)

syv's "fast variant" goes further: int4 GPTQ lm_head + int4 MTP (calibrated on the model's own hidden states), GSM8K 96.5%, ~114/124 tok/s on their box. This is a separate, larger project (GPTQ calibration pipeline + W4A16 kernel path) and **not** part of T-1's scope. Recorded here so it isn't re-discovered later; the draft-vocab work (CAT-1) already captures most of the drafter-side value on our box.

## 5. Bugs/inefficiencies found in OUR code during this recon

(Tracked and being fixed on this branch — see devlog follow-up list.)
1. kchunk table gaps: o_proj K=3072 and down K=8704 hit the default kc (unmeasured) — follow-up #4.
2. bf16→fp16 cast on the M≤4 path — follow-up #3.
3. per-layer bound closure not fully hoisted (residual attribute lookups in `_t1_apply`) — follow-up #2 rest.
4. M>4 dequant cache VRAM cost — T-1.5 (follow-up #5) removes it.

## 6. Verdict

- **SYV-3 itself: nothing to port.** Same principle, better kernel for our HW, parity is the expected and correct outcome on gfx906. The upstream +12% does not transfer because its enabler (stock Marlin WNA16) does not exist on gfx906 ROCm.
- **One real gap found: `embed_tokens` int8** (§4.1) — VRAM win (~1.27 GB/rank), small speed win, ported from syv's recipe (offline requant + 2-line model patch), adapted to our checkpoint format.
- **T-1 follow-ups #2-rest/#3/#4/#5 remain the throughput-side work** (devlog list).
