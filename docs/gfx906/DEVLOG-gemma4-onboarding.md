# Gemma-4-26B-A4B-AWQ onboarding (2026-08-19)

Branch: on `gfx906/main` (docs-only changes; no code). Checkpoint:
`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (17.2 GB, base
`google/gemma-4-26B-A4B-it`), moved to the standard HF layout under
`/local/cache/huggingface/hub/` (relative symlinks survived the move;
offline resolution verified).

**VERDICT: ONBOARDED — loads, serves with quality, 37.6 t/s decode
(graph mode).** Our custom gfx906 MoE kernel does NOT serve it (the
checkpoint has no zero points — the oracle's AWQ-only gate is correct
behavior); the MoE runs on the generic Triton WNA16 path, which is 46%
of the decode step and the clear next optimization target (a no-zp /
GPTQ-style W4A16 expert kernel).

## 1. Model facts (config + tensor index)

- `Gemma4ForConditionalGeneration`, text model `gemma4`, 30 layers,
  hidden 2816, vocab 262144, 256K context. Multimodal (image; the
  vision tower loads and warms up — text-only serving unaffected).
- **All 30 layers are MoE**: E=128, topk=8, moe_inter 704, PLUS a
  per-layer shared expert (the plain `mlp.*` tensors; README: "8 active
  / 128 total and 1 shared"). The shared experts of layers 0–26 are
  unquantized bf16 (config `ignore` list); routed experts are quantized
  in all 30 layers.
- Quantization: compressed-tensors W4A16 pack-quantized, **group 32,
  `symmetric: true`, NO qzeros** (`weight_packed` + `weight_scale`
  only). Attention q/k/o projections are also quantized.
- Attention: hybrid — 25 `sliding_attention` (window 1024, head_dim 256,
  16 q / 8 kv heads) + 5 `full_attention` (head_dim 512,
  **`attention_k_eq_v: true`** — V aliased from K, no `v_proj` tensors,
  `num_global_key_value_heads: 2`, proportional RoPE,
  `partial_rotary_factor 0.25`). `final_logit_softcapping: 30.0`.
- `tie_word_embeddings: true`; `dtype: bfloat16` in the config.

## 2. What routes where (measured, from the load log)

- dtype: checkpoint bf16 → **our auto-dtype gfx906 fallback
  (`69f615b98a`) fires** — the shared bf16→fp16 resolver whose detailed
  record is `DEVLOG-qwen38.md` §Fix (this is the second model validated
  on it); warning "Checkpoint dtype is bfloat16, but this device has no
  native bfloat16 support; auto-selecting float16".
- Attention: `Gemma4Config.verify_and_update_config` detects
  heterogeneous head dims (256/512); FA4 unavailable → forces
  **TRITON_ATTN** (the unified Triton attention). k_eq_v is handled in
  model code (K weights loaded into both K and V slots); sliding window
  supported by the backend.
- MoE: `CompressedTensorsWNA16MoEMethod` → WNA16 oracle → **TRITON**
  backend (`TritonWNA16Experts`, `fused_moe_kernel_gptq_awq`). Our
  gfx906 backend is rejected by the design gate
  (`"zero points are required (AWQ-style checkpoints)"`) — the correct
  outcome for a symmetric no-zp checkpoint.
- Dense W4A16 (quantized attention projections + shared-expert-adjacent
  linears): `CompressedTensorsWNA16` with **ExllamaLinearKernel**
  (our `triton_matmul` / `LLGemm1` paths).

## 3. Quality gate — PASSED (with one big caveat found along the way)

**Caveat (raw prompts):** `llm.generate` with bare continuation prompts
produces confident degenerate repetition ("The capital of France is" →
" capital of France is capital of France is…", p≈0.95 per repeat token,
greedy AND sampled, all prompts). This is a **prompt-format artifact of
the thinking-mode instruct model, not a dtype/attention bug**: with the
model's own chat template (`apply_chat_template` / the OpenAI chat API)
output is coherent and correct in fp16 — "Tokyo", "Paris", "Two",
distinct coherent haikus per sample. Any gate on this model must use
templated messages; raw prompts will read as "broken".

- Chat API smoke test: `--dtype float16 --enforce-eager
  --gpu-memory-utilization 0.9 --max-model-len 8192 --max-num-seqs 16`
  → "What is the capital of Japan?" → **"Tokyo"**.
- bf16 (the family's nominal dtype) is **structurally blocked** on this
  stack: the Exllama dense path hits our fp16-only
  `triton_matmul` assert ("Matrices A and B must have the same dtype
  (assuming fp16)"). fp16 is the only runnable dtype here — and it is
  numerically fine for this model (unlike gemma2/3, which are on the
  `_FLOAT16_NOT_SUPPORTED_MODELS` blocklist; gemma4 is not, and the
  quality gate above is the empirical confirmation).
- KV pool at 0.95: 53,434 tokens (11.27 GiB); weights 18.2 GiB.

## 4. Benchmarks (graph mode, AGENTS.md recipe: pp=2048 tg=256,
   BENCH_MAX_SEQS=32, util 0.95, 4 samples, local venv)

| sample | t/s |
|---|---|
| 0 | 37.623 |
| 1 | 37.600 |
| 2 | 37.565 |
| 3 | 37.567 |

**Record: 37.6 t/s (mean 37.59, spread 0.06).** For reference:
Qwen3.5-35B-A3B (custom-kernel MoE) 66.5 t/s; dense 27B ~25.3 t/s.

## 5. Per-kernel census (eager, `kernel_prof_probe.py`, GPU-busy
   30,192 µs/step, 256 decode + 1 prefill)

| µs/step | cnt/step | kernel | note |
|---|---|---|---|
| 13,915 (46%) | 59.77 | `fused_moe_kernel_gptq_awq` | MoE expert GEMMs, 2/layer × 30 layers. 232.8 µs/call ≈ ~17.8 MB active expert weights/layer at **~38 GB/s effective** — an order of magnitude off the MI50 HBM floor. The dominant lever. |
| 5,214 (17%) | 29.88 | `kernel_unified_attention` | 30 hybrid-attention layers, 174.5 µs/call (2048+ctx). |
| 3,495 (12%) | 90.30 | `LLGemm1_kernel` | shared-expert dense GEMMs (3/layer × 30), 38.7 µs/call. |
| 1,755 (6%) | 59.53 | `gemm_half_q_half_gptq_4bit_m1mi` | quantized q/k/o projections (2/layer), 29.5 µs/call. |
| 1,749 (6%) | 269.95 | `rms_norm_kernel` | 9/layer. |
| ~1,400 | — | elementwise/add/gelu glue | |
| 262 (1%) | 29.88 | `_gemma4_routing_kernel` | the upstream routing kernel (8.8 µs/call — no S2-class gap). |

## 6. Optimization target (recorded for the roadmap)

**No-zp (GPTQ-style) W4A16 MoE expert kernel for gfx906.** Our
`moe_gemm_q4` family requires stored AWQ zero points; the oracle gate
is correct as-is. Extending the kernel family to symmetric no-zp
(dequant = `(q - 8) * scale` with a constant offset — the zero-point
machinery collapses to a per-group bias) plus a compressed-tensors
repack would let gemma4's MoE use the custom path. Standalone
expectation from the 35B kernel's ~285 GB/s/call: gemm1
(8 × 1408×2816 int4 ≈ 17.8 MB) ~60–90 µs vs 232.8 µs today, i.e.
**~10 ms/step recoverable** (30 layers) → ~50–55 t/s class. Gate
sequence per house protocol: harness (add no-zp variant +
symmetric-check), unit tests, PPL, serving A/B 4 samples.

Secondary (smaller): Triton `E=128,N=704` MoE config file for
gfx906 (`Using default MoE config` warning) — autotune once the
custom-kernel question is settled.

## 7. State

- Model at `/local/cache/huggingface/hub/models--cyankiwi--gemma-4-26B-A4B-it-AWQ-4bit`
  (standard layout, offline-resolvable).
- No code changes; the only stack behavior exercised was the existing
  dtype fallback. Docs: this log + `roadmap-more-models.md` and
  `CHANGELOG.md` (the 2026-08-18 note "gemma-4 is GGUF-only" is superseded).
