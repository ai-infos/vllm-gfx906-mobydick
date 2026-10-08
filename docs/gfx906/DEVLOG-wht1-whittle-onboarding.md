# WHT-1 — Whittle-Qwen-3.8-35B-A3B: onboarding recon (GGUF arm closed, bf16 arm pending)

Ticket: [#36](../../issues/36) · Roadmap: `ROADMAP.md` → WHT-1 · Status **2026-09-28:
postponed** pending the bf16 download; the GGUF arm is closed on this stack.

## What the model is

A second, independent implementation of the architecture we ship as `qwen4_exp`
(Qwen3.8-Flash-Next format), distilled by David Aylward (logic65) from Qwen3.8-27B.

| | |
|---|---|
| artifact | `/local/models/logic65/Whittle-Qwen-3.8-35B-A3B-GGUF/Whittle-Qwen-3.8-35B-A3B-Q8_0.gguf` (37,828,807,904 B, sha256 matches HF) |
| GGUF arch | `qwen4exp` (GGUF v3, 1022 tensors, 55 kv pairs, `general.file_type=7` Q8_0, name "Agentfix2 Bf16") |
| geometry | 40 blocks, hidden 2048; 180 routed experts × ffn 512 (8 used) + shared expert; `full_attention_interval=4`; 16 heads / 2 KV, head_dim 256; SSM conv 4 / group 16 / inner 4096 / state 128 / rank 32; indexer `top_k 262144`; hyper-connections count 4 / low_rank 320; PLE ngram_size 3, heads_per_ngram 4, 8 tables × 4,880,000 rows; ctx 262144 |
| the lever | **9.994 e9 of 35.55 e9 parameters (28.5 %) are one embedding table** — `per_layer_token_embd.weight`, (256, 39,040,000) Q8_0. The card claims it can live in host RAM while the GPU holds a 27 B-class footprint at 3 B-active decode |

HF-side names for the same tensors (from `bf16-agentfix2/model.safetensors.index.json`,
979 params): `model.ngram_embedding.shard_0..4.weight` plus
`model.ple_embedding.{layer_multipliers,ngram_heads_offsets,ngram_heads_vocab_sizes}`.

## Arm A1 (GGUF on vLLM) — NO-GO on this stack, five blockers

Method: installed `vllm-gguf-plugin==0.0.5` + PyPI `gguf==0.19.0` into the fork venv
(`--no-deps`), built a serve dir, and drove the plugin's config parser and name map against
the real file. Probe: `/local/tmp/wht1/map_probe.py`. Output of the last step is the
decisive number: **0 of 1022 tensors mapped.**

1. **`qwen4_exp_text` is not in transformers' `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`.**
   `GGUFConfigParser.parse` requires it immediately after `HFConfigParser` returns:
   `RuntimeError: Can't get gguf config for qwen4_exp_text`. Our fork *does* register the
   config (`transformers_utils/configs/__init__.py` → `Qwen4ExpConfig`/`Qwen4ExpTextConfig`;
   `config.py` maps `qwen4_exp`/`qwen4_exp_text`), so this is the plugin bypassing vLLM's
   registry, not a missing implementation.
2. **No GGUF→HF name map exists for `qwen4exp`.** llama.cpp's gguf-py carries the arch
   constant (`gguf.MODEL_ARCH_NAMES['qwen4exp']` ✓, HC_* tensor names in `constants.py`), but
   `gguf.get_tensor_name_map(arch, 40)` resolves **0 of 1022** names, and PyPI `gguf` 0.19.0
   does not know the arch at all (so the plugin needs a llama.cpp checkout on `PYTHONPATH`).
   Authoring the map means hyper-connections (`hc_*` ↔ `*_hyper_connection.*`), PLE
   (`per_layer_token_embd`, `ple_*` ↔ `ngram_embedding.shard_*`, `ple_embedding.*`), GDN
   (`ssm_a` ↔ `linear_attn.A_log`, including undoing llama.cpp's load-time transforms), the
   QSA indexer, MoE + shared experts. `localweights/vllm-gguf-plugin` (qwen35 hybrid GDN)
   is the shape of that work — a project, not a patch.
3. **The plugin's adapter keys the arch off `model_type`** (`weights_adapter/default.py`
   matches `gguf.MODEL_ARCH_NAMES` values against `config.model_type`), so
   `qwen4_exp_text` → `Unknown gguf model_type: qwen4_exp_text`; the `qwen2_moe`/`qwen3_moe`
   rows show the `.replace("_", "")` trick this would need.
4. **The plugin's kernels are CUDA-only.** `_C_gguf.abi3.so` links `libcudart.so.13`,
   `libtorch_cuda.so`, `libc10_cuda.so` (all unresolved on this box) — there is no ROCm
   dequant/GEMV path, so even a complete mapping would have nowhere to run. **Checked
   2026-09-28: nothing is in flight upstream either** — the plugin's recent commits are
   Gemma4 / iq3xs / mellum (#136, #122–#124), and its open PRs target Qwen-VL, DeepSeek-V4,
   Kimi-K3, MiniMax-H3, diffusion models and MoE kernel dispatch (#135, #134, #125…): no
   `qwen4exp` work, no ROCm work. The reopen gate therefore starts with an upstream change
   that does not exist yet, not with a review queue we can wait on.
5. `gguf-py` must come from llama.cpp (blocker 2), which also means pinning that checkout.

Side finding: llama.cpp's **C++** does know `LLM_ARCH_QWEN4EXP` (`src/llama-arch.h:48`,
loader handling in `src/llama-model.cpp`, PLE KV keys in `src/llama-arch.cpp`), so the
ticket's reference arm A0 remains available — only vLLM's GGUF path is closed.

## Fork-side blocker that hits every route

`Qwen4ExpConfig._validate_qsa_config` (`vllm/models/qwen4_exp/config.py:152-158`) requires
`indexer_budget // indexer_compress_ratio ∈ {512, 2048}`. This model configures
`indexer_budget 262144`, `indexer_compress_ratio 4` → `block_topk = 65536` →
`ValueError: QSA requires indexer_budget / indexer_compress_ratio to be 512 or 2048, got 65536`.

This blocks the bf16 checkpoint as well as the GGUF, so it is the first thing to resolve
once the weights are local: either parameterise the check against what the QSA indexer
kernel actually supports (its tile/selection sizes) or clamp the budget at runtime
(`topk = min(block_topk, n_blocks)`), and validate the resulting selection against a
reference (llama.cpp, which runs the same model).

## Disk arithmetic for the bf16 download (why it is not a one-liner)

`logic65/Whittle-Qwen-3.8-35B-A3B` → `bf16-agentfix2/` = **66.2 GiB** in 14 shards.

| volume | free | fits 66.2 GiB? |
|---|---|---|
| `/local` (NVMe) | 38 GiB | no |
| `/` | 28 GiB | no |
| `/data` (NFS 192.168.33.240) | 453 GiB | yes |

The n-gram/PLE tensors live in **model-00012/13/14.safetensors = 19.4 GiB total** — that is
the per-token read path (one row per token per head; PR #34 measured its cost on the big
model as the whole prefill wall). Put those three on NVMe (19.4 GiB < 38 GiB free) and the
other eleven (46.8 GiB) on `/data`, with symlinks in the model dir. They fit host RAM
(48 GiB) as page cache at bf16, which is what makes the off-GPU residency plausible at all.

## Reusable artifacts from this pass

- `/local/models/whittle-serve/` — config.json + tokenizer (from the HF repo root) + a
  `model.gguf` symlink. The config is the real geometry, so the dir also serves the bf16 arm
  (swap the symlink for the shards).
- `/local/tmp/wht1/map_probe.py` — GGUF name-map coverage probe (re-run after any upstream
  llama.cpp update; it is the cheapest check of blocker 2).
- `/local/tmp/wht1/whittle-hf-index.json` — the 979 HF parameter names and their shards.
- Env: `vllm-gguf-plugin==0.0.5` + `gguf==0.19.0` are installed in the fork venv. Inert
  unless a `.gguf` model is passed to vLLM; remove with `pip uninstall` if that is preferred.

## Next steps (in order)

1. Kevin fetches `bf16-agentfix2/` (66.2 GiB) as above — 12/13/14 to `/local`, 11 to `/data`.
2. Relax/parameterise the QSA `block_topk` validation, with a reference check against
   llama.cpp for the same model (selection sizes must agree).
3. Arm A2/A3: TP=2 bf16 with the n-gram table off-GPU (`VLLM_PLE_MMAP`), then measure peak
   VRAM/card at init and the largest `max_model_len` that fits at `gpu_memory_utilization
   0.90`, memory resident vs host RAM (expected delta ≈ 18.6 GiB at bf16), decode t/s at
   16k. Sampler per the card: temperature 0.7, top_p 0.8, top_k 20, repeat_penalty 1.05 —
   sample, do not decode greedily (the family loops on greedy).
4. Optional: A0 (llama.cpp HIP build, `-ot per_layer_token_embd=CPU`) as the card-recipe
   cross-check — its C++ supports the arch today.
5. Cheap, independent: `tests/models/qwen4_exp/test_ple_mmap_shards.py` still skips with
   *"qwen4_exp AMD model code is not importable in this build"*, which is false on today's
   tree (only the `nvidia` path fails, correctly). Fix the skip reason while in the area.

## Attribution

Whittle-Qwen-3.8-35B-A3B by **David Aylward (logic65)** with Claude (Anthropic),
Apache-2.0; distilled from Qwen3.8-27B; n-gram memory contents from
Qwen/Qwen3.8-Flash-Next. Anything published from this work carries that attribution
(repo convention: main `README.md` + the ported file).
