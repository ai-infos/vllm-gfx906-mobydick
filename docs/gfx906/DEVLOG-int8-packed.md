# INT8-PACKED-1 — compressed-tensors `pack-quantized` W8A16 on gfx906

## 2026-09-16 — the first INT8 W8A16 checkpoint on this fork: a two-line wiring gap, then it loads and
## generates coherent output

**VERDICT:** VIABLE (numerics gate still pending) · **GATE:** `lued/Qwen3.8-27B-INT8-W8A16-DFlash2`
served at TP=2, `--dtype float16`, `--kv-cache-dtype float16` (see the trap below), 8k ctx, no spec
decode — loads with **zero skipped tensors**, allocates a 252,196-token KV pool, and answers templated
chat prompts correctly.

## HYPOTHESIS

Our fork cannot serve compressed-tensors `pack-quantized` INT8/W8A16 checkpoints (the blocker recorded
in `DEVLOG-dflash2.md` as the DFlash2 INT8 route's gate). Falsify: load one and generate.

## What was done

1. **Checkpoint contract** (`lued/Qwen3.8-27B-INT8-W8A16-DFlash2`, the VL variant
   `Qwen3_5ForConditionalGeneration`): `format: pack-quantized`, two config groups — `group_0`
   (`Linear`) and `group_embed` (`re:.*embed_tokens$`) — both **int8, symmetric, strategy `group`,
   group_size 128**; compressed-tensors schema `version 0.18.0`; `lm_head`, `in_proj_a/b`, all `mtp`
   and the whole visual tower are in `ignore`. 1986 tensors total: 401 `weight_packed` + 401
   `weight_scale` + 401 `weight_shape` + 521 `.weight` + 166 `.bias`.
2. **Scheme resolution probe** (`tools/ct_scheme_probe.py`, CPU-only): our tree already routes both
   groups to `CompressedTensorsWNA16(strategy=group, num_bits=8, group_size=128, symmetric=True)`,
   and a dedicated embedding scheme exists — `compressed_tensors_embedding.py`
   (`CompressedTensorsEmbeddingWNA16Int`), whose Triton `_dequant_gather_kernel` gathers rows by token
   id, unpacks int32-packed INT weights and dequantises in one pass, with both `GROUP_SIZE == 0`
   (channel) and group-scale paths.
3. **Root cause of the load failure was model wiring, not quantisation support:** `Qwen3_5Model`
   built `embed_tokens` as `VocabParallelEmbedding(vocab, hidden)` — no `quant_config`, no `prefix` —
   so the layer was never handed to the quant config, never registered the packed parameters, and the
   loader died with `ValueError: There is no module or parameter named 'embed_tokens.weight_packed'`.
   `VocabParallelEmbedding` resolves its method as `quant_config.get_quant_method(self, prefix=prefix)`,
   so the checkpoint's *name* patterns (`re:.*embed_tokens$`, `ignore: lm_head`) need the prefix too.
   **Fix: two lines** (AFMoe/AXK1/bailing_moe already do this):
   `quant_config=self.quant_config, prefix=maybe_prefix(prefix, "embed_tokens")`.
4. Serve the checkpoint (TP=2, 8k ctx, no spec decode) and sanity-generate through the chat template.

## Evidence FOR (what the gate measured)

- **Load:** `Application startup complete` after 520 s (cold Triton/inductor compile), **no**
  `not initialized` / missing / unexpected-key lines — i.e. every checkpoint tensor, packed ones
  included, was consumed.
- `GPU KV cache size: 252,196 tokens`; 29.1 GB resident on GPU0 (29.6 GB checkpoint at TP=2).
- Coherent templated answers (greedy, temperature 0):
  - *"What is the capital of France? Answer with one word."* -> `The user asks for the capital of
    France and requests a one-word answer. The capital of France is Paris.\n</think>\n\nParis`
  - *"Write a Python function that reverses a string."* -> opens a proper `\`\`\`python` block.
- The resolved scheme in the server log is `CompressedTensorsWNA16` (linear + embedding).
- The compressed-tensors *library* here is **0.17.0** while the checkpoint's schema says 0.18.0 —
  no incompatibility surfaced at load, so no library bump is forced.

## Evidence AGAINST / still open

- **Numerics are not gated yet.** Two coherent answers are an output-sanity gate, not a correctness
  gate. The gate to run: first-token top-k logprobs of the INT8 model against **bf16
  `Qwen/Qwen3.8-27B`** (both on `/data`) via `benchmarks/kernels/gfx906/ift_chat_gate.py` — templated,
  because this is an instruction-tuned checkpoint and prompt-logprob PPL is not a gate for IFT models.
- **Performance is unmeasured.** The only numbers seen (3.3 prompt / 1.9 generation t/s) come from the
  first request and include compile + warmup; a proper serving A/B is needed before any claim.
- **`qwen3_5_mtp.py` has the same wiring gap** (`embed_tokens` without `quant_config`): latent for this
  checkpoint (all `mtp` weights are `ignore`d), required for one that quantises MTP.
- Upstream `main` has the same gap in `qwen3_5.py` (checked 2026-09-16), so this fix looks upstreamable
  — but that also means the club-3090 INT8 recipe cannot work as written on stock vLLM; worth asking
  the other box to confirm empirically (it is in `HANDOVER-dflash2.md`'s scope).
- 8-bit `int_quantized` (the ecosystem's usual 8-bit form) and non-group strategies are untested here;
  this log covers the `pack-quantized`/group-128 path only.

## Why it failed (and why it now doesn't)

The packing, the group-128 scales and the embedding dequant-gather were all already implemented; the
model simply never asked for a quant method on its embedding. That is why the error was a *key-name*
error rather than a kernel or dtype error — the layer had no packed parameters to load into.

## Interactions / superseded-by

- Unblocks the **DFlash2 INT8 arm** (`DEVLOG-dflash2.md`, ROADMAP DFL2-1): the matched pair
  (INT8 W8A16 target + `lued/Qwen3.8-27B-DFlash2-W8`) can now be served on this fork.
- Trap worth keeping: `--kv-cache-dtype` does **not** accept `fp16` (choices: `auto`, `bfloat16`,
  `float16`, `fp8*`). Passing `fp16` makes `vllm serve` die in argparse before any logger exists,
  which is what silently killed the earlier chained DFlash2 INT8 session (empty server log, `rows=0`).
- This boot (2026-09-15 13:20) wedged once during the first INT8 load (half-wedge, `reset(4)`
  recovered, `degradation.md` #96); the authorized retry loaded cleanly, i.e. that failure was the
  boot's load lottery rather than this change.

## 2026-09-16 (promotion) — cherry-picked to `main` and re-gated there

**VERDICT:** on `main` · **GATE:** the same load+sanity gate, re-run on main's tree.

The INT8/W8A16 work (`309862a0b8` + the dev-log correction `20f356dbc3`) was cherry-picked from
`gfx906/int8-packed` onto `main` (clean, no conflicts; plus a follow-up fixing the lint fallout in
`tools/ct_scheme_probe.py`, which had been created by header injection into a scratch script).
Re-gated on main: `Application startup complete` after 300 s, `GPU KV cache size: 240,493 tokens`,
no skipped/unexpected-key lines, and the same two coherent templated answers as on the branch
("...The capital of France is Paris."; a fenced python block). The DFlash2 records stay on
`gfx906/dflash2`; only the quantisation work is promoted.
