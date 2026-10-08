# WHT-1: Whittle-Qwen-3.8-35B-A3B loads and serves on gfx906 (TP=2)

Date: 2026-10-05. Item: **#36 WHT-1**. Branch: `gfx906/wht1-qsa-topk`.

## Result

The bf16 checkpoint (`logic65/Whittle-Qwen-3.8-35B-A3B`, root = Phase-2 step 32010,
14 shards / 66.26 GiB, in `/biglocal/cache/hf`) **loads and serves** on 2x MI50
32 GB. First end-to-end success after five blockers, four of them real.

| Fact | Value |
| --- | --- |
| launch | TP=2 + EP=2, `--dtype float16`, `--gpu-memory-utilization 0.85` |
| limits | `--max-model-len 16384 --max-num-seqs 2 --max-num-batched-tokens 512 --block-size 64` |
| graphs | `--enforce-eager` (see blocker 5) |
| spec decode | none (`speculative_config=None`; see blocker 4) |
| weight load | **24.38 GiB / rank**, 146.6 s and 147.0 s |
| KV cache | 2.0 GiB → **117,964 tokens**, 7.20x concurrency at 16k |
| VRAM during load | 25.03 GiB / card |
| throughput (B=1, eager) | 2.75 t/s rep 1 (includes JIT warmup), **5.61 t/s rep 2** |
| coherence | `2+2=` → `4, 2+2+2=6, 2+2+2+2=8. So 4.`; "one sentence about the sea" → one sensible sentence |
| canary before/after | 38.8 / 38.7 t/s (healthy band) |

Repro: `/local/tmp/wht1/smoke_load2.sh` (log `/local/tmp/wht1/smoke2/`).

## Blockers, in the order they bit

### 1. `NotImplementedError: Qwen4Exp QSA requires FlashAttention`

Not the model. On ROCm that gate is literally "did `from flash_attn import
flash_attn_varlen_func` succeed at import time", and this venv's `flash_attn` is
the `/local/git/flash-attention-gfx906` fork **without a built C extension** --
it only imports with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (Triton-AMD path),
which `docs/gfx906/running.md` documents. Verified directly: the flag flips
`is_flash_attn_varlen_func_available()` False -> True.

Three recipe deltas, all in `docs/gfx906/_serve_qsa_flash_gfx906.sh`: that env
var, `--dtype float16`, and `VLLM_USE_V2_MODEL_RUNNER=1` ("Qwen4Exp hides the
PLE/ngram inputs behind the V2 model states; on V1 the PLE layer raises"). All
three were missing from the first attempt.

### 2. `torch.OutOfMemoryError: Tried to allocate 4.00 GiB` (per layer, at layer 7)

`vllm/models/qwen4_exp/amd/qsa.py` sizes the indexer's `topk_indices_buffer` as
`(max_num_batched_tokens, indexer.output_width)` int32 **per attention layer**.
This checkpoint carries `indexer_budget = 262144` -- its whole
`max_position_embeddings` -- so `output_width = budget + ratio - 1 = 262144`, and
at MBT 4096 that is 4 GiB per layer, 40 layers. Arithmetic checks out: 27.69 GiB
resident at the failure is 6-7 layers of buffers plus their weights.

Root cause is geometry, not a leak: a row can only ever address
`ceil(seq_len / compress_ratio)` compressed tokens, and `seq_len <= max_model_len`.
Fixed with `addressable_token_topk()` in `amd/indexer_qsa.py`, which clamps the
selection width to that ceiling (rounded up to a multiple of the ratio, which the
expansion kernel requires). Lossless -- it selects the same set -- and it pulls
`block_topk` back from 65536 to 1024, inside the kernel's measured 8192 ceiling,
so the fast path runs instead of the reference fallback.

### 3. `ValueError: There is no module or parameter named 'ngram_embedding'` (shard 11/14)

A checkpoint-side re-layout. The family nests the PLE embedding table under its
layer:

```
model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0..N.weight
model.language_model.layers.1.ple.ple_embedding.layer_multipliers
```

which matches the fork's module tree exactly. Whittle ships the same table at the
**checkpoint root**:

```
model.ngram_embedding.shard_0..4.weight
model.ple_embedding.layer_multipliers
```

so the loader had nothing to bind them to. Fixed with
`_remap_packed_ple_table_name()`, an anchored root->layer remap in
`Qwen4ExpModel.load_weights`, driven by `ple_layer_ids` (1-based: `[2]` -> layer
1) and inert when the model does not own exactly one PLE layer. Parent-style names
already carry a layer prefix, so the anchored match leaves them untouched: both
layouts load.

### 4. MTP drafter: `Following weights were not initialized from checkpoint`

The uninitialised list is the drafter's own parameters (`pre_fc_norm_embedding`,
`pre_fc_norm_hidden`, `hyper_connection_mixer.hc_norm`, `layers.0.self_attn.*`).
The checkpoint's index contains **zero** `mtp`/`nextn`/`draft`/`eagle` tensors
while `config.json` advertises `mtp_num_hidden_layers: 1` -- the config lies about
what it ships, so speculative decode is impossible on this checkpoint. Served with
`speculative_config=None`. The fork's auto-MTP paths are architecture-gated to
qwen3_5 / intern_s2 / deepseek, so `qwen4_exp` does not auto-enable it; it only
bit because the recipe's `--speculative-config` was passed.

### 5. `ConstraintViolationError: Constraints violated (L['query_start_loc'].size()[0])`

Raised in `_initialize_kv_caches -> determine_available_memory`, i.e. the
memory-profiling dummy run, after weight load succeeded (144.3 s, "Model loading
took 24.38 GiB"). First suspected the explicit `cudagraph_capture_sizes` ladder
(`[1,2]` cannot cover the profiler's `max_num_seqs + 1` decode rows), but vLLM's
own default power-of-two ladder fails identically, so it is the V2 model runner's
compile path meeting the profiler's shapes on this build, not the ladder.
`--enforce-eager` sidesteps it for the smoke test. Graph capture is a follow-up.

## Ruled out

| Hypothesis | Verdict |
| --- | --- |
| VRAM ceiling | No -- 24.38 GiB/rank of weights + 2.0 GiB KV fits at util 0.85 |
| PLE/ngram table on the GPU | No -- 48.8 GiB of weights across 2 cards vs a 66.26 GiB checkpoint, so ~17 GiB of table stays in host RAM (host RAM had 33 GiB available, memory PSI 0.7) |
| Broken/truncated download | No -- 14/14 shards, every root file matches the hub size, all headers readable, 1059 tensors |
| `--compilation-config` ladder | No -- default ladder fails the same way (blocker 5) |
| MTP weights present but misnamed | No -- 0 candidate keys in the index; drafter params are simply absent |

## Next

- **Graph-capture arm** (CLOSED 2026-10-05 — `DEVLOG-wht1-graph-capture.md`):
  drop `--enforce-eager` once the V2-runner profiler
  interaction is understood, then re-measure. The tester's validated 4x MI50
  config (TP=4, util 0.91, MBT 4096, MTP k=3) is **46.8 t/s at B=1, 25.4 without
  MTP**; our 5.61 t/s is eager, TP=2, MTP-less, so it is not comparable yet.
- **Envelope**: largest `max_model_len`/MBT that fits at util 0.85, TP=2. The
  index buffer scales with MBT (`MBT x (token_topk + ratio - 1) x 4 B x 40`), so
  MBT is the lever -- which is exactly why the tester's MBT 4096 does not transfer
  to TP=2 (their per-card weights are half ours).
- **Report to the uploader**: the root-packed PLE table (the fork now adapts) and
  the `mtp_num_hidden_layers` claim with no MTP tensors in the checkpoint.
- Quality gates once the perf arm settles (no quality-changing behavior shipped
  here: the width clamp is lossless by construction and the name remap only
  relocates weights).
