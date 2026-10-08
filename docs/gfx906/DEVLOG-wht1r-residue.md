# WHT-1r — WHT-1 residue: the graphed envelope, the dynamic-shapes bisect, the AMD warmup gap

**Issue:** KIntegrated/vllm-gfx906-mobydick#39 (residue of #36 / WHT-1, closed 2026-10-05).
**Status:** campaign complete for items 1, 2 and 3; **item 4 is blocked by a new bug found
while measuring it — issue #40** (the PLE n-gram lookup uses the wrong shard stride, which kills
the engine on longer generations). Bisect answered (the flag stays); envelope mapped.
**Box:** mi50-01, 2x MI50 32 GB gfx906, driver `wht1r-driver.service` / `wht1r-sweep`, port 8192.

Four items were open when WHT-1 closed. They are independent of each other except that all four
need the same load, so they are measured in one campaign of arms. Recipe and prior numbers:
`DEVLOG-wht1-whittle-load.md`, `DEVLOG-wht1-graph-capture.md`.

## Method (and why arms are expensive)

One arm = canary-gated load of `logic65/Whittle-Qwen-3.8-35B-A3B` (bf16, 66.26 GiB over 14
shards) at TP=2 + `--enable-expert-parallel`, fp16, util 0.85, `--max-num-seqs 2`, block 64, no
MTP, `VLLM_USE_V2_MODEL_RUNNER=1`, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` — about 13 minutes to
`/health`, then probes, then teardown. Discipline carried over from the serving A/B work:

- **`~/.cache/vllm/torch_compile_cache` is deleted before every arm.** Dynamic-shapes mode is a
  compile-time change; a stale artifact from the previous mode would silently answer the wrong
  question. (This is why arm B costs a full recompile.)
- **Per-arm log windows** — the byte offset of `server.log` is recorded before launch and only
  that window is analysed, so an error in arm *n* cannot be attributed to arm *n+1*.
- **Canary before and after the campaign** (`mtp1canary@mtp.service`, STOP below 35 t/s):
  before = **38.8 t/s** (healthy, 14:42:26), matching the 37.9–38.8 t/s band.
- Rep 1 of any timed series is discarded per the established rule (it carries the unwarmed
  first-call cost — which is item 3, so it is recorded rather than dropped silently).
- mclk is sampled every 2 s through the timed windows; `rocm-smi --showclocks` blocks under load.

Probes per arm: cold first call (1 token, timed — item 3), coherence at temperature 0, then
3 x 256 tokens at temperature 0 (item 4; the old numbers were 26-token probes).

## Item 3 — the AMD warmup gap: confirmed, and it is a silent no-op

**Finding.** `vllm/model_executor/warmup/kernel_warmup.py:201` calls
`qwen4_exp_qsa_triton_warmup(worker)` under `if enable_jit_warmup:` — with **no platform gate**.
The function itself resolves *nvidia* modules by name:

```python
qsa_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.indexer_qsa")
attn_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.qsa")
if qsa_module is None or attn_module is None:
    return
```

and on ROCm those names are never in `sys.modules`: `vllm/models/qwen4_exp/__init__.py:32-37`
imports `from .amd.model import (...)` and `from .amd.mtp import Qwen4ExpMTP` when
`current_platform.is_rocm()`. So on gfx906 the call returns on its second statement, without a
log line — which is why the missing warmup is invisible in the server log.

**Why it cannot simply be pointed at the AMD tree.** The warmup body calls
`warmup_qsa_mqa_paged_decode` (`vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py:309`) and
`warmup_qsa_sparse_paged_attention` (`nvidia/ops/qsa.py:622`). Neither helper exists anywhere
under `vllm/models/qwen4_exp/amd/` — the AMD kernels live in a single monolithic
`amd/ops/qsa.py` (seven `@triton.jit` kernels) with no warmup entry points at all. Wiring this
up is a port, not a rename: the AMD helpers would have to be written against the AMD kernel
signatures, and the block-table plumbing (`block_table_for()`, which reaches into
`runner_v2.block_tables.input_block_tables`) is V2-runner-specific in the same way.

**Cost.** Not correctness — Triton compiles per specialization on first launch, so the price is
a cold first request. Measured by the timed cold 1-token call and by rep 1 vs reps 2–3 of the
256-token series (arm B and the sweep arms). Prior evidence it is real: in the eager control the
26-token probe read 2.75 t/s at rep 1 against 5.61 t/s at rep 2.

**Options recorded for the item:** (a) port the two warmup helpers to `amd/ops/qsa.py` and
dispatch by platform in the warmup module; (b) record it as harmless once the cold-call cost is
quantified. The measurement decides; a sub-second cold call makes (a) pointless work.

**Verdict: (b) — recorded as harmless, with the cost on the record.** The cold 1-token call is
0.16–0.18 s across every arm that measured it, including the post-#40 validation arm (`FIX_graphed`,
0.18 s), and rep 1 of a 512-token generation runs 21.64 t/s against 25.55/25.57 t/s for reps 2–3 —
so the missing warmup costs one first-request latency (tens of milliseconds) and about 4 s of
first-request throughput on a 512-token generation. That is a first-request cost, not a serving
cost, and porting the two helpers would add AMD-side warmup code whose absence is not observable
in steady state. Reopen gate: if a future AMD `qsa`/`indexer` kernel lands with a *large*
first-call compile (multi-second cold call, measurable as a p99 latency spike on the first
request after a restart), port (a) then, with this section as the before-state.

## Item 2 — the envelope, on paper before burning arms

`addressable_token_topk()` (`amd/indexer_qsa.py:35`) caps the QSA selection width at
`ceil(max_model_len / compress_ratio)` rounded up to a multiple of the ratio (4), and the
indexer's selection buffer is `(max_num_batched_tokens, token_topk + ratio - 1)` int32 **per
attention layer**, 40 layers — `MBT x (token_topk + 3) x 4 B x 40`.

| arm | maxlen | MBT | token_topk | index buffer | KV @len | sum |
| --- | --- | --- | --- | --- | --- | --- |
| A_flag (baseline) | 16384 | 512 | 4096 | 0.313 GiB | 0.279 GiB | 0.59 GiB |
| E_eager | 16384 | 512 | 4096 | 0.313 GiB | 0.279 GiB | 0.59 GiB |
| E16k_mbt2048 | 16384 | 2048 | 4096 | 1.251 GiB | 0.279 GiB | 1.53 GiB |
| E32k_mbt512 | 32768 | 512 | 8192 | 0.625 GiB | 0.557 GiB | 1.18 GiB |
| E32k_mbt1024 | 32768 | 1024 | 8192 | 1.250 GiB | 0.557 GiB | 1.81 GiB |
| E64k_mbt512 | 65536 | 512 | 16384 | 1.250 GiB | 1.115 GiB | 2.37 GiB |

Headroom at util 0.85 on a 32 GiB card: 27.2 GiB − 24.38 GiB weights − 0.66 GiB graph pool ≈
**2.16 GiB**, and the graphed run allocated 1.07 GiB of KV at 16k (62,914 tokens). So the paper
prediction is that `E64k_mbt512` is over the line and `E32k_mbt1024` is near it — but the paper
model is a *lower bound*: what actually bounds KV is the profiled peak activation memory at
`--max-num-seqs 2` with MBT tokens in flight, which is what `determine_available_memory`
measures. That is the reason the sweep exists rather than an arithmetic answer.

## Item 1 — the bisect

Before the PLE split fix, default dynamic shapes died in `determine_available_memory` with
`ConstraintViolationError: Constraints violated (L['query_start_loc'].size()[0])`, and
`-cc.dynamic_shapes_config.type=backed_size_oblivious` was the mode that reached capture. The
split fix changed which ops are captured, not the dynamo/constraint path, so the prior is that
the flag is still required — but the prior was never tested against the current tree.

- **Arm B (`B_default`)**: no dynamic-shapes flag, split fix in place, compile cache cleared.
- **Arm A (`A_flag`)**: `-cc.dynamic_shapes_config.type=backed_size_oblivious` — the control.

### Result: the flag is still required (arm B, 14:42:32 → 14:48:13)

Arm B — no dynamic-shapes flag, split fix in place, compile cache cleared — **died before
`/health`**, 341 s after launch:

```
Worker_TP0_EP0/Worker_TP1_EP1 ... ConstraintViolationError: Constraints violated
  (L['query_start_loc'].size()[0])! For more information, run with TORCH_LOGS="+dynamic".
  ...
  vllm/v1/worker/gpu_worker.py:573 in determine_available_memory
    -> self.model_runner.profile_run()
    -> vllm/v1/worker/gpu/model_runner.py:1106
EngineCore: vllm/v1/core.py:1366 available_gpu_memory = self.model_executor.determine_available_memory()
RuntimeError: Worker failed with error 'Constraints violated (L['query_start_loc'].size()[0])'
RuntimeError: Engine core initialization failed. Failed core proc(s): {'EngineCore': 1}
```

14:47:44 — after the weights are up, inside the profiling run, which is exactly where this failed
before the split fix existed. Two things make the arm worth its 5.7 minutes:

- **The guard fired in this arm too** (`ple_ngram_embedding` present in `splitting_ops`, the
  `downgrading cudagraph_mode` warning present), so the capture-side change was applied and the
  failure simply did not move: the split fix and the dynamic-shapes mode address **different**
  failures — *which op may be captured* versus *what the dynamo guards will accept in the
  profiling run* — and the recipe needs both.
- `journalctl -k` over the arm's window: **0** kernel hits (no amdgpu/BACO/ring-timeout line), so
  this is a software assertion, not a wedge — no degradation-log entry.

**Conclusion for item 1: keep `-cc.dynamic_shapes_config.type=backed_size_oblivious`.** Arm A
(the flagged baseline) is re-measured in the sweep below for the throughput table and the cold
first-call number.

Canaries around the bisect: **38.8 t/s** before, **38.7 t/s** after.

## Results

### Items 2 + 4 — arms (all TP=2 + EP, fp16, util 0.85, `--max-num-seqs 2`, block 64, graphs on
unless noted; every arm clears the compile cache first)

| arm | maxlen | MBT | healthy | load | capture | KV | capacity | coherent | 26-tok reps 2/3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `B_default` (no flag) | 16384 | 512 | **no** | 341 s | — | — | — | — | — (died in `profile_run`) |
| `A_flag` | 16384 | 512 | yes | 701 s | 26 s / 0.66 GiB | 1.07 GiB | 62,914 tok (3.84x) | yes | 21.84 / 21.84 t/s |
| `E32k_mbt512` | 32768 | 512 | yes | 712 s | 26 s / 0.66 GiB | 0.97 GiB | **72,089 tok (2.20x)** | yes | 21.51 / 21.53 t/s |
| `E32k_mbt1024` | 32768 | 1024 | yes | 691 s | 24 s / 0.66 GiB | 0.82 GiB | 60,620 tok (1.85x) | yes | 21.68 / 21.58 t/s |
| `E64k_mbt512` | 65536 | 512 | yes | 712 s | 26 s / 0.66 GiB | 0.81 GiB | 68,344 tok (**1.04x**) | yes | 20.79 / 20.40 t/s |
| `E16k_mbt2048` | 16384 | 2048 | yes | 712 s | 24 s / 0.66 GiB | 0.82 GiB | 48,496 tok (2.96x) | yes | 21.28 / — |
| `E_eager` (no graphs) | 16384 | 512 | yes | **541 s** | — (eager) | 2.0 GiB | 117,964 tok (7.20x) | yes | 5.60 / 5.61 t/s |

**Eager vs graphed, same recipe, same probe.** The control reproduces the pre-existing eager
numbers exactly (2.0 GiB / 117,964 tokens / 7.20x at 16k, 5.61 t/s at rep 2 — the peer's own
reading), which is what makes the pair usable:

- **Throughput: 5.61 -> 21.6 t/s (3.9x)** on the identical 26-token probe. Graphs are the
  single biggest serving win in this recipe.
- **Graphs cost VRAM at the allocator level, not at the wheels:** init VRAM is *lower* graphed
  (27.9 GiB/card) than eager (29.2 GiB/card) — the eager path holds a bigger torch caching pool —
  while graphed reserves its 0.66 GiB graph pool up front and settles at 1.07 GiB of KV instead of
  2.0 GiB. Net: graphed is both faster and *roomier* for KV, which is why the envelope arms all
  ran graphed.
- **mclk differs between the modes under load**: the graphed arms sample 1000 MHz through the
  timed window, the eager arm 800 MHz. Same box, same clocks-on-demand policy, so the 3.9x is
  not a DVFS artifact — if anything eager was given *more* clock headroom and still lost.
- Eager load is faster (541 s vs ~700 s) because there is nothing to capture, and its cold first
  call is slower (0.38 s vs 0.17 s).

**Build note:** the engine logs `v0.28.0rc2` in `core.py` on this box even though the tree is the
`gfx906/v0.30.0` line — worth knowing when a recipe is quoted by version label.

**Envelope verdict (util 0.85, TP=2, `--max-num-seqs 2`, graphs on).** Every arm in the sweep
loads and captures; nothing in this range OOMs or wedges. What the numbers say:

| maxlen | MBT | KV | capacity | full-concurrency reading |
| --- | --- | --- | --- | --- |
| 16384 | 512 | 1.07 GiB | 62,914 tok | 3.84x — three 16k streams |
| 16384 | 2048 | 0.82 GiB | 48,496 tok | 2.96x |
| 32768 | 512 | 0.97 GiB | **72,089 tok** | **2.20x — two 32k streams** |
| 32768 | 1024 | 0.82 GiB | 60,620 tok | 1.85x |
| 65536 | 512 | 0.81 GiB | 68,344 tok | 1.04x — one 64k stream |

- **MBT is the lever, exactly as the load devlog predicted** — but it costs KV *capacity*, not
  the load: at 16k, going MBT 512 -> 2048 costs 23 % of the token pool (62,914 -> 48,496). The
  index buffer is paid for out of the same profiled headroom as the KV cache.
- **`max_model_len` is nearly free at MBT 512.** 16k -> 32k -> 64k moves capacity 62,914 ->
  72,089 -> 68,344 tokens: the extra-wide selection buffers fit inside headroom that the 16k arm
  was leaving idle.
- **The best cell on this box is 32768 / MBT 512**: 72,089 tokens is the only configuration that
  covers two *full* 32k requests at once (2 x 32,768 = 65,536 <= 72,089). 64k/512 serves a single
  full-context stream (1.04x) — usable, but the second concurrent long request does not fit.
- The paper model's "sum" column was indeed a lower bound: 64k/512 predicted 2.37 GiB against
  2.16 GiB of headroom and still loaded with 0.81 GiB of KV, because the profiled peak — not the
  arithmetic sum — is what `determine_available_memory` subtracts.

Two things already fall out:

- **Reproducibility is tight.** A_flag re-ran the peer's graphed recipe and returned exactly its
  numbers: capture 26 s / 0.66 GiB, KV 1.07 GiB / 62,914 tokens / 3.84x at 16k, and the same
  coherence string (`4, 2+2+2=6, 2+2+2+2=8. So 4.`) — on a wiped compile cache, four hours later.
  The graph-capture win is not a one-off.
- **Doubling `max_model_len` to 32,768 costs almost nothing:** KV went *down* (1.07 -> 0.97 GiB)
  and capacity *up* (62,914 -> 72,089 tokens), because the extra 4096-wide selection buffer
  (0.625 GiB by the paper model) still fits in the headroom that the 16k arm was not using. The
  paper model's "sum" column is thus a lower bound, as flagged: what bounds KV is the profiled
  peak, and at MBT 512 there is room to spare at 32k.

Per-request server metrics are `null` on this build, so all timings above are wall-clock (curl
around the request). The 26-token numbers are the sentence-prompt probes and stop at EOS by
themselves (`finish_reason: stop`) — they are *not* throughput measurements, which is why the
decode-length probe runs separately below.

The decode-length arms (`G_graphed`, `E_eager`) both died on their first long request — see the
new finding below. Item 4 therefore has no number yet: the probe is the reproducer.

## New finding, outside the four items: the engine dies on longer generations

The decode-length probe did not return an empty response — **it killed the engine** (HTTP 500
`EngineCore encountered an issue`), and the traceback is the same signature as the kills that
were previously attributed to `non_blocking=True` on the pinned-id copy:

```
ple_layer.py:1238 in qwen4_exp_amd_ple_ngram_embedding
    result = ple_embedding.ngram_embedding(pinned_ids).flatten(-2)
ple_layer.py:202  in MmapShardedNGramEmbedding.forward
    out[mask] = tensor.index_select(0, local_idx[mask])
IndexError: index out of range in self
```

### Cause: the runtime's shard stride does not match the checkpoint's shard layout

`MmapShardedNGramEmbedding.__init__` is handed
`shard_row_capacity = ceil(padded_vocab_size / split_ngram_parts)`, computed in the PLE owner's
`__init__` from **the runtime's own prime-based vocab layout** — 8 heads of prime size >=
4,880,000 sum to **39,040,640**, over 5 shards that is a stride of **7,808,128** and
`table_rows = 39,040,640`. The checkpoint says otherwise:

- its layout buffers (read out of `model-00012.safetensors`) are
  `ngram_heads_vocab_sizes = [4880000] * 8` and
  `ngram_heads_offsets = [0, 4880000, 9760000, ..., 34160000]` -> id space **39,040,000**;
- its shard files are a plain contiguous split: **7,812,500 x 4 + 7,790,000 = 39,040,000** rows.

The loader *does* overwrite those two buffers with the checkpoint's values (they are in
`persistent_buffers`, `ple_layer.py:519-533`), so the hash produces ids in `[0, 39,040,000)` —
but `shard_row_capacity` was fixed at construction time and is never re-derived from the shards
that are actually loaded. Two consequences:

1. **The crash.** Ids in `[4 x 7,808,128 + 7,790,000, 39,040,640)` = **[39,022,512, 39,040,640)**
   map to shard 4 with `local_idx >= 7,790,000`, past shard_4's rows.
2. **Silent wrong rows for most ids.** The stride mismatch (7,808,128 vs 7,812,500) means every
   id >= 7,808,128 is read from the wrong row: id 7,810,000 -> runtime (shard 1, local 1,872),
   files (shard 0, local 7,810,000). The existing out-of-range guard does not catch these — it
   folds ids outside `[0, table_rows)`, and these are inside it. So the PLE lookup has been
   returning trained rows belonging to *other* ids; the text stays coherent (PLE is one additive
   contribution) but this is not the model's own output.

### Why short probes passed and long ones die

The lookup produces one id per (token, head) — `ngram_ids` is `[tokens, 8]` for this checkpoint —
and the eight heads tile the id space in blocks of 4,880,000. Only the last block,
`[34,160,000, 39,040,000)`, reaches the unaddressable tail, so exactly one of a token's eight ids
can fall in the window: 17,488 / 4,880,000 = **0.36 % per token** with a near-uniform hash. A
short request usually survives (a 26-token prompt plus its 26-token answer: ~83 %), while a
512-token generation fails with ~84 % probability. That is the shape of the day: dozens of
26-token probe requests, none of which hit.

It also explains why the two decode-length arms died at the *same frame*. Both arms sent the
identical prompt and the n-gram ids are a deterministic function of the token ids, so the prompt's
ids are one draw, not two: the counting prompt sits inside the window and dies on its first long
request (12.8 s graphed, 45.9 s eager — that difference is load, not luck).

### Evidence, reproducible without a GPU

`/local/tmp/wht1r/pin_ple_range.py` drives the class's own `forward()` with the real capacity and
a short last shard: with capacity 4 and files `[4,4,4,4,3]`, id 19 raises
`IndexError: index out of range in self`, while the ids the guard is designed for (-1, and one
beyond the table) are folded to row 0 without an exception. Its crash id is outside that mini's
own id space (19 rows, ids 0-18), which is the one thing it does not mirror; the mechanism is
faithful. The faithful mirror is the regression test below, where the crash ids are *inside* the
id space: capacity 5 with four-row files (id space 20) raises on ids 4, 9, 14 and 19, and
`test_real_checkpoint_split_numbers` asserts this checkpoint's own arithmetic — crash window
`[39,022,512, 39,040,000)`, and a wrong row on any id >= 7,808,128.

### Why the AMD port did not catch it at load time

The nvidia implementation *validates the shard shape* before storing it
(`nvidia/ngram_embedding.py:896-920`):

```python
shard_size = (embedding.org_vocab_size + self.split_ngram_parts - 1) // self.split_ngram_parts
checkpoint_start = shard_index * shard_size
expected_rows = max(0, min(shard_size, embedding.org_vocab_size - checkpoint_start))
if tuple(loaded_weight.shape) != (expected_rows, embedding.embedding_dim):
    raise ValueError(...)
```

The AMD port kept the `embedding_dim` check and the `shard_index < split_ngram_parts` check but
dropped the row-count check, so a non-canonical split loads silently. On this checkpoint the
nvidia check would fire too — it expects `ceil(39,040,640 / 5) = 7,808,128` rows per shard, and
the files hold 7,812,500 — i.e. upstream fails loud where this port fails at 16:34 with an engine
death. (Two consequences worth separating: the *checkpoint's* split is non-canonical, and the
port is *silent* about it. The fix has to accept a non-canonical-but-contiguous split by taking
the boundaries from the files, and can mirror nvidia's validation for the canonical case.)

**Attribution correction.** The 00:26/00:31 kills were recorded as caused by `non_blocking=True`
on the pinned-id copy. `non_blocking=False` remains right (the host must not read stale or
uninitialized ids) but it is not the whole story — this padded-stride mismatch produces the same
exception text and is provable on CPU. The two mechanisms were conflated.

### Fix options (not implemented here)

- **(a) Correctness.** Build the id -> (shard, row) map from what is on disk: per-shard cumulative
  offsets taken from the loaded shard shapes (or the safetensors headers), validated to sum to the
  model's id space, failing loudly if they do not. Only this fixes the silent wrong-row reads.
- **(b) Containment.** Re-derive the capacity after the layout buffers are loaded and fold ids
  that exceed the *actual* shard rows, so any residual mismatch degrades to row 0 instead of
  killing the engine.
- (b) alone stops the crash; (a) is what makes the served model the model.

## The fix (option (a), implemented in the fork)

The decision above was to fix the fork rather than ask the uploader to re-split, so the boundaries
now come from the shards that were loaded and the id space from the checkpoint's own layout buffers,
with the two required to agree.

- `build_ple_shard_layout(rows, num_shards=..., shard_row_capacity=..., vocab_size=...)` (new) turns
  the *loaded* shards' row counts into cumulative `starts`, refuses a split whose rows do not add up
  to the model's id space, and warns when the split is not the one vLLM's own splitter would have
  written (`canonical_ple_shard_rows`) -- so a non-canonical checkpoint is served correctly *and*
  says so in the log.
- `MmapShardedNGramEmbedding.finalize_shard_layout(vocab_size)` records that layout;
  `Qwen4ExpNGramEmbedding.load_weights` calls it at the end of every call, so a chunked delivery
  completes on the last chunk. Until then -- dummy weights, or a harness that constructs the class
  directly -- the configured uniform split stands, and a shard that never arrives still raises on
  use instead of being papered over.
- `forward` masks each shard by `flat_ids in [start, start + rows)` instead of
  `id // shard_row_capacity`, and folds against the table's real row count (`table_rows`, now the
  loaded rows: 39,040,000 for this checkpoint, not the runtime layout's 39,040,640). The loop stays
  fixed-length over all shards, so CUDA-graph capture safety is not traded away.
- The id space is `max(ngram_heads_offsets + ngram_heads_vocab_sizes)`, read *after* the checkpoint's
  buffers have replaced the runtime's; when it is unavailable the rows still define the mapping and
  only the consistency check is skipped, with a warning.

Containment (option b) is implicit -- the fold now uses the loaded row count -- but the mapping
itself is (a): every id reads the row the checkpoint stored for it.

What this deliberately does not do: it does not re-split the checkpoint, does not write to the
uploader's files, and does not touch the nvidia path (which already validates, and would refuse this
checkpoint outright -- `logic65`'s split is non-canonical relative to vLLM's formula, which stays a
note for the uploader).

### Tests

`tests/models/qwen4_exp/test_ple_mmap_shards.py`: 12 new tests, 34 in the file, all passing
(`test_ple.py` 50 and `test_ple_table_remap_amd.py` 9 still pass; ruff check + format clean).

- `test_uniform_stride_mis_locates_ids_when_the_split_differs` -- the bug, scaled: capacity 5 with
  four-row files (id space 20), id 9 raises before finalize.
- `test_finalized_layout_maps_every_id_to_its_own_row` -- after it, the lookup equals one dense table
  built from the shards for *every* id, including the four the stride crashed on (4, 9, 14, 19); the
  fold still covers ids outside the table.
- `test_layout_rejects_a_split_that_does_not_add_up_to_the_id_space`,
  `test_layout_rejects_a_non_positive_shard` -- load-time errors, not warnings.
- `test_layout_accepts_the_canonical_split_without_warning`, `test_layout_warns_on_a_non_canonical_split`,
  `test_layout_warns_when_the_id_space_is_unknown`.
- `test_real_checkpoint_split_numbers` -- this checkpoint's arithmetic asserted directly: starts
  `[0, 7,812,500, 15,625,000, 23,437,500, 31,250,000]`, the first unaddressable id 39,022,512 (17,488
  of them), and the mis-located id 7,810,000 (stride says shard 1 row 1,872; the checkpoint stores it
  as row 7,810,000 of shard 0).
- `test_load_weights_finalizes_the_layout_from_the_shards` (one shard per call, i.e. chunked) and
  `test_load_weights_refuses_a_split_that_does_not_match_the_id_space` -- the loader path.

### Validation on 2x MI50 (the prompt that killed the engine, twice)

One arm, `FIX_graphed`, same recipe as the campaign's `A_flag` cell (util 0.85, TP=2 + EP, fp16, 16k
/ MBT 512, graphs on, compile cache wiped, `backed_size_oblivious`), launched clean on an idle box
after the harness fix below. Canary 38.6 t/s before, 38.7 t/s after; VRAM back to baseline at the end.

| | value |
| --- | --- |
| load | 731 s (weights 145.3 s, 24.38 GiB/rank) |
| capture | 26 s / 0.66 GiB, then 4 s / 0.26 GiB |
| KV cache | 1.07 GiB / 62,914 tokens (3.84x @16k) -- identical to pre-fix `A_flag` |
| cold first call (1 token) | 0.18 s |
| 512-token probe | rep1 21.64, rep2 25.55, rep3 25.57 t/s |
| mclk under load | 1000 MHz |

The same prompt that produced `IndexError: index out of range in self` at 16:34:19 (graphed) and
16:39:23 (eager) now completes three full 512-token generations with coherent text, and no
`IndexError` appears anywhere in the run's log window. The load log carries the new line from both
workers, which is the fix working as designed:

```
WARNING [ple_layer.py:138] PLE ngram shards are not vLLM's canonical split (5 rows, id space
39040000, stride 7808128) but [7812500, 7812500, 7812500, 7812500, 7790000] rows totalling
39040000 ids. Serving the checkpoint's own boundaries, which is correct for the rows on disk; a
re-split would restore the uniform layout.
```

The KV geometry being byte-identical to the pre-fix arm matters: the layout fix changes which rows
are *read*, not how much memory the table or the cache takes, so the #39 envelope table stays valid.

This answers WHT-1r item 4 -- the real-probe throughput is **25.6 t/s at 512 tokens**, well above the
21.6 t/s the 26-token probes reported, because a long generation amortises the prefill instead of
ending at the prompt's own EOS.

### Harness bug found on the way (and banked in the skill)

The first attempt at this validation launched while the previous run's canary was still resident:
`canary()` in the harness starts the 27B canary server, reads its number, and never stopped it, so
the tail canary from the crashed eager arm was still holding 22 GiB of card 0 sixteen minutes later.
The arm was killed 30 s in and restarted on an idle box, and `probe2.sh` now (a) stops the canary and
waits for VRAM to return to baseline, (b) refuses to launch an arm when a card is not idle, and
(c) reads the canary's t/s from the log offset captured before the run, so a hung canary cannot be
reported with the previous run's number. The recorded campaign arms are unaffected -- their canaries
exited promptly, which is what a clean box looks like.
