# gfx906 changelog

## 2026-10-08 (ai-infos v0.30.0 integration)

`gfx906/v0.30.0.x` fast-forwards the existing fork to KIntegrated's pinned
v0.30.0 reference, then fixes demonstrated attention, FP8 conversion, GPTQ
initialization, native MoE validation, and local build defects. Optimized
defaults and the explicit-V1 fused-align gate are retained. Inputs, complete
change inventories, build pins, validation results, and GPU checks still
required are recorded in [UPGRADE-0.30.0.x.md](UPGRADE-0.30.0.x.md).

## Reference history

This file records roadmap items that are complete, rejected, superseded, or
otherwise closed. Active work, deferred work, and changes that are local but
still need upstream merging remain in the roadmap files. Dates are landing or
merge dates where the repository history provides one; they are not necessarily
the date an investigation began. Since 2026-09-27 open work is tracked as
GitHub issues on `KIntegrated/vllm-gfx906-mobydick`; `ROADMAP.md` is the ordered
index, and closing an item means recording it here and deleting its roadmap line.

## 2026-10-06 (GDN-1 closed — the "red" gate was the test's indexing, and all four guards are live)

### GDN-1 — SYV-10 bounds-port test coverage · [#20](../../issues/20)

The T4-2 suite first ran 2026-10-05 and reported **40 passed, 4 failed**, all four
failures in the new `test_spec_invalid_accepted_count_zero_fills`, read at the time
as `fused_sigmoid_gating` failing to zero-fill an out-of-range accepted count.

**It does zero-fill; the test sliced the wrong axis.** Both entry points return
`(B, T, HV, V)` with `B == 1`, and the test asserted on `out[invalid]` — the
length-1 *batch* axis — so the "invalid" assertion counted the whole tensor (16384
nonzero == the four valid tokens' output) and the "valid" assertion was that same
tensor. Indexing the sequence axis (`out[0, invalid]`) turns all four params green:
**44 passed, 3 skipped** across the four deferred files.

The four paths the deliverable names are gated, proven by mutation — each guard's
bound dropped to the always-valid form and the suite re-run:

| path | bound removed | result |
|---|---|---|
| `fused_sigmoid_gating` | `i_t < stride_indices_seq` | 2 failed (`past_row`, fp32+bf16) |
| `fused_recurrent` | `i_t < stride_indices_seq` | 2 failed (`past_row`, fp32+bf16) |
| `causal_conv1d` | `num_accepted > seqlen` | `test_causal_conv1d_...` FAILED |
| `mamba_ssm` | `init_token_idx < stride_state_indices_batch` | `test_selective_state_...` FAILED |

So the T4 worry that `fused_recurrent` had "no gate" was a coverage gap, not a
missing guard: the kernel has carried the same mask since `fd6895e789`.

**Landed:** `b53c9b3f2f` — test-only (the indexing fix plus the new
`tests/kernels/mamba/test_spec_decode_bounds.py`). Retires the last T4 "shipped
without a runnable check" item.

## 2026-10-05 (QSA-FN-12 closed — the `qwen4_exp` cudagraph mode is settled by measurement)

### QSA-FN-12 — the `full → PIECEWISE` cudagraph downgrade stays · [#7](../../issues/7)

The downgrade was written while no Qwen4Exp checkpoint would load on this host, so its own gate
(`FULL_AND_PIECEWISE` + splitting list vs current PIECEWISE, interleaved A,B,A) had never been run.
`logic65/Whittle-Qwen-3.8-35B-A3B` now serves locally, so it was run: **the downgrade is not a
caution, it is the only mode that runs here.**

Asking for full graphs (`-cc.cudagraph_mode=FULL_AND_PIECEWISE`) dies at the **first** capture,
inside `vllm/v1/engine/core.py:308 → determine_available_memory()`:

```
RuntimeError: Worker failed with error 'CUDA error: operation not permitted when stream is
capturing / hipErrorStreamCaptureUnsupported
```

Two same-box controls make this a property of the model rather than of the box or the driver: the
27B canary captures FULL graphs successfully on the same HIP, and this same checkpoint captures
PIECEWISE minutes earlier. The offending op is the PLE n-gram lookup — a host-side gather with a
blocking device→host copy, which HIP refuses inside a stream capture, exactly as the downgrade's
warning has claimed since it was written.

Shipped default, measured (A1/A2 interleaved, 2.4 % drift): **25.76 / 26.38 t/s** decode, ~6k
prefill 17.6–17.8 s, prefix reuse ×21.4, KV cache 72,089 tokens at 32k, mclk 1000 — on 2× MI50
(TP=2 + `--enable-expert-parallel`, fp16, util 0.85, `--max-num-seqs 2`, block 64, MBT 512, V2
runner), canary-gated at 38–39 t/s before every arm.

**Landed:** the rewrite stays, with an escape hatch —
`VLLM_GFX906_QWEN4_EXP_ALLOW_FULL_CUDAGRAPH=1` keeps the requested mode, so the question can be
re-asked (a newer HIP, or a reworked gather that prefetches ahead of the graph) with one
environment variable instead of a patch. Off by default; when on it fails loudly in ~11 minutes
with one clear HIP error rather than silently running the other mode.
`tests/config/test_qwen4_exp_ple_splitting_ops.py` covers both sides of the hatch.

**Prefix caching (QSA-FN-10 item 5, arm C)** was lost to the load-lottery wedge #111
(`degradation.md`) and re-ran clean the same night (22:41–22:57, load 701 s): decode is unchanged
— **26.32 t/s** median (22.19/26.32/26.38) against A1 25.76 / A2 26.38, i.e. inside the interleave
drift — while the KV pool grows 72,089 → **80,099 tokens** at the *same* 0.97 GiB available and the
same VRAM (27,956/27,996 MiB), and reuse collapses ×21.4 → **×1.35** (a repeated 2.2k-token prefix
costs 3.825 s instead of 0.237 s). Prefix caching on is what selects the mamba `align` cache mode,
so the single-arm 6k-prefill reading (15.12 s vs 17.6–17.8 s) is recorded as an observation, not a
win.

Record: `DEVLOG-qwen38-flash-qsa.md` (2026-10-05), `DEAD-ENDS.md`. Refrigerated: PR #34's
`ple_prefetch.py` is the shape that could make this capturable, and it is not ours to land.

### QSA-FN-15 arm A — the generic UVA offload path cannot carry the PLE table · [#9](../../issues/9)

`--cpu-offload-gb 24 --cpu-offload-params ngram_embedding` **engages and moves nothing**: both
halves of the test fired (`'cpu_offload_gb': 24.0, 'cpu_offload_params': ['ngram_embedding']`
accepted, `Offloader set to UVAOffloader` in every worker) and yet the KV cache (**72,089 tokens**)
and the VRAM footprint (**27,956/27,996 MiB**) are byte-for-byte the shipped default's, with decode
inside the 2.4 % interleave drift (25.53 vs 25.76 t/s). The PLE table is not a GPU-resident
parameter: `MmapShardedNGramEmbedding` (`vllm/models/qwen4_exp/amd/ple_layer.py:166`) is
CPU-resident by design, backed directly by mmap'd safetensors shards that every rank maps and the
page cache shares, and `set_shard()` refuses anything but a CPU tensor — the shards "are not
parameters". The mmap layout *is* the offload, so a per-parameter offloader has nothing to move.
**Arm A closed as a dead-end**; **arm B stays open** — pinned host RAM vs page cache is the
deciding datum and it needs the reporter's 128 GB machine (this box is 46 GB).

Record: `DEVLOG-qwen38-flash-qsa.md` (2026-10-05), `DEAD-ENDS.md` (both rows),
`ROADMAP.md` (arm A retired, arm B kept).

## 2026-10-05 (WHT-1 closed — Whittle serves on 2× MI50, graphed, with the n-gram memory in host RAM)

### WHT-1 — Whittle-Qwen-3.8-35B-A3B runs locally (a second `qwen4_exp` implementation) (#36)

Whittle-Qwen-3.8-35B-A3B (Logic65's Phase-2 step-32010 conversion, bf16, 66.26 GiB over 14
shards in `/biglocal/cache/hf`) **loads and serves on 2× MI50**, and it is the n-gram-memory
experiment this item existed to run: 10.0 B of its 35.1 B parameters is a hashed n-gram table
(8 tables × 4.88 M rows × 256) that stays in **host RAM** while each rank holds 24.38 GiB of
weights — the same offload lever as QSA-FN-9/11/15, and the datum that decides whether the
Flash-Next line serves at long context on 2×32 GiB.

**Serving recipe:** TP=2 + `--enable-expert-parallel`, fp16 (no native bf16 on gfx906), util
0.85 (0.90+ wedges GPU0 on this box), `max-model-len 16384`, MBT 512, block 64, no MTP (the
checkpoint ships `mtp_num_hidden_layers: 1` with zero MTP tensors), and no `--enforce-eager`
since 2026-10-05. `-cc.dynamic_shapes_config.type=backed_size_oblivious` is still in the
winning combination — see the residue item below.

| | eager (control) | graphed |
| --- | --- | --- |
| rep 1 / rep 2, 26-token probe | 2.75 / 5.61 t/s | 4.25 / **21.15 t/s** |
| capture | `--enforce-eager` | PIECEWISE 3/3 in 27 s / 0.66 GiB (2nd round 4 s / 0.26 GiB) |
| KV cache | 2.0 GiB / 117,964 tok (7.2× @16k) | 1.07 GiB / 62,914 tok (3.84× @16k) |
| output | coherent | coherent, byte-identical |
| peak VRAM/card at init | 25.03 GiB | 25.03 GiB |

Coherent at temperature 0 end to end; canaries 37.9–38.8 t/s before and after the runs (38.7
t/s post-load). 21.15 t/s is a **26-token probe — a signpost, not a benchmark**, and the
tester's 46.8 t/s with MTP / 25.4 without are TP=4 and not comparable.

**What blocked it, and what shipped** (four load fixes and one capture fix, all in the fork):

- **Launch recipe** — the gfx906 flash-attn Triton gate (`FLASH_ATTENTION_TRITON_AMD_ENABLE`),
  fp16 (QSA needs 2-byte floats), V2 model runner.
- **A 4 GiB-per-layer index buffer.** This checkpoint carries a full-context `indexer_budget`
  (262144), which sizes a 262144-wide int32 selection buffer per attention layer — 4 GiB each
  at MBT 4096, 40 layers — and OOMed the loader at layer 7. Clamped losslessly to what a row
  can address (`block_topk 65536 → 1024`), which also brings the decode selection back under
  the kernel's measured 8192-block ceiling instead of falling back to the reference path.
- **Root-packed PLE table.** The checkpoint ships `ngram_embedding.shard_N.weight` and
  `ple_embedding.layer_multipliers` at the config root where the family nests them under
  `layers.<ple>.ple.ple_embedding.`; the loader now remaps the names onto their owning layer.
- **The MTP drafter** — absent from the checkpoint despite the config, so it serves without
  speculative decode; `mtp_num_hidden_layers: 1` with no MTP tensors is a config/checkpoint
  inconsistency, reported to the uploader.
- **The cudagraph split guard — this is what unlocked capture.** The host-resident PLE lookup
  is a *blocking* device→host copy (`pinned_ids.copy_(ngram_ids, non_blocking=False)`,
  `vllm/models/qwen4_exp/amd/ple_layer.py:1237`; deliberate, because the host gathers rows out
  of the mmap'd shards, and the async form read stale ids), and HIP refuses that inside a
  stream capture — so the op must run in an eager region between captured pieces. The fork had
  that guard, but it matched `hf_config.model_type == "qwen4_exp"` exactly while this
  checkpoint declares `qwen4_exp_text`, so the op stayed inside the graph and engine init died
  at `Capturing CUDA graphs (PIECEWISE): 0/3` with `hipErrorStreamCaptureUnsupported`.
  Fixed family-wide rather than by widening one literal: the family's spellings now live in one
  place (`vllm/models/qwen4_exp/config.py`, derived from the config classes that declare them,
  plus the MTP rewrite `qwen4_exp_mtp`) behind `is_qwen4_exp_model_type` /
  `is_qwen4_exp_config`, and both matching sites use it — the compilation guard and
  `SpeculativeConfig`'s MTP rewrite, which had carried its own literal (the third such list,
  which is why the drift existed).
- **A self-review narrowed that widening before merge**, so it costs nothing it need not:
  the guard now also requires the checkpoint to actually use the memory
  (`uses_ngram_embedding`, the model state's own `bool(ple_layer_ids)` gate, which also
  excludes the MTP drafter and any PLE-less family member) **and** the platform to be ROCm
  (the op is AMD-only — `vllm::qwen4_exp_amd_ple_ngram_embedding` is registered in
  `amd/ple_layer.py`, while the CUDA path prefetches through streams over a device-side table
  and registers no such op), via `needs_ple_ngram_split(hf_config, is_rocm=…)`. A family
  config with no PLE layers, the drafter, and every CUDA model keep the capture mode they
  asked for. Regression cover: `tests/config/test_qwen4_exp_ple_splitting_ops.py` (the guard,
  both spellings, the negatives, no duplicate append) and
  `tests/config/test_qwen4_exp_family_match.py` (the predicate, the derived tuple vs inherited
  `model_type`, odd `architectures` inputs).

**Left open at closure — now #39 (WHT-1r, `ROADMAP.md`):** whether `backed_size_oblivious` is
still required once the split applies (cheap bisect); the largest `max_model_len`/MBT that
fits **with graphs on** (the graph pool takes KV from 2.0 to 1.07 GiB, so the envelope has to
be re-measured); the AMD warmup gap
(`model_executor/warmup/qwen4_exp_qsa_warmup.py` looks up `vllm.models.qwen4_exp.nvidia.*` in
`sys.modules` and returns early on gfx906, so the QSA/indexer warmup never runs here); and a
throughput number at a realistic probe length.

**The Q8_0 GGUF half of this item is parked** — `REFRIGERATOR.md`: the fork has no in-tree
GGUF loader (upstream moved GGUF to the out-of-tree `vllm-gguf-plugin`) and llama.cpp is not
installed on mi50-01. The file is retained with its sha256
(`fb13eb28…c52b1`, 37,828,807,904 B).

**One figure is an inference, not a measurement:** "~17 GiB of the n-gram table stays in host
RAM" is 66.26 GiB checkpoint − 2 × 24.38 GiB resident (the predicted bf16 delta was ≈ 18.6
GiB). Nothing measured host memory.

Records: `DEVLOG-wht1-whittle-onboarding.md`, `DEVLOG-wht1-whittle-load.md`,
`DEVLOG-wht1-qsa-topk.md`, `DEVLOG-wht1-graph-capture.md` (which carries the two concurrent
accounts folded into one, the ruled-out tables and the self-review). Commits `e3b6542393`,
`b6873ecc25`, `df22877eee`, `d1a647a78a` on `gfx906/wht1-graph-capture`, fast-forwarded into
the `gfx906/v0.30.0` line.

### QSA-PLE-1 (#40) — the PLE n-gram lookup used the runtime's stride, not the checkpoint's split

Whittle's host-resident n-gram table killed the engine on any longer generation:
`IndexError: index out of range in self` at `amd/ple_layer.py:202` from the lookup op at `:1238`,
graphed (16:34:19) and eager (16:39:23) alike. Two row layouts were in play. The runtime computed
`shard_row_capacity = ceil(39,040,640 / 5) = 7,808,128` from **its own** prime-based vocab layout
before any weight was loaded, while the checkpoint is a plain contiguous split — 7,812,500 × 4 +
7,790,000 = 39,040,000 rows, matching its own `ngram_heads_offsets` / `ngram_heads_vocab_sizes`
(8 × 4,880,000). Every id >= 7,808,128 therefore read the wrong row silently (id 7,810,000 →
runtime (shard 1, row 1,872) vs the checkpoint's row 7,810,000 of shard 0), and the last 17,488 ids
indexed off the end of shard 4 and killed EngineCore. That window is 0.36 % of a token's draws —
one per token, since only the last head reaches it — so short probes usually survive (~83 % for a
26-token prompt, which is why dozens passed all day), while the counting prompt sits inside the
window, which is why both decode-length arms died at the same frame.

The boundaries now come from the shards that were loaded and the id space from the checkpoint's own
layout buffers, required to agree: `build_ple_shard_layout` (new), recorded by
`MmapShardedNGramEmbedding.finalize_shard_layout`, called at the end of
`Qwen4ExpNGramEmbedding.load_weights` so a chunked delivery completes on the last chunk;
`forward` masks by `[start, start + rows)` per shard instead of `id // shard_row_capacity` and folds
against the loaded rows (39,040,000 here, not 39,040,640). A split that does not add up to the id
space is a load-time `ValueError`; a split that is merely non-canonical is served and **warned
about** — and this checkpoint is non-canonical, which the load log now says out loud. The nvidia path
already validates shard shapes (`nvidia/ngram_embedding.py:896-920`) and would have refused this
checkpoint; the AMD port had dropped that check, which is why it loaded silently.

**Validated on 2× MI50** (util 0.85, TP=2 + EP, fp16, 16k / MBT 512, graphs on, compile cache
wiped, single arm `FIX_graphed`): the 512-token prompt that killed EngineCore twice now completes —
**rep1 21.64, rep2 25.55, rep3 25.57 t/s** at mclk 1000 MHz, coherent, KV cache unchanged at
1.07 GiB / 62,914 tokens (3.84× @16k), load 731 s, capture 26 s / 0.66 GiB, and both workers log the
layout line (`id space 39040000 … [7812500, 7812500, 7812500, 7812500, 7790000]`). Canaries 38.6 t/s
before and 38.7 t/s after; VRAM back to baseline. This also answers WHT-1r item 4: **25.6 t/s at 512
tokens**, against the 21.6 t/s of the 26-token probes.

12 new tests in `tests/models/qwen4_exp/test_ple_mmap_shards.py` (34 in the file; `test_ple.py` 50,
`test_ple_table_remap_amd.py` 9, ruff check + format clean), including the checkpoint's own
arithmetic without allocating a 39 M-row table. Devlog: `DEVLOG-wht1r-residue.md`. Follow-up for the
uploader, not for this fork: a re-split to vLLM's canonical layout would silence the warning.

### WHT-1r — WHT-1 residue: graphed envelope, dynamic-shapes bisect, AMD warmup, real throughput (#39)

Closed with all four items answered, one of them by finding and fixing a release-blocking bug on the
way (QSA-PLE-1 / #40, above).

- **The dynamic-shapes flag stays.** `-cc.dynamic_shapes_config.type=backed_size_oblivious` is still
  required even with the PLE split guard firing: the control arm (`B_default`, split in place, flag
  absent, compile cache wiped) died 341 s in inside
  `determine_available_memory → model_runner.profile_run` with
  `ConstraintViolationError: Constraints violated (L['query_start_loc'].size()[0])`. Split and
  dynamic shapes fix different failures — the split is what lets graph capture happen at all, the
  dynamic-shapes mode is what lets the memory profile run.
- **The graphed envelope at util 0.85 is 32768 / MBT 512** — KV 0.97 GiB, 72,089 tokens, 2.20×, the
  only cell that covers two full 32k streams. `max_model_len` is nearly free at MBT 512
  (16k → 32k → 64k: 62,914 → 72,089 → 68,344 tokens); **MBT is what costs KV capacity** (at 16k:
  62,914 → 48,496 tokens going 512 → 2048); 64k at MBT 512 serves one stream (1.04×).
- **Graphs are worth 3.9×** — 5.61 → 21.6 t/s at the same 26-token probe, reproduced byte-for-byte on
  a wiped cache four hours later, and not a clock artifact (eager 800 MHz, graphed 1000 MHz under
  load). Graphed init VRAM is *lower* (27.9 vs 29.2 GiB/card) and its KV cache is smaller (1.07 vs
  2.0 GiB): the graph pool comes out of the pre-capture budget.
- **The AMD QSA/PLE warmup gap is real, and is recorded as harmless** (devlog): `kernel_warmup.py:201`
  calls the QSA warmup on every platform, but it resolves `vllm.models.qwen4_exp.nvidia.*`, which ROCm
  never imports, so it returns on its second statement with no log line; `amd/` has no `warmup_*`
  helpers at all. Measured cost of leaving it: cold 1-token call 0.16–0.18 s and rep 1 at 21.64 t/s
  against 25.55/25.57 steady — first-request latency, not throughput. Reopen if an AMD kernel ever
  lands with a multi-second first-call compile.
- **Real throughput, once #40 was fixed: 25.6 t/s at 512 tokens** (21.64 / 25.55 / 25.57), which is
  why the campaign's ~21.6 t/s figures are a floor — a 26-token probe ends at the prompt's own EOS
  and mostly measures prefill plus first-call warmup.

**Method notes worth keeping:** every arm was canary-gated, ran on a wiped compile cache (the
dynamic-shapes mode is a compile-time change), and was analysed only through its own byte-offset log
window; the 26-token probes everyone had been quoting were not measurements of the server at all. Two
harness bugs were found and fixed mid-campaign (a driver that stopped itself because its server unit
shared its name, and a results emitter that lost failed arms) and one afterwards (a `canary()` helper
that never stopped its own 27B server, so a later run loaded onto a card holding 22 GiB). Devlog:
`DEVLOG-wht1r-residue.md`.

## 2026-09-28 (P2P-1 closed NO-GO — the custom all-reduce "win" was measured on a broken reduction)

### P2P-1 — re-test vLLM's custom all-reduce under live PCIe P2P (#3) — **NO-GO, not a win**

Opened to re-measure the inherited `--disable-custom-all-reduce` now that PCIe P2P is
live. The throughput A/B produced what looked like the largest single result on the 0.30
line (+17.1 % B=1, +6.4 % B=4 concurrent decode; TP=2, 27B-AWQ-INT4, same boot, interleaved
A-B-A, mclk 1000). **All of those figures are void: the treatment arm was numerically
broken.**

- **Correctness gate (runs 5/6, `/local/tmp/p2p1/r6/`):** arm A (`NCCL_P2P_DISABLE=1`,
  PYNCCL) coherent and identical across repeats (2/2). Arms B (CUSTOM, P2P off) and C
  (CUSTOM, P2P live) produced degenerate loops (`post post post…`, `parallel parallel
  parallel…`) **4/4**, with top-1 logprobs −2.4…−5.8 against −0.1…−1.8 for the coherent arm
  and 1/96 token agreement. It reproduces **with and without** P2P, so the failure is the
  implementation, not a P2P interaction.
- **Why:** upstream `CustomAllreduce` is MI300/XGMI territory. The platform gate says so
  ("We only enable custom allreduce for MI300 series", `gfx94`/`gfx95`), and on ROCm the
  peer-write-ordering check is skipped (`custom_all_reduce.py`: `same_node and not is_rocm
  and not _can_p2p`, comment "p2p is always enabled between XGMI connected GPUs"). These two
  cards are PCIe-only, so nothing validates the ordering the flag-based spin synchronisation
  depends on.
- **Action:** the `VLLM_GFX906_CUSTOM_AR` override and the amdsmi `is_fully_connected` guard
  were reverted; upstream's MI300-only gate is restored and `vllm/platforms/rocm.py` is
  byte-identical to the line. RCCL gains nothing from P2P either: PYNCCL + P2P live = PYNCCL
  + P2P off (33.92 t/s both).
- **Method lesson (now enforced for every numerics-touching A/B):** `stop_agreement` /
  `rep_frac8` do **not** catch a broken distribution — the garbage arm scored a *better*
  repetition fraction (0.0392) than the coherent one (0.32–0.36). Gate on coherent text +
  sane top-1 logprobs from run 1.
- Records: `DEVLOG-tp2-dense.md` §S10 (verdict-first, figures marked VOID), `DEAD-ENDS.md`,
  the issue #3 correction comment, and the `mi50-vllm-serving-ab` skill.

### TRITON-1i — a triton-3.8.0-adopting docker image (#26)

*Record provenance:* this issue was closed **COMPLETED** on 2026-09-28 by `unverbraucht` and its
`ROADMAP.md` line was left in place until 2026-10-05, so the entry below is taken from that closing
comment rather than from a landing commit in this repository. The image identity was re-verified at
record time (`docker buildx imagetools inspect`, 2026-10-05).

- **Image** `unverbraucht/vllm-gfx906:0.30.0-1c4d1065ad`, digest
  `sha256:cccaa5925cd03acc930548dd7f8208539621685b1c97f75df54fef1a601708f1`, built from
  `gfx906/v0.30.0` @ `1c4d1065ad` (branch tag `gfx906/v0.30.0-final`). The floating
  `0.30.0-rocm-7.14` tag resolves to the **same** digest — confirmed at record time. The published
  `0.29.0-e730ef4066` image (fork Triton) is left untouched, and the earlier 3.6.0-based 0.30.0
  image is superseded by this build. The new tag shape the item asked for is `0.30.0-<commit>`.
- **Triton**: stock upstream `triton-lang/triton` **v3.8.0**, built from source with
  `TRITON_CODEGEN_BACKENDS=amd` and
  `TRITON_APPEND_CMAKE_ARGS=-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON`. Verified in-image as `triton
  3.8.0`; the fork `v3.6.0+gfx906` is rollback-only, with its pin and rationale in
  `vllm-v2/preset.0.30.0-rocm-7.14-kintegrated.sh`.
- **Runtime validation** (Qwen3.8-27B-AWQ-INT4, TP=2, 256k context, ngram n=5, batched 4096):
  `/health` 200; `vllm 0.30.0` on `torch 2.13.0+gfx906`; the custom `GFX906_FA` backend registered;
  KV cache 411,940 tokens (1.57× concurrency for 262,144-token requests); ~28 GB VRAM/GPU;
  generations correct (greedy "capital of France" → `Paris`, plus a 200-token structured
  generation).
- **Two build-side fixes**, both in `ML-gfx906`: `ARG MAX_JOBS` was declared before `FROM` and so
  never reached the `RUN` step (falling back to `nproc` = 16 and OOM-killing the host twice), and
  the branch's annotated `gfx906/vX.Y.Z-final` tag broke setuptools-scm version parsing. See
  `ML-gfx906/AGENTS.md` ("Build memory: the build can OOM-kill the host").

Roadmap line retired with this record; the underlying reconnaissance remains in
`docs/gfx906/RECON-triton-1.md` and the 2026-09-15 TRITON-1 entries below.

## 2026-09-27 (roadmap migration — closed items retired from ROADMAP.md)

The roadmap was pruned to open work only. This section banks the closed items
that had no changelog record yet (the rest were already recorded by date above)
and moves parked work to `REFRIGERATOR.md`. Open items now live as issues #3–#33.

### 0.30.0 release prep

- **REL30-1 — `moe_gemm_q4_kernel_gfx906<1,2>` page fault during V2 FULL-graph
  capture (FIXED; the release ships on the V1 pin).** The merged 0.30 base faulted
  reproducibly under the default V2 runner during FULL-decode capture
  (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`; 916 journal `no-retry page
  fault` lines in one window). Root cause: the fork's **fused M=1 align**
  (`torch.ops._rocm_C.moe_align_block_size_m1_gfx906`) leaves its output buffers
  **uninitialised** under the V2 compiled/captured path — instrumented ids held
  float bit patterns (e.g. `1067139072`) instead of `[0,256)`/`-1`, so
  `b_q_weight + expert_id * stride` became a wild pointer. Ruled out: kernel
  source, the merge's CMake change, and eager PPL under V2 (15.9840, 0 misses). A
  `register_fake` for the align op did **not** fix it (reverted). Fix:
  `_use_fused_align_m1()` now also requires V1 to be explicitly selected, so the
  fused align is skipped on V2 (fail-safe, including the auto case). **Measured:**
  V2 default `rc=0`, 0 faults, **58.90 t/s** (58.904/58.976/58.800/58.880); V1
  unchanged at **60.42 t/s**. V2 therefore costs ~2.5 % on this workload and the
  recommended serving config stays V1. Older-build workaround:
  `VLLM_GFX906_ALIGN_M1=0` (validated). Record: `degradation.md` #110,
  `degradation_details.md` 2026-09-26.
- **GLM53-QGEMM-1** and **QSA-FN-14** closed in the 2026-09-24 and 2026-09-26
  entries above.

### Attention / FA

- **VIT-2 — head_dim-96 instantiation for the ViT (DONE, DEFAULT ON).** The
  launcher instantiates 96 and the pad map is now `(64, 96, 128, 256)`; 72/80 pad
  onto 96, an exact 96 runs natively. Gates (same-boot A-B-A, mclk 1000): image
  TTFT @1024² pad128 **5.151** / pad96 **5.080** / pad128 **5.155** s =
  **−1.46 %** (order control +0.08 %, distributions disjoint); Phi-3-mini
  (head_dim 96) **+0.69 %**. Two findings: the inherited `(96,96)` tile row was
  wrong for this Q8 kernel (`nbatch_K=48` not a multiple of 32 scored only 64 of
  96 dims; now `nbatch_K=96` + `static_assert`), and the −5 % estimate was a cost
  model over-prediction (ViT attention is ~19 % of TTFT). `GFX906_FA_PAD96=0`
  rolls back. Record: `DEVLOG-fa-d96.md`.
- **FA-COVER-1 — CUSTOM-FA coverage census + fallback guard (RESOLVED).**
  `tools/fa_coverage.py` + `DEVLOG-fa-coverage.md` mapped every rejection class —
  text fallbacks were sinks (72 rows), `head_size` (61), encoder (36), non-causal
  (18). The guard (`_guard_gfx906_fa_fallback`, `VLLM_GFX906_FA_STRICT`) emits a
  loud once-per-engine warning. Head-size padding became the default: Phi-3-mini
  (D=96) went from silent ROCM_ATTN to CUSTOM — identical top-5, PPL within
  0.11 %, **+27 % decode** (36.41 vs 28.62 t/s, A-B-A), FA suite 101 passed.
  Fixes: the full-attention KV-spec branch routes through `customize_spec`, and
  the backend widens both halves of the fused row (`get_kv_cache_shape` vs spec).
  `GFX906_FA_PAD=0` is the kill switch. Remaining classes → roadmap FA-COVER-2.
- **FA-NONCAUSAL Stage 1 (SHIPPED, DEFAULT ON).** A non-causal decoder-shaped
  batch now runs on CUSTOM FA (`supports_non_causal()` True; the impl suppresses
  `q_abs_offset` and `window` for that batch → full bidirectional).
  `GFX906_FA_NO_NONCAUSAL=1` rolls back. Gate: Muse-Glimmer + the official DFlash
  assistant, TP=2, k=7, graphs on — drafter capture `dflash CUDA graphs (FULL)
  2/2`, mean acceptance 3.12/3.18 (ROCM_ATTN+eager 2.95), decode **43.1 t/s vs
  30.5 eager / 27.1 non-spec**; order-controlled ≥ **+29 %** (best +41 %).
  Stage 2 (symmetric ±window in the kernel) is refrigerated — the assistant did
  not need it. Record: `DEVLOG-fa-noncausal.md`.
- **MBT-1 / FIX-H2 — multi-batch prefill O(live-context) tax root-caused and
  fixed.** E1 was chunk-invariant (refuting the per-step-repeated model); the
  owner was the **kv_max pad-tile expansion** (per-prefill-token × live-context
  work). The clamp was validated on boot Y9: **120k×B4 wall 75.3 → 44.8 min
  (−41 %), prefill agg 108.6 → 182.9 t/s (+68 %)**, outputs fingerprint-identical,
  decode/spec unaffected (`ttft-prefill-stall.md` §13.16/§13.16.1). Residual
  own-context streaming → roadmap MBT-1r.
- **G1 — decode-graph per-node replay-cost probe (DONE, hypothesis killed).**
  ~**1.2 µs/node TP=1, ~1.1 µs/node TP=2**, linear over N ∈ {0,16,32,64} — an
  order of magnitude below the ~10 µs needed to own the 1.55 ms/step. 16–32
  extra nodes ≈ 0.02–0.04 ms/step (~2 %). Future adds-nodes-per-step proposals
  now carry a citable budget of ~1 µs/node. Record:
  `DEVLOG-fa-legacy0-b1-decode.md` (G1 addendum).

### MoE / kernels

- **C3 — fold the two MoE zeroings: NO-GO (measured, not merged).** Phase A
  folded `w1_out.zero_()` into the M=1 align kernel, bit-correct (78/78 + 67/67)
  and firing (~37 layers/step), but the FULL_DECODE_ONLY serving A/B was a wash
  (median 85.42 on vs 85.62 off t/s): `w1_out` is only 8 KB here, so ~40 tiny
  memsets save ~tens of µs/step. Record: `DEVLOG-moe-c3-zeroing-fold.md`.
- **C8 — expert-weight residency measurement (DONE).** Combined active W4 set =
  12.47 MB > 8 MB L2/TCC (not fully resident), but the production gemm1 `<1,4>`
  M=1 kernel reaches only ~195 GB/s ≈ **24 % of the HBM floor** — the binding
  constraint at M=1 is **latency/occupancy, not bandwidth**, which set C2's
  target. Record: `DEVLOG-moe-residency.md`.
- **N3 — GDN state-bookkeeping copies (CLOSED, no code change).** Copy-class op
  invocations per decode step drop **~214 → ~57 (−73 %)** under graph serving;
  the eager ~180 µs/step was CPU **launch overhead** on 192-B `[3,1,32]` copies,
  absorbed by CUDA-graph capture. Residual is well under 60 µs/step. Record:
  `DEVLOG-gdn-n3-state-copies.md`.
- **N1 — quiet the expected AutoAWQMoEMarlin fallback (SHIPPED).** On gfx906 the
  fallback to the custom WNA16 path is intentional; `get_quant_method` now emits
  one `info_once` line per process instead of a per-layer warning. Gate:
  `tests/quantization/test_auto_awq_gfx906_fallback.py` (fails if reverted).
- **DE-1 — dead-end register-spill / compiler-structural audit (DONE).** **Zero**
  dead-ends failed from register spills or measurable pressure —
  `vgpr_spill_count = 0` on every in-tree HIP-kernel dead-end (VGPRs 12–93).
  FA V2 = 16 VGPR vs shipped V1's 12 (both spill-free → the 7× serving loss is
  grid-shape/scheduler); gemm1 V1 single-wave was structural by construction.
  13 rows annotated `FULLY DEAD` in `DEAD-ENDS.md`; no branch opened. Record:
  `DEAD-ENDS-AUDIT.md`.

### Spec decode

- **MTP-1a (DONE) and MTP-1b-0 (CLOSED).** Crossover pinned at 32k–64k pp (n=3,
  cold prefill), and the FA `kv_split` clamp for k>1 replaced with a byte budget
  (`GFX906_FA_KVSPLIT_MAX_BYTES`, default 512 MiB). With the clamp fixed, k=2
  beats both the clamped baseline and k=1 at every long-context point — 65 536
  **37.95** (vs 15.95 clamped / 31.6 k=1), 98 304 **29.88**, 122 880 **25.70**
  (vs 9.18 / 22.1) — so the "crossover" was clamp-specific. Regression test
  `test_forward_sq_multi_kv_split_vs_fp32_ref` (7 cases). Record:
  `DEVLOG-mtp1.md`.
- **MTP-1b candidate cross-check (same day).** The external-repo candidate
  list (SYV/CAT/J2G) was re-checked against `DEAD-ENDS.md` and the spec-decode /
  FA dev logs. Two candidates are covered by prior work and removed from the open
  list: **SYV-2** (our own n-gram/prompt-lookup drafter probe was a 0.68×
  dead-end, L3 closed the CPU-proposer cost, GPU n-gram was rejected on draft
  quality, and the verify-extension form is SYV-12) and **SYV-9** (the custom FA
  already runs int8 Q8 K with `v_dot4_i32_i8` at full rate; the Q-side Q8_1 path
  is the rejected C6; the remaining format upside is M6 Part C, refrigerated).
  Corrections to the rest: **CAT-2**'s GQA head-packing half already shipped
  (only head-dim Split-D remains), **J2G-5**'s tuned-config mechanism already
  ships in-tree (`E=256,N=128,device_name=AMD_GFX906,dtype=int4_w4a16.json`),
  and **CAT-3/SYV-11** are gated by the custom FA's fp16-only
  `supported_kv_cache_dtypes` (the original list did not note this). CAT-4 is
  not applicable (wide aligned loads are already our latency-hiding rule).
  Detail: issue #18.
- **SYV-4 — sort-free small-k top-k/top-p sampler (MERGED).** One `torch.topk(k)`
  replaces the full-vocab sort when all rows' k ≤ 64 and B < 8; opt-out
  `VLLM_GFX906_SORT_FREE_SMALL_K=0`. GPU bench 1.17× @B=1 k=64, **6.55× @B=4
  mixed**; e2e B=1 +1.4 %, **B=4 +3.4 %** on a sampling workload.
- **SYV-7 — hybrid-model prefix caching (DONE, nothing to port).** The flag is
  already ON by default and the model config auto-promotes mamba cache mode to
  `align`; verified working with MTP. (`SYV-7b` mamba block size → roadmap.)
- **SYV-10 — GDN spec-decode bounds checks (PORTED).** Upstream #50021 applied
  verbatim; the fork carried the pre-PR unmasked `i_t = num_accepted − 1` load in
  all four kernels. No runnable GPU test on ROCm (upstream is CUDA-gated) →
  **GDN-1 test debt**, closed 2026-10-06 (this file).
- **SYV-13 — mamba/GDN chunked-prefill align fixes (CLOSED N/A).** Both patch
  parts diffed: the V1 `src_col` path already implements the same guards
  CPU-side; `chunk_o.py` is the faithful pre-patch upstream state and no NaN has
  been observed here.
- **J2G-1 — persistent all-reduce env knobs (DONE, default ON).**
  `NCCL_ALGO=Tree` + `NCCL_PROTO=LL` measured **+2.77 % @120k / +4.27 % @64k**
  greedy TP=2 (n=5/point, no wedges); channel pinning adds nothing. Applied
  default-on in `run_server.sh` (standard env vars, no code port).
- **FD-1 — MTP fused-draft path (EXECUTED: NEUTRAL, stack-confounded).** FIX arm
  2377.6 s vs non-FD serving 2447.8/2464.9 s at 4×122 880 (offline vs serving
  control). The flag's only reader (A3's opt-in) was stripped 2026-09-13, so the
  arm must not be re-queued as-is; revival needs the archive branch and a
  same-stack re-gate.

### Onboarding

- **NH-5 — Nemotron topk chain (SHIPPED).** Single-group degenerate fast path in
  `grouped_topk` + the (128,6) fused align: 3 kernels/layer removed. Serving
  A–B–A (boot O) **106.8 → 114.6 → 107.8 t/s = +7.3–7.8 %**; fast path bit-equal
  (19/19, incl. ties), align bit-equal (51/51), PPL 27.05 vs 27.00.
  `VLLM_GFX906_TOPK_SINGLE_GROUP` default ON. Record: `DEVLOG-nemotron-h.md`.

### Infra / startup

- **S1 — startup graph/inductor speed (COMPLETE).** The 234 s warm engine-init
  bottleneck was **not** compile/capture: the 27B checkpoint is multimodal and
  every startup pushed a max-feature-size dummy image through the ViT in
  `profile_run()` (~213 s). Fix = stock `--language-model-only` (now the arm
  default). **Warm engine init 233.85 s → 14.54 s (16×)**; dev boot is now
  weights-load-bound. Record: `DEVLOG-s1-startup.md`.
- **HK-1 — drop the legacy `~/env-rocm-7.14-gfx906.sh` sourcing (recipes DONE).**
  Single ROCm toolchain now; confirmed on boot N (TP=2 boots, 74/74 suite, FA
  micro-bench) and re-verified under `env -u ROCM_PATH -u LD_LIBRARY_PATH`.
  Removed from `running.md`, `docs/gfx906/README.md` and the mem-attribution
  skill; the `/local/git/AGENTS.md` + `canary.sh` lines remain → roadmap HK-1.
- **QSA-FN-3 — tiny `qwen4_exp` rig (SHIPPED).** `_qsa_tiny_model.py` +
  `_serve_qsa_tiny_gfx906.sh` keep the architecture identical (all four layer
  types, PLE/ngram, hyperconnection, QSA + sparse attention, MTP; QSA fraction
  1/4) with random weights on one MI50. Baseline fp16/V2: prefill
  1321/2641/3961 → 22/42/48 ms; decode 563/1023/1994 t/s at B=1/2/4. Corrected
  gate: every path *executes*; quality stays with the tester. Produced the
  V1-cannot-serve, V2-MAMBA-1 and bf16-broken findings.
- **QSA-FN-6 — gfx906 int8-`tl.dot` fault in the QSA kernel (CLOSED, dropped).**
  The serving config contains no int8, and int8-QK is 2.4× *slower* than fp16 at
  the profiles where it runs, so the IMA cost nothing. Mechanism remains
  unexplained (profile-dependent codegen at `block_n=64, num_splits=1,
  num_warps=2`; `num_warps=8` clears it); repro at
  `/local/tmp/qsaprobe/logs/ima_repro_G12.log`.

## 2026-09-26 (QSA-FN-14 gate: MiniMax-M3 `amd/ops` + `rocm_aiter_mla_sparse` fp16 paths validated without the model)

The 0.30.0 merge took upstream's gfx950-gated MiniMax-M3 selectors and rewrote
the `index_topk` kernels, leaving the gfx906 paths with **no model gate** - the
model does not fit the 32 GB MI50s. Running the suites that the merge never ran
here turned them red, and three real defects came out:

- **`AttentionConfig` never canonicalized `indexer_kv_dtype`.** The fork's
  Literal accepts the `fp16`/`fp32` spellings, but nothing mapped them, so
  `indexer_kv_dtype="fp16"` reached the indexer cache as `"fp16"` and missed the
  `"float16" -> torch.float16` map. Normalized in `__post_init__`.
- **The merged `amd/ops`/`common/ops` decode index-score kernels cast only `k`,
  not `q`,** before `tl.dot` - the gfx906 fp32-KV path failed Triton's same-dtype
  assertion (`Both operands must be same dtype. Got fp16 and fp32`). The `q` cast
  is now hoisted before the block loop (Triton loop-carried types must be
  stable), so it cannot live inside the loop.
- **The re-applied gfx906 launch kwargs duplicated a keyword.**
  `**score_kwargs, **_index_score_launch_kwargs()` raises `TypeError` whenever
  `score_kwargs` already carries `num_warps` (the multi-head spec-decode path);
  merged to `**{**score_kwargs, **_index_score_launch_kwargs()}`.

Gates now in tree (all model-free): the `_forward_mla` `VLLM_ROCM_MLA_SPARSE_FP16`
reference-Torch early-return and the gfx906 routing (`MiniMaxM3SparseBackend` +
Triton impl, `common.ops` sparse attn rather than CDNA/AITER) are pinned by unit
tests, and the `amd/ops` kernels are gated numerically by
`tests/kernels/attention/test_minimax_m3.py`. A real MiniMax-M3-AWQ weight load
remains untested here (32 GB cards).

**Results:** `test_minimax_m3.py` **122 passed / 13 skipped** (was 9 failing);
`tests/models/minimax_m3/` + `test_rocm_glm5next_sparse.py` +
`test_rocm_aiter_mla_sparse_metadata_sync.py` **32 passed / 3 skipped**; combined
**154 passed / 16 skipped**; `tests/test_config.py -k "attention or indexer or
cudagraph"` 43 passed; ruff F/I/format clean. The stale
`test_fp32_kv_config.py` (it imported a helper that never existed, so it had
never collected) now runs - its two pre-merge-signature tests are skipped with
reasons. Commit `98e96532b7`.

**VERDICT:** SHIPPED (code + gates) / **GATE:**
`tests/kernels/attention/test_minimax_m3.py` + the routing and
`_forward_mla`-fp16 unit tests.

### Release-notes cross-check (same day)

Read the v0.30.0 release notes' "Breaking Changes & Deprecations" against the
fork. One more silent regression, now fixed, and several items that do **not**
apply:

- **`CommonAttentionMetadata.seq_lens_cpu` / `num_computed_tokens_cpu` removed
  (#55353).** The merge had re-applied the fork's vllm#47042 chunked-continuation
  guard with `getattr(common_attn_metadata, "seq_lens_cpu", None)` - now always
  `None`, so the guard was **silently inactive**. Re-derived from the
  `seq_lens_cpu_upper_bound` replacement (host upper bound; conservative).
  `rocm_aiter_mla_sparse.py`; the MLA-sparse suites stay green.
- The `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` removal is **deliberate upstream**;
  the fork's `arg_utils.py` still reads it via `get_from_deprecated_env_if_set`,
  which is why the merge's env-table loss crashed. Kept the fork's read + env
  registration (decision recorded; the alternative is aligning with upstream's
  removal).
- **Not applicable** (checked): `VLLM_MM_HASHER_ALGORITHM` (unused anywhere),
  `use_fp4_indexer_cache` (field already gone), DCP declaration (#55780; no DCP
  used, `AttentionImpl.supports_dcp` defaults `False`), YaRN alignment (#56446;
  no YaRN models/recipes - `--max-model-len` is explicit), scale-out endpoints
  (#54579; unused), `all` Mamba cache mode deprecation (#55041; the fork uses
  `align`), CUDA_VISIBLE_DEVICES-on-ROCm removal (custom all-reduce is disabled
  on ROCm anyway), and `requirements/build/rocm.txt` still carries the fork's pin
  (the 0.29 toolchain lesson held).
- **Confirms earlier findings:** #54809 (GPTQ act-order) and #55353 (env
  removals) are the listed breaking changes; the ROCm section's MiniMax-M3
  indexer/top-k work (#54682/#52664/#55235/#56170), Kimi-K3 mixed-batch KDA
  (#56159) and the AITER sparse-MLA sinks (#54404) are the upstream changes whose
  merge forced the QSA-FN-14 re-ports.
- **Fork-relevant, no action yet (post-0.30 candidates):** Nemotron-H MTP with a
  separate/quantized lm_head (#54574) and the latent-MoE TP>1 all-reduce skip
  (~13 % decode, #52301); persistent top-k fallback on low-shared-memory GPUs
  (#54110, relevant to MI50's small LDS); "unavailable piecewise CUDA graphs now
  raise instead of garbling" (#54782); W4A16 packed zero-points (-26 % TPOT on
  Gemma-4 AWQ, #54965); and "GLM-5.2 kept on MRV1 on ROCm" (#53155), which
  supports the fork's V1 pin.

## 2026-09-24 (UP-3 done — upstream v0.30.0 merged onto the fork line)

`gfx906/v0.30.0` started from `main`/`gfx906/v0.29.0` (`524ac6f2d6`) and merged
upstream `v0.30.0` (`ced6857afa`) — merge commit `8893a50e54`, 2017 files,
**31 conflicted files** exactly as scoped in `MERGE-0.30.0-review.md`. Full
per-file resolution record: the merge commit body. Machine validation: all
resolved modules import, `tests/quantization/test_moe_wna16.py` 47 passed,
`tests/test_config.py` upstream suites green (only the no-CUDA-GPU engram and
network-dependent HF cases fail, both environmental).

The one non-negotiable upstream change was **#54809 (GPTQ group/dynamic
activation ordering removed)**: it rewrites `q_gemm.cu`, `marlin*`, CPU WNA16
and the RDNA3 kernels, so keeping the fork's `b_g_idx` API would have meant
forking the whole quant stack. Adopted the removal; the gfx906 M=1 4-bit
max-ilp dispatch and its twin were re-ported onto the new kernel signature (no
`b_q_perm`) and stay default-ON. Consequence: **gfx906 GPTQ act-order
checkpoints are no longer supported** (upstream's new supported set). The
gfx906 AWQ path (`auto_awq.py`) was updated to the new call signatures.

Post-merge consistency fixes (not conflict hunks; found by import/pytest):
`_process_weights_gfx906` returned the old 14-tuple, `auto_awq` used the old
`gptq_gemm`/`gptq_shuffle` arity, `envs.py` had lost
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` (the fork's `arg_utils.py` still reads
it), `use_v2_model_runner`'s fork cases now use defensive `getattr` for
upstream's `SimpleNamespace` tests, and the wna16 test dropped a `desc_act=True`
config the new validator rejects. Also noted for the next sweep:
`tests/test_config.py` defines `test_dflash2_draft_forces_v2_model_runner` twice
(pre-existing on `main`, F811).

**VERDICT:** validated on the model paths the merge touched; OPEN follow-ups in
[`ROADMAP.md`](ROADMAP.md) UP-3 for the areas without a gate (below).

**Functional validation (2026-09-24, after a full gfx906 extension rebuild —
the installed `.so` still carried the pre-merge `gptq_gemm(..., b_g_idx)` /
`gptq_shuffle(..., q_perm)` schemas and would have failed on first use):**

- **Qwen3.8-27B-AWQ-INT4 (compressed-tensors pack-quantized → `ExllamaLinearKernel`,
  the path #54809 rewrote): PPL 10.5472** (359 tokens, **0 top-20 misses**), in the
  recorded band (10.5516 / 10.5472). This exercises `ops.gptq_shuffle` +
  `ops.gptq_gemm` end-to-end on the new 2-arg / 7-arg signatures.
- **Qwen3.5-35B-A3B-AWQ (MoE, the flagship): PPL 15.9840** (359 tokens, **0 top-20
  misses**), inside the recorded C4 flip pair (15.9361 ON / 16.0169 OFF). This is
  the run that exercises the **gfx906 MoE WNA16 path** after the merge:
  `_process_weights_gfx906`'s new 10-tuple, the `int_wna16` oracle gates, and
  `moe_gptq_gemm_gfx906`.
- **TP=2 Qwen3.8-27B-AWQ-INT4 re-measure** (MTP k=3 default, filler, prefix caching
  OFF, bt4096, max-seqs 4, util 0.82, capture `[4,8,12,16]`, maxlen 131072):
  **2k 79.1 · 64k 42.0 · 120k 40.7 t/s** decode; TTFT 4.2 / 185.3 / 442.2 s;
  standard `vllm bench serve` (random, 2048→256) **52.9 t/s** @ TPOT 18.90 ms.
  Both READMEs updated with these numbers.
- **Operational note for re-measures:** three leftover profiling plugins
  (`agdn`/`pfk4`/`syv9`, armed from `/local/tmp/mtp1/*_arm.cfg`) install forward
  hooks that do `os.path.exists`/`open` inside the graph and break inductor AOT
  compile of any server. They are not part of the merge; disarm the arm cfgs or
  set `VLLM_PLUGINS=gfx906_fa` (the fork's own plugin) for a clean compiled boot.
- **Open follow-ups** (no gate yet): the `rocm_aiter_mla_sparse` gfx906
  fp16-sparse early-return and MiniMax-M3 `amd/ops` routing; the 8k/32k/96k
  long-context points were not re-measured.

## 2026-09-18 (two gated wins go default-on; stale-verdict sweep)

Kevin's decision after the 0.30.0 review (`MERGE-0.30.0-review.md`) established
that both were gated wins awaiting a flip, not parked code.

- **`VLLM_GFX906_SKINNY_M16` is now default-on** (`=0` is the kill switch).
  Record: `DEVLOG-fp16-skinny.md`, VERDICT SHIPPED — 35B MoE N=8 graph **191.0 vs
  166.9 t/s (+14.5 %)**, 27B (Qwen3.8) N=8 **104.2 vs 98.2 (+6.1 %)**, 27B N=4
  control flat (−0.6 %, flag inert), correctness + per-shape 2–7.5× PASS, and a
  passed 30-rep × 2-model soak. Covers the M=5..16 spec-verify / 5–16-seq
  concurrent-decode regime, which was falling back to the M-invariant Triton
  skinny path.
- **`VLLM_GFX906_QUANT_LAYER0_MOE` (C4) is now default-on** (`=0` is the kill
  switch). Record: `DEVLOG-c4-layer0-quant.md` — GO 2026-09-01 with every gate
  passed: unit 8/8; PPL 15.9531 → 15.9929 (Δ +0.04 against a 0.5 gate); greedy
  serving fingerprint bit-identical; serving A/B 84.95 → **87.51 t/s (+3.0 %)**
  against a ~1.8 % noise floor; ~1.5 GiB returned to graph capture.
  **Re-measured after the flip on the house reference workload** (Qwen3.5-35B-A3B-AWQ,
  `_bench_gfx906.py` pp2048/tg256, 4 samples, mclk 1000): **59.79 t/s**
  (59.76–59.84) vs **58.40** before the flip = **+2.4 %** — so future 35B numbers
  must be compared against the new baseline, not the 57.97–58.36 band.
  **Quality, same build and prompt set (`VLLM_GFX906_QUANT_LAYER0_MOE` ON vs the
  `=0` kill switch): PPL 15.9361 vs 16.0169** (359 tokens, 0 top-20 misses in both;
  Δ 0.08 in the ON-better direction, while the 2026-09-01 pair differed 0.04 the
  other way ⇒ the delta is at the probe's resolution, not a quality signal). Layer 0's
  experts are quantized at load, which is a quality trade-off the checkpoint
  author did not make — accepted on the measurement above.
- **NH-4 stays** (`VLLM_GFX906_MAMBA_FUSED_GROUP_NORM`, default off) — Kevin's
  call after the review; its neutral A/B result is unchanged, and its stale
  "pending the serving A/B gate" comment is now accurate about what was measured.
- **Stale-verdict sweep (first pass):** the three stale comments above
  (NH-4, C4, SKINNY_M16 docstring) plus `DEAD-ENDS.md`'s "`VLLM_GFX906_FUSED_DRAFT`
  has no reader in-tree" (the A3 opt-in was revived 2026-09-14 with three tests)
  and a stale `V1`-pin recipe in `docs/gfx906/README.md` that contradicted the
  same file's DFL2-2 closure note. Everything else in the ~60-flag gfx906
  namespace reconciled (flag default ↔ comment claim ↔ recorded verdict). The
  `GFX906_FA_LEGACY_ALLOW_UNVERIFIED` reference in `test_gfx906_fa.py` is a
  deliberate guard that the removed override stays inert — kept.
  New standing rule: [`AGENTS.md`](AGENTS.md) merge-train rule 6.

## 2026-09-17 (V2-MAMBA-1 — V2 runner + mamba `align` mode no longer faults on gfx906)

- **A GPU memory fault on any hybrid model with heterogeneous KV-group block
  sizes is fixed**, found via Qwen3.8-Flash-Next (the V2-runner + prefix-caching
  combination its recipe needs). `MambaHybridModelState.add_request` seeded the
  per-request running mamba block column with `cache_config.block_size` instead
  of `cache_config.mamba_block_size`; the engine narrows the former to the
  **finest** KV-cache group's block size (4, from Qwen4Exp's
  `CircularBufferSpec` indexer group) while the mamba geometry stays 192, so a
  prefix-cache hit seeded a column ~57× too far out and the align pre-copy
  followed a stale block-table entry to a wild address. One line (+assert) now
  uses the mamba block size — the value the V1 path already used
  (`mamba_utils.py`: `block_size = mamba_spec.block_size`).
- **Not gfx906-specific and not an upstream fix re-derived:** `upstream/main`
  (fetched 2026-09-17) still has the seed line verbatim. Upstream *did* narrow
  the trigger the same day (only `prefix_cacheable` groups contribute to the
  min), which masks it for this model rather than fixing it.
- **Gates:** the tiny Qwen4Exp rig with prefix caching ON runs the sequence that
  used to fault, with greedy tokens **and** top-5 logprobs bit-identical to the
  prefix-caching-OFF arm (worst |Δ| = 0.000000, 6 prompt/rep pairs), with and
  without MTP k=3; 12/12 requests OK per arm. New unit test
  (`test_add_request_seeds_running_column_with_mamba_block_size`) fails on the
  pre-fix code with `assert 287 == 5`. The two CUDA-gated mamba kernel tests are
  now ROCm-enabled: **195 passed** on gfx906.
- **Supersedes** the `--no-enable-prefix-caching` workaround in the Qwen3.8 serve
  recipe (that recipe lives on `gfx906/qsa-fn`; the flag is retired there and kept
  only as a fallback for older builds). Record:
  [`DEVLOG-v2-mamba-align.md`](DEVLOG-v2-mamba-align.md).

## 2026-09-16 (KVLAYOUT-1 — LEGACY=0 default flip)

- **`GFX906_FA_LEGACY=0` (Q8 side-buffer KV read path) is now the default.** Verified
  against 0.29's fused KV-cache layout (#51718): PPL **10.5472 / 10.5460** vs **10.5472** for
  LEGACY=1 (the same to within the probe's own ~0.001 run-to-run spread), 0 top-20 misses in
  every run, and **−15.5 % / −19.1 % ms/step** (MTP k=3, dense 27B
  AWQ, 64k/120k; acceptance unchanged at 1.7634/1.7634/2.0476; interleaved L1 → L0 → L1,
  so the order control is included — L1's own repeat ran faster and L0 still beat it).
- The fail-closed refusal and its `GFX906_FA_LEGACY_ALLOW_UNVERIFIED` override are
  removed as obsolete, and the always-on "experimental read path" warning is demoted to
  debug. `GFX906_FA_LEGACY=1` remains as the rollback: ~6 % faster for B=1 greedy decode,
  the one regime it wins (`DEVLOG-fa-legacy0-b1-decode.md`).
- Tests: the five LEGACY=1-assuming tests (gather-buffer lifecycle, q_pad capture grow,
  the A3 fused-loop contract, the gather-retire warning) now pin `GFX906_FA_LEGACY=1`
  themselves, and the fail-closed test became
  `test_legacy_default_is_side_buffer_after_kvlayout1`.
- **KVLAYOUT-2 is closed as stale**: the three capture/lifecycle tests it tracked
  (`test_q_pad_buffer_survives_capture_then_prefill_grow`,
  `test_gather_buffers_lifecycle_postfix`,
  `test_forward_mixed_batch_pad_tile_clamp_and_host_cu`) are collected and pass
  (`3 passed, 0 skipped`); the first two now pin `GFX906_FA_LEGACY=1` because the buffers
  they exercise exist only on that path. The one observation left *unexplained* is recorded
  as an open question in the roadmap: this path wins 15-19 % ms/step under MTP k=3 but loses
  ~6 % at B=1 greedy decode.

## 2026-09-15 (MUSE-1 first signals)

- **Muse-Glimmer loads and runs on both runners, and V1/V2 generation is
  byte-identical** (greedy, two raw-text prompts) — the first parity evidence for
  the model, and the same class of control as the triton greedy-identity test.
- **The in-process PPL probe cannot gate it**: Muse-Glimmer is a VLM
  (`MuseGlimmerForConditionalGeneration` + `vision_config`) and the probe renders
  its prompts through the chat template and profiles the encoder cache, reporting
  **362/363 top-20 misses on every arm** (PPL 36.12 V1 / 36.19 V2) while raw-text
  generation is sane — a prompt/template artifact, unlike Gemma-4's genuinely broken
  output. Its gate has to be a serving A/B.
- The `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` arm (V2) was killed by a GPU wedge
  (#93, `hipErrorLaunchFailure`), so whether stock Triton 3.8.0 makes that fork-era
  workaround unnecessary is still open; the arms need interleaving to survive the
  load lottery.

## 2026-09-15 (TRITON-1 rebuild + MUSE-1 gate + DFL2-1 prep)

- **The in-tree extension rebuild with stock Triton 3.8.0 passes** (task 2 of the triton
  adoption): `setup.py build_ext --inplace` rc=0, then `import vllm` (0.29.0 + triton 3.8.0)
  and the **FA suite 97/97**. The release build recipe therefore works on the adopted
  triton; no patches were needed.
- **MUSE-1 gate: V1/V2 parity at the first token, and the RBLOCK workaround is obsolete.**
  Clean arms (V1, V2, V1-repeat; `RBLOCK` unset) all start with the same confident token
  (`328` @ 0.00 — Glimmer's `to=self` recipient marker) and V1 reproduces its ranks 2–5
  exactly; V2 differs by ≤0.7 at −18…−25. Since the fork-era crash was a **triton 3.6.0**
  defect, `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` can be dropped from Muse-Glimmer recipes.
  Its answer-quality gate must be a **serving A/B** (its output is its own recipient/
  reasoning format, and text is not stable across processes), which is also the 0.32 V1
  deadline item.
- **Post-wedge NaN caveat recorded**: the first gate session's arms returned `nan` logprobs
  because they ran straight after wedge #94; the clean re-run gave rc=0 and 0 NaNs. NaN
  output is a post-wedge symptom — do not treat a post-reset session's numbers as evidence.
- **DFL2-1 prepared**: drafter `incoai/Qwen3.8-27B-DFlash2` downloaded (3.6 GB; 5 layers,
  hidden 5120, 32/8 GQA, head_dim 128, `is_causal: false`, sliding-2048), the chain patch
  touches one file, and `DFlash2DraftModel` is already registered. Subtasks added: a
  **baseline DFlash2-vs-MTP-k=3 comparison without the patch**, then **rocprofv3 kernel
  attribution**, **microbenches** at the real shapes, and a **HIP-feasibility assessment per
  top kernel** (starting with which attention backend the drafter selects — its attention is
  bidirectional, which our custom FA already serves).

## 2026-09-15 (0.29.0 release readiness)

- **Validation matrix on the 0.29.0 line (release candidate state):** FA suite **97/97**;
  in-process PPL on the dense 27B **10.5516** (bit-identical to V1 and the 0.28 line, so the
  V2 bring-up, VIT-1 and the `kv_split` override left the text path numerically untouched);
  MoE 35B 57.97 t/s (v2 restamp 58.36, 0.28 record 58.43); Nemotron PPL 26.9937 (band
  26.96–27.02); Ornith 16.6664 (fork 16.7824); Gemma-4 gated via its chat template
  (templated V1/V2 identical, logprobs ≤0.05); Muse-Glimmer gate in flight.
- **Default flips on this line:** V2 model runner (dense 27B, MoE 35B, Nemotron, Ornith,
  Gemma-4); **VIT-1** (ViT attention on the custom FA: −11.5 % image-prompt TTFT @1024²,
  −55 s fresh-boot Triton JIT); **stock Triton 3.8.0** (upstream gfx906 support, fork kept
  as rollback); MTP k=3 spec config. Muse-Glimmer remains the only V1 pin.
- **Published:** `unverbraucht/vllm-gfx906:0.29.0-rocm-7.14` + `:0.29.0-e730ef4066`
  (digest `sha256:bf3caead…`; built from `preset.0.29.0-rocm-7.14-kintegrated.sh` pinned to
  `gfx906/v0.29.0` @ `e730ef4066`; verified in-container: vllm 0.29.0, transformers 5.15.0,
  triton 3.6.0+gfx906, torch 2.13.0+gfx906, FA extension loads). Note the image still ships
  the triton fork; a triton-adopting image needs its own tag.
- **Open before publishing the next artifact:** push the branch (needs the `gh` `workflow`
  scope or the user's credentials — 18 commits ahead of the pushed `e730ef4066`), the
  in-tree extension rebuild against stock Triton 3.8.0, and a triton-adopting image build.

## 2026-09-15 (IFT gate tool + the PPL probe's real limit)

- **`BENCH_CHAT_TEMPLATE=1` does not make the PPL probe valid for IFT checkpoints** —
  verified, not assumed: templating Gemma-4's prompts makes the number *worse*
  (PPL **1278491** vs 84261 raw, 0 top-20 misses in both) because prompt-logprob PPL asks
  the model to predict the **user's** tokens, which an instruct model was never trained to
  model. Both are prompt-format artifacts.
- So the shipped gate for those models is a new repo tool,
  **`benchmarks/kernels/gfx906/ift_chat_gate.py`**: renders each prompt through the model's
  chat template and prints the greedy continuation plus the first-token top-k logprobs —
  a confident first token (≈0.00) with a sensible completion is the signal, and running it
  under two configurations gives a parity gate by comparing text and logprobs (that is how
  Gemma-4's V2 validation was done). The prompt-format note, `running.md` and the probe's
  own warning now point there instead of promising the flag fixes it.

## 2026-09-15 (GEMMA4-1 CLOSED + prompt-format guards)

- **Gemma-4's V2 gate passed**: a *templated* in-process V1-vs-V2 comparison gives
  identical text and logprobs agreeing to ≤0.05 (`'Paris//'` at 0.00), so its V1 pin is
  lifted on evidence. The model was never broken (raw text on an IFT checkpoint returns
  garbage; the same garbage reproduces on the 0.28 image).
- **Prompt-format guards shipped** so the trap cannot repeat: the PPL probe warns loudly
  when a tokenizer has a chat template while `BENCH_CHAT_TEMPLATE` is unset and renders
  prompts through the template when set; the throughput harness records
  `prompt_form`/`has_chat_template` and warns that tokens/s is not a correctness gate; and
  `docs/gfx906/README.md` now carries a "Prompt format — read before gating any model"
  block with the raw-text-valid vs template-required model split.
- Lesson recorded (with AGENTS.md): the Gemma-4 row already said "chat template
  required", but an aside in a notes column is not a guard — and a speed number is not a
  correctness gate.

## 2026-09-15 (GEMMA4-1: the model is fine, the gate was wrong)

- **Gemma-4 was never broken and there is no 0.28→0.29 regression.** Two probes misled
  this item: (a) the in-process PPL probe (degenerate 84261/108909, 0 top-20 misses —
  a prompt/format artifact), and (b) my raw-text generation probe on 0.29, which showed
  garbage and was then *reproduced byte-identically by the 0.28 image*, proving the
  lines agree rather than that the model is broken. Root cause of both: Gemma-4 is an
  **instruction-tuned** checkpoint that does not continue raw text. With its chat
  template it answers correctly and confidently — `'Paris//'` at first-token logprob
  0.00, and a correct Python function snippet at ≈0.00 (`/local/tmp/b4/gemma_diag.log`).
- So the model's "supported, 67.79 t/s" row is a *speed* claim that was never backed by
  an output gate — the same class of proxy error as the acceptance and CAT-1 lessons.
  The gate for it (and for its V2 flip) is a **templated** in-process comparison or a
  serving A/B; the templated V1-vs-V2 parity run is in flight.

## 2026-09-15 (0.29 line: missing-model investigation)

- **(CORRECTED 18:55 — see the next entry: Gemma-4 is *not* broken and it is not a 0.29
  regression; the raw-text probe below was invalid for an instruction-tuned
  checkpoint, and the 0.28 image reproduces the identical output.)**
  ~~Gemma-4 is broken on the 0.29 line — a regression, not a probe limitation.~~
  In-process greedy completions at temperature 0 (V1) are garbage
  (`' it it it it most is it it ...'`, `'<|||||||로. ...'`), which is what the
  degenerate PPL (84261.54 V1 / 108909.96 V2) was telling us. The same checkpoint
  was the fastest model on record on the 0.28 line (67.79 t/s). The 0.29 merge
  rewrote a lot of Gemma-4 code (`gemma4_mm.py` +181 lines, `gemma4.py` +64,
  new `gemma4_dspark.py`, the Gemma-4 MTP/unified paths, 43 files / +1702 lines
  including the quantization utils) — that is the suspect area. GEMMA4-1 is
  reframed accordingly; it stays V1-pinned but V1 is *also* broken for it.
- **Muse-Glimmer: the AWQ-INT4 checkpoint pull is in progress.** 24 GB / 16 files,
  throttled unauthenticated at ~2.5 MB/s (~2.7 h); it goes to
  `/data/cache/huggingface/hub` because the `/local` cache has only ~27 GB free.
  On arrival: the V2 parity run plus a test of whether stock Triton 3.8.0 makes
  Muse-Glimmer's `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` workaround unnecessary
  (that env exists because the rblock variant compile crashed in the triton fork).

## 2026-09-15 (TRITON-1 adopted)

- **Stock upstream Triton 3.8.0 is now the default** (the ai-infos v3.6.0+gfx906 fork
  is retained as rollback only). Upstream has carried gfx906 since `aa53dba7455`
  `ISAFamily::GCN5_1`; all gates pass (FA 97, dense/MoE/Nemotron/Ornith numerics,
  ViT fallback, serving ms/step parity) and the install docs
  (`README.md`, `requirements/build/rocm.txt`, `running.md`) now describe building
  from the upstream v3.8.0 tag instead of cloning the fork. Two build gotchas and
  the PyPI-wheel-import segfault are recorded; adoption therefore carries a small
  build step.
- **The 0.29.0 docker image was built and published**: `unverbraucht/vllm-gfx906:
  0.29.0-rocm-7.14` and `:0.29.0-e730ef4066` (digest `sha256:bf3caead…`, both the
  same image), from `preset.0.29.0-rocm-7.14-kintegrated.sh` pinned to
  `gfx906/v0.29.0` @ `e730ef4066`. Verified in-container: vllm 0.29.0,
  **transformers 5.15.0** (the pinset's stale `5.7.0` violated upstream's
  `>= 5.10.4` and was updated), triton 3.6.0+gfx906, torch 2.13.0+gfx906, FA
  extension loads. This image still ships the fork Triton; a triton-adopting image
  would be a separate build.

## 2026-09-15 (TRITON-1, model gates)

- **All per-model gates pass on stock Triton 3.8.0.** MoE 35B (layer-0 Triton
  `fused_moe`) **57.97 t/s** (58.03/57.97/57.96/57.90, mclk 1000) vs the recorded
  58.36; Nemotron **PPL 26.9937** vs 27.0066 (band 26.96–27.02); Ornith **PPL
  16.6664** vs 16.7824 — all 0 top-20 misses. Dense 27B and the FA suite were
  already green. Ornith's −0.7 % is the largest numeric shift (PPL is
  deterministic per build+model here, so it is a real codegen effect on that
  model's kernel mix) and is the one number worth remembering when adopting.
- **Artifact caveat:** the published PyPI `triton-3.8.0` wheel segfaults on import
  on this box (AMD backend and `gfx906` present; no missing libs, no runtime deps),
  so adoption means building the unpatched upstream tag with our documented recipe
  — no patches, reproducible, but a build we produce. AMD's ROCm-index wheel
  (`3.7.1+git0263a6a6.rocm7.14.0`, vLLM's own `rock.txt` pin) downloads and is an
  untested alternative.

## 2026-09-15 (TRITON-1)

- **Stock Triton supports gfx906 since v3.8.0 — validated at parity, so the
  patched fork is no longer needed.** Upstream's `aa53dba7455` "[AMD] Add GCN5.1 /
  gfx906 target (#9628)" introduces `ISAFamily::GCN5_1` (wave64, DPP broadcast,
  `supportsVDot`, no MFMA, deliberately not CDNA/RDNA); v3.7.1 predates it. Built
  stock v3.8.0 (recipe: gcc, *no* `TRITON_BUILD_WITH_CLANG_LLD`, plus
  `TRITON_APPEND_CMAKE_ARGS=-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON`) and ran the
  screens on the 0.29.0 line: **FA suite 97 passed**; **PPL 10.5472 vs the fork's
  10.5516 (−0.04 %)** with 0 top-20 misses and a **byte-identical** 32-token greedy
  completion (so the drift is fp accumulation rounding); the
  **`GFX906_FA_VIT=0` ViT fallback** (flash-attn/Triton-AMD) compiles and runs; and
  serving parity in the same boot (MTP k=3, agentic, 64k+120k, 2 reps) gives
  ms/step **85.4/127.9 vs 85.8/128.2**. Two claims were walked back by an
  interleaved A→B→A follow-up: (a) the A/B's acceptance difference is **not** a
  build effect — at one corpus body 3.8.0 gave 2.1875 / 1.8132 / 1.7634 / 1.7128
  over four processes (spread 0.475) against the fork's 1.7634 twice, i.e. the
  cross-build delta is *inside the same build's own spread* (lead with ms/step,
  interleave arms; the variance-asymmetry hint did not survive the second
  corpus body — the fork's own two samples there differ by 0.19 and straddle 3.8.0's
  range, so per-process variance is common to both builds); 128-token greedy probes are byte-identical across all
  runs of both builds on a code prompt yet differ between two runs of the *same*
  build on a prose prompt, so close-call flips are per-process and prompt-dependent;
  the one surviving hint is that the newer Triton may be more process-variable
  (candidate mechanism: timing-based `@triton.autotune` config choice in the
  chunked-prefill GDN kernels — the PPL probe, which does not chunk, stays
  bit-reproducible per build); (b) the `supportsDirectToLdsLoadBitWidth` "gap" is
  **inert** — direct-to-LDS is only created for async copies, gated on
  `{CDNA3,CDNA4,GFX1250}` in 3.8.0 and `{CDNA3,CDNA4}` in the fork, so the fork's
  `VEGA20` case was dead code and 3.8.0's missing `GCN5_1` case is unreachable. Remaining: the adoption decision, and one
  small parity gap (`supportsDirectToLdsLoadBitWidth` has no `GCN5_1` case —
  upstreamable against #9628). Full record: `RECON-triton-1.md`.

## 2026-09-15

- **0.29.0 line updated for release: `gfx906/v2-bringup` fast-forwarded into
  `main` and `gfx906/v0.29.0` (all three at `8c147037f3`, 23 commits, 0 behind).**
  The release branch now carries the V2 bring-up, VIT-1, the CAT-1 correction, the
  A3 revival, KVLAYOUT-2's closure and the k=3 spec-decode default.
- **Decisions recorded for the release.** V2 is the default runner for the
  validated models (dense 27B, MoE 35B, Nemotron 3.5 Lightning, Ornith);
  **Gemma-4 and Muse-Glimmer stay pinned to V1** (`VLLM_USE_V2_MODEL_RUNNER=0`)
  because their parity gate has not passed. **VIT-1 stays default ON** (kill
  switches `GFX906_FA_VIT=0` / `GFX906_FA_VIT_AUTO=0`, documented in the root
  README). **V1's removal is upstream 0.32.0**, which sets the deadline for the
  two pins — tracked in ROADMAP `DFL2-2` / `GEMMA4-1` / `MUSE-1`.
- **Release docs re-anchored on the 0.29.0/V2 basis:** the root README's install
  section was empty and now states the stack (ROCm 7.14 + the official AMD DKMS
  driver for TP=2 P2P), the mandatory `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`,
  the build recipe, the MTP k=3 + capture-ladder defaults and the wedge/canary
  protocol; `docs/gfx906/README.md` carries the release basis, the V1-pin list,
  the VIT-1 knobs and the 0.29/V2 performance rows.
- **Release validation on the merged tree passed.** FA suite **97 passed** and
  the in-process PPL probe on the dense 27B is **10.5516 (359 tokens, 0 top-20
  misses)** — bit-identical to V1 and to the 0.28 line, i.e. the V2 bring-up +
  VIT-1 + the `kv_split` override left the text path numerically untouched.
- **`VLLM_GFX906_FUSED_DRAFT` (A3) is revived but inert** (default OFF, neutral at
  k=3 and k=7) and **SMLA-1 is parked/inert** (fork fp16 sparse-MLA, default off,
  DeepSeek-MLA only) — neither blocks the release.

## 2026-09-14

- **The 0.29.0 line was promoted to `main`** (fast-forward, `main` ==
  `gfx906/v0.29.0` @ `7311119d67`; `gfx906/v0.28.0` stays as the previous release
  branch). Parity gate all green before the promotion: build + extensions, FA
  suite (88 pass / 3 tracked skips), PPL probe **bit-identical** to the 0.28 line
  (10.5516), V1 serving smoke, V1 restamp (MoE 65.40/58.17, dense 24.90/16.33),
  and the agentic corpus re-measured (greedy 19.91/13.25 — parity; MTP k=3
  33.30/24.54 — parity arm-level; MTP k=3 + CAT-1 34.62/25.55). V2 was shown
  viable on gfx906 in the same window (graph serve + PPL 10.5516); its
  performance/spec-decode parity work continues off `main` (`V2-bringup.md`).
- **V2 bring-up (`gfx906/v2-bringup`) — V2 now serves spec decode at parity, and
  CAT-1 works under it.** Dense 27B on the agentic corpus (same boot, 2 reps):
  greedy 20.37/13.27 vs V1 19.91/13.25; MTP k=3 33.62/23.75 vs 33.30/24.54;
  MTP k=3 + CAT-1 42.60/26.94 vs V1 34.62/25.55 — **RETRACTED 2026-09-14**: the
  A/B client put the arm name in the prompt header, so every arm ran a different
  prompt (the AGENTS.md trap). Re-measured same-boot with the fixed client, 3 reps:
  plain 34.41/24.00 vs CAT-1 35.44/24.61 t/s (ms/step 86.3→83.7 @64k, 128.5→124.4
  @120k ⇒ **+3.0 % / +2.5 %**) with **acceptance unchanged** — the shortlist's
  effect is the cheaper per-step head read, not agreement. The `MTP draft-vocab
  shortlist ACTIVE` marker fires under V2 with graph capture on and no eager
  fallback, so item 3's silent-loss risk is refuted both by code audit (V2 drafts
  go through the draft model's `compute_logits`; the one bypass path fails closed
  at init) and live. V2-CAT1-1's earlier three-arm reading (matched 2.44/2.49 vs
  control 1.72/1.71 vs none 2.05/1.93) is retracted with it; exactness still
  holds because the rejection sampler reads the same masked draft logits it
  sampled from. V2 reserves more VRAM for the same flags (KV pool 454,536 vs
  496,693 tokens greedy; 386,513 vs 442,368 spec; capture 2.49–3.02 GiB vs
  0.71 GiB), so capture ladders matter more on V2. M3's host-`cu_seqlens` path is
  length-agnostic and now has a full-length-slice guard test; KVLAYOUT-2's three
  skipped capture tests were stale skips and pass (suite: 91 passed, 0 skipped).
- **A3 revived (V2-only, opt-in, still default OFF) and gated: NEUTRAL at the
  serving k.** The fused-draft opt-in + no-op `update_draft_decode_metadata` are
  back in `gfx906_fa_backend.py` behind `VLLM_GFX906_FUSED_DRAFT` (default 0),
  with the three archive tests (3 passed; the reuse test migrated to the 0.29
  fused KV layout) and the no-op contract re-audited against 0.29 + V2. Serving
  A/B under V2 (agentic, 2 reps, ms/step lead): off 81.2/88.7 ms @64k and
  128.9/129.4 @120k vs on 81.9/89.1 and 128.5/129.6 — neutral, like the archive's
  k=4. k=7 stays open: both `mtp7` launches wedged at load (burst) and GPU work
  stopped.
- **V2 parity extended to Nemotron 3.5 Lightning (PPL 27.0066 vs V1 26.9986) and
  Ornith 1.5-35B-A3B (16.7824 vs 16.7724)** — both flipped to V2. Gemma-4 cannot
  be gated by the in-process probe (both runners degenerate, PPL ~10^5 over a
  multimodal load) → ROADMAP GEMMA4-1, stays pinned to V1.
- **VIT-1 DONE — the Qwen3.5-family ViT now runs on the custom FA by default**
  (`DEVLOG-vit1.md`). Serving gate, fresh image per rep, prefix cache OFF, same
  boot, identical prompts: **5.81 → 5.14 s TTFT @1024x1024 (−11.5 %)**, 1.71 →
  1.67 s @512, and the fresh-boot Triton JIT for the ViT is **−55 s** (330 → 275 s
  with an empty `TRITON_CACHE_DIR`; the remaining ~140 s is the GDN Triton kernel,
  so this does *not* make the triton-AMD flash-attn package droppable). The
  swap is not bit-equivalent: same greedy answer content, different wording, top-1
  preserved, max tail |dlogprob| 0.66 (Q8-K features vs fp16). Fixed en route: the
  adapter's non-production layout assumption (packed `[seq_len,1,hidden]` is what
  the VL towers pass — multi-image requests would have failed) and the decode-era
  `kv_split` default (32), now a per-call override.
- VIT-1 step 1 landed (custom-FA arm for the Qwen3.5 ViT, opt-in, unit-validated);
  the aiter/spec-decode blocker found on the 0.29 line was fixed.

## 2026-09-13

- **Upstream v0.29.0 merged** into `gfx906/v0.29.0` (merge `3c445dba56`; parents
  `4b7e0b7eb2` + tag `98dff2a81d`). Upstream delta vs the merge base: 561
  commits / 1848 files; 104 files touched by both sides, **23 conflicts**
  resolved by hand (record: `/local/tmp/b4/v0290-resolutions.md`, condensed in
  the merge message). Headlines: **Model Runner V2 is upstream's default for all
  models** — the fork pins **`VLLM_USE_V2_MODEL_RUNNER=0`** in every recipe until
  DFL2-2 brings V2 up; upstream now ships **dflash2** itself (the fork's backport
  converged and its two superseded files went to upstream: `qwen3_dflash2.py`
  and `dflash2/__init__.py` are byte-identical post-merge); native Hunyuan VL
  moved to the Transformers backend (file deleted, our compat rename dropped);
  FA4-hd256 block sizes and the FlashInfer CuTeDSL BF16 path arrive with our
  GEMV dispatch kept intact; the WNA16 oracle keeps the fork's tested qzeros
  repack helper and learns about upstream's new `EMULATION` backend.
  **Validation (2026-09-13, boot a27a894e, all three gates green):**
  worktree build `rc=0`; **FA suite 89/89**; **PPL probe bit-identical to the
  0.28.0 line — 10.5516 both** (Qwen3.8-27B-AWQ-INT4, fp16, 359 tokens, 0
  top-20 misses); **V1 serving smoke**: coherent completion + clean teardown
  (`VLLM_USE_V2_MODEL_RUNNER=0`). Three merge defects were found and fixed
  *after* the static pass, each only by running something:
  (1) `csrc/.../gptq/q_gemm.cu` duplicate `dot22_8_f` declaration (caught by the
  build); (2) `qwen3_vl.py` `NameError: video_max_pixels_per_frame` in
  `get_dummy_mm_data` — the merge kept one of the fork's lines inside the method
  upstream rewrote; the fork's per-item video pixel cap was restored on top of
  upstream's definitions (caught by the PPL probe); (3) the 0.29 **KV-cache
  layout standardisation (#51718)** broke the FA backend's K/V split —
  `kv_cache.unbind(1)` now sees heads on dim 1 because the content axis is
  fused `K||V` (`[B, H, N, 2*D]`). Ported: both split sites now use
  `kv_cache.transpose(1, 2).split(self.head_size, dim=-1)` (identical
  `[blocks, block_size, Hkv, D]` view our kernels already took) and the backend
  declares `supported_kv_cache_layouts()` = `(KVCacheLayout.LBHNC,)`; the
  engine logs `Using LBHNC KV cache layout` (caught by the first PPL run).
  `ops/rocm_aiter_mla_sparse.py` was taken wholesale from upstream (the fork's
  fp16 sparse-MLA is half-ported, tracked as SMLA-1).
  **Spec-decode blocker found on the 0.29 line (2026-09-14):** the MTP/EAGLE
  drafter's base proposer imports `vllm.v1.attention.backends.rocm_aiter_fa`
  after a `find_spec` existence check, but that module imports AITER's gluon PA
  kernel at module scope — on a box without the `aiter` package (the gfx906
  venv; AITER is unusable here) **every spec-decode serve died at drafter init
  with `ModuleNotFoundError: No module named 'aiter'`**, i.e. our production MTP
  k=3 config could not start. Fixed by making that import optional (skip the
  metadata type with a one-shot warning); MTP k=3 then serves normally
  (`MTP3 READY ~395 s`). The deeper, upstream-worthy fix is to guard the
  module-level aiter import in `rocm_aiter_fa.py` itself.
  **Agentic-corpus re-measure on 0.29 (boot eefacc1e, V1 pinned, 2 reps/cell,
  mclk 1000):** greedy 20.39 @64k / 13.26 @120k, MTP k=3 32.54 / 24.53, MTP k=3
  + CAT-1 **34.62 / 25.55** — the CAT-1 shortlist reproduces here (+6.4 % /
  +4.2 % over plain k=3 in-session) and its `MTP draft-vocab shortlist ACTIVE`
  marker is logged in serving, which also de-risks the V2 variant of that check.
  An arm-level re-check of bare MTP k=3 (3 reps) then **retired that reading**:
  64k {35.20, 35.14, 29.56} → mean 33.30 (0.28: 33.28), 120k {24.26, 25.14,
  24.21} → mean 24.54 (0.28: 24.74, -0.8 % inside the spread), acceptance
  2.05/2.05/1.55 and 2.06/2.15/2.06 — the 2-rep "-1-2 %" was the arm's own
  acceptance variance, and spec decode is **at parity on 0.29**. Two load
  wedges hit the session (#81 mtp3 attempt 1, #82 CAT-1 attempt 1); both retried
  clean and are recorded in `degradation.md`.

  **Parity + V2 status (boot eefacc1e, same day).** V1 restamp of the 0.29 line
  against the 0.28 numbers: MoE **65.40 warm / 58.17 cold**, dense **24.90 /
  16.33** t/s — all within ±0.8 % (mclk 1000 in every window), i.e. **0.29 V1
  parity with 0.28 is established**. And **V2 is viable on gfx906**: the graph-mode
  V2 serve returned a coherent completion with a clean teardown, and V2
  in-process PPL = **10.5516**, bit-identical to V1's and to the 0.28 baseline
  (the Y16 "unsupported-by-design" record was the box's load lottery — a BACO
  reset landed in the same second as the eager attempt that failed). Remaining
  V2 work is performance/spec-decode parity: `V2-bringup.md`.

## 2026-09-02

- **MTP-1a closed: Qwen3.8-27B dense MTP k=2 crossover pinned at 32k–64k pp**
  (clean boot Q, TP=2, n=3, cold prefill, arms sequential). MTP wins ≤32k
  (1.03×), loses ≥64k (0.85× → 0.72× @120k); acceptance 2.0 stable through
  120k so the tax is O(Sk) step cost, not draft rejection. Bracket is ~2×
  wider than S9's ~20k estimate. Optz microbench: lm_head-per-draft lead
  DEAD (memory-bound, +322 µs/step = 0.4% of the 78 ms @120k step); attention
  K-multiplier ~1.0 at S=120k (KV bytes shared). Open residue: the 78 ms vs
  ~12 ms bandwidth-floor gap needs a rocprofv3 kernel breakdown (MTP-1b gate;
  blocked by a zombie KFD VRAM handle left by the old-vLLM wedge below).
  Old-vLLM 0.23.1 A/B abandoned: that code path wedged GPUs loading this AWQ
  model at shard ~2/5 on BOTH userlands (docker ROCm 7.2, in-process ROCm
  7.14) — incompatible with the model+host, not an env issue. Record:
  `DEVLOG-mtp1.md`, degradation entries 2026-09-01/02.

## 2026-09-01

- **Post-C4 maintainability sweep: removed the two FULLY-DEAD M=1 routing
  experiments (S2 topk kernel + C1 stage 2 fused routing).** Per the
  post-C4 inventory directive (inventory all merged-but-not-default-enabled
  gfx906 work; remove dead code for maintainability with a preservation
  branch first), branch `gfx906/preserve-dead-kernels` was cut at main so
  every removed line stays in git history, then: S2 (`moe_topk_gfx906.cu`,
  the `VLLM_GFX906_TOPK_M1` dispatch, bindings, `_custom_ops` wrapper +
  fake, tests, bench) and C1 stage 2
  (`moe_routing_fused_m1_gfx906.cu`, the `VLLM_GFX906_ROUTING_FUSE_M1`
  dispatch, bindings, tests) were deleted, along with the
  `_fused_align_meta` router→expert plumbing they orphaned (C1 stage 2 was
  its only producer; verified zero non-None assignments remain) — touching
  `moe_runner.py`, `routed_experts.py`, `modular_kernel.py`,
  `fused_moe_modular_method.py`, `gfx906_w4a16_moe.py`, and
  `fused_topk_router.py`. Net −406/+9 lines, 13 files. Both were already
  recorded FULLY DEAD in `DEAD-ENDS.md` (3rd isolated→serving flip pattern;
  C1 stage-1 — the align+count kernel that shipped default-on — is
  untouched). The structural probe `c1_routing_structural_probe.py` is kept
  (hasattr-guarded, degrades gracefully) as the cited evidence record.
  Verification: incremental build rc=0; unit suites green; live M=1/M=4
  `fused_topk` through the edited router OK; removed ops confirmed absent
  from the compiled extension; full-engine e2e probe (Ornith-1.5-35B-A3B,
  C4 ON) loading + coherent decode. Remaining default-off knobs after the
  sweep: `SKINNY_M16` (cleared to flip default-on — pending decision),
  `MAMBA_FUSED_GROUP_NORM` NH-4 (+0.4 % = noise; keep, revisit when a
  non-MoE-GEMV-bound config exists), the int8 W8A16 family
  (`W8A16_INT8*`, T1 PROBE GO dependency — keep), and C4
  `QUANT_LAYER0_MOE` (GO, opt-in pending soak). The stale S5 row in
  `DEAD-ENDS.md` was corrected: that kernel is live default-on as
  `VLLM_GFX906_MOE_M1` since the C2 combined A/B (+2.72 %).

- **C4: load-time int4 quantization of the unquantized first MoE layer (GO,
  measured).** Qwen3.5-35B-A3B-AWQ leaves `model.layers.0.` in fp16
  (`modules_to_not_convert`), so its routed experts ran on the Triton
  unquantized path at ~4× per-call cost (740 vs 182 µs, C2 profile). New
  method loads fp16 as usual, then quantizes to int4 in
  `process_weights_after_loading` (asymmetric AWQ, codepoints against the
  stored fp16 scale) and delegates to the shared gfx906 WNA16 repack — no new
  kernel code; fp16 storage released (~1.5 GiB). Gates: unit 8/8; PPL Δ +0.04
  (noise); greedy serving fingerprint bit-identical across arms; serving A/B
  (M=1, pp2048/tg256, same boot) **84.95 → 87.51 t/s = +3.0%** (above the
  ~1.8% noise floor). Pre-merge review (self + Claude CLI): runner-visible
  quant-method state sync + `supports_eplb` path-following fixed; one finding
  rejected with evidence. Ships opt-in (`VLLM_GFX906_QUANT_LAYER0_MOE=1`)
  pending a soak window. See `DEVLOG-c4-layer0-quant.md`.

## 2026-08-30

- **NH-4: mamba2 grouped gated-norm fused path (SHIPPED, env default
  OFF).** `Mixer2RMSNormGated.forward_cuda` routes the n_groups>1 case
  through the existing fused Triton `rms_norm_gated` kernel behind
  `VLLM_GFX906_MAMBA_FUSED_GROUP_NORM=1`, gated on
  `per_rank_hidden_size % group_size == 0`. Isolated ~68 → ~55 µs/layer
  (~0.29 ms/step over 23 mamba layers); serving A–B–A (TP=2+EP, fresh
  boot per arm) 109.8 → 110.05 → 109.37 t/s (+0.4 %, within inter-arm
  noise — step is MoE-GEMV-bound at this batch) with PPL 24.9034 vs
  24.8944 (Δ +0.04 %). Correctness: 11/11 unit tests (incl. production
  TP=2 geometry and TP-driven partial-group refusal), TP=2 regression
  driver 6/6 bit-equal, ruff clean. Review protocol: self-review +
  Claude CLI review of branch vs main, both findings fixed before merge.
  See `DEVLOG-nemotron-h.md` (NH-4 section) and `ROADMAP.md`.

## 2026-08-29

- **NH-1 + NH-3: Nemotron-3.5-Lightning-30B-A3B mixed INT4/INT8
  onboarding (`gfx906/nemotron-h-onboard`, unmerged).** Serves at
  70.4 tok/s (graph, pp2048/tg256, 4 samples) from 4.95 tok/s at first
  load (14.2×): fp32-router LLMM1 dtype guard; ssd_chunk_scan
  pointer-yield restructure working around the triton-gfx906
  CanonicalizePointers fat-pointer assertion (94/94 SSD reference
  tests); new `CompressedTensorsW8A16ChannelDequant` scheme replacing
  Conch for int8-channel dense layers (3.79 ms → 62 µs per M=1 GEMV,
  +1.8 GiB VRAM); gfx906 W4A16 MoE oracle gate widened to any positive
  multiple of 32 (group-64) + RELU2_NO_MUL experts (+88.8% vs Triton
  WNA16); fp32 router-gate GEMV on hipBLAS sgemv (+18.4%). PPL gate
  26.96–27.02 band across all arms. Open follow-ups NH-2 (int8 GEMV),
  NH-4/5 (mamba2/topk tails) in `ROADMAP.md`; records in
  `DEVLOG-nemotron-h.md`.
- **M2: per-q-tile prefill clip merged to `main` (`06c0614379`).** Two
  bit-identical per-q-tile scan bounds in both FA kernels — a window
  raise of `k0_base` (the tile's first row has the smallest window
  start; keys below it are masked for every row) and a causal cap of
  `k_VKQ_max` (the tile's last valid row bounds the scan tail) — plus
  the DIRECT_PAGED backend clip extended from decode-only to prefill
  chunks; knob `GFX906_FA_TILE_CLIP` (default on). Kernel A/B at the
  pp4096/full-context shape: 3.19×/2.81× (windowed, both kernels) and
  2.22×/1.96× (causal-cap-only, first-chunk full-attention geometry —
  the cap is a general chunked-prefill win, not a window feature).
  Review-gated e2e: Muse pp16384/B=2 windowed **+11.8 % wall / +14.8 %
  prefill**; Qwen3.8-27B pp2048 full-attention +0.73 % (GEMM-dominated;
  its FA component is the 1.96–2.22× above). Decode/spec paths provably
  unchanged (cap = seq_len; raise ≡ the existing floor). Residual:
  per-row granularity within a 64-row tile (~1/32 of the effect) left
  open. Records: `DEVLOG-fa-kernel-batches.md` (M2 + 2026-08-29 review-fix
  entries, `m2-code-rev-glm5.md` findings closed by `04e6ab7c60`).
- **M3: kernel hygiene batch merged (`feat/fa-m3-hygiene`).** #8
  device-side `k0_base = max(0, kv_start[seq])` clamp (a negative start
  walked the paged k-loop into token-negative space — illegal access /
  wedge, not a wrong number); #10 overflow-free window cutoff
  (`q_abs_row - k_pos_abs >= window`, provably equivalent for all int32
  window; the old form could not actually wrap — hardening/clarity);
  #4b `o_meta` `[B,Sq,Hq,2]` allocation dropped entirely (the kernel's
  only `dst_meta` write is guarded by `gridDim.y != 1` and
  `gridDim.y == kv_split`, so the buffer is dead at `kv_split==1` too
  — ~300 KB/layer at Sq=1568/Hq=24); amplified-V window-boundary
  regression pin (~400× discriminative). The branch's dot2 P·V rewrite
  premise was REFUTED by ISA and the item closed: objdump of the
  production build shows the P·V accumulate already compiles to
  `v_pk_fma_f16` (1024× in `flash_attn_tile_q8<128,128,16,2>`, 0×
  `v_pk_add_f16`), so `v_dot2_f32_f16` buys zero instruction count —
  precision-only candidate, revisit only behind a numerics gate
  (`dequant-instructions.md` corrected, old paragraph SUPERSEDED).
  Post-merge suite 70/70 (60 base + 5 M2 + 5 M3 parametrized cases);
  both review rounds (`m3-code-rev-glm5.md`, external fold) closed at
  `cf5ccbd685`/`9d98aca9ab`.
- **M4: long-context split-K accuracy point closed (qwen review #4a).**
  Production split defaults are safe — in fact MORE accurate — at
  16k–32k context: in-process probe (sk 16384/32768, D=256/Hq16/Hkv2
  + D=128/Hq32/Hkv2, seed 20260829) shows gather kv_split=16 (the B=1
  default) at 5.2e-3/6.6e-3 rel vs fp32 ref and direct-paged
  kv_split=8 (the B≥2 clamp default) at 4.0e-3/5.0e-3 — all ≤ half the
  5e-2 tolerance, and the no-split baseline is WORSE (1.9e-2/2.6e-2):
  the split partials are fp32 (the M4 "unscaled fp16 partials" framing
  was stale) and the fp16 P·V accumulator error scales with
  accumulator length, which splitting shortens. Suite 74 → 78 (two
  16k gather arms + direct-paged L=16384 split-8 pin, both
  geometries); probe kept at
  `benchmarks/kernels/gfx906/m4_splitk_accuracy_probe.py`.
  Records: `DEVLOG-fa-splitk-accuracy.md`.
- **B=1 LEGACY=1-vs-0 decode gap closed (roadmap item #1): LEGACY=0
  stays OFF.** Same-boot (boot O) serving A/B, Qwen3.8-27B TP=2 B=1
  pp2048/tg256: LEGACY=1 40.11/40.12 vs LEGACY=0 37.61/37.56 (−6.3 %)
  t/s; the M5-era direct-paged B=1 config lands within 0.2 % of the
  Q8-gather config (37.55/37.54) despite very different FA/gather
  subcomponents (kernel probe: the Q8-gather read path is 22–45 %
  FASTER per step than fp16-gather+quantize, growing with Sk; direct
  paged is +8–35 % slower). The serving gap is therefore a
  LEGACY=0-common per-step cost, not FA/gather: the append-time Q8
  side-buffer write is +60–105 us/step eager (16 full-attn layers;
  q8-alone ×16 = 105.6 us bound), and
  the ~1.55 ms/step remainder is a serving-harness/graph-node
  interaction (unmeasured). M5's "LEGACY=0 LOSES, default stays 1"
  verdict confirmed by a proper same-boot adjudication. Probes kept
  (`benchmarks/kernels/gfx906/legacy0_b1_step_probe.py`,
  `legacy0_append_cost_probe.py`); `_serve_tp2_gfx906.sh` gained
  EXTRA_SERVE_ENV passthrough. Records:
  `DEVLOG-fa-legacy0-b1-decode.md`.
- **Roadmap reorganization: three per-topic roadmaps → single
  priority-ordered `ROADMAP.md` + `REFRIGERATOR.md`.** The per-topic
  split had leaked (G1/housekeeping in more-models, non-MoE N-items and
  the upstream queue in the MoE file) and the spec-decode roadmap was
  100 % parked work. Closures folded in: the spec-decode file is deleted
  (all four items → REFRIGERATOR with reopen gates); the Muse follow-ups
  section is empty and gone (LEGACY-flip closed this date, Part C →
  REFRIGERATOR); DeepSeek-V4-Flash → REFRIGERATOR (not an active
  target); Qwen3-30B-A3B → DEAD-ENDS (SUPERSEDED — not an active goal,
  model superseded by the supported Qwen3.5/3.8 line). C4 stays active
  (70 t/s target active, user decision 2026-08-29). Item IDs (C*, G*,
  L*, N*, U*, HK*, SD-*) are stable; README/AGENTS references updated;
  the C8 L2/residency open question is folded into C2. Historical
  filename mentions inside devlogs/plans are left as records.
- **MoE C1 stage 1: M=1 fused align+count kernel landed (opt-in flag
  defaulted ON after gate).** The M=1 decode routing chain is 3 kernels
  per layer (topk + align 2-block + count_and_sort = 120 graph nodes/step
  ≈ 0.8 ms); the new 1-CTA kernel replaces the align pair (120 → 80
  nodes), bit-equal to the generic chain. Structural probe + S2
  re-validation: isolated-graph kernel numbers can flip sign in the
  production graph (S2 topk swap: −1.03% serving vs −28% per node in
  isolated graphs), but **node removal transfers** — serving A/B
  (in-process MoE 35B, pp2048/tg256, 4 samples/arm, back-to-back):
  **+1.18% (207 µs/step), +1.73% on the second session**; within 8% of
  the isolated prediction. `VLLM_GFX906_ALIGN_M1=0` opts out. Stage 2
  (fused topk+align+count, 120 → 40 nodes) is the follow-up. Records:
  `DEVLOG-moe-c1-routing-fusion.md`,
  `benchmarks/kernels/gfx906/c1_routing_structural_probe.py`.
- **MoE C1 stage 2: fused topk+align+count — DEAD-END in production
  (flag OFF, kernel + plumbing + tests landed).** The one-CTA fused
  routing kernel (`moe_routing_fused_m1_gfx906`) is bit-equal to the
  3-kernel chain (27/27 tests) and 28 % faster in isolated graphs
  (40 nodes: 10.0 µs/node vs 13.8 µs/layer for the stage-1 pair) — yet
  the A-B-A serving gate shows **−1.10 %** (57.42 → 56.79 → 57.46
  control t/s, Qwen3.5-35B, pp2048/tg256): the third S2-pattern flip,
  and the stage comparison pinpoints it: node REMOVAL transfers
  (stage 1, +1.2–1.7 %), REPLACING the proven production topk does not
  (S2: −1.0 %, stage 2: −1.1 %). Router→expert meta plumbing
  (optional `fused_align_meta` kwarg, signature-gated, dropped for
  unquantized/ignored layers) is in place and production-neutral with
  the flag off. Records: `DEVLOG-moe-c1-routing-fusion.md` (stage-2
  section), `tests/kernels/moe/test_moe_routing_fused_m1_gfx906.py`.

## 2026-08-27–28

- **Muse-Glimmer-30B-AWQ-INT4 onboarding + window FA + M1 gather clip
  merged to `main` (2026-08-28, `feat/muse-glimmer` fast-forward).**
  Sliding-window support in the custom Q8 FA (window arg, both kernel
  copies; all-CUSTOM 1.59× vs hybrid at B=1), direct-paged split-K +
  Phase C clip, LEGACY=0 Q8 side view aliased into the fp16 K half
  (zero extra KV memory, COW-safe; prefix-cache fail-closed removed),
  and the M1 gather-path window clip (absolute-position gather layout,
  +8.1% e2e at pp8192/B=1). Root-caused and fixed the boot J/K
  first-prefill OOM: the q_pad buffer was per-impl (v1 creates one
  backend impl per attention layer) — 52 × 256 MiB = 13.3 GiB;
  ClassVar share cut the transient 3.785 → 1.285 GiB and made bt4096
  TP=2 serving viable (the bt2048 workaround is droppable). Records:
  `DEVLOG-muse-glimmer.md` (rounds 1–5), `degradation*.md` boots I–L,
  working TP=2 recipe in `README.md`. Review rounds 1–3 + the
  post-boot-L review set (`fa_oom_fix_clip_code_rev_*.md`) closed;
  the two robustness gaps they flagged (raw-fp16 branch assert,
  GATHER_CLIP_MARGIN/config-table static_assert) landed in
  `52ff21f9d9`.
- **M6 Part B: LEGACY=0 B≥2 default route flipped to the fused-Q8
  gather (2026-08-28, round 10).** The M5 bake's B=4 @2k
  −27…−31 % deficit was Sq>1-specific (the in-process Sq=1 A/B on
  the identical strided-read path was a wash; mechanism — strided
  Q8-slice reads leading but unconfirmed, Sq>1 machinery at least a
  co-contributor — round-10 erratum). `GFX906_FA_DIRECT_PAGED_Q8`
  (default `0` since the flip) routes LEGACY=0 B≥2 through the
  fused-Q8 gather: B=4 @2k aggregate 35.7 → 46.3 t/s (parity with
  the 46.7 LEGACY=1 control within cross-boot uncertainty; B=1 and
  prefill unchanged; 60/60 suite). A no-op under the production
  LEGACY=1 default; direct-paged stays opt-in (=1). The M5
  LEGACY-flip gate's B=4 half is now green; the flip itself still
  needs the B=1 same-boot adjudication.
- **M5: LEGACY read-path bake executed — keep `GFX906_FA_LEGACY=1`
  (`a6780408a8`).** The TP=2 ngram bake measured LEGACY=0 slower at
  every controlled point (B=1 −2.5…−3.7 %, B=4 −27…−31 %, prefill
  wash); per the flip rule only a win flips, so the default stands.
  The bake's original "no int8 path / fp32-ALU" reading was refuted by
  the SCEV-proof dot-rate probe (`v_dot4_i32_i8` full-rate, 4.44×
  fp32 FMA — AMD's 53 TOPS INT8 figure is this instruction; rates in
  `dequant-instructions.md`); the deficit is read-path/layout, not
  the dot. LEGACY=0 remains an experimental opt-in.
- **M6 Part A: planar Q8 repack executed — DEAD-END for the flip
  question; merged to main 2026-08-29 as loader hygiene** (merge
  `02d197189f`). The rev-2 plan's hard stop-rule fired: loader global
  loads 10→6 per tile-row (1.67× < the 2× rule) despite a −2.4 %
  standalone B=1 win, so the B=1 gap is not load-instruction-bound.
  Merged for the aligned-loader win (production LEGACY=1 shares the
  loader) and Part C groundwork: merged-tree suite 74/74 (incl. 4
  byte-level layout pins), same-boot B=1 decode-step A/B (Muse
  geometry D=128/Hq=32, NC2=1/KVSPLIT=1, boot N): slope 36.0→34.4
  ns/token (−4.3/−4.8 %), @Sk=2176 83.6→79.1 us (−5.0/−5.6 %),
  bit-identical (maxerr equal at every Sk) — gate PASS. **Caveat
  (post-merge review): both A/B arms ran under contention — the same
  merged `.so` measures 42.0 us @Sk=2048 / slope 12.86 ns/token on an
  idle GPU (1.6–1.8× faster absolute), so the recorded µs/ns are
  contended-boot numbers; the −4…−5 % delta is directionally
  supported (16/16 points, round-11 −2.4 %, ISA mechanism) but the
  merge never depended on it (abort condition was slower-than-noise;
  bit-identical).** Record:
  `DEVLOG-muse-glimmer.md` round 11, `DEAD-ENDS.md` MG row, plan
  `plan_fa_part_A.md`.
- **M6 Part C (Q4-KV via `v_dot8_i32_i4`): SHELVED (`5d8d4c7f59`).**
  Quality unproven (Q4 K *and* Q requant; 7-level codebook ≈ doubles
  KQ quantization error with no PPL evidence). Reopens only behind a
  dedicated accuracy gate that must pass before any kernel work.
- **MI50 vLLM memory-attribution skill.** Personal skill
  (`~/.agents/skills/gfx906-mem-attribution/SKILL.md`) + in-repo probe
  (`docs/gfx906/_probe_mem_attribution_gfx906.py`): the 3-arm OOM
  attribution recipe, per-layer hooks, bisection, and the env traps
  (AOT workers, inductor fork/spawn HSA). Validated on the M0 hunt.

## 2026-08-14–16

- **Phase 3 gfx906 performance stack.** The custom W4A16 MoE grouped GEMM
  fixed the 3.49 t/s routed-MoE regression, and the custom Q8 FlashAttention
  backend was integrated and made serving-viable. The dense M=1 GEMV path,
  FA GQA head packing/KV split, fused KV gather and quantization, NC2/kv-split
  guards, and bit-exact fill/copy reductions were landed. The resulting
  Qwen3.5-35B decode progression reached 64.08 t/s before the later sprint
  work. See `DEVLOG-moe-opt.md`, `DEVLOG-fa-attention.md`, and
  `DEVLOG-dense-decode.md`.
- **Phase 2 prefill close-out.** The useful prefill tuning and its negative
  results were recorded; the remaining persistent-CTA prefill idea is still
  parked as an open item in `moe-decode-roadmap.md`.

## 2026-08-17

- **Initial gfx906 roadmap and review close-out.** The Qwen3.5 improvement
  branch was merged into `gfx906/main` (`e861d0b30f`). The twelve parked
  pre-merge review items were resolved: direct-paged fp32-Q handling,
  LEGACY=0 prefix-cache guarding, bounded gather-buffer retention, MoE caller
  validation, non-gfx906 plugin/build tolerance, duplicated FA helper cleanup,
  gather-buffer reuse, MoE workspace-alias documentation, stale comments,
  lint debt, and the FA debug switch. See `DEVLOG-moe-opt.md`.
- **S2 M=1 top-k experiment.** The dedicated E=256/topk-8 softmax kernel was
  bit-equal and faster in isolation, but lost in CUDA-graph serving replay.
  It was retained behind `VLLM_GFX906_TOPK_M1` with the default off; the
  standalone-kernel approach is rejected for the gapless serving regime.
  See `DEVLOG-moe-m1-sprint.md`.

## 2026-08-18

- **MoE M=1 sprint results.** The gemm2 lane-column re-tile shipped behind
  `VLLM_GFX906_MOE_M1` (default off), improving the graph result by about
  0.60 t/s; the gemm1 version did not yet have a serving-gated win. The
  shared-expert down-projection moved to the gfx906 dense GEMV path, with the
  default-on decision recorded as provisional. See `DEVLOG-moe-m1-sprint.md`.
- **Speculative decoding Phase 0 and L1'.** The n-gram, GPU-n-gram, suffix,
  and prompt-lookup experiments established that n-gram quality is
  repetition-bound and GPU n-gram has a draft-selection mismatch. Suffix was
  deferred because its dependency and dynamic-length path were not justified
  for the current target; the k sweep was likewise not pursued. The fp16
  M<=4 GEMV-family extension (L1') shipped
  and moved the dense draft step from 66.6 ms to 53.2 ms eager; the first
  serving A/B was 0.945x. See `DEVLOG-spec-decode.md`.
- **Speculation cost-model correction.** Kernel-path census overturned the
  original GDN-small-M attribution: the required sequential GDN kernel was
  already in-tree. The dominant draft cost was the AWQ and fp16 GEMM mix, not
  a missing GDN kernel. The old L1/L2 plan was consequently re-scoped. The
  capture-safe FA uniform-batch rails required by speculative decoding were
  also landed. See `DEVLOG-spec-decode.md`.
- **C6 activation-quantization disposition.** The proposed Q8_1 activation
  path was rejected on gfx906: it adds quantization launches and lacks DP4A or
  int8 matrix hardware, so its expected net cost is negative.
- **Layer-0 MoE attribution.** The residual Triton expert calls were resolved
  to layer 0's fp16 routed experts, not the shared expert. Layer-0 quantization
  remains an open conditional candidate in `moe-decode-roadmap.md`. See
  `DEVLOG-moe-opt.md`.
- **Upstream vLLM merge.** `gfx906/main` absorbed upstream `main` in
  `38ceb5d957`; the gfx906 attention, quantization, and platform behavior was
  retained and revalidated.

## 2026-08-19

- **Speculative decoding rails completed.** The small-capture-size fix (L5)
  removed the graph padding penalty from one-token no-draft steps. MTP k=2
  support was added for the Qwen3.5-27B target, and the final agentic result
  was 39.74 t/s, 1.503x over the no-spec arm, with 90.95% draft acceptance.
  The dispatch-only L1'' investigation was closed as a non-issue: the
  relevant fp16 GEMMs already reached the dispatcher. See
  `DEVLOG-spec-decode.md`.
- **Per-file max-ilp split.** The q_gemm 4-bit build was split by M: M=1
  uses max-ilp while M>=2 does not (`cfe09d8611`). This resolved the build
  concern without regressing the MTP result. See `DEVLOG-spec-decode.md`.
- **Gemm1 re-tiling close.** The V1 design and the NPT surface were measured;
  the apparent isolated gain did not transfer to TP=1 serving, so no gemm1
  dispatch change shipped. The V3/V4 follow-ups were closed by the same
  evidence. See `DEVLOG-moe-gemm1-retiling.md`.
- **Gemma-4 onboarding.** Gemma-4 26B-A4B was loaded and characterized on
  gfx906. Its raw-prompt degeneration and hybrid-attention logprob issue
  were documented, and the model was retained as a supported test target.
  See `DEVLOG-gemma4-onboarding.md`.

## 2026-08-20

- **Upstream release `v0.28.0rc1`.** The tag was merged into
  `gfx906/v0.28.0rc1` on 2026-08-20 (`fc777b87dd`).
- **Gemma-4 symmetric no-zero-point MoE support.** The existing gfx906 W4A16
  kernel was extended through Python-side compressed-tensors gates and the
  GPTQ-K-first repack path; no new kernel was required. Serving improved from
  37.81 to 67.79 t/s (1.793x), and the flagship Qwen3.5-35B result was
  unchanged. See `DEVLOG-gemma4-moe.md`.
- **Gemma-4 review follow-ups.** The bit-width/group-size/strategy gate,
  activation-ordering (`g_idx`) guard, no-fabricated-zero-point storage, and
  the numerical divergence record were all closed. See
  `DEVLOG-gemma4-moe.md`.
- **Prefill/TTFT and build investigations closed.** The MTP2 TTFT question
  was resolved, the Qwen3.8 launch failures were attributed to the NAS rather
  than a model or kernel defect, and gfx906 auto-dtype fallback to fp16 was
  landed for bf16 checkpoints. See `DEVLOG-spec-decode.md` and
  `DEVLOG-qwen38.md`.

## 2026-08-21

- **TP=2 transport diagnosis.** The TP=2 investigation closed the initial
  RCCL/P2P failure hypotheses and identified the driver/topology issue. The
  official amdgpu DKMS driver made TP=2 serving viable; the remaining
  communication-bound ceiling and capture behavior were recorded rather than
  treated as a decode-kernel regression. See `DEVLOG-tp2-dense.md`.
- **N4 capture-width diagnosis.** The long-context decode tax was traced to
  `max_model_len` being baked into the captured FA gather dimensions, and the
  persistent live-bounded gather design was selected. See
  `DEVLOG-masked-fa.md` and `tp_decode_investigation.md`.

## 2026-08-22

- **N4 persistent gather shipped.** The persistent gather+quantize path removed
  the max-context replay tax. TP=2 serving improved from 22.4 to 40.9 t/s at
  131k context and from 15.9 to 40.9 t/s at 262k; the short-context tax was
  within noise. `GFX906_FA_PERSIST` is on by default. See
  `DEVLOG-masked-fa.md` and `DEVLOG-tp2-dense.md`.
- **C2-V validation completed.** The additional TP=2 and batch-regime tests
  showed that the M=1 re-tiles are positive at TP=2 (about +1.47% for gemm2
  and +1.23%/+1.32% for gemm1) but neutral at TP=1 and in the tested batch
  arm. The TP=2-scoped follow-up and the default-on decision remain open in
  `moe-decode-roadmap.md`. See `DEVLOG-moe-c2v.md`.
- **Upstream release `v0.28.0rc2`.** The tag was merged into
  `gfx906/v0.28.0rc2` on 2026-08-22 (`19e23ffedd`). Subsequent gfx906 work
  was periodically merged into that release branch; the release merge date
  is recorded here so version provenance is unambiguous.
- **Qwen3.8-27B support.** TP=1 and TP=2 execution was brought to a
  functional, measured state, including the fp16 dtype fallback and the
  long-context FA validation. See `DEVLOG-qwen38.md` and
  `DEVLOG-tp2-dense.md`.

## 2026-08-23

- **W2: Qwen3.5-35B MTP2.** The speculative-decoding rails transferred to the
  MoE model without code changes. Graph serving measured 89.9 vs 76.2 t/s
  (1.18x) and eager serving 45.5 vs 24.5 t/s (1.86x), with 80.4% acceptance.
  MTP2 is the recommended 35B configuration; MTP3 was not viable. See
  `DEVLOG-moe-spec-decode.md`.
- **W4: skinny fp16 M=5..16 GEMM.** The weight-row-parallel GEMV extension
  shipped behind `VLLM_GFX906_SKINNY_M16` (default off). The original
  all-decode-step gain estimate was falsified by the x-L2 re-read bound, but
  concurrent decode improved by 14.5% on 35B and 6.1% on Qwen3.8-27B; a
  30-repetition soak passed. See `DEVLOG-fp16-skinny.md`.
## 2026-08-24

- **FA gather-buffer lifecycle fix.** Capacity-width reuse and a
  per-generation capture flag replaced the unbounded retired-generation
  behavior. The Qwen3.8 250k prefill completed with needle retrieval and
  flat decode A/B; the fix was merged in `21c69a8ead`. See
  `DEVLOG-fa-attention.md`, `oom-256k-prefill.md`, and
  `plan-gfx906-fa-fix.md`.
- **Release-branch sync.** The completed FA lifecycle fix and the W2/W4/C2-V
  results were merged from `gfx906/main` into `gfx906/v0.28.0rc2` on
  2026-08-24 (`7e4567053e`), keeping the release branch aligned with the
  feature line.
- **Final-build records.** The dense and MoE serving numbers were restamped
  on the max-ilp split build, including the MTP and skinny-GEMM comparisons.
  See `README.md` and `DEVLOG-spec-decode.md`.

## 2026-08-25

- **Ornith asymmetric compressed-tensors W4A16 support.** The stored-int8
  zero-point checkpoint was admitted through the gfx906 oracle and repacked
  safely; no kernel change was needed. The model reached 65.03 t/s versus
  3.50 t/s on the Triton arm. The Triton `has_zp` performance problem remains
  an open portability/kernel issue, while the supported gfx906 path is
  complete. See `DEVLOG-ornith-wna16.md`.
- **TP CPU-spin mitigation.** The HIP blocking-sync shim was added after the
  TP stuck-thread investigation. Two independent ROCR-Runtime fixes were
  documented as unmerged upstream candidates; they remain in
  `moe-decode-roadmap.md`. See `cpu-stuck-threads.md` and
  `moe-decode-roadmap.md`.

## 2026-08-26

- **W1: mixed-request GDN decode peel.** Non-spec sequences in spec-mixed
  batches no longer take the expensive one-token chunk path. The 27B
  two-request serving A/B improved from 55.60 to 59.35 t/s (+6.7%), and the
  review follow-ups were closed before merge. W1 was merged into
  `gfx906/main` as `5b15152431`. See `DEVLOG-gdn-mixed-decode.md`.
- **L3 n-gram proposer follow-up closed.** The proposer-cost battery and
  revalidation did not justify replacing the CPU proposer; GPU n-gram remains
  rejected for draft-quality reasons. See `DEVLOG-spec-decode.md`.
- **Ornith review hardening.** The shared `g_idx` gate, fail-closed qzeros
  repack checks, and stored-zero-point backend capability set were merged as
  `d160fb2ad0`. See `DEVLOG-ornith-wna16.md`.
- **Upstream release `v0.28.0`.** The final upstream tag was merged into the
  gfx906 fork on 2026-08-26 as `a4cb86c4aa`, after `v0.28.0rc2` had already
  been merged into the gfx906 release branch. The merge brought the three
  upstream rc2-to-final commits listed in that merge commit.

---

## Carried from the QSA-FN (0.29) line — `CHANGELOG.md`

The Qwen3.8-Flash-Next/QSA work landed on this line from `gfx906/qsa-fn` (merged 2026-09-27). This section is that branch's content for this file, kept verbatim so the records survive the union; the entries above are this line's own.

## 2026-09-21 (Qwen3.8-Flash-Next serves on gfx906 — tester report; PR #2 under review)

- **The FN-8 gate is met.** An external tester (4× MI50 32 GB, PCIe-only, no XGMI)
  served `Qwen3.8-Flash-Next` with our fp16 QSA patches: `TP=4`, fp16,
  `max-model-len=147456`, `max-num-seqs=3`, MTP k=3, piecewise cudagraphs —
  **46.8 t/s at B=1**. First end-to-end result on the real checkpoint; the fp16
  enablement (QSA-FN-1) is confirmed on the model, not just in kernel probes.
- Their measurements add two knobs we did not have: **drafter CUDA graphs cost
  2.3 % of the KV cache and are worth +6 %** on MTP k=3 here, and `enforce_eager`
  inside `--speculative-config` is a **no-op on this build** (read only by the
  legacy `v1/spec_decode/` proposer, never propagated to the draft `ModelConfig`).
- They also hit a **boot-time segfault under MTP + graphs** that we have never
  reproduced: death inside `gc.collect()`'s traversal, i.e. GC as the victim of an
  unidentified heap corruption (their suspect: a stale tvm_ffi torch-C-DLPack
  addon keyed without a torch-version hash). `gc.freeze()`+`gc.disable()` do not
  prevent it; shadowing the `gc.collect` attribute for the region does. Recorded as
  **OPEN** — the guard is a mitigation.
- Their PR #2 is reviewed in [`REVIEW-pr2-qsa-fn.md`](REVIEW-pr2-qsa-fn.md):
  merge-worthy in parts, with three required edits (invert the drafter-graph
  default, gate the PLE host-table replacement, trim the essay comments) and two
  drops (their `.cu` clang-format commit, their `degradation.md` ops rows).

## 2026-09-18 (two gated wins go default-on; stale-verdict sweep)

Kevin's decision after the 0.30.0 review (`MERGE-0.30.0-review.md`) established
that both were gated wins awaiting a flip, not parked code.

- **`VLLM_GFX906_SKINNY_M16` is now default-on** (`=0` is the kill switch).
  Record: `DEVLOG-fp16-skinny.md`, VERDICT SHIPPED — 35B MoE N=8 graph **191.0 vs
  166.9 t/s (+14.5 %)**, 27B (Qwen3.8) N=8 **104.2 vs 98.2 (+6.1 %)**, 27B N=4
  control flat (−0.6 %, flag inert), correctness + per-shape 2–7.5× PASS, and a
  passed 30-rep × 2-model soak. Covers the M=5..16 spec-verify / 5–16-seq
  concurrent-decode regime, which was falling back to the M-invariant Triton
  skinny path.
- **`VLLM_GFX906_QUANT_LAYER0_MOE` (C4) is now default-on** (`=0` is the kill
  switch). Record: `DEVLOG-c4-layer0-quant.md` — GO 2026-09-01 with every gate
  passed: unit 8/8; PPL 15.9531 → 15.9929 (Δ +0.04 against a 0.5 gate); greedy
  serving fingerprint bit-identical; serving A/B 84.95 → **87.51 t/s (+3.0 %)**
  against a ~1.8 % noise floor; ~1.5 GiB returned to graph capture.
  **Re-measured after the flip on the house reference workload** (Qwen3.5-35B-A3B-AWQ,
  `_bench_gfx906.py` pp2048/tg256, 4 samples, mclk 1000): **59.79 t/s**
  (59.76–59.84) vs **58.40** before the flip = **+2.4 %** — so future 35B numbers
  must be compared against the new baseline, not the 57.97–58.36 band.
  **Quality, same build and prompt set (`VLLM_GFX906_QUANT_LAYER0_MOE` ON vs the
  `=0` kill switch): PPL 15.9361 vs 16.0169** (359 tokens, 0 top-20 misses in both;
  Δ 0.08 in the ON-better direction, while the 2026-09-01 pair differed 0.04 the
  other way ⇒ the delta is at the probe's resolution, not a quality signal). Layer 0's
  experts are quantized at load, which is a quality trade-off the checkpoint
  author did not make — accepted on the measurement above.
- **NH-4 stays** (`VLLM_GFX906_MAMBA_FUSED_GROUP_NORM`, default off) — Kevin's
  call after the review; its neutral A/B result is unchanged, and its stale
  "pending the serving A/B gate" comment is now accurate about what was measured.
- **Stale-verdict sweep (first pass):** the three stale comments above
  (NH-4, C4, SKINNY_M16 docstring) plus `DEAD-ENDS.md`'s "`VLLM_GFX906_FUSED_DRAFT`
  has no reader in-tree" (the A3 opt-in was revived 2026-09-14 with three tests)
  and a stale `V1`-pin recipe in `docs/gfx906/README.md` that contradicted the
  same file's DFL2-2 closure note. Everything else in the ~60-flag gfx906
  namespace reconciled (flag default ↔ comment claim ↔ recorded verdict). The
  `GFX906_FA_LEGACY_ALLOW_UNVERIFIED` reference in `test_gfx906_fa.py` is a
  deliberate guard that the removed override stays inert — kept.
  New standing rule: [`AGENTS.md`](AGENTS.md) merge-train rule 6.

## 2026-09-17 (QSA-FN-8 — tester bundle for Qwen3.8-Flash-Next on gfx906)

- **Packaged and validated the external gate.** `docs/gfx906/qsa-tester-build/`
  holds the README (quick start A, our branch; B, stock upstream 0.30.0; the
  smoke rig; what to report) and `make_patches.sh`, which derives four patches
  from the branch commits and bundles the Qwen3.8 tokenizer + tiny config for an
  offline smoke run. Artifact: `/local/tmp/qsa-tester-build/` + `.tgz`, with
  `BUILD-INFO.txt` (branch/head/base/generated).
- **Bundle self-validation:** on `gfx906/v0.29.0` the four patches apply clean and
  reproduce the branch's **63 shipped files byte-for-byte**; on
  `upstream/releases/v0.30.0` 0002/0003/0004 apply clean, and 0001 applies with
  `common/qsa_cache.py` excluded (upstream moved it 103/25 — five mechanical
  edits listed in the README; `git apply -3` does not resolve it). A scratch
  checkout of the base + the patches then **served the tiny rig from the patched
  tree** and ran the 1344/2016/4031-token prefix-cached sequence that used to
  fault (3/3 OK), with `test_qsa_amd.py` + `test_mamba_hybrid_model_state.py` 26
  passed and `test_qsa_reference.py` 19.
- Both serve recipes hardcoded this checkout's absolute path; they now resolve the
  repo root from their own location, so the bundle works from any tree.
- Excluded on purpose: int8 KV / int8-QK (QSA-FN-5/6).

## 2026-09-17 (QSA-FN-4 — the tiled QSA indexer lands, fp16-gated)

- **The one CDNA2 patch that ports is in**: `_qsa_mqa_paged_tiled_kernel` plus the
  uniform-request dispatch in `vllm/models/qwen4_exp/amd/ops/qsa.py`, with the gate
  the CDNA version lacks — `dot_is_native = q.dtype == torch.float16 or
  current_platform.supports_native_bf16`. The win is the *hardware* `tl.dot` (the
  tiling only amortizes the key load), so fp16 gains everywhere (gfx906
  `v_dot2_f32_f16`, CDNA MFMA) while bf16 must not enter on gfx906, where the dot
  is emulated per-scalar.
- **Gate (launch-regime, one MI50, uniform mapping, same inputs, interleaved
  ×3, two runs):** fp16 dispatch **1.33-1.35×** (4084/4061 vs 5439/5465 µs) with top-2048 agreement
  **1.00000** and logits NRMSE 1.28e-07; **bf16 stays on the per-row route**
  (6958 vs 6959 µs = 1.00×, versus the ungated CDNA kernel's 0.42×/16648 µs on the
  same shape).
- **Tests:** `tests/models/qwen4_exp/test_qsa_amd.py` **16 → 22 passed** — the
  scoring test now runs both routes against the torch reference, and the new
  `test_qsa_mqa_paged_route_selection` pins the gate with recording kernel
  stand-ins (fp16 uniform 64 rows → tiled; 32 rows or mixed requests → per-row;
  bf16 uniform → per-row). Non-regression in the same boot: `test_qsa_reference.py`
  19 passed, `test_config.py` 7, `test_ple.py` 10, `test_gfx906_fa.py` 104, PPL
  10.5472 (the recorded value for this build), MoE-35B reference workload 58.40
  t/s mean (58.38-58.41) — at the top of the recorded 57.97-58.36 band.
- **Not measured:** the indexer's serving share of prefill (the tiny rig cannot
  transfer shares) — the tester's report (QSA-FN-8) is the only end-to-end number.

## 2026-09-17 (V2-MAMBA-1 — V2 runner + mamba `align` mode no longer faults on gfx906)

- **A GPU memory fault on any hybrid model with heterogeneous KV-group block
  sizes is fixed**, found via Qwen3.8-Flash-Next (the V2-runner + prefix-caching
  combination its recipe needs). `MambaHybridModelState.add_request` seeded the
  per-request running mamba block column with `cache_config.block_size` instead
  of `cache_config.mamba_block_size`; the engine narrows the former to the
  **finest** KV-cache group's block size (4, from Qwen4Exp's
  `CircularBufferSpec` indexer group) while the mamba geometry stays 192, so a
  prefix-cache hit seeded a column ~57× too far out and the align pre-copy
  followed a stale block-table entry to a wild address. One line (+assert) now
  uses the mamba block size — the value the V1 path already used
  (`mamba_utils.py`: `block_size = mamba_spec.block_size`).
- **Not gfx906-specific and not an upstream fix re-derived:** `upstream/main`
  (fetched 2026-09-17) still has the seed line verbatim. Upstream *did* narrow
  the trigger the same day (only `prefix_cacheable` groups contribute to the
  min), which masks it for this model rather than fixing it.
- **Gates:** the tiny Qwen4Exp rig with prefix caching ON runs the sequence that
  used to fault, with greedy tokens **and** top-5 logprobs bit-identical to the
  prefix-caching-OFF arm (worst |Δ| = 0.000000, 6 prompt/rep pairs), with and
  without MTP k=3; 12/12 requests OK per arm. New unit test
  (`test_add_request_seeds_running_column_with_mamba_block_size`) fails on the
  pre-fix code with `assert 287 == 5`. The two CUDA-gated mamba kernel tests are
  now ROCm-enabled: **195 passed** on gfx906.
- **Supersedes** the `--no-enable-prefix-caching` workaround in the QSA-FN-2
  tester recipe (retired; kept there as a fallback for older builds). Record:
  [`DEVLOG-v2-mamba-align.md`](DEVLOG-v2-mamba-align.md).

## 2026-09-17 (QSA-FN-1 — Qwen3.8-Flash-Next / QSA runs in fp16 on gfx906)

- **The reported `NotImplementedError: Qwen4Exp QSA currently requires BF16` is
  fixed, and the fix is a 4.4× kernel win.** Admission of fp16 was added to every
  place the QSA path stores or reads a 2-byte float, through one shared pair of
  constants (`QSA_ACTIVATION_DTYPES` / `QSA_KV_CACHE_DTYPES`, `common/qsa_cache.py`):
  the attention/backend/Impl guards, `forward_qsa`'s Q/K/V check, `get_kv_cache_spec`'s
  storage check, the indexer's activation check and its raw/compressed key caches
  (now the model dtype), the `qsa_sparse_paged_attention` assert, and the three AMD
  `HyperConnectionConfig(params_dtype=…)` sites. `common/qsa_cache.py` is shared with
  the NVIDIA implementation, so its edits are dtype-*general* (`self.dtype` / model
  dtype) and **no NVIDIA/CUDA path changes**; the NVIDIA HC sites keep their bf16
  literal.
- **Why it was 4.4× and not a guard fix only:** gfx906 has no bf16 instruction, so
  every bf16 `tl.dot` lowers to scalar `v_fmac_f32` (+ converts) while fp16 lowers to
  `v_dot2_f32_f16`. Measured on MI50, launch-regime: sparse attention **26.5 ms fp16
  vs 116.5 ms bf16** (interleaved reps, stable to ±0.2 %); the per-row indexer kernel
  **5426 vs 6928 µs**.
- **Gates (all green, one boot):** `test_qsa_amd.py` **9 → 16 passed** and
  `test_qsa_reference.py` **16 → 19 passed** — the sparse-attention reference test is
  now parametrized `bf16|fp16`, plus a new indexer-scoring reference test
  (`qsa_mqa_paged`, bf16+fp16) and a new int64-MRoPE-packing test for the shared
  state caches; FA suite **104 passed**; in-process PPL (Qwen3.8-27B-AWQ-INT4, fp16,
  359 tokens) **10.5472, 0 top-20 misses**, identical to the value recorded for this
  build; MoE 35B `_bench_gfx906.py` pp2048/tg256 4 samples **58.31/58.35/58.29/58.23
  t/s** (mean 58.30, mclk 1000) — parity.
- **Not covered — the end-to-end gate is QSA-FN-3's.** No served request has exercised
  the fp16 path (the model needs ~60 GB of W4A16 weights), so the config-shape
  plumbing is uninstantiated and the indexer's compress/store/selection kernels are
  untested in fp16. Record + limits:
  [`DEVLOG-qwen38-flash-qsa.md`](DEVLOG-qwen38-flash-qsa.md); pre-work evidence:
  [`RECON-qwen38-flash-qsa.md`](RECON-qwen38-flash-qsa.md).
- By-product baseline, first record: Qwen3.5-27B-AWQ in-process PPL **14.3750**
  (359 tokens, 0 top-20 misses) — a different model from the 10.55 band, so not
  comparable to it.

