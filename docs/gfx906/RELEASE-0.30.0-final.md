# gfx906 fork — 0.30.0-final release snapshot (2026-09-26)

**What this is.** The fork's first release on the upstream **0.30.0** base:
branch `gfx906/v0.30.0`, cut from the 0.29.0-final `main` (`524ac6f2d6`) and
merged with upstream `v0.30.0` (`ced6857afa`) as merge commit **`8893a50e54`**
— 2017 files, **31 conflicted files**. Per-file resolution record: the merge
commit body; theme log: `CHANGELOG.md` 2026-09-24/26.

**Release basis — V1 model runner for peak throughput; V2 now works too.**
`REL30-1` (the fork's fused M=1 MoE align left its out-params uninitialized under
V2) is **fixed** — the fused align is now used only when V1 is explicitly selected.
Measured, house config: **V1 60.07 t/s** (fused align on) vs **V2 58.90 t/s**
(fused align skipped, `rc=0`, 0 faults). Both are correct; the fork's serve recipes
keep pinning V1 for the ~2 % edge.

## What this snapshot adds over 0.29.0-final

| item | change | measured |
|---|---|---|
| **Upstream 0.30.0 base** | 772 upstream commits; `gfx906/v0.30.0` merge train (31 conflicts, ~11 live-code hand-merges) | build rc=0; op schemas verified |
| **GPTQ act-order adopted-out** | #54809 removed group/dynamic activation ordering stack-wide; the fork adopted it and re-ported its **M=1 4-bit max-ilp** dispatch onto the new kernel skeleton | both fork model families validated (below) |
| **MiniMax-M3 gfx906 re-port** | upstream's indexer/backend rewrite with the fork's fp16 casts / launch kwargs re-applied; `#47042` chunked-continuation guard re-derived on the `seq_lens_cpu_upper_bound` replacement | 122 MiniMax-M3 kernel tests + routing / `_forward_mla`-fp16 gates (QSA-FN-14) |
| **`upstream-merge` skill** | merge procedure + lessons + `post-merge-sweep.sh` | — |

## Breaking changes the base brings

- **GPTQ group/dynamic activation ordering removed (#54809)** — `g_idx` is
  ignored; the Marlin/GPTQ/CPU/RDNA3 act-order kernels are gone. The validator
  now fails loudly instead of mis-dequantizing. No fork model uses act-order.
- **Env removals (#55353)**: `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`,
  `VLLM_MM_HASHER_ALGORITHM`, the `use_fp4_indexer_cache` alias, the ROCm
  `CUDA_VISIBLE_DEVICES` fallback, and the `seq_lens_cpu` /
  `num_computed_tokens_cpu` attention-metadata properties. The fork keeps a
  deprecated-env read for the first via `get_from_deprecated_env_if_set` (fork
  decision; upstream removed it).
- Scale-out endpoints are opt-in (`--enable-scale-out`); the `all` Mamba cache
  mode is deprecated (the fork uses `align`); YaRN is aligned with Transformers.
  Per-item applies / not-applicable check: `CHANGELOG.md` 2026-09-26.
- **New in this release: `REL30-1`** — V2 + 35B MoE + FULL-graph capture
  faults (see below). Not a change to V1 behaviour.

## Gates measured on this snapshot

- **35B house bench, V1** (`docs/gfx906/_bench_gfx906.py`,
  `BENCH_EAGER=0 BENCH_GPU_UTIL=0.95 BENCH_SAMPLES=4 BENCH_PP=2048 BENCH_TG=256
  BENCH_MAX_SEQS=32`, mclk 1000): **60.42 t/s** (60.446 / 60.444 / 60.404 /
  60.388) — the anchor number, ~0.9 % above the 0.29 house number
  (59.77–59.86).
- **PPL probe, dense** Qwen3.8-27B-AWQ-INT4: **10.5472** (359 tokens, 0 top-20
  misses; band 10.5516 / 10.5472) — V2, eager.
- **PPL probe, MoE** Qwen3.5-35B-A3B-AWQ: **15.9407** (359 tokens, 0 top-20
  misses; C4 band 15.9361–16.0169) on the final build — V2, eager, and with the
  `REL30-1` gate active it exercises the **generic** align (15.9840 was measured
  pre-gate with the fused align; the two are documented bit-equal for these
  shapes). Exercises the gfx906 MoE WNA16 path (`_process_weights_gfx906`
  10-tuple, `moe_gptq_gemm_gfx906`).
- **35B house bench on V2** (default env, gate active): **58.90 t/s**
  (58.904 / 58.976 / 58.800 / 58.880), `rc=0`, 0 faults, `Using V2 Model
  Runner` — so V2 is correct as well; V1 stays the recommended serving config.
- **TP=2 Qwen3.8-27B-AWQ-INT4**, MTP k=3, filler, prefix caching OFF, bt4096,
  util 0.82, capture `[4,8,12,16]`, maxlen 131072 (V1): 2k **79.1** · 64k
  **42.0** · 120k **40.7** t/s decode; TTFT 4.2 / 185.3 / 442.2 s. Standard
  `vllm bench serve` (random 2048→256): **52.9 t/s** @ TPOT 18.90 ms.
- **Suites**: `test_gfx906_fa.py` **104 passed**; mamba kernel suite
  (`test_memcpy_u64_tiled` + `test_precopy_mamba_align` + `test_mamba_mixer2` +
  `test_mamba_hybrid_model_state`) **200 passed / 6 skipped**; C4
  (`test_c4_layer0_quant.py`) **8 passed**; `test_moe_wna16.py` 47 passed;
  MiniMax-M3 kernels 122 passed / 13 skipped; MiniMax-M3 model+sparse 32
  passed / 3 skipped; `test_config.py` (attention/indexer/cudagraph) 43 passed;
  `test_gptq.py` 2 passed; `post-merge-sweep.sh` PASS.

## `REL30-1` — fixed in this snapshot (was: V2 + 35B MoE graph capture)

Under the default **V2** runner the 35B MoE faulted during capture with
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in the fork's
`moe_gemm_q4_kernel_gfx906`. Root cause: the fork's **fused M=1 align**
(`_moe_align_block_size_fused_m1` -> `_rocm_C.moe_align_block_size_m1_gfx906`)
left its out-params **uninitialized** on V2's compiled/captured path, so the M=1
GEMM indexed the expert tables with garbage ids (~1e9). M>=2 used the generic
align and were always fine.

Fixed by gating the fused align to an explicit V1 selection
(`envs.VLLM_USE_V2_MODEL_RUNNER is False`; the auto/`None` default also skips it,
so it is fail-safe). A `register_fake` for the align op was tried first and does
**not** fix it (recorded). Details and the bisect: `ROADMAP.md` `REL30-1`,
`degradation_details.md` 2026-09-26.

## What is *not* in this snapshot

- The **Qwen3.8-Flash-Next / QSA** work stays on `gfx906/qsa-fn` (merge-review
  sequencing); a 0.30.0 release ships without it.
- **MiniMax-M3-AWQ** cannot be loaded here (needs > 32 GB); its gfx906 paths are
  gated model-free (QSA-FN-14) but not weight-validated.
- **kimi-k3** is not runnable here (its `kda.py` merge is reasoning-only).
- V2 serving for the MoE model (`REL30-1`).

## Next (release actions)

1. Fast-forward `main` to the `gfx906/v0.30.0` tip.
2. Push `main` + the branch; tag **`gfx906/v0.30.0-final`**.
3. Publish the image.
4. Post-release: `REL30-1` (V2 MoE capture) under `DFL2-2`; the `U30-1..3`
   candidates from the release-notes cross-check.
