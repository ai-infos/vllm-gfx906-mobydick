# gfx906 fork — 0.29.0-final release snapshot (2026-09-18)

**What this is.** The last snapshot of the fork's 0.29.0 line: `main`
(`kintegrated/main` = `gfx906/v0.29.0`) plus the non-QSA half of the
Qwen3.8-Flash-Next train. Tagged `gfx906/v0.29.0-final`; the code is frozen here
and all further work moves to the 0.30.0 base (`UP-3`) or to the QSA branch below.

## What this snapshot adds over the previous main

| item | change | measured |
|---|---|---|
| **V2-MAMBA-1** | mamba `align`-mode seed fix in `MambaHybridModelState.add_request` (used `cache_config.block_size`, now the mamba block size) — with V2 + prefix caching on, any prefix-cache hit could kill the engine with `Memory Fault Error … precopy_mamba_align_fused_kernel` | the prefix-cached request sequence that faulted now runs; greedy tokens **and** top-5 logprobs bit-identical to the prefix-caching-off arm; new unit test fails pre-fix (`assert 287 == 5`); mamba suite **195 passed** on gfx906 (the two CUDA-gated kernel tests are ROCm-enabled here) |
| **`VLLM_GFX906_SKINNY_M16` default on** | W4 skinny fp16 M=5..16 GEMV rail (`=0` kill switch) | 35B MoE N=8 graph **191.0 vs 166.9 t/s (+14.5 %)**, 27B (Qwen3.8) N=8 **104.2 vs 98.2 (+6.1 %)**, 27B N=4 control flat, 30-rep × 2-model soak passed (`DEVLOG-fp16-skinny.md`) |
| **`VLLM_GFX906_QUANT_LAYER0_MOE` (C4) default on** | load-time int4 quantization of the deliberately-unquantized layer-0 MoE experts (`=0` kill switch); ~1.5 GiB returned to graph capture | serving A/B **84.95 → 87.51 t/s (+3.0 %)**; house reference workload (35B, pp2048/tg256) **58.40 → 59.79 t/s (+2.4 %)**; PPL same-build ON vs OFF 15.9361 vs 16.0169 (0 top-20 misses both) |
| **Docs** | `DEVLOG-v2-mamba-align.md`, `MERGE-0.30.0-review.md` (31 conflicts, not 148 — per-file triage + the off-by-default inventory), corrected `DEAD-ENDS.md` FD-1 row, corrected `README.md` V1-pin recipe, `AGENTS.md` rule 6 | — |

NH-4 (`VLLM_GFX906_MAMBA_FUSED_GROUP_NORM`) stays **off**: its gate ran and was
neutral (+0.4 %, inside inter-arm noise) because the decode step is
MoE-GEMV-bound. Its comment now says so.

## Gates measured on this snapshot

One MI50 (GPU 0 unless noted), `~/env-rocm-7.14-gfx906.sh`, local editable `.venv`:

- `docs/gfx906/_bench_gfx906.py /data/models/QuantTrio/Qwen3.5-35B-A3B-AWQ`
  (`BENCH_EAGER=0 BENCH_GPU_UTIL=0.95 BENCH_SAMPLES=4 BENCH_PP=2048 BENCH_TG=256
  BENCH_MAX_SEQS=32`, mclk 1000): **59.77 t/s** (59.70 / 59.75 / 59.79 / 59.86) — the C4
  flip is the +2.4 % step over the 58.40 pre-flip baseline, and it reproduces the
  59.79 measured on the QSA branch, i.e. the backport is behaviour-identical.
- `benchmarks/kernels/gfx906/ppl_probe.py` — 35B PPL 15.9361 (359 tokens, 0 top-20
  misses); the C4 kill-switch arm 16.0169.
- Suites: mamba **195 passed**, `test_mamba_hybrid_model_state.py` +
  `tests/kernels/moe/test_c4_layer0_quant.py` **12 passed / 3 skipped**,
  `tests/kernels/attention/test_gfx906_fa.py` **104 passed**.

## What is *not* in this snapshot

The Qwen3.8-Flash-Next / QSA work lives on **`gfx906/qsa-fn`** (branched from the
pre-backport main; it contains everything here plus these), documented in
`DEVLOG-qwen38-flash-qsa.md` and shipped to the tester as
`qsa-tester-build/` (patch bundle, artifact at `/local/tmp/qsa-tester-build`):

- `QSA-FN-1` fp16 enablement — the reported `NotImplementedError: Qwen4Exp QSA
  currently requires BF16`, and a 4.4× kernel win on gfx906 (bf16 `tl.dot` is
  emulated per-scalar; fp16 lowers to `v_dot2_f32_f16`).
- `QSA-FN-4` tiled indexer, gated on a native `tl.dot` (**1.33–1.35×** fp16;
  **bf16 stays on the per-row route** — the ungated version measured 0.42×).
- `QSA-FN-2/3/8` the serve recipe, the tiny-config smoke rig, and the tester
  bundle. **Untested end-to-end here:** the real checkpoint (~60 GB W4A16 + a PLE
  ngram table) does not fit on 2× MI50, so quality and serving numbers are the
  tester's. The bf16 QSA arm is independently broken
  (`rocm_unquantized_gemm` dtype assert) — fp16 is the arm.

## Next

1. **`UP-1`** — upstream the mamba align seed fix (`main` and `releases/v0.30.0`
   still have the buggy line; upstream's newer narrowing masks the trigger for the
   Qwen4Exp case but leaves the seed wrong). Human-owned PR; §1 checks first.
2. **`UP-2`** — upstream the fp16 QSA enablement (a feature, not a bugfix: 0.30.0
   still declares `supported_dtypes = [torch.bfloat16]`).
3. **`UP-3`** — the `gfx906/v0.30.0` fork-merge train: 31 conflicted files
   (≈11 our live code, ≈10 upstream carries to take theirs, NH-4, glue) per
   `MERGE-0.30.0-review.md`; run the merge-prep stale-verdict sweep first
   (`AGENTS.md` rule 6).

Build/serve recipes, the reference-workload numbers and the model-support table
live in [`README.md`](README.md); the per-item history in
[`CHANGELOG.md`](CHANGELOG.md); the roadmap for what is still open in
[`ROADMAP.md`](ROADMAP.md).
