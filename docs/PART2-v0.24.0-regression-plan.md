# NOTE: see also PART2-v0.25.0-regression-plan.md if it exists
# PART2: v0.24.0 gfx906 regression triage & fix plan

Status: COMPILED + SMOKE-TESTED on gfx906 (MI60/MI50, ROCm 7.2.1, HSA_OVERRIDE_GFX_VERSION=9.0.6).
Remaining: broader kernel-path fuzz and long-context / fp8-KV / MLA-sparse validation.

This follows the established fork workflow: each version bump merges upstream and
then requires `[PARTx]` regression fixes because upstream refactors silently
clobber gfx906 kernels (see the v0.23.1 `PART1/2/3` commits restoring the fp8
path, `rocm_unquantized_gemm_impl`, and `rocm_aiter_mla_sparse`).

## Execution & accountability

- Per AGENTS.md: human must review every changed line and run the listed tests.
- Each fix lands as its own `[PART2]` commit with an attribution trailer.
- Do NOT jump to 0.25.0 until 0.24.0 is validated on gfx906.

---

## High-risk regression areas (grounded in actual merge diffs)

### R0. MLA sparse Triton kernels — `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`

Upstream 0.24 changed 69 lines here (fork-only file, auto-merged). Concrete changes:

- **int64 block-id** for the packed KV layout:
  `block_id = (slot_id // block_size)` -> `.to(tl.int64)` (32-bit overflow guard).
- **scale broadcast fix**: `score = einsum(...) * scale` -> `* scale.reshape(-1)`
  (per-KV-token scale broadcast alignment).
- **FNUZ/OCP fp8 split**: `IS_FNUZ_MAIN` vs `IS_FNUZ_EXTRA`; gfx942 uses FNUZ
  (`tl.float8e4b8`), gfx950 uses OCP. The C++ encoder writes FNUZ on gfx942; the
  Triton encoder writes OCP.
- **gfx942 aiter#3257 workaround**: `fp8_mqa_logits_gfx942` routed from
  `triton_fp8_mqa_logits` while gfx942, `rocm_aiter_ops.is_enabled()`.

**Why it matters for gfx906**: even though these branches are mostly gfx942/gfx950
guarded, the shared int64 block-id and scale-broadcast rewrites affect the
`indexer_k_*` / logits paths gfx906 uses. The fp8 write/read path must be rechecked
for gfx906. Note: `is_fp8_fnuz()` is True only for `gfx94` (MI300); gfx906 is NOT
gfx94, so gfx906 uses **OCP `torch.float8_e4m3fn`** (regular E4M3FN), not fnuz. The
FNUZ path in this merged code is gfx942-only. Verify gfx906's write/read encoding
consistency.

**Tests**: `tests/kernels/attention/test_attention_selector.py` (resolved in PART1),
plus a real gfx906 decode of a MiniMax M3 / DeepSeek V4 model. Watch for
"garbage output at 1k-15k+ tok" and infinite loops (the known gfx906 failure modes).

### R1. `indexer_k_quant_and_cache` stride semantics — `csrc/libtorch_stable/cache_kernels.cu`

Upstream changed the kernel from per-token `cache_stride = kv_cache.size(2)` to
per-block `cache_block_stride = kv_cache.stride(0)`:

- `dst_offset = block_idx * cache_block_size * cache_stride` -> `block_idx * cache_block_stride`
- bindings (`_custom_ops.py`, `torch_bindings.cpp`, `sparse_attn_indexer.py`) all
  updated consistently.

This is the **fp8 index-K cache write** used by gfx906 MLA. The offset math is
`block_idx * cache_block_stride + block_offset * head_dim + head_dim_idx`.
Verify numerically that for a `[num_blocks, block_size, ...]` cache,
`cache_block_stride == cache_block_size * per_token_stride` still holds so the fp8
scale index (`dst_scale_idx`) lands correctly.

**Tests**: stress a MiniMax M3 fp8 KV decode on gfx906; compare selections/scores
between `cache_stride != block_size*per_token` layouts.

### R2. MiniMax M3 fused KV insert — `csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu`

Merged with 221+/80- lines: added direct float->FP8 E4M3 conversion
(`rocm_cvt_float_to_fp8_e4m3` with `HIP_FP8_TYPE_OCP` vs `__hip_fp8_e4m3_fnuz`),
and a split-`scalar_t`/`cache_t` store path (`kAuto` = unquantized, fp8 cache uses
scaled-convert + identity scale). This is the fused kernel behind the PART1
`AMD/model.py` `indexer_kv_dtype` + `_fp8_kv` handling.

**Why it matters**: gfx906 fp32 kv / fp8 path. The `HIP_FP8_TYPE_OCP` vs fnuz
branch is the key correctness fork: gfx906 must match what the Python side
(`_indexer_kv_dtype_to_torch_dtype`) and the read kernels expect.

**Tests**: MiniMax M3 fp32 kv and fp8 kv decodes; verify fp32-dtype indexer writes
(introduced in the fork for gfx906, "no more infinite loops" fix) still hold.

### R3. Per-tensor quant API change — `vllm/_aiter_ops.py`

Upstream changed `rocm_aiter_per_tensor_quant` from returning
`(out, scale)` with `quant_dtype` arg to an out-param `(out, x, scale, is_dynamic)`
call with dynamic/static split (`dynamic_per_tensor_quant` vs `static_per_tensor_quant`).
The fork's `per_tensor_quant` wrapper was updated to the new signature.

**Why it matters**: gfx906 quantization/fp8. Verify the identity of the returned
scale tensor and that `static` vs `dynamic` (scale is None) dispatches correctly.

**Tests**: run an AWQ/GPTQ/moe quantized decode on gfx906; unit-check
`rocm_aiter_ops.per_tensor_quant` output dtype/shape/scale for both dynamic and
static scale.

### R4. Quantization dispatch consolidation (AWQ/INC) — PART1 already handled

PART1 adopted upstream `AutoAWQConfig`/`inc/` refactor and re-applied gfx906
routing. This is the one area with real code added (not just merge), so double-check
on hardware:

- AWQ gfx906 -> `AutoAWQLinearMethod` gptq_gemm path (fp16/fp32 dtypes, min-cap 60).
- INC gfx906 -> marlin disabled, gptq_gemm fallback.
- `moe_wna16.py` + fused_moe oracle `int_wna16`: gfx906 uses the config JSONs
  (`E=...,device_name=AMD_GFX906,...`) — verify these still resolve after the
  fused_moe refactor.

**Tests**: one AWQ model, one AutoGPTQ model, one INC model, and a DeepSeek MoE
(int4_w4a16 gfx906) on gfx906.

---

## Verification workflow

1. Static: `pre-commit run --all-files` on the touched files
   (per AGENTS.md — use `.venv`, not system python).
2. Build the ROCm wheel (out of scope here; needs gfx906 build machine).
3. Smoke test per model family on gfx906, watching for:
   - garbage output at long context (1k-15k+ tok)
   - infinite loops / hang (the MiniMax M3 symptom the fork fixed)
   - detection/selection mismatches (R0/R1/R2 fp8)
4. Perf sanity — the fork's purpose is speed; a build-time regression check on the
   aiter / unquantized gemm paths.

## Out of scope

- Full 0.25.0 migration (PagedAttention removal) — defer until 0.24.0 validated.
- Non-gfx906 platforms (leave upstream behavior untouched; the gfx906 guards are additive).

## Files to watch if a `[PART2]` source-fix is needed

(Validated clean below — kept as the target set should fp8-KV / MLA-sparse tests regress.)

- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`
- `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`
- `csrc/libtorch_stable/cache_kernels.cu`
- `csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu`
- `vllm/_aiter_ops.py`
- `vllm/models/minimax_m3/amd/model.py` (indexer_kv_dtype handling)
---

## Validation log (real gfx906, 2026-08-13)

Environment: host gfx906 card #0 (MI60/MI50), ROCm 7.14 host driver; container
`aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0` as toolchain
base (ROCm 7.2.1, torch 2.11, triton-gfx906 v3.6.0, flash-attn-gfx906),
`HSA_OVERRIDE_GFX_VERSION=9.0.6`, `PYTORCH_ROCM_ARCH=gfx906`, `HIP_VISIBLE_DEVICES=0`.
The merged `gfx906/v0.24.0rc0.x` branch was bind-mounted and rebuilt in-place with:
`pip install --no-build-isolation --no-deps -e .` + `VLLM_REQUIRE_RUST_FRONTEND=0`
(reusing the prebuilt `_rust_tool_parser.abi3.so`).

### Compile
- All native extensions rebuilt from the merged source against ROCm 7.2.1/gfx906
  cleanly: `_C`, `_C_stable_libtorch`, `_moe_C_stable_libtorch`, `_rocm_C`,
  `spinloop`, `cumem_allocator`. **R0/R1/R2 `.cu`/`.cpp` merge changes compile.**
- Editable install succeeded: `vllm 0.24.1.dev71+g5267f8350.rocm721`.

### Runtime (import + platform)
- `from vllm import LLM` OK; `RocmPlatform`, `on_gfx906() == True` confirmed.
- All C++ op tables load (`_C_stable_libtorch`, `_moe_C_stable_libtorch`, `_rocm_C`).
- **`fp8_dtype() == torch.float8_e4m3fn` (OCP)** on gfx906. Confirms R0/R2 note:
  gfx906 uses OCP E4M3FN, NOT fnuz (fnuz is gfx94-only via `is_fp8_fnuz()`).

### End-to-end generation (all via `python3 -c` to satisfy engine-core spawn)
- **Qwen/Qwen3-0.6B** (float16): "The capital of France is" ->
  "Paris, and the President of France is Gaston de Voltaire." (correct)
- **cyankiwi/Qwen3.5-9B-AWQ-INT8-INT4** (AWQ, exercises R4 gptq_gemm path):
  "The capital of Japan is" -> "Tokyo." (correct)

Both paths completed CUDA-graph capture, warmup, and inference on gfx906 without
crash or garbage output (the known gfx906 failure modes).

### Verified PART2 fixes already correct in the merge
- R3 `_aiter_ops.per_tensor_quant` dynamic/static split: loaded and ran (AWQ/dequant
  path exercised).
- R4 AWQ gfx906 gptq_gemm routing: correct output on a real AWQ model.

### Still needs coverage (next validation pass)
- MiniMax M3 / DeepSeek V4 **fp8-KV** and **fp32-KV** decodes (MiniMax M3 model not
  in the local HF cache; would need to pull it) — validates R1/R2 stride + fused
  minimax kernel end to end.
- MLA-sparse top-k selection consistency (R0) on a sparse model.
- Long-context (1k-15k+ tok) decode — the known gfx906 garbage-output regime.
- A `[PART2]` source-fix commit is NOT yet needed: the merge already validates.
  Revisit if the fp8-KV/MLA-sparse tests above regress.

---

## v0.25.0 merge + validation log (gfx906, 2026-08-13)

Bumped the fork to upstream v0.25.0 (702f4814) on branch gfx906/v0.25.0rc0.x,
based on the 0.24.0 branch. 0.25 highlights handled:

- PagedAttention removed (#47361): paged_attention_v1/v2 deleted; ops.h dropped
  their declarations but kept gfx906 gptq_shuffle_awq_qweight.
- MiniMax-M3: 0.25 split common/ops vs amd/ops index_topk; AMD vendor-routes
  via indexer.py. Ported gfx906 fp16 casts + _index_score_launch_kwargs into the
  amd/ops/index_topk.py copy (used by gfx906).
- sparse_attn: 0.25 routes ALL rocm to amd/ops (CDNA gfx942/950). Corrected to
  route gfx906 (MI50, not CDNA) to common/ops (keeps gfx906 LDS-safe kwargs).
- fused_moe/deepseek_v2/auto_gptq: kept gfx906 fp16/MI50-tile/marlin-disable
  paths; added 0.25 SWAP_AB/n_head_scale/allow_tile_padding.

Validation on real gfx906 (ROCm 7.2.1 container toolchain, HSA 9.0.6):
- Full C++ rebuild OK: vllm 0.25.1.dev92+g0d122f77c.rocm721.
- platform: RocmPlatform, on_gfx906 True, on_gfx942/gfx950/mi3xx False,
  fp8 == float8_e4m3fn (OCP).
- Standard (Qwen3-0.6B float16): "The capital of France is" -> "Paris."
  (spawn multiproc note: 0.25 enforces spawn; run tests via `python3 -c exec(...)`
   or set VLLM_WORKER_MULTIPROC_METHOD=fork so __main__ is importable).
- AWQ (cyankiwi/Qwen3.5-9B-AWQ-INT8-INT4, R4 gptq_gemm): "Tokyo." correct.
- First AWQ run was blocked by a stale orphaned VLLM::EngineCore hogging gfx906
  VRAM (left by a prior timed-out `docker run`); killed it and reran clean.

No [PART2] source-fix needed yet for the exercised paths. Remaining: MiniMax M3
fp8/fp32 KV, MLA-sparse, DeepSeek V4 long-context, and the new 0.25 MRv2 default
path on gfx906.

---

## v0.26.0 merge + validation log (gfx906, 2026-08-13)

Bumped the fork to upstream v0.26.0 (568afb3) on branch gfx906/v0.26.0rc0.x,
based on the 0.25.0 branch.

0.26 highlights handled:
- MiniMax-M3 AITER sparse paged attention + spec decode (amd/model.py
  use_aiter_sparse_pa + skip_index_topk + new named-arg
  fused_minimax_m3_qknorm_rope_kv_insert kernel API).
- MXFP8 GEMM for MiniMax-M3; csrc/rocm/skinny_gemms_int4.cu (new ROCm INT4
  integer GEMM) + ngram_embedding kernel; both compile on gfx906.
- KV-cache layout refactor: per KV-group backend (config/attention.py
  backend_per_kind) + k/v content-dim packing; KV_SCALE_MODE per-token scales
  in sparse_attn.py.
- Keep gfx906 indexer fp16/fp32 dtype + indexer_kv_dtype validator.
- fp32 lm_head head_dtype ROCm torch.mm fast path.

15 conflicts resolved (ops.h, config/attention, kv_cache_manager, int_wna16,
fused_moe/config, kernels/linear, compressed_tensors_moe_wna16, qwen3_dflash,
deepseek_v32/mtp, minimax_m3 amd/nvidia/sparse_attn, rocm_aiter_mla_sparse).

Validation on real gfx906 (ROCm 7.2.1 toolchain, HSA 9.0.6):
- Full C++ rebuild OK incl. new skinny_gemms_int4 + ngram:
  vllm 0.26.1.dev111+g8388aa8ea.rocm721.
- on_gfx906 True, fp8 == float8_e4m3fn.
- Qwen3-0.6B (float16): "The capital of France is" -> "Paris."
- cyankiwi/Qwen3.5-9B-AWQ-INT8-INT4 (R4 gptq_gemm): "Tokyo."

No [PART2] source-fix needed for the exercised paths. Remaining: MiniMax M3
fp8/fp32 KV + AITER sparse PA, DeepSeek V4 long-context, MRv2 default path on
gfx906.

---

## Testing the MRv2 (Model Runner V2) default path on gfx906

MRv2 is selected per-model at engine init. Relevant knobs:

- Default: non-MoE models => MRv2 (via `or not model_config.is_moe` in
  `config/vllm.py::use_v2_model_runner`). MoE models => MRv2 ONLY if the arch is
  in `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`
  (DeepseekV2/GraniteMoe/Inkling/LongcatFlashNgram/Qwen2Moe), else MRv1.
- Force one way: `VLLM_USE_V2_MODEL_RUNNER=1` (V2) / `=0` (V1).
- Confirm at runtime: worker logs `[gpu_worker.py] Using V2 Model Runner`.

Validation on gfx906 (0.26, ROCm 7.2.1 toolchain):
1. Selection probe: Qwen3-0.6B (non-MoE) -> MRv2; Qwen3.5-9B-AWQ (MoE) -> MRv1.
2. Qwen3-0.6B (MRv2 default): log "Using V2 Model Runner", correct output.
3. FORCED MRv2 on AWQ MoE (`VLLM_USE_V2_MODEL_RUNNER=1`): log "Using V2 Model
   Runner", correct output "Tokyo."  => validates MRv2 quantized-MoE path on
   gfx906 (the 0.25 #46535 / 0.26 MRv2 maturation).

To test any other gfx906 path under MRv2: rerun with VLLM_USE_V2_MODEL_RUNNER=1
and grep for the "Using V2 Model Runner" log line + output correctness. Non-MoE
gfx906 models are already on MRv2 by default.
