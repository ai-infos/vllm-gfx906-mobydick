# gfx906 -> upstream main merge strategy

Date: 2026-08-13

## Rationale (investigation)

Upstream main dropped gfx906 from aiter support: aiter is built with
AITER_ROCM_ARCH=gfx942;gfx950, and gating is `get_cdna_version() > 2` + a
separate RDNA4 path. The fork's own 0.26 build already compiled aiter only for
gfx942/950, so on gfx906 the fork's "rocm_aiter_mla_sparse" backend already
falls back to Triton/torch (reference_mla_sparse_prefill / use_fp16_sparse).

=> Most of main's changes to kvcache dtype / minimax / MoE / MLA prefill are
   vLLM structural refactors converging on the CDNA3+/RDNA4 substrate, plus
   real aiter-scope tightening. Not gfx906-targeted removals.

## Strategy: build on main's newer structures, keep only minimal gfx906 kernel deltas

ADOPT main's structures wholesale for the big refactors:
- FusedMoEFactory MoE layer (also gives bailing_moe_v3 / Ling-3 for free)
- MLA prefill chunking (plan_mla_context_chunks), gather kernel, minimax module layout
- aiter scoping to CDNA3+/RDNA4

RE-APPLY only the true gfx906 kernel deltas as thin adaptations on top:
- fp32->fp16 casts before tl.dot in Triton minimax/sparse kernels
- waves_per_eu / LDS-safe num_warps/num_stages launch params
- fp32-KV / fp16 IndexerKVDType IF still needed (verify; main's fp32 GEMV path may supersede)
- ExllamaLinearKernel priority, gptq_shuffle_awq_qweight, cp_gather_indexer_k_cache_fp16
- the gfx906 NCCL workaround

## Verification
Rebuild native ext + smoke test on gfx906 docker (Qwen3-0.6B / AWQ / MRv2 / MoE),
plus MiniMax-style fp32-KV decode to settle the IndexerKVDType question.

## Validation status on the pytorch-2.11 / ROCm-7.2.1 toolchain (2026-08-13)

The main merge (v0.27.2rc1) builds and imports cleanly on the gfx906 docker
(pytorch 2.11 / ROCm 7.2.1): on_gfx906 True, fp8==e4m3fn, no undefined symbols,
cross-type gather + fp16 indexer kernels compile. Ling-3.bailing_moe_v3 resolves.

However, runtime inference fails with `hipErrorIllegalState` in a BASIC kernel:
`rms_norm` (vllm/kernels/vllm_c.py -> torch.ops._C.rms_norm). This op worked
on vLLM 0.26.0 with the same toolchain, so it is NOT a gfx906 kernel bug and NOT
a graph-capture issue (it reproduces with enforce_eager, in the warmup forward).

Root cause hypothesis: upstream vllm main (0.27.x) is developed/built against
torch 2.13 + ROCm 7.14 (mixa3607/pytorch-gfx906:v2.13.0-rocm-7.14). Building
main's C++ with the older torch-2.11 / ROCm-7.2.1 produces subtly broken kernels
(rms_norm), even though it compiles. The torch-2.13 base lacks triton,
flash_attn, cmake/ninja, setuptools_rust, so validating main properly requires a
full rebuild (triton-gfx906 + flash-attn-gfx906 + vllm) on the 2.13 base via the
ML-gfx906/vllm-v2 recipe.

State: main merge fully resolved & committed; static (build/import) OK on
torch-2.11; runtime rms_norm failure pending the 2.13 toolchain rebuild.
