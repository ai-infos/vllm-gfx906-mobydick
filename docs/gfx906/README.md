# gfx906 (MI50/MI60) optimization hub

This fork optimizes vLLM for AMD **gfx906 (Vega 20)** — MI50/MI60, 60 CUs,
32 GB HBM2 (~800 GB/s), **no MFMA, no int8 matrix cores**. All numbers are
measured on a single MI50 with ROCm 7.14, torch 2.13, single request,
`pp=2048`/`tg=256`, `cudagraph_mode=FULL_DECODE_ONLY`.

## Release snapshots

- **[`0.30.0-final`](RELEASE-0.30.0-final.md)** (2026-09-26) — the first release on
  the merged upstream **0.30.0** base (31-conflict merge train): GPTQ act-order
  adopted-out, the MiniMax-M3 gfx906 re-port, and the 35B house bench at
  **60.07 t/s** on the **V1** runner (V2 also works now — `REL30-1` fixed: the
  fork's fused M=1 MoE align is V1-only, ~2 % slower on V2).
- **[`0.29.0-final`](RELEASE-0.29.0-final.md)** (2026-09-18) — the last snapshot of
  the 0.29.0 line: the V2 mamba `align` fix, `SKINNY_M16` and C4 default-on, the
  0.30.0 conflict review, and the merge-prep sweep rule. The Qwen3.8-Flash-Next /
  QSA work continues on `gfx906/qsa-fn`.

## Fork heritage

- This repository is the gfx906 vLLM port
  [**ai-infos/vllm-gfx906-mobydick**](https://github.com/ai-infos/vllm-gfx906-mobydick),
  itself based on [**nlzy/vllm-gfx906**](https://github.com/nlzy/vllm-gfx906),
  the original gfx906 port of vLLM.
- The custom Q8 FlashAttention kernels come from
  [**cassettesgoboom/gfx906-fa-vllm**](https://github.com/cassettesgoboom/gfx906-fa-vllm),
  vendored early in this fork's history and substantially extended for the
  decode path (see §What changed).

Test models:

- **MoE:** `QuantTrio/Qwen3.5-35B-A3B-AWQ` (40 layers, 256 experts × 8
  active, hybrid: 30 GDN linear-attn + 10 full-attention)
- **Dense:** `Qwen3.5-27B-AWQ` (64 layers: 48 GDN + 16 full-attention,
  Hq=24/Hkv=4/D=256)

Reference point: llama.cpp (Q4_K_XL GGUF, full offload) — **70.3 t/s decode,
806.5 t/s prefill** on the same hardware.

## Headline results

| workload | fork base (`gfx906/main`) | now | Δ | reference |
|---|---|---|---|---|
| MoE serving decode | 3.49 t/s | **67.39 t/s** | 19.3× | llama.cpp 70.3 (1.04× gap) |
| MoE prefill (pp=2048) | ~450 t/s | **~2140 t/s** | 4.7× | llama.cpp 806.5 (2.7× ahead) |
| Dense serving decode | 18.89 t/s | **25.60 t/s** | +35% | — |
| MoE concurrent decode (N=8) | 166.9 t/s (W4 off) | **191.0 t/s** | +14.5% | W4 skinny fp16 M≤16 (`VLLM_GFX906_SKINNY_M16`; **default on 2026-09-18**, `=0` kill switch; soak-verified; `DEVLOG-fp16-skinny.md`) |
| MoE serving decode, layer-0 experts quantized | 84.95 t/s | **87.51 t/s** | +3.0% | C4 (`VLLM_GFX906_QUANT_LAYER0_MOE`; **default on 2026-09-18**, `=0` kill switch; PPL 15.9531 → 15.9929, fingerprint bit-identical; `DEVLOG-c4-layer0-quant.md`). **Reference-workload effect:** the pp2048/tg256 4-sample house bench read **59.79 t/s** after the flip vs 58.40 before (+2.4 %); **re-measured on the 0.30.0 line, V1: 60.42 t/s** (60.446/60.444/60.404/60.388) |

Correctness gates: PPL on a fixed 442-token probe — MoE band 6.6817–6.6942,
dense band 6.6993–6.7197; on the 0.29.0 line the in-process probe is
**bit-identical across V2 / V1 / the 0.28 line (10.5516)**. Kernel suites: **97 FA
tests** (2026-09-15, default config) and 43/43 MoE GEMM (2026-08-24, not re-run
since).

**Release basis — 0.30.0 line (2026-09-26).** The upstream base defaults to Model
Runner **V2**; the fork's fused M=1 MoE align is now **V1-only** (`REL30-1`
fixed), so V2 is correct but ~2 % slower on the 35B (58.90 vs 60.07 t/s) and the
serve recipes keep pinning **V1** (`VLLM_USE_V2_MODEL_RUNNER=0`). V2 remains the
default for the small models (Qwen3.8-27B dense, and the PPL probes above run V2);
Muse-Glimmer's V1 pin was lifted 2026-09-16 and V1 is removed upstream in
**0.32.0**. VIT-1 (ViT attention on the custom FA) is on by default. See ROADMAP
`REL30-1` / `DFL2-2` / `GEMMA4-1` / `MUSE-1`.

## Model support status (single MI50, MI60 numbers similar)

All numbers: serving decode t/s, graph mode, pp=2048/tg=256, single
request (4 samples) unless noted. Recipes: §Bench recipes +
`DEVLOG-spec-decode.md` (spec-decode arms).

**Runner (0.29.0):** every model below runs the V2 model runner, including
**Gemma-4** (V2-validated 2026-09-15) and **Muse-Glimmer** (V2-validated 2026-09-16:
TTFT parity, decode −1.8 % @2k / −1.0 % @8k vs V1, `DEVLOG-muse-glimmer.md`). Upstream
removes the V1 runner in **0.32.0**; no model here is pinned to it any more
(ROADMAP `DFL2-2`, `GEMMA4-1`, `MUSE-1`).

**Prompt format — read before gating any model.** *Instruction-tuned* checkpoints answer
only in their own chat template; fed **raw text** (a `/v1/completions` continuation, or a
raw-text PPL probe) they emit fluent garbage that looks like a broken model or kernel.
This cost a full investigation on 2026-09-15: **Gemma-4-*-it scored PPL 84261 with 362 of
363 top-20 misses on raw text, while answering `Paris` correctly and confidently
(first-token logprob 0.00) through its chat template** — and the garbage reproduced
byte-identically on the 0.28 image, so nothing had regressed. Muse-Glimmer behaves the
same way (raw text: 362/363 misses; templated: correct). The Gemma-4 row below already
carried this caveat; it is now *enforced* by the harnesses.

| model class | raw-text probe / generation | chat template |
|---|---|---|
| Qwen3.5/3.8 dense + MoE, Nemotron 3.5 Lightning, Ornith | **valid** — the recorded reference bands were measured this way | fine (and correct for real traffic) |
| **Gemma-4-*-it, Muse-Glimmer** | **invalid: garbage that is not a defect** | **required** |

**The PPL probe cannot gate these models at all** — not even templated: its protocol scores
the *user's* tokens, which an instruct model is not trained to model (Gemma-4: raw text
84261, templated **1278491** — both artifacts). Use the **templated generation gate**,
`benchmarks/kernels/gfx906/ift_chat_gate.py` (greedy continuation + first-token top-k
logprobs per prompt; run it twice and compare text/logprobs), or a serving A/B.

Enforcement: `benchmarks/kernels/gfx906/ppl_probe.py` warns loudly when the tokenizer has
a chat template while `BENCH_CHAT_TEMPLATE` is unset, and renders prompts through the
template when it is set — while telling you those numbers are still not a gate for IFT
checkpoints;
`_bench_gfx906.py` records `prompt_form`/`has_chat_template` in every row and warns that
**tokens/s is a speed measurement, not a correctness gate** — Gemma-4 sat in this table as
"supported, 67.79 t/s" for weeks without ever having been gated, which is how the trap
survived.

| model | status | decode t/s | prefill t/s | notes |
|---|---|---|---|---|
| **Qwen3.5-35B-A3B-AWQ** (MoE) | **flagship, optimized** | **67.39** (record; band 65.3–67.0; final-build restamp 66.1, 2026-08-24) | **~2140** | full custom stack (W4A16 MoE GEMM + Q8 FA); 19.3× over fork base; llama.cpp parity on decode, 2.7× ahead on prefill |
| ↳ same, **N=8 concurrent decode** | W4 (`VLLM_GFX906_SKINNY_M16=1`, soak-verified) | 191.0 (baseline 166.9; **+14.5 %**; soak 189.9 ± 0.4; final-build restamp 192.9/194.0, 2026-08-24) | — | first N=8 record for this model (off arm = C2-V t1n8 steady); skinny fp16 M=5..16 kernel, M-dependent gate |
| ↳ same, **MTP k=2 spec decode** | recommended spec config (35B, W2) | 88.6 (1.16× vs 76.7 greedy graph; 1.83× vs eager 44.9) | — | **final-build restamp 2026-08-24** (the pre-W4 re-measure debt): 78.7 % acceptance, 1.57 tok/step; record 89.9 (1.18× vs 76.2; 80.4 % / 1.61 tok/step, pre-W4 build) |
| **Qwen3.5-27B-AWQ** (dense) | **well supported, optimized** | **25.60** (official-harness record; 27.99 no-spec on the current max-ilp split build, in-process metric) | ~257 (chunked) | GEMV + CUSTOM FA + max-ilp; serving needs `--gpu-memory-utilization 0.93` |
| ↳ same, **MTP k=2 spec decode** | recommended spec config | **39.4** (1.41×; 1.50× no-max build) | ~250 (neutral) | 90.9% draft acceptance, 1.82 tok/step; `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` |
| ↳ same, ngram-3 | works, weak | 28.0–28.9 (1.0–1.09×) | — | agentic prompts only break even; MTP preferred |
| **Gemma-4-26B-A4B-it-AWQ-4bit** (MoE) | **well supported, optimized; V2-validated 2026-09-15** | **67.79** (speed) | — | no-zero-point W4A16 expert kernel (`gfx906/gemma4-moe-nzp` work, 1.79× over Triton); **chat template required (thinking model)** — raw-text probes/generation return garbage (see the prompt-format note above); templated V1/V2 parity is exact; PPL/prompt_logprobs unreliable on this model — gate on coherent text + logprob A/B |
| **cyankiwi/Ornith-1.5-35B-A3B-AWQ-INT4** (MoE VLM) | **supported (2026-08-25, in main via `gfx906/moe-ct-asym-zp`)** | **65.03** (A/B mean; band 64.995–65.079; decode-only 81.1) | TTFT 0.77 s @2048 | first **asymmetric** (stored int8 zp) CT W4A16 checkpoint: oracle gate + pass-through zp repack, no kernel change; 18.6× over the Triton arm (3.50) — but the Triton W4A16 `has_zp` branch is pathologically slow on gfx906 (267 ms/tok, both zp layouts) — `DEVLOG-ornith-wna16.md`; PPL 16.67 gfx vs 16.45 triton (fp16-noise band); class-parity with the flagship 67.39 |
| **cyankiwi/Muse-Glimmer-30B-AWQ-INT4** (dense hybrid: GDN + full + sliding-2048, CT W4A16) | **supported (2026-08-27; in main 2026-08-28; TP=1 + TP=2). V2-validated 2026-09-16** (TP=1, util 0.90, maxlen 8192, greedy, chat template, filler body — the first non-ngram-filler serving numbers: **27.1 t/s decode @2k / 26.7 @8k, TTFT 4.77 / 11.74 s**; V1 measures 27.6 / 26.9, i.e. V2 is at TTFT parity and −1–2 % decode, KV pool 53 k vs 68 k tokens) | TP=1 in-process: **27.90** @B=1 all-CUSTOM window FA (1.59× vs hybrid 17.54) · **20.53** @B=4 (1.23× vs 16.75). TP=2 ngram n=5 serving (repetitive filler, **100 % acceptance ceiling**): bt2048 (boot K): **114.6** @2k/256 · 112.0 @2k/512 · **79.1** @8k/256 · 79.2 @8k/512 · **57.0** @16k/256 · 56.8 @16k/512 · B=4 @2k/256: **45.3** aggregate (~11.3/req); bt4096 (boot L, post q_pad fix + M1 gather clip): **111.5** @2k/256 · **~99** @8k/256 · B=4 @2k/256: **46.7** aggregate · real-prompt checks ~11.5/req (1–4 parallel) | TP=2 prefill (prefix-cache WARM): bt2048 **542** @2k · **491** @8k · **438** @16k; bt4096 (boot L): 496.9 @8k (cold first-chunk 452.4); TP=1 in-process prefill baseline: **~240 t/s** (32k prompt pass 135–137 s, bt4096 — the TRUE TP=1 rate, re-measured on the fresh boot of 2026-08-29 after boot M's ~2×-vs-TP=2-records scare; the ~450–540 t/s figures are TP=2, prefill scales ~2× with TP; `degradation*.md` 2026-08-29 rows); TP=1 gates: pp2048 tg256, 4 samples, prefix cache off | **Working TP=2 example** (boot K, 2026-08-27): `HIP_VISIBLE_DEVICES=0,1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE vllm serve <snap> --served-model-name Muse-Glimmer-30B --tensor-parallel-size 2 --dtype float16 --max-model-len 131072 --max-num-seqs 4 --max-num-batched-tokens 4096 --kv-cache-memory-bytes 6442450944 --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":2}' --compilation-config '{"cudagraph_capture_sizes":[6,12,18,24]}' --enable-auto-tool-choice --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer --generation-config auto` — pool 848–904k tok (6.5–7× the 128k max), weights 12.7 GiB/GPU, graphs ~0.9 GiB; 256k ctx: add `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (boot J, 1.36M-tok pool). First-prefill memory: boot J/K's "bt4096 OOMs the first 4096-chunk prefill" was the **per-impl q_pad bug** (52 impls × 256 MiB [num_seqs,Hq,Sq_pad,D] fp32 grown per impl on first prefill — DEVLOG-muse-glimmer.md round 4; fixed 2026-08-27, ClassVar) — **bt4096 re-validated clean on boot L** (cold 452 t/s @8k, 8.7 GiB headroom/GPU); the byte cap is still required (at util 0.82 the pool is sized from that profile — the engine logs the correction itself: 7.59 GiB fits the line, 9.02 was allocated). The filler is 100 % ngram-accepted (acceptance length 6.0) — decode numbers are a ceiling, not real-text. 52 layers (13 full + 39 sliding window-2048): sliding-window Q8 FA (window arg, both kernel copies) + direct-paged split-K + Phase C clip (direct-paged B≥2 decode dispatch; `GFX906_FA_LEGACY` orthogonal); TP=1 bench gates need an explicit 0.75 GiB KV cap + bt1024; `muse_glimmer` tool/reasoning parsers; records: `DEVLOG-muse-glimmer.md`, `degradation*.md` boot J/K |
| **Qwen3.8-27B-AWQ-INT4** (dense) | **fully functional (TP=1 + TP=2)** | MTP k=2 TP=2 (2026-08-24 final): **59.2** @2k / 44.9 @8k / 25.2 @32k · **long-context post kv_split fix (2026-09-03): 37.95 @64k / 29.88 @96k / 25.70 @120k** · **depth default since 2026-09-11: k=3** (mixed corpus: 27.44 @64k / 24.76 @120k vs k=2 27.30/22.65 = +9.3 % @120k, tie @64k; k=4 a loss on real payloads — `DEVLOG-mtp-depth-matrix.md`) · **agentic Python-coding headline (our CAT-1 corpus, 2026-09-13): 33.3 @64k / 25.0 @120k t/s with k=3 + CAT-1 (greedy 19.8/13.2 = 1.68×/1.89×)** · greedy TP=2: 40.8/38.1/30.5/24.1 @2k/8k/32k/64k · 28.62 @4k TP=1 (record) · 104.2 (N=8, W4 on) | — | `--dtype float16` required (auto-bf16→fp16 fallback landed); **kv_split clamp fix (`a6ff64a71b`) removed the old "MTP < greedy beyond ~20k ctx" live-ctx tax — MTP k=2 now ≈2× greedy at 64k+** (pre-fix 15.95/11.19/9.18 @64k/96k/120k vs post-fix 37.95/29.88/25.70; §Long-context decode with MTP below); MTP k=2 41.41 TP=1 (record, 2026-08-23; lifecycle fix byte-identical in graph mode); N=8 needs `--gpu-memory-utilization 0.90` (64 layers, FA KV 655 KB/token); TP=2 needs the official amdgpu DKMS driver + trimmed capture `[1,2,3,4]`; 445k-token KV pool — **256k context validated** (FA gather fix 2026-08-24, `oom-256k-prefill.md` §9); non-deterministic at temp=0 (token-identity gates unusable); records: `DEVLOG-tp2-dense.md` S1–S9, `DEVLOG-masked-fa.md`, `DEVLOG-qwen38.md` |
| ↳ **same, post-0.30.0 re-measure (TP=2, 2026-09-24)** | **MTP k=3 (current default) + `GFX906_FA_LEGACY=0` KV read** | **2k 79.1 · 64k 42.0 · 120k 40.7 t/s** (filler, prefix caching OFF, bt4096, max-seqs 4, util 0.82, capture `[4,8,12,16]`, maxlen 131072; TTFT 4.2/185.3/442.2 s) | — | PPL **10.5472** (359 tok, 0 top-20 misses) on the same build; standard `vllm bench serve` random 2048→256 **52.9 t/s** @ TPOT 18.90 ms (acceptance 59.9 %/2.80); KV pool 365,661 tok; logs `/local/tmp/bench_lc_030.log`, `/local/tmp/bench_2k_k3.log` |
| **Qwen3.6-27B / 3.6-35B-A3B** (fp16) | **not supported** | — | — | 52/67 GB fp16 checkpoints do not fit a 32 GB card; 3.6 GGUF only used as a llama.cpp reference point |
| small AWQ models (e.g. Qwen3.5-9B-AWQ, 0.8B) | supported | — | 590–1483 (9B, eager) | fine on ≤0.85 util; FA prefill benchmarks in top-level README |


## Contents of this directory

| file | what it is |
|---|---|
| `README.md` | this hub: changes, numbers, recipes, knobs |
| `CHANGELOG.md` | chronological record of completed roadmap work and release merges |
| `DEAD-ENDS.md` | one-pass index: hypothesis → gate → verdict → commit for what was tried (grep-able) |
| `_devlog-template.md` | the `VERDICT:`/`HYPOTHESIS:`/`GATE:` devlog convention + worked example |
| `DEVLOG-*.md` | topic-specific experiment records; see the changelog and roadmap entries for the index |
| `DEVLOG-int8-transfer.md` | int8-transfer session log: checkpoint fp16-mass scan, P1/P2 probes, the `v_dot8_i32_i4` scoping question |
| `int8-investigation-qwen.md` | gfx908 int8-vllm fork transfer analysis (T0–T5) + the P1/P2 probe records that gated it (source of the `INT8` rows in `DEAD-ENDS.md`) |
| `ROADMAP.md` | the single active queue: open work priority-ordered (Tier 0–2, onboarding + upstream queues) |
| `REFRIGERATOR.md` | parked/shelved items, each with its reopen gate |
| `running.md` | how to run/build/bench: local venv (canonical) + docker images |
| `_bench_gfx906.py` | end-to-end pp/tg serving bench harness (BENCH_* env knobs) |
| `_pp_bench.py` | prefill/decode split harness |
| `latency-hiding.md`, `lds-layout.md`, `dequant-instructions.md` | measured gfx906 ISA facts (kernel-writing guide, linked from AGENTS.md) |

## What changed vs `gfx906/main`

### 1. Custom Q8 FlashAttention backend (`CUSTOM`)
`vllm/gfx906_fa/`, `csrc/gfx906_fa/` — vendored from
`cassettesgoboom/gfx906-fa-vllm`, built into the wheel when gfx906 is a
target arch (`CMakeLists.txt`, `setup.py`), and the **default** attention
backend on gfx906 (`vllm/platforms/rocm.py`). Escape hatch:
`--attention-backend ROCM_ATTN`.

- Q8_0 KV quantization in-kernel (LEGACY path) or fused during the paged-KV
  gather (`GFX906_FA_FUSED_QUANT`, default on, bit-equal).
- **B=1 decode parallelism**: GQA head-packing (NC2) + KV split with
  split-combine — 245 → 58.3 µs/layer @Sk=2176 (4.2×). Fixed three vendor
  bugs en route (null-mask deref, NC2×prefill OOB guard, OOB-tail masking).
- **NC2 fail-closed**: only NC2∈{1,2,8} are instantiated; invalid explicit
  values error, default 8 auto-downgrades (8→2 when ratio%2==0, 8→1 for MHA).
- kv_split clamped to 1 for prefill (seq_q>2).
- Native **BSHD** output (no transpose copy); decode per-layer copy pile cut
  7→2 (dedicated decode q-pad buffer, fused fp16→fp32 casts, deferred
  `cu_seqlens_q.to(long)`).
- Direct-paged decode path for B≥2/Sq≤16 (`GFX906_FA_DIRECT_PAGED*`).
- LEGACY=1 decode verified FULL-capture-safe; CGSupport default is
  `UNIFORM_SINGLE_TOKEN_DECODE` (this flip alone: 22.44 → 52.90 t/s MoE).

### 2. Custom W4A16 MoE grouped GEMM
`csrc/rocm/moe_q_gemm_gfx906.cu` + `vllm/.../fused_moe/experts/gfx906_w4a16_moe.py`
+ oracle entry in `fused_moe/oracle/int_wna16.py`. Fixes the upstream modular
pipeline's −71% MoE regression (3.49 t/s) on gfx906. AWQ int4, 128-group,
load-time repack (~65 s). Handles both MoeWNA16 (N-first uint8) and AutoAWQ
(K-first int32) layouts.

### 3. Dense M≤16 W16A16 GEMV family
`csrc/rocm/dense_gemv_gfx906.cu`, dispatched from
`vllm/model_executor/layers/utils.py` (`_llmm1_tiny_m`, `_gfx906_gemv_long_k`,
`_gfx906_spec_gemv_m4`; kill switch `VLLM_GFX906_DENSE_GEMV=0`):

- m<4 decode GEMMs → padded LLMM1 / GEMV (−401 µs/step MoE).
- m==1 rows → GEMV RPT=1 (kills the per-step `F.pad` of the shared-expert
  gate weight; 4.7× isolated, bit-equal).
- Long-K n==1 GEMV (K=17408 down_proj, KCHUNK=1024/RPT=2, fp16 atomic
  K-split): 227.6 µs = 100% of HBM floor vs 794 µs triton_matmul.
- Verified at the HBM floor on all dense fp16 shapes (K=5120 lm_head probe:
  neutral — LLMM1 already at floor there).
- **W4 (2026-08-23): M=5..16 skinny rail** — weight-row-parallel kernel
  (RPT=1, exact-M template, grid (N, ksplit), fp16 `atomicAdd` K-split
  epilogue), behind `VLLM_GFX906_SKINNY_M16` (default off at merge).
  x-L2-re-read bound at M·B/1.6 TB/s; M-dependent gate (M≤7 all sizes;
  M=8 ≤32 MB; M≥9 ≤10 MB) — big shapes stay on triton. Serving: +14.5 %
  35B / +6.1 % 27B at N=8; flag-on 30-rep soak passed. `DEVLOG-fp16-skinny.md`.

### 4. Other landed fixes
- **GemmaRMSNorm fused-kernel dispatch** (`layernorm.py`): Gemma's `(1+w)`
  factorization dispatches the fused RMS-norm kernel with `w' = 1+w` in the
  input dtype instead of an fp32 decomposition.
- **GDN `core_attn_out` zero-fill removed** on the packed-decode fast path
  (`GFX906_GDN_EMPTY_CORE_OUT`, default on; the Triton kernel stores
  unconditionally).
- **fastsafetensors GDS fallback** (`model_loader/weight_utils.py`): catch
  bare `Exception` (GDS-unsupported raises non-`RuntimeError`) — 2.6× faster
  loads, was engine-death before.
- **hipify in-source guard** (`cmake/hipify.py`): same-dir copytree crash on
  Py3.12 in-source rebuilds.
- **ROCm platform** (`platforms/rocm.py`): CUSTOM backend default +
  registration; device-name derivation from GCN arch (amdsmi returns 0
  handles after torch import on ROCm 7.14).
- Fill/copy pile reductions (P3-4): three bit-exact launch removals
  (+1.15%), attributed the rest (MoE gemm zeroings are required by grid.z
  atomic K-splits; runner H2D micro-copies are upstream).


**Spec-decode recommendation (2026-09-13, Kevin).** All local models use **MTP**
(MTP k=3 where available, e.g. the Qwen3.5/3.8 family; Muse-Glimmer included).
The `ngram` configs still shown in some rows above — and the "+15 % at tg256"
and Muse-Glimmer "100 % filler acceptance" figures — are the historical
filler-corpus measurements (acceptance ceilings, not real-payload results);
ngram is **deprecated for now**, with a Muse-Glimmer MTP re-measure tracked as
MUSE-1 in `ROADMAP.md`.
## Performance history (serving, pp=2048/tg=256)

### MoE — Qwen3.5-35B-A3B-AWQ

0.29.0/V2 restamp (in-process harness, same boot, single GPU, mclk 1000): **V2
58.36 t/s vs V1 57.86** (+0.9 %, recorded reference 58.43) — the V2 runner is at
parity, no code change. Dense 27B: **V2 24.90 / 16.33 vs V1 24.82 / 16.27** (warm /
cold). Agentic TP=2 dense (3 reps, V2): greedy **20.37/13.27**, MTP k=3
**33.62/23.75**, MTP k=3 + CAT-1 **35.44/24.61** @64k/120k. Vision-tower path
(VIT-1): image-prompt TTFT −11.5 % @1024×1024, fresh-boot −55 s.

| milestone | t/s | commit |
|---|---|---|
| fork base (upstream modular pipeline, Triton WNA16) | 3.49 | — |
| custom W4A16 MoE kernel (eager 18.88; prefill 2140) | 41.5 (graphs) | `85eacaeed9`…`f770b9f446` |
| P3-1 tiny-m gemv routing | 44.09 | `3e7c4f2252` |
| P3-3a CUSTOM FA FULL-decode capture default | 52.90 | `2cd52b6f4a` |
| fused fp16 KV gather (Route B stage 1) | 57.09 | `01526dfc69` |
| FA kernel track: NC2 packing + KV split (B=1 parallelism) | 62.8 | `e8b3293554` |
| fused gather-and-quantize | 63.56 | `225448d93f` |
| fill/copy pile fixes (P3-4) | 64.08 | `9bdd9f4639` |
| kv_split prefill clamp + NC2 fail-closed; NC2=2 + GDN flip | 65.36 | `b4873459f8`, `1a895e8a01` |
| GemmaRMSNorm fused dispatch; down_proj GEMV | 66.36 | `19c1d41cf5`, `2cd5b4cafa` |
| FA decode copy pile 7→2 (BSHD native output) | 67.02 | `d63b3ab464` |
| max-ilp scheduler per-file (W4/FA/skinny) | **67.39** | `c6247f729e` |

### Dense — Qwen3.5-27B-AWQ

| milestone | t/s | decode-only t/s |
|---|---|---|
| baseline (Triton FA, GEMV off) | 18.89 | 22.55 |
| CUSTOM FA (NC2=1 fallback) | 23.15 | 28.10 |
| NC2=2 for ratio-6 GQA | 23.55 | 28.69 |
| down_proj K=17408 GEMV | 23.85 | — |
| FA copy pile 7→2 | 24.06 | — |
| max-ilp scheduler per-file (W4/FA/skinny) | **25.60** | `c6247f729e` |

### Concurrent decode (N=8, graph, Δ-metric A/B — W4, 2026-08-23)

| model | shape | W4 off | W4 on | Δ |
|---|---|---|---|---|
| Qwen3.5-35B-A3B-AWQ (MoE) | pp=2048/tg=256 | 166.9 | **191.0** (soak 189.9 ± 0.4; final-build restamp 192.9/194.0) | **+14.5 %** |
| Qwen3.8-27B-AWQ-INT4 (dense) | pp=1024/tg=160 (KV cap, util 0.90) | 98.2 | **104.2** | **+6.1 %** |

`VLLM_GFX906_SKINNY_M16=1` (default off); 27B N=4 control flat (−0.6 %, flag
inert). Kernel, gate rationale, and soak: `DEVLOG-fp16-skinny.md`.

### Long-context prefill sweep (TP=2, 2× MI50, 2026-08-29, boot N)

Deep-prompt prefill for the two prime dense models at max context
(256k / 128k). B=1, tg=128, 2 samples, prefix caching OFF, bt4096,
float16, capture `[1,2,3,4]`, no spec decode; KV 6 GiB (Muse,
783,892-token pool) / 10 GiB (Qwen3.8, 323,414 — 256k max-len needs
≥ 8.09 GiB). Canary 38.9 t/s healthy. csrc @ `cf5ccbd685` (2026-09-13 tree `bbb087b65a`; M2 + M3
hygiene, bit-identical). Harness: `_serve_tp2_gfx906.sh` +
`_bench_serve_grid_gfx906.py` (`'[[32768,128],[65536,128],[112640,128]]' 2`).

| model | pp | prefill t/s (s0 / s1) | TTFT (s0 / s1) | decode t/s |
|---|---:|---|---|---:|
| Qwen3.8-27B (256k) | 32768 | 442.1 / 445.7 | 74.12 / 73.52 | 25.5 / 25.8 |
| Qwen3.8-27B (256k) | 65536 | 365.1 / 364.5 | 179.49 / 179.78 | — (out=1, EOS on filler) |
| Qwen3.8-27B (256k) | 112640 | 289.0 / 289.0 | 389.70 / 389.80 | 13.3 / 13.3 |

**Post-0.30.0 (2026-09-24):** same model/TP=2 on the merged base with MTP k=3 +
`GFX906_FA_LEGACY=0`, prefix caching OFF: 64k **354 t/s** (TTFT 185.3 s), 120k
**271 t/s** (TTFT 442.2 s) — within the sweep's band, no prefill regression.
| Muse-Glimmer-30B (128k) | 32768 | 500.6 / 499.4 | 65.45 / 65.62 | 30.5 / 30.5 |
| Muse-Glimmer-30B (128k) | 65536 | 442.5 / 441.6 | 148.09 / 148.41 | 26.3 / 26.3 |
| Muse-Glimmer-30B (128k) | 112640 | 380.7 / 378.5 | 295.88 / 297.57 | 21.9 / 21.9 |

Sample-to-sample spread ≤ 0.5 % on TTFT. Live-ctx tax per doubling:
Qwen3.8 −17.8 % / −20.8 % (head_dim 256); Muse −11.6 % / −14.1 %.
Cross-check: matches the ~500 t/s TP=2 32k prefill records (TP=1
in-process is ~240 t/s, ~2× scaling holds). Raw logs:
`/local/tmp/lcbench_{muse,qwen38}_grid*.log`.

### Long-context decode with MTP k=2 (TP=2, 2× MI50, 2026-09-02/03)

Companion to the prefill sweep above — same hardware, but **spec decode on**
and at the regime where it used to hurt. The `seq_q>2` kv_split hard clamp
(added with the 256k OOM fix) disabled KV-split for every spec-decode verify
step (Sq = k+1 = 3), collapsing MTP long-context decode to ~half of greedy.
The **kv_split byte-budget guard** (`a6ff64a71b`, branch
`gfx906/mtp1b0-kvsplit-verify`; `GFX906_FA_KVSPLIT_MAX_BYTES`, default 512
MiB; regression-tested in `tests/kernels/attention/test_gfx906_fa.py` with a
32 MiB budget pin forcing the clamp path)
replaces the hard clamp: verifies keep split parallelism when their KV bytes
fit the budget; prefill OOM protection preserved. Correctness: kv_split
1/8/16 bit-identical Sq≤1024 both paths + torch-ref match; PPL gate PASS.

Qwen3.8-27B-AWQ-INT4, TP=2 (util 0.85, bt 1024, max-seqs 4, capture
`[1,2,3,4]`, `disable_custom_all_reduce=True`, `--max-model-len 131072`),
tg=256, temp 0, n=3 reps, filler corpus s9.

| ctx (pp) | MTP k=2 pre-fix (clamp) | **MTP k=2 post-fix** | greedy TP=2 | fix gain | MTP vs greedy |
|---:|---:|---:|---:|---:|---:|
| 65,536 | 15.95 t/s | **37.95 t/s** | 18.86 | **2.38×** | 2.01× |
| 98,304 | 11.19 t/s | **29.88 t/s** | 14.80 | **2.67×** | 2.02× |
| 122,880 | 9.18 t/s | **25.70 t/s** | 12.74 | **2.80×** | 2.02× |

**Post-0.30.0 re-measure (2026-09-24, merge `8893a50e54`), MTP k=3 + the
`GFX906_FA_LEGACY=0` KV-read default (both current defaults), same filler /
prefix-caching-OFF basis:** 64k **42.0** / 120k **40.7 t/s** (2 samples; 8k/32k/96k
not re-measured), 120k prefill **271 t/s** (TTFT 442.2 s) at
`--max-model-len 131072`. The k=2 rows above stay the 2026-09-02/03 record. Same
build PPL gate: **10.5472**. Raw: `/local/tmp/bench_lc_030.log`.

The old "MTP < greedy beyond ~20k ctx" live-ctx tax (FA gather/attention
O(Sk)) is gone at k=2: post-fix MTP leads greedy by ~2× at 64k+ and still
leads at short context (59.2 t/s @2k). Raw data:
`/local/tmp/mtp1/data_mtp_bootQ.jsonl` (pre-fix, boot Q) /
`data_mtp_k2fix_bootS.jsonl` (post-fix, boot S).

**Depth (3 vs 2): k=3 is the long-context default — 2026-09-09/11, same-corpus
A/B on the production payload** (`DEVLOG-mtp-depth-matrix.md`; v2 corpus = 20 %
chat). 120k: **k=3 24.76 > k=2 22.65 (+9.3 %)**, and > k=4 (21.87, v1) by
13.2 %; 64k: k=3 27.44 vs k=2 27.30 = tie. On the pure s9 filler both are
perfect-acceptance (s9 says nothing about depth — the k=5 s9 "+11.4 %" is a
*ceiling*, not a prediction). Mechanism: k=3's 4-row verify pads into the same
occ-2 FA tile as k=2; k=4 (5 rows → pad 8) crosses to the occ-1 slow tile.
**Enable MTP k=3 for long-context serving on TP=2** (capture sizes multiples of
k+1 = 4); k=2 for short/copy-light.

### Headline: agentic Python coding on our own corpus (2026-09-13)

The tables above use synthetic filler; this is the workload we actually serve.
Qwen3.8-27B-AWQ-INT4, TP=2, `--max-model-len 131072`, tg=256, temp 0, **our own
CAT-1 corpus** — 15.5 M tokens of pi/hermes **agentic Python coding** traffic
(`docs/gfx906/CAT1-corpus-build.md`), 8 distinct bodies per point, prefix cache
off so every rep pays its full prefill, 2 reps per cell (boot f27e8058, mclk
verified 1000 MHz):

| ctx | prefill | greedy | MTP k=3 | **MTP k=3 + CAT-1** | uplift |
|---|---:|---:|---:|---:|---:|
| 64k | 277 t/s | 19.80 | 33.28 | **33.26** | **1.68×** vs greedy |
| 120k | 226 t/s | 13.17 | 24.74 | **24.95** | **1.89×** vs greedy |

Acceptance (mean accepted per step) at 120k is the strongest in the fork's
records — 2.05/2.15 for plain k=3, 1.99/2.07 with CAT-1 — because a long agentic
tail is copy-heavy, which is exactly CAT-1's operating point. CAT-1 and plain
k=3 are a tie here (within the arm's own rep spread); its benefit is the
**per-step** one, measured under control on 11 identical 8k prompts × 2 reps:
**−2.52 ms/step [−2.91, −1.94] ⇒ +4.8 % mean / +5.9 % median t/s** (Note 2026-09-14: the same arm-labelled client was in use here, so the *acceptance* column of that A/B is void — the arms ran different prompts. The ms/step result is unaffected (at fixed k the per-step cost does not depend on acceptance), and "no acceptance penalty" is now independently supported by the clean V2 re-measure: acceptance 1.98 vs 1.98 @64k, 2.07 vs 2.10 @120k.), acceptance
no detectable penalty. So quote **~33 t/s @64k / ~25 t/s @120k** for agentic
coding on TP=2, and read the CAT-1 gain from the controlled A/B, not from this
session's 2-rep cells.

## Bench recipes

Canonical environment is the **local editable `.venv`** (docker images are
legacy; both documented in `running.md`). Harness: `docs/gfx906/_bench_gfx906.py`
— prints `BENCH: {json}` with per-sample t/s and `mclk_median_mhz`.

**Two things that change the answer more than the code does:**

1. **Metric basis — `BENCH_PREFIX_CACHE` (default `0` = OFF since 2026-08-27).**
   `=1` reproduces the historical DEVLOG numbers (the 4 samples share a prompt,
   so samples 2-4 reuse the prefix and their prefill is nearly free); `=0` bills
   every sample's full 2048-token prefill. Same build/boot: MoE 65.91 vs 58.43,
   dense 24.82 vs 16.27 t/s. Always state which basis a number came from.
2. **mclk — sampled per timed window by the harness** (idle 350 MHz inflates
   cold-clock numbers ~3×; a median < 900 MHz prints a loud warning).
   `dvfs-mi50.md` has the ground rules.

Profiler plugins are installed as entry points (`agdn_phase`, `syv9_phase`,
`t1_cap`, `mtp1_phase`, `pfk4_phase`); their hooks do file I/O in the hot path
and make dynamo AOT fail (`Attempted to call function marked as skipped:
posix.stat`). Allowlist them away for any bench:

```bash
export VLLM_PLUGINS=quark_online_quant
# The V2 model runner is the default and is validated on gfx906 since 2026-09-16
# (DFL2-2 closed; V1 was unpinned in this file's recipe above). V1 is the
# rollback: VLLM_USE_V2_MODEL_RUNNER=0 (Qwen4Exp needs V2 — see QSA-FN-2).
```

MoE (**`/data/models/QuantTrio/Qwen3.5-35B-A3B-AWQ`** — the `/local` copy is
gone since 2026-09; a wrong path makes vLLM treat it as an HF repo id and abort
with `HFValidationError`):

```bash
HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HF_HUB_OFFLINE=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 BENCH_EAGER=0 BENCH_SAMPLES=4 \
BENCH_PP=2048 BENCH_TG=256 BENCH_GPU_UTIL=0.95 BENCH_MAX_SEQS=32 \
BENCH_PREFIX_CACHE=1   # omit/0 for the cold-prefill basis \
.venv/bin/python docs/gfx906/_bench_gfx906.py /data/models/QuantTrio/Qwen3.5-35B-A3B-AWQ
```

Dense 27B (`/data/models/qwen/Qwen3.5-27B-AWQ`, NFS): identical except
`BENCH_MAX_SEQS=4` (the GDN state pool needs the room; higher OOMs on the first
chunk). Harness knobs: `BENCH_PP/TG/GPU_UTIL/BATCHED_TOKENS/MAXLEN/SAMPLES/
WARMUP/EAGER/MAX_SEQS/PREFIX_CACHE/CG_MODE/CG_MAX/KV_MEM/NREQS/DTYPE/
SPEC_CONFIG/MOE_BACKEND/ATTN_BACKEND(+_KIND)/MODEL`.

Long-context TP=2 serve sweep (2026-08-29 recipe):
`_serve_tp2_gfx906.sh start <tag> <snap> <name> <max-len> <tool> <reason>`
(`KVBYTES=10737418240` for Qwen3.8-27B @ 256k) → `wait` →
`LC_MODEL=<name> LC_SNAP=<snap> .venv/bin/python
docs/gfx906/_bench_serve_grid_gfx906.py '[[32768,128],[65536,128],[112640,128]]' 2`
→ `stop` (SIGTERM + VRAM-release wait — never SIGKILL a TP=2 server).

Rules: serving benches run sequentially; check `uptime` first (background CPU
contention invalidates numbers); `BENCH_ATTN_BACKEND` must NOT be set
(it forces Triton). PPL gate: `docs/gfx906/` probe convention (442-token
fixed set; bands above). PPL run-to-run noise: MoE ±0.3%, dense ±2% —
multi-batch greedy token identity is NOT a valid gate.

## Tests

```bash
.venv/bin/python -m pytest tests/kernels/attention/test_gfx906_fa.py -v      # 15
.venv/bin/python -m pytest tests/kernels/moe/test_gfx906_moe_gemm.py -v      # 12
.venv/bin/python -m pytest tests/model_executor/layers/test_rocm_unquantized_gemm.py -v
```

All three suites are green on real gfx906 hardware. In the last file the
arch-simulation tests patch every platform predicate (incl. `on_gfx906`)
and 2 wvSplitK real-kernel tests skip on gfx906 by design (wvSplitK
targets matrix cores; gfx906 has none).

## Environment knobs

| env | default | effect |
|---|---|---|
| `GFX906_FA_VIT` | 1 | **VIT-1** (0.29.0): custom Q8 FA for the Qwen3.5/3.8 VL **vision tower**; `0` = kill switch back to the upstream flash-attn ViT path |
| `GFX906_FA_VIT_AUTO` | 1 | whether the ViT path is chosen *automatically*; `0` opts out of auto-selection only (an explicit `--mm-encoder-attn-backend custom` still selects CUSTOM) |
| `VLLM_USE_V2_MODEL_RUNNER` | (upstream) | `0` forces the V1 runner; **no model needs it any more** (Muse-Glimmer was the last pin, lifted 2026-09-16; V1 is removed upstream in 0.32.0) |
| `GFX906_FA_LEGACY` | **0** (flipped 2026-09-16) | **Default since 2026-09-16 (KVLAYOUT-1)**: `0` = Q8 pre-quantized at KV write into a side **view aliased into the fp16 K half** — zero extra KV memory, COW-safe; verified on 0.29's fused layout (PPL 10.5472/10.5460 vs 10.5472 for LEGACY=1 — within the probe's own ~0.001 run-to-run spread; 0 top-20 misses) and **−15.5 % / −19.1 % ms/step** with MTP k=3 at 64k/120k (acceptance unchanged). `1` = fp16 KV cache + in-kernel Q8 quantize — the validated rollback, and the winner in the one regime it beats the side-buffer (~6 % at B=1 greedy decode, 2026-08-29). History: `0` = Q8 pre-quantized at KV write into a side **view aliased into the fp16 K half** — zero extra KV memory, COW-safe (page copies move the Q8 bytes); attention reads the Q8 directly instead of re-quantizing per read. 2026-08-27: 46/46 suite + default-config and prefix-cache smokes clean. **Orthogonal to the read pattern** (`GFX906_FA_DIRECT_PAGED*`): direct-paged is LEGACY=0-only (round-7 erratum); LEGACY=0's distinct contribution is the Q8 read (no repeated inline quantize). **TP=2 bake executed 2026-08-28 (roadmap M5, boot M): LEGACY=0 was SLOWER than the LEGACY=1 control at every point** (B=1 decode −2.5…−3.7 % @2k/8k; B=4 @2k aggregate −27…−31 %; prefill wash). ****Default stays `1`**; `0` remains an experimental opt-in (zero-extra-KV-memory alias, COW-safe). Round 10 (M6 Part B, same day): rerouting LEGACY=0 B≥2 to the fused-Q8 gather (`GFX906_FA_DIRECT_PAGED_Q8=0`, now the default) recovered B=4 @2k to 35.7 → 46.3 t/s — parity with the 46.7 LEGACY=1 control (same-boot adjudication since run and closed: −6.3 %, see the row end) — with B=1/prefill unchanged; direct-paged is opt-in (=1) with no measured advantage. Mechanism (review-softened 2026-08-28): NOT an int8-compute gap (`v_dot4_i32_i8` is full-rate on gfx906, 4 int8 MAC/cyc, 2× packed fp16; both arms share the same dot — `DEVLOG-fa-kernel-batches.md` M5 entry). The loss is **Sq>1-specific** — the in-process Sq=1 A/B on the identical strided-read path was a wash, which the read-layout theory alone cannot explain; direct-paged's strided Q8-slice reads (136 B slices inside 256-B row strides, 34-B block strides → sector waste) remain the **leading but unconfirmed** hypothesis, with the Sq>1 machinery (round-8 Q-pad/unpad fast paths, graph-capture interaction) at least a co-contributor — devlog round-10 erratum. The flip gate's B=4 half is green; the B=1 same-boot adjudication ran 2026-08-29 (boot O): LEGACY=0 −6.3 % (−6.4 % with direct-paged) — flip CLOSED, default stays 1 (`DEVLOG-fa-legacy0-b1-decode.md`) |
| `GFX906_FA_FUSED_QUANT` | 1 | fuse quantize into the decode KV gather (bit-equal); `0` kill switch |
| `GFX906_FA_NC2` | 8 (auto-downgrade) | GQA heads packed per KV block; instantiated {1,2,8}; invalid explicit value = error |
| `GFX906_FA_KVSPLIT` | shape-aware per path (2026-09-06): gather **32 @ Sq≥4 / 16 @ Sq<4** (`dfed62f133`); direct-paged `fa_paged_kv_split_default(seq_q, batch)` = 32 @ Sq≥4, its own batch clamp (8/8/5/2/2 at B=1/2/3/4/8) below that (`fa7e1e20b9`) | decode KV-split factor (one knob, two paths — an exported value pins BOTH; safe explicit values: gather `16`, direct-paged `8` at B≤2 / `2` at B≥4); `1` disables; `0` is NOT the unset default (it clamps to 1). The gather path's fixed `16` is B=1-tuned: at B=4 it costs ~+18 % extra combine-traffic vs the batch-scaled formula (documented in `csrc/gfx906_fa/gfx906_fa.cpp`) — the round-10 reroute's +29.7 % net win is *despite* this; a batch-aware gather split is a follow-up candidate |
| `GFX906_FA_CG` | UNIFORM_SINGLE_TOKEN_DECODE | FA cudagraph-support mode |
| `GFX906_FA_DIRECT_PAGED` / `_MIN_BATCH` / `_MAX_SQ` | auto / 2 / 16 | direct-paged decode path gating |
| `GFX906_FA_DIRECT_PAGED_Q8` | 0 | LEGACY=0-only B≥2 route selector: `0` (default since round 10, 2026-08-28) = fused-Q8 gather (B=4 @2k ngram serving 35.7 → 46.3 t/s, parity with the 46.7 LEGACY=1 control within cross-boot uncertainty; the M1 gather clip stays active); `1` = direct-paged (opt-in experiment route — it lost the M5 bake −27…−31 % at B=4 @2k and was a wash in the in-process Sq=1 A/B; the in-process harness cannot see the loss, only the ngram serving regime). Validated on Muse-Glimmer TP=2 ngram n=5 only — other LEGACY=0 configs (TP=1, MTP Sq=2) are unmeasured. `=0` gives up direct-paged's zero-gather-allocation property: a one-time bounded shared-buffer grow (~0.1–0.4 GiB at B=8/Sk=61k — the same grow-only buffer production LEGACY=1 already runs at every batch size; the stale 24 GiB OOM figure in the header comment predates the ClassVar sharing fix) |
| `GFX906_FA_WINDOW_CLIP` | 1 | Phase C window-start clip (kernel floors to the KV-tile boundary when the gained keys are mask-killed, so the result is bit-identical to the masked full scan); `0` kill switch (numerically identical, slower). **Since M2 (2026-08-28) it covers prefill chunks too, both dispatch paths**: decode (direct-paged, auto B≥2/Sq≤16 — LEGACY=0 only, round-7 erratum) and the DIRECT_PAGED prefill chunk-start clip (the `max_seqlen_q == 1` gate was dropped); under the LEGACY=1 default prefill B=1 the gather path's clip (`GFX906_FA_GATHER_CLIP`) is the active source. e2e gate +3.6% @ pp8192/B=2 (DEVLOG-muse-glimmer 2026-08-27 round 3) |
| `GFX906_FA_GATHER_CLIP` | 1 | M1: the same window-start clip for the **gather path** (persistent gather, B≤16, any Sq — the B=1 decode default and all prefill): the persistent gather writes only rows `[kv_start, seq_len)` at absolute positions (a 128-row margin covers the kernel's tile-boundary floor) and the FA kernel starts its k-loop there — rows `[0, kv_start)` are never written or read. Bit-identical to the unclipped full gather + scan (unit-gated, `_GATHER_CLIP` 0/1 A/B at Sq=1/6, B=1/2, unaligned L/W); `0` kill switch. The gather work and the FA k-loop both shrink by `min(seq_len, seq_len - 1 + 1 - window)` rows |
| `GFX906_FA_TILE_CLIP` | 1 | M2 (2026-08-28, branch `feat/fa-m2-tile-clip`): the two **per-q-tile** FA scan bounds — (1) raise `k0_base` to the tile's own window start (clip mode only, complements the sequence-level clips above, which stay the kill switches for the *source*), (2) cap `k_VKQ_max` at the tile's last row + 1 (fires for **any** chunked prefill with q_abs_offset, window or not — a general prefill win: the causal cap alone cuts first-chunk FA work ~2×). Skipped tiles are exact no-ops (P=0, KQ_max unchanged) → bit-identical; `0` disables both M2 bounds only (the M1 floors stay). A/B: read per call — eager A/B works in-process; FULL-captured graphs bake the value (M2 is prefill-only, so that never matters today). Gate (DEVLOG-fa-kernel-batches.md 2026-08-28/29 M2 entries): 65/65 suite + kernel A/B 3.19×/2.81× windowed, **2.22×/1.96× causal-only first-chunk (a general prefill win for any model)** + e2e A/B **+11.8 % wall / +14.8 % prefill @ pp16384 windowed**, +0.73 % @ pp2048 full-attention-hybrid (GEMM-dominated; both samples agree in direction) |
| `GFX906_FA_NO_WINDOW` | 0 | truthy = disable sliding-window masking on all layers (wrong output beyond the window; warns; perf A/B arm only) |
| `VLLM_GFX906_DENSE_GEMV` | 1 | M=1 dense GEMV dispatch; `0` kill switch |
| `GFX906_GDN_EMPTY_CORE_OUT` | 1 | skip the dead GDN core_attn_out zero-fill |
| `GFX906_FA_GATHER_V` | auto | gather-kernel V handling variant |
| debug: `GFX906_FA_DOUBLE_CHECK`, `_DUMP`, `_FWD_DEBUG`, `_NO_BUF_REUSE`, `_QPAD_EMPTY`, `_TORCH_GATHER`, `_ZERO_KTAIL`, `GFX906_FA_FUSED` | off | validation/debug paths |

## Known issues / limitations

- **CPU stuck-threads in TP=2 serving** (host-level, 2026-08-24): 2
  threads per worker freeze at 100% user CPU in the HSA P2P-IPC handshake
  (`libc __poll` / `libhsa IPCClientImport`) within ~15-20 min of start;
  reboot does not clear it; serving unaffected so far. Full write-up +
  options (NCCL_P2P_DISABLE A/B, AMD escalation): `cpu-stuck-threads.md`.
- **rocprofv3 finalization race**: dense-model traces consistently fail
  (ring buffer invalid at exit; EngineCore teardown races HSA). Use
  three-anchor inference or eager torch-profiler attribution instead.
- **LEGACY=0** (Q8 side view) had its warmup/COW/graph-replay desync
  fixed 2026-08-27 (`b98bb329f0`) and is validated (46/46 suite,
  default-config + prefix-cache smokes) but stays experimental — no
  long-context soak / net win in a realistic config; the M5 bake
  (2026-08-28) and the same-boot B=1 adjudication (2026-08-29, −6.3 %)
  both lost, so LEGACY=1 stays the default
  (`DEVLOG-fa-legacy0-b1-decode.md`).
- **Layer 0's routed experts ship fp16** (checkpoint
  `modules_to_not_convert`) → Triton `fused_moe`, 414 µs/step MoE. Options
  catalogued in `ROADMAP.md` (C4, Tier 2).
- **GDN/mamba state copy pile**: ~180 µs/step of `[3,1,32]` copies —
  upstream state bookkeeping, deferred.
- **B>1 decode**: one 192 KB reshape copy per FA layer remains (zero-copy
  needs a decode-specialized kernel store; only matters for batched decode).
- **Eager decode is launch-bound**: eager A/B of kernel improvements can tie
  even when the kernels differ; always gate in graph/serving mode.
- **256k-context prefill OOM on Qwen3.8-27B — RESOLVED 2026-08-24
  (branch `gfx906/fa-gather-lifecycle`; history: 7 OOMs on 2026-08-23,
  full mechanism + verbatim evidence in `oom-256k-prefill.md`, fix +
  validation in `DEVLOG-fa-attention.md` "Gather-buffer lifecycle fix").**
  The unprofiled long-context transient was the custom-FA
  `_gather_retired` keep-alive dict: pre-fix, every chunked-prefill chunk
  with a larger max-seq-len reallocated the gather buffers at exact Sk and
  retired the previous capture-flagged generation (measured 7.79 GB / 152
  generations by the 60k-token OOM point — vs ~1.94 GiB headroom; the AWQ
  `temp_dq` scratch was the allocation that landed on the remains). Fixed:
  capacity-width grow-only buffers + per-generation capture flag
  (`GFX906_FA_GATHER_EXACT=1` restores the old policy). The 250k run-4
  prefill now completes with the needle retrieved (148 tok/s incl.
  prefill), decode A/B flat; **the 131k ceiling is lifted** on this model.
  Not a serving-time leak (W4 soak flat at 98-99 %). KV sizing facts:
  Qwen3.8-27B 64 KB/token (TP=1) / 32 KB/rank (TP=2); Qwen3.5-27B 20 KB;
  Qwen3.5-35B 10 KB.
- Dense GEMM dispatch (exllama gptq + LLMM1) is at its measured optimum —
  a purpose-built W4A16 dense GEMV is the top remaining dense lever
  (roadmap item).
