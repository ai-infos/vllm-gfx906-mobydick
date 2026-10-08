# 1CatAI/1Cat-vLLM — recon notes (2026-09-03)

**Source fork:** https://github.com/1CatAI/1Cat-vLLM ("Make Volta Fast Again"),
fetched 2026-09-03, cloned at `/local/git/1cat-vllm-clone` (depth 50). Target:
**NVIDIA Tesla V100 / SM70** — same silicon generation class as our gfx906
(Volta-class: no `cp.async`, no TMA, no FP8 Tensor Cores; software must rebuild
the memory/compute overlap the newer HW gives for free).

**Central index:** ideas extracted here are registered in
`docs/gfx906/ROADMAP.md` → "Ideas from 1CatAI/1Cat-vLLM" (IDs **CAT-1 … CAT-9**).

## Why this fork is unusually close to us

Unlike the syv (RTX 3090) and joe2gaan (TP=8 host) recon docs, 1Cat runs
**Qwen3.6-27B-AWQ on V100 TP2** — the *same model family* as our Qwen3.8-27B-AWQ
(GDN hybrid: 48 linear-attn + 16 full-attn, D=256 heads, native MTP), at the
*same TP=2* we run. Their measured breakdowns are therefore directly comparable
to our SYV-9 prefill profile and MTP-1b decode profile. **Absolute numbers do NOT
transfer** (different silicon, different clocks, TurboMind AWQ backend vs our
custom WNA16 path, CUDA FA vs our Triton FA) — the *techniques* and *relative
shares* are what we mine.

### Their reference breakdowns (sanity anchors for us)

MTP4 round, Qwen3.6-27B-AWQ TP2 V100 (`sm70_qwen36_27b_awq_mtp4_optimization.md`):
target verifier forward **45.1%** / target logits 4.1% / rejection sampling 2.8% /
MTP draft GPU **24.2%** / bookkeeping 0.2% / uninstrumented remainder 23.7%.

Target-forward composition (`qwen36_27b_awq_mtp_target_verifier_fastpaths.md`):
TurboMind AWQ GEMM **44.3%** / TP all-reduce 9.1% / copy-cast 7.7% / GDN recurrent
core + causal conv 6.9% / FP16 GEMM 6.4% / Flash-V100 attention 4.2% (decode —
attention is small at short ctx, grows O(Sk) like ours).

This matches our own SYV-9 prefill finding (FA kernel → **45%** of prefill at 120k,
GEMM ~33%, GDN conv1d+rec ~3%) and MTP-1b decode finding (full_attn **73%** of the
greedy @120k step). The two forks independently converge on **attention is the
long-context owner; GEMM is the short-context owner** — strong external validation.

## Ideas, ranked for gfx906 TP=2 long-context relevance

### CAT-1 — Draft-vocabulary shortlisting (GO — feeds SYV-3)
Their single biggest MTP win: shrink the *drafter's* lm_head vocabulary from full
248k to a **static 131K** or **dynamic 98K + 2×512** shortlist. Measured on their
near-identical model at TP2: static-131K = **+21.9% e2e throughput** (80.1→97.7
tok/s), dynamic-98K = the current default (100.6 tok/s, A=4.02 acceptance, −1.67%
vs static). Method: one-time target-logits `topk=2048` bootstrap during prefill
builds a global shortlist; the drafter lm_head runs only over it (`torch.mm` on
the reduced rows); the **target** distribution stays full-vocab for accept/recover
(never renormalize p onto the shortlist → rejection sampling stays lossless).
**Why it matters to us:** this is *exactly* SYV-3 lever 3 ("small draft vocab"),
which we had scoped but not implemented. We already have the Triton K=1 skinny-GEMV
at 98% of BW ceiling (SYV-3 lever 1); shortlisting **halves the bytes read** on top
of that → stacks to ~7× combined at the roofline ceiling, and their e2e number
(+21.9%) shows it survives the full serving path. **Estimate: GO — high value,
moderate effort (scheduler + sampler glue, no new params), lossless by
construction.** Next step: scope the bootstrap + reduced-draft-lm_head integration
on our `step3p5.py` proposer; A/B with acceptance gate (our model is
non-deterministic at temp=0 → use t/s + PPL/coherence, not token identity).

### CAT-2 — FA prefill: D256 Split-D + GQA multi-head packing (GO / ANALYZE — feeds SYV-9)
Their long-context **prefill** attention is the direct target of our SYV-9 (FA
kernel = 45% of prefill at 120k). Their Volta D=256 kernel (`sm70_fa2_d256_prefill_pipeline.md`)
measures **1.66–2.2×** over generic FA2 on the *same D=256 shape* (their TP4 Hq=6/Hkv=1;
ours is TP2 Hq=24/Hkv=4). Techniques: **Split-D** (D=256 → four D64 slices, paired warps
share QK work while increasing PV parallelism), **N32 online-softmax** (a *quality*
requirement — their N64 variant's 1.27e-4 L2 error amplified across layers and changed
sampled tokens; N32 drops it to ~4.6e-6), **GQA multi-head packing** (pack six GQA query
heads into wider Tensor-Core work — directly applicable to our Hq/Hkv=6 ratio), K-stage
ping-pong, prefix/causal-tail separation, 128-bit conflict-aware V stores.
**Estimate: GO/ANALYZE — high value (biggest prefill cost), HIGH effort (a real FA-kernel
rebuild for gfx906 Triton).** The GQA-packing and Split-D ideas are the most transferable;
N32-as-quality-gate is a caution we should adopt. Next step: analyze our `gfx906_fa_forward`
Triton kernel against these axes (head packing, D-split) before any port.

### CAT-3 — FP8 E5M2 KV cache via one-pass expansion (ANALYZE)
Their biggest *prefill* speedup on the FP8-KV route: **4.5–4.9×** by doing a single
vectorized `fp8_e5m2_paged_kv_to_fp16` gather/expansion pass into a shared FP16 page-784
workspace, instead of converting E5M2 inside *every* query CTA (the old path re-converted
the same KV values for each of the many query CTAs → 96 KiB smem, 1 CTA/SM, 25% occupancy,
~4% tensor activity). Decode side: vectorized E5M2 XQA. **Why it matters to us:** we run
**fp16 KV today**, so this is a bigger change — but it halves KV bytes (memory + bandwidth)
and the "expand once, share across layers" workspace pattern is exactly what our long-context
decode (memory-bound FA gather) and SYV-7 prefix caching would benefit from. Their workspace
cost: ~512 MiB/rank at 256K/Hkv=2/D=256 (reused serially by all full-attn layers).
**Estimate: ANALYZE — high value for long-context, HIGH effort (KV-dtype change + new
expansion op), needs its own quality gate (FP8-vs-FP16 KV is a model-level decision).**

### CAT-4 — 128-bit wide aligned KV loads in decode XQA (ANALYZE)
`sm70_flash_v100_fp8_kv_long_context.md` PR #268: replace narrow `half8` KV fragment
loads with one aligned **128-bit** cache load, reusing the page ID and merging two
conversion groups. NCU evidence: L1 global-load requests **−41.5%**, warp instructions
−14.7%, long-scoreboard stall 39→30%, kernel duration **−23%** (B16/17.8K).
**Why it matters to us:** our long-context decode is memory-bound on the FA gather — wider
aligned loads cut address/dependency pressure and L1 traffic without changing DRAM bytes.
On gfx906 the equivalent is `v_load_dwordx4` (128-bit); we'd need to check whether our Triton
FA kernel already emits wide aligned loads or restructures the KV access. **Estimate: ANALYZE —
moderate value, LOW-MOD effort (kernel micro-optimization), good first probe for the decode FA path.**

### CAT-5 — Prefix / causal-tail separation for chunked prefill (ANALYZE)
Their root-cause of superlinear cold prefill (`sm70_tp4_nomtp_long_context_decode.md`):
fixed-size 1024-token chunks each attend over an *increasingly long* KV prefix → O(L²) work
scheduled as O(L/chunk) paged calls; the last 32K of a 64K request alone was **75%** of the
prefix-attention sum. Fix: schedule the fully-visible long **prefix** separately from the exact
**causal tail**, then merge online-softmax state. **Why it matters to us:** our SYV-9 profile
shows FA kernel dominant at 120k prefill; if we chunk-prefill, this is the structural fix for
the same O(L²) blowup. **Estimate: ANALYZE — high value for long-doc TTFT, MOD-HIGH effort
(scheduling + kernel merge), pairs with CAT-2.**

### CAT-6 — CTA-local K-parallel small-M GEMM (ANALYZE)
Their M=5 verify AWQ GEMM runs only **6–12% occupancy** on V100 (68 CTAs on 72 SMs, one 4-warp
CTA/SM). Their fix: intra-CTA K-split (thread-group map `1x4x1`→`1x4x2`, two warp groups
partition the CTA K-loop, FP32 partials reduced in shared memory) — no extra global split-K
workspace or launch. **Why it matters to us:** our decode GEMMs are memory-bound and we're near
the BW ceiling on the drafter GEMV (SYV-3), so this is *less* obviously a win here — but their
target-forward AWQ GEMM is 44% of the verifier forward, same as ours, and small-M occupancy
loss is a real gfx906 shape too (40 waves/CU). **Estimate: ANALYZE — uncertain value for us,
MOD effort; only worth it if a decode-GEMM profile shows occupancy loss at our shapes.**

### CAT-7 — DFlash2 block drafter + LABD / ngram lookup (POSTPONE)
Their headline ~260 tok/s (and 316 tok/s on q16 repeated-context lookup) uses the **DFlash2**
whole-block non-autoregressive drafter on NVFP4, plus optional lookup-augmented block drafting
(LABD) and a prompt-ngram assistant. **Why it's a postpone for us:** we run AWQ-INT4 dense;
their DFlash2 checkpoint is NVFP4 (doesn't transfer), the drafter arch differs from our MTP, and
it needs the V2 runner which conflicts with our FULLGRAPH path. This is the same idea already
parked as **SYV-8**. The *lookup/ngram* sub-idea (verbatim-context drafting) does map to our
SYV-2 lookahead-drafting candidate. **Estimate: POSTPONE — revisit only if MTP (MTP-1b-0 + SYV-3)
stops delivering; the ngram/lookup piece can be folded into SYV-2.**

### CAT-8 — Persistent partition-grid cap (NO-GO — dead-end reference)
They tried capping Flash-V100's fixed decode partition grid to remove empty CTAs; it was
**bitwise exact but slower** (register growth + persistent control consumed the saving, and a
fixed cap regressed at 65K/262K). **Relevance:** our kv_split work is already done (MTP-1b-0)
and we don't have a fixed-grid decode FA to cap. Recorded so we don't re-tread this path.
**Estimate: NO-GO (their own rejection; dead-end reference).**

### CAT-9 — FP8 prefill tile-selection pitfall (ANALYZE — caution for SYV-9)
Their FP8 prefill *regressed with context* until they fixed two things: page 1568 wasn't
selecting the fast BM32 phase kernel, and the generic FP8 kernel did per-CTA E5M2 expansion.
**Relevance:** a direct caution for our SYV-9 int8-QK port — ensure the right tile/phase config
is selected at every page size and never convert inside each query CTA. **Estimate: ANALYZE —
adopt as a design constraint on CAT-3 / SYV-9, not a standalone task.**

## No-go / not-applicable (recorded for completeness)
- **NVFP4 / TurboMind / MXFP4 operator work** (`sm70_nvfp4_turbomind_*`, `sm70_quasar_nvfp4_*`,
  DeepSeek-V4 MXFP4 experts): SM70 FP4/FP8 Tensor-Core paths we don't have; we're AWQ-INT4 on
  gfx906. Not applicable.
- **DeepSeek-V4 / GLM-5.3 sparse-MLA + DSA indexer** (`sm70_deepseek_v4_sparse_mla_*`): different
  model family (sparse MLA, not GDN-hybrid). Not applicable to our dense Qwen3.8.
- **PP2×TP4 / TP8 multi-GPU scaling** (`sm70_tp4_mtp4_multigpu_scaling.md`, `*_tp8_graph_allreduce`):
  we run TP=2 (one edge); their NVLink pairwise findings are reference only — the useful takeaway
  is that *many small sync-bound P2P reductions* hurt (+11% comm going TP2→TP4), which reinforces
  our J2G-1/J2G-2 allreduce work.
- **TileRT-inspired runtime** (`sm70_tile_runtime_exploration.md`): a closed-runtime clone, big
  architectural change; not worth it over our existing FULLGRAPH path.

## Cross-fork validation (the meta-finding)
Three independent forks now converge on the same bottleneck ranking for this model family:
- **syv** (RTX 3090): split-KV for multi-query verify → our MTP-1b-0 (done).
- **joe2gaan** (TP=8 host): persistent all-reduce / RCCL knobs → our J2G-1 (done, +2.77%).
- **1Cat** (V100 TP2, same model): draft-vocab shortlist (CAT-1→SYV-3), FA prefill rebuild
  (CAT-2→SYV-9), FP8 KV one-pass expansion (CAT-3).

The consistent story: **long-context = attention-bound** (FA kernel / O(Sk)), **short-context =
GEMM-bound**, and the MTP drafter's lm_head is a memory-bound GEMV. Our SYV-3 (drafter GEMV) and
SYV-9 (prefill FA) are exactly the two forks' top levers — good confirmation we're attacking the
right things.
