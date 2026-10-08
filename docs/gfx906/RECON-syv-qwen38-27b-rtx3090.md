# syv-ai/qwen38-27b-rtx3090 — recon notes (2026-09-02)

**Source fork:** https://github.com/syv-ai/qwen38-27b-rtx3090 (primary doc:
`docs/optimizations.md`, fetched 2026-09-02). Target: **Qwen3.8-27B on RTX 3090
(sm86, single GPU)** — same model family as ours (GDN hybrid 48 lin + 16 full-attn,
single-layer MTP head), different silicon. Their numbers don't transfer directly;
the *techniques* are the point of this recon.

**Central index:** ideas extracted from this doc are registered in
`docs/gfx906/ROADMAP.md` → "MTP-1b — ideas from syv-ai/qwen38-27b-rtx3090" (IDs SYV-*).

## Ideas, ranked for gfx906 TP=2 long-context relevance

### SYV-1 — Split-KV attention for the multi-query VERIFY step (MOST PROMISING)
FA2 only splits the KV sequence across thread blocks when a request has ONE query
token. Spec-decode verify has k+1 query tokens (k=2 → 3), so their 24-head model ran
attention on 24 of 82 SMs — 57 µs/layer @1.5k ctx, 1.3 ms @16k. Their Triton split-KV
kernel: 23 µs / 120 µs (2.5–11×).
**Why it matters to us:** our validated breakdown says full_attn = 73% of the greedy
step at 120k; MTP verify runs 3 query tokens through the same 16 O(Sk) layers, and our
GFX906_FA decode kernel grid is (num_q_blocks × num_kv_heads) — with B=1, k+1 queries
that's a single q-block per KV head → 4 KV heads × 1 = 4 workgroups on GPU. Same
underutilization shape as their 24/82 SMs. At 120k context this is potentially the
single biggest MTP-side kernel win available. **Status: candidate to explore (see
ROADMAP SYV-1).**

### SYV-2 — Lookahead/context drafting ("draft from the prompt")
Scans the request's own token history for the most recent occurrence of the longest
suffix of what was just generated, proposes the tokens that followed it. Point-mass
draft distribution → rejection sampling stays lossless (acceptance = p(x)). Verify
block can exceed drafter block because context tokens cost the drafter nothing.
Their numbers: +55% on verbatim reproduction, +2–3% prose, gated so long blocks only
schedule while a copy is actually running.
**Why it matters to us:** our workload (Kevin's docs/code) has heavy verbatim
reproduction; acceptance at 120k was already 2.0/2.0 stable (s9 corpus), so the
upside is in *content-shape* gains on real workloads. No new drafter params — pure
scheduler + sampler glue. **Status: candidate (ROADMAP SYV-2).**

### SYV-3 — Quantize the MTP draft module + small draft vocabulary
Their MTP draft was bf16 (850 MB) with full 248k-row lm_head per draft token → ~3 ms
per extra draft; MTP-3 already slower than MTP-2. They requantized the drafter to
int8/int4 and built a 40k-token draft vocab (97.5% coverage over model outputs) →
draft cost ~0.5–1 ms, four drafts pay off.
**Why it matters to us:** our step3p5.py proposer runs full lm_head per draft token —
confirmed in-tree. Our 2026-09-02 microbench measured the *marginal* cost of extra draft
rows (K1→K3) at +322 µs/step = 0.4% at 120k — that is what led to the original "closed"
verdict, and it was the wrong number to judge on: the **absolute** B=1 lm_head read
(~6.9 ms/GPU fp16 proxy, ~18% of MI50 peak BW) is memory-bound (AI = B ≪ ridge) and is
exactly what int8/int4 quantization cuts 2–4×. Drafts are target-verified → perturbs
acceptance rate, not correctness.
**Status: RE-OPENED 2026-09-03 — OPEN.** Full roofline + ruled-out table + next steps:
`/local/tmp/mtp1/drafter_memory_bound.md`; execution queued behind J2G-1 (see ROADMAP.md
SYV-3 entry).

### SYV-4 — Sort-free small-k top-k/top-p sampler
vLLM's top-k/top-p sorts the whole 248k vocab per row + one-thread-block softmax
(140 µs for a single 248k row, several calls/step). With top-k ≤ 64 known on host:
one torch.topk + multi-block softmax. Their gain: +4% at default sampling.
**Status: IMPLEMENTED + MERGED to main (2026-09-03, `e402e85192`..`3d78149c6d`, default ON; opt-out `VLLM_GFX906_SORT_FREE_SMALL_K=0`).**
GPU bench: 12/12 exact vs reference at V=248k; B=4 mixed-k 6.55× (2.212→0.338 ms/call), ~neutral at B=1. End-to-end A/B (TP=1 dense 27B, tg256): +0.6–3.4% t/s on sampling workloads (k-disabled control arm = noise floor). Full detail: ROADMAP.md SYV-4 entry.

### SYV-5 — fp16 GDN recurrent state (`--mamba-ssm-cache-dtype float16`)
48 GDN layers keep fp32 recurrent state per request (~150 MB/req) read+written every
step. Halving to fp16 doubles concurrency at unchanged PPL (to 3 decimals).
**Status: config flag — A/B is trivial; value depends on whether we're state-bound
(we run B=1, so mainly a KV-pool/concurrency win for future multi-request work).**

### SYV-6 — int8 activations for batched GEMMs (W4A8 Marlin) + negative-scale bug fix
At 40–64 concurrent seqs decode is fp16 tensor-core bound; W4A8 int8 MMA = 4× rate.
Their checkpoint had ~50% negative group scales read as unsigned → garbage; fixed by
folding sign into int4 codes at load. **Status: batch-mode only (we run B=1); park
until multi-request work resumes. NOTE the bug fix is model-portable if we ever hit
it.**

### SYV-7 — Hybrid-model prefix caching (`PREFIX_CACHE=1`)
vLLM keeps it opt-in for mamba/GDN hybrids; 24k-doc follow-up turn 23 s → 1 s; 64
requests sharing a system prompt 222 s → 17 s. Recurrent state resumes from last
cached block boundary. **Status: config flag — our sweep methodology uses cold prefill
per rep (unique headers), so this doesn't change MTP-1 numbers, but it's the single
biggest real-workload win for Kevin's chat-on-docs usage.**

### SYV-8 — DFlash2 block drafter (5-layer non-autoregressive, whole-block in one pass)
Different drafter architecture (Inco, Aug 2026): predicts the whole 7-token block from
target hidden states in ONE pass + selector over 16 candidates/slot. Their result:
4.80 vs 4.28 tok/step at same block size. Backport = vLLM PR #52816 (V2 model runner)
+ quantization to fit VRAM. **Status: big effort, needs V2 runner (conflicts with our
FULLGRAPH constraint path); park as a future option — revisit only if SYV-1/SYV-2
don't deliver.**

### SYV-9 — int8-QK prefill attention (SageAttention-style Triton)
QK^T on int8 tensor cores at 2× fp16 for the 16 full-attn layers during PREFILL
(head_dim 256 pins FA2 at 54–57 TFLOPS sm86). +2.7% @16k, +5.3% @51k end-to-end;
prefill-only by construction. **Status: prefill is not our bottleneck (decode is);
park — relevant if prefill-heavy workloads come back.**

## Notable negatives from their campaign (don't re-walk)
- `SPEC=off` prefills ~20% SLOWER than spec-on — the drafter demotes V2→V1 runner;
  drafter's own prefill cost is nil on V2. (Caveat: we don't use V2 runner.)
- Marlin tile tuning washes out end-to-end under sustained power throttling (+0.4%).
- Bigger prefill chunks (4k/8k) fail engine init under pinned KV_MEM — 2048 stays.

## Where the comm-related ideas came from (user question, 2026-09-02)
**Custom RCCL + persistent all-reduce are NOT from syv-ai — they're from a DIFFERENT
repo: joe2gaan/localaiservers qwen36-gfx906**, cloned at `/local/git/localaiservers`,
recon in `/local/tmp/mtp1/joe2gaan_recon.md` (source fork cited there too). Details:
- **Persistent all-reduce** (`VLLM_GFX906_PERSISTENT_AR=1` + prebuilt
  `libgfx906_persistent_tree_ll_ar_default_*.so`): a persistent tree low-latency AR
  kernel pre-initialized at graph capture, with multi-row/multi-work variants. Attacks
  TP all-reduce cost — the per-step comm that our breakdown couldn't attribute (it's
  outside any hookable module). Found in their `profiles/vnext/hf-dense27b-tp8.env`.
- **Custom RCCL overlay** (`/rccl-overlay/install/lib/librccl.so.1` + hand-tuned
  `RCCL_TREES`, `NCCL_ALGO=Tree NCCL_PROTO=LL NCCL_MIN/MAX_NCHANNELS=4`): replaces
  stock RCCL with a tuned build for their TP=8 topology. Their host is TP=8 full-BAR;
  ours is TP=2 bifurcated-x8 — the env-only knobs (NCCL_ALGO/PROTO/CHANNELS) are the
  transferable part, the prebuilt tree .so is topology-specific.
- **Relevant because:** our phase-profile shows hooked modules = only ~38% of the MTP
  step wall time; the rest is comm + CPU dispatch. Before porting persistent AR (a big
  lift), an env-only RCCL A/B is the cheap first probe.

## Deep review follow-up (2026-09-07)

Beyond the initial recon (optimizations.md → SYV-1..9), the patch set review
added ROADMAP items: **SYV-10** (GDN spec-decode bounds, upstream PR #50021 —
verify), **SYV-11** (KVarN 4-bit-K/2-bit-V KV compression + stock
int4/int8-per-token-head KV — capacity/context-extension tier; their data:
PPL +0.16 %, needle OK to 240k, decode ~20 % slower than fp8), **SYV-12**
(context-lookup verify-block extension for MTP — the new idea: verify
k+8 from prompt-lookup fills when a copy is running, +47 % on reproduce
tasks, lossless), **SYV-13** (mamba chunked-prefill align fixes — verify
only; our fork has the newer V2-align path). Not new: their sort-free
sampler = our shipped SYV-4 (2026-09-03); their Marlin-tune wash under
power cap matches our DVFS standalone-≠production trap.
