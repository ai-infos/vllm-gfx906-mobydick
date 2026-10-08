# MTP-1 — Qwen3.8-27B dense MTP long-context decode: crossover pinned, optz leads quantified

> Branch `gfx906/main` · model `Qwen3.8-27B-AWQ-INT4` (dense GDN hybrid) ·
> date 2026-09-02 · roadmap item MTP-1 (a/b/c).

**VERDICT:** `OPEN` — crossover pinned; optz leads quantified (lm_head DEAD,
attn K-multiplier ~1.0); rocprofv3 kernel breakdown + dynamic-MTP design
pending (blocked by zombie KFD handle from old-vLLM wedge).

**GATE:** same-boot TP=2 graph-mode streaming A/B, pp sweep 1k–120k, n=3 reps,
cold prefill per rep (unique headers defeat prefix cache), tg=256, temp=0.
Data: `/local/tmp/mtp1/data_{mtp,greedy}_bootQ.jsonl` (33 entries each).

---

## HYPOTHESIS

S9 (boot E) showed MTP k=2 transitioning from faster to slower than greedy
between ~8k and ~32k live context. The exact crossover bracket was unpinned
(n=2–3 per point, single boot, prefix-cache confound). Pin it on a clean boot
with separated prefill/live-context, then quantify the remaining optimization
leads (lm_head-per-draft, attention K-scaling, GDN reframe) to identify what
actually causes the long-context MTP tax.

## What was done

- **MTP-1a sweep (boot Q, post-reboot 2026-09-01 ~16:35 UTC):**
  - 11 pp points: [1k, 2k, 4k, 8k, 12k, 16k, 24k, 32k, 64k, 96k, 120k]
  - n=3 reps per point per arm, cold prefill (unique headers)
  - TP=2, util 0.85, chunk 1024, capture [1,2,3,4], maxlen 131072
  - Arms run SEQUENTIALLY (two TP=2 servers cannot coexist: each reserves
    ~85% of both GPUs at init)
  - MTP arm: `mtp1srv@mtp.service` port 8123, completed 20:09 UTC
  - Greedy arm: `mtp1srv@greedy.service` port 8124, completed 20:09 UTC
  - Both arms match S9 within ≤4% at every shared point (MTP) / ~15% (greedy,
    config difference; ratio SHAPE matches)

- **Optz microbench** (`/local/tmp/mtp1/microbench.py`, no full model):
  - lm_head GEMM at batch=1, K∈{1,2,3} rows, TP=2 sharded (124160×5120 fp16)
  - Full-attention decode step at S=122880, K∈{1,3} rows, GQA 24q/4kv hd=256

- **Old-vLLM A/B attempt** (0.23.1rc0.x via docker + in-process PYTHONPATH):
  - Docker: GPU hang at weight load shard ~2/5 → dual-GPU wedge (03:12)
  - In-process: fixed platform detection (sitecustomize RocmPlatform pin),
    still `hipErrorLaunchFailure` at weight load shard ~2/5 → GPU0 wedge (06:10)
  - **Abandoned:** old code path incompatible with this AWQ model + host on
    both userlands. Left zombie KFD handle (dead PID holds 10.99 GB GPU0).

- **joe2gaan/localaiservers qwen36-gfx906 recon** (`/local/tmp/mtp1/joe2gaan_recon.md`):
  - gfx906 Qwen3.6-27B TP=8 deploy bundle, ROCm 7.2
  - Has profiling scripts + vnext profiles with several gfx906-specific knobs

## Evidence — FOR (crossover pinned)

| pp | MTP t/s | Greedy t/s | ratio | winner |
|---:|---:|---:|---:|---|
| 1024 | 58.77 | 40.68 | **1.44×** | MTP |
| 2048 | 55.31 | 39.86 | **1.39×** | MTP |
| 4096 | 48.44 | 38.37 | **1.26×** | MTP |
| 8192 | 53.27 | 35.97 | **1.48×** | MTP |
| 12288 | 45.64 | 33.83 | **1.35×** | MTP |
| 16384 | 39.92 | 31.90 | **1.25×** | MTP |
| 24576 | 31.95 | 28.61 | **1.12×** | MTP |
| 32768 | 26.63 | 25.93 | **1.03×** | MTP (barely) |
| 65536 | 15.95 | 18.86 | **0.85×** | Greedy ← crossover |
| 98304 | 11.19 | 14.80 | **0.76×** | Greedy |
| 122880 | 9.18 | 12.74 | **0.72×** | Greedy |

- **Crossover: between 32k and 64k pp.** MTP wins at ≤32k (barely at 32k),
  loses at ≥64k. Reconciles with S9's ~8k–32k estimate (S9 had MTP losing *at*
  32k; we have it barely winning there — within n=3 noise).
- Acceptance rate = **2.0 stable through all points** (no collapse at 120k).
  The loss is pure O(Sk) step cost, not draft rejection.

## Evidence — AGAINST / optz leads quantified

### lm_head per-draft-token lead: DEAD

| K rows | us (per GPU, TP=2 sharded) |
|---:|---:|
| 1 | 6880 |
| 2 | 7059 |
| 3 | 7202 |

- lm_head at batch=1 is **memory-bound on the weight read** (~1.27 GB/GPU):
  ~6.9 ms regardless of K. Marginal per extra draft-token row = **161 µs**.
- MTP k=2 adds 2 rows → +322 µs/step ≈ **0.4% of the 78 ms step @120k**.
- The cost exists identically in greedy; it is NOT the MTP tax.

### Attention K-multiplier at S=120k: ~1.0 (surprise)

| K rows | us (SDPA GQA proxy, 1 full-attn layer) |
|---:|---:|
| 1 | 307.5 |
| 3 | 314.1 (×1.021) |

- Extra query rows over a long KV cache are nearly free (KV bytes shared).
- The MTP long-context loss is **NOT** attention-compute scaling with K in the
  ideal kernel. It lives in how vLLM executes the multi-row verify + draft path
  (kernel selection, graph mode, or draft-layer cost).

### Budget puzzle at 120k greedy (78.7 ms/step = 12.7 t/s)

Memory-bandwidth floors (MI50 ~1 TB/s HBM):
- AWQ INT4 weight read: ~13–14 GB total / 2 GPUs → ~7 ms floor
- KV read, 16 full-attn layers @ S=122880: 4 kv-heads × 122880 × 256 × 2(K+V)
  × 2B = ~504 MB/layer → ~4 GB/rank → ~4 ms floor
- **Floor total ~11–13 ms vs measured 78.7 ms → ~6× gap unexplained.**

Something in the real vLLM decode path is far from bandwidth-bound. This is
what the rocprofv3 kernel breakdown must find (torch.profiler captures zero
device events on this build — verified twice).

## Why old-vLLM A/B failed

The 0.23.1rc0.x code path wedges GPUs loading this AWQ model at shard ~2/5
(~40%) on BOTH userlands (ROCm 7.2 docker, ROCm 7.14 in-process). The old fork
predates our gfx906-specific weight-load and FA fixes. Not an env issue:
platform detection was fixed (sitecustomize RocmPlatform pin), triton confirmed
active post-CUDA-init, all build artifacts present. The code itself is
incompatible with this model+host at load time.

**Residual:** zombie KFD handle from the 06:10 wedge — dead PID 44222 holds
10.99 GB GPU0, kill -9 no-op, kernel-level amdgpu leak post-reset. Only a
reboot clears it. GPU0 now ~23 GB free → cannot host TP=2 @0.93 (~29.7 GB).

## Interactions / superseded-by

- Supersedes S9 crossover estimate (8k–32k) with pinned bracket (32k–64k).
- The lm_head lead was the #1 candidate from source analysis of `step3p5.py`;
  microbench kills it. Redirects MTP-1b toward the budget puzzle (what IS the
  78 ms made of?).
- The attn K-multiplier ~1.0 finding means the O(Sk) tax is NOT in the FA
  kernel's query-row scaling — it's in the vLLM execution path around it.
- Old-vLLM comparison was motivated by user report that "0.20.0 kept MTP
  benefit longer." Cannot verify on this host (code incompatible). The pinned
  crossover (32k–64k) + acceptance=2.0 throughout is the ground truth for our
  current build regardless.

## Refrigerated residue

- **rocprofv3 decode kernel breakdown @120k** — ready to run post-reboot.
  Script prepared (`/local/tmp/mtp1/profile_steps.py` needs rocprofv3 wrapper).
  Will answer: what is the 78 ms made of? (attention vs GEMM vs comm vs dequant
  vs graph overhead). This is the real MTP-1b gate.
- **Dynamic-MTP feasibility** — source analysis complete (variable-K wired,
  per-request K=0 expressible but NOT zero-overhead, ctx available at decision
  point; costs: forces PIECEWISE cudagraph + batch-size-only keying). Write-up
  pending in this devlog.
- **joe2gaan vnext profiles** — TP=8 configs with several gfx906 knobs worth
  mining for our TP=2 setup (see `/local/tmp/mtp1/joe2gaan_recon.md`).
- **syv-ai fork split-KV + fp16 SSM-state** — external lead, not yet checked.

## Search keys

`HYPOTHESIS: MTP k=2 crossover between 32k and 64k pp on clean boot Q`
`VERDICT: OPEN (crossover pinned; optz leads quantified; kernel breakdown pending)`
`CROSSOVER: 32768→65536 pp, ratio 1.03→0.85`
`ACCEPTANCE: 2.0 stable through 122880 pp (no collapse)`
`LM_HEAD: DEAD (memory-bound, 161us/tok marginal = 0.4% of step)`
`ATTN_K_MULT: 1.021 at S=122880 (KV bytes shared, query rows nearly free)`
`BUDGET_PUZZLE: 78ms/step vs ~12ms BW floor = 6x unexplained`
`OLD_VLLM: ABANDONED (code incompatible, GPU wedges at load, zombie KFD handle)`

## 2026-09-02 (boot S) — MTP-1b: k=1 arm + kv_split verify fix (SUPERSEDES the 32k-64k crossover verdict)

```
K1_ARM:   k=1 beats greedy 1.68-1.73x and k=2 1.98-2.40x at 64k/96k/120k (n=3 each)
          -> the "crossover" was K=2-specific; no crossover for k=1 in this range
KV_SPLT:  root cause found — both FA paths clamped `if (seq_q > 2) kv_split = 1`
          (a PREFILL OOM guard); spec-decode verify presents k+1 query tokens as
          ONE sequence, so k=2 verify (seq_q=3->pad 4) lost ALL KV-split
          parallelism on the O(Sk) full-attention that is ~73% of a long-ctx step
FIX:      byte-budget guard (GFX906_FA_KVSPLIT_MAX_BYTES, default 512 MiB) in
          both paths — verify keeps kv_split>1, real prefill still forced y=1
A/B:      k=2 fixed vs baseline @64k/96k/120k: 37.95/29.88/25.70 vs
          15.95/11.19/9.18 t/s = 2.38x / 2.67x / 2.80x (n=3, cold prefill)
          -> k=2 is now the BEST static config at >=64k (beats k=1 at every point)
CORRECT:  kv_split in {1,8,16} bit-identical for Sq in {1,2,3,4,256,1024}, both
          paths; torch causal-ref match <=2.2e-4 (kvsplit_verify_test.py recipe)
REVIEW:   self + claude CLI pre-merge review — no blockers, 2 minor fixed
MERGED:   a6ff64a71b to main (branch gfx906/mtp1b0-kvsplit-verify)
GPU0:     wedge #7 this session (SetDevice, pre-FA) — 7 non-deterministic wedges
          across two boots, canaries passing between each -> HW-degradation signal,
          RMA/replace question raised with Kevin

## 2026-09-03 (boot T) — PPL/coherence gate for the kv_split merge: GATE-PASS

```
GATE:     ppl_gate.sh on clean boot T (post-reboot, GPUs settled ~60s)
ARMS:     greedy vs mtp k=2, in-process TP=2, enforce_eager, maxlen 32768,
          12-prompt set (same as /local/tmp/ppl_nh5_090.py), prompt_logprobs=20
PPL:      greedy 10.517902 vs mtp2 10.517902 -> rel drift 0.0
COHERENCE: first naive check FAILED (14.8% token match, first diff @32) —
          root cause: the two arms use DIFFERENT attention numerics (greedy
          seq_q=1 keeps kv_split=8; mtp verify seq_q=4 uses kv_split=8 with a
          different tile reduction order). Control: two PURE-GREEDY runs
          differing only in GFX906_FA_KVSPLIT (8 vs 1) diverge IDENTICALLY
          (first diff @32, 218/256 diffs) -> FP non-associativity across
          KV-tile split orders is the floor; MTP adds ZERO extra divergence.
VERDICT:  GATE-PASS (ppl_gate_verdict.py v2: mtp signature == noise floor)
DATA:     /local/tmp/mtp1/ppl_gate_{greedy,mtp2}.jsonl, pk1_greedy.jsonl
```

**Wedge #8** at 06:47:08 (control-run mtp2 arm startup, BACO-recovered,
canary passed after). 8 non-deterministic GPU0 wedges on 2026-09-02/03;
4 of them at engine/server startup since boot R. HW-degradation assessment
unchanged: escalate for RMA/replace decision.

**MTP-1b-0 kv_split merge is now FULLY CLOSED**: fix + docs (a6ff64a71b),
review fixes (7eb8b5d08e), DEVLOG (4d9a1d3b13), degradation logs
(c183407df2 + this entry). k=2 on the fixed build is the best static config
at >=64k context.

DONE:     PPL/coherence gate GATE-PASS on clean boot T (see 2026-09-03 entry below);
          SYV/J2G recon ideas recorded in ROADMAP (SYV-1 superseded by fix)
```
