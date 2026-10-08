# gfx906 roadmap — open work, priority-ordered

**Open work is tracked as GitHub issues** on
[`KIntegrated/vllm-gfx906-mobydick`](../../issues). This file is the ordered
index: one entry per open item, with the issue link, the deliverable and the
gate. Item IDs (C*, G*, L*, N*, U*, HK*, QSA-FN-*, MTP-*, …) are stable across
reorganizations — cite them, not filenames.

- **Closed work → [`CHANGELOG.md`](CHANGELOG.md)** (positive/negative verdicts,
  rejected and superseded items). Closed items must not appear here.
- **Parked work → [`REFRIGERATOR.md`](REFRIGERATOR.md)** (parked with a reopen
  gate; no issues are opened for these).
- **Closed negatives → [`DEAD-ENDS.md`](DEAD-ENDS.md)**.

**Workflow contract:** create an issue for new work; close it only together
with a `CHANGELOG.md` entry and the removal of its line here. Measurements
follow `AGENTS.md` (identical prompts, `prompt_sha1` asserted, interleaved
arms, mclk logged).

Reference workload: Qwen3.5-35B-A3B-AWQ on one MI50 — 40 MoE layers,
E=256, topk=8, hidden=2048, W4A16 group-128 experts; B=1 decode step
≈ 15 ms at 66.5 t/s. Priority = expected gain × confidence ÷ effort+risk.

## Now — user-requested / high priority

### QSA-FN-9 — land Qwen3.8-Flash-Next PR #2 and close the validation · [#4](../../issues/4)

External tester gate met: 4× MI50, TP=4, fp16, 147 456 ctx, MTP k=3, PLE table
mmapped to host RAM → **46.8 t/s B=1**. Their six commits are cherry-picked onto
`gfx906/qsa-fn` under their authorship; ours are the cleanups. **Remaining:**
land PR #2, and the PLE comparison (QSA-FN-15). **Gate:** PR #2 review discharge
+ QSA-FN-15; QSA-FN-7 non-regression stays green. Refs:
`REVIEW-pr2-qsa-fn.md`, `DEVLOG-qwen38-flash-qsa.md`.

### QSA-FN-13 — report the pinned-id hazard upstream (#57497) · [#8](../../issues/8)

Two PLE id-bug classes (stale/uninitialised pinned ids; the capture-time
`0xff80ff80…` variant) transfer to upstream's pinned mechanism. **Deliverable:** a
comment/issue on `vllm-project/vllm#57497` with the mechanism and the
fold-and-warn recommendation. Cheap, upstream value.

### QSA-FN-15 — PLE offload comparison (bespoke mmap vs upstream pinned/UVA) · [#9](../../issues/9)

Decides whether the tester's `MmapShardedNGramEmbedding` survives QSA-FN-11.

**Arm A is closed as a dead-end** (2026-10-05, `DEAD-ENDS.md`): the generic UVA path cannot express
this table. With `--cpu-offload-gb 24 --cpu-offload-params ngram_embedding` the flags are accepted
(`'cpu_offload_params': ['ngram_embedding']`) and the path engages (`Offloader set to UVAOffloader`
per worker) — yet the KV cache (**72,089 tokens**) and the VRAM footprint (**27,956/27,996 MiB**)
are byte-for-byte the no-flag arm's, and decode is inside the 2.4 % interleave drift. The table is
CPU-resident by design: `MmapShardedNGramEmbedding` maps the shard files directly, every rank shares
one page-cache copy, and `set_shard()` refuses anything but CPU tensors — the shards "are not
parameters", so a per-parameter offloader has nothing to move. The mmap *is* the offload.

**Arm B remains open, and it is why this item stays**: the deciding datum is pinned host RAM
(26 GiB now) vs page cache, on the tester's 128 GB box. This host has 46 GB, so no honest
eviction-pressure experiment exists here — the only local contribution is the served host-memory
profile. Ref: `DEVLOG-qwen38-flash-qsa.md` (2026-10-05).

### UP-1 — upstream the mamba `align` seed fix (V2-MAMBA-1) · [#10](../../issues/10)

One line (+assert) in `mamba_hybrid.py` `add_request`; the bug is live upstream
and only masked there. Lead the PR with the failing unit test (no GPU model
needed). Human owner required (`AGENTS.md` §1). Ref:
`DEVLOG-v2-mamba-align.md`.

### UP-2 — upstream fp16 QSA for Qwen3.8-Flash-Next · [#11](../../issues/11)

Feature, not bugfix; on gfx906 also a 4.4× kernel win. Shared
`common/qsa_cache.py` edits must stay dtype-general (no CUDA path change); the
upstream copy has moved (103/25). Needs a model eval arm we cannot run here.

## Next — 0.30-post and 0.31

### QSA-FN-10 — next tester measurement list · [#5](../../issues/5)

Prefill breakdown, MTP depth k=2/3/4, long-context quality, drafter graphs,
prefix caching at 147k — one ~30 min session on the tester's box.

### QSA-FN-11 — adopt upstream's official PLE CPU offload · [#6](../../issues/6)

Blocked on a 0.30.1-based line (`vllm-project/vllm#57497`). Verify
`VLLM_PLE_CPU_OFFLOAD=1` engages with the tiny rig, port #57497's AMD half,
validate on the real checkpoint, then delete the bespoke mmap path.

### U30-1 — Nemotron-H: separate/quantized MTP lm_head + latent-MoE TP>1 AR · [#12](../../issues/12)

Re-run the Nemotron TP=2 decode A/B on the 0.30 base; confirm the lm_head loads
and whether #52301's ~13 % skip fires under EP. **Gate:** decode t/s + PPL
26.96–27.02.

### U30-2 — persistent top-k on low-shared-memory GPUs · [#13](../../issues/13)

Check which top-k kernel the gfx906 build selects and whether #54110's fallback
triggers; measure only if it does.

### U30-3 — W4A16 packed zero-points for Gemma-4 AWQ · [#14](../../issues/14)

Confirm the served checkpoint's zero-point layout; the fork's symmetric no-zp
expert kernel may make upstream #54965 inapplicable.

### TP-1 — TP-scaling probe (prefill + decode vs TP) · [#15](../../issues/15)

TP=2 is the ceiling on 2× MI50. Two wedge-light TP=1 loads at pp ∈ {32768,
65536} × tg 256, greedy and MTP k=3; the 120k point does not fit TP=1. **Gate:**
prefill TTFT ratio + long-ctx decode scaling vs the recorded TP=2 anchors. Ref:
`ttft-prefill-stall.md` §13.11.

### FA-COVER-2 — remaining CUSTOM-FA coverage classes · [#16](../../issues/16)

`FA-COVER-1` mapped the fallbacks and shipped the guard + head-size padding;
what remains is the **text-path head-dim padding** (61 census rows), attention
sinks (72) and encoder attention (36). Ref: `DEVLOG-fa-coverage.md`,
`tools/fa_coverage.py`.

### MBT-1r — residual own-context per-token KV streaming · [#17](../../issues/17)

FIX-H2 cut the B=4/120k wall 75.3 → 44.8 min; the residual ~1.2 ks slope is the
own-context per-token streaming. Also re-anchor the published B=1 prefill sweep
(one point, 64k). Ref: `ttft-prefill-stall.md` §13.14/§13.16.1.

### MTP-1b — remaining Qwen3.8 MTP optimization opportunities · [#18](../../issues/18)

Candidate list cross-checked against `DEAD-ENDS.md` and the spec-decode / FA
dev logs (2026-09-27; full table in the issue). **Covered, do not re-open:**
SYV-2 (our n-gram/prompt-lookup drafter probe is a 0.68× dead-end; the
block-extension form is SYV-12, parked) and SYV-9 (the FA already runs int8 Q8
K at full rate; the Q-side is C6, rejected; the format upside is M6 Part C,
refrigerated). **Partial:** CAT-2 (GQA head packing already shipped; only
head-dim Split-D remains) and J2G-5 (a tuned AMD_GFX906 MoE config already
ships in-tree). **Not applicable:** CAT-4 (the source XQA change has no
counterpart; wide aligned loads are already our latency-hiding rule). **Open:**
SYV-7b, CAT-5, CAT-6 (profile first), J2G-2/3/4/6/7, and SYV-11 + CAT-3 — the
last two blocked by the custom FA's fp16-only `supported_kv_cache_dtypes`.
`MTP-1a`/`MTP-1b-0` are closed (k=2 wins at long context once the `kv_split`
clamp is fixed).

### MTP-1c — dynamic MTP depth / per-request policy · [#19](../../issues/19)

Design work only until MTP-1b shows a real remaining static win; the fork's
dynamic-SD keys on batch tokens, not context length. **Gate:** beat the best
static config on a mixed-context workload.

### C5 — fuse the shared-expert chain · [#21](../../issues/21)

One chain kernel removes two launches per layer (~150–250 µs after the
critical-path discount). Bit-correctness + serving A/B. Ref:
`DEVLOG-moe-m1-sprint.md`.

### N2 — B>1 FA direct store · [#22](../../issues/22)

Store directly into `[B,Hq,D]` for batched single-token decode; B=1 is already
copy-free. Ref: `DEVLOG-fa-attention.md`.

### INT8-PACKED-1 — finish pack-quantized int8/W8A16 support · [#23](../../issues/23)

The checkpoint loads and generates (2-line `embed_tokens` wiring fix). Remaining:
templated numerics gate vs bf16, a serving/perf measurement, the same wiring gap
in `qwen3_5_mtp.py`, then the DFlash2 INT8 pairing arm. Ref:
`DEVLOG-int8-packed.md`.

### MUSE-2 — Muse-Glimmer + official DFlash assistant · [#25](../../issues/25)

Now working with graphs on (≥ +29 % decode, acceptance 2.82–3.18). Remaining:
B=4, the TP=1 path, and dropping the obsolete RBLOCK workaround. Refs:
`DEVLOG-muse-glimmer.md`, `DEVLOG-fa-noncausal.md`.

### FA-STRUCT — remaining FA decode headroom (structural) · [#28](../../issues/28)

KV re-read elimination across q-tiles (~6–9 %), online-softmax rescale batching
(~2.5–4 %), and the PPL-gated M6 Q4-KV bet, plus four housekeeping items.
Ref: `DEVLOG-fa-verify-sq8.md`, `DEVLOG-fa-kernel-batches.md`.

### C7 — persistent/cooperative MoE block · [#29](../../issues/29)

Verify HIP cooperative-launch support and resident-grid capacity on Vega 20
first; follows C1–C3.

## Later / backlog

### LING-1 — Ling-3.0-tiny onboarding (L1–L5) · [#30](../../issues/30)

`BailingMoeV3ForCausalLM`, ~7.5 B BF16: load, profile, then decide on a W16A16
expert kernel, routing specialization and the KDA/MLA workstreams. Stop rule: a
hard MLA/KDA blocker parks Ling.

### NH-LEFT — Nemotron-3.5-Lightning leftovers · [#31](../../issues/31)

NH-4 default flip (after a non-GEMV-bound config), NH-2′ M≤6 revival, NH-6,
TP=2 validation, the batched fp32 GEMV, and the 6 GQA layers on ROCM_ATTN.
Ref: `DEVLOG-nemotron-h.md`.

### HK-1 — drop the legacy env sourcing from `/local/git/AGENTS.md` · [#27](../../issues/27)

In-repo recipes are done; the remaining edit is the protected file plus the
session `canary.sh` line.

### GEMMA4-1f — templated reference gate for Gemma-4 / Muse-Glimmer · [#24](../../issues/24)

Record a templated reference (or decide `ift_chat_gate.py` is the gate and drop
the PPL form). Ref: `benchmarks/kernels/gfx906/ift_chat_gate.py`.

### UPR-1 — upstream queue (vLLM): U1–U5 · [#32](../../issues/32)

fastsafetensors GDS fallback, hipify in-source guard, GemmaRMSNorm fused
dispatch, asymmetric W4A16 qzeros repack, review hardening. Human owner + §1
duplicate checks per item.

### UPR-2 — upstream queue (ROCR/TheRock): ROCR-1, ROCR-2 · [#33](../../issues/33)

`IPCRecvHandle` EOF spin and the EventPool permanent allocation latch. Ref:
`cpu-stuck-threads.md`.

### UPR-3 — report upstream: top-k kernel memory corruption above width 8192 · [#37](../../issues/37)

`top_k_per_row_decode` (runtime `topK`) raises an illegal memory access at selection widths
12288/16384 instead of a clean error; 8192 is verified good, so upstream's `{512, 2048}`
config rule is hiding a safety limit behind a geometry reason. Evidence in
`DEVLOG-wht1-qsa-topk.md`.

## Standing requirements (not issues)

- **QSA-FN-7 — non-regression gate** runs with every QSA item: FA suite,
  in-process PPL on Qwen3.8-27B-AWQ-INT4 (recorded **10.5472**, 0 top-20
  misses; do not use Qwen3.5-27B, which reads 14.3750), Nemotron band
  26.96–27.02, MoE 35B `_bench_gfx906.py` pp2048/tg256. Any QSA change to a
  shared file also needs a bf16 arm.
- **Measurement hygiene** — `docs/gfx906/AGENTS.md` (identical prompts,
  `prompt_sha1`, interleaved arms, mclk, fresh-boot canary). The 2026-09-13/15
  lessons are the reason this is a standing rule, not a per-item note.

## Open questions

- Why does the Q8 side-buffer KV read path win big with MTP k=3
  (−15.5 %/−19.1 % ms/step at 64k/120k) but lose ~6 % at B=1 greedy decode?
  Both same-boot, acceptance unchanged; the sector-waste theory does not
  predict the spec-decode win. (Matters only for a non-spec-decode default;
  `DEVLOG-fa-legacy0-b1-decode.md`.)
- What exact call site accounts for the remaining roughly 158 µs/step of
  MoE-adjacent `[1,2048]` copies?
- What is llama.cpp's component-level kernel budget on the same MI50?
- Does `topkGating`'s cost come from structure or a hidden memory round trip?
  (feeds C1)
