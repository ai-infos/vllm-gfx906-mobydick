# Dev log — gfx906 custom Q8 FA & decode backend

> Topic: the custom Q8 FlashAttention backend, the B=1 decode-parallelism
> track, and the fused-gather/fill pile. Split from `DEVLOG-moe-opt.md`
> (2026-08, topic consolidation) and re-split 2026-09-13 (size budget).
> MoE → `DEVLOG-moe-opt.md`; dense 27B / the P3-2(b) GEMV →
> `DEVLOG-dense-decode.md`; the FA *sub-tracks* → the pointers below.

**VERDICT (top-level):** the custom Q8 FA backend went from dead code to the
gfx906 default; the decode-stack attention+copy work is what took MoE to
67.39 t/s and dense to 25.60 t/s. `CUSTOM` is both the win and several hard-won
traps (stride bugs, capture lifecycle, the V1/V2 gather serving degradation).

**GATE (all serving A/Bs here, unless a row says otherwise):**
`_bench_gfx906.py`, pp=2048/tg=256, single request, `FULL_DECODE_ONLY`
(bench-env recipe in `running.md`). GEMV = the P3-2(b) `dense_gemv` op.

**Full detail (pre-split, 131 KB):**
`git show e963fd8c62:docs/gfx906/DEVLOG-fa-attention.md`

## Where the sub-tracks now live (moved 2026-09-13, nothing deleted)

| topic | log |
|---|---|
| M5 ISA-rate refutation, M3 hygiene batch, M2 per-q-tile prefill clip | `DEVLOG-fa-kernel-batches.md` |
| Sq=8 verify deep-dive, KVSPLIT shape-aware default, R3, ncols1=5/6 | `DEVLOG-fa-verify-sq8.md` |
| Real-payload MTP depth matrix (k=2/3/4/5), k=3 serving default | `DEVLOG-mtp-depth-matrix.md` |
| SYV-12 (fill-from-draft-buffer) + SYV-13, incl. the V1-port history | `DEVLOG-syv12.md` |
| FD-1 (`VLLM_GFX906_FUSED_DRAFT` offline gate at mtp3b4) | `DEVLOG-spec-decode.md` |
| A3 (same flag, serving A/B @k=4: NEUTRAL, **stripped** 2026-09-13) | `archive/a3-fused-draft` + `A3-REVIVAL.md` on that branch |
| T-1 int8 mass (drafter lm_head) | `DEVLOG-t1-int8-fp16-mass.md` |
| N4 max-model-len decode tax (persistent live-bounded gather) | `DEVLOG-masked-fa.md` |
| Split-K long-context accuracy + tile-clip test rework | `DEVLOG-fa-splitk-accuracy.md` |
| LEGACY=0 vs LEGACY=1 B=1 flip adjudication | `DEVLOG-fa-legacy0-b1-decode.md` |
| FIX-H2 pad-tile prefill tax + host cu_seqlens (M3) | `DEVLOG-fa-multibatch-prefill.md` |
| Bench recipes for the FA decode/gather micro-benches | `BENCH-fa-decode.md` |

---

## 2026-08-15 — P3-3: the custom (Q8 FA) backend was dead code; making it live surfaced three stride bugs

**Starting point:** attention 1.94 ms/step (10 FA layers × 194 µs) vs
llama.cpp 0.19 ms. The repo vendors a custom Q8 FA backend
(`vllm/gfx906_fa/` + `csrc/gfx906_fa/`, llama.cpp's `flash_attn_tile_q8`,
head_size 256), integrated as the gfx906 default — **but dead at runtime**:
the `vllm.general_plugins` entry point is missing from the stale
`vllm.egg-info` *and* the image's dist-info, so `CUSTOM.is_overridden()` was
always False and everything silently fell to Triton.

**Fix:** `vllm/platforms/rocm.py::_get_backend_priorities` registers the plugin
explicitly on gfx906 when not already (idempotent; entry-point installs still
win) → `[CUSTOM, ROCM_ATTN, TRITON_ATTN, TURBOQUANT]`.

**Root cause of the "garbage output" hunt (a reusable trap).** With the
backend live, `GFX906_FA_LEGACY=0` (Q8 side-buffer + fused gather) produced
`'!!!!!...'`. Isolation ladder (each step verified): `reshape_and_cache_q8` ≡
`quantize_q8_0` byte-identical; synthetic gather ≡ torch `_gather_kv_q8`;
synthetic end-to-end vs fp32 SDPA correct (rel ~2.5e-3) — the first two
"references" were wrong (einsum axis bugs; pairwise A/B/C identity checks kept
it honest); in-model double-check `K identical, V NaN` — synthetic passed only
because the test caches were contiguous. **Real cause:** `value_cache =
kv_cache.unbind(1)` of `[num_blocks, 2, block, Hkv, D]` is non-contiguous
(block stride 2×), and `gather_paged_kv_q8` / `forward_paged_direct` /
`reshape_and_cache_q8` **computed strides from shapes, ignoring real tensor
strides** → K bytes read as V; only block 0 looked sane. Fixed all three sites
to `tensor.stride(i)` (× element size for fp16 V) + contiguity TORCH_CHECKs on
the last dim.

**Eager single-req (superseded by the serving matrix below):** Triton 19.49
best; CUSTOM LEGACY=1 18.49 (FA kernel 194→72 µs/layer, **2.7×**); LEGACY=0
FUSED 19.33; DIRECT 19.21 — at B=1 eager the FA win is eaten by gather/
conversion tax (eager is launch-bound anyway). Also fixed: `_bench_gfx906.py`
counted tokens by re-encoding output *text* (garbage re-encodes shorter — the
"mystery 32 tokens"); now counts `token_ids`.

**Learnings:** stride bugs hide from synthetic tests that build contiguous
caches — mirror the real allocation path (`unbind` views); silent registration
fallbacks make dead backends invisible — assert the backend you expect; Q8 K
quant shifts logits ~1e-3 (greedy diverges from fp16 after ~10–25 tokens, both
fluent — the same trade llama.cpp makes); pairwise A/B identity checks beat
building a math reference from scratch.

## 2026-08-15 — Day 1: gather go/no-go GO; aiter structurally a no-op; LLGemm1 retune not worth it

`gather_paged_kv_q8` (B=1, Hkv=2, D=256, bs=16, pre-allocated) byte-identical
to torch incl. V-tail zeroing: **Sk 2048 18.6 µs (2.3× floor) · 2816 21.7 µs
(2.0×) · 3328 25.3 µs (1.9×)**. Q-fp32 side costs/layer (Sq=1): `q.float` 3.9 +
`q_pad.zero_` 2.7 + q copy 7.0 + out unpack 8.3 = **21.9 µs** — as big as the
gather itself. Combined tax 43.6 µs/layer vs the 122 µs/layer FA win → net
**~0.78 ms/step** over 10 layers.

- **P3-2(a) aiter probe — STOPPED (structurally a no-op on gfx906).**
  `VLLM_ROCM_USE_AITER` defaults False and every aiter gemm path sits inside
  `if not on_gfx906()` (`gemm_a16w16`, `wvSplitKrc`; `ops.wvSplitK` explicitly
  excluded — "matrix cores not supported"); the triton-gemm whitelist is
  GPT-OSS shapes only. **Arch exclusion, not shape/dtype selection.**
- **LLGemm1 rows_per_block sweep** (dispatch hardcodes 4): rpb8 best weighted
  (5523 vs 5604 µs/step, +1.4 %), rpb2 worst (6626). No shape moves ≥20 % →
  retune not worth it.
- **P3-2(b) scoping:** big rows (in_proj/LM/qkv/o_proj) are BW-bound at rpb=4
  (0.95–1.29× floor, ~0.2–0.5 ms/step); small rows (gate_up/down/router/
  GDN-small, 150 calls/step) are launch/latency-bound (3.6–14× floor). Realistic
  ceiling ~0.7–1.1 ms/step.

**(P3-2(b) itself — the W16A16 dense GEMV for M=1 — is a dense-decode topic:
see `DEVLOG-dense-decode.md`. Summary: v1's K-split hypothesis was WRONG
(2.4–4.2× slower than LLMM1); v2 (RPT=2, kc=4096 single pass) won −23 % qkv,
−17 % router, −6 % in_proj/LM head → 5203 vs 5604 µs/step = **−401 µs
(−7.2 %)**; shape rule K=2048 ∧ (N==256 ∨ N≥2048); kill switch
`VLLM_GFX906_DENSE_GEMV=0`.)**

## 2026-08-15 — Serving-mode findings: `VLLM_ATTENTION_BACKEND` is gone; the FULL_DECODE_ONLY downgrade bug; PIECEWISE+CUSTOM = first real win

Two silent-knob facts: (1) `VLLM_ATTENTION_BACKEND` **no longer exists** in
0.27.2rc1.dev — the backend is the priority selector
(`attention_config={"backend": ...}` was the new knob); (2) the gfx906 FA
plugin now wins by default.

Then the surprise that set the serving config: the same binary measured 22.44 t/s
(GEMV off) / 22.58 (on) when **FULL_DECODE_ONLY was requested and downgraded**,
but **52.07 t/s when PIECEWISE was requested** — beating the 44.09 Triton-FULL
reference. **Mechanism:** a FULL_DECODE_ONLY request is a non-piecewise graph;
the `CGSupport.NEVER` downgrade then runs PIECEWISE *runtime* over it → decode
degrades toward eager. Requested-PIECEWISE compiles attention out → proper
piecewise graphs + the 72 µs/layer CUSTOM FA win. The downgrade path is a real
engine bug (kept in P3-3a scope); **requested-PIECEWISE + CUSTOM became the new
best config**.

## 2026-08-15 — P3-3a correctness + the serving A/B matrix (GATE)

`/bench/probe_custom_fa.py` (2048-tok filler, 128 greedy): ROCM_ATTN+FULL vs
CUSTOM+PIECEWISE **identical 128/128** (degenerate-repetition fingerprint —
breaks any V-stride bug instantly; Sk 2048→2176). `GFX906_FA_CG=decode` then
showed FULL decode capture works on LEGACY=1 (**53.09 t/s**, no downgrade, no
crash) → hypothesis confirmed: the decode fast path (`max_seqlen_q==1`) takes
no dangling host loop.

| # | attention | requested CG | GEMV | FA_CG | t/s | notes |
|---|---|---|---|---|---|---|
| 1/2 | CUSTOM | FULL_DECODE_ONLY | off/on | never | 22.436 / 22.584 | downgrade bug; GEMV +0.7 % |
| 3 | CUSTOM | PIECEWISE | on | never | **52.074** | probe-verified correct |
| 4 | CUSTOM | PIECEWISE | off | never | 50.877 | clean GEMV A/B: **+0.45 ms/step (+2.3 %)** |
| 5 | CUSTOM | FULL_DECODE_ONLY | on | decode | **53.094** | M2 capture; correctness probe pending |
| 6/8 | ROCM_ATTN | FULL / PIECEWISE | off/on | — | 43.986 / 43.955 | reproduces 44.09; "piecewise penalty" refuted |
| 9 | CUSTOM | FULL_DECODE_ONLY | on | default | 52.90 (σ 0.06, 5 samples) | W8 default flip |
| 10 | CUSTOM | FULL_DECODE_ONLY | on | default | 49.56 | **V2 fused gather — REGRESSION** |
| 11/12 | CUSTOM | FULL_DECODE_ONLY | on | default | 56.92 / **57.09** (σ 0.09) | V1 fused gate (`GATHER_V=1`) — new best |

**M2 closed:** FULL-path correctness PASSED (probe2 Triton-FULL vs CUSTOM-FULL,
128/128), `get_cudagraph_support` now returns UNIFORM_SINGLE_TOKEN_DECODE by
default (no more downgrade here; the bug remains for other NEVER backends),
5-sample 52.90 ± 0.06, and T3 `test_cudagraph_capture_replay_legacy_decode_path`
(warmup@small-Sk → capture@capacity; multi-size B=1→2 with B=1 replay; live
seq_lens 100→200 with K/V refill). Debug detour worth keeping: a `.tolist()`
inside a debug print raises "Cannot copy between CPU and CUDA" — the error is
the *print*; `arange(n).view(2,-1)` silently loses half the block-table columns.
**Headline: 22.44 → 52.90 t/s (2.36×).**

## 2026-08-15 — Re-baselined decode budget @52.90 (rocprofv3): the fused-gather track was demoted too early

Per-dispatch overhead inflates absolutes ~10–15 % under the tracer; shares are
reliable. Top (µs/step): `dense_gemv` 4366 (qkv/in_proj/router/LM head),
FA kernel 3272 (11.3 % ≈ 327 µs/layer @Sk~2176 ≈ 4.3× the 72 µs at Sk~500 —
Sk-linear), LLMM1 2505, MoE wna16 2390 + routing ~1900, GDN rec/conv ~590, and
the **LEGACY FA gather+side pile** (torch gather + copies + Fill + quantize +
q_pad zero) ≈ 4–5 ms.

**M0-3 resolved:** the LEGACY=1 attention slice is not "FA 327 + gather ~40 µs"
— the torch fancy-index `_gather_kv` costs **128–190 µs/layer** (189.5 @2048,
128.3 @2816) vs the fused gather's 19–25. **The v3 demotion of the M1
fused-gather track was premature: it was the biggest remaining lever
(~0.9–1.4 ms/step).** Route B chosen — stage 1: fused fp16-K gather
(`gather_paged_kv_fp16`, byte-generic, bytes_per_row=2D; K stays fp16, quantize
still runs on gathered K); stage 2 (if quantize is visible): fused fp16→q8
quantize-during-gather.

## 2026-08-16 — Route B stage 1 LANDED (V1 default); stage 2 LANDED (fused quantize); 63.56 t/s record

**Stage 1 — fused fp16 gather, and the V1-vs-V2 serving trap.** Built
`gather_paged_kv_fp16` (C++ over the byte-generic kernel) with a Python LEGACY
branch (`GFX906_FA_TORCH_GATHER=1` reverts). Bug en route: **stride-domain
mixup** — K is `const uint8_t*` (byte strides ×2), V is `__half*` (element
strides); L=32 "worked" by luck, L=512 faulted. Correctness: probe3 128/128
**bit-exact**. But serving regressed: **V2 fused 49.56 vs torch 52.83** —
isolated the kernel is 27–42 µs in every state, while the serving profile shows
**~285 µs/call uniform** (p10–p90 282–287) on the full decode graph; the
per-token **V1** kernel (`GFX906_FA_GATHER_V=1`, grid(B,Hkv,Sk), 64 thr, no
barriers, 16× more workgroups) hit **56.92** → launcher default flipped to V1
(`V=2` selects V2). 5-sample confirm **57.09 ± 0.09**; llama.cpp gap 1.23×.
Probe artifact to avoid: one L2 test put the evictor `zero_()` inside the timed
window (256 MB of zeroes ≈ 320 µs masquerading as gather cost). Also:
rocprofv3 grid-axis columns are untrustworthy in this build — use
timestamps/durations.

**Post-FA-track trace** (`rocprofv3 --kernel-trace`, 55.48 t/s under tracer,
17.49 ms/step kernel budget, GPU 99.5 % busy; shares reliable): `dense_gemv<2,2048>`
3936 µs (LM head 1 call ≈1138 µs at 0.9× floor — nothing left) · `LLGemm1<Half,4>`
2021 (micro-bench already adjudicated AGAINST GEMV) · `moe_gemm_q4` 2662 ·
`FillFunctor`+copyBuffer **1178** (uncharacterized pile; FA ~10 small q_pad
zeros — candidate P3-4) · `topkGating`+align+count_sort 1044 ·
`flash_attn_tile_q8<256,256,2,8>` 475 (10 calls; FA stack ≈621 vs 3272 pre-track)
· `fa_split_combine` 146 · `quantize_q8_0_dense` 284 ← stage-2 target ·
`gather_paged_kv_q8` (V1) 174 ← stage-2 target.

**Stage 2 — quantize-during-gather (`GFX906_FA_FUSED_QUANT`, default on).**
Replaces `gather_paged_kv_fp16` + `quantize_q8_0` (174 + 284 = 458 µs/step at
B=1, both launch-bound at 78–18 GB/s) with one kernel/layer: V fp16 copy (V1
semantics, tail zeroed) + K read from the fp16 paged cache quantized to q8_0
in-kernel (bit-exact shared `quantize_block_q8_0_halfwarp`). Isolated: Sk=2176
41.7→25.6 µs; Sk=3328 64.3→36.9 (−27.4 × 10 ≈ **−274 µs/step**). Serving A/B:
OFF 62.594/62.695 vs fused **63.534/63.581 → 63.56 t/s (+1.47 %)**. K_q8 is
asserted **bit-equal** to quantize(gather) on the production `unbind(1)` layout
(3 shapes).

**Phase-3 code review absorbed** (the same session): C1 arch-gates the GEMV
dispatch (`on_gfx906()` at both `_llmm1_tiny_m` call sites); F1/F6 make the
q_pad/gather buffer lifecycle capture-safe (no free-then-realloc on grow under
capture; retired keep-alive list — **superseded by the 2026-08-24 design
below**); F4 hardens `VLLM_GFX906_GEMV_RPT`; F5 adds the V1
`block_tab_idx` guard + V1→V2 auto above `Sk > 65535`; F7 makes LEGACY=0 +
prefix-caching a loud ERROR; F9/F10 hygiene (dead `gathered_sk`, translated
vendor comments, SPDX, gitignores). F3 evidence: PPL CUSTOM 6.6811 vs Triton
6.6775 (**+0.05 %** — Q8-K is PPL-negligible), and the multi-batch greedy
non-determinism is **engine-level** (MoE routing near-ties) — pure Triton shows
the same run-to-run spread. H3/M3 root-caused the V2 in-graph regression:
V2 needs the full decode graph to show; its 416 workgroups co-reside and
interleave with MoE/GDN/elementwise (V1's 6656 saturate) → V2 dropped, V1
default. Bench no regression: 56.73 post-fix vs 56.7–56.8 HEAD.

## 2026-08-16 — FA kernel track: B=1 decode parallelism — LANDED (57.1 → 62.7 t/s, +9.8 %)

`flash_attn_tile_q8` was the largest remaining non-MoE decode cost (3.27
ms/step, Sk-linear). At B=1 the launcher hardcoded NC2=1 (no GQA head-packing)
and `gridDim.y=1` (no KV split) → 16 blocks = 6.7 % of the wavefront slots. The
vendor kernel already supported both; the work was the dispatch ladders
(`GFX906_FA_NC2` / `GFX906_FA_KVSPLIT`) + a new `fa_split_combine_kernel`
(flash-decoding merge of per-split m/l partials, one warp/row; y≤1 no-op).

**Three vendor bugs found & fixed:** (1) null-mask deref — `(ncols2 > 1 ||
mask)` derefs `mask` unconditionally at NC2>1 (GPU fault at 0x0; 4 sites);
(2) NC2=8 × prefill fault (ncols=64 OOB at large Sq; guard `nc2>1 && seq_q>2`
→ NC2=1); (3) OOB-tail — the strided KV loop never enabled `oob_check` for the
tail tile, letting padding into the softmax (rel 0.24–0.60).

**Micro-bench** (B=1, Hq16/Hkv2/D256, Sq=2, maxerr ≤ 0.0048): @Sk=2176
NC2=1/y=1 245 µs → NC2=1/y=8 82.9 → **NC2=8/y=16 58.3 µs (4.2×)**; y=16 is the
knee (32/64 regress). **Serving A/B:** 57.08/57.16 → NC2=1/y=8 62.13/62.15 →
**NC2=8/y=16 62.81/62.92**. PPL legacy 6.6999 vs new 6.6895 (−0.15 %, Triton
6.6775). **Greedy is not a valid gate here** — legacy/new/Triton all diverge
across launches (engine non-determinism); PPL is the metric. Default flipped
(kill: `GFX906_FA_NC2=1 GFX906_FA_KVSPLIT=1`); local venv 62.677/62.668/62.671
(σ 0.005); decode 57.09 → **~62.7 (+9.8 %)**, llama.cpp gap 1.12×.

## 2026-08-19 / 08-24 — Gather-buffer lifecycle: a UAF, then the 256k-prefill OOM it caused

**2026-08-19 (UAF).** Qwen3-0.6B init faults 100 % in post-capture warmup;
`gather_paged_kv_quant_kernel` appears in the HW record with a garbage grid
(proved by cross-checks: LEGACY-independent constancy, a no-FA control naming
it, a launch-API spy seeing no such dispatch). **Cause:** `_ensure_gather_
buffers` allocated one exact-shape K+V pair per batch size; FULL capture bakes
35 pairs' VAs (B sweep 1..256) but the keep-alive list held only 4 generations
→ the descending sweep freed the first-captured (B=256) pair and warmup
replayed `graph_256` into freed segments. **Fix:** smaller-B requests slice the
current buffer `[:B]` (same base VA, one generation for all sizes), real growth
retires into a keep-alive dict keyed by `data_ptr` (a `(shape, device)` key let
same-shape generations collide), latch `_gather_captured` on the slice path
too. Verified: repro 4/4 clean (was 10/10 fault); suite 18/18; plus a no-view
fast path when `b.shape[0] == num_seqs` (the exact-size decode case was making
a fresh TensorImpl per FA layer per step). **MoE production (max_num_seqs=32 →
7+ captured sizes > old bound 4) had been exposed to silent corruption.**
Serving re-validation: dense 4-seq 25.33 t/s (band), MoE 65.98/65.81.
**This fix's "unbounded dict + sticky latch" design was itself the next bug.**

**HYPOTHESIS (08-24).** If the pre-fix exact-Sk reallocate + sticky-latch policy
is the 256k-prefill OOM cause, then (a) the pre-fix policy reproduces the OOM
with the retired dict dominating, and (b) a capacity-width + per-generation-flag
policy completes the same prefill with the dict flat and the needle intact.

**Evidence.** *Arm A (pre-fix, `GFX906_FA_GATHER_EXACT=1`, TP=2 util 0.82
maxlen 262144 chunk 1024 k=2 prefix-caching, 249,991-token prompt, needle at
125,000):* OOM 3.3/3.35 min in on two boots, failing alloc **178,257,920 B,
free: 0, `gptq_gemm`** — byte-exact to the original run-4 signature — with the
retired dict at **152 gens / 7.79 GB @Sk=60k** and 137 gens / 6.46 GB @53.6k
(growth ≈2.15 GB per 15k tokens; the dict, not the AWQ scratch, drained the
~1.94 GiB headroom). *Arm B (the fix, same harness):* 4 OOMHUNT lines total
(warmup + full-width capture), `retired=0` throughout, 250k prefill completes
(1692.4 s ≈ 148 tok/s incl. prefill) and retrieves the needle
`XQ47-KF92-PL08` from token 125k. **Decode gate:** post-fix 66.16/65.21/66.12/
66.16 (mean 65.92) vs EXACT 66.17/66.12/66.11/66.10 (66.13) — flat, as
predicted (at B=1 the EXACT policy's ~5 MiB reallocation churn is sub-ms; the
policy difference only shows at GiB-scale generations). **Fix (4 parts):**
(1) Sk is a *capacity* — grow-only `>=` reuse, grow-to-exact-need, and FULL
capture runs at max_model_len so one generation spans every later eager shape;
(2) per-generation `_gather_buf_captured` (reset at allocation) instead of a
sticky OR latch, so only graph-baked generations retire; (3) persistent-branch-
only wide reuse (non-persistent call sites keep the exact-Sk contract); (4)
`GFX906_FA_GATHER_EXACT=1` = pre-fix policy byte-for-byte. 25/25 unit suite
(capture-sweep keepalive, poisoned-tail width≫live bit-equality, B=17
fused-quant no-leak, exact-killswitch policy).

**Review follow-up (3 issues, 28/28):** (1) the §2.2b capture-order warning was
dead code (its condition required `not capturing` after the preceding
assignment set `capturing`) → moved to the retire-insertion site, pinned by
`test_gather_multi_retire_warns`; (2) grow-only `max()` per axis also applied
to *freeable* generations (32-way short decode then one 250k prefill left a
`[32, 262144]` standing buffer, ~13 GB/rank — a new OOM class) → freeable
replacements allocate at exact need (realloc frequency unchanged by
construction; FULL modes unaffected); (3) mixed-width k/v reuse required only
both-non-None, not equal width (a hand-set pair passed `Sk` = K's width and
silently dropped V to a per-call allocation) → now requires equal widths,
pinned by `test_gather_mixed_width_buffers_not_reused`. Second round: the
paged exact/capacity selections collapsed into one derived comparison (hot path
2 fit-checks instead of 4), and the duplicated kill switch is kept **by design**
(byte-for-byte pre-fix policy = the arm-A repro value) with drop notes at both
read sites and in `plan-gfx906-fa-fix.md` §6 (drop at the next
gather-lifecycle change, re-gated on a serving A/B).

**VERDICT: SHIPPED** — `_gather_retired` growth fixed and validated on the
exact situation that failed; Qwen3.8-27B TP=2 256k prefill now works at the
run-4 config; the 131k ceiling in `oom-256k-prefill.md` is lifted for this
consumer. Design + safety case: `plan-gfx906-fa-fix.md`; four adversarial
reviews in `gfx906-fa-fix-code-review-{claude,ds4,glm,gwen}.md`. Ops note: the
first arm-B attempts were blocked by boot C's wedge flap (12 resets, two
same-millisecond dual-card) — degraded-state territory, cleared by reboot; arm
A2 ran with the kill switch only because `window_watch.sh` passed `1` as the
EXACT arg (a launcher bug, caught by the log's `mode=exact` field).

---

## Refrigerated residue (cheap calls not taken, cross-linked not restated)

- The FA decode Sq=8/verify option space (pad-row skip, NC2=2, native
  ncols1=5/6, dual-tile) — all measured dead → `DEVLOG-fa-verify-sq8.md`.
- `GFX906_FA_NC1_OVERRIDE` + the host-side `fa_pick_ncols1` mirror — reusable
  scaffolding for any future tile variant (recipe in the pre-split log).
- The V2 fused-Q8 gather kernel (`GFX906_FA_GATHER_V=2`) — kept for completeness
  but never a default; it degrades only in serving (co-residency).
- `GFX906_FA_GATHER_EXACT=1` — kept as the arm-A repro value; drop notes are in
  code and in the fix plan.
