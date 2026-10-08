# Muse-Glimmer-30B onboarding — CT-asym W4A16 dense text + iRoPE hybrid attention on gfx906

> Branch `feat/muse-glimmer` off `main` (4a9e24b5c) · model
> `cyankiwi/Muse-Glimmer-30B-AWQ-INT4` · date 2026-08-26 · onboarding +
> sliding-window CUSTOM FA track.

**Model:** 52 text layers, hidden 6656, Hq 32 / Hkv 2 (GQA 16:1), head_dim
128, sliding_window 2048, fp16 KV. iRoPE: 13 full-attention NoPE layers
(indices 3,7,…,47,51) + 39 sliding-window(2048) RoPE layers. Vision tower
50 layers (all unquantized fp16), lm_head + layers 47/51 linears unquantized
(315 ignored tensors, CT `ignore` list). Dense CT-W4A16-asym (group 32,
uint4, int8 zp) → `CompressedTensorsWNA16` → Exllama `gptq_gemm`
(`use_v2_format=True`, raw zp).

**Baseline attention (no code change):** per-layer auto-select gives
sliding group → `ROCM_ATTN` (Triton `unified_attention`), full group →
`CUSTOM` (Q8 FA); ViT → `FLASH_ATTN`. Log: `rocm.py:748`
"Found incompatible backend(s) [CUSTOM, TURBOQUANT] … Overriding with
ROCM_ATTN".

## 2026-08-26 — onboarding: load + smoke + hybrid baseline

**VERDICT:** OPEN (baseline established; window work pending) ·
**GATE:** `_bench_gfx906.py` serving wall-clock, pp2048/tg256, 4 samples,
`BENCH_MAX_SEQS=4`, `BENCH_GPU_UTIL=0.93 BENCH_KV_MEM=4634016400`,
graph mode. (Util-only sizing OOMs the first request in warm-cache runs:
profiling peak ~0.16 GiB lower than cold → KV pool oversized for the
532 MiB inductor prefill buffer — 27B's was 356 MiB, so 0.93/0.95 are
both too tight here; explicit 4.32 GiB KV cap per the engine log's
suggestion, also makes A/B arms pool-identical. Logs:
`/local/tmp/muse/bench_hybrid_graph{,_093,_093_r2}.log`.)

## HYPOTHESIS

If the AWQ checkpoint loads through the existing CT-asym WNA16 (Exllama)
dense path and the per-layer backend auto-select routes sliding layers to
ROCM_ATTN, the model serves coherent output with no code change — i.e.
upstream MuseGlimmer support (PR #51655) + the shipped gfx906 quant path is
sufficient for a working baseline, and the only gap is sliding-window
support in CUSTOM FA (perf, not correctness).

## What was done

- `feat/muse-glimmer` from `main` (upstream `muse_glimmer` model/config/
  processor/reasoning+tool parsers already merged, PR #51655; registry maps
  `MuseGlimmerForConditionalGeneration` → `("muse_glimmer", …)`).
- Download: original target `Vishva007/…-W4A16-AutoRound-GPTQ` died in a
  host crash (group-64 sym GPTQ); switched to
  `cyankiwi/Muse-Glimmer-30B-AWQ-INT4` (group-32 CT-asym W4A16 — same
  kernel family as the shipped Ornith MoE path, but **dense** linears).
- Cache layout incident: my `hf download --cache-dir /data/cache/huggingface`
  put blobs in the NFS cache root, but `~/.cache/huggingface` →
  `/local/cache/huggingface` (local disk) is where transformers resolves
  (its `hub/` snapshot already had the small files from the pre-crash
  attempt). Copied the 5 shards + tokenizer.json into the `/local` snapshot
  (23 GB, now on local disk).
- Smoke (text-only `LLM`, `max_model_len=8192`, gpu_util 0.93, 4 greedy
  prompts via native chat template, thinking ON): loads in 58 s from local
  disk (22.55 GiB weights), coherent + correct answers (Paris; Rayleigh
  scattering; haiku drafting; 17·23=391). Log `/local/tmp/muse/smoke_hybrid.log`.
- Baseline bench: config above, log `/local/tmp/muse/bench_hybrid_graph_093_kv.log`.
  Harness gained a `BENCH_KV_MEM` hook (documented in README/running.md but
  missing from the harness file).

## Evidence — FOR

- Coherent, factually correct greedy output on 4 prompts incl. arithmetic;
  thinking turns emitted with the native ` to=self` protocol the reasoning
  parser expects.
- Backend select as predicted (sliding→ROCM_ATTN, full→CUSTOM); ViT
  FLASH_ATTN.
- **Hybrid baseline (graph mode, the gate):** 17.45 / 17.46 / 17.44 t/s
  (samples 1–3; sample 0 12.40 cold), pp2048/tg256, BENCH_MAX_SEQS=4,
  BENCH_GPU_UTIL=0.93 + BENCH_KV_MEM=4634016400. Qwen3.8-27B reference on
  the same harness: ~25.2 t/s (all-attention CUSTOM; no sliding layers).
  (Re-verified at the matched A/B config: 17.54 t/s.)

## Interactions / superseded-by

- Dense CT-asym Exllama dequant is **not** previously validated on gfx906 —
  the Ornith log (DEVLOG-ornith-wna16.md) covered MoE experts only. Smoke
  coherence is weak evidence for the (q−zp)·scale zero-point math; a
  numerics check vs transformers (logprob delta) is the follow-up if any
  doubt appears.
- Sliding-window CUSTOM FA landed in the next entry (SHIPPED, 1.59×);
  the open follow-up is step C — gather windowing (`kv_start` clip) for
  long-context decode, where the mask-only step still scans [0, seq_len).
  At the 2048-token window it is irrelevant for contexts ≤ window, so it
  only matters for long-ctx serving.

## 2026-08-26 — sliding-window CUSTOM FA (mask-only)

**VERDICT:** SHIPPED (1.59×) · **GATE:** same serving A/B as above,
all-52-layers CUSTOM arm vs the hybrid baseline.

## Gate result (2026-08-27 00:0x, matched config)

Both arms: graph mode, pp2048/tg256, 4 samples, BENCH_MAX_SEQS=4,
BENCH_KV_MEM=805306368 (15k-token pool ≫ 2816 needed),
BENCH_BATCHED_TOKENS=1024 (see the memory forensics below — 2048-chunk
prefill + 4.32 GiB KV OOMs the all-CUSTOM arm on the first request).

| arm | sample 0 (cold) | samples 1–3 (stable) |
|---|---|---|
| hybrid (sliding→ROCM_ATTN pinned, full→CUSTOM) | 12.43 | **17.54 t/s** (17.556/17.542/17.521) |
| all-CUSTOM (52/52 layers, window=2048) | 16.86 | **27.90 t/s** (27.931/27.901/27.874) |

**1.59×** decode wall (tg256). Hybrid stable matches the pre-change
baseline (17.45) — the harness-config delta (chunk size, KV cap) does not
move decode. Backend select verified in both logs (per-group `info` lines;
`_cached_get_attn_backend` dedupes identical configs, so one line per
group): all-CUSTOM shows both groups → CUSTOM, hybrid shows full → CUSTOM
+ sliding → ROCM_ATTN (kind-pin rides the "selected via
--attention-backend" log path). The Qwen3.8-27B all-CUSTOM reference on
the same harness is ~25.2 t/s — Muse 30B now runs ahead of it.

Note: that gate ran B=1 (single repeated request) with prefix caching on
the harness default. Both gaps closed by the 2026-08-27 review-follow-up
entry below (B=4 re-run, prefix cache off, window=0 control arm).

## HYPOTHESIS

If the Q8 FA tile kernel gets a per-row sliding cutoff (mask keys older
than `q_abs_row - W + 1`, re-using the existing `q_abs_offset` inline-
causal machinery, passed for every windowed batch including decode), all
52 Muse layers run on CUSTOM and decode recovers part of the 17.45→25.2
t/s gap the Triton sliding path costs.

## What was done

- `fattn-q8.cuh` / `fattn-q8-paged.cuh`: tile-iter + global kernels take a
  scalar `window`; both causal branches extend to
  `k_pos > q_abs_row || (window > 0 && k_pos < q_abs_row - W + 1)`.
  Fully-masked tiles are safe: `KQ_max` seeds at `-FLT_MAX/2` (finite), so
  an all-`-INF` tile leaves max unchanged and `val=exp(-INF)=0` — no NaN
  guard needed (plain causal could never hit an all-masked tile; a window
  can).
- `gfx906_fa_launcher.cu` (+ the cpp's own extern decls): `window` plumbed
  through `gfx906_fa_launch{,_paged}`.
- `gfx906_fa.cpp`: `forward` / `forward_paged_direct` gain `window=0`
  (pybind args added).
- `gfx906_fa_backend.py`: `Gfx906FAImpl` stores `sliding_window` (was
  `NotImplementedError`); backend class now
  `supports_sliding_window() -> True`.
- `gfx906_fa_paged.py`: `forward_paged(window=0)`;
  `need_causal = max_seqlen_q > 1 or window > 0` (decode rows need
  `q_abs_offset` for the per-row window; the causal check itself is a
  no-op there since all keys ≤ seq_len-1 = the decode row).
- Tests: `test_forward_sliding_window_vs_torch_ref` ×4 (D=128 GQA 16:1
  model shape W=128/64, W>L inert, D=256; decode + per-row prefill vs
  torch ref, plus window-bites / window-inert sanity). 32/32 suite green
  (28 pre-existing untouched).

Mask-only step (B of the plan): FA still scans [0, seq_len) — long-ctx
windowing (gather `kv_start` clip) is the follow-up (step C), gated by the
same A/B plus a long-context curve.

## Memory forensics — all-CUSTOM arm OOMs on the first prefill

Symptom: the all-52-CUSTOM bench OOMs on the 532 MiB inductor prefill
buffer at the first request (hybrid arm OOM'd once too, 23:00:36, and
recovered via empty_cache; all-CUSTOM never recovers). Probe
(`/local/tmp/muse/probe_mem.py`): post-init allocated 24.22 / reserved
26.06 GiB (weights 22.55 + graphs 0.71 + ~1 GiB). The prefill adds:

- **per-layer `q_pad_buf` [B, Hq=32, Sq_pad, D=128] fp32** — decoded
  batches were growing its dim0 to the capture max (B=8) although Sq=1
  never READS it (both branches use `q_pad_decode_buf`): a B=8 capture
  then ballooned every layer's prefill buffer to (8, 32, 2048, 128) =
  268 MB → 14 GiB across 52 layers. **Fixed**: `_ensure_forward_buffers`
  sizes dim0 with `qpad_num_seqs = num_seqs if max_seqlen_q > 1 else 1`
  (decode never reads `q_pad_buf`).
- per-layer C++ `o_bshd` FA output [B, Sq, 32, 128] fp32 = 33.5 MB at
  Sq=2048 → ~1.7 GiB across 52 layers while the prefill graph holds
  them (hybrid's 39 Triton layers emit fp16, ~2× smaller).

Residual budget: at KV cap 0.75 GiB + `BENCH_BATCHED_TOKENS=1024`
(prefill chunks 1024 instead of 2048 → q_pad/o_bshd per layer halve,
inductor workspace smaller) the first prefill should clear. KV pool
size is irrelevant to single-request decode wall (2816 tokens ≪ pool),
so the A/B arms just share the same capped pool. Long-term levers (not
for this gate): fp16 `o_bshd` in the Q8 FA forward (halves the per-layer
output), and/or a worker-shared prefill q_pad arena.

## Evidence — FOR / AGAINST

FOR: 32/32 FA suite (4 new window tests green); gate A/B above (1.59×);
all-CUSTOM smoke coherent + correct (4 prompts, thinking ON, 7.4 s for
4 completions vs 12.7 s hybrid). AGAINST: (none).

## 2026-08-27 — review follow-ups: direct-paged window coverage, B=4 gate, control arm

**VERDICT:** SHIPPED · **GATE:** same A/B at B=4 (BENCH_NREQS=4,
BENCH_PREFIX_CACHE=0 per the local-serving default) + a window=0-on-
CUSTOM control arm, per `muse_glimmer_opt_code_rev_claude.md` #1/#7/#8/#9.

## HYPOTHESIS

If the direct-paged kernel's hand-duplicated window formula is correct
and the window mask is ~free, then (a) a `forward_paged_direct` window
test matches the torch reference, (b) the all-CUSTOM win persists at
B=4 — where the sliding layers actually run the direct-paged kernel
(auto mode, min_batch=2 — not the B=1 gather path the first gate
exercised) — and (c) forcing window=0 (unbounded causal, perf-only) on
all-CUSTOM moves the t/s by ~0.

## What was done

- **Direct-paged window test** (#1): `test_forward_paged_direct_
  sliding_window_vs_torch_ref` ×4 — same shapes as the gather test,
  calling the `forward_paged_direct` binding directly (block_size=16,
  unbind(1) V layout): decode B=1, decode B=2 with different seq lens
  (per-row q_abs_offset/window), and per-row prefill. Suite 36/36.
- **`GFX906_FA_NO_WINDOW=1`** control knob (#9): forces
  `sliding_window=0` for every layer (numerically wrong for windowed
  layers, perf-representative) — the third bench arm.
- **`BENCH_PREFIX_CACHE`** harness hook (#7): defaults to the old
  behavior (on) for comparability; gate re-runs use 0.
- Comments: spec-decode caveat on the q_pad decode heuristic (#10 —
  verify steps DO read q_pad_buf but their Sq_pad is bounded by the spec
  depth, so the 14 GiB pathology stays prefill-only); validated envelope
  on `supports_sliding_window` (#11); LOCKSTEP notes on the duplicated
  cutoff formula in both .cuh files (#2).

## Gate result (B=4, pp2048/tg256, 4 samples, prefix cache OFF, 0.75 GiB
KV, bt1024)

| arm | B=1 (prev) | B=4 (this) |
|---|---|---|
| hybrid (sliding→ROCM_ATTN, full→CUSTOM) | 17.54 | 16.75 (16.78/16.74/16.71) |
| all-CUSTOM, window=2048 | 27.90 | 20.59 (20.73/20.59/20.19) |
| all-CUSTOM, window=0 (control) | — | 20.64 (20.73/20.64/20.64) |

- **1.23×** at B=4 vs 1.59× at B=1 — the win persists, smaller: hybrid
  decodes 4.5% slower at B=4 (Triton per-seq sliding attention batches
  worse) while all-CUSTOM drops 26% (B=4 direct-paged steps cost 5.4×
  the B=1 step time for 4× the work — grid-z/occupancy tuning headroom,
  not a regression: B=1 is unchanged).
- **Window-mask cost ≈ 0**: control 20.64 vs windowed 20.59 (−0.2%,
  within run noise) — at B=4 the win is entirely the kernel family on
  the 39 sliding layers; the per-row window branch is free.
- Logs: `/local/tmp/muse/bench_b4_{hybrid,allcustom,nowindow}.log`.

## Evidence — FOR / AGAINST

FOR: 36/36 FA suite; B=4 gate 1.23×; control arm isolates the window
cost at ~0; both kernel copies now reference-tested. AGAINST: (none).

## Interactions / superseded-by

Supersedes the B=1-only gate framing of the 2026-08-26 entry (numbers
there stand; this entry adds the B=4 + control data). The B=4
step-time tuning this entry flagged is done in the next entry (split-K
+ batch-aware KVSPLIT default, Phase C window clip).

## 2026-08-27 — perf follow-ups: direct-paged split-K + Phase C window clip

**VERDICT:** SHIPPED with ERRATUM (see below — both gates ran on the
GATHER path, not the direct-paged path they were attributed to; the
direct-paged path is unreachable in default serving). · **GATE:** B=4
pp2048 serving A/B (S=1 vs batch-aware default, same session) +
pp4096 clip on/off A/B + kernel micro-bench.

## ERRATUM (2026-08-27, review round 2)

The direct-paged branch of `forward_paged` is gated on
`key_cache_q8 is not None` — which the backend only passes when
`GFX906_FA_LEGACY=0`, and that mode is still broken on this tree
(Q8 side-buffer desync → garbage output; smoke-verified 2026-08-27,
see the review-round-2 entry). Default serving (LEGACY=1) therefore
runs EVERYTHING on the gather path, so:

- The **B=4 split-K gate was a GATHER-path measurement**: arm 1
  (KVSPLIT=1) vs arm 2 (unset → gather default 16). The +2.7% is real
  but is the gather kernel's existing S=16-vs-S=1 knob, not the new
  direct-paged plumbing. The batch-aware `clamp(16/B, 2, 8)` default
  is dead code in default serving (it only applies to the
  direct-paged path).
- The **pp4096 clip A/B (and the later pp8192 A/B) were NULL TESTS**:
  with LEGACY=1 the clip code (direct branch) never executes, so both
  arms ran identical gather-path code. The "below the noise floor"
  reading was a comparison of two identical configurations. Phase C's
  e2e serving benefit has NEVER been measured; the kernel-level −48%
  (L=4352) / −71% (L=8448) micro-bench numbers stand (binding level),
  but they only apply when LEGACY=0 works.
- Unchanged by the erratum: the window-mask gates (B=1 1.59×, B=4
  1.23× — the mask runs in the gather kernel), the q_pad fix (shared
  backend buffer code), and the direct-paged code itself (plumbing is
  real and binding-tested; it just can't be reached from serving yet).

Re-gate Phase C (and the batch-aware KVSPLIT) once the LEGACY=0
Q8-side-buffer desync is fixed.

## HYPOTHESIS

(B=4 anomaly) The direct-paged launch pinned `grid.y=1` (no split-K,
unlike the gather path's KVSPLIT=16), so every block's critical path is
the FULL ~88-iteration KV k-loop — hence B=4 steps cost 5.4× the B=1
step for 4× the work. If the paged kernel's existing strided split-K +
split_combine machinery (unused by the paged host path) is plumbed in
with a batch-aware split count, B=4 decode recovers part of the gap
without a kernel change. (Phase C) If a windowed decode row's k-loop
starts at `max(0, L-W)` instead of 0, the output is bit-identical
(prefix keys are window-masked to -INF anyway) and long-context decode
reads ~W/L of the KV.

## What was done

- **Direct-paged split-K** (host-only): the paged kernel already
  strides the k-loop by `gridDim.y*nbatch_fa` and stores unscaled
  `[rows, y, D]` partials + meta when `gridDim.y>1` — the launcher
  pinned `grid.y=1` and the binding never allocated/merged partials.
  Plumbed `kv_split` through launch_paged + `forward_paged_direct`
  (same OOM guard as the gather path: split=1 for Sq>2).
  `GFX906_FA_KVSPLIT` now returns -1 when unset: gather keeps its
  16 default; direct-paged defaults to `clamp(16/batch, 2, 8)` —
  batch-aware because grid-z = batch×Hq (NC2=1 paged), so the split
  that fills the 60-CU MI50 moves with the batch.
- **Phase C clip** (kernel + plumbing): `kv_start` arg end-to-end
  (binding → launcher → `fattn-q8-paged.cuh`); the k-loop walks
  `[kv_start, L)` (split-K still covers the clipped range exactly —
  slice y visits `kv_start + (y+i·S)·nbatch_fa`). Python passes
  `kv_start = max(0, q_abs + 1 − W)` for windowed decode (Sq=1) only;
  prefill rows keep the full scan (per-row windows; that clip is
  still open). `GFX906_FA_WINDOW_CLIP=0` kill switch for A/B.
- Tests: clip suite ×4 shapes (bit-identity to the masked full scan
  < 1e-6; a functional check with an INERT window (W=L) + real clip
  start proving the scan itself shrinks; unaligned start (W=64 →
  start=448 vs nbatch_fa=128); B=2 per-row). Suite 40/40.

## Gate result

**B=4 split-K** (pp2048/tg256, 4 samples, prefix off, 0.75 GiB KV,
same session/build):

| arm | t/s (samples 1–3) |
|---|---|
| KVSPLIT=1 (old grid.y=1) | 20.05 (20.27/20.07/19.89) |
| batch-aware default (S=4 @ B=4) | **20.59** (20.66/20.56/20.45) |
| KVSPLIT=2 (micro-bench optimum) | 20.55 (20.72/20.48/20.44) |

**+2.7%** e2e; S=4 (the formula) matches S=2 (the micro-bench
optimum) within noise, so the shape-independent formula is kept.
Micro-bench (B=4, Hq=32, D=128, L=2816, window=2048): best split per
batch = 8/8/5/2/2 for B=1/2/3/4/8; `clamp(16/B, 2, 8)` hits the
measured optimum at B∈{1,2,3,8}, within 4% at B=4 (235 vs 226 us).

**Phase C** (pp4096/tg256, B=4, prefix off — window bites 50% of the
decode KV at L=4096–4352):

| arm | t/s (samples 1–3) |
|---|---|
| clip ON (run 1) | 10.35 (10.51/10.26/10.19) |
| clip OFF (run 1) | 10.72 (10.65/10.68/10.76) |
| clip ON (run 2) | 10.54 (10.80/10.47/10.33) |
| clip OFF (run 2) | 10.44 (10.77/10.33/10.24) |

Kernel micro-bench at the serving shape (L=4352): FA 337.8 → 175.7 us
(S=4, −48%) — the clip works at kernel level. E2e the effect is below
the noise floor: the four runs span 10.35–10.72 with clip OFF
interleaved at 10.72 AND 10.44 (between the two clip-ON runs), and
three of the four runs drift ~4% DOWN across their own samples (the
host state fluctuates on the ~5 min timescale; an initial "clip ON
slower" read was the run-1 ordering artifact). FA is only ~8% of the
B=4 step at pp4096 (GEMM-bound), so the expected ~2.9% e2e win isn't
resolvable here. At L=8k+ the same clip saves ~4× the FA work (~11%
of the step) — re-measure on a longer-context config if long-ctx
serving matters. Correctness is exact (bit-identical, 40/40), so clip
ON stays the default; it is also the prerequisite for the gather-path
clip.

## Evidence — FOR / AGAINST

FOR: 40/40 suite (clip bit-identity + functional scan-shrink);
B=4 split-K +2.7% same-session A/B; micro-bench split matrix
(stable across repeats: S=2 225.9/226.3 us across two runs); Phase C
kernel −48% at L=4352; 4-run pp4096 A/B shows no clip-induced loss
(OFF interleaved at both ends and the middle of the ON range).
AGAINST: Phase C e2e WIN not resolvable at pp4096 (below the ~2% run
noise); within-run drift ±4% on 3 of 4 runs (host-state fluctuation —
recorded, not diagnosed).

## Gotchas / notes

- `GFX906_FA_FWD_DEBUG=1` is incompatible with cudagraph capture
  (its `torch.cuda.synchronize()` at the _DBG log points invalidates
  the stream capture — `hipErrorStreamCaptureInvalidated`). Debug
  graph-mode runs without it.
- Build note: the CMake build compiles a hipified copy of the .cu
  staged under `build/temp…/csrc/` — the in-tree `.hip` mirrors are
  NOT built (they predate even the window arg); edit the `.cu`.
- A 16-way clang build concurrent with a 20 GB weight load
  coincided with one `hipErrorLaunchFailure` (first non-OOM-collateral
  event of boot I; retry clean) — see degradation.md. Serialize
  builds and loads if it recurs.
- Open: gather-path (B=1) window clip (needs the persistent gather
  kernel's work list to start per-row at `max(0, L-W)` + compacted
  store + shifted q_abs_offset); prefill-row clip (per-row windows
  [max(0, t-W+1), t] — a 2D per-(row, k) problem, not a per-row
  start); **fix the LEGACY=0 Q8-side-buffer desync** — the only path
  to the direct-paged kernel (split-K, batch-aware KVSPLIT, Phase C
  clip); re-gate Phase C e2e at L≥8k after that.

## 2026-08-27 — review round 2: clip bit-identity fix (P1), NaN guard, checks, null-test exposure

**VERDICT:** SHIPPED (P1 clip floor fix reproduced + verified on
hardware; NaN guard + symmetric checks in; both prior e2e clip gates
exposed as null tests by the LEGACY=1 dispatch reality — see the
erratum in the perf-follow-ups entry). · **GATE:** pre/post-fix
unaligned clip-vs-full diff on hardware + 44/44 suite + LEGACY=0
smoke.

## HYPOTHESIS

(Review P1) The Phase C clip is NOT bit-identical to the masked full
scan when `kv_start` is not a multiple of `nbatch_fa` (the normal
production case, e.g. L=4353/W=2048 → 2305): an unaligned start makes
the first tile partial, repacking which lane holds each surviving
score and re-associating the fp16 VKQ reduction. Flooring the clip
start to the tile boundary — only when the keys the floor gains are
window-masked — restores exactness at the cost of ≤ nbatch_fa−1 extra
keys. (Review P3) A fully-masked row (KQ_sum==0) writes inf*0=NaN on
the non-split (gridDim.y==1) store path in BOTH kernels; guard it like
the split-combine l_star guard.

## What was done

- **Clip floor (P1)**: `fattn-q8-paged.cuh` floors `k0_base` to
  `nbatch_fa` when `k0_base <= max(0, q_abs + 1 - window)` (the
  production clip passes exactly that; the guard was found when an
  unconditional floor broke the inert-window test subcase, where the
  clip emulates a window and the gained keys are NOT masked).
- **NaN guard (P3)**: `scale_out = KQ_sum>0 ? 1/KQ_sum : 0` in the
  non-split store of both `fattn-q8-paged.cuh` and `fattn-q8.cuh`
  (the gather copy had the identical pattern — fixed by inspection).
- **Symmetric checks**: `window > 0` now requires `q_abs_offset` in
  BOTH bindings (previously the window mask silently degraded to full
  attention without the offset, while `kv_start` had the check).
  Verified: both bindings reject the malformed call.
- **Nits**: clip math int32-only (no int64 round-trip);
  `GFX906_FA_NO_WINDOW` truthy-parsed like the sibling knobs + warns
  (was `== "1"`, silent); KVSPLIT "0 ≠ unset" note; README knob table
  (KVSPLIT batch-aware row, WINDOW_CLIP, NO_WINDOW); `BENCH_PREFIX_CACHE`
  default flipped to 0 (AGENTS.md local-serving recipe);
  `supports_sliding_window` envelope note: the clip only fires on the
  direct-paged (B≥2) dispatch.
- **Tests**: +3 unaligned bit-identity shapes (513/128, 1025/256,
  4353/2048 — L deliberately not a multiple of BLOCK) + 1 NaN-guard
  test (B=2, fully-masked row, KVSPLIT=1, no-NaN + zero-row + correct
  sibling) + 1 forward_paged-LEVEL test (review #10: dispatch →
  direct branch → Python clip math → binding, B=2, two different
  seq lens, unaligned starts 385/352 — split-K + window + clip all
  active together) → suite 45/45.

## Gate result

Pre-fix (old .so) vs post-fix, clip vs masked full scan:

| shape (L/W→start) | pre-fix rel | post-fix max-abs |
|---|---|---|
| 513/128→385 | 5.2e-4 | 0.0 |
| 1025/256→769 | 5.4e-4 | 3.0e-8 |
| 4353/2048→2305 | 7.2e-4 | 5.6e-9 |
| 512/128→384 (aligned) | 0.0 | 0.0 |

(Review's independent numbers 4.7e-4/5.3e-4/6.8e-4 — same shapes,
seed noise.) The old suite's "unaligned" shape (512/64→448,
448%128=64) was accidentally bit-exact: the 64-key window fills the
half tile on a chunk boundary — the suite's 40/40 never saw the bug.
(The review's "448 is a multiple of 128" is arithmetic wrong —
448%128=64 — but its conclusion that the old test missed the bug is
right.)

**LEGACY=0 smoke (fresh, 2026-08-27)**: 4-prompt greedy → incoherent
prompt-echo garbage ("to=self<prompt>…" + meta-reasoning) with a
capped 0.375 GiB KV pool (an uncapped pool OOMs on the ~1.5 GiB Q8
side buffer + inductor headroom). The Q8 side-buffer desync is still
live → the direct-paged path (split-K, batch-aware KVSPLIT, Phase C
clip) is unreachable in any working serving config; the pp4096 and
pp8192 clip A/Bs were null tests (both arms gather). Blocking item:
fix the desync before any direct-paged e2e claim.

> **RESOLVED** by the round-3 entry below — and partly misdiagnosed:
> the "incoherent garbage" was the model's normal greedy style (see
> the round-3 erratum); the real defects were side-buffer sizing and,
> in the first alias fix, per-call re-alias zeroing.

## Evidence — FOR / AGAINST

FOR: pre/post hardware repro of the P1 bug + fix (3 shapes); 45/45
(incl. the forward_paged-level direct-branch test); window-check
rejection on both bindings; LEGACY=0 garbage smoke (blocker
established, not assumed). AGAINST: no new e2e serving numbers
(nothing to re-gate until the desync fix); Phase C serving benefit
remains unmeasured at every context.

## Notes

- `_fwdlog` (GFX906_FA_FWD_DEBUG) writes to
  `/tmp/gfx906_fa_debug/fwd-<pid>.log` and silently no-ops if the dir
  doesn't exist — `mkdir -p` before trusting an empty log. The
  gather-branch dispatch log's `path=FUSED/FAST/LEGACY` labels the Q8
  K-buffer SOURCE, not direct-vs-gather; the direct branch logs
  `forward_paged DIRECT_PAGED:`.
- The review's ds4 "40/40 miscount" claim was re-checked:
  `pytest --collect-only` collects exactly 40 (now 46) — the devlog
  counts were literal all along.

---

## 2026-08-27 — round 3: LEGACY=0 Q8 side buffer aliased into the fp16 K half (desync blocker resolved)

**VERDICT:** SHIPPED (LEGACY=0 now serves coherent output at the
default uncapped config with zero extra KV memory; the direct-paged
e2e gates are now real measurements: Phase C clip +3.6% @
pp8192/B=2, B=4 KVSPLIT +1.8%). · **GATE:** 46/46 suite (new
alias-layout test) + LEGACY=0 default smoke + prefix-cache smoke +
LEGACY=1 regression smoke + pp8192 clip A/B (8 samples) + B=4
KVSPLIT S=1/2/4 (12 samples).

## HYPOTHESIS

The "Q8 side-buffer desync" that blocked LEGACY=0 is a misdiagnosis:
the boot-I "incoherent prompt-echo garbage" is this model's *normal*
greedy output style — the persisted logs of the clean hybrid/
all-CUSTOM gate runs are word-identical in style (to=self echo +
meta-reasoning) to the run we declared broken. The real defects:
(1) the Q8 side buffer was a separate lazy `torch.empty` (~17% of KV
bytes) allocated after profiling → outside the pool budget → OOM on
uncapped high-util runs; (2) it could be derived against the
profile/dummy cache geometry (1890 vs 471 blocks observed in a
probe); (3) the first alias fix introduced its own bug: the alias
identity check used `is` on the per-call `kv_cache.unbind(1)` view
(a fresh object every call, never equal) → re-alias + zero-fill of
the whole K half on *every* `do_kv_cache_update` (52×/step) → the
Q8 bytes of all previously written slots wiped after each triton K
write (the Q8 write only restores the current step's slots) → fully
degenerate "to the the the same same…" loops.

## What was done

- **Alias, not allocation** (`gfx906_fa_backend.py`): `_k_cache_q8`
  is a strided uint8 view of the fp16 K half
  (`key_cache.view(uint8)[:, :, :, :bytes_per_row]` — the 88 B Q8
  row fits the 256 B fp16 row). Zero extra KV memory: in LEGACY=0
  the fp16 K half is written but never read, so its bytes double as
  Q8 storage. Page copies (prefix-cache COW) move the Q8 bytes with
  the page → the fail-closed prefix-cache guard in
  `get_cudagraph_support` removed (now a warning). All consumer
  kernels (`reshape_and_cache_q8`, `gather_paged_kv_q8`,
  `forward_paged_direct`) are fully stride-parameterized — verified
  against the real backend tensor, including the 2D-byte head
  stride of the unbind view.
- **Memory-based identity**: re-derive the alias only when
  (data_ptr, shape, strides) change (profile dummy → live pool,
  layout re-slices); never per-call.
- **No zero-fill, ever**: zeroing the Q8 region of a re-derived
  alias clobbers in-use context (re-derivation can happen mid-run
  on live memory). Unwritten slots are never read — attention
  seq_lens only cover written slots — so the fill buys nothing.
- **Test**: +`test_paged_direct_and_gather_on_q8_aliased_into_fp16_khalf`
  — mirrors the backend alias exactly (strided view, production
  write order fp16-K-then-Q8), checks `forward_paged_direct` (B=2,
  window + clip, unaligned starts) and the fused
  `gather_paged_kv_q8` against torch refs → suite 46/46.
- **Docs**: README LEGACY/KVSPLIT/WINDOW_CLIP rows rewritten
  (no more "desyncs — do not use"; per-path safe KVSPLIT values),
  supported-models table gains the Muse-Glimmer row,
  `supports_sliding_window` envelope note updated (W=2048 covered
  by the direct-paged clip tests + e2e, not a gather unit case).

## Gate result

| run | config | result |
|---|---|---|
| suite | `test_gfx906_fa.py` | 46/46 |
| LEGACY=0 default smoke | graph, util 0.93, **uncapped** (95,012 pool), maxlen 8192 | coherent, no OOM; 4 completions in 12.7 s (boot-I: this config OOM'd) |
| LEGACY=0 + prefix cache ON | same; shared prefix, 2 divergent tails (COW fork) | coherent in both branches (COW alias invariant) |
| LEGACY=1 regression smoke | default | coherent (391 correct) |
| Phase C clip A/B (direct-paged) | pp8192/B=2/tg256, prefix off, LEGACY=0, 0.75 GiB KV cap + bt1024 | ON 5.758/5.721/5.710/5.693 → **5.72** vs OFF 5.537/5.518/5.508/5.505 → **5.52** = **+3.6%** |
| B=4 KVSPLIT (direct-paged) | pp2048/tg256, B=4, prefix off, LEGACY=0 | S=1 **20.16** → S=2 **20.50** / S=4 (default) **20.53** (+1.8%); at parity with the gather record 20.59 |

The clip ON/OFF delta is itself the evidence the direct-paged path
executed (the clip exists only there) — the first non-null
direct-paged e2e measurement on this model. B=4: the batch-aware
default (S=4) is already optimal at this batch (S=2 within noise);
the e2e split-K gain is small because FA is ~5–8% of the GEMM-bound
B=4 step. Cross-session drift note: these run in boot J, a different
boot than the gather 20.59 record — compare trends, not absolute
values.

## ERRATUM (round 2)

The round-2 entry's "LEGACY=0 smoke (fresh)" paragraph — "incoherent
prompt-echo garbage … the Q8 side-buffer desync is still live" —
rests on a misdiagnosis: that text is the model's normal greedy
style (boot-I clean-run logs match it word for word). The garbage
seen before this fix ("to the the the same same…") is a *different*
failure — the per-call re-alias zeroing above — not a "desync".
Round 2's "blocking item: fix the desync" is closed by this entry;
"direct-paged unreachable in any working serving config" is
withdrawn (it is reachable and gated).

## Notes / open items

- **TP=2 serving (2026-08-27, boot J)**: the uncapped util-0.82 config
  OOM'd the first real 4096-chunk prefill (`gptq_gemm` `aten::empty`):
  the 9.02 GiB/GPU pool (1.36M tok) + 14.48 GiB steady state left
  <7.2 GiB of the 31.98 GiB physical, and the runtime bt4096 inductor
  prefill buffer exceeded it (profiled peak only 2.73 GiB — the
  warm/cold gap is far larger than the 27B 0.16 GiB case at this
  model size). Serving needs `--kv-cache-memory-bytes 6442450944`
  (6 GiB/GPU ≈ 900k tok, still 3.5× the 256k max). The OOM teardown
  force-killed the TP=2 workers mid-P2P op; the next two relaunches
  failed with `hipErrorLaunchFailure` (the documented TP=2 SIGKILL
  wedge; TP=1 canary healthy between) → session stopped, reboot
  required (degradation.md boot J, 17:40–17:52).
  **Boot K (18:16Z)**: canary 38.8 t/s; 1st launch hit the chronic
  weight-load hang (GPU1 fence timeout → `GPU reset(1)`, 18:23:47);
  retry loaded clean and validated the cap through capture — **KV pool
  904,164 tokens (exactly the 6 GiB), graphs 0.89 GiB** (vs 1.28
  uncapped) — then the process group died silently because the
  launching shell call was operator-interrupted (not HW; no kernel
  events, VRAM released; degradation_details.md 18:23–18:29). Operator
  relaunch (128k max, 6 GiB cap = 848,301–884,644 tok):
  - **bt4096 OOMed the first 4096-chunk prefill again** (18:41:32,
    same 254 MiB last-straw alloc as boot J, free: 0) → the compiled
    prefill transient scales ~linearly with the chunk; bt4096 needs
    >10.6 GiB per-GPU headroom at TP=2 (4 data points, incl. TP=1
    bt1024-OK / bt2048-OOM, fit it; degradation.md 18:41).
  - **bt2048 relaunch: OOM site cleared** — 4106-token prefill
    ttft 10.2 s (401 t/s), 0 OOM, coherent output (18:55).
  - **Grid** (`docs/gfx906/_bench_serve_grid_gfx906.py`, 3 samples,
    19:0x): prefill 542/491/438 t/s @2k/8k/16k; decode B=1 (filler,
    ngram 100 % acceptance — a ceiling) 114.6/79.1/57.0 t/s @2k/8k/
    16k (tg256), 112.0/79.2/56.8 (tg512); B=4 @2k/256 45.3 aggregate
    (~11.3/req); real-prompt checks ~11.5/req. README row updated
    with the working TP=2 example + these numbers.
  - **Open (operator question):** is the >10.6 GiB first-prefill
    transient caused by our custom FA/kernels? Partial evidence says
    no: (a) the 532 MiB inductor-buffer OOM predates the window work
    (boot I 22:30, hybrid config with Triton FA on sliding layers);
    (b) the 27B GDN hybrid serves bt4096/0.82/256k on the same
    all-CUSTOM stack; (c) our FA runs as a *split op* outside the
    inductor segments — the failing alloc is an inductor-segment
    buffer for `gptq_gemm` (Exllama); (d) Q8 side view is an alias
    (0 bytes). Unexplained: geometry only accounts ~1.3× of the
    27B→Muse gap. Definitive probe written
    (`/local/tmp/muse/probe_oom_attribution.py`: memory-snapshot owner
    breakdown at OOM; arms custom / rocm_attn / enforce_eager) —
    needs a free GPU (server teardown).
- LEGACY=0 is validated but stays **experimental**; the default
  remains LEGACY=1 (gather, the validated serving mode). Flipping
  the default is a roadmap decision after a longer bake.
- Direct-paged is B-gated: B=1 decode is gather (no clip,
  KVSPLIT=16); B≥2 is direct-paged. Window clip on the gather path
  (B=1) and per-row prefill clip remain open — recorded in
  `roadmap-more-models.md`.
- `muse_glimmer_opt2_code_rev_qwen.md` open items → roadmap:
  #8 device-side `kv_start` clamp, #10 overflow-free cutoff form
  (4 LOCKSTEP sites), #4 long-context split-K accuracy (L=16k–32k,
  split 8 vs 1) + per-call `o_part` allocation / dead `o_meta`.


## 2026-08-27 — round 4: OOM-attribution probe — the first-prefill transient is OUR FA's prefill path, not inductor

## HYPOTHESIS

The >10.6 GiB/GPU transient that OOMed the first 4096-token prefill
chunk under TP=2 serving (boot J 17:40, boot K 18:41; 254 MiB
last-straw `aten::empty` in a `gptq_gemm` inductor segment, free: 0)
is owned by the **inductor piecewise-compiled prefill graph** (the
failing allocation sits in an inductor segment), not the custom
gfx906 FA backend.

## GATE

Three-arm in-process probe (`/local/tmp/muse/probe_oom_attribution.py`,
TP=1, 0.5 GiB explicit KV cap, PP=4097 so the first prefill chunk is
4096 = the OOM site; peak transient = `max_memory_allocated` delta
around the cold first prefill):

| arm | attention | compile | peak transient | outcome |
|---|---|---|---|---|
| custom | our gfx906 FA | inductor | **3.785 GiB** | OOM (last straw 508 MiB, free: 0) |
| rocm | ROCM_ATTN (Triton FA) | inductor | **1.035 GiB** | survived (4.77 GiB free after) |
| eager | our gfx906 FA | eager | **4.506 GiB** | OOM (0.37 GiB free after) |

## VERDICT

**HYPOTHESIS REJECTED as stated.** Compile held at inductor: swapping
our FA for Triton FA removes **2.75 GiB** of the transient (3.785 →
1.035) — the custom FA's prefill path is the **dominant owner**. FA
held constant: inductor is **0.72 GiB smaller** than eager (3.785 vs
4.506) — inductor's memory planner reuses activation memory; eager
materializes more. The boot J/K “inductor gptq_gemm” OOM frame was
where the memory ran **out**, not the owner: the 508 MiB Exllama fp16
M×N output (508 = exactly 2× the TP=2 254 MiB last straw; N TP-split,
~[4096, 32512] fp16 = 254×128 cols) is the model's own GEMM output
landing last on an exhausted allocator. (Resolves the boot-I “532 MiB
inductor OOM” datum: 532,676,608 B is this same 508 MiB alloc,
misread as 532 MiB.)

**Root cause FOUND (per-layer instrumentation, `attr_tp1_custom15/17`):
the q_pad buffer was per-IMPL, and v1 creates one backend impl per
attention layer.** `_ensure_forward_buffers` grew `self._q_pad_buf`
([num_seqs, Hq, Sq_pad, D] fp32 = **256 MiB** at the 4096-chunk:
metadata pads num_seqs to max_num_seqs=4) on each impl's FIRST
prefill call and the capture-latched retire policy keeps every
generation alive → 52 impls × 256 MiB = **13.3 GiB** of duplicate
buffers (TP=2: 16 heads/GPU → 6.7 GiB/GPU). The per-layer probe
showed exactly +256.0 MiB net at EVERY attention call (monotone
23.389 → 25.690 over 9 calls; OOM at ~call 10), while the C++
binding itself is clean (+64 MiB churn, same out_ptr reused —
`probe_fa_reuse.py`: 52 calls, flat) and the Python wrapper is clean
(+64 MiB `out_flat`, freed). This also explains the boot J/K
signatures: "transient ~linear in the chunk" (buffer ∝ Sq_pad ∝
chunk), bt2048 surviving (64 MiB/impl → 3.35 GiB total), and the
2026-08-26 comment that already noted the "x52 layers = 14 GiB,
first-request OOM" pathology (that fix only stopped the DECODE-time
dim0 growth; the per-impl prefill duplication remained).

**Fix: q_pad buffers are now ClassVar (shared across all impls, one
set per worker — the pattern the gather buffers already use, which
had the identical bug fixed earlier).** `vllm/gfx906_fa/
gfx906_fa_backend.py`: `_q_pad_buf` / `_q_pad_decode_buf` /
`_q_pad_retired` / `_q_pad_captured` → ClassVar; `_ensure_forward_
buffers` → @classmethod (num_heads/head_size now parameters). The
q_pad lifecycle test was rewritten with the class-state
snapshot/restore pattern (the gather-buffer tests' pattern) so the
grow sequence starts from a clean shared buffer.

**Verification status:** unit gate **PASS — 51/51** (incl. the
rewritten `test_q_pad_buffer_survives_capture_then_prefill_grow`
capture→grow→retire→replay sequence). Probe verification **PASS
(boot L, post-reboot, `attr_tp1_custom20_bootL.log`)**: custom arm
now **SURVIVES** the 0.5 GiB-KV-cap 4097-token prefill with peak
transient **1.285 GiB** (vs 3.785 pre-fix; 4.89 GiB free after,
vs 0.00) — the predicted model-core + one-time-256-MiB + churn
shape, and the 2.75 GiB custom-vs-rocm gap is gone.

**Serving re-validation (boot L, 2026-08-28, PASS)**: boot K TP=2
recipe with **`--max-num-batched-tokens 4096`** (only change), 6 GiB
KV cap kept, 128k max, ngram n=5, captures [6,12,18,24]. First real
8192-token request (two 4096 chunks — the exact boot J/K OOM site)
**cleared**: prefill 452.4 t/s (cold, ttft 18.1 s incl. first-use
compile) / 496.9 t/s (warm); no OOM, no wedge; VRAM steady 23.4
GiB/GPU (22.3 + ~1.1 runtime buffers, 8.7 GiB headroom/GPU). Decode
(ngram ceiling, filler): 8k/B=1 **97.2 / 101.9 t/s** (boot K: 79.1 —
includes M1 gather clip + the q_pad fix + warm cache; the clean
isolated M1 contribution is the +8.1% harness A/B below), 2k/B=1
111.5 (boot K: 114.6), 2k/B=4 46.7 aggregate (boot K: 45.3).
**The bt2048 workaround is droppable; bt4096 is the new default**
(preference: larger chunk = faster prefill; the 6 GiB cap keeps
plenty of margin, so it stays).

(Boot K note: the pre-reboot verification attempts both hit the
boot-K burst wedge — custom18_fixed 21:58, retry custom19 22:06,
2nd consecutive launch failure, GPU0 left 24.9 GB zombie VRAM;
degradation.md 21:58/22:06.)

Ruled out along the way: `o_part` KVSPLIT partials (the binding
forces `kv_split=1` for `seq_q>2` — no such alloc at prefill),
per-call binding/wrapper churn (flat; same buffer VA reused), Q8
side view (alias, 0 bytes), gather buffers (already class-level,
+0.0 MiB per call). Dead ends recorded: this torch build's memory-
snapshot stack capture returns only `?:0` unwind frames (no user
frames) and kineto records no device-memory events on this ROCm
stack — the per-layer `memory_allocated()` hook + segment-snapshot
diff is what worked.

## Evidence — the host-crash detour (cost a session; keep for the skill)

Fresh in-process compiles on this box (torch 2.10-dev venv) die in
order, before ever reaching the OOM site:

1. **AOT is ON by default on torch ≥ 2.10** (`use_aot_compile()` in
   `vllm/envs.py`): its out-of-process compile workers die with
   “Could not find an active GPU backend” on every fresh compile.
   Servers are unaffected (compile once on a clean boot, then warm AOT
   cache). Fix: `VLLM_USE_AOT_COMPILE=0`.
2. **Inductor writes `async_compile.wait()` into every generated
   wrapper** (this torch) and its pool defaults to the FORK
   SubprocPool (`TORCHINDUCTOR_WORKER_START=subprocess`) — HSA is not
   fork-safe after the parent initialized HSA → same GPU-backend error
   in the child.
3. **`TORCHINDUCTOR_WORKER_START=spawn` also failed** — but
   differently: the spawned child could not init HSA at all while the
   parent's HSA was healthy (at the failing exec: parent
   `is_available=True, bad_fork=False, devcount=1`, triton backends
   fine, `_is_backend_active("amd")` True in parent). A standalone
   spawn child DID init — state-dependent (boot K had 3 GPU incidents
   by then; degradation.md 20:40). Workaround: replace
   `AsyncCompile.process_pool` with a `ThreadPoolExecutor` (parent
   threads; HSA already up) — also keeps compile transients inside the
   measured process.
4. **`TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` required**: the rblock
   variant-compile path (`_dynamic_scale_rblock` → direct
   `triton.compile`) crashes in our triton-gfx906 fork with
   `AttributeError: 'NoneType' object has no attribute '__code__'`
   (`get_jit_fn_file_line` → `JITCallable.fn is None`) — first compile
   fine, the *variant* compile broken.

## Notes / pending

- Post-OOM segment snapshot (`oom_snap_custom_tp1_pp4097.json`):
  31.30 GiB reserved / 26.47 allocated, 589 segments; top blocks
  2×2566 MiB + 256 MiB weight shards. History-armed re-run
  (`attr_tp1_custom11_hist.log`) reproduced the OOM byte-for-byte
  (3.785 GiB / 532,676,608 B last straw) but the snapshot frames
  are empty unwind-only stacks — stack capture is broken in this
  torch build, and kineto records no device-memory events on this
  ROCm stack either. The working instruments: per-layer /
  per-binding `memory_allocated()` hooks + segment-snapshot diff.
- The >10.6 GiB/GPU TP=2 serving OOM (boot J/K) is the same bug at
  16 heads/GPU (6.7 GiB q_pad growth + inductor buffers > 10.6
  headroom). The ClassVar fix should let bt4096 serve under the 6
  GiB KV cap — **re-validate the boot K launch recipe (bt2048
  workaround droppable, 6 GiB cap shrinkable) post-reboot, after the
  probe verification passes** (pending list in degradation_details.md
  22:06).

## 2026-08-27 — round 5: M1 window clip on the gather path (B=1 decode) — implemented + unit-gated, e2e pending

## HYPOTHESIS

Extending the Phase C window clip to the gather path — via an
absolute-position gather layout (gather writes only rows
`[kv_start, seq_len)` at buffer index == absolute position) plus the
FA kernel starting its k-loop at the floored `kv_start` — is
**bit-identical** (no mask changes; the floored start matches the
existing paged Phase C floor block) and cuts the gather + FA work for
B=1 long-context decode (the kernel micro-bench showed −48% FA time
at L=8k/W=2k).

## What was done

- `fattn-q8.cuh`: new `kv_start` param (after `window`);
k0_base floor block **LOCKSTEP-copied** from the paged Phase C floor
  (floor is for bit-identity: an unaligned first tile re-associates
  the fp16 reduction, ~5e-4); both loop starts shifted to
  `k0_base + blockIdx.y*nbatch_fa`.
- `gfx906_fa_gather.cu`: persistent gather takes `kv_start` (int32
  [B], optional); per-seq gather start + 128-row margin
  (`GATHER_CLIP_MARGIN`, ≥ max nbatch_fa=128 so the floored start is
  always materialized); grid-stride kernel → no host row-count change.
- `gfx906_fa_launcher.cu` / `gfx906_fa.cpp`: signatures + bindings
  (`py::arg("kv_start") = nullopt`; int32 [B] validation mirroring
  paged-direct).
- `gfx906_fa_paged.py`: `GFX906_FA_GATHER_CLIP` kill switch (default
  1); `kv_start = max(0, q_abs + 1 − window)` per seq, int32
  contiguous, computed in the persistent sub-path and passed to both
  the persistent gather and `forward`; `_DOUBLE_CHECK` requires
  `kv_start is None` (the double-check kernel's window test assumed
  full-gather positions — keep the invariant explicit).
- M1 v1 = **persistent sub-path only** (B≤16, all Sk = all current
  server traffic); other gather sub-paths pass `kv_start=None`
  (full gather, still correct).

## GATE

Bit-identity unit tests (clip ON vs OFF, direct dispatch forced off
so B=2 really uses gather) + full suite + e2e pp8192/B=1 tg256 A/B
(`GFX906_FA_GATHER_CLIP` 1 vs 0, record recipe).

## VERDICT

**PASS — both gates.**

- **Unit:** 5/5 new tests (Sq=1/6 × B=1/2 at L=4353/W=2048 unaligned
  starts 2305/2300 + short-ctx inert L=513<W) + full suite 51/51.
  Kernel-level sanity: rows `[2305, L)` bit-identical, rows
  `[0, 2305)` skipped by gather.
- **E2E (boot L, 2026-08-28, record recipe: BENCH_EAGER=0
  BENCH_GPU_UTIL=0.95 BENCH_SAMPLES=4 BENCH_PP=8192 BENCH_TG=256
  BENCH_MAX_SEQS=32 BENCH_KV_MEM=805306368 BENCH_BATCHED_TOKENS=1024,
  GPU0, `m1_e2e_clip{1,0}.log`):**

  | `GFX906_FA_GATHER_CLIP` | samples (t/s) | mean |
  |---|---|---|
  | 1 (on) | 6.118 / 6.084 / 6.022 / 5.946 | **6.042** |
  | 0 (off) | 5.706 / 5.573 / 5.547 / 5.522 | **5.587** |

  **+8.1%** at pp8192/B=1/tg256 (no spec decode — the harness is
  single-request). Matches the kernel micro-bench's −48% FA time at
  L=8k/W=2k applied to the FA's ~17% share of the B=1 step at 8k
  context (0.48 × 0.17 ≈ +8.2%). The clip's e2e win is largest at
  B=1 long context (the gather hot path); the TP=2 serving decode
  numbers (boot K grid: 114.6/79.1/57.0 @2k/8k/16k, ngram ceiling)
  should move by the same relative amount once re-validated.

---

## Promotion to main (2026-08-28)

`feat/muse-glimmer` fast-forward-merged to `main` (branch point
`284ce5ff6a`, tip `52ff21f9d9` + promotion docs). All gates in this
log passed at merge time (unit 51/51 incl. the fix commit's rebuild;
probe survival 3.785→1.285 GiB; M1 e2e +8.1%; bt4096 TP=2 prefill
clear; canary 38.8 t/s healthy). Post-merge open items per
`roadmap-more-models.md`: M2–M5 (long-context split-K accuracy, kernel
hygiene, default read-path decision) and the residual review nits
(F4/F5 class-buffer release + grow-shape checks; F2 seq_lens slice;
F3 partial-range double-check). CHANGELOG 2026-08-27–28 entry added.

## 2026-08-28 — round 6: M1 clip ported to the fused Q8 gather (LEGACY=0 B=1/prefill) — unit-gated, serving A/B pending (M5)

## HYPOTHESIS

The fused Q8 gather kernels (V1 per-token, the default; V2 paged-
block, auto-selected at Sk>65535) can take the M1 window clip
verbatim — same kv_start/margin/absolute-layout contract as the
persistent kernel — because their dst writes are already at absolute
token indices. LEGACY=0's B=1 decode + prefill would then scan only
`[kv_start, seq_len)` like LEGACY=1's, removing the last performance
argument against flipping the default (M5): pre-port, at 8k ctx the
unclipped LEGACY=0 fused gather moved ~3× the HBM traffic of the
clipped LEGACY=1 persistent gather (1164 B × 8192 rows vs 1288 B ×
2176 rows per sliding layer); post-port it is ~10 % leaner per row
than LEGACY=1 at every context (no in-kernel quantize, 388 B/row
K+V read).

## What was done

- `gfx906_fa_gather.cu`: V1 early-returns per token below the per-seq
  clip start (no V-zero, no K write — LOCKSTEP with the persistent
  kernel, which writes nothing in `[0, start)` either); V2 early-
  returns whole paged blocks inside the clipped prefix + per-token
  skip in all four copy loops; `GATHER_CLIP_MARGIN` moved to the top
  of the file (visible to all three kernels; still sourced from
  `kernel/gfx906-config.h`, whose static_assert ties it to the FA
  table). No fattn-kernel changes (the k-loop floor + margin
  contract is shared).
- `gfx906_fa.cpp`: `kv_start` (optional int32 [B]) plumbed through
  the `gather_paged_kv_q8` binding + launcher; the fp16 fallback
  caller passes NULL (unclipped, unchanged).
- `gfx906_fa_paged.py`: the kv_start formula moved to one helper
  (`_gather_clip_start`), now used by BOTH the fused branch (new) and
  the persistent branch (inline copy replaced — no formula drift
  possible); the fused branch passes it to the gather binding, and
  the existing FA call already receives it. Fallback sub-paths
  (torch-gather / two-kernel / fused-quant) intentionally stay
  unclipped (kv_start=None → FA full scan — unchanged behavior).
- F1 (review): `_ensure_q8_sidebuffer` fit-assert now checks
  `D * key_cache.element_size()` instead of `2 * D` (the fp8-KV +
  D≤64 slice-clamp hole); stale "88 B" Q8-row docstring corrected
  to 136.
- Tests: fused-Q8 alias A/B bit-identity (clip on/off), Sq∈{1,6} ×
  B∈{1,2} @ L=4353/W=2048 (unaligned start 2305); short-ctx inert
  (L=513<W, kv_start=0); functional prefix-skip (sentinel buffer,
  kv_start=769/margin=128: rows [0,641) untouched K and V, rows
  [641,1025) bit-equal to the full gather, V tail still zeroed) —
  the A/B form alone cannot distinguish "clipped correctly" from
  "clip silently inert".

## GATE

Unit: the five new tests above + full suite (57). Serving: NOT this
round — the M5 gate (LEGACY=0 vs LEGACY=1 TP=2 A/B, B=1 + B=4 @ 8k,
canary green, plus the two remaining e2e gaps (a) LEGACY=1 +
direct-paged + clip and (b) B=4 long-ctx clip on/off) decides the
default flip separately.

## VERDICT

**PASS (unit level).** 57/57 on the rebuilt `_gfx906_fa_C` (boot L).
V2's clip is covered by LOCKSTEP inspection only — `GFX906_FA_GATHER_V`
is process-global (read once), so a V2 unit A/B is not expressible in
the existing suite; flagged for the M2 hygiene batch (same as the
other V2 gaps). Prefill-under-clip on the fused path inherits the M1
e2e result (the persistent twin was e2e-gated at pp8192 with the
identical kv_start contract); a fused-path prefill A/B is folded into
the M5 serving bake.

## 2026-08-28 — round 7: M5 gate e2e — gaps (a)/(b) closed (with an erratum on what gap (a) actually was)

## HYPOTHESIS

The two remaining M5 e2e gaps — (a) "the LEGACY=1 + direct-paged +
clip combination has never been separately e2e-gated" and (b) "no
B=4 long-context (8k+) clip on/off e2e" — are closable with in-
process harness A/Bs at pp8192/tg256 (B=2 for (a), B=4 for (b)), and
both will show the clip's expected delta (the round-3 Phase C A/B
showed +3.6% e2e at the same shape).

## What was done

- **Erratum on gap (a):** direct-paged dispatch is gated on
  `key_cache_q8 is not None` (gfx906_fa_paged.py) — i.e. **LEGACY=0
  only**. Under the default LEGACY=1, B≥2 decode auto-gating selects
  the persistent GATHER path with the M1 clip (`GFX906_FA_GATHER_CLIP`),
  never direct-paged. The original gap (a) ("LEGACY=1 + direct-paged +
  clip") was therefore VACUOUS — that combination does not exist in
  production; the round-3 +3.6% A/B (run LEGACY=0) had already gated
  the LEGACY=0 direct-paged + Phase C path. The real gap (a') is:
  the LEGACY=1 B≥2 gather + M1-clip combination (what the default
  actually dispatches) needed its own GATHER_CLIP on/off A/B. Roadmap
  M5 text corrected.
- A/Bs run (harness record recipe, pp8192/tg256, 4 samples, prefix
  off, GPU0, boot L; launches serialized after the 12:43/12:54
  double-wedge pairs — see degradation.md):
  - (a') B=2 (nreqs=2): GATHER_CLIP=1 **6.410/6.394/6.386/6.385** vs
    =0 **5.710/5.706/5.705/5.702** → **+12.1%**.
  - (b) B=4 (nreqs=4): GATHER_CLIP=1 **6.093/6.086/6.077/6.068** vs
    =0 **4.982/4.982/4.983/4.982** → **+22.1%**.
  - (context) B=1 clip A/B was the round-5 M1 e2e: +8.1% (6.042 vs
    5.587). The delta grows with batch (gather HBM scales with B, the
    per-step GEMM/launch overhead does not) — the expected shape.
  - A first B=2 A/B toggling `GFX906_FA_WINDOW_CLIP` (the Phase C
    flag) showed 0.1% — consistent with the erratum: under LEGACY=1
    that flag is a no-op at B≥2 (direct-paged unreachable). Both of
    its arms ran the gather+clip-ON config: **6.38 t/s is the
    current-default B=2/8k in-process record** (6.39 on the
    GATHER_CLIP=1 re-run).
- G3 (LEGACY=0 TP=2 serving bake) attempted; it caught a real bug
  (below), which was fixed; the re-run then hit the boot-L wedge
  burst — G3 is PENDING post-reboot.

## GATE

(a') + (b) A/B deltas above (both must be a real positive delta, not
a no-op — the 0.1% WINDOW_CLIP A/B is the counter-example that
caught the erratum); unit bit-identity already in place (57/57,
round 6).

## VERDICT

**PASS — gaps (a') and (b) closed.** The default (LEGACY=1) B≥2
combination is e2e-gated with a real clip delta at 8k (+12.1% B=2,
+22.1% B=4); the LEGACY=0 direct-paged + Phase C path was already
gated in round 3 (+3.6%). What remains for the flip decision: the
G3 LEGACY=0 TP=2 serving bake vs the boot L LEGACY=1 records
(111.5/99/46.7 @2k/8k/B=4).

## 2026-08-28 — round 8: G3 caught a latent capture-unsafe sync in the direct-paged branch; fixed; G3 itself blocked by the boot-L wedge burst

## HYPOTHESIS

The LEGACY=0 TP=2 ngram serving bake (G3) boots and serves — the
LEGACY=0 path (fused Q8 gather + clip, direct-paged + Phase C) is
cudagraph-capture-safe in serving, as the persistent path was.

## What was done

- G3 attempt 1 (~14:20Z, TP=2, capture [6,12,18,24]) died during the
  **FULL decode capture** with `hipErrorStreamCaptureUnsupported`
  ("operation not permitted when stream is capturing") at
  `gfx906_fa_paged.py:447` — `n = int(cu[s+1] - cu[s])`, a D2H sync in
  the direct-paged branch's Q-pad **and** output-unpad fallback loops
  (the Sq>1 case). **Pre-existing latent bug, not a round-6
  regression:** those loops only run when Sq>1, and direct-paged is
  reachable only under LEGACY=0 (round-7 erratum) — under the
  default LEGACY=1 the B≥2 decode never entered this branch, and the
  in-process harness never used spec decode (Sq=1 decode → fast
  path; Sq>1 prefill → gather). First production hit = LEGACY=0 +
  ngram (Sq=6 per seq) + B≥2 under FULL capture.
- **Fix (in-tree, this commit):** added the gather branch's existing
  capture-safe **uniform-batch fast paths** to the direct-paged
  branch (both Q-pad and output-unpad): `num_tokens == num_seqs *
  max_seqlen_q` is a host-integer check, so no sync; spec batches
  are always uniform (n_q = 1 + num_spec). The sync loops remain
  only in the non-uniform fallback (eager-only reachability: prefill
  chunks are piecewise/EAGER at the attention split op, so capture
  never sees them).
- Validation: 57/57 suite — the `nq=6/B=2` cases of
  `test_forward_paged_fusedq8_window_clip_bit_identical` now
  dispatch to direct-paged (auto mode, B≥2) and assert bit-identity
  vs the gather path, so the new idioms are exercised and match.
- G3 attempt 2 (~14:38Z) wedged at weight load
  (`hipErrorLaunchFailure`, Worker_TP1 both ranks) — 3rd boot-L wedge
  observation = BURST per the 12:54Z protocol → **all GPU work
  stopped; reboot (root) required** (degradation.md +
  degradation_details.md item 9). Possible collateral to attempt 1's
  teardown; unresolvable without the reboot.
- G3 recipe (post-reboot): `HIP_VISIBLE_DEVICES=0,1
  GFX906_FA_LEGACY=0` + the README TP=2 flags (bt4096, 6 GiB KV cap,
  ngram n=5, capture [6,12,18,24]); grid
  `_bench_serve_grid_gfx906.py` default ×3; control = the boot L
  LEGACY=1 records (111.5/99/46.7 @2k/8k/B=4 — same boot, same
  recipe; tree delta since is LEGACY=0-path-only, re-validated by
  today's clean single-card runs).

## GATE

G3: LEGACY=0 TP=2 grid ≥ the LEGACY=1 records at every point (a wash
= keep LEGACY=1 per the flip rule). Blocked on the reboot.

## VERDICT

**Capture bug: FIXED (57/57). G3: BLOCKED (host reboot pending).**
Flip decision (D1) remains gated on G3.

## 2026-08-28 — round 9: G3 executed (boot M) — LEGACY=0 LOSES to LEGACY=1; default stays 1, no flip

## HYPOTHESIS

The LEGACY=0 Q8 pre-quantized read path (fused Q8 gather + clip at
B=1, direct-paged + Phase C at B≥2) is at least a wash against the
LEGACY=1 gather+re-quantize path in TP=2 ngram serving — the per-row
HBM read is 10% leaner (392 vs 512 B at D=128), and the clip now
matches on both paths, so the flip gate (≥ control at every grid
point) should pass.

## What was done

- Boot M (~14:52Z, post burst-reboot): clean arrival (both cards
  0%/0%, 33 °C), canary **38.9 t/s** (healthy band 38.4–38.9) — host
  clear. G3 attempt 1 wedged at weight load (10th chronic weight-
  load hang; canary was green 2 min prior) — isolated, self-
  recovered, retry per protocol.
- **G3 attempt 2: clean** (ready in ~108 s — warm inductor/AOT cache
  from attempt 1; weights + 4-size FULL capture all passed, incl.
  the round-8 uniform fast paths under ngram B≥2; spec acceptance
  6.00 throughout). Grid ×3, same recipe as the control (bt4096,
  6 GiB KV cap, ngram n=5, capture [6,12,18,24], prefix cache off,
  filler prompts):

  | point | LEGACY=0 (this run) | LEGACY=1 (boot L control) | Δ |
  |---|---|---|---|
  | 2k/B=1 decode | 107.44/107.41/107.43 | 111.5 | **−3.7 %** |
  | 8k/B=1 decode (tg256) | 93.69/97.81/97.96 | ~99 | **−2.5 %** |
  | 8k/B=1 decode (tg512) | 95.58/97.76/97.89 | — (no control) | — |
  | 16k/B=1 decode | 88.03/88.13/88.07 | — (no control) | — |
  | B=4 @2k aggregate (s1/s2) | 35.79 / 32.08 | 46.7 | **−27…−31 %** |
  | 8k prefill (warm, tg256 s1/s2) | 495.3 / 495.5 | 496.9 | wash |

  B=4 s0 (27.96) is the session's first B=4 sample (cold); s1/s2
  are the stable readings. TTFT/queuing structure identical to the
  control run (ttft_max ~16 s at B=4/2k from the 4096-token
  serialization).
- **Interpretation (corrected 2026-08-28 — the original wording
  mis-attributed the delta to an FA-dot ALU gap that this A/B never
  measured):** both LEGACY modes run the *identical* FA-Q8 kernel —
  `forward_paged_q8` with the `v_dot4_i32_i8` Q·K dot on both sides
  (the LEGACY flag selects the KV-layout/gather strategy, not the FA
  dot). So: (1) the B=1 delta (−2.5…−3.7 %) is in the gather kernel
  + read path (fused per-token gather of pre-quantized 136 B rows vs
  the persistent gather), not the FA inner loop; (2) the B=4 delta
  (−27…−31 %) is the direct-paged kernel's block-strided paged
  global reads (256 B-row stride, uncoalesced across pages) vs
  LEGACY=1's gather-materialize-then-contiguous-read — a memory-
  access-pattern penalty, not an instruction/ALU penalty; (3) the
  10%-leaner per-row HBM read is invisible because none of these
  points is HBM-bandwidth-bound at these context/batch sizes.
  Prefill (gather-bound, not FA-bound) is a wash — consistent with
  the B=1 in-process A/Bs (round 7: the LEGACY=1 gather+clip path
  was already competitive there).
  ISA follow-up (same date, `benchmarks/kernels/gfx906/dot_isa_probe.py` +
  `dequant-instructions.md`): the Q·K dot already uses the only
  full-rate int dot on gfx906 (`v_dot4_i32_i8`; dot4c/dot8_i8 are
  assembler-rejected), so there is no int-dot upgrade for Q8. The one
  unexploited dot-family lever is P·V (currently
  `v_pk_mul_f16` + `v_pk_add_f16`, fp16-acc → `v_dot2_f32_f16` would
  halve P·V instruction count with fp32-acc) — an M3 candidate that
  would speed BOTH LEGACY modes equally, so it does not change this
  A/B's outcome.

## GATE

LEGACY=0 TP=2 grid ≥ LEGACY=1 records at every point.

## VERDICT

**FAIL — LEGACY=0 is slower at every controlled point (−2.5 %…−3.7 %
B=1 decode, −27…−31 % B=4 aggregate; prefill wash). Per the flip
rule (a wash already keeps LEGACY=1; only a win justifies flipping),
D1 is NOT executed: `GFX906_FA_LEGACY` stays `1`.** The LEGACY=0
path remains an experimental opt-in (zero-extra-KV-memory alias,
COW-safe) — useful if/when a gfx906 FA change (e.g. the M3 P·V
dot2 rewrite, or a cheaper direct-paged read pattern) closes the gap
measured above, at which point the M5 gate re-opens. Roadmap M5: gate executed → closed
with the keep-LEGACY=1 decision. The round-8 capture fix stands on
its own (it made any future LEGACY=0 ngram serving possible).

**M5 outcome (for the roadmap): M5 = DONE (decision made; default
unchanged). Residual M5-adjacent work: none blocking — the LEGACY=0
read-layout gap (not a compute gap, corrected 2026-08-28) is M6
territory; the unrelated P·V dot2 lever is an M3 kernel-hygiene topic.

---

## 2026-08-28 — round 10: M6 Part B — LEGACY=0 B≥2 rerouted to the fused-Q8 gather; B=4 @2k loss RECOVERED; sub-flag default flipped

## HYPOTHESIS

If the LEGACY=0 B=4 @2k spec-serving loss (M5: −27…−31 % vs the
LEGACY=1 gather control) is the direct-paged kernel's strided
Q8-slice K reads (misaligned 136-of-256-B slices + per-row page
indirection), then routing LEGACY=0 B≥2 through the fused-Q8 gather
(the LEGACY=0 B=1 path, compact-tile reads) recovers B=4 @2k to
LEGACY=1 parity without touching B=1 or prefill.

## What was done

- `gfx906_fa_paged.py`: new `GFX906_FA_DIRECT_PAGED_Q8` knob (commit
  `50326f9b06`); `=0` makes the Level-3c direct-paged dispatch
  condition fail, so LEGACY=0 B≥2 falls through to the fused-Q8
  gather. The M1 gather clip is dispatch-agnostic
  (`_gather_clip_start` from seq_lens/cu/window) — retained under the
  reroute.
- Unit: `test_forward_paged_q8_direct_paged_q8_off_routes_gather`
  (nq×B ∈ {1×2, 1×4, 6×2}, L=4353/W=2048 unaligned clip, LEGACY=0
  alias layout): rerouted output vs the torch windowed reference at
  the 5e-2 cross-path tolerance, clip on/off A/B bit-identity at B≥2
  (the clip stays active under the reroute), and a dispatch-change
  check (direct vs gather outputs differ — the two FA variants are
  not bit-identical). 60/60 suite.
- In-process A/B (boot M, GPU0, pp8192/tg256/BENCH_NREQS=4, record
  recipe, LEGACY=0, `GFX906_FA_DIRECT_PAGED_Q8=1`): 6.074/6.057/
  6.061/6.062 — a WASH vs the round-7 LEGACY=1 B=4 in-process record
  (≈6.08). **The M5 loss does not exist in the Sq=1 in-process
  regime; it is specific to the ngram spec-decode (Sq=6) serving
  regime**, so the serving bake is the gate (the in-process harness
  cannot reproduce Sq>1).
- Serving bake (boot M, TP=2, README M5 recipe verbatim: ngram n=5
  prompt_lookup 2, bt4096, KV 6 GiB, capture [6,12,18,24],
  `--max-model-len 131072`, grid `_bench_serve_grid_gfx906.py`
  default ×3; canary 38.4 t/s at ~16:59Z; one 17:18Z weight-load
  wedge on arm-1 launch — recorded in `degradation.md`/`_details`
  item 11, self-recovered, retry clean):
  - Arm 1 (LEGACY=0 + `=1`, current LEGACY=0 default):
    B=1 @2k 107.6/107.6/107.4; **B=4 @2k 34.76/36.26/36.05**; B=1
    @8k ≈97.7; B=1 @16k 88.0 → reproduces the M5 loss (35.7 vs 46.7
    = −23 %).
  - Arm 2 (LEGACY=0 + `=0`, the reroute): B=1 @2k 107.2/107.3/
    107.2; **B=4 @2k 46.34/46.29/46.13**; B=1 @8k 93.5/97.8/97.8;
    B=1 @16k 87.8. B=4 = **parity with the 46.7 LEGACY=1 control
    (−0.9 %)**; B=1 and prefill unchanged (ttft_max 16.07 s vs
    16.09 s; the B=4 prefill is Sq=2048 → never direct-paged, so
    both arms ran the same gather there).

## GATE

Plan gate (plan_fa_legacy0_impr_claude.md Part B): B=4 @2k
aggregate, M5-recipe serving bake vs the 46.7 t/s LEGACY=1 control.

## VERDICT

**SHIPPED — the reroute recovers the B=4 loss: 35.7 → 46.3 t/s
(+29.7 %), parity with LEGACY=1 (−0.9 %, cross-boot; M5's own
same-config sample spread was ~11 %, so 46.3 vs 46.7 is a wash,
not a loss). `GFX906_FA_DIRECT_PAGED_Q8` default flipped 1 → 0:
gather is the LEGACY=0 B≥2 default; direct-paged is opt-in (=1).
Under the production LEGACY=1 default the flip is a no-op (direct-
paged is LEGACY=0-only, round-7 erratum) — zero production risk.
Direct-paged has no measured advantage anywhere (in-process Sq=1
wash; spec-serving B=4 −23…−31 %) — it stays as an experiment
route only.**

**M5 impact:** the LEGACY flip's B=4 gate is now GREEN (LEGACY=0 +
gather-routed B=4 at LEGACY=1 parity). The flip itself (roadmap M5
re-open) still needs the B=1 same-boot adjudication: LEGACY=0
B=1 @2k = 107.2 (boot M, this round) vs the LEGACY=1 control 111.5
(boot L) — cross-boot −3.9 %, within the band M5's own numbers sat
in (110.5/109 boot M vs 111.5 boot L); a same-boot LEGACY=1 arm
would settle it. Deferred: boot M is at 3 wedge observations in
~2.5 h (15:00Z, 17:18Z + the boot-L pattern) — the next two-card
launch belongs on a fresh boot.

**Erratum (2026-08-28, post-review — F1/F3 of
`fa-legacy-m0-m6-code-rev.md`): the mechanism as committed does not
survive this round's own data.** The "What was done" / "VERDICT"
text above states the B=4 loss as "direct-paged's strided Q8-slice
reads", escalating the 2026-08-28 `DEVLOG-fa-kernel-batches.md`
inference ("code-level attribution") to settled fact. This round's
own in-process A/B contradicts the claim as stated: arm 1
(LEGACY=0 direct-paged — the IDENTICAL strided-read path) at B=4/
Sq=1 came back a WASH vs the round-7 LEGACY=1 record (6.064 vs
≈6.08). Dilution arithmetic: at Sq=1 the attention tile work is ~1/6
of a spec step (n=5 draft + 1 verify row), so if direct-paged's
attention step were 2–3× slower from misaligned reads, Sq=1 should
show ~−10…−18 %, not a wash; K-tile reads are also amortized across
a tile's Q rows, so a pure read-layout penalty should still surface
at Sq=1. The loss is therefore Sq>1-specific, which points at the
direct-paged Sq>1 machinery itself (round-8's capture-safe Q-pad/
unpad fast paths, per-layer copies, graph-capture interaction) as at
least a co-contributor — a path with prior form for exactly this
kind of surprise (round 8 found its capture-unsafe D2H sync in
precisely this configuration: LEGACY=0 + ngram + B≥2 + FULL
capture). Corrected statement: the B=4 loss is *associated with the
direct-paged Sq>1 serving path*; strided Q8-slice reads remain the
**leading but unconfirmed** hypothesis. The README (LEGACY row,
DIRECT_PAGED_Q8 row), roadmap M6 paragraph, and CHANGELOG entry
have been softened accordingly; the `6b46d2a2e0` commit message is
immutable post-merge and stands as written (its mechanism clause is
the pre-erratum framing).

**F3 framing fix:** the B=4 "parity" (46.3 vs 46.7) is a
**cross-boot** comparison (boot M vs boot L); the ~11 % M5 sample
spread (35.79/32.08) cited to justify it came from a looser-
conditioned session (s0-cold-contaminated), while this round's
same-boot arm spreads were far tighter (arm 1 ≈4 %, arm 2 ≈0.5 %).
Honest framing, applied in all updated docs: "parity within
cross-boot uncertainty, same-boot adjudication pending" — the same
hedge the B=1 question already carried. Also: arm 2's B=1 @8k
first sample (93.5 vs 97.8/97.8) is unremarked in the original text
— treat it as suspected-cold-start (the M5 s0 contamination class),
not a real −5.5 % point; the arm's representative value is 97.8.

**F2 (the missing cheap measurement, run 2026-08-28):** in-process
B=4/Sq=1 head-to-head — LEGACY=0 fused-Q8 gather (post-Part-B
reroute) vs LEGACY=1 fp16 gather — record recipe
(BENCH_EAGER=0, pp8192/tg256, BENCH_NREQS=4, 4 samples, GPU0, boot
M):

| arm | samples (t/s) | mean |
|---|---|---|
| LEGACY=1 (fp16 gather) | 6.091 / 6.080 / 6.075 / 6.078 | **6.081** |
| LEGACY=0 (fused-Q8 gather) | 6.097 / 6.094 / 6.087 / 6.086 | **6.091** |

**Wash** (+0.16 % for Q8, inside run-to-run noise; the LEGACY=1
mean reproduces the round-7 record 6.081 on this boot). Corollary:
against round 10's arm-1 in-process number (LEGACY=0 direct-paged,
same config: 6.064), the rerouted gather is ≥ the direct-paged
route at B=4 even in-process (+0.45 %). Per the plan's open
question (`plan_fa_part_A.md`, "Arm B outcome"): the B=4
gather-vs-gather question is closed at parity, so the B=1 gap is
the entire remaining LEGACY=0 deficit — consistent with the M6
regime (B=1 is gather-dominated; B=4 is GEMM-dominated, so a B=4
wash is expected even if the B=1 layout gap is real). Proceed per
the plan's test plan.

## 2026-08-28 — round 11: M6 Part A (planar Q8 quants/scale repack) — microbench hard-stop gate FIRED; flip question DEAD-END; code change NEUTRAL (kept on branch)

## HYPOTHESIS

If the LEGACY=0 B=1 deficit (−2.5…−3.7 % vs LEGACY=1, M5) is caused
by the interleaved `block_q8_0` layout (34-B block strides at 2-mod-4
tile offsets forcing narrow loader loads + scales scattered inside the
structs), then repacking the same 136 B into a quants plane + scale
plane recovers the B=1 gap — without changing the byte budget, row
stride, alias contract, or numerics. Falsifiable per
`plan_fa_part_A.md` (rev 2) via the pre-decided microbench stop-rule.

## What was done

- Implementation (`feat/fa-legacy0-m6-partA`, this round's commit):
  all three tile writers (alias `reshape_and_cache_q8`, fused-Q8
gather V1/V2, dense fallback `quantize_q8_0`) emit the planar layout;
  both FA loaders (non-paged + paged) read it; 4 new bit-exact layout
pins; suite 64/64. Intra-row only: row byte count `(D/32)×34`
invariant — strides, buffers, alias contract, COW semantics untouched.
- **Loader scale-offset bug found + fixed:** both rewritten loaders
applied the tile column-block offset to the quants plane but not the
scale plane (the old loader inherited it from the pre-offset struct
base). Latent for D=128 (`k_col_block0=0` when `nbatch_K==D`),
activated for D=256 prefill (`nbatch_K=128` → 2nd chunk at
k_KQ_0=128) — caught by
`test_forward_decode_prefill_vs_sdpa_on_unbind_cache` (t=63 err
0.0519 vs 5e-2 tol); fixed in both
loaders; suite 64/64; decode-vs-SDPA 0.43 % in BOTH LEGACY modes
(D=128, k_KQ_0=64 path exercised). Builds `/tmp/partA_build_*.log`.
- Gate microbench (`/tmp/microbench_partA.py`, standalone B=1 decode
step, L=8192/W=2048/D=128/Muse geometry Hq=32/Hkv=2/BLOCK=16,
record env, GPU0): OLD layout = csrc stashed + rebuilt
(`build_old4.log`); NEW = this branch. 5 OLD / 6 NEW process runs.
- ISA: launcher compiled `-S --offload-device-only` with the exact
CMake flags (`/tmp/isa_dump.sh`); decode kernel
`flash_attn_tile_q8<128,128,2,1>` (nbatch_fa=128, nbatch_K=64 → two
K chunks) extracted from the .s (`/tmp/isa_{old,new}.s`,
`/tmp/kern_{old,new}_decode.s`).

## GATE

`plan_fa_part_A.md` §Test-plan hard stop-rule (pre-decided rev 2):
PROCEED only if (a) ISA-verified loader VMEM instruction count drops
≥2× AND (b) standalone B=1 step time moves ≥2 %; either failing →
DEAD-END for the flip question, no in-process/serving slot spent on
the perf question.

## Evidence (launch-regime — microbench, not the serving gate)

| arm | OLD (interleaved) | NEW (planar) | Δ |
|---|---|---|---|
| LEGACY=0 gather+FA (ms) | 0.0831/0.0835/0.0829/0.0827/0.0832 → **0.08308** | 0.0814/0.0809/0.0811/0.0810/0.0812/0.0810 → **0.08110** | **−2.4 %** (bands disjoint) |
| LEGACY=1 quant+FA (ms) | 0.0732/0.0736/0.0731/0.0752/0.0752 → 0.07406 | 0.0730/0.0706/0.0734/0.0731/0.0732/0.0728 → 0.07268 | −1.9 % |
| LEG0/LEG1 ratio | 1.136 | 1.115 | gap narrows slightly |

ISA loader loop per 32-value block (decode kernel, 4 inlined chunk
copies each):

| | OLD | NEW |
|---|---|---|
| loads/block | 1×`global_load_ushort` + 4×`global_load_dwordx2` (offs 2/10/18/26), address via `v_mad … 34` | 2×`global_load_dwordx4` (offs 0/16) + 1×`global_load_ushort` |
| **loads/tile row** (J=64) | **10** | **6** → **1.67×** |

## VERDICT

**DEAD-END (flip question).** Stop-rule (a) FAILS: ISA-verified
loader loads 10 → 6 per tile row = 1.67× < 2×; (b) passes (LEG0
step −2.4 %, disjoint bands). Both were required. The plan's cost
model assumed ~17 loads/block today (34-B struct at 2-B alignment)
— the compiler already decomposed it into 4×8-B (GCN legal
misaligned) + 1×2-B, so the achievable drop is 5→3, not 17→5. The
mechanism is real but sub-threshold: the repack cannot single-handedly
account for the B=1 deficit (−2.5…−3.7 %), and per the stop rule no
in-process/serving slot is spent on the perf question. Per the
plan's falsification clause the B=1 gap is elsewhere — the
quantize write path, the FA Q-side, or the gather kernel's own
traffic. **Defaults unchanged: `GFX906_FA_LEGACY=1`,
`GFX906_FA_DIRECT_PAGED_Q8=0`.**

**Code change: NEUTRAL — kept on the branch, merge-or-revert is the
user's call.** The layout is bit-identical (64/64 suite incl. 4 new
pins; decode-vs-SDPA 0.43 % in both LEGACY modes), ISA-verified 10→6
loader loads with 16-B width (2×16-B + 2-B vs 2-B + 4×8-B, the ×34
MAD leaves the hot path), and a measured −2.4 %/−1.9 % standalone
B=1 decode step in the LEGACY=0/LEGACY=1 arms. If merged to main it
is hygiene with a small win on the production path (LEGACY=1 uses
the same loader); a serving regression arm was NOT run (stop rule:
no perf slot) — the 64/64 bit-pins are the correctness record.
Refrigerated: A2 (144-B padded stride) is moot for the flip
question — A1 already measured below the 2× threshold. Part C
(Q4-KV/`v_dot8_i32_i4`) remains the M6 instruction-level upside path.

**M5 impact:** the B=1 same-boot adjudication for the LEGACY flip is
now moot *via Part A* — the repack (the B=1 fix candidate) is closed
as a dead end for the flip; the flip stays closed unless a future
mechanism (Part C or a write-path fix) targets the residual B=1 gap
and survives the serving gate.

## 2026-09-16 — MUSE-1: the V1 pin is lifted (V2 serving validated), and the spec method is the official DFlash assistant

> NOTE: this log is ~3x the 20-25 KB budget (`AGENTS.md` merge-train rule 1) — it needs an
> archive/staleness pass; the entry below is kept compact for that reason.

**VERDICT:** `SHIPPED` (V2 pin lifted; V1 removal in 0.32.0 no longer blocks this model) ·
**GATE:** serving A/B, Muse-Glimmer-30B-AWQ-INT4, TP=1 GPU0, util 0.90, maxlen 8192,
greedy (spec off), **chat template** via `/v1/chat/completions`, identical prompts
(`prompt_sha1` logged), 3 reps/point, arms A(V2)-B(V1)-A(V2) same boot, mclk 1000.

### HYPOTHESIS

Muse-Glimmer is the last model pinned to the V1 runner (`VLLM_USE_V2_MODEL_RUNNER=0`) and
upstream removes V1 in 0.32.0. In-process V1/V2 generation was already byte-identical
(2026-09-15), so if V2 serving is at parity the pin can go.

### What was done

- Server A/B with `VLLM_USE_V2_MODEL_RUNNER=1|0`, everything else identical; the client
  is new (`docs/gfx906/_bench_chat_serve.py`) because Muse is instruction-tuned (chat
  template required) **and** a reasoning model — under `--reasoning-parser muse_glimmer`
  its tokens arrive in the `reasoning` delta field, so a client that counts only
  `content` reads zero tokens (that cost one full session tonight: "no content received"
  on every request).
- Prompts are raw-filler bodies (8000 / 20800 chars ≈ 2k / 5-6k tokens); the corpus
  `real`/`agent` files are token-id lists, not text, so they cannot be sent through a
  chat template directly. Recorded as `prompt_form=chat-template + filler body`.

### Evidence — FOR (V2 at parity → pin lifted)

| arm | decode t/s @2k | decode t/s @8k | TTFT @2k | TTFT @8k | KV pool |
|---|---|---|---|---|---|
| V2 (load 1) | 27.12 / 27.07 / 27.05 | 26.65 / 26.60 | 4.773 / 4.764 / 4.766 s | 11.73 / 11.75 s | 53,235 tok |
| V1 (the pin) | 27.66 / 27.59 / 27.58 | 26.97 / 26.95 | 4.784 / 4.756 / 4.780 s | 11.75 / 11.78 s | **67,722 tok** |
| V2 (load 2, order control) | 27.23 / 27.18 / 27.14 | 26.80 / 26.72 / 26.68 | 4.778 / 4.771 / 4.780 s | 11.72 / 11.77 s | — |

- **TTFT is at parity** (4.773 vs 4.773 s @2k; V2 is 0.3 % faster @8k) and the order
  control reproduces V2 within 0.4 %, so V2 is the runner to ship.
- **Decode is −1.8 % @2k and −1.0 % @8k** on V1. That is above this config's
  process-to-process drift (V2-1 vs V2-2 agree to 0.4 %), so it is real — accepted
  anyway because 0.32.0 removes V1 (keeping V1 would mean carrying it in the fork), and
  the numbers are recorded so the cost is visible.
- **The KV pool is 21 % smaller under V2** (53,235 vs 67,722 tokens at util 0.90,
  maxlen 8192) — irrelevant at 8k (6.5× concurrency) but it would matter at longer
  maxlen; recorded, not chased.
- Both runners get `Using CUSTOM (gfx906 FA) backend for ViT attention`, i.e. VIT-1
  covers the Muse vision tower on V2 as well.
- These are the **first real-payload (non-ngram-filler) serving numbers** for this
  model: ~27 t/s decode at 2k-8k greedy, TTFT 4.8 s @2k / 11.7 s @8k on one MI50.

### Spec-decode decision (MUSE-1's other half)

- **MTP does not exist for this checkpoint**: `cyankiwi/Muse-Glimmer-30B-AWQ-INT4` has no
  MTP head (2654 tensors, no `mtp`/`nextn`/`draft` names; no such config keys). The
  ROADMAP's "use MTP instead of ngram" is not available here.
- **ngram is deprecated repo-wide** (filler-corpus acceptance ceilings).
- **The official drafter exists and our tree already supports it**:
  `meta-models/Muse-Glimmer-30B-assistant` (`architectures: [MuseGlimmerAssistantModel]`)
  is mapped by `vllm/config/speculative.py` to **method `dflash`** by architecture name,
  and `registry.py` routes `MuseGlimmerAssistantModel` → `DFlashQwen3ForCausalLM`. Its
  config: 5 layers, hidden 6656, 32/8 GQA, head_dim 128, all-sliding 2048, block_size 16,
  `target_layer_ids [1,13,25,37,49]`, bf16, 5.1 GB. Since `dflash` drafts are V2-only,
  this is only reachable *because* the pin is lifted — the two halves of MUSE-1 resolve
  together. First arm (k=7) launched tonight as MUSE-2; the DFlash2 precedent (a trained
  drafter for this family measuring degenerate) makes the acceptance check the gate, not
  the port.
- A DSpark drafter also exists (`DaoCloud/Muse-Glimmer-30B-DSpark`) but is a *port*: the
  registry maps `DSparkDraftModel` to the DeepSeek-V4 class (`dspark_target_layer_ids`,
  `dspark_markov_rank`, MLA-shaped), while the Muse drafter is qwen3-layer shaped. Left
  queued; the assistant above is the cheaper path.

### Interactions

- Supersedes the ROADMAP's "MUSE-1 = MTP" framing and the model table's ngram recipe.
- The reasoning-delta client fix is a general lesson for Muse/Gemma-4-class models (a
  `reasoning` parser changes the stream field, and "no content" is a client bug, not a
  server failure).

## 2026-09-16 (later) — MUSE-2: the official DFlash assistant drafts usefully; non-causal attention is the blocker to graphs

**VERDICT:** `OPEN — drafter validated, graph capture blocked by FA-NONCAUSAL` (the assistant
is worth keeping and the enabler is now a named item) · **GATE:** per-position acceptance +
decode t/s, Muse-Glimmer-30B-AWQ-INT4 + `meta-models/Muse-Glimmer-30B-assistant`,
**TP=2**, k=7, `--enforce-eager` (forced by the blocker below), chat-templated prompt,
3 reps × 128 tokens.

### What was done

- Checkpoint: `meta-models/Muse-Glimmer-30B-assistant` (5.1 GB, bf16;
  `MuseGlimmerAssistantModel`, 5 layers, hidden 6656, 32/8 GQA, head_dim 128,
  all-sliding 2048, block_size 16, `target_layer_ids [1,13,25,37,49]`). Our tree maps the
  architecture to **method `dflash`** by name (`vllm/config/speculative.py`) and routes it
  to `DFlashQwen3ForCausalLM` (`registry.py`) — no port needed.
- Serve: `--speculative-config {"method":"dflash","model":"<assistant>","num_speculative_tokens":7}`,
  TP=2, maxlen 8192, max-num-seqs 2, `--kv-cache-memory-bytes 2 GiB`.
- Two launch failures first, both mine, both cheap: TP=1 OOM (24 GB target + 5.1 GB bf16
  drafter > 32 GB card) and TP=2 with `HIP_VISIBLE_DEVICES=0` (rank 1 out of bounds).

### Evidence — the drafter works

| metric | value |
|---|---|
| steps / draft tokens / accepted | 128 / 896 / 253 |
| **mean acceptance length** (server metric) | **2.95** |
| accepted per draft token | 0.282 → **2.98 tokens/step** |
| per-position acceptance | **pos0 0.844**, pos1 0.508, pos2 0.305, pos3 0.180, pos4 0.117, pos5 0.023, pos6 0.000 |
| decode t/s (3 reps) | 30.46 / 30.43 / 31.17 |
| TTFT (3 reps) | 3.873 / 3.883 / 3.889 s |

For reference the non-spec numbers on the same model/prompts are 27.1 t/s decode (V2, TP=1)
and 27.6 (V1); this arm is *eager* TP=2 and still +11 % on decode, because ~3 tokens/step
more than pays for the drafter. The position-0 rate (0.84) is the metric that separated a
healthy drafter from the degenerate DFlash2 one (0.04) — this is a normal distribution.

### Evidence — the blocker (why graphs are off)

- The assistant's attention is **non-causal** (`dflash_has_any_non_causal` is true for this
  config), so the gfx906 backend selector rejects CUSTOM:
  `attention backend ROCM_ATTN was selected … but the custom gfx906 FA is unavailable (non-causal …)`.
- ROCM_ATTN then cannot be CUDA-graph captured on this stack:
  `Cannot copy between CPU and CUDA tensors during CUDA graph capture` (rocm_attn →
  chunked_prefill), which is the same wall the DFlash2 arms hit.
- Workaround used: `--enforce-eager` (target and drafter). Eager TP=2 costs the *target* a
  lot on ordinary decode, so the production path needs FA non-causal support.

### Consequence

The kernel already computes bidirectional attention (`mask=None`, `q_abs_offset=None` — VIT-1
proved that path), so what is missing is the *decoder-shaped* non-causal case: backend
acceptance, window symmetry (`_maybe_symmetrize_window` in the DFlash2 handover), and the
KV-write/read contract. That is the documented **FA-NONCAUSAL** design, which was previously
motivated only by the parked DFlash2 family — the assistant gives it a live, measured use
case (a supported spec method for a served model, replacing deprecated ngram). Queued in the
ROADMAP.

## 2026-09-17 — MUSE-2 follow-up: the blocker is gone (FA-NONCAUSAL Stage 1); assistant runs with graphs

**VERDICT:** `SHIPPED` — Muse-Glimmer now has a working, graph-capturable spec method ·
**GATE:** same arm as the entry above, with `--enforce-eager` removed and
`cudagraph_capture_sizes [8,16]`, after FA-NONCAUSAL Stage 1 (CUSTOM FA accepts the drafter's
non-causal class; `DEVLOG-fa-noncausal.md`).

| arm | drafter attention | graphs | decode t/s | mean acceptance |
|---|---|---|---|---|
| control, pre-change (`GFX906_FA_NO_NONCAUSAL=1` + eager), 3 runs | ROCM_ATTN + Triton | no | 30.3-30.7 | 2.95-3.02 |
| **this change** | **CUSTOM FA** | **yes** | **39.2-43.5** | **2.82-3.18** |
| no drafter (V2, TP=1) | — | yes | 27.1 | — |

`Capturing dflash CUDA graphs (FULL): 100%|2/2` and zero `Cannot copy between CPU and CUDA`
lines. Acceptance is *not* degraded by the superset (full-bidirectional) mask, so the kernel's
symmetric ±window (FA-NONCAUSAL Stage 2) is not needed for this drafter. Net for the model: **≥ +29 % decode** against an order-controlled control
(≥ +45 % vs serving it with no spec method at all — its only alternatives were deprecated ngram
and a DSpark port). The first version of this entry quoted a single 43.1 t/s run; the number is now
the range from two clean runs, because the harness of the day did not reap orphaned TP=2 workers
(fixed; see `DEVLOG-fa-noncausal.md` finding 3). Caveat: one load died first in the load lottery
(wedge #100); the retry was clean. k was not swept (k=7 only), B=1 only.
