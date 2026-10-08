# DEVLOG: MoE M=1 expert-kernel sprint (week of 2026-08-18)

Branch: `gfx906/moe-m1-sprint` (off `gfx906/main` @ 53f5a79429).
Plan: the 3-4 day set proposed in the former MoE roadmap. It covered
cheap expert-kernel wins, vectorized loads, gate/GEMV status, RMSNorm-gate
fusion, shared-expert GEMV dispatch, and GDN/FA follow-ups.

Stop rule (agreed before starting): if the day-1 cluster lands < 0.8 ms
combined, re-profile before continuing — the roadmap may be stale.

**GATE (all serving A/Bs in this log):** `_bench_gfx906.py`, pp=2048/
tg=256, 4 samples, same day, this build. Per-kernel census / standalone
harness results are evidence, not the gate — this log repeatedly shows
launch-regime wins failing to transfer (see the top-level takeaway).

**Top-level takeaway (reinforced three ways below):** standalone / census
kernel-cost wins do NOT reliably transfer to serving wall-clock on the
decode-size kernels; the mode-dependent outcome (eager wins, graph loses)
is the recurring pattern. A named serving A/B is the only trustworthy
gate for µs-scale kernels.

## Day 1 (2026-08-18): re-anchoring — the roadmap is stale

**Stop rule triggered immediately.** The roadmap's A1 premise ("w13 16.1 ms
+ w2 9.7 ms = 27.2 ms/step, 59% of the step") does not match the kernel:

- Micro-bench (`bench_moe_gemm_gfx906.py`, random, M=1, bm=1): **w13
  42.0 µs/call, w2 33.0 µs/call**. w13 M=1 traffic = 8 experts × 1 MB =
  8 MB → **~190 GB/s ≈ DRAM floor** (the harness `bw` column assumes the
  full 256-expert table; 8 MB/42 µs is the right number). w2: 8 × 0.5 MB /
  33 µs ≈ 121 GB/s (76% floor).
- ×40 layers ⇒ expert GEMMs ≈ **3.0 ms/step**, not 27.2 ms (roadmap
  numbers predate the Phase-3 NPT-split/single-stage prefetch/NC2=2/BM
  work).
- A1a is **phantom**: no `gate_k`/`gate_gather` kernel or "by = stride-1
  coalesced write" comment exists anywhere; the BM=1 prologue loads the
  token row 1D-coalesced into LDS.
- A2 is done: the [1,2048] shared_expert_gate converted in P3-1
  (DEVLOG-moe-opt.md); the router [256,2048] is already GEMV-eligible
  (`_llmm1_tiny_m` m==256). A2 = a profiler check, not a kernel.

So the first deliverable is a fresh in-model kernel table. New probe:
`benchmarks/kernels/gfx906/kernel_prof_probe.py` (the old
`fillprof_probe.py` was deleted in the 0defa92551 docs consolidation; the
reboot wiped /tmp/bench). Probe gotcha: the v1 engine runs in a separate
process → profiler sees zero kernels; needs `VLLM_ENABLE_V1_MULTIPROCESSING=0`
(the probe now refuses to run doctrinally without it).

### Fresh in-model table (eager, 256 decode steps, GPU-busy 17.59 ms/step)

Top decode rows (prefill rows excluded; µs/step):

| µs/step | cnt/step | µs/call | kernel |
|---|---|---|---|
| 3,775 | 80.4 | 47.0 | dense_gemv<2, 2048> |
| 2,149 | 77.4 | 27.8 | moe_gemm_q4<1, 4> (experts) |
| 2,095 | 148.8 | 14.1 | LLGemm1<Half, 4> (LLMM1) |
| 713 | 39.8 | 17.9 | topkGating<8, 256, ...> (router) |
| 586 | 79.7 | 7.4 | fused_add_rms_norm |
| 493 | 9.9 | 49.7 | flash_attn_tile_q8<256,256,2,8> (FA decode) |
| 414 | 29.8 | 13.9 | GDN packed_decode recurrent |
| 382 | 79.7 | 4.8 | act_and_mul (silu-mul) |
| 390+331 | ~80 | ~5-10 | elementwise/casts (unattributed) |
| 372 | 38.9 | 9.6 | moe_align_block_size |
| 308 | 9.96 | 30.9 | FA gather_paged_kv_q8 |
| 303 | 49.8 | 6.1 | sigmoid |
| 269 | 59.0 | 4.6 | copyBuffer |

Key readings:
- MoE expert kernel is **27.8 µs/call in-model** (vs 42+33 cold) — L2 is
  friendlier; expert GEMMs ≈ 2.15 ms/step, not 27.2.
- A1 ("59% of step") is dead; the real top-3 = dense_gemv 3.78 + experts
  2.15 + LLMM1 2.09 = 8.0 ms/step (46%).
- Gap to llama.cpp parity (70.3 t/s = 14.22 ms/step) from 67.39 t/s
  (14.84) is **~0.6 ms/step** — the sprint needs ~0.6 ms, not 5.

### Re-ranked sprint (post-profiling)

1. **S1** — attribute/optimize dense_gemv<2,2048> (3.78 ms; 47 µs/call far
   off floor; router [256,2048]=1 MB should be ~10-15 µs).
2. **S2** — router + topk pair (713 µs topkGating + gate's S1 share).
3. **S3** — LLMM1 2.09 ms bucket: attribute shapes, check GEMV-eligibility
   (K<=8192 n==1 gate), K=2048 joining dense_gemv dispatch.
4. **S4** — expert-kernel A1 leftovers (A1c zp L1, w2 121 GB/s → floor).
5. B1/B2 (GDN chunk, FA page) demoted to stretch (both smaller than the
   roadmap claimed).

### Shape spy + GEMV calibration → final re-ranking (S2 is the play)

Shape spy (`/tmp/shape_spy.py`, monkey-patched op counters, 32-step eager)
attributes every per-layer linear of the 35B. dense_gemv<2,2048> = router
[256,2048]×40 + GDN in_proj [12288,2048]×30 + FA qkv [9216,2048]×10 +
lm_head [248320,2048]×1; LLMM1 = out_proj [2048,4096]×40 + shared gate_up
[1024,2048]×40 + shared down [2048,512]×40 + GDN in_proj_ba [64,2048]×30.

Bandwidth correction: MI50 HBM ≈ 1 TB/s nominal, ~800 GB/s effective
floor (K=17408 GEMV: 178 MB/223.8 µs = 795 GB/s = "100.2% floor"). My
initial 160 GB/s was the consumer-Vega number (wrong by 5×). Recomputed,
the big GEMV shapes are AT FLOOR:

| shape (m×k) | x/step | floor µs | best µs/call | cfg | verdict |
|---|---|---|---|---|---|
| 12288×2048 GDN in_proj | 30 | 63.1 | 60.9 | GEMV kc2048 | 1.0× — no lever |
| 248320×2048 lm_head | 1 | 1274.6 | 1127.4 | GEMV kc2048 | 0.9× — no lever |
| 9216×2048 FA qkv | 10 | 47.3 | 46.8 | GEMV kc2048 | 1.0× — no lever |
| 2048×4096 o_proj | 40 | 21.0 | 21.4 | LLMM1 rpb4 | 1.0× — no lever |
| 1024×2048 shared gate_up | 40 | 5.3 | 7.3 | LLMM1 rpb4 | 1.4× — small lever |
| 2048×512 shared down | 40 | 2.6 | 6.7 | LLMM1 rpb4 | 2.5× — lever |
| 256×2048 router | 40 | 1.3 | 4.6 | GEMV kc2048/r2 | 3.5× — lever |
| 64×2048 GDN in_proj_ba | 30 | 0.3 | 4.6 | GEMV kc2048/r2 | 14× but launch-bound |

**Real top-2 non-floor costs:**
1. **topkGating 17.9 µs × 40 = 713 µs/step** — pure launch/shape; a fused
   router+topk (1 CTA, 256 thr, one 1 MB gate read, in-CTA top-8) should
   land ~3-5 µs → ~550-600 µs/step = the whole 0.6 ms parity gap.
2. **MoE expert M=1: 64 CTAs (8 rows × K/256), 32-iter latency chain** —
   cold 42 µs (190 GB/s of 800) vs in-model 27.8; a better-parallelized M=1
   expert GEMV worth ~0.5-1.0 ms/step (the A1 cluster re-diagnosed as
   occupancy, not load width).

**Final order:** (1) S2 fused router+topk; (2) S5 M=1 expert parallelism;
(3) S3 shared-down K=512 GEMV; (4) S4 shared gate_up; B1/B2 stretch.

## S2 — dedicated M=1 topk kernel (implementation + VERDICT: NEUTRAL)

**Generic topkGating semantics extracted** (topk_softmax_kernels.cu,
SCORING_SOFTMAX, no bias/padding): per-lane (VPT=8) `max()` (fmaxf,
NaN-ignoring) → width-32 xor(16..1) butterfly; `expf(x - row_max)` with
per-lane sum in expert order → width-32 xor butterfly sum;
`p = e * (1.f/row_sum)`; `isnan||isinf → 0.f`; 8× (local argmax strict `>`
i-ascending, xor-butterfly argmax with lowest-expert tie-break, blank
winner to -10000.f, `selected_sum += winner_p` in k order); renorm
`scale = 1.f/denom` (denom≤0→1.f), `out[k] *= scale`.

**Kernel** (`csrc/rocm/moe_topk_gfx906.cu`): one 64-lane CTA; lanes 32-63
exit immediately; the 32 active lanes each own 8 consecutive experts (the
generic's exact TPR=32/VPT=8 partition, one 16B load), reduced with the
generic's exact width-32 `VLLM_SHFL_XOR_SYNC_WIDTH` butterflies. (The
earlier 64-lane/-inf-dummy + LDS-scan draft was dropped — see the variant
table.)

**Bit-equality argument (why shipping default-OFF was safe), gist:**
with finite inputs the kernel is **instruction-identical to the generic**
on the only operations that affect the result — the max/sum width-32
xor-butterfly trees are reproduced operation-for-operation, and the
per-lane pieces (expf, local sum order, recip multiply, NaN clamp,
selected_sum k-order, renorm scale) are byte-identical ops on
byte-identical operands. The argmax is a total order (value desc, expert
asc) so any reduction order yields the same winner; blanking is local to
the owner lane. Hence bit-equal output, no divergence from the generic
path.

**gfx906 ISA finding:** every `__shfl_xor` pattern — width 32 or 64 —
lowers through the E32 extended-register file on gfx906 (width 32: 558
v_readlane + 357 v_writelane + 526 v_mov_b32_e32; width 64: 155
ds_bpermute + 183 v_mov_b32_e32); `v_perm_b32` only for constant-lane
shuffles. (See `latency-hiding.md` for the general shuffle guidance; this
is the shfl-specific measurement.)

| variant | GPU self-time | notes |
|---|---|---|
| generic topkGating<8,256,4,16,64> | 17.3–17.5 µs | 4-warp block, row machinery |
| width-32 cut (this file, SHIPPED) | **12.5 µs** | 1 wave, 32 active lanes, generic's exact shuffles |
| width-64 + 32 dummy -inf lanes | (rejected, unmeasured) | 2× per-lane work; shuffles no cheaper |
| LDS scans + forced s_barrier | 35.5 µs | 22 barriers + 8×32 ds_read2; 2.8× SLOWER |

The shuffle path wins despite the ugly lowering — one wave, no row
machinery. The LDS variant verified correct (3000-trial `topk_harness.cu`,
bit-equal to CPU incl. tie/sparse) but is dead weight; keep the harness.
Side finding: the backend elides `__syncthreads()` for single-wave CTAs
(per-wave LDS FIFO ordering) — that's why forced-barrier cost was isolated
as the 35.5 µs suspect.

**Dispatch** (fused_topk_router.py): `VLLM_GFX906_TOPK_M1` env, default
OFF; gates = on_gfx906 + M==1 + E==256 + k==8 + fp16 + int32 outputs +
contiguous + softmax + no bias + no padding.

**Build gotcha (this week):** the in-tree `.deps/triton_kernels-subbuild`
is root-owned (docker-era); workaround in `running.md`
(`FETCHCONTENT_BASE_DIR=/tmp/vllm-deps`); added
`TRITON_KERNELS_SRC_DIR=.../triton_kernels-src` to skip the ROCm/triton
clone after reboots.

### S2 result — mode-dependent, shipped default-OFF

Bit-equality gates: pytest `test_gfx906_m1_topk_bit_equal_to_generic` 4/4
(random×6 + all-equal + all-zero + 64-way tie, renorm on/off), micro-bench
bit-equal PASS.

Serving A/B (GATE):

| regime | generic (TOPK=0) | new kernel (TOPK=1) | Δ |
|---|---|---|---|
| isolated GPU self-time | 17.3 µs/call | 12.5 µs/call (64t) | −4.8 ✓ |
| eager serving | 23.98 t/s | 24.09 t/s | **+0.11** ✓ |
| CUDA-graph serving | 66.31 t/s | 65.32 / 65.38 | **−0.95** ✗ |

The same kernel gains in eager (CPU-gapped) and loses in graph replay
(gapless). Thread count (64 vs 256) changes nothing. The in-model
profiler is NOT reliable here (<~10 µs kernels: 256t read 19.9-20.1 µs/call
in-model vs 7.2 solo vs 1.8 interleaved-solo) — activity records depend on
surrounding pipeline state. Wall-clock A/B is the only gate for µs-scale
kernels.

Hypothesis (unverified): the width-32 E32-footprint is cheap when CPU
launch gaps let the wave set up lazily, but exposed in graph replay's
back-to-back dispatch. The LDS variant's true graph number is unknown.

**Decision: `VLLM_GFX906_TOPK_M1` default OFF** (opt-in eager-only);
kernel/op/binding/test/bench ship as the experimental record; generic
topkGating stays default. The 713 µs/step budget stays open — a real fix
must be fast in the gapless regime (candidate: fuse topk into the router
GEMV epilogue, roadmap S2').

## S5 — M=1 expert kernel re-tile: gemm2 shipped default-OFF  (VERDICT: NEUTRAL)

### Design (V2-B, lane-based columns)

`moe_gemm_q4_v2_kernel_gfx906<THREADS,NPT,SLICE>`: one 512-thread CTA (8
waves) covers `64*NPT` output columns; columns **lane-based** (`n =
offset_n + tl*NPT`) so each wave owns a disjoint K-slice (`offset_k =
w*size_k/8`); per-wave fp32 partials `[wave][64][NPT]` reduce through LDS
(two `__syncthreads`); **only wave 0 runs the epilogue** — direct store
(gemm1) or packed CAS into the pre-zeroed token row (gemm2, all 8 slot
x-blocks share the row).

Design bug caught in harness: with lane-based columns all 8 waves hold the
*same* value for the *same* cells; if every wave CAS'd the value would be
added 8× (measured exactly 8.02× single-slot). Direct stores hide it
(idempotent) — why gemm1 looked correct while gemm2 blew up. Wave-0-only
epilogue fixes both.

### Harness bugs (moe_m1_harness.cu — kept fixes, details live in the harness)

1. `uint2` weight load only read 2 of 4 NPT=4 cols (`b_w[j][2..3]` stale) →
   need `uint4`.
2. Direct store dropped r[2..3] for NPT=4 — the "gemm1 512t4col max err
   0.2511" pass was a **stale-buffer artifact**; fixed the store.
3. Test bookkeeping: a `d_npp=8` leftover (deterministic 14.6 "error") and
   a 2-D-grid launch (missing grid.z → 12.1 partial sum).

### Machine fact: intra-wave CAS contention is pathological

Standalone probes with ≥2 lanes of one wavefront CAS'ing the *same*
address lose updates (deterministic saturation at 2^11 for 32-/64-bit CAS)
and can raise `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (rocprofv3 7.14).
Hardware `atomicAdd`, plain stores, single-thread CAS loops, and the
production regime (per-lane distinct cells; ≤16-CTA fan-in per cell) are
all fine. **Rule: never let multiple lanes of a wavefront CAS the same
address; keep per-lane distinct cells.** (Also `global atomic_cmpswap_x2`
retry logic updates `old := packed+old` not the CAS return — harmless in
production contention, don't rely on it under heavy same-address spin.)

### Standalone A/B (harness, M=1, 35B shapes)

| K (gemm) | config | µs/launch | vs current |
|---|---|---|---|
| 2048 (gemm1) | current <1,4> | 32.00 | — |
| 2048 | v2 <512,2,256> | 27.11 | 1.18× |
| 2048 | v2 <512,4,256> | 28.14 | 1.14× |
| 512 (gemm2) | current <1,4> | 21.36 | — |
| 512 | **v2 <512,4,256>** | **10.80** | **1.98×** |
| 512 | v2 <512,2,256> | 11.39 | 1.88× |

Correctness: current vs cpu-ref 0.25 (dequant noise); v2 vs current 0.02
(gemm1) / 0.65 (gemm2, fp16 CAS-chain); single-slot npp=1 0.21/0.10;
production gemm2 convention (top_k=1, per-slot a-rows, 8× CAS) all three
kernels agree within 0.29/0.73 abs.

### Production integration + in-model

Dispatch: `block_size_m == 1`, `output_topk > 0` (gemm2) and
`size_m == output_topk` (EM==topk ⟺ M==1), behind `VLLM_GFX906_MOE_M1`
(default OFF). Launcher guard `size_n%256==0 && size_k%256==0 &&
size_k<=2048 && groupsize%32==0` (R1: a non-256-multiple size_k gives
slice_k<32 → reads past end_k; non-32 groupsize skips group refreshes).
The env flag only affects launches after the flip — no effect on an
already-captured graph. Caller zeroing unchanged (gemm1 zeroing now
redundant but harmless; gemm2 zeroing required for CAS).

In-model (kernel_prof_probe, direction only): gemm2 26.8 → 22.3 (−17%);
gemm1 v2 <512,2,256> 27.5 vs 26.8 (neutral — standalone 1.18× didn't
transfer; kept <1,4>).

Wall-clock A/B (GATE):

| regime | OFF | ON (gemm2 v2) | Δ |
|---|---|---|---|
| CUDA-graph serving | 66.43 t/s | **67.03 t/s** | **+0.60 (+0.9%)** |
| eager serving | 23.50 t/s | **23.96 t/s** | **+0.46 (+2.0%)** |
| greedy (12×128) | `868ad09e…` | `868ad09e…` | identical |

Both-V2 (gemm1+gemm2): graph 66.81 / eager 24.00 — worse/equal both,
confirming the gemm1 re-tile isn't worth keeping.

**Decision: ship gemm2-only M=1 re-tile, `VLLM_GFX906_MOE_M1` default
OFF** (opt-in; positive and token-identical in both modes). The gemm1
budget (~1070 µs/step) stays open: its premise (standalone 1.18×) doesn't
transfer; next idea = 256-thread 4-col tile or folding gemm1 into the
activation epilogue — roadmap C2 follow-up.

## S3 — shared-expert down_proj GEMV (shipped default-ON)  (VERDICT: SHIPPED, PROVISIONAL)

The re-anchor table flagged shared down [2048,512] ×40/step at 2.5× floor
(6.7 µs LLMM1 rpb4 vs 2.6) and shared gate_up [1024,2048] at 1.4×. The
dense GEMV already supported both shapes; S3 = a dispatch extension + one
RPT rule.

**Micro-bench (`bench_dense_gemv_gfx906.py`):**

| shape | LLMM1 rpb4 | best GEMV | verdict |
|---|---|---|---|
| shared down [2048,512] | 6.7-7.7 µs | **kc512/r2 5.6-5.7 µs** | ship (−15..25%) |
| shared gate_up [1024,2048] | 7.3 µs | kc2048/r4 8.0 µs (r2: 46.4!) | **no lever — stays LLMM1** |

The gate_up lever doesn't exist: GEMV is 8% slower at RPT=4 (default) and
6× slower at RPT=2. The down win is real but the 2.5×-floor gap doesn't
materialize: the winning GEMV is itself 2.2× floor.

**Integration:** `_llmm1_tiny_m`: K=512, m=2048 → `dense_gemv_gfx906(w,x,512)`
under the existing `VLLM_GFX906_DENSE_GEMV` gate + kill switch
`VLLM_GFX906_DOWN_GEMV=0`. Kernel default RPT: `kchunk==512 && N==2048 →
RPT=2` (measured; RPT=4 old fallback at 8.0-8.2 µs).

**Gates (2026-08-18; NOTE: thermal noise — fans not ideal, serving numbers
directional only):**
- standalone: GEMV kc512/auto(=r2) 5.64 vs LLMM1 6.71-7.68 µs (first-proc
  "auto" ~6.1 µs = GPU warmup bias, not config).
- in-model (kernel_prof_probe): dense_gemv_kernel<2,512> 39.7/step at
  8.4 µs/call (L2-cold inflation, same as S5); GPU-busy 17593→17523 µs/step (−70).
- PPL (`ppl_probe.py`, recreated 12-prompt, 359 tok, 0 top-20 misses; — values
  NOT comparable to the lost 6.69-era probe): OFF 16.009±0.01 vs ON
  15.993±0.01 = −0.16%, inside the 2% bar.
- greedy 12×128: **token-identical** with GEMV on (standalone 0.031 maxdiff
  stays below the argmax margin).
- serving graph (2 runs × 4 samples): 65.88/65.87 (OFF) vs 66.07/65.98 (ON)
  = +0.19/+0.11 t/s, all positive; expect cleaner re-measure with thermals.

**Decision: default ON — PROVISIONAL** (kill switch
`VLLM_GFX906_DOWN_GEMV=0`). ~40 µs/step expected (≈0.25% of the 17.5 ms
GPU step) — the smallest lever left. Provisional (post-sprint review R4)
because: this is the branch's only default-ON numerics change, not bit-equal
to LLMM1, and rests on a thermally-flagged serving A/B. Uncontaminated
evidence: micro-bench −15..25%, in-model −70 µs/step, greedy token-identical,
PPL −0.16%, four positive (noisy) serving samples. **Reopen conditions:** a
clean serving A/B (fans confirmed, both directions, 2 runs each) going
negative, or a greedy-margin analysis showing the 0.031 maxdiff isn't below
the smallest in-model winning-logit margin. **House rule (applied
retroactively): a default-ON numerics change requires a clean wall-clock
A/B — a per-kernel number may motivate, not justify, flipping a default.**

## Post-sprint code review fixes (2026-08-19)

Two independent reviews (records deleted after fixing) → six findings, five
commits:
- **R1+R2** (`979e72c925`): v2 M=1 launcher guard now
  `size_n%256==0 && size_k%256==0 && size_k<=2048 && groupsize%32==0`; dropped
  the dead gemm1 dispatch branch; noted env flag doesn't affect captured
  graphs.
- **R5** (`b7ec0306ee`): this log's S2 section rewritten to the shipped
  width-32 kernel (was the dropped draft); dispatch env default correction.
- **R4** (`e06f484c0e`): S3's default-ON marked provisional with reopen
  conditions (see S3).
- **R3/R6** (`b3e7139aaa`): harnesses moved to `benchmarks/kernels/gfx906/harness/`.
- **R5-hardening** (`8e7935edd4`): S2 docstring pins bit-equality
  assumptions; Python gate adds `gating_output.shape[0]==1`; new e2e dispatch
  test (M=1 bit-equal, M>1 guarded off).

Not done (deliberately): S5's `diff < 0.3·max+0.05` gate stays — S5 is
default-OFF; a stronger high-precision check only before a future
default-ON flip. S3 clean re-A/B deferred until thermals (provisional flag
is the record).

Verification: incremental rebuild + `test_gfx906_moe_gemm.py` (25 passed) +
`test_fused_topk.py -k gfx906` (6 passed).