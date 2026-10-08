# Dev log — Qwen3.5-27B dense decode on gfx906

> Split from DEVLOG-moe-opt.md (2026-08, topic consolidation). The
> dense 27B handover, elementwise/rmsnorm, W4A16-dense-rejected,
> down_proj GEMV, max-ilp, and the docker load test.

**VERDICT (top-level):** dense decode 18.89 → 25.60 t/s serving
(+35%); several proposed levers were measured and rejected (W4A16
dense kernel, K=5120 GEMV — LLMM1 is already at the HBM floor).
Individually labelled inline.

## Dense Qwen3.5-27B handover (2026-08-16)

The other agent's dense-model session handed over mid-flight; companion
doc `qwen35_dense_opt.md`. Dense 27B now *runs* in serving mode (it OOMed
on every attempt at session start). Its two FA fixes (NC2 GQA fail-closed
guard + kv_split prefill clamp) are commit `b4873459f8`. This section
tracks the profiling/baseline/improvement work taken over.

### FA fixes on this build: MoE gates

- PPL (MoE model, prompt logprobs = prefill path): 6.6942 / 6.6918 (two
  draws, build `b4873459f8`) vs 6.6889 on the previous build. The
  difference (+0.003..+0.005) is explained: the probe measures the
  **prefill** path, and `b4873459f8` changed prefill FA from kv_split=16
  to kv_split=1 (same math, different float summation order — ulp-level
  reordering, same class as the accepted P3-4 decode shift +0.0027).
  Decode path is bit-identical for the MoE model (ratio 8 % 8 == 0 →
  NC2=8 unchanged). Two same-build draws agree within 0.0024 (the known
  ~0.003 probe noise). **Gate: PASS.**
- Serving A/B (2 samples vs the 63.2–64.1 distribution): pending.

### rocprofv3 is unusable for full-model dense runs on this box (rocprofv2 gone)

rocprofv2 no longer ships in ROCm 7.14 (`/opt/rocm-7.14/bin` has only
rocprofv3/rocprof-sys-*). rocprofv3 1.3.2's *exit finalization* is broken
for full-model runs in every mode tried:

| mode | result |
|---|---|
| spawn + `-f csv` (their prof1) | CSV opened at finalization, `ring_buffer: munmap EINVAL` → CHECK → abort → 0-byte CSV |
| in-proc + `-f csv` (their prof2) | same CHECK at natural exit after the bench completed |
| spawn + SQLite (their prof3) | EngineCore's shutdown TERM beat its own clean-exit flush; signal-finalization wrote all info tables but **0 kernel dispatches** |
| in-proc + SQLite + external TERM (prof4) | rocprofv3 `exec`s into the target (the `$!` is the python itself; `pgrep -P` found no child); TERM never sent; **natural exit hit the same ring-buffer CHECK — and the process survived the abort** (torch's chained SIGABRT handler swallowed it), **stuck holding 28 GiB VRAM** until `kill -9` |

The MoE `trace_fa` run (complete 275 MB DB) succeeded because the
EngineCore's clean-exit flush finished *before* the parent's shutdown
TERM arrived — a race the dense runs lose (faster parent shutdown).
The ring-buffer CHECK itself is rocprofv3 tearing down a kernel-dispatch
ring the HSA runtime already unmapped during process exit.

**Pitfall: a crashed profile finalization can leave the python process
alive holding all VRAM** (state S, ~90 threads) — every subsequent
bench OOMs with "Free memory 3.26/31.98 GiB". Check VRAM after any
profiled run; `kill -9` the straggler if needed.

**Fallback (used):** in-proc eager torch.profiler decode-window budget
(`dense_budget_probe.py`, the devlog-documented fallback; same kernels as
cudagraph decode, launch overhead differs). Kernel times transfer.

### Dense model facts (Qwen3.5-27B-AWQ, runs at dtype=float16)

- 64 layers = 48 GDN + 16 FA (every 4th), hidden 5120, inter 17408.
- FA: Hq=24/Hkv=4 (ratio **6**), D=256, **attn_output_gate=True** →
  fused `qkv_proj` = [24×2×256 + 2×4×256, 5120] = **[14336, 5120]** fp16.
- AWQ `modules_to_not_convert`: visual, `linear_attn.in_proj_a/b`,
  `self_attn.q/k/v_proj`, `model.layers.0.*`, mtp. lm_head is fp16
  [248320, 5120] (2.37 GiB). GDN `in_proj_a` [2048, 5120] /
  `in_proj_b` [6144, 5120] fp16; GDN in_proj_qkv/z + out_proj +
  gate/up/down + o_proj are AWQ (exllama gptq path).
- **fp16 GEMV floor (M=1, 798 GB/s): lm_head 3193 µs + qkv×16 2944 µs +
  in_proj_b×48 3787 µs + in_proj_a×48 1253 µs ≈ 11.2 ms/step** — vs a
  ~39.5 ms/step total at 25.3 t/s decode. All of it currently runs
  LLMM1 (the GEMV gate hardcodes K==2048). Extending `dense_gemv` to
  K=5120 is the top lever (handover §6b#3 estimated 0.2–0.5 ms/step;
  the floor math says the real room is larger — the budget will tell).
- K=5120 options (MI50 max workgroup = 256 threads → KCHUNK ≤ 2048):
  kchunk=512 (KSPLIT=10, existing) or **kchunk=1024 (KSPLIT=5, added)**;
  2048/4096 don't divide 5120. KSPLIT>1 uses the fp16-CAS accumulation
  path (P3-2(b): "supported but no model shape used it" — this will be
  its first production use → dense greedy/PPL gate mandatory).

### Resolved (same day, handover-takeover session)

**Budget (in-proc eager torch.profiler, 96-step decode window, Sk≈2050):**
per-step kernel time (µs/step; the probe's raw µs were off by 1e3 due to
a unit bug in my aggregation — cross-validated against LLMM1/gptq
micro-benches and the eager wall time, which the corrected numbers
reproduce):

| bucket | µs/step | note |
|---|---|---|
| gptq W4 GEMM (exllama `gemm_half_q_half_gptq_4bit<1>`) | ~19,500–24,400 | 64×(gate+up+down) + 48×GDN-out; standalone probe: 87/102/39 µs per 44.6/44.6/15.6 MB call = 144–198% of HBM floor |
| LLGemm1 (fp16: in_proj a/b/qkv, qkv, o) | ~7,100 | **already 98–114% of floor — no lever** |
| elementwise/reduce pile (vec_elem 753 + reduce 159 + unroll ~300 + act_and_mul 63 calls/step) | ~9,000–11,000 | launch-latency class (P3-4 style); next attribution target |
| FA decode (tile 1.32 + gather 0.72 + combine 0.30 ms) | ~2,300 | NC2=1 fallback for ratio 6 |
| GDN recurrent + conv1d | ~1,100 | 48 layers |
| triton_matmul (1/step, ~8 µs) | ~740 | small-N path, not lm_head |

**Gates (all PASS):**
- MoE PPL (b4873459f8): 6.6942/6.6918 vs 6.6889 — the prefill kv_split
  16→1 ulp reorder (documented above).
- MoE serving: **65.361/65.392 t/s — new record** (prev 64.08); includes
  the GDN empty-out default flip (MoE 30 GDN layers ≈ 90 µs/step) —
  comfortably above the 63.2–64.1 distribution.
- Dense FA (CUSTOM vs Triton reference, 12 prompts × 128 greedy):
  10/12 sequences identical; the 2 divergences are exact/near ties —
  logprob margins 0.000 (perfect tie "entered"/"walked") and 0.016–0.063
  at the flip position. PPL 6.7197 (CUSTOM) vs 6.7036 (Triton) is inside
  the dense probe's ~2% run-to-run band (4 draws: 6.7000×3, 6.7197).
- GDN empty-out flip (default 0→1): the C config (empty=1) is
  deterministic (rerun 0/12 greedy divergence); the old config
  (empty=0) itself varies 2/12 between identical runs — the fill is
  load-bearing at the noise level on the old path, so greedy identity
  was never a valid gate here (same as MoE). PPL in band.

**Serving A/B (dense 27B, tg256, 2 samples each):**

| stack | t/s | decode-only t/s |
|---|---|---|
| current (CUSTOM FA, NC2=1 fallback) | 23.15 / 23.12 | 28.10 (35.6 ms/step) |
| baseline (Triton FA + GEMV off) | 18.89 / 18.89 | 22.55 (44.4 ms/step) |
| eager current / eager base | 15.74 / 15.77 | — |

Custom FA = **+22.5% serving / −19.7% decode step time** on dense; the
eager A/B is a tie because eager decode is launch-bound (both backends
leave comparable idle bubbles; graph mode collapses them).

**NC2=2 (new instantiation) for ratio-6 GQA:** micro-bench at the dense
shape (24/4/256), NC2=1 vs NC2=2, KVSPLIT=16: Sk=2048 102.8→76.1 µs
(−26%), Sk=3328 132.1→101.2 (−23%), maxerr 0.0038 (same as NC2=1).
25% off the FA tile kernel; the gather (per-Q-tile KV reads) halves the
same way → ~0.6 ms/step total. Default nc2=8 now auto-downgrades
8→2 (ratio%2==0) or 8→1 (MHA ratio 1, preserves pre-b4873459f8
behavior); an explicit GFX906_FA_NC2=2 that is GQA-invalid is an error.
NC2=2 is the largest valid packing for ratio 6 (divisors of 6 that are
powers of 2: 1, 2).

**GEMV K=5120: NEGATIVE.** `dense_gemv` (new KCHUNK=1024 path,
KSPLIT=5) vs LLMM1 rpb4 at all dense fp16 shapes: LM head
[248320,5120] 3114 vs 3128 µs (tie, both ~98% of floor); FA qkv
[14336,5120] 194 vs 184 (LLMM1 wins); in_proj_b 90 vs 81; in_proj_a
38 vs 28; kv rows 23 vs 15. LLMM1 is at the HBM floor for every dense
shape — the "lm_head GEMV" lever from the handover estimate does not
exist. KCHUNK=1024 stays as a bench-only path (also documents the MI50
256-thread workgroup limit for KCHUNK=4096).

**W4A16 dense: moe-kernel reuse = ~3%, rejected; new kernel = roadmap.**
Cross-kernel probe (`/tmp/bench/dense_qgemm_probe.py`, synthetic AWQ-
layout weights): `moe_gptq_gemm_gfx906` with a 1-expert identity-routing
view is only 3–8% faster than the exllama gptq kernel on the dense
shapes (its tuning targets the MoE N/K). A purpose-built W4A16 dense
GEMV (fp16-GEMV structure, int4 rows) could plausibly reach ~80–90% of
floor → ~6 ms/step off the 19.5 ms bucket — the top remaining dense
lever. The (probe) kernels' outputs disagreed on synthetic data
(maxdiff ~2–4) — layout subtlety, moot unless the new kernel lands.

**Remaining dense work (priority):** (1) land NC2=2 + GDN flip (this
commit); (2) attribute + cut the elementwise/reduce pile (~9–11
ms/step, 753+159+300 small kernels/step); (3) W4A16 dense kernel
(roadmap); (4) hybrid-KV auto-sizing over-commit note (~0.7 GiB).

## Elementwise attribution + production-path recharacterization (dense)

**EWP probe (eager, in-proc torch.profiler, 96-step decode window, NC2=2
installed):** the eager elementwise/reduce pile breaks down as —
aten::copy_ 4.73 ms/step (944 calls, [1,5120]/[5120] shapes, 5.0 µs each =
launch-latency-bound), elementwise vec<4> ~5.3 ms, reduce/mean 1.87 ms
(159/step), manual_unroll 1.38 ms (191/step), mul 1.80 + add 1.64 ms
(782 calls), silu_and_mul 0.65 ms (63/step). The mean/rsqrt/pow/mul
quadruple (~159/step) is the unfused RMSNorm decomposition (see below).

**GemmaRMSNorm was dispatching the torch decomposition on CUDA/ROCm.**
`forward_native` builds `weight.float() + 1.0` (fp32), which fails the
vllm_c fused impl's `weight.dtype == x.dtype` supports_args check → both
`rms_norm` and `fused_add_rms_norm` fell back to native (8–13 kernels per
call). Fixed `forward_cuda` to dispatch with `weight.to(x.dtype) + 1.0`
(a plain scaled RMS norm with w' = 1+w) → fused `vllm::rms_norm_kernel` /
`fused_add_rms_norm` selected. Unit: maxdiff 0.002 (fp16 rounding),
residual exact; 8–13 kernels → 2 per call. PPL: dense 6.6993 (band
6.7000–6.7197), MoE 6.6817 (band 6.6889–6.6942, probe noise).

**Production-path recharacterization (important):** the serving stack is
`VLLM_COMPILE (mode 3) + inductor + FULL_DECODE_ONLY` graphs. In that
mode (a) the platform IR priority is `['native']` for both norm ops, so
inductor fuses the norm decompositions into codegen kernels — the
GemmaRMSNorm fix is an eager-path improvement (enforce_eager users +
graph-fallback steps), not a production one; (b) the eager elementwise
pile is largely fused by inductor in production. Torch-profiler on the
production path only sees the ~5–6 eager-fallback steps per 256
(92 FA-tile calls, 1165 gptq calls ≈ 4.6/step) — graph-internal kernels
are not per-kernel-attributed, so its 862 µs/step "total" is NOT the
production kernel budget.

**Three-anchor inference for the production decode budget** (all
consistent): eager budget 47 ms/step (kernel, under-profiler) ≈
41 ms true; production wall 34.9 ms/step (28.69 t/s decode-only);
eager − fused-pile (~8.7 ms) + codegen (~1.7 ms) ≈ 34 ms ≈ wall at
95–99% busy. So production decode IS ~97% GPU-busy at ~34 ms/step, and
the eager kernel map is the right optimization map:

| bucket (eager, true µs/step) | prod est. | lever |
|---|---|---|
| gptq W4 (exllama) | ~19–20 ms (56%) | W4A16 dense kernel (~6 ms) |
| LLGemm1 fp16 | ~6–7 ms (18%) | at floor, none |
| elementwise residual after inductor fusion | ~2–4 ms | copy pile, per-site |
| FA tile+gather (NC2=2) | ~1.7 ms | done |
| GDN recurrent+conv | ~1.1 ms | small |
| down_proj triton_matmul [5120,17408] | ~0.8 ms | GEMV kc1024 rpt2 (224 µs, 100.2% floor) → −0.57 ms |

Combined remaining levers ≈ 7.5–8.5 ms/step → dense decode-only
28.7 → ~37 t/s if all land.

**down_proj GEMV K=17408:** the dense 27B ships the MLP down proj fp16
(2.0 GB, [5120, 17408]); it hits `rocm_unquantized_gemm` →
`triton_matmul` (794 µs/step; the `n==1 and k<=8192` GEMV gate excludes
K=17408). LLMM1 crashes at K=17408 (NUM_THREADS = K*2/16 = 2176 > MI50's
256-thread workgroup limit; K=5120 → 640 works empirically, i.e. the
effective LLMM1 K ceiling is ~2048). `dense_gemv` kc=1024 KSPLIT=17
rpt2: 223.8 µs = **100.2% of HBM floor** (err 0.0016). rpt2 beats rpt4
here (259.7, 116%) — the auto-RPT (N%4→4) is suboptimal for mid-N
shapes; a per-shape rpt override is needed.

**rocprofv3 dense trace: FAILED, not pursued.** The MoE trace
(719850_results.db) succeeds because the agent's SIGTERM handler writes
the DB (incl. the 2.6 s ring-buffer read) before chaining to vLLM's
handler, keeping HSA alive. Dense EngineCore consistently dies during
finalization: first attempt hit `ring_buffer: munmap failed: Invalid
argument` at the dispatch-read (buffer already invalid before the read);
subsequent runs die before "writing SQL database". No in-run HSA error
found. Not needed: the three-anchor inference above is anchored to
(eager budget, production wall, NC2=2 A/B delta mapping 1:1 to wall).

**Remaining dense work (priority):** (1) W4A16 dense kernel (~6
ms/step, top lever) — bundle with the down_proj GEMV gate extension
(both need a `_rocm_C` rebuild); (2) copy-pile call-site attribution
(eager stacks) for the ~1–2 ms residual; (3) hybrid-KV over-commit note.

### W4A16 dense kernel: correctness RESOLVED, lever REJECTED (2026-08-17)

**Correction to earlier entries:** the "ROCm compiler codegen bug"
narrative (kernarg-pointer multiplier in `v_mad_i64_i32`, all kernel
variants corrupted) was **false**. It was two bugs in my test harness,
both invisible to the uniform random-ish test data:

1. The pybind signature is `(x, qw, scales, qzeros, rpt, kchunk)`;
   my probe/bench scripts passed scales and qzeros **swapped**. The
   kernel read an int32 zero-tensor as fp16 scales → output exactly
   0.0.
2. The dequant reference flattened `[k8, n8, i]` without the
   `permute(0, 2, 1)` before `.reshape(K, N)` → the reference itself
   was a scrambled (transposed) matrix, so even a perfect kernel
   "failed" against it.

With the correct call order and a correct reference, the original
kernel (as of the very first build) passes **18/18** correctness
configs (K=5120/6144, N=128..5120, rpt=8/16/32; maxabs 1.7-2.8 = the
expected z1×s fp16 pre-rounding, same as the production MoE kernel).
The kernel reads q in the **exllama shuffle** order (bit
[0,16,4,20,8,24,12,28]) and qzeros in **sequential** order
(bit 4·i) — matching `gptq_shuffle_awq_qweight`'s output. Verified
with targeted bit-layout probes (all-9 q → 98.0; all-8 z →
−576×z_c).

**Performance: REJECTED.** exllama's tuned gptq_gemm (the one the
MoE path already uses via the repacked AWQ format) beats my
purpose-built kernel on every dense shape:

| shape (N×K) | exllama | W4A16 (best rpt) | floor |
|---|---|---|---|
| 10240×5120 (gate_up, ×2/layer) | 87 µs | 209 µs (rpt16, 187%) | 111.7 µs |
| 16384×5120 (gdn_in) | — | 120.9 µs (rpt32, 230%) | 52.7 µs |
| 6144×5120 (gdn_out) | 39 µs | 43.4 µs (r16, 220%) | 19.8 µs |

Extrapolated: mine is ~2.5 ms/step **slower** than exllama across the
model. The exllama kernel's vectorized load layout wins on skinny
GEMV shapes; my 256-thread rpt-rows × kchunk-k layout does not get
close (187-230% of floor vs exllama 87-97%). **W4A16 stays rejected;
exllama gptq remains the dense W4 path.** Prototype lives in
`/tmp/bench/w4a16/` (not in-tree) for the record.

### 2026-08-15 — M=1 W16A16 GEMV (`dense_gemv_gfx906.cu`, P3-2b): K-split refuted, RPT=2 single-pass LANDED

*(moved 2026-09-13 from DEVLOG-fa-attention.md — same kernel as the entries below)*

Built for the small-row pile (gate_up/down/router/GDN-small, ~150 calls/step,
3.6–14× floor, launch/latency-bound). Op `_rocm_C.dense_gemv_gfx906(weight[N,K]
fp16, x[1,K], kchunk)`: row-parallel LLGemm1-style, `__ockl_fdot2`, fp32 acc,
templated `RPT×KCHUNK`; K-split >1 accumulates packed fp16 CAS (pre-zeroed out).

- **v1 (K-split) hypothesis WRONG:** kc=512 splits are 2.4–4.2× *slower* than
  LLMM1 on every K=2048 shape (o_proj 3.5×) — fp16-CAS + `zero_` + tiny-block
  latency dominates at M=1. Small rows are launch/latency-bound, not
  CU-occupancy-bound, so a better GEMM cannot close them.
- **v2 (RPT=2, kc=4096 single pass) won:** qkv 9216×2048 **−23 %**, router
  256×2048 **−17 %**, in_proj −6 %, LM head −6 % (0.9× floor); o_proj / shared
  gate_up / down keep LLMM1 (RPT=2 is pathological at N=1024). **Weighted step
  5203 vs 5604 µs = −401 µs/step (−7.2 %)** — the plan's 0.7–1.1 ms estimate
  missed because of the launch-bound reframing. Shape rule
  `K==2048 ∧ (N==256 ∨ N≥2048)` through the `_llmm1_tiny_m` choke point; kill
  switch `VLLM_GFX906_DENSE_GEMV=0`.
- v1 review bugs caught pre-landing: 64-thread K-split CAS overcount (4 lanes
  CAS the same 8 B → guard `t==0`), 256-thread OOB LDS read in the sibling-row
  epilogue, host accepting `N%4!=0` with K-split.
- **Build gotcha:** a missing `\` on an interior macro line silently truncated
  `LAUNCH_BY_RPT`; `clang -E | grep` on the file alone found it — use that for
  macro surgery, not a full rebuild.

### down_proj GEMV (K=17408) LANDED (2026-08-17)

- `csrc/rocm/dense_gemv_gfx906.cu`: auto-RPT rule now
  `kchunk==1024 → rpt = (N%2==0) ? 2 : 1` (auto-RPT N%4→4 is
  suboptimal at K=17408: rpt4 259.7 µs vs rpt2 223.8 µs).
- `vllm/model_executor/layers/utils.py`: new `_gfx906_gemv_long_k()`
  gate (gfx906 + fp16 + contiguous + [5120,17408]) invoked from both
  skinny and general dispatch branches when k>8192; falls back to
  triton_matmul otherwise. Kept separate from `_llmm1_tiny_m` so the
  GEMV kill switch (`VLLM_GFX906_DENSE_GEMV=0`) cannot drag the
  K=17408 case into LLMM1 (which crashes there: 2176 threads).
- Fresh `_rocm_C` rebuild (local; see build note below).

**Numbers:**

| gate | result |
|---|---|
| kernel micro | 227.6 µs vs triton 795.4 µs (3.5×); 101% of HBM floor |
| dispatch n=1 | GEMV (maxabs 0.75 = expected fp16 atomic K-split accumulation, 17 chunks); n=4/kill-switch → triton (0.125) |
| PPL dense | 6.7153 (band 6.6993–6.7197) |
| PPL MoE | 6.6895 (band 6.6817–6.6942) |
| serving dense | **23.845 / 23.801 t/s** (record; was 23.553/23.524, +1.3%) |
| serving MoE | 66.36 t/s (no regression; record 65.36) |
| pytest `test_rocm_unquantized_gemm.py` | identical failure set before/after (9 pre-existing env failures: platform-mock tests assume a non-gfx906 host — `on_gfx906` is not monkeypatched) |

The fp16 atomic accumulation adds ~0.75 maxabs on ±130-scale outputs
(~0.6%) — a reordering-class change per the kernel header; PPL
confirmed no measurable shift.

### Build + hardware notes (2026-08-17)

- **Local `_rocm_C` rebuild works** with the `FETCHCONTENT_BASE_DIR`
  env override (setup.py honours it): the repo's default
  `.deps/triton_kernels-*` dirs are **root-owned** (from docker builds)
  and unwritable; `FETCHCONTENT_BASE_DIR=/tmp/vllm-deps` sidesteps it.
  Warm `build/` tree → only the changed .cu recompiles (~3 min).
- **MI50 HBM:** 138 correctable UMC (ECC) errors, stable across long
  bench runs (not climbing); one transient 0x77077777 readback
  observed during the W4A16 bisect, 40/40 clean reads after. Watch
  item, not active corruption.
- Recurring glibc heap crashes at **teardown** of probe processes
  ("corrupted double-linked list", "malloc_consolidate(): unaligned
  fastbin chunk") *after* clean test output — host heap quirk of the
  torch/HIP runtime on this box, not kernel-related.

### TODO (user-requested, 2026-08-17)

- [x] **Measure the impact of `CMAKE_HIP_FLAGS='-mllvm -amdgpu-sched-strategy=max-ilp'`** — RESOLVED: measured; adopted per-file (see "max-ilp scheduler strategy: measured, adopted per-file" below).

### FA decode per-layer copy pile: attributed + cut 7→2 (2026-08-17)

Isolated per-call probe (`/tmp/bench/fa_percall_probe.py`) profiles a
single `forward_paged` call at the production decode shape (B=1, Sq=1,
Hq=24/Hkv=4/D=256, Sk=2048, legacy path) — deterministic op census, no
windowing. Found **7 copies per FA layer (~47 µs, launch-bound)**:

| copy | shape | µs | fix |
|---|---|---|---|
| backend `q.float()` cast | [1,24,256] | 7.0 | drop: copy_ into the fp32 q_pad buffer casts |
| q_pad_buf slice `.contiguous()` | [1,24,2,256] | 9.0 | dedicated decode buffer [maxB,Hq,2,D]; `[:B]` prefix slice is contiguous |
| q→q_pad assign | [1,24,1,256] | 8.2 | KEPT (now carries the cast) |
| `cu_seqlens_q.to(long)` | [2] | 6.9 | deferred into the multi-seq branch only |
| C launcher `.transpose(1,2).contiguous()` | [1,24,2,256] | 8.8 | C returns native BSHD |
| Python `[:,:,0,:].reshape().contiguous()` | [1,24,256] | ~7 | gone: BSHD `[:, 0, :, :]` is a contiguous [B,Hq,D] view |
| `out_flat.to(fp16)` + `copy_` | [1,6144]×2 | 6.7 | one `out_view.copy_(out_flat)` (cast fused) |

Result: **2 copies/layer** (the two unavoidable ones), ~425 µs/step in
eager. The B>1 decode path goes from 2 copies/layer (384 KB each at
B=8) to 1 (192 KB); the fully-zero-copy B>1 path needs a
decode-specialized kernel store ([B,Hq,D] direct, j==0 guard) —
follow-up, only matters for batched decode.

**Interim bug (caught by the PPL gate):** my first BSHD edit left a
`.permute(1, 0, 2)` in the general output path — with BSHD,
`out_padded[s, :n]` is already a contiguous [n, Hq, D], so the permute
scrambled the reshape. PPL dense dropped to 11.89 (was 6.70); removed
the permute → fixed. PPL is the right gate for prefill-path layout
changes (the eager kernel tests cover the Sq=1 fast path but the
probe's prompts exercise the general path).

**[3,1,32] copy pile (32/step, ~180 µs) — attributed to GDN, not FA.**
Timeline-context probe (`/tmp/bench/dense_ewp_timeline.py`): the copies
sandwich `_causal_conv1d_update` + `fused_recurrent_gated_delta_rule`
in the GDN layers — vLLM upstream mamba state bookkeeping. Deferred
(upstream code, small).

**Gates (all green):** FA tests 15/15; PPL dense **6.7026** (band
6.6993–6.7197), MoE **6.6863** (band 6.6817–6.6942); serving dense
**24.061 / 24.028 t/s** (record; was 23.845/23.801, +0.9% — the in-graph
gain is smaller than the eager estimate because launch overhead is
cheap inside the graph; the GPU copy time itself is ~0.15–0.25
ms/step); serving MoE **67.024 t/s** (record; was 66.36).

**Build note:** FA `.so` rebuilt locally via `/tmp/bench/build_fa_local.py`
(only `gfx906_fa.cpp` recompiled — kernels untouched). Lint: 37 ruff
findings after vs 40 before on the touched files (pre-existing debt
only, no new ones).

### GemmaRMSNorm (1+w) eager cache (2026-08-17)

`forward_cuda` (and `forward_native`) recomputed `weight.to(dtype) + 1.0`
on every call: 131 [5120]+scalar adds per dense step (~0.78 ms/step in
eager, launch-bound). Now cached in `_one_plus_weight(dtype)`, keyed on
`(data_ptr, dtype, device, _version)`:

- Inference weights are frozen after loading, and vLLM loads weights
  before the first forward (which is where the cache is created), so
  the version/data_ptr key is sufficient. Note: in-place ops through
  `.data` do NOT bump `_version` (verified); that path only matters
  pre-forward here.
- No stream-capture guard: vLLM always warms up eagerly before cudagraph
  capture, so the cached allocation comes from the normal pool. (A
  `torch.cuda.is_current_stream_capturing()` guard was tried and had to
  be removed: this torch build's dynamo cannot trace the op —
  `Unsupported: torch.* op returned non-Tensor`.)
- Unit: cache hit returns the same tensor; in-place op on the parameter
  and parameter replacement invalidate; fp16/bf16 both correct; fused
  add path residual exact.

Eager-only gain (production inductor folds the +1 into the norm codegen;
the value is bit-identical, just computed once). PPL on the final build:
dense 6.7122, MoE 6.6832 — both in band.

### max-ilp scheduler strategy: measured, adopted per-file (2026-08-17)

User-requested measurement of `-mllvm -amdgpu-sched-strategy=max-ilp`.
The flag demonstrably changes gfx906 codegen (dummy-kernel ISA diff:
instruction reordering) and is applied per source file in
CMakeLists.txt (`VLLM_NO_MAX_ILP=1` to disable).

**Micro-bench A/B** (same session, local builds):

| kernel | no flag | max-ilp | delta |
|---|---|---|---|
| FA decode dense 24/4, Sk=3328 | 93.3 µs | 91.0 | −2.5% |
| FA decode MoE 16/2, Sk=13312 | 186.9 | 177.7 | −4.9% |
| GPTQ W4 gate/up 17408×5120 | 86.2 | 73.1 | **−15.2%** |
| GPTQ W4 down 5120×17408 | 105.3 | 77.8 | **−26.1%** |
| GPTQ W4 gdn_out 6144×5120 | 40.4 | 37.2 | −7.9% |
| MoE W4 kernel (same shapes) | 82.8/105.4/37.9 | 71.7/77.1/36.1 | −13/−27/−5% |
| GEMV K=17408 | 227.6 µs | 229.6 | neutral (HBM floor) |

**Serving A/B** (within the same build environment — required, because
the local no-flag environment itself runs ~2% lower than the
docker-built no-flag on dense, 23.70 vs 24.06 t/s):

| build | dense t/s | MoE t/s |
|---|---|---|
| no flag | 23.55 / 23.86 | 67.08 |
| global max-ilp (CMAKE_HIP_FLAGS) | **25.54 / 25.51** | 65.65 / 65.56 |
| per-file max-ilp (CMakeLists) | 25.14 / 25.60 | **67.33 / 67.39** |

Findings:
- The flag is a big win on the W4 GEMM kernels (exllama gptq, −8 to
  −26%) — the source of the dense +7–8% e2e.
- It regresses the MoE routed-expert kernel at decode shapes: global
  flag → MoE −2.2% (65.6 vs 67.1, two samples each). Presumed register
  pressure: max-ilp raises VGPR usage, which hurts occupancy in the
  wavefront-limited decode regime (the probe shapes above are M=1 rows
  with large N; the production MoE launch is 256 experts × tiny M).
- Per-file adoption keeps both wins: flag on q_gemm (gptq),
  gfx906_fa_{launcher,quant,gather}, skinny_gemms{,_int4}; excluded:
  moe_q_gemm_gfx906, dense_gemv (neutral), attention, all others.

Gates on the final per-file build: PPL dense 6.7122 ✓ / MoE 6.6832 ✓;
serving dense 25.14/25.60 (record; was 23.70), MoE 67.33/67.39 (record;
was 67.08). Build mechanics (local setup.py recipe, CMAKE_HIP_FLAGS
cache gotcha) are documented in docs/gfx906/running.md.


---

## 2026-08-17 — Dense 27B load test in the 0.27.99rc0-rocm-7.14 docker image

Image `unverbraucht/vllm-gfx906:0.27.99rc0-rocm-7.14` (built from the merged
`gfx906/main` tip `e861d0b30f`) served Qwen3.5-27B-AWQ (48 GDN + 16 FA)
single-GPU MI50. Load test: chat + completions endpoints, 25..2124-token
prompts, bursts/sustained/long-context, ~48 requests, **zero failures,
zero hipError/assert in server logs**.

### Findings

- **Hybrid-model steady-state footprint on MI50 32GB** (the reason the
  first three configs OOMed):
  - weights 18.9 GiB (visual tower dropped via
    `--limit-mm-per-prompt '{"image":0,"video":0}'`);
  - mamba (GDN) state workspace ≈ 72 MB *per sequence* (48 linear layers ×
    48 v-heads × 128×128) — `max_num_seqs` is the biggest lever;
  - the custom-FA **capture-generation gather buffers**
    (`_ensure_gather_buffers` in `gfx906_fa_backend.py`) are sized
    `max_num_seqs × pad32(max_model_len)` and linger in the R3 retirement
    list after the first reallocate — 1.64 GiB at 16×32768, 83 MB at
    8×3328. This is what made max-model-len 131072 unbootable (8.5-9.1 GiB
    allocation during cudagraph profiling) and what ate the headroom at
    16 seqs × 32768 (engine died on ~2K prefills, `free: 0`, fixed
    340/600 MiB inductor allocations);
  - inductor AOT static workspace scales with the compiled range
    (`max_num_batched_tokens`).
  - `--kv-cache-memory-bytes` skips memory profiling entirely → all of the
    above must be budgeted manually (vLLM's auto-sizing over-commits by
    ~0.7 GiB at gpu_util ≥ 0.92).
- **Working config** (32K context, stable): `--max-model-len 32768
  --gpu-memory-utilization 0.92 --max-num-seqs 8
  --max-num-batched-tokens 512 --kv-cache-memory-bytes 5368709120
  --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}'
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY",
  "max_cudagraph_capture_size":8}'`. KV pool: 85,333 tokens
  (125 × 784-token blocks, mamba page aligned), 2.6× concurrency at 32K.
  Idle VRAM 28.8 GiB, ~3.2 GiB headroom.
- **Prefix caching works**: 2124-token prompt cold 11.1 s → warm 4.7 s;
  metrics `prefix_cache_hits_total 1568 / queries 6129` (1568 = two
  784-token mamba-aligned blocks).
- **Latency reality check** (512-token chunk cap for stability):
  - prefill ≈ 250-300 tok/s (chunk-capped; TTFT median 56 s for an
    8×2048 burst — the 512 cap, not the GPU, is the TTFT bottleneck);
  - decode ≈ 25 t/s single-stream (matches the 25.60 t/s record),
    ≈ 30 t/s aggregate at batch 8;
  - sustained capacity ≈ 0.18 rps for 1024-in/128-out (2 rps saturates the
    queue: 24 peak concurrent, 16 waiting, P99 TTFT 101 s).
- Qwen3.5-27B is a thinking model: use
  `chat_template_kwargs: {"enable_thinking": false}` for concise answers.
- Bench driver gotchas: `vllm bench serve --backend openai --model
  <served-name> --tokenizer <model-path>` (model name must match the
  served name; tokenizer needs the path).
### Chunk-cap A/B: 512 vs 1568 at 8K context

Single-stream 8205-token prompt, cold (2 different prompts) + warm:

| config | chunk | KV pool | max seqs | cold TTFT | prefill | warm TTFT |
|---|---|---|---|---|---|---|
| load-test baseline | 512 | 5 GiB (85,333 tok) | 8 | 35.3-35.6 s | 231-233 tok/s | 1.74 s |
| raised cap | 1568 | 4.5 GiB (64,170 tok) | 4 | 32.2-32.3 s | 254-255 tok/s | 1.76 s |

- 1568 only fits after giving back 0.5 GiB of KV pool **and** 4
  sequences: at 8 seqs / 5 GiB the first 1568-chunk forward OOMs
  (`free: 0`, same 340 MiB signature) — the per-chunk forward peak
  needs ~3.2-4.5 GiB over the ~27.5-28.8 GiB idle footprint.
- **The TTFT gain is only ~10% (35.4 s -> 32.3 s), not the expected
  2-3x**: prefill on this hybrid model is *token-bound* (48/64 layers
  are GDN), not chunk-bound. The 512 cap limits how much prefill work
  overlaps, not the per-token rate. Earlier note ("the 512 cap, not the
  GPU, is the TTFT bottleneck") is superseded for single-stream TTFT.
- Warm (prefix-cache) TTFT is config-invariant (~1.75 s for 8K).
- The 1568/4-seq/4.5-GiB config is stable (4x2048 burst, 4/4 OK).

---

