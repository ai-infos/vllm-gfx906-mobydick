# Dev log — Qwen 3.5 quantized MoE decode/prefill on gfx906

> The MoE kernel train (W4A16 grouped GEMM, prefill tuning, the
> fill/copy pile, layer-0 attribution, branch merge/archival). The
> custom FA backend moved to DEVLOG-fa-attention.md; the dense 27B
> trail to DEVLOG-dense-decode.md. Historical development record:
> the plan/review documents this log referenced were consolidated
> into README.md at merge-prep and removed; one-off probe scripts
> likewise.

Branch: `gfx906/moe-opt` (from `gfx906/fa-integration` = fork main +
custom FA). **Merged into `gfx906/main` 2026-08-18** (fast-forward,
71 commits, 52 files, tip `1691d1dd29`).

**VERDICT (top-level): SHIPPED** — MoE decode 3.49 → 67.39 t/s on the
flagship (19.3×); prefill ~2140 t/s. Individual experiment verdicts
labelled inline. (The post-branch C2-V/W2/W4 work that reaches 67.39
lives in DEVLOG-moe-c2v.md / DEVLOG-moe-spec-decode.md /
DEVLOG-fp16-skinny.md.)

## Problem statement

End-to-end bench (`_bench_gfx906.py`, pp=2048/tg=256, MI50 32 GB, eager):

| Version | dense 9B-AWQ tok/s | MoE 35B-A3B-AWQ tok/s |
|---------|--------------------|-----------------------|
| 0.23.0  | 27.47              | (unsupported)         |
| 0.26.0  | 28.03              | 12.16                 |
| main    | 32.31              | **3.49 (−71%)**       |

Profile (main, torch profiler in-proc, pp=512/tg=64):
`fused_moe_kernel_gptq_awq` = **91.2% of GPU time**, 3.495 ms/call —
the whole regression is one kernel. MoE GEMM per decode step ≈ 3% of
achievable HBM bandwidth: a kernel problem, not a bandwidth wall.

Dispatch fact (both versions, still true): `check_moe_marlin_supports_layer`
is **always False on ROCm** → AWQ MoE falls back to `MoeWNA16Config`/
`MoeWNA16Method` (the warning "not supported by AutoAWQMoEMarlin" ×39 =
all quantized MoE layers). 0.26 ran the legacy monolithic Triton
`fused_experts`; main ran the modular pipeline + TritonWNA16Experts
(separate align/gemm1/act/gemm2/sum kernels) = the −71%. Dense AWQ has
a custom gfx906 `gptq_gemm` fast path with **no MoE equivalent**.

## PROBE PITFALLS — read before ANY monkeypatch/counter/measurement probe

> Learned the hard way twice (P3-0 Q3/Q4). vLLM V1 makes naive
> in-process probes silently wrong or loudly crashing.

1. **The model does NOT live in your process by default.** V1 runs
   EngineCore in a spawned process (`SyncMPClient`); wrapping
   `module.forward` in the main process measures NOTHING (probe exits
   0, all-zero tables — a silent lie). Fix:
   `VLLM_ENABLE_V1_MULTIPROCESSING=0` BEFORE wrapping anything.
2. Every probe script needs `if __name__ == "__main__": main()` —
   spawn re-imports the module in the child.
3. In-proc mode changes memory accounting: with this model's
   `max_model_len=262144`, util 0.9 OOMs the KV cache in-proc. Probes
   that only need short gens: `max_model_len=4096` — never copy a
   config you haven't checked for seq-len.
4. **Never trust a probe that prints zeros.** Assert ≥1 counter
   non-zero (and right order of magnitude, ~40 router gemms/step here)
   before believing results.
5. Profiling ≠ probing. torch-profiler traces work in MP mode for
   EAGER runs; in cudagraph mode op-correlation breaks (kernels come
   from `hipGraphLaunch`) — attribute via `hipGraphLaunch` boundaries.
6. Counter keys built in a loop need default-arg binding
   (`def make_fwd(orig, _k=key)`) — late binding puts all calls in the
   last bucket.
7. Windows that include prefill/capture must be separated by timestamp
   before dividing by step count (see the P3-0 aten::mm artifact).

## Model facts (QuantTrio/Qwen3.5-35B-A3B-AWQ)

- 40 layers: **30 linear-attn (GDN) + 10 full-attn** (every 4th).
  hidden=2048, head_dim=256. **256 routed experts/layer, top-k=8**,
  moe_intermediate=512; shared expert (fp16) + **layer 0** + attn +
  linear_attn not quantized (`modules_to_not_convert`).
- AWQ int4, group_size=128, zero_point=true. Active routed
  ≈ 25.2M params/token → ~507 MB/token over 40 layers.
- Roofline ≤1 TB/s theoretical HBM → >1000 tok/s decode. Measured 12
  (0.26) / 3.5 (main) = ~100× off roofline. (The old "~700 GB/s"
  figure was wrong — see P2-0.)

## Phase 1 — custom W4A16 grouped GEMM (SHIPPED)

The fork already had this design for RDNA3
(`csrc/rocm/moe_q_gemm_gfx906.cu`'s parent `moe_q_gemm_rdna3.cu`:
sorted_token_ids/expert_ids + exllama-shuffled [E,K/8,N] weights,
pre-zeroed C + fp16 CAS atomic epilogue, fused topk-weight sum).
Ported it (fp16-only `__ockl_fdot2`, runtime zero_offset,
BLOCK_SIZE_M ∈ {1,2,4,8,16}), through the standard modular pipeline
so shared experts work (the RDNA3 path bypasses the pipeline and drops
shared experts — not copied). Repack in torch/Python at load
(~65 s); oracle backend `GFX906_HIP` first on gfx906.

**Layout gotchas (learned the hard way):**
1. **Exllama shuffle is even/odd interleaved, not natural order.** The
   dequant masks put k0,k2,k4,k6 in the lower half, k1,k3,k5,k7 upper
   ("77775555 33331111 66664444 22220000"). Packing natural order
   produced silently wrong output — a self-consistent (wrong-both-sides)
   torch test passed until nibbles were extracted with the *kernel's*
   masks.
2. **AWQ MoE on main is K-first int32 [E,K,N/8], not MoeWNA16 N-first
   uint8** (that layout is 0.26-legacy only). Consequence: scales/zp
   are *already in kernel layout* → repack = weights only.
3. Nibble-interleave reshape: `stack(lo,hi,dim=2)` then
   `reshape(E,N,G)` is correct; stacking at the last dim merges
   (G,2) not (N/2,2). Same trap in the test reference — twice.
4. **Docker import shadowing**: the image's dist-packages vllm is not
   removed by `pip install -e .` — run with
   `PYTHONPATH=/workspace/vllm` and clear stale `__pycache__`.
5. **Backend gate**: the MoeWNA16 fallback passes group-scale weight
   keys, not `kInt4StaticGroupScale` — the gate must accept all four
   int4 keys. `is_supported_config` rejections log at DEBUG only.

Micro-bench (E=256, topk=8, Qwen shapes, per-call µs, w13+w2 speedup):

| M | Triton | gfx906 | speedup |
|---|--------|--------|---------|
| 1 | 3667+564 | 35.5+33.0 | **62×** |
| 8 | 25036+1923 | 135+80 | **125×** |
| 512 | 169723+13567 | 3063+1618 | **39×** |

Decode w13 3.5 ms → 35.5 µs; effective ~225 GB/s (latency-bound).
Correctness ALL PASS M ∈ {1..64}, both layouts, both gemm passes.

Full model (pp=2048/tg=256, eager): **3.49 → 18.79 tok/s** (+54% vs
0.26's 12.16); prefill 450 → **2140 tok/s** (4.7×), decode 3.72 →
19.7 (5.3×). Post-fix profile: MoE 14.95% of GPU (was 91.2%);
CPU-launch-bound in eager (Self CPU 4.3 s > Self CUDA 2.0 s).
Pre-fix A/B via `git worktree` of the parent commit — the worktree has
no built .so's; copy them from the image's dist-packages (symlinking
the *new* tree's .so into the old tree fails: ops moved in main).

## PHASE 2 (plan: `plan-moe-phase2.md`, adversarially reviewed)

### P2-0 — diagnostics

**Hardware correction:** 240 SIMDs = **60 CUs → MI50, not MI60**
(the header's MI60 label was VRAM-based). Measured `v_dot2_f32_f16`
peak **~20 TFLOPS** (standalone, ILP≥2, sclk 930) — not the 29.5
datasheet; at ILP=1 only ~8.5 (dependency-sensitive). The plan's
"occupancy=1 pinned by ~80 VGPRs" hypothesis **REFUTED**: no spills at
any BM, measured occupancy 6–13 blocks/CU (true table via
`hipOccupancyMaxActiveBlocksPerMultiprocessor` — the P2-0 "occupancy"
column was mislabeled). PMC (gfx906 counter names differ from CDNA;
CDNA-style names silently return 0): bank conflicts 0, LDS waits <1%
→ issue-bound at large M, not LDS-bound. **Launch baseline: ~1,500
hipLaunchKernel per decode step** (rocprofv3 `--kernel-trace`; note
`--hip-trace` does NOT include kernel dispatches). Tooling: host LLVM
tools extract gfx906 objects from `_rocm_C.abi3.so`
(`llvm-objcopy --dump-section .hip_fatbin=` + scan \x7fELF +
`llvm-readobj`); **no PC sampling on gfx906** (CDNA2+ only).

### P2-0b — zero-fill launch fusion: **DEAD-END (design is racy)**

Folding `w1_out.zero_()` into the GEMM via `blockIdx.z==0`
clear-before-CAS is a data race: K-slice blocks run concurrently with
no grid.z ordering — a z>0 atomic can land before z=0's clear-store,
silently dropping the partial sum. Correctness tests pass thousands of
times (tiny window). Alternatives rejected: spin-wait flag (undocumented
dispatch assumptions, hang risk), per-element epoch CAS (4B atomic per
output element, likely slower than the memset), zero-on-read in the
consumer (cross-cutting shared op). ~80/1500 launches not worth it.

### P2-1 — prefill tuning

1. **b128 LDS reads** (SHIPPED, neutral): 3027→3049 µs M=512 (noise;
   P2-0 showed zero bank conflicts). Kept (lower LDS instruction
   pressure).
2. **N_PER_THREAD=2** (SHIPPED, +3.7%): M=512 3027→2917; occupancy
   doubles at BM=16 (4→8 waves/CU). Modest → not purely
   latency-bound.
3. **Double-buffered weight prefetch: DEAD-END — reverted.** 256 VGPRs
   + 214 spills → 5.8× slower; even spill-free <4,4> slower
   (occupancy halved). The compiler keeps both chunk buffers live
   across the whole consume phase — software prefetch is
   register-infeasible at BM=16.
4. **BM=8 for prefill (SHIPPED, BIG WIN):** `em > 512 → BM=8, NPT=2`
   = 3 blocks/CU (12 waves) → M=512 w13 **−23%**, cumulative **−26%**
   vs Phase 1 at M=512. BM=8 loses below em~1024 (padding waste) →
   heuristic ≤32→1, ≤512→4, else 8. Full model +0.5% (prefill is ~7%
   of bench time).
5. **ISA finding:** K-loop body 512 v_dot2 = 57.5% of ~890
   instructions, but S+V share one issue port → **issue-bound**: dots
   alone cap at ~9.6 TF (48% of dot peak). The <1.5 ms M=512 goal is
   unreachable in this kernel shape (needs the persistent-CTA
   redesign — deferred, out of Phase-2 scope).

### P2-2 — cudagraph ceiling (SHIPPED, config-only 2.2×)

Setup gotchas (hybrid GDN model): default `FULL_AND_PIECEWISE`
capture OOMs (graphs to 512) → `FULL_DECODE_ONLY` +
`max_cudagraph_capture_size=8`; capture requires
`max_num_seqs ≤ mamba cache blocks` (91) → `max_num_seqs=32`.

| mode | tok/s | decode est. |
|------|-------|-------------|
| eager | 18.88 | ~19.7 (50.8 ms/step) |
| **FULL_DECODE_ONLY graphs** | **41.51** | ~49 (20.3 ms/step) |

2.2× from configuration alone — confirms the CPU-launch-bound
diagnosis. `_bench_gfx906.py` gained `BENCH_EAGER=0` for this.

### P2-3/4/5 — go/no-go from graph-mode profile

Self CUDA ≈ 22.6 ms/step: aiter LLGemm1 7.2 (32%), aten::mm 2.2 (10%
— later refuted, see P3-0), paged attn 1.95 (9%), **MoE kernel 1.77
(8%)**, triton_matmul 1.6 (7%), GDN ~1.15 (5%), routing ~1.0 (4.5%),
shared expert 0.55 (2.5%), elementwise ~2 (9%).
- **P2-3 (decode MoE latency): SKIP** — MoE is 8% of the step; the
  residual is non-MoE kernels (a separate project).
- **P2-4 (fused topk+align): DEFERRED** — 4.5%, highest
  correctness-risk item.
- **P2-5 (shared-expert Triton elimination): DEFERRED** — 2.5%
  (premise later corrected by O1).

End-to-end at Phase 2: **3.49 → 18.88 eager (5.4×), 41.5 serving
(11.9×)**.

### llama.cpp baseline (Q4_K_XL, full offload) — done

`llama-bench -ngl 99 -p 2048 -n 256 -r 2` (Qwen3.6-35B-A3B, same
arch; the Q5_K_XL 36 GiB file cannot fit the 32 GB card):
**prefill 806.5 t/s, decode 70.3 t/s** (14.2 ms/step) vs vLLM
eager 19.7 / graph ~49. vLLM wins prefill 2.6×; llama.cpp wins decode
1.43× — the gap lives in dense/GDN/attention (its dense weights are
Q8_0 = half our fp16 bytes; its attention is 15 µs/layer vs our
194). Host gotcha: run with `LD_LIBRARY_PATH=/opt/rocm-7.14/lib` —
the ldconfig cache's 6.4.3 hsa-runtime is ABI-incompatible (SIGSEGV in
`hipStreamCreateWithFlags`).

## PHASE 3 (plan: `plan-decode-phase3.md`, v3 after external adversarial review)

Goal: close the ~6 ms/step decode gap vs llama.cpp in non-MoE
kernels. Primary metric: serving mode decode tok/s + ms/step.

### P3-0 — diagnostics

**HBM read BW measured 798 GB/s** (8 GiB double4 stream). L2: paged
attn KV 82% hit (resident), dense m=1 gemms ~13–15% — dense-gemm
floors at HBM are real.

**Negative result (important):** the P2-3 "aten::mm 4×532 µs = 2.2
ms/step" row **does not exist in steady-state decode** — a
warmup/capture artifact of that profile window (zero M=1 `aten::mm`
rows in the shape-aware profile). Lesson: timestamp-separate
prefill/capture windows before dividing by step count.

**Reconciled per-step budget (31 graph steps, kernel-level):** LLGemm1
5.83 ms (230 calls, incl. shared expert + LM head) ·
**triton_matmul 1.63 ms = ONE [1,2048] `shared_expert_gate` Linear ×40
at 41 µs/call** (Q3 below) · paged attn 1.94 · MoE 1.75 · routing
1.06 · GDN ~0.5 (faster than llama.cpp's — leave alone) ·
elementwise ~2.3 · kernel total ~17.6 ms + **~2.7 ms inter-kernel
gap** (llama.cpp ≈ 0: 14.32 kernel = 14.2 wall).

**Q3 (the big one):** in-proc probe of all 270 Linear modules — the
other 230 (router [256,2048], GDN in_proj/out_proj, FA qkv/o_proj,
shared-expert gate_up+down) all run LLGemm1/LLMM1; 40 go Triton, and
the entire Triton row is
`shared_expert_gate` (`ReplicatedLinear(hidden→1)`, one rank-1 dot
per layer): `rocm_unquantized_gemm_impl` sends m=1 to Triton because
LLMM1 requires `m % 4 == 0`. The "biased out_projs fall back"
hypothesis was WRONG — no module has bias. Fix surface: one dispatch
branch; floor ~5 µs/call → **~1.4 ms/step recoverable**.

**llama.cpp per-step comparison:** its lead = Q8_0 dense weight bytes
(3.85 ms vs our 5.83+1.2 fp16) + attention (0.19 vs 1.94 ms — the
biggest kernel-level difference); MoE at parity; GDN we're already
faster. Its activation-quant tax (1.31 ms) we don't pay. GGUF v3
parser note: the new type enum (STRING=8, ARRAY=9) — hand-rolled
parsers with the old enum misparse silently.

Revised priorities: P3-1 gate fix (~1.4 ms) → P3-3 attn (~1.5–1.7 ms,
moved to the FA track) → P3-2 LLGemm1 surface → inter-kernel gap via
launch-count cuts.

### P3-1 — `shared_expert_gate` tiny-m gemv (SHIPPED)

Micro-bench (n=1, k=2048 fp16): Triton 42.8 µs · torch F.linear
(rocBLAS gemv) **281.1 — terrible, rejected** · **LLMM1 + zero-pad to
4 rows 7.3 — chosen** · `(x*w).sum(-1)` 12.6. Change: `_llmm1_tiny_m()`
in `vllm/model_executor/layers/utils.py` (pad weight to 4 rows,
`ops.LLMM1(w, x, 4)`, slice; both LLMM1 dispatch sites accept
`m % 4 == 0 or m < 4`). Generic, not model-specific. Correctness:
greedy A/B — 2/3 prompts token-identical, prompt 2 diverges at token
~11 (fp16 reorder sensitivity of the sigmoid gate, logit delta ~1e-3,
both fluent) — accepted. (That test file has 8 pre-existing base
failures in this env; passes 1→5.)

| mode | before | after |
|------|--------|-------|
| eager | 18.88 | **19.49** (+3.2%) |
| serving (cudagraph) | 41.51 | **44.09** (+6.2% ≈ 1.6 ms/step) |

llama.cpp gap (serving): 1.70× → 1.59×.

### P3-4 — the fill/copy pile: three fixes LANDED (SHIPPED)

Post-FA-track trace: **1178 µs/step** in small fills + D2D copies
(246.8 launches/step, median 4.64 µs — 100% launch-latency-bound; the
only lever is removing launches, not bytes). Attribution via
eager-mode torch profiler with **correlation-id join** (cpu_op →
kernel → python stack; kernel-timestamp stack attribution is
unreliable — GPU-queue lag; CPU-op-timestamp windows are the
reliable one; `docs/gfx906/fillprof_probe.py`, since removed).
Top rows: fill [4,2048] = `F.pad` of the *constant*
shared-expert-gate weight every layer/step (P3-1's pad) · fill
[8,1024] `w1_out.zero_()` (MoE gemm1 atomic K-split) · fill [1,2048]
`output.zero_()` (aliases w1_out — see REJECTED) · fill [1,32,128]
GDN `core_attn_out = torch.zeros` (upstream PR #28182 spec invariant)
· 80.3 copy [1,2048]/step (MoE + LLMM1 pads).

Fixes (all bit-exact or provably output-identical):
1. **gate [1,K] → GEMV RPT=1** (utils.py): kills the F.pad fill + pad
   copy per layer/step; 22.2 → 4.7 µs/call, bit-equal to pad+LLMM1 at
   N=1 (call sites guarantee n==1, no bias).
2. **FA q_pad zero_ skip on the decode fast path** (LEGACY branch,
   `max_seqlen_q==1 and num_tokens==num_seqs`): pad rows never
   consumed (per-row-independent tiles; Python keeps row 0; q8_0
   quantization clamps NaN garbage to int8 range). Prefill keeps the
   zero_. 15/15 FA tests incl. the capture test.
3. **GDN core_attn_out `torch.empty` on the non-spec packed-decode
   fast path**, env-gated `GFX906_GDN_EMPTY_CORE_OUT=1` (default 0):
   the packed-decode kernel stores unconditionally for every cell
   (explicit zero store on the invalid-state branch) — no output byte
   survives from the allocation.

**REJECTED: MoE zero_ reorder (aliasing trap).** Overlapping
`output.zero_()` with `w1_out.zero_()` corrupted prefill (PPL 6.69 →
1.09e7): modular_kernel `_allocate_buffers` **reuses one
common_workspace for gemm1 out and fused output** — the gemm2
zeroing must happen after gemm1+activation finish using the same
memory. Both gemms K-split via grid.z atomics → the two zeroings are
genuinely required and cannot merge. (Documented at the call site in
`gfx906_w4a16_moe.py`.)

Deferred (measured, not worth it): FA fp16 q/out casts (~80 µs/step,
touches vendor kernel for 0.5%); the MoE zeroings (required); runner
H2D micro-copies (~60 µs/step, upstream); GDN prefill/spec fill
(upstream invariant).

Gates: PPL 6.6862 vs 6.6889 (inside MoE-atomic run-to-run noise); MB
greedy B=2 — one token-0 flip between two runs of the *same* build
(known multi-batch near-tie nondeterminism); unit 15/15 + 12/12.
Serving A/B (util 0.95, FULL_DECODE_ONLY, 2 samples): baseline
63.175/63.53 → fixes (1)+(2)+(3, env on) **64.102/64.062 → record
64.08 t/s** (min-fixed > max-baseline). Net +0.7 t/s (+1.15%) vs the
63.56 record — smaller than the ~350 µs/step isolated per-op numbers
predict because at 99.5% GPU busy part of the removed fills was
already overlapped; only the critical-path share is recovered.
llama.cpp gap: 70.3/64.08 ≈ **1.10×**. Phase 3 declared done; the
MoE-side residual (~3.8 ms/step vs ~1.0 floor) catalogued in
`plan-moe-decode-future.md`.

### O1 — the Triton residual is layer 0 (resolved)

The 2×/step `fused_moe_kernel` (206.8 µs/call, 414 µs/step) is
**layer 0's routed MoE**: `modules_to_not_convert` includes
`"model.layers.0."` → its 256 experts ship fp16 → unquantized oracle
→ Triton. **The P2-5 premise ("shared expert runs the generic Triton
fused_moe, ~40 launches/pass") was a misattribution of this same
signal — there was never a Triton shared expert** (all shared experts
are dense fp16 on the LLGemm1/LLMM1/GEMV surface). Roadmap updated
(O1 resolved, C4/C5 rewritten).

## 2026-08-17 — Pre-merge review fixes (4 findings, all fixed)

1. **Repack UnboundLocalError on symmetric checkpoints** (qzeros=None)
   + second latent bug: the symmetric zp fill was `torch.full(..., 8)`
   — a packed zp word needs 8 in *every nibble* (**0x88888888**); 8
   only set nibble 0 (columns 1–7 would have used zp 0). Same fix in
   the AWQ-K-first variant.
2. **GDN empty-core-out not platform-gated** — added `on_gfx906()`.
3. **Oracle accepted GPTQ-style zp checkpoints** — the kernel/repack
   implement AWQ-encoded zps only; now excludes
   `(AutoGPTQConfig, QuantizationArgs)` explicitly.
4. **`moe_gptq_gemm_gfx906` missing `top_k > 0` guard** (on-device int
   div-by-zero is UB) — `TORCH_CHECK` added, verified.

Tests: 12 new sym-layout cases (the `wna16_sym` case crashed
UnboundLocalError pre-fix, wrong output with the first-draft fill);
oracle rejection/acceptance cases; M=1 cases use tol 1e-1 (fp16
accumulation ~2× noisier over TOPK rows only; 5e-2 flaked once in ~50).
Gates: 72 passed / 2 skipped; PPL 6.6827 in band (6.6817–6.6832).

## 2026-08-18 — Parked review items R1–R12 resolved

All 12 roadmap-§9 items; mostly edge-path/hygiene, three touch the
default LEGACY=1 path (R8, R7, R11):
- **R1** `forward_paged` direct branch used the fp32 q_pad fallback
  (the fp16 one is rejected by TORCH_CHECK); Sq=1 uses
  `q_pad_decode_buf` (no per-layer .contiguous()). LEGACY=0-only.
- **R2** cudagraph support fails closed on LEGACY=0 + prefix caching.
- **R3** `_gather_retired` bounded (max 4 pairs, oldest evicted).
  The review's "grow-only capacity + exact view" rejected: the gather
  kernels address output from SHAPES, not strides — a non-contiguous
  view corrupts silently (a real capacity buffer needs stride-based
  output addressing in C++). Bounding trades worst-case memory for a
  stale graph touching an evicted buffer — only reachable in the
  already-inconsistent LEGACY=0+capture combo. (Superseded 2026-08-24
  by the capacity-Sk + per-generation capture lifecycle in
  DEVLOG-fa-attention.md: it over-allocates contiguous width — no
  views — so R3's shape-vs-stride objection never applies.)
- **R4** kernel host checks now include all silent-garbage violations
  (K/groups/N%8/scales_N/zeros_N) + matching oracle shape gate.
- **R5** tolerant `_gfx906_fa_C` import; `register()` no-op off
  gfx906. **R6** CMake no-op target when gfx906 absent from arches.
- **R7** ncols1 ladder consolidated to `fa_pick_ncols1()` +
  `_pick_ncols1()` (was 4 code copies) with keep-in-sync pointers.
- **R8** `gather_paged_kv_quantized` gained the same grow-buffer
  params as `gather_paged_kv_q8`; the default LEGACY=1 path previously
  allocated a fresh 24–200+ MiB K+V pair per layer per step on long
  contexts. (Default-path change — covered by the PPL gate below.)
- **R9** workspace13/output aliasing + load-bearing order documented.
- **R10** docstring/help/header fixes (kchunk 1024, BSHD, NPT in the
  grid formula, dead `mask_buf` param).
- **R11** ruff-clean: `vllm/gfx906_fa/`, gfx906 bench scripts, two
  test files; per-file-ignores for the bench timeit closures.
- **R12** `GFX906_FA_DEBUG=1` master switch for the six debug hooks.

Gates: 74 passed / 2 skipped; PPL MoE **6.6825** (band; covers R8
end-to-end). Serving benches not re-run: R8 changes allocation source,
not kernel work; the rest are dormant in the default config.

## 2026-08-18 — Merged into gfx906/main

Fast-forward (main was a direct ancestor; 71 commits, 52 files,
+13818/−21, tip `1691d1dd29`); linear history kept (fork convention:
merge commits only for upstream merges). Gates current at the merge
tip.

## 2026-08-18 — Upstream merge: vllm-project/main into gfx906/main

158 upstream commits; 661 files, +31783/−7248. Four content
conflicts, resolved by hand:
1. `config/attention.py` `IndexerKVDType`: upstream "auto" + logger
   kept alongside our fp16/fp32 spellings (Minimax M3 gfx906 work).
2. `auto_awq.py`: took upstream's deletion of
   `is_awq_marlin_compatible` (dead; our guard on it moot).
3. `rocm_aiter_mla_sparse.py` (ops): kept our env-gated fp16 logits
   gate (gfx906 has no FP8).
4. MLA backend `get_supported_kernel_block_sizes`: upstream's
   `[1, MultipleOf(16)]` is a superset of our `[1,32,64]` — took
   upstream.

Upstream of note: Rust frontend/gRPC, spec-decode MTP fusions,
`[ROCm]` Triton W4A16 transpose bugfix, gfx942 FlyDSL. **Zero
csrc/rocm changes from the merge.** Validation: build clean; M3
config tests byte-identical pre/post (3F pre-existing both sides);
gfx906 suites 99 passed / 2 skipped; greedy probe **token-identical**
(`868ad09e…`); serving 4-sample: MoE 66.56 mean (band 65.87–67.03),
dense 27B (max_num_seqs=4) 25.25 mean. No regression.

**Finding — dense 27B graph mode OOMs at max_num_seqs=32
(post-merge):** first 1568-token prefill chunk, 340 MiB inductor
static-buffer allocation, `free: 0` (the known 340/600 MiB signature).
At 32 seqs the GDN mamba-state pool (block 784, mamba-page alignment)
+ weights 19.77 GiB + KV 7.13 GiB leave no headroom; at 4 seqs the
pool shrinks ~2 GiB and it runs clean (KV 7.39 GiB / 43,366 tok).
Not proven merge-caused (post-merge compile config registers many
more splitting_ops → plausibly larger inductor footprint).
**Production dense config is 4–8 seqs.** `BENCH_MAX_SEQS` knob added
to `_bench_gfx906.py`. (Later refined: serving servers also need
`--gpu-memory-utilization 0.93` with a warm inductor cache —
DEVLOG-spec-decode.md 2026-08-19 prefill/TTFT entry.)
