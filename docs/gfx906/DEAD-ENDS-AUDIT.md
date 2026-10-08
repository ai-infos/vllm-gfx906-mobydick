# DE-1 — dead-end register-spill / compiler-structural audit (2026-08-31)

Scope (roadmap **DE-1**, user request 2026-08-31): re-examine every row in
[`DEAD-ENDS.md`](DEAD-ENDS.md), classify HIP-kernel vs non-HIP, and for the
HIP-kernel rows determine whether the failure could be a compiler/structural
artifact — VGPR/SGPR pressure, register spills, occupancy loss, bad
vectorization — i.e. *fixable by restructuring* rather than dead by hypothesis.

**Bottom line up front: zero dead-ends failed from register spills or
measurable register pressure.** Every in-tree HIP-kernel dead-end was compiled
standalone for gfx906 with the repo's exact build flags and its HSA kernel
notes read back (VGPR/SGPR counts, spill counts, LDS). **Every kernel shows
`vgpr_spill_count = 0`** and modest VGPR usage (12–79). The failures were
what the original verdicts already said: HBM floor, isolated→serving transfer
flip, structural design trade-offs, Triton codegen gap, or scope caps. No
branch is opened; every HIP-kernel row is annotated `FULLY DEAD` in
`DEAD-ENDS.md` with its failure class.

## Method / instrument

- **Compiler:** `/opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch=gfx906
  -O2` (build parity: the build uses `-O2`) with the exact defines/includes
  from a real `_rocm_C` compile line (`/local/tmp/nh2c_clang_flags.txt`).
  Source = the **hipified `.hip` copies** under
  `build/temp.linux-x86_64-cpython-312/csrc/` (what the build actually
  compiles; the in-tree `.cu` files need the hipify pass for the torch header
  chain).
- **Read-back:** `-S` → AMDGPU assembly → per-kernel HSA notes
  (`.vgpr_count`, `.vgpr_spill_count`, `.sgpr_count`,
  `.group_segment_fixed_size`). Harness: `/local/tmp/spill_audit.py`
  (kept on durable storage; rerunnable).
- **Hardware constants (rocminfo-verified, this boot):** gfx906 = 60 CU ×
  4 SIMD, wavefront 64, `maxThreadsPerCU=2560` (40 waves/CU),
  `regsPerBlock=65536`, LDS/block cap 65536 B.
- **Caveat on the occupancy column:** the runtime
  `hipOccupancyMaxActiveBlocksPerMultiprocessor` returned a constant
  32 blocks/CU for every probe kernel (even at 256 forced VGPRs), so it is
  NOT a reliable per-kernel signal here — treat the harness's `waves/CU`
  column as an estimate from the register-file size, not ground truth. The
  team's own occupancy measurements (`DEVLOG-moe-opt.md`: "measured occupancy
  6–13 blocks/CU" via the same API on the MoE kernel; `DEVLOG-spec-decode.md`:
  "~130+ VGPRs → ~35% occupancy loss at KC=1024") are the calibrated anchors.
  For this audit only the **spill count** (exact, from HSA notes) and the raw
  VGPR/LDS numbers matter; neither shows pressure on any dead-end kernel.

## Measured register profile — every in-tree HIP-kernel dead-end

| Dead-end row | Kernel (compiled instance) | VGPR | SGPR | V-spill | LDS (B) | Failure class |
|---|---|---|---|---|---|---|
| gemm1 V1 full-K single-wave (G1, REJECTED 2.2×) | *reverted; not in tree* — design: 64 blocks × 1 wavefront, full K=2048 | n/a (not recoverable; no spill mechanism claimed) | | | | **structural**: 64 long 128 KB streams can't stay in flight; kills atomics but loses all transfer. Measured 60.4 µs vs 27 µs baseline. Not a register issue by construction (one wavefront per block, K-looped). |
| gemm1 NPT=2 z-split (G1, DEAD-END neutral) | `moe_gemm_q4_v2_kernel<512,4,256>` (the re-tile family it tuned) | **79** | 40 | **0** | 12416 | **transfer flip**: −186 µs/step census did not move wall-clock in eager *or* graph. 3rd consecutive G1 transfer failure. No spill, no occupancy loss (LDS-limited to ~5 blocks/CU — same regime as the shipped kernel). |
| gemm1 V3 fp32-scratch K-split (G1, closed w/o build) | never built | — | | | | **structural** (design): +184 µs/step scratch+launch on top of a design whose best point doesn't transfer. |
| gemm1 V4 halve CAS fan-in (G1, DEAD-END) | `moe_gemm_q4_kernel` family sweep | 51–93 across NPT/slice instances | 44 | **0** | 528–2112 | **structural** (measured monotone): CAS fan-in degrades monotonically as count rises (27.8→33.4 µs z=16→32). No spill at any point in the sweep. |
| S5 gemm2 V2 M=1 re-tile, lane-based cols (M1, NEUTRAL in-model) | `moe_gemm_q4_v2_kernel<512,4,256>` | **79** | 40 | **0** | 12416 | **transfer flip**: standalone 1.18× win (gemm2 K=512: 21.4→10.8 µs), in-model NEUTRAL — the recurring isolated→serving flip. Shipped default-OFF (`VLLM_GFX906_MOE_M2V2` flag later removed). LDS-limited occupancy (~5 blocks/CU) is a *property of the design* (512-thread CTA + per-wave partials), not pressure: zero spills, and the standalone win proves the kernel runs fine. |
| S2 dedicated M=1 topk kernel (M1, NEUTRAL) | `topk_softmax_m1_gfx906_kernel` | **37** | 16 | **0** | 0 | **transfer flip / mode-dependent**: standalone win, graph-replay loss. Bit-equal to generic chain (4/4). No spill; 37 VGPR = no pressure at all. |
| dense W4A16 purpose-built GEMV (D, REJECTED) | prototype `/tmp/bench/w4a16/` — **lost** (not in tree, /tmp wiped) | n/a | | | | **structural** (load layout): exllama's vectorized gptq_gemm load order wins on skinny GEMV shapes; the 256-thread rpt-rows × kchunk-k layout sits at 187–230% of floor vs exllama 87–97%. The devlog attributes it to *load layout*, not registers; and the prototype is gone, so a spill measurement is impossible. Rebuilding it to chase spills would be new kernel work with no evidence base — see verdict below. |
| dense K=5120 / lm_head GEMV (D, NEUTRAL) | `dense_gemv` LLMM1 family (in tree; measured at floor) | 22–65 across the i8/fp16 family instances recompiled today | 22–44 | **0** | 0–512 | **HBM floor**: 3114 vs 3128 µs (tie, ~98% of floor). No kernel change can beat the memory system. |
| FA V2 fused gather, 416 WG + barriers (FA, REJECTED — V1 wins) | `gather_paged_kv_q8_kernel_v2` vs shipped `_kernel` | **16** vs **12** | 52 vs 44 | **0** / **0** | 12 vs 0 | **wave-scheduling**, not registers: V2's 7× serving degradation (285 vs 42 µs) with only +4 VGPR and zero spills on either side. The "low-WG effect" is real but it is a scheduler/grid-shape effect, not compiler pressure — restructuring the kernel body cannot fix a grid that launches 416 workgroups; shrinking the grid to V1's shape *is* V1 (already shipped). |
| C1 stage-2 fused topk+align+count (C1, DEAD-END) | `moe_routing_fused_m1_gfx906_kernel` | **38** | 26 | **0** | 3072 | **transfer flip** (3rd instance): 28% faster per node isolated, −1.10% in serving. Mechanism already attributed by the stage comparison (removing nodes transfers; replacing the production topk in place does not). 38 VGPR + 3 KB LDS = no pressure. |
| A8W8 Triton `tl.dot` prefill GEMM (INT8, NO-GO) | *Triton, not HIP* — codegen gap: 19% of the dot4 record vs hipBLAS fp16's 57% | n/a | | | | **Triton compiler/codegen gap** (out of scope for a HIP restructure; reopening requires beating W4A16 head-to-head, which the next row shows is scope-capped). |
| Hand-written HIP int8 A8W8 GEMM (INT8, DEAD-END scope-capped) | never built (free-scope arithmetic) | — | | | | **scope cap**: BF16 mass is only 7.9–8.9% of per-token GEMM MACs → even a perfect kernel wins ~3% wall; int8-ing the int4 mass costs +11.4/+15.8 GB. (The same tensors ARE the decode lever — that's T1, open in DEAD-ENDS.md.) |

Non-HIP rows (no branch possible by definition): P2-0b zero-fill
(descoped-racy design), fused cast micro-copies (upstream-attributed),
llama.cpp Q5_K_XL baseline (36 GiB > 32 GB memory fit), LEGACY=0 B=1 flip
(serving-harness interaction, not a kernel — the Q8 gather is 22–45% *faster*
per step; the gap is graph-node/serving cost), M6 Part A planar-Q8 repack
(loader hygiene merged; flip question closed by same-boot A/B), P3-2(b)
K-split GEMV (M=1 latency/launch-bound — occupancy was explicitly ruled out
in the original verdict, consistent with today's zero-spill measurement of
the family), GEMV V2 RPT=2+kc=4096 shape-rule (N=1024 pathological case is a
shape-scheduling artifact, kept per-shape).

## Verdicts

**Fixable by compiler/structural restructure → open branch: none.**

The audit's question was "which dead-ends might have failed due to register
spills or similar compiler issues?" and the answer is **none**:

1. **Spills:** `vgpr_spill_count = 0` on every measurable in-tree kernel,
   including the two primary suspects (FA V2: 16 VGPR; gemm1 re-tile family:
   ≤93 VGPR). The one historical high-pressure case in the devlogs (runtime-M
   m4 at ~130+ VGPR, `DEVLOG-spec-decode.md`) was already fixed by the M-templated
   kernel and is *shipped*, not a dead-end.
2. **Occupancy/pressure:** no dead-end kernel sits near the register-file
   limit; the highest (gemm1 v2 at 79 VGPR) is LDS-limited by design, and its
   failure was wall-clock transfer, not throughput-per-wave.
3. **What actually killed them** (all recorded in the original verdicts, now
   compiler-confirmed as non-pressure): HBM floor (K=5120/lm_head), the
   isolated→serving transfer flip (gemm1 NPT=2, S5 V2, S2 topk, C1 stage-2 —
   four instances of one recurring lesson), structural design trade-offs
   (gemm1 V1 stream depth, V4 CAS fan-in, W4A16 GEMV load layout, FA V2 grid
   shape), scope caps (int8 A8W8 prefill), and the Triton codegen gap (A8W8).

**Marked `FULLY DEAD` in `DEAD-ENDS.md`:** all HIP-kernel rows above that were
already REJECTED/DEAD-END/NEUTRAL — annotated with the compiler evidence so a
future "could we fix the spill?" question is closed by data, not re-litigation.
The two rows that stay *open* are unchanged: T1 (int8 W8A16 decode mass —
byte-side lever, probe GO) and T5 (dot8 prefill — nothing may be built before
the P3a probe). Neither is a dead-end; both were correctly excluded from the
`FULLY DEAD` sweep.

**No branch opened.** The only row where "restructure" could theoretically
mean something new is the W4A16 GEMV prototype (load-layout loss to exllama) —
but its source is gone, no spill evidence exists, and the measured gap
(187–230% vs 87–97% of floor) is a layout-class problem where exllama's
tuned kernel is already in production; rebuilding a competitor without a new
hypothesis would violate the repo's "no build before probe" rule. If it ever
reopens, it reopens as a fresh design item with a micro-bench gate, not as a
spill fix.

## Artifacts

- Harness: `/local/tmp/spill_audit.py` (rerunnable: `python3 spill_audit.py
  <file.hip> [-I… -D…]`; reads HSA notes from `-S` output).
- Raw per-kernel tables: `/local/tmp/de1_audit_results.txt`.
- Build-flag snapshot used: `/local/tmp/nh2c_clang_flags.txt`.
- NH2C (this audit's side context): the new int8 GEMV family also compiles
  spill-free — 22–65 VGPR across all RPT/KC/M instances, `vgpr_spill_count=0`
  everywhere.
