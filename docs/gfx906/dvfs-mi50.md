# MI50 (gfx906) DVFS / clock facts — benchmarking ground rules

Measured 2026-09-04, boot U, GPU0, ROCm 7.14. Scripts:
`/local/tmp/mtp1/syv3_dvfs_test.py`, `/local/tmp/mtp1/syv3_dvfs_gap.py`.

## The facts

| state | mclk | effective HBM2 BW |
|---|---|---|
| idle | **350 MHz** | ~358 GB/s (≈35% of peak) |
| any sustained kernel load | **1000 MHz** | ~1 TB/s theoretical (4096-bit @ 2 Gbps) |

- The boost is fast: mclk was at 1000 MHz within the first 0.5 s sample after
  load started, for BOTH a well-utilizing kernel and a poorly-utilizing one
  (ATen `torch.mm` at ~19% of peak still boosts).
- Per-iteration `torch.cuda.synchronize()` does **not** drop mclk between
  calls (sampled every 0.5 s through a per-sync bench: flat 1000 MHz, times
  flat to within 2%). So "hot" standalone numbers are real.
- COLD benches run right after idle ARE inflated ~3× by the 350 MHz memory
  clock. If a standalone number looks off vs an in-context one, sample
  `rocm-smi --showclocks` (mclk line) during the bench before drawing
  conclusions — do not assume downclocking, and do not assume it's absent.

## The standalone-≠production trap (SYV-3, 2026-09-04)

The biggest error this cost: benchmarking ATen `torch.mm` standalone and
concluding "the stock path leaves 3.3× on the table". Wrong on two counts:

1. **ATen mm is not what vLLM runs.** On gfx906 the Linear dispatch goes to
   custom kernels first — see `vllm/model_executor/layers/utils.py`:
   - n=1, k≤8192 → `_llmm1_tiny_m` (LLMM1)
   - n=1, long-k (5120×{10240,17408}) → `_gfx906_gemv_long_k`
   - n=2–4 → `_gfx906_spec_gemv_m4` = `dense_gemv_m4_gfx906`
   - kill-switches: `VLLM_GFX906_DENSE_GEMV=0`, `VLLM_GFX906_SPEC_GEMM=0`
2. **ATen mm at N=248320 K=5120 fp16 is pathological on MI50** (~194 GB/s ≈
   19% of peak, even at full clock), while the fork's GEMV family runs ~822
   GB/s in-context (≈80% of peak). The "gap" was two different kernels, not
   two clock states.

**Rule:** before benchmarking any "stock" op, read the `utils.py` dispatch and
identify which kernel actually runs (profiler cpu-op stream or CUDA-event
timers on the live path). A standalone ATen number is evidence about ATen mm
only.

## Missing rung CLOSED — direct measurement of the dispatched kernels (2026-09-04, post-review)

The root-cause chain above was originally established by elimination (DVFS
ruled out + dispatch code + cpu-op stream identity), never by directly timing
the *actual* production kernels standalone. External review flagged exactly
that gap; it is now closed (`/local/tmp/mtp1/syv3_gemv_standalone.py` — calls
the **dispatcher functions themselves**, not hand-picked ops, hot loop,
deciles, concurrent mclk sampling with a ≥900 MHz hard gate):

| path (drafter lm_head [248320, 5120] fp16) | standalone median | BW | vs in-context / prior audit |
|---|---:|---:|---|
| ATen `torch.mm` (the misleading reference) | 12.83 ms | 198 GB/s (19.8%) | — |
| n=1 `_llmm1_tiny_m` → LLMM1 | **3.098 ms** | **821 GB/s (82%)** | in-context 3.09 ms; DEAD-ENDS audit 3114 µs "HBM floor" — all three agree within 1% |
| n=4 `_gfx906_spec_gemv_m4` → `dense_gemv_m4_gfx906` | **4.997 ms** | 509 GB/s (51%) | verify-rows path; weight-read bound as designed for M=2–4 |

mclk gate: median 1000 MHz in all three timed windows → data valid.
**Conclusion hardened:** the production kernels are directly measured at/near
the HBM floor standalone AND in-context — SYV-3 shelve stands on direct
measurement, not elimination. The only remaining lm_head lever is quantization
(see DEAD-ENDS T1: int8 lm_head probe GO 1.93×), not kernel selection.

## PROPOSED: dispatcher-faithful standalone benchmark protocol (2026-09-04)

Proposed as the standard for any "is this op at the floor / how much headroom"
question on gfx906 (external review + SYV-3 lessons). Not yet promoted to
standard — needs validation at 2–3 context lengths before that.

1. **Benchmark the dispatcher, never a hand-picked op.** Import and call
   `vllm.model_executor.layers.utils._llmm1_tiny_m` / `_gfx906_spec_gemv_m4`
   (or whatever the live dispatch is) so standalone and in-context exercise
   the identical call path. Assert the result is not None (shape supported);
   a silent fallback invalidates the run.
2. **Hot loop, no per-iteration sync; decile stats, not just median.** 30+
   warmup iters; report p05/p25/median/p75/p95 of per-call CUDA-event times
   (block-of-10 event pairs amortize event overhead). Median hides bimodality
   (autotune re-triggers, prefill/decode mix) — the decile spread is the canary.
3. **Concurrent mclk sampling with a HARD gate.** Read
   `/sys/class/drm/card*/device/pp_dpm_mclk` every 0.25 s during each timed
   window — the active level is the line carrying `*`. **Per card, and not via
   `rocm-smi --showclocks`**: that call blocks while a deck is under load, its
   max-across-cards hides one deck's down-shift, and any single median hides the
   excursion that matters. Gate on a **time share**, not a median: FAIL LOUDLY
   if the median is < 900 MHz, **and** warn when ≥ 10 % of *loaded* time
   (≥ 400 MHz; the 350 MHz idle state excluded) is held at 800 MHz — a
   sustained 64k prefill measured **41–44 %** there while its median still read
   a clean 1000 (DEVLOG-fa-multibatch-prefill.md, 2026-10-06). Cold-clock data
   must not be silently reusable. (Discipline-only mitigation is how the ~3×
   cold artifact almost poisoned SYV-3.)
   Reference implementation: `_bench_gfx906.py::_MclkSampler`.
4. **Cross-validate against ≥2 independent anchors** before trusting: an
   in-context CUDA-event number and/or a prior audit figure. Agreement within
   ~1–2% across instruments = floor confirmed; disagreement = investigate
   queue-wait contamination (in-context module spans include stream wait) or
   kernel-path mismatch.
5. **Label evidentiary weight.** cpu-op-stream dispatch identity ≠ timing
   evidence. Conclusions resting on it say "identity confirmed; timing
   inferred" until rung 4 closes.

Reference implementation: `/local/tmp/mtp1/syv3_gemv_standalone.py`.

## Profiling methods — official vLLM vs our CUDA-event harness

Compared 2026-09-04 (boot U) against the upstream docs
(<https://docs.vllm.ai/en/latest/contributing/profiling/>). Three official
methods exist; status on THIS host:

1. **vLLM torch-profiler** — `--profiler-config '{"profiler":"torch",
   "torch_profiler_dir":...}'` + `/start_profile`/`/stop_profile` (or
   `vllm bench serve --profile`). Writes a real trace but with **zero GPU
   kernel events** (cpu_op only; verified 2026-09-04: 19 MB trace, 1.1M
   cpu_ops, 0 kernels — same in `--enforce-eager`) → same broken HSA tracing
   layer as rocprofv3. Still useful: the cpu-op stream names the exact
   custom-op dispatch per decode step (parse between
   `execute_context_*_generation_*` annotations) — this is how SYV-3 found
   that production runs the fork GEMV family, not ATen mm.
2. **Nsight Systems** (`nsys profile --trace-fork-before-exec=true
   --cuda-graph-trace=node --capture-range=cudaProfilerApi ...` +
   `vllm bench serve --profile`) — upstream's recommended low-overhead path
   for per-kernel timing under CUDA graphs. **NOT INSTALLED on this host**
   (no nsys binary). Untested; if the HSA tracing layer is broken for
   torch-profiler/rocprofv3 it may be broken here too, but nsys is a
   different capture stack — worth trying before assuming dead. Requires a
   user install (~100 MB, NVIDIA package on ROCm).
3. **cProfile helpers** (`vllm.utils.profiling.cprofile`) — CPU-side Python
   only; irrelevant to GPU kernel questions.

**Verdict (2026-09-04): the self-built CUDA-event forward-hook harness
STAYS.** It is the only method on this host that yields per-module *GPU*
time attribution (validated 0.4% vs wall clock, `phase_profile_results.md`).
Official methods cover a different question (which kernel family dispatches)
and are complementary, not replacements: torch-profiler cpu-op stream =
kernel identity; CUDA-event harness = kernel/module timing. Remove the
harness only if nsys is installed AND verified to produce GPU kernel events
on this ROCm build — re-test then and update this section.
