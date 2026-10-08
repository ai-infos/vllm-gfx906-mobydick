# Ornith WNA16 MoE (asymmetric compressed-tensors W4A16) — the CT
# asymmetric expert checkpoint class now loads on gfx906 and runs the
# custom kernel: 65.03 t/s harness A/B (decode-only 81.1), 18.6× over
# the Triton fallback arm.

Model: `cyankiwi/Ornith-1.5-35B-A3B-AWQ-INT4` (Qwen3.5-MoE VLM, 40 layers,
256 experts × 8 top-k, moe_inter 512, 24.33 GiB checkpoint — the same
decode-shape class as the flagship Qwen3.5-35B-A3B-AWQ). Quant:
compressed-tensors **asymmetric** W4A16, group-32, int8 zero points,
pack-quantized, routed experts only. Branch `gfx906/moe-ct-asym-zp`
(base `gfx906/v0.28.0rc2` @ 67ae6c3f96), MI50 GPU0, 2026-08-25.

**VERDICT:** SHIPPED (branch finalized 2026-08-25, unmerged) ·
**GATE:** serving wall-clock A/B, `_bench_gfx906.py`, graph
(`BENCH_EAGER=0`), pp2048/tg256, maxlen 2816, `BENCH_MAX_SEQS=8`,
gpu_util 0.95, 4 samples/arm

## 2026-08-25 — onboarding + CT-asym W4A16 MoE support

## HYPOTHESIS

If the on-disk zps of a compressed-tensors asymmetric pack-quantized
W4A16 checkpoint are the kernel's native K-first `[E, G, N/8]` int32
layout (as the `is_transposed` MoE loader presents them), then opening
the oracle gate + a pass-through repack — no kernel change — lets
Ornith load on gfx906 and decode at flagship-class speed; if not, the
zp path in the repack or the kernel needs real work.

## What was done

Pre-change state on stock v0.28.0rc2: **Ornith cannot load at all** —
`AssertionError: Only symmetric quantization is supported for MoE`
(`compressed_tensors_moe_wna16.py:127`). CT-asymmetric MoE was
unreachable on this platform: the Marlin branch is CUDA-only, and the
non-Marlin branch asserted symmetric for every other backend.

Changes (all Python; no kernel work — CT-asym dequant `(q − zp)·scale`
is the kernel's existing AWQ-with-zp convention, `zero_offset=0`):

1. `oracle/int_wna16.py` — `_gfx906_asym_ct_reason()`: accept
   QuantizationArgs asymmetric 4-bit, static scales, GROUP strategy,
   group size ∈ {32, 128}, no g_idx actorder, for GFX906_HIP; AutoGPTQ
   still rejected ("GPTQ-style zero-point encoding is not supported").
2. `oracle/int_wna16.py` — `_repack_w4a16_gptq_kfirst_layout`: zps
   validated against `[E, G, N/8]` (wrong shape fails closed) and
   passed through bit-exact (`.to(int32).contiguous()`).
3. `Gfx906WNA16Experts._supports_quant_scheme`: accept
   `kInt4StaticAsym` / `kInt4Static32Asym`.
4. `CompressedTensorsWNA16MoEMethod.__init__`: the symmetric assert is
   scoped to zp-consuming backends (TRITON, GFX906_HIP); any other
   backend still fails closed for asymmetric.
5. `TritonWNA16Experts._supports_quant_scheme`: accept the asym keys
   (the Triton WNA16 kernel consumes `w1_zp`/`w2_zp` in both gemm
   passes) — required for the A/B baseline arm.
6. `oracle/int_wna16.py` TRITON branch of
   `convert_to_wna16_moe_kernel_format` — `_repack_qzeros_kfirst_for_triton`:
   the passthrough read a **transposed** layout (garbage text before
   the fix); the kernel indexes column n from word `n // 2` (axis 1),
   nibble `(n % 2) * 4`, group `g` (axis 2) — i.e. `[E, N/2, G]`,
   2 zps per int32. Repack converts the checkpoint's 8-zp-per-word
   K-first packing; result stored physically `[E, G, N/2]` and returned
   as a transposed view (coalesced along the kernel's n-walk).
7. Probes: `BENCH_MODEL` + `BENCH_MOE_BACKEND` env support in
   `benchmarks/kernels/gfx906/ppl_probe.py` and `greedy_probe.py`.

One load-time GPU wedge on the first graph-mode launch of the boot
(22:19 UTC, GPU0, BACO reset, self-recovered) — recorded in
`degradation.md`/`degradation_details.md`; retried once per house
recipe, clean.

## Evidence FOR

- Unit: oracle suite 34/34; GPU kernel suite 51/51 (incl. 8 new
  `gptq_kfirst` E2E asym cases + repack pass-through/shape-failure
  tests).
- Smoke (eager, max_len 4096, gpu_util 0.95): coherent text on both
  arms. gfx906: "Paris. Paris is located in the north-central part of
  the country, on the Seine River…" triton pre-repack: garbage
  (`::::::::ALES::::::::对付::::::zi`). KV 123,699 tokens @4096.
- PPL probe (12 prompts, 359 tokens, 0 top-20 misses, eager):
  gfx906 **16.6716** / **16.6876** (run-to-run 0.016); triton
  **16.4539**. Inter-arm 0.218 ≈ fp16-noise-level divergence between
  the two kernel implementations (cf. the gemma-4 ΔLP note), not a
  correctness failure; both arms coherent.
- **GATE — serving A/B (graph, config above):** gfx906 arm
  **65.079 / 65.063 / 64.999 / 64.995 t/s** (mean 65.03, band 0.08 %);
  triton arm **3.498 / 3.501 / 3.500 / 3.500** (mean 3.50). Ratio
  **18.6×**.
- TTFTI probe (same graph config, 2048-prefill + 256-decode split):
  gfx906 TTFT **0.77 s**, decode **12.3 ms/tok = 81.1 t/s**; triton
  TTFT **4.38 s**, decode **267.6 ms/tok = 3.7 t/s**. Decode-only A/B
  **21.9×**; prefill 5.7×.
- Class parity: 65.03 harness vs the flagship's 67.39 record — the
  asym-zp path performs on par with the symmetric path on identical
  decode shapes.

## Evidence AGAINST

None against the fix itself. The Triton arm's slowness (Entry 2) is a
property of the upstream Triton kernel's `has_zp` branch, not of this
change — and that path was unreachable before this work.

## Why it worked

The loader already presents the checkpoint zps in the kernel's native
layout, so the gate + pass-through is sufficient; the only real
layout work was for the Triton baseline arm (Entry 2).

## Interactions / superseded-by

- The Triton int4-zp path is now a *reachable* (was: assert-failed)
  fallback for CT-asym checkpoints on ROCm; its gfx906 slowness is
  tracked below and cross-linked to `moe-decode-roadmap.md` U4.
- Precedent: gemma-4 no-zp CT W4A16 onboarding (`DEVLOG-gemma4-*.md`,
  1.79× over Triton) — same oracle/repack machinery, symmetric branch.

VERDICT: SHIPPED

## 2026-08-25 — Triton W4A16 int4 `has_zp` branch is pathologically slow on gfx906

## HYPOTHESIS

The Triton arm of the A/B is a fair no-zp-equivalent baseline — i.e.
within ~2× of the no-zp class (gemma-4 Triton baseline 37.8 t/s).

## What was done

Measured the A/B baseline arm; when it read 3.50 t/s (harness) / 3.7
t/s (decode-only), decomposed with a TTFT/decode probe and a 9B dense
control (`cyankiwi/Qwen3.5-9B-AWQ-INT8-INT4`, same GDN-Triton-fallback
stack, no MoE, same boot/config).

## Evidence AGAINST

- 9B dense control (no MoE): decode **13.7 ms/tok = 73.2 t/s**,
  TTFT 0.61 s → the stack, GDN Triton fallback, and host state are
  healthy on this boot (the canary concern was refuted by the
  healthy dense control + the 81.1 t/s gfx906 arm).
- The excess over the control concentrates in the 40 MoE blocks:
  ~6 ms/block in the Triton zp path vs ~0.2 ms/block class.
- **Layout-independent:** the repack was measured with both physical
  zp layouts (contiguous `[E, N/2, G]` and transposed-view
  `[E, G, N/2]`) — identical 267–270 ms/tok. Not a storage-layout
  artifact of this work.
- Values are correct (PPL 16.4539, coherent text) — pure speed
  pathology in the `has_zp` branch of `fused_moe_kernel_gptq_awq` on
  gfx906. Suspect: the per-element data-dependent shift
  `(b_zp >> (offs_bn % 2) * 4) & 0xF` over `[BLOCK_K, BLOCK_N]` blocks
  on triton-hip/CDNA1 — **unmeasured mechanism**.
- Prefill affected too (TTFT 4.38 s vs 0.77 s) — consistent with the
  same branch at large M.
- Not a regression: the path was unreachable before this work (the CT
  scheme asserted symmetric), so this is new visibility.

## Why it failed

Upstream Triton kernel characteristic on gfx906; out of scope for the
backend fix (the whole point of the work is the fast custom kernel,
which the oracle selects by default).

Refrigerated residue: a standalone microbench of
`fused_moe_kernel_gptq_awq` with `has_zp` True/False at M=64
(256 experts, w13 N=1024/K=2048) would pin the mechanism in minutes —
worth doing only if the Triton zp path ever becomes a supported
fallback on gfx906.

VERDICT: OPEN

## 2026-08-26 — code-review fixes (g_idx actorder gate + qzeros repack
fail-closed + capability constant + doc expert-count)

## HYPOTHESIS

An independent high-effort code review (`/tmp/moe-ct-asym-zp-code-rev-claude.md`,
folding a second review) found the Triton fallback's g_idx gate and the
qzeros repack could silently mis-dequant on checkpoint shapes this diff
made newly reachable. If so, the gate should be a single source of truth
and the repack should fail closed, not patch each site in isolation.

## What was done

- **Finding 1 (real, blocking) — Triton g_idx gate missed `dynamic`.**
  The Triton gate checked `actorder == "group"` only, but
  compressed-tensors `ActivationOrdering` has four distinct members
  (`group`, `weight`, `dynamic`, `static`) with **no** alias collapsing
  `dynamic`→`group` (verified against the installed `compressed_tensors`
  package: `qa.actorder == 'group'` is `False` for a dynamic config). An
  asymmetric CT W4A16 checkpoint with `actorder=dynamic` was rejected
  from GFX906_HIP, fell through Marlin (unconditionally False on ROCm),
  and landed on TRITON with the actorder check silently passing → silent
  mis-dequant (missing g_idx reordering). This diff is what made asym CT
  reachable on TRITON at all, so it introduced the path.
- **Finding 3 (root cause) — three near-duplicate g_idx checks drifted.**
  Factored the shared predicate into
  `_gidx_actorder_reason(quant_config, family)` (single source of truth:
  `group`/`dynamic` carry a runtime g_idx the kernels lack; `weight`/
  `static` are natural-order safe), now used by the gfx906 symmetric/no-zp
  gate, the asymmetric-CT gate, and the Triton gate. The Triton gate
  message becomes `"the Triton WNA16 MoE backend does not support g_idx
  activation ordering"` (still matches the `"activation ordering"`
  test substring).
- **Finding 2 (real, lower reachability) — qzeros repack silently
  accepted input it didn't validate.** `_repack_qzeros_kfirst_for_triton`
  derived the output width from the tensor's last dim with no check, so a
  qzeros tensor in an unexpected layout/packing (a different quantization
  source's convention) would mis-dequant silently. It now takes the
  weight's output width `n_out` (K-first axis 2) and fails closed unless
  `dtype==int32`, `words == n_out//8`, and `n_out % 8 == 0`. Note: for the
  two supported sources the packed width is in fact consistent (both
  auto-gptq `pack_cols2ints` and compressed-tensors pack-quantized store
  8 zps per word along N, and the fused w13 width `2*inter` is handled
  because the function derives it from the tensor) — the review's
  "wrong shape for auto-gptq w13" analysis was re-derived and does not
  hold; this is a fail-closed guard, not a behavior change for the
  supported path.
- **Finding 4 (maintainability) — hardcoded backend tuple in the CT
  assert.** Moved the zp-capable-backend set to a single source of truth,
  `WNA16_BACKENDS_WITH_STORED_ZP = frozenset({TRITON, GFX906_HIP})`, in
  the oracle; `CompressedTensorsWNA16MoEMethod.__init__` now references
  it instead of a literal tuple, so a future zp-capable backend is
  declared once.
- **Finding 6 (docs) — expert count.** Ornith is **256** experts
  (verified against the live checkpoint config.json: `num_experts=256`,
  `num_experts_per_tok=8`, `moe_intermediate_size=512`), not 128 as the
  devlog stated. Corrected the model line and the refrigerated-microbench
  shape. A/B perf numbers are unaffected (they ran against the real
  checkpoint).

Rejected/deferred review items: Finding 5 (repack's `[E,G,N]` int32
intermediate) is load-time-only transient memory — deferred per the
review's own "optional cleanup, only if a practical OOM". The
`pytest.mark.gpu` marker item: this file has no such markers anywhere and
its sibling GPU tests in `tests/kernels/moe/` don't use them either, so
adding it would not match convention.

## Evidence FOR

- Oracle unit suite: **40/40** (was 34; +TRITON `dynamic` reject,
  +TRITON `weight` asym accept, +repack fail-closed layout cases). The
  new TRITON `dynamic` case is the regression test for Finding 1 — it
  fails on the pre-fix gate (`actorder == "group"`).
- gfx906 GPU kernel E2E suite: **51/51** (one transient
  `M2-bm1-N1536-gs128-gptq_kfirst_sym` failure on a random-seed run
  re-passed in isolation and on full rerun — pre-existing flake,
  unrelated to this diff, which touches no gfx906 kernel path).
- Ornith real-checkpoint smoke (eager, gpu_util 0.85): **auto
  (GFX906_HIP) arm** — loads and decodes coherently ("Paris is the
  largest city in France..."); **explicit `moe_backend=triton` arm** —
  loads and decodes coherently ("Paris. A. True..."), exercising the new
  repack width validation end-to-end on a real asymmetric checkpoint.
- No behavior change on the supported path: the gfx906 gate and the CT
  repack pass-through are logic-identical (only the actorder predicate is
  now shared); the Triton gate and the repack strictly add
  rejection/validation. The 65.03 t/s A/B (Entry 1) is unaffected.

## Evidence AGAINST

None. The one GPU-suite failure was a seed flake that re-passed; the
supported-path logic is unchanged.

## Verdict

**VERDICT: SHIPPED.** The review's blocking finding (Triton g_idx gate
missing `dynamic`) is a real silent-mis-dequant path introduced by this
diff and is fixed by construction via the shared
`_gidx_actorder_reason` helper (Finding 3); the repack now fails closed
on unexpected layouts (Finding 2); the zp-capable-backend set is a single
source of truth (Finding 4); the doc expert count is corrected (Finding
6). GATE: oracle unit suite (40/40) + gfx906 E2E (51/51) +
real-checkpoint smoke on both the auto (gfx906) and explicit-triton arms
— all green.

## Files

- `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py`
- `tests/quantization/test_moe_wna16.py`
- `docs/gfx906/DEVLOG-ornith-wna16.md`
