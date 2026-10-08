# MoE C1 — M=1 routing pipeline fusion — stage 1 SHIPPED, stage 2 DEAD-END

**VERDICT:** SHIPPED (stage 1) · **GATE:** in-process
`_bench_gfx906.py`, Qwen3.5-35B-A3B-AWQ, TP=1 B=1, pp2048/tg256, graph
regime, 4 samples/arm, `VLLM_GFX906_ALIGN_M1` 0 vs 1, same boot (O),
2026-08-29

## HYPOTHESIS

If the ~0.8 ms/step M=1 routing cost is per-kernel graph-node latency
(rather than the raw node floor or kernel work), then replacing the
two-kernel `moe_align_block_size` + `count_and_sort_expert_tokens` pair
with one 1-CTA kernel (120 → 80 nodes/step) saves ~200 µs/step in serving
wall clock, not just in isolated graph replay.

## What was done

- **Structural probe** (`benchmarks/kernels/gfx906/c1_routing_structural_probe.py`):
  the production M=1 routing chain is **3 kernels/layer** (topkGating +
  align 2-block + count_and_sort = 120 graph nodes/step ≈ 772–800 µs,
  M-independent — latency-bound). Node floor (dummy `add_`): 1.3 µs/node.
  Per-node: topk 11.8 µs, align 3.8, sort 3.8.
  S2 re-check in the same regime: the dedicated M=1 topk is 28% faster per
  node in isolated graphs (8.5 vs 11.8 µs) yet **loses −1.03% in serving**
  (56.36 vs 56.95 t/s, 4 samples/arm) — reproducing the sprint's
  −0.95 t/s graph-regime loss. Isolated-graph kernel numbers do not predict
  production-graph cost; they can flip sign. (S2 was a kernel *swap*;
  node count unchanged.)
- **New kernel** `csrc/rocm/moe_align_m1_gfx906.cu`: one 128-thread CTA,
  LDS-atomic counts + 5-step `shfl_down` warp scan over the 256 expert
  counts (exclusive prefix = total − suffix, total ≡ 8 slots), lane-atomic
  placement. No grid work, no D2H, no global atomics → capture-safe.
  Bit-equal to the two-kernel chain (26/26 tests, tie-heavy included;
  within-expert slot order matches the production single-warp atomic order).
- **Call site** `gfx906_w4a16_moe.py`: gated branch (M=1, topk=8, E=256,
  block_size=1, int32, no expert_map), flag `VLLM_GFX906_ALIGN_M1`,
  **default ON** after the gate PASS; `=0` to opt out.

## Evidence FOR

- Isolated graph (launch-regime evidence): fused align **2.0 µs/node vs
  3.8** for the pair (79.9 vs 300.7 µs / 40 layers) → ~224 µs/step
  predicted.
- **GATE — serving A/B (boot O):**

  | run | off | on | Δ |
  |---|---|---|---|
  | 1 | 56.53 t/s | 57.51 t/s | +1.73% |
  | 2 (back-to-back) | 56.66 t/s | 57.33 t/s | +1.18% (207 µs/step) |

  Back-to-back step: 17.649 → 17.442 ms — 207 µs/step, within 8% of the
  isolated prediction; both sessions consistent; within-arm spread ≤0.7%.
- Narrowed suite 524 passed (gfx906 MoE gemm, align-m1, generic align,
  topk) both with flag off and with the default flipped on.

## Evidence AGAINST

- None in serving. The S2 flip is a warning about *kernel-swap*
  mechanisms, not a counterexample to node removal — the back-to-back gate
  shows node removal transfers (207 vs 224 µs/step predicted).

## Stage 2 — DEAD-END in production (same boot, 2026-08-29)

Fused topk + align + count into ONE kernel (120 → 40 nodes/step) per the
stage-2 plan: `csrc/rocm/moe_routing_fused_m1_gfx906.cu` (S2's bit-exact
topk phase + stage-1's LDS count/scan/place in one 128-thread CTA).
Router-side fused mode in `FusedTopKRouter._compute_routing` (flag
`VLLM_GFX906_ROUTING_FUSE_M1`, **default OFF**); the (sorted_token_ids,
expert_ids, ntp) meta is plumbed `moe_runner → forward_modular →
FusedMoEModularMethod.apply → FusedMoEModularKernel.apply →
Gfx906WNA16Experts.apply` (optional kwarg, dropped when the quant method
or impl doesn't declare it — unquantized/ignored layers re-align from the
same topk_ids, which is exact since the fused kernel's topk phase is
bit-equal). Tests `tests/kernels/moe/test_moe_routing_fused_m1_gfx906.py`
27/27: bit-equal to the 3-kernel production chain on all six outputs
(24 seeds × renormalize, tie-heavy included), graph capture/replay,
router dispatch + gate shape checks.

**Gate — serving A/B (boot O, A-B-A back-to-back, 4 samples/arm):**

| arm | t/s |
|---|---|
| A (stage 1 only = current default) | 57.42 |
| B (routing fuse ON) | **56.79 (−1.10%, −175…−195 µs/step, runs 1–2)** |
| A2 (control after B) | 57.46 |

A2 ≈ A ⇒ not boot drift: the flip is real, despite the isolated-graph
prediction of +152 µs/step (40-node fused routing 400.0 µs = 10.0
µs/node vs stage-1's 80-node 552 µs ≈ 13.8 µs/layer).

**Verdict: DEAD-END — third confirmation of the S2 flip pattern.** The
stage comparison pinpoints the mechanism: stage 1 (SHIPPED, +1.2–1.7%)
REMOVED redundant kernels while keeping the proven topk; stage 2
REPLACED the proven production topk with a new 1-CTA kernel — the exact
S2 failure mode. **Node removal transfers to serving; replacing a
working production kernel does not** (S2 topk: 28% faster isolated →
−1.0%; stage-2 fused routing: 28% faster isolated → −1.1%).

State: kernel + plumbing + tests committed behind the OFF flag
(production behavior unchanged; re-runnable for future kernel-design
iterations). Stage 1 remains the C1 outcome; C1 is closed.

## Notes / logs

- Logs (boot O; /tmp wiped on reboot):
  `/local/tmp/c1_structural_run{1..5}.log` (run5 = stage-2 probe with
  routing_fused arm), `/local/tmp/c1_ab_default.log`,
  `/local/tmp/c1_ab_topkm1.log`, `/local/tmp/c1s1_ab_arm{A,B,A2,B2}.log`,
  `/local/tmp/c1s2_ab_arm{A,B}.log`, `/local/tmp/c1s2_ab2_arm{A2,B2,A3}.log`.
- The UnquantizedFusedMoEMethod TypeError found in the first A/B arm
  (ignored/unquantized layers have a real router but a method without the
  kwarg) is guarded by the signature check in `RoutedExperts` and
  `FusedMoEModularKernel`; the dynamo-trace exclusion in the router gate
  keeps the fused branch out of compile tracing.
