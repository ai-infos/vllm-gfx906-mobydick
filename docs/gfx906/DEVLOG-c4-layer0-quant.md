# C4 — load-time int4 quantization of the unquantized first MoE layer

**Date:** 2026-09-01 · **Branch:** `gfx906/c4-layer0-quant` · **Gate:** GO (measured)

## Symptom / target

Qwen3.5-35B-A3B-AWQ lists `model.layers.0.` in `modules_to_not_convert`, so
layer 0's routed experts load as fp16 and run on the generic Triton
unquantized MoE path. On gfx906 that path is ~4× slower per call than the
custom W4A16 kernel because it reads 4× more weight bytes (memory-bound at
M=1). C2's profile: layer-0 Triton = **740 µs/call** vs layers 1–39 gfx906
W4A16 = **182 µs/call**; net saving ≈ **558 µs/step ≈ 4.8%** of the ~11.7 ms
step — above the ~1.8% serving A/B noise floor, so a real gate was
mandatory.

## Approach (C4-A; C4-B rejected at scoping)

Load fp16 exactly as the unquantized path does, then in
`process_weights_after_loading`:

1. Quantize each expert to int4 with the **asymmetric AWQ convention**
   (`(q − zp) * scale`, group size from the checkpoint's AWQ config).
   Codepoints are chosen against the **stored fp16 scale** (the value the
   kernel actually applies), so reconstruction error is bounded by half a
   step of that scale with no extra rounding term.
2. Emit the MoeWNA16 N-first uint8 layout that the gfx906 repack
   auto-detects (`w [E,N,K/2]`, `scales [E,N,G]`, `qzeros [E,N/2,G]`).
3. Free the fp16 storage via `register_parameter(name, None)` (not just
   dropping references — unregisters from the module registry so graph
   capture / reloads never see stale fp16; ~1.5 GiB back).
4. Delegate to `MoeWNA16Method.process_weights_after_loading` → shared
   gfx906 repack + kernel setup. **No new kernel code.**

Gated by `VLLM_GFX906_QUANT_LAYER0_MOE=1` (default off until soak) —
quantizing a layer the checkpoint author deliberately left unquantized is a
quality trade-off.

## Correctness gates

- **Unit 8/8** (`tests/kernels/moe/test_c4_layer0_quant.py`): bit-exact
  packing vs an independent reference (exllama-shuffle orientation
  `[E, N, K/8, 8]`, int32 zp shifts to match kernel wrap semantics),
  round-trip error bounds, cross-check against the production
  `_repack_w4a16_gfx906_expert`.
- **PPL:** off 15.9531 → on 15.9929 (Δ +0.04 — noise; gate threshold < 0.5).
- **Coherence/fingerprint:** greedy serving fingerprint bit-identical across
  arms (`d2e5262183c6b92f`, 256 tokens); coherent text all samples.
  Notably int4 layer-0 flips no decode token on this workload.

## Serving A/B (the gate)

M=1 decode, pp2048/tg256, 3 reps/arm, same boot, `moe_multireq_ab.py`
(FULL_DECODE_ONLY):

| arm | layer-0 path | t/s (mean ± std) | fingerprint |
|---|---|---|---|
| OFF | fp16 → Triton | 84.95 ± 0.08 | `d2e5262183c6b92f` |
| ON | int4 → gfx906 WNA16 | **87.51 ± 0.33** | `d2e5262183c6b92f` |

**+2.56 t/s = +3.0%** — above the noise floor (C3's wash was 0.3%).
Firing confirmed in the ON-arm log: `C4: quantizing 256 routed experts to
int4` + `GFX906_HIP WNA16 backend`; OFF arm shows layer-0 on `TRITON
Unquantized`.

## Pre-merge review (self + Claude CLI, validated)

Claude's first pass cited a nonexistent `moe_runner.py` path; the real file
is `fused_moe/runner/moe_runner.py`, and its reads of the installed method
were checked line by line:

1. **REAL — runner-visible state not synced.** The runner reads
   `moe_kernel` (→ `_fused_output_is_reduced`, `supports_internal_mk`,
   `mk_can_overlap_shared_experts`, `topk_indices_dtype`) off
   `self._quant_method` (C4's method), not the delegate. Fixed: sync
   `moe_kernel` / `moe_quant_config` / `experts_cls` after successful
   repack.
2. **REAL — `supports_eplb` inherited True** from
   `UnquantizedFusedMoEMethod`; the active WNA16 path does not support EPLB.
   Fixed: property follows the active path (WNA16 → base False; unquantized
   fallback unchanged).
3. **REJECTED — "fp16 freed before the delegate's N%8 gate".** The gfx906
   repack branch (`_process_weights_gfx906`) has no N%8 early-return that
   would leave a half-converted layer; shape gates already ran in C4's own
   preconditions (K divisibility), and the A/B arm proved the full path.

After fixes: unit 8/8 re-run green, A/B re-run green.

## Open

- Default-on decision after soak (keep opt-in until ≥1 full-day serving
  window).
- T1 (int8 W8A16 family, PROBE GO) targets the same unquantized mass at a
  different bit-width; if it lands, C4's layer-0 quantizer may be superseded
  or subsumed — revisit then.
