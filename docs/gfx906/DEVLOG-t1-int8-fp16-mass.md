# T-1 int8-mass (drafter lm_head int8-W8A16) + T-1.5 (M≤8 kernel) + A5 (method-layer strip)

**VERDICT:** DEAD-END (T-1 NOT PASS at k=2 AND k=4 at the serving gate;
A5 NO-WIN; method layer removed) · **GATE:** serving A/B, same-boot,
greedy, mtp3/mtp4 arms (Qwen3.8-27B-AWQ, TP=2)
**Branch:** gfx906/fa-decode-fp16 (`e1a57e9a17..a0bb358670`)
**Full detail:** this file pre-rewrite at
`git show 6bf288d1b7:docs/gfx906/DEVLOG-t1-int8-fp16-mass.md` (55 KB);
kernel preserved as W8A16 infrastructure (`dense_gemv_i8_m4_gfx906`,
M≤8 since T-1.5).

## 2026-09-04/05 — HYPOTHESIS: int8-quantize the MTP drafter's unquantized
lm_head mass (849 MB plain-BF16 across 15 mtp.* tensors) → the drafter's
per-step lm_head GEMM gets cheaper → ~+5–8% decode on Qwen3.8

## What was done

Quality gate FIRST (real captured hidden states): **0/120 argmax flips,
KLD ≤ 2.75e-4 nats, per-row margin ≥7×** — structurally speed-only under
greedy verify. Implementation: Int8W8A16 quant methods + load-time
transform, m4 CUDA kernel (M≤4, later M≤8), env-gated
`T1_INT8_MASS=1` default OFF; fc rows excluded (dominant-element rel-L2
12.5%) behind a sub-flag. Root-caused + fixed en route: graph-break
via the bare-Triton M=1 path (`da4f8bc431`), AOT cache-key collision
(`84356a4f59`), M>4 per-call dequant (`7929085e03`), VRAM transients
halved, kchunk TP-sharding.

## Evidence (serving A/B — the gate)

- **k=2: NOT PASS** — +6 ms/step regression → +3.1 ms/step after the
  graph-break fix. Still not negative.
- **k=4: NOT PASS** — the "+15.4/17.6/18.4% DECISIVE WIN" was a
  single-rep cold-start misread; the 9-rep re-test measured **parity
  −1%** with **+~2 GiB VRAM**.
- Wedges #18–#22/#25 (#20/#21, #22: T-1 boot attempts in the chronic
  weight-load family) — not T-1 faults.

## Why it failed (mechanisms)

1. **The ceiling was acceptance-dependent**: M = accepted+1 ranges
   3–9 at k=4, so the int8 kernel's small-M GEMV regime (where dequant
   overhead dominates) never got the large-M bandwidth win.
2. **Fixed costs the paper ceiling missed**: launch/graph-break recovery,
   the M>4 per-call dequant (+18 ms/step before caching), dispatch/kchunk
   hoists — each fix recovered ground but never crossed zero.
3. **The bf16 GEMV baseline is strong at these shapes** (HBM-floor,
   max-ilp-tuned): int8 halves weight traffic but adds dequant + split
   combine that cancel it at M≤9.

## 2026-09-06 — T-1.5: the M≤8 kernel extension KEPT as infrastructure

The int8 GEMV (`dense_gemv_i8_m4_gfx906`) extended M≤4 → M≤8 (T-1.5):
NO-WIN at k=4 in-tree, but kept committed as W8A16 infrastructure —
consumed by the opt-in W8A16 channel-dequant quant scheme
(`compressed_tensors_w8a16_channel_dequant.py:416`). Kept in the
closure (not T-1-exclusive; the scheme is its own caller).

## 2026-09-07 — A5 (method-layer strip): CLOSED NO-WIN, layer removed

Stripping the method layer (head-only mode, M-aware kchunk) measured
NO-WIN at k=4 (−0.40/−0.44/−0.13%, token-identical). The k=7 stacking
condition died structurally: k=4 itself loses on the real payload and
B≥2 forces M = B×(k+1) = 10 > 8, off the in-kernel path. Method layer
(`t1_int8_w8a16.py`) **removed** at `e289ff17dc`; the kernel stays.

## 2026-09-13 — final trace removal (branch closure)

The last in-tree reference (the decorators.py AOT cache-key factor —
no-op when T-1 off) removed at branch closure. The dense_gemv M≤8
kernel stays (see T-1.5 above). Post-closure:
`grep -rn T1_INT8_MASS vllm/` → 0.

## Revival preconditions

1. A W8A16 GEMV kernel that beats the bf16 GEMV at M=3–9 (the m4 kernel
   did not).
2. A model whose drafter-head cost share is materially larger (Qwen3.5's
   5.71 GB drafter — the original figure that motivated the scope).
3. A depth/k regime where M stays ≥5 without B≥2 (B≥2 kills it
   structurally at M≤8).
