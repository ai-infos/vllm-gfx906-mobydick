# C3 — fold the two MoE zeroings: MEASURED, NO SERVING GAIN (NO-GO)

**Branch:** `gfx906/c3-zeroing-fold` off main `df0e8f7a26`.
**Model:** Qwen3.5-35B-A3B-AWQ (E=256, topk=8, moe_intermediate=512 →
`w1_out` = [8,512] = 4096 halves = **8 KB**; 40 hidden layers, of which ~37
take the M=1 MoE fold path per step — measured).
**Gate (user, verbatim):** "If this does not add gains or does not get
enabled by default, do not merge it."

## What was built (Phase A — the low-risk half)

Folded `w1_out.zero_()` into the single-CTA M=1 align kernel
(`csrc/rocm/moe_align_m1_gfx906.cu`), removing one memset graph node per MoE
layer. The fold is **stream-ordered**: the align CTA fully completes before
gemm1's K-split CAS blocks dispatch, so it is *not* the racy in-GEMM
clear-before-CAS design P2-0b rejected (DEAD-ENDS.md). New binding
`moe_align_block_size_m1_zero_gfx906`; Python gate `VLLM_GFX906_MOE_ZERO_M1`
(default ON, `=0` opts out) in `gfx906_w4a16_moe.py`, gated on the C1 stage-1
shape set + a no-tail buffer-size check (numel % 256 == 0).

The second zeroing (`output.zero_()`) was **not** folded: `workspace13`
(w1_out) and `output` alias one storage and gemm1 dirties output's region via
that alias, so it must stay after activation (the P3-4 "REJECTED" aliasing
trap). Folding it would require de-aliasing — a bigger change deferred because
Phase A already failed the gain gate.

## Correctness (PASS)

- Unit: `tests/kernels/moe/test_moe_align_m1_zero_gfx906.py` — 78/78 green
  (align outputs bit-equal to the plain fused M=1 kernel AND the production
  two-kernel chain for random + tie-heavy topk_ids on both served shapes;
  pre-dirtied buffer comes back fully zero; dispatch gate exact; CUDA-graph
  capture-safe and replay-stable).
- Regression: `test_moe_align_m1_gfx906.py` (C1 stage 1) +
  `test_gfx906_moe_gemm.py` end-to-end — all green.
- **Serving fingerprint identical** across off/on arms (`d2e5262183c6b92f`).

## Fold-firing confirmation (in-process probe)

`benchmarks/kernels/gfx906/c3_fold_firing_probe.py` (in-process so the
expert-module monkeypatch is visible; eager single-request, 16 profiled decode
tokens):

| flag | c3_zero/step | plain/step | generic/step |
|---|---|---|---|
| `VLLM_GFX906_MOE_ZERO_M1=1` (on) | **36.56** | 0.00 | 2.44 |
| `VLLM_GFX906_MOE_ZERO_M1=0` (off) | 0.00 | **36.56** | 2.44 |

So C1 stage-2 fused routing is **not** active on this model — the M=1 decode
path goes through the align call, and the on-arm genuinely ran the fold on ~37
MoE layers/step. The A/B arms demonstrably differed. (The `generic` 2.44/step
are non-MoE/vision align calls, present in both arms.)

## Serving A/B (the deciding gate) — WASH WITHIN NOISE

`benchmarks/kernels/gfx906/moe_multireq_ab.py`, FULL_DECODE_ONLY graphs,
N=1 M=1 decode, pp2048/tg256, TP=1, util 0.88 (coexisting with the ~2.25 GiB
llama-server reranker tenant on GPU0), 3 repeats/arm, decode-only t/s via the
long-minus-short difference:

| arm | reps (t/s) | mean | median | stdev |
|---|---|---|---|---|
| off (baseline) | 85.68, 85.62, **82.98** | 84.76 | 85.62 | 1.54 |
| on (C3 fold) | 85.42, 85.46, 85.21 | 85.36 | 85.42 | 0.13 |

The +0.7% "mean" is an artifact of one noisy off rep (82.98). The on-arm is
rock-solid (stdev 0.13) but sits **at-or-below** the off-arm's two clean reps
(85.6) → a wash within noise, possibly a marginal loss. Not a gain.

## Why no gain (coherent explanation)

This model's `w1_out` is only 8 KB (moe_intermediate=512). Removing ~40 tiny
memset nodes under graph replay saves on the order of tens of µs/step — but the
fold *adds* an 8 KB store to the single align CTA, roughly offsetting it. Net:
cost-neutral, which is exactly the wash we measured. (Contrast C1 stage 1,
which removed ~40 nodes carrying real routing work and shipped at +1–2%; those
were larger-node removals that cleared the noise floor.) This is the same
"isolated win doesn't transfer to serving" pattern G1 and N3 documented — here
the node removal is simply too small to clear the A/B's ~1.8% noise floor on a
~14 ms step.

## Verdict: NO-GO (not merged)

Bit-correct and confirmed firing, but **no measurable serving gain** under
FULL_DECODE_ONLY graphs → fails the user gate. Not merged to main. The
`output.zero_()` half (de-aliasing) is not pursued for the same reason: it
would remove ~40 more tiny nodes on a model where the first 40 didn't clear
the bar.

**Follow-up note:** the fold *could* clear the bar on a model with a larger
moe_intermediate (bigger `w1_out` memset), but there is no such gfx906 serving
target in scope, so it is not worth keeping default-on. The branch is retained
unmerged as a reference; recommend discarding unless a larger-intermediate MoE
model enters the serving set.

## Environment note (transient)

The first fold-firing probe run aborted with `hipErrorLaunchFailure` ~4 min in
(preceded by an `amdsmi_shut_down failed` warning); both A/B arms had just
completed cleanly (~30k fold invocations). GPUs recovered to 0% idle with no
dmesg GPU errors → treated as a transient driver hiccup, not a kernel bug.
Re-run of the probe was clean. Logged in `/local/tmp/gpu_degradation.log`.
