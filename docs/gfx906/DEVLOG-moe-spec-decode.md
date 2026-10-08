# DEVLOG-moe-spec-decode (W2): port the spec-decode rails to the 35B MoE

**VERDICT:** SHIPPED (2026-08-23: the rails port with no code changes —
all four were in-tree from the merged 27B phase; mtp2 k=2 is +16–20 %
graph / ~1.84× eager on the 35B MoE. No dispatch change to ship; the
recommendation is config: `--speculative-config '{"method":"mtp",
"num_speculative_tokens":2}'` + cg-small for 35B serving.)
**Date:** 2026-08-23 (started)
**Build:** `vllm 0.28.0rc2.dev318+gfed585110` (C2-V .venv; both C2-V flags
default off → dispatch identical to the fed585110 baseline)
**Branch:** `gfx906/moe-spec-decode` (off `gfx906/main` d608aa40a5)

## HYPOTHESIS

The Phase-1 spec-decode rails (merged from the 27B dense phase: MTP
k=2 drafter + `Gfx906FAMetadata` proposer allowlist `68243a61b2`,
`UNIFORM_BATCH` cg support, cg-small capture sizes, drafter fc
GEMV dispatch, GDN `fused_seq`/sequential spec kernel) function on
the Qwen3.5-35B-A3B-AWQ MoE and raise serving decode throughput
versus the 67.39 t/s greedy record (band 65.9–67.0).

Model-specific facts (checked in checkpoint, 2026-08-23):

- text_config: hidden 2048, 40 layers (30 GDN + 10 FA), KV heads 2,
  linear heads 32v/16k, 256 experts, topk 8, moe_inter 512.
- **MTP head present**: `mtp_num_hidden_layers=1`,
  `mtp_use_dedicated_embeddings=false`, 785 `mtp.layers.0.*` tensors.
  The MTP layer is itself MoE (`mtp.layers.0.mlp.experts.*`); `mtp`
  is in `modules_to_not_convert` → drafter is **fp16** (same as the
  27B), but sparse: a draft forward reads only 8/256 experts
  (8 × 3 × 2048×512 × 2 B ≈ 25 MB) + fc (2048×4096×2 ≈ 16 MB) +
  attn (~25 MB) + shared lm_head (2048×V×2 ≈ 1 GB) ≈ **~1.1 GB
  per draft forward** vs the 27B drafter's 3.4 GB dense read →
  draft cost should be a small fraction of the step.
- `Qwen3_5MoeMTP` is registered (`registry.py:682`); fc K=4096 is
  inside the in-tree GEMV KCHUNK set {512,1024,2048,4096}.

**The one structural risk (predicted before measuring):** the
target verify step at k=2 runs 3 tokens through the grouped MoE path
(em = 3×8 = 24 slots, a C2-V-unmeasured point between em=32/BM=1 at
21.7 ms and em=64/BM=4 at 47.8 ms). The MoE grouped path is
M-sensitive (unlike the weight-bound dense AWQ GEMM), so the
break-even acceptance for k=2 on the 35B is far above the ~1.82
tok/step the 27B dense achieved: break-even ≈ (verify + 2×draft)
/ baseline_step tok/step. If verify ≈ 25–35 ms and draft ≈ 3–5 ms,
break-even ≈ 2.6–3.4 tok/step vs the k=2 maximum of 3.0. **W2 may
function and still not pay off** — that outcome is itself the
roadmap-level finding (dense vs MoE spec-decode economics).

## GATE

- Serving wall-clock A/B, same build/session: greedy baseline (must
  reproduce ~67.4 t/s harness metric) vs
  `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
  with cg-small trimmed capture sizes. Graph + eager.
- Token-identity of spec vs greedy outputs (same prompts, temp 0).
- Draft acceptance + tokens/step counters reported.
- Positive = spec t/s > baseline t/s; the 27B reference is 1.50×.

## Matrix

| arm | TP | spec | regime | notes |
|---|---|---|---|---|
| base | 1 | off | graph | anchor: expect ~65.6 (Δ-metric era) / 67.4 harness |
| mtp2 | 1 | k=2 | graph | the rail under test |
| base | 1 | off | eager | |
| mtp2 | 1 | k=2 | eager | |
| (k=3 / ngram3) | 1 | k=3 | graph | only if k=2 pays off |

## What was done

- 2026-08-23: branch created; checkpoint MTP-head + quant config
  audited (above); in-tree rail audit (all four rails present from
  the merged 27B phase — see Interactions); smoke next.

## Results (interim — graph A/B; session interrupted by FW wedge)

Post-reset#1 (05:18Z), host canary 38.9 t/s (healthy-for-this-boot):

| arm | steady t/s (prefill-incl, 256 tok) | notes |
|---|---|---|
| base graph (cycle 1) | 76.09 / 76.09 / 76.23 / 76.10 | rep0 cold (fused_moe JIT mid-run) |
| base graph (cycle 2) | 76.87 / 76.75 / 75.27 / 76.69 | |
| **mtp2 graph (k=2)** | **86.85 / 88.22 / 86.23 / 89.67** | **≈1.14–1.17× baseline** |
| base eager | 24.51 / 24.65 / 24.53 / 24.72 | |
| mtp2 eager | — | crashed into the 05:47Z reset#2 (re-run pending) |

**≈+15 % on the graph regime before the wedge ended the session.**
The pre-wedge 0.49× smoke was host artifact, confirmed: post-reset
the rail is solidly positive. Break-even fear (em=24 verify cost) did
not materialize — the sparse MoE MTP drafter (~1.1 GB/forward) is
cheap enough that k=2 pays off even with the M-sensitive grouped
verify path.

Acceptance counters are missing from this pass (the scheduler's
`log_stats` gate suppresses `observe_draft`; the driver env lacked
`VLLM_LOG_STATS=1`) — mtp2 arms re-run with the flag pending the
reboot. Token-identity gate: unusable on this model (baseline
non-reproducible at temp=0 — see degradation_details 05:47Z section).

2026-08-23 06:08Z: GPU0 full wedge (PSP -62) — session paused;
reboot required (Kevin). **Note: /tmp is tmpfs on this box — the
reboot wiped /tmp/moespec and /tmp/c2v (raw logs); the numbers above
were already committed here and in the degradation files.**

### Post-reboot completion (2026-08-23 ~06:30Z reboot, canary 38.6 = healthy)

Acceptance re-runs (the missing piece: `LLM()` defaults
disable_log_stats=True — llm.py:228 — which suppressed
observe_draft; fixed in the bench, `LOG_STATS=1`):

| arm | steady t/s (prefill-incl) | tok/step | acceptance |
|---|---|---|---|
| base graph (w2b_base_g) | 76.27 / 76.10 / 76.25 / 76.09 | — | — |
| **mtp2 graph (w2b_mtp2_g)** | **87.75 / 91.33 / 91.56 / 89.86** | **1.609** | **80.44 %** |
| **mtp2 eager (w2b_mtp2_e)** | **44.83 / 46.14 / 44.90 / 45.71** | 1.566 | 78.31 % |
| base eager (pre-reboot) | 24.51–24.72 | — | — |

- **Graph: 89.9 vs 76.2 steady = 1.18×** (per-rep 1.15–1.20);
  consistent with the pre-reboot mtp2_g (86.2–89.7) — reproduces
  across the reboot. **Eager: 45.5 vs 24.5 = 1.86×** — spec
  amortizes the launch-bound eager step (40.8 ms) across 1.57
  tokens, so the relative win is much bigger than graph's.
- **The 35B MTP drafter is a weaker proposer than the 27B's**: 80.4 %
  acceptance / 1.609 tok/step vs the 27B's 90.95 % / 1.819 (same
  k=2, both MoE-vs-dense MTP layers; the 35B MTP layer is itself
  MoE, fp16). Break-even arithmetic: baseline decode ≈ 11.8
  ms/step; mtp2 step ≈ 16.4 ms (1.39×) → break-even 1.39 tok/step;
  measured 1.609 → +16 % margin. k=3 needs ~2.4 tok/step here (two
  more draft forwards + a 4-token verify at em=32) — not viable at
  80 % first-draft acceptance; leave k=2 as the config.
- **Verdict:** the rails (MTP k=2 + UNIFORM_BATCH cg + cg-small +
  fc GEMV + GDN spec path) work on the 35B MoE with **zero code
  changes** — everything was already in-tree from the merged 27B
  phase (`Qwen3_5MoeMTP` registered, fc K=4096 inside the GEMV
  KCHUNK set, allowlist + UNIFORM_BATCH model-agnostic). W2's real
  deliverable is the measured config recommendation: mtp2 + cg-small
  for 35B serving (graph: +18 %; eager: +86 %).
- Host: two clean canaries post-reboot window (38.6), no wedges in
  the 06:30–06:52Z completion window (3 engine cycles). The
  100 %-acceptance DEG symptom (Kevin's report) did NOT appear —
  healthy-state acceptance is 78–80 %.

## Refrigerated residue

- If k=2 loses: the lever is the verify-side MoE grouped-path M
  sensitivity (C2-V's unmeasured em=24 point; a BM=2/4 re-tile is
  C2 territory, now with a concrete consumer) — not the drafter
  (already ~1/4 the 27B's cost).
- k=3 needs ≥2.17 tok/step on the 27B; on the 35B the bar is
  ~break-even + 1/2.

## Interactions / superseded-by

- `DEVLOG-spec-decode.md` (the 27B Phase-1 log; the rails this
  ports) and the completed W2 entry in `CHANGELOG.md`.
- `DEVLOG-moe-c2v.md` — C2-V (same day): the grouped-path step-cost
  data (em 8/32/64 → 12.3/21.7/47.8 ms) that motivates the risk
  above.
- `68243a61b2` (MTP k=2 + GEMV dispatch + cg-small), `4e40e3eee2`
  (FULL cudagraphs for spec steps), `5d960a503c` (UAF fix).

## Search keys

moe spec decode, mtp2 35b, Qwen3_5MoeMTP, draft acceptance, em=24,
break-even acceptance, moe verify cost, W2
