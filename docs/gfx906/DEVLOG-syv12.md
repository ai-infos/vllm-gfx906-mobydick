# SYV-12 — context-lookup verify extension for MTP (fill-from-draft-buffer)

**VERDICT:** DEAD-END (not workable in the current kernel/architecture
shape — corpus-independent, per the 2026-09-12 corpus-agent finding;
earlier: v1 NET LOSS, v2 payload-conditional PARKED) ·
**GATE:** production serving A/B @120k/64k
**Branch:** gfx906/fa-decode-fp16 (2026-09-10 → 2026-09-11)
**Archive:** `archive/syv12` (branch; see REFRIGERATOR.md)
**Full detail:** hunt-combined doc + stall-log §S13.x at
`git log e2ed0b1706..dea38eb14c` (pre-restructure)

## 2026-09-10 — HYPOTHESIS: for payloads with high copy-density
(prompt text that recurs in the expected continuation), an extra verify
slot filled from the previous step's draft buffer (context lookup)
raises acceptance at MTP k=2 cheaply

## What was done

V1-runner port (`vllm/v1/spec_decode/syv12.py` + worker + scheduler
plumbing, `syv12_ext` slots), fill kernel (Triton → the fill-contract
fix `7b2fba0754`), probe gates, boot-Y3 A/B.

## Evidence / verdicts

- **v1 (2026-09-10): NET LOSS on the production payload — CLOSED**
  (`4b6450865e`). Gate: boot-Y3 serving A/B @120k.
- Attribution correction (verdict review `9bae8c4d3f`): the fill yield
  was never validated — the "structural" framing retracted.
- **Fill-contract root cause + fix**: lookup window anchored at the
  wrong offset for k=2; fix validated (pos3 identity 512/512,
  `7b2fba0754`).
- **Gate variable adjudicated** (`b08f2dddec`): the gate is payload
  POSITION (copy density), NOT context length — window copy density
  measured: serving-64k 12.6% / serving-120k 16.9% / same-payload
  first-64k 13.6% / last-64k 21.5%.
- **v2 production A/B (2026-09-11, `dea38eb14c`): +6.4% @120k,
  −5.3% @64k → payload-conditional; PARKED by Kevin.**

## History — the V1-port era (moved 2026-09-13 from DEVLOG-fa-attention.md)

**Why this took three boots and two runners** (the part worth keeping — it is a
runner-selection trap, not an SYV-12 fact):

- **2026-09-08 boot Y — v1 landed on the WRONG runner.** The committed worker
  side went into the **V2** runner (`vllm/v1/worker/gpu/model_runner.py`), but
  the live path on this stack is the **V1** runner (`gpu_model_runner.py`;
  `gpu_worker.py` selects V2 only when `VLLM_USE_V2_MODEL_RUNNER` is set). The
  probe's ON arm was a **100 % no-op**: the jit_monitor never logged
  `_syv12_fill_kernel` during warmup, and the committed kernel could not even
  compile (Triton rejects `break` in that loop context — `unsupported AST node
  type: Break`). OFF/ON identity was therefore vacuous, not losslessness. With
  `GFX906_SYV12=1` the *scheduler*-side pad would still have applied while the
  V1 worker did nothing — an **inconsistent state, not a no-op** — so the
  staged A/B driver was withheld. Fixes that session: `break` removed (scan is
  ascending with last-occurrence overwrite), the occurrence-window off-by-4
  fixed (`j..j+4` + continuation `j+5`, matching the gate scans), 9 kernel
  cases PASS on GPU, probe `log_stats=True`, result parsing moved to
  `parse_result.py`.
- **V1 port design (the durable decision):** the fill must be computed
  **GPU-side** — under async scheduling the worker's CPU token history is
  unusable (`token_ids_cpu` / `req_state.output_token_ids` carry `-1`
  placeholders; the real tokens live on the GPU and the scheduler's history is
  in another process). So: a small per-request GPU history
  (`_syv12_hist [max_num_reqs, max_model_len]` int32 + `_syv12_hist_len`, rows
  stable per request, row mapping rebuilt per launch) + a new
  `_syv12_append_history_kernel` that compacts each step's accepted tokens, then
  the fill writes the ext column. Sizing/plumbing: `uniform_decode_query_len =
  1+k+ext`, `draft_token_ids_cpu` widened, **both** `async_scheduler.py`
  placeholder sites widened (the v1 commit missed them), and
  `MambaSpec.num_speculative_blocks = k+ext` (the GDN state pool needs a slot
  per draft token for the extra verify row). Kernels moved to a runner-neutral
  `vllm/v1/spec_decode/syv12.py`; kernel cases 9 → **16/16** (append/fill flows).
- **Boot Y2 — two software bugs and a wear-based stop:** retry 2 hit a
  **scheduler spec-stats sizing** bug, retry 3 a **GDN state-slot
  under-provisioning** bug (zero output). Wedges #41/#42 (chronic weight-load
  family) stopped the session by the wear rule. OFF arm measured clean at that
  point: **decode 25.335 t/s** (120k s9, 512 tokens, decode-only from
  `RequestStateStats` timestamps).
- **Boot Y3 — 120k probe PASS:** identity **512/512** lossless (the fill now
  actually fires), then the v1 serving A/B = **NET LOSS** (see the verdicts
  above, `4b6450865e`).
- **SYV-13 (same window, `CLOSED as N/A`):** the mamba/GDN chunked-prefill
  align fixes turned out to be verify-only — no production effect, closed
  without a code change.
- **Resurrection gate (post-v1 attribution correction):** the fill's lookup
  suffix targeted the **d0 slot — off by k**. Fixed (`7b2fba0754`) and validated
  by probe (pos3 fire 0.987, mean ~3.99 fills/step); a compile-assert bug was
  found and fixed on the way. Only then was the v2 A/B meaningful.
- **Generation-time gate (from the MTP depth campaign's replay, 39 convos /
  205,091 positions): GO** — fill hit_frac **25.3 %** vs the ≥15 % gate (the
  corpus proxy's 10–18 % was an underestimate: generations re-quote context
  more than the corpus self-repeats). Recorded here because it is SYV-12's
  demand-side evidence; the replay itself is in
  `DEVLOG-mtp-depth-matrix.md`.

Full pre-split detail (boot-by-boot, incl. the probe/harness bug list):
`git show e963fd8c62:docs/gfx906/DEVLOG-fa-attention.md` (entries of
2026-09-08/09).

## Why parked

The win exists only for payloads whose continuation copies the prompt
at ≥ ~15–20% density (measured at the FA lookup window); the general
payload loses. Post-FIX-H2 the fill also no longer pays for itself on
the residual path it was meant to recover (D1c-style remainder ~1.9 s
per 2-step probe, later re-measured smaller).

## Closure (2026-09-12)

The corpus agent's finding: SYV-12 is **not workable in the current
kernel/architecture shape, independent of payload copy density** — the
corpus work cannot rescue it. Deep-scrubbed from the line on branch
`gfx906/fa-decode-fp16-scrub` (all three layers: payload, scheduler/
runner plumbing, sizing terms; `grep -ri syv12 vllm/ tests/` = 0).
The code lives grouped on `archive/syv12` (incl. the V2 runner wiring).
Revival is only rational if the SHAPE changes (a different fill
mechanism or verify layout) — not a corpus or config matter.
