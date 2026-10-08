# CAT-1 draft-vocab shortlist — shrink drafter lm_head read 248k → ~131K rows (lossless)

> Branch `cat1-draft-vocab` off `main` @ `4fa513098d` · model `cyankiwi/Qwen3.8-27B-AWQ-INT4` · date 2026-09-04 ·
> roadmap item CAT-1 (see [RECON-1cat-vllm.md](RECON-1cat-vllm.md) §CAT-1, [ROADMAP.md](ROADMAP.md)).

**VERDICT:** `PASS (k=2 pilot + k=4 stacking A/B) — merge candidate; final list awaits Kevin's corpus`

**GATE:** serving A/B, TP=2 + MTP depth-2 (port 8123 config, `run_server.sh`),
graph mode: baseline arm = current main behavior (full-vocab draft head); test
arm = same server pointed at a working-copy model dir with
`mtp_draft_vocab_ids.pt` + `model_extra_tensors.safetensors` present. Measure
t/s at the standard sweep contexts + acceptance rate (server logs) + coherence
spot-checks. Token-identity checks are NOT valid (model non-deterministic at
temp=0). Standalone probe numbers below are launch-regime evidence only — per
the transfer-failure history in [DEVLOG-moe-gemm1-retiling.md](DEVLOG-moe-gemm1-retiling.md) §3, they do not decide the verdict.

---

## HYPOTHESIS

The MTP drafter's lm_head read ([248320, 5120] bf16, unquantized — in the AWQ
ignore list) is the dominant byte-read of the draft step, and SYV-3 closed the
kernel side (fork GEMV family already at ~82% HBM peak), so only byte count is
left. If the drafter scores a static shortlist of frequent tokens via a reduced
second lm_head (out-of-list logits = -inf; target model stays full-vocab for
accept/recover — lossless by construction), then serving t/s must improve in the
A/B without measurable acceptance/coherence cost above the held-out coverage gap.

## External review (Claude Code, 2026-09-04 — handover /local/tmp/mtp1/cat1_handover.md)

Findings validated against the code before acting (per pre-merge protocol):

- **F1 "TP=2 shard corruption for N % (tp×pad) != 0" — REJECTED.** The premise
  contradicts `VocabParallelEmbedding.__init__`: it pads FIRST
  (`num_embeddings_padded = pad_vocab_size(N, 64)`), then shards the padded size
  (`divide(padded, tp)`), which always succeeds for tp ∈ {1,2,4} since every
  multiple of 64 is divisible by them. Verified numerically for N ∈
  {131072, 131009, 130945, 40960, 40900, 98304, 12345} × TP ∈ {1,2,4}: all pad
  cleanly; gather is a rank-ordered concat of contiguous shards + trim `[:N]`,
  order preserved for any N. No reindexing exists because none is needed.
- **F2 argmax-bypass undefended — VALID.** Fixed: hard `ValueError` in
  `Qwen3_5MTP.__init__` when the shortlist is active and
  `use_local_argmax_reduction` is on.
- **F3 builder N unpredictable — no functional issue** (`--n` was always an
  upper bound); documented in the builder docstring incl. the TP note from F1.
- **F4 lazy CPU→GPU id migration under cudagraphs — safe by construction**
  (fires once during eager warmup, before capture; branch is dead in replays);
  documented in `compute_logits`.
- **F5 probe doesn't cover TP=2 — already caveated** (handover + this log).

## PILOT RESULT (complete 13:17 UTC — both arms torn down clean, `vllm procs left: 0`)

| point | arm0 stock-MTP | arm1 CAT-1 (40,960 head) | Δ |
|---|---|---|---|
| pp=65536 | 38.85 t/s [38.856/38.847/38.81] | **40.31** [40.403/40.308/40.298] | **+3.86%** (same boot) |
| pp=122880 | 25.70 (measured, boot S, same config) | **26.83** [26.796/26.856/26.828] | +4.38% (cross-boot) |

- **Activation verified (no silent fallback):** arm1 logged `MTP drafter uses a 40960-token
  draft head` on both TP workers; arm0 has no such line (full head).
- **Zero coverage loss:** mean acceptance length **3.00 on both arms** throughout — k=2 → both
  drafts accepted every step, at 64k *and* 120k. The shortlist never cost an acceptance.
- **Internal consistency (strong):** dec_s 6.56→6.31 = ~0.98 ms/token saved, matching the
  standalone probe's predicted ~1 ms/token head-read saving under TP=2 almost exactly → gain is
  attributable to the reduced-head read, not noise. Same-boot @64k gap = **75× arm0 std**.
- **Coherence:** samples fluent + on-topic both arms; cat1 sample is a stylistically different
  (equally correct) phrasing — expected under greedy + reduced-head dynamics, no degeneration.
- **Wedge bookkeeping:** wedge #18 was the only incident and it hit arm0's @120k rep (that row's
  ttft=179 ms is a client artifact: first chunk arrived before the hang). CAT-1 side ran ~40 min
  bench load with **zero wedges**.

**Caveat (honest):** corpus s9 is synthetic filler, so *both* arms saturate at perfect
acceptance — this pilot validates the **mechanism + head-read speed gain** at N=40,960 on our
traffic, NOT the real-world acceptance tradeoff (that needs Kevin's request/response corpus).

**PILOT VERDICT: PASS.** Mechanism validated end-to-end (sliced head → index/weight_map → TP=2
sharding → scatter-back under cudagraphs); direction + magnitude real on our traffic at N=40,960.
Satisfies C3 "does it add gains" for the pilot. **Not a merge decision** — final number awaits
our own corpus + ~131K list (branch `cat1-draft-vocab` stays open, everything committed).

## k=4 STACKING A/B (2026-09-05, complete — arm torn down clean)

Question: does the shortlist gain scale with depth? At k=4 the drafter lm_head runs ~3× per step
(vs 1× at k=2), so a fixed head-read saving should compound. Setup: `cat1k4` arm = CAT-1 pilot
work-dir (same 40,960 head) + `num_speculative_tokens=4` + capture [1,2,3,4,5], port 8129;
T-1 OFF (clean single axis). Baseline = plain k=4 (`mtp4`, full head): 44.82 / 35.79 / 30.95 t/s.

| pp | CAT-1 × k=4 median (n=3) | plain k=4 | Δ | per-pos acceptance (min across reps) |
|---|---|---|---|---|
| 65536 | **47.31** [47.48/47.31, rep0 cold 30.23] | 44.82 | **+5.6%** | p0–p3 = 1.000 |
| 98304 | **37.19** [37.24/37.19/37.13 — rock stable] | 35.79 | **+3.9%** | p0–p3 = 1.000 |
| 122880 | **32.13** [32.06/32.13/32.20] | 30.95 | **+3.8%** | p0–p3 = 1.000 |

- **Quality gate PASS:** perfect 4/4 acceptance at EVERY draft position, every rep — the 40K
  shortlist shows no coverage gap even at depth (positions 2–3 are where a too-small list would
  first bite; it didn't).
- **Gain shape matches the step-economics model:** CAT-1 saves a fixed ~4 ms/step off the drafter
  head read, so the percentage compresses as context grows (attention is O(S) and dilutes a fixed
  saving): +5.6% → +3.9% → +3.8%. All three points clear the ±1 t/s noise floor (unlike T-1's −1%).
- **Stacked story:** k=4 depth (+15–18% over k=2) + CAT-1 (+3.8–5.6% on top) ≈ **+20–24% total
  over plain k=2 decode**, zero acceptance loss — the same neighborhood as 1CatAI's headline
  +21.9%, reached via depth+shortlisting rather than their exact recipe (their number is a k≈2
  shape where the drafter is a bigger step fraction).
- **Interaction check:** T-1 int8 does NOT stack with CAT-1 on this model (drafter lm_head +
  mtp.* are already unquantized bf16 — see Interactions); the two levers act on different bytes
  (head row count vs weight dtype) but the drafter head is the shared target, so int8-on-top of a
  shortlisted head has diminishing/negative returns (the quant tax is a bigger fraction of a
  smaller GEMV — exactly what the T-1 @ k=4 re-test measured).

**k=4 VERDICT: PASS.** The shortlist gain holds at depth and stacks on the k=4 win. Merge story:
CAT-1 is env-gated by construction (requires the sliced work-dir; plain model dir = full-vocab
path unchanged) → clears the "optional/gated" bar. Remaining before merge to main: Kevin's real
corpus → our own ~131K list → final A/B acceptance gate on real traffic (the s9 filler saturates
both arms at perfect acceptance, so it validates mechanism+speed, not the coverage tradeoff).

## RESUME (state as of 2026-09-04 ~13:20 UTC)

**Where we are:** pilot A/B **DONE** (driver proc `proc_5cb93bec805e` exited 0 at 13:17).
Sequence: canary PASS 39.2 t/s → arm0 `mtp` ready ~585s (12:12) → bench → teardown+quiesce
→ arm1 `cat1` (:8125, work-dir `/local/tmp/mtp1/cat1_pilot/`) → same bench → clean teardown.

**STATUS 13:20 UTC:** both arms complete. arm0 DONE at pp=65536 (median 38.85 t/s, n=3);
arm0 **pp=122880 lost to wedge #18** (mid-decode @120k, silent hang, no kernel event — recorded
in degradation.md + degradation_details.md; accepted HW-related per Kevin's ruling). arm1 (CAT-1
pilot) loaded weights CLEANLY (50.6 s + 18.1 s incl. the extra 419 MB draft-head file), served
both points, zero wedges. **No re-run of arm0@122880 needed** — the @120k baseline uses the
measured boot-S stock number (25.70 t/s, same config); cross-boot drift bounded by the +2.4%
observed at 64k today.

**Check progress (no GPU needed):**
- driver log: `tail -f /local/tmp/mtp1/cat1_pilot_driver.log`
- per-arm bench JSONL: `/local/tmp/mtp1/cat1_pilot_arms.jsonl`
- coherence samples: `/local/tmp/mtp1/cat1_sample_mtp.log`, `cat1_sample_cat1.log`
- server logs: `/local/tmp/mtp1/server_mtp.log`, `server_cat1.log`

**After the driver exits (DONE):** both arms' JSONL read; t/s + acceptance compared. PILOT PASS
criteria met: (a) arm1 served with no crash/wedge, coherent samples, acceptance delta = 0 (both
saturated at 3.00 — synthetic filler); (b) t/s direction positive (+3.86% @64k same-boot).

**Still owed after corpus lands:** `build_draft_vocab.py count` on Kevin's JSONL → pick N
(target ~131K static or 98K+2×512 dynamic) → `slice` → final TP=2 A/B with our own list →
acceptance gate (real traffic, not filler) → verdict + merge. README attribution for the ported
syv hunks is owed (see below). If arm1 wedges/crashes in a future re-run: record in degradation.md
+ degradation_details.md, inspect server_cat1.log, fix, re-run driver (canary first if >30 min
since last pass).

## README attribution OWE...[truncated]

## Pilot (2026-09-04, launched ~11:50 UTC — driver /local/tmp/mtp1/cat1_pilot_driver.sh)

**False-abort incident (run 1):** canary actually PASSED (`CANARY: 256 tok / 6.5s =
39.1 t/s`, healthy band), but the driver grepped for a `CANARY[mtp]` tag that
canary_probe.py never prints (it emits plain `CANARY:`) → empty parse →
"DEGRADED ()" abort before any bench arm. Fixed: correct grep + truncate canary
log at driver start (probe appends; stale lines could mask a real crash).
No GPU damage; re-launched 12:0x UTC.

**Driver bug 2 (run 2):** `local start_ts=$(( $(date +%s) )) deadline=$(( start_ts + 1350 ))` —
bash evaluates ALL `$(( ))` on the line before any assignment binds, so under
`set -u` `start_ts` was unbound at expansion time → died before arm0. Fixed with
separate statements; added SKIP_CANARY env (canary had passed 39.2 t/s minutes
earlier on the same clean system). Run 3 launched 12:02 UTC.

Kevin directive: corpus capture takes time → use syv's **shipped 40,960-id list**
(`/tmp/syv_draft_vocab_ids.json`) to (a) validate the implementation end-to-end and
(b) run a pilot A/B now; final A/B with our own ~131K list comes when the corpus
lands. Compatibility verified against OUR tokenizer: 40960 unique ids, all in
[0, 248076], sample decodes cleanly (code tokens dominate).

- Work-dir `/local/tmp/mtp1/cat1_pilot/`: original snapshot + sliced draft head
  (419 MB bf16) + updated index (real file) + ids .pt. Original snapshots
  verified untouched.
- **INCIDENT (mine, fixed):** first slice test ran with `--snapshot` = the NFS
  mirror; the builder symlinked the index into the work-dir and then wrote it —
  O_TRUNC write-through added `mtp.draft_lm_head.weight` to the shared NFS
  snapshot's index (dangling: extra file never written there). Repaired from the
  clean local copy (byte-identical, backup `/tmp/nfs_index_corrupted.bak`);
  builder now copies (never symlinks) files it overwrites — commit 38ef11244d.
- Arms (sequential, wedge-safe per #12/#13): canary pre-flight → arm0 `mtp`
  (stock, :8123) → teardown+quiesce → arm1 `cat1` (pilot work-dir, :8125).
  Bench = syv3_client.py @ 65536/122880 ctx, tg=256, 3 reps + acceptance deltas;
  coherence sample per arm. Pilot readout answers: does the mechanism work in
  serving (load, scatter under cudagraphs, TP=2 sharding) and what is the
  direction/magnitude at N=40960 on OUR traffic? NOT the final verdict — our
  own list + corpus coverage is still owed before merge.

## What was done

- Ported the "syv patch" (source: `syv-ai/qwen38-27b-rtx3090`,
  `patches/qwen3_5-mtp-draft-vocab.patch`, written for vLLM 0.27.1; ours
  0.28.0rc2) into `vllm/model_executor/models/qwen3_5_mtp.py` — 4 hunks:
  reduced `draft_lm_head` creation (adapted: `quant_config=None`, our rows are
  bf16 not int8-packed), `draft_logits_processor`, `compute_logits` scatter-back
  (`new_full(-inf)` + `index_copy_`), load_weights skip when disabled.
  Attribution: inline comments here + README entry (pending merge).
- New `tools/build_draft_vocab.py`: `count` (tokenize model-generated JSONL
  corpus → top-N ids + all_special_ids, held-out coverage) and `slice`
  (row index_select → working-copy dir: symlinked shards +
  `model_extra_tensors.safetensors` + updated index.json +
  `mtp_draft_vocab_ids.pt`). Both phases tested on synthetic corpus; sliced rows
  verified byte-equal to checkpoint rows.
- Corpus spec written for Kevin to capture real traffic:
  [CAT1-draft-vocab-corpus.md](CAT1-draft-vocab-corpus.md) (request/response
  pairs incl. reasoning traces + raw tool-call blocks; ≥15M generated tokens;
  JSONL format defined). **Corpus not yet captured — no real id list exists.**
- Standalone probe `/local/tmp/mtp1/cat1_head_probe.py` (output
  `/tmp/cat1_head_probe.out`): production dispatcher kernels
  (`_llmm1_tiny_m` n=1, `_gfx906_spec_gemv_m4` n=4) at full/131K/40960 rows,
  real checkpoint rows bf16→fp16, hot loop + CUDA-event deciles, mclk hard gate.

## Evidence — FOR

- **Launch-regime (not the gate).** Probe, GATE PASS (mclk 1000 MHz all
  windows): full 248320 n=1 = 3096 µs (821 GB/s, 82.1% peak); static 131072
  n=1 = 1637 µs; shipped-40960 n=1 = 515 µs. n=4: 4940 / 2616 / 817 µs.
  Full-head anchor matches SYV-3 in-context 3.09 ms within 0.2% → dispatch path
  confirmed identical to production. Static-131K cuts the draft head read by
  47%; dispatchers hold ~82% peak at shortlisted Ns (no kernel cliff).
- Lossless-by-construction: acceptance math unchanged, only the draft proposal
  distribution is restricted; upstream reports +21.9% e2e on their TP=2 Qwen3.6
  int8-head config (their numbers — not transferable as evidence here).

## Evidence — AGAINST

- Probe is TP=1-shaped: under TP=2 each rank holds half the rows, so absolute
  savings halve per rank; the step is also attention-dominated at long context
  (73% full_attn @120k, [phase_profile_results.md](phase_profile_results.md)),
  so the head read's share of an MTP step is smaller than the probe suggests.
  Upstream's +21.9% was on a 3090 with an int8 (1.3 GB) head — our bf16
  full-head is 2.54 GB, and their N=40960 vs our planned ~131K means they cut
  more bytes than we do at comparable coverage.
- Acceptance cost unmeasured: held-out coverage of our own traffic unknown until
  the corpus lands (upstream hit 95% at N=40960 on 8.8M tokens; our ~131K list
  should clear ≥97%, but that is a prediction, not a measurement).

## Why it failed (if applicable)

(none yet — OPEN)

## Interactions / superseded-by

- SYV-3 (shelved, root cause confirmed): this is the byte-count follow-up to
  "no kernel can beat ~82% for this shape".
- T1 int8 lm_head: does NOT stack here — our drafter lm_head + mtp.* are already
  unquantized bf16 (AWQ ignore list); shortlisting is the only byte cut for the
  drafter on this model.
- Constraint documented: `use_local_argmax_reduction` (default False) would
  silently bypass the shortlist via `get_top_tokens()`; serving config keeps it
  off. Flagged in code comment + corpus doc.
- K2-Horizon-MoVA-36B-A4B-Q8_0.gguf download completed 2026-09-04 (39.8 GB,
  `/data/models/K2/`) — unrelated to this task; GPU was idle during the probe.

## Refrigerated residue

- Dynamic shortlist variant (98K + 2×512 per-request top-k bootstrap from target
  prefill logits): no corpus needed; only worth it if static's acceptance cost is
  too high. Cross-link: RECON-1cat-vllm.md §CAT-1.
- syv's shipped `draft_vocab_ids.json` (40960 ids, Danish-web-tuned) fetched to
  `/tmp/syv_draft_vocab_ids.json` — usable as a stopgap list for an early A/B
  shape test, NOT as the final list (wrong traffic distribution).

## 2026-09-13 — own-corpus list (35,251 ids) + the parsed-log blind spot

**VERDICT:** PASS with the markup fix — **−2.5 ms/step [95 % CI −2.9, −1.9]
and no detectable acceptance penalty** (Δ mean −2.1 pp, CI [−7.5, +3.4]; Δ median
−0.4 pp) · **GATE:** serving A/B, TP=2 MTP k=2, 11 distinct 8192-token
agentic prompts × 2 reps per arm, identical prompts, arm-level distributions
(boot Y14).

### HYPOTHESIS

A shortlist built from our own traffic beats the pilot's foreign 40K list, and
the acceptance cost measured on 09-12 was caused by the *corpus*, not by CAT-1.

### What was done

- Corpus rebuilt from pi api-logs + pi sessions + hermes (own model only,
  content-hash deduped): 15.5 M tokens ⇒ `cat1_ids_v3.json`, **35,251 ids**
  (every observed id + the tokenizer's 33-id control family), work dir
  `cat1_v3` (361 MB head vs 2.54 GB full).
- **Root cause of the 09-12 acceptance loss: parsed logs cannot contain the
  markup the model emits.** pi records `tool_calls` as structured JSON, so
  `<tool_call>`/`</tool_call>`/`<tool_response>`/`<think>` (ids 248058/59/66/67/68)
  were absent from the corpus and from `all_special_ids` (9 of 33 control ids) —
  the drafter could never propose them. Measured on a captured raw continuation:
  97.7–98.0 % of positions covered before, **100 %** after.
- Builder hardened (`tools/build_draft_vocab.py`): added-token family forced by
  default, truncation warned instead of silent, loud failure on an empty count,
  JSONL detected by content, block-style `messages` content counted,
  session-level holdout by default. Replication guide + scripts:
  `docs/gfx906/CAT1-corpus-build.md`, `tools/build_draft_vocab_corpus.py`,
  `tools/check_draft_vocab_list.py`, `tools/pi-extensions/`.

### Evidence FOR

- **Per-step cost is the robust signal** (22 samples/arm): −2.52 ms mean
  [−2.91, −1.94], median −2.79 ms [−2.98, −2.66], P(Δ<0) = 1.000 — i.e.
  ~6.4 % of a 41 ms step, matching the standalone probe's ~2.7 ms for the
  7.6× smaller head read (39.25 vs 41.77 ms/step).
- **Acceptance: no detectable penalty.** Δ mean −2.07 pp (CI [−7.5, +3.4]),
  Δ median −0.44 pp (CI [−4.2, +2.0]), Mann–Whitney z = −0.83 on 22 vs 22
  samples. The full-head arm's own rep-to-rep spread (sd 8.7 pp) is larger than
  the effect.
- Net t/s (paired per-prompt medians) +4.8 % mean / +5.9 % median, positive on
  9/11 prompts — consistent with the per-step saving with a wash on acceptance.
- Artifact checks clean (`check_draft_vocab_list.py`): must-have tokens present,
  rows byte-identical to `lm_head[ids]`, index adds exactly one key,
  raw-continuation coverage 100 %.

### Evidence AGAINST / caveats

- The 09-12 numbers (86.6/75.7/79.7/77.5 % acceptance) were **invalid**: the
  client put the arm name in the prompt header, so each arm ran a different
  prompt; plus `reps` was forced to 1 for multi-body payloads, and the three
  `agent` bodies are the *same* prompt after the 65,536-token cut.
- Acceptance on this stack is chaotic (±7 pp run-to-run on the full head with an
  identical prompt) — single-sample A/Bs cannot see a 2 pp effect, and **pairing
  one sample per prompt across arms is invalid** because the two arms do not
  reproduce the same trajectory.
- The first pass looked like two "weak prompts" (−8.1 / −9.6 pp). Investigation
  closed it as variance, not a list property: the full-head arm's two reps on
  those very prompts differed by 4.5–10 pp, and fresh logprobs captures of both
  prompts had **0/256 out-of-list positions** (a control had 1). Since a loss
  requires the target token to be outside the list, no loss is possible on those
  continuations. No "weak prompt class" exists in this sample; ~1.5–2 pp of
  residual acceptance cost remains only as an upper bound.
- Measurement hygiene is now recorded in `/local/git/AGENTS.md`.

### Interactions

- Corpus is **use-dependent**: this list is a snapshot of this box's agentic
  coding traffic; chat/RAG/other-language services need their own corpus
  (only the control-token family is workload-independent).
- SUPERSEDES the 40K-pilot headline for the shipping decision: the "zero
  acceptance loss" claim came from the synthetic s9 payload, which cannot
  express the markup loss at all.

## Search keys

`HYPOTHESIS:` drafter lm_head byte cut via static token shortlist, lossless scatter-back.
`VERDICT:` PASS — own-corpus list (35,251 ids, control-token family forced): **−2.5 ms/step [−2.9, −1.9]**, no detectable acceptance penalty (Δ median −0.4 pp), net +4.8 % mean t/s at 8k context. The earlier 40K pilot (+3.86/4.38 % k=2, +5.6/3.9/3.8 % k=4) and its "zero acceptance loss" came from the synthetic s9 payload and are **superseded** for the shipping decision; the 09-12 "loss" numbers were an invalid harness (arm name in the prompt) plus the parsed-log blind spot. (Note 2026-09-14: the same arm-labelled client was in use here, so the *acceptance* column of that A/B is void — the arms ran different prompts. The ms/step result is unaffected (at fixed k the per-step cost does not depend on acceptance), and "no acceptance penalty" is now independently supported by the clean V2 re-measure: acceptance 1.98 vs 1.98 @64k, 2.07 vs 2.10 @120k.)
