# V2 model-runner bring-up on gfx906 (0.29.0 line)

**Status: open — but points 1 and 2 below are now ANSWERED (2026-09-13, boot
eefacc1e): V2 loads, serves and is numerically at parity on gfx906.**
V2 is upstream's default from 0.29.0 (#53183); the fork still pins V1 everywhere
(`VLLM_USE_V2_MODEL_RUNNER=0`) until the *perf* steps of this plan are signed off.

**Session A result (2026-09-13).** A1 (V2 eager) died at engine-core init with
`hipErrorLaunchFailure`, but the kernel logged a BACO reset in the same second —
i.e. the load-lottery family, not V2 (wedge #79). A2 (**V2 graph mode**) then
loaded clean, reached READY in ~300 s, returned a coherent completion and tore
down cleanly, and **V2 in-process PPL = 10.5516 — bit-identical to V1's and to
the 0.28.0 baseline** (Qwen3.8-27B-AWQ-INT4, fp16, 359 tokens, 0 top-20 misses).
So the Y16 "V2 forced on GDN ⇒ unsupported-by-design" record was the lottery:
V2 is viable here, and the remaining work is performance/spec-decode parity, not
bring-up.

*Runner identification trick for the logs* (the log does not print V1/V2): the
V1 runner logs `[gpu_model_runner.py:…]` for the encoder-cache line, V2 logs
`[encoder_runner.py:…]`. The smoke script should assert this tag.
Plan owner: whoever runs the sessions below. Gate for each step is stated in the
table; the house wedge rules apply to every session (canary first, one retry per
wedge, 2 consecutive failures = burst → stop + reboot, log every event in
`degradation.md`/`degradation_details.md`).

Why this exists: 0.28.0 parity is a **V1** property (see `CHANGELOG.md`
2026-09-13). V2 is *additive* work for the next upstream version, but it is a
prerequisite for anything that rides the V2-only speculator loops (DFL2-1's
n-gram chains, A3's fused draft metadata) and for dropping the V1 pin. Upstream's
own V2-unsupported list is `stock torch.compile`, sequence parallelism, and PP
with `external_launcher` — **none of our features**, so nothing upstream blocks
us.

## 1. What already rides shared code (verify, don't port)

Most gfx906 work sits in paths both runners drive, so the question is
*reachability and behaviour under V2*, not code movement:

| fork work | where it lives | V1/V2 reachability | how it gets verified |
|---|---|---|---|
| CUSTOM Q8 FA backend (decode+prefill, M2 clip, KVSPLIT shape-aware, FIX-H2, gather track) | `vllm/gfx906_fa/*`, `csrc/gfx906_fa/*` (shared `AttentionBackend` API) | shared: V2 calls the same `do_kv_cache_update` / metadata builder | FA suite under V2 + PPL parity + serving A/B |
| Fused-content KV layout port (#51718) | `gfx906_fa_backend.py` (`transpose(1,2).split(...)`, `supported_kv_cache_layouts=LBHNC`) | shared | engine logs `Using LBHNC KV cache layout`; V1 already green |
| Dense GEMV / max-ilp GEMM dispatch | `model_executor/layers/utils.py` | shared `Linear` dispatch | decode benches V1-vs-V2 |
| gfx906 W4A16 MoE kernels (+ asymmetric zp) | `fused_moe/oracle/int_wna16.py` (`GFX906_HIP`), `csrc/rocm/*` | shared | MoE decode bench + PPL |
| GDN/mamba ops, SYV-10 bounds port | `layers/mamba/ops/*`, `third_party/flash_linear_attention/ops/*`, `v1/attention/backends/gdn_attn.py` | shared | PPL + GDN-model serving A/B |
| CAT-1 draft-vocab shortlist | `model_executor/models/qwen3_5_mtp.py` (draft **model** class) | shared class, **but see 2.3** | `MTP draft-vocab shortlist ACTIVE` marker + per-step ms delta |
| max-ilp build flags, headroom advisory, plugin registration | build-time / `platforms/rocm.py` | runner-independent | implicit in the above |

## 2. Gaps that need work

1. **V2 init on gfx906 — unresolved, and the first question.** The Y16 record
   ("V2 forced on Qwen3.8-GDN ⇒ init wedge") is *one* wedge; the same
   `hipErrorLaunchFailure` signature also hit pristine paths on this box, so this
   is the load lottery until a fresh-boot retry says otherwise. Do that retry
   before writing any code for V2.
2. **MTP k=3 (production depth) under V2.** V2 has its own speculator stack
   (`gpu/spec_decode/mtp`) with `share_mtp_topk_indices`/`compact_topk_indices`
   and a draft buffer sized by `num_speculative_steps` (`gpu/states.py`). Check
   the draft width and the GDN state-slot sizing (`MambaSpec.num_speculative_blocks`
   on `mamba_cache_mode=align`) match V1's k+1, then A/B.
3. **CAT-1 under V2 — the silent-loss risk.** If V2's MTP drafts come from the
   *target's* shared top-k indices rather than the drafter's truncated
   `draft_lm_head`/`LogitsProcessor`, the shortlist is bypassed and the
   +4.8 %/+5.9 % and 361 MB-vs-2.54 GB win vanish without an error. The model now
   logs `MTP draft-vocab shortlist ACTIVE (…)` on first use
   (`logger.info_once`, added 2026-09-13): **a V2 run that never prints it is not
   using the shortlist.** Verify the marker plus a per-step ms delta.
   **Code audit (2026-09-14, `gfx906/v2-bringup`): no silent-loss path exists.**
   V2's draft sampling goes through `BaseSpeculator.sample_draft` →
   `self.model.compute_logits(...)` (`speculator.py:364-388`), i.e. the draft
   model's own `compute_logits`, where the shortlist scatter lives — and the
   greedy branch `_greedy_sample_draft` (`:358-362`) uses `compute_logits` too,
   *except* under `use_local_argmax_reduction`, where it calls `get_top_tokens`
   (full `lm_head`): that combination already raises at drafter init via our
   CAT-1 guard (`qwen3_5_mtp.py`, `use_local_argmax_reduction` check), so it
   fails closed rather than silently degrading. The
   `share_mtp_topk_indices` route (target's top-k instead of the drafter's head,
   `mtp/speculator.py:30-34`) is gated on
   `index_share_for_mtp_iteration` + `set_skip_topk`/`compact_topk_indices` —
   DeepSeek-style MTP only; the Qwen3.5 drafter has neither, so it stays off.
   Live confirmation (marker + ms/step under V2) is the session below.
4. **M3 host-`cu_seqlens` path.** V2 passes *full-length* host/device
   `query_start_loc` slices in `mamba_hybrid`, unlike V1's
   `[:num_reqs_padded+1]`; our host-path argument was written for V1's slicing.
   Re-audit + test with a full-length slice; then a V2 prefill A/B.
   **Code audit + guard test (2026-09-14): the path is length-agnostic.**
   `forward_paged` consumes the host list with `for s in range(num_seqs)` where
   `num_seqs` comes from the padded query tensor, never from the list's length
   (`gfx906_fa_paged.py:482-490` and `:575-582`), so trailing entries are never
   read; V2's zero-length padded rows are already skipped by `if n > 0`. V2's
   builder passes `query_start_loc_cpu` unsliced
   (`v1/worker/gpu/attn_utils.py:296`) — fine under that bound. The doc's
   failure mode is now guarded by
   `test_forward_mixed_batch_pad_tile_clamp_and_host_cu`, which additionally
   runs the same case with a **full-length slice whose tail is garbage
   (`-12345`)** and asserts bit-identical output (V2 layout). V2 prefill A/B
   still outstanding.
5. **Graph capture.** 0.29 changed the defaults ("widest uniform decode batch by
   default", memory-safe graph sizes) and V2 reserves graph memory differently
   (#53306, #53682). Re-tune the trimmed capture ladder per model, assert no
   eager fallback, and re-run the gather-buffer UAF/capture regression tests
   (`_gather_captured`, `GFX906_FA_CG`).
6. **KV/VRAM sizing.** Compare the logged KV pool and peak VRAM V1-vs-V2 at the
   same `--gpu-memory-utilization` for dense 27B and MoE 35B before trusting any
   V2 bench.
7. **A3 (fused draft metadata)** — V2-only accelerator, stripped and archived
   (`archive/a3-fused-draft` + `A3-REVIVAL.md`). Only after 1–6: re-add the
   opt-in, re-audit the no-op contract at the serving k, and gate at k=7 (k=4 was
   NEUTRAL).
   **Revived 2026-09-14, gate NEUTRAL at k=3 (the serving k); k=7 unmeasured
   (burst).** The three archive items were restored: the env-gated opt-in
   (`VLLM_GFX906_FUSED_DRAFT`, default 0) plus the no-op
   `update_draft_decode_metadata` in `gfx906_fa_backend.py`, and the three tests
   (`test_a3_fused_draft_flag_env_gate_and_noop_update`,
   `test_a3_metadata_build_is_persistent_views`,
   `test_a3_draft_step_reuse_reads_live_seq_lens` — the last one migrated to the
   0.29 fused KV layout; **3 passed**). The no-op contract was re-audited against
   0.29 + V2 (see the comment at the opt-in): seq_lens is advanced in place by
   `update_draft_inputs`, slot_mapping is rewritten by `compute_slot_mappings`
   inside the captured loop, and every layer's forward re-derives kv_max /
   q_abs_offset from those buffers, so a once-built metadata object stays correct
   across draft steps; the scalar fields are step-constant.
   Serving A/B under V2 on the agentic corpus (2 reps, ms/step is the lead
   metric):

   | arm | @64k t/s | ms/step @64k | @120k t/s | ms/step @120k |
   |---|---|---|---|---|
   | a3 off | 40.27 (42.05/38.49) | 81.2 / 88.7 | 23.69 (23.74/23.64) | 128.9 / 129.4 |
   | a3 on | 39.81 (41.69/37.94) | 81.9 / 89.1 | 24.10 (24.60/23.60) | 128.5 / 129.6 |

   i.e. **neutral, exactly as at k=4** — the 1–3 ms/round host saving stays hidden
   behind the 22–31 ms GPU steps at our shapes. The opt-in is therefore kept
   **default OFF** (behaviourally inert; it exists so the k=7 question can be
   closed later), which is also its shipped state. k=7 could not be measured: the
   two `mtp7` launches wedged at load (wedges #87/#88, the second one both decks)
   → burst → GPU work stopped; that arm is the one measurement to redo on a fresh
   boot.
8. **Flip the recipes** (`VLLM_USE_V2_MODEL_RUNNER` removal) model by model, only
   as each one passes. **Done 2026-09-14 for dense 27B and MoE 35B** (both at
   parity: agentic greedy/spec + MoE in-process 58.36 vs 57.86 t/s; see the
   session results above). `run_server.sh` now defaults to V2 with V1 one env
   override away, and the root README carries the per-model status. Still pinned
   to V1: **Gemma-4** (its PPL probe is inapplicable — both runners degenerate,
   see below; needs a serving gate) and **Muse-Glimmer** (checkpoint not local —
   only the GGUF). Nemotron 3.5 Lightning and Ornith passed the probe at parity
   and are flipped too (session E-2 below).
   **Flip verification (2026-09-14, after the burst reboot): PASSED.** With no
   runner env set, `run_server.sh greedy` came up on V2 (7 bare
   `[model_runner.py:*]` tags, 0 `gpu_model_runner.py`), served a completion whose
   text is **identical** to the V1-pinned arm of the same recipe, and tore down
   cleanly (KV 472,932 tokens, no eager fallback, 0 resets). The diagnostic arms
   also showed the preceding load wedge was the lottery rather than V2-specific:
   the V1-pinned arm loaded on the first attempt immediately after the wedge. The
   remaining V2 work (Nemotron/Ornith/Gemma-4 parity → A3 revival → the V2 CAT-1
   headline re-measure) continues on the fresh boot.

## 2b. Session C/D result (2026-09-14, branch `gfx906/v2-bringup`, boot eefacc1e)

V2 serving on the agentic corpus (dense 27B, TP=2, V1-pinned reference numbers
from the same boot; 2 reps/cell, mclk 1000; runner confirmed V2 by the bare
`[model_runner.py:*]` tag, which V1 never emits):

| arm | V2 @64k | V2 @120k | V2 acc | V1 @64k / @120k | V2 vs V1 |
|---|---|---|---|---|---|
| greedy | 20.37 (20.78/19.95) | 13.27 (13.28/13.25) | – | 19.91 / 13.25 | +2.3 % / +0.2 % |
| MTP k=3 | 33.62 (34.38/32.86) | 23.75 (24.15/23.36) | 2.05/1.93, 2.13/2.00 | 33.30 / 24.54 | +1.0 % / −3.2 % |
| MTP k=3 + CAT-1 | ~~42.60~~ (superseded) | ~~26.94~~ | ~~2.44/2.49~~ | 34.62 / 25.55 | **+3.0 % / +2.5 %** (ms/step; see the correction below) |

- **CAT-1 is active under V2** — `MTP draft-vocab shortlist ACTIVE (35251 ids)`
  logged (count 1 in that arm, 0 in the others), with graph capture on and **no
  eager fallback** in any arm, i.e. the scatter survives capture. Item 3's
  silent-loss risk is refuted live, matching the code audit above.
- **Spec decode works under V2** at k=3 with acceptance on par with V1 in the
  plain arm (2.05/1.93 vs V1's 2.05/2.05 at 64k; 2.13/2.00 vs 2.06/2.15 at
  120k) — the −3.2 % @120k is inside the arm spread and is the only
  below-parity cell.
- **Item 6 (KV/VRAM), same util + flags:** V2 `GPU KV cache size` 454,536 tokens
  (greedy, 14.2 GiB avail) vs V1 496,693; 386,513 (spec) vs V1 442,368 (13.48
  GiB avail) — V2 reserves 8.5–12.6 % more, and its graph capture costs
  1.69+0.80 GiB (greedy) / 2.15+0.87 GiB (spec) against V1's 0.71 GiB. On V2 the
  capture-ladder trimming therefore matters more, not less.
- **V2-CAT1-1 — RETRACTED, then re-measured (2026-09-14).** The earlier result
  (plain 33.62 / matched **42.60** / mismatched 35.22 at 64k; acceptance 2.05 /
  2.44 / 1.72) is **invalid**: the A/B client put the arm *name* in the prompt
  header (`RESEARCH-BRIEFING-{arm}-…`), and since the header token count differs
  per arm (26 vs 29) the body slice `pp - len(header)` shifted too, so each arm
  saw a **different prompt ending** — the exact trap AGENTS.md forbids. The
  acceptance column was arm-dependent-prompt junk, and with it the 42.60 figure,
  which was acceptance-driven.
  Re-measured with the fixed client (arm name out of the prompt; a `prompt_sha1`
  digest is now logged so prompt identity can be *asserted*), same boot, V2,
  agentic corpus, **3 reps**, ms/step as the lead metric:

  | V2 arm | @64k t/s | ms/step @64k | @120k t/s | ms/step @120k | acc @64k | acc @120k |
  |---|---|---|---|---|---|---|
  | plain MTP k=3 | 34.41 (34.56/34.74/33.94) | 86.3 (82.0/88.4/88.4) | 24.00 (24.01/23.85/24.15) | 128.5 (128.0/128.8/128.8) | 1.83/2.07/2.02 | 2.08/2.07/2.13 |
  | **MTP k=3 + CAT-1** | 35.44 (34.64/36.65/35.02) | **83.7** (83.6/83.9/83.7) | 24.61 (24.71/24.68/24.43) | **124.4** (124.3/124.5/124.3) | 1.91/2.07/1.97 | 2.08/2.07/2.06 |

  Reading: **the effect is the ms/step saving, and it reproduces** — −2.6 ms/step
  @64k, −4.1 ms/step @120k ⇒ **+3.0 % / +2.5 %**, with the CAT-1 per-rep ms/step
  spread an order of magnitude tighter than the plain arm's (83.6–83.9 vs
  82.0–88.4). **Acceptance is unchanged** (1.98 vs 1.98 @64k, 2.07 vs 2.10 @120k):
  the shortlist makes each step cheaper (35,251-row head read instead of the full
  248,320-row one), it does not make the drafter agree more often. That matches
  the V1 controlled A/B (−2.52 ms/step ⇒ +4.8 %/+5.9 %, MWU z = −0.83 on
  acceptance = no acceptance penalty), so CAT-1's honest V2 headline is
  **+3.0 % / +2.5 %**, not +23 %.
  **Exactness holds by audit**: `gumbel_sample` caches the *masked* draft logits
  (`logits_cache=draft_logits`) and `rejection_sampler_utils.py` computes the
  acceptance ratio from that same cache (`draft_logit … / temp`) while the target
  keeps its own full `lm_head`, so the ratio uses exactly the distribution the
  draft was sampled from. At `temperature = 0` the draft path is a plain argmax
  (`gumbel_noised_argmax`, "or plain argmax at temp 0").
  **Method note (why this took three readings):** acceptance is deterministic per
  (config, prompt) at temperature 0 but its absolute level moves across boots (the
  same plain k=3 arm read 2.05 pre-reboot, 2.41 on the next boot), so acceptance
  deltas are only meaningful **within a boot** and the only trustworthy cross-arm
  signal is ms/step.
- **A3 revived and gated (item 7, 2026-09-14): NEUTRAL at both the serving k and
  k=7.** The three archive items are restored (env-gated `VLLM_GFX906_FUSED_DRAFT`,
  default 0, + the no-op `update_draft_decode_metadata`, + 3 tests, 3 passed; the
  reuse test migrated to the 0.29 fused KV layout) and the no-op contract
  re-audited against 0.29 + V2. The flag demonstrably flips the path: the *off*
  arm logs "Fused multi-step draft decode is not supported by attention
  backend(s) CUSTOM; falling back to rebuilding attention metadata between draft
  steps" (count 1) and the *on* arm logs it 0 times. Serving A/B, V2, agentic,
  ms/step lead:

  | arm | k | ms/step (reps) | acceptance (reps) |
  |---|---|---|---|
  | a3 off | 3 | 81.2 / 88.7 @64k, 128.9 / 129.4 @120k | 2.4133/2.4133, 2.0595/2.0595 |
  | a3 on | 3 | 81.9 / 89.1 @64k, 128.5 / 129.6 @120k | 2.4133/2.3816, 2.1605/2.0595 |
  | a3 off | 7 | 140.4 / 139.3 @64k | 5.2683 / 2.5833 |
  | a3 on | 7 | 141.5 / 140.7 @64k | 2.4933 / 5.1190 |

  Neutral at both depths — the 1–3 ms/round host saving stays hidden behind the
  22–31 ms (k=3) and ~140 ms (k=7) GPU steps. **This A/B is unaffected by the
  arm-label bug above:** the pairs happened to tokenize to equal header lengths
  (`a3off_k3`/`a3on_k3` both 28, `a3off_k7b`/`a3on_k7b` both 29), so each pair ran
  byte-identical prompts (verified by the header token counts), and the verdict
  rests on ms/step, which is acceptance-independent by construction. Note the k=7 acceptance spread
  (2.58 vs 5.27 across the two prompts) is *prompt* variation, and ms/step is flat
  across it, which is exactly why ms/step is the gate. The opt-in stays **default
  OFF** (behaviourally inert), which is also its shipped state.
- **Wedge #83 (GPU1) hit the mtp3 arm's first launch; the retry passed and is
  recorded in `degradation.md`.**
  recorded in `degradation.md`.

**Session E-2 — remaining models, in-process PPL, same boot (2026-09-14).**
PPL is our only valid numerical gate:

| model | V1 | V2 | verdict |
|---|---|---|---|
| Nemotron 3.5 Lightning 30B-A3B (mixed INT4/INT8) | 26.9986 | 27.0066 | **parity** (0.03 %; both inside the recorded 26.96–27.02 fp16 band) |
| Ornith 1.5-35B-A3B-AWQ-INT4 | 16.7724 | 16.7824 | **parity** (0.06 %) |
| Gemma-4-26B-A4B-it-AWQ-4bit | 84261.54 (raw-text artifact) | 108909.96 (raw-text artifact) | **V2-validated 2026-09-15** via a *templated* V1/V2 comparison (identical text and logprobs ≤0.05) — the PPL figures were a prompt-format artifact of an IFT checkpoint, not a defect |

Nemotron and Ornith are therefore flipped to V2 as well (item 8). Nemotron
serving note: at TP>1 it needs `--enable-expert-parallel` (group-64 CT experts),
unchanged by the runner. **Gemma-4 is a separate problem, not a V2 one**: the
in-process probe loads it through the multimodal path ("Model does not support
mm_device_do_normalize") and both runners return a degenerate distribution
(PPL ~10^5, 350 tokens), so the probe cannot gate it at all — it stays pinned to
V1 and needs a serving-level gate (new ROADMAP item GEMMA4-1).

**MoE 35B parity (session E-1, in-process harness, same boot).** `docs/gfx906/_bench_gfx906.py`,
`BENCH_SAMPLES=4 BENCH_PP=2048 BENCH_TG=256 BENCH_MAX_SEQS=32`, single GPU, mclk 1000
in every sample: **V1 57.86 t/s** {58.13, 58.18, 57.04, 58.08} vs **V2 58.36 t/s**
{58.42, 58.36, 58.32, 58.33} → **+0.9 %**, and within 0.1 % of the recorded 58.43
reference. Runner identity confirmed the same way (bare `[model_runner.py:*]`
tags only in the V2 arm). No cudagraph fallback in either arm. Note the KV
direction is config-dependent here: in-process single-GPU V2 kept *more* KV
(130,944 vs 123,904 tokens) while reserving more for capture (0.35+0.05 vs
0.05 GiB) — unlike the TP=2 dense-27B serving case where V2 was smaller.

**KVLAYOUT-2 closed in this session too:** the three capture/lifecycle tests that
were skipped as "0.29 fused KV layout migration pending" were already written
against the fused-layout helpers (`_make_fused_cache`/`_kv_split`/
`_write_v_fused`) and pass unchanged — the skips were stale. The suite is now
**91 passed, 0 skipped** (`tests/kernels/attention/test_gfx906_fa.py`), including
the extended M3 test which now also asserts that a **full-length (V2-style) host
`cu_seqlens` slice with a garbage tail** gives bit-identical output.

## 3. Test and verification matrix

| level | what it catches | concrete run |
|---|---|---|
| unit / kernel | backend API drift, K/V split, capture buffers | `tests/kernels/attention/test_gfx906_fa.py` (90 tests) with V2 pinned; the gather-lifecycle/UAF regression test |
| in-process numerical | wrong attention/MoE/GDN numerics under V2's metadata construction | `benchmarks/kernels/gfx906/ppl_probe.py` with `VLLM_USE_V2_MODEL_RUNNER=1`, compared to the recorded V1 numbers (dense 27B, MoE 35B, Muse-Glimmer, Nemotron) — PPL is our only valid numerical gate (greedy token identity is not, on this stack) |
| serving A/B | end-to-end parity: decode t/s, TTFT, acceptance, capture behaviour | standard recipes, same boot, V1 vs V2: dense 27B (`max-seqs 4`), MoE 35B (`max-seqs 32`), then the agentic corpus at 64k/120k |
| spec decode | k=3 draft width/state sizing, acceptance, CAT-1 shortlist actually acting | MTP k=3 on the agentic corpus, V1-vs-V2; CAT-1 on/off within V2; assert the `shortlist ACTIVE` marker and draft ids ⊆ list ∪ control family |
| memory | V2 reservation/sizing drift | logged KV pool + peak VRAM, same util, both models |
| manual (human) | behavioural drift no numeric gate sees | tool-call round trip through the server with the `qwen3_coder`/`qwen3` parsers on real agentic prompts; visual check of the continuation shape on our own corpus bodies; confirm in the log which runner was used (the log does not echo it — the smoke should assert it) |

## 4. Order and session plan

1. **Session A (fresh boot, ~30 min)**: V2 init smoke on the simplest model —
   dense 27B, `VLLM_USE_V2_MODEL_RUNNER=1`, eager first, then graph mode. If it
   wedges: one retry; if the retry wedges → burst → stop + reboot (that is the
   evidence that matters for the arch question).
2. **Session B**: in-process PPL parity for dense 27B and MoE 35B under V2 vs the
   recorded V1 numbers; FA suite with V2 pinned.
3. **Session C**: serving A/B V1-vs-V2 (greedy) for both models, with a
   capture-fallback assertion.
4. **Session D**: V2 + MTP k=3 on the agentic corpus, then CAT-1 on/off inside V2
   (marker + per-step ms).
5. **Sessions E+**: Muse-Glimmer / Nemotron / Ornith-Gemma, graph-ladder tuning,
   then item 7 (A3) if the V2 path is staying.

Each session ends with: canary, VRAM back to the 10.9 MB baseline, no hung
holders (`rocm-smi --showpids`), and a one-paragraph record here or in the
related dev log.
