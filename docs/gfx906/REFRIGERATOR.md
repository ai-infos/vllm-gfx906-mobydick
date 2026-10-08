# gfx906 refrigerator — parked work with reopen gates

The index of shelved/parked/deferred items: not active targets, but
cheap to reopen if their gate fires. Symmetric with `DEAD-ENDS.md`
(closed negatives) and `CHANGELOG.md` (closed positives). Every entry
states **why parked** and the **reopen gate** — an item leaves this file
only by meeting its gate (or by a user decision). Dev-log-level
refrigerated levers stay in their `DEVLOG-*.md` residue sections; this
file indexes roadmap-level items only (see Cross-references).

## WHT-1 GGUF half — Q8_0 onboarding for Whittle-Qwen-3.8-35B-A3B

**Parked: tooling-blocked (user decision 2026-10-05, when WHT-1 was closed with the bf16
path serving).** The item was opened to test `Whittle-Qwen-3.8-35B-A3B-Q8_0.gguf`
(`fb13eb28e82d0bd44394f886e119424e584a82dd0883c79c64cf06d8f48c52b1`, 37,828,807,904 B, kept
at `/local/models/logic65/Whittle-Qwen-3.8-35B-A3B-GGUF/`) so the `qwen4_exp` line could be
optimized against this box's VRAM. Two blockers: the fork has **no in-tree GGUF loader**
(upstream moved GGUF to the out-of-tree `vllm-gguf-plugin`; `gguf_loader.py` and
`layers/quantization/gguf.py` are absent and the venv has no `gguf` module), and
**llama.cpp is not installed** on mi50-01, so there is no reference implementation to compare
against either. **Reopen gate:** the OOT `vllm-gguf-plugin` can be made to build and load on
gfx906 (ROCm 7.x, `extra-quant: ["vllm-gguf-plugin>=0.0.2"]`) *and* its quant path passes a
coherence gate on this box, or llama.cpp becomes available for the reference side. On reopen:
onboard the Q8_0 file, gate coherence against the bf16 serving path, and measure what the
quantized file does to the host-resident-PLE footprint — that comparison, not the throughput,
is why the item existed.

## DeepSeek-V4-Flash

**Parked: hardware-blocked, not an active target (user decision
2026-08-29 — others can test it for us).** 43 layers plus MTP,
E=256/topk=6 MoE with FP4 experts, FP8 dense weights, DSA sparse-indexer
attention, compressed KV, hidden size 4096. Expert memory ≈ 140 GB at
FP4 (more at FP8) vs 64 GB across the two cards; TP=2 is not a reliable
sharding path on this machine and DP would replicate the model.
**Reopen gate:** a smaller variant, a substantially smaller checkpoint,
or a working multi-card sharding path. If reopened, validate format
conversion, the K=4096 W4A16 extension, DSA/MLA attention, and
sqrtsoftplus/topk-6 routing independently.

## M6 Part C — Q4-KV via native `v_dot8_i32_i4`

**Parked: user decision 2026-08-28 (`5d8d4c7f59`).** Quality unproven —
Q4 K *and* Q4 Q (or Q8→Q4 requant) accuracy is unvalidated on this
model family; the 7-level q4_0 codebook roughly doubles the KQ
quantization error with no measured PPL evidence. **Reopen gate:** a
dedicated accuracy gate (PPL probe bands on the 442-token set, Q4-KV vs
Q8-KV arms) that passes *before* any kernel work. The measured ISA rates
motivating it are in `dequant-instructions.md` (`v_dot8_i32_i4`
49.6 T MAC/s, 2× dot4 at half the operand bytes).

## C9 — overlap shared and routed MoE work

**Parked: no overlap window.** The shared-expert chain is independent of
routed work, so a multi-stream fork/join might hide part of its cost,
but vLLM currently captures a single stream. **Reopen gate:** a
concurrent/batched decode project where the overlap window is large
enough to measure.

## P2-1(e) — persistent-CTA MoE prefill GEMM

**Parked: out of decode scope.** The earlier prefill effort stalled well
below the practical dot2 peak. **Reopen gate:** prefill becomes a
performance target.

## Speculative decoding (former spec-decode-roadmap.md)

Shipped recommendation on record: MTP k=2 for Qwen3.5-27B/-35B
(`running.md`, `DEVLOG-spec-decode.md`); Qwen3.8-27B serving uses ngram
n=5 (repo serving defaults). Completed phases are in `CHANGELOG.md`;
detailed evidence in `DEVLOG-spec-decode.md`,
`DEVLOG-gdn-mixed-decode.md`, `DEVLOG-fp16-skinny.md`.

### SYV-12 — context-lookup verify extension (MTP fill slot)

**Parked: v2 (post fill-fix) +6.4 % @120k / −5.3 % @64k, payload-conditional — parked by Kevin 2026-09-11; earlier v1 (k=2 + EXT=1) CLOSED negative for v1-as-built (2026-09-08,
boot Y3 same-boot A/B: −9.3 % @64k / −5.3 % @120k, mixed-v2, 5/6 reps
negative; lossless 512/512; code stays env-gated `GFX906_SYV12`
default-OFF). The close does NOT establish the mechanism fails
structurally — its yield was never measured. As of 2026-09-09 the
root-caused fill fix is VALIDATED on s9 (pos3 0.987, mean ~3.99;
see reopen gate below) — parked pending the production A/B.** Attribution correction
(2026-09-08, verdict review of the close): per-position acceptance
shows the fill row at **0.000 in every s9 probe window** (drafted
3/step, accepted never; mean acceptance capped 3.00) and 0.006
step-weighted on mixed-v2 — while a static reconstruction of the fill
kernel over the real s9 history proves the correct fill value is
511/511 = 100 % there, and an end-to-end wiring audit (gate, history
append, fill kernel, input scatter, sampler alignment) found no static
flaw (the "sampler caps at k" candidate is refuted) — a **runtime
defect** was declared open at that point. **2026-09-08 (pre-probe,
boot Y3): root cause found + fixed — it was a static CONTRACT bug the
wiring audit and the 16/16 unit test had both baked in:** the v1
fill's lookup suffix was the trailing history only, which ends at the
step's ANCHOR, so its continuation targets the d0 slot while the value
is stored in the FILL slot (after d0, d1) — off by k=2 positions. That
single off-by-k explains both observed regimes: s9 (period-9 loop) →
old fill ≈ the d0 value, never the FILL-slot argmax → 0.000 in every
window; mixed-v2 → accepted only when a token repeats across the
anchor→d1 span → the ~0.6 % incidental hits. (The "511/511 static
reconstruction" measured the kernel's own contract — d0-continuation
vs the next token — not FILL correctness.) Fix: suffix = last
(MIN_MATCH−k) history tokens + the k base drafts (which end at
d_{k-1}, the token before the FILL position); unit-verified 16/16
under the corrected contract. The measured A/B loss is
the pure always-paid 4th-row GDN cost (+33 % per-step state traffic;
FA unchanged — both blocks pad to Sq_pad=4) with ~zero fill yield. The
point-mass draft-probs fallback (all drafts, while SYV-12 is active)
showed no measurable base-draft degradation in the recorded A/B but
must be removed in any v2. Note the revival regime is NOT s9-class
(s9 is where MTP already saturates: pos1/pos2 = 1.000) — it is
verbatim-span-heavy generation where the MTP head misses (syv's +47 %
reproduce-a-document case). **Reopen gate (in order, updated 2026-09-09 post-validation):** (1)
the instrumented s9 in-process probe (per-step dump landed in-tree,
env-gated default-OFF; driver + analyzer under /local/tmp/syv12/) —
now a VALIDATION run of the fixed fill, with the dump's
(A) kernel-write / (B) input-path / (C) sampler-verdict decomposition
as the fallback localizer if anything still misses — **PASSED
2026-09-09 (boot Y4)**: (A) 96/96, (B) 65/65, gate/append 0/0;
pos3 = 0.987 steady state (single miss = the prefill->decode boundary
step, gate off by design), mean acceptance ~3.99 (vs ~3.0 under v1),
identity perfect, eager decode 31.65 t/s (vs 24.56 v1-as-built,
+28.9%). Two boot-Y4 defects found on the way, both fixed: a Triton
compile assert (mixed int32/int64 phi in the match loads — the unit
test had compiled only the all-int32 signature; `fb971c54a0`, test now
uses the live int64 draft dtype) and the TP-rank dump interleaving
(`2eacd8e8ed`, per-pid suffix). (2) — **DONE 2026-09-09 (boot Y4): PAYLOAD-CONDITIONAL.** Same-boot
production A/B, boot-Y3 protocol (mtp k=2 TP=2, mixed-v2 64k+120k
×3, OFF first): ON 27.99/23.80 vs OFF 29.55/22.37 → **−5.3 % @64k,
+6.4 % @120k (medians)**; @120k clean separation (min ON 23.28 > max
OFF 23.06), @64k overlapping. Realized fill acceptance ≈25 % of
steps @120k (above the review's 15–20 % bar) vs ≈0–5 % @64k (fill
effectively never fires there; ON = OFF + the always-paid 4th-row
GDN cost). Lossless in serving (text probes clean both arms). The
bar "ON ≥ OFF at the mixed points" is met at 120k only — the
mechanism is payload-conditional exactly as the review predicted,
so SYV-12 v2 stays PARKED — **park confirmed by Kevin 2026-09-09
until further review; the B=4 campaign runs first (same boot)**;
(2b) **gate variable adjudicated 2026-09-09 (research agent,
CPU-only; handover
`/local/tmp/handover-syv12-context-vs-payload.md`, repro
`/local/tmp/mtp1/syv12_content_vs_context.{py,log}`)**: the 64k→120k
flip is a **payload-position (copy-density) effect, not context
scaling** — the 8,192-token lookup window cannot see context length.
Windowed 5-gram repeat density on the served corpus: served 64k
bodies 12.6 %; same 120k bodies first-65,536 13.6 %; same 120k
bodies LAST 65,536 **21.5 %**; realized acceptance tracks the tail
(≈25 % @120k vs the 21.5 % tail; ≈0–5 % @64k vs the 12.6 %).
Amended parked verdict (one line): *payload-conditional,
position-of-generation-tail-gated (+6.4 % @120k, tail density
≈21 %; −5.3 % @64k, ≈13 %); NOT context-length-gated.*
Consequence: any enablement gate is on payload/runtime density (the
fill path already computes the match statistics — gate on sustained
recent fill-hit rate > ~10 %), NOT on a context-token threshold
(which breaks silently on front-sliced payloads, e.g. the B=4
16k/32k/64k envelope);
(3) if pursued, in order: the crossover A/B as a 2×2 (content ×
depth) — FRONT-SLICES of the 122880-point bodies at 80k+100k (same
text, shallow vs deep; only front-slices isolate depth) + the served
64k point as content control (2 loads) + compiled-mode s9 120k
(2 loads), then the B=4-era default discussion
(the 4th row adds the same cost at B=4 — the campaign should know
whether SYV-12 is in the picture). Small-n caveat (8 bodies/point)
currently fine: per-body spread ±0.002 on the A/B/C/D ordering. Dev
log: `DEVLOG-fa-attention.md` SYV-12 entries (boot Y/Y2/Y3 +
correction); ROADMAP SYV-12 entry.

### SD-L2 — AWQ M≤4 draft-step GEMV

**Parked: estimate says small win.** The M=1-to-M=4 AWQ cost is
approximately 17 ms per agentic draft step, but the q_gemm family is
dequant-ALU-bound: M=1 already reads about 44 MB in 75 µs (roughly
590 GB/s), and the existing tiled M=4 kernel shares dequantization
across rows. An exllama-style four-row GEMV or a q_gemm re-tile is
therefore expected to save only the atomics/LDS/M-tiling overhead,
estimated at 2–8 ms. **Reopen gate:** MTP is no longer the preferred
drafter, or a serving profile shows the estimate is materially wrong.
Gates if reopened: per-shape microbenchmark, the gfx906 MoE/GPTQ tests,
a PPL or greedy gate as appropriate, and an agentic serving A/B.

### T-1.5/A5 — M≤8 W8A16 GEMV kernel (orphaned infrastructure)

**Parked: method layer removed, kernel intentionally kept (2026-09-07, A5
CLOSED — `e289ff17dc`).** `dense_gemv_i8_m4_gfx906`
(`csrc/rocm/dense_gemv_gfx906.cu`, in-kernel fast path for M≤8 int8-W/
fp16-A GEMV) has NO in-repo caller anymore — reachable only via the
`_custom_ops` binding. Do not re-derive it and do not delete it as dead
code: it is the only in-tree W8A16 GEMV for the verify head at M=5..8
(the M>8 path is a per-call dequant fallback). A/B was NO-WIN at k=4
(−0.40/−0.44/−0.13 %, token-identical — the int8 head flipped no
argmax). Full transform spec + probe recipe in
`DEVLOG-t1-int8-fp16-mass.md` (2026-09-07 closure entry). **Reopen
gate:** a B=1 deep-k≤3 head-quant need (the k=2/3 operating point keeps
M ≤ 8, where the kernel's in-kernel path applies) or a serving profile
showing the verify head GEMV back in the critical path.

### SD-suffix — suffix draft-quality probe

**Parked: dependency-blocked.** The suffix proposer needs
`arctic-inference==0.1.1` and has dynamic draft length, so it remains
PIECEWISE-only and does not use the uniform speculative-decode graph
rails. Its only useful result would be a draft-quality comparison
against MTP/ngram. **Reopen gate:** a better drafter is needed AND the
dependency can be installed and verified on ROCm.

### SD-gram — GPU n-gram proposer match-selection fix

**Parked: not an adoption candidate.** The GPU proposer produced 0.428
accepted tokens per draft step versus 1.08 for the CPU proposer and
diverged in repeated-match tie breaking. A line-by-line match-selection
fix could make it useful for deployments without an MTP head.
**Reopen gate:** a no-MTP-head deployment need, behind a draft-quality
and serving A/B gate. Keep the CPU proposer as the default until then.

### SD-future — future drafter models (EAGLE etc.)

**Parked: unplanned.** Would use the existing FA/speculative-decode
rails. **Reopen gate:** a checkpoint exists locally AND a target model
AND an acceptance gate AND a memory budget. Do not add implementation
work before all four.

## Parked at the 2026-09-27 roadmap migration

Items that were carried in `ROADMAP.md` but are not active work. They are kept
here with their reopen gates; no GitHub issues are opened for them.

### DFlash2 family (DFL2-1, DFL2-3, DFL2-4, DFL2-5, DFL2-6, DFL2-7, DFL2-8)

**Parked finally 2026-09-16 (external result), confirmed by the arm-C
measurement.** Upstream vLLM 0.29 is degenerate on the card's own matched pair,
and so is our bf16+bf16 control (`HANDOVER-dflash2.md` §8/§9; per-draft
acceptance 0.0408 / 0.0079). Arm C (DFlash2 without the chain patch) measured
**2.48/2.52 t/s at 64k** (acceptance 0.045/0.063) and **1.37 t/s at 120k**
(acceptance 0.0) against the MTP k=3 + CAT-1 band (35.97/35.44 at 64k,
24.58/24.82 at 120k, reproduced within +0.8 %/+0.4 % in the same boot). The
gap is ~14×, so no downstream patch closes it; `DFL2-8` (the drafter's attention
backend) was the gate for the whole family. **Reopen gate:** a non-degenerate
drafter appears for this family (a fresh matched bf16 pair that is not
degenerate on a stock vLLM). Work branch: `gfx906/dflash2`. Record:
`DEVLOG-dflash2.md`, `HANDOVER-dflash2.md`.

### QSA-FN-5 — int8 `per_token_head` KV for QSA

**Parked: the capacity argument moved.** The item existed to buy KV capacity at a
measured 2.5–2.7× attention-prefill cost, but the tester's working config already
serves 3 × 147 456 tokens on 4 × 32 GB via the PLE ngram table mmapped to host
RAM (26 GB at zero attention cost). **Reopen gate:** a real need appears (more
concurrency at 147k, or fewer cards); then measure the attention share of prefill
on the tester's box first. Record: `RECON-qwen38-flash-qsa.md` §5.2.

### S2 — shard loading time

**Parked: deprioritized by Kevin 2026-09-04** ("not high priority, do not spend
too much time"); fastsafetensors, the main lever, reserves more VRAM than the
usual loaders. **Reopen gate:** load > 3 min cold, or NFS-backed serving becomes
a target.

### SMLA-1 — fp16 sparse-MLA on 0.29's ROCm path

**Parked: inert, half-ported.** The fork's fp16 variant (`VLLM_ROCM_MLA_SPARSE_FP16`,
default off) is split across the backend/hook files while 0.29's restructured ops
file took upstream's implementation, so the path is not expected to work if
enabled. **Reopen gate:** we serve a DeepSeek/GLM sparse-attention model on
gfx906 (needs AITER + gfx942/950 upstream). Preserved in git history.

### FA-NONCAUSAL Stage 2 — symmetric ±window in the kernel

**Parked: not needed.** Stage 1 (full bidirectional for a non-causal batch) shipped
default ON and gave the Muse-Glimmer assistant **≥ +29 %**; the symmetric-window
mask was only a refinement. **Reopen gate:** a windowed-bidirectional drafter whose
acceptance moves when the pre-window is honoured. Record:
`DEVLOG-fa-noncausal.md`.

### SYV-6 — int8 activations (W4A8 Marlin) + negative-scale bug fix

**Parked: batch-mode only.** We run B=1; park until multi-request resumes. The
bug fix is model-portable if the negative-scale issue is ever hit.

### SYV-8 — DFlash2 block drafter

**Parked: covered by the DFlash2 family park above** (big effort, needs V2,
conflicts with the FULLGRAPH path).

## Cross-references (dev-log refrigerated levers, not restated)

- LEGACY=0 Q8-write fusion into `triton_reshape_and_cache_flash` —
  parked in `DEVLOG-fa-legacy0-b1-decode.md`; gated on ROADMAP G1
  (node-overhead measurement).

## Archive branches (gfx906/fa-decode-fp16 closure, 2026-09-12)

Dead-ended / parked experiments stripped-or-stripped-at-merge from the
main line, kept as named branches. Revival = checkout, rebase on then-
main, re-gate per the preconditions.

- **`archive/syv12`** — constructed branch (main + SYV-12 code + dev
  log). 2026-09-12 corpus-agent finding: NOT workable in the current
  kernel/architecture shape (corpus-independent). Revival only if
  the fill/verify SHAPE changes. See `DEVLOG-syv12.md`.
- **`archive/t1-int8-mass`** (pointer @ `a0bb358670`) — int8 W8A16
  drafter mass, NOT PASS at k=4 serving. Revival: only if the serving
  depth regime or the lm_head cost share changes materially; quality
  gate was clean (0/120 argmax flips). Range `e1a57e9a17..a0bb358670`.
- **`archive/a3-fused-draft`** (pointer @ `3fbb4e3f81`; revival recipe in
  `A3-REVIVAL.md` on that branch) — fused
  multi-step draft metadata, NEUTRAL @k=4; **the fork's opt-in was stripped from
  the branch 2026-09-13** (`vllm/gfx906_fa/gfx906_fa_backend.py` reverted to
  upstream's default `False`; brief `/local/tmp/b4/a3-strip-decision.md`).
  Revival: V2 becomes the consuming runner (the fused loop is upstream, V2-only)
  and the draft-step host cost share grows (bigger k or B) — then re-add the
  ~7-line opt-in and re-audit the no-op contract at the serving k. The branch is
  the permanent home of the A3 work (code + tests + the measured verdict + the
  revival recipe) — `archive/*` branches are never deleted. 
- **`archive/fd1-fused-draft-meta`** (pointer @ `abc9d8ded9`) —
  FD-1 decode-metadata path, NEUTRAL @B=4/120k offline (stack-confounded:
  offline arm vs serving control). **FD-1's code IS A3's code** — one flag,
  two gates; FD-1 was the offline B=4/120k gate of it, and the flag has **no
  reader** in-tree since the 2026-09-13 strip. Roles: the A3 entry above is
  the **code + revival-recipe home** (`A3-REVIVAL.md`); this branch is the
  **era snapshot** (full tree @ `abc9d8ded9`, superset — it also holds the
  `bdcbd8ec3a` kv_max clamp fix). Revival: pair with A3; re-gate **same-stack**
  (never offline-vs-serving again). Analysis: `/local/tmp/b4/fd1-keep-strip-decision.md`.
