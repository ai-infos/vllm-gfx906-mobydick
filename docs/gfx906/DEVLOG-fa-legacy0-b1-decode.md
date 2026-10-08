# FA LEGACY=0 B=1 decode gap — kernel-level localization + serving
# adjudication

## 2026-09-16 — KVLAYOUT-1: LEGACY=0 verified on 0.29, and it is 15-19 % FASTER with MTP k=3

**VERDICT:** ADOPT `GFX906_FA_LEGACY=0` for spec-decode serving — numerics verified, and
the ms/step win is far outside this box's per-process drift. The 2026-08-29 B=1 *greedy*
reading below (LEGACY=1 better by 6.3 %) stands for that regime; this is a regime split,
not a contradiction.

- **Gate 1 (numerics).** `GFX906_FA_LEGACY=0` gives **PPL 10.5472 / 10.5460** across runs
  (359 tokens, **0 top-20 misses** in every run) against **10.5472** for this build's LEGACY=1
  (the pre-triton-3.8.0 value was 10.5516) — i.e. the same value to within the probe's own
  run-to-run spread (<= 0.0012), which the two LEGACY=0 runs and the LEGACY=1 run establish by
  themselves. The Q8
  side-buffer byte-alias under 0.29's fused `[B, H, N, 2*D]` layout is therefore correct,
  not merely plausible — the static reasoning in ROADMAP KVLAYOUT-1 is now confirmed by the
  only numerical gate this stack has.
- **Gate 2 (serving A/B, interleaved L1 -> L0 -> L1).** MTP k=3, dense 27B AWQ, TP=1,
  agentic corpus, 2 reps, ms/step:

  | arm | 64k ms/step | 64k t/s | 120k ms/step | 120k t/s |
  |---|---|---|---|---|
  | L1 (legacy, default) | 87.8 / 87.5 | 31.54 / 26.47 | 128.1 / 128.2 | 23.42 / 23.69 |
  | L0 (Q8 side-buffer) | **71.4 / 76.8** | **38.43 / 35.73** | **103.7 / 103.8** | **29.29 / 28.54** |
  | L1 repeat (order control) | 82.0 / 87.8 | 33.43 / 31.21 | — | — |

  Acceptance is identical across arms (1.7634 / 1.7634 / 2.0476), so this is a pure
  ms/step effect: **−15.5 % @64k, −19.1 % @120k** (means 74.1/103.8 vs 87.7/128.2). The
  interleave control earned its keep: L1's own second run was 6-7 % faster than its first
  (82.0 vs 87.8 @64k), yet L0 still beat even that fast L1 sample on every point but one
  (76.8 vs 82.0) — i.e. the win exceeds the drift that has invalidated earlier A/Bs here.
- **Regime split, plainly:** B=1 greedy decode pays ~6 % for this path; spec-decode serving
  at 64-120k saves 15-19 %. Production on this box is MTP k=3, so the default belongs at
  LEGACY=0, with LEGACY=1 kept as the rollback.
- The unit suite is not the instrument here (4 preconditions/diagnostics fail under
  LEGACY=0 — see ROADMAP KVLAYOUT-1).
- **Flipped 2026-09-16.** `_resolve_legacy_mode()` now defaults to the side-buffer path; the
  fail-closed refusal and `GFX906_FA_LEGACY_ALLOW_UNVERIFIED` are removed as obsolete, and the
  always-on "experimental read path" warning is demoted to a debug line. Five LEGACY=1-only
  tests (gather-buffer lifecycle, q_pad capture grow, the A3 fused-loop contract, the
  gather-retire warning) now pin `GFX906_FA_LEGACY=1` themselves, and the fail-closed test
  became `test_legacy_default_is_side_buffer_after_kvlayout1`. Gates on the flipped tree:
  **FA suite 97 passed**; PPL under the default **10.5472 / 10.5460 / 10.5472** across three
  runs (LEGACY=1: 10.5472, 0 top-20 misses throughout).

## 2026-08-29 — B=1 LEGACY=1-vs-0 decode gap (roadmap item #1)

**VERDICT:** DEAD-END (flip question closed: LEGACY=1 stays the
default) · **GATE:** same-boot (boot O) TP=2 serving A/B,
Qwen3.8-27B B=1 pp2048/tg256, 2 samples/arm — A 40.11/40.12 vs
B 37.61/37.56 (−6.3 %) vs C 37.55/37.54 (−6.4 %) t/s.

## HYPOTHESIS

If LEGACY=0 B=1 loses to LEGACY=1 (the M5-era 2.5–3.7 % serving gap,
never same-boot adjudicated: 107.2 boot M vs 111.5 boot L), the loss
sits in a named suspect — quantize write path, FA Q-side, or gather
traffic — and the kernel-level decomposition will assign it.

## What was done

1. Dispatch audit (no GPU): with today's defaults (`GFX906_FA_LEGACY=1`,
   `GFX906_FA_DIRECT_PAGED=auto` min_batch=2,
   `GFX906_FA_DIRECT_PAGED_Q8=0` since M6 Part B), the direct-paged
   branch is **never taken** — LEGACY=0 B=1 runs the fused-Q8 gather
   (`gather_paged_kv_q8`, reads the pre-quantized aliased side buffer)
   + the same FA gather kernel as LEGACY=1, which instead runs
   `gather_paged_kv_quant_persistent` (fp16 K read + in-kernel
   quantize). The M5-era gap was measured when LEGACY=0 B=1 ran
   direct-paged FA (`forward_paged_direct`, internal kv_split=8).
2. Kernel-level probe (GPU0, in-process, eager — launch-regime
   evidence): `benchmarks/kernels/gfx906/legacy0_b1_step_probe.py`,
   B=1, Sk ∈ {2048, 16384, 32768}, both prime geometries; log
   `/local/tmp/b1_step_probe_run1.log`.
3. Serving bake (TP=2, Qwen3.8-27B, maxlen 32768,
   `_serve_tp2_gfx906.sh` + new `EXTRA_SERVE_ENV` passthrough +
   `_bench_serve_grid_gfx906.py [[2048,256]] 2`), boot O (canary
   39.3 t/s; boot N arm A 39.76/40.12 kept as cross-boot anchor):
   all three arms ran clean on boot O after two intermittent GPU1
   qcm-fence load wedges (15:51 arm A 1st try, 16:09 arm C 1st try —
   recorded in degradation.md/_details; bare two-card RCCL probe +
   clean runs between break the chain per the boot-L 12:52
   precedent).
4. Append-path cost probe (`legacy0_append_cost_probe.py`, eager,
   B=1 D=256 Hkv=4 — full-model Hkv; per-shard at TP=2 is Hkv=2, so
   the probe writes 2× the per-token K bytes; launch-dominated at 1
   token, immaterial): the LEGACY=0 per-layer append adds
   `reshape_and_cache_q8` (6.6 us alone; 16.4 % on the 36.1 us
   triton write), ×16 full-attn layers = **+60–105 us/step eager**
   (recorded run +94.6; review re-run +59.6 with the q8-alone ×16 =
   105.6 us no-overlap bound — eager launch-overlap variance).
   Log: /local/tmp/b1ab_*_bootO.log, /local/tmp/b1_step_probe_run1.log.

## Evidence FOR (the framing is superseded — launch-regime)

Per-step decomposition (us; A = LEGACY=1 gather+quant+FA, B = LEGACY=0
Q8-gather+FA, C = LEGACY=0-era direct-paged FA):

| Sk | D=256 A / B / C | B−A | C−A | D=128 A / B / C | B−A | C−A |
|---|---|---|---|---|---|---|
| 2048 | 92.1 / 58.5 / 120.9 | −36.4 % | +31.4 % | 64.9 / 50.7 / 72.0 | −21.9 % | +10.9 % |
| 16384 | 496.3 / 273.0 / 664.9 | −45.0 % | +34.0 % | 414.6 / 301.7 / 446.2 | −27.2 % | +7.6 % |
| 32768 | 970.7 / 533.5 / 1307.2 | −45.0 % | +34.7 % | 815.6 / 588.9 / 878.3 | −27.8 % | +7.7 % |

The FA term is identical in A and B (same kernel, same compact
buffers) — the entire A−B delta is the gather kernel: 50.9/307.5/605.5
us (A) vs 17.4/84.1/168.3 us (B) at D=256. The M5-era gap maps onto
C: direct-paged B=1 is +7.6…+34.7 % slower than A, growing with Sk
(block_table indirection + strided aliased-Q8 reads in the FA kernel
vs compact reads — the wrapper header's old "B=1: gather faster by
~3-6 %" A/B was measured at short Sk, where the penalty is smallest).

## Evidence AGAINST (the gate fired — launch-regime numbers did not
## transfer)

Same-boot (boot O) serving, Qwen3.8-27B TP=2, B=1 pp2048/tg256,
2 samples/arm (B=4 aggregate in parens):

| arm | path | t/s | vs A |
|---|---|---|---|
| A | LEGACY=1 (production default) | 40.11 / 40.12 (41.15/41.04) | — |
| B | LEGACY=0 (Q8-gather dispatch) | 37.61 / 37.56 (38.24/38.14) | **−6.3 %** |
| C | LEGACY=0 + direct-paged (M5 era) | 37.55 / 37.54 (38.20/38.12) | **−6.4 %** |

Sample spread ≤0.1 % per arm; cross-boot A consistency: boot N
39.76/40.12 vs boot O 40.11/40.12. The kernel probe's B win (−36 % on
the gather+FA subcomponent at 2k) did NOT transfer — the serving step
at 2k context is ~25 ms, of which the subcomponent is ~92 us (0.4 %).

## Why LEGACY=0 loses (mechanism, bounded)

- B and C differ hugely in the FA/gather subcomponent (−36 %/+31 % vs
  A in the kernel probe) yet land within 0.2 % of each other in
  serving → the serving gap is NOT in the FA kernel or the gather.
  It is in what both LEGACY=0 arms share per step.
- Measured share: the append-time Q8 side-buffer write
  (`reshape_and_cache_q8` + slot cast, 16 full-attn layers/step) is
  `reshape_and_cache_q8` + slot cast, 16 full-attn layers/step) is
  +60–105 us/step eager (q8-alone ×16 = 105.6 us bound) — real but an
  order of magnitude below the ~1.55–1.70 ms/step serving delta.
- The unexplained remainder is a serving-harness interaction specific
  to LEGACY=0's per-step path: the ~16–32 extra captured graph nodes
  per decode step (Q8 writes + slot casts) add graph-replay node
  overhead that is invisible in eager timing, ± TP=2 sync-placement
  effects. Not further decomposed (a step trace on this stack has
  the documented wall-alignment caveat; eager TP=2 is not a valid
  isolator — it collapses ~3× from launch overhead).
- The steady-state READ path win (Q8 gather 22–45 % below fp16
  gather+quantize per step, growing with Sk) is real but swamped at
  2k context; even at 32k it is ~440 us/step vs the ~1.55 ms
  fixed LEGACY=0 cost — no crossover at reachable context lengths on
  this step-time shape.

## Interactions / refrigerated residue

- M5's verdict ("LEGACY=0 LOSES, default stays 1") is CONFIRMED by a
  proper same-boot adjudication; the M5-era 2.5–3.7 % gap does not
  reproduce at 6.3–6.4 % on the current build (different era —
  direction unchanged). The wrapper-header "B=1: direct loses 3–6 %"
  note is superseded by the kernel probe numbers (+8–35 % at
  2k–32k, growing with Sk) for future reference.
- Refrigerated: fusing the Q8 write INTO
  `triton_reshape_and_cache_flash` (one kernel writes fp16 K + Q8
  bytes) would cut the append delta to ~0 and the graph nodes in
  half — but the node-overhead remainder would still stand, so this
  alone would not close a 6 % gap; revisit only with a graph-node
  overhead measurement (now tracked: `roadmap-more-models.md` →
  "Decode-graph node-overhead point", G1).

VERDICT: DEAD-END (flip question closed; records: DEAD-ENDS.md,
CHANGELOG; branch stays unmerged for review)

## G1 addendum — per-node replay cost measured (2026-08-31, boot of this session)

The "node-overhead point" (G1) above is now MEASURED, closing the last
unmeasured hypothesis. Probe: `benchmarks/kernels/gfx906/
g1_node_replay_probe.py` — captures a decode-shaped CUDA graph (40 layers ×
3 real work kernels + 3 sink bookkeeping = 240 base nodes; TP=2 adds one
allreduce per layer) and re-captures with N dummy no-op Triton launches
appended per layer (N ∈ {0,16,32,64}); replays WARM=50 + ITERS=300, host wall
time per replay incl. device sync (the vLLM decode-step shape).

Results (same boot, GPUs idle at start):

| config | N=0 base | slope us/node | linearity |
|--------|----------|---------------|-----------|
| TP=1   | 3206 µs/replay (240 nodes) | **1.20–1.23** | ~1.2 across all ΔN |
| TP=2   | 3976 µs/replay (280 nodes) | **0.77–1.10** | ~1.1 across all ΔN |

Per-node replay cost ≈ **1 µs/node**, an order of magnitude below the ~10 µs
that would be needed for node count to own the 1.55 ms/step. At the real
LEGACY=0 delta (16–32 extra nodes/decode step) this is only
**~0.02–0.04 ms/step** — ~2 % of the unexplained cost, not the owner.

Consequence for the roadmap:
- G1's falsifiable branch taken: **node count is NOT the owner** of the
  ~1.55 ms/step. The remainder lives in TP=2 sync placement / other
  LEGACY=0-common per-step work (eager TP=2 can't isolate it — documented).
- The refrigerated Q8-fusion lever (fuse the Q8 write into
  `triton_reshape_and_cache_flash`) does NOT reopen on this evidence: halving
  ~16–32 nodes saves at most ~0.03 ms/step, which cannot close a 6 % gap. It
  stays refrigerated.
- Budget for future adds-nodes-per-step proposals (MoE routing fusion,
  spec-decode extensions, KV-side writes): **~1 µs/node** of replay overhead
  on this stack — cheap, but now bounded and citable rather than a guess.

Note: `torch.cuda.device_count()` misreports 0 on this ROCm 7.14 stack even
with GPUs present (rocm-smi + real HIP init both fine); the TP=2 launcher
(`/local/tmp/g1_tp2_launch.sh`) pins each rank to its own GPU via
HIP_VISIBLE_DEVICES before torch import and inits NCCL without device_id, so
the probe runs despite that quirk.

