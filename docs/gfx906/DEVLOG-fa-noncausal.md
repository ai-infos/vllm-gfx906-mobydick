# FA non-causal attention (for DFlash2 drafters)

## 2026-09-16 — design: what the drafter needs, what our FA lacks, and the cheapest way to know it matters

**VERDICT:** OPEN (design complete; no code yet) · **GATE:** DFlash2 per-position acceptance with
the recommended pair — currently 0.0 because the drafter cannot use our FA at all.

## Why this exists

`DEVLOG-dflash2.md` (2026-09-16) root-caused DFlash2's acceptance collapse to the drafter's
attention path rather than to the model, the checkpoint or the quantisation pairing: with the
*matched* pair (AWQ-INT4 target + `syvai/Qwen3.8-27B-DFlash2-W4A16`, k=7) acceptance is still 0.0,
and the reasons line added for DFL2-8 says why:

```
Found incompatible backend(s) [CUSTOM, TURBOQUANT] with AttentionType.DECODER.
  Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN', 'TRITON_ATTN'].
  Reasons: {CUSTOM: [non-causal attention not supported], ...}
```

The predicate is `vllm/v1/attention/backend.py`: `if use_non_causal and not cls.supports_non_causal()`
— and our `Gfx906FABackend` simply does not define `supports_non_causal`, so it is rejected for any
layer built with non-causal attention. The DFlash2 drafter is such a layer: all 5 of its layers are
`sliding_attention` with `sliding_window: 2048` and `is_causal: false`, and `DFlashQwen3Attention`
(`qwen3_dflash.py`) is built with `causal=False` and drives a vLLM `Attention` layer
(`self.attn = Attention(...)`, then `self.attn(q, k, v)` — the context K/V are pre-inserted into the
paged cache by `precompute_and_store_context_kv`). The fallback it gets instead (ROCM_ATTN with the
Triton paged kernel) cannot be CUDA-graph captured — `RuntimeError: Cannot copy between CPU and CUDA
tensors during CUDA graph capture` through `rocm_attn.py` -> `chunked_prefill_*` — and drafts nothing.

## The semantics to implement

Match the reference helper `_maybe_symmetrize_window` (`vllm/v1/attention/backends/flash_attn.py`):
a causal window `(w, 0)` becomes **symmetric `(w, w)` when attention is non-causal**, so
bidirectional queries attend in both directions; already-symmetric windows and full attention are
left alone. For the drafter that means: **no causal clip, and a symmetric ±(window) clip** — i.e.
keys are visible when `|k_pos - q_pos| <= window`, with `window = sliding_window` (2048).

## What our kernel does today

`csrc/gfx906_fa/kernel/fattn-q8.cuh` (and the three sibling kernels) mask with the absolute
position, gated on `q_abs_offset`:

```cpp
if (q_abs_offset) {
    const int q_abs_row = q_abs_base + j;
    if (k_pos_abs > q_abs_row || (window > 0 && q_abs_row - k_pos_abs >= window)) {
        KQ_acc[jc0] = -INFINITY;    // future keys, and keys older than the window
    }
}
```

That is a causal clip plus a **left-only** window: there is no right-side clip and no way to allow
future keys. Non-causal needs exactly two changes at each site: drop `k_pos_abs > q_abs_row`, and add
the symmetric right clip `k_pos_abs - q_abs_row > window`.

## Edit inventory (measured, not estimated)

- **Kernel mask sites, 8** — `k_pos_abs > q_abs_row` appears twice per file (the two inner-loop
  variants) in each of `fattn-q8.cuh`, `fattn-q8-paged.cuh`, `fattn-q8_hip.cuh`,
  `fattn-q8-paged_hip.cuh`.
  - Caveat: the two `*_hip.cuh` sites have the causal test **without** the window clause, so those
    variants do not implement sliding windows at all. Before wiring non-causal in, confirm which
    variant actually serves this shape (the launcher picks between the generic and hip paths), or the
    flag would be honoured on a path that is never taken.
- **Kernel signatures**: 4 files, plus the launcher (`gfx906_fa_launcher.cu`, ~8 sites: two launch
  wrappers and their `flash_attn_tile_q8*` calls) and the op (`gfx906_fa.cpp`, ~6 sites incl. the
  `window > 0 requires q_abs_offset` style `TORCH_CHECK`s).
- **Python**: one call site (`gfx906_fa_backend.py`, `window=self.sliding_window`) plus the impl
  constructor (read `vllm_config.attention_config.use_non_causal`) and the backend class
  (`supports_non_causal()` -> True, only once the path above is real).
- The plumbing is a flat argument list, not a struct, which is why this is ~25-30 edits rather than
  a handful.

## Cheaper decisive experiment first

Before spending that change: **does correct non-causal attention actually restore acceptance?** If it
does not, the kernel work buys nothing and the DFlash2 family stays parked.

Plan: an env-gated (`VLLM_DFLASH2_EAGER_ATTN=1` or similar) path in `DFlashQwen3Attention` that
computes the drafter's block attention in torch with a **symmetric-window mask** instead of calling
`self.attn`. The context K/V are available densely at the moment they are projected:
`precompute_and_store_context_kv` builds `all_k, all_v` (`_project_context_kv`, `qwen3_dflash.py`) —
so the experiment can stash those and run
`scaled_dot_product_attention(q, cat([ctx_k, blk_k]), cat([ctx_v, blk_v]), attn_mask=window_mask)`.
Slow (eager, per step) — but it answers the scientific question, and its acceptance number is the
justification (or the refutation) for the kernel work.

## Gates

1. Experiment: per-position acceptance with the eager symmetric-window path, 2 reps, ~12k prompt,
   compared against 0.0 today and syv-ai's 3.1-3.6 tokens/step reference.
2. If positive: implement the kernel flag + the `supports_non_causal()` flip, then re-run 1 with the
   real backend, plus the FA suite (must stay green; the flag defaults off so behaviour is unchanged
   until the flip).
3. Only then do DFL2-3 (lookup drafting) and DFL2-1 (chains) make sense.

## 2026-09-17 — Stage 1 implemented: CUSTOM FA serves the non-causal drafter, graphs work, acceptance holds

**VERDICT:** `SHIPPED` (default on; `GFX906_FA_NO_NONCAUSAL=1` is the rollback) · **GATE:**
Muse-Glimmer-30B-AWQ-INT4 + `meta-models/Muse-Glimmer-30B-assistant` (method `dflash`, k=7),
TP=2, util 0.90, maxlen 8192, `cudagraph_capture_sizes [8,16]`, chat-templated prompt, 3 reps
× 128 tokens — the drafter's **mean acceptance length** (server metric) plus decode t/s, with
the same arm run one day earlier through ROCM_ATTN + `--enforce-eager` as the control.

### HYPOTHESIS

If the non-causal batch is served by CUSTOM (no causal clip), the DFlash assistant's layers
stop landing on ROCM_ATTN, CUDA-graph capture succeeds, and decode beats the eager arm while
acceptance does not degrade (the mask changed from the trained symmetric ±window to its
superset).

### What was done (Python only — no kernel change, no rebuild)

- `Gfx906FAMetadata.causal` (from `CommonAttentionMetadata.causal`, per batch); a *tensor*
  causal (per-token masks) is not expressible in this kernel and stays causal with a
  one-time warning (fail-closed to today's behaviour).
- `Gfx906FAImpl.forward` passes `non_causal=not attn_metadata.causal` to `forward_paged`.
- `forward_paged(..., non_causal=)`: when set, zeroes `window` and suppresses
  `q_abs_offset`/`need_causal`. Both causal mechanisms in this stack are driven by exactly
  those two variables (the kernel's inline clip by `q_abs_offset`; the window mask and the
  Phase-C/M1 `kv_start` clips by `window` > 0), so one point turns them all off in every
  dispatch branch (direct-paged, fused gather, persistent gather, legacy).
- `Gfx906FABackend.supports_non_causal() -> True` (the selector's gate at
  `v1/attention/backend.py:349`), with `GFX906_FA_NO_NONCAUSAL=1` restoring the rejection.
- Tests: `test_non_causal_batch_is_bidirectional_vs_torch_ref` — a query *block* on top of a
  64-token cache, both arms (causal=True/False) against their own fp32 references, plus an
  amplified-future-keys control that proves the flag reaches the kernel (the first version of
  that control passed spuriously on diffuse softmax).

### Evidence — FOR

| drafter attention | graphs | decode t/s (3 reps, mean) | mean acceptance | runs |
|---|---|---|---|---|
| ROCM_ATTN + Triton fallback (`--enforce-eager`; `GFX906_FA_NO_NONCAUSAL=1`) | **no** | 30.46/30.43/31.17 (30.7), 30.46/29.71/31.16 (30.4), 30.49/29.73/30.69 (30.3) | 2.95 / 3.00 / 3.02 | **3 clean** |
| **CUSTOM FA (this change)** | **yes** | 43.13/43.10/44.14 (43.5), 39.27/39.22/39.19 (39.2) | 3.12 / 3.18 and 2.95 / 2.82 | **2 clean** |
| (non-spec reference: V2 TP=1, no drafter) | yes | 27.1 | — | 1 |

**Order-controlled result: +29 % to +41 % decode, acceptance at parity.** The control arms are
extremely stable (two interleaved runs 30.44 and 30.30 mean, order control −0.5 %, acceptance
2.95-3.02); the graph arm read 43.5 t/s once and 39.2 t/s once, so the honest claim is the
**lower bound, ≥ +29 %**, with acceptance indistinguishable from the control (2.82-3.18 against
2.95-3.02 — and acceptance is documented as chaotic on this stack, ±7 pp run to run).

- The drafter's own capture line is the direct evidence the blocker is gone:
  `Capturing dflash CUDA graphs (FULL): 100%|2/2 … Graph capturing finished in 18 secs`, and
  `Cannot copy between CPU and CUDA tensors during CUDA graph capture` appears **zero** times
  (it was the eager arm's failure mode).
- **Acceptance is not degraded** (3.12/3.18 vs 2.95): the superset mask costs this drafter
  nothing at 8k context, so Stage 2 (the symmetric ±window in the kernel) is not needed for
  the live use case.
- Net: **≥ +29 %** decode over the eager control (best clean run +41 %), **≥ +45 %** over
  non-spec. The two clean graph runs bracket the effect; no single-run headline is quoted.

### Evidence — AGAINST / caveats

- One launch failed first with `hipErrorLaunchFailure` during weight load (wedge #100, the
  chronic load lottery — the retry was clean), so the arm's numbers are from the second load.
- The superset mask has *not* been validated on a drafter with a short window at long context;
  Muse's assistant is sliding-2048 and this arm ran at 8k. A drafter that genuinely needs the
  ±window would show it as an acceptance drop; that is the guard to re-run if another
  non-causal drafter appears.
- A pre-existing capture-hygiene warning fires twice now (`2 retired capture-baked gather
  generations (expected <= 1)`): with a target *and* a drafter capturing, two generations is
  the expected count, so the warning's threshold is stale for the spec-decode case (cosmetic;
  left alone).
- k was not swept (k=7 only) and B=1 only; the histogram at the eager arm was still
  productive at positions 3-4, so a k sweep is a cheap follow-up.

### Interactions / superseded-by

- This is the enabler MUSE-2 asked for; the ROADMAP's FA-NONCAUSAL item moves from "design" to
  "Stage 1 shipped, Stage 2 refrigerated".
- DFlash2 (the original motivation) stays parked for its own reasons — its degeneration was
  never the mask.

## 2026-09-17 (later) — k sweep on the graph path: throughput is flat, the verify step is the cost

**VERDICT:** `NEUTRAL` on the k axis (k=7 stays) · **GATE:** same arm as above with
`num_speculative_tokens=4` (`cudagraph_capture_sizes [5,10]`, i.e. multiples of k+1), 3 reps.

| k | capture ladder | decode t/s (3 reps) | mean acceptance | tokens/step |
|---|---|---|---|---|
| 4 | [5,10] | 44.29 / 43.30 / 41.47 (43.0) | 2.65 | 3.65 |
| 7 | [8,16] | 43.13 / 43.10 / 44.14 (43.5) | 3.12 / 3.18 | 4.12 |

Throughput-equivalent (within this box's run-to-run spread), so **k is not the lever**: with
tokens/step 4.12 vs 3.65 at the same t/s, the extra k=7 drafts pay for themselves exactly and
the step cost scales with the number of verified tokens, not with the drafter's own work (whose
5 layers are clearly not dominant). k=7 stays the working value on the higher acceptance; a
real optimisation would have to attack the *verify* step (the target's M=1..k+1 forward) or the
drafter's attention, not the draft depth.

## 2026-09-17 — self-review before merge: two findings, one of them a bug I had just written

**VERDICT:** `SHIPPED` (unchanged verdict) · **GATE:** re-run of the MUSE-2 arm with the
corrected code, plus an interleaved control/graph/control A-B-A (below).

### Finding 1 — the "scope guard" was based on a false premise, and it broke the validated path

I added a guard so that a `causal=False` batch on a **non-draft** model would fail loudly rather
than be served a mask nobody gated (motivated by `model_executor/models/config.py`:
DiffusionGemma sets `use_non_causal=True` for *mixed* causal/bidirectional attention). The guard
keyed off `vllm_config.model_config.runner_type == "draft"`.

It fired on the real drafter — the MUSE-2 arm died with the guard's message instead of serving.
Reason: the draft attention build does **not** carry the draft ModelConfig. The speculator's
`attn_vllm_config` is `copy.copy(super().attn_vllm_config)` with only `attention_config`
`replace`d, so `model_config` there is the **target's** (`runner_type` is not `"draft"`).

Second look at the premise: the guard is unnecessary. A **bool** `causal=False` has the same
meaning in the reference backends — `triton_attn.py` and `rocm_attn.py` pass
`common_attn_metadata.causal` straight into their mask logic, so the contract is "no causal
clip", which is exactly what this kernel now serves. (The *tensor* form is the weaker claim: it
is per-token and is refused — warn once, serve causal, i.e. the pre-FA-NONCAUSAL behaviour.)
So the guard was replaced by a comment recording that equivalence, and the only remaining
asymmetry is the **window** (see the caveat below).

### Finding 2 — the headline t/s was a single optimistic run

The first graph arm read 43.1/43.1/44.1 t/s; the same code minutes ago read 39.2/39.2/39.2 with
acceptance 2.88 vs 3.15. Both are the *same* effective configuration (the guard never served a
token), so that spread is process/run variation — the repo's standing warning. The gate is
therefore re-read as an interleaved **control → graph → control** A-B-A in one boot, and the
headline is quoted from that, never from the best single run.

### Caveats recorded (not fixed)

- **Windowed drafters pay full-KV reads under stage 1.** Suppressing `window` turns off the
  mask *and* the M1/M2 clips, so a sliding-2048 drafter scans the whole context per block
  instead of ~2048 keys. At the 8k context measured here that is small; at 32k+ the drafter's
  attention cost grows linearly and stage 2 (the symmetric ±window, which would restore the
  clip) becomes a *performance* item as well as a fidelity one.
- The design doc's "8 mask sites" counts both `*_hip.cuh` twins, which are **never included**
  (`fattn-q8.cuh` + `fattn-q8-paged.cuh` are the live pair) — the real inventory is 4 sites, and
  stage 1 needed none of them.
- `k` is not the lever (k=4 vs k=7 throughput-equal), and `B=1` only.

### Finding 3 — my own measurement scripts were the confound (fixed in the harness, and it is why the headline is a range)

The Muse arm scripts killed the APIServer and waited, but **never reaped orphaned workers** — the
hazard already fixed for the other arm scripts in this repo. With TP=2 that is ~20 GiB per rank
left resident, so the next arm could either fail to init or run on a partially-occupied GPU. Both
were observed: the graph arm's first numbers (43.5) and its 11:16 re-read (39.2) bracket exactly
the kind of drift an un-reaped predecessor would produce, and the string of load failures
(#101-#103) all followed un-reaped teardowns. All four Muse arm scripts now share the
`reap_gpu_procs` helper (KFD VRAM > 1 GiB -> TERM then KILL), so the next session's numbers are
not confounded by it. The range above is quoted *because* of this: a single 43.5 t/s run cannot be
distinguished from an orphan-free lucky run.
