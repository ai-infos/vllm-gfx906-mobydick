# gfx906 custom FA — gather-buffer lifecycle fix for the 256k prefill OOM

2026-08-23. Target: `vllm-gfx906-mobydick` @ `gfx906/main` (4a9e24b5ca).
Companion docs: `docs/gfx906/oom-256k-prefill.md` (OOM analysis),
`docs/gfx906/DEVLOG-masked-fa.md` (N4 persistent gather),
`degradation_details.md` (2026-08-23 OOM cluster).

**STATUS (2026-08-24): IMPLEMENTED + VALIDATED** — branch
`gfx906/fa-gather-lifecycle` @ `090673ad21`; the run-4 situation OOMs
byte-exact under `GFX906_FA_GATHER_EXACT=1` (×2 boots, OOMHUNT retired
7.79 GB at the OOM point) and completes under the fix (250k prefill,
needle retrieved, `retired_B=0`), decode A/B flat. Record:
`DEVLOG-fa-attention.md` (Gather-buffer lifecycle fix, 2026-08-24).
(Originally blocked on the weight-load `hipErrorLaunchFailure`,
`DEVLOG-boot-failure.md` — cleared by the 2026-08-24 05:20 reboot.)

**Post-landing review follow-up (2026-08-24):** a code review of the
branch found three issues, fixed on the same branch: (1) the §2.2b
capture-order guard was dead code (its condition required
`not capturing` while checking a flag just set to `capturing`) — moved
to the retire-insertion site, one-shot at >1 retired capture-baked
generation; (2) §2.2a's grow-only `max()` per axis also applied to
never-captured (freeable) replacements, pinning the B×Sk high-water
product (~13 GB/rank at `[32, 262144]`) — freeable replacements now
allocate at exact need (realloc frequency unchanged; FULL-mode
capture-time sizing identical, so the validated arm-B path is
unaffected); (3) the persistent branch's k/v capacity reuse now
requires equal K/V widths (an unequal hand-set pair was half-reused,
silently dropping V to per-call C++ allocations). Unit gate: 28/28.
See the Review follow-up section of the dev-log entry.

**Revision note:** this supersedes `/local/tmp/gfx906-fa-fix.md`
(now stale/gone from `/tmp`). It was rewritten after four independent
adversarial reviews (`gfx906-fa-fix-code-review-{claude,ds4,glm,gwen}.md`,
kept in this directory as the review record) found two must-fix
correctness bugs and one unclamped-growth bug in the original design.
The kernel-level safety case (width-shaped gather + FA execution is
live-bounded, matches decode's shipped status quo) was independently
verified by three of the four reviews and is treated as settled below
— see `gfx906-fa-fix-code-review-claude.md` §"Adjudicating the
ds4-vs-gwen contradiction" for the one genuine dispute among the
reviews and how it was resolved (in favor of the original design: no
`gridDim.z` limit applies to the persistent kernel, and decode
genuinely does replay at capture-time width).

## 0. Root-cause framing (read before treating this as OOM-closing)

The project's own same-day diagnosis (`oom-256k-prefill.md`,
`degradation_details.md` §"2026-08-23... verified diagnosis")
identifies the *failing* 250k-prefill allocation as the AWQ
`gptq_gemm` **`temp_dq` dequant scratch** — weight-shaped, **not
token-count-scaled** (178 MB per MLP call, 2.37 GiB for the quantized
lm_head, identical across every chunk-size/MTP/util arm tested) — and
states plainly "not a serving-time leak." That doc lists **unbounded
gather-buffer growth as one of several unprofiled headroom
consumers**, not the sole or proven cause.

This plan fixes a real, independently-confirmed bug (an unbounded
`_gather_retired` keep-alive dict, quantified below) that **may**
compound the AWQ-scratch OOM by shrinking the ~1.9–2.5 GiB headroom
the scratch needs. It is not established to be *sufficient by itself*
to close the 256k OOM — the missing artifacts and the un-reconciled
"131k is the validated ceiling" claim (§3 below) mean this must ship
as a hypothesis with a control arm, not a proven fix. **Land it as a
worthwhile bugfix on its own merits; treat "closes the 256k OOM" as
an open question the validation plan (§4) is designed to answer.**

The most informative missing experiment is a `--attention-backend
TRITON_ATTN` A/B on the same 250k request (bypasses this backend
entirely). If that arm OOMs identically, the FA-growth theory is dead
regardless of what this fix measures, and it should be run *before*
investing in the 27B repro.

## 1. The bug (quantified, confirmed against source)

The 256k first-prefill OOM cluster on Qwen3.8-27B TP=2 coincides with
a real, verified memory-lifecycle bug:

1. FULL cudagraph capture calls `_ensure_gather_buffers` with
   `max_seqlen_k = max_model_len = 262144`
   (`build_for_cudagraph_capture` forces `max_seq_len = self.max_model_len`
   specifically `for_cudagraph_capture`, `gpu_model_runner.py:2387-2390`)
   → one full-width generation (B=cap, Hkv=2/GPU, D=256: K 272 B/row +
   V 512 B/row = 784 B per (seq,head,row)).
2. At serving, chunked prefill's `CommonAttentionMetadata.max_seq_len`
   grows every chunk; `_ensure_gather_buffers`
   (`vllm/gfx906_fa/gfx906_fa_backend.py:~540`) demands an **exact**
   `Sk_pad` match (`b.shape[2] != Sk_pad` → realloc) → a new generation
   every growth step.
3. Since the Aug-19 UAF fix (`5d960a503c`), every replaced generation
   is retired into `_gather_retired` — an **unbounded** keep-alive
   dict. Correct for UAF safety (a captured graph's baked VA must
   never be freed), but the retire condition is gated on a single
   rolling `_gather_captured` class-level latch that is **never reset
   once set** — so once any FULL capture has happened (which it always
   does in FULL/FULL_AND_PIECEWISE serving), *every* subsequent
   eager-serving replacement gets retired too, not just the
   graph-baked one.
4. The scale of this: at B=1, chunk 1024, growing `Sk_pad` in 1024-token
   steps from 1024 to 262144 is 256 generations; each generation is
   `B × Hkv × Sk_pad × 784 B`. Order-of-magnitude, this is tens of GiB
   by 250k tokens — the exact multiplier depends on B, chunk schedule,
   and whether the capture-time generation is counted, and **no
   reproducible script for the originally-cited "89.4 GiB" figure
   exists on disk** (`/local/tmp/oom_hunt/verify_theory.py` is absent).
   Re-derive and commit a small script alongside this plan before
   citing a specific number in a dev log — see §4 gate 1.

The pinned test `test_gather_buffers_capture_sweep_keepalive`
(`tests/kernels/attention/test_gfx906_fa.py` ~L558) asserts
retire-on-every-Sk-change — correct for the UAF it was written against,
but never evaluated for the memory consequence; it must be rewritten
(§5) alongside the fix.

## 2. The fix: grow-only (≥) reuse + width-decoupled Sk, scoped correctly

**Principle:** the buffer's `Sk` dimension becomes a capacity, not an
exact shape, **only on the code paths that can safely execute at a
width wider than the live sequence length**. Reuse any buffer whose
width ≥ needed `Sk_pad`; never slice K/V on the Sk axis; pass the
buffer's own width as `Sk` to the persistent gather kernel specifically
— not to every gather kernel indiscriminately (§2.3 explains why that
distinction is load-bearing, unlike in the original design).

### 2.1 Why passing `Sk = buffer width` (≥ live length) is safe — persistent path only

- The persistent gather kernel (`csrc/gfx906_fa/gfx906_fa_gather.hip`,
  `gather_paged_kv_quant_persistent_kernel` + its launcher
  `launch_gather_paged_kv_quant_persistent`, ~L660-698) uses a **fixed
  1-D grid** (`dim3 grid_d(grid, 1, 1)`, `grid` defaulting to 1024) and
  grid-strides over a flat row space computed from the **device-side**
  `seq_lens` tensor: `rph[s] = min(seq_len[s], Sk) + margin`, margin
  clamped to `Sk - min(seq_len[s], Sk)` (~L560-572). `Sk` bounds the
  row-count math only — it is never a launch dimension, so there is
  **no `gridDim.z` limit on this kernel** (that 65535 cap is real but
  applies only to the *non-persistent* V1/V2 kernels — see §2.3).
  Work is live-bounded, width-independent (the N4 property;
  `DEVLOG-masked-fa.md`).
- The FA kernel (`fattn-q8_hip.cuh:963`) cuts at
  `k_VKQ_max = KV_max[...] = seq_len` — never at `Sk`; `kv_max_tensor
  = seq_lens` is always passed by `forward_paged`. Tail-tile OOB rows
  are masked in LDS without a global read (`oob_check`); rows past
  `seq_len + margin` are garbage in V/uninitialized in K but never
  read — same situation as today's mixed-length batches.
- **Decode genuinely already runs this way — verified precisely, not
  by analogy.** `max_seqlen_k=attn_metadata.max_seq_len`
  (`gfx906_fa_backend.py:688` etc.) is a plain Python `int` that
  becomes the literal `Sk` kernel-launch argument. CUDA graph capture
  bakes Python-side scalar launch arguments into the graph as fixed
  constants; a replay does not re-run the Python code that computed
  them. Since capture always runs with `max_seq_len = max_model_len`
  (§1.1), the **baked** `Sk` argument for every persistent-gather and
  FA launch inside a FULL graph is `max_model_len`-derived and is
  replayed unchanged on every decode step — only `seq_lens`/`kv_max`
  (genuine device tensors, read *inside* the kernel) carry the live
  length at replay time. Width-shaped gather+FA execution is the
  shipped, soaked status quo for decode; this fix makes eager chunked
  prefill use exactly the buffers decode already uses.
- Addressing: `v_dst_base`/`k_dst` use int64 and stride `Sk` = actual
  tensor width (the C++ `use_or_alloc` exact-match then passes
  trivially because Python passes `Sk = kbuf.shape[2]` on this one
  call site). No C++ changes required for the persistent path.

### 2.2 Code changes

**(a) `vllm/gfx906_fa/gfx906_fa_backend.py` — `_ensure_gather_buffers` (~L497-594)**

```python
# exact match -> grow-only capacity check
if (b.shape[1] != num_kv_heads
        or b.shape[3] != bytes_per_row
        or b.device != device
        or b.shape[0] < num_seqs
        or b.shape[2] < Sk_pad):          # was: b.shape[2] != Sk_pad
    need_realloc = True
```

- New allocation: allocate at
  `(max(num_seqs, cur.shape[0] if cur is not None else 0), num_kv_heads,
  max(Sk_pad, cur.shape[2] if cur is not None else 0), bytes_per_row)`
  — grow **both axes independently to the max of what's needed and what
  already existed**, mirroring the in-tree `_q_pad_buf` pattern exactly
  (`gfx906_fa_backend.py:449-469`, which already does `max(num_seqs,
  cur.shape[0])` / `max(Sq_pad, cur.shape[2])` and is proven capture-safe
  in production).
- **No doubling hysteresis.** The original design's `max(Sk_pad, 2 *
  old_width)` was found to overshoot `max_model_len` permanently and
  unboundedly whenever `old_width` isn't power-of-two-aligned (e.g.
  `old_width=200,064` → next grow allocates 400,128, a 52.6% permanent
  overshoot that is never released, paid out of the exact headroom
  this fix protects). Grow-to-exact-need has a real but bounded cost:
  more reallocs in no-capture (PIECEWISE-only) serving modes — but
  those reallocs are now cheap frees under policy (b) below, and this
  is the same trade `_q_pad_buf` already makes safely in production.
  Under FULL capture the first capture call already allocates at full
  width, so serving never reallocs at all — the hysteresis question
  only ever mattered for no-capture modes, where "no hysteresis, more
  small reallocs" is strictly safer than "hysteresis, unbounded
  overshoot."
- The `[:num_seqs]` leading-dim slice reuse stays (same base VA,
  contiguous).
- Update the docstring at `_gather_retired`'s declaration
  (~L305-320): it currently states "Replacements after the first
  capture only happen on Sk/Hkv/D growth ... so this set stays small"
  — the Sk-growth clause **is the bug being fixed here** and must not
  survive verbatim. Rewrite to: replacement happens only on
  Hkv/D/B-below-capacity change post-fix (Sk is a capacity, not an
  exact shape), so the retained set holds at most the captured
  generation(s) — normally exactly one, the VA the FULL graphs replay.

**(b) Retire policy — free never-captured generations (fixes the unbounded dict)**

Track captured-ness **genuinely per generation** — the earlier draft
of this fix used a single sticky class-level bool that inherited
`True` from any prior captured generation and never reset, which all
three independent reviews caught as functionally unchanged from
today's bug. The corrected version:

```python
_gather_buf_captured: ClassVar[bool] = False

# on replacement, BEFORE allocating the new generation:
if cls._k_gather_buf is not None and cls._gather_buf_captured:
    cls._gather_retired[cls._k_gather_buf.data_ptr()] = (
        cls._k_gather_buf, cls._v_gather_buf)
# else: simply drop the reference — the caching allocator reuses the block.

capturing = torch.cuda.is_current_stream_capturing()
cls._k_gather_buf = torch.empty(new_shape_k, ...)
cls._v_gather_buf = torch.empty(new_shape_v, ...)
cls._gather_buf_captured = capturing   # NOT `or`ed forward — the new
                                        # generation starts from its own
                                        # state, not the old one's.

# in the no-realloc (reuse) path only, latch forward within ONE generation:
elif not cls._gather_buf_captured:
    cls._gather_buf_captured = torch.cuda.is_current_stream_capturing()
```

The critical difference from the original draft: `_gather_buf_captured`
is **reset to `capturing` (not OR'd)** at the moment a new generation is
allocated. The `or`-forward pattern is correct *only* within a single
generation's lifetime (accumulating "was this generation ever current
during a capture" across repeated no-grow calls) — carrying it across a
replacement conflates two different generations' capture history.

Rationale: a freed generation is a UAF only if its VA is baked into a
replaying graph. FULL graphs bake the VA (attention captured inside the
graph); PIECEWISE graphs never contain attention by default
(`vllm::unified_attention_with_output` is in the default
`splitting_ops` set, `vllm/config/compilation.py:766`) — so
eager-serving generations are never baked → safe to free. **Caveat,
not covered by the flag fix:** `splitting_ops` is a user-settable
config field and **can be set to `[]`** explicitly today
(`vllm/config/compilation.py:1135-1200`, which warns but does not
block this for PIECEWISE/FULL_AND_PIECEWISE) — that configuration
would put attention inside a piecewise-captured graph, defeating this
policy's safety argument regardless of the flag-reset fix. This is a
present-day reachable misconfiguration, not a hypothetical future
one; either assert against it when this backend is selected with a
non-default `splitting_ops`, or document it as unsupported.

**Coupling to capture order (new, from review):** the "one generation
survives the whole capture sweep" property depends on
`get_capture_descs` sorting batch sizes **largest-first**
(`vllm/v1/cudagraph_dispatcher.py:326-344`, verified). If a future
vLLM upgrade changes that order, ascending-B capture would retire a
full-width generation at every B-growth step *during capture itself*,
potentially OOMing at capture time instead of prefill time. Cheap
guard: assert/warn if `len(_gather_retired) > 1` immediately after
capture completes (§5 test list).

**(c) `vllm/gfx906_fa/gfx906_fa_paged.py` — buffer checks + kernel Sk (~L446-528)**

**This must NOT be a single shared capacity-check change** — the
original draft changed the shared `kbuf`/`vbuf` selection block
(~L451-466) to `>=` for all consumers, but only changed the `Sk`
argument passed at the *persistent* call site. That silently breaks
buffer reuse on the other three call sites: `gather_paged_kv_q8`
(LEGACY=0/`_FUSED`), `gather_paged_kv_quantized` (`num_seqs > 16`,
`Sk_pad <= 65535`), and `gather_paged_kv_fp16` (`num_seqs > 16`,
`Sk_pad > 65535`). All three go through C++ `use_or_alloc` lambdas
(`gfx906_fa.cpp`, three near-identical copies) that check
`t.size(2) == Sk` **exactly** — passed a wide `kbuf`/`vbuf` alongside
logical `Sk_pad`, the check silently fails and C++ falls back to a
fresh `torch::empty(...)` **per layer per step**, with no exception
and no log. That is strictly worse than today (today's exact-width
buffer reuses fine on those paths) — a regression, not merely an
unfixed gap.

Correct scoping — capacity-check the persistent branch's buffer
selection only, leave the other three call sites on today's
exact-match selection:

```python
# Persistent-branch buffer selection (>= capacity, reuse the wide buffer):
kbuf_pers = k_gather_buf if (
    not _NO_BUF_REUSE and k_gather_buf is not None
    and k_gather_buf.dtype == torch.uint8 and k_gather_buf.dim() == 4
    and k_gather_buf.shape[0] >= num_seqs
    and k_gather_buf.shape[1] == hkv_k
    and k_gather_buf.shape[2] >= Sk_pad            # capacity, not exact
    and k_gather_buf.shape[3] == bytes_per_row_expected
    and k_gather_buf.is_contiguous()) else None
vbuf_pers = ...  # same shape[2] >= Sk_pad condition; keep k/v decisions
                  # coupled (use the pair only if BOTH pass — a mixed
                  # state, k reused/v fresh-per-layer, is a pre-existing
                  # footgun worth closing here) — pre-existing risk noted
                  # by review, not introduced by this change, but this
                  # rewrite is the point to fix it.

Sk_buf = kbuf_pers.shape[2] if kbuf_pers is not None else Sk_pad

K_q8, V_bhsd = gfx906_fa.gather_paged_kv_quant_persistent(
    key_cache, value_cache, bt_i32, sl_i32, Sk_buf,     # buffer width
    k_out=kbuf_pers, v_out=vbuf_pers)

# gather_paged_kv_q8 / gather_paged_kv_quantized / gather_paged_kv_fp16:
# UNCHANGED — keep today's exact-tuple `k_gather_buf`/`v_gather_buf`
# selection (== Sk_pad) so these call sites see kbuf=None on a wide
# buffer, exactly as before. No silent divergence, no regression.
```

- `gfx906_fa.forward(q_padded, K_q8, V_bhsd, ...)` is shape-driven and
  needs **no change**: `seq_kv` = width only feeds the `ne11` fallback
  (unused — `kv_max` always passed) and the k_q8/v_fp16 consistency
  checks (both width).
- `kv_max_tensor = seq_lens` unchanged — attention cut stays
  per-sequence live length.
- Inline causal via `q_abs_offset` unchanged (independent of Sk).

**Consequence of this scoping, stated explicitly (not left implicit
as in the original draft):** batches with `num_seqs > 16` (the
`_PERSIST_MAX_SEQS` gate, `gfx906_fa_paged.py:120`) and any LEGACY=0
forward get **no benefit** from this fix — they keep today's per-step
allocation churn on the non-persistent paths, unchanged. This is a
known, accepted scope limit, not a silent gap: those paths' tail-zero
semantics are width-proportional work (§2.2d), so extending them to
wide-buffer reuse is a separate, larger change. Confirm before
validation (§4) whether the 27B repro's effective concurrent-seq count
stays at or below 16 — if it does, this scope limit does not affect
that specific validation run, but it will resurface at higher
concurrency (e.g. the 35B-A3B `max_num_seqs=32` decode config) and
should be tracked as a follow-up, not silently absorbed into "the fix
covers 256k."

**(d) What NOT to change**

- `csrc/gfx906_fa/gfx906_fa.cpp` `use_or_alloc` (persistent-path copy,
  ~L933-950; the other two copies at ~L614-634/L719-737): exact match
  stays everywhere — Python now always passes `Sk = buffer width` only
  on the persistent call, so that one check passes; the other two
  continue to see exact-match Python inputs unchanged. Keeping C++
  exact everywhere guards against stride/addressing mismatches without
  needing any C++ diff at all.
- The **fused q8 path** (`gather_paged_kv_q8`, `key_cache_q8 is not
  None` + `_FUSED`) and **`gather_paged_kv_quantized`**: both zero the
  entire V tail up to `Sk` in-kernel — at width 262144 that is a
  ~800 MB–3.3 GB memset per layer per step. Keep exact-match (`Sk =
  Sk_pad` logical) for these paths per (c) above; do not attempt wide
  reuse here without a tail-bounded kernel rewrite (out of scope).
- `_q_pad_buf` pattern (slice + `.contiguous()`): fine for a kernel-*read*
  staging input (copy cost only), wrong for kernel-*written* outputs —
  a `.contiguous()` would duplicate the whole gather. That's why K/V
  reuse must be "pass the wide buffer, write in place" instead of
  slicing.

### 2.3 Why the original "no C++ change" framing needed the (c) correction

The original design read "pass buffer width as Sk, no C++ changes" as
a blanket statement covering every gather call site. It is correct
**only** for the persistent kernel (§2.1) — the non-persistent V1/V2
kernels have a real `gridDim.z <= 65535` limit
(`gfx906_fa_gather.hip:401`, `gfx906_fa.cpp:821`
`TORCH_CHECK(Sk <= 65535, ...)`) that a wide `Sk` argument would
violate outright, which is exactly why (d) keeps those paths on exact
logical `Sk_pad` rather than buffer width. The corrected plan is
narrower and more precise than the original: "no C++ change" applies
to one call site, not three, and (c)'s scoped Python change is what
keeps that distinction intact end to end.

### 2.4 Memory accounting

| | today (run 4) | after fix (persistent path, `num_seqs <= 16`) |
|---|---|---|
| capture-time full-width gen | allocated, then retired at first chunk | allocated once, **reused forever** |
| serving-time gens | 1 per Sk growth step (~256 gens to 262k at 1024-chunk) | **0** (width covers all) |
| `_gather_retired` | unbounded: order tens of GiB by 250k (unverified exact figure — see §1.4) | bounded to graph-baked gens only (expected: exactly one, under FULL_AND_PIECEWISE with descending capture order) |
| steady-state cost | — | the capture-time allocation already paid today (B=cap × Hkv × 262144 × 784 B; ~822 MB at B=2/TP=2, ~3.3 GiB at B=8) + zero growth on the persistent path |
| no-capture (PIECEWISE-only) serving | one live gen at ~live width, freed on replacement (latch never set) | **new cost**: a long request leaves a full-width buffer resident **permanently** once grown (grow-only, never shrinks) — same ~822 MB–3.3 GiB range. Almost certainly the right trade (it's what FULL modes already pay) but is a genuinely new standing cost in PIECEWISE-only mode, not "free everywhere" as the memory table implied in the original draft. |
| `num_seqs > 16` or LEGACY=0 | unbounded retire, same as "today" row | **unchanged** — out of scope, see §2.2(c) consequence note |

Headroom verdict: **conditional, not established.** If the AWQ-scratch
mechanism (§0) is the true binding constraint, this fix buys back
whatever headroom the unbounded retire dict was consuming but does not
by itself guarantee the 1.9–2.5 GiB run-4 headroom survives to the
lm_head's 2.37 GiB scratch — that depends on the magnitude of the
retire-dict consumption, which §1.4 flags as unverified. Treat "256k
prefill completes" as the hypothesis under test (§4), not the
predicted outcome.

## 3. Open questions carried from the original diagnosis (unresolved, must be addressed before claiming success)

1. **"131k is the validated context ceiling on this model"**
   (`oom-256k-prefill.md`) is not reconciled with this fix's mechanism.
   If a 131k-token prefill completed on post-`5d960a503c` code (which
   introduced the unbounded retire), the same unbounded-growth
   mechanism should have exhausted headroom near the same ~30-50k
   token mark this plan's (unverified) simulation predicted — unless
   that 131k validation predates the UAF fix, ran in an eager/PIECEWISE
   mode where the sticky-latch bug didn't apply, or the growth
   magnitude is smaller than estimated. The OOM doc gives no run
   config or date for the 131k claim. Resolve this before writing
   "this fix closes 256k" into any dev log.
2. **The `TRITON_ATTN` control arm is the single most informative
   missing experiment** and should be run before the 27B repro: one
   250k prefill with `--attention-backend TRITON_ATTN` (bypasses this
   backend, and its config-level escape hatch is already listed in
   §6). If it OOMs identically, the FA-growth theory contributes
   nothing to the 256k OOM and this fix should be reframed purely as
   "worthwhile bugfix," with the AWQ-scratch mechanism (§0) pursued
   separately.
3. **A fixed-token-count probe** (needle at 60k/100k) tests the sharp
   prediction implied by any retire-dict-growth theory: headroom
   exhaustion at a roughly fixed *token count*, independent of the
   target prompt length. If a 100k-token prefill OOMs at the same
   point a 250k one does, that's strong evidence for the growth
   mechanism; if it doesn't OOM at all, the mechanism's magnitude was
   overestimated.

## 4. Validation plan (gated on the weight-load failure)

The host currently cannot boot real weights reliably
(`hipErrorLaunchFailure`, intermittent GPU die/fabric flap with
multi-minute good windows — `DEVLOG-boot-failure.md`). Sequence once
boot is stable enough for multi-minute runs:

1. **Re-derive the growth estimate** from the actual buffer-size
   formula (`B × Hkv × Sk_pad × 784 B` per generation, summed over the
   real chunk schedule) and commit the script under
   `docs/gfx906/` or a persistent (non-`/tmp`) scratch location — do
   not cite a number that isn't reproducible from a script in the
   repo.
2. **Unit**: `pytest tests/kernels/attention/test_gfx906_fa.py -k gather`
   + the rewritten keepalive test (§5) + the new width≫live test,
   including the `num_seqs > 16` non-regression smoke test (§5).
3. **9B probe**: expect **one** `alloc` line at capture, **zero**
   during a 100k prefill, `retired=` constant at 1, VRAM ramp flat.
   Also run one **multi-request mixed batch** step with `num_seqs > 16`
   through the probe — this is where the §2.2(c) scoping bug would
   have shown up if missed, so it's the cheapest regression check for
   that specific fix.
4. **TRITON_ATTN control arm** (§3.2): run *before* step 5, not after.
5. **27B run-4 repro** (TP=2, util 0.82, MTP n=2, chunk 1024, prefix
   caching, FULL_AND_PIECEWISE, capture sizes matching the actual
   `--max-num-seqs` used — state which capture-size list is used, since
   `[1,2,4,8]` vs. the house-recipe-trimmed `[1,2]` changes the
   capture-time generation size and thus the §2.4 headroom table row
   that applies): 250k-token first prefill completes without OOM.
6. **Fixed-token-count probe** (§3.3): needle at 60k/100k.
7. **Accuracy**: needle retrieval at 256k sanity-checks long-context
   attention with the wide buffers (width≫live correctness, not just
   memory).
8. **Perf A/B**: decode t/s (persistent path untouched — expect flat);
   prefill t/s at 32k+ (removes per-chunk hipMalloc/hipFree churn on
   the persistent path — expect ≥ parity, likely improvement); a 35B
   `max_num_seqs=32` decode t/s A/B specifically to confirm the
   `num_seqs > 16` scope-limited paths are unaffected (should be
   exactly flat, since they're unchanged).
9. This needs a dev log (`DEVLOG-fa-gather-lifecycle.md` or folded into
   `DEVLOG-fa-attention.md` per the topic-grouping convention in
   `docs/gfx906/AGENTS.md`) with the serving-wall A/B as the gate, per
   house rules. Any `OOMHUNT_LOG` instrumentation stays in-tree only
   for validation and must be reverted before merge.

## 5. Test updates (`tests/kernels/attention/test_gfx906_fa.py`)

`test_gather_buffers_capture_sweep_keepalive` (~L558) currently asserts
retire-on-every-Sk-change. Rewrite to pin the NEW contract:

1. shrink / ping-pong Sk (130 → 100 → 130) → **no realloc** (`_k_gather_buf
   is` unchanged, `retired` count unchanged);
2. **(corrected — the original item 2 is unconstructible in the real
   flow, since FULL capture always runs at `max_model_len`, so
   `Sk_pad > captured width` cannot occur post-capture organically)**:
   drive the retire branch directly — capture against a test-forced
   narrow `max_seq_len` (override the dummy metadata, not real FULL
   capture semantics), then grow past it; assert the previous
   **captured** gen lands in `_gather_retired` and stays alive;
3. grow in eager mode (never captured) → replaced gen is **freed**:
   assert `data_ptr() not in _gather_retired` AND that a subsequent
   allocation reuses the freed VA (allocator-level evidence) — do not
   assert on CPython refcount, which is not reliable evidence of a
   free;
4. capture-baking invariants unchanged (slice `[:B]` same base VA
   across the capture sweep);
5. **new**: assert `_gather_buf_captured` is reset (not OR'd) across a
   replacement — construct a captured-then-eager-then-captured-again
   sequence and check the middle generation is freed, not retained;
6. **new**: end-of-capture assertion that `len(_gather_retired) <= 1`
   under the current (largest-first) capture order, documented as
   coupled to `get_capture_descs` sort order (§2.2b).

Plus new units:

- Persistent gather + FA forward with buffer width ≫ live seq_len
  (e.g. width 4096, seq_lens [37, 1000]): bitwise vs the exact-width
  reference (in-range rows), **and** the stale region
  `[seq_len + margin, width)` pre-filled with NaN/Inf before the
  forward (poisons exactly the data that would leak through if the
  tail mask were ever broken) — assert bitwise-equal in-range rows and
  no NaN/Inf in the output. This pins the invariant currently resting
  on a kernel comment.
- Mixed `seq_lens` at `num_seqs > 1` with width ≫ live, to exercise the
  margin-clamp arithmetic (`extra = min(margin_zeros, Sk -
  min(seq_len, Sk))`) per-sequence, not just at `num_seqs == 1`.
- **`num_seqs > 16` non-regression smoke test**: an eager forward with
  `num_seqs = 17..32` against a wide (`>` `Sk_pad`) class buffer,
  asserting the non-persistent path still gets a working buffer (no
  crash) and, separately, that per-call `torch.cuda.memory_allocated`
  deltas match today's baseline (i.e. confirming §2.2(c)'s scoping fix
  actually prevents the regression it was designed to prevent).
- `GFX906_FA_GATHER_EXACT=1` A/B: verify it restores byte-identical
  behavior at **all** three sites it needs to gate (the backend
  grow-only check, the paged.py persistent-branch capacity check, and
  the `Sk_buf` choice) — a switch that only gates one of the three
  produces a meaningless A/B.
- MTP draft-layer forward included in the width≫live unit (shares the
  same class-level buffers via matching Hkv/D) — the one forward shape
  the original test plan omitted.

## 6. Fallbacks / kill switches

- `GFX906_FA_NO_BUF_REUSE=1` already exists (disables buffer reuse
  entirely — keep as-is; it makes things worse, only for debugging).
- Add `GFX906_FA_GATHER_EXACT=1` to restore the old exact-match policy
  for A/B (default: new grow-only), gated at all three sites listed
  in §5's last bullet. **Lifecycle note (2026-08-24):** this is a
  temporary A/B arm, NOT a permanent knob like `GFX906_FA_PERSIST`/
  `GFX906_FA_LEGACY`. Its value is that the pre-fix policy stays
  byte-for-byte (that is what made the arm-A OOM repro byte-exact) —
  which is why the EXACT branches are deliberately NOT deduplicated
  with the capacity policy. Drop the switch (both files, both read
  sites, the kill-switch test, and the `k_exact` derivation) at the
  NEXT gather-lifecycle change, re-gated on a serving A/B; until then
  any lifecycle edit must touch both policies and keep them
  divergence-free.
- `GFX906_FA_CG=never` removes the captured-gen path entirely (retire
  set stays empty) — cheapest bisect knob if anything regresses.
- Config-level (no code): `--attention-backend TRITON_ATTN` bypasses
  the custom FA entirely (loses N4 wins; known-good escape hatch, and
  per §3 should be run as a control arm, not just documented as a
  fallback); `--cudagraph-mode PIECEWISE` avoids the capture latch (no
  unbounded retire) but keeps per-chunk alloc/free churn; `NONE`
  similarly at large decode cost.

## 7. Risks

- **Wide-V tail garbage read by the FA tail tile (masked)**: identical
  to today's mixed-length batches; re-probed explicitly by the new
  NaN-poisoned width≫live unit test (§5), not just a comment-level
  claim.
- **int overflow**: `Sk=262144` fits int32; kernel addressing is
  int64; decode already executes at this width every step.
- **Retire-free correctness**: hinges on "attention is never inside a
  PIECEWISE graph," which is the *default* `splitting_ops` behavior
  but **not a hard invariant** — `splitting_ops=[]` is a reachable
  user config (§2.2b) that defeats it independent of the flag-reset
  fix. Document or assert against this combination.
- **Capture-order coupling**: the single-generation-survives-the-sweep
  property depends on descending-size capture order remaining true
  upstream (§2.2b) — guarded by a cheap end-of-capture assertion, not
  currently enforced anywhere else in the codebase.
- **Scope limit, not a risk but must not be forgotten**: `num_seqs >
  16` and LEGACY=0 get zero benefit from this fix (§2.2c) — track as a
  known follow-up, do not let it silently disappear from future
  status updates.
