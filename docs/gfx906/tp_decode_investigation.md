# TP=2 dense decode — max_model_len / context-length decode-speed investigation

**VERDICT:** OPEN · **GATE:** none run yet — this is a code-analysis
investigation, no serving A/B has confirmed the mechanism live. See
"What needs to be checked" for the decisive experiments.

## Origin

During TP=2 dense-27B serving work (`DEVLOG-tp2-dense.md` S5, 39.7 t/s
record at `--max-model-len 131072`), a session testing
`--max-model-len 262144` on the same server/model/flags measured
**29.9 t/s decode vs a 39.9 t/s control at 131072** — a ~25% regression
reproduced both ways (clean restart, acceptance unchanged). This matches
the already-documented finding in `DEVLOG-tp2-dense.md` line 194-195
("Where TP=2 still pays: 131k-context capacity... at 262144 max_model_len
boots... at ~7-12 t/s decode" — a larger gap than 29.9, but same
direction) and `DEVLOG-dense-decode.md` line 476 ("max-model-len 131072
was unbootable [under an older config]... the gather buffers are sized
`max_num_seqs × pad32(max_model_len)`").

Two independent hypotheses were floated before code was read:

1. **Buffer-size theory** — bigger `max_model_len` → bigger block
   tables / FA gather buffers → some fixed per-replay cost scales with
   buffer capacity.
2. **Frozen split-count theory** (informed by generic ROCm/CUDA FA
   knowledge, not gfx906-specific) — FULL cudagraph capture bakes
   Flash-Decoding-style KV-split counts computed from the worst-case
   `max_model_len`; every replay pays for the frozen (inflated) split
   count regardless of real per-request context length.

Both were investigated against the actual code paths active in this
deployment (confirmed via `/tmp/vllm-tp2-mtp3b.log`: `GFX906_FA` backend
registered as `AttentionBackendEnum.CUSTOM`, `--max-model-len 131072`
control run, MTP `num_speculative_tokens=3`).

## What was ruled out

- **Block-table H2D copy size** (`vllm/v1/worker/block_table.py`
  `BlockTable.copy_to_gpu` → `CpuGpuBuffer.copy_to_gpu`). Row width
  (`max_num_blocks_per_req`) does scale with `max_model_len`
  (`vllm/v1/kv_cache_interface.py` `max_num_blocks_per_req()`), but the
  extra bytes moved per step from doubling `max_model_len` are in the
  hundreds-to-low-thousands range at `max_num_seqs=4` — sub-microsecond
  even unpinned; this buffer is pinned + `non_blocking=True`. Cannot
  explain an ~8.4 ms/token regression (25.1 ms → 33.4 ms at 39.9 → 29.9
  t/s). **Ruled out on magnitude.**
- **Triton `kernel_paged_attention_2d`** (`vllm/v1/attention/ops/
  chunked_prefill_paged_decode.py`, the `ROCM_ATTN`/`rocm_attn.py`
  fallback backend). Its inner loop is `num_blocks =
  cdiv_fn(seq_len, BLOCK_SIZE)` (line ~159) — driven by the real
  per-request `seq_len` at kernel-launch time, not by
  `max_num_blocks_per_req`/`block_table.shape[1]`. `block_table_stride`
  is used only as a pointer-arithmetic multiplier for row indexing, at
  zero extra cost regardless of row width. **Not the mechanism — and
  not the active backend for this deployment anyway** (gfx906 always
  has `use_rocm_custom_paged_attention()` return `False`, see
  `vllm/platforms/rocm.py:412-413`, so this path is dead code on
  gfx906 hardware for the custom-kernel branch; `ROCM_ATTN` is only
  selected as a fallback below `CUSTOM`, and the log confirms `CUSTOM`
  was active).
- **Generic split-K/Flash-Decoding "frozen split count baked into the
  graph"** (`vllm/v1/attention/ops/triton_unified_attention.py`,
  `NUM_SEGMENTS_PER_SEQ`). This mechanism is real *in that file*, but
  that kernel belongs to `TRITON_ATTN`/`rocm_aiter_unified_attn.py`,
  which is not the backend in use here. **Right shape of theory, wrong
  kernel for this deployment.**
- **`max_seq_len` frozen to `max_model_len` at replay time.**
  `Gfx906FAMetadataBuilder.build()` (`vllm/gfx906_fa/gfx906_fa_backend.py`
  line ~153) sets `max_seq_len=common_attn_metadata.max_seq_len`, which
  for real (non-capture) steps is computed fresh every step
  (`vllm/v1/worker/gpu_model_runner.py:2393`:
  `self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()`).
  `optimistic_seq_lens_cpu` itself
  (`gpu_model_runner.py:2145-2148`) is `num_computed_tokens_cpu +
  num_scheduled_tokens` — genuinely live per-request data. Freezing to
  `self.max_model_len` (`gpu_model_runner.py:2391`) only happens when
  `for_cudagraph_capture=True`, i.e. the one-time capture/warmup pass,
  not real decode replays. **Ruled out**: there is no code path where
  a request with genuinely short real context is charged
  `max_model_len`-sized work at decode time.

## What was confirmed as the likely mechanism

`GFX906_FA` (`vllm/gfx906_fa/`) is a **gather-then-dense-attention**
design, structurally different from the two Triton kernels above:

1. Decode (`GFX906_FA_LEGACY=1`, then the default — **now the rollback**, see the 2026-09-16 note at the end of this section) gathers K/V from the
   paged fp16 cache into a **dense** buffer of shape
   `[B, Hkv, Sk_pad, D]` via the fused HIP kernels
   `gather_paged_kv_quantized`/`gather_paged_kv_fp16`
   (`vllm/gfx906_fa/gfx906_fa_paged.py` `forward_paged`, `Sk_pad =
   pad32(max_seqlen_k)`, line ~317).
2. `max_seqlen_k` passed in is `attn_metadata.max_seq_len`
   (`gfx906_fa_backend.py:663,675,690`) — confirmed live, not frozen
   (see above).
3. The FA kernel then runs dense attention over the full `Sk_pad`
   width — there is no per-request early-exit/mask-driven skip the way
   `kernel_paged_attention_2d`'s `j == num_blocks - 1` tail masking
   works; the whole padded buffer is real compute.
4. **This cost is empirically linear in `Sk`, already measured and
   recorded**: `DEVLOG-fa-attention.md` line 282-283 — "FA kernel 3272
   (11.3 ≈ 327 µs/layer @ Sk~2176 ≈ 4.3× the Sk~500 72 µs — Sk-linear)."
   Multiplied across every layer (10 FA layers in the Qwen3.5-27B
   config referenced in that devlog; layer count for the TP=2 dense
   model under test needs confirming) and every decode step, a batch
   whose live `max_seq_len` grows 4x produces ~4x the FA-kernel time.

**Conclusion so far:** the ~25% decode regression at `--max-model-len
262144` is not caused by the config value itself (no code path charges
`max_model_len`-sized work at decode time for short-context requests).
It is much more likely caused by **the actual accumulated context
length reached by the 262144 test's requests being longer than the
131072 control's**, combined with this backend's `Sk`-linear
gather-then-dense design (which has no sparsity benefit within the
padded buffer, unlike the Triton paged kernels). This would be expected
behavior for this kernel architecture — "longer real context costs
more decode time" — not a `max_model_len`-triggered bug.

## Secondary candidate: gather-buffer realloc thrash (unconfirmed, likely minor)

`Gfx906FAImpl._ensure_gather_buffers` (`gfx906_fa_backend.py` line
~479) uses an **exact-match** realloc condition on `Sk_pad`
(`b.shape[2] != Sk_pad`, line ~526) rather than a monotonic grow-only
policy like its sibling `_ensure_forward_buffers`/`_q_pad_buf` (which
uses `shape[2] < Sq_pad`, only grows). This means every time the live
batch's `max_seq_len` crosses a 32-token `pad32` boundary — plausible
under MTP (`num_speculative_tokens=3`, variable accepted length) or a
mixed batch where requests finish/start — the K/V gather buffers
reallocate (`torch.empty`), not just resize a view.

This was flagged in code review as **R3**/**R8** and is recorded in
`CHANGELOG.md`; it is **already partially addressed**: R3 capped the
retired-buffer keep-alive list at
`_gather_retired_max = 4` (bounding the leak, not the realloc
frequency), and R8 made the class-level gather buffers shared on both
`LEGACY=0` and `LEGACY=1` paths (previously `LEGACY=1` allocated fresh
per call). Neither fix changes the exact-match shrink-and-regrow
behavior itself. Whether this thrash is a measurable contributor on
top of the expected `Sk`-linear cost, or is negligible next to it
(reallocation is a device `cudaMalloc`-class op, not free, but may be
dwarfed by the 327 µs/layer FA cost at realistic sizes) — **not yet
measured**.

## What needs to be checked

In priority order (cheapest/most decisive first), per the gate rule in
`docs/gfx906/AGENTS.md`: **serving wall-clock A/B is the gate**, not
kernel census.

1. **Sweep real context length at fixed `--max-model-len 262144`.**
   Measure ITL at ~2k, ~30k, ~120k tokens of actual conversation/prompt
   length, same server, same flags as the S5 config
   (`-tp 2 --gpu-memory-utilization 0.93 --max-num-seqs 4
   --max-model-len 262144 --compilation-config
   '{"cudagraph_capture_sizes":[1,2,3,4]}'`).
   - ITL flat across all three → something is still charging
     `max_model_len`-sized (or otherwise config-fixed) work regardless
     of real content; re-open the ruled-out theories above and check
     for a missed freeze point.
   - ITL rises with real length, offset upward vs. an equivalent
     131072-config sweep at matched real length → confirms the
     `Sk`-linear gather-attention story; the "regression" is really
     "the 262144 test ran with longer real context than the control,"
     which is expected, not a bug.
2. **Matched-length control.** Run the 131072 config and the 262144
   config at the *same* real accumulated context (e.g. both servers
   actually holding ~100k tokens of real conversation). If decode
   speed still differs at matched real `Sk`, something beyond the
   `Sk`-linear FA cost is at play — most likely candidate is #3 below,
   or KV-cache/`gpu_memory_utilization` budgeting differences between
   the two `max_model_len` values changing concurrency/preemption
   behavior.
3. **Gather-buffer realloc thrash isolation.** With `GFX906_FA_DEBUG=1`
   (master debug switch, `gfx906_fa_backend.py` R12) or a targeted
   counter, log `_ensure_gather_buffers` realloc events per step during
   an MTP-enabled decode run. Compare realloc frequency/cost at 131072
   vs 262144 configs at matched real context — if thrash is common and
   its cost is non-trivial relative to the 327 µs/layer FA-kernel
   floor, it's a distinct, fixable lever (grow-only capacity buffer
   with an exact-size view, same pattern already used for `_q_pad_buf`
   — noting R3's finding that a naive grow-only capacity buffer risks
   silent corruption here because the gather kernels address output
   from shapes, not strides; any fix must preserve exact-shape
   addressing).
4. **`--enforce-eager` at both max_model_len values** (supporting
   check, not primary — `GFX906_FA`'s `get_cudagraph_support` already
   declares `UNIFORM_SINGLE_TOKEN_DECODE`/`UNIFORM_BATCH`, so this
   mainly rules out any residual capture-vs-eager metadata-freezing
   difference not covered above). Gap collapses → capture-related
   after all despite the code-read above; gap persists → confirms the
   mechanism is the live-context-length-driven FA cost, not a
   graph-capture artifact.
5. **`rocprofv3 --kernel-trace` diff** on both configs at matched real
   context: confirm the FA/gather-kernel durations account for the
   full measured gap (per the `Sk`-linear coefficient from
   `DEVLOG-fa-attention.md`), rather than some other kernel growing.
   Per `docs/gfx906/AGENTS.md`, treat this as launch-regime evidence,
   not the gate — the serving A/B in #1/#2 is authoritative.

## Cross-references

- `DEVLOG-tp2-dense.md` — TP=2 serving record and the original
  131072-vs-262144 observation (S4/S5 sections).
- `DEVLOG-fa-attention.md` — source of the `Sk`-linear FA-kernel
  measurement (327 µs/layer @ Sk~2176 vs 72 µs/layer @ Sk~500) and the
  fused-gather buffer design/capture-safety notes.
- `DEVLOG-dense-decode.md` — earlier note on gather-buffer sizing vs
  `max_model_len` in a different (hybrid GDN+FA) deployment.
- `CHANGELOG.md` — the R3/R8 gather-buffer retired-list cap and
  shared-buffer fixes that partially bear on the thrash candidate in this doc.

---
## RESOLUTION (2026-08-21 late, pi agent) — experiment #4 decisive: capture-baking CONFIRMED

**VERDICT:** CONFIRMED mechanism / FIXED — SUPERSEDED ANALYSIS BELOW
(kept for history; the fix that shipped is the persistent live-bounded
gather, `DEVLOG-masked-fa.md` — FULL-cudagraph capture freezes
`max_seq_len = self.max_model_len` (`gpu_model_runner.py:2390`,
`for_cudagraph_capture` branch), so the GFX906_FA gather-then-dense
kernels get `Sk_pad = pad32(max_model_len)` baked into their launch
dims; **every decode replay pays dense attention proportional to
max_model_len regardless of live context**. The "ruled out" above was
correct for live/piecewise steps (metadata is fresh) but incomplete:
FULL graph replay re-executes capture-time launch dims.

Evidence (identical prompts ~1.5k real tokens, same offline harness):

| max_model_len | graph decode | eager decode |
|---|---|---|
| 131072 | 39.9 t/s (server) | 18.9-19.5 t/s |
| 262144 | 29.9 t/s (server) | 19.7-19.9 t/s |

Eager gap at matched real context: none (256k marginally faster = noise).
Graph gap: -25%, tracking pad32(max_model_len) doubling.

### ⚠ CORRECTION (2026-08-21, later same day) — the bounded-capture proposal below is broken for its stated purpose

**Gross oversight, caught in review before any code was written.** The
bound-check in the mechanism below compares against
`num_computed_tokens_cpu + num_scheduled_tokens`
(`optimistic_seq_lens_cpu`'s definition, confirmed in
`vllm/v1/core/sched/scheduler.py:881-884`: `num_computed_tokens =
num_new_local_computed_tokens + num_external_computed_tokens`,
asserted `<= request.num_tokens`) — this is **total accumulated
sequence length including any prefix-cache hit**, not the size of the
newly-added increment. A prefix-cache hit sets `num_computed_tokens`
to a large value on the *first* scheduling of the request; it does not
start at 0 and grow into the cached portion over time.

Consequence: a long-running or deep multi-turn conversation crosses
the bound **once** (e.g. at turn N where accumulated history passes
32k tokens) and then **every subsequent decode step for that
conversation falls back to PIECEWISE/eager permanently** — even though
the actual new work per step is a completely ordinary single-token
decode against a KV cache vLLM already has fully paged and resident.
This is not a corner case; it is the **common case** for exactly the
workload `--max-model-len 262144` was raised to serve. As designed,
the single-bound version of this fix helps only short-lived/low-depth
conversations on a server configured with a large `max_model_len` for
headroom, and is neutral-to-useless for the long-context traffic that
motivated raising `max_model_len` in the first place. It does **not**
generalize to "all decode gets faster" as the original proposal
implied — that claim is retracted.

This isn't fixable by reading a different counter — the constraint is
structural: decode attention legitimately attends over the *entire*
accumulated KV cache every step, so `Sk` is inherently a total-length
quantity, not an increment quantity, and any fix that keys off total
`Sk` inherits this cliff. See the two alternatives below (multi-tier
capture, masked early-exit) for how to actually serve long
conversations at reasonable speed; the single-bound version is kept
here only as the "cheap partial win for short-context-heavy traffic"
option, not the primary recommendation.

### Fix lever — bounded-capture Sk + dispatch fallback

> **SUPERSEDED (2026-08-22):** none of the three proposals below was
> built. `gather_paged_kv_quant_persistent` (branch
> `gfx906/fa-masked-gather`, `DEVLOG-masked-fa.md`) implements the
> masked-early-exit route — one live-bounded kernel at every `Sk`, all
> gates passed, serving A/B 131k 22.4→40.9 / 262k 15.9→40.9 t/s — and
> the devlog's Interactions section records why it supersedes the
> bounded-capture and multi-tier designs. Retained as analysis only.

**Status: design only, and per the correction above, only a partial
fix (short-context traffic) — not a general one.** Nothing below has
been implemented or gated; treat every number in the
performance-impact section as an estimate from a two-point linear
extrapolation, not a measurement. House protocol
(`docs/gfx906/AGENTS.md`) applies before any of this ships:
micro-bench, PPL/greedy gates, serving A/B, separate commit.

#### Mechanism

Two coordinated changes, both plain Python/config — no vendored HIP
kernel touched:

1. **Bound what capture bakes in.** In `_build_attention_metadata`
   (`gpu_model_runner.py:2387-2391`), replace the unconditional
   `max_seq_len = self.max_model_len` (used whenever
   `for_cudagraph_capture=True` — confirmed this fires for *every*
   FULL-mode capture call, not just the `i==0` buffer-sizing warmup
   pass, which is a separate mechanism keyed off `profile_seq_lens`)
   with `max_seq_len = min(self.max_model_len,
   self.cudagraph_capture_max_seq_len)`, a new runner attribute set
   once at init from an env knob (e.g. `VLLM_GFX906_FA_CAPTURE_SK`,
   default something like 32768). Generic runner-level change, inert
   for backends whose kernels are already live-`seq_len`-driven
   (`ROCM_ATTN`'s `kernel_paged_attention_2d`, `TRITON_ATTN`'s unified
   kernel) — only `GFX906_FA`'s gather-then-dense design is sensitive
   to this value. Keep the existing sliding-window comment's intent:
   bound should be `max(configured_bound, max sliding window across
   layers)`.
2. **Refuse the FULL graph once live context exceeds the bound.**
   Before `_determine_batch_execution_and_padding` is called
   (`execute_model`, ~line 4389), `self.input_batch.
   num_computed_tokens_cpu[:num_reqs]` and `num_scheduled_tokens_np`
   are already available (used one line earlier for cascade-attn
   prefix lens) — compute `live_max_seq_len` from them the same way
   `optimistic_seq_lens_cpu` does later, and set `exceeds_capture_bound
   = live_max_seq_len > self.cudagraph_capture_max_seq_len`. Thread it
   into the existing `disable_full` plumbing exactly like
   `use_cascade_attn` already does: `disable_full=use_cascade_attn or
   has_encoder_output or exceeds_capture_bound`. This reuses
   `CudagraphDispatcher.dispatch`'s existing `invalid_modes=
   {CUDAGraphMode.FULL}` parameter — no dispatcher change needed. The
   step then runs PIECEWISE (if compiled that way) or eager, where
   `Gfx906FAMetadataBuilder.build()` already computes `max_seq_len`
   live and correctly (confirmed in "What was ruled out" above) — no
   new correctness risk, since the fallback path never bakes launch
   geometry, it just doesn't get the FULL-graph speedup for that step.

Capture count/sizes (the batch-size buckets) are unaffected — only
`Sk_pad` within each captured graph shrinks. The `i==0` profiling-run
memory-sizing pass should stay tied to `self.max_model_len` (not the
new bound) since it exists to budget the allocator for true worst
case; conflating the two "worst case" uses would under-provision
memory.

Correctness note: there is nothing to guard against beyond "don't
dispatch FULL when it doesn't apply" — a request over the bound simply
never enters a FULL graph, so there is no truncated-context/garbage-
output failure mode the way, e.g., R2's LEGACY=0+prefix-caching
corruption was. The only real risk is a **batch-wide performance
cliff**: dispatch is per-step batch-wide, not per-request, so one
long-context request in a batch drops the *whole* batch to
PIECEWISE/eager for that step, even if the other requests are
well within bound. `logger.warning_once` when this fires, so it's
visible in production, not silently absorbed.

The masked-early-exit alternative (rewrite the HIP gather+attention
kernel to skip padded tail rows within a fixed-shape graph) would let
one captured graph serve any live `Sk` up to `max_model_len` at true
cost — the ideal fix — but grid dims are still frozen at capture, so
it needs the kernel body itself to branch/skip on a runtime tensor
value: a real kernel-design project, not a knob. The bounded-capture
approach is the cheap, low-risk first step; the kernel rewrite is the
natural follow-up if the batch-wide-cliff fallback rate proves too
high in practice.

#### Performance impact (estimated, NOT measured — needs the gate)

Using the `Sk`-linear coefficient from `DEVLOG-fa-attention.md` (327
µs/layer @ Sk~2176, 72 µs/layer @ Sk~500 → ≈0.152 µs/layer per token of
`Sk`) and the S8 evidence (131072-capture graph decode 39.9 t/s ≈ 25
ms/step), assuming ~10 FA layers:

| capture Sk bound | est. FA-kernel time (10 layers) | est. step time | est. decode t/s |
|---|---|---|---|
| 131072 (today) | ~19.7 ms | ~25 ms (measured) | 39.9 (measured) |
| 32768 | ~4.9 ms | ~10.3 ms | ~97 (**~2.4×**) |
| 4096 | ~0.6 ms | ~6.0 ms | ~166 (**~4×**, dubious — see caveat) |

**Caveat:** this is a two-point linear fit extrapolated 40-60× below
its calibration range (Sk~500-2176 → bound 4096-32768), and it ignores
non-attention step time possibly having its own floor/overhead that
doesn't shrink with Sk. Treat only the qualitative direction (large
win for a well-chosen bound) as trustworthy; the specific multipliers
need a real capture-bound sweep to confirm.

**For requests that fall back (exceed the bound):** no worse than
today in absolute terms — they already never benefited from the
FULL-graph shape (capture always baked the worst case for everyone).
Per S8, eager decode at matched short context runs ~19.5 t/s regardless
of `max_model_len` — call this the fallback-path floor. Today, a
long-context request already pays close to `max_model_len`-wide FULL
attention cost every step; whether that is currently faster or slower
than the eager floor at genuinely long real context is **not yet
measured** (S8's eager numbers are at ~1.5k tokens only) — needed
before claiming the fallback path is strictly non-regressive at long
context, not just at short context.

**Batch-wide cliff caveat:** dispatch is per-step, not per-request, so
realized gains depend heavily on concurrency and context-length
distribution. Single-request benchmarks (the existing
`_bench_gfx906.py` harness, `max_num_seqs=4` in the S5 config) will
read close to the ideal per-request numbers above; heterogeneous
multi-tenant serving with even occasional long-context requests will
see a smaller realized win, since one long request poisons FULL-graph
eligibility for the whole batch that step.

**Second-order effects not in the table:** capture time/VRAM should be
roughly unchanged or slightly better (smaller gather buffers during
capture); mode-switching overhead (FULL this step, PIECEWISE next) is
unmeasured but likely small next to the attention-cost swing; smaller
`Sk_pad` gather buffers free VRAM that could fund more KV-cache blocks
(more concurrency) — a real secondary win not captured in the t/s
table at all.

#### What the gate needs to confirm before this ships

1. Bounded-capture serving A/B (`_bench_gfx906.py`-style, per house
   protocol) across a sweep of bounds (e.g. 4k/8k/16k/32k/64k) at
   fixed real context below each bound — confirms/corrects the
   `Sk`-linear extrapolation above.
2. The same sweep at real context *above* each bound — confirms the
   fallback path is non-regressive (not just non-improving) relative
   to today's baseline at that context length.
3. A concurrency/mixed-length A/B (`BENCH_MAX_SEQS`-style, per the
   C2 precedent in `ROADMAP.md`) — quantifies the
   batch-wide-cliff discount versus the single-request numbers above.
4. PPL/greedy correctness gate on the fallback path itself (should be
   a no-op numerically since eager/piecewise `GFX906_FA` is already
   the LEGACY=1 default path in non-FULL execution, but must be run,
   not assumed).

### Alternative — multi-tier capture (several Sk buckets)

Prompted by the correction above: instead of one bound with a binary
eager fallback, capture FULL graphs at several `Sk_pad` tiers (e.g.
4k/32k/128k/262k) and dispatch to the smallest tier that covers the
batch's live `max_seq_len`, the same way `cudagraph_capture_sizes`
already buckets batch size today. This directly attacks the cliff:
a 33k-token conversation lands in the 128k tier instead of falling
all the way to eager.

**Why this is a worse ceiling than masked early-exit, not just a
cheaper stopgap:**

- **Cost multiplies per tier, not just once.** Every additional Sk
  tier is a full additional set of captured graphs (crossed with the
  existing batch-size buckets), each with its own gather-buffer
  generation. Per the capture-safety design in
  `_ensure_gather_buffers` (a buffer baked into a graph is retired,
  never freed, for the worker's lifetime), N Sk tiers means N
  simultaneously-resident gather-buffer generations, each sized to
  its tier's `Sk_pad`. On a 32 GB MI50 already fighting for KV-cache
  headroom (the entire reason `--max-model-len` capacity work
  happened), this directly competes with the KV cache the feature
  exists to serve. Capture time also multiplies with tier count.
- **Still a step function, not a fix.** Each tier still pays its
  *bucket's* `Sk_pad` cost, not the request's real `seq_len` — a 33k
  request in a 128k tier pays 128k-wide dense attention, ~4x its real
  cost. Finer tiers shrink average waste but push VRAM/capture cost up
  further; coarser tiers keep VRAM/capture cost down but leave more
  waste per request. There is no tier count that removes the
  structural `O(bucket_size)` tax — only ones that trade its size
  against resource cost.
- **The cliff shrinks but does not disappear.** Dispatch is still
  batch-scalar: one long request in a batch still forces the whole
  batch to that request's tier (or to eager, above the largest tier).
  More tiers make the *average* cliff smaller but every tier boundary
  is still a boundary.
- **What it doesn't touch:** the underlying kernel is unchanged — this
  is purely a capture/dispatch-side mitigation, same engineering
  category (and similar risk/cost profile) as the single-bound
  proposal above, just repeated N times.

**Verdict:** viable as an incremental improvement over a single bound
(fewer requests hit the eager cliff, at proportionally more VRAM/capture
cost), but it is not a real fix for the long-context case — it only
narrows the gap between "capture-time worst case" and "real cost,"
never closes it. Not recommended as the target design; could be a
pragmatic interim step if the masked-early-exit kernel rewrite (below)
turns out to be infeasible or is deprioritized.

### Alternative — masked early-exit in the kernel (the actual fix for long context)

Instead of bucketing `Sk_pad` at all, make the attention kernel's
per-request work data-driven at replay time, the same way
`kernel_paged_attention_2d` (the `ROCM_ATTN`/Triton fallback backend,
analyzed early in this investigation) already works: grid *shape*
(batch dim, head dim) is fixed at capture — legal to bake into a
graph — but the **inner K-loop trip count** is `cdiv(real_seq_len,
tile_size)`, read from the live `seq_lens` tensor at replay time, not
from a capture-time constant. Varying a data-dependent loop bound
inside a captured kernel is ordinary, legal CUDA/HIP; what CUDA graphs
freeze is launch *geometry* (grid/block dims), not data values a
kernel reads and branches on.

Applied to `GFX906_FA`: keep the gather buffer allocated at
`Sk_pad = max_model_len` width as today (one-time VRAM cost, unchanged
from the current design, paid once regardless of live traffic), but
change the gather and attention kernels so the attention loop iterates
`cdiv(real_seq_len, tile_size)` times instead of `cdiv(Sk_pad,
tile_size)` — using the live `seq_lens` tensor already passed into the
kernel today. This gets **true `O(real_seq_len)` cost at every context
length, in a single capture, no tiers, no fallback, no batch-wide
cliff** — because the grid no longer needs to depend on `Sk` at all
once the K-loop is internal and data-driven. It also fully fixes "even
131k configs overpay for short contexts," for every context length,
not just below some bound — the qualitative win the original proposal
claimed but, per the correction above, cannot actually deliver.

**A closely related, possibly cheaper path already exists in this
backend:** `forward_paged_direct`
(`gfx906_fa_paged.py`/`gfx906_fa.cpp:845`, dispatched via
`_should_use_direct_paged`) reads directly from the paged KV cache via
`block_table`/`seq_lens` with **no dense gather buffer at all** —
same category as `kernel_paged_attention_2d`, i.e. already
`seq_len`-driven rather than capacity-driven. If its cost profile is
confirmed flat/linear in real `seq_len` (not yet measured — needs the
same `Sk`-sweep gate as everything else here), routing `GFX906_FA`'s
FULL-graph decode through this path would sidestep writing a new
kernel entirely. **It is not free to enable today**: it requires the
Q8 K side-buffer (`key_cache_q8 is not None`, i.e. `GFX906_FA_LEGACY=0`),
and `GFX906_FA_LEGACY=0` was, at the time of this investigation, flagged experimental and
**failing closed** (`RuntimeError`); both no longer apply — it is the default, the refusal is gone
(2026-09-16)
(`get_cudagraph_support`, per the completed R2 review item in
`CHANGELOG.md`) —
because the Q8 side-buffer misses COW'd prefix-cache blocks and
produces corrupt output. Since prefix caching is exactly what makes
long multi-turn conversations affordable, that correctness gap would
need closing first (making the Q8 side-buffer COW-aware) before this
route is usable — real work, but likely less than a from-scratch
kernel rewrite, since the direct-paged kernel and its causal/
block-table addressing already exist and are validated for the
non-prefix-caching case.

**Cost:** real kernel-design work either way — either extending
`forward_paged_direct`'s Q8 side-buffer to stay COW-consistent, or
rewriting the dense gather-attention kernel's inner loop to be
`seq_len`-driven. Both are correctness-critical changes to vendored
HIP code (`csrc/gfx906_fa/`), not the cheap Python/config change the
bounded-capture proposal is. Needs the same house-protocol rigor as
any gfx906 kernel change: standalone kernel tests, PPL/greedy gates,
serving A/B — likely a multi-session effort, not a quick follow-up.

**Recommendation:** masked early-exit (via either route) is the
correct target — it is the only option that actually serves long
conversations fast, matching the reason `--max-model-len` capacity
work happened at all. Multi-tier capture and the single-bound proposal
are both capture/dispatch-side mitigations that cap the *worst* case
tax without removing the structural `O(capacity)` cost; they may still
be worth shipping first as a cheap, low-risk win for short-context
traffic while the kernel work is scoped, but neither should be
described as "fixing" the `max_model_len` decode tax — only as
narrowing which traffic still pays it.

## RESOLUTION UPDATE (2026-08-22) — fix shipped on branch `gfx906/fa-masked-gather`

The "masked early-exit in the kernel" route (final section above) was
implemented as the plan `plan_masked_fa.md` §2.2 design — a persistent
grid-stride fused gather+quantize kernel (`gather_paged_kv_quant_persistent`)
with a fixed capture-time grid and work bounded by the live `seq_lens`
tensor. The single-bound bounded-capture design and the multi-tier
alternative were NOT built (superseded; see the CORRECTION above).
All house-protocol gates passed, including the serving A/B (THE gate):
131k 22.4→40.9 t/s, 262k 15.9→40.9 t/s (plain-greedy harness; P0 tax
−28.8% here vs the −25% of the mtp2 S8 numbers), residual tax at P1
0.07% (noise). `GFX906_FA_PERSIST` default ON. Full record:
`DEVLOG-masked-fa.md`. Status of this doc: diagnosis final; fix lever
realized as the kernel route; roadmap N4 → RESOLVED (pending merge).

---

## Update 2026-09-16 (KVLAYOUT-1 resolved)

`GFX906_FA_LEGACY=0` (the Q8 side-buffer read path this report analyses) is now the **default**,
not an experimental opt-in: verified against 0.29's fused KV-cache layout (#51718) — PPL within
the probe's ~0.001 run-to-run spread of LEGACY=1 (0 top-20 misses) and **−15.5 % / −19.1 % ms/step**
with MTP k=3 at 64k/120k (acceptance unchanged, interleaved A/B). The fail-closed refusal and
`GFX906_FA_LEGACY_ALLOW_UNVERIFIED` are removed; `GFX906_FA_LEGACY=1` is the rollback.
See `DEVLOG-fa-legacy0-b1-decode.md`.
