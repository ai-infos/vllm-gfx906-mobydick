# FA coverage — where gfx906 does *not* get our FA, and why

## 2026-09-16 — enumeration: the fallback map, ranked

**VERDICT:** recon complete, four actionable classes identified · **GATE:** none yet (this entry is a
map; the gate belongs to whatever we change — FA suite plus a PPL/serving check for a predicate fix).

## Why

Twice in one day a model silently lost the MI50-tuned FA and nobody was told why: the DFlash2 drafter
(`non-causal attention not supported` -> ROCM_ATTN, which cannot even be CUDA-graph captured) and
Gemma-4 (`TRITON_ATTN`). The reason strings existed the whole time but were logged at `debug_once`.
With them promoted to INFO (`b06a09e978`), the coverage question becomes enumerable off-GPU — the
selector is a pure function of the attention config — so this entry answers it for the synthetic
space and for every checkpoint in the local zoo, including TP widths we cannot run locally (1/2/4/8)
and vision towers (a separate selector).

Tool: `tools/fa_coverage.py` (CPU-only, no model loads, no GPU lottery). It drives
`RocmPlatform.get_valid_backends` for text attention and `vit_unsupported_reason` for the ViT, so it
reports exactly what the engine would choose and why not CUSTOM.

## What the enumeration found

**Synthetic text matrix** (192 configs over head size x block size x sliding x non-causal x sinks x
attention type; 174 do not get CUSTOM):

| reason | rows | real-world class | our exposure |
|---|---|---|---|
| `attention sinks not supported` | 72 | GPT-OSS-style sink models | none in the zoo yet; real models exist |
| `head_size not supported` | 61 | text models with head_dim not in {64,128,256} — e.g. 96/112 | hypothetical as text, see the asymmetry below |
| `attention type encoder not supported` | 36 | encoder / encoder-decoder attention | we serve decoder-only |
| `non-causal attention not supported` | 18 | DFlash2 drafters, lookahead/bidirectional spec decode | DFlash2 parked; FA-NONCAUSAL design exists |

**Vision tower** (30 head-dim x dtype combinations; 12 fall back): `torch.bfloat16` (10) — the Q8 ViT
kernel is fp16/fp32 only — and `head_size > 256` (2). Everything else, including 72/80/96/112, is
served because the ViT path **pads** to the next instantiated kernel dim.

**Real checkpoints** (78 config reads across `/local/models`, `/data/models` and both HF caches, at
tp=1/2/4/8): 13 reads hit a non-CUSTOM config, all small text models with head_dim 40 (4 reads ->
TRITON_ATTN) or head_dim 32 (9 reads -> ROCM_ATTN). Those are the BERT/BGE-shaped embedding and
reranker checkpoints, which we run under `llama-server`, not vLLM — so the practical impact today is
nil, but they are the honest map of what a vLLM bring-up of them would cost.

## The asymmetry worth acting on

`supports_head_size` on the text backend accepts **exactly** `(64, 128, 256)`; the ViT path accepts
anything `<= 256` and pads via `_pad_head_dim`. So a *vision* tower with head_dim 96 is served by our
kernel (at the padded cost — that is VIT-2's subject), while a *text* model with head_dim 96 is
rejected outright and runs ROCM_ATTN. Mirroring the ViT's padding rule in the text path turns a
silent ~3-10x step cost into a bounded arithmetic overhead, and the paging is ours to declare
(`get_kv_cache_shape`), so the padded layout is expressible.

## Ranked next steps

1. **The guard** (durable, cheap): make a non-CUSTOM selection for a gfx906 text attention layer a
   loud, once-per-engine *warning* naming the reason, plus an opt-in fail-closed switch. This is what
   turns the next silent fallback into a log line — it is the piece that pays forward on every future
   model onboarding.
2. **Text head-dim padding** (mirror `_pad_head_dim`): removes the whole `head_size not supported`
   class. Gate: FA suite + PPL on a padded dim + a serving A/B.
3. **Sinks** (72 rows, the largest class): needs a kernel feature (an extra learned logit per head),
   so it is real work — worth it only when a sink model (GPT-OSS-style) is actually wanted.
4. **Non-causal** (18 rows): already designed and *experimentally refuted as an acceptance fix* for
   DFlash2 (`DEVLOG-fa-noncausal.md`); implement only if DFlash2 is revived for performance.
5. **Encoder attention / enc-dec** and **bf16 ViT**: leave documented; both are out of our serving
   envelope.

## Reproduce

```
.venv/bin/python tools/fa_coverage.py                      # matrix + every local checkpoint
.venv/bin/python tools/fa_coverage.py --skip-matrix        # checkpoints only
.venv/bin/python tools/fa_coverage.py --models /data/models --tp 1,2,4,8
```

## 2026-09-16 (step 1 of the plan) — the guard is in

**VERDICT:** ADOPTED (the durable piece of this topic) · **GATE:** FA suite 97 passed with the guard
live on the selection path, plus 4 GPU-free unit tests.

A non-CUSTOM pick for a gfx906 attention layer is now a loud, once-per-engine warning naming the
reason (`_guard_gfx906_fa_fallback` in `platforms/rocm.py`, called from the rejection branch that now
also carries the reasons). `VLLM_GFX906_FA_STRICT=1` turns it into a `RuntimeError`, so a deployment
that must not lose the tuned kernel fails closed instead of degrading quietly.

It fires only when CUSTOM was *actually rejected* (so no false alarms when some other backend was
rejected for an unrelated reason, e.g. a cache-dtype mismatch), and only on gfx906. The unit tests
cover exactly those four cases: warns when CUSTOM is rejected, raises under strict, silent when
CUSTOM was never a candidate, silent off gfx906. Both of today's silent losses — the DFlash2 drafter
and Gemma-4 — would have been a one-line warning with this in place.

Next per the rank: mirror the ViT's `_pad_head_dim` in the text path to delete the
`head_size not supported` class (61 synthetic rows), then reconsider sinks only if a sink model is
wanted.

## 2026-09-16 (step 2) — text head-dim padding: helpers landed, the gap pinned

**VERDICT:** design landed, implementation next (behind `GFX906_FA_PAD`) · **GATE:** unit tests only
(13 passed) — no local decoder LM has head_dim outside {64,128,256}, so the real-model gate has to
wait for one to be onboarded.

The ViT path already pads (any head dim up to 256 -> the next instantiated kernel dim) and its
docstring carries the exactness argument: padded Q dims add 0 to the QK dot, padded K dims quantise
to zero q8_0 blocks, padded V dims add 0 to P*V, and the padding is inside the head dim, so the
softmax denominator is unchanged. The text path refuses those dims outright instead, and falls back
to ROCM_ATTN or TRITON_ATTN.

This session adds the mirror of that rule as **helpers only** (`_pad_head_dim`,
`_padded_head_size`, plus the `GFX906_FA_PAD` kill switch) and a test that pins both the map
(32->64, 72/80/96/112->128, 160->256, 288->None) and the current restriction, so the flip has an
assertion to update. Nothing calls them yet: a padded dim also needs the KV-cache layout to carry the
pad, and wiring the predicate first would feed unpadded tensors to the kernels.

Edit list for the implementation (measured, not estimated):

- `get_kv_cache_shape` must return the **padded** dim — the row width is `head_size` today, and the
  "identical to TritonAttentionBackend" note only holds for unpadded dims;
- `do_kv_cache_update` and `forward` split the fused K||V content axis by `head_size`; both need the
  padded dim, and the write path must **zero the pad on every store** (the cache is allocated once,
  so stale bytes there would quantise to a non-zero q8_0 block and corrupt K);
- `forward_paged` must receive the padded dim and a zero-padded Q, with the output sliced back;
- the Q8 side view derives from the cache tensor, so it follows the new shape automatically;
- `supports_head_size` accepts pad-able dims only once all of the above is in.

Tests: `tests/kernels/attention/test_gfx906_head_dim_pad.py` (13 cases, no GPU needed).

## 2026-09-16 (blind spots closed) — forced backends and the inventory's own gaps

**VERDICT:** both fixes in · **GATE:** FA suite 97 passed (the rocm.py selection path), 21 GPU-free
unit tests, and the tool re-run.

Asked why Gemma-4 and Muse-Glimmer were missing from the inventory; the answer was three different
things, only one of which was a tool defect:

- **Muse-Glimmer was correctly absent.** It *is* read (head_dim 128, sliding, Hq 32 / Hkv 2) and the
  selector picks CUSTOM, matching its dev log's shipped all-CUSTOM arm (**27.90 vs 17.54 t/s, 1.59x**).
  The ROCM_ATTN line quoted earlier was the *pre-change baseline* row of that A/B table.
- **Gemma-4's checkpoint is not on this box at all**: its HF cache entry is a `refs`-only stub with no
  `snapshots/`, and there is no `config.json` anywhere under `/data/models`, `/local/models` or
  `/local/cache`. Nothing to enumerate.
- **Its fallback is forced by model code**, not by the selector: `Gemma4Config.verify_and_update_config`
  detects heterogeneous head dims (256/512), finds FA4 unavailable and picks `TRITON_ATTN` outright, so
  `get_valid_backends` is never consulted. That is a blind spot in the *guard* too, not just the tool:
  the guard fires on selector rejections, and a forced backend never reaches one.

Fixes:

1. `_guard_gfx906_forced_backend` (next to the fallback guard, called from the explicit-selection
   branch of `get_attn_backend_cls`): warns once — or raises under `VLLM_GFX906_FA_STRICT=1` — whenever
   a non-CUSTOM backend is forced on gfx906, naming the fact that it came from `--attention-backend`
   or the model's own config. Four more GPU-free unit tests.
2. `tools/fa_coverage.py`: reports `refs`-only repo stubs (21 at the time of writing, which is exactly
   how Gemma-4 disappeared), recurses past depth 1 for non-hub roots (93 checkpoint config reads now
   vs 78 before), evaluates **per `layer_types` group** so hybrid models are checked group by group,
   and includes a >256 head-size class (288/512) that prints the distinction between "gate can be
   widened" and "needs a new kernel instance" — the class Gemma-4's 512 is in.

Ranking consequence: step 2 (padding up to 256) has **no serving-relevant beneficiary** in the zoo
today, while Gemma-4's class needs a 512-head instance plus heterogeneous dispatch plus a config
change. Keep step 2 as insurance; do it after anything with a real model behind it.

### Step 2 implementation notes (measured while starting it, 2026-09-16)

Reading the plumbing before touching it produced a precise edit list, and one prerequisite that
landed first: **`GFX906_FA_PAD` is opt-in (default 0)**. Until the KV layout carries the pad, serving
a non-instantiated dim would feed unpadded tensors to the kernels — silent garbage rather than a
fallback — so the default must stay off until a real model validates the path. The tests now pin that
(unset -> only instantiated dims; `=1` -> the pad map; 14 cases).

Where the change actually goes (all in `gfx906_fa_backend.py`):

- `get_kv_cache_shape` (static): the row width is `head_size`; it must return the padded dim. The
  "identical to TritonAttentionBackend, so backends can be swapped without re-allocating" note only
  holds for unpadded dims.
- `Gfx906FAImpl.__init__`: keep `self.head_size` as the *real* dim and add `self.padded_head_size`;
  the fused K||V split (`kv_cache.transpose(1, 2).split(self.head_size, dim=-1)`) appears twice —
  `do_kv_cache_update` (write) and `forward` (read) — and both must split by the padded dim.
- `do_kv_cache_update`: `triton_reshape_and_cache_flash` writes only the first D channels of a
  padded row, so the pad must be **zeroed explicitly**. Cheap approach: zero the pad slices of the
  cache once per cache tensor (identity on the underlying storage, the same trick
  `_ensure_q8_sidebuffer` already uses) — after that nothing writes those bytes, and a zero pad is
  exactly what keeps the math exact.
- the Q8 side view needs no change in principle: its row ((D/32)*34 bytes) fits the fp16 K row
  (2D bytes) — 136 <= 256 at D=128 — and it derives from the cache tensor, so it follows the padded
  shape. Worth an assertion that the fit still holds for a padded dim.
- `forward`/`forward_paged`: pass the padded dim to the op, zero-pad the query, and slice the output
  head dim back to `head_size` (the reshape/slice happens where the fp32 result is copied into
  `output`, alongside the existing `q_pad` machinery).
- `supports_head_size` returns pad-able only once all of the above is in.

Verification path stays as recorded: unit tests against a torch reference at 72/80/96/112 (no local
decoder LM has such a dim, so a real-model gate waits for one to be onboarded — the guard will say so
when that happens).

### Step 2 implementation (2026-09-16) — the padded write/read path is in, behind the opt-in gate

**VERDICT:** implemented, unreachable by default, new-path test outstanding · **GATE:** the FA suite
(97 passed) proves no regression at D=64/128/256, since GFX906_FA_PAD defaults to 0 and instantiated
dims return unchanged. The *new* path's own verification - a D=96 case against a torch reference - is
the remaining piece, and the default flip stays blocked on it plus a real model.

What landed in `gfx906_fa_backend.py`:

- `get_kv_cache_shape` returns the padded row width (the "identical to
  TritonAttentionBackend" property now holds only for instantiated dims);
- `Gfx906FAImpl.__init__` keeps `head_size` as the real dim and adds `padded_head_size` /
  `_head_pad`;
- `_pad_last_dim` (zero-pad [.., D]) and `_zero_cache_pad` (zero a padded cache's pad channels once
  per cache tensor, keyed on storage identity - the trick `_ensure_q8_sidebuffer` already uses,
  because `triton_reshape_and_cache_flash` only writes the first D channels and a non-zero K pad
  would quantise to a non-zero q8_0 block);
- `do_kv_cache_update` splits the fused content axis by the padded dim, zeroes the pad once and
  writes zero-padded K/V (the Q8 write inherits the padded rows, so its block count matches);
- `forward` splits by the padded dim, zero-pads the query, grows the q_pad/gather buffers at the
  padded width, and slices the fp32 result back to the real head dim before the output copy;
- `supports_head_size` returns pad-able, gated by `GFX906_FA_PAD` inside `_padded_head_size`.

Cost noted honestly: the padding is per-write `torch.zeros` (opt-in path only), and the KV cache
grows by the pad ratio (96 -> 128 is +33 % of K and V bytes).

### Step 2 verified against a torch reference (2026-09-16)

**VERDICT:** the padded path is verified · **GATE:** `test_padded_head_dim_matches_torch_ref`
(D=72/80/96/112) plus the full FA suite at 101 passed.

The test drives the *impl* rather than the raw op, so the padding runs end to end: the cache is the
backend-level fused layout `[N, Hkv, BLOCK, 2*D]` (vLLM fuses the spec's logical head dim into the
row), K/V go in through `do_kv_cache_update`, the query is padded on the way in and the result sliced
on the way out. Two details make it a real test rather than a smoke:

- the cache is **pre-filled with garbage**, so an unzeroed pad shows up as a mismatch (the test also
  asserts the pad channels are exactly zero after the write);
- it compares against the existing `_windowed_ref` torch reference at the *real* head dim, which is
  the right expectation precisely because a zero pad contributes nothing.

The default is still 0, and the guard now makes the opt-in discoverable: when CUSTOM is rejected for
`head_size` on a dim that `_pad_head_dim` can reach, the warning says so and names `GFX906_FA_PAD=1`
(Phi-3-mini / Phi-3.5-mini are 96, Phi-2 is 80, SigLIP-style ViTs are 72 — the class this serves).

### Step 2, real-model gate: Phi-3-mini (head_dim 96) — WIP, one allocator interaction left

**VERDICT:** OPEN (diagnosed, not fixed) · **GATE:** `GFX906_FA_PAD=1` vs `0` on Phi-3-mini-4k-instruct,
same model and prompts; the fallback arm is green (PPL **15.0090**, 421 tokens, 0 top-20 misses) and
is the reference the padded arm must match.

What the real model showed that the unit test could not (the unit test hand-builds the cache, so it
never exercises vLLM's allocation):

- with padding opted in, the gate *opens* — Phi-3's head_dim 96 is accepted and the log says
  `Overriding with CUSTOM`;
- it then dies in the Q8 gather with `bytes_per_row must equal (D/32)*34, got 136 vs expected N`,
  where N moved from 68 to 102 as the fix below landed — i.e. the failure is a **width disagreement
  between the two views the kernel receives**;
- the shapes (env-gated `GFX906_FA_DEBUG_SHAPES`) are the smoking gun:
  `kv_cache=(2, 32, 16, 224)`, `head=96 padded=128`, `K=(2, 16, 32, 128)`, `V=(2, 16, 32, 96)`.
  224 is **neither** 2*96 (real) nor 2*128 (padded) — it is 128 + 96, i.e. the allocator built a
  2*112 row.

Mechanism: vLLM sizes a page from the **spec** (`page_size_bytes` = block_size * num_kv_heads * 2 *
head_size * dtype_size) and builds the tensor from the backend's `get_kv_cache_shape`. Padding the
spec's `head_size` (via `customize_spec`, which the sliding-window branch already calls and the
full-attention branch now does too) makes those two disagree, and the reconciliation yields a third
row width. So the padded dim has to be expressed through vLLM's **page-size** mechanism that the spec
already has (`page_size_padded`, used by the sliding branch), not by widening the shape and the spec
independently.

Landing here as WIP on `gfx906/fa-cover-1`: the `customize_spec` route for full attention (a 3-line
change in `attention.py` mirroring the sliding branch, plus the backend override), the env-gated
shape debug prints (`GFX906_FA_DEBUG_SHAPES`, kept because this class of bug is otherwise invisible),
and a test pinning that the spec widens only when opted in. Defaults are unchanged: `GFX906_FA_PAD`
is still 0, so nothing serves a padded dim by accident.

### Step 2 RESOLVED — Phi-3-mini gate green, padding on by default (2026-09-16)

**VERDICT:** ADOPTED · **GATE:** Phi-3-mini-4k-instruct (head_dim 96), two arms, same model and
prompts: IFT logprobs (the templated comparability tool) + PPL + an interleaved speed A/B.

| arm | backend | PPL | top-20 misses | tokens/s (A-B-A, 4 samples) |
|---|---|---|---|---|
| padded (`GFX906_FA_PAD=1`) | CUSTOM | 15.0254 | 0 | **36.41 / 36.18** |
| fallback (`=0`) | ROCM_ATTN | 15.0090 | 0 | 28.62 |

- **Numerical:** PPL within 0.11 %, and the IFT gate gives **identical top-5 tokens in the same
  order** (3681/450/1459/25343/3579 and 315/7521/822/2266/18585) with logprobs within 0.06 — two
  different attention implementations agreeing, which is the gate this tool was built for.
- **Speed:** **+27 %** decode (36.41 vs 28.62 t/s), order-controlled by the A-B-A repeat (36.18,
  within 0.6 % of the first padded run).
- **Regression:** FA suite 101 passed.

**What the real model caught that no unit test could.** vLLM sizes a KV *page* from the **spec**
(`block_size * num_kv_heads * (head_size + head_size_v) * dtype_size`) and builds the *tensor* from
the backend's `get_kv_cache_shape`. Widening only `head_size` in the spec gave a fused row of
128 + 96 = **224**, the allocator built exactly that third width, and the Q8 gather rejected it
(`bytes_per_row must equal (D/32)*34, got 136 vs expected 102`). The fixes:

1. the **full-attention** branch of `Attention.get_kv_cache_spec` now routes through
   `attn_backend.customize_spec` (the sliding-window branch already did — a 3-line change);
2. our override widens **both** `head_size` and `head_size_v`, since the fused row is their sum.

Two tests now pin that: the spec must widen both halves, and the spec's page size must equal the row
width our `get_kv_cache_shape` declares (the invariant whose absence caused this).

**Broader reach than expected:** `_pad_head_dim(32)` is 64 and `_pad_head_dim(160)` is 256, so the
padding serves **any** head dim up to 256 — including the small encoder-shaped checkpoints the
inventory flagged (they still fall back on `attn_type`, since this backend is DECODER-only).

**Cost, stated plainly:** KV bytes grow with the pad ratio (96 -> 128 is +33 % of K and V; 72 -> 128
is +78 %). `GFX906_FA_PAD=0` restores the old fallback behaviour, and the guard still names the knob
when a dim cannot be padded at all (> 256, e.g. Gemma-4's 512).

Kept for the next such bug: `GFX906_FA_DEBUG_SHAPES=1` prints the spec in/out and the first few
kernel-boundary shapes (this bug was invisible without it).
