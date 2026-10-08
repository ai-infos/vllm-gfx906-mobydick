# VIT-1 — gfx906 custom FA for the Qwen3.5-family ViT (image-prompt TTFT)

> Branch `gfx906/v2-bringup` · model Qwen3.8-27B-AWQ-INT4 (VL) · authed 2026-09-15 ·
> roadmap item VIT-1.

**VERDICT:** `SHIPPED` (opt-in → default ON, kill switch `GFX906_FA_VIT=0`)

**GATE:** serving, single GPU (MI50 GPU0), dense 27B VL ckpt
(`cyankiwi/Qwen3.8-27B-AWQ-INT4`), `--max-model-len 8192 --max-num-seqs 4
--gpu-memory-utilization 0.85 --no-enable-prefix-caching`, 1024×1024 PNG prompt,
TTFT (time to first streamed token), 3 fresh images per arm + a cache probe, V2
runner. Standalone kernel numbers below are **launch-regime evidence**, not the
gate.

---

## HYPOTHESIS

The VL towers run bidirectional, cache-free, ragged fp16 attention at
`head_dim = 72` (16 heads, hidden 1152), prefill-only, **on the critical path of
every image-bearing prompt**. Upstream serves it through flash-attn, which on
gfx906 is the Triton-AMD path (not MI50-tuned, and it JIT-compiles its kernels
per triton cache). If the custom `gfx906_fa` dense entry serves those shapes, both
the image-prompt TTFT and the fresh-boot cost should drop — and a prefill-shaped
call must not use the decode-era `kv_split` default.

## What was done

- **Adapter** (`vllm/gfx906_fa/gfx906_fa_mm_encoder.py`, `forward_vit`): pad
  head_dim 72 → 128 (launcher dispatches {64,128,256}), Q fp32, K quantised
  q8_0, V fp16, `mask=None` + `q_abs_offset=None` = full bidirectional.
- **Production layout fixed** (`d41ff72840`): `Qwen3VLModel.forward` does
  `hidden_states.unsqueeze(1)`, so attention sees `[seq_len, 1, hidden]` with
  `cu_seqlens[-1] == seq_len` — a **packed** stream, one entry per image (the same
  contract flash-attn's varlen wrapper asserts). The first adapter asserted
  `cu.numel() == B+1` with B=1, i.e. it would have failed on **every multi-image
  request**, and the test it came with used a `[B,S,H,D]`/B=2 layout production
  never passes. Rewritten around an explicit `(batch_row, start, length)` plan;
  equal-length neighbours group into one kernel call through zero-copy views.
- **`kv_split` override** (`5d14d96be9`): the shape-aware default is 32 for any
  `Sq >= 4` (a DECODE rule). At `Sq` in the hundreds-to-thousands the per-split
  partial buffer `[B, Sq, Hq, y, D]` fp32 grows with `Sq` and the split-combine
  dominates; only the 512 MiB transient budget accidentally saved the large-Sq
  cases. Adding an optional per-call `kv_split` to the dense binding (value 1 also
  bypasses the budget, being always safe) and passing 1 from the adapter.
- **Grouping fix**: the whole-row fast path was taken for a packed sequence that
  merely *started* at offset 0, computing attention for every other image's
  queries too (right answer — later groups overwrite — but ~2× work).

## Evidence — FOR

Gate (same boot, identical prompts per arm, `prompt_sha1` asserted equal, prefix
caching OFF so the ViT cannot be served from the KV cache; fresh image content per
rep because the **mm encoder cache** — not configurable in 0.29 — otherwise serves
repeats and skips the ViT entirely):

| image (fresh) | upstream ViT | gfx906 ViT | Δ |
|---|---|---|---|
| 1024×1024 (4096 patches) | 5.81 / 5.80 / 5.81 s | **5.15 / 5.13 / 5.14 s** | **−0.67 s (−11.5 %)** |
| 1024×1024, 256-tok probe | 6.063 s | **5.421 s** | −0.64 s (−10.6 %) |
| 512×512 | 1.71 / 1.71 s | 1.67 / 1.68 s | −0.04 s (−2.5 %) |
| 1024×1024, 2nd send (encoder-cache hit, ViT skipped) | 4.149 / 4.154 s | 4.144 / 4.147 s | — (control) |

Fresh-boot cost, `TRITON_CACHE_DIR` pointed at an empty dir, same image,
same `--no-enable-prefix-caching` config (READY = `Application startup complete`):

| arm | cold triton cache | warm triton cache |
|---|---|---|
| upstream ViT | **330 s** | 135 s |
| gfx906 ViT | **275 s** | 135 s |

⇒ the ViT's Triton share of a fresh boot is **−55 s**; the remaining ~140 s of
that 195 s penalty is *other* Triton (the log shows
`qwen_gdn_linear_attn.py:526 GDN decode kernel: triton` — the LLM's GDN path
compiles a Triton kernel too), so **VIT-1 does not make the triton dependency
droppable on its own** (VIT-1 item 4 stays open, now with a measurement).

Launch-regime evidence (standalone, `bench_vit_prod.py`, GPU0, mclk 800–1000 MHz,
25 iters back-to-back; upstream arm = production `vit_flash_attn_wrapper`):

| shape | upstream (Triton-AMD) | gfx906 | speedup |
|---|---|---|---|
| 1 × 2304 patches | 14.65 ms | 7.68 ms | 1.91× |
| 3 × 1536 (3 images) | 19.80 ms | 10.77 ms | 1.84× |
| 2304 + 576 + 1728 (ragged) | 24.83 ms | 13.24 ms | 1.88× |

`kv_split` pathology found by the same harness (before the override):

| Sq | kv_split | time |
|---|---|---|
| 576 | 32 | 4.97 ms → **0.95 ms** at 1 |
| 1728 | 32 | 41.85 ms → **4.47 ms** at 1 |
| 2304 | 1 (budget forced) | 7.05 ms → 7.06 ms |

Correctness: rel err vs SDPA 1.6–2.2e-2 (Q8-K quantisation; upstream path 4e-4),
**zero cross-image leakage** on every layout (packed ragged, packed equal, padded
per-item, single image, `cu=None`). FA suite 95 passed.

## Evidence — AGAINST

- **The swap is not bit-equivalent, and the surface form changes.** Same image,
  same prompt, 256 greedy tokens: content identical ("two shapes … a red square"),
  **wording differs** from char 156 ("On it are two square shapes. First shape: a
  red square …" vs "There are two distinct shapes. Shape 1: A red square …").
  Mean logprob/token −0.13484 (upstream) vs −0.12220 (ours), i.e. ours is slightly
  *more* confident on its own path. First-token top-20: top-1 `The` identical
  (−0.015 both), 18/20 tokens shared, **max tail |Δlogprob| 0.66** (rank 4+:
  `-`, `Let`, `Thinking`). So the Q8-K feature perturbation is visible in the tail
  of the distribution, not in the top.
- The first serving A/B read **"TTFT unchanged"** (1.498 → 1.511 s @512,
  4.108 → 4.126 s @1024). That run was measuring the **mm encoder cache**: a
  repeated image is served from it, so the ViT never ran (the cache probe above
  reproduces the artifact: 4.15 s vs 5.81 s fresh). Both arms reported identical
  TTFT *because neither was running a ViT*.

## Why the win is smaller than the kernel ratio

1.91× on the ViT attention buys −11.5 % of TTFT at 1024×1024, not 1.9×: the ViT is
~29 % of that TTFT (measured as fresh 5.81 s vs encoder-cached 4.15 s) and the
LLM prefill + decode of the merged image tokens dominate the rest. At 512×512 the
ViT share is small enough that the win is 2.5 %.

## Interactions / superseded-by

- Supersedes the "VIT-1 step 1" opt-in entry (adapter validated against a
  non-production layout, no serving gate).
- Reinforces: **a serving A/B must assert that the work under test actually runs**
  — here via the encoder-cache probe; the `prompt_sha1` guard (AGENTS.md) catches
  prompt drift but not a skipped computation.
- The `kv_split` finding is the **third** decode-tuned default that misbehaves on
  prefill-shaped input; see also the MTP-1b-0 budget note in `gfx906_fa.cpp`.
- **The fall-through is now loud.** `ROCmPlatform.get_vit_attn_backend` used to
  drop to the upstream path silently, which is how a model quietly loses the
  MI50-tuned ViT kernel; it now logs `gfx906 CUSTOM ViT attention UNAVAILABLE
  (<reason>)…` at WARNING, with the reason (dtype, head size, or the kill switch)
  from `vit_unsupported_reason()`. Worth knowing for that path: `on_cdna()` is a
  substring test (`"gfx9" in arch`) that is **TRUE on gfx906**, so the usual
  upstream pick is the "CDNA" flash-attn branch — and if `flash_attn` is not
  installed the chain ends at unfused `TORCH_SDPA`.

## Refrigerated residue

- **fp16-K kernel variant**: the ViT has fp16 K/V natively, yet the only
  instantiated kernel is `fattn-q8` (K quantised to q8_0 then dequantised
  in-kernel). A fp16-K instantiation would remove both the ~2e-2 error (→ the
  upstream path's 4e-4 class) and the quantise pass. Kernel work, not adapter
  work; not attempted.
- **head_dim 96 instantiation — quantified, not yet attempted.** Cost tracks the
  **padded** head dim, not the useful one: with the D=128 instantiation,
  head_size 72 / 80 / 96 / 112 / 128 all cost **7.68–8.00 ms** (96–100 % of each
  other, `bench_vit_dscale.py`, H=16 S=2304, mclk 800 MHz) while the D=64
  instantiation is **3.205 ms**, i.e. 0.050 ms/dim at 64 vs 0.063 ms/dim at 128.
  So the 56 zero dims of the 72 → 128 pad are pure cost, and a 96-wide instance
  would remove exactly 25 % of the head-dim arithmetic: at the measured per-dim
  costs that is **~5.5–6.0 ms vs 7.78 ms → −22 … −29 % on the ViT attention**,
  worth **≈ −0.25 s of TTFT at 1024×1024 (−5 %)** and −0.7 % at 512×512 (the ViT
  is ~19 % of the custom path's TTFT there).
  *Accuracy is unaffected*: q8_0 blocks are 32-wide, so D=96's three blocks are
  the first three of today's D=128 (dims 64–95 = 8 real + 24 zeros either way) —
  only the redundant all-zero fourth block disappears. Measured rel err is
  padding-independent: 0.0156 / 0.0172 / 0.0199 / 0.0182 at head dim 64 / 72 / 96 /
  128.
  *Risks*: (a) the `(DKQ=96, DV=96)` tile-config entry (nthreads, occupancy,
  nbatch_fa, nbatch_K) is new and untuned — the ±25 % per-dim spread between the
  64 and 128 entries shows config quality dominates, and a bad entry can end up
  **slower** than the padded 128 path; (b) build size/time grows per
  (DKQ, DV, ncols) instance, so instantiate only the ncols1 the ViT picks (64 for
  Sq > 32); (c) the launcher's `switch (head_dim)` is shared — a mis-keyed table
  entry could shadow an existing case, and any other caller asking for 96 (the
  paged paths derive 256) moves onto the new kernel; (d) `_pad_head_dim` must learn
  96, which likewise moves any caller with head_size 80–96 onto it.
- Standalone `Sq=1536/1728` single-row calls stay 4–9× slower than the
  budget-forced large-`Sq` case *without* the override — any future caller of the
  dense entry with prefill-shaped input needs the same `kv_split=1`.

## Search keys

`HYPOTHESIS:` ViT-custom-FA; `VERDICT:` SHIPPED; `GATE:` image-prompt TTFT;
`kv_split` prefill pathology; mm-encoder-cache artifact.
