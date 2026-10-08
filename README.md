## Mini Install Guide for GFX906

**Branch `gfx906/v0.30.0.x`** integrates the pinned
[KIntegrated v0.30.0 reference](https://github.com/KIntegrated/vllm-gfx906-mobydick/tree/gfx906/v0.30.0)
and preserves its gfx906 optimizations. See the
[upgrade audit](docs/gfx906/UPGRADE-0.30.0.x.md) for inputs, fixes, validation,
and unresolved runtime checks.

- **Primary build target:** Linux x86_64, Python 3.12, gfx906-native ROCm 7.14,
  PyTorch 2.13.0, and source-built Triton 3.8.0. Immutable base-image and source
  revisions are in [docker/gfx906-build.env](docker/gfx906-build.env).
- **Build the selected checkout:** `bash build_and_push_docker.sh`. This creates
  `aiinfos/vllm-gfx906-mobydick:v0.30.0.x-rocm7.14-pytorch2.13.0` locally;
  publishing requires an explicit `--push`. No image from this branch has been
  published or GPU-validated during this upgrade.
- **Custom Q8 attention stays enabled**, with the reference's rollback switches.
  Q8 Q/K and fp16 intermediates change numerical results; retaining the defaults
  does not establish device correctness or model quality here.
- **Attribution:** benchmark numbers and host-specific serving recipes below
  are historical results from the KIntegrated reference, chiefly Kevin Read's
  MI50/MI60 environment. They were not rerun for this branch. Its
  [release review](docs/gfx906/RELEASE-0.30.0-final.md) pins V1 for the house bench
  and documents the fused-align V2 graph-capture fault.

- **Serving defaults:** MTP **k=3** spec decode plus a cudagraph capture ladder of
  multiples of `k+1` up to `max_num_seqs × (k+1)`; an undersized ladder silently
  collapses to B=1. Recipe + rule: `running.md` §1 and [`AGENTS.md`](AGENTS.md).
- **Operational caveats:** gfx906 hosts accumulate GPU wedges and a degraded state
  that only a reboot clears. Run the canary probe before trusting spec-decode
  numbers and follow the two-strike burst rule — [`docs/gfx906/degradation.md`](docs/gfx906/degradation.md).

## Fork heritage

This repository is the gfx906 vLLM port
[**ai-infos/vllm-gfx906-mobydick**](https://github.com/ai-infos/vllm-gfx906-mobydick),
based on [**nlzy/vllm-gfx906**](https://github.com/nlzy/vllm-gfx906), the
original gfx906 port of vLLM. The custom Q8 FlashAttention kernels below are
vendored from
[**cassettesgoboom/gfx906-fa-vllm**](https://github.com/cassettesgoboom/gfx906-fa-vllm).
See [`docs/gfx906/`](docs/gfx906/) for the full optimization record.

### Ported optimizations (external sources)

- **SYV-4 — sort-free small-k top-k/top-p sampler** (`vllm/v1/sample/ops/topk_topp_sampler.py`,
  default ON, opt-out `VLLM_GFX906_SORT_FREE_SMALL_K=0`): technique ported from
  [**syv-ai/qwen38-27b-rtx3090**](https://github.com/syv-ai/qwen38-27b-rtx3090)
  (`docs/optimizations.md`, fetched 2026-09-02); implementation is ours for the
  gfx906 ROCm path. +0.6–3.4% serving t/s on sampling workloads (TP=1 dense 27B
  A/B, 2026-09-03). Recon: [`docs/gfx906/RECON-syv-qwen38-27b-rtx3090.md`](docs/gfx906/RECON-syv-qwen38-27b-rtx3090.md).

- **CAT-1 — MTP draft-vocab shortlist** (`vllm/model_executor/models/qwen3_5_mtp.py`,
  `tools/build_draft_vocab.py`; **default OFF**, enabled per-deploy via a work-dir + env — see
  [`docs/gfx906/DEVLOG-draft-vocab.md`](docs/gfx906/DEVLOG-draft-vocab.md)): technique ported from
  [**syv-ai/qwen38-27b-rtx3090**](https://github.com/syv-ai/qwen38-27b-rtx3090) (draft-LM-head
  vocabulary reduction for the MTP drafter; arXiv 2506.22694 VocabTrim is the training-free
  reference), adapted to this fork's bf16/TP=2 path — implementation ours. Final A/B on our own
  traffic (Qwen3.8-27B, TP=2, k=2, list = **35,251 ids** = every token observed in the corpus plus
  the added-token control family): **−2.52 ms/step [−2.91, −1.94]** ⇒ **+4.79 % t/s mean / +5.90 %
  median** (9/11 prompts positive), acceptance **no detectable penalty** (mean −2.07 pp,
  Mann-Whitney z = −0.83), raw-continuation coverage **97.7–98.0 % → 100 %**. The list is **paired to its
  draft head by a manifest** (`cat1_manifest.json`: ids_sha1 + head sha1 +
  snapshot) that the serving loader verifies at startup — an equal-length ids
  swap or a mismatched head fails loudly instead of serving wrong draft logits
  (`tools/draft_vocab_manifest.py`). The corpus-provenance block is written by
  `count` when a list is rebuilt; the shipped 35,251-id list predates that field
  (list built 08:42, manifest support landed 10:35 same day), so it carries the
  ids↔head pairing and its snapshot but no provenance block — a fresh build
  records the corpus. Building it correctly requires a **raw-continuation** capture —
  parsed logs can never contain the markup the model emits, which is what an earlier parsed-log list
  silently missed (`docs/gfx906/CAT1-corpus-build.md`). Still **not enabled by default**
  (per-workload list).

## Custom FlashAttention backend (gfx906 FA, `CUSTOM`)

This fork vendors a custom Q8 FlashAttention attention backend for gfx906
(`AttentionBackendEnum.CUSTOM`, built from
`https://github.com/cassettesgoboom/gfx906-fa-vllm`) and makes it the **default**
for attention on gfx906 (prefill and decode). No `--attention-backend` flag is needed;
`CUSTOM` is automatically selected and the extension ships inside this wheel.

### What it accelerates

The vendor kernels originally target **prefill** on **long contexts** of
**full-attention models**; on *any*-attention hybrids (e.g. Qwen3.5, few
full-attention layers) the prefill gain is small. This fork adds the decode
path (B=1 parallelism via GQA head-packing + KV split, fused
gather-and-quantize, native BSHD output), making `CUSTOM` the default for
**decode** as well: 18.9 → 25.6 t/s serving on dense Qwen3.5-27B, and the
MoE flagship at 66.1 t/s single-request (67.4 record) / 193 t/s concurrent
(N=8, 191.0 record) — final-build restamps 2026-08-24. **Basis:** these are
*historical-bench* numbers (prefix caching at vLLM's default ON, so samples 2-4
reuse the prompt prefix); the harness default has been prefix caching OFF since
2026-08-27 — same build, cold basis: MoE 58.4 / dense 16.3 t/s (2026-09-13,
mclk-verified). See the note under the model table.
See [`docs/gfx906/`](docs/gfx906/) for the full change inventory, numbers,
and bench recipes.

### Vision-tower (ViT) attention — custom FA by default on 0.29.0

The Qwen3.5/3.8 VL **vision tower** (bidirectional, cache-free, ragged fp16
attention at head_dim 72, prefill-only, 27 layers, on the critical path of every
image-bearing prompt) now also runs on the custom FA. Upstream served it through
flash-attn's Triton-AMD path, which JIT-compiles its kernels per Triton cache.
Measured on 0.29.0 (serving, a fresh image every rep so the mm encoder cache
cannot skip the ViT, `--no-enable-prefix-caching`, identical prompts with
`prompt_sha1` asserted, 3 reps/arm): image-prompt **TTFT 5.81 → 5.14 s @1024×1024
(−11.5 %)**, 1.71 → 1.67 s @512×512, and **−55 s of fresh-boot time** (330 → 275 s
with an empty `TRITON_CACHE_DIR`). That −55 s is the ViT's own Triton JIT; the rest
of a cold boot's Triton cost is the LLM's GDN kernel, so this does **not** make the
triton dependency droppable.

The swap is **not bit-equivalent**: the custom path quantises K to q8_0 (rel err
~2e-2 vs SDPA; flash-attn is 4e-4), so the image-conditioned distribution moves in
the tail (top-1 preserved, max |Δlogprob| 0.66 at rank 4+, mean logprob
+0.0126/token) and a 256-token greedy description keeps its content but not its
wording. If you want the upstream path back:

```bash
GFX906_FA_VIT=0        # kill switch — upstream flash-attn ViT path outright
GFX906_FA_VIT_AUTO=0   # opt out of auto-selection only (an explicit
                       # --mm-encoder-attn-backend custom still selects CUSTOM)
```

Unsupported head dims/dtypes (bf16, head_size > 256, …) fall back to the upstream
backend automatically and now log a WARNING that names the reason. Full record:
[`docs/gfx906/DEVLOG-vit1.md`](docs/gfx906/DEVLOG-vit1.md).

### Model support and performance on gfx906 (single MI50/MI60)

| model | status | decode t/s |
|---|---|---|
| Qwen3.5-35B-A3B-AWQ (MoE) | flagship, fully optimized | **66.1** (restamp; 67.4 record; ~2140 t/s prefill) |
| ↳ N=8 concurrent decode | W4 (`VLLM_GFX906_SKINNY_M16=1`) | **193** (restamp; 191.0 record; +14.5 % vs 166.9) |
| ↳ with MTP k=2 speculative decoding | recommended spec config | **88.6** (restamp; 89.9 record; 1.16× vs 76.7 greedy) |
| Qwen3.5-27B-AWQ (dense) | optimized | **25.6** |
| ↳ with MTP k=2 speculative decoding | recommended spec config | **39.4** (1.41×) |
| Gemma-4-26B-A4B-it-AWQ-4bit | optimized | **67.8** |
| Qwen3.8-27B-AWQ-INT4 (dense) | fully functional (TP=1 + TP=2) | **59.2** (MTP k=2, TP=2, 2k ctx; 2026-08-24 final) |
| ↳ MTP k=2 context curve (TP=2) | **kv_split fix 2026-09-03 — MTP ≥ greedy at 64k+** | 8k/32k: 44.9/25.2 (2026-08-24); **64k/96k/120k: 37.95/29.88/25.70** (post-fix; greedy 18.86/14.80/12.74) |
| ↳ **post-0.30.0 re-measure (TP=2, 2026-09-24)** | **MTP k=3 (current default depth) + `GFX906_FA_LEGACY=0` KV read** | **2k 79.1 · 64k 42.0 · 120k 40.7 t/s** (filler, prefix caching OFF, bt4096, max-seqs 4, util 0.82, capture `[4,8,12,16]`; TTFT 4.2/185.3/442.2 s; PPL 10.5472). Standard `vllm bench serve` random 2048→256: **52.9 t/s** @ TPOT 18.90 ms, acceptance 59.9 % / 2.80 |
| ↳ N=8 concurrent decode | W4 (`VLLM_GFX906_SKINNY_M16=1`) | **104.2** (TP=1, util 0.90) |
| ↳ 256k context | FA gather fix (2026-08-24); kv_split fix (2026-09-03) | 250k needle PASS; **37.95 t/s MTP @ 64k ctx** (was 16.6 pre-fix) |
| Qwen3.6 fp16 checkpoints (52–67 GB) | do not fit 32 GB | — |

**Decode-t/s basis (2026-09-13).** The rows above use the *historical bench
basis*: `enable_prefix_caching` left at vLLM's default (ON), so in the 4-sample
protocol samples 2-4 reuse the prompt prefix and their prefill is nearly free.
`_bench_gfx906.py` defaults to prefix caching **OFF** since 2026-08-27
(`BENCH_PREFIX_CACHE=0`, per the "prefix caching off for benchmarks" rule), which
bills the full prefill to every sample. One build, one boot, mclk verified at
1000 MHz in every timed window: **MoE 65.91 warm / 58.43 cold · dense 27B 24.82
warm / 16.27 cold** — identical hardware, ~11 %/~34 % metric difference, no
regression. Use `BENCH_PREFIX_CACHE=1` to compare with the table, `=0` for
serving-shaped prefill-honest numbers; the fork-base deltas (3.49 → 67.39,
18.89 → 25.60) are unaffected because both arms were measured on one basis.

Details, per-model caveats, and bench recipes:
[`docs/gfx906/README.md`](docs/gfx906/README.md) §Model support status.

### Long-context performance (TP=2, 2× MI50 32 GB)

Prefill sweep for the two prime dense models at their max context
(Qwen3.8-27B: 256k; Muse-Glimmer-30B: 128k). Deep prompts, B=1,
tg=128, mean of 2 samples; prefix caching OFF, `--max-num-batched-tokens
4096`, float16, trimmed cudagraph capture `[1,2,3,4]`. 2026-08-29,
boot N (canary 38.9 t/s healthy); csrc @ `cf5ccbd685` (M2 merged + M3
hygiene, bit-identical) — tree as of 2026-09-13 `bbb087b65a`; the later
FIX-H2 / host-`cu_seqlens` work only affects multi-batch prefill, so these
B=1 numbers stand (one-point re-verify pending). Re-run recipe: `docs/gfx906/_serve_tp2_gfx906.sh`
(start/wait/stop; Qwen3.8 at 256k needs `KVBYTES=10737418240`) +
`docs/gfx906/_bench_serve_grid_gfx906.py` with
`'[[32768,128],[65536,128],[112640,128]]' 2`.

| model (max ctx) | 32k prefill | 64k prefill | 110k prefill | TTFT @ 32k/64k/110k |
|---|---:|---:|---:|---|
| Qwen3.8-27B-AWQ-INT4 (256k) | **443.9** | **364.8** | **289.0** | 73.8 s / 179.6 s / 389.7 s |
| Muse-Glimmer-30B-AWQ-INT4 (128k) | **500.0** | **442.1** | **379.6** | 65.5 s / 148.1 s / 296.7 s |

**Post-0.30.0 note (2026-09-24, merge `8893a50e54`):** on a k=3-spec,
`GFX906_FA_LEGACY=0` build with prefix caching OFF, 64k prefill reads **354 t/s**
(TTFT 185.3 s) and 120k **271 t/s** (TTFT 442.2 s) — same order as the rows above;
the sweep itself remains the spec-free 2026-08-29 record.

Prefill t/s = pp / TTFT. Live-ctx tax: prefill rate falls ~12–14 % per
doubling for Muse and ~18–21 % for Qwen3.8 (head_dim 256 makes its
attention share scale harder). Decode (byproduct, tg=128, no spec
decode): Muse 30.5 → 26.3 → 21.9 t/s; Qwen3.8 25.6 → (–) → 13.3 t/s
(the 64k sample ended at out=1 — the model hit EOS on the repetitive
filler, a content artifact; TTFT is unaffected). KV budgets: 6 GiB
(783,892-token pool) for Muse, 10 GiB (323,414-token pool) for
Qwen3.8 — the 256k max-len needs ≥ 8.09 GiB of KV.

### Long-context DECODE with MTP k=2 (TP=2, 2× MI50, 2026-09-02/03)

The prefill sweep above ran without spec decode; here is the same regime
with **MTP k=2 enabled** — and the kv_split clamp fix
(`a6ff64a71b`, merged 2026-09-03) that made it fast: the old `seq_q>2`
hard clamp forced KV-split off for every spec-decode verify step (Sq=3),
so MTP long-context decode ran ~2× slower than greedy. The fix's byte-budget
guard (`GFX906_FA_KVSPLIT_MAX_BYTES`, default 512 MiB) keeps split
parallelism on k≥2 verifies while preserving the prefill OOM protection.

Qwen3.8-27B-AWQ-INT4 (dense), TP=2, `--max-model-len 131072`, tg=256,
temp 0, n=3 reps, synthetic filler corpus s9 (serving config: util 0.85,
bt 1024, max-seqs 4, capture `[1,2,3,4]`, `disable_custom_all_reduce`).

| ctx (pp) | MTP k=2 pre-fix (clamp) | **MTP k=2 post-fix** | greedy | fix gain | MTP vs greedy (post-fix) |
|---:|---:|---:|---:|---:|---:|
| 65,536 | 15.95 t/s | **37.95 t/s** | 18.86 | **2.38×** | 2.01× |
| 98,304 | 11.19 t/s | **29.88 t/s** | 14.80 | **2.67×** | 2.02× |
| 122,880 | 9.18 t/s | **25.70 t/s** | 12.74 | **2.80×** | 2.02× |

**Re-measured on the 0.30.0 base (2026-09-24, merge `8893a50e54`), MTP k=3 +
`GFX906_FA_LEGACY=0` (both current defaults), same filler / prefix-caching-OFF
basis:** 64k **42.0** / 120k **40.7 t/s** (2 samples each; 8k/32k/96k not
re-measured). Prefill on the same runs: 64k **354 t/s** (TTFT 185.3 s), 120k
**271 t/s** (TTFT 442.2 s). The k=2 rows above remain the 2026-09-02/03 record;
the k=3 rows are faster because depth 3 adds accepted tokens/step and the
KV-read layout changed. PPL gate on this build: **10.5472** (359 tokens, 0
top-20 misses). Raw logs: `/local/tmp/bench_lc_030.log` (filler),
`/local/tmp/bench_2k_k3.log` (standard CLI).

The old "MTP < greedy past ~20k ctx" live-ctx tax is gone: with the fix,
MTP beats greedy by ~2× at 64k+ context (it still leads at short context —
59.2 t/s @2k). Raw data: `/local/tmp/mtp1/data_mtp_bootQ.jsonl` (pre-fix,
boot Q) and `data_mtp_k2fix_bootS.jsonl` (post-fix, boot S).

**Depth: k=3, not k=2 (2026-09-09/11, same-corpus A/B, mixed agent+chat
payload — `docs/gfx906/DEVLOG-mtp-depth-matrix.md`).** The k=2 row above is the
kv_split-fix evidence; the *depth* call came later, on the production payload,
where k=2's perfect-acceptance advantage does not transfer:

| arm | 64k | 120k | note |
|---|---:|---:|---|
| greedy | 19.76 | 13.11 | v1 corpus |
| k=2 | **27.30** | 22.65 | v2 (20 % chat) |
| **k=3** | 27.44 | **24.76** | v2 — **+9.3 % vs k=2 @120k**, tie at 64k |
| k=4 | 29.37 | 21.87 | loses on real payloads (s9 win did not transfer) |

Mechanism: k=3's verify block (1 anchor + 3 drafts = 4 rows) pads to the same
occ-2 FA tile as k=2 (3 rows), while k=4 (5 rows → pad 8) crosses to the slow
occ-1 tile. **Recommendation for long-context serving of Qwen3.8-27B on TP=2:
enable MTP k=3** (`--speculative-config
'{"method":"mtp","num_speculative_tokens":3}'`, capture sizes multiples of 4);
k=2 remains the choice for short-context / copy-light workloads.

### Headline: agentic Python coding on our own corpus (2026-09-13)

The tables above use synthetic filler; this is the workload we actually serve.
Qwen3.8-27B-AWQ-INT4, TP=2, `--max-model-len 131072`, tg=256, temp 0, **our own
CAT-1 corpus** — 15.5 M tokens of pi/hermes **agentic Python coding** traffic
(`docs/gfx906/CAT1-corpus-build.md`), 8 distinct bodies per point, prefix cache
off so every rep pays its full prefill, 2 reps per cell (boot f27e8058, mclk
verified 1000 MHz):

| ctx | prefill | greedy | MTP k=3 | **MTP k=3 + CAT-1** | uplift |
|---|---:|---:|---:|---:|---:|
| 64k | 277 t/s | 19.80 | 33.28 | **33.26** | **1.68×** vs greedy |
| 120k | 226 t/s | 13.17 | 24.74 | **24.95** | **1.89×** vs greedy |

Acceptance (mean accepted per step) at 120k is the strongest in the fork's
records — 2.05/2.15 for plain k=3, 1.99/2.07 with CAT-1 — because a long agentic
tail is copy-heavy, which is exactly CAT-1's operating point. CAT-1 and plain
k=3 are a tie here (within the arm's own rep spread); its benefit is the
**per-step** one, measured under control on 11 identical 8k prompts × 2 reps:
**−2.52 ms/step [−2.91, −1.94] ⇒ +4.8 % mean / +5.9 % median t/s**, acceptance
no detectable penalty. So quote **~33 t/s @64k / ~25 t/s @120k** for agentic
coding on TP=2, and read the CAT-1 gain from the controlled A/B, not from this
session's 2-rep cells.

**Re-measured on the 0.29 line (2026-09-14, boot eefacc1e, V1 pinned, same
protocol).** Greedy **20.39 @64k / 13.26 @120k** (the 0.28 line: 19.80 / 13.17);
**MTP k=3 + CAT-1 34.62 / 25.55** (33.25 / 24.95) — i.e. the CAT-1 gain
reproduces on 0.29 (+6.4 % @64k / +4.2 % @120k over plain k=3 in-session).
Bare MTP k=3 was then re-measured **arm-level (3 reps)**: 64k {35.20, 35.14,
29.56} → mean **33.30** (0.28: 33.28) and 120k {24.26, 25.14, 24.21} → mean
**24.54** (0.28: 24.74, i.e. −0.8 % inside the spread); acceptance 2.05/2.05/1.55
and 2.06/2.15/2.06. So **spec decode is at parity on 0.29** — the earlier 2-rep
"−1–2 %" was the arm's own acceptance variance (which is why the re-check was
run before merging).

### Benchmarks

**gfx906 fork — dense AWQ `QuantTrio/Qwen3.5-9B-AWQ` (few full attention
layers), pp = prefill throughput (tok/s), single MI60, eager, pp/tgen two-phase:**

| pp | `CUSTOM` prefill | stock `ROCM_ATTN` prefill | Δ |
| ---: | ---: | ---: | ---: |
|  256 | 590 | 575 | +2.6% |
|  512 | 757 | 764 | −0.9% |
| 1024 | 1483 | 1427 | +3.9% |
| 2048 | 1399 | 1288 | **+8.6%** |

Decode throughput in that table reflects the vendor baseline; this
fork's decode path (above) changes these models' decode numbers
substantially.

**Upstream gfx906-fa-vllm — full-attention `MiniMax-M2.7-AWQ-4bit` (8× MI50,
TP=8, BS=1, from the upstream repo's README):**

| ctx | `CUSTOM` TG (tok/s) | Δ vs stock `TRITON_ATTN` |
| ---: | ---: | ---: |
|  1K | 27.7 | — |
|  32K | 7.7  | +6% |
| 100K | 3.9 | **+32%** |
| 130K | 3.0 | **+29%** |

On a full-attention model at long context the custom kernels give roughly
**+20–40%** prefill/overall throughput and stay functional where the stock
Triton kernels stall.

### Escaping back to the default attention backend

To bypass `CUSTOM` and use vLLM's stock ROCm backend instead:

```bash
vllm serve ... --attention-backend ROCM_ATTN
# or in Python:
# LLM(..., attention_backend="ROCM_ATTN")
```

Set the env `VLLM_ATTENTION_BACKEND=ROCM_ATTN` as well for earlier-stack paths.
If you built without gfx906 (no FA extension compiled), or the backend is not
registered, vLLM automatically falls back to the stock ROCm/TRITON backends.

### Recommended serving configuration (gfx906, 32 GB MI50/MI60)

Environment variables:

| variable | value | when | why |
|---|---|---|---|
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | **every run** | the ROCm platform aborts at import without it ("ROCm platform requires upstream flash-attn to be installed"); selects the Triton-AMD flash-attn path |
| `LD_LIBRARY_PATH` | `…/rocm/lib` | venv runs | system ldconfig knows no ROCm — `source` your ROCm env script (or export `LD_LIBRARY_PATH=/opt/rocm/lib`) before launching |
| `HF_HUB_OFFLINE` | `1` | optional | set only if the model is already in the local HF cache — without it vLLM downloads/refreshes from the hub on import |
| `VLLM_GFX906_HIP_LIB_PATH` | `…/rocm/lib/libamdhip64.so.7` | **TP≥2 only** | with the blocking-sync `.pth` shim below (see note) — without both, every TP worker permanently pegs a host core at ~100 % (HIP active-wait) |
| `VLLM_GFX906_SKINNY_M16` | `1` | optional | dense N=8 concurrent decode (+14.5 % W4 skinny GEMV); default off |
| `GFX906_FA_TILE_CLIP` | default `1` | optional | M2 per-q-tile window raise + causal cap in the prefill FA (bit-identical, skips masked k-tiles); `0` = disable (A/B arm) |
| `HSA_OVERRIDE_GFX_VERSION` | — **do not set** | — | ROCm 7.14 has native gfx906 (older 7.2.1 images needed `9.0.6`) |
| `VLLM_ATTENTION_BACKEND` | — **do not set** | — | the custom Q8 FA backend (`CUSTOM`) is already the gfx906 default |

Validated serve command (2× MI50, Qwen3.8-27B-AWQ-INT4, 2026-08-25/29):

```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
vllm serve <model> \
  --served-model-name <name> \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.82 \
  --compilation-config '{"cudagraph_capture_sizes":[4,8,12,16]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --generation-config auto
```

- **Model Runner: V2 is the validated default on 0.29.0+, per model.** Upstream
  defaults to V2; on gfx906 it is now brought up and at parity for the models
  listed below, so no pin is needed for them. `VLLM_USE_V2_MODEL_RUNNER=0` still
  selects V1 (used for A/B reference arms), and **the two models whose parity run
  has not passed must keep it set**: Gemma-4 (its in-process PPL gate is
  inapplicable — both runners return a degenerate distribution, so it needs a
  serving-level gate) and Muse-Glimmer (no local AWQ checkpoint). Validated on V2:
  Qwen3.8-27B, MoE 35B, Nemotron 3.5 Lightning (PPL 27.0066 vs 26.9986 — see
  `docs/gfx906/V2-bringup.md` for the exact pairs) and Ornith (16.7824 vs 16.7724).
  Caveats measured on V2: in the TP=2 serving config it reserves more VRAM for
  graph capture (KV pool 454,536 vs V1's 496,693 tokens at `--gpu-memory-utilization
  0.82`), and the **CAT-1 shortlist buys a ms/step saving, not acceptance** —
  same-boot 3-rep A/B on V2: 34.41 → 35.44 t/s @64k and 24.00 → 24.61 @120k
  (−2.6 / −4.1 ms/step ⇒ **+3.0 % / +2.5 %**), acceptance unchanged. (An earlier
  +23 % figure came from an A/B client that put the arm name in the prompt header
  — retracted, see `docs/gfx906/V2-bringup.md`.)
  **V1 sunset:** upstream removes the V1 model runner in **0.32.0**, and we track
  that schedule. Gemma-4 cleared its V2 gate on 2026-09-15 (templated V1/V2 comparison:
  identical answers and logprobs), so **Muse-Glimmer is the only remaining pin** —
  tracked as ROADMAP `DFL2-2` (V2 up to speed) and `MUSE-1`. **Prompt format matters:**
  instruction-tuned checkpoints (Gemma-4-*-it, Muse-Glimmer) must be prompted through
  their chat template — raw `/v1/completions` text or a raw-text PPL probe returns
  garbage that looks like a broken model yet is a prompt-format artifact; see the
  prompt-format note in `docs/gfx906/README.md`.
- `--dtype float16` is required: gfx906 has no bf16 hardware; bfloat16
  checkpoints would fall back to fp32 math.
- **cudagraph capture sizes = multiples of `num_speculative_tokens + 1`, up to
  `max_num_seqs × (k+1)`.** MTP k=3 (width 4) with the 4-seq default →
  `[4,8,12,16]`; ngram n=5 (width 6) with 4 seqs → `[6,12,18,24]`. The engine
  rounds explicit entries **up** to that multiple, dedups and lowers
  `max_cudagraph_capture_size` to the last entry, so an undersized list silently
  leaves multi-request steps eager (`[1,2,3,4]` with k=3 collapses to `[4]` =
  B=1). Use `[1,2,3,4]` only for prefill/TTFT-focused or spec-free serving.
  Plain TP does not enter this calculation; **sequence parallelism** does
  (`multiple_of = max(k+1, tp)`, which rejects some depths at tp=4).
- EAGLE is too heavy for these GPUs; **MTP k=3 is the default spec config for
  the Qwen3.5/3.8 family** (same-corpus A/B: +9.3 % over k=2 @120k, tie at 64k;
  agentic-corpus headline 33.3 @64k / 25.0 @120k t/s). k=2 stays useful for
  short-context/copy-light work, k=4 is a loss on real payloads, and both beat
  greedy by ~2× at 64k+ after the kv_split fix. **MTP is the spec config for
  every local model, Muse-Glimmer included** (Kevin, 2026-09-13). ngram is
  deprecated for now: the ngram numbers in this tree (e.g. +15 % decode at
  tg256, the Muse-Glimmer "100 % filler acceptance" rows) were measured on
  filler corpora and are ceilings, not real-payload results — a Muse-Glimmer MTP
  A/B is queued as MUSE-1 in `docs/gfx906/ROADMAP.md`.
- Tool/reasoning parsers: Qwen 3.5/3.6/3.8 → `qwen3_coder` + `qwen3`;
  Muse-Glimmer → `muse_glimmer` for both.
- `--gpu-memory-utilization`: 0.82 with the spec config above; 0.93 for
  dense TP=1 serving (0.95 OOMs on the second request with a warm
  inductor cache); the in-process bench harness uses 0.95.
- **Benchmarks:** add `--no-enable-prefix-caching` (replayed prefixes
  poison TTFT-derived numbers).
- **TP=2 prerequisites:** the official AMD DKMS `amdgpu` driver (the
  stock Ubuntu driver stalls or hangs RCCL P2P/IPC on dual-root-port
  topologies). One-time shim: `cp docs/gfx906/gfx906-blocking-sync.pth
  .venv/lib/python3.12/site-packages/` and export
  `VLLM_GFX906_HIP_LIB_PATH` (absolute path to `libamdhip64.so.7`);
  re-copy the `.pth` after any fresh venv. TP=1: drop
  `--tensor-parallel-size`, set `HIP_VISIBLE_DEVICES=0`, skip the shim.
- **In-process (harness) only:** `VLLM_ENABLE_V1_MULTIPROCESSING=0
  VLLM_USE_AOT_COMPILE=0 TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` —
  not needed for `vllm serve`.

Deep-dive recipes, docker images, and build instructions:
[`docs/gfx906/running.md`](docs/gfx906/running.md).

### Building this checkout

Use Linux with Docker BuildKit and an x86_64 host. The build context is this
checkout, including the vendored `csrc/gfx906_fa` extension; the helper never
clones another vLLM branch to determine the contents.

```bash
bash build_and_push_docker.sh
# Reduce parallelism if necessary:
MAX_JOBS=4 bash build_and_push_docker.sh my-gfx906-test
```

See [the build and validation record](docs/gfx906/UPGRADE-0.30.0.x.md) for
dependency pins, editable builds, package checks, and remaining GPU tests.
The old ROCm 6.3.x / PyTorch 2.11 / Triton 3.6 combination is **unverified for
this release**; the v0.30.0 API and build metadata target PyTorch 2.13.

After a successful local build, an example GPU container invocation is:

```bash
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" \
  --ipc=host -v "$PWD/models:/models:ro" \
  aiinfos/vllm-gfx906-mobydick:v0.30.0.x-rocm7.14-pytorch2.13.0
```

Use the drivers and device access appropriate to your host. The reference's
dual-GPU topology notes in `docs/gfx906/running.md` are historical environment
evidence, not a driver-installation requirement established by this upgrade.

### Quickstart example (with Qwen3.5-0.8B)

```code
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE VLLM_LOGGING_LEVEL=DEBUG vllm serve Qwen/Qwen3.5-0.8B \
  --dtype float16 \
  --kv-cache-dtype float16 \
  2>&1 | tee log.txt
```

NB: --dtype float16 is recommended to add for this gfx906 fork. If not set, vllm will take the dtype from config.json model which might be bfloat16, not natively supported on gfx906 (with potential fallback to float32, leading to slower inference)

CREDITS
-------

- https://github.com/nlzy/vllm-gfx906
- https://github.com/Said-Akbar/vllm-rocm
- https://github.com/vllm-project/vllm

---

<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

The upstream PyPI quickstart below is for official vLLM. For this gfx906
branch, use the source build above. Upstream installs can use
[`uv`](https://docs.astral.sh/uv/):

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
