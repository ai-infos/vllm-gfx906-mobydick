# A6 — GDN bucket breakdown (k=3 MTP, TP=2, mixed-v2 payload)

**Status:** DONE. The GDN (linear-attention) bucket decomposes into
**GEMM-dominated** per-layer cost (out_proj 110 + qkvz 80 + ba 19 = 209 µs
of 360 µs/layer = 58 %), with the recurrent core (conv 13.5 + rec 29 µs)
only 12 % and the rest eager launch-gap/unpack (~99 µs/layer, an
eager-mode artifact that cudagraphs largely remove). The bucket is
**context-flat** (360.2 → 362.6 µs/layer, 64k→120k, +0.7 %) — all context
growth is in full_attn (×1.80). **SYV-5 (fp16 GDN recurrent state) is a
confirmed dead end**: the rec kernel runs 3.7× the pure-bandwidth state-
traffic floor (latency-bound, not state-BW-bound), so fp16 state saves
≤ 0.4–1.6 % of the step even in the impossible fully-traffic-bound case.
**Date:** 2026-09-08 · boot Y3 (3rd reboot; 0 wedges on the profile; one
wedge #43 on the aborted attempt 2 launch — chronic weight-load family,
see degradation.md) · branch `gfx906/fa-decode-fp16`

---

## Question

The pfk4 k=4 ledger (`PROFILE-k4-step-ledger.md`) measured the GDN bucket
as a flat ~518–525 µs/layer (48 layers ≈ 25 ms/step) across context, with
no sub-bucket detail. A6 (per the combined doc's order, `fa-decode-fp16-
hunt-combined.md` item A6) decomposes the bucket into per-layer sub-
buckets at the **winning depth k=3** (boot Z/W' verdict) to size:
(a) SYV-5 — fp16 GDN recurrent state (`--mamba-ssm-cache-dtype float16`),
and (b) whether targeted GDN kernel work (rec/conv/unpack) has headroom.

## Method

`pfk4`-style CUDA-event profiler, new plugin `agdn_phase_plugin.py`
(harness files under `/local/tmp/mtp1/`; the pfk4 plugin survived and was
used as the structural template — its `per_step` accounting and the
wall-anchor calibration carry over):

- **Server:** `run_agdnsrv.sh` — serve-based TP=2 `vllm serve`, Qwen3.8-27B-
  AWQ-INT4, k=3 MTP spec decode, `--enforce-eager` (Python forward hooks
  fire every step), `--max-num-batched-tokens 1024`, util 0.85, maxlen
  131072, port 8130. Eager because hooks don't survive graph capture.
- **Client:** `agdn_client.py` — the **mixed-v2 production payload** (20 %
  chat), two windows: c64k (65,536 ids) + c120k (122,880 ids), each
  256-token generation. Per window: warmup request (8k prefill + 64
  decode) to stabilize clocks/jit, then rearm the plugin config
  (`skip=10 steps=12` — the 12 decode steps after 10 warm steps) and run
  the windowed request. Both ranks flush `agdn_{arm}_p{pid}.json`.
- **Buckets:** representative-layer hooks (pfk4 convention — one GDN
  layer, one FA layer, one MLP; per-step = median × layer count) for
  `gdn_qkvz`, `gdn_ba`, `gdn_norm`, `gdn_out` (module pre/post hooks on
  the linear-attention submodules) plus `full_attn`, `mlp`, `d_attn`,
  `d_fc`, `t_lm_v`, `t_lm_d`; wrapped module functions (same-process
  CUDA events) for `gdn_conv` (`causal_conv1d_update`) and `gdn_rec`
  (`fused_sigmoid_gating_delta_rule_update`); `gdn_core_rest` derived =
  lin_attn(median) − (qkvz+ba+norm+out)(medians) per layer. `lin_attn`
  (whole GDN layer forward) demoted to REFERENCE. Step bracket =
  prev-decode-end → draft-end events (the pfk4 wall anchor).
- **Aggregation:** `agdn_ledger.py` — averages both ranks, prints the
  per-step ledger (ms, % of summed), the per-layer µs table, and the
  SYV-5 sizing.

Model facts (Qwen3.5/3.8-27B, TP=2): 64 layers = 48 GDN + 16 FA; GDN
state = 24 v-heads/rank × 128 k × 128 v × 4 B fp32 = **1.50 MiB/slot/
rank**; the rec kernel reads one state slot + writes one checkpoint per
token → (1+ROWS) = 5 slots/layer/step at k=3 (ROWS=4) ≈ 360 MiB/step/
rank of state traffic.

## Results

Per-step ledger (eager, both ranks averaged, 12-step windows):

| bucket | 64k (ms) | % | 120k (ms) | % | 64k→120k |
|---|---|---|---|---|---|
| **step bracket** | **88.3** | — | **128.6** | — | ×1.46 |
| full_attn (16 layers) | 42.04 | 46.8 | 75.82 | 58.7 | ×1.80 |
| mlp (48) | 15.75 | 17.5 | 15.97 | 12.4 | ×1.01 |
| **GDN total (48)** | **17.26** | 19.2 | **17.43** | 13.5 | **×1.01** |
| — gdn_core_rest | 6.81 | 7.6 | 6.91 | 5.4 | ×1.01 |
| — gdn_out | 5.21 | 5.8 | 5.23 | 4.0 | ×1.00 |
| — gdn_qkvz | 3.87 | 4.3 | 3.88 | 3.0 | ×1.00 |
| — gdn_ba | 0.89 | 1.0 | 0.89 | 0.7 | ×1.00 |
| — gdn_norm | 0.48 | 0.5 | 0.48 | 0.4 | ×1.00 |
| d_attn (MTP draft) | 6.40 | 7.1 | 11.60 | 9.0 | ×1.81 |
| t_lm_d (verify logits) | 4.84 | 5.4 | 4.84 | 3.8 | ×1.00 |
| t_lm_v (draft logits) | 2.48 | 2.8 | 2.47 | 1.9 | ×1.00 |
| d_fc (MTP fc) | 1.10 | 1.2 | 1.09 | 0.8 | ×1.00 |

Summed = 89.9 / 129.2 ms vs bracket 88.3 / 128.6 ms → closure
−1.6 / −0.6 ms (the pfk4 calibration property holds).

**GDN layer decomposition (µs/layer, median; ranks agree within noise):**

| component | 64k | 120k |
|---|---|---|
| lin_attn (whole GDN layer) | 360.2 | 362.6 |
| — qkvz GEMM | 80.4 | 80.5 |
| — ba GEMM | 18.6 | 18.4 |
| — norm | 10.1 | 9.9 |
| — out_proj GEMM | 110.2 | 108.5 |
| — **core_rest** (derived) | **141** | **145** |
| &nbsp;&nbsp;of which: conv1d | 13.4 | 13.6 |
| &nbsp;&nbsp;of which: rec (delta-rule update) | 28.6 | 29.0 |
| &nbsp;&nbsp;of which: unpack/zero/index + eager gaps | ~99 | ~102 |

Sample counts: 576 conv/rec samples per rank (48 layers × 12 steps),
12 step brackets; both ranks within ~1 µs on every component.

## Findings

1. **GDN is context-flat, as predicted.** 360.2 → 362.6 µs/layer
   (+0.7 %) from 64k to 120k; every GDN component flat. The whole
   64k→120k step growth (88.3 → 128.6 ms, +40.3 ms) sits in full_attn
   (+33.8 ms) + d_attn (+5.2 ms) — the FA-side buckets. This confirms
   the pfk4 "flat ~25 ms bucket" observation at k=3 (the bucket is
   17.3 ms here — the k=4→k=3 M drop, 5→4 verify rows, plus the k=4
   ledger's boot-state; the flatness is the load-bearing fact).
2. **The GDN layer is GEMM-dominated, not state-traffic-dominated.**
   209 of 360 µs (58 %) is in_proj_qkvz + in_proj_ba + out_proj GEMMs;
   the actual recurrence (conv 13.5 + rec 29 µs) is 12 %. Zeroing the
   rec+conv kernels entirely would save 48 × 42.5 µs ≈ 2.0 ms/step
   (2.3 % of the 64k step) — and that's the absolute ceiling, not a
   realistic kernel-improvement target.
3. **core_rest (~99 µs/layer) is mostly eager launch-gap.** Between the
   sub-GEMMs and the rec kernel the forward runs ~10–15 small ops
   (qkvz/ba unpacking, rearrange_mixed_qkv, z/gate splits,
   core_attn_out zero, index math) at ~5–8 µs/launch in eager. In the
   graphed production path these collapse to kernel time only, so the
   graphed GDN layer ≈ 260–290 µs and the graphed GDN bucket ≈ 13–14
   ms/step (~15–17 % of the graphed step). The remaining GDN levers are
   the **M≤8 launch-regime GEMMs** (out_proj 5.3 ms/step + qkvz 3.9
   ms/step at M=4, both weight-read-bound) — the same family as T-1.5/
   A4, not a new standalone project.
4. **SYV-5 (fp16 SSM state) — dead end, CLOSED.** rec = 28.6–29.0
   µs/layer vs a pure-bandwidth state-traffic floor of 7.7 µs/layer
   (5 slots × 1.50 MiB at 1024 GB/s): the kernel runs **3.7× the floor**
   → latency-bound, not state-BW-bound (implied 272 GB/s ≪ HBM). fp16
   halves state bytes; the time attributable to state traffic is
   between the floor (BW-perfect) and all of rec (traffic-bound), so
   savings ∈ [0.5×floor, 0.5×rec] per layer. Doubling for the
   unmeasured gdn_copy (state copies are at most the same order as rec):
   **−0.37 … −1.40 ms/step = 0.4 … 1.6 % of the step** (64k), 0.3–1.1 %
   (120k). The realistic end (latency-bound rec) is the bottom of that
   bracket. Sub-1 % lever on a flag that needs a PPL gate → not worth
   the gate. `--mamba-ssm-cache-dtype float16` is confirmed live for
   this model family (`get_mamba_state_dtype_from_config`,
   `models/qwen3_5.py:378/590`) — the lever exists, it's just small.
5. **The MTP draft attention scales with context** (d_attn 6.4 → 11.6
   ms, ×1.81, tracking the FA layers) — expected (draft full attention)
   and folded into the FA-side story (kv_split/A5 territory, closed).

## Caveats

- **Eager mode** (required for the hooks): per-op launch gaps inflate
  op-count-heavy buckets, especially core_rest. Absolute ms/step values
  are UPPER bounds vs the graphed production path; ratios and the
  context-flatness finding transfer. The step bracket (88.3 ms @64k) is
  consistent with graphed production (k=2/3 serving 27–31 t/s ≈ 36–44
  ms/step at ~3.5–3.8 tok/step — eager adds the launch-gap tax on top).
- **gdn_copy unmeasured** on this run: the plugin's mamba-state-copy
  wrapper referenced a non-existent class name (`MambaStateManager`;
  the methods live on `MambaSpecDecodeGPUContext`,
  `vllm/v1/worker/mamba_utils.py:644`). Fixed in the installed plugin
  (`/local/tmp/mtp1/agdn_phase_plugin.py`); no re-run — the SYV-5
  bracket already doubles for an at-most-equal copy, and the copy's
  ceiling cannot move any verdict.
- **Representative-layer extrapolation** (pfk4 convention): assumes
  homogeneous layers of each type — holds architecturally here (all 48
  GDN layers identical shapes; all 16 FA identical). Per-step totals are
  anchored by the step-bracket closure (−1.6/−0.6 ms), which bounds the
  extrapolation error.
- 12-step windows (256-token generations); clocks DVFS-warm after the
  warmup request (pfk4 protocol).

## Verdict

**GDN breakdown = GEMM-dominated (58 %), context-flat, rec/conv ≤ 12 %
of the layer.** SYV-5 fp16 state **CLOSED as a dead end** (≤ ~1.5 % of
the step even in the impossible traffic-bound case; realistic ≪ 1 %).
The residual GDN levers fold into the M≤8 GEMM launch-regime story
(out_proj + qkvz ≈ 9.2 ms/step eager) — no standalone GDN kernel
project is justified by these numbers. **GATE:** this doc IS the gate
(profiling sizing step); it sizes SYV-5 to "not worth the PPL gate" and
down-grades "targeted GDN kernel work" to "part of the GEMM-M-tax
story". Next in the combined-doc order: SYV-13 (mamba/GDN chunked-
prefill align fixes — verify the claim) or the cheap A8 controls.

## Files

- Harness (persistent, `/local/tmp/mtp1/`): `agdn_phase_plugin.py`
  (+ `agdnpkg/` pip package, installed in the repo venv under the
  `agdnphase` name), `agdn_client.py`, `run_agdnsrv.sh`, `run_agdn.sh`
  (driver), `agdn_ledger.py`, results `agdn_c{64k,120k}_p{8340,8341}.json`
  + `agdn_result.txt`, `agdn_driver.log`, `agdnsrv.log`.
- Wedge #43 (aborted attempt 2, chronic weight-load family, GPU1,
  self-recovered): `degradation.md` / `degradation_details.md`.
- One harness bug found + fixed on this boot (not a GPU issue): the
  conv/rec wrapper's `self=None` positional swallowed the first arg of
  the non-self functions (arg shift → `weight=None` → AttributeError at
  the first decode step). Fixed + unit-checked; attempt 3 ran clean.
  A second aggregator units bug (state-floor µs calc) fixed before the
  final table.
