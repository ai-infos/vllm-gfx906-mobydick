# How to run the gfx906 FA decode + gather micro-benches

Standalone (no server boot) CUDA-event micro-benches for the custom FA
attention path, at the **Qwen3.8-27B per-rank shape** (TP=2). Used to size
kernel changes before a full serve A/B — see
`PROFILE-k4-step-ledger.md` "Kernel-level localization" for the findings they
produced.

## Scripts

| script | what it measures |
|--------|------------------|
| `benchmarks/kernels/gfx906/bench_gfx906_fa_decode_qwen27.py` | the FA decode kernel (`gfx906_fa.forward`, Q8 K + fp16 V, contiguous BHSD) — time vs Sk, effective KV bandwidth, HBM-floor ratio, maxerr vs torch reference at small Sk |
| `benchmarks/kernels/gfx906/bench_gfx906_fa_gather_qwen27.py` | the paged-KV gather kernels (`gather_paged_kv_q8` fused + LEGACY persistent path) + per-layer Q-fp32 side costs (cast/pad/unpack) + the torch `_gather_kv` fallback for comparison |

Both are **zero production impact**: they import only `vllm._gfx906_fa_C`
(the already-built extension) and a couple of pure-Python helpers; no model
weights, no engine, no server. They allocate ~1 GB at Sk=122880 on ONE GPU and
exit. Safe to run while a server is idle (it shares the GPU but adds no
steady-state load).

## Run

```bash
cd /local/git/vllm-gfx906-mobydick

# FA decode kernel, greedy shape (Sq=2) — bandwidth-bound reference:
.venv/bin/python -u benchmarks/kernels/gfx906/bench_gfx906_fa_decode_qwen27.py

# FA decode kernel, k=4 VERIFY shape (Sq_pad=8 = max_seqlen_q 5 rounded to
# ncols1) — the production spec-decode shape; this is the one with headroom:
BENCH_FA_SQ=8 .venv/bin/python -u benchmarks/kernels/gfx906/bench_gfx906_fa_decode_qwen27.py

# KV gather + Q-side costs (long-context sweep by default):
.venv/bin/python -u benchmarks/kernels/gfx906/bench_gfx906_fa_gather_qwen27.py
```

## Env knobs

FA decode bench:
- `BENCH_FA_SQ` — Sq_pad. **2 = greedy decode, 8 = k=4 verify (k+1=5 rows →
  ncols1=8).** Always run both when evaluating a kernel change; the two shapes
  are compute-bound vs bandwidth-bound respectively and can move in opposite
  directions.
- `BENCH_FA_HQ` / `BENCH_FA_HKV` / `BENCH_FA_D` — shape (default 12/2/256 =
  Qwen3.8-27B per-rank; the upstream 35B-A3B shape is 16/2/256).
- `BENCH_FA_SK` — comma list of Sk values (default `8192,32768,65536,98304,122880`).
- `BENCH_FA_CHECK_MAX` — max Sk for the torch-reference correctness check
  (default 8192; the per-head Python reference is O(Hq×Sk) and too slow at
  long Sk). Correctness beyond this relies on the merged test suite.
- `GFX906_FA_NC2` / `GFX906_FA_KVSPLIT` — launcher config knobs (head-packing,
  gridDim.y KV-split). Parsed once per process → **one process per config**
  when sweeping.

Gather bench:
- `G_SK` — comma list of Sk (default `65536,98304,122880`).
- `BENCH_G_HQ` / `BENCH_G_HKV` / `BENCH_G_D` / `BENCH_G_BLOCK` — shape
  (default 12/2/256/16).

## Interpreting output

- **`eff GB/s` vs HBM floor:** the "floor" uses bytes = n_tiles × Sk ×
  (Q8-row + 2·D) at 798 GB/s (P3-0 measured MI50 read BW). At Sq=2 the ratio
  can exceed 100% — that's the Q8 K rows being ~half the fp16 byte width, i.e.
  "at/beyond the read-bound limit", not an error. At Sq=8 a ~46% ratio means
  **compute-bound** (KV re-read per query row) → optimization headroom.
- **`linear fit` slope (ns/token):** the O(S) coefficient — compare across
  kernel changes at the SAME Sq.
- **maxerr:** vs the fp32 torch reference; must stay ≤ ~0.05 (Q8 quantization
  noise floor is ~5e-4). A bit-exact kernel change should keep maxerr equal to
  the pre-change value at every checked Sk.

## DVFS caveat

MI50 idles at mclk 350 MHz and ramps to ~1 GHz within ~0.5 s of load; cold
bench numbers inflate ~3×. These benches warm up before timing, so they are
fine as-is, but if you compare against a *serving* number, sample
`rocm-smi --showclocks` during the run — do not mix cold-clock standalone
numbers with warm serving numbers.

## When to re-run

- Before/after any change touching `csrc/gfx906_fa/kernel/fattn-q8*.cuh`,
  `gfx906_fa_launcher.cu`, or the gather kernels.
- When changing spec depth k (verify Sq_pad = round-up of k+1 to ncols1).
- As the baseline step before a serve A/B (this is the fast gate; the serve
  A/B is the final adjudication — kernel-level wins do not always transfer,
  see `DEVLOG-fa-legacy0-b1-decode.md`).
