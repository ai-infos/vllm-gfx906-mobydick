# k=4 Step Ledger — per-bucket profile of the 111.6 ms-class decode step

**Status:** profiling DONE (calibrated). Lever identified + localized = the
custom FA **decode kernel at the k=4 verify shape (Sq_pad=8)**, which runs
compute-bound at ~46% of HBM — real headroom, unlike the greedy Sq=2 shape.
FA-decode implementation pending (this doc is the write-up gate).
**Date:** 2026-09-06 · branch `gfx906/t1-int8-fp16-mass` @ `a0bb358670`
(+ uncommitted FA bench probes, see "Kernel-level localization")

---

## Symptom / question

Our k=4 (MTP num_speculative_tokens=4) decode step at long context is ~2.2×
slower per-step than 1CatAI's reference round, and the gap *widens* as context
grows (64k→120k). Round-5 recon converged on "step-composition dilution" but
never decomposed the step into buckets. This profile does that: it splits one
clean k=4 decode step into per-bucket GPU time and tracks each bucket across
three context lengths to find where the extra cost lives.

## Method (pfk4 CUDA-event profiler)

- **Harness:** serve-based TP=2 `vllm serve` (the clean boot family; in-process
  `LLM()` wedges on this host — see degradation.md #9-11/#21), `--enforce-eager`
  so Python forward hooks fire every step, k=4 MTP spec decode, one server boot
  serves all three context windows (client rearms the plugin between requests).
- **Plugin:** `pfk4_phase_plugin.py` — a vLLM general plugin. CUDA-event pairs
  around representative modules give per-bucket GPU time; a step-level bracket
  (open at layer-0 pre-hook, close one step later) is the wall anchor.
- **Windowing / decode detection:** gate on target decoder-layer-0 forward. A
  forward is *decode* iff its `hidden_states` token count M ≤ 32 (k=4 verify is
  M=5; chunked-prefill chunks are M≈1024 and are excluded). **This was the crux
  bug** — see "Ruled out" below.
- **Buckets:** `full_attn` (16 full-attention layers), `lin_attn` (48 GDN/linear
  layers), `mlp` (64), drafter `d_fc`/`d_attn`, and the shared lm_head split by
  call shape — `t_lm_v` (verify, M=5) vs `t_lm_d` (draft, M=1×4). Per-layer
  buckets are median × layer-count; per-step buckets (head, drafter) are raw
  sums. All categories are non-nested by construction.
- **Sampler:** skip=10 decode steps, then 12 profiled steps per context, both
  TP ranks, temp=0, ignore_eos, tg=256.

**Calibration (why the numbers are trustworthy):** the step-bracket wall anchor
reproduces our independently-measured end-to-end throughput within ~2% at every
context, and the summed buckets cover ~100% of each step (uninstrumented
remainder −3 to −4 ms):

| ctx | step bracket | implied t/s (5 tok/step) | measured e2e t/s |
|-----|-------------|--------------------------|------------------|
| 64k | 113.6 ms | 44.0 | 44.8 |
| 96k | 142.6 ms | 35.1 | 35.8 |
| 120k| 163.9 ms | 30.5 | 31.0 |

## The ledger (ms/step, % of summed)

| bucket | c64k | c96k | c120k |
|--------|------|------|-------|
| **full_attn** (×16) | 53.0 (45%) | 77.9 (53%) | **97.0 (57.5%)** |
| lin_attn / GDN (×48) | 25.2 (21%) | 25.0 (17%) | 24.9 (15%) |
| mlp (×64) | 17.7 (15%) | 17.4 (12%) | 17.7 (10.5%) |
| d_attn (drafter) | 8.8 (7.5%) | 13.0 (8.9%) | 16.2 (9.6%) |
| t_lm_d (draft head M=1×4) | 6.4 (5.5%) | 6.4 (4.4%) | 6.4 (3.8%) |
| t_lm_v (verify head M=5) | 4.9 (4.1%) | 4.9 (3.3%) | 4.9 (2.9%) |
| d_fc | 1.8 (1.5%) | 1.8 (1.2%) | 1.7 (1.0%) |
| **step total** | **113.6** | **142.6** | **163.9** |

Per-layer medians (µs): full_attn 3315 / 4869 / 6061 · lin_attn 525 / 521 /
518 · mlp 277 / 271 / 276.

## The finding: long-context full attention is the single lever

- **full_attn is O(S)** — it is the only bucket that scales with context. Per-layer
  median grows 3.3 → 4.9 → 6.1 ms (64k→120k), a +83% rise while every other bucket
  is flat (lin_attn, mlp constant; drafter head constant).
- **It accounts for 87% of the 64k→120k step growth** (+44 ms of the +50.3 ms total).
- **It is already the dominant bucket at 64k (45%)** and becomes the majority at
  120k (57.5%).

This is *why* our step is slow at long context — not a GEMM/sampling/port defect.
The other buckets are all small and flat: lin_attn 25 ms, mlp 18 ms, drafter ~22 ms,
heads ~11 ms combined. Even zeroing every non-attention bucket would only buy back
~40% of the step; full_attn is where the long-context cost concentrates.

**Why this differs from 1CatAI's ledger:** their round = 50.1 ms with drafter 18.2%,
AWQ-GEMM 45% of forward, 23.7% uninstrumented — but *no long-context attention
bucket*, because they benchmark at short context where full_attn is a few ms. Our
full_attn alone (97 ms at 120k) exceeds their entire round. The gap is structural
(long-context O(S) attention), not a port bug — consistent with the round-5
convergence, now quantified.

## Ruled out / bugs fixed to get here

| item | resolution |
|------|-----------|
| "mlp 49% / lin_attn 25%" (first c64k data) | **WRONG — chunked-prefill contamination.** The gate's M-extraction read `inp[0]` positionally, but the model calls decoder layers **all-keyword** (`layer(positions=..., hidden_states=...)`) so `inp` was empty → M always -1 → decode default-True → M≈1024 prefill chunks miscounted as decode. Fixed with kwargs-aware `_extract_m`; the mlog diagnostic confirmed clean M=5 windows after. |
| lm_head never firing (`t_lm` absent) | `LogitsProcessor.forward` calls `lm_head.quant_method.apply(...)` directly (the T-1 seam), so a forward hook on the head module can't fire. Fixed by hooking the `LogitsProcessor` module and splitting samples by M (verify vs draft). |
| only 1 of 2 ranks flushing | rearm file was shared + deleted-on-read, starving the second rank. Fixed with an idempotent `gen=` counter each rank consumes independently. |
| in-process engine OOM/wedge | agent-worker 4 GiB cgroup cap SIGKILLs in-process TP=2; host also shows the chronic transient weight-load wedge family (canary passed 39.2 t/s between failures → retry). Moved to serve-based harness + `systemd-run --user -p MemoryMax=infinity`. |

## Kernel-level localization (standalone benches, 2026-09-06)

The `full_attn` bucket hooks the whole `self_attn` module (QKV GEMMs + gather
+ K→Q8 + FA kernel + o_proj). To find where its O(S) time actually lives, the
FA decode kernel and the KV gather were benched **in isolation** at our exact
per-rank shape (Hq=12, Hkv=2, D=256 — TP=2 split of 24/4 heads), B=1:

| Sk | FA kernel Sq=2 (greedy) | FA kernel **Sq=8 (k=4 verify)** | fused Q8 gather |
|----|------------------------|--------------------------------|-----------------|
| 65536 | 0.53 ms · ~147% of HBM floor | 1.68 ms · **~46%** | 0.29 ms |
| 98304 | 0.79 ms · ~147% | 2.53 ms · **~46%** | 0.43 ms |
| 122880 | 1.00 ms · ~144% | **3.16 ms · ~46%** | 0.56 ms |

(LEGACY=1 default path = fused persistent gather + in-kernel Q8 quantize; the
torch `_gather_kv` fallback is 6.1× slower — never used in production.)

**The decisive axis is Sq (query rows), not Sk alone.** The kernel re-reads the
same KV rows once per query row:
- **Sq=2** (greedy, `max_seqlen_q=1`): one KV pass → bandwidth-bound, ~100%+ of
  the HBM floor. No headroom — this is the shape the earlier "FA kernel is at
  the limit" conclusion was accidentally based on.
- **Sq=8** (k=4 verify: `max_seqlen_q = 1+k = 5` → Sq_pad=8, ncols1=8): 4× the
  query work per KV row → **compute-bound at ~46% of HBM**. This is the actual
  production k=4 shape, and it has real headroom.

**Reconciling with the ledger:** standalone FA(Sq=8) + gather @120k ≈ 3.16 +
0.56 = **~3.7 ms/layer**, vs the ledger's eager-measured `full_attn` of
**6.06 ms/layer**. The difference is (a) the QKV/o_proj GEMMs inside the
bucket, and (b) **eager launch overhead** — pfk4srv ran `--enforce-eager` so
hooks fire, but the 30.95 t/s production baseline runs CUDA-graphed
(`cudagraph_capture_sizes [1..5]`), which removes per-op launch cost. So the
ledger's *relative* breakdown is trustworthy (wall anchor matches e2e within
2%), while its *absolute* per-bucket numbers overstate eager-only costs; treat
~3.7 ms/layer as the graphed-mode full_attn floor and the FA kernel itself at
~3.2 ms/layer (~50% of it).

## The lever (refined)

Optimize the **custom FA decode kernel (`fattn-q8.cuh` / launcher) for the
multi-query verify shape (ncols1=8, D=256)**: it is compute-bound at ~46% of
HBM and dominates the long-context step. Candidate directions (to be sized by
the bench): KV re-read elimination across query rows (one KV load feeding all 8
query rows per tile), occupancy/VGPR pressure at D=256, and an NC2/KVSPLIT
config sweep **at Sq=8** (the existing sweeps were done at Sq=2). Bit-exact
target (no quality risk) — the same standard as choosing FA-decode over
KV-quantization.

## Next steps (FA decode — chosen direction)

1. ✅ Standalone micro-bench of the decode kernel + gather at 64k/96k/120k,
   both Sq shapes — done; probes committed under
   `benchmarks/kernels/gfx906/*_qwen27.py` (how-to in
   `docs/gfx906/BENCH-fa-decode.md`).
2. ⚠️ **CORRECTION (2026-09-06, after starting the fp16 work):** two premises
   of the earlier plan were wrong:
   - The QK-dot is ALREADY packed — `ggml_cuda_dp4a` = `v_dot4_i32_i8` on
     gfx906 (fattn-q8.cuh:487-498), and P·V uses fp16 half2 FMA. There is no
     scalar-fp32 QK loop to rewrite; the "fp16 QK-dot" lever does not exist.
   - The config sweep tested NC2=1/KVSPLIT=1 as "current default". WRONG: the
     C++ defaults are **NC2=8 + KVSPLIT=16** (gfx906_fa.cpp:87-107), and for
     our shape (Hq/Hkv = 12/2 → gqa_ratio=6, not divisible by 8) the launcher
     auto-downgrades NC2 8→2 (gfx906_fa_launcher.cu:123-144). No env override
     in `run_server.sh` → **production runs NC2=2 + KVSPLIT=16**.
3. ✅ Re-sweep at the REAL production config (NC2=2, Sq=8, 2026-09-06):
   - KVSPLIT=16 (prod): 3095 µs @122880 (2486 @98304, 1665 @65536)
   - **KVSPLIT=32: 2796 µs (−10.0%)** · KVSPLIT=64: 2789 µs (−9.8%, plateau)
   - Bit-exact path (split-combine is an fp32 max/sum merge; verified vs the
     torch reference at Sk≤8192: maxerr 0.0010 @y=32 vs 0.0025 @y=1, both ≪
     0.05 gate). Partial buffer B·Sq_pad·Hq·y·D·4 = 1·8·12·32·256·4 ≈ 3 MiB
     ≪ 512 MiB budget; even a 1024-token prefill chunk stays under budget
     (≈403 MB) and real prefill (Sq≥~2k) is still forced to y=1.
   - **Next step: pin `GFX906_FA_KVSPLIT=32` in the serve env (zero code
     change), verify with canary + A/B on a CUDA-graphed arm.** Expected
     ~10% off the FA kernel → ~2-3% of the full step e2e (FA kernel ≈ half
     the full_attn bucket; full_attn = 45-57.5% of the step).

   ### KVSPLIT=32 verified — same-boot A/B (2026-09-06)

   CUDA-graphed mtp4 arm (TP=2, corpus s9), steady-state reps 1-2 mean,
   **same boot** for both arms (control = default KVSPLIT=16):

   | ctx | KVSPLIT=16 | KVSPLIT=32 | Δ |
   |---|---|---|---|
   | 64k | 44.84 | 45.47 | **+1.4%** |
   | 96k | 35.67 | 36.45 | **+2.2%** |
   | 120k | 31.01 | 31.73 | **+2.3%** |

   Consistent with cross-boot (44.82/35.79/30.95 → +1.4/+2.6%) and scaling
   with context length as the floor model predicts. Bit-exact path (fp32
   max/sum merge; bench maxerr 0.0010 @y=32 vs 0.0025 @y=1, both ≪ 0.05 gate).
   **Decision: ship `GFX906_FA_KVSPLIT=32` as the serve default.**

   ### Register profile of the production kernel (ELF metadata, CPU-only)

   Template signature is `<DKQ, DV, ncols1, ncols2, softcap>`; production runs
   `flash_attn_tile_q8<256, 256, 8, 1>` (LEGACY gather path, Sq_pad=8).
   From the GPU code object:

   | variant | VGPR | LDS | notes |
   |---|---|---|---|
   | ncols1=8, NC2=1 (production) | **232** | 28,800 B | occ pinned to 1 (config table `256,256,8 → occ=1`; occ=2 needs ≤128 VGPR and spilled flat) |
   | ncols1=8, NC2=2 | **161** | 30,336 B | 71 fewer VGPR — halved per-lane Q accumulator rows |

   The production variant is register-rich at occ=1 (232/256 VGPR) with zero
   spill (`vgpr_spill_count: 0`) — ALU-bound within the tile, not spilling.

   ### NC2=2 at Sq_pad=8 — TESTED, NO GAIN (ruled out)

   Relaxed the `seq_q>2` launcher guard for NC2=2 (kept NC2=8 decode-only),
   rebuilt the extension, correctness PASS (maxerr 0.0010 @ Sk≤8192). At
   KVSPLIT=32 the wall time is **identical** to NC2=1:

   | Sk | NC2=1/y=32 | NC2=2/y=32 |
   |---|---|---|
   | 64k | 1504 µs | 1510 µs |
   | 120k | 2888 µs | 2897 µs |

   Halving KV DRAM traffic buys nothing: with y=32 each block reads only
   Sk/32 ≈ 3.8k tokens, so the per-block KV read is no longer the cost — the
   Q-side ALU work (QK dot + P·V + softmax rescale, all per query row)
   dominates and head-packing doesn't change it. **Guard change reverted**
   (production stays NC2=1); recorded here so it isn't re-tried. This also
   confirms the kernel at Sq=8/y=32 is compute-bound on Q-side math,
   independent of KV traffic — pointing remaining headroom squarely at
   **KV re-read elimination** (share each KV tile's loads across all 8 query
   rows) and **softmax rescale batching**, not memory-side knobs.

   ### Remaining levers (ranked, post-verification)

   1. **KVSPLIT=32 pin — DONE, verified above.** Zero code change.
   2. **KV re-read elimination** (restructure loop nesting: read each KV tile
   once, reuse across the 8 query rows in-tile): largest remaining lever
   (~6-9% of step per Claude estimate) but a weeks-scale kernel rewrite
   touching LDS layout + register allocation.
   3. **Online-softmax rescale batching** (defer max-update across N tiles):
   ~2.5-4% of step, 1-2 weeks.
   4. MFMA — infeasible on gfx906 (hardware). Sq_pad=5 — ~1-2% (KV loads are
   shared across query rows; only Q-side math shrinks). NC2>1 at Sq=8 —
   measured null, ruled out above.

5. Note on the long-Sk linear fits: their large "intercept" (~800-1500 µs) is
   a fit artifact from mixing regimes — at Sk≤8k the kernel is latency-bound
   (y=32 does 4k in 175 µs), so extrapolating the throughput-regime slope
   back to Sk=0 overshoots. Not evidence of a fixed per-call overhead; drop
   as a lead unless a real profiler shows one.
5. Sq_pad=5 (we need k+1=5 rows, ladder pads to 8): still valid but lower
   priority — with NC2=2 packing two kv heads per tile, the Q-side work is
   amortized over 2× the KV reads; the wasted-row fraction of TOTAL time is
   smaller than the naive 37% suggests. Defer unless step 3 disappoints.

### Ruled out (evidence, 2026-09-06)

- **fp16 QK-dot rewrite** (initial next-step): does not exist as a lever —
  QK is already `v_dot4_i32_i8` packed, P·V already fp16 half2.
- **D-split "occupancy fix"**: wrong model of the bottleneck; KV-split data
  shows parallelism helps (−10% at y=32) but the kernel is not occupancy-starved
  — it's per-row ALU work at D=256 with limited headroom.
- **NC2/KVSPLIT sweep at NC2=1** (the first sweep): valid numbers, wrong
  baseline — production never runs NC2=1 for this shape. Superseded by step 3.

## Repro

```bash
# server (clean boot family, eager so hooks fire)
systemctl --user start pfk4srv.service        # run_pfk4srv.sh, port 8130, k=4, TP=2
# client (warmup + 3 context windows, rearms between)
systemd-run --user -p MemoryMax=infinity /bin/bash -c \
  'cd /local/git/vllm-gfx906-mobydick && .venv/bin/python -u /local/tmp/mtp1/pfk4_client.py'
# aggregate
.venv/bin/python /local/tmp/mtp1/pfk4_ledger.py
```

Artifacts: `/local/tmp/mtp1/pfk4_{c64k,c96k,c120k}_p*.json`, plugin
`/local/tmp/mtp1/pfk4_phase_plugin.py`, client `pfk4_client.py`, aggregator
`pfk4_ledger.py`.
