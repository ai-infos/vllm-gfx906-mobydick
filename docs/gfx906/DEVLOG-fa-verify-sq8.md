# FA verify-shape deep-dive (Sq=8) — KVSPLIT shape-aware default shipped; every other config lever closed

**VERDICT:** KVSPLIT shape-aware default SHIPPED (−10.6 % kernel @Sq=8; +1.4–2.3 %
serving) · pad-row skip NO-GO (+5 % regression) · NC2=2 NO-WIN · native
ncols1=5/6 tiles NO-WIN (spill 171/167) · dual-tile **+29 % worse** · MFMA /
fp16 QK-dot / Hq=24 all NO-GO · R3 paged-direct default alignment SHIPPED
· **GATE:** standalone kernel bench (Sk=122880) + same-boot serving A/B
(mtp4, CUDA-graphed)
**Branch:** `gfx906/fa-decode-fp16` · 2026-09-06 · HEAD at write `dbc081eff6`
**Full detail:** `git show e963fd8c62:docs/gfx906/DEVLOG-fa-attention.md`
(2026-09-06 entries, pre-split) · `PROFILE-k4-step-ledger.md`

**Context.** The k=4 step ledger put full attention at **57.5 % of the decode
step @120k**, with the FA decode Sq=8 (verify) kernel at ~3 ms — 3× the Sq=2
cost. This log is the full option space explored for that kernel: shipped,
measured-dead, and what is left (structural only).

## SHIPPED — shape-aware KVSPLIT default (`dfed62f133`)

`GFX906_FA_KVSPLIT` flat default 16 → `fa_kv_split_default(seq_q)`: **32 for
Sq≥4, 16 for Sq=2** (env override unchanged).

| evidence | Sq=2 | Sq=8 |
|---|---|---|
| standalone, NC2=1, Sk=122880, y=16 vs y=32 | 1644 vs 1707 µs = **+3.8 % slower** at y=32 (Sq=2 is ~88 % HBM-bound; split-combine traffic exceeds what the split saves) | 3094 vs 2765 µs = **−10.6 %** at y=32 |

Same-boot serving A/B (mtp4, graphed): **+1.4/+2.2/+2.3 % @64/96/120k**;
split merge is bit-exact (maxerr ≤ 0.0025); suite 85/85.

## CLOSED NO-WIN — pad-row QK^T/softmax/P·V skip (the "KV re-read" work)

**Premise corrected twice:** (1) "KV is re-read per query row" is **FALSE** —
K loads DRAM→LDS once per tile, V once per sub-tile, and every row reads LDS;
no redundant KV movement exists. (2) The MTP pad rows *are* computed (no
valid-row guard in the compute loops; only the epilogue store skips), but the
cost is not what the estimate assumed.

**Reframing measurement** (standalone, Sk-fit slope, ns/token): Sq=2 6.09 ·
Sq=4 8.64 · Sq=5/6/8 **~19** — i.e. **Sq=6 ≈ Sq=8 (+0.8 %)**: inside an
ncols1=8 tile, row count does not move wall time (per-row ALU is latency-hidden
under the KV pipeline at occ=1/np=1).

| build (Sq=5 @120k, 3 pad rows) | µs | vs original |
|---|---|---|
| original (no skip) | **2785** | — |
| QK^T-only skip | ~2860 | +2.7 % |
| full skip (all stages) | ~2924 | **+5 %** |

**NO-GO:** the branch guards in the unrolled hot loops cost more than the
off-critical-path work they remove (an independent review predicted 25–30 %
savings — refuted). All kernel changes reverted. **The Sq=4→Sq=8 2.2× jump is
a tile-config cliff, not pad waste** (below).

## CLOSED NO-WIN — NC2=2 at Sq=8 (guard relaxed to sq≤8)

Correctness PASS (maxerr 0.0010), wall **identical** to NC2=1 at y=32
(1504 vs 1510 µs @64k): halving KV DRAM traffic buys nothing once the KV range
is already split (each block reads ~3.8k tokens; Q-side ALU dominates).
Register profile (GPU ELF metadata): NC2=1 VGPR 232/256, LDS 28.8 KB, 0 spill;
NC2=2 VGPR 161 — no spill risk either way. Guard reverted.

## CLOSED NO-GO — MFMA / fp16 QK-dot / head parallelism

MFMA is infeasible on gfx906 (Vega 20 predates MFMA; gfx908+). The QK dot is
already packed `v_dot4_i32_i8` (nothing to rewrite — see
`DEVLOG-fa-kernel-batches.md` M5). An Hq=24 head-parallelism probe was flat.

## CLOSED NO-WIN — native ncols1=5/6 tiles (the config cliff)

**Why it looked promising:** the cliff is a config effect (fattn-q8.cuh L87-95)
— `(256,256,4)` runs occ=**2** (nbatch_fa 64), `(256,256,8)` runs occ=**1**;
Sq=5 pads to ncols1=8 and pays the 2.2× slope.

**Structure is feasible at non-power-of-2 ncols:** row mapping
`j = (jc0 + (threadIdx.y/np)*cpw)/ncols2` with `cpw = ncols>nwarps ?
ncols/nwarps : 1`, `np = nwarps>ncols ? nwarps/ncols : 1` is clean iff
`(nwarps//np)*cpw == ncols` and np is a power of 2 → **ncols=5 @160 thr
(5 warps)** and **ncols=6 @192 thr (6 warps)** both give np=1 (no cross-warp
max exchange); the 320-thread variants are broken or force the exchange;
`nbatch_fa ∈ {64,128}`; the tile loaders are grid-stride and tolerate
non-power-of-2 nthreads.

**Built and benched — the register gate fired:**

| variant | VGPR | LDS | spill | wall @120k | vs baseline 2776 µs |
|---|---|---|---|---|---|
| ncols1=4 (occ=2 ref) | 126 | 29136 | 0 | — | Sq=4 = 8.64 ns/token |
| **ncols1=5** (160 thr, occ=2) | **128** | 29536 | **171** | 2769 / 2779 µs | flat (±0.3 %) |
| **ncols1=6** (192 thr, occ=2) | **128** | 30336 | **167** | 2788 / 2783 µs | flat (±0.2 %) |
| ncols1=8 (occ=1, production) | 232 | 18944 | 0 | 2776 µs | — |

Both new tiles land *exactly* on the occ=2 VGPR ceiling (128) by **spilling
~170 registers** (every other variant in the family spills 0) — the occupancy
win is entirely eaten by spill traffic. **Bonus finding:** this resolves the
non-monotonic VGPR data (ncols1=8 → 232 but ncols1=16 → 163) that made
register prediction useless for the interpolated 5/6 case — per-warp state
scales with **cpw**, not ncols; 5/6 sit at the boundary and spill.

**Also measured dead — dual-tile (Sq=5 as 2×ncols1=4, occ=2):** correctness
PASS (maxerr 0.0012) but **+29 % worse** (3588 vs 2776 µs @120k) — each q-tile
independently streams the full KV range, so grid_x=2 doubles KV DRAM traffic.
**That is the real "KV re-read" in this kernel and it is now measured dead at
long context.** (`GFX906_FA_NC1_OVERRIDE` + the `fa_pick_ncols1` host mirror
are the reusable scaffolding if a future tile variant is tried — re-add from
the pre-split log. All experiment code reverted.)

## Open — structural only (from the review, ranked)

1. **KV re-read elimination across q-tiles** — ~6–9 % of step; weeks-scale
   rewrite (ref llama.cpp/llaminar FA). Only pays when grid_x>1, i.e. moot at
   Sq=5 until a tile change lands.
2. **Online-softmax rescale batching** — defer the max update across N KV
   tiles; ~2.5–4 % of step, 1–2 weeks.

## SHIPPED — R3 paged-direct KVSPLIT default aligned (`fa7e1e20b9`)

The direct path defaulted to `clamp(16/batch, 2..8)` for **every** shape while
the gather path had moved to the shape-aware rule — flipping `GFX906_FA_LEGACY`
1→0 would silently have moved the Sq≥4 verify shapes from split 32 to ≤8 (a
measured ~10 % FA regression). New `fa_paged_kv_split_default(seq_q, batch)`:
shape-aware for Sq≥4, and for Sq<4 the direct path's **own** measured batch
clamp (8/8/5/2/2 for B=1/2/3/4/8, MI50 micro-bench in `DEVLOG-muse-glimmer.md`)
— its grid is B·heads_q blocks at NC2=1, so the gather path's 16 is the wrong
value for its decode shapes. Pinned by the pure probe test
`fa.kv_split_default(seq_q, batch, direct)` (per-call env read, unlike the
launch-site static). Suite 89/89. **VERDICT: SHIPPED; n/a under LEGACY=1** —
production behavior is byte-identical; the change protects a future LEGACY=0
flip.

## Net position

Every **config-level** lever on Sq=5/8 verify FA is now measured
(KVSPLIT shape-aware = shipped, −10 % already captured; pad-row skip, NC2=2,
dual-tile, native ncols1=5/6 all no-win). Remaining headroom is structural
items 1–2 above. **The k=4 ledger's `full_attn` bucket should be re-baselined
after the KVSPLIT default ships in a serve A/B.** Kill switches:
`GFX906_FA_KVSPLIT`, `GFX906_FA_NC2`.
