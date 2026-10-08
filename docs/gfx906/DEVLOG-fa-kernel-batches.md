# FA kernel batches — ISA-rate refutation (M5), hygiene batch (M3), per-q-tile prefill clip (M2)

**VERDICT:** M5 "salvage LEGACY=0 with a better dot" DEAD-END (analysis-only) ·
M3 SHIPPED (hygiene; dot2 P·V rewrite DEAD-END) · M2 SHIPPED (+11.8 % wall
@pp16384 windowed) · **GATE:** unit suite for hygiene + same-boot kernel/e2e
prefill A/B for M2 (no serving-decode claim in this log)
**Branch:** `gfx906/fa-decode-fp16` · 2026-08-28/29
**Full detail:** `git show e963fd8c62:docs/gfx906/DEVLOG-fa-attention.md`
(entries of 2026-08-28/29, pre-split); ISA rates live in
`dequant-instructions.md` ("Measured dot-instruction rates").

## 2026-08-28 — M5 premise refuted: the Q8 KQ dot is *already* the chip's fastest dot

**HYPOTHESIS.** M5 read the LEGACY=0 loss as architectural — "gfx906 has no
int8 matrix path, so the Q8 dot runs as fp32 ALU where fp16 gets FMA" — and
proposed salvaging LEGACY=0 with a better dot (`dp4a`, `v_dot2_f32_f16`,
`v_dot8_i32_i4`).

**Refuted on both halves (probe + roofline, no kernel code touched):**

1. `ggml_cuda_dp4a` → `__builtin_amdgcn_sdot4` → **`v_dot4_i32_i8` is already
   in use** in the KQ loop (8 dp4a per 32-element block). Nothing to swap in.
2. **dot4 is full-rate on gfx906**: 4 int8 MAC/lane/cycle = 4.44× fp32 FMA =
   2× packed fp16 (25.9 T MAC/s ≈ AMD's 53 TOPS INT8). dot8 also full rate
   (8.52×); the "expansion composite" paths are dead (0.17×/0.24×). *Launch-
   regime evidence by construction* (pure-ALU probe).
3. **Roofline (D=128, B=1 decode):** the gather moves 512 B/row (LEGACY=1:
   K fp16 256 + V 256) vs 392 B nominal (LEGACY=0: K q8 136 + V 256); ALU is
   ~96 pipe-cycles/row. Machine balance 800 GB/s ÷ ~5.8 T lane-cycles/s ⇒ the
   read path is **HBM-bound by ~2.7×** — no instruction substitution can
   surface at B=1.
4. **Where the loss actually comes from (read layout, code-level):** the Q8
   alias packs 4×34 B q8_0 blocks into the first 136 B of every 256-B fp16 K
   row. The fused-Q8 gather reads 136 of 256 B — 5 sectors per row for 136 B
   useful (≤85 % efficiency; 416/512 = 1.23× effective lean, not the nominal
   1.31×) plus per-token tail handling. LEGACY=1 reads 16 aligned uint4 and
   quantizes: more bytes, more ALU, clean bursts. At B≥2, direct-paged reads
   the same misaligned slices with per-row indirection → the −27…−31 %.

**VERDICT: DEAD-END** for "salvage via a better dot instruction" (analysis-only,
nothing reverted). The M5 *decision* (LEGACY=1 stays the default) is untouched
and later confirmed same-boot at B=1 (−6.3 %) in
`DEVLOG-fa-legacy0-b1-decode.md`. **M6 reframed:** the per-block rescale tax
cannot be the B=1 cause (ALU is off the critical path); the deficit is the
aliased-Q8 read layout. Salvage paths are layout work (aligned quants / scale
planes) or B≥2 via gather; the only instruction-level upside left is a **Q4-KV
format** that unlocks native `v_dot8_i32_i4` — roadmap M6, PPL-gated.

## 2026-08-28/29 — M3 hygiene batch SHIPPED; the dot2 P·V rewrite DEAD-END

**HYPOTHESIS.** #8 a negative `kv_start[sequence]` can walk the k-loop into
token-negative space (illegal access, not a wrong number); #10 the window
cutoff wraps for absurd windows; #4b the `[B,Sq,Hq,2]` `o_meta` buffer is
allocated even when it is dead; and the P·V `v_dot2_f32_f16` rewrite halves the
P·V instruction count.

**Shipped:**
- **#8** device-side `k0_base = max(0, kv_start[sequence])` at both LOCKSTEP
  entry sites — closes a real latent wedge class.
- **#10** all four cutoff sites → `q_abs_row - k_pos_abs >= window` (both
  operands non-negative ⇒ wrapping-free by construction). Note: the roadmap's
  "overflows for absurd windows" is not literally true for int32 (min
  INT_MIN+2); the new form is kept for proof-carrying clarity.
- **#4b** final form: pass `nullptr` at `kv_split==1`, allocate only
  `o_meta_split` (the reviewer's inversion claim was refuted against the code:
  the only `dst_meta` write is guarded by `gridDim.y != 1`, and at
  `kv_split>1` the live buffer is `o_meta_split`, which #4b never removed).
  Drops a ~300 KB/layer prefill-sized dead allocation.
- **Tests (3 functions, 5 parametrized cases):** `window=INT_MAX` bit-identical
  to plain causal (pins the new form's inertness — the rewrite's equivalence
  rests on the algebraic proof + the existing unaligned bit-identity suite);
  `kv_start=-L` bit-identical to `kv_start=0` (pre-#8 this test risks *wedging*
  the GPU rather than failing); amplified-V boundary test (first out-of-window
  key's V ×400 → wrong-cutoff outputs are ~1.0 away from the shifted reference).

**Refuted — dot2 P·V (not implemented).** The production build's P·V region is
in-place fused **`v_pk_fma_f16 vd, v, p, vd`** (1024× in the NC2 prefill
instantiation vs 54× `v_pk_mul_f16` for QK-path dequant scales and **0×
`v_pk_add_f16`**), i.e. already 1 instruction per 2 MACs at full packed rate —
the same as `v_dot2_f32_f16`. The rewrite buys zero instruction count and zero
rate, only fp16→fp32 accumulation, which would need P·V row-ownership
restructuring. `dequant-instructions.md`'s P·V paragraph is corrected/marked
SUPERSEDED. If revisited it is a **precision** change, not a perf one.

**Gates.** Hygiene: 65/65 → 70/70 after the merge with main (60 base + 5 M2 + 5
M3); no perf claim, no serving slot (hygiene rides along by rule).

**Trap recorded (cost a session):** the first M3 build linked a **stale
`gfx906_fa_quant.hip.o`** — hipify reported "[skipped, already hipified]" for a
`.cu` whose build-tree `.o` came from another branch, producing a Frankenstein
`.so` (planar quantizer + main FA kernel) with all-NaN FA output while the
sources on disk were consistent. **Countermeasure: after switching branches
that touch `csrc/`, delete the extension's build state** (`build/temp.*/CMake
Files/_gfx906_fa_C.dir` + the hipified `csrc/gfx906_fa/*.hip` and `kernel/*.cuh`
copies), or the hipify skip-check will lie to you.

## 2026-08-28 — M2 per-q-tile prefill clip SHIPPED (2.8–3.2× kernel, +11.8 % e2e wall)

**HYPOTHESIS.** If the skipped k-tiles are provably fully masked (window or
causal) for every row of a q-tile, then moving each q-tile's scan to its own
window start and capping it at its own last row is bit-identical and cuts the
sliding-window prefill scan from ~chunk to ~window per q-tile.

**Shipped** (`GFX906_FA_TILE_CLIP`, default 1; `0` = A/B arm):
(1) per-q-tile window raise `k0_base = max(k0_base, floor16(q_abs + col_Q_0 +
1 - window))` (clip mode only); (2) per-q-tile causal cap `k_VKQ_max =
min(k_VKQ_max, q_abs + min(col_Q_0 + ncols1, ne01.z))`. The backend's
DIRECT_PAGED window clip gate dropped `max_seqlen_q == 1`, so **prefill chunks
now get the clip on both paths**.

**Evidence (kernel-level, in-process, same boot):**

| shape | gather-path | direct-paged |
|---|---|---|
| A windowed (L=131072, Sq=4096, W=2048) | 62.286 → 19.519 ms = **3.19×** | 83.791 → 29.892 ms = **2.80×** |
| B causal-only first chunk (L=Sq=4096, W=0) | 43.218 → 19.465 ms = **2.22×** | 61.246 → 31.317 ms = **1.96×** |

Theory matches: without M2 a q-tile scans 6143 keys; with M2 ≈2047 (3.0×).
Cross-path bit-identity: DIRECT_PAGED clip-on vs LEGACY clip-on `max|diff| = 0.0`.
Decode is provably unchanged (cap = seq_len; raise = the existing floor).
Suite 64/64 → 65/65 with the paged cap arm.

**Review-fix e2e gate (2026-08-29,** review's F1 condition: one e2e A/B covering
both halves; the 130k e2e was re-scoped off after a boot-M wedge + ~2× slow
prefill — covered by bench shape A):**

| e2e (in-process, 2 samples/arm, tg=256) | clip=1 | clip=0 | delta |
|---|---|---|---|
| Muse pp16384/B=2 (windowed, both bounds) | 134.094/134.401 s | 152.007/152.387 s | **+11.8 % wall / +14.8 % prefill** |
| Qwen3.8-27B pp2048 (full-attn layers, cap only) | 15.817/15.848 s | 15.931/15.966 s | +0.73 % e2e (both samples agree) |

**Erratum (boot N):** the "~2× slow" prefill that made boot-M absolutes
suspect was in fact the **true TP=1 rate** (135.48 + 136.51 s for the same 32k
shape, 0 wedges) — the ~450–540 t/s records are TP=2. The A/B ratios stand.

**VERDICT: SHIPPED** (in main since 2026-08-29; F1 satisfied before merge;
F2 paged cap test, F3 the deliberate non-memoization of the per-call env read,
F4 canary record, F5 README env rows all closed). The causal cap is a
**general chunked-prefill win** for any model (window-independent), called out
as such in the README. Kill switches: `GFX906_FA_TILE_CLIP=0`,
`..._WINDOW_CLIP=0`, `GFX906_FA_GATHER_CLIP=0`.

## Cross-links

- KVSPLIT shape-aware default, the Sq=8 verify deep-dive, R3 →
  `DEVLOG-fa-verify-sq8.md`.
- LEGACY=0 flip adjudication (the M5 decision's gate) →
  `DEVLOG-fa-legacy0-b1-decode.md`.
- Split-K accuracy + tile-clip test rework → `DEVLOG-fa-splitk-accuracy.md`.
- Backend/gather-track history → `DEVLOG-fa-attention.md`.
