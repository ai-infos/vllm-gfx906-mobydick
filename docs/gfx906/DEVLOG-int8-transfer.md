# INT8 transfer from the gfx908 fork — what survives on gfx906

> Branch `main` · models Qwen3.5-27B-AWQ (dense), Qwen3.5-35B-A3B-AWQ (MoE) ·
> dates 2026-08-30 → 2026-08-31 · roadmap `I1`–`I5` (proposed) ·
> analysis + probes, **no engine code changed**.

**VERDICT:** `OPEN` — the byte-side item (int8 the BF16 weight mass, **T1**)
passed its probe gate and awaits the serving A/B that is the real gate; the
Triton A8W8 GEMM item is `DEAD-END` (below); the hand-kernel successor
question resolves to `v_dot8_i32_i4`, not to int8.

**GATE (for T1, still ahead):** serving graph, `_bench_gfx906.py`, dense
4-seq pp2048/tg256 + MoE 32-seq, both arms same boot. Every number in this log
is **launch-regime evidence** (standalone probes) except where a recorded
serving number is quoted for comparison.

Analysis and full surface list: `int8-investigation-qwen.md` (T0–T5, §6 has
the probe records). This log is the experiment record only.

---

## 2026-08-30 — how much of *our* decode traffic is unquantized?

### HYPOTHESIS

If the gfx908 fork's int8 wins came from int8 *arithmetic*, they need
MFMA-class hardware and die on gfx906; if they came from *bytes*, they
transfer wherever our own checkpoints still ship fp16 weights.

### What was done

Safetensors-header scan (dtype × shape × count) of both served checkpoints
plus their `quantization_config`; per-token MAC and byte mass computed for
the prefill view (all experts touched) and the decode view (top-k 8/256);
`modules_to_not_convert` read to see what is quantizable at all. Fork side:
`INT8_AUDIT_RESULTS.md`, `docs/recipes/README.md`,
`docs/MI100_Optimization_Attempts.md`, `triton_w8a16.py`, `aiter_w8a16.py`,
`act_quant_rn.py`, the int8 embedding/lm_head commits, `kld_probe_v2.py`,
P82 lossy-acceptance record.

### Evidence — FOR

- Dense 27B per decode token: **5.71 GB BF16** (lm_head 2.543 + FA q/k/v
  2.349 + layer 0 0.767 + `in_proj_a/b` 0.047), **6.56 GB with the MTP draft
  layer**. At 798 GB/s = 7.17 ms/step.
- MoE 35B: **3.88 GB/token** (lm_head 1.017 + GDN in_proj_qkv/z + out_proj
  2.013 + FA q/k/v/o 0.545 + shared expert 0.252 + BF16 layer-0 experts
  ~0.05) = 4.86 ms/step.
- Reconciles with the recorded floors: analytic lm_head 3186.5 µs vs recorded
  3114–3193 µs (98–101 %); MoE floor table sum 4.80 ms vs analytic 4.86 ms.
- **60.3 %** of the MoE decode weight *bytes* are BF16 (vs 8.9 % of its
  prefill MACs) — a byte-side lever with no prefill-side equivalent.

### Evidence — AGAINST / corrections found in our own records

- The fork's entire GEMM/attention/AR layer is aiter + Composable-Kernel
  (MFMA-shaped) → **zero code transfer**; only doctrine, sampler, KV format
  and methodology move.
- `mtp.*` ships **entirely unquantized** (0.849 GB dense: q/k/v/o 0.210 +
  `fc` 0.105 + MLP 0.535) and is read every MTP step — no existing budget
  line accounts for it (`int8-investigation-qwen.md` §7 C-3).
- Both checkpoints carry a **0.92 GB BF16 vision tower** in the same shards
  under a `*ForConditionalGeneration` arch (worth confirming it is not
  resident in text-only runs).
- `modules_to_not_convert` differs sharply by model: dense leaves q/k/v +
  `layers.0.*` + `mtp` unquantized; MoE leaves **all** `linear_attn`,
  `self_attn`, `shared_expert`, `mlp.gate` + layer 0 + mtp unquantized. Any
  "the fp16 mass" claim must name the model.

### Why the fork's number does not transfer

MI100 got int8 from MFMA (374 TOPS vs 184.6 TFLOPS fp16) *and* a tuned CK
library. Our equivalent arithmetic (dot4) is 1.96× packed fp16 but has no
library behind it, so int8 only pays where we are byte-bound — which fp16
weights, by definition, are.

**VERDICT:** `OPEN` (analysis; feeds T1/`I1`, no code).

---

## 2026-08-31 — P1: Triton A8W8 GEMM (int8 act × int8 weight)

### HYPOTHESIS

If Triton lowers int8 `tl.dot` to `v_dot4_i32_i8` on gfx906 and dot4 is 1.96×
packed fp16, then an A8W8 prefill GEMM beats our fp16 prefill GEMM by
**≥ 1.3× at [4096, 34816, 5120]** with the activation-quant pass charged.

### GATE

Standalone Triton GEMM vs the same-codegen fp16 Triton GEMM and vs hipBLAS
fp16 at five production prefill shapes (**launch-regime evidence by design —
a kernel-level NO kills the item before any serving time is spent**). Probe:
`benchmarks/kernels/gfx906/int8_triton_dot_probe.py`, log
`/local/tmp/int8-probes/p1-full.log` (torch 2.13.0+gfx906, Triton 3.6.0,
`HIP_VISIBLE_DEVICES=0`, GPU idle).

### What was done

ISA dump of the emitted `amdgcn` (asm dict, with cache-scan + `llvm-objdump`
fallback); exactness of int32 accumulation vs an fp64 reference at 64³/32³;
int8-vs-fp16 `tl.dot` rate sweep over three tiles; the gfx908 blockscale A8W8
kernel re-expressed with dense int8 weights (per-token A scale × per-128
weight-group scale, one descale per tile) vs fp16 Triton vs hipBLAS at
M ∈ {4096, 1024, 256}; act-quant rounding (trunc / `floor(x+0.5)` /
`tl.extra.hip.libdevice.round`) with payload-disagreement counts.

### Evidence — FOR

- `v_dot4_i32_i8` **is emitted** (16 occurrences; zero `v_mac_f32`, zero
  `v_fma_f32`), and the A8W8 GEMM emits **2048 `v_dot4` against 4096
  `v_dot2_f32_f16`** for the same-work fp16 kernel — the 2:1 the ISA ceiling
  predicts. No spills.
- Accumulation **exact** (maxdiff 0 vs fp64) — the "one descale per tile"
  structure costs nothing in correctness.
- Act-quant is not the problem: 0.13 ms at M=4096 = **0.14 %** of the GEMM.
- Best int8/fp16 tile ratio 2.51× (> the 1.96× ceiling ⇒ the fp16 comparator
  was latency-stalled, not issue-bound).

### Evidence — AGAINST (decisive rows)

| M×N×K | A8W8 (+quant) | Triton fp16 | hipBLAS fp16 | A8W8/hipBLAS |
|---|---|---|---|---|
| 4096×34816×5120 | 145.75 ms | 167.35 | **96.24** | **0.66×** |
| 4096×5120×17408 | 66.93 | 78.73 | **43.75** | 0.65× |
| 4096×14336×5120 | 60.13 | 69.07 | **39.80** | 0.66× |
| 1024×34816×5120 | 35.55 | 39.10 | **24.17** | 0.68× |
| 256×34816×5120 | 10.56 | 12.35 | **6.26** | 0.59× |

Only **1.10–1.18×** over the same-codegen fp16 kernel (gate needed 1.30×).

### Why it failed

Triton reaches **5.0 T MAC/s = 19 %** of the 25 877 GMAC/s dot4 record, while
hipBLAS fp16 reaches 7.57 T = **57 %** of its record. The compiler deficit
(1.5–1.7×) is larger than everything int8 arithmetic can buy, so dot4's edge
is spent catching up instead of moving the model. The instruction set was
never the constraint on this path.

**VERDICT:** `DEAD-END` (Triton route). Nothing to revert — probe scripts are
the only artifact and stay in `benchmarks/kernels/gfx906/` as the evidence.
Indexed in `DEAD-ENDS.md` (`INT8`).

Side results worth keeping: `tl.extra.hip.libdevice.round` **is** available on
our Triton/ROCm 7.14 (contradicting the fork's "no rint on HIP" workaround
note) and costs +12 % over `floor(x+0.5)`, disagreeing with it on 0.007 % of
elements (round-half-even at exact `.5`, so the fork's "bit-identical" is
0.007 % off); truncation disagrees on **49.7 %** of the payload — a correctness
issue in any in-tree truncating act-quant, not a perf one.

---

## 2026-08-31 — P2: int8-weight GEMV (W8A16) at the BF16 mass

### HYPOTHESIS

If our fp16 GEMVs are at the HBM floor, then halving their bytes converts
almost 1:1 into decode time: **≥ 1.7× at lm_head and ≥ 1.6× at the ×30 GDN
shapes** for int8 weights + per-channel scale, fp16 activations.

### GATE

Probe = launch-regime gate for spending the implementation budget.
**Shipping gate is the serving A/B** (top of file) and has *not* been run.
Probe: `benchmarks/kernels/gfx906/int8_gemv_probe.py`, log
`/local/tmp/int8-probes/p2-full.log`.

### What was done

M=1 Triton GEMVs — fp16, int8 + per-output-channel scale (scale applied after
the reduction), int8 + per-128-group scale, and an int8 `tl.dot` variant — at
24 shapes whose per-step counts come from the checkpoint scan (incl. layer 0
and the MTP draft legs); baselines `torch.mv` (hipBLAS) and the recorded HBM
floor; correctness spot-check per shape (rel-err ≤ 3 × 10⁻⁴).

### Evidence — FOR (launch-regime)

| shape (N×K) | ×/step | fp16 µs | int8 µs | ratio | int8 % of floor |
|---|---|---|---|---|---|
| 248320×5120 lm_head dense | 1 | 3315.7 | **1716.9** | **1.93** | 93 % |
| 248320×2048 lm_head MoE | 1 | 1316.8 | 712.6 | 1.85 | 89 % |
| 12288×5120 FA q_proj dense | 16 | 224.8 | 112.5 | 2.00 | 70 % |
| 10240×5120 L0 in_proj_qkv | 1 | 169.8 | 84.8 | 2.00 | 78 % |
| 17408×5120 L0/mtp mlp gate,up | 2+2 | 274.0 | 134.9 | 2.03 | 83 % |
| 8192×2048 GDN in_proj_qkv MoE | 30 | 134.5 | 55.5 | **2.42** | 38 % |
| 8192×2048 FA q_proj MoE | 10 | 126.3 | 52.5 | 2.40 | 40 % |
| 5120×17408 L0/mtp mlp down | 1+1 | 290.9 | 191.6 | 1.52 | 58 % |
| 4096×2048 GDN in_proj_z MoE | 30 | 143.4 | 94.3 | 1.52 | 11 % |
| 512×2048 shared gate/up, k/v MoE | 80+10 | 14.4–14.8 | 14.7–15.1 | 0.96–0.99 | 9 % |

- lm_head int8 (1716.9 µs) vs the **recorded production fp16 floor**
  (3114–3193 µs, `DEVLOG-dense-decode.md`) = **1.81× against what we run
  today**, at 93 % of the int8 floor of its own.
- Pattern: **fp16 side ≥ 55 % of floor ⇒ 1.93–2.03×**; below that both sides
  are launch-bound and the ratio is 1.0 ± 0.05 — **never worse than 0.96×**.
- Per-128-group scales cost **+28 %** vs per-channel at lm_head → the cheap
  granularity is also the accurate-enough one, and it keeps the inner loop
  scale-free (1 byte/weight + 1 cvt + 1 fma).
- `torch.mv` fp16 = 85 GB/s at lm_head (9× off our own kernel) — confirms
  hipBLAS is not our GEMV path and comparisons must be kernel-vs-kernel.

### Evidence — AGAINST / limits of this evidence

- The probe's own fp16 baseline is **2–3× off the production kernel at mid
  shapes** (e.g. [8192,2048] at 31 % of floor where `DEVLOG-moe-m1-sprint`
  records 100 %), so its self-reported "4.60 + 0.67 ms/step saved (dense),
  6.18 (MoE)" **overstates**. Floor-based projection used instead: dense
  −4.0 ms, MoE −2.2…2.4 ms → ceiling dense +10–13 %, MoE +14–19 %.
- Auto-geometry (BK=128/SPLIT=8) left 30–40 % on the table at
  [5120,17408] (1.52×) and mtp `fc` (1.45×) — tuning debt, not a wall.
- Nothing here is a serving number. Three prior sessions (G1, C1, S2) show
  standalone wins of this size failing to transfer.

### Why the byte lever works where the ALU lever did not

Decode M=1 GEMVs sit at the HBM floor, so time = bytes / bandwidth and the
only available lever is bytes; the dequant ALU (cvt + fma per weight) hides
under the stream, which is why the int8 rows keep ~90 % of their own floor.
This does **not** contradict the `DEAD-ENDS` row "the lm_head GEMV lever does
not exist" — that closed *kernel-side* wins at a fixed byte count.

**VERDICT:** `OPEN` — probe gate passed; serving A/B + KLD/PPL pending
(`I1`). Highest-risk tensor is lm_head per-channel W8: KLD it alone.

---

## 2026-08-31 — is a hand HIP int8 GEMM the real version of P1?

### HYPOTHESIS

Since our hot path is hand HIP and not Triton, a purpose-built int8 GEMM
reaches near the dot4 ceiling and wins prefill.

### Gate

Scope arithmetic first (checkpoint MAC shares), because it can kill the
hypothesis for free: an int8 GEMM can only touch tensors we can *store* in
int8.

### Evidence

- The BF16 mass is only **7.9 % (dense) / 8.9 % (MoE) of per-token GEMM
  MACs** → even a perfect 1.96× kernel on it wins 4.4 % of prefill compute
  (~3 % wall). Same tensors, right lever at decode (60 % of MoE bytes), wrong
  lever at prefill.
- int8-ing the int4 mass = 2× its bytes: **+11.4 GB dense / +15.8 GB MoE**,
  TP=2-only, and at TP=2 dense that is ~178 k tokens of the 445 k KV pool.
- llama.cpp's dot4-based multi-token GEMM on gfx906 measured **+21…+51 %**
  over its own baseline (`dequant-instructions.md`) — the realistic band for a
  hand int8 GEMM is ~1.3×, not 1.96×.

**VERDICT:** `DEAD-END` (hand int8 GEMM as a prefill lever — scope-capped,
not measured; kept so the "just write the HIP kernel" proposal meets the 8 %
number instead of re-deriving it).

---

## 2026-08-31 — the only >2× prefill ceiling left: `v_dot8_i32_i4`

### HYPOTHESIS

If our prefill GEMM runs at 5.46 T MAC/s = **0.93× the scalar fp32-FMA
record**, then we are at ~1 MAC per issued instruction, and any inner loop
getting ≥ 2 MACs per instruction (packed fp16, dot4, dot8) has headroom
independent of data format; dot8 (49 600 GMAC/s = 3.75× packed fp16, **int4 on
both operands, so weights stay int4**) is the only candidate that both clears
2× and costs no bytes.

### Evidence

- Ceiling ladder measured on this box: `v_mac_f32` 5 824 · `v_pk_fma_f16`
  13 210 · `v_dot4_i32_i8` 25 877 · **`v_dot8_i32_i4` 49 600** (all full-rate,
  `dequant-instructions.md`).
- Break-even vs our 5.46 T: 1.3× needs 14 % of the dot8 record, 2× needs 22 %.
- Counter-evidence that operand prep is the whole game: our own
  `i4→i8 unpack + 2×sdot4 = 0.24×` row — unfree operand preparation is
  punished by ~4× on this ISA.
- Cost side: dot8 needs **int4 activations** (W4A4 on an already-int4 model,
  per-group act scales, AWQ `qzeros` → epilogue correction terms). No
  evidence yet that quality survives; the fork's int8 act-quant result says
  nothing about int4 acts.

### Refrigerated residue (cheap, do before any kernel)

- **ISA audit of the W4A16 prefill inner loop**: does it use packed/fdot2
  MACs, or is it scalar-FMA-bound at the 5.8 T row? Free, zero accuracy risk,
  candidate band 1.2–1.5×. Skill: `gfx906-isa-disassembly`.
- **P3a** (~200 lines): LDS-staged BM=64/BN=64/BK=32 rate probe with dot4 and
  dot8 inner loops at [4096, 34816, 5120], **operand prep charged**, vs
  hipBLAS fp16 (7.57 T). GO bars: int8 ≥ 1.3×, dot8 ≥ 2×.
- **P3b** (only if P3a clears): W4A4 KLD + PPL gate.
- T1 follow-ups: M=2…4 leg (MTP verify pass), per-shape geometry tuning,
  lm_head-only KLD.

**VERDICT:** `OPEN` — T5/`I5`, gated by P3a; nothing may be built on this
hypothesis before the probe runs.

---

## Interactions / superseded-by

- Supersedes nothing; **reframes** the `DEAD-ENDS` rows "dense K=5120 GEMV /
  lm_head GEMV → NEUTRAL, the lm_head GEMV lever does not exist" and
  "dp4a/dot2/dot8 swap in the KQ loop → DEAD-END": both are byte/rate facts
  that hold; T1 attacks bytes, T5 attacks MACs-per-instruction.
- Cross-links: `int8-investigation-qwen.md` (analysis, T0–T5, §6 probe
  records), `DEAD-ENDS.md` (`INT8` rows), `dequant-instructions.md` (dot-rate
  table this log leans on four times), `DEVLOG-dense-decode.md` +
  `DEVLOG-moe-m1-sprint.md` (the fp16 GEMV floors), `oom-256k-prefill.md`
  (VRAM pressure that makes the −4.5/−4.1 GB T1 side-benefit interesting),
  `ROADMAP.md` (`I1`–`I5` proposed, not yet entered).
