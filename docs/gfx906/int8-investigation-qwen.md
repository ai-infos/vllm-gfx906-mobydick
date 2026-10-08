# INT8 / Q8 on gfx906 — what transfers from the gfx908 int8-vllm fork

Investigation, 2026-08-30. Inputs: `/local/git/int8-vllm`
(gfx908/MI100, `curvedinf/int8-vllm` @ `0dc62c98e`), `/home/kread/git/llama.cpp`
@ `b7386eeae` (gfx906 Q8_0 repack), the two Qwen checkpoints we serve, and
this hub. §1–§5 are analysis; **§7 is measured** — probes P1/P2 were run on
MI50 #0 on 2026-08-31 (logs `/local/tmp/int8-probes/p1-full.log`,
`p2-full.log`; no engine code was modified by them).

**VERDICT:** **T1 (int8 the fp16 weight mass) GO on the probe gate** —
1.93–2.03× at every floor-bound shape, never < 1.0×, so it goes to a serving
A/B. **T3 (A8W8 prefill) NO-GO**: Triton `tl.dot` int8 does emit
`v_dot4_i32_i8` and is exact, but the Triton-vs-tuned-kernel gap on gfx906
(1.5–1.7×) is bigger than dot4's edge over packed fp16 (1.96×), so a Triton
A8W8 GEMM loses to hipBLAS fp16 at every prefill shape we ran (§3.3, §6.2).
**The HIP-kernel successor question is answered by scope, not by compiler:**
the fp16 mass is only 7.9 %/8.9 % of per-token GEMM MACs, so no int8 GEMM —
hand-written or otherwise — is worth more than ~3 % of prefill, while
`v_dot8_i32_i4` (3.75× packed fp16, int4 weights stay on tape) is the only
route with >2× headroom (§3.10). T0 (P82 lossy acceptance) and T2 (int8 KV)
stay open — neither is gated by these probes.

## 0. Bottom line

Ranked by (expected gain × confidence) ÷ effort. "fp16 mass" = weights that
**our checkpoints ship unquantized** (§2.2) — the class where int8 is a strict
byte reduction over what we read today, with no W4 comparison problem.

| # | item | what transfers | expected gfx906 effect | conf | gate |
|---|---|---|---|---|---|
| **T1** | int8 for the **fp16 weight mass** (lm_head, FA q/k/v/o, GDN in_proj_qkv/z/out, shared expert, layer 0, the BF16 MTP draft layer) — load-time quant, W8A16 GEMV | yes, fully (byte-bound, no MFMA/aiter needed); their load-time quant pattern (`_quantize_embedding_int8_`) is the template | **dense 27B +10–13 % decode, MoE 35B +14–19 % ceiling**; ~4.5/4.1 GB of resident weights freed | **high — P2 measured 1.93–2.03×** at floor-bound shapes, ≥ 0.96× everywhere (§7) | serving A/B + KLD/PPL |
| **T0** | **P82 lossy spec-decode acceptance** (rejection-sampler OR-clause) | yes, arch-free (pure sampler) | their measured **+19…+26 % decode, GSM8K neutral**; ours rides on MTP k=2 (78.7–90.9 % acceptance today) | high (imported) | GSM8K + greedy-coherence + serving A/B; **explicit opt-in only** |
| **T2** | **int8 KV storage** (`int8_per_token_head` doctrine, but in our Q8_0 layout) | partial: format+doctrine yes, their Triton/aiter kernel no | **1.88× KV capacity** (256k → ~480k-token pool at fixed bytes); decode +1–3 % (halves FA gather traffic, kills the read-side quantize); V-side accuracy is the only new loss | med | FA-side accuracy gate + serving A/B (the LEGACY=0 lesson, §3.2) |
| **T3** | **A8W8 (int8 act × int8 weight) GEMM for prefill only** (Triton route; HIP successor → T5, §3.10) | **as Triton source** — our Triton port does lower int8 `tl.dot` to `v_dot4_i32_i8` (§2.3, measured) | **0.59–0.68× of hipBLAS fp16** at our five prefill shapes (measured) — dot4 is real but Triton's GEMM codegen gives up more than int8 wins | — | **closed by P1: NO-GO → `DEAD-ENDS.md`** |
| **T5** | **hand-kernel prefill dot** — the HIP successor to T3: `v_dot8_i32_i4` (49 600 GMAC/s = 3.75× packed fp16, int4 on both operands → **weights stay int4, no KV cost**), not int8 | nothing to port — the datapath is ours to write (no upstream/MFMA equivalent); int8 has no prefill scope here (§3.10) | ceiling 3.75× packed fp16, and our W4A16 prefill runs at 5.46 T = 0.93× the *scalar fp32-FMA* record (~1 MAC per issued instruction); only 14–22 % of the dot8 record is needed for 1.3–2× — risks are operand supply (`i4→i8 unpack + 2×sdot4 = 0.24×` is our own warning) and W4A4 accuracy | low-med | free ISA audit of the prefill inner loop → **P3a** rate probe → **P3b** accuracy (§3.10) |
| **T4** | accuracy + measurement methodology (KLD probe, record-once replay budget, paired boots, metric definitions) | yes, pure tooling | our dense PPL band is ±2 % — too coarse to gate a 0.1 % weight perturbation; KLD is ~100× more sensitive | high | adopt on first use |

The single most important framing: **the int8 fork's GEMM layer does not
transfer at all (aiter + CK, both MFMA-based), but the int8 fork's *byte*
doctrine transfers better than it did on MI100** — because on our Qwen
checkpoints a third of the decode-step weight traffic is still fp16.

## 1. Hard constraints (do not re-litigate)

1. **No aiter, no CK on gfx906.** Every GEMM, the act quant, unified
   attention and the CAR in the gfx908 stack are aiter/Composable-Kernel
   objects; aiter does not support gfx906. All of it is out.
2. **No MFMA/WMMA** (`README.md` line 4). Integer compute on this chip is
   `v_dot4_i32_i8` only: full rate, 4.44× fp32 FMA, **1.96× packed fp16**
   (`dequant-instructions.md`, SCEV-proof probe). `v_dot4c_i32_i8`,
   `v_dot8_i32_i8`, `v_dot8c_i32_i8` are assembler-rejected; the i4
   `v_dot8_i32_i4` exists but needs int4 operands on both sides.
3. **Weights are already int4** (AWQ) for the MLP/expert/GDN-in_proj bulk.
   W8 for those tensors is **2× the bytes** → strictly worse on byte-bound
   decode. Any A8W8 proposal must live where the workload is
   compute-bound, not where it is byte-bound (§2.1).
4. Our custom Q8 FA backend declares `supported_kv_cache_dtypes = ["auto",
   "float16", "half"]` (`vllm/gfx906_fa/gfx906_fa_backend.py:171-175`).
   `--kv-cache-dtype int8_per_token_head` today silently leaves the CUSTOM
   backend → Triton attention (the 18.89-vs-25.60 t/s class of gap). "Just
   turn on int8 KV" is not available.
5. The int8 fork's numbers are **their** metric definitions
   (`docs/recipes/README.md` §NUMBER DEFINITIONS: steady-state TG =
   concurrency ÷ mean TPOT vs `vllm bench serve` wall-clock ≈ 0.55× of it),
   on a box with a 24 % per-rank sclk skew. Never mix them with ours.

## 2. Three facts that decide most of the surface list

### 2.1 int8 GEMM pays only where compute-bound — and the fork measured the boundary

`vllm/model_executor/kernels/linear/mixed_precision/triton_w8a16.py:583-587`
(gfx908 fork) is the whole doctrine in one comment:

```
_W8A8_DISPATCH_MIN_M = 256      # decode (M<256) stays on W8A16 (bandwidth-bound)
_W8A8_DISPATCH_MIN_N = 8192     # microbench 2026-08-22: 17408-wide MLP up/gate 1.62x,
                                # 5120-wide 0.81-0.95x, small-N 0.61x
```

MI100's int8 advantage over its own fp16 path is the same ~2× ours is
(MI100: INT8 374 TOPS vs FP16 184.6 TFLOPS; MI50: dot4 25.9 T MAC/s vs
packed fp16 13.2 T MAC/s). So those ratios are the best available prior for
A8W8 on gfx906: **the win is real on fat-GEMM prefill and negative
everywhere byte-bound.** Independent corroboration from the llama.cpp gfx906
Q8 repack PR (commit `b7386eeae`, recorded in `dequant-instructions.md`):
layout/repack deltas were **neutral (−1.1…+6.4 %) on single-token mat-vec**
and **+21…+51 % on the LDS-staged multi-token GEMM**.

Our own dense prefill sits at ~41 % of the fp16 MAC ceiling (442 t/s @32k at
TP=2 from the `README.md` long-context sweep, vs 13.2 T MAC/s/GPU taking
~24.5 B non-vocab params → 49 GFLOP/token), i.e. there is
headroom for a higher-ceiling dot — but only on the fat shapes:
`gate/up = [34816, 5120]` ✓, `qkv = [14336, 5120]` ✓, `down/o_proj =
[5120, 17408]/[5120,5120]` ✗ (their 0.81–0.95× band), MoE experts
`N=512/1024` ✗✗.

**P1 (2026-08-31) closed that headroom for Triton:** on exactly those two ✓
shapes a Triton A8W8 kernel reaches 5.0 T MAC/s = 19 % of the dot4 ceiling —
below hipBLAS fp16's 7.6 T (57 % of the fp16 ceiling) and level with what
our W4A16 prefill already achieves. The arithmetic ceiling is real and
unreachable in this compiler; the headroom is a *codegen* problem, not an
instruction-set one (§3.3, §6.2).

### 2.2 Our fp16 weight mass (checkpoint scan, 2026-08-30)

Safetensors-header scan of the two models we actually serve. This is the
number the int8 fork's whole `INT8_AUDIT_RESULTS.md` exists to find, and on
our checkpoints it is **big**.

**Qwen3.5-27B-AWQ** (`/data/models/qwen/Qwen3.5-27B-AWQ`, 21.85 GB:
11.47 GB packed int4, 10.02 GB BF16, 0.36 GB fp16 scales):

| fp16/bf16 tensor group | size | read per decode token |
|---|---|---|
| `lm_head.weight` [248320, 5120] | 2.543 GB | **yes** |
| `embed_tokens.weight` [248320, 5120] | 2.543 GB | no (row gather) |
| FA `q_proj` [12288, 5120] ×16 layers | 2.013 GB | **yes** |
| FA `k_proj`/`v_proj` [1024, 5120] ×16 | 0.336 GB | **yes** |
| layer 0 (`modules_to_not_convert`: GDN projs 0.232 + 3×178 MB MLP) | 0.767 GB | **yes** |
| `mtp.*` draft layer (q/k/v/o 0.210 + `fc` 0.105 + MLP 0.535) | 0.849 GB | **yes, in every MTP arm** |
| GDN `in_proj_a`/`in_proj_b` [48, 5120] ×96 | 0.047 GB | yes (negligible) |
| **fp16 read per decode step** | **≈ 5.71 GB** (6.56 GB with the MTP draft) | **≈ 7.17 ms @ 798 GB/s** |
| `model.visual.*` (27-block ViT + merger, BF16) | 0.922 GB | no (multimodal tower; `Qwen3_5ForConditionalGeneration` ships it in the same shard set) |

That reconciles with the measured `LLGemm1 … ~7,100 µs/step` bucket in
`DEVLOG-dense-decode.md` to within 1 %. (It does **not** reconcile with that
devlog's *labelled* floor table, which attributes 3.79 + 1.25 ms/step to
`in_proj_b×48`/`in_proj_a×48` — those tensors are 47 MB total, i.e. ~60 µs.
The labelled rows are stale; the total is right. §7 correction C-1.)

**Qwen3.5-35B-A3B-AWQ** (`/data/models/QuantTrio/…`, 25.45 GB: 15.83 GB
packed int4, 9.05 GB BF16): per-step fp16 read = `lm_head` 1.017 GB +
GDN `in_proj_qkv` [8192,2048]×30 = 1.007 + `in_proj_z`×30 = 0.503 +
`out_proj`×30 = 0.503 + FA `q_proj`×10 = 0.336 + `o_proj`×10 = 0.168 +
shared experts ×40 = 0.246 + layer-0 active experts ≈ 0.05 →
**≈ 3.8 GB/step ≈ 4.8 ms @ 800 GB/s**. This matches the per-shape floor
table in `DEVLOG-moe-m1-sprint.md` ("S3 LLMM1 2.09 ms" + the `dense_gemv`
floors: lm_head 1274.6 µs, GDN in_proj 63.1 µs×30, FA qkv 47.3×10,
o_proj 21.0×40, shared gate_up 5.3×40, shared down 2.6×40 = **4.80 ms**)
line for line.

Consequences, and they are the reason this document exists:

* **32 % of the MoE decode step and ~20 % of the dense decode step is
  fp16 weight streaming that is already at 98–101 % of the HBM floor**
  (`DEVLOG-moe-m1-sprint.md` "no lever" column; `DEVLOG-dense-decode.md`
  "LLMM1 is at the HBM floor for every dense shape"). Nothing in our
  roadmap can win there — except moving fewer bytes.
* int8 halves those bytes. At the floor the convertible mass is worth
  **dense −3.58 ms (−4.0 ms with the draft leg) of a ~35.5 ms decode step,
  MoE −2.4 ms of a 15.0 ms step**, i.e. a ceiling of **dense +10–13 % and
  MoE +14–19 %** (the top of the MoE band is the 70 t/s roadmap target in
  one move). P2's conversion factors (§7) say the halving is realized
  wherever the fp16 GEMV is actually at the floor, and buys nothing where
  the shape is launch-bound (N ≤ 2048 → ~1.0×), which is why the band is
  14–19 and not a flat 19.
* It also halves the resident non-visual BF16 set: **−4.5 GB (dense,
  9.1 GB loaded) / −4.1 GB (MoE, 8.2 GB)** — directly relevant to the
  `--gpu-memory-utilization 0.93` ceiling and the 256k-prefill pressure in
  `/local/git/AGENTS.md` and `oom-256k-prefill.md`.
* **Their doctrine matches ours on what NOT to int8**: they keep `A_log`,
  `dt_bias`, depthwise conv1d, norms, KV scales, GDN recurrent state and
  softmax/P·V float, and they explicitly refuse to quantize `in_proj_a/b`
  because those feed `exp(A)` gates (`INT8_AUDIT_RESULTS.md` #5). On our
  checkpoints those tensors are 47 MB — nothing to win there anyway.

### 2.3 Triton on gfx906 lowers int8 `tl.dot` to `v_dot4_i32_i8`

Source-verified in our Triton (`/local/git/triton-gfx906`, 3.6.0, the tree
`requirements`/venv resolves to):

* `third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp:48-53` — for
  `i8 × i8 → i32` the chosen intrinsic is `llvm.amdgcn.sdot4`,
  `vectorSize = 4` (operand pack bitcast to `i32`).
* `.../TargetUtils.cpp:44-56` — `supportsVDot()` includes `VEGA20`.
* `.../TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp:1663-1669` — the
  I8×I8→I32 v_dot form is legal when `k % 4 == 0`.
* `.../DotOpToLLVM.cpp:53-55` — a blocked-layout `DotOp` goes to
  `convertAMDFMADot` (the path above); `f16×f16→f32` takes `fdot2` there.

So the gfx908 fork's **portable** int8 GEMM (`triton_w8a8_gemm_kernel`,
`triton_w8a16.py:248-341` — `tl.dot(a_q, b_q, out_dtype=tl.int32)` with
`BLOCK_K == group_size == 128` and one A-scale/B-scale descale per tile,
exact by construction) can be lifted as *source* with no aiter dependency.
**Measured on our toolchain (P1, 2026-08-31), so this is no longer a
source-reading claim:** a `16×16×16`-tile int8 `tl.dot` compiles to
`v_dot4_i32_i8` (16 occurrences in `kernel.asm['amdgcn']`, zero `v_mac_f32`,
zero `v_fma_f32`), accumulates **exactly** vs an fp64 reference (maxdiff 0 at
64³ and 32³), and the A8W8 GEMM kernel emits 2048 `v_dot4_i32_i8` against
4096 `v_dot2_f32_f16` for the same-work fp16 comparator — the 2:1
instruction ratio the ISA ceiling predicts, with no spills.

What the probe *also* established, and it is the reason T3 died: neither
path gets near its ISA rate in Triton (int8 14–19 % of the 25 877 GMAC/s
dot4 record, fp16 12–21 % of the 13 210 `v_pk_fma_f16` record, vs hipBLAS
fp16 at 57 % of its record). The compiler, not the instruction set, is the
binding constraint on gfx906 — see §3.3.

## 3. Surface by surface

Format: what gfx908 did (file/commit) → why it won there → gfx906 transfer →
verdict → gate.

### 3.1 T1 — int8 the unquantized projections (incl. lm_head + embedding)

* **They did:** `VLLM_GFX908_INT8_EMBEDDING` (default ON, `d9e11f233`) —
  load-time per-row int8 + fp16 scale via `replace_parameter`, dequant fused
  into the gather; `VLLM_GFX908_INT8_LM_HEAD` (production ON, `64627cb78`) —
  untied lm_head → per-channel int8 + W8A8. Their audit's "ship int8
  storage" item halves a 30 GB checkpoint toward 18 GB.
  **Trap they hit first** (`ab2139abf`): the initial version cast the whole
  table to fp16 *before* `F.embedding`, materializing 2.5 GB per call. The
  fix gathers int8 rows and dequantizes only the selected rows.
* **Us:** the same load-time conversion applies to the §2.2 mass, not just
  the vocab pair. No checkpoint work, no third-party quant pipeline, and we
  already do load-time repacks (W4 MoE repack ~65 s). **P2 settles the
  quantization choice: per-output-channel, scale applied after the
  reduction.** The per-channel inner loop is scale-free (one `v_cvt` +
  `v_fma` per weight, scale multiplied into the accumulator once), and P2
  measured the per-128-group variant at **+28 % over per-channel** at
  lm_head (2202 vs 1717 µs) for no bandwidth benefit — group granularity is
  only worth paying for accuracy, and only where per-channel accuracy fails
  the gate. Note this path is **W8A16**: activations stay fp16, there is no
  act-quant pass, and no GPTQ packing is involved.
* **Kernel work:** one int8 weight-stationary GEMV in the
  `csrc/rocm/dense_gemv_gfx906.cu` family (M=1, K≤8192) reading int8 +
  per-row scale, fp16 `x`, `__ockl_fdot2` accumulate. Byte-bound ⇒ the
  dequant ALU is free (the `dequant-instructions.md` warning about
  i8→half2 expansion costing 0.17× applies to *dot-throughput*-bound loops,
  not to a stream at 800 GB/s). The `v_dot4` alternative (quantize `x` per
  token, pad M to 16, exact int32 accumulate) is **measured dead in Triton**:
  277 ms vs 1.7 ms at lm_head (161×), 107 ms vs 0.7 ms at lm_head-MoE — the
  broadcast-`tl.dot` form does not generate a usable M=1 loop. If dot-through
  weight is ever wanted at M=1 it has to be hand-written (the llama.cpp
  `q8_repack` MMV design, `b7386eeae`, is the reference); P2 says it is not
  wanted — bytes, not MACs, are what we are paying for.
  `LLMM1`-served shapes (`out_proj`, shared expert) can be routed to the new
  GEMV rather than porting an int8 LLMM1.
* **Verdict: GO on the probe gate (P2 passed 1.93× / 2.00× against a 1.7× /
  1.6× bar; §7).** Remaining gate, in order: (1) serving A/B
  (`_bench_gfx906.py`, MoE + dense, both arms same boot) — the probe's
  ratios are geometry-clean but its own fp16 baseline is 2–3× off the
  production kernel at mid shapes, so only the serving number counts
  (`AGENTS.md` transfer rule); (2) accuracy: PPL bands (6.6817–6.6942 /
  6.6993–6.7197) **plus** a KLD gate (§3.8) because the dense PPL band is
  ±2 % and cannot see a 0.1 % weight change — per-channel W8 on lm_head is
  the highest-risk tensor and should be KLD'd alone; (3) an M=2…4 leg for
  the MTP verify pass (weight bytes are per-step, not per-token, so the
  saving should carry, but that is an assumption until measured).

### 3.2 T2 — int8 KV cache (their `int8_per_token_head`, our Q8_0)

* **They did:** `--kv-cache-dtype int8_per_token_head` (fp32 inline scales)
  on target *and* draft, through aiter unified attention / the Triton unified
  kernel; per-token-head dynamic quant at cache-write; replay error "0.85 %
  per token-head"; the int8-KV + int8-mamba-state **combination** corrupts
  generation (bisected; fp16 state blew states to 63k → fp32 state is
  mandatory).
* **Us:** our FA already computes on **Q8_0 K** (per-32-block, inline half
  scale — *finer* granularity than per-token-head, so equal-or-better
  numerics per byte) and reads V in fp16. Per `README.md`, LEGACY=1 (the
  default) quantizes K on every read; LEGACY=0 stores a Q8 view aliased into
  the fp16 K half. The thing **nobody has tried** is making Q8 the *primary
  pool*:
  * capacity: per (layer, token) fp16 K+V = 2·(D·2 B·Hkv) = 4096 B vs Q8_0
    2·((D/32)·34 B·Hkv) = 2176 B → **1.88× pool** (K-int8-only: 1.31×);
  * cost: the append-time Q8 write we measured at **+94.6 µs/step** in
    LEGACY=0 — but there it was an *extra* write on top of the fp16 write.
    In an int8-primary pool the fp16 write disappears, so that specific tax
    is structurally gone (the G1 per-node tax question stays open and is
    the same measurement);
  * read: gather traffic halves and the in-kernel quantize disappears —
    i.e. LEGACY=0's stated benefit without LEGACY=0's measured cost.
    Note the honest counter-evidence: LEGACY=0's kernel-level wins
    (Q8 gather 22–45 % faster per step) did **not** transfer (−6.3 % B=1).
  * **V is the new loss and the new kernel work.** Storing V in int8 does
    *not* let P·V use `v_dot4` (P is fp16 probabilities; the P·V accumulate
    is 1024× in-place `v_pk_fma_f16` — `dequant-instructions.md` 2026-08-29
    objdump). V must be dequantized back to fp16 in the kernel, adding ALU
    to an issue-bound loop at Sq>1. So: **stage A = int8 K only (zero new
    accuracy loss vs the production read path — K is already Q8 there),
    stage B = int8 V with its own gate.**
* Plumbing cost is real: `int8_per_token_head` is already plumbed for the
  Triton backend in our tree (`vllm/v1/kv_cache_interface.py`,
  `triton_attn.py`) but the attention *pool* is a single
  `[blocks,2,bs,Hkv,D]` tensor — a mixed/staged int8 layout for the CUSTOM
  backend is new allocation + `do_kv_cache_update` + COW work
  (`GFX906_FA_LEGACY=0` already proved the COW-safe alias pattern).
* **Verdict: OPEN, medium.** GATE: (i) PPL/coherence gate on V-int8
  (K-int8 needs none beyond a bit-equality check vs the current Q8 read);
  (ii) serving A/B at B=1/B=4 and pp2048/pp16384; (iii) the win is mostly
  *capacity*, so the acceptance criterion is tokens-in-pool and
  long-context TTFT, not t/s.

### 3.3 T3 — A8W8 prefill GEMM

* **They did:** aiter CK W8A8 (`gemm_a8w8_CK`, per-token act scale ×
  per-channel weight scale) as the production path at **every** M, with a
  load-time requant of the GPTQ gs128 scales to per-channel
  (`triton_w8a16.py:674-697`, `aiter_w8a16.py:380-500`), plus a Triton
  blockscale fallback and their own Triton `tl.dot` int8 kernel for
  large-M/fat-N (`d9e11f2..16e2b51`, `199db58`, `89e808d`).
* **Us:** only the portable Triton kernel is on the table (§2.3), only for
  **prefill / spec-verify shapes with M ≥ 256 and N ≥ 8192** (their
  measured gate, §2.1) — i.e. dense `gate/up` [34816,5120] and `qkv`
  [14336,5120]. Two extra costs they also paid and we would too:
  a per-call activation quant (§3.4) and a **2× weight copy** — int8 weights
  for prefill on top of the int4 weights we need for decode is 1.9× the
  weight footprint, which only fits at TP=2 and only if we free the §2.2
  fp16 mass first (T1). Also note the accuracy asymmetry: int8-ing an int4
  checkpoint's *activations* is fine, but requantizing group scales is not
  free — their own per-channel requant is lossy and they kept the gs128
  scales around "for the blockscale fallback and the fused context-KV
  dequant".
* **Verdict: NO-GO for the Triton route, closed by P1 (2026-08-31); the HIP
  route is a separate question and is scoped in §3.10.** All three ISA-level
  preconditions passed — `v_dot4_i32_i8` in the ISA, exact int32
  accumulation, dot4 2:1 on instruction count vs the fp16 comparator — and
  it still loses, at every shape, by 1.5–1.7×:

  | M×N×K | A8W8 (+act-quant) | Triton fp16 | hipBLAS fp16 | A8W8/hipBLAS |
  |---|---|---|---|---|
  | 4096×34816×5120 (gate/up) | 145.75 ms | 167.35 | **96.24** | **0.66×** |
  | 4096×5120×17408 (down) | 66.93 ms | 78.73 | **43.75** | 0.65× |
  | 4096×14336×5120 (qkv) | 60.13 ms | 69.07 | **39.80** | 0.66× |
  | 1024×34816×5120 | 35.55 ms | 39.10 | **24.17** | 0.68× |
  | 256×34816×5120 (their MIN_M gate) | 10.56 ms | 12.35 | **6.26** | 0.59× |

  A8W8 beats the *same-codegen* fp16 Triton kernel by only 1.10–1.18× (the
  gate needed 1.30×), i.e. dot4's 1.96× ISA edge is spent closing Triton's
  own codegen deficit instead of buying speed. Three independent reasons it
  cannot work here, any one of which is sufficient:
  1. **Triton's GEMM on gfx906 runs at 19 % of the dot4 record** (5.0 T
     MAC/s) — the same deficit exists in fp16 (33 % of record) and the
     hand-written/hipBLAS path is at 57 %; the compiler gap (1.5–1.7×)
     exceeds the int8 arithmetic edge.
  2. Our production prefill GEMM is **W4A16**, which reads **half the weight
     bytes** A8W8 would; A8W8 would *increase* prefill weight traffic over
     what we do today while only matching its arithmetic rate (~5.0 vs the
     ~5.4 T MAC/s our dense prefill already achieves, §2.1).
  3. The 2× weight copy (int8 prefill weights alongside the int4 decode
     weights) does not fit at TP=1 and competes with T1's own savings.

  Reopen conditions and why they are narrow: a hand-tuned int8 GEMM (the
  llama.cpp `q8_repack` idiom, or a T1-style LDS-staged kernel) would have to
  beat W4A16 head-to-head **and** first clear the §3.10 scope argument (8 %
  of prefill MACs). **Action: move to `DEAD-ENDS.md` with `p1-full.log` as
  the evidence.**

### 3.4 Activation quantization numerics (round-to-nearest) — cheap, and it is a numerics lesson not a perf lever

Their `pertoken_quant` replacement (`act_quant_rn.py`) is a two-sweep Triton
per-token int8 quant that **rounds to nearest instead of truncating toward
zero**: replay convicted the truncation leg at 10–15 % mean rel-L2 per GEMM
output, rounding recovers ~half of it, and it was *also faster* than the
4-pass aiter eager chain (TPOT 14.4 → 12.5 ms came with it). Portable
gotcha they hit: **`libdevice.rint` is not available on HIP** — they use
`floor(x + 0.5)`. Our in-tree `_per_token_quant_int8`
(`vllm/model_executor/layers/quantization/utils/int8_utils.py:47-61`) uses
`tl.extra.hip.libdevice.round`.

**Measured (P1 part D, 4096×5120 fp16 → int8, per-token symmetric):**

| rounding | time | disagreement vs `floor(x+0.5)` |
|---|---|---|
| trunc (`.to(tl.int8)`) | 133.1 µs | 49.7 % of elements |
| `floor(x + 0.5)` | 131.9 µs | — |
| `tl.extra.hip.libdevice.round` | 148.3 µs | 0.007 % |

So: `libdevice.round` **is** available on our Triton 3.6 / ROCm 7.14 build
and costs +12 % over `floor(+0.5)`; it differs from it on 0.007 % of
elements, consistent with round-half-to-even at exact `.5` (their "100 %
bit-identical to floor(x+0.5)" claim is off by that tie-breaking, which is
harmless but should not be quoted as bit-equality). Truncation differs on
**half the payload** — which is the actual numerics finding of theirs worth
adopting, and it is a correctness issue, not a perf one.
Perf relevance for us: only as a component of T3, and T3 is closed. At
M=4096 the whole pass costs 0.13 ms = 0.14 % of the GEMM, i.e. their
act-quant overhead never was the problem (§2.1's N-gate is about the GEMM,
not the quant). **Verdict: adopt RN rounding as method wherever an act-quant
exists; no standalone perf item.**

### 3.5 int8 GDN state / conv cache / attention internals — matches our doctrine, no action

Their verdicts: GDN recurrent state **must stay fp32** (int8 corrupts with
int8 KV; fp16 round-trip broke delta-rule cancellation, states → 63k); conv
cache fp16; softmax and P·V stay float. We already keep all of these float.
Nothing to do; recorded so a future "let's int8 the mamba state" proposal
meets their bisection record instead of re-running it.

### 3.6 T0 — P82 lossy spec-decode acceptance (no int8, biggest non-kernel win)

`VLLM_GFX908_MTP_ACCEPT_THRESHOLD` (`vllm/v1/sample/rejection_sampler.py:99`)
adds an OR-clause: also accept a draft token whose *target* probability is
above a threshold. Their measurement (4×MI100, full bench + GSM8K 300 Q,
temp 0.6, seeded, default budget): **+19 % (27B dense decode), +24 %
(35B MoE decode), GSM8K 91.7 → 92.0 % (neutral)**; ~75–80 % of tokens
diverge from strict sampling; lossy and explicitly opt-in; they also
record an n=5 long-context buffer overflow that they sidestepped by
shipping n=3.
For us this is a sampler-only change that rides MTP k=2 (78.7–90.9 %
acceptance, `README.md`) and ngram n=5. Two caveats before believing the
+19 % here: (a) our spec numbers already assume exact verification, and
`DEVLOG-spec-decode.md` gates are acceptance-based — an OR-clause inflates
"accepted tokens" while silently changing output distribution, so the
accuracy gate must be task-level (GSM8K-style) **and** a
stop-probability/coherence check (see §3.8); (b) our own record warns that
spec-decode numbers are the first casualty of host-state degradation
(`/local/git/AGENTS.md` MI50 degradation note) — run the canary first.
**Verdict: OPEN, high value / low cost. GATE:** canary probe → seeded
task-accuracy A/B + first-token-stop probability + serving A/B; ship
default-OFF behind an env flag exactly as they do.

### 3.7 Fused AR+RMSNorm+quant epilogue — their result is a *negative*, and one warning

They built the full chain (aiter `fused_ar_rms_int8_per_token_quant`, a vLLM
fusion pass, an eager seam) and shipped it **OFF**: neutral on TPOT
(prefill-only by design), and — the transferable warning — enabling the
fused norm+quant **dropped spec-decode acceptance 71.2 → 46.5 %** while
being numerically "close" (`docs/recipes/README.md` §Current production
status). Second instance of the same lesson in this repo: quant-adjacent
numerics changes show up in acceptance rates long before they show up in
PPL. For our TP=2 work the corollary is cheap: keep vLLM CUSTOM all-reduce
(they measured 63.49 vs aiter CAR 58.34 vs PYNCCL 53.04 tok/s at TP4/C8 —
the same ordering conclusion we reached independently for TP=2), and do not
add per-layer fused epilogues without first pricing the graph-node cost
(roadmap **G1** — this is exactly the measurement G1 was built for).
**Verdict: no action; cross-linked.**

### 3.8 Their accuracy + measurement methodology is the most portable artifact

* `scripts/kld_probe_v2.py`: greedy continuations + per-position top-K
  logprob dumps along the *reference's* greedy path (so supports always
  intersect — v1's temp=1.0 sampling produced ∞ KLD), fixed 52-prompt
  corpus (code/math/strict-format/chat/long-form), `capture`/`compare`
  modes, `KLD_OUT_DIR/<tag>.npz`. Reported gate format: median KLD, greedy
  agreement n/m, **first-token stop probability vs the BF16 reference** —
  that last one is how they caught a 10× first-token-stop inflation that
  produced empty responses and that PPL never showed.
* `quant_audit_recorder.py` + `scripts/quant_replay_gemms.py`: record-once
  (pinned ring, non-blocking copies, ARM-file arming so warmup records
  nothing) then replay offline to attribute error per quantization leg.
  This is how they *convicted* act-quant and the fp16 GDN state and
  *exonerated* weights/AR/KV.
* Measurement doctrine worth copying verbatim: paired same-regime boots
  (their box is bimodal from DVFS skew, sub-15 % single-boot deltas are
  unmeasurable), one named metric per claim, and "audited exceptions are
  the optimum at the current measurement floor" as a stopping rule.
  Our equivalents of these are the `_bench_gfx906.py` graph A/B (the gate),
  PPL (±0.3 % MoE / ±2 % dense — too coarse here) and `greedy_probe.py`.
  **KLD is the missing gate in our kit** and T1 needs it.
* **Verdict: ADOPT (tooling, no kernel risk).**

### 3.9 Autotuned per-shape kernel tables

Their aiter tuning program produced a 299-row gfx908 tuned a8w8 CSV + a
45-variant `module_gemm_a8w8.so`: **1.37× geo-mean (max 2.64×) kernel-level
decode GEMM speedup, +3.5 % median wall-clock paired A/B** — i.e. a large
kernel-level win that moved end-to-end by single digits (the same transfer
ratio we report constantly). The method (per-(N,K,M-bucket) measured table,
shipped next to the kernel) is applicable to our own `q_gemm`/GEMV/skinny
dispatch — several of our dispatch gates are analytic (K≤8192, M≤7/≤32 MB)
rather than table-driven. **Verdict: OPEN (method), low priority; gate = a
per-shape table for the §3.1 int8 GEMV, which we need anyway.**

### 3.10 T5 — the HIP-route question: what a hand-written int8/int4 GEMM could win here

P1's NO-GO is a verdict about **Triton codegen**, not about int8 on this
chip — and our hot path is hand HIP, so the right question is what a
purpose-built kernel could do. Answer in one line: **int8 (dot4) is scoped
out by the checkpoint, not by the compiler; `v_dot8_i32_i4` is the only >2×
ceiling left, and there is a zero-accuracy-risk measurement to do first.**

**The ISA ladder we already measured** (`dequant-instructions.md`, 2026-08-28,
SCEV-proof probes, MI50) against the rates we actually achieve:

| inner loop | ceiling (GMAC/s) | × packed fp16 | where we sit |
|---|---|---|---|
| `v_mac_f32` | 5 824 | 0.44× | **our W4A16 prefill = 5 460 (0.93× this row)** |
| `v_pk_fma_f16` | 13 210 | 1.00× | hipBLAS fp16 = 7 570 (57 %, P1) |
| `v_dot2_f32_f16` | 13 210-class, latency-sensitive (~10 T in the MoE harness) | 1.0× | — |
| `v_dot4_i32_i8` | **25 877** | **1.96×** | nothing in-tree uses it for GEMM |
| `v_dot8_i32_i4` | **49 600** | **3.75×** | nothing in-tree uses it at all |

The first row is the uncomfortable one: at 41 % of the *fp16* ceiling our
prefill GEMM runs at **0.93× the scalar fp32-FMA rate — about one MAC per
issued instruction**. Any inner loop that does not get ≥ 2 MACs per
instruction is capped near where we already are, which is why three sessions
of W4A16 GEMM work moved prefill so little. That makes "does our prefill
inner loop use packed/fdot2 MACs at all?" the cheapest open question on this
list (ISA audit with the `gfx906-isa-disassembly` skill, **zero accuracy
risk**, candidate band 1.2–1.5×).

**Why the int8 (dot4) route is scoped out — the checkpoint, not the ISA.**
The fp16/BF16 mass is only **7.9 % of the dense and 8.9 % of the MoE
per-token GEMM MAC mass** (checkpoint scan: fp16 24.72 / 34.46 G MAC-token vs
26.05 / 34.79 G total, excluding `visual`/`embed`/`lm_head`). Even a *perfect*
1.96× kernel restricted to those tensors wins
`0.089 × (1 − 1/1.96) = 4.4 %` of prefill compute → ~3 % wall. It is the
right lever for **decode** (where the same tensors are 60.3 % of the MoE
per-token *byte* mass — that is T1, and it is a GEMV, no new GEMM needed) and
the wrong one for prefill.

**Going after the int4 mass with int8 weights costs bytes we do not have:**
int8 is 2× int4, so +11.4 GB (dense) / +15.8 GB (MoE routed experts) resident
— TP=2-only, and at TP=2 dense it eats ~178 k tokens of the 445 k KV pool
(32 KB/token/GPU) to buy at most 1.96× on a path we run at 41 % of the fp16
ceiling. Plus the lossy AWQ-gs128 → per-channel int8 requant they needed and
we would too.

**The one route with real ceiling headroom is `v_dot8_i32_i4`** — 8 int4
MACs/lane/cycle, 3.75× packed fp16, native packed-nibble operands (no unpack
ALU), and critically it keeps **int4 weights on tape**: no byte increase, so
it works at TP=1 and costs no KV. Break-even against our measured 5.46 T
MAC/s: 1.3× needs 7.1 T = **14 % of the dot8 record**; 2× needs 10.9 T =
22 %. Those are low bars *if operand supply holds* — and that is the whole
question, because our own `i4→i8 unpack + 2×sdot4 = 0.24×` row shows how fast
this ISA punishes operand prep that is not free. What it costs:

1. **int4 activations** (dot8 needs i4 on *both* operands). On a model whose
   weights are already int4 this is W4A4 territory: per-group activation
   scales, and AWQ carries zero-points (`qzeros`), so the correction terms
   (`A·z`, `B·z`, `z_A·z_B·k`) land in the epilogue. Expect a real quality
   hit and an honest KLD + PPL gate — their int8 act-quant result is *not*
   evidence for int4 act-quant.
2. **A no-MFMA GEMM that reaches ~20 % of a full-rate instruction** while
   feeding it nibbles from LDS. Nearest working precedent on this ISA family:
   llama.cpp's dot4-based multi-token GEMM, measured **+21…+51 %** over its
   own baseline (`dequant-instructions.md`) — i.e. in practice ~1.3×, not the
   1.96×/3.75× of the ISA table. That is the number to beat in our own probe,
   not the record.

**Recommended order (cheapest first):** (a) ISA-audit the existing prefill
inner loop for packed-MAC usage — free, no accuracy risk; (b) **P3a**, a
~200-line HIP rate probe: LDS-staged BM=64/BN=64/BK=32 GEMM with a dot4 and a
dot8 inner loop at [4096, 34816, 5120], *including* operand prep, against
hipBLAS fp16 (7.57 T). GO bars: int8 ≥ 1.3× hipBLAS fp16, dot8 ≥ 2×. (c)
**P3b**, only if P3a clears: W4A4 accuracy (KLD corpus + PPL bands). Do not
start from a vLLM integration.

### 3.11 Why not "just port their aiter kernels to plain HIP" — recorded so it stays closed

The aiter/CK kernels are MFMA (Matrix Core) shaped: `v_mfma_*` does not exist
on gfx906 (`dequant-instructions.md`), so their tile structure, LDS layouts
and scale handling have no instruction-level home here. What ports is the
*algorithm* (per-token A scale × per-group B scale, one descale per tile,
exact int32 accumulation) — and P1 shows that algorithm's value on this chip
is bounded by the scope argument in §3.10, not by how well we implement it.

## 4. Do-not-transfer list (their own negative verdicts)

| tempting idea | their verdict (evidence) |
|---|---|
| quantize the draft model's `fc` GEMM | REJECT — "not a bottleneck", TPOT +3.4 % paired despite 0.49 % GEMM err |
| int8 ctx-KV projection in the drafter | REJECT — +10.4 % TPOT from acceptance drop, even with RN act-quant |
| int8 GDN/mamba state | corrupts generation (with int8 KV); fp16 state blew up to 63k; fp32 mandatory |
| fused norm+quant for spec decode | acceptance 71.2 → 46.5 % → shipped OFF |
| int8 all-reduce transport | research-only; fp16 payload stays (AR is ~10 % of decode) |
| A8W8 at decode M / narrow N | their dispatch gate excludes it (§2.1): 0.61–0.95× |
| quantizing `in_proj_a/b` (exp(A) gates) | doctrine-excluded on sensitivity; on our checkpoints it's 47 MB anyway |
| "enable int8 KV" without a CUSTOM-backend path | lands on Triton attention → multi-× slowdown for us |
| quoting their tok/s against ours | different metric definitions + a skewed box (§1.5) |

## 5. Suggested sequence and roadmap entries

1. ~~**P1 + P2 probes** (§6)~~ **done 2026-08-31**: P2 → GO, P1 → NO-GO
   (results in §6).
2. **T1** int8 GEMV for the §2.2 mass, staged by expected value:
   `lm_head` first (single shape, 2.5 GB→1.27 GB dense / 1.02→0.51 GB MoE),
   then FA q/k/v (2.35 GB dense), then GDN in_proj_qkv/z + out_proj
   (MoE 2.0 GB), then shared expert / layer 0. Kill switch
   `VLLM_GFX906_INT8_F16_MASS=0`; per-stage serving A/B + KLD.
   → propose **ROADMAP `I1`**.
3. **T0** P82-style acceptance threshold (independent track, opt-in).
   → **ROADMAP `I2`**.
4. **T2** int8 KV: stage A (K only, capacity), then stage B (V).
   → **ROADMAP `I3`** (cross-ref G1: it must price the per-node tax first).
5. ~~**T3** A8W8 prefill via Triton~~ **closed by P1 (NO-GO)** — →
   `DEAD-ENDS.md`, not `ROADMAP.md`. Its HIP successor (**T5**, §3.10) is
   scoped by the same checkpoint arithmetic down to `v_dot8_i32_i4`, and its
   first step is a free ISA audit of our prefill inner loop, not a kernel.
6. **T5 step (a)** — ISA-audit the W4A16 prefill inner loop (packed-MAC
   usage?) → **P3a** HIP rate probe (dot4 / dot8, operand prep charged) →
   **P3b** W4A4 accuracy. → propose **ROADMAP `I5`** (Tier 2, probe-gated).

## 6. Probes (run on MI50 #0, 2026-08-31)

Both are standalone (torch + triton only, no vLLM import), take
`HIP_VISIBLE_DEVICES` from the caller, and print an explicit `GO`/`NO-GO`.
Logs: `/local/tmp/int8-probes/{p1-full,p2-full,p1-quick,p2-quick}.log`
(persistent path per `/local/git/AGENTS.md`). Environment: torch
2.13.0+gfx906.20260802001858, Triton 3.6.0, `HIP_VISIBLE_DEVICES=0`, GPU
idle before/after, host up 16:47 h (no degradation canary run — these are
bandwidth-bound, not sync-cadence-bound, so the MI50 host-degradation mode
documented in `/local/git/AGENTS.md` does not apply).

* **P1 `benchmarks/kernels/gfx906/int8_triton_dot_probe.py`** — decides T3
  and §3.4. (a) compiles an int8 `tl.dot` kernel, asserts exact equality vs
  an int32 reference, and dumps whether `v_dot4_i32_i8` appears in the
  emitted ISA (asm dict `kernel.asm['amdgcn']`, with the
  `objcopy --only-section`/`llvm-objdump` fallback from the
  `gfx906-isa-disassembly` skill); (b) measures int8-vs-fp16 `tl.dot`
  GMAC/s (expect ~1.96× if dot4 is real: records 25 877 vs 13 210);
  (c) runs the ported blockscale A8W8 GEMM against a same-quality fp16 Triton
  GEMM at M ∈ {4096, 1024, 256} × our production N/K, charging the
  activation-quant pass; (d) trunc vs `floor(x+0.5)` vs
  `tl.extra.hip.libdevice.round` in the act-quant kernel.
  **GO iff** dot4 present ∧ exact ∧ ≥ 1.3× at [4096, 34816, 5120].
  *(Ran 2026-08-31: dot4 ✓, exact ✓, **0.66×** → NO-GO, §6.2.)*
* **P2 `benchmarks/kernels/gfx906/int8_gemv_probe.py`** — decides T1.
  Triton M=1 int8-weight GEMV with per-row and per-128-group scales, at the
  real shapes ([248320,5120], [12288,5120]×16, [8192,2048]×30,
  [2048,4096], [512,2048]), against the fp16 baseline and the recorded HBM
  floor; prints achieved GB/s and the projected serving delta.
  **GO iff** ≥ 1.7× at lm_head and ≥ 1.6× at the ×30 GDN shapes.

### 6.1 P2 result — **GO** (1.93× at lm_head, 2.42× at the ×30 GDN shape)

M=1, Triton, int8 weights + fp16 activations (W8A16), per-output-channel
scale applied after the reduction; `BN=32, BK=512, SPLIT=1` unless noted;
int8-row correctness spot-checked at rel-err ≤ 3 × 10⁻⁴ against a dequantized
fp32 reference.

| shape (N×K) | ×/step | fp16 µs | int8 µs | ratio | int8 GB/s (% of 798 floor) |
|---|---|---|---|---|---|
| **248320×5120** lm_head dense | 1 | 3315.7 | **1716.9** | **1.93** | 741 (93 %) |
| 248320×2048 lm_head MoE | 1 | 1316.8 | 712.6 | 1.85 | 714 (89 %) |
| 12288×5120 FA q_proj dense | 16 | 224.8 | 112.5 | 2.00 | 559 (70 %) |
| 10240×5120 L0 in_proj_qkv | 1 | 169.8 | 84.8 | 2.00 | 619 (78 %) |
| 17408×5120 L0/mtp mlp gate,up | 2+2 | 274.0 | 134.9 | 2.03 | 660 (83 %) |
| 12288×5120 mtp q_proj | 1 | 205.3 | 106.2 | 1.93 | 592 (74 %) |
| 8192×2048 GDN in_proj_qkv MoE | 30 | 134.5 | 55.5 | **2.42** | 302 (38 %) |
| 8192×2048 FA q_proj MoE | 10 | 126.3 | 52.5 | 2.40 | 319 (40 %) |
| 5120×17408 L0/mtp mlp down | 1+1 | 290.9 | 191.6 | 1.52 | 465 (58 %) |
| 5120×10240 mtp fc | 1 | 279.9 | 192.5 | 1.45 | 272 (34 %) |
| 4096×2048 GDN in_proj_z MoE | 30 | 143.4 | 94.3 | 1.52 | 89 (11 %) |
| 2048×4096 GDN out_proj MoE | 30 | 128.9 | 95.3 | 1.35 | 88 (11 %) |
| 2048×8192 FA o_proj MoE | 10 | 175.1 | 181.4 | 0.96 | 92 (12 %) |
| 1024×5120 FA k_proj/v_proj | 16+16 | 47.0 | 26.4 | 1.78 | 198 (25 %) |
| 512×2048 shared gate/up, k/v MoE | 80 | 14.8 | 15.1 | 0.98 | 69 (9 %) |
| 2048×512 shared down MoE | 40 | 17.1 | 15.1 | 1.13 | 69 (9 %) |

Reads:

1. **Whenever the fp16 side is ≥ 55 % of the HBM floor, int8 lands
   1.93–2.03×** — the byte halving converts to wall time almost exactly. The
   two shapes above that bar that miss (down-proj 1.52×, mtp fc 1.45×) are
   this probe's own auto-geometry (`BK=128/SPLIT=8`: 465 and 272 GB/s where
   the tuned rows reach 660–741), i.e. tuning debt, not a wall.
2. **Below ~100 GB/s both sides are launch/occupancy-bound and the ratio
   goes to 1.0±0.05, never worse.** So the shapes where int8 does nothing are
   exactly the shapes where *nothing* byte-side does anything — the correct
   conclusion is "route them elsewhere / batch them", not "int8 fails".
3. **lm_head, the single biggest item, is also the cleanest**: int8 at
   1716.9 µs against the *recorded production* fp16 floor of 3114–3193 µs
   (`DEVLOG-dense-decode.md`) = **1.81× vs what we run today**, and 93 % of
   the int8 floor of its own. This is the result that makes T1 worth doing.
4. Per-128-group scales cost **+28 %** at lm_head (2202 vs 1717 µs) → use
   per-output-channel (§3.1).
5. `tl.dot`-based int8 GEMV (M padded to 16, dot4): **161× slower** (277 ms
   at lm_head). Dead; recorded so nobody retries it.
6. `torch.mv` fp16 is 85 GB/s at lm_head (30.6 ms) — 9× slower than the
   Triton fp16 GEMV. Confirms the `DEVLOG` line that hipBLAS is not our GEMV
   path and that these comparisons must be against *our* kernels.

**Reconciling the projection.** P2's self-reported saving (dense 4.60 +
0.67 ms/step, MoE 6.18 ms/step) is computed against this probe's own fp16
times, which are 2–3× off the production kernel at mid shapes — so it
overstates. §2.2 uses the floor-based number instead (dense −4.0 ms,
MoE −2.2…2.4 ms), and P2's role is only to establish the **conversion
factor** (1.9–2.0× at floor, ≥ 1.0 everywhere) that made that projection
legitimate.

### 6.2 P1 result — **NO-GO** for A8W8 (but the ISA question is settled yes)

* int8 `tl.dot` → `v_dot4_i32_i8` ✓ (16 in the ISA, no `v_mac_f32`/`v_fma_f32`),
  exact int32 vs fp64 ✓, and the A8W8 GEMM emits half the dot instructions of
  the same-work fp16 kernel (2048 `v_dot4` vs 4096 `v_dot2_f32_f16`, no
  spills) — the 2:1 the ISA ceiling predicts.
* Rates: int8 3.76–4.79 T MAC/s (14.5–18.5 % of the 25 877 dot4 record),
  fp16-Triton 1.53–2.82 T (11.6–21.3 % of the 13 210 record), and the best
  int8/fp16 tile ratio was **2.51×** — above the 1.96× ISA ceiling, which
  means the fp16 comparator was latency-stalled rather than issue-bound.
  Neither path is anywhere near its record; ratios at this level of
  efficiency are not portable facts, instruction counts are.
* Production-shape A8W8: **1.10–1.18× vs same-codegen fp16 Triton, 0.59–0.68×
  vs hipBLAS fp16** at all five shapes including the two that their dispatch
  gate was designed to catch (table in §3.3). Act-quant charged is 0.13 ms =
  0.14 % of the GEMM, so overhead was never the issue.

## 7. Corrections and open questions

* **C-3 (new, checkpoint scan + P1/P2 run):** the **MTP draft layer ships
  entirely unquantized** — 0.849 GB BF16 on the dense 27B (q/k/v/o 0.210 +
  `fc` 0.105 + MLP 0.535) and ~0.08 GB/step active on the MoE — and it is
  read on **every** MTP/spec-decode step. No `DEVLOG-*` budget line accounts
  for it, which matters twice over: it is a T1 leg in its own right, and any
  spec-decode step budget that omits it is ~1 ms short of the truth. Same
  class of finding: both checkpoints carry a **0.92 GB BF16 vision tower**
  (`model.visual.*`, 333 tensors) under a `*ForConditionalGeneration`
  architecture — worth confirming it is not resident in our text-only serving
  runs before sizing anything else (it would explain VRAM we attribute to
  other pools).

* **C-1 (our record):** `DEVLOG-dense-decode.md` §"Dense model facts"
  attributes 3.79 + 1.25 ms/step to `in_proj_b×48` / `in_proj_a×48` fp16
  GEMVs; the checkpoint ships those as `[48, 5120]` bf16 × 48 = 47 MB
  (~60 µs). The bucket *total* (LLGemm1 ≈ 7.1 ms) is confirmed by the
  §2.2 scan, so only the attribution is wrong. Worth a one-line fix in that
  log next time it is touched (do not re-derive the budget from those rows).
* **C-2 (our record):** `README.md`'s KV sizing line ("Qwen3.5-27B 20 KB;
  Qwen3.5-35B 10 KB per token") disagrees with the config arithmetic
  (27B: 16 FA layers × 4 KB = 64 KB/token TP=1; 35B: 10 layers × Hkv 2 ×
  4 KB/2 = 20 KB/token). Recompute before sizing anything for T2.
* **Q-1:** does Triton's blocked-layout int8 dot stage operands well enough
  to matter (P1a/b)? If it emits dot4 but at < 60 % of the recorded rate,
  T3 needs the llama.cpp `q8_repack` structure (BM=64, BK=4 sub-blocks,
  `sW_lo/sW_hi` double buffer, k-major `sWdh` scales, `sXd[BN][BK+1]`
  padding, DPP reduce with `s_nop` padding — `repack-common.cuh`) ported
  instead, which is a hand-kernel project. **Update (P1):** for the *GEMM*
  the answer is the worse one — dot4 is emitted but Triton reaches only 19 %
  of its rate, and that deficit (1.5–1.7× vs hipBLAS fp16) exceeds everything
  int8 can buy (§3.3). For the *GEMV* the answer is the good one: see Q-2.
* **Q-2 (answered by P2):** no — an int8 lm_head GEMV does *not* fall below
  the HBM floor once it carries a per-channel scale and a `v_cvt`: 741 GB/s
  (93 % of floor) in plain Triton, with the scale applied after the reduction
  so the inner loop stays 1 byte/weight + 1 cvt + 1 fma. **No `q8_repack`
  port is needed for T1**; the repack structure is only relevant if someone
  later wants dot-through int8 at M ≥ 16 (§3.3's reopening condition).
* **Q-3:** int8 K/V with **Q8_0 inline scales** (finer than
  `int8_per_token_head`) — do we keep it, or adopt the upstream
  per-token-head format so upstream ops (paging tests, `triton_attn`,
  `kv_cache_interface`) keep working? Keeping Q8_0 is better numerics per
  byte and zero extra stores, but it is a bespoke pool layout.
* **Q-4:** their `VLLM_GFX908_CK_FREE_GS128` pattern (free the superseded
  weight copy after requant) is required for any T1/T3 combination — do we
  quantize in place (`replace_parameter`) or keep an fp16 master?

## 8. Cross-links

`dequant-instructions.md` (dot rates, P·V contraction), `lds-layout.md`
(power-of-two strides, the q8 repack padding result), `latency-hiding.md`,
`README.md` (model records, knobs), `DEVLOG-dense-decode.md` /
`DEVLOG-moe-m1-sprint.md` (the fp16 GEMV floors), `DEVLOG-fa-attention.md` +
`DEVLOG-fa-legacy0-b1-decode.md` + `plan_fa_part_A.md` (Q8 read path,
LEGACY=0 negative), `oom-256k-prefill.md` (fp16 lm_head scratch),
`ROADMAP.md` (G1, C4, C7 = the 70 t/s target), `DEAD-ENDS.md`,
`REFRIGERATOR.md` (M6 Part C Q4-KV gate — the T2 neighbour).
gfx908 side: `docs/recipes/README.md` (canonical stack + numbers),
`INT8_AUDIT_RESULTS.md` (coverage inventory + experiment ledger),
`docs/mi100_decode_opt/p82_lossy_acceptance.md`, `scripts/kld_probe_v2.py`.
llama.cpp side: `ggml/src/ggml-cuda/q8_repack/README.md` @ `b7386eeae`.

Probes (this document): `benchmarks/kernels/gfx906/int8_triton_dot_probe.py`
(P1) and `benchmarks/kernels/gfx906/int8_gemv_probe.py` (P2); logs
`/local/tmp/int8-probes/`. Both are re-runnable and self-gating; P2 is the
one to re-run if the int8 GEMV geometry or the HBM floor assumption is ever
in doubt. Results are indexed in `DEAD-ENDS.md` (A8W8 row = DEAD,
T1 row = OPEN/IN-FLIGHT).
