# Nemotron-H onboarding (Nemotron-3.5-Lightning-30B-A3B mixed INT4/INT8) — serves at 70.4 tok/s on one MI50 after five gfx906 fixes

> Branch `gfx906/nemotron-h-onboard` off `main` (97a6fbe11a) · model
> `primitive-ai/Nemotron-3.5-Lightning-30B-A3B-mixed-INT4-INT8` (NFS HF
> cache snapshot `1a1521fb`) · 2026-08-29 · roadmap item `NH-1`–`NH-3`.

Model: 52 layers = 23 mamba2 + 23 MoE (E=128, topk 6, 1 shared expert,
non-gated relu² experts, sigmoid+bias router) + 6 GQA attention (2 KV
heads). compressed-tensors mixed: routed+shared experts INT4 **group-64**
symmetric; mamba in/out_proj, qkvo, lm_head INT8 **channel-wise**; MTP
head BF16 (unused). 19.2 GiB on disk, 3B active.

---

## 2026-08-29 — bring-up: three blockers, all fixed on this branch

**VERDICT:** SHIPPED (decode path) · **GATE:** greedy two-prompt smoke +
PPL probe A/B (365 tokens, 0 top-20 misses) + `_bench_gfx906.py` graph
serving below.

### Blocker 1 — fp32 router gate crashed `ops.LLMM1`

`GateLinear(force_fp32_compute=True)` stores fp32 weights on ROCm (no
specialized tiers apply) → tier-6 `F.linear` fp32 → the fork's
`rocm_unquantized_gemm` tiny-M interception fed fp32 into LLMM1, whose
TORCH_CHECK only accepts fp16/bf16. First MoE layer crashed the engine.

Fix: `_llmm1_tiny_m` (vllm/model_executor/layers/utils.py) returns None
for non-fp16/bf16 operands; callers fall back to `triton_matmul` /
`torch.nn.functional.linear`. Generic correctness fix — an unquantized
fp32 GEMM must not enter the fp16 skinny-kernel family.

### Blocker 2 — mamba2 SSD chunk-scan does not compile on triton-gfx906

`_chunk_scan_fwd_kernel` with `HAS_INITSTATES=True` aborts the triton
compile worker:

```
TritonAMDGPUTransforms/CanonicalizePointers.cpp:1430:
Assertion `(fatPtrs.at({thenFatPtrBase,...}) == fatPtrs.at({elseFatPtrBase,...}))
&& "expected then fat ptr canNarrow and else fat ptr canNarrow to be equal"' failed
```

Mechanism: the runtime `scf.if` selecting the previous-state source
yielded **two different base pointers** (`initstates_ptr` vs
`states_ptr`) plus per-branch strides; the fat-pointer narrowing pass
cannot merge that. `HAS_INITSTATES=False` (constexpr-folded branch)
compiled fine — the shape of the clue.

- `AMDGCN_USE_BUFFER_OPS=0` works around it globally (repro
  `/tmp/repro_ssd_chunk_scan.py`, both variants OK) but disables buffer
  ops for *every* triton kernel — not acceptable as a serving default.
- **Shipped fix:** restructure `ssd_chunk_scan.py` so each branch loads
  its own tile and the `scf.if` yields the *tile*, never a pointer
  (both the `BLOCK_SIZE_DSTATE <= 128` fast path and the k-loop path).
  Zero double loads; addresses are branch-local.
- Gate: `tests/kernels/mamba/test_mamba_ssm_ssd.py` — **94/94 passed**
  against the `ssd_minimal_discrete` reference (initial-states variants
  included). Engine multi-chunk prefill (init-states path) verified by
  the smoke tests.

This is a triton-gfx906 (3.6.x base) compiler limitation, not a vLLM
bug; the kernel-side restructure avoids depending on a triton bump.
Worth reporting upstream against their fork regardless.

### Blocker 3 — INT8-channel dense layers landed on Conch at 3.8 ms/GEMV

`choose_mp_linear_kernel` on ROCm has no kernel for channel-wise 8-bit
except `ConchLinearKernel` (installed in the venv). It "worked" — and
cost **3.79 ms per M=1 GEMV** (46 dense projections per decode step =
75% of GPU time; 4.95 tok/s total). Profile:
`/tmp/nemotron_prof.log` (eager, 24-step window).

Fix: new scheme `CompressedTensorsW8A16ChannelDequant`
(compressed_tensors/schemes/compressed_tensors_w8a16_channel_dequant.py)
— on gfx906 only, symmetric pack-quantized int8-channel dense layers
dequantize to fp16 at load (bias-128 convention, [N, K/4] i32 →
[N, K] fp16, chunked over N; packed tensors freed) and run the
optimized unquantized GEMV family via `dispatch_unquantized_gemm`.
Selection wired in `compressed_tensors.py` `_get_scheme_from_parts`.

Cost: +1.8 GiB VRAM (18.05 → 19.87 GiB weights+non-torch at util 0.90).
Gate below.

### Evidence (gates)

| config | result |
|---|---|
| greedy smoke (short + 300-tok multi-chunk) | coherent, `Paris.` ✓ (all arms) |
| PPL fp16 + gfx906 MoE + conch dense | 27.0216 |
| PPL fp16 + triton MoE + conch dense | 27.0026 |
| PPL bf16 + triton MoE + conch dense | 27.0026 |
| **PPL fp16 + gfx906 MoE + dequant dense (shipped)** | **26.9555** |

All within ±0.25% — noise band; the dequant path is exact int8→fp16.

---

## 2026-08-29 — MoE group-64 on the custom gfx906 W4A16 kernel

**VERDICT:** SHIPPED · **GATE:** serving A/B, graph mode, pp=2048/tg=256,
4 samples, same boot, `_bench_gfx906.py`, GPU0, util 0.95.

## HYPOTHESIS

If the oracle's 32/128 group gate is widened to any positive multiple of
32 (the kernel derives `groupsize = K/groups` arithmetically and the CT
repack passes `[E, G, N]` scales through unchanged), the Nemotron g64
experts run on the custom kernel bit-compatibly with the Triton WNA16
fallback, and the serving gain matches the historical Qwen g128 gap.

## What was done

- `oracle/int_wna16.py` `_gfx906_no_zp_reason`: (32, 128) → positive
  multiple of 32. Asymmetric gate left at 32/128 (no asym-g64 checkpoint
  to validate).
- `Gfx906WNA16Experts._supports_activation`: + `RELU2_NO_MUL` (kernel is
  activation-agnostic; `apply` already routes non-gated activations
  through `apply_moe_activation`). Without this the oracle silently fell
  to Triton even with the gate open.
- Tests: `test_gfx906_moe_gemm.py` + Nemotron shapes
  (1856/2688 × 2688/1856, gs 64, identity inter-GEMM for the non-gated
  layout) → 63 passed; `test_moe_wna16.py` gfx906 oracle gates → 24
  passed (g64 moved to the accept list, g48 is the new rejection case).

## Evidence — FOR

- PPL: 26.9555 (gfx906) vs 27.0026 (triton) — Δ0.17%, accumulation-order
  noise, same magnitude as the historical g128 A/B.
- **Serving A/B (the gate):** gfx906 **59.44 tok/s** (59.42–59.46) vs
  triton **31.48 tok/s** → **+88.8%**. Per-kernel census (launch-regime
  evidence): 38.4 µs/gemm × 46 gemms/step = 1.77 ms/step on the custom
  kernel.

## Evidence — AGAINST

None at the gate. Standalone numbers not measured (shape covered by the
unit tests instead).

## Interactions

- Reinforces: dispatch by weight format + shape, not model family — the
  only model-specific fact that mattered was relu²-no-mul.
- The `_supports_activation` gap is the second time a silent oracle
  fallthrough (not a crash) hid the custom kernel; cf. Gemma-4 no-zp
  (180f030ee3). Lesson: when a CT model "runs slow", grep the backend
  selection log line first.

## Refrigerated residue

- Asymmetric CT g64 widening: one-line gate change once a checkpoint
  exists to validate against.
- `moe_backend` override made it easy to A/B — keep using
  `BENCH_MOE_BACKEND=triton` as the control arm for future expert-kernel
  work on this model.

---

## 2026-08-29 — W8A16-channel dequant vs Conch (dense INT8 path)

**VERDICT:** SHIPPED · **GATE:** same serving A/B harness as above.

## HYPOTHESIS

If Conch's 3.79 ms M=1 GEMV is structural (generic triton, untuned for
MI50 skinny shapes), dequantizing the 71 int8-channel tensors to fp16 at
load and running the fork's GEMV family must recover decode throughput
at modest VRAM cost.

## What was done

New `CompressedTensorsW8A16ChannelDequant` scheme (see blocker 3).
Control arm: same tree with the scheme disabled (conch selected).

## Evidence — FOR

- **Serving (the gate):** 4.95 tok/s (conch) → **59.44 tok/s**
  (dequant) — **12.0×**, same MoE backend both arms.
- PPL: 26.9555 vs 27.0216 (conch arm) — dequant marginally *better*
  (exact dequant vs in-kernel rounding), both in the noise band.
- VRAM: weights+non-torch 18.05 → 19.87 GiB (+1.8); KV pool still
  ~8.2 GiB at util 0.90 / maxlen 8k.

## Evidence — AGAINST

- Decode reads 2× the weight bytes vs a native int8 GEMV: LLMM1 family
  is now 3.57 ms/step (57 × 62 µs, ≈890 GB/s — near the HBM floor for
  fp16, so the *only* way further down is an int8 GEMV, roadmap NH-2).
- Prefill pays a full fp16 GEMM on dequantized weights — measured TTFT
  impact not isolated; total pp2048+tg256 wall 4.31 s/sample.

## Interactions

- Removes the last Conch dependency from this model's serving path.
- Same pattern applies to any future CT int8-channel dense checkpoint on
  gfx906 (selection is platform+format-gated, not model-gated).

## Refrigerated residue

- `AMDGCN_USE_BUFFER_OPS=0` (global triton knob) also unblocks the
  chunk-scan compile; rejected as a default, kept as a diagnostic.
- Exllama channel-wise acceptance (groups=1 via `group_size=K`) was
  drafted and dropped: M>32 reconstructs per call (dequant + hipBLAS per
  invocation) would tax prefill; the fp16-dequant path dominates it.

---

## Search keys

## 2026-08-29 — fp32 router-gate GEMV on hipBLAS sgemv (NH-3)

**VERDICT:** SHIPPED · **GATE:** serving A/B, graph mode, pp=2048/tg256,
4 samples, same boot, plus the PPL probe.

## HYPOTHESIS

If the 128 µs fp32 triton matmul per router gate is a skinny-shape
artifact (1.3 MB weight read ⇒ ~10 µs at HBM speed), routing the M=1
fp32 GEMV to hipBLAS sgemv (`torch.mv`) must cut ~3 ms/step and show
≥ +10% in serving.

## What was done

`rocm_unquantized_gemm_impl` (vllm/model_executor/layers/utils.py):
fp32 single-token GEMVs take `torch.mv(weight, x_view[0])` before the
fp16 GEMV family (which rejects fp32). Micro-bench at the gate shape
[128, 2688]: F.linear fp32 141.8 µs · triton 117.6 µs · mv **8.0 µs**.

## Evidence — FOR

- **Serving (the gate):** 59.44 → **70.40 tok/s** (+18.4%), samples
  70.34–70.44 (`/tmp/bench_arm4.log`).
- PPL 26.9757 vs 26.9555 — Δ0.07%, fp32 accumulation-order change,
  same band as every other kernel swap on this model.

## Evidence — AGAINST

- M=2..32 fp32 (batched/spec-decode steps) still take the ~118 µs
  triton path — batched fp32 GEMV left open (roadmap NH-3 residue).

## Interactions

- The path only fires for fp32 operands — previously those *crashed*
  at LLMM1 on gfx906 (blocker 1), so no previously-working model's
  route changes. Blast radius: fp32-router models only.

---

## 2026-08-30 — TP=2 crash reports investigated: EP flag required + amdsmi wrapper fix

**VERDICT:** SHIPPED (wrapper fix + verification) · **GATE:** TP=2 +
`--enable-expert-parallel` serving boot + greedy A/B vs TP=1 (7 prompts,
token-ids/text diff) + multi-chunk prefill.

## HYPOTHESIS

If a user launches this model at TP=2, (1) the default no-EP config dies
in `CompressedTensorsWNA16MoEMethod.create_weights` because the g64 scale
groups do not divide the per-rank intermediate (1856/2 = 928, 928 % 64 = 32),
and (2) on boots whose post-torch amdsmi is broken (0 handles, `shut_down`
returns NOT_INIT), the first unprotected `get_device_name` (SSD config
lookup in the cudagraph-profile dummy run) kills the rank-1 worker inside
`with_amdsmi_context`'s `finally`, leaving rank 0 hung in an NCCL spin.

## What was done

- Reproduced (1) exactly: TP=2 default → `ValueError: ... intermediate
  size per tensor-parallel partition (928) to be divisible by group_size
  (64)` in `create_weights` (pre-existing upstream check, untouched by the
  onboard branch — the message itself prescribes the fix).
- Reproduced (2): rank-1 worker died in
  `selective_state_update → get_ssm_configs → get_device_name` with
  `AMDSMI_STATUS_NOT_INIT` from the wrapper's `finally: amdsmi_shut_down()`
  (`/local/tmp/nemotron_tp2_ep_server.log`); rank 0 then spun at 100 % CPU
  + GPU (NCCL) until SIGTERM. Fresh-process probe: post-torch
  `amdsmi_init()` returns success with **0 handles** and `shut_down()`
  raises NOT_INIT — deterministic this boot (single- and dual-GPU).
- Fixed `with_amdsmi_context` (vllm/platforms/rocm.py): cleanup failure in
  `finally` no longer masks a successful query (warning only); the
  0-handles GCN-arch fallback (`AMD_GFX906`) then carries the lookup.
  This is the permanent fix the 2026-08-22 C2-V entry in
  `degradation_details.md` called for (same crash signature, 35B-MoE
  rank-1 death in `get_device_name` during `profile_run`; that boot's
  workaround was a per-run sitecustomize shim). Test:
  `tests/test_rocm_amdsmi_context.py` (2/2).
- Verified TP=2 + `--enable-expert-parallel`: boots (64 local experts/rank,
  `GFX906_HIP` MoE backend still selected under EP), multi-chunk prefill
  (5041 tok = 2 chunks) + decode coherent.

## Evidence — FOR

- Greedy A/B TP1 vs TP2+EP, same tree (7 prompts, temp 0):
  5/7 token-identical; 2/7 (`story`, `mito`) diverge mid-output and stay
  coherent (quantized-kernel + all-reduce float diffs — expected). The
  whitespace-loop `transformer` prompt loops **identically** on both —
  model behavior, not EP corruption. Probe: `/local/tmp/nemotron_tp_ab.py`
  (JSONs `nemotron_tp_ab_tp{1,2}.json`).
- Steady-state decode TP2+EP B=1: ~100 tok/s (128-tok runs 99.4/101.4/
  101.0; 1.26–1.29 s) — above the 70.4 TP=1 healthy-host record, as
  expected (EP halves per-GPU expert traffic). **Caveat:** this boot's
  amdsmi is broken (fallback marker in every log) — the number stands but
  host state is suspect; re-confirm on a clean boot.

## Evidence — AGAINST

- None at the gate. Note the TP=1 arm on this same boot ran abnormally
  slow (first request ~7 tok/s, warming to ~35) while TP=2 ran at
  record level — a TP-dependent perf anomaly on a suspect boot, not
  adjudicated (canary + clean boot pending; see degradation.md rows).

## Interactions

- **Operational rule:** this model at TP>1 needs
  `--enable-expert-parallel` (g64 + pure-TP shard is unsatisfiable:
  1856/TP must be a multiple of 64, which only holds at TP=1 — or TP=29,
  which is not a deployment; 128 % 2 == 0, so EP=2 is exact).
  Recorded in the workspace AGENTS.md local-serving notes.
- The wrapper fix is generic (any `get_device_name`/`get_device_uuid`/
  memory query on a broken-amdsmi boot); blast radius: none on healthy
  boots (shut_down succeeds → no warning path).

## Refrigerated residue

- This boot's perf anomaly (TP1 decode ~7–35 tok/s vs TP2 ~100, boot O,
  amdsmi broken since ~08-30 morning) is an open question for the
  degradation TP-dependence line — needs the canary on a clean boot.
- Tuned SSD-decode configs for `AMD_GFX906` do not exist (lookup falls to
  the default launch config on every boot here); tuning is a perf item,
  orthogonal to this incident.

## 2026-08-30 (later) — review follow-up on the wrapper fix

**VERDICT:** SHIPPED · **GATE:** `tests/test_rocm_amdsmi_context.py` (2/2,
with cleanup + diagnostic-contract asserts) + module import + plugin probe
on this box.

Self-review of `e4a4dd0edc` found four diagnostic-integrity gaps, all in
the exact scenario the fix exists for; fixed same day:

- The "after a successful query" warning also fired on the failed-query
  path (finally runs both ways). Now branched in `_shut_down_amdsmi()`:
  warning on success, debug on failure — the propagating query exception
  dominates there.
- `warning_once` defaulted to `scope="local"` → dropped on non-first
  local ranks, i.e. the very rank that died in the TP=2 incident. Now
  `scope="process"`.
- Same `finally: amdsmi_shut_down()` pattern in
  `platforms/__init__.py::rocm_platform_plugin` fed the NOT_INIT error
  into the outer except and misattributed the debug log ("not available
  because: AmdSmiException" instead of "no GPU found"); return value was
  unaffected (assignment precedes the finally). Fail-soft swallow+debug
  applied symmetrically; no dedicated test — the only observable delta
  is log text, which is brittle to pin.
- Tests now pin the cleanup half of the contract (`init/shutdown
  assert_called_once`) and the warning contract (fires once at process
  scope on success; absent on the failure path); failure test switched
  to `pytest.raises`.

Not actioned (deliberate): decorator order on `get_device_name`
(`with_amdsmi_context` outside `lru_cache` → per-call init/shutdown even
on cache hits) and broken-boot `_GCN_ARCH` initializing CUDA at import —
both pre-existing design, not regressions; noted for the roadmap if the
broken-amdsmi boots persist. No commit trailer on `e4a4dd0edc` (policy
miss, local fork, left as-is).

---

## 2026-08-30 (later) — NH-2: int8 in-kernel W8A16 GEMV/GEMM — NO-GO

**VERDICT:** NO-GO (Triton int8 in-kernel, measured on MI50 2026-08-30;
branch `gfx906/nh2-int8-gemv`, code lands with the env default OFF)
· **GATE:** probe at Nemotron's exact dense shapes vs the actual
production dispatch — the gate failed at the M>1 arms; no serving A/B
was needed.

## HYPOTHESIS

Nemotron's per-step dense weight traffic is 1.32 GiB of int8 across the
71 CT W8A16 channel tensors (23× mamba in_proj [10304, 2688] + 23×
out_proj [2688, 4096] + 6× GQA q/k/v/o + lm_head [131072, 2688]). The
scheme dequants to fp16 at load (2× the traffic, +1.3 GiB VRAM) and
serves M=1 via LLMM1 rpb=4 (K=2688/4096 miss the custom
dense_gemv_gfx906 whitelist). If in-register dequant is cheap, serving
the packed bytes halves the traffic; the P2 probe
(`DEVLOG-int8-transfer.md`) had measured this pattern at 1.93–2.03× on
HBM-bound shapes, "never worse than 0.96×".

## What was built (kept on the branch, env-gated, default off)

- Scheme: int8 path pre-shifts the packed bytes in-place at load
  (`byte ^= 0x80` → two's-complement of (byte−128), 1-op/weight, no
  copy) so `weight_i8` is a signed int8 view and the kernels run the
  3-op element chain. New Triton kernels `k_w8a16_gemv` (M=1) /
  `k_w8a16_gemm` (M>1; tile dequant bit-identical to the load-time
  dequant, so M>1 numerics = dequant path numerics). `VLLM_GFX906_W8A16_INT8=1`
  enables (default 0); the dequant path is bit-identical to before.
- Probe: `benchmarks/kernels/gfx906/bench_w8a16_gfx906.py` — full
  config sweep (BN×BK×SPLIT / BM×BN×BK×warps) at all six shape
  families × M=1/4/4096 vs `rocm_unquantized_gemm_impl`, correctness
  vs fp32 reference with 0x80 zero-codes pinned. 18/18 correctness
  checks pass (2.0–3.9e-4 rel-err).

## Measured (MI50, torch 2.13+gfx906, triton 3.6; /local/tmp/w8a16_probe_full.log)

M=1, best config per shape (cur = production dispatch on dequanted fp16):

| shape | N×K | ×/step | cur µs | i8 µs | ratio | i8 % floor |
|---|---|---|---|---|---|---|
| mamba in_proj | 10304×2688 | 23 | 70.1 | 49.3 (BN64 BK128 S1) | **1.42×** | 142 % |
| mamba out_proj | 2688×4096 | 23 | 36.4 | 51.3 (BN16 BK512 S1) | 0.71× | 372 % |
| GQA o_proj | 2688×4096 | 6 | 37.1 | 51.7 | 0.72× | 375 % |
| GQA q_proj | 4096×2688 | 6 | 36.3 | 28.2 (BN32 BK128 S1) | **1.29×** | 204 % |
| GQA k/v_proj | 256×2688 | 12 | 11.8 | 16.9 (BN16 BK128 S4) | 0.69× | launch-bound |
| lm_head | 131072×2688 | 1 | 850.2 | 530.8 (BN32 BK128 S1) | **1.60×** | 120 % |
| **step total** | | | **3882.7** | **3526.9** | **1.10×** | |

M>1 (best config of 12 each): M=4 best 0.55–0.80× vs cur (in_proj
395.7 vs 239.3; out_proj 234.6 vs 62.2; lm_head 3158.7 vs 2511.5);
M=4096 best 0.19–0.47× vs hipBLAS (in_proj 50.1 vs 14.1 ms; out_proj
21.6 vs 7.2 ms; lm_head 929.8 vs 301.0 ms). num_warps=8 was 1.5–3×
slower than 4 on every GEMM config.

## Why the P2 GO did not transfer

- P2's evidence set was N ≥ 10K at M=1 (248320/17408/12288/10240);
  its own table already showed the mid-N collapse (4096×2048 int8 at
  **11 % of floor**) and flagged its fp16 baseline 2–3× off production
  at mid sizes. Nemotron's dense set is exactly the mid-N band where
  the hand-tuned CUDA (LLMM1 rpb=4 / dense_gemv_m4) runs at 400–790
  GB/s and this Triton at 170–600 GB/s. The two only cross at
  N ≳ 10K — hence lm_head/in_proj/q_proj win, out_proj/o_proj/k_v
  lose.
- The serving mode is the deal-breaker: local Nemotron config runs
  ngram spec n=5 → **M=6/step** (to 24 at batch 4), where the int8
  path is the Triton GEMM (0.5–0.8×), plus M=4096 prefill (0.19–0.47×).
  An M=1-only int8 hybrid is impossible without re-materializing the
  fp16 weight for the M>1 path (3× VRAM).
- The byte savings only materialize with a hand-tuned int8 CUDA
  kernel: M=4 in_proj cur = 239 µs (464 GB/s fp16); an int8 kernel at
  ~700 GB/s would be ~79 µs (3×). That is **NH-2′** — a CUDA port of
  the hand-tuned GEMV family (LLMM1 / dense_gemv_m4 / dense_gemv) with
  byte loads, not Triton; parked in the ROADMAP with this table as the
  evidence base.

## Interactions

- Nothing merges as a default change: env default OFF, dequant path
  bit-identical. The branch is safe to merge for the probe + tests +
  opt-in kernel only.
- If a future model has an N ≥ 10K int8-channel GEMV-dominated decode
  (Qwen-class lm_head) and plain M=1 serving, `VLLM_GFX906_W8A16_INT8=1`
  is a real win there (lm_head 1.60× measured).

---

## 2026-08-30 (evening) — NH-5: topk chain node removal — SHIPPED

**VERDICT:** SHIPPED · **GATE:** serving A/B (A–B–A), graph mode, TP=1,
~2048-token prompt / tg256, 4 samples/arm, same boot (boot O, post-
amdsmi-flip), streaming client, GPU0, util 0.90.

## HYPOTHESIS

If Nemotron's `grouped_topk` chain is node-heavy only because
`n_group=1/topk_group=1` makes two of the three `aten::topk` calls and
the group-mask machinery provable no-ops, and if C1's rule holds (node
REMOVAL transfers to serving; topk REPLACEMENT does not), then removing
those nodes — without touching the surviving topk kernel — recovers most
of the roadmap's ~1.2 ms/step topk-chain cost.

## What was done

Serving-mode census (torch trace of the running server,
`/local/tmp/nh5_prof/`): per MoE layer per step the chain is
`triton_poi_fused_add_sigmoid_unsqueeze_0` (sigmoid+bias) + `aten::topk`
×3 + `triton_red_...gather/sum` (weights) + `moe_align_block_size`
(2 kernels). The 3 topks are the torch-compiled `grouped_topk`
(`grouped_topk_router.py`) — the fully-fused `ops.grouped_topk` single
kernel is unreachable on this fork: its gate requires
`current_platform.is_cuda()`, which is **False** on this ROCm fork
(dead path on ROCm here; not touched).

Two folds, both pure node removal (3 kernels/layer):

1. `grouped_topk_router.py`: new `_grouped_topk_single_group` fast path
   for `n_group==1 and topk_group==1` (Nemotron: E=128/topk=6/sigmoid/
   bias/renorm/scale=2.5). In the generic chain, group_idx is always 0,
   group_mask all ones, masked_fill the identity — so it degenerates to
   one topk over `sigmoid(logits)+bias` with weights gathered from the
   pre-bias scores. The surviving selection is the SAME `aten::topk`
   kernel on a bit-identical input; the weights path is byte-identical.
   Env `VLLM_GFX906_TOPK_SINGLE_GROUP`, default ON.
2. `moe_align_m1_gfx906.cu`: the C1 stage-1 fused align+count kernel
   (E=256/topk=8 only) templated on (E, topk) with a (128, 6)
   instantiation; dispatcher gate `_ALIGN_M1_SHAPES` in
   `gfx906_w4a16_moe.py` now admits both pairs. One 128-thread CTA
   replaces the align + count_and_sort pair.

## Evidence FOR

- **Gate — serving A/B (A–B–A, same boot):**

  | arm | folds | decode t/s (4 samples) | mean | ms/step |
  |---|---|---|---|---|
  | A (OFF) | both 0 | 107.1 / 107.1 / 106.1 / 106.9 | 106.8 | 9.36 |
  | B (ON) | defaults | 114.7 / 114.6 / 114.5 / 114.6 | 114.6 | 8.73 |
  | A2 (OFF control) | both 0 | 107.8 / 107.8 / 107.8 / 107.7 | 107.8 | 9.28 |

  **+7.3 % to +7.8 % decode (0.63 ms/step removed)**; the ON arm sits
  squarely between two OFF arms, samples stable to ±0.1 %.
- Launch-regime (isolated, eager, MI50): group topk#1 (top2/128 + sum)
  24.3 µs, group topk#2 (k=1 over 1) 7.2 µs, generic align+count pair
  19.4 µs → fused 3.2 µs ⇒ ~47.6 µs/layer × 23 layers =
  **1.09 ms/step predicted** (the roadmap's ~1.2 ms). In-graph transfer
  ~58 % (the in-graph topk nodes are cheaper than eager launches; C1
  stage 1 transferred within 8 %).
- Correctness: `tests/kernels/moe/test_grouped_topk_single_group.py`
  **19/19** — fast path bit-equal (`torch.equal`) to the generic chain
  for random + tie-heavy (all-equal, block-tie) inputs across
  sigmoid/softmax × bias × renormalize × scale at (1,128,6),
  (4,128,6), (1,256,8), plus the compiled env-toggle (inductor) pair;
  `tests/kernels/moe/test_moe_align_m1_gfx906.py` **51/51** — (128,6)
  bit-equal to the 2-kernel chain + cudagraph-capture replay, (256,8)
  unchanged. ruff clean.
- PPL (`ppl_probe.py`, in-process, util 0.90 — see AGENTS note; the
  0.95 default no longer fits beside the grown llama-server): OFF
  27.0041 / 27.0177, ON 27.0492 / 27.0688 — Δ +0.18 %, the same
  magnitude as the established inter-arm noise band (08-29 arms
  26.9555–27.0216). Mechanism if real: the smaller ON graph lets
  inductor re-fuse the renormalize reduction differently → ULP jitter in
  weights, NOT fold logic (eager bit-equality proven above). Coherent
  greedy output on both arms.
- Boot note: boot O's 14:2x TP=1 collapse (7→35 t/s, DEG row in
  `degradation.md`) did NOT reproduce ~4 h later — all three A/B boots
  ran 106–115 t/s steady, with the amdsmi-fallback lines still in the
  logs. TP=1 numbers from this window are usable.

## Evidence AGAINST

- Greedy serving output is run-to-run mult-modal for this model
  (expert-gemm atomic accumulation: two consecutive OFF-arm runs matched
  on only 1/4 of 128-token prompts), so cross-arm token-identity is not
  a valid gate here — ON outputs sat inside the OFF mode set
  (`/local/tmp/nh5_{on,off,off2}_out.json`), but identity is unprovable
  e2e; the unit-level bit-equality + PPL are the correctness record.
- PPL ON samples read at/above the top of the historical 26.96–27.02
  band (27.05–27.07); see mechanism above. Magnitude = historical
  inter-arm Δ, not an outlier.

## Interactions / follow-ups

- `ops.grouped_topk` (single-kernel sigmoid+bias+grouped-topk) remains
  dead on this fork (is_cuda() gate). Enabling it for ROCm would be a
  topk REPLACEMENT — C1's evidence (S2 −1.03 %, stage 2 −1.10 %) says
  that loses in serving; parked, not NH-5 scope.
- The (128,6) align instantiation is shape-gated (fail-closed
  TORCH_CHECK); other (E, topk) pairs keep the generic 2-kernel chain.
- Remaining topk-chain cost per layer: the surviving top-6/128
  (`aten::topk`, ~19 µs isolated) + sigmoid/bias + weights + fused align
  ≈ 4 kernels — next fold would need to replace that topk (C1-forbidden
  without new design) or fold topk into the gate GEMV epilogue (the
  original C1 scope, open).

---

## Search keys

`HYPOTHESIS:` `VERDICT:` Nemotron, nemotron_h, group-64, g64, relu2,
RELU2_NO_MUL, W8A16 channel, int8 channel, dequant, Conch, ConchLinearKernel,
CanonicalizePointers, HAS_INITSTATES, chunk_scan, ssd_chunk_scan, fp32 router,
LLMM1, GateLinear, force_fp32_compute, AMDGCN_USE_BUFFER_OPS, TP=2,
tensor-parallel, enable-expert-parallel, EP, expert parallel, 928,
with_amdsmi_context, AMDSMI_STATUS_NOT_INIT, amdsmi, get_device_name,
_shut_down_amdsmi, rocm_platform_plugin, scope=process,
profile_run, rank-1 worker, NCCL hang, NH-2, NH-2 prime, NO-GO,
weight_i8, bias-128, 0x80, k_w8a16_gemv, k_w8a16_gemm, SPLIT,
VLLM_GFX906_W8A16_INT8, bench_w8a16_gfx906, mid-N, spec M=6,
dense_gemv_m4, 464 GB/s, NH-5, topk chain, grouped_topk, single-group,
VLLM_GFX906_TOPK_SINGLE_GROUP, VLLM_GFX906_ALIGN_M1, moe_align_m1_gfx906,
(128,6), (256,8), node removal, fold don't replace, is_cuda False,
ops.grouped_topk dead path, 114.6, 106.8, 0.63 ms, A-B-A, mult-modal
greedy, atomic accumulation, PPL 27.05.

---

## 2026-08-30 (night) — NH-4: mamba2 grouped gated-norm fused path — SHIPPED

**VERDICT:** SHIPPED · **GATE:** serving A/B (A–B–A), graph mode,
TP=2 + EP (`--enable-expert-parallel`), max-len 8192 / tg256, 4
samples/arm, fresh boot per arm (gate read at worker init), streaming
client, both GPUs, util 0.90.

## HYPOTHESIS

If Nemotron-H's 23 mamba layers each run `Mixer2RMSNormGated` with
`n_groups=8` through the ~8-launch eager chain (per-group norm + gate
muls) instead of the fused Triton `rms_norm_gated` kernel, then routing
the grouped case through that existing kernel — env-gated, only when
`per_rank_hidden_size % group_size == 0` — removes launch-tail overhead
without changing numerics.

## What was done

One file changed: `mamba_mixer2.py`. `forward_cuda` now checks
`VLLM_GFX906_MAMBA_FUSED_GROUP_NORM=1` and, when the per-rank hidden
size divides evenly into groups (algebraically identical to
`n_groups % tp_size == 0`, which excludes the redundant all-gather
case), calls the existing fused Triton kernel with `group_size` — the
same call shape as the n_groups==1 path. No changes to
`layernorm_gated.py`. New test file
`tests/kernels/mamba/test_mixer2_grouped_gated_norm.py`, 11 tests.

## Evidence FOR

- **Gate — serving A/B (A–B–A, fresh boot per arm):**

  | arm | gate | decode t/s (4 samples) | mean | PPL (377 tok) |
  |---|---|---|---|---|
  | A | 0 | 109.8 / 109.8 / 109.8 / 109.8* | ~109.8 | — |
  | B | 1 | 110.0 / 110.1 / 110.1 / 110.1 | 110.05 | 24.9034 |
  | A2 | 0 (control) | 109.3 / 109.4 / 109.3 / 109.4 | 109.37 | 24.8944 |

  (*arm A per-sample values from the pre-compaction run; recorded mean
  ~109.8 t/s.) B sits +0.4 % above the two OFF arms — inside inter-arm
  noise, i.e. **no serving regression** from the fused path. The
  isolated kernel bench (`/local/tmp/nh4_bench_gated_norm.py`) showed
  eager ~68 µs/layer vs fused ~55 µs (1.2–1.6× per layer, ~0.29
  ms/step over 23 layers); that does not move end-to-end t/s at this
  batch because the decode step is MoE-GEMV-bound, not mamba-tail-bound.
- Correctness: unit suite **11/11** — grouped fused output bit-equal
  (fp16 tolerance) to `forward_native` across geometries incl.
  production TP=2 shape (8×1024 groups, per-rank 4096); gate OFF leaves
  every path byte-unchanged; TP-driven partial-group case (8 groups at
  TP=16 → per-rank 512 < 1024) provably refuses the fused path
  (sentinel never fires). ruff clean.
- **TP=2 regression driver** (`/local/tmp/nh4_tp2_regr.py`): 6/6 — env
  OFF unchanged vs main, env ON bit-equal at (64,1)/(64,2)/(64,4) under
  real TP=2 process groups.
- PPL: B 24.9034 vs A2 24.8944 — Δ +0.04 %, zero top-20 misses on both;
  well inside inter-arm noise. (Serving-side prompt-logprob PPL, 12
  fixed prompts, same estimator both arms: `/local/tmp/nh4_ppl_client.py`.)

## Evidence AGAINST

- No measurable end-to-end t/s gain in this config (+0.4 %, within
  noise). The win is launch-count reduction (~0.29 ms/step isolated),
  which becomes visible when the step is not GEMV-bound (smaller batch,
  spec-decode mid-N, or after NH-2′ shrinks the MoE GEMVs). Gate default
  stays OFF until such a config shows it; flipping is a one-line env
  default change.

## Interactions / follow-ups

- EP requirement reconfirmed for serving A/B: plain TP=2 crashes on this
  model's g64 scale groups (`create_weights`); arms ran with
  `--tensor-parallel-size 2 --enable-expert-parallel`. Mamba layers stay
  tensor-parallel under EP (per rank: 4 groups of 1024) — the fused path
  is exercised per-rank, which is exactly what the TP=2 unit test covers.
- Arms launched via `systemd-run --user -p MemoryMax=infinity`
  (`/local/tmp/nh4_launch.sh`): the background-terminal worker cgroup
  (~4 GB cap) OOM-kills vLLM NFS weight loads; foreground scope is
  unlimited but a 300 s+ boot would eat the call.
- Review protocol (self-review + Claude CLI review of branch vs main,
  merged): both found the same two test gaps (no real tp_size=2 test;
  non-physical partial-group geometry) — both fixed, suite re-run green.

---

## Search keys (NH-4 additions)

`VLLM_GFX906_MAMBA_FUSED_GROUP_NORM`, Mixer2RMSNormGated, n_groups=8,
group_size, rms_norm_gated fused kernel, mamba_mixer2.py, forward_cuda,
per_rank_hidden_size % group_size, redundant all-gather exclusion,
TP-driven partial groups, sentinel _FusedCalled, 109.8, 110.05,
109.37, PPL 24.90, systemd-run MemoryMax=infinity, cgroup OOM weight
load, nh4_launch.sh, nh4_ppl_client.py, prompt_logprobs API dict shape,
decoded_token lookup, EP serving A/B, no t/s gain GEMV-bound step.

---

## 2026-08-31 — NH-2′: CUDA int8 W8A16 GEMV serving A/B gate — NO-GO (M-mismatch)

**VERDICT:** NO-GO at serving level (measured on MI50, 2026-08-31; branch
`gfx906/nh2c-int8-cuda`, checkpoint `8b0c2e38b9`). The CUDA kernel is
correct and fast *in isolation* but the serving A/B is a **−61% regression**
because the M distribution it was gated on (M=4) is not the M distribution
serving actually produces.

· **GATE:** serving A-B-A, spec-decode ngram n=5, TP=2+EP, port 8931, local
NVMe weights, per-arm isolated `VLLM_CACHE_ROOT`, warm (untimed pre-pass) +
median-of-3 timed runs; PPL via prompt_logprobs on both arms.

### Symptom / paradox

The M=4 micro-bench gate passed with a strong win (in_proj [10304,2688]
239 µs → 72 µs = **3.3×**, out_proj 245 → 74 µs = 3.3×). Yet the serving A/B
measures armB (CUDA int8 active) at **46.1 t/s** vs armA (baseline dequant)
at **119.2 t/s** — a **−61% regression**, uniform across every prompt. The
kernel is provably faster on the shape it was tuned for, but serving gets
slower. That is the contradiction this section resolves.

### Run recipe (what actually worked after 3 infra blockers)

- Launch: systemd **user service unit** `nh2p-arm{A,B}.service`
  (`Type=simple`, `MemoryMax=infinity`) → `/local/tmp/nh2p_serve.sh <arm>`.
  This escapes the 4 GiB cgroup cap that OOM-killed vLLM workers during
  host-RAM weight staging (kernel log: `usage 4194304kB, limit 4194304kB`).
  Do **not** set `VLLM_ENABLE_V1_MULTIPROCESSING=0` (collapses the engine
  into the API process and breaks TP=2 worker spawn).
- Weights: local NVMe copy `/local/cache/huggingface/nemotron35` (NFS
  safetensors concurrent-mmap flaked with `UntypedStorage` on shard 2; local
  copy removes NFS as a variable entirely).
- Required flags for TP=2+EP: `--tensor-parallel-size 2
  --enable-expert-parallel --disable-custom-all-reduce`. Missing the last one
  → NCCL-init hang (workers sleep at 0% CPU after "using nccl==2.30.4").
- **Per-arm `VLLM_CACHE_ROOT=/local/tmp/vllmcache_<arm>` is REQUIRED.** The
  torch.compile AOT cache hash does NOT include the `VLLM_GFX906_W8A16_INT8`
  gate, so armB replayed armA's cached graph (traced under the dequant path)
  and crashed with `KeyError: 'weight'` during `determine_available_memory →
  profile_run → _dummy_run`. Isolate each arm's cache or the A/B is unsound.

### Measured (MI50, torch 2.13+gfx906; /local/tmp/nh2p_gate_arm{A,B}.json)

| | armA (baseline dequant) | armB (CUDA int8 active) | Δ |
|---|---|---|---|
| Decode t/s (warm, median×3, 7 prompts) | **119.2** | **46.1** | **−61%** |
| PPL (12 prompts, 377 tok, 0 top-20 misses) | 24.9260 | 24.8826 | +0.002 (noise) |

Per-prompt t/s is uniform in the regression direction (no single outlier
swamps it — that was a v1 protocol artifact, fixed by adding an untimed
warmup pass so per-prompt cudagraph/JIT first-request cost lands there).
PPL delta is within noise → **armB is not "slow AND wrong", just slow.**

### The M distribution (root cause) — captured in eager mode

Instruments inside the compiled `apply_weights` region are impossible under
this fork's `aot_compile_fullgraph`: file I/O, `print`, and even a
`@torch.compiler.disable`d recorder each trip Dynamo
(`Unsupported: Attempted to call function marked as skipped` /
`Failed to trace builtin operator`). The only way to read the live M
distribution was an **eager-mode** boot (`--compilation-config
'{"mode":"NONE"}'`) with a one-line `print` (safe when uncompiled), then a
128-token decode. Result (both TP ranks, per (m,n,k)):

```
3422  m=1 n=2688 k=2048     708  m=1 n=2304 k=2688    120  m=1 n=65536 k=2688
2714  m=1 n=5152 k=2688     264  m=6 n=2304 k=2688     58  m=5 n=2688 k=2048
1276  m=6 n=2688 k=2048     1012 m=6 n=5152 k=2688      56  m=4 n=2688 k=2048
  46  m=5 n=5152 k=2688      44  m=6 n=65536 k=2688     44  m=4 n=5152 k=2688
  12  m=5 n=2304 k=2688      12  m=4 n=2304 k=2688       2  m=4 n=65536 k=2688
```

Totals: **m=1 = 72%**, **m=6 = 28%**, m=4 + m=5 = ~2%. This is the smoking
gun. The kernel's M≤4 gate fires on m=1 (72%) and m=4/m=5 (~2%), but the
**m=6 case — 28% of all calls, including every full-acceptance spec step —
falls through to the Triton int8 GEMM**, which the NH-2 devlog already
measured at **0.5–0.8×** of the fp16 baseline (in-register per-tile dequant).

### Why the M=4 micro-bench gate did not predict this

The kernel was tuned and gated on **M=4 in_proj = 3.3×**, but under ngram
spec-decode the verification step runs at **M=5–6** (1 real token + up to 5
drafts), and single-token steps run at M=1. M=4 is only ~1% of calls. So:

- The 72% m=1 traffic *does* hit the fast CUDA kernel, but M=1's absolute win
  is far smaller than M=4's (lm_head M=1 was actually *below* Triton in the
  micro-bench: 544 vs 730 GB/s).
- The 28% m=6 traffic hits the **slow** Triton int8 GEMM fallback.
- Net across the real mix: −61%.

The gate measured an operating point (M=4) that essentially never occurs in
this serving mode, and did not measure the two that do (M=1, M=6).

### Ruled out

| hypothesis | status | evidence |
|---|---|---|
| GPU1 `hipErrorLaunchFailure` wedge | ruled out (transient) | probe: matmul + int8 launch both pass; retry boots clean; no dmesg errors |
| stale `_rocm_C.so` (BUILD_RC=1) | ruled out | BUILD_RC=1 was a **versioning** failure (`vcs_versioning` can't parse tag `gfx906-main-pre-promotion`; `SETUPTOOLS_SCM_PRETEND_VERSION` is ignored by it), not a compile error. `.so` (01:14) predates the failed build and contains all 3 kernel symbols (`dense_gemv_i8_gfx906`, `_m4_`, `_gfx906`) |
| armB crash = bug in my dispatch | ruled out | `KeyError 'weight'` was armA's stale AOT cache replayed under INT8=1; per-arm `VLLM_CACHE_ROOT` fixes it. My int8 path correctly uses `layer.weight_i8` (no `['weight']` lookup) |
| M-histogram instrumentation bug | ruled out (but unusable in-graph) | the recorder is correct, but fullgraph compilation rejects any graph break / disable-marked call / print inside the region — must run eager |
| "slow AND wrong" (int8 dequant shifts logits) | ruled out | PPL armA 24.9260 vs armB 24.8826, 0 misses both; delta is noise |

### Next steps / how this could become a GO

- **The kernel must serve M=5–6.** That is the actual serving operating
  point (28% of calls, and the full-acceptance spec step). A CUDA int8 GEMM
  at M≤6 that beats the fp16 baseline (not just the Triton-int8 fallback) on
  in_proj/out_proj would flip the sign. This is a new kernel target, not a
  config tweak — the existing `dense_gemv_i8_m4` family caps at M=4.
- **Re-gate on the real M mix**, not M=4: measure m=1 and m=6 in_proj +
  out_proj against the fp16 baseline (the production dispatch), weighted by
  the 72/28 split above. If m=1 is ~parity and m=6 is a loss, there is no win
  to capture without fixing m=6 first.
- Keep the kernel + tests on the branch env-gated default-OFF (safe to merge
  as opt-in); do **not** enable it by default.

### Interactions

- Nothing merges as a default change: `VLLM_GFX906_W8A16_INT8` /
  `_INT8_CUDA` stay OFF; the dequant path is bit-identical.
- The M-distribution finding (m=1 72% / m=6 28%, m=4 ~1%) is reusable for any
  future NH-2′/int8-decode work: it supersedes the "M=4 in_proj" framing in
  the ROADMAP and the NH-2 section above.

*Keywords: NH-2', CUDA int8 W8A16 GEMV, serving A/B gate NO-GO, M-mismatch,
ngram spec m=1/m=6 distribution, aot_compile_fullgraph no graph break,
per-arm VLLM_CACHE_ROOT KeyError weight, vcs_versioning tag parse BUILD_RC=1,
systemd user service MemoryMax=infinity, local NVMe nemotron35 weights, eager
mode MLOG, PPL 24.9 noise, dense_gemv_i8_m4 M≤4 cap.*
