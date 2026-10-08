# Dev log — Gemma-4 MoE: no-zero-point W4A16 expert kernel for gfx906

Branch: `gfx906/gemma4-moe-nzp` (from `gfx906/main` @ `934de9b910`, 2026-08-19)

## Goal

Serve Gemma-4-26B-A4B-AWQ's MoE layers (30 × {gemm1 N=1408 K=2816, gemm2
N=2816 K=704}, E=128, topk=8, AWQ-style W4A16 group-32 **symmetric, no zero
points**) on our `moe_gemm_q4` gfx906 kernel instead of Triton WNA16.

Target (the Gemma-4 onboarding item in `roadmap-more-models.md`): Triton
232.8 µs/call × 59.8 calls/step
(13.9 ms, 46% of GPU-busy) → ~60–90 µs/call class → ~10 ms/step recovered →
37.6 t/s → ~50–55 t/s class.

House gate sequence: format/repack probe → unit tests → PPL (chat-templated —
see traps) → greedy/token A/B → serving A/B (4 samples) → flagship regression.

## Known traps

- **Gemma-4 is a thinking-mode instruct model.** Raw continuation prompts
  degenerate into confident repetition loops. All quality gates (greedy / PPL)
  MUST use the chat template (`apply_chat_template` / chat API).
- Load recipe: fp16 (auto-fallback from bf16 config, `69f615b98a`),
  `HF_HUB_OFFLINE=1`, local snapshot
  `/local/cache/huggingface/hub/models--cyankiwi--gemma-4-26B-A4B-it-AWQ-4bit/snapshots/0ef577a5710035bd2d3a3f27e4f5cb2e86a9a9ba`.
- Baseline record: 37.6 t/s (graph, pp=2048/tg=256, 4 samples, util 0.95,
  max-seqs 32). KV pool 53,434 tokens.
- **PPL on this model is unreliable in our stack (both arms, see
  Experiments/PPL).** Gate on logprob A/B + coherent text instead.

## Phase 0 — format & dispatch study (DONE)

### On-disk quant format

- Tensors per routed expert: `weight_packed` int32 `[N, K/8]` (N-first,
  8×uint4 packed) + `weight_scale` fp16 `[N, K/32]`. **No zero-point tensor
  anywhere in the index** (confirmed: no `*zero*` / `*zp*` keys for MoE
  experts).
- Symmetric dequant derived by `benchmarks/kernels/gemma4_wna16_probe.py`:
  with scale s and packed uint4 q, `(q - 8) * s` matches vLLM's own
  compressed-tensors emulation reference (`_unpack_and_dequant_int4_gptq`)
  to 9.8e-4 (fp16 noise) across the expert; max|w|/scale p99 = 8.0 confirms
  the codes saturate at the uint4 midpoint → **codes are uint4b8 (0..15),
  zero point = 8, dequant = `(q-8)*scale`** (exactly what the Triton kernel
  does in its `not has_zp` branch: `b_zp_num = 8`).
- Repack: raw N-first layout; the loader transposes to K-first for the
  kernel. 200-cell random round-trip (pack→repack→dequant) OK.

### Oracle gate — exactly three blockers

`vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`:
1. **Gate 1** required `may_have_zp` (AWQ-style stored zero points) — gemma4
   has none → rejected.
2. **Gate 2** rejected `isinstance(quant_config, (AutoGPTQConfig,
   QuantizationArgs))` — gemma4's compressed-tensors config surfaces as
   `QuantizationArgs` → rejected.
3. The gfx906 repack helper had no branch for the GPTQ-style K-first packed
   layout (only the AutoAWQ K-first and MoeWNA16 N-first shapes).

The repack already fabricated `0x88888888` (q=8) zero points when
`qzeros is None` — i.e. the C++ side already implements `(q-8)` dequant for
missing zp; only the Python-side gates and repack layout were missing.

### Kernel applicability

`csrc/rocm/moe_q_gemm_gfx906.cu` `moe_gemm_q4` handles gemma4's shapes:
gemm1 N=1408 = 5×256 + 128 (partial gridY N-block, covered by the existing
tail handling), K=2816/704 multiples of 64, group 32, em = topk = 8 →
M=1 decode path. No kernel changes needed.

## Code changes

- `oracle/int_wna16.py`
  - `_is_symmetric_no_zp()`: recognizes `QuantizationArgs` with
    `.symmetric` (compressed-tensors surface) as a no-zp W4A16 config.
  - Gate 1 relaxed: accept `may_have_zp` **or** symmetric-no-zp.
  - Gate 2 relaxed: GPTQ-style rejection now exempts symmetric-no-zp.
  - `_repack_w4a16_gptq_kfirst_layout()`: same exllama nibble shuffle as the
    existing AWQ K-first branch, collision-free layout detection
    (`w.shape[2] == N` distinguishes K-first from N-first), raises on
    asymmetric GPTQ (asymmetric no-zp is mathematically meaningless for us).
  - (Fixed a permute/reshape ordering bug found while writing the new
    branch: shuffle must happen on the packed axis before the K/N split.)
- `compressed_tensors_moe_wna16.py`
  - The fabricated-zp `replace_parameter` now fires on `GFX906_HIP` too
    (the fabricated `0x88888888` zp tensor must reach the op on gfx906, not
    just on the Triton path).
  - **Fixed a latent upstream bug**: `_setup_kernel` was not forwarding
    `backend=` to `make_wna16_moe_kernel`, so the gfx906 backend fell
    through to `assert experts_cls in allowed_experts` at load time. The
    other three `_setup_kernel` callers all pass it; this one was missed
    upstream. Without this, no gfx906 W4A16 MoE load could succeed.

## Experiments

All under `HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 LD_LIBRARY_PATH=`,
`dtype=float16`, `enforce_eager=True`, `max_num_seqs=4`, util 0.9.

### Format/repack probe — PASS

`benchmarks/kernels/gemma4_wna16_probe.py`: repack + dequant matches the
vLLM emulation reference at 9.8e-4 (fp16 noise); 200-cell pack→repack→
dequant round-trip OK. (Committed, durable.)

### Unit tests — 58/58

- `tests/kernels/moe/test_gfx906_moe_gemm.py`: added a
  `gptq_kfirst_sym` layout case (per-case group size; gemma4-shape cases
  N=1408/K=2816 and N=2816/K=704 with gs=32, no zp) + a repack unit test.
  42/42.
- `tests/quantization/test_moe_wna16.py`: 16/16 (unchanged, still green with
  the `_setup_kernel` fix).

### PPL gate — INVALID on this model (documented, not a kernel signal)

Chat-templated PPL (`benchmarks/kernels/gemma4_chat_probe.py`, k=20
top-k protocol) returns **~1.3e6 in BOTH arms** (auto 1,289,863 / triton
1,396,470; re-run with a BOS-robust tokenization: 1,297,094 both ways —
the engine's prompt tokenization length matches, so it is not a
tokenization off-by-one).

Root investigation (triton arm, 20-token prompt, per-position
`prompt_logprobs`): at many prompt positions the model outputs **confident
garbage** (top-1 logprob -0.02 for tokens like " own", " ability" against
the template text), while at other positions it is near-certain on the right
token (" you" lp -0.11, "?" lp -0.035), and the actual token is in the
top-20 at every position (so a healthy sum would cap PPL at ~e^3). The
generated (decode-side) text is coherent in both arms. Conclusion: the
prefill-time logprobs path for this hybrid sliding-window + `k_eq_v`
attention on our stack is anomalous — equally in both MoE arms, so it is
**not** a signal about the MoE kernel. (Separate investigation, out of
scope here.) PPL is retired as a gate for this model.

### Numerical A/B (arms: `--moe-backend auto` → GFX906_HIP vs `triton`)

- 6-token greedy on one templated prompt: **token ids identical in both
  arms** (`[236777, 236789, 236757, 3490, 1822, 236764]`); top-20 overlap
  16–18/20, max |ΔLP| 0.5–4.5 — and in that prompt's flat-garbage prefill
  regime (top-1 logprob ~-20.7, i.e. ~1e-9) the top-20 boundary is
  arbitrary, so overlap is the meaningful metric.
- 12 chat-templated prompts × 64 tokens, logprobs=10, per-step comparison:
  - token match: 4/12 prompts 100% identical, mean 0.64 — divergence is
    from near-tie argmax flips (first-diff positions spread 0–61, consistent
    with softcap-30 + 262k-vocab hypersensitivity, not a constant offset).
  - **|ΔLP| of the sampled token at matching steps (n=491): median 0.0017,
    p90 0.071, p99 0.307, max 0.59** — fp16 accumulation-order noise.
  - top-10 set overlap at matching steps: median 1.00.
  - 128-token probe: coherent, on-topic text in **both** arms.

Verdict: the gfx906 kernel path computes the same MoE function as Triton
WNA16 to fp16 precision. (A systematic dequant error would show O(1–10)
signed |ΔLP|, not p99 0.31.)

### Serving A/B (the gate)

`docs/gfx906/_bench_gfx906.py`, graph mode (BENCH_EAGER=0), util 0.95,
pp=2048/tg=256, max-seqs 32, 4 samples; new `BENCH_MOE_BACKEND` env knob
added to the harness for the arm switch. Logs confirm the auto arm used
`'GFX906_HIP' WNA16 MoE backend` / `Gfx906WNA16Experts`.

| arm | samples (t/s) | mean |
|---|---|---|
| triton (baseline) | 37.836 / 37.808 / 37.797 / 37.791 | **37.81** |
| auto → gfx906 | 67.802 / 67.795 / 67.778 / 67.771 | **67.79** |

**Speedup 1.793×.** 37.81 → 67.79 t/s; step time 26.4 ms → 14.75 ms
(~11.6 ms/step recovered vs the 10 ms estimate). Both bands tight
(±0.02 t/s). Gemma-4 now decodes faster than the Qwen3.5-35B flagship.

### Flagship regression (oracle gate change could touch Qwen35)

Qwen3.5-35B-A3B-AWQ, same recipe: 66.322 / 66.308 / 66.249 / 66.211
(mean 66.27) — inside the standing band ~65.9–67.0. **No regression.**

## Verdict

**VERDICT: SHIPPED.** No kernel changes were needed — the C++ side already
implemented `(q-8)` dequant for missing zero points; the entire blocker was
Python-side (two oracle gates + one missing repack layout branch + one
latent `_setup_kernel` bug that would have crashed any gfx906 W4A16 MoE
load). Gemma-4-26B-A4B-AWQ: **37.81 → 67.79 t/s (1.793×)**, numerics at
fp16-noise level vs the Triton reference, flagship unaffected.

Follow-ups (not done here):
- The prefill-logprob anomaly on gemma4's hybrid attention (affects
  prompt_logprobs quality gates; both MoE arms equally).
- Triton `E=128,N=704` MoE config file for gfx906 (autotune warning) —
  now only relevant to non-gfx906-eligible models.

## Post-review no-zp gate hardening (2026-08-20)

The post-`180f030ee3` review flagged two fails-open paths in the symmetric
no-zp exemption. Both fixed on
`gfx906/no-zp-gate-hardening`:

- **`253942905c`** — `_is_symmetric_no_zp()` accepted *any* symmetric
  `QuantizationArgs`; a W8 / dynamic-scaled / odd-group-size /
  non-group-strategy checkpoint would pass the oracle and either crash
  late in the repack or mis-dequant silently. New
  `_gfx906_no_zp_reason()` (GFX906_HIP branch of
  `_backend_incompatibility_reason`) rejects symmetric no-zp configs
  that are not 4-bit, static (non-dynamic) scaled, group-strategy,
  and group size 32 or 128; they fall through to the Triton backend.
  (The kernel's per-32-K-slice group tracking would in fact accept any
  group size that is a multiple of 32; 32/128 are the checkpoint-
  validated values — widen the gate with a per-shape micro-bench, not
  by assumption.)
- **`5e3cf6d780`** — the symmetric exemption skipped the act-order
  check the GPTQ-style rejection carries. Per the compressed-tensors
  `ActivationOrdering` contract, `group` (and its `dynamic` alias)
  stores weights in original column order and requires a runtime g_idx
  reordering — which the kernel and the gfx906 repack do not implement
  (the repack drops g_idx), i.e. a silent mis-dequant. The helper now
  rejects `actorder in (group, dynamic)`; `weight`/`static` are
  format-identical to no activation ordering and keep passing, matching
  how the Marlin/Triton paths treat them. (The loader's pre-existing
  `assert actorder != "group"` in the non-Marlin branch would have
  crashed such a load anyway; the oracle gate now fails closed to
  Triton with a specific reason instead.)

Evidence: oracle-gate unit tests in `tests/quantization/
test_moe_wna16.py` — 25 passed (new: W8, dynamic, group-64, channel,
actorder=group, actorder=dynamic all rejected with specific reasons;
group-32/128 and actorder=weight accepted). The real gemma-4
checkpoint weight args still pass the gate at the real model shapes
(E=128, hidden 2816, moe_inter 704, group 32). Model path untouched
(oracle-level only): MoE kernel suites 48 passed (`test_gfx906_moe_
gemm.py` + `test_fused_topk.py`), WNA16 conversion suites 121 passed.
No serving A/B: the gate only changes backend selection for configs
that were previously unservable or silently wrong on gfx906; the
accepted gemma-4 path is unchanged.

## Post-review MED items (2026-08-20)

### MED-1 — fabricated zp tensor eliminated (`20df23b80f`)

The symmetric repack no longer materializes a `[E, G, N/8]`
`0x88888888` zp tensor (~16 MiB per gemma-4 MoE layer, ~0.45 GiB
across the ~29 MoE layers), nor streams it per group: the repack
returns `None`, the op wrapper passes an empty tensor, and both
kernels (base and the M=1 v2 re-tile) inline the constant zero point 8
(uint4 midpoint) when `b_qzeros` is null. `zero_offset` semantics are
unchanged; symmetric gfx906 layers now carry no `zero_point`
parameters (CPU still requires the tensor).

No-numeric-change evidence: the MED-2 re-run below is post-MED-1 and
agrees with the pre-MED-1 Numerical A/B record within run-to-run
variance (median |ΔLP| 0.0012 vs 0.0017, p90 0.071 both, p99 0.286 vs
0.307, first-diff spread 0–61 in both, same 4/12 fully-matching
prompts), and the kernel e2e suite exercises the nullptr path against
a torch reference with z = 8 (43 passed).

### MED-2 — first-diff vs prefill regime (numerics record)

Re-ran the 12-prompt × 64-token A/B with per-step logprobs and
per-position prompt logprobs
(`benchmarks/kernels/gfx906/gemma4_divergence_probe.py`; same
protocol and prompt set as the Numerical A/B):

- 8/12 divergent; 4/12 (indices 3, 7, 8, 9) 100% token-identical over
  all 64 tokens; mean match 0.633 — reproduces the original record
  (0.64, 4/12).
- First-diff positions `[14, 10, 60, 1, 38, 0, 61, 26]`: spread
  through the decode (median 20; only 2/8 in the first two steps) —
  **not** concentrated at the prefill→decode boundary.
- Prefill flatness per prompt (mean top-1 logprob over the model's own
  prompt tokens): −0.26…−0.43 for the divergent prompts vs
  −0.29…−0.56 for the non-divergent — overlapping ranges; the
  garbage-prefill regime (PPL section) does **not** correlate with MoE
  arm divergence.
- |ΔLP| of the sampled token at first-diff steps (median 0.031, max
  0.181) is inside the matching-step |ΔLP| band (median 0.0012, p99
  0.286, n=486); the tie gap at the diff (median 0.219, min 0.172) is
  a narrow but real top-1 lead.

Record: the divergence is a near-tie argmax flip in pure decode
(consistent with the softcap-30 + 262k-vocab hypersensitivity noted
above); the anomalous prefill logprob regime is equally present in
both arms and is not a driver of the divergence. Verdict unchanged: no
systematic dequant error.

Probe note: an early iteration of the probe ranked tokens by *most
negative* logprob (the 10th of the top-10) and reported 12/12
"divergences at position 0" — an artifact of comparing the arbitrary
top-10 tails across arms (the top-10 sets themselves match with
median overlap 1.00). The record above uses the true top-1, which is
the greedy sample.

### Serving (post-MED-1)

auto arm: 68.245 / 68.239 / 68.240 / 68.221 → **68.24 t/s mean**
(graph mode, util 0.95, pp 2048 / tg 256, max-seqs 32, 4 samples) vs
the original 67.79 band — holds, with the ~16 MiB/layer zp stream
gone. Triton arm unchanged by MED-1 (37.81 from the original A/B).
