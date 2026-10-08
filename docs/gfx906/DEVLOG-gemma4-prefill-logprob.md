# Dev log — Gemma-4 prefill-logprob anomaly (confident garbage prompt logprobs)

Branch: `gfx906/gemma4-prefill-logprob` (from `gfx906/main` @ `180f030ee3`, 2026-08-20)

## Symptom

Discovered during the no-zp MoE A/B (DEVLOG-gemma4-moe.md): chat-templated
PPL on Gemma-4-26B-A4B-AWQ (fp16, TRITON_ATTN, eager) is implausible —
~1.3e6 in BOTH MoE arms. Per-position investigation (20-token templated
prompt, `prompt_logprobs=20`, triton arm):

- Many prompt positions get **confident garbage**: top-1 logprob ≈ -0.02
  for wrong tokens (" own", " ability") against the template text
  `<bos><|turn>user\nHello there,...`; the actual token scores -16..-30
  (near the softcap-30 floor) yet still lands in the top-20.
- Other positions are near-certain on the right token (" you" lp -0.11,
  "?" lp -0.035).
- The actual token is in the top-20 at **every** position (0 misses) — so
  a healthy sum would cap PPL at ~e^3; the measured 1.3e6 means the
  average actual-token logprob is ~-14.
- **Decode-side generation is coherent** (both MoE arms, 128-token greedy
  on 12 templated prompts).
- 6-token A/B: at the FIRST generated position on the 20-token prompt the
  whole top-5 sits at lp ≈ -20.7 (flat garbage, top token prob ~1e-9) in
  both arms — yet on the 12 longer (~40-60 token) probe prompts the
  greedy text is coherent from the visible start.

Equally present in both MoE arms ⇒ NOT the MoE kernel. Candidate
subsystem: the prefill path of the hybrid attention (25 sliding hd-256
window-1024 + 5 full hd-512 `attention_k_eq_v` layers, proportional RoPE,
partial_rotary 0.25) on TRITON_ATTN, or the engine's prompt_logprobs
collection.

## Questions

1. Is the anomaly in prefill hidden states (attention), or only in the
   logprob *collection* path? (Compare prompt_logprobs[i] at i=last vs the
   first decode step's logprobs — same forward, should agree exactly.)
2. Length dependence: 20-token prompt → flat first-token distribution and
   garbage prefix logprobs; ~40-60 token prompts → coherent decode. Map
   the boundary and per-position pattern.
3. Layer-type dependence: which layer class is broken — sliding (25) or
   full/k_eq_v (5)? Bisection by config edit (all-sliding / all-full).
4. Is it TRITON_ATTN-specific, or shared by any attention backend that
   serves gemma4 on this stack?

## Hypotheses (ranked, to be falsified)

- H1: TRITON_ATTN prefill kernel bug for one of the gemma4 layer classes
  (k_eq_v V-pointer, sliding-window current-block handling, or the
  hd-256/hd-512 mixed path).
- H2: vLLM's gemma4 hybrid config mapping is wrong (window/layer-type
  attribution, num_global_key_value_heads, proportional RoPE head_dim).
- H3: engine prompt_logprobs path mishandles this model class (e.g.
  per-position logits taken from a different tensor than decode uses).
- H4: fp16-specific (some prefill-only buffer overflow/overflow-adjacent
  numerics) — low prior: decode uses the same weights fine.

## Protocol

- Model: local snapshot
  `/local/cache/huggingface/hub/models--cyankiwi--gemma-4-26B-A4B-it-AWQ-4bit/snapshots/0ef577a5710035bd2d3a3f27e4f5cb2e86a9a9ba`,
  fp16, eager, `max_num_seqs=4`, util 0.9, `max_model_len=4096`,
  `moe_backend=triton` (MoE path proven fine — keep the variable out).
- All prompts chat-templated (thinking-mode trap).
- "Confident-correct" position: actual token is top-1 and lp < -0.5.
  "Confident-garbage" position: top-1 lp < -0.5 but actual token lp < -5.

## Experiments

Full narrative, artifacts and safety rules (HF-forward GPU hang + power
button, 2026-08-20 ~10:04): see `/local/tmp_gemma4/gemma4_findings.md`.
Summary of the closure sequence (2026-08-20, agent handoff to Pi):

1. **In-engine verification sweep (pre-handoff)**: all 30 attention
   layers' outputs == exact full-causal softmax over their captured
   q/k/v (fp16 precision); KV write path exact; in-model RoPE wiring
   exact (both layer types); prompt-logprob collection path internally
   consistent; row retrieval clean (each prompt-logit row == softcap30(
   lm_head @ final_norm row i), permutation search = identity).
2. **Cross-model control (probe6)**: Qwen3.5-35B on the same harness —
   SANE (template positions rank 1-2, lp >= -3; only first-content-word
   ranks ~16k, normal). The anomaly is gemma4-specific, not the harness
   or shared numerics.
3. **Prompt controls (probe7)**: gemma4 instruction and prose prompts
   break EQUALLY at word starts (`What` rank 149669, `Once` rank 254631,
   template head rank 129296) — kills "OOD fruit prompt" and
   "model characteristic on weird prompts". Greedy from bare `<bos>`:
   `' own ox.0.5.5.5...'` — degenerate. This confirms the pos-1 anomaly
   OUTSIDE the logprob path (pure decode from bos).
4. **Chat template check (CPU)**: the checkpoint's community template
   renders byte-identically to the official gemma-4 template for a
   simple user message — not a template issue.
5. **Absolute fp32 CPU reference (ref_model.py)**: full 30-layer
   reference computed from the safetensors (W4A16 dequant `(q-8)*s`
   group-32, dense mlp bf16, HF Gemma4 semantics: dual-FFN + router on
   the residual + layer_scalar; attention scaling=1.0, proportional
   RoPE exponents /512 zero-padded). Compared against per-layer vLLM
   captures (probe8, caps8.pt): **all 30 layers match at rel 6e-4 to
   2e-2** — exactly the fp16-vs-fp32 band with normal MoE-routing chaos
   growth, no divergence. End-to-end: ref top-1 from `<bos>` = `' own'`
   lp **-2.697** vs vLLM's **-2.700**. **vLLM's gemma4 path is
   numerically correct.**
6. **fp16 bound (ref_layer0_fp16.py)**: simulated-fp16 layer 0 differs
   from fp32 by rel 5e-4; the initial 2.9% layer-0 mismatch that
   motivated the deep dive was **a bug in the reference itself**
   (attention loop flattened (pos, kv-head) into one key axis — the
   softmax attended over all 16 kv-head copies). After the fix the
   mismatch collapsed into the fp16 band. Recorded here as a trap for
   future reference-implementation reviews.

## Verdict

**VERDICT: CLOSED — not a vLLM bug. The anomaly is a property of the model
weights** (the community cyankiwi W4A16 AWQ quantization of
gemma-4-26B-A4B-it): from short/low-context prefixes the checkpoint
genuinely produces flat degenerate distributions (top-1 `' own'`/`'-'`
at lp ~-2.7, entropy ~2-3 nats, template tokens at rank 50k-130k).
Two fully independent implementations — vLLM fp16 on GPU and a
from-safetensors fp32 CPU reference — agree to fp16 precision and
produce the same degenerate distributions. Coherent decode survives
because distributions sharpen once context accumulates.

Consequences:
- **PPL / prompt_logprobs stay retired as gates for this checkpoint.**
   The DEVLOG-gemma4-moe numerics verdict (|dLP| p99 0.31 vs Triton)
   stands unchanged — both arms compute the same (correct) function.
- The 37.81 -> 67.79 t/s no-zp serving result is unaffected.
- If a future official (bf16 or Google-released) gemma-4 checkpoint
  arrives, re-run the reference comparison to separate "community
  quant damage" from "base model behavior" (the fp32 reference of
  THIS checkpoint cannot distinguish those).

Upstream note (separate): `Gemma4Attention.forward`'s dead
`num_kv_shared_layers > 0` branch skips k_norm+RoPE on K, disagreeing
with HF — latent, unreachable for this checkpoint (0 shared layers).
