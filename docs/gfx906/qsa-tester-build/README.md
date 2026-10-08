# Qwen3.8-Flash-Next (`qwen4_exp` QSA) on gfx906 — tester bundle (QSA-FN-8)

For whoever can run the **real** `Qwen/Qwen3.8-Flash-Next` checkpoint: this is
the gfx906 work needed to serve it, plus a tiny-config rig that validates the
same code paths in ~2 minutes without the ~60 GB of weights.

**What we need back:** does the real model load and serve, and how fast — the
checklist in [What to report back](#what-to-report-back). Nothing in this
bundle has been run against the real checkpoint (it does not fit on our 2×32 GB
MI50 box), so your report *is* the end-to-end gate.

**Short on time? Read [`HANDOVER.md`](HANDOVER.md)** — a one-page
version of this file with just the steps, the gates and what to send back.

## What is included

| file | what / why |
|---|---|
| `patches/0001-qsa-fp16-enablement.patch` | **the fix for the reported error.** QSA's guards only accepted bf16; gfx906 has no bf16 instruction, so models run fp16 and the guards rejected it. Also a 4.4× kernel win (bf16 `tl.dot` is emulated per-scalar, fp16 lowers to `v_dot2_f32_f16`) |
| `patches/0002-v2-mamba-align-seed.patch` | crash fix needed for **prefix caching** (which the V2 runner turns on by default for hybrid models): the mamba align pre-copy was seeded with the wrong block size and faulted on any prefix-cache hit |
| `patches/0003-qsa-tiled-indexer.patch` | the indexer scoring kernel's tiled route: **1.33–1.35×** on the indexer in fp16, gated off for bf16 (where it is 2.4× slower on gfx906) |
| `patches/0004-qsa-harness.patch` | the tiny-config rig (`_qsa_tiny_model.py`, `_serve_qsa_tiny_gfx906.sh`) and the real-model recipe (`_serve_qsa_flash_gfx906.sh`) |
| `tiny-tokenizer/` | Qwen3.8-Flash-Next tokenizer + the tiny config, so the smoke rig works offline |
| `patches/../BUILD-INFO.txt` | branch, commit, base and generation date of this bundle |

Not included on purpose: the int8 KV / int8-QK experiments. int8 KV costs
2.5–2.7× in attention prefill on gfx906 (decode-neutral) for a capacity win we
do not need, and the int8-QK path faults at the profile every real prefill uses.

## Apply it

### A. On our branch (validated: 4/4 clean, byte-identical to the branch)

```bash
git fetch <remote> gfx906/qsa-fn        # or start from gfx906/v0.29.0
git checkout gfx906/qsa-fn              # nothing to apply — this IS the result
```
Equivalently, from `gfx906/v0.29.0`: `git apply patches/000{1,2,3,4}-*.patch` in
order — that was validated here and reproduces the branch's 63 shipped files
byte-for-byte.

**On `gfx906/v0.29.0-final` (our 2026-09-18 release), patch 0002 is already in
the tree** (the release backported it) and so is `0004`'s harness; apply only
`0001` + `0003`, or apply all four to the pre-release base `f79ebf2d44` as the
`BUILD-INFO.txt` records.

### B. On stock upstream (verified against `releases/v0.30.0`)

```bash
git apply patches/0002-v2-mamba-align-seed.patch                 # clean
git apply patches/0003-qsa-tiled-indexer.patch                   # clean
git apply patches/0004-qsa-harness.patch                         # clean
git apply --exclude=vllm/models/qwen4_exp/common/qsa_cache.py \
          patches/0001-qsa-fp16-enablement.patch                 # everything else clean
```

`vllm/models/qwen4_exp/common/qsa_cache.py` moved upstream (103/25 lines) and is
the **one file that needs a manual port** — `git apply -3` does not resolve it
either. It is five mechanical edits; compare with
`git show 0001:…` or just add:

1. next to the other module constants:
   `QSA_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16)` and
   `QSA_KV_CACHE_DTYPES = ("auto", "float16", "bfloat16")`;
2. `QSAStateBackend.supported_dtypes / supported_kv_cache_dtypes` →
   `list(...)` of those two constants (upstream: `[torch.bfloat16]` /
   `["auto", "bfloat16"]`);
3. `_QSAStateCache.bind_kv_cache`: `kv_cache.dtype != torch.bfloat16` →
   `!= self.dtype`;
4. `QSAKeyStateCache._BF16_PER_INT64` → `_ELEMS_PER_INT64` (same value, 4 —
   8 bytes per int64 over 2-byte elements; it is only a name);
5. the three bf16 wordings in docstrings ("Raw BF16 key" → "Raw 2-byte-float
   key" etc.).

## Smoke test without the real model (2 minutes, one GPU)

```bash
python docs/gfx906/_qsa_tiny_model.py model        # writes the tiny config;
                                                   # offline: copy tiny-tokenizer/* into $MODEL_DIR
docs/gfx906/_serve_qsa_tiny_gfx906.sh start pc     # prefix caching ON (the fix above)
docs/gfx906/_serve_qsa_tiny_gfx906.sh wait  pc
.venv/bin/python /local/tmp/v2mamba/v2mamba_repro.py 8341 qsa-tiny 1343 2015 4030
docs/gfx906/_serve_qsa_tiny_gfx906.sh stop  pc
```

The three prompts share a prefix (each is a prefix-cache hit on the previous
one) — before `0002` the second or third one killed the engine with
`Memory Fault Error … precopy_mamba_align_fused_kernel`. Random weights, so the
text is garbage by design; the point is that every path executes. Expected on
one MI50 (fp16, V2, no spec decode): prefill 1321/2641/3961 tokens ≈ 22/42/48 ms,
decode ≈ 563/1023/1994 t/s at B=1/2/4 after warming each shape (the first
request at a new shape pays a one-time Triton autotune, ~20 s — warm up before
timing).

Unit tests for the same code, if you want them: `tests/models/qwen4_exp/`
(`test_qsa_amd.py` 22 passed, `test_qsa_reference.py` 19, `test_config.py` 7,
`test_ple.py` 10 — run one file at a time; the directory cannot be collected as
a whole because `amd/` and `nvidia/` register the same custom op twice) and,
for the mamba fix, `tests/v1/worker/test_mamba_hybrid_model_state.py`
(+ `tests/kernels/mamba/`, ROCm-un-gated here).

## Run the real model

```bash
docs/gfx906/_serve_qsa_flash_gfx906.sh start /path/to/Qwen3.8-Flash-Next [tag]
docs/gfx906/_serve_qsa_flash_gfx906.sh wait  [tag]
docs/gfx906/_serve_qsa_flash_gfx906.sh report      # prints the checklist below
```

The script documents every flag and why it differs from the MI210 production
recipe; the two that matter:

- **`VLLM_USE_V2_MODEL_RUNNER=1`** — required: on the V1 runner the PLE layer
  raises `PLE inputs were not prepared` (its inputs come from the V2 model
  states). This is the one recipe here that must *not* pin V1.
- **`--dtype float16`, no `--mamba-cache-dtype bfloat16`** — the gfx906 fp16
  rule; also `--max-model-len 262144` native RoPE (no YaRN/rope scaling).

It needs ~60 GB of W4A16 weights plus the PLE ngram table (the CDNA recipe
offloads 60 GB of it with `--cpu-offload-gb 60 --cpu-offload-params
ngram_embedding`, exposed here as `OFFLOAD_GB=60`), so raise `TP` to 2 and lower
`GPUTIL` if init OOMs. Known-broken, do not chase: the **bf16 arm** dies in
`rocm_unquantized_gemm_impl` (`Matrices A and B must have the same dtype`) inside
the hyperconnection chain — pre-existing and unrelated to these patches, fp16 is
the arm to use.

## What to report back

1. your launch line, card model and per-card VRAM, and `rocm-smi` output;
2. did it load: the `Resolved architecture`, `GPU KV cache size` and
   `Available KV cache memory` lines from the server log;
3. one short greedy completion and one ~2 000-token prompt: TTFT and decode
   tok/s (state the config: pp/tg, batch size, prefix caching on/off);
4. one long-context request (100 k+ if the KV pool allows): does it complete, is
   the output coherent, and is a needle retrievable at start/middle/end;
5. if it broke, which came first: init OOM / a dtype or `NotImplementedError` /
   a kernel fault (paste the kernel name and grid) / garbage-but-running;
6. if you enable speculation (`SPEC='{"method":"mtp","num_speculative_tokens":3}'`),
   the acceptance rate and decode t/s with and without, on identical prompts.

## Evidence and limits (why you can trust the parts we claim)

- Kernel-level measurements on MI50, all in `docs/gfx906/DEVLOG-qwen38-flash-qsa.md`
  (entries 1, 4) and `DEVLOG-v2-mamba-align.md`: sparse attention 26.5 ms fp16 vs
  116.5 ms bf16; indexer 5426 → 4061 µs with the tiled route (top-2048 agreement
  1.00000, logits NRMSE 1.3e-7); the mamba fix validated by bit-identical
  logprobs against the prefix-caching-off path.
- Non-regression, same boot: FA suite 104 passed, PPL 10.5472 (our recorded
  value for Qwen3.8-27B-AWQ-INT4), MoE-35B bench 58.40 t/s (house band
  57.97–58.36).
- **Limits:** quality on the real checkpoint is unverified (no box here can load
  it); the fp16 path is exercised only through the tiny config, whose dimensions
  are 10–24× off the real model, so *shares* of time (e.g. the indexer's ~23 % of
  prefill) do not transfer; decode numbers here are per-kernel, not serving.

Full context: `docs/gfx906/RECON-qwen38-flash-qsa.md` (pre-work recon incl. the
ISA facts), `DEVLOG-qwen38-flash-qsa.md` (per-item records),
`DEVLOG-v2-mamba-align.md` (the crash fix), `ROADMAP.md` (`QSA-FN-*`), and the
branch `gfx906/qsa-fn` (5 code commits + docs).
