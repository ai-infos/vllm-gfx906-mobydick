# Handover — Qwen3.8-Flash-Next (`qwen4_exp` QSA) on gfx906

**Ask:** run the real checkpoint and tell us whether it works and how fast. Nothing
in this bundle has been tested end-to-end — the model needs ~60 GB and we have
2×32 GB, so **your report is the gate**. Full detail: `README.md` next to this file.

**Provenance:** branch `gfx906/qsa-fn` @ `885ff49234`; release `gfx906/v0.29.0-final`
@ `524ac6f2d6`. Bundle: this directory (or `/local/tmp/qsa-tester-build.tgz`).

## 1. Apply (pick one)

- **On `gfx906/v0.29.0-final`** (our release): apply `patches/0001` + `patches/0003`
  only — 0002 and the harness are already there.
- **On the QSA branch:** nothing to apply, `git checkout gfx906/qsa-fn`.
- **On stock upstream:** `git apply` 0002, 0003, 0004, then 0001 with
  `--exclude=vllm/models/qwen4_exp/common/qsa_cache.py`; that one file needs five
  mechanical edits (listed in `README.md`).

Then rebuild if your tree needs it, and check `python -c "import vllm"` works.

## 2. Smoke test first (~3 min, one GPU, no big download)

```bash
python docs/gfx906/_qsa_tiny_model.py model   # or copy bundle tiny-tokenizer/* into $MODEL_DIR
docs/gfx906/_serve_qsa_tiny_gfx906.sh start pc && docs/gfx906/_serve_qsa_tiny_gfx906.sh wait pc
.venv/bin/python /local/tmp/v2mamba/v2mamba_repro.py 8341 qsa-tiny 1343 2015 4030
docs/gfx906/_serve_qsa_tiny_gfx906.sh stop pc
```

Expect 3× `OK` (random weights ⇒ garbage text; the point is that every path runs).
Before the 0002 fix the 2nd or 3rd request killed the engine.

## 3. Run the real model

```bash
docs/gfx906/_serve_qsa_flash_gfx906.sh start /path/to/Qwen3.8-Flash-Next t1
docs/gfx906/_serve_qsa_flash_gfx906.sh wait t1     # 60 GB load: tens of minutes
docs/gfx906/_serve_qsa_flash_gfx906.sh report      # the checklist below
```

Two flags matter: `VLLM_USE_V2_MODEL_RUNNER=1` (on V1 the model cannot run) and
`--dtype float16`. If init OOMs: `OFFLOAD_GB=60`, `TP=2`, lower `GPUTIL`.

**Do not chase:** the bf16 arm dies in `rocm_unquantized_gemm_impl` (pre-existing),
and int8 KV/QK is deliberately excluded.

## 4. Send back

1. Launch line, card(s) and VRAM.
2. `Resolved architecture`, `GPU KV cache size`, `Available KV cache memory` lines.
3. One short greedy completion + one ~2 000-token prompt: TTFT and decode t/s
   (state pp/tg, batch size, prefix caching on/off).
4. One 100 k+ request: does it finish? coherent? needle retrievable at
   start/middle/end?
5. If it breaks: what came first — init OOM / dtype or `NotImplementedError` /
   kernel fault (paste kernel name + grid) / garbage-but-running?
6. Optional: with `SPEC='{"method":"mtp","num_speculative_tokens":3}'`, acceptance
   rate and decode t/s vs without, on identical prompts.

**Limits to keep in mind:** quality is unverified here (no loadable checkpoint); the
tiny rig's dimensions are 10–24× off the real model, so its *timings* do not
transfer — only its pass/fail does.
