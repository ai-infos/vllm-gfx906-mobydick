# V2 runner · mamba `align` state migration — topic log

> Prefix caching on hybrid mamba models: `mamba_cache_mode='align'`, the
> pre-copy/post-copy kernels and the per-request running-block column.
> Branch `gfx906/qsa-fn` (off `gfx906/v0.29.0`) · 2× MI50 (gfx906), ROCm 7.14,
> Triton 3.8.0+gitc01b6774 · roadmap items [`V2-MAMBA-*`](ROADMAP.md),
> `DFL2-2` (V2 adoption). Newest entry first.
>
> Related: [`DEVLOG-qwen38-flash-qsa.md`](DEVLOG-qwen38-flash-qsa.md) (the
> Qwen4Exp train that found this), `degradation.md` (wedges are *not* this).

## 2026-09-17 — V2-MAMBA-1: `add_request` sized the align column with the wrong block size

**VERDICT:** `SHIPPED` · **GATE:** the tiny Qwen4Exp rig with prefix caching ON
(→ align mode) runs the request sequence that used to fault, and produces
**bit-identical** greedy tokens *and* top-5 logprobs to the same rig with prefix
caching OFF — with and without MTP k=3.

```bash
# repro (before the fix): 3 requests, each a prefix hit on the previous one
docs/gfx906/_serve_qsa_tiny_gfx906.sh model && docs/gfx906/_serve_qsa_tiny_gfx906.sh start pc
.venv/bin/python /local/tmp/v2mamba/v2mamba_repro.py 8341 qsa-tiny 1343 2015 4030
```

### HYPOTHESIS

If the align pre-copy reads a block column past the end of the sequence, the
load of that (stale) block-table entry yields a garbage physical block id and
the copy follows it to a wild address — i.e. the fault is a *wrong running
column*, not a copy-arithmetic bug in the kernel.

### What was done

- Dumped every launch of `precopy_mamba_align_fused_kernel` (num_reqs, src_col,
  dst_col, token_bias, the flattened per-state metadata, the block-table
  pointers/strides) and the `_mamba_state_idx_gpu` / `_mamba_src_col_gpu`
  lifecycle around `preprocess_state` / `postprocess_state` / `add_request`
  (temporary instrumentation, removed before commit).
- Fixed `MambaHybridModelState.add_request`, one line + comment: seed the
  running state block column from `cache_config.mamba_block_size` instead of
  `cache_config.block_size` (plus `assert ... is not None`).
- New regression test
  `tests/v1/worker/test_mamba_hybrid_model_state.py::test_add_request_seeds_running_column_with_mamba_block_size`
  (num_computed 0 / 1152 / 4032 → -1 / 5 / 20).
- ROCm-enabled the two CUDA-gated mamba kernel tests (see Evidence).

### Evidence FOR

**The fault** (before the fix, `tests/models/qwen4_exp` tiny config, one MI50,
prefix caching ON, V2 runner):

```
:0:rocdevice.cpp :3678: Memory Fault Error [host: mi50-01, GPU index: 0,
  faulting addr: 0x94237cc28000, kernel: precopy_mamba_align_fused_kernel]
  -> c10::AcceleratorError / EngineDeadError -> HTTP 500
```

Flaky by request, as expected for a stale-entry read: three runs faulted on the
4031-token request, on the 2016-token one, and once not at all (the garbage id
happened to be benign). A fresh request never faults; **the trigger is
`num_computed_tokens > 0` at `add_request`** — a prefix-cache hit (also a
resumed/preempted request).

**The mechanism** (launcher dump, the faulting step):

```
[v2m-dbg] add_request slot=0 num_computed=1152 block_size=4 mamba_block_size=192
          spec=192 -> 287
[v2m-dbg] 64 preprocess.in  n=1 state_idx=[287, 0, 0, 0] src_col=[10, 0, 0, 0]
```

`(1152 - 1) // 4 = 287` (buggy) vs `(1152 - 1) // 192 = 5` (correct): the seeded
column landed ~57× past the true one. `precopy`'s temporal path is

```
actual_src_block_id = tl.load(block_table_base + src_col + token_bias)
src_addr = state_base_addr + actual_src_block_id * state_block_stride
```

so column 287 (the sequence has ~6 columns) is loaded from a stale region of the
block table, and the garbage id is then scaled by the page stride.

**The three block sizes** in one run (measured, all in the same config object):

| value | source | used by |
|---|---|---|
| 192 | `cache_config.mamba_block_size`, `MambaSpec.block_size` | the align kernels, block tables, `max_num_blocks_per_req` |
| 4 | `cache_config.block_size`, **after** engine startup narrows it to `min(g.kv_cache_spec.block_size for g in kv_cache_groups)` | scheduler granularity |
| 192 | attention block size, *raised* earlier by `_align_hybrid_block_size` ("Setting attention block size to 192 tokens … Padding mamba page size by 37.14%") | attention KV |

The 4 comes from a **`CircularBufferSpec` group with `prefix_cacheable=False`**
(Qwen4Exp's QSA indexer cache) — logged per group at engine startup.

**Gates (after the fix), tiny rig, greedy, one MI50:**

| arm | requests | greedy vs pcOFF | top-5 logprobs vs pcOFF |
|---|---|---|---|
| pcOFF (align off) | 12/12 OK | baseline | baseline |
| pcON (align on) | 12/12 OK | 12/12 identical | worst \|Δ\| = **0.000000** |
| pcON + MTP k=3 | 12/12 OK | 12/12 identical | worst \|Δ\| = **0.000000** |

(`v2mamba_equiv.py` = 6 prompts × 2 reps; `lp.py` = 6 prompt/rep pairs of
top-5 logprobs. Random weights make the greedy text degenerate, so the logprob
comparison is the sharp one.) Not measured: any *real-model* align-mode run —
no loadable Qwen4Exp checkpoint here (QSA-FN-3/8).

**Unit + kernel tests** (this box, `HIP_VISIBLE_DEVICES=0`):

- new seed test: **fails on the pre-fix code** (`assert 287 == 5`,
  `assert 1007 == 20`) and passes after; 4 passed / 3 skipped (the two other
  CUDA-gated tests in that file).
- `tests/kernels/mamba/test_memcpy_u64_tiled.py` +
  `test_precopy_mamba_align.py`, ROCm-un-gated in this branch:
  **195 passed** (120 + 75) in 56 s. They already passed on gfx906 when run
  un-gated from a scratch copy — the gate was just `is_cuda()`.

### Evidence AGAINST / limits

- **The ROCr report's grid is not trustworthy here.** It printed
  `grid=[256, 7, 16], workgroup=[256, 1, 1]` while the launcher dump for the
  faulting step shows grid `(1, 7, 16)` and *no* launch in the run used
  num_reqs=256 (dims 1/2 match the kernel's `(num_states, _TEMPORAL_TILES)` =
  `(7, 16)` exactly). Same class as the un-aligned trace timestamps in
  `AGENTS.md` — use the launcher-side dump, not the fault line, to attribute.
- Upstream `main` (fetched 2026-09-17) still has the seed bug verbatim, so
  nothing here is a re-derivation of an upstream fix. It *did* change the
  narrowing the same day (min over `prefix_cacheable` groups only, with a
  comment that a small-block group would "desync it from mamba") — for our
  model that filters out the bs=4 `CircularBufferSpec`, i.e. upstream currently
  **masks** this rather than fixing the seed. A prefix-cacheable group finer
  than the mamba block would still trip it (no concrete repro on this box).
- **The upstream E2E guard was not run:**
  `tests/v1/e2e/general/test_mamba_prefix_cache.py` loads Qwen3-Next-80B-FP8
  config + tokenizer and the `heheda/a_long_article` dataset, neither cached
  here, and Qwen3-Next-FP8 is not a validated model on this box. It would not
  exercise the trigger anyway: it runs every group at one uniform
  `BLOCK_SIZE = 560`, so the two divisors coincide. The new unit test is the
  guard for this class; the E2E arm remains unvalidated on gfx906.
- No perf claim: the fix is a divisor, and the arms above differ in prefix
  caching, so no t/s comparison is reported (and none is the gate).

### Interactions / superseded-by

- Supersedes the **`--no-enable-prefix-caching` workaround** that QSA-FN-2's
  tester recipe carried; the recipe now leaves prefix caching on (upstream
  default). Kept as a documented fallback for builds predating the fix.
- Why only Qwen4Exp hit it locally: it needs a hybrid align-mode model with a
  KV group finer than the mamba block size. Our other local recipes have
  uniform block sizes, where the two divisors coincide — the fix is a no-op
  there (no behaviour change to re-validate).
- `V2-MAMBA-1` was listed as a prerequisite for the V2-runner adoption
  (`DFL2-2`); only the *Qwen4Exp-with-prefix-caching* half of it was live, and
  that half is now fixed.

### Refrigerated residue

- `_postprocess_recoverssm_align_kernel` (`recoverssm.py`, Kimi-K3 RecoverSSM,
  `use_kda_recoverssm`) writes `(num_computed + num_sampled) // MAMBA_BLOCK_SIZE`
  — **no `-1`**, unlike every other align site (`aligned_num // block_size - 1`
  in `postprocess_mamba_fused_kernel`). Same family of suspicion (a column that
  can point one block past the writer's intent). Not touched: no RecoverSSM
  model is loadable here, so it cannot be gated — flagged for whoever can run
  one.
- `precopy_mamba_align_fused_kernel` has no bounds check on `src_col` against
  the block-table width. A cheap belt-and-braces guard (clamp or early-exit)
  would turn any future mis-seed into a no-op instead of a wild read; not added
  because the seed is now provably in range and a silent clamp could hide the
  next bug of this class.
