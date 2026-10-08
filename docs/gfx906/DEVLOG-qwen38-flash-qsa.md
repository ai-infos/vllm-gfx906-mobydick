# QSA-FN — Qwen3.8-Flash-Next (Qwen4Exp QSA) on gfx906

> Branch `gfx906/qsa-fn` off `gfx906/v0.29.0` · model `Qwen/Qwen3.8-Flash-Next`
> (`qwen4_exp`) · 2026-09-17 · roadmap [`QSA-FN-*`](ROADMAP.md) · pre-work
> evidence: [`RECON-qwen38-flash-qsa.md`](RECON-qwen38-flash-qsa.md) (the
> bf16-only site inventory, the ISA facts and the kernel timings live there; not
> restated per entry). Newest entry first.

## 2026-10-05 — QSA-FN-12 (cudagraph mode) and QSA-FN-15 arm A (PLE offload), decided on a real checkpoint

**VERDICT:** `DEAD-END` for full-graph capture of the Qwen4Exp path on this HIP ·
`SHIPPED` for the escape hatch `VLLM_GFX906_QWEN4_EXP_ALLOW_FULL_CUDAGRAPH` ·
`DEAD-END` for the generic per-parameter UVA path as a way to express the PLE table.

**GATE:** graph-serving A/B on `mi50-01` (2× MI50, gfx906), `logic65/Whittle-Qwen-3.8-35B-A3B`
served as `whittle`, TP=2 + `--enable-expert-parallel`, fp16, util 0.85, `--max-model-len 32768`,
`--max-num-seqs 2`, block 64, MBT 512, V2 runner, `backed_size_oblivious` dynamic shapes,
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, 27B canary gate (38–39 t/s) before **every** arm.
Arms interleaved A1/B/A2/C/D via `/local/tmp/qsa12/driver.sh`; per-arm records in
`/local/tmp/qsa12/arms.jsonl` (numbers restated here — `/tmp` is not the record).

### HYPOTHESIS

1. *If* the `full → PIECEWISE` rewrite is only a caution written while no Qwen4Exp checkpoint
   would load here, *then* a real checkpoint will capture full graphs and the rewrite costs us
   decode t/s.
2. *If* the PLE table can be expressed by the generic UVA per-parameter offload path, *then*
   `--cpu-offload-gb 24 --cpu-offload-params ngram_embedding` will move bytes off the cards and
   change the KV cache budget.

### What was done

Arm B ran `-cc.cudagraph_mode=FULL_AND_PIECEWISE` **with** the new hatch (so the rewrite could
not silently undo the experiment); arm A1/A2 ran the shipped default; arm D ran the offload
flags. The hatch was added first precisely so (1) could be *measured* rather than argued.

### Evidence — FOR (hypothesis 1 refuted, hypothesis 2 refuted)

| arm | decode | prefill-6k | prefix reuse | KV cache | VRAM | offload log |
|---|---|---|---|---|---|---|
| A1 default | 25.76 t/s | 17.76 s | ×21.42 | 72,089 tok | 27,956/27,996 MiB | — |
| A2 default (control) | 26.38 t/s | 17.60 s | ×21.51 | 72,089 tok | same | — |
| B full graphs | **did not boot** — `hipErrorStreamCaptureUnsupported` at the first capture (691 s) | | | | | |
| D `--cpu-offload-gb 24 …` | 25.53 t/s | 18.34 s | ×21.17 | **72,089 tok** | **identical** | `Offloader set to UVAOffloader` |
| C prefix caching off | 26.32 t/s | 15.12 s | **×1.35** | **80,099 tok** | 27,956/27,996 MiB | — |

* **Hypothesis 1 is dead.** The capture was refused inside
  `vllm/v1/engine/core.py:308 → determine_available_memory()` — i.e. at the first graph build,
  not later in serving. `RuntimeError: Worker failed with error 'CUDA error: operation not
  permitted when stream is capturing` / `hipErrorStreamCaptureUnsupported`. Two same-box controls
  make it model-specific, not box-specific: the 27B canary logs `Capturing CUDA graphs (FULL):
  0/1` completing, and A1/A2 captured *this* checkpoint in PIECEWISE minutes earlier. The only
  variable is the requested mode.
* **Hypothesis 2 is dead.** Both halves fired and nothing moved: the flags were accepted
  (`'cpu_offload_gb': 24.0, 'cpu_offload_params': ['ngram_embedding']`), the path engaged
  (`Offloader set to UVAOffloader` per worker), and yet the KV cache and VRAM footprint are
  byte-for-byte A1's. A1 ships no offload lines at all.
* Interleave drift A1↔A2 is **2.4 %**, so D's −0.9 % decode is noise and is reported as such.

### Evidence — AGAINST / limits

* The alternative reading of hypothesis 2 — "the offload worked but the table is small" — is
  excluded by the KV cache figure: freeing 24 GiB would move it, and it did not move by a token.
* C is **answered** (re-run 22:41–22:57 the same night; the 20:19 wedge #111 had killed the first
  attempt — the resume driver re-ran only C, since D was already healthy). Prefix caching **off**
  does not move decode: **26.32 t/s** median (22.19/26.32/26.38) against A1 25.76 / A2 26.38, i.e.
  inside the 2.4 % interleave drift. It moves two other things: the KV pool rises 72,089 →
  **80,099 tokens (+11.1 %)** at the same 0.97 GiB available and the *same* VRAM footprint
  (27,956/27,996 MiB, byte-for-byte A1's), and reuse collapses — a repeated 2.2k-token prefix costs
  **3.825 s instead of 0.237 s** (×21.4 → ×1.35), so ~3.6 s per reuse. The 6k prefill reads 15.12 s
  against 17.76/17.60 with caching on; that direction is *not* expected, it is a single
  non-interleaved arm, and prefix caching on is what sets the mamba cache mode to `align` — recorded
  as an observation, not a win.
* `--gpu-memory-utilization` was held at 0.85 throughout (≥ 0.90 wedges GPU0 on this box).

### Why it failed

* Capture: the PLE n-gram lookup is a **host-side gather with a blocking D2H copy** (the shards
  are mmap'd on the host), and HIP refuses that inside a stream capture. The split-op guard keeps
  the op in the eager region, which is only meaningful in PIECEWISE.
* Offload: the table is not a GPU-resident parameter. `MmapShardedNGramEmbedding`
  (`vllm/models/qwen4_exp/amd/ple_layer.py:166`) is "CPU-resident … backed directly by mmap'd
  safetensors shard tensors, with no TP sharding and no copying"; every rank maps the same files
  and the page cache shares one physical copy across the node. `set_shard()` raises if a shard
  arrives on anything but CPU, and the shards "are not parameters" — so a per-parameter offloader
  finds nothing to move. The mmap layout *is* the offload.

### Interactions / superseded-by · Refrigerated residue

* Supersedes the "no loadable Qwen4Exp checkpoint here" caveat that gated QSA-FN-12 and QSA-FN-15
  in `ROADMAP.md`; those entries move to `CHANGELOG.md` (see below).
* The hatch is the durable part: a future HIP with capture support, or a reworked PLE gather that
  prefetches ahead of the graph (PR #34's `ple_prefetch.py`, not ours to land), can be re-measured
  with one environment variable instead of a patch.
* Refrigerated: pinned-vs-page-cache residency for the mmap is the *real* remaining shape of
  QSA-FN-15 — the decision datum needs the reporter's 128 GB machine (this box is 46 GB, so no
  honest eviction-pressure experiment exists here). Only the host-memory profile is contributable.
* Pruned to budget in the 2026-10-06 staleness pass (rule 1, `AGENTS.md`), which is also when the
  two PROBE/coverage entries below were compressed.

## 2026-09-27 (1) — the 0.30 merge check: one build break, one supersession

**VERDICT:** `SHIPPED` for the merge prep (both findings acted on).

**GATE:** the merge of `gfx906/qsa-fn` onto `gfx906/v0.30.0-final` resolves to 10
conflicts (8 docs, 2 code), builds, boots the tiny rig with MTP k=3 + the PLE host
table, serves 3/3 requests; `test_qsa_amd.py` 22, `test_qsa_reference.py` 19 (+52
skipped), the PR #2 harnesses 52.

### The PLE commit's `.cu` hunk is a build break, not a perf tweak

`csrc/rocm/dense_gemv_gfx906.cu` arrived with the mmap-PLE commit carrying a file-scope
`atomicAdd(__half*, __half)` in an anonymous namespace (plus a duplicate `hip_fp16.h`
include and a comment typo). Nothing in the PLE path uses it and on this toolchain it
**does not compile** — HIP already provides that overload, so the ksplit>1 epilogue's
call is ambiguous:

```
.../dense_gemv_gfx906.hip:621:7: error: call to 'atomicAdd' is ambiguous
```

Reproduced through the real build (ninja → hipify → clang++ with the build's flags), then
reverted and re-compiled clean. The call site is deliberate in the base ("compiler-lowered
fp16 atomicAdd … the HSA aperture violation on an odd 32-bit CAS was observed,
2026-08-23"), so the hunk was **dropped** rather than fixed: a hand CAS substitute would
swap the M<=4 GEMV rail's accumulation unmeasured, and belongs in its own change with its
own numerics/perf gate.

### The GC guard is superseded by upstream #54646 — no consolidation needed

The merge report's remaining item ("fold the tester's shadowing into `gc_utils.py`, or keep
two mechanisms") resolves to *neither*: the 0.30 line already does the job upstream.

- `vllm/utils/gc_utils.py::freeze_gc_for_cudagraph_capture` (`c28feab989`, "[Core][MRV2]
  Freeze gc during V2 CG capture; skip per-descriptor cleanup (#54646)") freezes +
  disables GC around capture.
- `vllm/compilation/breakable_cudagraph.py::_capture` skips its per-descriptor
  `gc.collect()` while GC is disabled (`if gc.isenabled():`) — the same insurance the
  tester's shadowing provided, at the call site that matters.

The 0.29 base (this branch) has neither the helper nor the skip, which is why the guard
was needed here and stays. On 0.30 it is off by default (`GFX906_GC_FREEZE=1` enables;
unset is byte-identical to upstream), so the merge carries no stacked behaviour. Residual,
not acted on: the guard's exit traversal (upstream's `finally` does `gc.unfreeze();
gc.collect()`) was the tester's crash site on 0.29; 0.30's release gates ran it clean.

## 2026-09-24 (1) — the PLE id guard: fold, warn, never refuse a boot

**VERDICT:** `SHIPPED` (fold-and-warn adopted from the tester's third revision, our one
local delta retained); the implementation decision behind the guard — their mmap host table
vs upstream's pinned/UVA offload — stays `OPEN` (QSA-FN-11).

**GATE:** `test_ple_mmap_shards.py` (24) + `tests/v1/worker/test_gfx906_gc_freeze.py` (28)
+ `test_qsa_amd.py` (22) = **74 passed**; the tiny rig boots with MTP k=1 and k=3 and logs
**no** out-of-range id.

### Why the guard changed shape

The cherry-picked guard (`6c26a8edd6`, raise on any out-of-range id) was itself the second
boot-breaker of the day on the tester's box, in two steps:

1. **`-1` is not corruption.** `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`
   pre-fills `sample_idx_mapping` with `-1` for slots holding no real sample, and that
   tensor reaches the lookup on every MTP drafter warmup/capture. Their original reasoning
   enumerated only the ids the ngram arithmetic generates, not the other producer of the
   same tensor. Two boots died at `speculator.propose → qwen4_exp_amd_ple_ngram_embedding`.
2. **Unwritten pinned buffers are producers too.** Folding only the sentinel, keeping the
   raise, then died three more times on `0xff80ff80ff80ff80` repeated
   (`-35747867511423104`, `min == max`) arriving on the compile/graph path only — which is
   why `--enforce-eager` hid it.

The general lesson: **a guard that can refuse a boot has to be justified against producers
it cannot enumerate** — warn-and-fold by default, raise behind a flag.

### The adopted shape (theirs, verbatim)

One host-side range test over the whole tensor (ids are CPU) before the shard loop: every
out-of-range id folds onto row 0 with a warn-once per module carrying the span and count;
`VLLM_GFX906_PLE_STRICT=1` restores the raise. Folding rather than clamping matters: clamping a
large positive id would land on the *last* row, a real embedding and therefore a plausible wrong
answer, whereas row 0 costs one wrong row for slots whose result is discarded anyway.

### Our delta, and why their failure does not reproduce here

We keep `dummy_weights=` on top (a `--load-format dummy` load never delivers shard tensors, so a
missing shard serves zero rows instead of raising; a real load still raises).

Their poison run happened **without** our two gfx906 PLE fixes — their tip has neither the
splitting op in `vllm/config/vllm.py` nor the piecewise downgrade, so there the lookup sits inside
the captured graph and reads capture-time dummy inputs. On ours it is a splitting op and runs
outside the captured pieces, making the fold *defensive* rather than load-bearing — but
load-bearing for anyone serving this model without the splitting-op change. (Single tiny rig,
dummy weights, 4 layers: a weak proxy; the statement is about our configuration.)

### Interactions

- `dummy_weights` and the splitting-op/piecewise fixes are ours, made earlier.
- Their operational notes (BOOT_TRIES, VRAM reaper, `TVMFFI=disable`) are not carried; the
  generic part is in `DEVLOG-spec-decode.md` ("the GC guard").
- Upstream's pinned path stages ids through a pinned buffer too — the stale- and
  uninitialised-id hazard is worth reporting on `vllm-project/vllm#57497`.

## 2026-09-17 (1)–(5) — the QSA-FN sprint, settled (index)

Five items landed in one sprint, all `SHIPPED`; each has its full settlement entry in
`CHANGELOG.md`, so per the index rule (`AGENTS.md` rule 1) they are consolidated here — what
survives is what exists nowhere else: the FN-4 probe path and its gate figures, the FN-3 rig
constants and its three findings, and the residue. Pre-work evidence (bf16-only site inventory,
ISA facts, kernel timings) is in `RECON-qwen38-flash-qsa.md`; per-item detail in the CHANGELOG.

### 2026-09-17 (5) — QSA-FN-8: the tester bundle · `SHIPPED`

`docs/gfx906/qsa-tester-build/` (`README.md` + `make_patches.sh`, which derives the patch set from
committed state rather than hand-copying it and bundles tokenizer + tiny config so the smoke rig
needs no HF access) + artifact `/local/tmp/qsa-tester-build{,.tgz}` + `BUILD-INFO.txt`. **Gate:** a
clean base + the four patches (0001 fp16, 0002 the mamba `align` seed fix, 0003 the tiled indexer,
0004 the harness) serves the tiny rig and passes the sequence that used to fault — 4/4 `git apply`
clean against `gfx906/v0.29.0` (f79ebf2d44) reproducing **63 shipped files byte-for-byte**, and
0002/0003/0004 clean against upstream `releases/v0.30.0` (0001 needs `--exclude=…/qsa_cache.py`;
its five mechanical edits are listed in the README). The scratch-tree run was verified through
`/proc/<pid>/{cwd,environ}`; `v2mamba_repro.py … 1343 2015 4030` → 3/3 OK. **Limits:** quality and
serving numbers are the tester's (the ~60 GB W4A16 checkpoint + PLE table cannot be gated here);
re-run `make_patches.sh` after any commit in the four scopes. **Hazard:** `pkill -f <script>` killed
the tool's own shell — the pattern matched the command line; use a pidfile or a `[v]llm serve`
pattern.

### 2026-09-17 (4) — QSA-FN-4: the tiled indexer, fp16-gated · `SHIPPED`

Two CDNA hunks in `vllm/models/qwen4_exp/amd/ops/qsa.py` (`_qsa_mqa_paged_tiled_kernel` with
`BLOCK_M=16`, plus the dispatch carrying the uniform-request precondition inside the
`q.shape[0] >= 64` gate), and — the part the CDNA version lacks — the guard
`dot_is_native = q.dtype == torch.float16 or current_platform.supports_native_bf16`: the win is the
*hardware dot*, not the tiling, so gating on `supports_native_bf16` keeps CDNA's 6.57× and drops
the gfx906 regression. **Gate**
([`benchmarks/kernels/gfx906/probe_fn4_indexer_route.py`](../../benchmarks/kernels/gfx906/probe_fn4_indexer_route.py),
2048 rows, L=30720, uniform mapping, interleaved reps ×3, two runs agreeing to <1 %): fp16
**4061/4084 µs** against a 5424 µs per-row op = **1.35×/1.33×**, top-2048 agreement **1.00000**,
NRMSE 1.28e-07; bf16 selects the per-row route (1.00×, not the ungated kernel's 0.42×).
`test_qsa_amd.py` 16 → **22 passed** including `test_qsa_mqa_paged_route_selection`; same-boot
non-regression: FA 104 passed, PPL 10.5472 / 0 misses, MoE 35B 58.40 t/s. **Limit:** launch-regime
only — the indexer's serving share is not re-measured (dims 10–24× off), and when the route is not
taken the dispatch costs one device sync per prefill ≥ 64 rows.

### 2026-09-17 (3) — QSA-FN-2: the tester serve recipe · `SHIPPED`

`docs/gfx906/_serve_qsa_flash_gfx906.sh` (`start|wait|stop|report`) with two **required** gfx906
deviations: `VLLM_USE_V2_MODEL_RUNNER=1` (this model cannot run on V1 — the PLE inputs come from the
V2 model states) and, then, `--no-enable-prefix-caching` (retired 2026-09-17 with the V2-MAMBA-1
fix). Plus `--dtype float16` (no `--mamba-cache-dtype bfloat16`), maxlen 262144 native RoPE, block
64, seqs 4, bt 4096, ladder `[4,8,12,16]`, `method:"mtp"`, `qwen3_xml`/`qwen3` parsers,
expert-parallel; validated on the tiny rig with `--load-format dummy`. **Caveat:** with random
weights the chat message returns `content=None` with everything in `reasoning` — parser behaviour
on garbage; `content` vs `reasoning` on real output is the tester's to report, and nothing about
quality is implied.

### 2026-09-17 (2) — QSA-FN-3: the tiny-config harness · `SHIPPED`

`docs/gfx906/_qsa_tiny_model.py` keeps the **architecture identical** (four layer types, PLE with a
real ngram table, hyperconnection, QSA indexer + sparse attention, MTP) and shrinks only dimensions:
hidden 2560→256, 48→4 layers (QSA fraction stays 1/4), head_dim 256→64, E=512→8, MoE inter 640→64,
`ngram_vocab_size_base` 20 M→4096, max_position 262144→4096 — `compress_ratio` cannot be shrunk (the
config requires 512 or 2048) and `vocab_size` must stay 248 320. Served by
`_serve_qsa_tiny_gfx906.sh` with `--load-format dummy`. Weights are random, so it is an
**execution + A/B harness**: log `prompt_sha1`s, never quote its t/s as quality. **Baseline** (one
MI50, V2, prefix caching off): prefill 1321 tok **22 ms** (60.0 k tok/s), 2641 **42 ms**, 3961
**48 ms**; decode **B=1 563 / B=2 1023 / B=4 1994 t/s**. **Three findings:** (a) the model cannot run
on V1 at all — `Qwen4ExpModel.forward` takes `query_start_loc`/`ngram_context` with `None` defaults
and PLE raises `PLE inputs were not prepared`; the plumbing is V2's `Qwen4ExpModelState`, so the
house V1 pin is not an option for `qwen4_exp`; (b) with V2 + prefix caching on
(`mamba_cache_mode='align'`) `precopy_mamba_align_fused_kernel` faulted on a 2015-token prefill (grid
`[256,7,16]`, 1343-token prefills pass) → filed `V2-MAMBA-1`, **fixed the same day** as a wrong
divisor in `add_request`, not a kernel bug, workaround retired; (c) the bf16 arm is independently
broken (`rocm_unquantized_gemm_impl`, "Matrices A and B must have the same dtype" in the
hyperconnection chain), so the whole-model fp16-vs-bf16 A/B is unavailable and the kernel numbers
stand as the evidence. **Hygiene:** warm each batch shape — a one-time Triton-autotune storm
measured 6.5 t/s and 563 t/s in the same process — and pin `VLLM_PLUGINS=` (the venv carries five
stale profiler plugin entry points; A/B'd inert, B=1 466 vs 448).

### 2026-09-17 (1) — QSA-FN-1: fp16 enablement · `SHIPPED`

The reported failure was the gfx906 bf16→fp16 auto-fallback meeting QSA's bf16-only guards; the CDNA2
patch set did not fix it. **Guard/dtype generalisation only — no kernel code changed.**
`common/qsa_cache.py` gains `QSA_ACTIVATION_DTYPES`/`QSA_KV_CACHE_DTYPES` as the single source of
truth (`_BF16_PER_INT64` → `_ELEMS_PER_INT64`, same value 4; `bind_kv_cache` checks `self.dtype`);
`amd/qsa.py` / `amd/indexer_qsa.py` / `amd/ops/qsa.py` take `model_config.dtype` at every guard and
cache, including the attention activation guard that was **the reported error**; `amd/model.py` (×2)
+ `amd/mtp.py` (×1) pass it to `HyperConnectionConfig(params_dtype=…)` (pulled in from QSA-FN-2 —
HC linears must match the activations). `common/` stays dtype-general: it is shared with the NVIDIA
implementation, which is untouched. **Gate:** `tests/models/qwen4_exp/` both dtype arms **plus** the
FN-7 regression — FA suite 104 passed, PPL probe 10.5472, MoE 35B 58.30 t/s mean (= parity).
**Kernel win (launch-regime evidence, not the gate):** sparse attention 26.5 ms fp16 vs 116.5 ms
bf16 (**4.39×**), per-row indexer 5426 vs 6928 µs (1.28×). **Why it works:**
`v_dot2_f32_f16` is a real gfx906 instruction and Triton 3.8.0 (`GCN5_1`) emits it for fp16 dots,
while Vega20 has no bf16 instruction — a codegen fact, which is why guard edits alone bought the
win. **Limits:** no served request existed in this entry (the tiny rig covered it later); the
indexer's compress/store/selection kernels are not exercised in fp16; the config-shape plumbing
needed a VllmConfig fixture, which QSA-FN-3 provided. **Residue:** `QSAKeyStateCache`'s int64 packing
width is still hardcoded 4 (`8 // element_size` is the generic form, needed only for a 1-byte cache
dtype); `HyperconnectionConfig.params_dtype`'s default is now dead (all six sites pass it);
`mamba_cache_dtype` for this model is still pinned bf16 by the CDNA recipe.
