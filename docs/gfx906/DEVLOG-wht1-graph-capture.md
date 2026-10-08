# WHT-1 graph capture: a one-line family-match bug was the whole blocker

Date: 2026-10-05. Item: **#36 WHT-1**, arm "graph capture" (follows
`DEVLOG-wht1-whittle-load.md`). Branch: `gfx906/wht1-graph-capture`.
*This log consolidates two accounts of the same arm written concurrently by the
terminal and container agents (originally `DEVLOG-wht1-graph-capture.md` and
`DEVLOG-wht1-cudagraph-guard.md`); the duplicate regression file is folded in
below.*

## Result

Graph capture works on this model, and decode at B=1 goes from 5.61 t/s to
**21.15 t/s** — same probe, same flags, one variable changed (`--enforce-eager`
removed; the split bug below fixed).

| | eager (control) | graphed |
| --- | --- | --- |
| rep 1 (includes warmup) | 2.75 t/s | 4.25 t/s |
| **rep 2** | **5.61 t/s** | **21.15 t/s** |
| KV cache | 2.0 GiB / 117,964 tok | 1.07 GiB / 62,914 tok (3.84x at 16k) |
| output | coherent, identical | coherent, **identical** |

The KV difference is the cudagraph pool reserving device memory up front, not a
capability loss.

Timeline of the verified run (`/local/tmp/wht1/smoke2/`, 13:01–13:14):

```
13:01:42 WARNING [vllm.py:1937] Qwen4Exp's host-resident PLE n-gram lookup cannot run inside a
                           full cudagraph (blocking D2H mid-graph); downgrading cudagraph_mode to PIECEWISE.
13:12:42 Graph capturing finished in 27 secs, took 0.66 GiB     (PIECEWISE 3/3)
13:13:06 Graph capturing finished in 4 secs, took 0.26 GiB      (CUDA graph pool 0.26 GiB actual)
13:13:28 throughput 26 tok in 6.1s = 4.25 t/s (rep1, warmup); 26 tok in 1.2s = 21.15 t/s (rep2)
13:13:36 teardown; post-load canary per protocol
```

## Root cause: `model_type` was matched exactly, and this checkpoint spells it differently

The failure signature was a capture-time abort on both TP workers:

```
File ".../vllm/models/qwen4_exp/amd/ple_layer.py", line 1237, in ...
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
Search for `hipErrorStreamCaptureUnsupported' ...
```

Finding *which* op is the whole triage: the engine-level `RuntimeError` names no
op, but the **innermost Python frame of the worker traceback** does —
`torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding`, registered at
`amd/ple_layer.py:1271`, dying in that op's body at line 1237.

Line 1237 is `pinned_ids.copy_(ngram_ids, non_blocking=False)` — a **blocking
device->host copy**. The PLE n-gram table is host-resident (mmap'd shards), so the
ids must land on the host before the CPU gathers rows; the comment above it
explains that a blocking copy is required for correctness, and the H2D of the
result stays async. That makes the op structurally uncapturable.

The fork already knows this. `vllm/config/vllm.py` appends
`"vllm::qwen4_exp_amd_ple_ngram_embedding"` to `splitting_ops` for Qwen4Exp
models and downgrades FULL capture to PIECEWISE with a warning. The guard was:

```python
getattr(self.model_config.hf_config, "model_type", None) == "qwen4_exp"
```

Whittle's `config.json` says **`qwen4_exp_text`**, so the guard was False, the op
was never split, and the capture swallowed a host round-trip. The fork's own
engram table (`vllm/config/engram.py`) avoids this trap by keying on the
architecture `Qwen4ExpForCausalLM`; this line keyed on a per-checkpoint spelling.

Three independent observables confirm the guard never fired (each is what the
guard's own comment predicts when it *does* fire):

| expected if the guard had fired | observed in the failing run |
| --- | --- |
| `splitting_ops` contains `…ple_ngram_embedding` | absent (it does contain `qwen4_exp_ple_short_conv`) |
| `cudagraph_mode` downgraded to PIECEWISE | stayed `FULL_AND_PIECEWISE (2,1)` |
| the downgrade warning is logged | absent |

The fix landed as commit `e3b6542393` on branch **`gfx906/wht1-cudagraph-guard`**,
written by the terminal agent that picked this thread up concurrently (same root cause
found independently, same 13:16 minute). Its implementation is the better one: a reusable
`is_qwen4_exp_config()` helper in `vllm/models/qwen4_exp/config.py`, consulted for both
`hf_config` and `hf_text_config`, so a multimodal wrapper config is matched too. The
earlier draft of this document described a narrower fix (architecture prefix first,
`model_type` prefix as fallback) that never landed — this measurement was taken against
the working tree while that draft was in place, and the two are semantically equivalent
for this checkpoint, but the helper is what is in the branch.

The helper is anti-drift rather than a widened literal: the spellings
(`qwen4_exp`, `qwen4_exp_text`, plus the drafter rewrite `qwen4_exp_mtp`) are
**derived from the config classes** that declare them, in one place, and both
matching sites use it — the compilation guard (`vllm/config/vllm.py`) and
`SpeculativeConfig`'s MTP rewrite (`vllm/config/speculative.py`), which until now
carried its own literal. Adding a family config class extends the family
automatically, so the third site cannot drift the way this one did.

Regression cover, kept knowingly as two layers rather than one:

- `tests/config/test_qwen4_exp_ple_splitting_ops.py` — the guard's contract: the
  op lands in `splitting_ops` for **both** spellings, and repeated `VllmConfig`
  builds do not accumulate duplicate entries.
- `tests/config/test_qwen4_exp_family_match.py` — the helper itself: every
  declared spelling accepted (asserted against the config classes, so a new class
  must be covered), `qwen4_exp_mtp` accepted, `qwen3_5` / `qwen4_exp_extra` / `""`
  / non-strings rejected, the architecture fallback, and missing-`model_type`
  tolerance.
- The overlap between the two files is the layer each pins (config plumbing vs
  the compiled decision), not a duplicated assertion.

`ruff check` clean on the touched files; `import vllm.config.vllm` unchanged at
~13.5 s (the arch config module pulls 5 light modules and never imports
`vllm.config`, so there is no cycle).

## The three failures on the way there, in order

1. **Default compile** — `ConstraintViolationError: Constraints violated
   (L['query_start_loc'].size()[0])` in `determine_available_memory`. Not the
   capture ladder: vLLM's own default power-of-two ladder fails identically.
   `--enforce-eager` sidestepped it at the cost of graphs.
2. **`-cc.dynamic_shapes_config.type=unbacked`** (the remedy documented in
   `docs/design/debug_vllm_compile.md` for exactly this error) got past the
   profiler and died in the GDN layer instead: `Could not guard on data-dependent
   expression Ne(128*u0, 0)`, at
   `qwen_gdn_linear_attn.py:956` — `z.reshape(z.size(0), -1, self.head_v_dim)`.
   A `-1` reshape against a symbolic extent forces dynamo to reason about the
   shape product. **Not applied**: see "ruled out" below.
3. **`-cc.dynamic_shapes_config.type=backed_size_oblivious`** cleared dynamo and
   the profiler both, and reached capture — where it hit the `model_type` bug.
   With the bug fixed, this is the mode that serves.

## Ruled out

| Hypothesis | Verdict |
| --- | --- |
| Capture ladder too small | No — the default ladder reproduces failure 1 exactly |
| The GDN `-1` reshape must be rewritten first | Not needed for this arm — it is a latent dynamo hazard in *unbacked* mode only; `backed_size_oblivious` never reaches it. Worth fixing on its own merits if the unbacked mode is ever wanted. |
| A capture-unsafe op in the attention/GDN kernels | No — the abort names `ple_layer.py:1237`, the host-read PLE lookup |
| The PLE "16 of 16 ids were outside the table; folded onto row 0" warning is a serving defect | No — 2 occurrences, both during `Graph capturing finished`, none during inference; the same warning appears in the eager run. It is the capture-time dummy batch over a pinned buffer that is not written yet, exactly as the message says. `VLLM_GFX906_PLE_STRICT=1` turns the fold into a raise if it ever needs proving again. |
| The fix changes numerics | No — the output text is byte-identical to the eager run on both probes; splitting only moves capture boundaries, and the regression test pins the config-level effect |
| A host sync in the QSA path | No — the only `.item()` under `qwen4_exp/` is `common/qsa_cache.py:684`, and it reads **CPU** tensors (`query_start_loc_cpu`), so there is no device sync |
| Cross-stream / prefetch streams stuck in the region | No — no `Stream`/`Event`/`wait_stream` anywhere in `qwen4_exp/amd/**`; the `nvidia/ngram_embedding.py` prefetch streams do not run on gfx906 |
| Capture-illegal HIP API in the fork's C++ | No — the `cudaMalloc`/`cudaMemcpy`/`hipMalloc` hits are in `custom_all_reduce`/`quickreduce`, both out of play (`--disable-custom-all-reduce`) |
| EP/MoE all-to-all host syncs | No — the `fused_moe` `.item()`/`.tolist()` hits are init-time (`expert_map`) or `routed_experts_capturer` (off by default) |

## Self-review: what the widened match must not cost

Widening a guard is only safe if every added match *needs* it. Reviewing the wider
match against the family's own runtime gates found three ways it could have cost
something it must not -- all now closed, all covered by tests:

| risk | why it would have happened | fix |
| --- | --- | --- |
| A family checkpoint with **no PLE layers** (and the **MTP drafter**, whose config is rewritten to `qwen4_exp_mtp`) got downgraded `FULL_AND_PIECEWISE` → `PIECEWISE` | the guard asked only "is this the family?", while the op exists only when the text config declares `ple_layer_ids` -- the model state's own gate (`bool(config.ple_layer_ids)`) | `uses_ngram_embedding()` is now required as well |
| **Non-ROCm platforms** lost full-graph capture for nothing | the op is AMD-only: `vllm::qwen4_exp_amd_ple_ngram_embedding` is registered only in `amd/ple_layer.py`, and the CUDA path prefetches the lookup through streams over a device-side table (`nvidia/ple_layer.py`) with no such op, so it is capturable by construction; on CUDA the append matched nothing while the downgrade still applied | `needs_ple_ngram_split(hf_config, is_rocm=...)` -- the guard is now ROCm-only |
| A future family config class that omits its own `model_type` **dragged a parent's spelling into the family** (`qwen3_next`, `qwen3_vl_vision`) | the tuple was built by attribute lookup, so a missing class attribute silently inherited the parent's value, and the compilation guard would then fire for unrelated models | the tuple reads each class's own `__dict__`, and a test fails if a family class does not declare its own `model_type` |

Two smaller hardenings: `is_qwen4_exp_config()` now only iterates known container
types for `architectures` (it runs for *every* model built, so a malformed
third-party value must not raise during engine init), and the `architectures`
string form is accepted rather than iterated character-wise.

The tests were the other finding: the guard's regression test built its config from
the Whittle snapshot's `config.json` on a hard-coded path and skipped as a *module*
when that was absent, so the fix was unpinned on any other machine. It now builds
synthetic config dirs (runs anywhere), keeps a real-checkpoint test that runs when
the snapshot is cached, and asserts the negatives too -- a PLE-less family config and
a non-family config must **not** be split, and must **not** have their capture mode
touched.

Resulting behaviour, through the real `ModelConfig`/`VllmConfig` path (the one the
engine uses at init), on this ROCm box:

| checkpoint config | op split out | cudagraph_mode |
| --- | --- | --- |
| `qwen4_exp_text` + `ple_layer_ids: [2]` (Whittle) | yes | `PIECEWISE` |
| `qwen4_exp` + `ple_layer_ids: [2]` | yes | `PIECEWISE` |
| `qwen4_exp_text` + `ple_layer_ids: []` | no | `FULL_AND_PIECEWISE` (untouched) |
| `qwen4_exp_text` without the key | no | `FULL_AND_PIECEWISE` (untouched) |
| `llama` | no | `FULL_AND_PIECEWISE` (untouched) |

Verified after the change: 54 tests pass across the two files (including the
real-Whittle-config case), `ruff check`/`ruff format --check` clean, and
`import vllm.config.vllm` unchanged at ~12.5 s. The narrowing does not touch the
measured behaviour above: Whittle still appends the op and still downgrades, which is
why the verified capture run stays valid.

## Open

- **Is `backed_size_oblivious` still required now that the split is applied?**
  The winning combination has two changes; which one carries failure 1 is
  untested. A cheap bisect (split fix + default dynamic shapes) would say.
- **Envelope arm**: largest `max_model_len`/MBT at util 0.85, TP=2. Note the
  graph pool now reserves VRAM up front, so re-measure the KV budget alongside.
- **AMD warmup gap**: `model_executor/warmup/qwen4_exp_qsa_warmup.py` looks up
  `vllm.models.qwen4_exp.**nvidia**.*` in `sys.modules` and returns early on AMD,
  so the QSA/indexer warmup never runs on this box. Not the cause of the capture
  failure (a JIT-latency concern, not a capture-illegal op), but first-call cost is
  currently unwarmed here. Confirm whether it is measurable, then either wire the
  AMD modules in or record it as harmless.
- **Proper throughput measurement**: the 21.15 t/s is a 26-token probe; the arm
  needs the same shape as the eager A/B before any number is quoted. Never quote
  eager or short-probe timings as kernel time (see `dvfs-mi50.md`).
- **MTP is absent from the checkpoint** (`mtp_num_hidden_layers: 1` with zero MTP
  tensors in the index), so there is no speculative-decode multiplier to add; the
  remaining throughput work is the envelope and batch shape, not a drafter.
- Quality gates once the envelope settles. Reported to the uploader: the
  root-packed PLE table and the MTP config/checkpoint inconsistency.
