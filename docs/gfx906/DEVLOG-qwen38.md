# Qwen3.8-27B on gfx906 — onboarding, dtype fix, NAS-crash postmortem

Branch: `gfx906/main`. Model: `cyankiwi/Qwen3.8-27B-AWQ-INT4`.
Merged (2026-08-20) from `qwen3.8_crash.md` so one file carries the
whole model train (onboarding + dtype fix + the NAS-crash postmortem).

**VERDICT: PROVISIONAL — loads & serves in eager (experimental).** The
bf16 crash is fixed (`--dtype float16` auto-fallback). No graph-mode /
evalk-it number yet; MTP on this model is open.

## TL;DR (the 3-sentence version)

1. **Two "machine-killing" boots were a NAS outage, not a GPU/kernel/
   vLLM bug** — the GPU was healthy, the power cycles were forced by
   unkillable D-state NFS I/O threads (postmortem in §Crash).
2. **The Qwen3.8 init crash was a bf16-vs-fp16 dtype mismatch** (gfx906
   has no native bfloat16; the fp16-only kernel stack can't feed a
   bf16-forwarded residual). Fixed by the shared auto-fallback
   (§`69f615b98a`): bf16 checkpoint → float16 with a warning.
3. **Loads clean with `--dtype float16 --enforce-eager`** (60 s NVMe,
   KV pool 92,521, coherent); graph-mode + speed numbers still open.

## Crash postmortem — the two NAS outages (unrelated to the dtype fix)

> This is the merged content of `qwen3.8_crash.md` (2026-08-19). Kept
> in full — the findings are the durable value.

**Verdict:** neither crash is a GPU fault, a Qwen3.8 kernel bug, or a
vLLM bug. The NAS (`192.168.33.240:/volume2/ai` → `/data`, NFSv4
**hard** mount) became unresponsive while vLLM loaded the 20 G model
through `np.memmap`'d safetensors. The amdgpu HMM page-pinning path
(H2D weight copy) blocked in D-state on NFS reads; the GPU "fence
timeout"+"unrecoverable state" lines are *downstream symptoms*. The
BACO reset **succeeded in both**; the power cycles were forced by
unkillable D-state threads + the dead NFS share blocking unmount.

**Decisive evidence (hung-task stacks):** both boots' dumps bottom out
in NFS page reads, not GPU waits:

```
Crash #1 (18:45:08, pid 170137)          Crash #2 (20:02:34, pid 34003)
task:VLLM::EngineCor state:D             task:VLLM::EngineCor state:D
  io_schedule                              io_schedule
  folio_wait_bit_common                    folio_wait_bit_common
  filemap_fault                            filemap_fault
  hmm_range_fault                          hmm_vma_fault / hmm_range_fault
  amdgpu_hmm_range_get_pages [amdgpu]      amdgpu_hmm_range_get_pages [amdgpu]
```

`amdgpu_hmm_range_get_pages` pages in host pages for a DMA (H2D) copy;
`filemap_fault → folio_wait_bit` = the page is file-backed (mmap'd
safetensors on NFS) and the kernel waits for an NFSv4 READ. On a hard
mount that wait is unbounded → D-state → unkillable.

Supporting facts:
- NAS "not responding ... still trying ... timed out" in **both** boots
  (18:44:42–18:45:37; 20:03:37–20:04:29). These lag the outage (they
  appear when RPC timeout backoff expires).
- **Zero** GPU page faults in either boot; **zero** second fence
  timeouts after the resets.
- Both crashes fell in the 20 G NFS weight-load window (fence timeout
  at T+115 s / T+46 s).

**Mechanism, step by step:** safetensors `np.memmap` over NFS → H2D copy
calls `amdgpu_hmm_range_get_pages` → uncached page → NFSv4 READ → NAS
dead + hard mount → D-state on folio bit → H2D fence can never signal →
`fence wait loop timeout` → queue preemption fails → "unrecoverable
state" → **BACO reset succeeds** → D-state thread survives (NFS I/O)
→ systemd poweroff can't unmount dead `/data` → hang → power cycle.

**The two incidents:**

| | Crash #1 (boot -2) | Crash #2 (boot -1) |
|---|---|---|
| Launch | 18:39:47 (graphs, default) | 19:59:42 (`--enforce-eager`) |
| Block start | ~18:43:06 | ~19:59:52 (T+10 s) |
| Fence timeout | 18:41:42 (T+115 s) | 20:00:28 (T+46 s) |
| BACO reset | succeeded 18:41:44 | succeeded 20:00:30 |
| NAS not-responding | 18:44:42–18:45:37 | 20:03:37–20:04:29 |
| Hung task | `VLLM::EngineCor:170137` | `VLLM::EngineCor:34003` |
| Shutdown | poweroff hang → power cycle | SIGKILL (no effect) → power cycle |

Eager and graph failed identically → retires the CUDA-graph-capture
hypothesis; the trigger is the NFS load window, independent of mode.

**What this is *not*:**
- **Not the 0.6B FA UAF** (`5d960a503c`): real GPU memory bug, no-retry
  *write* page faults, SIGABRT exit-134, no D-state. Different
  mechanism. (Conversely, this journal's kernel faults at
  `0x7584fc000000`–`0x75850c400000` bracketing the runtime-reported
  `0x7584ff900000`, `RW:0x1`=write, unmapped — *confirmed* the UAF.)
- **Not proven innocent of Qwen3.8**: both died during weight load,
  before model kernels meaningfully ran. The D=256 serving question
  (gfx906 FA/gather, 3.8 block sizes, GDN params) remains open.

**Why Qwen3.8 (correlation):** the 20 G AWQ checkpoint lives on NFS
(`/data/cache/huggingface/...`) → each load is a multi-minute NFS
stream. The same NAS served the 15 G 27-B load cleanly many times; MoE
from local `/local/models` are immune — exposure scales with load
size/duration.

**Actions:**
1. Check the NAS (Synology `192.168.33.240`, `/volume2/ai`) logs 18:39–18:46 / 19:59–20:06 + disk health; fix if flaky.
2. Copy the model to local disk and re-validate from local weights (removes the failure class, still answers the D=256 question).
3. Robustness: hard NFS + mmap'd loads means *any* NAS hiccup wedges the box. Keep weights local (preferred) or mount the share `soft,timeo=` so reads fail cleanly.
4. Operational signature: if a fence timeout comes with D-state `VLLM::EngineCor` bottoming at `amdgpu_hmm_range_get_pages`/`filemap_fault`, treat as I/O — check the NAS first, power-cycle to recover.

**State (2026-08-19, post-reboot):** boot since 20:06:54, no amdgpu
errors; `/data` is `auto`/`nofail`. The reboots wiped `/tmp/bench/*`,
`/tmp/fa-analysis.md`, `/tmp/HANDOVER-vllm-27b.md` — the committed
repo copies are the durable record.

## Model facts (gfx906-relevant deltas vs the 3.5-27B family)

> Family architecture (hidden 5120, 24 q / 4 kv heads, GDN) in
> `DEVLOG-dense-decode.md` §Dense model facts. Only the 3.8 deltas:

- **compressed-tensors W4A16** (`pack-quantized`, I32 packed weights +
  I32 zero points) — *not* the auto_awq of 3.5-27B/MoE. Dense linears
  on `TritonW4A16LinearKernel`; GDN on Triton/FLA; **q/k/v projections
  quantized** (3.5-27B left them unconverted).
- **FA head_dim 256** (3.5-27B is 128); GDN head_k_dim 128; MTP head
  present (`mtp_num_hidden_layers: 1`).
- Unquantized tensors (norms, embed, layer-0 GDN, mtp) are **BF16**;
  config has **no `torch_dtype`** — *why* the auto-fallback matters.
- Local copy: `…/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4` (standard
  HF layout, offline-resolvable).

## The bf16 crash — root-cause chain (the fix's canonical record)

Server launch (`--enforce-eager`) crashed in `_dummy_run`:
`fused_add_rms_norm …/layernorm_kernels.hip:320` =
`input.scalar_type() == residual.scalar_type()` — dtype mismatch at a
decoder `post_attention_layernorm` (qwen3_next.py:533).

1. No `torch_dtype` → `get_torch_dtype` falls back to the safetensors
   weight dtype → **bf16** (dominant unquantized dtype).
2. `_resolve_auto_dtype` trusts it: bf16 ∈ ROCm `supported_dtypes` →
   model dtype = bf16.
3. **gfx906 has no native bfloat16** + the fp16-only kernel stack → the
   attention-output / residual-stream dtypes diverge in-layer → the
   host check fires.

Cross-check: another user runs this model fp16 on gfx906 (plus
tool/reasoning parser flags) — fp16 is workable (model not in
`_FLOAT16_NOT_SUPPORTED_MODELS`).

## The fix (canonical here — gemma4 onboards onto it)

> Signpost: commit **`69f615b98a`**. `DEVLOG-gemma4-onboarding.md` §2 is
> a *second, smaller* mention (same auto-dtype fallback, "second model
> validated"). Keep the detailed description here.

- `Platform.supports_native_bf16` (base `True`); `RocmPlatform`
  override `not _ON_GFX906` (safe at config-parse; `_GCN_ARCH` via
  amdsmi, no CUDA init).
- `_resolve_auto_dtype`: bf16 config + fp16 in the model's supported set
  + platform lacks native bf16 → **fall back to float16 with a warning**
  (explicit `--dtype` overrides). The `float16 in supported_dtypes` test
  keeps fp16-forbidden models (gemma2/3, glm4, plamo2) on bf16.
- `_get_and_verify_dtype`: explicit `--dtype bfloat16` on such a
  platform warns (still honored).
- `tests/config/test_dtype_resolution.py`: 4 unit tests. All pass.

## Validation

- `auto` → warning + **float16** (was: silent bf16 → crash);
  explicit `bfloat16` → warning + bfloat16 (honored).
- Server `--dtype float16 --enforce-eager`
  (`/local/tmp_q38/fp16_eager.log`): loads clean, KV pool 92,521,
  coherent Eiffel-Tower probe. ~9 min startup = vision-encoder
  profiling (VL model) + Triton JIT warmup.
- D=256 FA path: eager forward passes; **graph-capture + speed numbers
  still open** (eager chosen first to de-risk post-NAS).

## Open items

- Graph-mode + single-request serving speed (D=256 capture path).
- MTP k=2 on 3.8 (checkpoint has the head; recipe from the 27B work) —
  spec decode on a bf16-native model running fp16.
- PPL/coherence sanity probe (`ppl_probe.py`).

## References (avoid re-documenting here)

- Shared auto-dtype mechanism: `DEVLOG-gemma4-onboarding.md` §2
- 3.5-27B architecture facts: `DEVLOG-dense-decode.md` §Dense model facts
- `--dtype float16` + support recipe: `README.md` model-support table
- 0.6B FA UAF (distinct mechanism): the FA gather-buffer record in
  `DEVLOG-fa-attention.md`