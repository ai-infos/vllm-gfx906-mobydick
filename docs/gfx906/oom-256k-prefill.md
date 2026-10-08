# 256k-context prefill OOM on Qwen3.8-27B — mechanism and evidence

2026-08-23, 2× MI50 (gfx906), ROCm 7.14, `vllm-gfx906-mobydick` @
`gfx906/main`. Model: `/local/cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4`
(64 layers: 16 FA + 48 GDN, `full_attention_interval=4`, hidden 5120,
intermediate 17408, FA 24 q-heads × head_dim 256 / 4 kv-heads, GDN
16 k / 48 v × 128, vocab 248,320, compressed-tensors AWQ INT4, **no
`ignored_layers` — the lm_head is quantized too**).

**VERDICT:** a ~250k-token first prefill OOMs at `free: 0` on the
exllama AWQ **per-call dequant scratch** — a token-count-*independent*
allocation (170 MB for the MLPs, **2.37 GiB for the lm_head on every
forward**). Chunk size, `gpu_memory_utilization`, MTP and
prefix-caching/mamba-align all fail to fix it (each tested). The
post-capture headroom left for runtime transients is util-independent
by construction (~1.9–2.5 GiB/rank here) and is drained by
unprofiled request-time consumers plus an unidentified ~1–2 GiB
long-context in-flight transient. **131k is the validated context
ceiling on this model** (dense Qwen3.5-27B serves 256k fine — smaller
footprint). Not a serving-time leak.

## 1. Symptom (verbatim)

Every failed 250k-needle request (250,000-token prompt,
`/tmp/needle_256k.py`), ~2.5–4.5 min into the prefill, both ranks:

```
[rank0]:[W823 12:13:20.229436006 HIPCachingAllocator.cpp:3934] memory
allocation failed with OOM on device 0 while trying to allocate
178257920 bytes (free: 0, total: 34342961152).
(Worker_TP0 pid=20182) ERROR 08-23 12:13:20 [multiproc_executor.py:1060]
    buf10 = torch.ops._C.gptq_gemm.default(buf9, arg8_1, arg9_1, arg10_1, arg11_1, True, True, 4)
(Worker_TP0 pid=20182) ERROR ... RuntimeError: torch_call_dispatcher( "aten::empty", "memory_format", ...
```

The worker dies, the engine shuts down, the API server returns HTTP 500.
Identical byte count (178,257,920) and `free: 0` in all four 256k TP=2
arms (12:13:20, 12:45:14/18, 13:06:18); earlier TP=2 arms died in
`gptq_gemm` (chunk 8192) or the FA `_q_pad_buf` 186 MB (chunk 4096) —
different first straws, same drained headroom.

## 2. What the 178,257,920 B allocation is

`csrc/libtorch_stable/quantization/gptq/q_gemm.cu:1833` — every
`gptq_gemm` call allocates **two** buffers, the second of which is a
dequant scratch sized purely by the *weight* shape:

```cpp
auto c = torch::stable::empty({a.size(0), b_q_weight.size(1)},
                              a.scalar_type(), std::nullopt, a.device());
auto temp_dq =
    torch::stable::empty({b_q_weight.size(0) * 32 / bit, b_q_weight.size(1)},
                         a.scalar_type(), std::nullopt, a.device());
```

With the exllama-v2 packed layout (`b_q_weight = [N, K×bit/32]`, bit=4 →
`[N, K/8]`), `temp_dq = [N×8, K/8]` fp16:

| weight | N | K | temp_dq | size |
|---|---|---|---|---|
| MLP gate_up | 17,408 | 5,120 | [139,264, 640] | **178,257,920 B** (the observed OOM) |
| MLP down | 5,120 | 17,408 | [40,960, 2,176] | 178,257,920 B |
| GDN in_proj_qkvz | 16,384 | 5,120 | [131,072, 640] | 167,772,160 B |
| **lm_head** | **248,320** | 5,120 | [1,986,560, 640] | **2,542,796,800 B = 2.37 GiB** |

The scratch is freed after each call (the caching allocator normally
reuses the block), so it only fails when the remaining headroom has no
contiguous block of that size. The lm_head's 2.37 GiB scratch is
*larger than the entire headroom* in most runs — it can only succeed
when almost all of the headroom is free and contiguous at the end of
the forward. The per-layer 170 MB scratches fail first (model layers
precede the lm_head).

**This is why no launch-flag combination moved the needle:** the
failing allocation does not depend on `max_num_batched_tokens`,
sequence length, batch size, MTP, or the mamba cache mode. (An earlier
session red herring: 178,257,920 = 5,120 × 17,408 × 2 also parses as
"[5,120 tokens × gate_up] fp16" — 5,120 is the *hidden size*; the
allocation is the weight scratch, not a token buffer.)

The failing graph in the AOT inductor cache
(`~/.cache/vllm/torch_compile_cache/torch_aot_compile/9431246e…/
inductor_cache/jk/cjkcvb…py`, dynamic-shape region) shows
`buf10 = torch.ops._C.gptq_gemm.default(buf9, arg8_1, …)` followed by
`assert_size_stride(buf11, (s18, 17408), (17408, 1))` — i.e. the
17,408-wide MLP (s18 = dynamic token count) — and the file contains no
MTP references, so it is the **backbone** (the MTP head graph, compiled
separately, has its own 24×128-head attention and `fc [2560, 10240]`).

## 3. The memory accounting (verbatim, from logs still on disk)

Run 4 (TP=2, util 0.82, MTP k=2, chunk 1024) — `/tmp/tp2_27b_serve.log`:

```
11:59:18 [config.py:605] Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default when prefix caching is enabled
12:01:08 [gpu_model_runner.py:5520] Model loading took 10.0 GiB memory and 75.502855 seconds
12:01:08 [interface.py:919] Setting attention block size to 800 tokens to ensure that attention page size is >= mamba page size.
12:09:10 [gpu_model_runner.py:6946] Estimated CUDA graph memory: 1.59 GiB total
12:09:10 [gpu_worker.py:579] Available KV cache memory: 12.29 GiB
12:09:10 [kv_cache_utils.py:1875] GPU KV cache size: 364,688 tokens, Maximum concurrency for 262,144 tokens per request: 1.39x
12:09:22 [gpu_worker.py:742] CUDA graph pool memory: 2.0 GiB (actual), 1.59 GiB (estimated), difference: 0.4 GiB (20.2%).
12:09:22 [gpu_worker.py:805] Free memory on device (31.53/31.98 GiB) on startup. Desired GPU memory utilization is (0.82, 26.23 GiB).
```

Run 5 (same, MTP off) — `/tmp/tp2_27b_nomtp.log`: weights 9.6 GiB,
graph 1.45 est / 1.82 actual, KV 12.93 GiB (415,125 tokens), block size
784.

vLLM sizes the KV pool to *fill* the budget:
`KV = budget − weights − profiled_peak − graph_estimate`, so the
headroom left after graph capture for everything runtime is

```
H = budget − weights − KV − graph_actual
  = profiled_peak + graph_estimate − graph_actual
```

— the budget (hence `--gpu-memory-utilization`) **cancels out**:

| run | budget | weights | KV | graph (est/act) | H |
|---|---|---|---|---|---|
| 4 (MTP) | 26.23 | 10.0 | 12.29 | 1.59 / 2.0 | **1.94 GiB** |
| 5 (no MTP) | 26.23 | 9.6 | 12.93 | 1.45 / 1.82 | **1.88 GiB** |

(An earlier 0.82+MTP boot with chunk 2048 read 11.72 GiB KV / H ≈ 2.51
GiB; the ~0.5 GiB spread is warm-cache profile variance.) So at the
failing moment, the 250k-sequence forward working set had consumed the
entire ~1.9–2.5 GiB headroom, and the next 170 MB scratch hit
`free: 0`.

## 4. Why a 250k sequence is the trigger

The contrast that isolates the context as the common factor: the W4
A/B + soak on this same model (TP=1, util 0.90, **maxlen 1536**,
`--max-num-seqs 8`, 30 reps × 2 models, ~20 min) ran at 98–99 % flat
VRAM with **zero OOM** — with a KV pool of comparable size (~400k
tokens), i.e. the same lazy Q8 side-buffer and the same per-call
scratches, but no long in-flight sequence.

Unprofiled consumers that materialize at request time (the profiling
dummy run never sees them at this scale):

- **Q8 KV side-buffer**, `vllm/gfx906_fa/gfx906_fa_backend.py:385`
  `_ensure_q8_sidebuffer` — lazy-allocated to the *full* K-cache size
  on the first forward (`[num_blocks, block_size, Hkv, D/32×34]`
  uint8 ≈ half the fp16 K cache, ~0.4 GiB for a ~400k-token pool).
- **`_ensure_forward_buffers`** (same file, :401) — `_q_pad_buf` /
  `_q_pad_decode_buf` / gather buffers grow on demand to the largest
  forward shape seen and never shrink (retired buffers are kept alive
  once a capture has used them — required for graph VA stability).
- **Inductor dynamic-shape buffers** — the profile run compiles one
  token-count shape; serving at a different chunk size (800 with
  mamba-align, 1024 without) runs a different graph with fresh
  allocations.
- **An unidentified ~1–2 GiB long-context in-flight transient** — the
  remainder of the drain. Candidate sources considered and excluded by
  size/shape: FA paged-KV reads (no score materialization), GDN state
  (per-sequence, fixed), block tables (313 entries), MTP hidden
  states (MTP arm ruled out). **Open: capture a worker
  `torch.cuda.memory._record_memory_history()` snapshot at OOM time.**

## 5. What was ruled out, and the evidence

| hypothesis | test | result |
|---|---|---|
| `mamba_cache_mode: align` bumps the chunk cap to a multiple of the mamba block size | code + `--no-enable-prefix-caching` arm | **false at the scheduler** (below) — and the npc arm (align off) OOMed on the identical 170 MB |
| MTP / drafter footprint | MTP-off arm | OOM identical (12:45:18, same bytes) |
| prefill chunk size | 8192 → 4096 → 2048 → 1024 | all OOM (allocation is token-independent) |
| `gpu_memory_utilization` | 0.93 → 0.90 → 0.82 | all OOM (H is util-independent, §3) |
| serving-time VRAM leak | W4 30-rep soak (this + 35B MoE) | flat 98–99 %, zero growth — negative |

**mamba-align detail** (the hypothesis deserves the full record): the
mode *is* auto-enabled (log line above; it goes away with
`--no-enable-prefix-caching`) and it *does* change chunking — the
hybrid KV manager sets the attention block size to 784/800 tokens so
the attention page matches the mamba page, and
`Scheduler._mamba_block_aligned_split`
(`vllm/v1/core/sched/scheduler.py:366`) forces prefill chunk ends to
block-aligned positions. But it only ever *clips*:

```python
aligned_end = end // block_size * block_size
if aligned_end > start or block_size <= max_prefill_tokens:
    end = aligned_end          # end can only move backwards
...
end = min((s for s in stops if start < s < end), default=end)
return max(end - start, 0)
```

With block 800 and cap 1024, chunks become 800 — *smaller*, never a
multiple of the block size larger than the cap. Empirically the
align-off arm OOMed identically, so align is a co-passenger (it
changes chunk count and reserves per-block mamba state slots in the
pool), not the cause.

## 6. Arm matrix (this session, 2026-08-23 UTC)

| # | TP | util | maxlen | chunk | MTP | prefix cache (align) | result |
|---|---|---|---|---|---|---|---|
| W4 A/B | 1 | 0.93 | 4096 / 2560 | 8192 | – | on | OOM 356 MB inductor buf @ warmup |
| W4 A/B/soak | 1 | 0.90 | 1536 | 8192 | off | on | **PASS**, 30-rep soak flat |
| 1 | 2 | 0.90 | 262144 | 8192 | on | on | OOM in `gptq_gemm`, mid 250k prefill |
| 2 | 2 | 0.90 | 262144 | 4096 | on | on | OOM `_q_pad_buf` 186 MB, `free: 0` |
| 3 | 2 | 0.82 | 262144 | 2048 | on | on | OOM 170 MB gptq scratch |
| 4 | 2 | 0.82 | 262144 | 1024 | on | on | OOM 170 MB gptq scratch (12:13:20) |
| 5 | 2 | 0.82 | 262144 | 1024 | **off** | on | OOM 170 MB gptq scratch (12:45:18) |
| 6 | 2 | 0.82 | 262144 | 1024 | off | **off** (align off) | OOM 170 MB gptq scratch (13:06:18) |

## 7. Fix directions

1. **Identify the long-context transient** (the open item): worker
   memory snapshot at OOM time (`torch.cuda.memory._record_memory_history()`
   injected before the failing prefill, or `VLLM`'s profiler envs) and
   attribute the ~1–2 GiB.
2. **Honest profiling**: the KV pool is sized from a short-context
   profiled peak; profiling a long-context prefill shape would shrink
   the pool and make `H` reflect reality — but it does not create
   headroom, it reallocates it. Any real fix must *reduce* the
   runtime peak (item 1) or *pre-reserve* the big transients (item 3).
3. **AWQ scratch**: making `temp_dq` a persistent per-shape buffer
   removes the fragmentation risk but permanently reserves up to 2.37
   GiB (lm_head) — net-negative unless combined with 1/2. The
   lm_head being quantized at all is worth a separate look (fp16
   lm_head costs ~2.4 GiB of weights but removes the 2.37 GiB
   per-forward scratch and the 170 MB ones are unchanged).

## 8. Artifacts

- Logs still on disk: `/tmp/tp2_27b_serve.log` (run 4, MTP on —
  accounting + OOM verbatim in §1/§3), `/tmp/tp2_27b_nomtp.log`
  (run 5), `/tmp/tp2_27b_nopc.log` (run 6; also contains the 12:48
  `hipErrorLaunchFailure` boot attempt and the ~12:58 shm-broadcast
  init hang that self-recovered — see `degradation.md`).
- Launch scripts: `/tmp/launch_tp2_27b.sh` (+ `_nomtp`, `_npc`),
  needle: `/tmp/needle_256k.py` (and `/tmp/needle_100k.py`, never run —
  the server dies with each OOM).
- Inductor graph with the failing call: §2 path (cache dir is
  content-hashed; re-derive from `~/.cache/vllm/torch_compile_cache`
  after any recompile).
- Cross-refs: `degradation_details.md` § 2026-08-23 OOM cluster
  (summary), `README.md` § Known issues (pointer),
  `DEVLOG-tp2-dense.md` (131k/256k dense context records).

## 9. Resolution — the transient was `_gather_retired` (2026-08-24)

The "unidentified ~1–2 GiB long-context transient" (§6/§7) was the
custom-FA gather-buffer keep-alive dict. `GFX906_OOMHUNT_LOG` probe on
the 250k run-4 config (pre-fix policy via `GFX906_FA_GATHER_EXACT=1`):

- pre-fix, every chunked-prefill chunk with a larger max-seq-len
  reallocated the gather buffers at exact Sk and retired the previous
  (sticky-capture-latched) generation;
- retired dict: 1.19 GB @13.6k → 2.45 GB @28.8k → 4.62 GB @44k →
  **7.79 GB (152 generations) @60k tokens — OOM** at
  `178,257,920 bytes (free: 0, total: 34,342,961,152)` in
  `torch.ops._C.gptq_gemm`, 3.3 min into prefill (run-4 band 2.5–4.5
  min). Reproduced byte-exact on two independent boots.

So the dict — not the token-independent AWQ `temp_dq` — drained the
~1.94 GiB headroom; the 178 MB dequant scratch was the allocation that
landed on the remains. Fix: capacity-width grow-only gather buffers +
per-generation capture flag (branch `gfx906/fa-gather-lifecycle`,
`090673ad21`; design + reviews in `plan-gfx906-fa-fix.md` +
`gfx906-fa-fix-code-review-*.md`). **Validated on the §1 situation
itself:** pre-fix policy OOMs byte-exact (×2); fixed policy completes
the 250k prefill (148 tok/s incl. prefill) with the needle at token
125k retrieved, `retired_B=0` throughout; decode A/B flat (35B MoE
65.92 vs 66.13 t/s). **The 131k ceiling (§7 era) is lifted** on
Qwen3.8-27B; 256k TP=2 works at the run-4 config. Full record:
`DEVLOG-fa-attention.md` (Gather-buffer lifecycle fix, 2026-08-24).
