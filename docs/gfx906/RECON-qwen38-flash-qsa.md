# QSA-FN recon — Qwen3.8-Flash-Next (`qwen4_exp` QSA) on gfx906

> 2026-09-17, branch `gfx906/qsa-fn` (cut from `gfx906/v0.29.0`). Roadmap
> [`QSA-FN-*`](ROADMAP.md). Patches inspected: `../qsa-cdna2-vllm-patches`
> (`github.com/davetha/qsa-cdna2-vllm-patches`, 4 commits, head `bf1e73b`,
> developed on MI210 / gfx90a). Triton in the venv: **stock v3.8.0 (`GCN5_1`)**
> — the TRITON-1 adoption of 2026-09-15.
>
> All numbers below are **launch-regime evidence** (standalone kernel probes on
> one MI50, `HIP_VISIBLE_DEVICES=0`), not serving numbers: the model cannot be
> loaded on this box (QSA-FN-3), so no serving gate exists yet. Probes and logs:
> `/local/tmp/qsaprobe/` (survives reboot), `…/scratch` = the CDNA patch set
> applied to a copy of our `amd/` files.

**Question.** A tester reports gfx906 fails with
`NotImplementedError: Qwen4Exp QSA currently requires BF16`. The CDNA2 patch set
touches the same two files (`amd/qsa.py`, `amd/ops/qsa.py`) and makes QSA run with
an int8 KV cache and faster kernels. Is that patch set the fix, and does any of it
port to gfx906 (no MFMA, no native bf16, 60 CU of `v_dot2`/`v_dot4`)?

**Answer, in two parts.**

1. **The reported error is not what the patch set fixes.** It is the gfx906
   bf16→fp16 auto-fallback (`Platform.supports_native_bf16=False` on gfx906,
   `vllm/config/model.py:2294`) tripping QSA's hard bf16-only guards; the CDNA
   patches *keep* the bf16 requirement (they only reword its message, §3). The
   fix is a **gfx906 fp16 enablement** of the QSA path — which is also by far the
   largest measured win available here: **3.9×** on the QSA kernel pair vs bf16
   (§5.2), because gfx906 emulates every bf16 `tl.dot` as scalar fp32 FMA.
2. **Of the CDNA patch set's three changes, one ports as-is.** The **tiled
   indexer** ports, measured **+39 %** kernel (fp16 only — it must be gated, bf16
   costs −58 %). **`int8_per_token_head` KV** ports but buys **capacity only** on
   gfx906: attention prefill costs **2.5–2.7×** for the same shapes while decode
   is neutral (§5.3). **int8-QK does not port**: it is slower than the fp16
   baseline at every shape it survives, and it faults (IMA) at the exact dispatch
   profile vLLM picks for any real prefill (§5.4, §6).

Neither half can be gated end-to-end yet: the model is ~120 B params of MoE
(§2), so a **tiny-config harness** (QSA-FN-3) has to exist before anything
gfx906-side is judged on serving wall-clock.

## 1. What fails, exactly

The checkpoint is `bfloat16` (`Qwen/Qwen3.8-Flash-Next` `text_config.dtype`).
On gfx906 that is silently rewritten to fp16 at startup:

```
vllm/platforms/rocm.py:645   supports_native_bf16 -> not _ON_GFX906   (False)
vllm/config/model.py:2294    "Checkpoint dtype is bfloat16, but this device has no
                              native bfloat16 support; auto-selecting float16"
```

`Qwen4ExpQSAAttention.__init__` then sees `model_config.dtype == float16` and
raises `Qwen4Exp QSA currently requires BF16` (`amd/qsa.py:188`). The indexer has
its own copy of the same guard (`amd/indexer_qsa.py:95`), so the message can come
from either.

This is not a QSA bug and not a CDNA-patch issue: it is the documented gfx906
policy (bf16 checkpoints run as fp16) meeting a subsystem that upstream only ever
ran in bf16. **Every bf16-only site on the QSA path is a gfx906 blocker**, and they
are all in `vllm/models/qwen4_exp/`: See
[`DEVLOG-qwen38-flash-qsa.md`](DEVLOG-qwen38-flash-qsa.md) for what was changed
(QSA-FN-1, landed 2026-09-17) — the table below is the pre-edit inventory.

| file | line | guard |
|---|---|---|
| `amd/qsa.py` | 70 | `supported_dtypes = [torch.bfloat16]` (backend) |
| | 71 | `supported_kv_cache_dtypes = ["auto","bfloat16"]` |
| | 113 | Impl: `kv_cache_dtype not in ("auto","bfloat16")` |
| | 151 | `forward_qsa`: `key_cache.dtype != bf16 or query.dtype != bf16` |
| | **188** | **`model_config.dtype != bf16` — the reported error** |
| | 190 | `cache_config.cache_dtype not in ("auto","bfloat16")` |
| | 272 | `kv_cache_torch_dtype != bf16` (cache storage) |
| `amd/indexer_qsa.py` | **95** | `model_config.dtype != bf16` — the second copy |
| | 130, 139 | `raw_key_cache` / `compressed_key_cache` `dtype=torch.bfloat16` |
| `common/qsa_cache.py` | 662, 663 | `QSAStateBackend.supported_dtypes` / `…_kv_cache_dtypes` |
| | 725 | `bind_kv_cache`: `kv_cache.dtype != bf16` |
| `amd/ops/qsa.py` | 849 | `assert q.dtype == k_cache.dtype == v_cache.dtype == bf16` |
| `amd/model.py`, `amd/mtp.py` | 260, 436, 229 | `HyperConnectionConfig(params_dtype=torch.bfloat16)` — the NVIDIA variant passes the model dtype here; AMD hardcodes bf16 |

`common/qsa_cache.py` is **shared with the NVIDIA path**, so its edits must be
dtype-*general* (`self.dtype` instead of a bf16 literal), never gfx906-conditional,
or the CUDA path is put at risk for no reason. `_BF16_PER_INT64 = 4` (`:736`) is
the int64-MRoPE packing width — 4 × 2 B = 8 B, i.e. correct for fp16 as-is; it only
needs to become dtype-derived if a 1-byte cache dtype is ever added.

## 2. What the model is (why there is no local end-to-end gate)

`Qwen/Qwen3.8-Flash-Next`: 48 layers (36 linear-attention + 12 full-attention),
hidden 2560, **E=512 / topk=10**, `moe_intermediate_size` 640, head_dim 256,
24 Q / 2 KV heads, 1 MTP layer, `hc_count=4`, partial RoPE 0.25 (no YaRN, native
256 K). W4A16 puts the weights alone at ~60 GB, and the PLE ngram table is large
enough that the CDNA recipe ran `--cpu-offload-gb 60 --cpu-offload-params
ngram_embedding` on **2× MI210 (64 GB)**. It does not load in 2× MI50.

The CDNA launcher recipe, mapped onto our tree (all flags exist here except the
int8 KV, which is §5.3):

| CDNA prod launch | gfx906 delta |
|---|---|
| `--dtype bfloat16` | fp16 (auto-fallback) — QSA-FN-1/FN-2 |
| `--kv-cache-dtype int8_per_token_head` | not ported; capacity-only on gfx906 (§5.3) |
| `--mamba-cache-dtype bfloat16` | drop / auto (gfx906 stack is fp16) |
| `--speculative-config '{"method":"qwen4_exp_mtp",…}'` | fine — normalized to `mtp` (`config/speculative.py:1090`); our documented `{"method":"mtp"}` form is equivalent |
| `--tool-call-parser qwen3_xml`, `--reasoning-parser qwen3` | both present |
| `--kv-cache-memory`, `--block-size 64`, `--max-num-seqs 4`, `--max-num-batched-tokens 4096`, `--enable-expert-parallel` | present |

## 3. Does the CDNA patch set apply here?

Yes, both patches apply **cleanly and unmodified** (`patch -p0 --dry-run` prints
only `checking file`, i.e. every hunk matches at zero offset):

```
scripts/…  cd /tmp && cp $REPO/vllm/models/qwen4_exp/amd/{qsa.py,} …   # see §9
patch -p0 --dry-run < ../qsa-cdna2-vllm-patches/patches/qsa.py.patch       # clean
patch -p0 --dry-run < ../qsa-cdna2-vllm-patches/patches/ops_qsa.py.patch   # clean
```

That is itself a finding: **our `amd/` files are the CDNA author's base revision**
(same line numbers, same hunks), so there is no adaptation layer to write — the
port is a review of the diff, not a re-derivation. The one fragile spot is
deliberate: `Qwen4ExpQSAFlashAttentionImpl.__init__` bypasses
`FlashAttentionImpl`'s quantized-KV rejection by feeding the parent `"auto"` and
restoring the real dtype afterwards, with `_POS = 6` hard-coding the positional
slot of `kv_cache_dtype`. We call the parent with 10 positional args
(`amd/qsa.py:283`) and `kv_cache_dtype` is indeed index 6
(`backends/flash_attn.py:836`), so the hack is correct *for this revision* — flag
it in review as version-brittle, not as wrong.

What the three changes are, and what survives gfx906:

| change | CDNA claim (MI210) | gfx906 verdict |
|---|---|---|
| tiled row-tiled indexer (`_qsa_mqa_paged_tiled_kernel`, `BLOCK_M=16`, `tl.dot` scoring, uniform-request prefill only) | 6.57× kernel | **ports — 1.39× in fp16, must be dtype-gated** (§5.2) |
| `int8_per_token_head` KV (`customize_spec` + scale views + inline dequant in the read kernels) | 1.67× capacity, prefill faster with int8-QK | **ports as capacity-only: 2.5–2.7× prefill cost, decode-neutral** (§5.3) |
| int8-QK (`tl.dot(q_i8, keys_i8) -> int32`, `QSA_INT8_QK=1`) | 30 K prefill 30.8 s → 18.6 s | **does not port: faults at the production dispatch profile, and is not faster where it runs** (§5.4, §6) |

## 4. gfx906 ISA facts (measured, this session)

Compiled with the venv's Triton (stock v3.8.0 `GCN5_1`), dumped from
`kernel.asm['amdgcn']`:

| `tl.dot` operand dtype | lowering on gfx906 | instruction count for a 16×16×16 dot |
|---|---|---|
| fp16 × fp16 → fp32 | `v_dot2_f32_f16` | 32 |
| **bf16 × bf16 → fp32** | **scalar `v_fmac_f32` (upcast, no bf16 instruction)** | 60 `v_fmac_f32` + 4 `v_fma_f32` |
| int8 × int8 → int32 | `v_dot4_i32_i8` (+`v_perm_b32` packing) | 16 |

So the CDNA design maps onto gfx906 *at the instruction level* (Triton does emit
VOP3P dots for both dtypes — `supportsVDot` is already on for `GCN5_1`), and bf16
is the worst of the three on this chip. That is the whole reason the fp16
enablement is worth more than everything else in this file. No `v_mfma` is
emitted anywhere (`getMfmaVersion` = 0), so the CDNA kernels' *tiling* ideas
transfer but their matrix-core rationale does not.

## 5. Measured kernels

### 5.1 Indexer (`_qsa_mqa_paged_kernel` vs the new tiled kernel)

`probe_indexer.py` — L=30720, CL=7680, NROWS=2048, NH=4, HD=128, CR=4,
PAGE=64, BUDGET=2048, `num_warps=4` (the CDNA author's own test shapes):

| dtype | per-row kernel | tiled kernel | speedup | tiled vs bf16 per-row |
|---|---|---|---|---|
| **fp16** | 5426 µs | **3902 µs** | **1.39×** | 0.56× |
| bf16 | 6928 µs | 16648 µs | **0.42×** | 2.40× |

Quality (the reason this change is safe to take): logits NRMSE 1.3e-7 (fp16),
`-inf` mask agreement 1.0000, visible-blocks 1.0000, **top-2048 membership
agreement 1.00000**. Identical selection, so it cannot change output quality.

The tiled kernel is a win in fp16 and a catastrophe in bf16 (the tiled `tl.dot`
loses the vector-ALU `tl.sum` path and pays bf16 emulation *per tile*). Gate the
route on the operand dtype, not just on `q.shape[0] >= 64`.

### 5.2 Sparse attention, fp16 vs bf16 cache (the headline)

`probe_attn5.py` — T=1024, G=12, NKV=1, HS=256, TOPK=2048, L=16384, PAGE=64,
default dispatch profile, arms interleaved, 3 reps:

| K/V cache dtype | kernel | vs fp16 |
|---|---|---|
| fp16 | **26.5 ms** | 1.00× |
| bf16 | 116.5 ms | **4.39×** |

Rep-stable to ±0.2 %. Combined with the indexer at the same scale (fp16 5.4 ms +
26.5 ms vs bf16 6.9 ms + 116.5 ms), the QSA kernel pair is **~3.9× cheaper in
fp16 than in bf16** on gfx906. With the tiled indexer gated to fp16 the pair is
30.4 ms. This is the single most valuable item in this document, and it needs no
kernel writing at all — only the guard edits in §1.

### 5.3 int8 KV cache: capacity only on gfx906

`probe_decode.py` — fp16 cache vs `int8_per_token_head` cache with the patch's
inline dequant, identical shapes (L=65536, NBLK=1024, TOPK=2048, G=12, HS=256):

| query rows T | fp16 cache | int8 + dequant | ratio |
|---|---|---|---|
| 1 (decode) | 100.7 µs | 100.3 µs | **0.99×** |
| 4 | 115.4 | 165.5 | 1.43× |
| 16 | 322.4 | 546.1 | 1.69× |
| 64 | 2064.7 | 5163.6 | 2.50× |
| 256 | 7441.2 | 18728.5 | 2.52× |
| 513 | 14102.5 | 38223.4 | 2.71× |
| 1024 | 26537.5 | 71187.6 | 2.68× |

Accuracy is as documented by CDNA: NRMSE 0.0081 against the fp16 cache (they
report 0.0070 roundtrip + 0.0006 for int8-QK), so the *quality* risk is the known
~1 % attention-value one, not something new.

Mechanism check (three variants, same shapes): moving the dequant to fp16 math
(`keys.to(fp16) * k_scale.to(fp16)`) changes nothing (2.68×), and **removing the
scale multiply entirely still leaves 2.68×**. So the cost is not the dequant
arithmetic but **loading int8 and converting it into a `v_dot2`-legal operand
(register packing / layout) per tile**, which the fp16 path gets for free from the
load. This is a Triton-codegen property on gfx906, not something a smarter scale
kernel fixes — and it is what makes int8 KV a *capacity-only* feature here: 1.67×
KV tokens bought with ~2.5× on the attention kernel at prefill, ~0 at decode.

(The same trick is already costed elsewhere in this hub for the custom FA path —
`int8-investigation-qwen.md` T2 / ROADMAP SYV-11, "1.88× capacity, ~20 % decode
cost on their box". This session adds the QSA-path measurement: no decode cost,
large prefill cost.)

### 5.4 int8-QK: no win, and it faults

`probe_dispatch.py` (default dispatch, L=65536), int8-QK vs the fp16 baseline for
the same shapes:

| T | fp16 | int8-QK | ratio |
|---|---|---|---|
| 1 | 100.7 | 103.5 | 1.03× |
| 4 | 117.7 | 150.6 | 1.28× |
| 16 | 325.3 | 430.2 | 1.32× |
| 64 | 2068.8 | 4987.9 | 2.41× |
| 256 | 7500.4 | 17943.3 | 2.39× |
| 513 | 14080.0 | **IMA** | — |
| 1024 (L=16384, BN=16/W=4) | 26408 | 22185 | 0.84× |

The one sub-1× point is a profile where the dequant arm is *also* slow; at the
profile vLLM actually selects for prefill it is 2.4× *slower* than just using an
fp16 cache. NRMSE(int8-QK, int8-dequant) = 0.0037, i.e. the arithmetic itself is
fine — the value is not there.

## 6. The int8-QK IMA (gfx906-specific, open)

The int8-QK branch faults in `_qsa_sparse_paged_gqa_splitk_kernel`:

```
env P_GROUP=12 P_T=1024 P_TOPK=2048 P_L=65536 P_ONLYQK=1 HIP_VISIBLE_DEVICES=0 \
    .venv/bin/python /local/tmp/qsaprobe/probe_attn2.py
  → Memory Fault Error … kernel: _qsa_sparse_paged_gqa_splitk_kernel
    (log: /local/tmp/qsaprobe/logs/ima_repro_G12.log)

same shapes with QSA_FORCE_BN=16 QSA_FORCE_WARPS=4  →  int8-qk OK True
```

The CDNA author's own test (`tests/test_int8qk_attn.py`, HS=256/G=8/T=16/L=256)
**passes** on gfx906, and G=8 passes at every size — so this is not "the port is
wired wrong", it is a shape/profile-dependent failure of the int8 path. The
dispatch profile matters, not just the tile: the wrapper's
`block_n / target_splits / partial_warps` table is explicitly "Tuned on GB300"
(`amd/ops/qsa.py`, "Tuned on GB300 for the Qwen-Air TP1, TP2, and TP4 attention
shapes"), and for `base_programs = rows × kv_heads > 512` — i.e. **every prefill
chunk over 512 tokens, which is every real prefill** — it picks
`block_n=64, num_splits=1, num_warps=2`, which is one of the failing profiles.

(rows, `num_warps`) matrix at `block_n=64, num_splits=1`, T=1024/TOPK=2048,
BLOCK_M = `next_power_of_2(group_size)`, forced via the probe's `QSA_FORCE_BM`:

| BLOCK_M | w=2 | w=4 | w=8 |
|---|---|---|---|
| 4 | IMA | IMA | OK |
| 8 | OK | OK | OK |
| 16 | IMA | OK | OK |
| 32 | IMA | IMA | OK |

`num_warps=8` clears every combination, as does `BLOCK_M=8`; nothing else does.
This is register-pressure-shaped (the signature of a miscompile), but it is **not
root-caused**: I could not reduce it to a minimal standalone `tl.dot` repro — a
stripped loop kernel with the same (M=16, D=256, N=64, loop=32, w=2) shape, a live
fp32 accumulator and a per-iteration int8 load computes *correctly*
(maxdiff 1e-6 vs an int64 reference) and does not fault on repeated runs, and one
earlier variant faulted once and then passed on re-runs (address-dependent or
nondeterministic in that form). So: a gfx906 int8-`tl.dot` codegen hazard that
needs this kernel's register pressure to trigger. Candidate mechanisms to check
first: a wild/near-bound LDS or VGPR-pair packing for `v_dot4_i32_i8` operands
under spills; the AMD-side `supportsVDot` path in Triton's int8 dot lowering.
Two acceptable resolutions, and the cheap one is chosen (QSA-FN-6).

## 7. Gates that exist locally

* **`tests/models/qwen4_exp/test_qsa_amd.py` — 9 passed on MI50, this session**
  (bf16 path, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HIP_VISIBLE_DEVICES=0`).
  It contains a CPU reference for `qsa_sparse_paged_attention` and parametrizes
  TP1/TP2/TP4 shapes — this is the ready-made fp16 gate (add an fp16
  parametrization) *and* the regression gate for the existing bf16 path.
* `test_qsa_reference.py` (16), `test_ple.py` (10), `test_config.py` (7) pass
  individually.
* Pre-existing quirk: collecting the whole `tests/models/qwen4_exp/` directory at
  once dies with a duplicate custom-op registration error (`RuntimeError: Tried to
  register…`), because the `amd` and `nvidia` modules register the same op names.
  Run the files one at a time; not caused by and not fixed by this work.
* The hub's PPL probe (`benchmarks/kernels/gfx906/ppl_probe.py`) cannot gate this
  model — it needs a loadable checkpoint (hence QSA-FN-3).

## 8. What cannot be measured locally, and what that costs the verdict

Everything in §5 is a kernel probe on synthetic tensors. The two things a kernel
probe cannot tell us:

* **The attention kernel's share of prefill wall-time on this model.** That single
  number decides whether the int8-KV prefill cost (2.5×, §5.3) is affordable for
  the 1.67× capacity. On MI210 the CDNA author measured QSA sparse attention at
  57.7 % of prefill — if that transfers, int8 KV is unattractive on gfx906 even
  before int8-QK is considered. **Do not port int8 KV for QSA before measuring it**
  (QSA-FN-5).
* **Whether the fp16 QSA path is numerically acceptable end to end.** Region-level
  accuracy is measured (§5.3-style NRMSE), model-level is not. The gfx906 PPL band
  is ±2 % and this model has no local checkpoint.

## 9. Recommended order

1. **QSA-FN-1 (fp16 enablement)** — unblocks the reported error, and is the 3.9×
   win. Editing list is §1; gate is an fp16 parametrization of
   `test_qsa_amd.py` plus the shared-cache dtype tests, with the bf16 arm kept
   green (no CUDA/NVIDIA path may change: `common/qsa_cache.py` edits stay
   dtype-general). **DONE 2026-09-17** —
   [`DEVLOG-qwen38-flash-qsa.md`](DEVLOG-qwen38-flash-qsa.md).
2. **QSA-FN-3 (tiny-config harness)** — the enabler for every end-to-end gate
   after this point, and cheap: a `--load-format dummy` + `--hf-overrides`
   qwen4_exp config (4 layers, E=8, hidden ~512, PLE present but tiny) so the
   indexer/PLE/QSA paths all execute on one MI50 in fp16.
3. **QSA-FN-4 (tiled indexer, fp16-gated)** — 1.39×, quality-identical, already
   written and cleanly applicable.
4. **QSA-FN-2 (model-level fp16 sweep + tester launch recipe)** — the remaining
   bf16 literals outside the QSA files (`HyperConnectionConfig` ×3, the
   `--mamba-cache-dtype` in the recipe) and what the tester must run.
5. **QSA-FN-5 / QSA-FN-6 (int8 families)** — evidence-first, both default OFF;
   the measured case for either on gfx906 is weak (capacity-only at 2.5× prefill
   cost; int8-QK slower and unsafe).

Repro scratch (this session, all under `/local/tmp/qsaprobe/`); the QSA-dependent
probes move into `benchmarks/kernels/gfx906/` when QSA-FN-4 lands the patched
`ops/qsa.py` in-tree (they need it):

```
dot_probe.py          # §4 ISA scan (fp16/bf16/int8 tl.dot → asm)
probe_indexer.py      # §5.1 (uses scratch/ops/qsa.py)
probe_attn5.py        # §5.2 fp16-vs-bf16 sparse attention
probe_decode.py       # §5.3 int8-dequant sweep vs T
probe_dispatch.py     # §5.4 int8-QK vs fp16 at the real dispatch
probe_attn2.py        # §6 IMA repro (P_GROUP/P_T/P_TOPK/P_L/P_ONLYQK, QSA_FORCE_*)
logs/                 # ima_repro_G12.log, probe_*.log
```
