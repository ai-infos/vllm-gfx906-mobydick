# Handover — DFlash2 pairing test on the other (NVIDIA) box

**This is the in-repo copy; the live copy the other agent reads is
`/data/docs/dflash2-handover.md` (shared mount, identical text).** It is a work order, not a dev
log: our own measurements and the open verdict live in `DEVLOG-dflash2.md`, and the queue items in
`ROADMAP.md` (DFL2-*, INT8-PACKED-1). Written 2026-09-16.

## 1. The question we need answered

We measured DFlash2 speculative decoding on gfx906 (2x MI50, our vLLM fork, 0.29 base) and got
a **pathologically low acceptance**: 2.48/2.52 t/s @64k and 1.37/1.37 t/s @120k, with
**per-position acceptance 0.041 / 0.0625 @64k and exactly 0.0 @120k, and pos1+ = 0** (an MTP
control run in the same boot gave 35.97/24.58 t/s, acceptance 1.78-2.11, pos0 ~0.85).

A drafter whose *first* token is essentially never right is mis-conditioned or mis-mapped, not
merely badly tuned. We have ruled out (details in §7): the target-side FA, the DFlash2 PR itself
(upstream vLLM#52816 is already in our tree), the mamba `drop_eagle_block` bug (inert with prefix
caching off, which is how we run), and a GDN async-scheduling race (that one crashes, it doesn't
lower acceptance).

**Two candidates remain**, and your box discriminates them:

1. **Drafter/target quantisation pairing.** Drafters ship *matched to the target's quantisation*
   — `...-DFlash2` for bf16 targets, `...-DFlash2-W8` for W8A16 targets. Our pair was the **bf16**
   drafter against an **AWQ-INT4** target, and the drafter consumes the target's hidden states
   (layers `[5, 19, 33, 47, 61]`), so a mismatch can plausibly collapse acceptance.
2. **The drafter's attention path.** All 5 drafter layers are `sliding_attention` with
   `sliding_window: 2048` and **`is_causal: false`** (windowed-bidirectional). On our box it is
   rejected by our custom FA and runs an upstream backend plus a Triton fallback. A wrongly
   masked / unwindowed drafter would also explain our per-step cost growing with context
   (422 ms @64k -> 728 ms @120k).

**Decisive test: run the matched pairs on your stack and report per-position acceptance.**
Normal acceptance (~2-3) => the pairing was the cause. Degenerate (~0) => the drafter's
attention/mechanism is the problem and we stop investing in this family.

## 2. Artifacts — everything is already on the shared `/data` mount

**The pair we actually ran (use this to reproduce our numbers)** — both already on `/data`, no
downloads needed:

| role | path on `/data` | revision |
|---|---|---|
| 4-bit target (our production quant) | `/data/cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea` | `63768c10df38c0395e12ef49edac1bd539eaeeea` |
| W4A16 drafter (**the model syv-ai publishes**, GPTQ of `incoai/Qwen3.8-27B-DFlash2`) | `/data/cache/huggingface/hub/models--syvai--Qwen3.8-27B-DFlash2-W4A16/snapshots/4d30ec736ffc6b8688dc2ae2b5...` | 1.19 GB, 4 files (`config.json`, `model.safetensors`) |

This is the pairing syv-ai themselves measure at **3.14 / 3.34 tokens per step** on a 3090
(their target is `Qwen3.8-27B-Uncensored-W4A16`, a 4-bit requant, with their patch for the quantized
`lm_head`; ours stores `lm_head` unquantized, so that patch does not apply to us).

**What we measured with exactly these two models on our fork** (TP=2, `--enforce-eager`, 64k agentic,
k=7, 1 rep), and why the drafter is *not* the variable:

| arm | accepted/draft | drafts | t/s |
|---|---|---|---|
| backend fallback (our FA refuses the drafter -> ROCM_ATTN + Triton) | 0.045 | — | 2.48 |
| eager symmetric-window (non-causal) attention in torch | **0.0282** | 248 | 2.52 |

Against syv-ai's 3.1-3.6 tokens/step, i.e. ~1 token/step. The **matched** drafter did not change
anything versus the bf16 one, and two independent attention implementations agree — so neither the
quantisation pairing nor the drafter's masking is the cause on our fork. See `DEVLOG-dflash2.md`
and `DEVLOG-fa-noncausal.md`.


Both repos below are complete and verified (target: 6 shards, 1986 tensors, 29.53 GB = the index's
`total_size`; drafter: 6 files, 2.2 GB). Revisions, so you load byte-identical files:

| role | path on `/data` | revision |
|---|---|---|
| INT8 W8A16 target (second arm; see the table above for what we ran) | `/data/cache/huggingface/hub/models--lued--Qwen3.8-27B-INT8-W8A16-DFlash2/snapshots/2971c64ba386dd3faa6884cc215f67b3b2477a3e` | `2971c64ba386dd3faa6884cc215f67b3b2477a3e` |
| W8 drafter for it | `/data/cache/huggingface/hub/models--lued--Qwen3.8-27B-DFlash2-W8/snapshots/f454fa8e6a84387bf006f849584f72541cc29118` | `f454fa8e6a84387bf006f849584f72541cc29118` |
| bf16 target (card's own pairing) | `/data/cache/huggingface/hub/models--Qwen--Qwen3.8-27B` | (already present) |
| bf16 drafter for it | **not on /data** — `incoai/Qwen3.8-27B-DFlash2` (3.6 GB, pull it if you want the card's exact pair) | — |

Notes:
- The INT8 target is a **VL** checkpoint (`Qwen3_5ForConditionalGeneration`, keys nested under
  `model.language_model.*`, plus `model.visual.*`) and uses compressed-tensors
  **`pack-quantized`** (`weight_packed`/`weight_scale`/`weight_shape`). A recent vLLM nightly
  handles it. Our fork could not until 2026-09-16 — the failure was model wiring, not quantisation
  support (`ValueError: no module or parameter named 'embed_tokens.weight_packed'`) — and that is
  **fixed on `main`** now: the checkpoint loads with zero skipped tensors and generates coherent
  output (`DEVLOG-int8-packed.md`). If it fails to load on your stack, that is a different bug.
- Sanity-check what you actually loaded: `ls -l <snapshot>/model.safetensors.index.json` and
  compare the revision against the table above.

## 3. Serve command (your box — Kevin's recipe, unchanged)

```bash
hf download lued/Qwen3.8-27B-INT8-W8A16-DFlash2   # already on /data, will resolve instantly
hf download lued/Qwen3.8-27B-DFlash2-W8

git clone https://github.com/noonghunna/club-3090.git   # for the vendored PR patches, if needed
export CLUB3090="$HOME/club-3090"

podman run --rm --replace --device nvidia.com/gpu=all --ipc=host -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$CLUB3090/models/qwen3.6-27b/vllm/patches/vllm-pr52816-dflash2":/etc/club3090/pr52816:ro \
  -v "$CLUB3090/models/qwen3.6-27b/vllm/patches/vllm-pr48375-mamba-drop-eagle-block":/etc/club3090/pr48375:ro \
  --entrypoint bash docker.io/vllm/vllm-openai:nightly-5a4c8d99242e9e069b604d0e9b969e77f7dd501d \
  -c 'bash /etc/club3090/pr48375/install.sh || exit 1; bash /etc/club3090/pr52816/install.sh || exit 1; exec vllm serve "$@"' -- \
  lued/Qwen3.8-27B-INT8-W8A16-DFlash2 \
  --tensor-parallel-size 2 --max-model-len 262144 --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"dflash","model":"lued/Qwen3.8-27B-DFlash2-W8","num_speculative_tokens":7}'
```

- **Check whether you need those two patches at all**: upstream PR #52816 ("DFlash2: local
  convolution + candidate selector") **merged 2026-08-21** and the current club-3090 head no
  longer ships a `vllm-pr52816-dflash2` directory (we cloned it today and it isn't there) — evidence
  the pinned nightly may already contain it. If the mount fails, grep the image for
  `qwen3_dflash2.py` / `dflash2/speculator.py` and skip that patch. PR #48375 stays unmerged.
- Add the Qwen parsers if you want tool-calling/reasoning, and **always prompt through the chat
  template** (see §6).
- If a single GPU has enough memory, TP=1 is fine; TP=2 as above is what Kevin used.

### The command we ran (for reproducing our numbers)

```bash
# on our box: /local is fast, /data is NFS; the drafter lives on /data, the target has both copies
vllm serve <AWQ-INT4 target dir> \
  --served-model-name qwen27 --dtype float16 --tensor-parallel-size 2 \
  --max-model-len 131072 --max-num-seqs 2 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype float16 --no-enable-prefix-caching --enforce-eager \
  --speculative-config '{"method":"dflash","model":"<syvai W4A16 drafter dir>","num_speculative_tokens":7}'
```

- `--kv-cache-dtype float16` is *our* spelling; their card uses `--kv-cache-dtype bfloat16` (and
  `fp8_e4m3` in Kevin's recipe). Note `vllm serve` rejects `fp16` and dies in argparse *before any
  logger exists* — a silent-looking empty log.
- **A *quantized* drafter needs a patch on vLLM 0.29.0** (we hit it as `'QKVParallelLinear' object has
  no attribute 'weight'`, and the 5070 Ti box hit the same): `qwen3_dflash.py` builds the fused
  context-KV weight from `qkv_proj.weight`, which a pack-quantized layer does not have. syv-ai's
  `_dense_kv_rows` (in their `dflash2-lookup-drafting.patch`) dequantizes from
  `weight_packed`/`weight_scale` and derives the shape from the tensors, because vLLM's fused
  `weight_shape` keeps only the last shard's. A bf16 drafter loads without it; ours is on
  `gfx906/dflash2` (port validated: q_proj (4096,5120), k/v (1024,5120), `[q_size:]` -> (2048,5120)).
- `--enforce-eager` is needed **on our box only**, because the drafter's ROCM_ATTN fallback cannot be
  CUDA-graph captured (`Cannot copy between CPU and CUDA tensors during CUDA graph capture`, via
  `rocm_attn.py` -> `chunked_prefill_*`). With real FA it is unnecessary.
- Keep `--max-model-len` >= your prompt: a 65k prompt against an 8k/32k limit returns HTTP 400,
  which the sweep client reports as an incomplete run rather than an error.

## 4. Instrumentation — the number we need is per-position acceptance

Total acceptance alone is not enough (our own single-rep numbers were chaotic on this stack), so
log the **per-position** histogram. vLLM exposes it as a counter:

```bash
curl -s localhost:8080/metrics | grep -E 'spec_decode_(num_accepted_tokens|num_draft_tokens)' | grep -v '^#'
```

A compact client that reports what we compare on (adapt paths/ports as needed):

```python
import json, time, hashlib, urllib.request
import openai  # or plain HTTP against /v1/completions

def metrics(port=8080):
    txt = urllib.request.urlopen(f"http://localhost:{port}/metrics").read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith(("vllm:spec_decode_num_accepted_tokens_per_pos",
                            "vllm:spec_decode_num_draft_tokens",
                            "vllm:spec_decode_num_accepted_tokens")) and not line.startswith("#"):
            name = line.split("{")[0]
            label = line[line.find("}") + 1:].strip().split()[0] if "}" in line else line.split()[1]
            pos = line[line.find('position="') + 10: line.find('"', line.find('position="') + 10)] \
                  if 'position="' in line else "-"
            out.setdefault(name, {})[pos] = float(line.split()[-1])
    return out

def run(prompt, port=8080, max_tokens=256):
    m0, t0 = metrics(port), time.perf_counter()
    c = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="x")
    r = c.completions.create(model="<served-model-name>", prompt=prompt,
                             max_tokens=max_tokens, temperature=0, extra_body={"ignore_eos": True})
    dt = time.perf_counter() - t0
    m1 = metrics(port)
    drafted = m1["vllm:spec_decode_num_draft_tokens"]["-"] - m0["vllm:spec_decode_num_draft_tokens"]["-"]
    per_pos = {p: m1["vllm:spec_decode_num_accepted_tokens_per_pos"][p]
                  - m0["vllm:spec_decode_num_accepted_tokens_per_pos"].get(p, 0.0)
               for p in m1["vllm:spec_decode_num_accepted_tokens_per_pos"]}
    accepted = m1["vllm:spec_decode_num_accepted_tokens"]["-"] - m0["vllm:spec_decode_num_accepted_tokens"]["-"]
    ntok = len(r.usage.completion_tokens) if hasattr(r.usage, "__len__") else r.usage.completion_tokens
    return dict(prompt_sha1=hashlib.sha1(prompt.encode()).hexdigest()[:12],
                acceptance=accepted / max(drafted, 1), accepted=accepted, drafted=drafted,
                per_pos=per_pos, ms_per_step=dt / max(drafted, 1) * 1000,
                tps=ntok / dt)
```

Log **per request**: `prompt_sha1`, `accepted`, `drafted`, `per_pos` (the histogram), `ms_per_step`,
`tps`. The headline we need is `per_pos[0]` (first draft token) — ours was 0.04/0.06/0.0 versus MTP's
~0.85.

Also capture from the server log, for both the target and the drafter:
- `Found incompatible backend(s) [...] with AttentionType.<X>` / `Overriding with <BACKEND>` — which
  attention backend each model got. (On our box the drafter was rejected from our custom FA and ran
  `ROCM_ATTN` with a `Cannot use ROCm custom paged attention kernel, falling back to Triton
  implementation` line.)
- `GPU KV cache size: N tokens`, and any `Mean acceptance length` summary line per request.

## 5. Protocol (this box burns people who skip it — same rules apply on yours)

1. **Identical prompts across arms.** Never put the arm name in the prompt: a tokenized prefix shifts
   the body window and the arms then see *different* prompts. Log `prompt_sha1` and assert it is
   equal across arms before comparing anything.
2. **Interleave arms** (A, B, A) or repeat the first arm last. Acceptance and t/s here vary
   **per server process** by as much as the effects we chase (observed 2.19/1.81/1.76/1.71 on
   identical configs), so same-boot is *not* sufficient; same order-position is what's missing.
3. **≥2 reps per (arm, prompt), ≥10 distinct prompts** before believing an acceptance delta.
4. Lead with **ms/step** for same-configuration comparisons; for method-vs-method (e.g. DFlash2 k=7
   vs MTP k=3) report t/s at each method's recommended depth.
5. A throughput number is **not** a correctness gate. Before any of this, check the model actually
   answers: one greedy, templated request ("What is the capital of France?") with speculation off,
   then again with it on.
6. Teardown: SIGTERM the server and wait for it to drain before the next arm.

## 6. Prompt format — do not skip this

These are **instruction-tuned** checkpoints (the INT8 target is also VL). Raw text via
`/v1/completions` returns garbage that is *not* a bug. Use the model's chat template
(`/v1/chat/completions`, or `--chat-template "$(cat <snapshot>/chat_template.jinja)"`). We lost a
day to exactly this on another checkpoint: both builds produced identical garbage, which proves the
builds agree, not that the model works.

## 7. What we already excluded (so you don't re-test it)

- **Target-side attention**: excluded by the MTP control run in the same boot/session
  (35.97/24.58 t/s, acceptance 1.78-2.11 with the same target and our FA).
- **DFlash2 support being absent/incomplete in our tree**: PR #52816 is already merged upstream
  (2026-08-21) and present in our 0.29 tree (`74a6576b9b`).
- **vLLM#48375 (`drop_eagle_block` in MambaManager)**: its own note says it is inert unless prefix
  caching is enabled; all our arms run `--no-enable-prefix-caching`. Also relevant only to hybrid
  GDN + MTP/EAGLE + prefix caching.
- **club-3090's GDN+MTP async-spec-order patch**: a cross-stream race that manifests as a CUDA
  illegal memory access, not as low acceptance (and needs prefix caching on).
- **Draft-vocab mapping**: the bf16 drafter has `draft_vocab_size: None` (full 248,320 vocab), so
  there is no shortlist to mis-map.
- **The drafter/target quantisation pairing** (2026-09-16): the *matched* W4A16 drafter against the
  4-bit target gives the same degenerate result as the bf16 drafter against it (0.0282 vs 0.045
  accepted/draft), so "which drafter" is not the variable.
- **The drafter's attention masking**: an eager torch path implementing the reference semantics
  (no causal clip, symmetric ±2024 window — `_maybe_symmetrize_window`) also gives ~0
  (0.0282, 248 drafts, 2.52 t/s), and both implementations agree.
- **Remaining suspects** on our side, in order of cheapness: which target layers' hidden states the
  drafter actually receives (`combine_hidden_states` validates only the *width* 25600 = 5 x 5120, so a
  wrong-but-equal-count layer set would pass silently), and the selector/vocab machinery under this
  fork's V2 runner (upstream #52816 is present, but our V2 runner is not upstream's). Your stack can
  separate those two from the model itself.

## 8. Report back (what to send)

For each arm (matched INT8 pair; optionally the card's bf16 pair; optionally a mismatched control if
you want to reproduce our failure):
1. `per_pos` histogram (position 0 is the one that matters) and mean acceptance, per rep.
2. `ms/step`, t/s, context length, `num_speculative_tokens`, and any `Mean acceptance length` line.
3. Which attention backend the **target** and the **drafter** got, plus any fallback warnings.
4. Whether the model answered sensibly with speculation off (the sanity gate).

**Interpretation:** acceptance ~2-3 at pos 0 => the quantisation pairing is confirmed as the cause
(and our fork's missing `pack-quantized` support becomes the blocker, not DFlash2). Acceptance ~0
there too => the drafter/mechanism is broken independently of our fork, and DFlash2 should be
de-prioritised rather than ported further.

**Updated (2026-09-16):** we have already excluded the pairing and the masking on our fork, so the
decisive question for you is now **fork vs upstream**: run the same drafter against the same 4-bit
target on your stack (syv-ai's patches + real FA) and report per-position acceptance.
- Normal (~3 tokens/step) => the fault is in our fork's DFlash2 plumbing — worth us checking the aux
  hidden-state layer selection next.
- Degenerate (~1 token/step) => the drafter/mechanism is broken for this target family generally, and
  DFlash2 should be parked everywhere, not just here.

**RESULT (2026-09-16, from the 2x 5070 Ti box, upstream vLLM 0.29.0, TP=2, real FLASHINFER attention):**
degenerate — **0 accepted of 665 / 889 / 1785 drafted**, per-position acceptance `0.000 x7`, mean
acceptance length 1.00, at 1,242 / 9,453 / 52,837 prompt tokens; the spec-off sanity gate answered
correctly. Our fork measured the same regime (0.045 accepted/draft on the fallback path, 0.0282 with an
eager symmetric-window path). So **the fork is exonerated and the mechanism is dead upstream too**:
DFlash2 is **parked** for this target family (`DEVLOG-dflash2.md`, "external, decisive").

**The remaining check is answered too — the target is exonerated.** We ran the model card's own
documented pairing, **unquantised on both sides** (bf16 `Qwen/Qwen3.8-27B` + bf16
`incoai/Qwen3.8-27B-DFlash2`, TP=2, eager, k=7): acceptance **0.0408** at 1,024 prompt tokens (245
drafted; per-position 0.0204 / 0.0204 / 0 elsewhere) and **0.0079** at 2,048 (253 drafted), 6.30 and
5.96 t/s — the same degenerate regime, against syv-ai's ~0.33 per-draft reference.

So all four suspects are now excluded by measurement — **fork** (upstream 0.29.0 degenerate), **attention
path** (eager symmetric-window 0.0282 here, upstream FlashInfer 0.000), **drafter quantisation** (matched
W4A16 drafter degenerate), **target** (this run) — and **DFlash2 is dead for this family**. Nothing
further to run; the topic is parked finally, not provisionally (`DEVLOG-dflash2.md`).

For reference, if you ever need the bf16 pair on a 32 GB-class card: at TP=2 it fits only with
`--gpu-memory-utilization 0.96 --max-model-len 4096 --max-num-batched-tokens 256` (KV 9,206 tokens,
and that is *with* `--enforce-eager`); 0.92 with 32k gives a **negative** pool. On 2x16 GB cards the
unquantised 54 GB target does not fit at all, which is why your W4A16 route was the right one.

## 9. Links

- Our measurements + this analysis: `docs/gfx906/DEVLOG-dflash2.md` (in the `vllm-gfx906-mobydick`
  repo on Kevin's MI50 box; the same content's conclusions are summarised in §1 and §7 above).
- Upstream: PR #52816 (merged), PR #48375 (open), issues #43559 / #50021 (GDN state + race).
- Kevin's recipe for this model pair (the podman command above).
