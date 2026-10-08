# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""W2: spec-decode rails A/B on the 35B MoE (MTP k=2 vs greedy).

In-process serving A/B, same method both arms (identical prompts,
temp 0, prefill-inclusive t/s = n_out / wall — the 27B-phase
convention). Acceptance counters are captured by patching
SpecDecodingStats.observe_draft (robust to in-process stats wiring).
Greedy output fingerprints per (prompt, rep): spec arms must match
the baseline arm; the baseline is also run twice to separate
model-boundary drift from arm-specific divergence.

Env: M (model), SPEC (speculative-config json or empty), EAGER,
REPEATS (default 3), TGT (default 256), MAXLEN, UTIL, MAX_SEQS, TAG.
Prints SPECAB-REP / SPECAB-SUMMARY JSON.
"""
import hashlib
import json
import os
import signal
import sys
import time

from vllm import SamplingParams
from vllm.v1.spec_decode.metrics import SpecDecodingStats

MODEL = os.environ.get("M", "/local/models/QuantTrio/Qwen3.5-35B-A3B-AWQ")
SPEC = os.environ.get("SPEC", "")
EAGER = os.environ.get("EAGER", "0") == "1"
REPEATS = int(os.environ.get("REPEATS", "3"))
TGT = int(os.environ.get("TGT", "256"))
MAXLEN = int(os.environ.get("MAXLEN", "4096"))
UTIL = float(os.environ.get("UTIL", "0.95"))
MAX_SEQS = int(os.environ.get("MAX_SEQS", "4"))
TAG = os.environ.get("TAG", "specab")
# LLM() defaults disable_log_stats=True (llm.py) which suppresses the
# scheduler's spec-decoding stats (and thus observe_draft); enable for
# the acceptance counters.
LOG_STATS = os.environ.get("LOG_STATS", "1") == "1"

SYSTEM = (
    "You are an expert terminal assistant. Reply concisely; when a "
    "command is needed, answer with a short explanation then the "
    "command on its own line."
)

PROMPTS = [
    # 1: billing bug investigation (tool loop, repetitive code)
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": ("Our nightly billing job produced wrong totals for "
                     "subscription invoices. The customer disputes tax on "
                     "discounted line items. Investigate and fix.")},
        {"role": "assistant",
         "content": '{"tool": "read_file", "arguments": '
                    '{"path": "billing/invoice.py"}}'},
        {"role": "user",
         "content": ("Tool result for read_file(billing/invoice.py):\n"
                     "```python\n"
                     "from decimal import Decimal, ROUND_HALF_UP\n"
                     "\n"
                     "TAX_RATES = {\n"
                     '    "US-CA": Decimal("0.0875"),\n'
                     '    "US-NY": Decimal("0.08875"),\n'
                     '    "DE-BE": Decimal("0.19"),\n'
                     "}\n"
                     "\n"
                     "class Invoice:\n"
                     "    def __init__(self, invoice_id, customer_id, "
                     "region):\n"
                     "        self.invoice_id = invoice_id\n"
                     "        self.customer_id = customer_id\n"
                     "        self.region = region\n"
                     "        self.lines = []\n"
                     "\n"
                     "    def add_line(self, desc, amount, discount=0):\n"
                     "        self.lines.append(\n"
                     "            (desc, amount, amount * discount))\n"
                     "\n"
                     "    def total(self):\n"
                     "        rate = TAX_RATES.get(self.region, "
                     "Decimal(\"0\"))\n"
                     "        sub = sum(l[1] - l[2] for l in "
                     "self.lines)\n"
                     "        return (sub * (1 + rate)).quantize(\n"
                     '            Decimal("0.01"), ROUND_HALF_UP)\n'
                     "```\n"
                     "What looks wrong, and give the fixed total() "
                     "method?")},
    ],
    # 2: slow cold-start triage (long ls output, repetitive paths)
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": "vLLM cold start takes 9 minutes on this box. "
                    "First I ran:\n"
                    "$ du -sh ~/.cache/vllm/* ~/.cache/huggingface "
                    "2>/dev/null\n"
                    "4.2G\t/root/.cache/vllm/torch_compile_cache\n"
                    "2.1G\t/root/.cache/vllm/inductor_cache\n"
                    "23.7G\t/root/.cache/huggingface\n"
                    "$ ls /local/models/QuantTrio/\n"
                    "Qwen3.5-35B-A3B-AWQ  Qwen3.8-27B-AWQ-INT4\n"
                    "Where should I look first, and what is the "
                    "cheapest fix?"},
        {"role": "assistant",
         "content": "The inductor/torch-compile caches are warm, so the "
                    "9 minutes is likely weight load from NFS. Check:\n"
                    "$ time dd if=/dev/zero of=/dev/null\n"
                    "$ ls -la /data/models | head\n"
                    "Mount /local/models if the weights live on NFS "
                    "and copy the 35B to local disk."},
        {"role": "user",
         "content": ("$ time dd if=/dev/zero of=/dev/null bs=1M count=1\n"
                     "1+0 records in, 1+0 records out\n"
                     "1048576 bytes (1.0 MB) copied, 0.000312 s, "
                     "3.4 GB/s\n"
                     "So local disk is fast. The model is on NFS. What "
                     "exactly do I run?")},
    ],
]

# ---- acceptance counters (patch observe_draft) -----------------------
_ACC = [0, 0, 0]  # [draft_tokens, accepted_tokens, steps]
_orig_observe = SpecDecodingStats.observe_draft


def _observe(self, num_draft_tokens, num_accepted_tokens):
    _orig_observe(self, num_draft_tokens, num_accepted_tokens)
    _ACC[0] += num_draft_tokens
    _ACC[1] += num_accepted_tokens
    _ACC[2] += 1


SpecDecodingStats.observe_draft = _observe


def _shutdown(*_):
    try:
        llm.llm_engine.engine_core.shutdown(timeout=20)  # noqa: F821
    except Exception:
        pass
    os._exit(124)


signal.signal(signal.SIGTERM, _shutdown)

from vllm import LLM  # noqa: E402

comp = None
if not EAGER:
    comp = {"cudagraph_capture_sizes": [1, 2, 3, 4]}

t0 = time.time()
llm = LLM(
    model=MODEL,
    max_model_len=MAXLEN,
    gpu_memory_utilization=UTIL,
    max_num_seqs=MAX_SEQS,
    enforce_eager=EAGER,
    speculative_config=json.loads(SPEC) if SPEC else None,
    compilation_config=comp,
    enable_prefix_caching=False,
    disable_log_stats=not LOG_STATS,
)
t_boot = time.time() - t0
print(f"SPECAB-BOOT: {t_boot:.1f}s spec={bool(SPEC)}", flush=True)

tz = llm.get_tokenizer()


def chat(text_msgs):
    return tz.apply_chat_template(
        text_msgs, tokenize=False, add_generation_prompt=True)


reps = []
for rep in range(REPEATS):
    for pi, msgs in enumerate(PROMPTS):
        prompt = chat(msgs)
        sp = SamplingParams(temperature=0.0, max_tokens=TGT)
        t1 = time.time()
        outs = llm.generate([prompt], sp, use_tqdm=False)
        wall = time.time() - t1
        toks = outs[0].outputs[0].token_ids
        fp = hashlib.sha1(
            " ".join(map(str, toks)).encode()).hexdigest()[:16]
        reps.append({
            "rep": rep,
            "prompt": pi,
            "n_out": len(toks),
            "wall_s": round(wall, 2),
            "tps": round(len(toks) / wall, 2),
            "fp": fp,
        })
        print(f"SPECAB-REP: {json.dumps(reps[-1])}", flush=True)

n_draft, n_acc, n_steps = _ACC
summary = {
    "tag": TAG,
    "spec": bool(SPEC),
    "eager": EAGER,
    "tps_mean": round(sum(r["tps"] for r in reps) / len(reps), 2),
    "tps_stdev": round((
        sum((r["tps"] - sum(x["tps"] for x in reps) / len(reps))**2
            for r in reps) / len(reps))**0.5, 2),
    "wall_mean": round(sum(r["wall_s"] for r in reps) / len(reps), 2),
    "draft_tokens": n_draft,
    "accepted_tokens": n_acc,
    "tok_per_step": round(n_acc / n_steps, 3) if n_steps else None,
    "acceptance_pct": round(100.0 * n_acc / n_draft, 2) if n_draft else None,
    "fp_set": sorted(set(r["fp"] for r in reps)),
    "reps": reps,
}
print(f"SPECAB-SUMMARY: {json.dumps(summary)}", flush=True)
