#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""gfx906 benchmark (0.23 vs 0.26 vs main). Usage: python3 /bench/_b.py <model>
Env: BENCH_PP (2048), BENCH_TG (256), BENCH_GPU_UTIL (0.85),
     BENCH_BATCHED_TOKENS (4096), BENCH_MAXLEN, BENCH_WARMUP
     (1=do untimed warmup), BENCH_SAMPLES (default 1).
Measures one pp-prefill + tg-decode request (after an untimed warmup).
Prints "BENCH: {json}". Robust to cross-version SamplingParams differences.
"""

import glob
import json
import os
import re
import sys
import threading
import time

WARMUP = os.environ.get("BENCH_WARMUP", "1") == "1"
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "1"))

# DVFS gate (docs/gfx906/dvfs-mi50.md): idle mclk is 350 MHz and cold-clock
# benches are inflated ~3x. Sample mclk concurrently with the timed windows and
# report it per card; a median < 900 MHz invalidates the number.
#
# Read it from sysfs, not `rocm-smi --showclocks`: that call blocks while a deck
# is under load, and a single median hides exactly the case this gate exists for
# -- a sustained bandwidth-bound prefill holds 1000 MHz but dips to 800 for a
# large share of the wall (measured 41-44 % of loaded prefill time at 64k/27B,
# DEVLOG-fa-multibatch-prefill.md 2026-10-06), which a per-sample median reports
# as a clean 1000.
class _MclkSampler:
    # pp_dpm_mclk lines look like "1: 800Mhz *" -- level, clock, active marker.
    _ACTIVE = re.compile(r"^\s*\d+:\s*(\d+)\s*Mhz\s*\*", re.M)

    def __init__(self, period=0.3):
        self.period, self.samples, self._stop = period, [], threading.Event()
        self.cards = sorted(glob.glob("/sys/class/drm/card*/device/pp_dpm_mclk"))
        self._t = threading.Thread(target=self._run, daemon=True)

    def _read(self):
        out = {}
        for path in self.cards:
            try:
                with open(path) as fh:
                    m = self._ACTIVE.search(fh.read())
            except OSError:
                continue
            if m:
                out[path.split("/")[4]] = int(m.group(1))
        return out

    def _run(self):
        while not self._stop.is_set():
            v = self._read()
            if v:
                self.samples.append((time.time(), v))
            self._stop.wait(self.period)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()

    def _span(self, t0, t1):
        """(t, {card: mhz}, held_seconds) for samples overlapping [t0, t1]."""
        hi = min(t1, self.samples[-1][0]) if self.samples else t1
        for i, (t, clocks) in enumerate(self.samples):
            if t > t1:
                break
            nxt = self.samples[i + 1][0] if i + 1 < len(self.samples) else hi
            held = min(nxt, hi) - max(t, t0)
            if held > 0:
                yield t, clocks, held

    def share_between(self, t0, t1):
        """{card: {mhz: seconds}} -- time-weighted, not per-sample."""
        out = {}
        for _, clocks, held in self._span(t0, t1):
            for card, mhz in clocks.items():
                out.setdefault(card, {})
                out[card][mhz] = out[card].get(mhz, 0.0) + held
        return out

    def median_between(self, t0, t1):
        spans = list(self._span(t0, t1))
        vals = [v for _, clocks, _ in spans for v in clocks.values()]
        return (sorted(vals)[len(vals) // 2], len(spans)) if vals else (None, 0)


def model_arg():
    if len(sys.argv) > 1 and sys.argv[1]:
        return sys.argv[1]
    m = os.environ.get("BENCH_MODEL", "")
    if m:
        return m
    raise SystemExit("BENCH: no model (argv[1] or BENCH_MODEL)")


def main():
    model = model_arg()
    pp = int(os.environ.get("BENCH_PP", "2048"))
    tg = int(os.environ.get("BENCH_TG", "256"))
    gpu_util = float(os.environ.get("BENCH_GPU_UTIL", "0.85"))
    maxlen = int(os.environ.get("BENCH_MAXLEN", str(pp + tg + 512)))

    from vllm import LLM, SamplingParams

    # BENCH_EAGER=0 runs with cudagraphs ("serving mode"); numbers are NOT
    # comparable to the eager tables in the README. FULL_DECODE_ONLY + small
    # capture size: this bench is single-request decode-dominated.
    eager = os.environ.get("BENCH_EAGER", "1") == "1"
    extra = {}
    # This vLLM dropped VLLM_ATTENTION_BACKEND; force the backend via
    # attention_config (AttentionConfig.backend). On gfx906 the default
    # resolves to the CUSTOM (Q8 FA) backend.
    extra["max_num_batched_tokens"] = int(
        os.environ.get("BENCH_BATCHED_TOKENS", "4096")
    )
    attn_backend = os.environ.get("BENCH_ATTN_BACKEND")
    attn_backend_kind = os.environ.get("BENCH_ATTN_BACKEND_KIND")  # JSON
    if attn_backend or attn_backend_kind:
        cfg = {}
        if attn_backend:
            cfg["backend"] = attn_backend
        if attn_backend_kind:
            # e.g. '{"sliding_window": "ROCM_ATTN"}' — pins one KV-cache
            # kind while the rest keep auto-selection.
            cfg["backend_per_kind"] = json.loads(attn_backend_kind)
        extra["attention_config"] = cfg
    # BENCH_MOE_BACKEND (e.g. triton) overrides the MoE backend selection
    # for A/B runs (default auto picks the gfx906 W4A16 kernel where gated).
    moe_backend = os.environ.get("BENCH_MOE_BACKEND")
    if moe_backend:
        extra["moe_backend"] = moe_backend
    # BENCH_SPEC_CONFIG (JSON) sets speculative_config, e.g.
    # '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":2}'.
    spec_config = os.environ.get("BENCH_SPEC_CONFIG")
    if spec_config:
        extra["speculative_config"] = json.loads(spec_config)
    # BENCH_KV_MEM (bytes) caps the KV pool explicitly. Needed when the
    # warm-cache profiling peak underestimates runtime inductor/prefill
    # buffers and gpu_memory_utilization alone OOMs the first request
    # (e.g. 30B-class models whose 532 MiB prefill buffer exceeds the
    # util-0.93 headroom); also makes A/B arms use an identical pool.
    kv_mem = os.environ.get("BENCH_KV_MEM")
    if kv_mem:
        extra["kv_cache_memory_bytes"] = int(kv_mem)
    # BENCH_PREFIX_CACHE (default 0): the local-serving bench default is
    # prefix caching OFF (AGENTS.md — it poisons TTFT-derived numbers).
    # The harness historically left the vLLM default (on); pre-flip
    # DEVLOG numbers used that, gate re-runs use BENCH_PREFIX_CACHE=0
    # explicitly.
    extra["enable_prefix_caching"] = \
        os.environ.get("BENCH_PREFIX_CACHE", "0") == "1"
    # BENCH_NREQS (default 1) runs that many identical prompts concurrently
    # (prefix caching is off, so prefills are real); totals are aggregated.
    nreqs = int(os.environ.get("BENCH_NREQS", "1"))
    if not eager:
        # Hybrid GDN model: cudagraph capture requires max_num_seqs <= number
        # of Mamba cache blocks. Single-request bench -> 32 is plenty.
        # BENCH_MAX_SEQS overrides (dense 27B needs 4: the GDN state pool is
        # ~72 MB/seq and 32 seqs OOMs the 1568-chunk prefill, 2026-08-18).
        # BENCH_CG_MODE overrides the cudagraph mode (P3-3a M0 needs
        # Triton in PIECEWISE for the mode-matched baseline).
        extra["max_num_seqs"] = int(os.environ.get("BENCH_MAX_SEQS", "32"))
        extra["compilation_config"] = {
            "cudagraph_mode": os.environ.get("BENCH_CG_MODE", "FULL_DECODE_ONLY"),
            # Spec decode: steps carry nreqs*(k+1) tokens; BENCH_CG_MAX must
            # cover that or mixed-batch steps fall back to eager.
            "max_cudagraph_capture_size": int(os.environ.get("BENCH_CG_MAX", "8")),
        }
    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_util,
        max_model_len=maxlen,
        # BENCH_DTYPE (e.g. float16): checkpoints whose config says bfloat16
        # need an explicit float16 to select the fp16-only gfx906 kernels.
        dtype=os.environ.get("BENCH_DTYPE", "auto"),
        enforce_eager=eager,
        **extra,
    )
    tok = llm.get_tokenizer()
    # 2026-09-15 (GEMMA4-1): this harness fills *raw text* to the target prompt
    # length and measures tokens/s only. A speed number is not a correctness gate —
    # Gemma-4-*-it sat in the docs as "supported, 67.79 t/s" while its raw-text
    # output was garbage (it is an IFT checkpoint and needs its chat template).
    # Recorded in every row so a number is never mistaken for validation.
    prompt_form = "raw-filler"
    has_chat_template = bool(getattr(tok, "chat_template", None))
    if has_chat_template:
        print(
            "WARNING: this tokenizer has a chat template, so its output for this "
            "raw-text filler prompt is NOT a quality signal (IFT-only checkpoints "
            "return garbage here). Tokens/s is valid; use a templated gate "
            "(BENCH_CHAT_TEMPLATE=1 in benchmarks/kernels/gfx906/ppl_probe.py) or a "
            "serving A/B for correctness.",
            flush=True,
        )

    # Build a prompt encoding to exactly pp tokens.
    filler = "The quick brown fox jumps over the lazy dog. "
    prompt = ""
    toks = []
    while len(toks) < pp:
        prompt += filler
        toks = tok.encode(prompt)
    prompt = tok.decode(toks[:pp])

    def gen_params(max_tokens):
        # enable_thinking may not exist across versions; probe harmlessly.
        try:
            return SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                ignore_eos=True,
                enable_thinking=False,
            )
        except TypeError:
            return SamplingParams(
                max_tokens=max_tokens, temperature=0.0, ignore_eos=True
            )

    prompts = [prompt] * nreqs

    # BENCH_MIXED=1 (with nreqs>=2): request 0 is the repetitive filler
    # (ngram always drafts -> spec), the rest are a diverse sentence pool
    # ending on a novel mid-sentence (ngram finds no match -> non-spec
    # decode) -> most decode steps are spec-mixed batches. For the W1 GDN
    # reclass A/B (the reclass pathology only exists in mixed batches).
    if os.environ.get("BENCH_MIXED") == "1" and nreqs >= 2:
        pool = (
            "The compiler first parses the source into an abstract syntax "
            "tree, then walks the tree emitting instructions while keeping "
            "register pressure within the limits of the target architecture. "
            "Marble statues weather slowly as acid rain etches their "
            "surfaces, turning sharp chiselled detail into soft shapes over "
            "centuries of exposure to the open air. "
            "Battery capacity fades with age because the solid electrolyte "
            "interface layer thickens on the anode, trapping lithium ions "
            "and reducing the charge the cell can deliver. "
            "Good tests describe intent: a name like test_overflow_refunds "
            "when total exceeds budget tells the reader what behavior is "
            "protected without opening the body of the function. "
            "The river braids and splits around gravel islands each spring, "
            "carrying snowmelt from the high valleys down to the delta where "
            "the marsh grass bends but does not break. "
            "Quantum computers exploit superposition and entanglement to "
            "explore many candidate solutions at once, though error "
            "correction remains the principal obstacle to practical machines. "
            "The lighthouse keeper climbed the spiral staircase each evening "
            "and trimmed the "
        )
        prompts = [prompt] + [pool] * (nreqs - 1)

    if WARMUP:
        llm.generate(prompts, gen_params(min(tg, 8)))
        print("BENCH warmup_pass done", flush=True)

    sampler = _MclkSampler().start()
    results = []
    for s in range(SAMPLES):
        t0 = time.time()
        outs = llm.generate(prompts, gen_params(tg))
        t1 = time.time()
        n_out = sum(len(o.outputs[0].token_ids) for o in outs)
        # token_ids-based: text re-encoding collapses on degenerate/garbage
        # output (e.g. '!!!!...') and undercounts.
        elapsed = t1 - t0
        mclk, nsamp = sampler.median_between(t0, t1)
        if mclk is not None and mclk < 900:
            print(f"BENCH WARNING: sample {s} median mclk {mclk} MHz < 900 "
                  f"(cold-clock; number INVALID) - see dvfs-mi50.md", flush=True)
        # A clean median is not a clean clock: report the per-card time share
        # held at <= 800 MHz of the loaded (>= 400 MHz) wall as well.
        shares = sampler.share_between(t0, t1)
        down = {}
        for card, secs in shares.items():
            loaded = sum(v for k, v in secs.items() if k >= 400)
            low = sum(v for k, v in secs.items() if 400 <= k < 900)
            if loaded > 0:
                down[card] = round(low / loaded, 4)
        worst = max(down.values()) if down else 0.0
        if worst > 0.10:
            worst_card = max(down, key=lambda c: down[c])
            print(f"BENCH WARNING: sample {s} held 800 MHz for {worst:.0%} of "
                  f"loaded time by card {worst_card} - the median "
                  f"({mclk} MHz) hides it; see dvfs-mi50.md", flush=True)
        results.append(
            {
                "sample": s,
                "nreqs": nreqs,
                "out_tokens": n_out,
                "elapsed_s": round(elapsed, 3),
                "tokens_per_s": round(n_out / elapsed, 3) if elapsed else 0.0,
                "mclk_median_mhz": mclk,
                "mclk_n": nsamp,
                "mclk_800_share_of_loaded": down,
            }
        )
    sampler.stop()

    print(
        "BENCH: "
        + json.dumps(
            {
                "model": model,
                "pp": pp,
                "tg": tg,
                "prompt_form": prompt_form,
                "has_chat_template": has_chat_template,
                "maxlen": maxlen,
                "samples": results,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
