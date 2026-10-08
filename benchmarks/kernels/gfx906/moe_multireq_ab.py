#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Multi-request decode A/B bench (originally C2-V (v2) on the 35B MoE,
docs/gfx906/moe-decode-roadmap.md; reused for the W4 skinny-M A/B on the
27B/35B FA layers, docs/gfx906/DEVLOG-fp16-skinny.md).

N concurrent prompts (C2V_PP tokens each) decoding C2V_TG tokens, one
engine per process; the driver alternates flag off/on engines to cancel
build/date drift. Decode throughput is the difference of two runs in
the SAME engine (tg and C2V_TG_SHORT, identical prompts): the
prefill/decode overlap pattern cancels in the difference, and the
result is decode-only t/s — comparable to the single-request decode
records (35B: 67.39 t/s band, 14.9 ms/step).

Env: C2V_MODEL (required), C2V_N (1/4/8/32), C2V_PP (2048), C2V_TG
(256), C2V_TG_SHORT (16), C2V_MAXLEN (4096), C2V_UTIL (0.95),
C2V_EAGER (0=graph, 1=eager), C2V_TP (1/2), C2V_REPEATS (3),
C2V_MAX_SEQS (32), C2V_TAG (label echoed in output).

Prints "C2V-REP: {json}" per repeat and one "C2V-SUMMARY: {json}" at
the end (the driver parses the summary). A SIGTERM handler attempts a
clean engine-core shutdown before exiting (TP=2 teardown protocol:
unclean kills can wedge GPU1 mid-P2P).
"""

import hashlib
import json
import os
import signal
import sys
import time

_STATE = {}


def _shutdown(signum, frame):
    try:
        llm = _STATE.get("llm")
        if llm is not None:
            core = getattr(llm.llm_engine, "engine_core", None)
            if core is not None and hasattr(core, "shutdown"):
                core.shutdown(timeout=20)
    except BaseException:
        pass
    os._exit(124)


def main():
    model = os.environ.get("C2V_MODEL", "")
    if not model:
        raise SystemExit("C2V: set C2V_MODEL")
    n = int(os.environ.get("C2V_N", "1"))
    pp = int(os.environ.get("C2V_PP", "2048"))
    tg = int(os.environ.get("C2V_TG", "256"))
    tg_short = int(os.environ.get("C2V_TG_SHORT", "16"))
    maxlen = int(os.environ.get("C2V_MAXLEN", "4096"))
    util = float(os.environ.get("C2V_UTIL", "0.95"))
    eager = os.environ.get("C2V_EAGER", "0") == "1"
    tp = int(os.environ.get("C2V_TP", "1"))
    repeats = int(os.environ.get("C2V_REPEATS", "3"))
    max_seqs = int(os.environ.get("C2V_MAX_SEQS", "32"))
    tag = os.environ.get("C2V_TAG", "")
    assert n in (1, 4, 8, 32), f"C2V_N must be 1/4/8/32, got {n}"
    assert tg > tg_short > 0 and maxlen > pp + tg

    signal.signal(signal.SIGTERM, _shutdown)

    from vllm import LLM, SamplingParams

    # Trimmed capture sizes (the TP=2 trimmed-capture convention): 1 +
    # powers of 2 up to the decode batch. Fewer shapes = less capture
    # VRAM and wall.
    sizes = [1]
    s = 2
    while s < n:
        sizes.append(s)
        s *= 2
    if n > 1:
        sizes.append(s)

    extra = {"max_num_seqs": max_seqs}
    if not eager:
        extra["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": sizes,
        }
    t0 = time.perf_counter()
    llm = LLM(
        model=model,
        gpu_memory_utilization=util,
        max_model_len=maxlen,
        dtype="auto",
        enforce_eager=eager,
        seed=0,
        tensor_parallel_size=tp,
        **extra,
    )
    _STATE["llm"] = llm
    load_s = time.perf_counter() - t0
    print(f"C2V-LOAD: {load_s:.0f}s tp={tp} eager={eager} n={n}", flush=True)

    tok = llm.get_tokenizer()
    filler = "The quick brown fox jumps over the lazy dog. "
    prompts = []
    for i in range(n):
        p = f"Prompt number {i}. "
        while len(tok.encode(p)) < pp:
            p += filler
        prompts.append(tok.decode(tok.encode(p)[:pp]))

    def params(mt):
        try:
            return SamplingParams(
                max_tokens=mt, temperature=0.0, ignore_eos=True,
                enable_thinking=False,
            )
        except TypeError:
            return SamplingParams(max_tokens=mt, temperature=0.0, ignore_eos=True)

    # Warmup: untimed short run (graph capture, kernel warm, page-in).
    t0 = time.perf_counter()
    llm.generate(prompts[: min(n, 4)], params(min(tg_short, 8)))
    warmup_s = time.perf_counter() - t0
    print(f"C2V-WARMUP: {warmup_s:.0f}s", flush=True)

    reps = []
    for r in range(repeats):
        t0 = time.perf_counter()
        llm.generate(prompts, params(tg_short))
        wall_short = time.perf_counter() - t0
        t0 = time.perf_counter()
        outs = llm.generate(prompts, params(tg))
        wall_long = time.perf_counter() - t0
        n_out = sum(len(o.outputs[0].token_ids) for o in outs)
        d = wall_long - wall_short
        decode_tps = n * (tg - tg_short) / d if d > 0 else 0.0
        # Greedy output fingerprint: A/B arms must produce identical
        # tokens (a re-tile that changes results is not a win).
        fp = hashlib.sha1(
            " ".join(
                " ".join(map(str, o.outputs[0].token_ids)) for o in outs
            ).encode()
        ).hexdigest()[:16]
        rep = {
            "rep": r,
            "wall_short_s": round(wall_short, 2),
            "wall_long_s": round(wall_long, 2),
            "decode_tps": round(decode_tps, 2),
            "out_tokens": n_out,
            "out_fp": fp,
        }
        reps.append(rep)
        print("C2V-REP: " + json.dumps(rep), flush=True)

    dt = [x["decode_tps"] for x in reps]
    mean = sum(dt) / len(dt)
    var = sum((x - mean) ** 2 for x in dt) / (len(dt) - 1) if len(dt) > 1 else 0.0
    summary = {
        "tag": tag,
        "model": model,
        "n": n,
        "tp": tp,
        "eager": eager,
        "pp": pp,
        "tg": tg,
        "tg_short": tg_short,
        "maxlen": maxlen,
        "max_seqs": max_seqs,
        "capture_sizes": sizes if not eager else None,
        "vllm_gfx906_moe_m1": os.environ.get("VLLM_GFX906_MOE_M1", "0"),
        "vllm_gfx906_moe_npt": os.environ.get("VLLM_GFX906_MOE_NPT", ""),
        "vllm_gfx906_skinny_m16": os.environ.get("VLLM_GFX906_SKINNY_M16", "0"),
        "load_s": round(load_s, 1),
        "warmup_s": round(warmup_s, 1),
        "decode_tps_mean": round(mean, 2),
        "decode_tps_stdev": round(var ** 0.5, 2),
        "decode_tps_min": round(min(dt), 2),
        "decode_tps_max": round(max(dt), 2),
        "wall_long_mean": round(sum(x["wall_long_s"] for x in reps) / len(reps), 2),
        "out_fps": sorted(set(x["out_fp"] for x in reps)),
        "reps": reps,
    }
    print("C2V-SUMMARY: " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
