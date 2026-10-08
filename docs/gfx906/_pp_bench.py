#!/usr/bin/env python3
"""Separates prefill from decode for the gfx906 FA (CUSTOM) vs stock backend.

Measures (per pp):
  prefill_tps  = pp / (time to process pp fresh tokens = a max_tokens=1 fresh
                 prompt). Dominated by prefill.
  decode_tps   = TG / (time to generate TG with a PREFIX-CACHED prompt, so the
                 prefill is a cache hit and elapsed ~= pure decode).

Usage: python3 /bench/_pp_bench.py <model>
Env: BENCH_PP list (default '256,512,1024,2048'), BENCH_TG (default 64),
     BENCH_ATTN (default auto; or CUSTOM/...), BENCH_GPU_UTIL.
Prints "BENCHPP: {json}".
"""
import json, os, sys, time

PP_LIST = sorted(int(x) for x in os.environ.get("BENCH_PP", "256,512,1024,2048").split(","))
TG = int(os.environ.get("BENCH_TG", "64"))
GPU_UTIL = float(os.environ.get("BENCH_GPU_UTIL", "0.6"))
ATTN = os.environ.get("BENCH_ATTN", None)  # None -> platform default

# Register CUSTOM at MODULE level so the spawn'd engine-core (which re-imports
# this module) also sees it. Only for the CUSTOM run.
if ATTN == "CUSTOM":
    from vllm.gfx906_fa.gfx906_fa_backend import register as fa_register
    fa_register()


def model_arg():
    if len(sys.argv) > 1 and sys.argv[1]:
        return sys.argv[1]
    raise SystemExit("no model")


def main():
    model = model_arg()
    maxlen = max(PP_LIST) + TG + 64

    from vllm import LLM, SamplingParams
    kw = dict(model=model, gpu_memory_utilization=GPU_UTIL,
              max_model_len=maxlen, dtype="auto", enforce_eager=True,
              enable_prefix_caching=True)
    if ATTN:
        kw["attention_backend"] = ATTN
    llm = LLM(**kw)
    tok = llm.get_tokenizer()

    filler = "The quick brown fox jumps over the lazy dog. "
    def make_prompt(pp):
        p, t = "", []
        while len(t) < pp:
            p += filler
            t = tok.encode(p)
        return tok.decode(t[:pp])

    # warmup (also warms the backend)
    w = make_prompt(min(min(PP_LIST), 128))
    llm.generate([w], SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))

    sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    spT = SamplingParams(max_tokens=TG, temperature=0.0, ignore_eos=True)

    results = []
    for pp in PP_LIST:
        prompt = make_prompt(pp)
        # Phase 1: fresh prefill (max_tokens=1) -> prefill time.
        t0 = time.time()
        out = llm.generate([prompt], sp1)
        t1 = time.time()
        prefill_s = t1 - t0
        prefill_tps = pp / prefill_s if prefill_s else 0.0

        # Phase 2: prefix-cached prompt, generate TG -> decode time.
        t0 = time.time()
        out = llm.generate([prompt], spT)
        t1 = time.time()
        decode_s = t1 - t0
        decode_tps = TG / decode_s if decode_s else 0.0

        results.append({
            "pp": pp, "prefill_s": round(prefill_s, 3),
            "prefill_tps": round(prefill_tps, 2),
            "decode_s": round(decode_s, 3),
            "decode_tps": round(decode_tps, 2),
        })
        print("BENCHPP: " + json.dumps({"model": model, "attn": ATTN or "default",
                                        "tg": TG, "r": results[-1]}), flush=True)


if __name__ == "__main__":
    main()