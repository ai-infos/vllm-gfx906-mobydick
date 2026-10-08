#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""In-model per-kernel GPU time table for the gfx906 decode bench.

Loads Qwen3.5-35B-A3B-AWQ at the standard bench config (pp=2048 prefill,
tg=256 decode, eager — the BENCH_EAGER default), warms up, then profiles
one full generation with the torch CUDA activity profiler and prints the
top kernels by measured GPU time:

  name | cnt/step | us/step | us/call

Steps = 1 prefill + tg decode. Prefill contamination is ~0.4 % of decode
time; prefill-only kernels show up with cnt/step ~ 1/tg and are easy to
spot. MoE kernels fire 40x/step (one per layer); GDN ~30x, FA ~10x.

Env:
  BENCH_PP    prefill tokens (default 2048)
  BENCH_TG    decode tokens to profile (default 256)
  TOP         rows to print (default 40)
  NO_WARMUP   skip the untimed warmup generation (default: run it)
"""
import os

import torch

MODEL = "/local/models/QuantTrio/Qwen3.5-35B-A3B-AWQ"


def make_prompt(tok, pp: int) -> str:
    filler = "The quick brown fox jumps over the lazy dog. "
    prompt = ""
    while True:
        prompt += filler
        toks = tok.encode(prompt)
        if len(toks) >= pp:
            break
    return tok.decode(toks[:pp])


def main():
    from torch.profiler import ProfilerActivity, profile

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    pp = int(os.environ.get("BENCH_PP", "2048"))
    tg = int(os.environ.get("BENCH_TG", "256"))
    top = int(os.environ.get("TOP", "40"))
    steps = 1 + tg  # one prefill step + tg decode steps

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt = make_prompt(tok, pp)

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=pp + tg + 512,
        enforce_eager=True,
    )

    if not os.environ.get("NO_WARMUP"):
        llm.generate([prompt], SamplingParams(max_tokens=16, temperature=0.0))

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
        raise SystemExit(
            "set VLLM_ENABLE_V1_MULTIPROCESSING=0: the v1 engine runs in a "
            "separate process and the torch profiler sees no kernels")

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        llm.generate([prompt], SamplingParams(max_tokens=tg, temperature=0.0))

    rows = []
    total_us = 0.0
    for ev in prof.key_averages():
        us = ev.self_device_time_total
        if us <= 0:
            continue
        total_us += us
        cnt = ev.count
        rows.append((us / steps, cnt / steps, us / max(cnt, 1), ev.key))
    rows.sort(reverse=True)

    print(f"steps={steps} ({tg} decode + 1 prefill)  "
          f"GPU-busy={total_us / steps:.0f} us/step")
    print(f"{'us/step':>9} {'cnt/step':>9} {'us/call':>9}  kernel")
    shown = 0.0
    for us_step, cnt_step, us_call, key in rows[:top]:
        shown += us_step
        print(f"{us_step:9.1f} {cnt_step:9.2f} {us_call:9.1f}  {key[:96]}")
    if total_us <= 0:
        raise SystemExit("no GPU kernel time captured — profiler mismatch")
    print(f"(top {len(rows[:top])} rows = {shown:.0f} us/step, "
          f"{100.0 * shown / (total_us / steps):.1f}% of GPU-busy)")


if __name__ == "__main__":
    main()
