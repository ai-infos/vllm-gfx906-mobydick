# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""N3 attribution probe: which ops produce the GDN state-copy pile?

ROADMAP N3: ~180 us/step of small `[3,1,32]` copies in the GDN layers
(sandwiching causal_conv1d_update + fused_recurrent_gated_delta_rule),
attributed to upstream mamba state bookkeeping (DEVLOG-dense-decode.md).

This probe runs an eager single-request decode on Qwen3.5-35B-A3B-AWQ and
reports, for copy-class aten ops in the DECODE phase: per-(op, shape)
self GPU time normalized to us/step.

Design note (v3): NO with_stack. Full Python stacks per event OOM-killed
runs 2-4 during post-processing even under MemoryMax=infinity. Launch
sites are identified from kernel/aten names + source inspection instead.
Aggregation uses key_averages(group_by_input_shape=True), which is
compact (unique keys only, no per-event objects).

Run:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 HIP_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
      FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
      .venv/bin/python benchmarks/kernels/gfx906/n3_state_copy_probe.py \
          > /local/tmp/n3/probe.log 2>&1
"""

import gc
import os
from collections import defaultdict

import torch

from vllm import LLM, SamplingParams

MODEL = "/data/models/QuantTrio/Qwen3.5-35B-A3B-AWQ"
# 32 tokens bounds the profiler buffer (per-step averages are stable well
# below this); 128 blew host RAM in post-processing under with_stack runs.
TG = 32           # profiled decode tokens
COPYISH = ("copy_", "_to_copy", "contiguous", "clone", "Memcpy")


def main() -> None:
    # N3_MODE=eager (default) vs graph (production FULL_DECODE_ONLY).
    # Graph mode is the production serving regime; under it the decode
    # forward is captured into a CUDA graph and replayed, so per-op CPU
    # dispatch (and its launch cost) disappears from the profiled window.
    mode = os.environ.get("N3_MODE", "eager")
    kwargs = dict(
        model=MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.93,
        seed=0,
    )
    if mode == "graph":
        # Hybrid GDN: capture requires max_num_seqs <= Mamba cache blocks;
        # single-request bench -> keep it small (GDN state pool is big).
        kwargs["max_num_seqs"] = 4
        kwargs["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "max_cudagraph_capture_size": 8,
        }
    else:
        kwargs["enforce_eager"] = True   # eager: copies are visible as ops
    llm = LLM(**kwargs)
    prompt = "The quick brown fox jumps over the lazy dog. " * 40

    # warmup (runs prefill once; also JITs triton kernels) so the profiled
    # call is pure decode
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8),
                 use_tqdm=False)

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = llm.generate([prompt],
                           SamplingParams(temperature=0.0, max_tokens=TG),
                           use_tqdm=False)
    torch.cuda.synchronize()

    n_tokens = len(out[0].outputs[0].token_ids)
    print(f"N3-PROBE: profiled {n_tokens} decode tokens "
          f"(mode={mode}, B=1)", flush=True)
    del out, llm
    gc.collect()
    print("N3-PROBE: engine released; starting aggregation", flush=True)

    # --- per-(op, input-shape) self device time -------------------------
    ka = prof.key_averages(group_by_input_shape=True)
    del prof
    gc.collect()

    agg = defaultdict(lambda: [0.0, 0])   # (name, shapes) -> [us, count]
    for evt in ka:
        name = evt.key
        if not any(c in name for c in COPYISH):
            continue
        shapes = tuple(evt.input_shapes or [])
        key = (name, str(shapes))
        dev_us = getattr(evt, "self_device_time_total", 0.0) or 0.0
        agg[key][0] += dev_us
        agg[key][1] += evt.count

    print(f"N3-PROBE: copy-class (op,shape) groups: {len(agg)}", flush=True)

    rows = []
    for key, (us, cnt) in agg.items():
        n_per_step = cnt / n_tokens
        if n_per_step < 0.25:            # drop sub-quarter-per-step noise
            continue
        rows.append((n_per_step, us / n_tokens, key))

    rows.sort(key=lambda r: -r[1])
    print("\n=== copy-class ops per decode step (count/step | us-GPU/step) ===")
    total = 0.0
    for nps, us_step, key in rows:
        total += us_step
        name, shapes = key
        print(f"  {nps:8.2f}/step  {us_step:7.1f} us  {name:32s} {shapes}")
    print(f"TOTAL copy-class GPU time: ~{total:.0f} us/step", flush=True)

    # --- context: where do the big non-copy steps go? (top-20 all ops) --
    top = sorted(ka, key=lambda e: -(e.self_device_time_total or 0))[:20]
    print("\n=== top-20 device-time ops in decode phase (us/step) ===")
    for evt in top:
        nps = evt.count / n_tokens
        print(f"  {evt.self_device_time_total / n_tokens:8.1f} us  "
              f"{nps:7.2f}/step  {evt.key[:60]}")

    print("N3-PROBE: done", flush=True)


if __name__ == "__main__":
    main()
