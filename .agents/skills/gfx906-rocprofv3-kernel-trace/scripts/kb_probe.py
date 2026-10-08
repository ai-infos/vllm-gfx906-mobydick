# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""TTFT stall: per-step GPU kernel breakdown (ttft-prefill-stall.md
§12.6). In-process LLM, TP=2, mtp3, util 0.93, pp=1024 (same per-step
terms as the matrix). One 8192-token prefill = 8 steps of ~2.6-3.3 s;
the torch profiler (kineto/rocprofiler) traces CPU+CUDA for the
generate call; the chrome trace is exported for offline per-kernel
bucketing (GPU-domain spans/ratios are usable on this stack — the
wall-alignment caveat per AGENTS.md 2026-08-22 does not affect
per-kernel durations/ratios).

Launch:  bash run_kernel_breakdown.sh
Wall: ~5-6 min load + ~30 s traced prefill.
"""
import json
import os
import time

MODEL = (
    "/local/cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/"
    "snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea"
)
CORPUS = "/local/tmp/mtp1/corpus.json"
TRACEDIR = "/local/tmp/b4/kbtraces"
PP = int(os.environ.get("PROF_PP", "1024"))
PREFILL = int(os.environ.get("KB_PREFILL", "8192"))


def main():
    from transformers import AutoTokenizer

    print(f"KB: pp={PP} prefill={PREFILL}", flush=True)
    bodies = json.load(open(CORPUS))["mixed"]["65536"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    header = tok("KB-BREAKDOWN-x7f3e9a2 ",
                 add_special_tokens=False)["input_ids"]
    prompt = header + bodies[0][: PREFILL - len(header)]
    assert len(prompt) == PREFILL

    from vllm import LLM, SamplingParams

    tp = int(os.environ.get("KB_TP", "2"))
    t0 = time.time()
    os.makedirs(TRACEDIR, exist_ok=True)
    for f in os.listdir(TRACEDIR):
        os.remove(os.path.join(TRACEDIR, f))
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=tp,
        enforce_eager=(os.environ.get("KB_EAGER") == "1"),
        max_model_len=int(os.environ.get("KB_MAXLEN", "131072")),
        max_num_seqs=4,
        max_num_batched_tokens=PP,
        gpu_memory_utilization=0.93,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        seed=0,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 3, 4]},
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": TRACEDIR,
            "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": False,
        },
    )
    print(f"KB: loaded in {time.time() - t0:.0f} s", flush=True)

    # warm (untimed): short prompt
    short = tok("KBWARM-x7f3e9a2 ",
                add_special_tokens=False)["input_ids"] + bodies[0][: 96]
    llm.generate([short], SamplingParams(temperature=0.0, max_tokens=4),
                 use_tqdm=False)
    print("KB: warm done", flush=True)

    llm.start_profile()
    t1 = time.time()
    llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=1),
        use_tqdm=False,
    )
    wall = time.time() - t1
    llm.stop_profile()
    print(f"KB: traced prefill wall={wall:.2f} s "
          f"(~{PREFILL // PP} steps)", flush=True)
    print(f"KB-DONE traces={TRACEDIR} wall={wall:.2f}", flush=True)
    # rocprofv3 flush window (2026-09-11): vLLM's shutdown force-kills the
    # EngineCore workers before rocprofv3 can flush their activity buffers
    # (output generation ran empty at §13.1 retry). Holding the process tree
    # alive lets the rocpd/sqlite flush complete on clean child exit.
    if os.environ.get("KB_FLUSH_WAIT"):
        print(f"KB: flush wait {os.environ['KB_FLUSH_WAIT']}s for rocprofv3",
              flush=True)
        time.sleep(float(os.environ["KB_FLUSH_WAIT"]))


if __name__ == "__main__":
    main()
