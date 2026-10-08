# Copyright Kevin Read <me@kevin-read.com>
"""Serial-vs-batch token comparison (review ds4 F2/F7 gate).

Runs two same-length prompts serially (M=1) and concurrently (M=2)
and reports divergence points. The concurrent run exercises the FA
uniform-prefill fast path (num_tokens == 2*max_seqlen_q) and the
non-spec M=2 fp16 decode path (m4 kernel / AWQ m_count=2). Divergence
is expected as a benign fp argmax flip near logit ties (S3-class bar:
coherence, not token identity across M); a scatter/gather or row-
mapping bug would show as incoherent text instead."""
import os
os.environ.setdefault("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
from vllm import LLM, SamplingParams

MODEL = "/data/models/qwen/Qwen3.5-27B-AWQ"
P0 = ("You are a senior Python engineer. Write a function that parses a "
      "cron expression and returns the next five fire times. Explain the "
      "tricky edge cases in your code comments.")
P1 = ("Explain how a lock-free queue works in a single-writer multi-reader "
      "setting. Cover the ABA problem, memory ordering requirements on "
      "weakly ordered architectures, and when a plain mutex is the better "
      "choice.")

def main():
    llm = LLM(model=MODEL, max_model_len=2816, gpu_memory_utilization=0.95,
              max_num_seqs=4, dtype="float16", trust_remote_code=True,
              enable_prefix_caching=False)
    sp = SamplingParams(temperature=0.0, max_tokens=128)
    s0 = llm.generate([P0], sp)[0].outputs[0].text
    s1 = llm.generate([P1], sp)[0].outputs[0].text
    b0, b1 = [o.outputs[0].text for o in llm.generate([P0, P1], sp)]
    for i, (s, b) in enumerate([(s0, b0), (s1, b1)]):
        if s == b:
            print(f"prompt {i}: identical")
            continue
        j = next(k for k in range(min(len(s), len(b))) if s[k] != b[k])
        print(f"prompt {i}: diverge at char {j}")
        print("  serial ...", repr(s[max(0, j-80):j+100]))
        print("  batch  ...", repr(b[max(0, j-80):j+100]))
        # coherence tail
        print("  batch tail:", repr(b[-120:]))
    import sys
    sys.exit(0)

if __name__ == "__main__":
    main()
    os._exit(0)
