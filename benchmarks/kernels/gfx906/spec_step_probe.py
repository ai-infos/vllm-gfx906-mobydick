# Copyright Kevin Read <me@kevin-read.com>
import json, time, urllib.request
BASE = "http://127.0.0.1:8931"
def counters():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        t = r.read().decode()
    out = {}
    for w in ("vllm:spec_decode_num_drafts_total{",
              "vllm:spec_decode_num_draft_tokens_total{",
              "vllm:spec_decode_num_accepted_tokens_total{"):
        for line in t.splitlines():
            if line.startswith(w):
                out[line.split("{")[0]] = float(line.rsplit(" ", 1)[1])
    return out
def post(prompt, n):
    before = counters()
    t0 = time.perf_counter()
    req = urllib.request.Request(BASE + "/v1/completions",
        data=json.dumps({"model": "qwen27", "prompt": prompt,
                         "max_tokens": n, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    dt = time.perf_counter() - t0
    after = counters()
    d = {k: after[k] - before[k] for k in before}
    n_out = resp["usage"]["completion_tokens"]
    steps_with_draft = d["vllm:spec_decode_num_drafts_total"]
    nodraft_steps = n_out - steps_with_draft - d["vllm:spec_decode_num_accepted_tokens_total"]
    print(json.dumps({"out_tokens": n_out, "elapsed_s": round(dt, 3),
        "tps": round(n_out / dt, 2),
        "draft_steps": int(steps_with_draft), "nodraft_steps": int(nodraft_steps),
        "draft_tokens": int(d["vllm:spec_decode_num_draft_tokens_total"]),
        "accepted": int(d["vllm:spec_decode_num_accepted_tokens_total"]),
        "acc_per_step": round(d["vllm:spec_decode_num_accepted_tokens_total"] / max(1, steps_with_draft), 3),
        "ms_per_token": round(1000 * dt / n_out, 2)}), flush=True)

post("Write a list of 250 unique 10-character hexadecimal identifiers, one per line, each different from every other. Start the first with a31f.", 256)
post("Repeat the following sentence exactly 60 times, once per line, with no changes: the quick brown fox jumps over the lazy dog", 128)
