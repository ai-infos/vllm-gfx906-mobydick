# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""W1 probe: 2-request mixed spec batch — GDN kernel routing + t/s.

Request A: repetitive prompt (ngram always matches -> spec decode).
Request B: unique-n-gram prompt (ngram never matches -> non-spec decode).
Most decode steps are MIXED batches (one spec + one non-spec decode seq),
the regime where the GDN metadata reclass pathology lives (no-draft seqs
run the chunk kernel as 1-token "prefills" instead of the per-seq
recurrent kernel).

Spies the GDN leaf kernel entry points (module-global wrap; must be
installed before the model runs) and reports per-path call totals:
  - chunk:       fla_chunk_gated_delta_rule (prefill path)
  - fused_seq:   fused_sigmoid_gating_delta_rule_update
                 (spec 2.1 with num_accepted_tokens; peeled decodes
                 2.2/2.3 without)
  - packed:      fused_recurrent_gated_delta_rule_packed_decode

Usage:
    HIP_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/kernels/gfx906/probe_gdn_mixed.py
Env: BENCH_MODEL, PROBE_NTOK (256), PROBE_PROMPT_TOK (2048)
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import collections

# --- spies BEFORE model load ---------------------------------------------
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as g

totals = collections.Counter()
fused_spec_calls = 0
fused_nonspec_calls = 0
step_comps = collections.Counter()  # (num_prefills, num_decodes, num_spec_decodes)
step_toks = collections.Counter()   # (num_actual_tokens, num_spec_decodes)
step_q = collections.Counter()      # (spec_mask, spec_lens, non_spec_lens)


def _wrap(name):
    real = getattr(g, name)

    def spy(*a, **kw):
        global fused_spec_calls, fused_nonspec_calls
        totals[name] += 1
        if name == "fused_sigmoid_gating_delta_rule_update":
            if kw.get("num_accepted_tokens") is not None:
                fused_spec_calls += 1
            else:
                fused_nonspec_calls += 1
        return real(*a, **kw)

    setattr(g, name, spy)


_wrap("fla_chunk_gated_delta_rule")
_wrap("fused_sigmoid_gating_delta_rule_update")
_wrap("fused_recurrent_gated_delta_rule_packed_decode")


sched_steps = collections.Counter()  # (reqs, tokens) per engine step


def _wrap_scheduler():
    """Ground truth: which requests the engine scheduled per step."""
    from vllm.v1.core.sched.scheduler import Scheduler

    real = Scheduler.schedule

    def spy(self, *a, **kw):
        out = real(self, *a, **kw)
        ns = out.num_scheduled_tokens or {}
        positions = {r.request_id: i for i, r in enumerate(self.running)}
        reqs = tuple(sorted(positions.get(r, -1) for r in ns))
        sched_steps[(reqs, sum(ns.values()))] += 1
        return out

    Scheduler.schedule = spy


_wrap_scheduler()


def _wrap_step_composition():
    """Record the per-layer per-step GDN batch composition (ground truth
    for which dispatch branch each step took)."""
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gl

    real = gl.QwenGatedDeltaNetAttention._forward_core

    def spy(self, mixed_qkv, b, a, core_attn_out):
        fc = get_forward_context()
        if fc.attn_metadata is not None:
            md = fc.attn_metadata[self.prefix]
            key = (md.num_prefills, md.num_decodes, md.num_spec_decodes)
            step_comps[key] += 1
            step_toks[(md.num_actual_tokens, md.num_spec_decodes)] += 1
            if md.spec_sequence_masks is not None:
                mask = tuple(int(x) for x in md.spec_sequence_masks.tolist())
                sl = md.spec_query_start_loc
                spec_lens = tuple(int(sl[i + 1] - sl[i]) for i in range(sl.numel() - 1))
                nl = md.non_spec_query_start_loc
                if nl is not None:
                    non_lens = tuple(
                        int(nl[i + 1] - nl[i]) for i in range(nl.numel() - 1)
                    )
                else:
                    non_lens = ()
            else:
                mask, spec_lens = (), ()
                nl = md.non_spec_query_start_loc
                non_lens = tuple(
                    int(nl[i + 1] - nl[i]) for i in range(nl.numel() - 1)
                ) if nl is not None else ()
            step_q[(mask, spec_lens, non_lens)] += 1
        return real(self, mixed_qkv, b, a, core_attn_out)

    gl.QwenGatedDeltaNetAttention._forward_core = spy


_wrap_step_composition()
# ---------------------------------------------------------------------------


def main():
    import time

    from vllm import LLM, SamplingParams

    model = os.environ.get(
        "BENCH_MODEL",
        "/local/cache/huggingface/hub/models--cyankiwi--Qwen3.5-9B-AWQ-INT8-INT4"
        "/snapshots/763be420f16be619241ed1bd1ac6b79deb4a986a",
    )
    ntok = int(os.environ.get("PROBE_NTOK", "256"))
    ptok = int(os.environ.get("PROBE_PROMPT_TOK", "2048"))
    b_temp = float(os.environ.get("PROBE_B_TEMP", "1.0"))
    b_seed = int(os.environ.get("PROBE_B_SEED", "0"))

    llm = LLM(
        model=model,
        max_model_len=ptok + ntok + 256,
        max_num_seqs=2,
        gpu_memory_utilization=0.95,
        dtype="auto",
        enforce_eager=True,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 2,
        },
        seed=0,
    )
    tok = llm.get_tokenizer()

    # A: repetitive (always drafts)
    filler = "The quick brown fox jumps over the lazy dog. "
    prompt_a = ""
    while len(tok.encode(prompt_a)) < ptok:
        prompt_a += filler
    prompt_a = tok.decode(tok.encode(prompt_a)[:ptok])

    # B: unique 2-grams ("t0 t1 t2 ...") at temp=0 (continues the pattern,
    # drafts often), or random gibberish (PROBE_B_GIBBERISH=1) for the PPL
    # run. Per-prompt temps: A drafts every step; B at temp=1 samples
    # tokens that rarely repeat a 2-gram -> B stays non-spec -> mixed
    # batches on (almost) every step.
    # Diverse sentence pool (each used once): B's context then has almost
    # no repeated 2-grams -> B drafts rarely -> mixed batches with near-tie
    # argmax structure (the PPL/identity gate).
    bpool = (
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
        # Novel mid-sentence ending: the greedy continuation of this has
        # never occurred in B's context, so B's 2-grams are new and the
        # ngram proposer finds no match -> B stays non-spec.
        "The lighthouse keeper climbed the spiral staircase each evening "
        "and trimmed the "
    )
    if os.environ.get("PROBE_B_TEXT") == "1":
        prompt_b = bpool
    elif os.environ.get("PROBE_B_GIBBERISH") == "1":
        import random

        rng = random.Random(b_seed)
        vocab = tok.get_vocab()
        vocab_ids = sorted(vocab.values())
        prompt_b = tok.decode(rng.sample(vocab_ids, ptok))
    else:
        b = " ".join(f"t{i}" for i in range(ptok * 3))
        prompt_b = tok.decode(tok.encode(b)[:ptok])

    spa = SamplingParams(max_tokens=ntok, temperature=0.0, ignore_eos=True)
    spb = SamplingParams(
        max_tokens=ntok,
        temperature=b_temp,
        top_p=0.95 if b_temp > 0 else 1.0,
        seed=b_seed,
        ignore_eos=True,
    )
    llm.generate([prompt_a, prompt_b], [spa, spb], use_tqdm=False)  # untimed

    t0 = time.perf_counter()
    outs = llm.generate([prompt_a, prompt_b], [spa, spb], use_tqdm=False)
    dt = time.perf_counter() - t0

    n_a = len(outs[0].outputs[0].token_ids)
    n_b = len(outs[1].outputs[0].token_ids)
    print(
        f"GDN-MIXED t/s total={(n_a + n_b) / dt:.1f} "
        f"A={n_a / dt:.1f} B={n_b / dt:.1f} ({n_a + n_b} tok / {dt:.1f}s)",
        flush=True,
    )
    for name in (
        "fla_chunk_gated_delta_rule",
        "fused_sigmoid_gating_delta_rule_update",
        "fused_recurrent_gated_delta_rule_packed_decode",
    ):
        print(f"GDN-MIXED kernel {name}: {totals[name]}", flush=True)
    print(
        f"GDN-MIXED fused_seq split: spec(natt)={fused_spec_calls} "
        f"non-spec={fused_nonspec_calls}",
        flush=True,
    )
    print(
        "GDN-MIXED step composition (num_prefills, num_decodes, "
        "num_spec_decodes): count",
        flush=True,
    )
    for key in sorted(step_comps):
        print(f"GDN-MIXED   comp={key}: {step_comps[key]}", flush=True)
    print(
        "GDN-MIXED step tokens (num_actual_tokens, num_spec_decodes): count",
        flush=True,
    )
    for key in sorted(step_toks):
        print(f"GDN-MIXED   toks={key}: {step_toks[key]}", flush=True)
    print(
        "GDN-MIXED per-req (spec_mask A,B / spec lens / non-spec lens): count",
        flush=True,
    )
    for key in sorted(step_q):
        print(f"GDN-MIXED   q={key}: {step_q[key]}", flush=True)
    print(
        "GDN-MIXED scheduler (running-req positions, tokens): count",
        flush=True,
    )
    for key in sorted(sched_steps):
        print(f"GDN-MIXED   sched={key}: {sched_steps[key]}", flush=True)
    print(f"GDN-MIXED outA[:80]={outs[0].outputs[0].text[:80]!r}", flush=True)
    print(f"GDN-MIXED outB[:80]={outs[1].outputs[0].text[:80]!r}", flush=True)
    import hashlib

    for tag, out in (("A", outs[0]), ("B", outs[1])):
        h = hashlib.sha1(
            b" ".join(str(t).encode() for t in out.outputs[0].token_ids)
        ).hexdigest()[:12]
        print(f"GDN-MIXED tohash{tag}={h}", flush=True)
        if tag == "B" and os.environ.get("PROBE_DUMP_B"):
            with open(os.environ["PROBE_DUMP_B"], "w") as f:
                f.write(" ".join(map(str, outs[1].outputs[0].token_ids)))
                f.write("\n")
            with open(os.environ["PROBE_DUMP_B"] + ".txt", "w") as f2:
                f2.write(outs[1].outputs[0].text)


if __name__ == "__main__":
    main()
