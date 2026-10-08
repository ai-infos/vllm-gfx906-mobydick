# CAT-1 draft-vocab corpus — required inputs

> Branch `cat1-draft-vocab` · model `cyankiwi/Qwen3.8-27B-AWQ-INT4` · date 2026-09-04 ·
> roadmap item CAT-1 (see [RECON-1cat-vllm.md](RECON-1cat-vllm.md)).

Purpose: define exactly what text we need, in what format, to build **our own**
`mtp_draft_vocab_ids` for the vocab-truncated draft head (syv patch, attributed —
see main README). The id list is the only input that has to come from real traffic;
everything else (row slicing, scatter-back, A/B) is mechanical.

> **Step-by-step replication guide + sample scripts:**
> [`CAT1-corpus-build.md`](CAT1-corpus-build.md) (capture → emit → count → slice → validate).

## Why request/response capture is the right basis

The MTP drafter proposes tokens at every decode step — i.e. it only ever drafts
**model-generated** text. Speculative decoding stays exact regardless of what the
drafter proposes (rejection sampling uses the target model), so the acceptance-rate
cost of a shortlist is ≈ the fraction of *generated* token mass that falls outside
the list. Consequences:

- **Self-generated output is the correct counting corpus.** Coverage measured on
  held-out text the model itself produced directly predicts the acceptance drop.
  Third-party corpora (web/wiki) systematically under-cover our format tokens
  (`think` blocks, tool-call XML/JSON scaffolding, code conventions), which are
  exactly the high-frequency, well-draftable tokens we most want in the list.
- **Prompts / system prompts do not need coverage** — input *content* tokens are never
  drafted. **But this does not extend to the model's own format/markup tokens:**
  parsed request logs (`reasoning`/`response`/`tool_calls`) never contain the raw
  markup the model emits (`<tool_call>`, `</tool_call>`, `<tool_response>`,
  `<think>`, `</parameter>`, …), which are emitted at **every tool boundary** in
  agentic traffic. A list built only from parsed text omits them, the drafter can
  never propose them, and the acceptance rate drops (measured 2026-09-13 on a raw
  tool-call continuation: 98.0 % of positions covered without them, **100 %**
  with). Capture the **raw** assistant stream where it is available (server output
  before the tool parser, or `logprobs`-style capture); `tools/build_draft_vocab.py`
  now forces the tokenizer's whole *added-token* family into the list by default
  (`--no-control-tokens` opts out). Reasoning traces and tool-call blocks are
  *high value*: structured, repetitive scaffolding that drafts very well and is
  under-represented in any public corpus.

## What to capture

1. **Generated text only**, per request or per full session, in original order:
   reasoning trace → response text → each raw tool-call block as emitted.
2. **Complete sessions** — no mid-session truncation (truncation biases the
   frequency distribution toward early-turn tokens).
3. **Raw as-served**: exactly what the model emitted, including `think` tags and
   tool-call markup. No post-processing, reformatting, redaction of format tokens,
   or markdown normalization.
4. **Distinct real traffic** — avoid synthetic repetitions of the same prompt;
   repeated boilerplate inflates its frequency and distorts the top-N selection.

## Recommended mix and volume

- **≥ 15 M generated tokens.** Reference point: the upstream list was counted over
  8.8 M tokens and reached 95% held-out coverage at N=40960; our static budget is
  ~131K rows (≈3.2× that), so a comparable corpus should clear ≥97%. More volume =
  better tail coverage; the marginal value drops off around 20–30 M tokens.
- **Mix** roughly proportional to production traffic. If unsure, start with
  ~60% agentic coding / ~25% chat Q&A / ~15% long reasoning traces.
- **Languages**: match production usage. Each additional language adds tail
  pressure (its common tokens must be covered), so prioritize by traffic share —
  a language that is rare in real requests can be low-priority.

## Format

**Primary: JSONL**, one object per line (per request, or one object per session):

```json
{"session_id": "optional", "prompt": "...", "reasoning": "...", "response": "...", "tool_calls": ["<raw block 1>", "<raw block 2>"]}
```

- **Counted** = `reasoning` + `response` + `tool_calls`, concatenated in original
  order. `prompt`/`session_id` are ignored for counting.
- A `messages` array (chat format) is also accepted in place of `prompt`/
  `response`; assistant-role content strings are used verbatim.
- **Alternative: plain `.txt`** — one session per file, or a single concatenated
  file; everything in it is counted as generated text.

**Do not pre-tokenize.** We tokenize with the model's own tokenizer; id lists
captured elsewhere risk tokenizer-version mismatch and silently corrupt coverage.

## Delivery

- Location: `/local/tmp/mtp1/corpus/` (durable — `/tmp` is tmpfs), or NFS
  `/data/models/cat1/` if preferred.
- Naming: `<YYYYMMDD>-<domain>.jsonl`, e.g. `20260904-coding.jsonl`,
  `20260904-chat.jsonl`, `20260904-reasoning.jsonl` — or a single mixed file.
- Size expectation: ~4 bytes/token → 15 M tokens ≈ 60 MB. Tiny; no compression
  needed (gzip fine if convenient).

## What we do with it (acceptance criteria up front)

1. Tokenize + count, with a held-out split (every 10th session held out — the
   upstream scheme). **Check the corpus is complete before trusting the count:**
   `count` warns when it truncates a file at `--max-text-bytes` (20 MB default) and
   refuses to emit a list when nothing was counted.
2. Top-N ids + special/**control** tokens; report **held-out token coverage** at
   N ∈ {32768, 65536, 98304, 131072}, overall and per domain (coding/chat/
   reasoning) so under-covered domains are visible. **Occurrence coverage is
   necessary but blind** — it cannot see a missing format-token family (parsed
   text scores 99.9 %+) nor the N-cap tail. Also report **distinct-id coverage**
   and, ultimately, gate on a serving A/B.
3. Selection rule: prefer **N ≥ the number of distinct ids observed in the
   corpus** (≈35 k for our traffic) rather than a round cap — the tokens a cap
   drops are exactly the rare ones copy-heavy agentic output needs. If coverage
   at that N is short of the target, identify the weak domains from step 2 and ask
   for a targeted top-up rather than blind more data.
3b. **Gate: a serving acceptance A/B on a real payload, not a coverage table.**
   Identical prompts across arms (never put the arm name in the prompt), ≥3 reps
   per prompt and ≥~10 distinct prompts, per-rep acceptance reported (acceptance
   is chaotic on this box: ±7 pp run-to-run on the full head). See
   `/local/git/AGENTS.md` §"Serving A/B measurement hygiene".
3c. Ship metadata: `slice` writes `cat1_manifest.json` (ids sha1, head-file sha1,
   corpus provenance) next to the artifacts; the server verifies and logs it at
   load, which is what makes "which corpus is this list from?" answerable and a
   mixed ids/head pair a load-time error instead of silent bad drafts.

4. Slice those rows from our bf16 `lm_head` → `mtp_draft_vocab_ids.pt` +
   `model_extra_tensors.safetensors`. A/B switch is file-presence based (file in
   directory = shortlist arm; absent or `MTP_DRAFT_VOCAB=0` = baseline) — no env
   var has to cross the EngineCore boundary.
5. A/B on the standard TP=2 / MTP depth-2 serving config: t/s + acceptance rate +
   coherence spot-checks (token-identity checks are NOT valid — model is
   non-deterministic at temp=0).

## Dynamic variant note

This corpus serves the **static** list, which we do first. The dynamic
98K + 2×512 shortlist needs no extra corpus — it bootstraps a per-request top-k
from target prefill logits — and is only evaluated if static's acceptance cost
turns out to be too high.
