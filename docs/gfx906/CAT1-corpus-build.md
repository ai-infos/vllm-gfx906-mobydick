# Building the MTP draft-vocab corpus and list — replication guide

How to go from "agent traffic on this box" to a `mtp_draft_vocab_ids.pt` shortlist
that the MTP drafter can use, plus the checks that keep it honest. Companion to
[`CAT1-draft-vocab-corpus.md`](CAT1-draft-vocab-corpus.md) (what the corpus is
for) and `tools/build_draft_vocab.py` (the builder). Scripts:
`tools/build_draft_vocab_corpus.py` (emit), `tools/check_draft_vocab_list.py`
(validate). Measured numbers below are from the 2026-09 CAT-1 work
(`DEVLOG-draft-vocab.md`, `/local/tmp/b4/cat1-ab-status.md`).

**The corpus is use-dependent.** The list is a frequency table of *what this
deployment's model actually has to draft*. Pi-agentic coding traffic on this box
gave ~15.5 M own-model tokens and a 35 k-id list at 100 % raw-continuation
coverage; a chat service, a RAG service quoting documents, or another language
mix needs **its own** corpus. The only workload-independent part is the
**format/control-token family** (below) — always force that in. Record the
corpus provenance (sources, window, token counts) next to every list you ship,
and rebuild when the workload shifts.

## Pipeline

```
1 capture  raw assistant stream from the serving traffic (pi api-logs / sessions, hermes sqlite)
2 emit     one JSON object per assistant message -> corpus JSONL  (dedupe by content hash)
3 count    tools/build_draft_vocab.py count  -> ids list (N >= distinct ids observed)
4 slice    tools/build_draft_vocab.py slice  -> work dir the server can load
5 validate raw-continuation coverage + a serving A/B (identical prompts, >=3 reps, >=10 prompts)
```

## Must-have tokens (contained in code, listed here for review)

Force these into every list regardless of corpus frequency. For the Qwen3.8
tokenizer they are the *added* tokens at ids **248,044–248,076**; `all_special_ids`
contains only 9 of the 33, which is what broke the first build:

| ids | tokens | why |
|---|---|---|
| 248044 `<\|endoftext\|>`, 248045 `<\|im_start\|>`, 248046 `<\|im_end\|>` | chat framing | emitted at every turn boundary |
| **248058 `<tool_call>`, 248059 `</tool_call>`** | tool calls | **emitted at every tool call** — absent from parsed logs |
| **248066 `<tool_response>`, 248067 `</tool_response>`** | tool results | emitted when the model writes/replays tool results |
| **248068 `<think>`, 248069 `</think>`** | reasoning | thinking delimiters |
| 248053/54 `<\|vision_start\|>`/`<\|vision_end\|>`, 248056/57 `<\|image_pad\|>`/`<\|video_pad\|>`, 248070/71 `<\|audio_start\|>`/`<\|audio_end\|>`, 248076 `<\|audio_pad\|>` | multimodal | VL/audio serving |
| 248047–52 `<\|object_ref_*\|>`, `<\|box_*\|>`, `<\|quad_*\|>`, 248055 `<\|vision_pad\|>` | grounding/VL pads | VL serving |
| 248060–65 `<\|fim_*\|>`, `<\|repo_name\|>`, `<\|file_sep\|>` | FIM/repo | only if you serve FIM or repo-map prompts |
| 248072–75 `tts_*` | TTS | only for TTS serving |

`build_draft_vocab.py count` now forces the whole added-token family by default
(`--no-control-tokens` opts out), so the common path needs no extra flags. For a
tokenizer where the markup is *not* an added token, pass
`--extra-ids <json list>` with the ids you care about.

## Capture: what counts as generated text

Per assistant message, in emission order: `reasoning` + `response` +
`tool_calls`. Prompts are *inputs* and are not drafted — **but** the model's own
*format* tokens are, and parsed logs don't contain them, so:

- prefer the **raw** stream (server output before the tool parser, or a capture
  with `logprobs`), and
- never ship a list built from parsed text without the control family forced.

## Importing from pi agent

**Enable the logger first.** The api-log store is produced by a **box-local,
private pi extension**, vendored here as
[`tools/pi-extensions/log-openai.ts`](../tools/pi-extensions/README.md) so it can
be installed on other machines:

```bash
mkdir -p ~/.pi/agent/extensions
cp tools/pi-extensions/log-openai.ts ~/.pi/agent/extensions/   # pi auto-loads it
# restart pi, then confirm ~/.pi/agent/api-logs/<date>.jsonl grows per turn
```

It writes one JSONL line per event (`request` = exact OpenAI payload,
`response_headers`, `response` = the assembled assistant message), correlated by
`id`. Without it there is no api-log store to import — the **pi sessions** store
below still works and reaches further back in time.

Two stores, both per-day JSONL:

- `~/.pi/agent/api-logs/<date>.jsonl` — HTTP-level: `direction: request|response|response_headers`.
  Generated text = the `response` record's `message.content[]` blocks
  (`thinking` / `text` / `toolCall.arguments`); the request carries the prompt.
  `provider`/`model` identify the model (see the aliasing lesson below).
  **The logged `message` is parsed** — the raw tool-call markup is gone, which is
  exactly why the control-token family must be forced into the list.
- `~/.pi/agent/sessions/<project>/<session>.jsonl` — `{type: "message"}` lines with
  `message.role == "assistant"` and the same block schema; **model attribution
  needs the preceding `{type: "model_change"}` line** (one session mixes models).
  This store often reaches further back in time than the api-logs.

```python
def blocks(msg):                      # -> (reasoning[], response[], tool_calls[])
    r, t, c = [], [], []
    for b in (msg or {}).get("content") or []:
        k = b.get("type")
        if k == "thinking" and b.get("thinking"): r.append(b["thinking"])
        elif k == "text" and b.get("text"):       t.append(b["text"])
        elif k == "toolCall":                     c.append(json.dumps(
            {"name": b.get("name"), "arguments": b.get("arguments")}, ensure_ascii=False))
    return r, t, c
```

Emit one record per assistant message — the shape `build_draft_vocab.py`
counts (extra keys are ignored):

```json
{"session_id": "...", "ts": "...", "provider": "rtx5070", "model": "...",
 "domain": "coding", "source": "session",
 "reasoning": "...", "response": "...", "tool_calls": ["{\"name\": ...}"]}
```

## Importing from hermes

`~/.hermes/state.db` (sqlite): `messages(role, content, reasoning,
reasoning_content, tool_calls)` with `sessions(id, model)`. Assistant rows carry
the generated text; `tool_calls` is a JSON string.

```sql
select m.session_id, m.timestamp, m.reasoning, m.reasoning_content,
       m.content, m.tool_calls, coalesce(s.model,'?')
from messages m left join sessions s on s.id = m.session_id
where m.role = 'assistant'
```

Concatenate `reasoning + reasoning_content + content + tool_calls`, then apply
the same record shape and the same content-hash dedupe.

## Dedupe, attribution, scale

- **Dedupe by exact content hash, across every source.** Hermes had **32,217 of
  42,236** assistant rows as exact duplicates (76 %); the pi api-log window is
  re-stored inside the session files (2,899 responses + 819 + 114). Without
  dedupe the volume and the frequency table are both wrong.
- Keep `session_id` (for the session-level holdout), `ts`, `provider`, `model`,
  `domain` — they make the per-domain coverage report and later re-slicing possible.
- **Aliasing trap:** pi logs this box's Qwen3.8-27B under the stale id
  `Qwen3.6-27B-UD-Q4_K_XL.gguf`. A `model` filter written from the model card
  silently drops the best third of the corpus.
- Volume reference: 15.5 M own-model tokens ⇒ 99.8 % held-out occurrence
  coverage and a 35,251-id list; ~10.79 % of real positions sit in ≥8-token copy
  runs (which is why copy-heavy agentic output needs the tail).

## Count and slice

```bash
SNAP=/path/to/Qwen3.8-27B-AWQ-INT4            # tokenizer for counting
.venv/bin/python tools/build_draft_vocab.py count --snapshot $SNAP \
    --corpus corpus/*.jsonl --n 40000 --out-ids cat1_ids.json
.venv/bin/python tools/build_draft_vocab.py slice --snapshot $SNAP \
    --work-dir /local/models/cat1_work --ids cat1_ids.json
```

`count` also writes `<ids>.provenance.json` (corpus files, token count, holdout
scheme, forced-token policy, timestamps) and `slice` folds it into
`<work dir>/cat1_manifest.json`, which carries the canonical hashes that bind the
ids file to the sliced head rows.

`--n` is an upper bound: the list is `top-(n - |forced|)` ∪ forced, so pass
**n ≥ the number of distinct ids you observed** rather than a round cap — the
ids a cap drops are exactly the rare ones copy-heavy output needs (measured:
a 32,768 cap cost 2.4 % of a raw tool-call continuation's positions; 35,251
"all observed" cost 0). Keep the corpus files under the per-file text limit
(default 20 MB; the tool warns when it truncates) or split them.

## Enable / disable on a server (and the workload caveat)

There is **no corpus input at serve time** and nothing is enabled by default. The
switch is **file presence in the served model directory**:

| served model dir | drafter |
|---|---|
| stock checkpoint (no `mtp_draft_vocab_ids.pt`) | full `lm_head` (baseline MTP) |
| a work dir from `slice` (ids `.pt` + `model_extra_tensors.safetensors` + one index entry) | shortlist head (`MTP drafter uses a N-token draft head`) |
| either, with `MTP_DRAFT_VOCAB=0` in the server env | baseline (kill switch, no rebuild needed) |

At load the server logs the manifest's provenance
(`MTP draft-vocab manifest: 35251 ids, sha1 …, corpus …, control tokens True`),
so the served list's origin is visible in every server log.

So "which corpus" is answered by **which work dir you serve**
(`vllm serve /path/to/cat1_<workload>`), and the unit to copy between machines is
the whole work dir (the ids file and the sliced rows in
`model_extra_tensors.safetensors` must stay a matched pair — the row order *is*
the id mapping; a count mismatch fails loudly at load, two lists with the same N
would not).

**The list is a frequency table of the corpus's traffic — keep them matched.**
Measured on this box: the own-corpus list (35,251 ids) covers 99.6 %+ of
generated positions on the traffic it was built from and costs no measurable
acceptance; a *web*-tuned 40 K list covers only 95.2–96 % of the same traffic
(4–5 % of positions outside the list). The per-step byte saving is
workload-independent (~6.4 % of a step), so a list from a *different* workload —
another language, a different code ecosystem, numeric/table-heavy output — can
approach break-even or lose. The format half is safe everywhere now (the
control-token family is forced); the content half is not.

Before enabling a list for a new workload, spend ~10 minutes: capture a few raw
continuations with `logprobs` on that workload and run
`check_draft_vocab_list.py --continuation …`; require ~99.5 %+ position coverage,
then confirm with the A/B hygiene rules. Otherwise rebuild the list from that
workload's traffic, or serve that deployment with `MTP_DRAFT_VOCAB=0`.
Note the head is **additive**: the full `lm_head` stays loaded (361 MB extra at
35,251 rows).

## Validate before shipping

0. **Manifest** (`check_draft_vocab_list.py` does it for you): the ids hash and
   the `model_extra_tensors.safetensors` hash in `cat1_manifest.json` must match
   the artifacts. The row order *is* the id mapping, so an ids file swapped for
   another list of the same length would otherwise serve wrong draft logits
   silently — the server verifies the same manifest at load
   (`MTP_DRAFT_VOCAB_STRICT=0` skips only the 300 ms head-file hash) and logs the
   provenance when it matches.
1. **Artifact checks** (all offline, seconds): the work dir's
   `model.safetensors.index.json` adds exactly one key; the draft head rows are
   **byte-identical** to `lm_head.weight[ids]`; ids are sorted and unique; the
   work-dir `.pt` equals the JSON list.
2. **List checks** (`tools/check_draft_vocab_list.py`): every must-have token
   present; `set(json list) == set(.pt)`; coverage of a **raw continuation**
   capture (not just of the corpus). Sanity bands from the 2026-09 run:
   parsed-text occurrence coverage is ~99.8 % and does **not** discriminate
   (the unfixed 32 k list scored 99.96 % and still missed `<tool_call>`);
   raw-continuation coverage was 97.7–98.0 % unfixed → **100 %** fixed.
3. **Serving A/B** — the only real gate, with the hygiene rules from
   `/local/git/AGENTS.md`: identical prompts across arms (never put the arm name
   in the prompt), ≥3 reps per prompt and ≥~10 distinct prompts, per-rep
   acceptance (acceptance is chaotic on this box: ±7 pp run-to-run on the full
   head), and compare **ms/step** — a 32 k-row head vs the full 248 k-row head
   was 75.7 vs 78.5 ms/step at 64 k (≈2.8 ms/step saved, matching the standalone
   probe's ~2.7 ms).
   Reference points from the same work: 35,251-id list ≡ 336–361 MB head vs
   2.54 GB full; acceptance with the fixed list indistinguishable from the full
   head within noise.

## Lessons learned (the short list)

1. Parsed logs cannot contain the markup the model emits → capture raw or force
   the control family. This single defect cost a full A/B round.
2. `all_special_ids` is not the control-token set (9 of 33 here).
3. Occurrence coverage is blind to a missing format-token family *and* to the
   N-cap tail. Check raw-continuation coverage; gate on a serving A/B.
4. Dedupe across sources before counting (76 % duplicates in one store).
5. Session-level holdout, not record-level (`--holdout session`, the default).
6. N ≥ distinct observed ids; a round cap quietly deletes the copy-regime tail.
7. Synthetic payloads (periodic filler, 100 % acceptance) cannot validate a
   list: they hide both the markup loss and the acceptance cost. Validate on real
   traffic with real tool calls.
8. Identical prompts across arms — a header carrying the arm name is enough to
   change the continuation and the acceptance.
9. Acceptance is chaotic (±7 pp); one sample per (arm, prompt) is meaningless.
10. Tooling traps in the builder/client paths: silent 20 MB/file truncation;
    a corpus file without `.jsonl` read as one blob; block-style
    `messages` content skipped; `run_server.sh` `exec`s `vllm serve` so
    `pkill -f "run_server.sh …"` never matches (kill by PID, verify VRAM).
11. Check that your bench "bodies" are distinct at the chosen prefill length
    (`corpus.json['agent']` bodies are the same prompt after a 65 k cut).
12. Record provenance with the list. The corpus is a snapshot of one workload's
    history; a different service needs a different list.
