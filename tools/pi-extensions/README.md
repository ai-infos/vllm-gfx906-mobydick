# pi API-log extension (vendored)

`log-openai.ts` is a **pi coding-agent extension** that logs every provider call
to `~/.pi/agent/api-logs/<YYYY-MM-DD>.jsonl`. It is a **box-local, private
extension** (not part of pi, not published) — it exists on this machine at
`~/.pi/agent/extensions/log-openai.ts` and is vendored here so other machines
(and future sessions) can install it without reverse-engineering the log format.

The file is a **verbatim copy** (md5 `16064640bb119c581348ff3c2447ac4d`); keep it
in sync with the local original rather than editing it here.

## Activate

```bash
mkdir -p ~/.pi/agent/extensions
cp tools/pi-extensions/log-openai.ts ~/.pi/agent/extensions/
# pi auto-loads every extension in that directory — restart pi, then verify:
ls -la ~/.pi/agent/api-logs/            # <date>.jsonl should grow with each turn
```

## What it writes (one JSON object per line)

| `direction` | fields |
|---|---|
| `request` | `id`, `ts`, `provider`, `model`, `session`, `payload` (exact OpenAI request JSON: `messages`, `tools`, sampling params) |
| `response_headers` | `id`, `ts`, `status`, `headers` |
| `response` | `id`, `ts`, `provider`, `model`, `session`, `message` (the fully assembled assistant message: `content[]` blocks of `thinking` / `text` / `toolCall` + usage + `stopReason`) |

Requests and responses are correlated by `id` (a UUID minted per provider call),
which is what lets a corpus builder pair a prompt with its generated text.

## Limitation that matters for draft-vocab corpora

`message` is the **parsed** assistant message: tool calls arrive as structured
`toolCall` blocks, so the raw markup the model emits (`<tool_call>`,
`</tool_call>`, `<tool_response>`, `<think>`, …) is **not** in the log. A draft
shortlist built from these logs therefore misses those tokens and the drafter can
never propose them. `tools/build_draft_vocab.py count` forces the tokenizer's
added-token family by default for exactly this reason, and
[`../../docs/gfx906/CAT1-corpus-build.md`](../../docs/gfx906/CAT1-corpus-build.md)
documents the checks. If you want the raw stream (the durable fix), capture it on
the serving side (provider payload / SSE) rather than relying on this extension.

## Related

- Corpus pipeline + lessons: [`../../docs/gfx906/CAT1-corpus-build.md`](../../docs/gfx906/CAT1-corpus-build.md)
- Corpus spec: [`../../docs/gfx906/CAT1-draft-vocab-corpus.md`](../../docs/gfx906/CAT1-draft-vocab-corpus.md)
- Emitter (reads these logs): `tools/build_draft_vocab_corpus.py`
- List validation: `tools/check_draft_vocab_list.py`
