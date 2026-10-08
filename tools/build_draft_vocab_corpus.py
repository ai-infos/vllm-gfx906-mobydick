#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Emit an MTP draft-vocab counting corpus from local agent traffic.

Sources (each optional, deduped against the others by exact content hash):
  pi api-logs   ~/.pi/agent/api-logs/<date>.jsonl   (needs the vendored pi
                extension, see tools/pi-extensions/README.md)
  pi sessions   ~/.pi/agent/sessions/<proj>/*.jsonl (reaches further back)
  hermes        ~/.hermes/state.db                  (sqlite messages/sessions)

Only *generated* text is emitted (reasoning + response + tool_calls); prompts are
inputs and are not drafted. Parsed logs cannot contain the model's raw format
markup (`<tool_call>`, `</tool_call>`, `<think>`, ...), so any list built from this
corpus MUST force the tokenizer's control-token family - see
check_draft_vocab_list.py (MUST_HAVE_TOKENS) and docs/gfx906/CAT1-corpus-build.md.

Usage:
  .venv/bin/python tools/build_draft_vocab_corpus.py --out corpus.jsonl \
      --models 'rtx5070|Qwen3.6-27B'
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import sqlite3

PI_LOGS = "~/.pi/agent/api-logs/*.jsonl"
PI_SESSIONS = "~/.pi/agent/sessions/*/*.jsonl"
HERMES_DB = "~/.hermes/state.db"


def load_jsonl(path):
    """Yield parsed JSON lines, skipping unparsable ones (live log files can
    carry a partially written last line)."""
    with open(path, errors="ignore") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except Exception:
                continue


def blocks(msg):
    """Split an assistant message into (reasoning[], response[], tool_calls[])."""
    reason, text, tools = [], [], []
    for b in (msg or {}).get("content") or []:
        if not isinstance(b, dict):
            continue
        kind = b.get("type")
        if kind == "thinking" and b.get("thinking"):
            reason.append(b["thinking"])
        elif kind == "text" and b.get("text"):
            text.append(b["text"])
        elif kind == "toolCall":
            tools.append(
                json.dumps(
                    {"name": b.get("name"), "arguments": b.get("arguments")},
                    ensure_ascii=False,
                )
            )
    return reason, text, tools


def record(session_id, ts, provider, model, source, reason, text, tools, domain=""):
    return {
        "session_id": session_id,
        "ts": ts,
        "provider": provider,
        "model": model,
        "source": source,
        "domain": domain,
        "reasoning": "\n".join(reason),
        "response": "\n".join(text),
        "tool_calls": list(tools),
    }


def keep(provider, model, patterns):
    if not patterns:
        return True
    return any(p in f"{provider}|{model}" for p in patterns)


def iter_pi_sessions(patterns):
    """One session file mixes models: attribute every message to the latest
    `model_change` line before it."""
    for path in sorted(glob.glob(os.path.expanduser(PI_SESSIONS))):
        session_id = os.path.basename(path)[:-6]
        project = path.split("/sessions/")[-1].split("/")[0]
        provider = model = "?"
        for d in load_jsonl(path):
            if d.get("type") == "model_change":
                provider = d.get("provider") or "?"
                model = d.get("modelId") or "?"
            elif d.get("type") == "message":
                m = d.get("message") or {}
                if m.get("role") != "assistant":
                    continue
                reason, text, tools = blocks(m)
                if not (reason or text or tools):
                    continue
                if not keep(provider, model, patterns):
                    continue
                yield record(
                    session_id,
                    d.get("timestamp"),
                    provider,
                    model,
                    "session",
                    reason,
                    text,
                    tools,
                    project,
                )


def iter_pi_apilogs(patterns):
    requests = {}
    for path in sorted(glob.glob(os.path.expanduser(PI_LOGS))):
        for d in load_jsonl(path):
            direction = d.get("direction")
            if direction == "request":
                requests[d.get("id")] = "".join(
                    m.get("content") or ""
                    for m in (d.get("payload") or {}).get("messages") or []
                    if isinstance(m, dict) and isinstance(m.get("content"), str)
                )
            elif direction == "response":
                provider, model = d.get("provider") or "?", d.get("model") or "?"
                if not keep(provider, model, patterns):
                    continue
                reason, text, tools = blocks(d.get("message") or {})
                if not (reason or text or tools):
                    continue
                session = d.get("session") or "?"
                project = (
                    session.split("/sessions/")[-1].split("/")[0]
                    if "/sessions/" in session
                    else "?"
                )
                yield record(
                    os.path.basename(session)[:-6],
                    d.get("ts"),
                    provider,
                    model,
                    "apilog",
                    reason,
                    text,
                    tools,
                    project,
                )


def iter_hermes(patterns, db):
    if not os.path.exists(db):
        return
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "select m.session_id, m.timestamp, m.reasoning, m.reasoning_content,"
        "       m.content, m.tool_calls, coalesce(s.model,'?')"
        " from messages m left join sessions s on s.id = m.session_id"
        " where m.role = 'assistant'"
    )
    for sid, ts, reasoning, rcontent, content, tools, model in rows:
        reason = "\n".join(x for x in (reasoning or "", rcontent or "") if x)
        if not keep("hermes", model, patterns):
            continue
        tools_list = [tools] if tools else []
        if not (reason or content or tools_list):
            continue
        yield record(
            sid,
            str(ts),
            "hermes",
            model,
            "hermes",
            [reason],
            [content or ""],
            tools_list,
        )


def emit(records, out_path, dedupe, limit):
    seen = set()
    stats = collections.Counter()
    written = 0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as out:
        for rec in records:
            body = "\x00".join([rec["reasoning"], rec["response"], *rec["tool_calls"]])
            if not body.strip():
                continue
            digest = hashlib.sha1(body.encode()).hexdigest()
            if dedupe and digest in seen:
                stats["duplicates_skipped"] += 1
                continue
            seen.add(digest)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            stats[(rec["source"], rec["model"])] += 1
            if limit and written >= limit:
                break
    print(f"wrote {written} records -> {out_path}")
    for key, count in stats.most_common(20):
        print(f"   {count:8d}  {key}")
    print(
        "NOTE: force the control-token family when counting "
        "(check_draft_vocab_list.py MUST_HAVE_TOKENS); prefer raw-stream capture."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--models",
        default="",
        help="comma-separated 'provider|model' substrings; empty = all",
    )
    ap.add_argument("--sources", default="sessions,apilogs,hermes")
    ap.add_argument("--hermes-db", default=os.path.expanduser(HERMES_DB))
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N records")
    args = ap.parse_args()
    patterns = [p for p in args.models.split(",") if p]
    wanted = set(args.sources.split(","))

    def chain():
        if "sessions" in wanted:
            yield from iter_pi_sessions(patterns)
        if "apilogs" in wanted:
            yield from iter_pi_apilogs(patterns)
        if "hermes" in wanted:
            yield from iter_hermes(patterns, args.hermes_db)

    emit(chain(), args.out, not args.no_dedupe, args.limit)


if __name__ == "__main__":
    main()
