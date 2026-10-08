# Copyright Kevin Read <me@kevin-read.com>
"""n-gram speculative-decode A/B on dense Qwen3.5-27B-AWQ (gfx906).

Runs against a locally-started vLLM OpenAI server (baseline arm and
ngram arm use the SAME server flags except --speculative-config).
For each prompt: POST /v1/chat/completions (greedy, 512 max tokens),
time it, then diff the vllm:spec_decode_num_* Prometheus counters for
the acceptance breakdown.

Usage:
  python spec_ngram_dense.py --port 8931 --arm baseline
  python spec_ngram_dense.py --port 8932 --arm ngram3
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request

BASE = None
OUT_TOKENS = 512

SYSTEM = (
    "You are a coding agent with access to tools. To call a tool, reply "
    "with a single JSON object of the form {\"tool\": \"<name>\", "
    "\"arguments\": {<key>: <value>}}. Available tools: read_file, "
    "write_file, run_tests. Wait for tool results before acting. Keep "
    "explanations brief; code changes go through write_file calls."
)

PROMPTS = [
    # --- 1: bug fix; traceback repeats the buggy code -------------------
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": ("Our nightly billing job produced wrong totals for "
                     "subscription invoices. The customer disputes tax on "
                     "discounted line items. Investigate and fix.")},
        {"role": "assistant",
         "content": ("{\"tool\": \"read_file\", \"arguments\": "
                     "{\"path\": \"billing/invoice.py\"}}")},
        {"role": "user", "content": (
            "Tool result for read_file(billing/invoice.py):\n"
            "```python\n"
            "from decimal import Decimal, ROUND_HALF_UP\n"
            "\n"
            "TAX_RATES = {\n"
            "    \"US-CA\": Decimal(\"0.0875\"),\n"
            "    \"US-NY\": Decimal(\"0.08875\"),\n"
            "    \"DE-BE\": Decimal(\"0.19\"),\n"
            "}\n"
            "\n"
            "\n"
            "class Invoice:\n"
            "    def __init__(self, invoice_id, customer_id, region):\n"
            "        self.invoice_id = invoice_id\n"
            "        self.customer_id = customer_id\n"
            "        self.region = region\n"
            "        self.line_items = []\n"
            "\n"
            "    def add_line_item(self, description, quantity, unit_price,\n"
            "                      discount_percent=Decimal(\"0\")):\n"
            "        unit_price = Decimal(str(unit_price))\n"
            "        quantity = int(quantity)\n"
            "        gross = unit_price * quantity\n"
            "        discount_percent = Decimal(str(discount_percent))\n"
            "        net = gross - (gross * discount_percent / Decimal(\"100\"))\n"
            "        self.line_items.append({\n"
            "            \"description\": description,\n"
            "            \"quantity\": quantity,\n"
            "            \"unit_price\": unit_price,\n"
            "            \"gross\": gross,\n"
            "            \"discount_percent\": discount_percent,\n"
            "            \"net\": net,\n"
            "        })\n"
            "\n"
            "    def calculate_totals(self):\n"
            "        subtotal = sum(item[\"net\"] for item in self.line_items)\n"
            "        tax_rate = TAX_RATES.get(self.region, Decimal(\"0\"))\n"
            "        tax = (subtotal * tax_rate).quantize(\n"
            "            Decimal(\"0.01\"), rounding=ROUND_HALF_UP)\n"
            "        total = subtotal + tax\n"
            "        return {\"subtotal\": subtotal, \"tax\": tax, \"total\": total}\n"
            "\n"
            "    def to_dict(self):\n"
            "        return {\n"
            "            \"invoice_id\": self.invoice_id,\n"
            "            \"customer_id\": self.customer_id,\n"
            "            \"region\": self.region,\n"
            "            \"line_items\": self.line_items,\n"
            "            **self.calculate_totals(),\n"
            "        }\n"
            "```\n"
            "Then I ran: {\"tool\": \"run_tests\", \"arguments\": "
            "{\"path\": \"tests/test_invoice.py\"}}\n"
            "Tool result for run_tests:\n"
            "```\n"
            "FAILED tests/test_invoice.py::test_discounted_line_tax\n"
            "    assert tax == Decimal(\"8.75\")\n"
            "E   AssertionError: assert Decimal('9.63') == Decimal('8.75')\n"
            "\n"
            "    def calculate_totals(self):\n"
            "        subtotal = sum(item[\"net\"] for item in self.line_items)\n"
            "        tax_rate = TAX_RATES.get(self.region, Decimal(\"0\"))\n"
            "        tax = (subtotal * tax_rate).quantize(\n"
            "            Decimal(\"0.01\"), rounding=ROUND_HALF_UP)\n"
            "```\n"
            "The test builds an invoice with one line item: quantity 100, "
            "unit price 10.00, discount 20%, region US-CA. Expected: net "
            "80.00 per the finance spec, tax 6.82 at 8.75%. What is the "
            "root cause and how should calculate_totals be fixed? Reply "
            "with the corrected calculate_totals method and a two-sentence "
            "explanation.")},
    ],
    # --- 2: add a third function by analogy ------------------------------
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": ("We need a Markdown export next to the existing CSV "
                     "and JSON exporters in reports/exporter.py. "
                     "Implement it in the same style and register it.")},
        {"role": "assistant",
         "content": ("{\"tool\": \"read_file\", \"arguments\": "
                     "{\"path\": \"reports/exporter.py\"}}")},
        {"role": "user", "content": (
            "Tool result for read_file(reports/exporter.py):\n"
            "```python\n"
            "import csv\n"
            "import json\n"
            "from datetime import datetime\n"
            "from typing import Any\n"
            "\n"
            "_FORMATS = {}\n"
            "\n"
            "\n"
            "def register(fmt: str):\n"
            "    def wrap(fn):\n"
            "        _FORMATS[fmt] = fn\n"
            "        return fn\n"
            "    return wrap\n"
            "\n"
            "\n"
            "def _rows_from_report(report: dict[str, Any]):\n"
            "    rows = []\n"
            "    for metric in report[\"metrics\"]:\n"
            "        rows.append({\n"
            "            \"name\": metric[\"name\"],\n"
            "            \"value\": metric[\"value\"],\n"
            "            \"unit\": metric.get(\"unit\", \"\"),\n"
            "            \"window\": report[\"window\"],\n"
            "        })\n"
            "    return rows\n"
            "\n"
            "\n"
            "@register(\"csv\")\n"
            "def export_to_csv(report: dict[str, Any], out_path: str) -> None:\n"
            "    rows = _rows_from_report(report)\n"
            "    fieldnames = [\"name\", \"value\", \"unit\", \"window\"]\n"
            "    with open(out_path, \"w\", newline=\"\") as fh:\n"
            "        writer = csv.DictWriter(fh, fieldnames=fieldnames)\n"
            "        writer.writeheader()\n"
            "        for row in rows:\n"
            "            writer.writerow(row)\n"
            "\n"
            "\n"
            "@register(\"json\")\n"
            "def export_to_json(report: dict[str, Any], out_path: str) -> None:\n"
            "    rows = _rows_from_report(report)\n"
            "    payload = {\n"
            "        \"window\": report[\"window\"],\n"
            "        \"generated_at\": datetime.utcnow().isoformat() + \"Z\",\n"
            "        \"metrics\": rows,\n"
            "    }\n"
            "    with open(out_path, \"w\") as fh:\n"
            "        json.dump(payload, fh, indent=2, sort_keys=True)\n"
            "\n"
            "\n"
            "def export_report(report: dict[str, Any], fmt: str,\n"
            "                  out_path: str) -> None:\n"
            "    try:\n"
            "        fn = _FORMATS[fmt]\n"
            "    except KeyError:\n"
            "        raise ValueError(f\"unknown format: {fmt!r}\") from None\n"
            "    fn(report, out_path)\n"
            "```\n"
            "Write the full new export_to_markdown function (with "
            "@register(\"markdown\")) that renders a header with the "
            "window, a generated_at line, and a pipe table with columns "
            "Name | Value | Unit, plus the write_file call that adds it "
            "to the file.")},
    ],
    # --- 3: refactor + tests; tests reuse the API verbatim ---------------
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": ("session.py leaks sockets when handlers raise. "
                     "Convert Session to a context manager and add "
                     "regression tests in tests/test_session.py in the "
                     "existing style.")},
        {"role": "assistant",
         "content": ("{\"tool\": \"read_file\", \"arguments\": "
                     "{\"path\": \"gateway/session.py\"}}")},
        {"role": "user", "content": (
            "Tool result for read_file(gateway/session.py):\n"
            "```python\n"
            "import socket\n"
            "import logging\n"
            "\n"
            "logger = logging.getLogger(__name__)\n"
            "\n"
            "\n"
            "class Session:\n"
            "    def __init__(self, host: str, port: int, timeout: float = 5.0):\n"
            "        self.host = host\n"
            "        self.port = port\n"
            "        self.timeout = timeout\n"
            "        self.sock: socket.socket | None = None\n"
            "        self.closed = False\n"
            "\n"
            "    def connect(self) -> None:\n"
            "        self.sock = socket.create_connection(\n"
            "            (self.host, self.port), timeout=self.timeout)\n"
            "        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)\n"
            "        logger.info(\"session open %s:%d\", self.host, self.port)\n"
            "\n"
            "    def send_frame(self, payload: bytes) -> None:\n"
            "        if self.sock is None or self.closed:\n"
            "            raise RuntimeError(\"session is not open\")\n"
            "        header = payload[:4]\n"
            "        self.sock.sendall(header + payload)\n"
            "\n"
            "    def close(self) -> None:\n"
            "        if self.closed or self.sock is None:\n"
            "            return\n"
            "        try:\n"
            "            self.sock.close()\n"
            "        finally:\n"
            "            self.sock = None\n"
            "            self.closed = True\n"
            "            logger.info(\"session closed %s:%d\", self.host, self.port)\n"
            "```\n"
            "Tool result for read_file(tests/test_session.py) [excerpt]:\n"
            "```python\n"
            "import pytest\n"
            "from gateway.session import Session\n"
            "\n"
            "\n"
            "def test_send_frame_requires_open_session():\n"
            "    session = Session(\"127.0.0.1\", 9000)\n"
            "    with pytest.raises(RuntimeError, match=\"session is not open\"):\n"
            "        session.send_frame(b\"\\x00\\x00\\x00\\x01hi\")\n"
            "\n"
            "\n"
            "def test_close_is_idempotent():\n"
            "    session = Session(\"127.0.0.1\", 9000)\n"
            "    session.close()\n"
            "    session.close()\n"
            "    assert session.closed is True\n"
            "    assert session.sock is None\n"
            "```\n"
            "Reply with: (1) the __enter__/__exit__ methods to add to "
            "Session, (2) the updated connect() so double-connect is "
            "safe, and (3) two new pytest functions: "
            "test_context_manager_closes_on_exception and "
            "test_context_manager_normal_close, matching the style of "
            "the existing tests.")},
    ],
]


def get_json(url, payload=None):
    req = urllib.request.Request(
        url, data=None if payload is None
        else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="GET"
        if payload is None else "POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def spec_counters(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",
                                timeout=30) as r:
        text = r.read().decode()
    want = ("vllm:spec_decode_num_drafts_total{",
            "vllm:spec_decode_num_draft_tokens_total{",
            "vllm:spec_decode_num_accepted_tokens_total{")
    out = {}
    for line in text.splitlines():
        for w in want:
            if line.startswith(w):
                name = line.split("{")[0]
                out[name] = float(line.rsplit(" ", 1)[1])
    return out


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--outdir", default="/tmp/spec_texts",
                    help="dir for per-prompt completion text dumps "
                         "(token-identity check vs another arm)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat the whole prompt sweep N times so the "
                         "summary has enough samples for a real CI "
                         "(roadmap Phase 0 gate: mean + 95%% CI lower "
                         "bound vs the baseline band, not a flat t/s "
                         "threshold — the baseline band is ~4%% wide)")
    args = ap.parse_args()
    assert args.repeats >= 1
    os.makedirs(args.outdir, exist_ok=True)
    BASE = f"http://127.0.0.1:{args.port}"

    # Warmup: trigger tokenizer/capture paths, unmeasured.
    get_json(BASE + "/v1/chat/completions", {
        "model": "qwen27", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8, "temperature": 0})

    results = []
    for rep in range(args.repeats):
        for i, msgs in enumerate(PROMPTS):
            before = spec_counters(args.port)
            t0 = time.perf_counter()
            resp = get_json(BASE + "/v1/chat/completions", {
                "model": "qwen27", "messages": msgs,
                "max_tokens": OUT_TOKENS, "temperature": 0})
            dt = time.perf_counter() - t0
            after = spec_counters(args.port)
            n_out = resp["usage"]["completion_tokens"]
            text = resp["choices"][0]["message"]["content"]
            text_sha = hashlib.sha256(text.encode()).hexdigest()[:16]
            if rep == args.repeats - 1:
                with open(f"{args.outdir}/spec_texts_{args.arm}_{i}.txt",
                          "w") as fh:
                    fh.write(text)
            drafts = after.get("vllm:spec_decode_num_drafts_total", 0) \
                - before.get("vllm:spec_decode_num_drafts_total", 0)
            draft_tok = after.get("vllm:spec_decode_num_draft_tokens_total",
                                  0) \
                - before.get("vllm:spec_decode_num_draft_tokens_total", 0)
            acc = after.get("vllm:spec_decode_num_accepted_tokens_total",
                            0) \
                - before.get("vllm:spec_decode_num_accepted_tokens_total",
                             0)
            rec = {
                "rep": rep, "prompt": i, "arm": args.arm,
                "out_tokens": n_out,
                "elapsed_s": round(dt, 3),
                "tokens_per_s": round(n_out / dt, 3),
                "drafts": drafts, "draft_tokens": draft_tok,
                "accepted_tokens": acc, "text_sha": text_sha,
                "accept_rate_pct": round(100.0 * acc / draft_tok, 2)
                if draft_tok else None,
                "accepted_per_step": round(acc / drafts, 3)
                if drafts else None,
            }
            results.append(rec)
            print(json.dumps(rec), flush=True)

    tps = [r["tokens_per_s"] for r in results]
    n = len(tps)
    mean = sum(tps) / n
    sd = (sum((x - mean) ** 2 for x in tps) / max(1, n - 1)) ** 0.5
    ci95_lo = mean - 1.96 * sd / (n ** 0.5) if n > 1 else mean
    tot_tok = sum(r["out_tokens"] for r in results)
    tot_t = sum(r["elapsed_s"] for r in results)
    print("SUMMARY " + args.arm + " "
          + json.dumps({
              "n": n, "mean_tps": round(mean, 3),
              "sd_tps": round(sd, 3),
              "ci95_lower_tps": round(ci95_lo, 3),
              "aggregate_tps": round(tot_tok / tot_t, 3),
              "mean_accept_rate_pct": round(
                  sum(r["accepted_tokens"] for r in results)
                  / max(1, sum(r["draft_tokens"] for r in results)) * 100, 2),
              "mean_accepted_per_step": round(
                  sum(r["accepted_tokens"] for r in results)
                  / max(1, sum(r["drafts"] for r in results)), 3),
          }), flush=True)


if __name__ == "__main__":
    main()
