import { appendFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { randomUUID } from "node:crypto";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

const LOG_DIR = join(homedir(), ".pi", "agent", "api-logs");
mkdirSync(LOG_DIR, { recursive: true });
const LOG_FILE = join(LOG_DIR, `${new Date().toISOString().slice(0, 10)}.jsonl`);

const write = (entry: unknown) =>
  appendFileSync(LOG_FILE, JSON.stringify(entry) + "\n");

export default function (pi: ExtensionAPI) {
  // Correlate request ↔ response within a turn. before_provider_request
  // fires right before each LLM call; message_end for the assistant message
  // fires after the stream finishes assembling.
  let pendingRequestId: string | undefined;

  pi.on("before_provider_request", (event, ctx) => {
    pendingRequestId = randomUUID();
    write({
      id: pendingRequestId,
      ts: new Date().toISOString(),
      direction: "request",
      provider: ctx.model?.provider,
      model: ctx.model?.id,
      session: ctx.sessionManager.getSessionFile() ?? null,
      payload: event.payload, // exact provider payload (OpenAI request JSON)
    });
  });

  pi.on("after_provider_response", (event, _ctx) => {
    write({
      id: pendingRequestId,
      ts: new Date().toISOString(),
      direction: "response_headers",
      status: event.status,
      headers: event.headers,
    });
  });

  // Fully assembled assistant message: text + thinking + tool_calls + usage
  // + stopReason + responseId. Equivalent to a parsed OpenAI response body,
  // including the final turn that prior-version replay was dropping.
  pi.on("message_end", (event, ctx) => {
    if (event.message.role !== "assistant") return;
    write({
      id: pendingRequestId,
      ts: new Date().toISOString(),
      direction: "response",
      provider: ctx.model?.provider,
      model: ctx.model?.id,
      session: ctx.sessionManager.getSessionFile() ?? null,
      message: event.message,
    });
    pendingRequestId = undefined;
  });
}