/**
 * Render a request's stored prompt and response as readable, bounded text.
 *
 * The platform stores the request's `messages`, its envelope (`request`: tools, sampling and
 * reasoning settings), and the response the caller received (`{"kind": "json", "body"}` or
 * `{"kind": "sse", "frames"}`). Chat Completions, Responses and Anthropic Messages shapes are
 * read; anything else is shown as JSON. Every part is cut to a budget with a visible marker.
 */

import { count, truncate } from "./format.js";
import type { CapturedContent } from "./requests.js";

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

function json(value: unknown): string {
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

/** The text of one content value: a string, or an array of typed parts. */
export function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return content === undefined || content === null ? "" : json(content);
  return content
    .map((part) => {
      if (typeof part === "string") return part;
      const object = record(part);
      if (object === undefined) return json(part);
      if (typeof object.text === "string") return object.text;
      if (object.type === "tool_use") return `[tool call ${String(object.name)} ${json(object.input)}]`;
      if (object.type === "tool_result") return `[tool result ${contentText(object.content)}]`;
      return `[${String(object.type ?? "part")}]`;
    })
    .join("\n");
}

/** One stored message as `role: text`, with its tool calls. */
export function messageText(message: unknown): string {
  const object = record(message);
  if (object === undefined) return json(message);
  const role = typeof object.role === "string" ? object.role : typeof object.type === "string" ? object.type : "message";
  const parts = [contentText(object.content)];
  if (Array.isArray(object.tool_calls)) {
    for (const call of object.tool_calls) {
      const fn = record(record(call)?.function);
      parts.push(`[tool call ${String(fn?.name ?? "?")} ${typeof fn?.arguments === "string" ? fn.arguments : json(fn?.arguments)}]`);
    }
  }
  if (object.type === "function_call") parts.push(`[tool call ${String(object.name)} ${String(object.arguments ?? "")}]`);
  if (object.type === "function_call_output") parts.push(String(object.output ?? ""));
  const id = typeof object.tool_call_id === "string" ? ` (${object.tool_call_id})` : "";
  return `${role}${id}: ${parts.filter((part) => part !== "").join("\n")}`;
}

/** The assistant's answer from a non-streamed response body of any of the three wire shapes. */
export function responseBodyText(body: unknown): string | undefined {
  const object = record(body);
  if (object === undefined) return undefined;
  const choice = record(Array.isArray(object.choices) ? object.choices[0] : undefined);
  if (choice !== undefined) {
    const message = record(choice.message);
    const text = message === undefined ? "" : messageText({ ...message, role: message.role ?? "assistant" });
    return `${text}${choice.finish_reason ? `\n(finish_reason: ${String(choice.finish_reason)})` : ""}`;
  }
  if (Array.isArray(object.output)) {
    return object.output.map((item) => messageText(record(item)?.type === "message" ? { role: "assistant", content: record(item)?.content } : item)).join("\n");
  }
  if (Array.isArray(object.content) && object.type === "message") {
    return `assistant: ${contentText(object.content)}${object.stop_reason ? `\n(stop_reason: ${String(object.stop_reason)})` : ""}`;
  }
  if (record(object.error) !== undefined || typeof object.error === "string") return `error: ${json(object.error)}`;
  return undefined;
}

/** Reassemble streamed text and tool-call arguments from stored SSE frames. */
export function framesText(frames: unknown[]): string | undefined {
  let text = "";
  const calls = new Map<string, { name: string; args: string }>();
  let seen = false;
  for (const frame of frames) {
    const raw = typeof frame === "string" ? frame : record(frame)?.data;
    let data: Record<string, unknown> | undefined;
    if (typeof raw === "string") {
      const payload = raw.replace(/^data:\s*/, "").trim();
      if (payload === "" || payload === "[DONE]") continue;
      try {
        data = record(JSON.parse(payload));
      } catch {
        continue;
      }
    } else {
      data = record(raw);
    }
    if (data === undefined) continue;
    const delta = record(record(Array.isArray(data.choices) ? data.choices[0] : undefined)?.delta);
    if (delta !== undefined) {
      seen = true;
      if (typeof delta.content === "string") text += delta.content;
      if (Array.isArray(delta.tool_calls)) {
        for (const call of delta.tool_calls) {
          const object = record(call);
          const key = String(object?.index ?? 0);
          const fn = record(object?.function);
          const entry = calls.get(key) ?? { name: "", args: "" };
          if (typeof fn?.name === "string") entry.name += fn.name;
          if (typeof fn?.arguments === "string") entry.args += fn.arguments;
          calls.set(key, entry);
        }
      }
      continue;
    }
    if (data.type === "response.output_text.delta" && typeof data.delta === "string") {
      seen = true;
      text += data.delta;
    } else if (data.type === "content_block_delta") {
      const inner = record(data.delta);
      seen = true;
      if (typeof inner?.text === "string") text += inner.text;
      if (typeof inner?.partial_json === "string") text += inner.partial_json;
    } else if (data.type === "error" || data.error !== undefined) {
      seen = true;
      text += `\n[error ${json(data.error ?? data)}]`;
    }
  }
  if (!seen) return undefined;
  const toolText = [...calls.values()].map((call) => `\n[tool call ${call.name} ${call.args}]`).join("");
  return `assistant: ${text}${toolText}`;
}

/** The response section's text. */
export function responseText(response: unknown): string {
  const object = record(response);
  if (object === undefined) return "(no response stored)";
  if (object.kind === "json") return responseBodyText(object.body) ?? json(object.body);
  if (object.kind === "sse" && Array.isArray(object.frames)) {
    const disconnected = object.client_disconnected === true ? "\n(the caller disconnected before the stream finished)" : "";
    return `${framesText(object.frames) ?? json(object.frames)}${disconnected}`;
  }
  return json(response);
}

/** The request envelope without its messages: tool names and settings. */
export function envelopeText(request: Record<string, unknown> | null | undefined): string | undefined {
  if (!request) return undefined;
  const shown: Record<string, unknown> = {};
  for (const [name, value] of Object.entries(request)) {
    if (name === "messages" || name === "input" || name === "system") continue;
    if (name === "tools" && Array.isArray(value)) {
      shown.tools = value.map((tool) => {
        const object = record(tool);
        return String(record(object?.function)?.name ?? object?.name ?? object?.type ?? "?");
      });
    } else {
      shown[name] = value;
    }
  }
  return Object.keys(shown).length === 0 ? undefined : truncate(json(shown), 2000);
}

/**
 * Keep the first and the most recent messages within `budget` characters, saying how many in
 * the middle were left out.
 */
export function boundedMessages(messages: string[], budget: number): string {
  const total = messages.reduce((sum, text) => sum + text.length + 2, 0);
  if (total <= budget || messages.length <= 1) return truncate(messages.join("\n\n"), budget);
  const first = messages[0]!;
  const head = truncate(first, Math.floor(budget / 4));
  const tail: string[] = [];
  let used = head.length;
  for (let index = messages.length - 1; index >= 1; index -= 1) {
    const text = messages[index]!;
    if (used + text.length + 2 > budget) {
      if (tail.length === 0) tail.unshift(truncate(text, Math.max(200, budget - used)));
      break;
    }
    tail.unshift(text);
    used += text.length + 2;
  }
  const omitted = messages.length - 1 - tail.length;
  const marker = omitted > 0 ? [`[... ${count(omitted)} earlier message${omitted === 1 ? "" : "s"} omitted ...]`] : [];
  return [head, ...marker, ...tail].join("\n\n");
}

/** Render a request's stored content within `maxChars` characters. */
export function renderCapturedContent(content: CapturedContent, maxChars: number): string {
  const messageBudget = Math.floor(maxChars * 0.6);
  const responseBudget = maxChars - messageBudget;
  const perMessage = Math.max(500, Math.floor(messageBudget / 3));
  const messages = (Array.isArray(content.messages) ? content.messages : []).map((message, index) => `[${index + 1}] ${truncate(messageText(message), perMessage)}`);
  const sections: string[] = [];
  const envelope = envelopeText(content.request ?? undefined);
  if (envelope !== undefined) sections.push(`Request settings: ${envelope}`);
  sections.push(`Prompt (${count(messages.length)} message${messages.length === 1 ? "" : "s"}, stored ${content.captured_at}):\n${boundedMessages(messages, messageBudget)}`);
  sections.push(
    content.response === undefined || content.response === null
      ? "Response: not stored."
      : `Response${content.response_captured_at ? ` (stored ${content.response_captured_at})` : ""}:\n${truncate(responseText(content.response), responseBudget)}`
  );
  return sections.join("\n\n");
}
