import { describe, expect, it } from "vitest";
import { boundedMessages, contentText, envelopeText, framesText, messageText, renderCapturedContent, responseText } from "./transcript.js";

describe("stored content rendering", () => {
  it("reads Chat, Responses and Messages message shapes", () => {
    expect(messageText({ role: "user", content: [{ type: "text", text: "hi" }, { type: "image_url", image_url: {} }] })).toBe("user: hi\n[image_url]");
    expect(messageText({ role: "assistant", content: null, tool_calls: [{ function: { name: "read", arguments: '{"path":"a"}' } }] })).toBe('assistant: [tool call read {"path":"a"}]');
    expect(messageText({ role: "tool", tool_call_id: "t1", content: "file text" })).toBe("tool (t1): file text");
    expect(messageText({ type: "function_call", name: "bash", arguments: "{}" })).toBe("function_call: [tool call bash {}]");
    expect(contentText([{ type: "tool_use", name: "grep", input: { q: 1 } }, { type: "tool_result", content: "ok" }])).toBe('[tool call grep {"q":1}]\n[tool result ok]');
  });

  it("reads non-streamed responses of each wire", () => {
    expect(responseText({ kind: "json", body: { output: [{ type: "message", content: [{ type: "output_text", text: "Hi" }] }] } })).toBe("assistant: Hi");
    expect(responseText({ kind: "json", body: { type: "message", content: [{ type: "text", text: "Yo" }], stop_reason: "end_turn" } })).toBe("assistant: Yo\n(stop_reason: end_turn)");
    expect(responseText({ kind: "json", body: { error: { message: "bad" } } })).toBe('error: {"message":"bad"}');
    expect(responseText({ kind: "json", body: { other: 1 } })).toBe('{"other":1}');
    expect(responseText(null)).toBe("(no response stored)");
  });

  it("reassembles streamed text and tool calls from frames", () => {
    const chat = [
      'data: {"choices":[{"delta":{"content":"Hel"}}]}',
      { data: '{"choices":[{"delta":{"content":"lo","tool_calls":[{"index":0,"function":{"name":"read","arguments":"{\\"p\\""}}]}}]}' },
      { data: { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: ":1}" } }] } }] } },
      "data: [DONE]"
    ];
    expect(framesText(chat)).toBe('assistant: Hello\n[tool call read {"p":1}]');
    expect(framesText([{ data: '{"type":"response.output_text.delta","delta":"A"}' }, { data: '{"type":"content_block_delta","delta":{"text":"B"}}' }])).toBe("assistant: AB");
    expect(framesText(["noise"])).toBeUndefined();
    expect(responseText({ kind: "sse", frames: chat.slice(0, 1), client_disconnected: true })).toContain("the caller disconnected");
  });

  it("summarizes the request envelope without the messages", () => {
    expect(envelopeText({ messages: [1], tools: [{ name: "a" }, { function: { name: "b" } }], max_tokens: 5 })).toBe('{"tools":["a","b"],"max_tokens":5}');
    expect(envelopeText({ messages: [] })).toBeUndefined();
    expect(envelopeText(null)).toBeUndefined();
  });

  it("keeps the first and latest messages within budget", () => {
    const messages = Array.from({ length: 20 }, (_, index) => `message ${index} ${"y".repeat(80)}`);
    const text = boundedMessages(messages, 600);
    expect(text.startsWith("message 0")).toBe(true);
    expect(text).toContain("message 19");
    expect(text).toMatch(/\[\.\.\. \d+ earlier messages omitted \.\.\.\]/);
    expect(text.length).toBeLessThan(800);
    expect(boundedMessages(["a", "b"], 100)).toBe("a\n\nb");
  });

  it("bounds the whole rendering", () => {
    const content = {
      request_id: "r",
      captured_at: "t",
      messages: Array.from({ length: 50 }, () => ({ role: "user", content: "z".repeat(2000) })),
      response: { kind: "json", body: { choices: [{ message: { content: "w".repeat(10_000) } }] } }
    };
    const text = renderCapturedContent(content, 4000);
    expect(text.length).toBeLessThan(5500);
    expect(text).toContain("Prompt (50 messages, stored t)");
    expect(text).toContain("truncated");
    expect(renderCapturedContent({ request_id: "r", captured_at: "t", messages: [] }, 1000)).toContain("Response: not stored.");
  });
});
