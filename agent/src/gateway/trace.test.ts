import { describe, expect, it } from "vitest";
import { conversationTraceHeaders } from "./trace.js";

describe("conversationTraceHeaders", () => {
  it("names the conversation as the session and tags the call with its conversation and turn", () => {
    const headers = conversationTraceHeaders({ conversationId: "c-1", turnId: "r-1" });
    expect(headers["X-Explabs-Session-Id"]).toBe("c-1");
    expect(JSON.parse(headers["X-Explabs-Tags"]!)).toEqual({ conversation: "c-1", turn: "r-1" });
  });
});
