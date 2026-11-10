import { describe, expect, it } from "vitest";
import { createTodoTool } from "../extension/todo.js";
import { createWebFetchTool } from "../extension/web-fetch.js";
import { isReplaySafe } from "./recovery.js";

describe("replay safety", () => {
  it("treats read-only built-ins and tools declaring read-only or idempotent hints as replay-safe", () => {
    expect(isReplaySafe("read", undefined)).toBe(true);
    expect(isReplaySafe("grep", undefined)).toBe(true);
    expect(isReplaySafe("todo_write", createTodoTool())).toBe(true);
    expect(isReplaySafe("web_fetch", createWebFetchTool({ shell: { exec: async () => ({ exitCode: 0 }) } }))).toBe(true);
  });

  it("never replays tools with side effects or no declaration", () => {
    expect(isReplaySafe("bash", undefined)).toBe(false);
    expect(isReplaySafe("write", { annotations: { readOnlyHint: false } })).toBe(false);
    expect(isReplaySafe("custom", {})).toBe(false);
  });
});
