import { describe, expect, it } from "vitest";
import { createTodoTool, normalizeTodos } from "./todo.js";

describe("todo_write", () => {
  it("reports progress and returns the normalized list as details", async () => {
    const result = await createTodoTool().execute(
      "c1",
      { todos: [{ content: "Plan", status: "completed" }, { content: "Build", status: "in_progress" }] },
      undefined,
      undefined,
      undefined as never
    );
    expect(result.content).toEqual([{ type: "text", text: "Task list updated (1 of 2 completed)." }]);
    expect(result.details).toEqual({ todos: [{ content: "Plan", status: "completed" }, { content: "Build", status: "in_progress" }] });
  });

  it("normalizes raw arguments from hosts", () => {
    expect(normalizeTodos([{ content: "a", status: "weird" }, { nope: 1 }, { content: "b", status: "completed" }])).toEqual([
      { content: "a", status: "pending" },
      { content: "b", status: "completed" }
    ]);
    expect(normalizeTodos("not a list")).toEqual([]);
  });
});
