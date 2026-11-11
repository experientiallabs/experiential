/**
 * `todo_write`: a visible task list the agent keeps for multi-step work.
 *
 * The whole list is sent on every call and replaces the previous one, so the latest call is the
 * complete state. Hosts that render progress (the headless runner, RPC clients) read it from the
 * tool's `details`.
 */

import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export type TodoStatus = "pending" | "in_progress" | "completed";

export interface TodoItem {
  content: string;
  status: TodoStatus;
}

const todoSchema = Type.Object({
  todos: Type.Array(
    Type.Object({
      content: Type.String({ description: "The task, in a short imperative sentence" }),
      status: Type.Union([Type.Literal("pending"), Type.Literal("in_progress"), Type.Literal("completed")], {
        description: "pending, in_progress (exactly one at a time), or completed"
      })
    }),
    { description: "The whole task list. Replaces the previous list." }
  )
});

/** Normalize an arbitrary `todos` argument into well-formed items (used by hosts reading raw args). */
export function normalizeTodos(value: unknown): TodoItem[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is { content: string; status?: unknown } => typeof (item as { content?: unknown })?.content === "string")
    .map((item) => ({
      content: item.content,
      status: item.status === "completed" || item.status === "in_progress" ? item.status : "pending"
    }));
}

export function createTodoTool() {
  return defineTool({
    name: "todo_write",
    label: "Task list",
    description:
      "Create or update your task list for the current request. Use it for any request with three or more steps: write the plan first, keep exactly one task in_progress, and mark tasks completed as soon as they are done. The user sees this list live. Send the whole list every time.",
    promptSnippet: "Keep a visible task list for multi-step work",
    promptGuidelines: [
      "Start any request with three or more steps with todo_write: write the plan first, mark each task in_progress when you start it and completed as soon as it is done."
    ],
    parameters: todoSchema,
    executionMode: "sequential",
    annotations: { readOnlyHint: false, idempotentHint: true },
    async execute(_toolCallId, { todos }) {
      const done = todos.filter((todo) => todo.status === "completed").length;
      return {
        content: [{ type: "text", text: `Task list updated (${done} of ${todos.length} completed).` }],
        details: { todos: normalizeTodos(todos) }
      };
    }
  });
}
