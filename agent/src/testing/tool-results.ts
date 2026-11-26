/** Shared helpers for the platform tool tests. */

export type ToolOutcome = { content: { type: string; text?: string }[]; details: Record<string, unknown>; isError?: boolean };

/** Run a tool's execute with arguments, as Pi would. */
export function exec(tool: { execute: (...args: never[]) => unknown }, args: Record<string, unknown>): Promise<ToolOutcome> {
  return (tool.execute as (id: string, args: unknown, signal?: AbortSignal) => Promise<ToolOutcome>)("c", args);
}

/** The text a tool result shows the model. */
export function text(result: ToolOutcome): string {
  return result.content.map((part) => part.text ?? "").join("");
}
