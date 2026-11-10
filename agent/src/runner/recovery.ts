/**
 * Continuing a run after the runner process died in the middle of it.
 *
 * Pi's session file holds every finished message, so after a crash the transcript can end with
 * an assistant message whose tool calls never got results. Each tool declares whether it is
 * replay-safe (read-only, or idempotent like `todo_write`): replay-safe calls are run again and
 * their results recorded; any other call gets an error result saying it was interrupted, so the
 * model checks the state before repeating it. A call replayed too many times is not run again.
 */

import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import type { SessionManager } from "@earendil-works/pi-coding-agent";
import type { RunnerState } from "./state.js";

/** Built-in Pi tools that only read. */
export const READ_ONLY_BUILTINS = new Set(["read", "ls", "grep", "find"]);

/** What the model sees after a restart, when its prompt was already in the session. */
export const RESUME_NOTE =
  "(The agent runner restarted while working on this request. Continue the request from where it stopped. Interrupted tool calls are marked in their results.)";

export const INTERRUPTED_TOOL_TEXT =
  "This tool call was interrupted because the agent runner restarted. Its effect is unknown: check the current state before running it again.";

/** Whether a tool may be run again after a crash interrupted it. */
export function isReplaySafe(name: string, definition: Pick<ToolDefinition, "annotations"> | undefined): boolean {
  if (READ_ONLY_BUILTINS.has(name)) return true;
  const hints = definition?.annotations;
  return hints?.readOnlyHint === true || hints?.idempotentHint === true;
}

interface DanglingCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

/** Tool calls of the last assistant message that have no result in the session. */
export function danglingToolCalls(sessionManager: SessionManager): DanglingCall[] {
  const messages = sessionManager.buildSessionContext().messages;
  let lastAssistant = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]!.role === "assistant") {
      lastAssistant = index;
      break;
    }
  }
  if (lastAssistant < 0) return [];
  const assistant = messages[lastAssistant]!;
  if (assistant.role !== "assistant") return [];
  const answered = new Set<string>();
  for (const message of messages.slice(lastAssistant + 1)) {
    if (message.role === "toolResult") answered.add(message.toolCallId);
  }
  return assistant.content
    .filter((part) => part.type === "toolCall" && !answered.has(part.id))
    .map((part) => (part.type === "toolCall" ? { id: part.id, name: part.name, args: part.arguments as Record<string, unknown> } : { id: "", name: "", args: {} }));
}

export interface RepairOptions {
  sessionManager: SessionManager;
  state: RunnerState;
  /** Tools the runner can execute outside a session, by name. */
  tools: Map<string, ToolDefinition>;
  /** Most replays of one call before it is reported as interrupted instead. */
  replayLimit: number;
  now: () => number;
}

/** Record a result for every dangling tool call. Returns the ids replayed and interrupted. */
export async function repairDanglingToolCalls(options: RepairOptions): Promise<{ replayed: string[]; interrupted: string[] }> {
  const replayed: string[] = [];
  const interrupted: string[] = [];
  for (const call of danglingToolCalls(options.sessionManager)) {
    const definition = options.tools.get(call.name);
    const attempts = options.state.replays[call.id] ?? 0;
    let content: { type: "text"; text: string }[] = [{ type: "text", text: INTERRUPTED_TOOL_TEXT }];
    let isError = true;
    if (definition !== undefined && isReplaySafe(call.name, definition) && attempts < options.replayLimit) {
      options.state.replays[call.id] = attempts + 1;
      try {
        // Replay-safe tools take everything they need from their own options, not the session context.
        const result = await definition.execute(call.id, call.args as never, undefined, undefined, undefined as never);
        content = result.content.map((part) => (part.type === "text" ? part : { type: "text", text: `[${part.type}]` }));
        isError = (result as { isError?: boolean }).isError === true;
        replayed.push(call.id);
      } catch (error) {
        content = [{ type: "text", text: `Replaying this call after a runner restart failed: ${(error as Error).message}` }];
        interrupted.push(call.id);
      }
    } else {
      interrupted.push(call.id);
    }
    options.sessionManager.appendMessage({ role: "toolResult", toolCallId: call.id, toolName: call.name, content, isError, timestamp: options.now() });
  }
  return { replayed, interrupted };
}
