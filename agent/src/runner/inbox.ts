/**
 * Reads steering and stop commands from inbox.jsonl.
 *
 * Only complete lines are read, from a persisted byte offset, so a restarted runner never
 * re-applies a command. A command with a `requestId` that was already applied is ignored, which
 * makes retried appends from the control plane safe.
 */

import { existsSync, readFileSync } from "node:fs";
import type { InboxCommand } from "./protocol.js";
import type { RunnerState } from "./state.js";

/** Parse one inbox line, or null for anything that is not a known command. */
export function parseInboxCommand(line: string): InboxCommand | null {
  try {
    const value = JSON.parse(line) as { type?: unknown; text?: unknown; requestId?: unknown };
    const requestId = typeof value.requestId === "string" && value.requestId !== "" ? value.requestId : undefined;
    if (value.type === "stop") return requestId === undefined ? { type: "stop" } : { type: "stop", requestId };
    if (value.type === "steer" && typeof value.text === "string") {
      return requestId === undefined ? { type: "steer", text: value.text } : { type: "steer", text: value.text, requestId };
    }
  } catch {
    // A torn or foreign line.
  }
  return null;
}

/**
 * Read commands appended since the state's offset, advancing it, and drop duplicates by
 * requestId. Returns whether the state changed.
 */
export function readInbox(path: string, state: RunnerState): { commands: InboxCommand[]; changed: boolean } {
  if (!existsSync(path)) return { commands: [], changed: false };
  const bytes = readFileSync(path);
  if (bytes.length <= state.inboxOffset) return { commands: [], changed: false };
  const fresh = bytes.subarray(state.inboxOffset);
  const lastNewline = fresh.lastIndexOf(0x0a);
  if (lastNewline < 0) return { commands: [], changed: false };
  state.inboxOffset += lastNewline + 1;
  const applied = new Set(state.appliedRequestIds);
  const commands: InboxCommand[] = [];
  for (const line of fresh.subarray(0, lastNewline).toString("utf8").split("\n")) {
    const command = parseInboxCommand(line);
    if (command === null) continue;
    if (command.requestId !== undefined) {
      if (applied.has(command.requestId)) continue;
      applied.add(command.requestId);
      state.appliedRequestIds.push(command.requestId);
    }
    commands.push(command);
  }
  return { commands, changed: true };
}
