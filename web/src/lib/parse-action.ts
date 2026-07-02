/**
 * Mirror of `wmh play`'s forgiving action grammar (wmh/engine/play.py:parse_action):
 *   - `get_user {"id": "u1"}` -> tool call with JSON arguments
 *   - `list_flights`          -> tool call, no arguments
 *   - `say hello there`       -> free-text message
 *   - anything else           -> free-text message
 */

import type { Action } from "./types";

function looksLikeToolName(token: string): boolean {
  // ASCII identifier-ish: [A-Za-z][A-Za-z0-9._-]* (same rule as the Python side).
  return /^[A-Za-z][A-Za-z0-9._-]*$/.test(token);
}

export function parseAction(line: string): Action {
  const text = line.trim();
  if (!text) throw new Error("empty action");

  if (text.startsWith("say ")) {
    return { kind: "message", content: text.slice(4).trim() };
  }

  const spaceAt = text.indexOf(" ");
  const head = spaceAt === -1 ? text : text.slice(0, spaceAt);
  const rest = spaceAt === -1 ? "" : text.slice(spaceAt + 1).trim();

  if (looksLikeToolName(head) && (!rest || rest.startsWith("{") || rest.startsWith("["))) {
    let args: Record<string, unknown> = {};
    if (rest) {
      try {
        const parsed: unknown = JSON.parse(rest);
        if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error("arguments must be a JSON object");
        }
        args = parsed as Record<string, unknown>;
      } catch (e) {
        throw new Error(
          `could not parse arguments as JSON: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    }
    return { kind: "tool_call", name: head, arguments: args };
  }
  return { kind: "message", content: text };
}
