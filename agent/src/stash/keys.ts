/**
 * Which keys stash the prompt. Ctrl+S by default; rebind or turn it off in the same
 * `keybindings.json` that holds every other key (`~/.exp-code/agent/keybindings.json`):
 *
 *     { "exp-code.prompt.stash": "alt+s" }          another key
 *     { "exp-code.prompt.stash": ["ctrl+s", "f2"] }  several keys
 *     { "exp-code.prompt.stash": [] }               no key (use /stash)
 *
 * Pi ignores ids it does not know, so the entry lives beside Pi's own without conflict.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

/** The keybindings.json id of the stash key. */
export const STASH_KEYBINDING_ID = "exp-code.prompt.stash";

/** The default stash key, as in Claude Code. */
export const DEFAULT_STASH_KEYS: readonly string[] = ["ctrl+s"];

const KEY_PATTERN = /^(?:(?:ctrl|shift|alt|super)\+)*[^\s+]+$/i;

/**
 * The stash keys configured in `<agentDir>/keybindings.json`, or the default when the file is
 * missing, unreadable, or does not set a valid value.
 */
export function loadStashKeys(agentDir: string): string[] {
  let raw: unknown;
  try {
    raw = JSON.parse(readFileSync(join(agentDir, "keybindings.json"), "utf8").replace(/^\uFEFF/, ""));
  } catch {
    return [...DEFAULT_STASH_KEYS];
  }
  if (typeof raw !== "object" || raw === null || !(STASH_KEYBINDING_ID in raw)) return [...DEFAULT_STASH_KEYS];
  return parseStashKeys((raw as Record<string, unknown>)[STASH_KEYBINDING_ID]) ?? [...DEFAULT_STASH_KEYS];
}

/** A binding value as a key list, or `undefined` when it is not a string or list of key ids. */
export function parseStashKeys(value: unknown): string[] | undefined {
  const keys = typeof value === "string" ? [value] : Array.isArray(value) ? value : undefined;
  if (keys === undefined || !keys.every((key): key is string => typeof key === "string" && KEY_PATTERN.test(key.trim()))) return undefined;
  return keys.map((key) => key.trim().toLowerCase());
}

/** A key id as people write it: `ctrl+s` becomes `Ctrl+S`. */
export function keyLabel(key: string): string {
  return key
    .split("+")
    .map((part) => (part.length <= 1 ? part.toUpperCase() : part[0]!.toUpperCase() + part.slice(1)))
    .join("+");
}
