/**
 * The default model for a CLI invocation.
 *
 * When the user names no model and is not resuming a session, exp-code starts on a gateway
 * model: `EXP_CODE_MODEL` when set, otherwise the first alias the gateway lists. A model saved
 * in exp-code's settings (chosen earlier with `/model`) is left to Pi.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Args } from "@earendil-works/pi-coding-agent";
import { GATEWAY_PROVIDER_ID } from "../gateway/config.js";

/** Pi subcommands that take no model. */
const SUBCOMMANDS = new Set(["auth", "install", "remove", "uninstall", "update", "list", "config", "mcp"]);

/** Whether this invocation should get an injected `--model`. */
export function wantsDefaultModel(args: string[], parsed: Args, agentDir: string): boolean {
  if (args[0] !== undefined && SUBCOMMANDS.has(args[0])) return false;
  if (parsed.help || parsed.version || parsed.export !== undefined || parsed.listModels !== undefined) return false;
  if (parsed.model !== undefined || parsed.provider !== undefined || (parsed.models?.length ?? 0) > 0) return false;
  if (parsed.continue || parsed.resume || parsed.session !== undefined || parsed.sessionId !== undefined || parsed.fork !== undefined) return false;
  return savedDefaultModel(agentDir) === undefined;
}

/** The `defaultModel` saved in exp-code's global settings, if any. */
export function savedDefaultModel(agentDir: string): string | undefined {
  try {
    const settings = JSON.parse(readFileSync(join(agentDir, "settings.json"), "utf8")) as { defaultModel?: unknown };
    return typeof settings.defaultModel === "string" && settings.defaultModel !== "" ? settings.defaultModel : undefined;
  } catch {
    return undefined;
  }
}

/** Pick the alias: the configured one when set, else the first listed. */
export function defaultModelId(listed: string[], configured: string | undefined): string | undefined {
  const wanted = configured?.trim();
  if (wanted) return wanted;
  return listed[0];
}

/** The `--model` argument pair for a gateway alias. */
export function modelArgs(modelId: string): string[] {
  return ["--model", `${GATEWAY_PROVIDER_ID}/${modelId}`];
}
