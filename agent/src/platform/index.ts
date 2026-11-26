/**
 * The Experiential platform tools: read-only access to the organization behind the session's
 * hosted gateway key (its request log, usage, models, and evaluations).
 *
 * Safety defaults. The tools can read the organization's stored prompts and responses, so a
 * prompt injection in anything the agent reads (a web page, a file) could try to make it fetch
 * traces and copy them somewhere. They are on by default in the interactive CLI, where the user
 * watches the session and owns the key, and can be turned off with `EXP_CODE_PLATFORM_TOOLS=0`.
 * The headless runner (hosted sandboxes) never enables them unless its host asks explicitly.
 */

import { tmpdir } from "node:os";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import { materializeSkills } from "../extension/skills.js";
import type { GatewayConfig } from "../gateway/config.js";
import { createModelsTool } from "./catalog.js";
import { PlatformClient, platformOrigin } from "./client.js";
import { createEvalsTool } from "./evals.js";
import { createGetRequestTool, createSearchRequestsTool } from "./requests.js";
import { TRACES_SKILL, TRACES_SKILL_NAME } from "./skill.js";
import { createUsageTool } from "./usage.js";

/** The environment variable that turns the platform tools off in the CLI (`0`, `false`, `off`, `no`). */
export const PLATFORM_TOOLS_ENV = "EXP_CODE_PLATFORM_TOOLS";

/** Tool names the platform tools register. */
export const PLATFORM_TOOL_NAMES = [
  "experiential_search_requests",
  "experiential_get_request",
  "experiential_usage",
  "experiential_models",
  "experiential_evals"
] as const;

/** Whether the CLI enables the platform tools: on unless the environment turns them off. */
export function platformToolsEnabled(env: NodeJS.ProcessEnv): boolean {
  const value = env[PLATFORM_TOOLS_ENV]?.trim().toLowerCase() ?? "";
  return !["0", "false", "off", "no"].includes(value);
}

export interface PlatformToolsOptions {
  fetch?: typeof globalThis.fetch;
  timeoutMs?: number;
}

/**
 * The platform tools for a gateway, or an empty list when it has no platform API (a local `exp`
 * gateway, or a base URL that does not end in `/v1`).
 */
export function createPlatformTools(gateway: GatewayConfig, options: PlatformToolsOptions = {}): ToolDefinition[] {
  const origin = platformOrigin(gateway);
  if (origin === undefined) return [];
  const client = new PlatformClient({ origin, apiKey: gateway.apiKey, fetch: options.fetch, timeoutMs: options.timeoutMs });
  return [createSearchRequestsTool(client), createGetRequestTool(client), createUsageTool(client), createModelsTool(client), createEvalsTool(client)] as unknown as ToolDefinition[];
}

/**
 * The directory holding the `experiential-traces` skill, written under `root` (it describes the
 * platform tools, so it is offered only where they are registered).
 */
export function tracesSkillDir(root: string = tmpdir()): string {
  return materializeSkills({ [TRACES_SKILL_NAME]: TRACES_SKILL }, root);
}

export { PlatformClient, PlatformError, platformOrigin } from "./client.js";
export { TRACES_SKILL, TRACES_SKILL_NAME } from "./skill.js";
