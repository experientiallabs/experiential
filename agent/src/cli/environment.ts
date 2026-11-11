/**
 * Process environment for the exp-code CLI, applied before Pi loads.
 *
 * exp-code keeps its own configuration directory (`~/.exp-code`, or `EXP_CODE_HOME`) so its
 * settings, sessions, saved credentials, and extensions never mix with a separate Pi install.
 * Pi reads its agent directory from `PI_CODING_AGENT_DIR`, so exp-code sets that variable to
 * `<home>/agent`. Project files keep Pi's `.pi/` and `AGENTS.md` conventions, which Pi fixes at
 * build time and which a project can share with Pi.
 */

import { homedir } from "node:os";
import { join } from "node:path";

/** The exp-code home directory. */
export function expCodeHome(env: NodeJS.ProcessEnv): string {
  const configured = env.EXP_CODE_HOME?.trim();
  return configured ? configured : join(homedir(), ".exp-code");
}

/**
 * Point Pi at exp-code's directories and default telemetry and version checks off.
 *
 * Explicit `PI_TELEMETRY` and `PI_SKIP_VERSION_CHECK` values are kept; the agent directory is
 * always exp-code's own.
 */
export function applyCliEnvironment(env: NodeJS.ProcessEnv): void {
  env.PI_CODING_AGENT_DIR = join(expCodeHome(env), "agent");
  env.PI_TELEMETRY ??= "0";
  env.PI_SKIP_VERSION_CHECK ??= "1";
  env.PI_CODING_AGENT = "true";
  env.AI_AGENT = "exp-code";
}
