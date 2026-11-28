/**
 * Process environment for the exp-code CLI, applied before Pi loads.
 *
 * Users only ever set exp-code names. Pi reads its own `PI_*` variables, so exp-code translates
 * each user-facing `EXP_CODE_X` to the `PI_X` Pi reads, after removing any `PI_X` already in the
 * environment: a `PI_*` name is never honored as a setting of exp-code.
 *
 * exp-code keeps its own configuration directory (`~/.exp-code`, or `EXP_CODE_HOME`), so its
 * settings, sessions, saved credentials, and extensions never mix with another agent install; Pi
 * reads it as `PI_CODING_AGENT_DIR=<home>/agent`. Pi's install telemetry and its own version check
 * are always off (exp-code checks for its own releases, see update.ts).
 */

import { homedir } from "node:os";
import { join } from "node:path";

/** The exp-code home directory. */
export function expCodeHome(env: NodeJS.ProcessEnv): string {
  const configured = env.EXP_CODE_HOME?.trim();
  return configured ? configured : join(homedir(), ".exp-code");
}

/**
 * The exp-code variable for each Pi variable a user may set, with what it does. The README's
 * variable table mirrors this list (a test keeps them equal).
 */
export const TRANSLATED_VARIABLES: readonly { name: string; pi: string; meaning: string }[] = [
  { name: "EXP_CODE_SESSION_DIR", pi: "PI_CODING_AGENT_SESSION_DIR", meaning: "Session storage directory (overridden by `--session-dir`)." },
  { name: "EXP_CODE_OFFLINE", pi: "PI_OFFLINE", meaning: "`1` disables startup network operations, as `--offline` does." },
  { name: "EXP_CODE_CACHE_RETENTION", pi: "PI_CACHE_RETENTION", meaning: "`long` asks for long prompt-cache retention where the model supports it." },
  { name: "EXP_CODE_EXPERIMENTAL", pi: "PI_EXPERIMENTAL", meaning: "`1` turns on experimental features." },
  { name: "EXP_CODE_CODEMODE_MAX_WORKERS", pi: "PI_CODEMODE_MAX_WORKERS", meaning: "Maximum parallel codemode script workers." },
  { name: "EXP_CODE_CLIPBOARD_OSC52", pi: "PI_CLIPBOARD_OSC52", meaning: "`1` also copies through the terminal (OSC 52), as over SSH." },
  { name: "EXP_CODE_HARDWARE_CURSOR", pi: "PI_HARDWARE_CURSOR", meaning: "`1` uses the terminal cursor instead of the drawn one." },
  { name: "EXP_CODE_CLEAR_ON_SHRINK", pi: "PI_CLEAR_ON_SHRINK", meaning: "`1` clears the screen when the UI shrinks." },
  { name: "EXP_CODE_TRUE_COLOR", pi: "PI_TRUE_COLOR", meaning: "`1` or `0` overrides true-color detection." },
  { name: "EXP_CODE_HYPERLINKS", pi: "PI_HYPERLINKS", meaning: "`1` or `0` overrides terminal hyperlink detection." },
  { name: "EXP_CODE_IMAGE_PROTOCOL", pi: "PI_IMAGE_PROTOCOL", meaning: "Inline image protocol (`kitty`, `iterm2`, or `none`)." },
  { name: "EXP_CODE_PROGRAM_STATUS", pi: "PI_PROGRAM_STATUS", meaning: "`1` or `0` states whether the terminal supports program status (OSC 7501) instead of asking it." },
  { name: "EXP_CODE_TUI_ESC_TIMEOUT", pi: "PI_TUI_ESC_TIMEOUT", meaning: "Milliseconds to wait after Esc for an escape sequence." },
  { name: "EXP_CODE_TIMING", pi: "PI_TIMING", meaning: "`1` prints startup timings (debugging)." },
  { name: "EXP_CODE_STARTUP_BENCHMARK", pi: "PI_STARTUP_BENCHMARK", meaning: "`1` measures interactive startup and exits (debugging)." },
  { name: "EXP_CODE_TUI_DEBUG", pi: "PI_TUI_DEBUG", meaning: "`1` writes terminal UI debug logs." },
  { name: "EXP_CODE_TUI_DEBUG_REDRAW", pi: "PI_TUI_DEBUG_REDRAW", meaning: "`1` logs full redraws (debugging)." },
  { name: "EXP_CODE_TUI_WRITE_LOG", pi: "PI_TUI_WRITE_LOG", meaning: "File that receives every terminal write (debugging)." }
];

/**
 * Pi variables exp-code owns or that only serve Pi's own services and providers: removed from the
 * environment (or set by exp-code), never read from the user.
 */
const OWNED_PI_VARIABLES = [
  "PI_CODING_AGENT_DIR",
  "PI_TELEMETRY",
  "PI_SKIP_VERSION_CHECK",
  "PI_PACKAGE_DIR",
  "PI_SELF_UPDATE_COMMAND",
  "PI_SHARE_VIEWER_URL",
  "PI_RADIUS_GATEWAY",
  "PI_OAUTH_CALLBACK_HOST",
  "PI_MANAGED_INSTALL_ROOT",
  "PI_INSTALLER_API_BASE",
  "PI_CODING_AGENT",
  "PI_SESSION_ID",
  "PI_SESSION_FILE",
  "PI_PROVIDER",
  "PI_MODEL",
  "PI_REASONING_LEVEL"
];

/** Whether an environment flag is on (`1`, `true`, `yes`). */
export function envFlag(value: string | undefined): boolean {
  const normalized = value?.trim().toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes";
}

/** Translate exp-code's variables to Pi's and point Pi at exp-code's directories. */
export function applyCliEnvironment(env: NodeJS.ProcessEnv): void {
  for (const name of OWNED_PI_VARIABLES) delete env[name];
  for (const variable of TRANSLATED_VARIABLES) {
    delete env[variable.pi];
    const value = env[variable.name];
    if (value !== undefined && value !== "") env[variable.pi] = value;
  }
  env.PI_CODING_AGENT_DIR = join(expCodeHome(env), "agent");
  env.PI_TELEMETRY = "0";
  env.PI_SKIP_VERSION_CHECK = "1";
  env.EXP_CODE_AGENT = "true";
  env.AI_AGENT = "exp-code";
}
