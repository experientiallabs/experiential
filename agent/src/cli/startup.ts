/**
 * Startup housekeeping for the interactive terminal UI, done before the UI draws.
 *
 * - The file-search helpers (`fd`, `rg`) are installed into exp-code's own bin directory the
 *   first time they are missing, with one short progress line, instead of the upstream UI's
 *   per-tool download messages in the chat area.
 * - The upstream package's changelog is never shown as exp-code's "what's new": exp-code records
 *   the bundled version as already seen. exp-code's own release notes live with its releases.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { VERSION as UPSTREAM_VERSION } from "@earendil-works/pi-coding-agent";

/** The upstream helpers that install `fd` and `rg` (internal to the pinned package). */
export interface ToolsManager {
  getToolPath(tool: "fd" | "rg"): string | null;
  ensureTool(tool: "fd" | "rg", onStatus?: (status: { type: "info" | "warning"; message: string }) => void): Promise<string | undefined>;
}

/** Load the pinned package's tools manager. It is not a public export, so it is loaded by file. */
export async function loadToolsManager(): Promise<ToolsManager> {
  const entry = import.meta.resolve("@earendil-works/pi-coding-agent");
  return (await import(new URL("./utils/tools-manager.js", entry).href)) as ToolsManager;
}

/**
 * Install any missing file-search helper, printing one line while it downloads. Failures are left
 * for the UI to report when the helper is first needed.
 */
export async function ensureSearchTools(offline: boolean, write: (text: string) => void = (text) => process.stderr.write(text)): Promise<void> {
  const offlineEnv = process.env.PI_OFFLINE?.toLowerCase();
  if (offline || offlineEnv === "1" || offlineEnv === "true" || offlineEnv === "yes") return;
  const tools = await loadToolsManager();
  const missing = (["fd", "rg"] as const).filter((tool) => tools.getToolPath(tool) === null);
  if (missing.length === 0) return;
  write("exp-code: setting up file search (one time)...");
  const results = await Promise.all(missing.map((tool) => tools.ensureTool(tool).catch(() => undefined)));
  write(results.every((path) => path !== undefined) ? " done\n" : " skipped\n");
}

/** Mark the bundled upstream changelog as seen in exp-code's settings, so it never pops up. */
export function markUpstreamChangelogSeen(agentDir: string): void {
  const path = join(agentDir, "settings.json");
  let settings: Record<string, unknown>;
  try {
    const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return;
    settings = parsed as Record<string, unknown>;
  } catch {
    // No settings yet: a first start records the version itself and shows nothing.
    return;
  }
  if (settings.lastChangelogVersion === UPSTREAM_VERSION) return;
  settings.lastChangelogVersion = UPSTREAM_VERSION;
  writeFileSync(path, `${JSON.stringify(settings, null, 2)}\n`);
}
