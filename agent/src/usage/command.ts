/**
 * `/usage` (and its alias `/cost`): this session's time, calls, tokens, cost per model, context
 * use, and the organization's balance, in one aligned block.
 *
 * In the terminal UI the report is a custom session entry drawn by its own renderer: it is not
 * part of the model's context, and it is drawn only by the process that wrote it, so a resumed
 * session never shows an old snapshot as if it were current. Elsewhere (RPC) it is a notice.
 */

import { randomUUID } from "node:crypto";
import type { ExtensionAPI, ExtensionCommandContext, Theme } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { platformUrl } from "../auth/loopback.js";
import type { GatewayConfig } from "../gateway/config.js";
import { summarizeSession } from "./aggregate.js";
import { type ReportLine, renderUsageReport } from "./format.js";
import { type OrgUsage, readOrgUsage } from "./org.js";

/** The custom entry type the report is stored under. */
export const USAGE_ENTRY_TYPE = "exp-code-usage";

/** Command names: `/usage`, and `/cost` for people used to that name (Pi defines neither). */
export const USAGE_COMMAND_NAMES = ["usage", "cost"] as const;

export interface UsageCommandOptions {
  gateway: GatewayConfig | null;
  fetch?: typeof globalThis.fetch;
  /** Web page with the organization's usage history; defaults to the platform's Spend page for a hosted gateway. */
  dashboardUrl?: string;
  /** Organization reads; injectable for tests. */
  readOrg?: (gateway: GatewayConfig | null) => Promise<OrgUsage | undefined>;
  now?: () => number;
}

interface UsageEntryData {
  instance: string;
  lines: ReportLine[];
}

function themed(line: ReportLine, theme: Theme): string {
  switch (line.style) {
    case "title":
      return theme.bold(line.text);
    case "header":
      return theme.fg("dim", line.text);
    case "dim":
      return theme.fg("dim", line.text);
    case "label": {
      // Dim the label column, keep the value plain.
      const match = /^(\s+\S+\s+)(.*)$/.exec(line.text);
      return match === null ? line.text : `${theme.fg("dim", match[1]!)}${match[2]!}`;
    }
    default:
      return line.text;
  }
}

/** Plain text of a report (RPC and tests). */
export function reportText(lines: ReportLine[]): string {
  return lines.map((line) => line.text).join("\n");
}

/** Build the report for the current session. */
export async function buildUsageReport(ctx: Pick<ExtensionCommandContext, "sessionManager" | "getContextUsage" | "isIdle" | "model">, options: UsageCommandOptions & { openedAt?: number }): Promise<ReportLine[]> {
  const now = options.now?.() ?? Date.now();
  const manager = ctx.sessionManager;
  const usage = summarizeSession(manager.getEntries(), {
    sessionId: manager.getSessionId(),
    sessionName: manager.getSessionName(),
    startedAt: manager.getHeader()?.timestamp,
    openedAt: options.openedAt
  });
  const contextUsage = ctx.getContextUsage();
  const org = await (options.readOrg ?? ((gateway) => readOrgUsage(gateway, { fetch: options.fetch, now })))(options.gateway);
  const dashboardUrl = options.dashboardUrl ?? (options.gateway?.kind === "hosted" ? `${platformUrl(process.env)}/spend` : undefined);
  return renderUsageReport(usage, {
    now,
    streaming: !ctx.isIdle(),
    context: contextUsage === undefined ? undefined : { ...contextUsage, model: ctx.model?.id },
    org,
    dashboardUrl
  });
}

/** Register `/usage`, `/cost` and the report's renderer. */
export function registerUsageCommand(pi: ExtensionAPI, options: UsageCommandOptions): void {
  const instance = randomUUID();
  let openedAt: number | undefined;
  pi.on("session_start", () => {
    openedAt = options.now?.() ?? Date.now();
  });
  pi.registerEntryRenderer<UsageEntryData>(USAGE_ENTRY_TYPE, (entry, _options, theme) => {
    const data = entry.data;
    if (data === undefined || data.instance !== instance || !Array.isArray(data.lines)) return undefined;
    return new Text(data.lines.map((line) => themed(line, theme)).join("\n"), 1, 1);
  });
  const handler = async (_args: string, ctx: ExtensionCommandContext) => {
    const lines = await buildUsageReport(ctx, { ...options, openedAt });
    if (ctx.mode === "tui") pi.appendEntry<UsageEntryData>(USAGE_ENTRY_TYPE, { instance, lines });
    else ctx.ui.notify(reportText(lines), "info");
  };
  pi.registerCommand("usage", { description: "Show this session's time, tokens, cost by model, context use and your organization's balance", handler });
  pi.registerCommand("cost", { description: "Same as /usage", handler });
}
