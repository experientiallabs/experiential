/**
 * The `/usage` report as aligned plain-text lines.
 *
 * Each line carries a style tag so the terminal renderer can theme it; the text itself has no
 * escape codes, which keeps the report testable and readable in RPC and print modes.
 */

import { NANO_PER_USD, type ModelUsage, type SessionUsage, type UsageFigures } from "./aggregate.js";
import type { OrgUsage } from "./org.js";

export type LineStyle = "title" | "label" | "header" | "plain" | "dim";

export interface ReportLine {
  style: LineStyle;
  text: string;
}

/**
 * Whole nano-dollars as dollars, from the integer (no float rounding): cents for large figures,
 * four decimals from one cent, and six for the sub-cent cost of a single call (nine when six
 * would read as zero). Matches the platform's money display.
 */
export function formatNano(nano: number): string {
  if (!Number.isFinite(nano)) return "-";
  if (nano === 0) return "$0";
  const sign = nano < 0 ? "-" : "";
  const abs = Math.abs(Math.round(nano));
  let digits = abs >= 100 * NANO_PER_USD ? 2 : abs >= NANO_PER_USD ? 3 : abs >= NANO_PER_USD / 100 ? 4 : 6;
  if (digits === 6 && abs < 500) digits = 9;
  const step = 10 ** (9 - digits);
  const units = Math.floor((abs + step / 2) / step);
  const whole = Math.floor(units / 10 ** digits);
  let fraction = String(units % 10 ** digits).padStart(digits, "0");
  // Sub-cent figures keep their significant digits but drop trailing zeros past the fourth.
  if (digits > 4) fraction = fraction.replace(/0+$/, "").padEnd(4, "0");
  return `${sign}$${whole.toLocaleString("en-US")}.${fraction}`;
}

/** An integer with thousands separators. */
export function formatCount(value: number): string {
  return Math.round(value).toLocaleString("en-US");
}

/** A duration as `850ms`, `4.2s`, `3m 05s` or `1h 02m`. */
export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "-";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const seconds = Math.floor(ms / 1000);
  const pad = (n: number) => String(n).padStart(2, "0");
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${pad(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${pad(Math.floor((seconds % 3600) / 60))}m`;
}

/** Output tokens per second over the calls that recorded a response time; "-" without one. */
export function formatSpeed(figures: Pick<UsageFigures, "timedMs" | "timedOutput">): string {
  if (figures.timedMs <= 0 || figures.timedOutput <= 0) return "-";
  return (figures.timedOutput / (figures.timedMs / 1000)).toFixed(1);
}

/** The prompt tokens a set of calls sent: uncached input plus cache reads and writes. */
export function promptTokens(figures: UsageFigures): number {
  return figures.input + figures.cacheRead + figures.cacheWrite;
}

/** A row's cost: the charged credits, "-" when nothing was priced, with a marker for partial pricing. */
function rowCost(figures: UsageFigures): string {
  if (figures.unpricedCalls === figures.calls) return "-";
  if (figures.chargedNano === 0 && figures.freeNano > 0 && figures.unpricedCalls === 0) return "free";
  return `${formatNano(figures.chargedNano)}${figures.unpricedCalls > 0 ? "*" : ""}`;
}

function costLine(totals: UsageFigures): string {
  if (totals.calls > 0 && totals.unpricedCalls === totals.calls) return "- (the gateway reported no price for these calls)";
  const parts = [`${formatNano(totals.chargedNano)} charged`];
  if (totals.byokListNano > 0) parts.push(`${formatNano(totals.byokListNano)} on your own provider key (list value, not charged)`);
  if (totals.freeNano > 0) parts.push(`${formatNano(totals.freeNano)} free (list value, not charged)`);
  if (totals.unpricedCalls > 0) parts.push(`${formatCount(totals.unpricedCalls)} unpriced call${totals.unpricedCalls === 1 ? "" : "s"} not included`);
  return parts.join(" · ");
}

function tokensLine(totals: UsageFigures): string {
  const prompt = promptTokens(totals);
  const promptParts = [];
  if (totals.cacheRead > 0) promptParts.push(`${formatCount(totals.cacheRead)} cached`);
  if (totals.cacheWrite > 0) promptParts.push(`${formatCount(totals.cacheWrite)} written to cache`);
  const input = `${formatCount(prompt)} input${promptParts.length > 0 ? ` (${promptParts.join(", ")})` : ""}`;
  const output = `${formatCount(totals.output)} output${totals.reasoning !== null && totals.reasoning > 0 ? ` (${formatCount(totals.reasoning)} reasoning)` : ""}`;
  return `${input} · ${output} · ${formatCount(prompt + totals.output)} total`;
}

/** Render rows into left-aligned first column and right-aligned numbers. */
function table(header: string[], rows: string[][]): string[] {
  const widths = header.map((cell, column) => Math.max(cell.length, ...rows.map((row) => row[column]?.length ?? 0)));
  const line = (cells: string[]) => cells.map((cell, column) => (column === 0 ? cell.padEnd(widths[column]!) : cell.padStart(widths[column]!))).join("  ");
  return [line(header), ...rows.map(line)];
}

function modelRow(row: ModelUsage): string[] {
  return [row.label, formatCount(row.calls), formatCount(promptTokens(row)), formatCount(row.cacheRead), formatCount(row.output), rowCost(row), formatSpeed(row)];
}

export interface ContextFigure {
  tokens: number | null;
  contextWindow: number;
  percent: number | null;
  model?: string;
}

export interface ReportOptions {
  now: number;
  /** A response is streaming: its usage arrives when it finishes. */
  streaming?: boolean;
  context?: ContextFigure;
  /** The organization section; undefined omits it entirely. */
  org?: OrgUsage;
  /** Where the history lives on the web dashboard. */
  dashboardUrl?: string;
}

const LABEL_WIDTH = 14;
const row = (label: string, value: string): ReportLine => ({ style: "label", text: `  ${label.padEnd(LABEL_WIDTH)}${value}` });

function titleLine(usage: SessionUsage): string {
  const name = usage.sessionName ? ` "${usage.sessionName}"` : "";
  const started = usage.startedAt === undefined ? "" : `, started ${new Date(usage.startedAt).toISOString().slice(0, 16).replace("T", " ")}Z`;
  return `Usage · session ${usage.sessionId.slice(0, 8)}${name}${started}`;
}

/** The whole report. */
export function renderUsageReport(usage: SessionUsage, options: ReportOptions): ReportLine[] {
  const lines: ReportLine[] = [{ style: "title", text: titleLine(usage) }];
  const { totals } = usage;
  if (usage.earlierCalls > 0) {
    lines.push({ style: "dim", text: `  Resumed session: includes ${formatCount(usage.earlierCalls)} call${usage.earlierCalls === 1 ? "" : "s"} from before it was opened here.` });
  }
  const time = [`${formatDuration(usage.startedAt === undefined ? Number.NaN : options.now - usage.startedAt)} since start`];
  if (usage.earlierCalls > 0 && usage.openedAt !== undefined) time.push(`${formatDuration(options.now - usage.openedAt)} since resumed`);
  if (usage.modelTimeMs > 0) time.push(`model ${formatDuration(usage.modelTimeMs)}`);
  if (usage.toolTimeMs > 0) time.push(`tools ${formatDuration(usage.toolTimeMs)}`);
  lines.push(row("Time", time.join(" · ")));
  const failed = totals.failedCalls > 0 ? ` (${formatCount(totals.failedCalls)} failed)` : "";
  lines.push(row("Activity", `${formatCount(usage.prompts)} prompt${usage.prompts === 1 ? "" : "s"} · ${formatCount(totals.calls)} model call${totals.calls === 1 ? "" : "s"}${failed}`));
  if (totals.calls === 0) {
    lines.push(row("Tokens", "No model calls yet"));
  } else {
    lines.push(row("Tokens", tokensLine(totals)));
    lines.push(row("Cost", costLine(totals)));
  }
  const context = options.context;
  if (context !== undefined && context.contextWindow > 0) {
    const used = context.tokens === null ? "unknown until the next response" : `${formatCount(context.tokens)} / ${formatCount(context.contextWindow)} tokens (${Math.round(context.percent ?? (context.tokens / context.contextWindow) * 100)}%)`;
    lines.push(row("Context", `${used}${context.model ? ` on ${context.model}` : ""}`));
  }
  if (options.streaming === true) lines.push({ style: "dim", text: "  A response is still streaming; its usage is added when it finishes." });

  if (usage.models.length > 0) {
    lines.push({ style: "plain", text: "" });
    const [header, ...body] = table(["Model", "Calls", "Input", "Cached", "Output", "Cost", "Out tok/s"], usage.models.map(modelRow));
    lines.push({ style: "header", text: `  ${header}` });
    for (const text of body) lines.push({ style: "plain", text: `  ${text}` });
    if (usage.models.some((model) => model.unpricedCalls > 0 && model.unpricedCalls < model.calls)) {
      lines.push({ style: "dim", text: "  * some calls on this model reported no price and are not included" });
    }
  }

  const org = options.org;
  if (org !== undefined) {
    lines.push({ style: "plain", text: "" });
    if (org.kind === "unavailable") lines.push({ style: "dim", text: `  Organization: ${org.reason}` });
    else {
      const parts = [org.orgName ?? "Organization"];
      if (org.balanceNano !== null) parts.push(`${formatCredits(org.balanceNano)} left (${formatCents(org.balanceNano)})`);
      if (org.today !== null) parts.push(`today (UTC) ${formatNano(org.today.chargedNano)} charged over ${formatCount(org.today.requests)} request${org.today.requests === 1 ? "" : "s"}`);
      lines.push(row("Organization", parts.join(" · ")));
      for (const note of org.notes) lines.push({ style: "dim", text: `  ${note}` });
    }
  }
  if (options.dashboardUrl !== undefined) lines.push({ style: "dim", text: `  History and per-request detail: ${options.dashboardUrl}` });
  return lines;
}

/** Nano-dollars as dollars and cents (a balance). */
export function formatCents(nano: number): string {
  const cents = Math.round(nano / (NANO_PER_USD / 100));
  const abs = Math.abs(cents);
  return `${cents < 0 ? "-" : ""}$${Math.floor(abs / 100).toLocaleString("en-US")}.${String(abs % 100).padStart(2, "0")}`;
}

/** Nano-dollars as platform credits (1 credit = $0.01), to two decimals. */
export function formatCredits(nano: number): string {
  const hundredths = Math.round(nano / (NANO_PER_USD / 10_000));
  const sign = hundredths < 0 ? "-" : "";
  const abs = Math.abs(hundredths);
  return `${sign}${Math.floor(abs / 100).toLocaleString("en-US")}.${String(abs % 100).padStart(2, "0")} credits`;
}
