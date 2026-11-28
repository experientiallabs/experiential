/**
 * Session usage, summed from the session file's entries for `/usage`.
 *
 * Every number comes from what the session recorded: token counts from each model call's
 * `usage`, money from the billing the Experiential gateway reported on that call
 * (`gatewayBilling`, see gateway/provider.ts), and durations from the monotonic `durationMs`
 * Pi stamps on finished responses and tool results. Money is summed as integer nano-dollars,
 * so totals carry no float drift, and a call with no reported price stays unpriced: a
 * catalog estimate is never shown as a cost.
 */

import type { SessionEntry } from "@earendil-works/pi-coding-agent";
import { GATEWAY_PROVIDER_ID } from "../gateway/config.js";
import { messageBilling } from "../gateway/provider.js";

/** Nano-dollars per dollar. */
export const NANO_PER_USD = 1_000_000_000;

/** A non-negative dollar amount as whole nano-dollars (the gateway's own unit). */
export function toNano(usd: number): number {
  return Math.round(usd * NANO_PER_USD);
}

/** Token and money figures, summed over a set of calls. */
export interface UsageFigures {
  calls: number;
  failedCalls: number;
  /** Uncached prompt tokens (the prompt minus cache reads and cache writes). */
  input: number;
  cacheRead: number;
  cacheWrite: number;
  /** Output tokens, reasoning included. */
  output: number;
  /** Reasoning tokens (a subset of `output`); null when no call reported a split. */
  reasoning: number | null;
  /** Credits the gateway charged, in nano-dollars, over the priced calls. */
  chargedNano: number;
  /** List value of calls served on the member's own provider key (not charged). */
  byokListNano: number;
  /** List value given away on calls charged below list (free or discounted). */
  freeNano: number;
  /** Calls with no reported price. */
  unpricedCalls: number;
  /** Summed response time of the calls that carried one, and their output tokens. */
  timedMs: number;
  timedOutput: number;
}

/** One row of the per-model table. */
export interface ModelUsage extends UsageFigures {
  /** Display name: the model id, with the provider for anything but the gateway. */
  label: string;
  provider: string;
  model: string;
}

export interface SessionUsage {
  sessionId: string;
  sessionName: string | undefined;
  /** When the session was created (the header timestamp), in ms. */
  startedAt: number | undefined;
  /** User prompts. */
  prompts: number;
  totals: UsageFigures;
  models: ModelUsage[];
  /** Summed model response time and tool execution time, in ms. */
  modelTimeMs: number;
  toolTimeMs: number;
  /** Calls recorded before `openedAt` (a resumed session's earlier turns). */
  earlierCalls: number;
  /** When this process opened the session, in ms. */
  openedAt: number | undefined;
}

export function emptyFigures(): UsageFigures {
  return {
    calls: 0,
    failedCalls: 0,
    input: 0,
    cacheRead: 0,
    cacheWrite: 0,
    output: 0,
    reasoning: null,
    chargedNano: 0,
    byokListNano: 0,
    freeNano: 0,
    unpricedCalls: 0,
    timedMs: 0,
    timedOutput: 0
  };
}

/** One model call as the session recorded it. */
interface CallRecord {
  provider: string;
  model: string;
  input: number;
  cacheRead: number;
  cacheWrite: number;
  output: number;
  reasoning: number | undefined;
  failed: boolean;
  chargedNano: number | null;
  byokListNano: number;
  freeNano: number;
  durationMs: number | undefined;
  timestamp: number;
}

interface RecordedUsage {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
  reasoning?: number;
  cost?: { total?: number; totalFromProvider?: boolean };
}

const count = (value: unknown): number => (typeof value === "number" && Number.isFinite(value) && value > 0 ? Math.round(value) : 0);

/**
 * The price of a call: the gateway's reported billing, else a cost another provider itself
 * reported as billed (`totalFromProvider`). A catalog estimate is not a price.
 */
function priceOf(message: unknown, usage: RecordedUsage | undefined): Pick<CallRecord, "chargedNano" | "byokListNano" | "freeNano"> {
  const billing = messageBilling(message);
  if (billing !== undefined) {
    const charged = toNano(billing.cost);
    return {
      chargedNano: charged,
      byokListNano: billing.byok && billing.byokListUsd !== null ? toNano(billing.byokListUsd) : 0,
      freeNano: billing.listUsd !== null ? Math.max(0, toNano(billing.listUsd) - charged) : 0
    };
  }
  const total = usage?.cost?.total;
  if (usage?.cost?.totalFromProvider === true && typeof total === "number" && Number.isFinite(total) && total >= 0) {
    return { chargedNano: toNano(total), byokListNano: 0, freeNano: 0 };
  }
  return { chargedNano: null, byokListNano: 0, freeNano: 0 };
}

function record(provider: string, model: string, usage: RecordedUsage | undefined, message: unknown, timestamp: number, extra: Partial<CallRecord> = {}): CallRecord {
  return {
    provider,
    model,
    input: count(usage?.input),
    cacheRead: count(usage?.cacheRead),
    cacheWrite: count(usage?.cacheWrite),
    output: count(usage?.output),
    reasoning: typeof usage?.reasoning === "number" && Number.isFinite(usage.reasoning) ? Math.max(0, Math.round(usage.reasoning)) : undefined,
    failed: false,
    durationMs: undefined,
    timestamp,
    ...priceOf(message, usage),
    ...extra
  };
}

function entryTime(entry: { timestamp?: unknown }): number {
  const parsed = typeof entry.timestamp === "string" ? Date.parse(entry.timestamp) : Number.NaN;
  return Number.isNaN(parsed) ? 0 : parsed;
}

function add(target: UsageFigures, call: CallRecord): void {
  target.calls += 1;
  if (call.failed) target.failedCalls += 1;
  target.input += call.input;
  target.cacheRead += call.cacheRead;
  target.cacheWrite += call.cacheWrite;
  target.output += call.output;
  if (call.reasoning !== undefined) target.reasoning = (target.reasoning ?? 0) + call.reasoning;
  if (call.chargedNano === null) target.unpricedCalls += 1;
  else target.chargedNano += call.chargedNano;
  target.byokListNano += call.byokListNano;
  target.freeNano += call.freeNano;
  if (call.durationMs !== undefined && call.durationMs > 0 && call.output > 0) {
    target.timedMs += call.durationMs;
    target.timedOutput += call.output;
  }
}

export interface SummarizeOptions {
  sessionId: string;
  sessionName?: string;
  /** Header timestamp (ISO). */
  startedAt?: string;
  /** When this process opened the session, in ms; earlier calls are counted as resumed. */
  openedAt?: number;
}

/**
 * Sum every model call in the session file: assistant responses on every branch (a fork or a
 * `/tree` jump still spent the tokens), compaction and branch summaries, and recorded
 * background usage such as cache warming.
 */
export function summarizeSession(entries: readonly SessionEntry[], options: SummarizeOptions): SessionUsage {
  const calls: CallRecord[] = [];
  let prompts = 0;
  let modelTimeMs = 0;
  let toolTimeMs = 0;
  for (const entry of entries) {
    switch (entry.type) {
      case "message": {
        const message = entry.message as { role?: string; timestamp?: number; durationMs?: number };
        if (message.role === "user") prompts += 1;
        else if (message.role === "toolResult") toolTimeMs += count(message.durationMs);
        else if (message.role === "assistant") {
          const assistant = message as { provider?: string; model?: string; usage?: RecordedUsage; stopReason?: string; durationMs?: number; timestamp?: number };
          const durationMs = count(assistant.durationMs) || undefined;
          modelTimeMs += durationMs ?? 0;
          calls.push(
            record(assistant.provider ?? "unknown", assistant.model ?? "unknown", assistant.usage, assistant, typeof assistant.timestamp === "number" ? assistant.timestamp : entryTime(entry), {
              failed: assistant.stopReason === "error",
              durationMs
            })
          );
        }
        break;
      }
      case "compaction":
      case "branch_summary":
        if (entry.usage !== undefined) calls.push(record("", entry.type === "compaction" ? "(compaction)" : "(branch summary)", entry.usage, entry, entryTime(entry)));
        break;
      case "usage":
        calls.push(record(entry.provider, `${entry.model} (${entry.kind.replaceAll("_", " ")})`, entry.usage, entry, entryTime(entry)));
        break;
      default:
        break;
    }
  }

  const totals = emptyFigures();
  const byModel = new Map<string, ModelUsage>();
  let earlierCalls = 0;
  for (const call of calls) {
    add(totals, call);
    if (options.openedAt !== undefined && call.timestamp > 0 && call.timestamp < options.openedAt) earlierCalls += 1;
    const key = `${call.provider}\u0000${call.model}`;
    let row = byModel.get(key);
    if (row === undefined) {
      const label = call.provider === "" || call.provider === GATEWAY_PROVIDER_ID ? call.model : `${call.provider}/${call.model}`;
      row = { ...emptyFigures(), label, provider: call.provider, model: call.model };
      byModel.set(key, row);
    }
    add(row, call);
  }
  const startedAt = options.startedAt === undefined ? Number.NaN : Date.parse(options.startedAt);
  return {
    sessionId: options.sessionId,
    sessionName: options.sessionName,
    startedAt: Number.isNaN(startedAt) ? undefined : startedAt,
    prompts,
    totals,
    // Most expensive first, then the busiest, so the table leads with what matters.
    models: [...byModel.values()].sort((a, b) => b.chargedNano + b.byokListNano - (a.chargedNano + a.byokListNano) || b.calls - a.calls),
    modelTimeMs,
    toolTimeMs,
    earlierCalls,
    openedAt: options.openedAt
  };
}
