/**
 * `experiential_usage`: the organization's spend and usage over a window, grouped one way.
 *
 * Reads (all `GET`, under `/api/orgs/{org}` unless noted):
 * - `model` and `time`: `/usage/timeseries` cells (bucket, model, lane), folded here;
 * - `provider`, `label`, `key`: `/usage/by-provider`, `/usage/by-label`, `/usage/by-key`;
 * - `day`: `/api/gateway/usage/daily?org_id=...&scope=org&group_by=day`, for longer history;
 * - `/usage/latency` adds the window's p50/p95 latency and time to first token.
 */

import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { type PlatformClient, PlatformError, type QueryValue } from "./client.js";
import { count, duration, failure, failureFrom, nanoUsd, result, table, timestamp, usd , literalUnion } from "./format.js";

export const USAGE_GROUPS = ["model", "provider", "key", "label", "time", "day"] as const;
export type UsageGroup = (typeof USAGE_GROUPS)[number];
export const USAGE_WINDOWS = ["24h", "7d", "30d"] as const;
export const MAX_USAGE_ROWS = 100;

/** The money and volume figures every rollup row carries. */
interface Figures {
  request_count: number;
  error_count: number;
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens?: number;
  cost_usd: number;
  estimated_cost_usd: number;
  free_usd: number;
}

interface Bucket extends Figures {
  bucket_start: string;
  model: string;
  lane: string | null;
}

interface Timeseries {
  window: string;
  bucket_seconds: number;
  buckets: Bucket[];
}

interface ProviderRow extends Figures {
  provider: string | null;
}

interface LabelRow extends Figures {
  attribution_label: string;
}

interface KeyRow {
  api_key_id: string | null;
  key_label: string | null;
  totals: Figures;
}

interface DailyRow {
  day: string | null;
  requests: number;
  input_tokens: number;
  output_tokens: number;
  spend_nano_usd: number;
  paid_nano_usd: number;
  free_nano_usd: number;
}

interface Percentiles {
  samples: number;
  p50_ms: number;
  p95_ms: number;
}

interface Latency {
  requests: number;
  latency: Percentiles | null;
  ttft: Percentiles | null;
}

const usageSchema = Type.Object({
  group_by: Type.Optional(
    literalUnion(USAGE_GROUPS, {
      description:
        "model (default), provider, key (per API key; needs the provisioning key), label (per end-user label), time (hourly for 24h, daily otherwise), or day (daily history over `days`)"
    })
  ),
  window: Type.Optional(literalUnion(USAGE_WINDOWS, { description: "24h (default), 7d or 30d. Not used by group_by=day." })),
  days: Type.Optional(Type.Number({ description: "For group_by=day: how many days back, 1 to 365 (default 30)" })),
  model: Type.Optional(Type.String({ description: "Only this model slug (model, provider and time groupings)" })),
  api_key_id: Type.Optional(Type.String({ description: "Only this API key id (model, provider and time groupings)" })),
  lane: Type.Optional(Type.Union([Type.Literal("platform"), Type.Literal("byok")], { description: "Only platform-credit or own-provider-key traffic (model, provider and time groupings)" }))
});

const FIGURE_HEADER = ["requests", "errors", "input tokens", "output tokens", "charged", "own-key est.", "free value"];

function figureCells(row: Figures): string[] {
  return [count(row.request_count), count(row.error_count), count(row.input_tokens), count(row.output_tokens), usd(row.cost_usd), usd(row.estimated_cost_usd), usd(row.free_usd)];
}

function emptyFigures(): Figures {
  return { request_count: 0, error_count: 0, input_tokens: 0, output_tokens: 0, cached_input_tokens: 0, cost_usd: 0, estimated_cost_usd: 0, free_usd: 0 };
}

function add(into: Figures, row: Figures): void {
  into.request_count += row.request_count ?? 0;
  into.error_count += row.error_count ?? 0;
  into.input_tokens += row.input_tokens ?? 0;
  into.output_tokens += row.output_tokens ?? 0;
  into.cached_input_tokens = (into.cached_input_tokens ?? 0) + (row.cached_input_tokens ?? 0);
  into.cost_usd += row.cost_usd ?? 0;
  into.estimated_cost_usd += row.estimated_cost_usd ?? 0;
  into.free_usd += row.free_usd ?? 0;
}

/** All-in value of a row: charged plus own-key estimate plus free list value, for ranking. */
function allIn(row: Figures): number {
  return (row.cost_usd ?? 0) + (row.estimated_cost_usd ?? 0) + (row.free_usd ?? 0);
}

/** Fold timeseries cells by `key`, biggest first (or in time order for buckets). */
export function foldBuckets(buckets: Bucket[], key: (bucket: Bucket) => string): Map<string, Figures> {
  const folded = new Map<string, Figures>();
  for (const bucket of buckets) {
    const name = key(bucket);
    const figures = folded.get(name) ?? emptyFigures();
    add(figures, bucket);
    folded.set(name, figures);
  }
  return folded;
}

function totalsLine(rows: Figures[]): string {
  const total = emptyFigures();
  for (const row of rows) add(total, row);
  return `Total: ${count(total.request_count)} requests (${count(total.error_count)} not completed), ${count(total.input_tokens)} input and ${count(total.output_tokens)} output tokens, charged ${usd(total.cost_usd)}, own-key estimate ${usd(total.estimated_cost_usd)}, free value ${usd(total.free_usd)}.`;
}

function latencyLine(latency: Latency): string | undefined {
  const parts: string[] = [];
  if (latency.latency) parts.push(`latency p50 ${duration(latency.latency.p50_ms)}, p95 ${duration(latency.latency.p95_ms)}`);
  if (latency.ttft) parts.push(`time to first token p50 ${duration(latency.ttft.p50_ms)}, p95 ${duration(latency.ttft.p95_ms)}`);
  return parts.length === 0 ? undefined : `Window ${parts.join("; ")}.`;
}

function limitRows<T>(rows: T[]): { shown: T[]; note: string } {
  return rows.length > MAX_USAGE_ROWS ? { shown: rows.slice(0, MAX_USAGE_ROWS), note: `\n(${rows.length - MAX_USAGE_ROWS} smaller rows not shown.)` } : { shown: rows, note: "" };
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** The UTC date `days` before today (`YYYY-MM-DD`). */
function utcDay(now: number, daysBack: number): string {
  return new Date(now - daysBack * 86_400_000).toISOString().slice(0, 10);
}

export function createUsageTool(client: PlatformClient, now: () => number = Date.now) {
  return defineTool({
    name: "experiential_usage",
    label: "Experiential usage",
    description:
      "Summarize this organization's Experiential gateway usage and spend over the last 24h, 7d or 30d: requests, errors, tokens, charged credits, own-provider-key cost estimate and free (promotional) value, grouped by model, provider, API key, end-user label, or over time, plus the window's p50/p95 latency and time to first token. Use it for questions like \"what did we spend this week and on which models\", \"which provider fails most\", or \"how did traffic change over the last 30 days\" (group_by=day). For individual calls use experiential_search_requests.",
    promptSnippet: "Summarize the organization's Experiential spend and usage by model, provider, key, label or time",
    parameters: usageSchema,
    annotations: { readOnlyHint: true, openWorldHint: true },
    async execute(_toolCallId, args, signal) {
      const group: UsageGroup = args.group_by ?? "model";
      if (!(USAGE_GROUPS as readonly string[]).includes(group)) return failure(`group_by must be one of ${USAGE_GROUPS.join(", ")}.`);
      const window = args.window ?? "24h";
      if (!(USAGE_WINDOWS as readonly string[]).includes(window)) return failure(`window must be one of ${USAGE_WINDOWS.join(", ")}.`);
      if (args.api_key_id !== undefined && !UUID.test(args.api_key_id.trim())) return failure("api_key_id must be an API key id (a uuid).");
      const cuts: Record<string, QueryValue> = { model: args.model?.trim(), api_key_id: args.api_key_id?.trim(), lane: args.lane };
      const cutsApply = group === "model" || group === "provider" || group === "time";
      const hasCuts = Object.values(cuts).some((value) => value !== undefined && value !== "");
      if (hasCuts && !cutsApply) return failure("The model, api_key_id and lane filters apply only to group_by model, provider and time.");
      try {
        if (group === "day") return await dailyUsage(client, args.days ?? 30, now(), signal);
        const scoped = cutsApply ? { window, ...cuts } : { window };
        const latencyRead =
          group === "key" || group === "label"
            ? Promise.resolve(undefined)
            : client
                .get<Latency>(await client.orgPath("/usage/latency", signal), scoped, signal)
                .then(latencyLine)
                .catch(() => undefined);
        let header: string[];
        let rows: { name: string; figures: Figures }[];
        let extra = "";
        switch (group) {
          case "model":
          case "time": {
            const series = await client.get<Timeseries>(await client.orgPath("/usage/timeseries", signal), scoped, signal);
            const buckets = Array.isArray(series?.buckets) ? series.buckets : [];
            const folded = foldBuckets(buckets, group === "model" ? (bucket) => bucket.model : (bucket) => bucket.bucket_start);
            rows = [...folded.entries()].map(([name, figures]) => ({ name: group === "time" ? timestamp(name) : name, figures }));
            if (group === "model") rows.sort((a, b) => allIn(b.figures) - allIn(a.figures));
            else rows.sort((a, b) => a.name.localeCompare(b.name));
            header = [group === "model" ? "model" : series?.bucket_seconds === 3600 ? "hour (UTC)" : "day (UTC)", ...FIGURE_HEADER];
            break;
          }
          case "provider": {
            const body = await client.get<{ providers: ProviderRow[] }>(await client.orgPath("/usage/by-provider", signal), scoped, signal);
            rows = (body?.providers ?? []).map((row) => ({ name: row.provider ?? "(not dispatched)", figures: row }));
            header = ["provider", ...FIGURE_HEADER];
            break;
          }
          case "label": {
            const body = await client.get<{ labels: LabelRow[] }>(await client.orgPath("/usage/by-label", signal), scoped, signal);
            rows = (body?.labels ?? []).map((row) => ({ name: row.attribution_label, figures: row }));
            header = ["end-user label", ...FIGURE_HEADER];
            if (rows.length === 0) extra = "\nNo request in the window carried an end-user label (OpenAI `user` / `safety_identifier`).";
            break;
          }
          case "key": {
            let body: { keys: KeyRow[] };
            try {
              body = await client.get<{ keys: KeyRow[] }>(await client.orgPath("/usage/by-key", signal), scoped, signal);
            } catch (error) {
              if (error instanceof PlatformError && error.status === 403) {
                return failure(`${error.message} Per-key usage reveals every key's spend, so it needs the organization's provisioning key; group_by=model with api_key_id reads one key's traffic with this key.`);
              }
              throw error;
            }
            rows = (body?.keys ?? []).map((row) => ({ name: `${row.key_label ?? "(deleted key)"}${row.api_key_id ? ` (${row.api_key_id})` : ""}`, figures: row.totals }));
            header = ["API key", ...FIGURE_HEADER];
            break;
          }
        }
        const latency = await latencyRead;
        const filters = hasCuts && cutsApply ? ` (${Object.entries(cuts).filter(([, value]) => value).map(([name, value]) => `${name}=${value}`).join(", ")})` : "";
        const { shown, note } = limitRows(rows);
        const lines = [`Usage over the last ${window} by ${group}${filters}.`, totalsLine(rows.map((row) => row.figures))];
        if (latency) lines.push(latency);
        lines.push("Charged = platform credits debited; own-key est. = list value of calls on the organization's own provider keys (never charged); free value = list value of promotional usage (never charged).");
        if (shown.length > 0) lines.push("", table(header, shown.map((row) => [row.name, ...figureCells(row.figures)])) + note);
        else lines.push("", `No usage in the last ${window}.`);
        return result(lines.join("\n") + extra, { group_by: group, window, rows: rows.map((row) => ({ name: row.name, ...row.figures })) });
      } catch (error) {
        return failureFrom(error, { group_by: group, window, rows: [] });
      }
    }
  });
}

async function dailyUsage(client: PlatformClient, days: number, now: number, signal: AbortSignal | undefined) {
  if (!Number.isInteger(days) || days < 1 || days > 365) return failure("days must be a whole number from 1 to 365.");
  const { org_id } = await client.whoami(signal);
  const from = utcDay(now, days - 1);
  const to = utcDay(now, 0);
  const body = await client.get<{ rows: DailyRow[] }>(
    "/api/gateway/usage/daily",
    { org_id, scope: "org", group_by: "day", from, to, limit: days },
    signal
  );
  const rows = (Array.isArray(body?.rows) ? body.rows : []).filter((row) => row.day !== null).sort((a, b) => String(a.day).localeCompare(String(b.day)));
  const total = rows.reduce(
    (sum, row) => ({ requests: sum.requests + row.requests, spend: sum.spend + Number(row.spend_nano_usd), paid: sum.paid + Number(row.paid_nano_usd), free: sum.free + Number(row.free_nano_usd) }),
    { requests: 0, spend: 0, paid: 0, free: 0 }
  );
  const lines = [
    `Daily usage from ${from} to ${to} (UTC).`,
    `Total: ${count(total.requests)} requests, spend ${nanoUsd(total.spend)} (charged ${nanoUsd(total.paid)}; the rest is the list value of own-provider-key calls), free value ${nanoUsd(total.free)}.`
  ];
  if (rows.length === 0) lines.push("", "No usage in this range.");
  else
    lines.push(
      "",
      table(
        ["day", "requests", "input tokens", "output tokens", "spend", "charged", "free value"],
        rows.map((row) => [String(row.day), count(row.requests), count(row.input_tokens), count(row.output_tokens), nanoUsd(row.spend_nano_usd), nanoUsd(row.paid_nano_usd), nanoUsd(row.free_nano_usd)])
      )
    );
  return result(lines.join("\n"), { group_by: "day" as const, from, to, rows });
}
