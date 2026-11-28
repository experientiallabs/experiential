/**
 * The organization section of `/usage`: the key's remaining credits and today's org spend.
 *
 * Read-only, with the session's own hosted gateway key, from two platform reads an organization
 * key may make: `GET /api/v1/credits` (granted and spent, in dollars) and
 * `GET /api/gateway/usage/daily` for today's UTC row (after `GET /api/whoami` names the org).
 * Both run at once under one short deadline; a failure or a local `exp` gateway turns the
 * section into a one-line note and never fails `/usage`.
 */

import type { GatewayConfig } from "../gateway/config.js";
import { PlatformClient, PlatformError, platformOrigin } from "../platform/client.js";
import { toNano } from "./aggregate.js";

/** The longest `/usage` waits for the organization reads. */
export const ORG_DEADLINE_MS = 3000;

export type OrgUsage =
  | { kind: "unavailable"; reason: string }
  | {
      kind: "available";
      orgName: string | undefined;
      /** Granted minus spent, in nano-dollars; null when the read failed. */
      balanceNano: number | null;
      /** Today's UTC row; null when the read failed. */
      today: { chargedNano: number; requests: number } | null;
      /** One line per read that failed. */
      notes: string[];
    };

interface CreditsBody {
  data?: { total_credits?: unknown; total_usage?: unknown };
}

interface DailyBody {
  rows?: { day?: unknown; requests?: unknown; paid_nano_usd?: unknown }[];
}

export interface OrgUsageOptions {
  fetch?: typeof globalThis.fetch;
  deadlineMs?: number;
  now?: number;
}

const finite = (value: unknown): number | null => {
  const parsed = typeof value === "string" ? Number(value) : value;
  return typeof parsed === "number" && Number.isFinite(parsed) ? parsed : null;
};

function reasonOf(error: unknown): string {
  if (error instanceof PlatformError) return error.message;
  return (error as Error)?.message ?? String(error);
}

/** Read the organization section for a gateway; never throws. */
export async function readOrgUsage(gateway: GatewayConfig | null, options: OrgUsageOptions = {}): Promise<OrgUsage | undefined> {
  if (gateway === null) return undefined;
  const origin = platformOrigin(gateway);
  if (origin === undefined) {
    return { kind: "unavailable", reason: gateway.kind === "local" ? "not shown for a local exp gateway (it has no platform API)." : "not shown: this gateway URL has no platform API." };
  }
  const deadlineMs = options.deadlineMs ?? ORG_DEADLINE_MS;
  const client = new PlatformClient({ origin, apiKey: gateway.apiKey, fetch: options.fetch, timeoutMs: deadlineMs });
  const day = new Date(options.now ?? Date.now()).toISOString().slice(0, 10);

  const credits = client.get<CreditsBody>("/api/v1/credits").then((body) => {
    const granted = finite(body?.data?.total_credits);
    const spent = finite(body?.data?.total_usage);
    if (granted === null || spent === null) throw new Error("the credits answer had no balance");
    return toNano(granted) - toNano(spent);
  });
  const whoami = client.whoami();
  const today = whoami.then(async ({ org_id }) => {
    const body = await client.get<DailyBody>("/api/gateway/usage/daily", { org_id, scope: "org", group_by: "day", from: day, to: day, limit: 1 });
    const rows = Array.isArray(body?.rows) ? body.rows.filter((row) => row.day === day) : [];
    return {
      chargedNano: rows.reduce((sum, row) => sum + Math.round(finite(row.paid_nano_usd) ?? 0), 0),
      requests: rows.reduce((sum, row) => sum + Math.round(finite(row.requests) ?? 0), 0)
    };
  });
  // A read still running after the deadline must not surface as an unhandled rejection.
  for (const pending of [credits, whoami, today]) pending.catch(() => {});
  // One deadline for the whole section, even if a fetch ignores its signal.
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<"timeout">((resolve) => {
    timer = setTimeout(() => resolve("timeout"), deadlineMs + 50);
  });
  const settled = await Promise.race([Promise.allSettled([credits, whoami, today]), deadline]).finally(() => clearTimeout(timer));
  if (settled === "timeout") return { kind: "unavailable", reason: `the Experiential API did not answer within ${Number((deadlineMs / 1000).toFixed(1))} s.` };
  const [balance, who, daily] = settled;
  if (balance.status === "rejected" && daily.status === "rejected") return { kind: "unavailable", reason: reasonOf(balance.reason) };
  const notes: string[] = [];
  if (balance.status === "rejected") notes.push(`Balance unavailable: ${reasonOf(balance.reason)}`);
  if (daily.status === "rejected") notes.push(`Today's spend unavailable: ${reasonOf(daily.reason)}`);
  return {
    kind: "available",
    orgName: who.status === "fulfilled" ? who.value.org_name || who.value.org_slug : undefined,
    balanceNano: balance.status === "fulfilled" ? balance.value : null,
    today: daily.status === "fulfilled" ? daily.value : null,
    notes
  };
}
