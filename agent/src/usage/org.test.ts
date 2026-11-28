import { afterEach, describe, expect, it } from "vitest";
import { type FakePlatform, startFakePlatform } from "../testing/fake-platform.js";
import { readOrgUsage } from "./org.js";

let platform: FakePlatform | undefined;
afterEach(async () => {
  await platform?.close();
  platform = undefined;
});

const NOW = Date.parse("2026-11-27T15:00:00Z");
const CREDITS = { "GET /api/v1/credits": { body: { data: { total_credits: 25, total_usage: 7.6543 } } } };
const DAILY = { "GET /api/gateway/usage/daily": { body: { rows: [{ day: "2026-11-27", requests: 42, paid_nano_usd: 1234500000, spend_nano_usd: 1300000000 }] } } };

async function hosted(routes: Parameters<typeof startFakePlatform>[0]) {
  platform = await startFakePlatform(routes);
  return { kind: "hosted" as const, baseUrl: platform.baseUrl, apiKey: "xpl_k" };
}

describe("readOrgUsage", () => {
  it("reads the balance and today's UTC spend with the session key", async () => {
    const gateway = await hosted({ ...CREDITS, ...DAILY });
    const org = await readOrgUsage(gateway, { now: NOW });
    expect(org).toEqual({ kind: "available", orgName: "Acme", balanceNano: 17_345_700_000, today: { chargedNano: 1_234_500_000, requests: 42 }, notes: [] });
    expect(platform!.calls("/api/gateway/usage/daily")[0]?.query).toEqual({ org_id: "org-1111", scope: "org", group_by: "day", from: "2026-11-27", to: "2026-11-27", limit: "1" });
    expect(platform!.requests.every((request) => request.method === "GET" && request.authorization === "Bearer xpl_k")).toBe(true);
  });

  it("keeps what it could read and notes the rest", async () => {
    const gateway = await hosted({ ...DAILY, "GET /api/v1/credits": { status: 403, body: { error: "no" } } });
    const org = await readOrgUsage(gateway, { now: NOW });
    expect(org).toMatchObject({ kind: "available", balanceNano: null, today: { requests: 42 } });
    expect(org?.kind === "available" && org.notes[0]).toMatch(/^Balance unavailable: This key is not allowed/);
  });

  it("turns a total failure into one line", async () => {
    const gateway = await hosted({ "GET /api/whoami": { status: 401 }, "GET /api/v1/credits": { status: 401 } });
    const org = await readOrgUsage(gateway, { now: NOW });
    expect(org).toEqual({ kind: "unavailable", reason: expect.stringContaining("did not accept this key (401)") });
  });

  it("gives up at the deadline", async () => {
    const gateway = await hosted({ ...CREDITS, ...DAILY, "GET /api/whoami": { delayMs: 2000, body: {} }, "GET /api/v1/credits": { delayMs: 2000, body: {} } });
    const started = Date.now();
    const org = await readOrgUsage(gateway, { now: NOW, deadlineMs: 200 });
    expect(Date.now() - started).toBeLessThan(1500);
    expect(org).toEqual({ kind: "unavailable", reason: expect.stringMatching(/did not answer within/) });
  });

  it("stops at the deadline even when fetch ignores its signal", async () => {
    const hang = (() => new Promise<Response>(() => {})) as typeof globalThis.fetch;
    const org = await readOrgUsage({ kind: "hosted", baseUrl: "https://api.example.test/v1", apiKey: "xpl_k" }, { fetch: hang, now: NOW, deadlineMs: 100 });
    expect(org).toEqual({ kind: "unavailable", reason: "the Experiential API did not answer within 0.1 s." });
  });

  it("omits the section without a gateway and notes a local gateway", async () => {
    expect(await readOrgUsage(null)).toBeUndefined();
    expect(await readOrgUsage({ kind: "local", baseUrl: "http://127.0.0.1:8000/v1", apiKey: "k" })).toEqual({ kind: "unavailable", reason: "not shown for a local exp gateway (it has no platform API)." });
  });
});
