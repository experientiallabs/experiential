import { afterEach, describe, expect, it } from "vitest";
import { type FakePlatform, startFakePlatform } from "../testing/fake-platform.js";
import { PlatformClient } from "./client.js";
import { exec, text } from "../testing/tool-results.js";
import { createUsageTool } from "./usage.js";

let platform: FakePlatform | undefined;
afterEach(async () => {
  await platform?.close();
  platform = undefined;
});

const ORG = "/api/orgs/org-1111";
const figures = (requests: number, cost: number, extra: Record<string, unknown> = {}) => ({
  request_count: requests,
  error_count: 1,
  input_tokens: requests * 1000,
  output_tokens: requests * 10,
  cached_input_tokens: 0,
  cost_usd: cost,
  estimated_cost_usd: 0,
  free_usd: 0,
  ...extra
});

async function tool(routes: Parameters<typeof startFakePlatform>[0]) {
  platform = await startFakePlatform(routes);
  return createUsageTool(new PlatformClient({ origin: platform.origin, apiKey: "xpl_k" }), () => Date.parse("2026-11-25T15:00:00Z"));
}

const LATENCY = { [`GET ${ORG}/usage/latency`]: { body: { window: "24h", measured_from: "x", requests: 3, latency: { samples: 3, p50_ms: 1200, p95_ms: 9000 }, ttft: { samples: 3, p50_ms: 300, p95_ms: 900 } } } };

describe("experiential_usage", () => {
  it("folds timeseries cells per model, biggest spend first, with the window latency", async () => {
    const usage = await tool({
      ...LATENCY,
      [`GET ${ORG}/usage/timeseries`]: {
        body: {
          window: "24h",
          bucket_seconds: 3600,
          buckets: [
            { bucket_start: "2026-11-25T13:00:00+00:00", model: "cheap", lane: "platform", ...figures(10, 0.1) },
            { bucket_start: "2026-11-25T14:00:00+00:00", model: "big", lane: "platform", ...figures(2, 2) },
            { bucket_start: "2026-11-25T14:00:00+00:00", model: "cheap", lane: "byok", ...figures(5, 0, { estimated_cost_usd: 0.05 }) }
          ]
        }
      }
    });
    const result = await exec(usage, { model: "cheap", lane: "platform" });
    const output = text(result);
    expect(output).toContain("Usage over the last 24h by model (model=cheap, lane=platform).");
    expect(output).toContain("Total: 17 requests (3 not completed)");
    expect(output).toContain("Window latency p50 1.20s, p95 9.00s; time to first token p50 300ms, p95 900ms.");
    expect(output.indexOf("| big |")).toBeLessThan(output.indexOf("| cheap |"));
    expect(output).toContain("| cheap | 15 | 2 | 15,000 | 150 | $0.1000 | $0.0500 | $0 |");
    expect(platform!.calls(`${ORG}/usage/timeseries`)[0]?.query).toEqual({ window: "24h", model: "cheap", lane: "platform" });
    expect(platform!.calls(`${ORG}/usage/latency`)[0]?.query).toEqual({ window: "24h", model: "cheap", lane: "platform" });
  });

  it("groups over time in bucket order", async () => {
    const usage = await tool({
      [`GET ${ORG}/usage/timeseries`]: {
        body: { window: "7d", bucket_seconds: 86400, buckets: [{ bucket_start: "2026-11-25T00:00:00+00:00", model: "a", lane: null, ...figures(1, 1) }, { bucket_start: "2026-11-24T00:00:00+00:00", model: "a", lane: null, ...figures(1, 1) }] }
      }
    });
    const output = text(await exec(usage, { group_by: "time", window: "7d" }));
    expect(output).toContain("| day (UTC) |");
    expect(output.indexOf("2026-11-24")).toBeLessThan(output.indexOf("2026-11-25"));
  });

  it("reads the provider and label rollups", async () => {
    const usage = await tool({
      ...LATENCY,
      [`GET ${ORG}/usage/by-provider`]: { body: { window: "24h", providers: [{ provider: null, ...figures(1, 0), last_used_at: "x" }, { provider: "openai", ...figures(3, 1), last_used_at: "x" }] } },
      [`GET ${ORG}/usage/by-label`]: { body: { window: "30d", labels: [] } }
    });
    expect(text(await exec(usage, { group_by: "provider" }))).toContain("| (not dispatched) |");
    const labels = await exec(usage, { group_by: "label", window: "30d" });
    expect(text(labels)).toContain("No request in the window carried an end-user label");
    expect(platform!.calls(`${ORG}/usage/by-label`)[0]?.query).toEqual({ window: "30d" });
    expect((await exec(usage, { group_by: "label", model: "x" })).isError).toBe(true);
  });

  it("explains that per-key usage needs the provisioning key", async () => {
    const usage = await tool({ [`GET ${ORG}/usage/by-key`]: { status: 403, body: { error: "A provisioning key is required" } } });
    const result = await exec(usage, { group_by: "key" });
    expect(result.isError).toBe(true);
    expect(text(result)).toMatch(/403.*needs the organization's provisioning key; group_by=model with api_key_id/);
  });

  it("lists per-key totals", async () => {
    const usage = await tool({ [`GET ${ORG}/usage/by-key`]: { body: { window: "24h", keys: [{ api_key_id: "k1", key_label: "ci", totals: figures(4, 0.4), models: [], last_used_at: "x" }] } } });
    expect(text(await exec(usage, { group_by: "key" }))).toContain("| ci (k1) | 4 |");
  });

  it("reads daily history from the gateway usage rollup", async () => {
    const usage = await tool({
      "GET /api/gateway/usage/daily": {
        body: {
          org_id: "org-1111",
          scope: "org",
          group_by: "day",
          rows: [
            { day: "2026-11-25", user_id: null, alias: null, requests: 4, input_tokens: 10, output_tokens: 2, spend_nano_usd: 3_000_000_000, paid_nano_usd: 2_000_000_000, free_nano_usd: 500_000_000 },
            { day: "2026-11-24", user_id: null, alias: null, requests: 1, input_tokens: 1, output_tokens: 1, spend_nano_usd: 0, paid_nano_usd: 0, free_nano_usd: 0 }
          ]
        }
      }
    });
    const output = text(await exec(usage, { group_by: "day", days: 7 }));
    expect(platform!.calls("/api/gateway/usage/daily")[0]?.query).toEqual({ org_id: "org-1111", scope: "org", group_by: "day", from: "2026-11-19", to: "2026-11-25", limit: "7" });
    expect(output).toContain("Total: 5 requests, spend $3.000 (charged $2.000");
    expect(output.indexOf("2026-11-24")).toBeLessThan(output.indexOf("| 2026-11-25"));
    expect((await exec(usage, { group_by: "day", days: 0 })).isError).toBe(true);
  });

  it("refuses unknown groupings and windows", async () => {
    const usage = await tool({});
    expect((await exec(usage, { group_by: "planet" })).isError).toBe(true);
    expect((await exec(usage, { window: "1y" })).isError).toBe(true);
    expect((await exec(usage, { api_key_id: "x" })).isError).toBe(true);
  });
});
