/**
 * Opt-in live smoke against the hosted gateway: `EXP_API_KEY=xpl_... npm run test:live`.
 * Skipped otherwise; the shared test setup removes credentials from every other test.
 */

import { describe, expect, it } from "vitest";
import { HOSTED_GATEWAY_BASE_URL } from "./gateway/config.js";
import { fetchGatewayModels } from "./gateway/models.js";

const key = process.env.EXP_CODE_LIVE_API_KEY;

describe.skipIf(!key)("live hosted gateway", () => {
  it("lists models and completes one chat request", async () => {
    const config = { kind: "hosted" as const, baseUrl: process.env.EXP_CODE_BASE_URL ?? HOSTED_GATEWAY_BASE_URL, apiKey: key! };
    const models = await fetchGatewayModels(config, { ttlMs: 0 });
    expect(models.length).toBeGreaterThan(0);
    const model = process.env.EXP_CODE_MODEL ?? models[0]!.id;
    const response = await fetch(`${config.baseUrl}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${config.apiKey}` },
      body: JSON.stringify({ model, messages: [{ role: "user", content: "Reply with the single word: ready" }], max_tokens: 16 })
    });
    expect(response.status).toBe(200);
  }, 60_000);
});
