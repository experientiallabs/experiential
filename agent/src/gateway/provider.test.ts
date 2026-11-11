import { describe, expect, it } from "vitest";
import type { GatewayConfig } from "./config.js";
import { gatewayProviderConfig, gatewayProviderModels, literalApiKey, reportedBilling, reportedCost } from "./provider.js";

const local: GatewayConfig = { kind: "local", baseUrl: "http://127.0.0.1:8000/v1", apiKey: "k" };

describe("gateway provider", () => {
  it("escapes keys so Pi never reads them as env or command syntax", () => {
    expect(literalApiKey("xpl_abc")).toBe("xpl_abc");
    expect(literalApiKey("a$b")).toBe("a$$b");
    expect(literalApiKey("!cmd")).toBe("$!cmd");
  });

  it("reads the settled cost from Chat Completions and Responses stream events", () => {
    expect(reportedCost({ choices: [], usage: { prompt_tokens: 1, cost: 0.002 } })).toBe(0.002);
    expect(reportedCost({ type: "response.completed", response: { usage: { cost: 0.5 } } })).toBe(0.5);
    expect(reportedCost({ usage: { cost: "free" } })).toBeUndefined();
    expect(reportedCost({ choices: [{ delta: { content: "hi" } }] })).toBeUndefined();
    expect(reportedCost(null)).toBeUndefined();
  });

  it("registers one model per alias under the gateway's endpoint", () => {
    const models = gatewayProviderModels([{ id: "a" }, { id: "gpt-6-luna" }], local);
    const config = gatewayProviderConfig(local, models);
    expect(config).toMatchObject({ baseUrl: local.baseUrl, apiKey: "k", api: "openai-completions", name: "Experiential (local gateway)" });
    expect(config.models?.map((model) => [model.id, model.api])).toEqual([
      ["a", "openai-completions"],
      ["gpt-6-luna", "openai-responses"]
    ]);
  });
});

describe("reportedBilling", () => {
  it("reads the credits charged and, for a call on the caller's own key, the provider's list price", () => {
    expect(reportedBilling({ usage: { cost: 0.002 } })).toEqual({ cost: 0.002, byok: false, byokListUsd: null });
    expect(reportedBilling({ usage: { cost: 0, is_byok: true, cost_details: { upstream_inference_cost: 0.0004 } } })).toEqual({ cost: 0, byok: true, byokListUsd: 0.0004 });
    expect(reportedBilling({ usage: { cost: 0, is_byok: true } })).toEqual({ cost: 0, byok: true, byokListUsd: null });
    expect(reportedBilling({ usage: { prompt_tokens: 1 } })).toBeUndefined();
  });
});
