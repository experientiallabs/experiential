import { describe, expect, it } from "vitest";
import type { GatewayConfig } from "./config.js";
import { gatewayProviderConfig, gatewayProviderModels, literalApiKey, readableGatewayError, registerGatewayProvider, reportedBilling, reportedCost } from "./provider.js";

const local: GatewayConfig = { kind: "local", baseUrl: "http://127.0.0.1:8000/v1", apiKey: "k" };

describe("gateway provider", () => {
  it("escapes keys so Pi never reads them as env or command syntax", () => {
    expect(literalApiKey("xpl_abc")).toBe("xpl_abc");
    expect(literalApiKey("a$b")).toBe("a$$b");
    expect(literalApiKey("!cmd")).toBe("$!cmd");
  });

  it("turns a raw gateway error body into one readable line that keeps status, type and code", () => {
    expect(
      readableGatewayError('403: {"message":"The requested model alias is not granted to this identity.","type":"permission_error","param":"model","code":"model_not_granted"}')
    ).toBe("403 (permission_error, model_not_granted): The requested model alias is not granted to this identity.");
    expect(readableGatewayError('429: {"error":{"message":"Slow down.","type":"rate_limit_error"}}')).toBe("429 (rate_limit_error): Slow down.");
    expect(readableGatewayError('500: {"message":"boom","type":"server_error","code":"server_error"}')).toBe("500 (server_error): boom");
    expect(readableGatewayError("Connection error.")).toBeUndefined();
    expect(readableGatewayError("502: <html>Bad Gateway</html>")).toBeUndefined();
    expect(readableGatewayError('400: {"detail":"no message field"}')).toBeUndefined();
  });

  it("reads the settled cost from Chat Completions and Responses stream events", () => {
    expect(reportedCost({ choices: [], usage: { prompt_tokens: 1, cost: 0.002 } })).toBe(0.002);
    expect(reportedCost({ type: "response.completed", response: { usage: { cost: 0.5 } } })).toBe(0.5);
    expect(reportedCost({ usage: { cost: "free" } })).toBeUndefined();
    expect(reportedCost({ choices: [{ delta: { content: "hi" } }] })).toBeUndefined();
    expect(reportedCost(null)).toBeUndefined();
  });

  it("sends the gateway's headers with every call, escaped like the key", () => {
    const config = gatewayProviderConfig({ ...local, headers: { "X-Explabs-Session-Id": "s", "X-Odd": "$HOME" } }, []);
    expect(config.headers).toEqual({ "X-Explabs-Session-Id": "s", "X-Odd": "$$HOME" });
    expect(gatewayProviderConfig(local, []).headers).toBeUndefined();
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
    expect(reportedBilling({ usage: { cost: 0.002 } })).toEqual({ cost: 0.002, byok: false, byokListUsd: null, listUsd: null });
    expect(reportedBilling({ usage: { cost: 0, is_byok: true, cost_details: { upstream_inference_cost: 0.0004 } } })).toEqual({ cost: 0, byok: true, byokListUsd: 0.0004, listUsd: null });
    expect(reportedBilling({ usage: { cost: 0, is_byok: true } })).toEqual({ cost: 0, byok: true, byokListUsd: null, listUsd: null });
    expect(reportedBilling({ usage: { cost: 0, cost_details: { list_cost: 0.003 } } })).toEqual({ cost: 0, byok: false, byokListUsd: null, listUsd: 0.003 });
    // A list figure at or below the charge is no saving.
    expect(reportedBilling({ usage: { cost: 0.003, cost_details: { list_cost: 0.003 } } })?.listUsd).toBeNull();
    expect(reportedBilling({ usage: { prompt_tokens: 1 } })).toBeUndefined();
  });
});

describe("the model a gateway message names", () => {
  /** Register the provider on a fake extension API and return its event handlers. */
  async function handlers(): Promise<Map<string, (event: unknown) => unknown>> {
    const on = new Map<string, (event: unknown) => unknown>();
    const pi = { registerProvider() {}, on: (name: string, handler: (event: unknown) => unknown) => on.set(name, handler) };
    await registerGatewayProvider(pi as never, local, { models: [{ id: "experiential-small-model" }] });
    return on;
  }
  const answer = { role: "assistant", provider: "experiential", model: "experiential-small-model", responseModel: "experiential/gpt-6-luna", usage: { cost: { total: 0 } } };

  it("is the model the user chose, not the one the gateway routed it to", async () => {
    const on = await handlers();
    const ended = on.get("message_end")!({ message: answer }) as { message: Record<string, unknown> };
    expect(ended.message.model).toBe("experiential-small-model");
    expect(ended.message).not.toHaveProperty("responseModel");
  });

  it("carries the settled cost on that model", async () => {
    const on = await handlers();
    on.get("message_start")!({ message: { role: "assistant" } });
    on.get("provider_stream_event")!({ provider: "experiential", data: { usage: { cost: 0.25 } } });
    const ended = on.get("message_end")!({ message: answer }) as { message: { usage: { cost: { total: number } }; responseModel?: string } };
    expect(ended.message.usage.cost.total).toBe(0.25);
    expect(ended.message.responseModel).toBeUndefined();
  });
});
