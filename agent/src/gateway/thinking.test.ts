/**
 * Thinking levels for gateway models: the default each model starts at, when "off" exists, and
 * the reasoning field each wire API sends for the level the footer shows. Models are built from
 * listing entries shaped like the hosted gateway's (`supports_reasoning`,
 * `supported_reasoning_efforts`), and sessions run against the loopback fake gateway.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createAgentSession, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { type FakeGateway, startFakeGateway } from "../testing/fake-gateway.js";
import { GATEWAY_PROVIDER_ID, type GatewayConfig } from "./config.js";
import { type GatewayWireApi, parseListedModel } from "./models.js";
import { gatewayProviderConfig, gatewayProviderModels } from "./provider.js";

/** deepseek-v4.1-flash-uncensored as an always-reasoning route lists it: no `none` effort. */
const ALWAYS_REASONS = {
  id: "deepseek-v4.1-flash-uncensored",
  context_window_tokens: 1_048_576,
  maximum_output_tokens: 393_216,
  supports_reasoning: true,
  reasoning_effort: "high",
  supported_reasoning_efforts: ["low", "high"],
  reasoning_output_exposed: true,
  reasoning_content_native: true
};
/** The same model as its live lane lists it today: thinking can be turned off (`none`). */
const DEEPSEEK = { ...ALWAYS_REASONS, id: "deepseek-v4.1-flash", supported_reasoning_efforts: ["none", "low", "high"] };
/** gpt-6-luna: none, low and medium. */
const LUNA = {
  id: "gpt-6-luna",
  context_window_tokens: 1_050_000,
  maximum_output_tokens: 128_000,
  supports_reasoning: true,
  reasoning_effort: "medium",
  supported_reasoning_efforts: ["none", "low", "medium"]
};
/** claude-sonnet-5.5 as the hosted gateway lists it: an effort ladder, no `none`. */
const SONNET = {
  id: "claude-sonnet-5.5",
  context_window_tokens: 1_000_000,
  maximum_output_tokens: 128_000,
  supports_reasoning: true,
  reasoning_effort: "high",
  supported_reasoning_efforts: ["low", "medium", "high", "xhigh", "max"]
};
/** A model without reasoning. */
const PLAIN = { id: "plain-model", context_window_tokens: 128_000, maximum_output_tokens: 16_000, supports_reasoning: false };

let gateway: FakeGateway;
let agentDir: string;

beforeAll(async () => {
  gateway = await startFakeGateway([], [ALWAYS_REASONS, DEEPSEEK, LUNA, PLAIN]);
  agentDir = mkdtempSync(join(tmpdir(), "exp-code-thinking-"));
});

afterAll(async () => {
  await gateway.close();
  rmSync(agentDir, { recursive: true, force: true });
});

const config = (): GatewayConfig => ({ kind: "hosted", baseUrl: gateway.baseUrl, apiKey: "xpl_test_key" });

/** A session on the gateway's models (Chat Completions for all, as the fake serves only that). */
async function session(startOn: string, settings: SettingsManager = SettingsManager.inMemory(), wire: GatewayWireApi = "openai-completions") {
  const listed = [ALWAYS_REASONS, DEEPSEEK, LUNA, PLAIN].map((entry) => parseListedModel(entry)!);
  const modelRuntime = await ModelRuntime.create({ authPath: join(agentDir, "auth.json"), modelsPath: join(agentDir, "models.json") });
  modelRuntime.registerProvider(GATEWAY_PROVIDER_ID, gatewayProviderConfig(config(), gatewayProviderModels(listed, config(), wire)));
  const model = modelRuntime.getModel(GATEWAY_PROVIDER_ID, startOn)!;
  const { session } = await createAgentSession({ cwd: agentDir, agentDir, modelRuntime, model, sessionManager: SessionManager.inMemory(agentDir), settingsManager: settings, noTools: "all" });
  return { session, model: (id: string) => modelRuntime.getModel(GATEWAY_PROVIDER_ID, id)! };
}

describe("the thinking level a model starts at", () => {
  it("is medium, or the closest level the model accepts", async () => {
    expect((await session(LUNA.id)).session.thinkingLevel).toBe("medium");
    // No medium on these routes: Pi takes the next level up.
    expect((await session(DEEPSEEK.id)).session.thinkingLevel).toBe("high");
    expect((await session(ALWAYS_REASONS.id)).session.thinkingLevel).toBe("high");
    expect((await session(PLAIN.id)).session.thinkingLevel).toBe("off");
  });

  it("offers off only on a route that accepts the none effort", async () => {
    const { session: s, model } = await session(ALWAYS_REASONS.id);
    expect(s.getAvailableThinkingLevels()).toEqual(["low", "high"]);
    await s.setModel(model(DEEPSEEK.id));
    expect(s.getAvailableThinkingLevels()).toEqual(["off", "low", "high"]);
    // An explicit off on an always-reasoning route becomes its lowest level, never "off".
    await s.setModel(model(ALWAYS_REASONS.id));
    s.setThinkingLevel("off");
    expect(s.thinkingLevel).toBe("low");
  });

  it("does not carry off from a model without reasoning to one with it", async () => {
    const { session: s, model } = await session(PLAIN.id);
    expect(s.thinkingLevel).toBe("off");
    await s.setModel(model(ALWAYS_REASONS.id));
    expect(s.thinkingLevel).toBe("high");
    await s.setModel(model(LUNA.id));
    expect(s.thinkingLevel).toBe("medium");
  });

  it("keeps a level the user chose for the model", async () => {
    const settings = SettingsManager.inMemory();
    settings.setModelThinkingLevel(GATEWAY_PROVIDER_ID, LUNA.id, "low");
    const { session: s, model } = await session(PLAIN.id, settings);
    await s.setModel(model(LUNA.id));
    expect(s.thinkingLevel).toBe("low");
    expect((await session(LUNA.id, settings)).session.thinkingLevel).toBe("low");
  });

  it("follows a default level the user saved", async () => {
    const settings = SettingsManager.inMemory({ defaultThinkingLevel: "low" });
    expect((await session(LUNA.id, settings)).session.thinkingLevel).toBe("low");
  });
});

describe("the level reaches the request", () => {
  async function sentEffort(modelId: string, level?: "off" | "low" | "medium" | "high"): Promise<unknown> {
    const { session: s } = await session(modelId);
    if (level !== undefined) s.setThinkingLevel(level);
    gateway.script([{ text: "ok" }]);
    const before = gateway.completions().length;
    await s.prompt("hi");
    const bodies = gateway.completions().slice(before);
    expect(bodies).toHaveLength(1);
    return bodies[0]!.reasoning_effort;
  }

  it("sends the level the footer shows", async () => {
    expect(await sentEffort(LUNA.id)).toBe("medium");
    expect(await sentEffort(LUNA.id, "off")).toBe("none");
    expect(await sentEffort(DEEPSEEK.id)).toBe("high");
    expect(await sentEffort(DEEPSEEK.id, "off")).toBe("none");
    expect(await sentEffort(ALWAYS_REASONS.id)).toBe("high");
    expect(await sentEffort(ALWAYS_REASONS.id, "off")).toBe("low");
    expect(await sentEffort(PLAIN.id)).toBeUndefined();
  });

  it("on every wire API", async () => {
    const { streamSimple } = await import("@earendil-works/pi-ai/compat");
    const payload = async (wire: GatewayWireApi, id: string, level: "off" | "medium") => {
      const listed = parseListedModel([LUNA, ALWAYS_REASONS, SONNET].find((entry) => entry.id === id))!;
      // A closed port: the payload is built and handed to onPayload before the request fails.
      const [model] = gatewayProviderModels([listed], { kind: "hosted", baseUrl: "http://127.0.0.1:9/v1", apiKey: "xpl_test_key" }, wire);
      let sent: Record<string, unknown> | undefined;
      const stream = streamSimple({ ...model!, provider: GATEWAY_PROVIDER_ID } as never, { messages: [{ role: "user", content: "hi", timestamp: 0 }] }, {
        apiKey: "xpl_test_key",
        reasoning: level === "off" ? undefined : level,
        maxRetries: 0,
        onPayload: (body: unknown) => {
          sent = body as Record<string, unknown>;
          return undefined;
        }
      });
      await stream.result();
      return sent;
    };
    expect((await payload("openai-responses", LUNA.id, "medium"))?.reasoning).toMatchObject({ effort: "medium" });
    expect((await payload("openai-responses", LUNA.id, "off"))?.reasoning).toEqual({ effort: "none" });
    expect((await payload("openai-completions", LUNA.id, "medium"))?.reasoning_effort).toBe("medium");
    // Claude on the Messages lane: adaptive thinking with the effort, never a numeric budget
    // (Claude 4.6 and later refuse `budget_tokens`).
    const claude = await payload("anthropic-messages", SONNET.id, "medium");
    expect(claude?.thinking).toMatchObject({ type: "adaptive" });
    expect(claude?.thinking).not.toHaveProperty("budget_tokens");
    expect(claude?.output_config).toEqual({ effort: "medium" });
  });
});
