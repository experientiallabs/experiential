import { mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { GatewayConfig } from "./config.js";
import { type Api, getSupportedThinkingLevels, type Model } from "@earendil-works/pi-ai";
import {
  agentCanUse,
  cacheKey,
  chooseWireApi,
  clearModelCache,
  DEFAULT_CONTEXT_WINDOW,
  fetchGatewayModels,
  parseModelListing,
  parseWireApi,
  thinkingLevelMap,
  toProviderModel
} from "./models.js";

const hosted: GatewayConfig = { kind: "hosted", baseUrl: "https://gw.example/v1", apiKey: "xpl_secret" };

let dir: string;
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "exp-code-models-"));
  clearModelCache();
});
afterEach(() => rm(dir, { recursive: true, force: true }));

describe("gateway models", () => {
  it("parses plain OpenAI listings and the gateway's limits and capabilities", () => {
    const listed = parseModelListing({
      object: "list",
      data: [
        { id: "plain", object: "model", created: 0, owned_by: "exp" },
        {
          id: "experiential-large-model",
          context_window_tokens: 1_050_000,
          maximum_output_tokens: 128_000,
          supports_reasoning: false,
          reasoning_effort: null,
          supported_reasoning_efforts: []
        },
        {
          id: "gpt-6-luna",
          context_window_tokens: 1_050_000,
          maximum_output_tokens: 128_000,
          supports_image_input: true,
          supports_reasoning: true,
          reasoning_effort: "medium",
          supported_reasoning_efforts: ["none", "low", "medium", "high", "xhigh", "max"]
        },
        { id: "bad-limits", context_window_tokens: -5, maximum_output_tokens: 1.5, supported_reasoning_efforts: ["high", 3] },
        { object: "model" }
      ]
    });
    expect(listed).toEqual([
      { id: "plain" },
      { id: "experiential-large-model", contextWindow: 1_050_000, maxTokens: 128_000, reasoning: false },
      { id: "gpt-6-luna", contextWindow: 1_050_000, maxTokens: 128_000, vision: true, reasoning: true, reasoningEfforts: ["none", "low", "medium", "high", "xhigh", "max"] },
      { id: "bad-limits" }
    ]);
    expect(() => parseModelListing({})).toThrow(/no data array/);
  });

  it("lists only aliases that serve chat with tools", () => {
    // exp-code sends its tools on every turn: the gateway refuses these aliases with a 400
    // (`unsupported_capability`) before any call, so the picker never offers them.
    const listed = parseModelListing({
      data: [
        { id: "coder", supports_completions: true, supports_tools: true },
        { id: "sonar", supports_completions: true, supports_tools: false },
        { id: "tts-1", supports_completions: false, supports_tools: null },
        { id: "text-embedding-3-small", supports_completions: false, supports_tools: false, supports_embeddings: true },
        // A local gateway states neither: kept.
        { id: "local-alias" }
      ]
    });
    expect(listed.map((model) => model.id)).toEqual(["coder", "local-alias"]);
    expect(agentCanUse({ id: "x", supports_tools: null, supports_completions: null })).toBe(true);
  });

  it("no longer reads the OpenRouter-style names neither gateway sends", () => {
    // Both gateways exp-code talks to (hosted, and a local `exp` gateway) publish
    // context_window_tokens / maximum_output_tokens; reading these names silently left every
    // model on the 128k default.
    expect(
      parseModelListing({ data: [{ id: "legacy", context_window: 200_000, context_length: 64_000, max_output_tokens: 8_000, top_provider: { max_completion_tokens: 4_000 } }] })
    ).toEqual([{ id: "legacy" }]);
  });

  it("states the gateway's window and output reservation on the Pi model", () => {
    const [large] = parseModelListing({ data: [{ id: "experiential-large-model", context_window_tokens: 1_050_000, maximum_output_tokens: 128_000, supports_reasoning: false, supported_reasoning_efforts: [] }] });
    expect(toProviderModel(large!, hosted)).toMatchObject({ contextWindow: 1_050_000, maxTokens: 128_000, reasoning: false });
    expect(toProviderModel(large!, hosted)).not.toHaveProperty("thinkingLevelMap");
  });

  it("offers only the thinking levels a route accepts", () => {
    const levels = (efforts: string[] | undefined, reasoning = true) => {
      const model = { ...toProviderModel({ id: "m", reasoning, reasoningEfforts: efforts }, hosted), provider: "experiential" } as unknown as Model<Api>;
      return getSupportedThinkingLevels(model);
    };
    expect(levels(["none", "low", "medium", "high", "xhigh", "max"])).toEqual(["off", "low", "medium", "high", "xhigh", "max"]);
    expect(levels(["none", "high", "max"])).toEqual(["off", "high", "max"]);
    // No `none`: thinking cannot be turned off on this route.
    expect(levels(["high", "max", "ultra"])).toEqual(["high", "max"]);
    expect(levels(["none"])).toEqual(["off"]);
    expect(levels([], false)).toEqual(["off"]);
    // A local gateway states no efforts: Pi's own levels apply.
    expect(levels(undefined)).toEqual(["off", "minimal", "low", "medium", "high"]);
    expect(thinkingLevelMap(["none", "high"])).toEqual({ off: "none", minimal: null, low: null, medium: null, high: "high", xhigh: null, max: null });
    expect(toProviderModel({ id: "m", reasoning: true, reasoningEfforts: ["none"] }, hosted)).toMatchObject({ reasoning: false });
  });

  it("chooses a wire API per model, with an override", () => {
    expect(chooseWireApi("deepseek-v4", "hosted")).toBe("openai-completions");
    expect(chooseWireApi("gpt-6-luna", "hosted")).toBe("openai-responses");
    expect(chooseWireApi("o3", "local")).toBe("openai-responses");
    expect(chooseWireApi("claude-sonnet-5", "hosted")).toBe("anthropic-messages");
    expect(chooseWireApi("claude-sonnet-5", "local")).toBe("openai-completions");
    expect(chooseWireApi("claude-sonnet-5", "hosted", "openai-completions")).toBe("openai-completions");
    expect(parseWireApi("responses")).toBe("openai-responses");
    expect(parseWireApi(undefined)).toBeUndefined();
    expect(() => parseWireApi("grpc")).toThrow(/Unknown wire API/);
  });

  it("builds Pi models with defaults and the right base URL per wire", () => {
    expect(toProviderModel({ id: "plain" }, hosted)).toMatchObject({ id: "plain", api: "openai-completions", baseUrl: hosted.baseUrl, contextWindow: DEFAULT_CONTEXT_WINDOW, input: ["text"], reasoning: false });
    expect(toProviderModel({ id: "claude-x" }, hosted)).toMatchObject({ api: "anthropic-messages", baseUrl: "https://gw.example", reasoning: true });
    expect(toProviderModel({ id: "m", contextWindow: 8_000, maxTokens: 50_000 }, hosted)).toMatchObject({ maxTokens: 4_000 });
  });

  it("fetches with the key, caches in memory and on disk without storing the key", async () => {
    let calls = 0;
    let auth = "";
    const fetch = (async (url: string, init?: RequestInit) => {
      calls += 1;
      auth = (init?.headers as Record<string, string>).authorization ?? "";
      expect(url).toBe("https://gw.example/v1/models");
      return new Response(
        JSON.stringify({ data: [{ id: "a", context_window_tokens: 1_050_000, maximum_output_tokens: 128_000, supports_reasoning: true, supported_reasoning_efforts: ["none", "high"] }, { id: "b" }] })
      );
    }) as typeof globalThis.fetch;
    const first = await fetchGatewayModels(hosted, { fetch, cacheDir: dir });
    expect(first.map((m) => m.id)).toEqual(["a", "b"]);
    expect(auth).toBe("Bearer xpl_secret");
    await fetchGatewayModels(hosted, { fetch, cacheDir: dir });
    clearModelCache();
    // The disk copy keeps every field, so a cached start states the same limits as a fresh one.
    expect(await fetchGatewayModels(hosted, { fetch, cacheDir: dir })).toEqual(first);
    expect(calls).toBe(1);
    const [file] = await readdir(dir);
    expect(await readFile(join(dir, file!), "utf8")).not.toContain("xpl_secret");
    await fetchGatewayModels(hosted, { fetch, cacheDir: dir, ttlMs: 0 });
    expect(calls).toBe(2);
  });

  it("explains a refused listing", async () => {
    const fetch = (async () => new Response(JSON.stringify({ error: { message: "bad key" } }), { status: 401 })) as typeof globalThis.fetch;
    await expect(fetchGatewayModels(hosted, { fetch })).rejects.toThrow(/refused the model listing \(401\): bad key\. Check EXP_API_KEY/);
  });
});

describe("listing cache key", () => {
  it("uses only the key's public prefix, never the secret", () => {
    const base = { kind: "hosted" as const, baseUrl: "https://gw.example/v1" };
    const a = cacheKey({ ...base, apiKey: "xpl_abcdefgSECRET-1" });
    expect(cacheKey({ ...base, apiKey: "xpl_abcdefgSECRET-2" })).toBe(a);
    expect(cacheKey({ ...base, apiKey: "xpl_zzzzzzzSECRET-1" })).not.toBe(a);
    expect(cacheKey({ ...base, apiKey: "local-virtual-key" })).toBe(cacheKey({ ...base, apiKey: "another-local-key" }));
    expect(cacheKey({ ...base, baseUrl: "http://127.0.0.1:8080/v1", apiKey: "xpl_abcdefgSECRET-1" })).not.toBe(a);
  });
});

describe("output reservation", () => {
  it("never reserves more than half the context window, so the prompt always fits", () => {
    const config = { kind: "hosted" as const, baseUrl: "https://gw.example/v1", apiKey: "xpl_k" };
    expect(toProviderModel({ id: "small", contextWindow: 8192 }, config)).toMatchObject({ maxTokens: 4096 });
    expect(toProviderModel({ id: "small-listed", contextWindow: 8192, maxTokens: 8192 }, config)).toMatchObject({ maxTokens: 4096 });
    expect(toProviderModel({ id: "big", contextWindow: 400_000, maxTokens: 128_000 }, config)).toMatchObject({ maxTokens: 128_000 });
    expect(toProviderModel({ id: "default", contextWindow: 1_000_000 }, config)).toMatchObject({ maxTokens: 32_000 });
  });
});
