import { mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { GatewayConfig } from "./config.js";
import { cacheKey, chooseWireApi, clearModelCache, DEFAULT_CONTEXT_WINDOW, fetchGatewayModels, parseModelListing, parseWireApi, toProviderModel } from "./models.js";

const hosted: GatewayConfig = { kind: "hosted", baseUrl: "https://gw.example/v1", apiKey: "xpl_secret" };

let dir: string;
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "exp-code-models-"));
  clearModelCache();
});
afterEach(() => rm(dir, { recursive: true, force: true }));

describe("gateway models", () => {
  it("parses plain OpenAI listings and optional limits", () => {
    const listed = parseModelListing({
      object: "list",
      data: [
        { id: "plain", object: "model", created: 0, owned_by: "exp" },
        { id: "rich", context_window: 200_000, max_output_tokens: 8_000, input_modalities: ["text", "image"], supports_reasoning: true },
        { id: "openrouter-shape", context_length: 64_000, top_provider: { max_completion_tokens: 4_000 } },
        { object: "model" }
      ]
    });
    expect(listed).toEqual([
      { id: "plain" },
      { id: "rich", contextWindow: 200_000, maxTokens: 8_000, vision: true, reasoning: true },
      { id: "openrouter-shape", contextWindow: 64_000, maxTokens: 4_000 }
    ]);
    expect(() => parseModelListing({})).toThrow(/no data array/);
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
      return new Response(JSON.stringify({ data: [{ id: "a" }, { id: "b" }] }));
    }) as typeof globalThis.fetch;
    expect((await fetchGatewayModels(hosted, { fetch, cacheDir: dir })).map((m) => m.id)).toEqual(["a", "b"]);
    expect(auth).toBe("Bearer xpl_secret");
    await fetchGatewayModels(hosted, { fetch, cacheDir: dir });
    clearModelCache();
    await fetchGatewayModels(hosted, { fetch, cacheDir: dir });
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
