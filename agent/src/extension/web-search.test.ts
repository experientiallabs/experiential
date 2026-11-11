import { describe, expect, it } from "vitest";
import type { GatewayConfig } from "../gateway/config.js";
import { citationsOf, createWebSearchTool } from "./web-search.js";

const gateway: GatewayConfig = { kind: "hosted", baseUrl: "https://gw.example/v1", apiKey: "xpl_k" };

function reply(body: unknown, init: ResponseInit = {}) {
  const seen: { url: string; body: Record<string, unknown>; auth: string }[] = [];
  const fetch = (async (url: string, request?: RequestInit) => {
    seen.push({ url, body: JSON.parse(String(request?.body)), auth: (request?.headers as Record<string, string>).authorization ?? "" });
    return new Response(JSON.stringify(body), init);
  }) as typeof globalThis.fetch;
  return { fetch, seen };
}

const ctx = (provider: string, id: string) => ({ model: { provider, id } }) as never;

describe("web_search", () => {
  it("runs on the session's gateway model with web_search_options and lists citations", async () => {
    const { fetch, seen } = reply({
      choices: [{ message: { content: "1. Pi docs", annotations: [{ url_citation: { url: "https://a.example", title: "A" } }, { url_citation: { url: "https://a.example" } }] } }]
    });
    const result = await createWebSearchTool({ gateway, fetch }).execute("c", { query: "pi agent" }, undefined, undefined, ctx("experiential", "m1"));
    expect(seen[0]).toMatchObject({ url: "https://gw.example/v1/chat/completions", auth: "Bearer xpl_k" });
    expect(seen[0]?.body).toMatchObject({ model: "m1", web_search_options: { search_context_size: "medium" } });
    expect(result.content[0]).toEqual({ type: "text", text: "1. Pi docs\n\nSources:\n[1] A - https://a.example" });
    expect(result.details).toEqual({ query: "pi agent", citations: [{ url: "https://a.example", title: "A" }] });
  });

  it("falls back to the configured model when the session model is not a gateway model", async () => {
    const { fetch, seen } = reply({ choices: [{ message: { content: "ok" } }] });
    await createWebSearchTool({ gateway, model: "fallback", fetch }).execute("c", { query: "q" }, undefined, undefined, ctx("anthropic", "x"));
    expect(seen[0]?.body.model).toBe("fallback");
    const none = await createWebSearchTool({ gateway, fetch }).execute("c", { query: "q" }, undefined, undefined, ctx("anthropic", "x"));
    expect((none as { isError?: boolean }).isError).toBe(true);
  });

  it("reports gateway errors and an ignored search parameter as tool errors", async () => {
    const refused = await createWebSearchTool({ gateway, model: "m", fetch: reply({ error: { message: "no credits" } }, { status: 402 }).fetch }).execute("c", { query: "q" }, undefined, undefined, undefined as never);
    expect(refused.content[0]).toEqual({ type: "text", text: "Web search failed (402): no credits" });
    const ignored = await createWebSearchTool({
      gateway,
      model: "m",
      fetch: reply({ choices: [{ message: { content: "x" } }] }, { headers: { "x-experiential-ignored-parameters": "web_search_options" } }).fetch
    }).execute("c", { query: "q" }, undefined, undefined, undefined as never);
    expect((ignored as { isError?: boolean }).isError).toBe(true);
  });

  it("deduplicates citations and ignores malformed annotations", () => {
    expect(citationsOf({ annotations: [{ url_citation: { url: "u" } }, { url_citation: { url: "u" } }, { other: 1 }] })).toEqual([{ url: "u" }]);
    expect(citationsOf(undefined)).toEqual([]);
  });
});
