import { afterEach, describe, expect, it } from "vitest";
import { type FakePlatform, startFakePlatform } from "../testing/fake-platform.js";
import { catalogModelRow, createModelsTool, listedModelRow, mergeModels } from "./catalog.js";
import { PlatformClient } from "./client.js";
import { exec, text } from "../testing/tool-results.js";

let platform: FakePlatform | undefined;
afterEach(async () => {
  await platform?.close();
  platform = undefined;
});

const price = (input: number, output: number) => ({ input_nano_usd_per_million_tokens: input * 1e9, cached_input_nano_usd_per_million_tokens: (input / 10) * 1e9, output_nano_usd_per_million_tokens: output * 1e9 });

const LISTING = {
  object: "list",
  data: [
    { id: "experiential-large-model", object: "model", context_window: 400000, pricing: price(1.25, 10) },
    { id: "gpt-6-luna", object: "model", context_window: 400000, pricing: price(1.25, 10) },
    { id: "cheap-coder", object: "model", context_window: 128000, pricing: price(0.1, 0.4) }
  ]
};

function entry(slug: string, name: string, context: number, input: number, output: number, provider = "openai") {
  return {
    model: { id: `m-${slug}`, slug, display_name: name, context_window: context, max_input_tokens: context - 1000, max_output_tokens: 1000, input_modalities: ["text"] },
    providers: [
      { id: `p-other-${slug}`, provider: "other", status: "active", input_nano_usd_per_million: 99e9, output_nano_usd_per_million: 99e9 },
      { id: `p-${slug}`, provider, status: "active", input_nano_usd_per_million: input * 1e9, cached_input_nano_usd_per_million: null, output_nano_usd_per_million: output * 1e9 }
    ],
    default_provider_ids: [`p-${slug}`]
  };
}

const CATALOG = {
  models: [entry("gpt-6-luna", "GPT-6 Luna", 400000, 1.25, 10), entry("cheap-coder", "Cheap Coder", 128000, 0.1, 0.4, "fireworks"), entry("listed-only", "Not Callable", 32000, 0.05, 0.1)],
  promotions: [],
  total: 3,
  limit: 500,
  offset: 0,
  integrated_total: 3
};

async function tool(routes: Parameters<typeof startFakePlatform>[0]) {
  platform = await startFakePlatform(routes);
  return createModelsTool(new PlatformClient({ origin: platform.origin, apiKey: "xpl_k" }));
}

describe("experiential_models", () => {
  it("parses both listings and takes the default route's price", () => {
    expect(listedModelRow(LISTING.data[0])).toMatchObject({ id: "experiential-large-model", global: true, callable: true, inputPrice: 1.25e9, contextWindow: 400000 });
    expect(listedModelRow({ id: "" })).toBeUndefined();
    expect(catalogModelRow(CATALOG.models[1])).toMatchObject({ id: "cheap-coder", provider: "fireworks", inputPrice: 0.1e9, outputPrice: 0.4e9, maxInputTokens: 127000 });
    const merged = mergeModels([listedModelRow(LISTING.data[0])!, listedModelRow(LISTING.data[1])!], [catalogModelRow(CATALOG.models[0])!, catalogModelRow(CATALOG.models[2])!]);
    expect(merged.map((row) => [row.id, row.callable])).toEqual([
      ["experiential-large-model", true],
      ["gpt-6-luna", true],
      ["listed-only", false]
    ]);
    expect(merged[1]?.cachedInputPrice).toBe(0.125e9);
  });

  it("lists callable models with global names first and prices per million", async () => {
    const models = await tool({ "GET /v1/models": { body: LISTING }, "GET /api/models": { body: CATALOG } });
    const result = await exec(models, {});
    const output = text(result);
    expect(output).toContain("3 models match.");
    expect(output).toContain("| experiential-large-model (global name) | - | - | 400,000 | $1.25 / $0.13 / $10.00 | yes |");
    expect(output).toContain("| cheap-coder | Cheap Coder | fireworks | 128,000 (in 127,000, out 1,000) | $0.10 / $0.010 / $0.40 | yes |");
    expect(output).not.toContain("listed-only");
    expect(platform!.calls("/api/models")[0]?.query).toEqual({ limit: "500", sort: "preferred" });
    expect(platform!.calls("/v1/models")[0]?.authorization).toBe("Bearer xpl_k");
    expect(text(await exec(models, { callable_only: false }))).toContain("| listed-only | Not Callable |");
  });

  it("sorts by price, filters by words, and passes catalog filters through", async () => {
    const models = await tool({ "GET /v1/models": { body: LISTING }, "GET /api/models": { body: CATALOG } });
    const cheapest = text(await exec(models, { sort: "price", min_context: 100000, max_input_price: 2 }));
    expect(cheapest.indexOf("cheap-coder")).toBeLessThan(cheapest.indexOf("gpt-6-luna"));
    expect(platform!.calls("/api/models")[0]?.query).toEqual({ limit: "500", sort: "price", order: "asc", min_context: "100000", max_input_nano_usd_per_million: "2000000000" });
    const coder = text(await exec(models, { query: "coder" }));
    expect(coder).toContain("1 model matches.");
    expect((await exec(models, { limit: 0 })).isError).toBe(true);
    expect((await exec(models, { min_context: -1 })).isError).toBe(true);
  });

  it("answers from the gateway listing when the catalog is down, and fails when both are", async () => {
    const models = await tool({ "GET /v1/models": { body: LISTING }, "GET /api/models": { status: 503, body: { error: "down" } } });
    const output = text(await exec(models, { sort: "price" }));
    expect(output).toContain("The catalog was unavailable");
    expect(output.indexOf("cheap-coder")).toBeLessThan(output.indexOf("gpt-6-luna"));
    platform!.route("GET /v1/models", { status: 401, body: { error: { message: "bad key" } } });
    const failed = await exec(models, {});
    expect(failed.isError).toBe(true);
  });
});
