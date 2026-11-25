/**
 * `experiential_models`: the models this organization can call, with prices and context windows.
 *
 * Two reads, in parallel: `GET /v1/models` (exactly what this key can call, including the global
 * names `experiential-small-model`, `experiential-medium-model` and `experiential-large-model`)
 * and the platform catalog `GET /api/models` (display names, context and output limits, and the
 * default route's provider and token prices). Either alone is enough to answer.
 */

import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { LARGE_MODEL_ALIAS, SMALL_MODEL_ALIAS } from "../gateway/config.js";
import type { PlatformClient } from "./client.js";
import { count, failure, failureFrom, NANO_PER_USD, ratePerMillion, result, table , literalUnion } from "./format.js";

/** The gateway's global model names: request-time names for the platform's default model per role. */
export const GLOBAL_MODEL_NAMES = [SMALL_MODEL_ALIAS, "experiential-medium-model", LARGE_MODEL_ALIAS] as const;
export const MODEL_SORTS = ["preferred", "price", "context"] as const;
export const DEFAULT_MODEL_ROWS = 30;
export const MAX_MODEL_ROWS = 100;

/** One model as the tool reports it. */
export interface ModelRow {
  id: string;
  name?: string;
  provider?: string;
  contextWindow?: number;
  maxInputTokens?: number;
  maxOutputTokens?: number;
  /** Nano-dollars per million tokens. */
  inputPrice?: number;
  cachedInputPrice?: number;
  outputPrice?: number;
  callable: boolean;
  global: boolean;
  modalities?: string[];
}

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

function wholeNumber(...values: unknown[]): number | undefined {
  for (const value of values) if (typeof value === "number" && Number.isFinite(value) && value >= 0) return value;
  return undefined;
}

/** Read one `/v1/models` entry (prices under `pricing.*_nano_usd_per_million_tokens`). */
export function listedModelRow(entry: unknown): ModelRow | undefined {
  const object = record(entry);
  if (typeof object?.id !== "string" || object.id === "") return undefined;
  const pricing = record(object.pricing) ?? {};
  return {
    id: object.id,
    contextWindow: wholeNumber(object.context_window, object.context_window_tokens, object.context_length),
    maxOutputTokens: wholeNumber(object.max_output_tokens, object.maximum_output_tokens),
    inputPrice: wholeNumber(pricing.input_nano_usd_per_million_tokens),
    cachedInputPrice: wholeNumber(pricing.cached_input_nano_usd_per_million_tokens),
    outputPrice: wholeNumber(pricing.output_nano_usd_per_million_tokens),
    callable: true,
    global: (GLOBAL_MODEL_NAMES as readonly string[]).includes(object.id)
  };
}

/** Read one `/api/models` entry: the model and its default route (`default_provider_ids[0]`). */
export function catalogModelRow(entry: unknown): ModelRow | undefined {
  const object = record(entry);
  const model = record(object?.model);
  if (typeof model?.slug !== "string" || model.slug === "") return undefined;
  const providers = Array.isArray(object?.providers) ? object.providers.map(record).filter((row) => row !== undefined) : [];
  const defaults = Array.isArray(object?.default_provider_ids) ? object.default_provider_ids : [];
  const route = providers.find((row) => row.id === defaults[0]) ?? providers.find((row) => row.status === "active") ?? providers[0];
  return {
    id: model.slug,
    name: typeof model.display_name === "string" ? model.display_name : undefined,
    provider: typeof route?.provider === "string" ? route.provider : undefined,
    contextWindow: wholeNumber(model.context_window),
    maxInputTokens: wholeNumber(model.max_input_tokens),
    maxOutputTokens: wholeNumber(model.max_output_tokens),
    inputPrice: wholeNumber(route?.input_nano_usd_per_million),
    cachedInputPrice: wholeNumber(route?.cached_input_nano_usd_per_million),
    outputPrice: wholeNumber(route?.output_nano_usd_per_million),
    callable: false,
    global: false,
    modalities: Array.isArray(model.input_modalities) ? model.input_modalities.filter((item): item is string => typeof item === "string") : undefined
  };
}

/** Merge the two listings: catalog facts, callability from `/v1/models`, global names first. */
export function mergeModels(listed: ModelRow[] | undefined, catalog: ModelRow[] | undefined): ModelRow[] {
  const callable = new Map((listed ?? []).map((row) => [row.id, row]));
  const rows: ModelRow[] = (listed ?? []).filter((row) => row.global);
  if (catalog === undefined) return [...rows, ...(listed ?? []).filter((row) => !row.global)];
  for (const row of catalog) {
    const live = callable.get(row.id);
    rows.push({
      ...row,
      callable: listed === undefined ? true : live !== undefined,
      contextWindow: row.contextWindow ?? live?.contextWindow,
      maxOutputTokens: row.maxOutputTokens ?? live?.maxOutputTokens,
      inputPrice: row.inputPrice ?? live?.inputPrice,
      cachedInputPrice: row.cachedInputPrice ?? live?.cachedInputPrice,
      outputPrice: row.outputPrice ?? live?.outputPrice
    });
  }
  return rows;
}

const modelsSchema = Type.Object({
  query: Type.Optional(Type.String({ description: "Words to match in the model id or name, e.g. \"claude\", \"gemini flash\", \"coder\"" })),
  provider: Type.Optional(Type.String({ description: "Only models served by this provider (catalog provider name, e.g. openai, anthropic)" })),
  min_context: Type.Optional(Type.Number({ description: "Only models with at least this many tokens of context window" })),
  max_input_price: Type.Optional(Type.Number({ description: "Only models whose input price is at most this many dollars per million tokens" })),
  sort: Type.Optional(literalUnion(MODEL_SORTS, { description: "preferred (default, the platform's recommended order), price (cheapest input first) or context (largest first)" })),
  callable_only: Type.Optional(Type.Boolean({ description: "Only models this key can call now (default true)" })),
  limit: Type.Optional(Type.Number({ description: `Rows to return, 1 to ${MAX_MODEL_ROWS} (default ${DEFAULT_MODEL_ROWS})` }))
});

function matches(row: ModelRow, words: string[]): boolean {
  const haystack = `${row.id} ${row.name ?? ""} ${row.provider ?? ""}`.toLowerCase();
  return words.every((word) => haystack.includes(word));
}

function modelCells(row: ModelRow): string[] {
  const context = row.contextWindow === undefined ? "?" : count(row.contextWindow);
  const limits = [row.maxInputTokens !== undefined ? `in ${count(row.maxInputTokens)}` : "", row.maxOutputTokens !== undefined ? `out ${count(row.maxOutputTokens)}` : ""].filter(Boolean).join(", ");
  return [
    row.global ? `${row.id} (global name)` : row.id,
    row.name ?? "-",
    row.provider ?? "-",
    limits === "" ? context : `${context} (${limits})`,
    `${ratePerMillion(row.inputPrice)} / ${ratePerMillion(row.cachedInputPrice)} / ${ratePerMillion(row.outputPrice)}`,
    row.callable ? "yes" : "no"
  ];
}

export function createModelsTool(client: PlatformClient) {
  return defineTool({
    name: "experiential_models",
    label: "Experiential models",
    description:
      "List and search the models available through the Experiential gateway for this organization, with each model's default provider, context window and input/output limits, and token prices (dollars per million input, cached input and output tokens), marking which ones this key can call. Includes the global names experiential-small-model, experiential-medium-model and experiential-large-model (they always mean the platform's current default for that role). Use it to pick a model (\"cheapest model with a 200k context\": sort=price, min_context=200000), check a price, or explain a request's cost.",
    promptSnippet: "List Experiential models with prices, context windows and callability",
    parameters: modelsSchema,
    annotations: { readOnlyHint: true, openWorldHint: true },
    async execute(_toolCallId, args, signal) {
      const limit = args.limit ?? DEFAULT_MODEL_ROWS;
      if (!Number.isInteger(limit) || limit < 1 || limit > MAX_MODEL_ROWS) return failure(`limit must be a whole number from 1 to ${MAX_MODEL_ROWS}.`);
      const sort = args.sort ?? "preferred";
      if (!(MODEL_SORTS as readonly string[]).includes(sort)) return failure(`sort must be one of ${MODEL_SORTS.join(", ")}.`);
      if (args.min_context !== undefined && (!Number.isInteger(args.min_context) || args.min_context < 1)) return failure("min_context must be a positive whole number of tokens.");
      if (args.max_input_price !== undefined && (!Number.isFinite(args.max_input_price) || args.max_input_price < 0)) return failure("max_input_price must be a non-negative number of dollars per million tokens.");
      const filtered = args.provider !== undefined || args.min_context !== undefined || args.max_input_price !== undefined;
      const catalogQuery = {
        limit: 500,
        sort: sort === "preferred" ? "preferred" : sort,
        order: sort === "price" ? "asc" : sort === "context" ? "desc" : undefined,
        provider: args.provider?.trim(),
        min_context: args.min_context,
        max_input_nano_usd_per_million: args.max_input_price === undefined ? undefined : Math.floor(args.max_input_price * NANO_PER_USD)
      };
      const [listedRead, catalogRead] = await Promise.allSettled([
        client.get<{ data?: unknown[] }>("/v1/models", {}, signal),
        client.get<{ models?: unknown[] }>("/api/models", catalogQuery, signal)
      ]);
      if (listedRead.status === "rejected" && catalogRead.status === "rejected") return failureFrom(catalogRead.reason, { models: [] });
      const listed = listedRead.status === "fulfilled" ? (listedRead.value?.data ?? []).map(listedModelRow).filter((row) => row !== undefined) : undefined;
      const catalog = catalogRead.status === "fulfilled" ? (catalogRead.value?.models ?? []).map(catalogModelRow).filter((row) => row !== undefined) : undefined;
      let rows = mergeModels(listed, catalog);
      // The catalog applied the filters; without it, or for global names (copies of a listed model), apply them here.
      rows = rows.filter((row) => {
        if (!filtered || (catalog !== undefined && !row.global)) return true;
        if (args.provider !== undefined && row.provider !== undefined && row.provider !== args.provider.trim()) return false;
        if (args.provider !== undefined && row.global) return false;
        if (args.min_context !== undefined && (row.contextWindow ?? 0) < args.min_context) return false;
        if (args.max_input_price !== undefined && (row.inputPrice === undefined || row.inputPrice > args.max_input_price * NANO_PER_USD)) return false;
        return true;
      });
      const words = (args.query ?? "").toLowerCase().split(/\s+/).filter(Boolean);
      if (words.length > 0) rows = rows.filter((row) => matches(row, words));
      if (args.callable_only !== false) rows = rows.filter((row) => row.callable);
      if (sort === "price") rows.sort((a, b) => (a.inputPrice ?? Number.MAX_SAFE_INTEGER) - (b.inputPrice ?? Number.MAX_SAFE_INTEGER));
      if (sort === "context") rows.sort((a, b) => (b.contextWindow ?? 0) - (a.contextWindow ?? 0));
      const shown = rows.slice(0, limit);
      const lines = [`${rows.length} model${rows.length === 1 ? "" : "s"} match${rows.length === 1 ? "es" : ""}${rows.length > shown.length ? ` (showing ${shown.length})` : ""}. Prices are list prices of the default route in dollars per million tokens (input / cached input / output); a request's billed cost can differ with promotions, plan discounts or fallbacks.`];
      if (listed === undefined) lines.push(`The callable model list was unavailable (${(listedRead as PromiseRejectedResult).reason?.message ?? "error"}), so callability is not checked.`);
      if (catalog === undefined) lines.push(`The catalog was unavailable (${(catalogRead as PromiseRejectedResult).reason?.message ?? "error"}), so names and some prices are missing.`);
      if (shown.length > 0) lines.push("", table(["model", "name", "provider", "context (limits)", "price in / cached / out", "callable"], shown.map(modelCells)));
      else lines.push("", "No model matched. Loosen the query or filters.");
      return result(lines.join("\n"), { models: shown, total: rows.length });
    }
  });
}
