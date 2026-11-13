/**
 * The gateway's model catalog, as Pi model definitions.
 *
 * `GET /v1/models` lists the aliases the key may call. The hosted and local gateways return plain
 * OpenAI model objects (`id`, `object`, `created`, `owned_by`); optional limits and modalities are
 * read when a listing carries them, with conservative defaults otherwise. Listings are cached
 * briefly in memory and, when a cache directory is given, on disk, so startup does not wait on the
 * network every time.
 */

import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import type { GatewayConfig, GatewayKind } from "./config.js";

/** One callable alias and whatever limits its listing states. */
export interface ListedModel {
  id: string;
  contextWindow?: number;
  maxTokens?: number;
  vision?: boolean;
  reasoning?: boolean;
}

/** Wire APIs the gateway serves that Pi can speak. */
export type GatewayWireApi = "openai-completions" | "openai-responses" | "anthropic-messages";

export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 32_000;
/** How long a listing is reused before the gateway is asked again. */
export const MODEL_CACHE_TTL_MS = 5 * 60_000;
const LISTING_TIMEOUT_MS = 10_000;

const WIRE_ALIASES: Record<string, GatewayWireApi> = {
  completions: "openai-completions",
  chat: "openai-completions",
  "openai-completions": "openai-completions",
  responses: "openai-responses",
  "openai-responses": "openai-responses",
  messages: "anthropic-messages",
  anthropic: "anthropic-messages",
  "anthropic-messages": "anthropic-messages"
};

/** Parse a wire override (`completions`, `responses`, `messages`, or a full Pi API name). */
export function parseWireApi(value: string | undefined): GatewayWireApi | undefined {
  if (value === undefined || value.trim() === "") return undefined;
  const wire = WIRE_ALIASES[value.trim().toLowerCase()];
  if (wire === undefined) {
    throw new Error(`Unknown wire API "${value}". Use completions, responses, or messages.`);
  }
  return wire;
}

/**
 * Pick the wire API for one alias.
 *
 * Chat Completions is the default because every gateway route serves it. OpenAI reasoning
 * families use Responses, which keeps encrypted reasoning across tool turns. Claude aliases use
 * the Anthropic Messages lane, which only the hosted gateway serves.
 */
export function chooseWireApi(modelId: string, kind: GatewayKind, override?: GatewayWireApi): GatewayWireApi {
  if (override !== undefined) return override;
  const id = modelId.toLowerCase().replace(/^[^/]+\//, "");
  if (kind === "hosted" && id.startsWith("claude")) return "anthropic-messages";
  if (/^(gpt-5|gpt-6|o[1-9](-|$)|codex)/.test(id)) return "openai-responses";
  return "openai-completions";
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : undefined;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

/** Read one listing entry, keeping only the fields exp-code uses. */
export function parseListedModel(entry: unknown): ListedModel | undefined {
  const object = record(entry);
  if (object === undefined || typeof object.id !== "string" || object.id.trim() === "") return undefined;
  const topProvider = record(object.top_provider);
  const architecture = record(object.architecture);
  const modalities = object.input_modalities ?? architecture?.input_modalities;
  const model: ListedModel = { id: object.id };
  const contextWindow = positiveInteger(object.context_window) ?? positiveInteger(object.context_length);
  const maxTokens =
    positiveInteger(object.max_output_tokens) ??
    positiveInteger(object.max_completion_tokens) ??
    positiveInteger(topProvider?.max_completion_tokens);
  if (contextWindow !== undefined) model.contextWindow = contextWindow;
  if (maxTokens !== undefined) model.maxTokens = maxTokens;
  if (Array.isArray(modalities)) model.vision = modalities.includes("image");
  if (typeof object.supports_reasoning === "boolean") model.reasoning = object.supports_reasoning;
  return model;
}

/** Parse a `GET /v1/models` body into listed models, dropping malformed entries. */
export function parseModelListing(body: unknown): ListedModel[] {
  const data = record(body)?.data;
  if (!Array.isArray(data)) throw new Error("The gateway's model listing has no data array.");
  return data.map(parseListedModel).filter((model): model is ListedModel => model !== undefined);
}

/** Convert one listed alias into a Pi chat model definition for the gateway provider. */
export function toProviderModel(listed: ListedModel, config: GatewayConfig, wireOverride?: GatewayWireApi): ProviderModelConfig {
  const api = chooseWireApi(listed.id, config.kind, wireOverride);
  const contextWindow = listed.contextWindow ?? DEFAULT_CONTEXT_WINDOW;
  const maxTokens = Math.min(listed.maxTokens ?? DEFAULT_MAX_TOKENS, contextWindow);
  return {
    type: "chat",
    id: listed.id,
    name: listed.id,
    api,
    // The Anthropic SDK appends /v1/messages itself, so its base is the gateway origin.
    baseUrl: api === "anthropic-messages" ? config.baseUrl.replace(/\/v1$/, "") : config.baseUrl,
    reasoning: listed.reasoning ?? api !== "openai-completions",
    input: listed.vision === true ? ["text", "image"] : ["text"],
    // The gateway reports the settled cost of every call; catalog prices are not needed.
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens
  };
}

export interface FetchModelsOptions {
  fetch?: typeof globalThis.fetch;
  /** Directory for the on-disk listing cache; omit to cache in memory only. */
  cacheDir?: string;
  ttlMs?: number;
  now?: () => number;
}

interface CachedListing {
  fetchedAt: number;
  models: ListedModel[];
}

const memoryCache = new Map<string, CachedListing>();

/**
 * The public part of an Experiential key (`xpl_` plus 7 characters, the prefix the API keys page
 * shows): enough to keep two organizations' listings apart, and never derived from the secret.
 */
const PUBLIC_KEY_PREFIX_LENGTH = 11;

/**
 * A cache key naming the endpoint and the key's public prefix. Nothing secret goes into it, not
 * even through a hash. A key without the hosted prefix (a local gateway's virtual key) shares the
 * endpoint's entry: a local gateway serves one listing.
 */
export function cacheKey(config: GatewayConfig): string {
  const endpoint = createHash("sha256").update(config.baseUrl).digest("hex").slice(0, 16);
  const prefix = config.apiKey.startsWith("xpl_") ? config.apiKey.slice(0, PUBLIC_KEY_PREFIX_LENGTH).replace(/[^A-Za-z0-9_]/g, "") : "";
  return prefix === "" ? endpoint : `${endpoint}-${prefix}`;
}

async function readDiskCache(path: string): Promise<CachedListing | undefined> {
  try {
    const parsed = JSON.parse(await readFile(path, "utf8")) as Partial<CachedListing>;
    if (typeof parsed.fetchedAt !== "number" || !Array.isArray(parsed.models)) return undefined;
    return { fetchedAt: parsed.fetchedAt, models: parsed.models.map(parseListedModel).filter((m): m is ListedModel => m !== undefined) };
  } catch {
    return undefined;
  }
}

async function writeDiskCache(dir: string, path: string, listing: CachedListing): Promise<void> {
  try {
    await mkdir(dir, { recursive: true, mode: 0o700 });
    const temp = `${path}.${process.pid}.tmp`;
    // Stored in the parsed shape so a later read goes through the same validation.
    const models = listing.models.map((model) => ({
      id: model.id,
      context_window: model.contextWindow,
      max_output_tokens: model.maxTokens,
      input_modalities: model.vision === undefined ? undefined : model.vision ? ["text", "image"] : ["text"],
      supports_reasoning: model.reasoning
    }));
    await writeFile(temp, JSON.stringify({ fetchedAt: listing.fetchedAt, models }), { mode: 0o600 });
    await rename(temp, path);
  } catch {
    // The cache is an optimization; a read-only home directory must not break startup.
  }
}

/** Describe a failed listing response in words a user can act on. */
async function listingError(response: Response, config: GatewayConfig): Promise<Error> {
  let detail = "";
  try {
    const body = (await response.json()) as { error?: { message?: string } };
    detail = body.error?.message ?? "";
  } catch {
    // Keep the status alone.
  }
  const hint =
    response.status === 401
      ? config.kind === "hosted"
        ? " Check EXP_API_KEY, or run `exp login` and launch with `exp code`."
        : " Check EXP_GATEWAY_KEY against the key your local gateway issued."
      : "";
  return new Error(`The gateway at ${config.baseUrl} refused the model listing (${response.status})${detail ? `: ${detail.replace(/\.$/, "")}.` : "."}${hint}`);
}

/** List the gateway's callable models, from a fresh cache entry when one exists. */
export async function fetchGatewayModels(config: GatewayConfig, options: FetchModelsOptions = {}): Promise<ListedModel[]> {
  const now = options.now ?? Date.now;
  const ttl = options.ttlMs ?? MODEL_CACHE_TTL_MS;
  const key = cacheKey(config);
  const fresh = (entry: CachedListing | undefined) => entry !== undefined && now() - entry.fetchedAt < ttl;
  const memory = memoryCache.get(key);
  if (fresh(memory)) return memory!.models;
  const diskPath = options.cacheDir === undefined ? undefined : join(options.cacheDir, `models-${key}.json`);
  if (diskPath !== undefined) {
    const disk = await readDiskCache(diskPath);
    if (fresh(disk)) {
      memoryCache.set(key, disk!);
      return disk!.models;
    }
  }
  const response = await (options.fetch ?? globalThis.fetch)(`${config.baseUrl}/models`, {
    headers: { authorization: `Bearer ${config.apiKey}`, accept: "application/json" },
    signal: AbortSignal.timeout(LISTING_TIMEOUT_MS)
  });
  if (!response.ok) throw await listingError(response, config);
  const listing: CachedListing = { fetchedAt: now(), models: parseModelListing(await response.json()) };
  memoryCache.set(key, listing);
  if (diskPath !== undefined && options.cacheDir !== undefined) await writeDiskCache(options.cacheDir, diskPath, listing);
  return listing.models;
}

/** Forget in-memory listings (tests and explicit refreshes). */
export function clearModelCache(): void {
  memoryCache.clear();
}
