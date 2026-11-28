/**
 * The gateway's model catalog, as Pi model definitions.
 *
 * `GET /v1/models` lists the aliases the key may call. The hosted and local gateways return OpenAI
 * model objects (`id`, `object`, `created`, `owned_by`) extended with the catalog's facts:
 * `context_window_tokens`, `maximum_output_tokens`, `supports_reasoning`, `supports_image_input`,
 * and on the hosted gateway `supported_reasoning_efforts`. Each is read when the listing carries
 * it, with conservative defaults otherwise. Listings are cached briefly in memory and, when a
 * cache directory is given, on disk, so startup does not wait on the network every time.
 */

import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { ThinkingLevelMap } from "@earendil-works/pi-ai";
import type { ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import type { GatewayConfig, GatewayKind } from "./config.js";

/** One callable alias and whatever limits its listing states. */
export interface ListedModel {
  id: string;
  /** The whole window: prompt plus the model's maximum output (`context_window_tokens`). */
  contextWindow?: number;
  /** The model's maximum output (`maximum_output_tokens`). */
  maxTokens?: number;
  vision?: boolean;
  reasoning?: boolean;
  /** The reasoning efforts the route accepts (`supported_reasoning_efforts`); absent when unstated. */
  reasoningEfforts?: string[];
}

/** Wire APIs the gateway serves that Pi can speak. */
export type GatewayWireApi = "openai-completions" | "openai-responses" | "anthropic-messages";

export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 32_000;
/**
 * The smallest context window exp-code offers a model with. Its system prompt and tool
 * definitions take several thousand tokens before the conversation starts, so on a smaller window
 * (a 4K model) the output allowance left was one token: every turn ended "length" with nothing to
 * show, and was still billed.
 */
export const MIN_AGENT_CONTEXT_WINDOW = 16_384;
/** How long a listing is reused before the gateway is asked again. */
export const MODEL_CACHE_TTL_MS = 5 * 60_000;
const LISTING_TIMEOUT_MS = 10_000;
/**
 * Bumped when the cached shape or its selection changes, so a listing an older release cached
 * (without the limits this one reads, or with aliases it can no longer run) is fetched again.
 */
const LISTING_CACHE_VERSION = "v4";

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
  const model: ListedModel = { id: object.id };
  const contextWindow = positiveInteger(object.context_window_tokens);
  const maxTokens = positiveInteger(object.maximum_output_tokens);
  if (contextWindow !== undefined) model.contextWindow = contextWindow;
  if (maxTokens !== undefined) model.maxTokens = maxTokens;
  if (typeof object.supports_image_input === "boolean") model.vision = object.supports_image_input;
  if (typeof object.supports_reasoning === "boolean") model.reasoning = object.supports_reasoning;
  const efforts = object.supported_reasoning_efforts;
  if (Array.isArray(efforts) && efforts.length > 0 && efforts.every((effort) => typeof effort === "string")) model.reasoningEfforts = efforts;
  return model;
}

/** Pi's thinking levels in order; `off` is the gateway's `none` effort. */
const PI_THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"] as const;

/**
 * The Pi thinking-level map for a route's supported efforts: each level the route accepts maps to
 * its gateway effort and every other level to null, which Pi reads as unsupported, so `/thinking`
 * and Shift+Tab offer only what the model takes. Efforts Pi has no level for (`ultra`) are left
 * out. Undefined when the listing states no efforts (a local gateway): Pi's defaults apply.
 *
 * `off` exists only when the route accepts `none`: the gateway refuses an effort a route does not
 * list (`unsupported_parameter`), and a route without `none` reasons whatever is sent, so a model
 * that always reasons never shows "thinking off". Pi then starts it at medium clamped to this map
 * (pi-patches.ts keeps a switch from carrying another model's level).
 */
export function thinkingLevelMap(efforts: string[] | undefined): ThinkingLevelMap | undefined {
  if (efforts === undefined) return undefined;
  const map: ThinkingLevelMap = {};
  for (const level of PI_THINKING_LEVELS) {
    const effort = level === "off" ? "none" : level;
    map[level] = efforts.includes(effort) ? effort : null;
  }
  return map;
}

/**
 * Whether exp-code can run on a listed alias. Every turn is a chat request that carries the agent's
 * tools, so an alias the listing states does not serve chat (`supports_completions: false`: speech,
 * transcription, embeddings, image and video models) or does not take tools (`supports_tools:
 * false`) can only fail: the gateway refuses it with a 400 before any call. Nor can one whose
 * context window is below MIN_AGENT_CONTEXT_WINDOW hold the agent's prompt. An unstated capability
 * or window (a local gateway's listing) keeps the alias.
 */
export function agentCanUse(entry: unknown): boolean {
  const object = record(entry);
  const window = positiveInteger(object?.context_window_tokens);
  return object?.supports_completions !== false && object?.supports_tools !== false && (window === undefined || window >= MIN_AGENT_CONTEXT_WINDOW);
}

/** Parse a `GET /v1/models` body into the aliases exp-code can run, dropping malformed entries. */
export function parseModelListing(body: unknown): ListedModel[] {
  const data = record(body)?.data;
  if (!Array.isArray(data)) throw new Error("The gateway's model listing has no data array.");
  return data
    .filter(agentCanUse)
    .map(parseListedModel)
    .filter((model): model is ListedModel => model !== undefined);
}

/** Convert one listed alias into a Pi chat model definition for the gateway provider. */
export function toProviderModel(listed: ListedModel, config: GatewayConfig, wireOverride?: GatewayWireApi): ProviderModelConfig {
  const api = chooseWireApi(listed.id, config.kind, wireOverride);
  const contextWindow = listed.contextWindow ?? DEFAULT_CONTEXT_WINDOW;
  // The output reservation and the prompt share one window: a small-window model (8K) asked for
  // its whole window as output can never fit the agent's prompt. Reserve at most half the window,
  // whatever the listing allows.
  const maxTokens = Math.max(1, Math.min(listed.maxTokens ?? DEFAULT_MAX_TOKENS, Math.floor(contextWindow / 2)));
  const levels = thinkingLevelMap(listed.reasoningEfforts);
  // A route whose only effort is `none` cannot think, whatever `supports_reasoning` says.
  const thinks = levels === undefined ? (listed.reasoning ?? api !== "openai-completions") : listed.reasoning !== false && Object.entries(levels).some(([level, effort]) => level !== "off" && effort !== null);
  // A Claude route that states an effort ladder takes adaptive thinking (`thinking.type: adaptive`
  // plus `output_config.effort`). Pi only knows that for its own model ids, so a gateway alias would
  // otherwise get budget-based thinking (`budget_tokens`), which Claude 4.6 and later refuse: every
  // turn on claude-sonnet-5.5 or claude-haiku-5.5 failed with a 400 at the default level.
  const adaptive = api === "anthropic-messages" && thinks && levels !== undefined;
  return {
    type: "chat",
    id: listed.id,
    name: listed.id,
    api,
    // The Anthropic SDK appends /v1/messages itself, so its base is the gateway origin.
    baseUrl: api === "anthropic-messages" ? config.baseUrl.replace(/\/v1$/, "") : config.baseUrl,
    reasoning: thinks,
    ...(thinks && levels !== undefined ? { thinkingLevelMap: levels } : {}),
    ...(adaptive ? { compat: { forceAdaptiveThinking: true } } : {}),
    input: listed.vision === true ? ["text", "image"] : ["text"],
    // The gateway reports the settled cost of every call; catalog prices are not needed.
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    // The whole window, as the gateway states it: the footer shows it (1.05M for a 1,050,000-token
    // model). It is NOT the prompt budget: the provider counts the model's whole output reservation
    // inside the window, so a prompt may only grow to `contextWindow - maxTokens`. Auto-compaction
    // reserves `maxTokens` on top of its own headroom (see the settings-manager edit in
    // pi-patches.ts), so it runs before the prompt reaches that budget.
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
  /** Called when a failed listing was answered from an older cached one: why, and the cache's age. */
  onStaleListing?: (error: Error, ageMs: number) => void;
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
      context_window_tokens: model.contextWindow,
      maximum_output_tokens: model.maxTokens,
      supports_image_input: model.vision,
      supports_reasoning: model.reasoning,
      supported_reasoning_efforts: model.reasoningEfforts
    }));
    await writeFile(temp, JSON.stringify({ fetchedAt: listing.fetchedAt, models }), { mode: 0o600 });
    await rename(temp, path);
  } catch {
    // The cache is an optimization; a read-only home directory must not break startup.
  }
}

/**
 * A listing that failed. `transient` marks failures a cached listing may stand in for (the gateway
 * unreachable or slow, a 408/429/5xx, a body that is not a listing); a refused key is never one.
 */
export class ListingError extends Error {
  constructor(
    message: string,
    readonly transient: boolean
  ) {
    super(message);
    this.name = "ListingError";
  }
}

/** Describe a failed listing response in words a user can act on. */
async function listingError(response: Response, config: GatewayConfig): Promise<ListingError> {
  let detail = "";
  try {
    const body = (await response.json()) as { error?: { message?: string } };
    detail = typeof body.error?.message === "string" ? body.error.message : "";
  } catch {
    // Keep the status alone.
  }
  const hint =
    response.status === 401 || response.status === 403
      ? config.kind === "hosted"
        ? " Check EXP_API_KEY, or run `exp-code login` to save a new key."
        : " Check EXP_GATEWAY_KEY against the key your local gateway issued."
      : "";
  const transient = response.status === 408 || response.status === 429 || response.status >= 500;
  return new ListingError(
    `The gateway at ${config.baseUrl} refused the model listing (${response.status})${detail ? `: ${detail.replace(/\.$/, "")}.` : "."}${hint}`,
    transient
  );
}

/** Why the listing request never got an answer, in words (the gateway down, DNS, TLS, a timeout). */
function unreachable(error: unknown, config: GatewayConfig): ListingError {
  const failure = error as { name?: string; message?: string; cause?: { code?: string; message?: string } };
  const reason =
    failure.name === "TimeoutError"
      ? `no answer within ${LISTING_TIMEOUT_MS / 1000} s`
      : (failure.cause?.code ?? failure.cause?.message ?? failure.message ?? String(error));
  return new ListingError(`Could not reach the gateway at ${config.baseUrl} to list models (${reason}).`, true);
}

/** How long a failed listing is remembered, so one start neither asks nor reports twice. */
const FAILURE_MEMO_MS = 15_000;
const failures = new Map<string, { at: number; error: ListingError }>();
const inFlight = new Map<string, Promise<CachedListing>>();

async function requestListing(config: GatewayConfig, options: FetchModelsOptions, now: () => number): Promise<CachedListing> {
  let response: Response;
  try {
    response = await (options.fetch ?? globalThis.fetch)(`${config.baseUrl}/models`, {
      headers: { authorization: `Bearer ${config.apiKey}`, accept: "application/json" },
      signal: AbortSignal.timeout(LISTING_TIMEOUT_MS)
    });
  } catch (error) {
    throw unreachable(error, config);
  }
  if (!response.ok) throw await listingError(response, config);
  let body: unknown;
  try {
    body = await response.json();
  } catch (error) {
    if ((error as { name?: string }).name === "TimeoutError") throw unreachable(error, config);
    const type = response.headers.get("content-type") ?? "no content type";
    throw new ListingError(`The gateway at ${config.baseUrl} answered the model listing with something that is not JSON (HTTP ${response.status}, ${type}).`, true);
  }
  try {
    return { fetchedAt: now(), models: parseModelListing(body) };
  } catch (error) {
    throw new ListingError(`The gateway at ${config.baseUrl} answered the model listing without a model list: ${(error as Error).message}`, true);
  }
}

/**
 * List the gateway's callable models, from a fresh cache entry when one exists.
 *
 * When the gateway cannot be asked (down, slow, a 5xx) and an older listing is cached, the older
 * listing is returned and `onStaleListing` is told why, so an outage of the listing alone does not
 * take every model away. A refused key always fails. A failure is remembered briefly, so the
 * several callers of one start share one request and one error.
 */
export async function fetchGatewayModels(config: GatewayConfig, options: FetchModelsOptions = {}): Promise<ListedModel[]> {
  const now = options.now ?? Date.now;
  const ttl = options.ttlMs ?? MODEL_CACHE_TTL_MS;
  const key = cacheKey(config);
  const fresh = (entry: CachedListing | undefined) => entry !== undefined && now() - entry.fetchedAt < ttl;
  const memory = memoryCache.get(key);
  if (fresh(memory)) return memory!.models;
  const diskPath = options.cacheDir === undefined ? undefined : join(options.cacheDir, `models-${LISTING_CACHE_VERSION}-${key}.json`);
  const disk = diskPath === undefined ? undefined : await readDiskCache(diskPath);
  if (fresh(disk)) {
    memoryCache.set(key, disk!);
    return disk!.models;
  }
  const fallback = (error: ListingError): ListedModel[] => {
    const stale = memory ?? disk;
    if (!error.transient || stale === undefined) throw error;
    options.onStaleListing?.(error, Math.max(0, now() - stale.fetchedAt));
    return stale.models;
  };
  const failed = failures.get(key);
  if (failed !== undefined && now() - failed.at < FAILURE_MEMO_MS) return fallback(failed.error);
  let pending = inFlight.get(key);
  if (pending === undefined) {
    pending = requestListing(config, options, now).finally(() => inFlight.delete(key));
    inFlight.set(key, pending);
  }
  let listing: CachedListing;
  try {
    listing = await pending;
  } catch (error) {
    const failure = error instanceof ListingError ? error : new ListingError((error as Error).message, true);
    failures.set(key, { at: now(), error: failure });
    return fallback(failure);
  }
  failures.delete(key);
  memoryCache.set(key, listing);
  if (diskPath !== undefined && options.cacheDir !== undefined) await writeDiskCache(options.cacheDir, diskPath, listing);
  return listing.models;
}

/** Forget in-memory listings (tests and explicit refreshes). */
export function clearModelCache(): void {
  memoryCache.clear();
  failures.clear();
  inFlight.clear();
}
