/**
 * The Experiential gateway as a Pi provider.
 *
 * Registration uses Pi's provider configuration form with one model per gateway alias, each on
 * the wire API `chooseWireApi` picks, and a `refreshModels` callback so `/model` refreshes see new
 * aliases. The gateway reports each call's settled cost as `usage.cost` in its final stream chunk;
 * Pi only trusts provider-reported cost for providers it knows, so this module records that number
 * from the provider stream and writes it onto the finished assistant message.
 */

import type { ExtensionAPI, ProviderConfig, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { GATEWAY_PROVIDER_ID, type GatewayConfig } from "./config.js";
import { type FetchModelsOptions, fetchGatewayModels, type GatewayWireApi, type ListedModel, toProviderModel } from "./models.js";

export interface GatewayProviderOptions extends FetchModelsOptions {
  /** Force one wire API for every alias. */
  wire?: GatewayWireApi;
  /** Register exactly these aliases instead of listing them (the headless runner knows its model). */
  models?: ListedModel[];
  /** Called with the reason when the listing fails; the provider is then registered without models. */
  onListingError?: (error: Error) => void;
}

/** Escape a literal key so Pi never reads it as `$ENV` or `!command` syntax. */
export function literalApiKey(key: string): string {
  const escaped = key.replaceAll("$", () => "$$");
  return escaped.startsWith("!") ? `$${escaped}` : escaped;
}

/** Read a provider-reported cost off a parsed stream event, when the event carries one. */
export function reportedCost(data: unknown): number | undefined {
  return reportedBilling(data)?.cost;
}

/**
 * What the gateway reported about a response's billing: `usage.cost` (the credits it charged),
 * and for a call served on the caller's own provider key (bring your own key) `usage.is_byok`
 * with that provider's list price as `usage.cost_details.upstream_inference_cost`.
 */
export interface ReportedBilling {
  cost: number;
  byok: boolean;
  byokListUsd: number | null;
}

export function reportedBilling(data: unknown): ReportedBilling | undefined {
  if (typeof data !== "object" || data === null) return undefined;
  const usage = (data as { usage?: unknown; response?: { usage?: unknown } }).usage ?? (data as { response?: { usage?: unknown } }).response?.usage;
  if (typeof usage !== "object" || usage === null) return undefined;
  const record = usage as { cost?: unknown; is_byok?: unknown; cost_details?: { upstream_inference_cost?: unknown } };
  const cost = record.cost;
  if (typeof cost !== "number" || !Number.isFinite(cost) || cost < 0) return undefined;
  const upstream = record.cost_details?.upstream_inference_cost;
  return {
    cost,
    byok: record.is_byok === true,
    byokListUsd: typeof upstream === "number" && Number.isFinite(upstream) && upstream >= 0 ? upstream : null
  };
}

/** The billing the gateway reported for an assistant message (see registerGatewayProvider), if any. */
export function messageBilling(message: unknown): ReportedBilling | undefined {
  return (message as { gatewayBilling?: ReportedBilling }).gatewayBilling;
}

/** Build the provider models for a listing. */
export function gatewayProviderModels(listed: ListedModel[], config: GatewayConfig, wire?: GatewayWireApi): ProviderModelConfig[] {
  return listed.map((model) => toProviderModel(model, config, wire));
}

/** The Pi provider configuration for the gateway with a given model list. */
export function gatewayProviderConfig(
  config: GatewayConfig,
  models: ProviderModelConfig[],
  refreshModels?: () => Promise<ProviderModelConfig[]>
): ProviderConfig {
  return {
    name: config.kind === "hosted" ? "Experiential" : "Experiential (local gateway)",
    baseUrl: config.baseUrl,
    apiKey: literalApiKey(config.apiKey),
    api: "openai-completions",
    models,
    refreshModels
  };
}

/**
 * Register the gateway provider on an extension API and attach settled-cost capture.
 *
 * Resolves once the initial model list is known, so Pi can select a gateway model at startup.
 */
export async function registerGatewayProvider(pi: ExtensionAPI, config: GatewayConfig, options: GatewayProviderOptions = {}): Promise<ProviderModelConfig[]> {
  let listed: ListedModel[] = options.models ?? [];
  if (options.models === undefined) {
    try {
      listed = await fetchGatewayModels(config, options);
    } catch (error) {
      options.onListingError?.(error instanceof Error ? error : new Error(String(error)));
    }
  }
  const models = gatewayProviderModels(listed, config, options.wire);
  pi.registerProvider(
    GATEWAY_PROVIDER_ID,
    gatewayProviderConfig(
      config,
      models,
      options.models === undefined ? async () => gatewayProviderModels(await fetchGatewayModels(config, options), config, options.wire) : undefined
    )
  );

  // Settled billing: the last reported value of the current response, applied at message_end.
  // The cost becomes the message's total; the whole report rides along as `gatewayBilling`.
  let pending: ReportedBilling | undefined;
  pi.on("message_start", (event) => {
    if (event.message.role === "assistant") pending = undefined;
  });
  pi.on("provider_stream_event", (event) => {
    if (event.provider !== GATEWAY_PROVIDER_ID) return;
    const billing = reportedBilling(event.data);
    if (billing !== undefined) pending = billing;
  });
  pi.on("message_end", (event) => {
    const message = event.message;
    if (message.role !== "assistant" || message.provider !== GATEWAY_PROVIDER_ID || pending === undefined) return;
    const billing = pending;
    pending = undefined;
    return {
      message: { ...message, usage: { ...message.usage, cost: { ...message.usage.cost, total: billing.cost, totalFromProvider: true } }, gatewayBilling: billing }
    };
  });
  return models;
}
