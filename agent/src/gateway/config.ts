/**
 * Where exp-code sends model calls: the hosted Experiential gateway or a local `exp` gateway.
 *
 * The selection is read from the environment first, so the same binary works when launched by
 * `exp code`, from a shell, or inside a sandbox; without a key there, the key `exp-code login`
 * saved is used. Keys never travel through command-line arguments.
 */

/** Provider id under which the gateway's models are registered in Pi. */
export const GATEWAY_PROVIDER_ID = "experiential";

/** The hosted gateway's OpenAI-compatible root. */
export const HOSTED_GATEWAY_BASE_URL = "https://api.experientiallabs.ai/v1";

/** The default local `exp` gateway's OpenAI-compatible root. */
export const LOCAL_GATEWAY_BASE_URL = "http://127.0.0.1:8000/v1";

export type GatewayKind = "hosted" | "local";

/** A resolved gateway endpoint with the key that authorizes calls to it. */
export interface GatewayConfig {
  kind: GatewayKind;
  /** OpenAI-compatible root, without a trailing slash (normally ending in `/v1`). */
  baseUrl: string;
  /** `xpl_` key for the hosted gateway, or a virtual key issued by the local gateway. */
  apiKey: string;
  /** Headers sent with every call (a hosted run's conversation trace, see trace.ts). */
  headers?: Readonly<Record<string, string>>;
}

/** Where the key came from: an environment variable or the key `exp-code login` saved. */
export type GatewayKeySource = "env" | "saved";

/** The outcome of resolving the gateway: a config and its key's source, or the reason there is none. */
export type GatewayResolution =
  | { config: GatewayConfig; source: GatewayKeySource; problem?: undefined; missingKey?: undefined }
  | { config: null; problem: string; source?: undefined; /** True when the only problem is that no key exists. */ missingKey: boolean };

/** The saved login, as resolution needs it (see auth/credentials.ts). */
export interface SavedGatewayKey {
  hostedKey: string;
  baseUrl?: string;
}

/** Environment variables that select and authorize the gateway. */
export const GATEWAY_ENV = {
  /** `hosted` or `local`; when unset the available key decides. */
  kind: "EXP_CODE_GATEWAY",
  /** Overrides the base URL of the selected gateway. */
  baseUrl: "EXP_CODE_BASE_URL",
  /** Hosted gateway key (`xpl_...`). */
  hostedKey: "EXP_API_KEY",
  /** Local gateway virtual key. */
  localKey: "EXP_GATEWAY_KEY"
} as const;

const MISSING_KEY_HELP =
  "No Experiential gateway key found. Run `exp-code login` to sign in through your browser, " +
  "or set EXP_API_KEY to an xpl_ key for the hosted gateway (EXP_GATEWAY_KEY for a local `exp` gateway).";

/** Drop trailing slashes in linear time (a regex like `/\/+$/` backtracks on long slash runs). */
export function trimTrailingSlashes(value: string): string {
  let end = value.length;
  while (end > 0 && value.charCodeAt(end - 1) === 47) end -= 1;
  return value.slice(0, end);
}

/** Remove trailing slashes so path joins stay predictable. */
export function trimBaseUrl(url: string): string {
  return trimTrailingSlashes(url.trim());
}

/**
 * Resolve the gateway from environment variables, then the saved login.
 *
 * An explicit `EXP_CODE_GATEWAY` wins. Otherwise a hosted key in the environment selects the
 * hosted gateway, a local key the local one, and with neither the key `exp-code login` saved
 * selects the hosted gateway. With none of them there is no gateway and the reason says how to
 * provide one. The environment always wins over the saved login.
 */
export function resolveGateway(env: NodeJS.ProcessEnv, saved: SavedGatewayKey | null = null): GatewayResolution {
  const envHostedKey = env[GATEWAY_ENV.hostedKey]?.trim() ?? "";
  const localKey = env[GATEWAY_ENV.localKey]?.trim() ?? "";
  const savedKey = saved?.hostedKey.trim() ?? "";
  const requested = env[GATEWAY_ENV.kind]?.trim().toLowerCase() ?? "";
  if (requested !== "" && requested !== "hosted" && requested !== "local") {
    return { config: null, problem: `${GATEWAY_ENV.kind} must be "hosted" or "local", not "${requested}".`, missingKey: false };
  }
  let kind: GatewayKind;
  if (requested === "hosted" || requested === "local") kind = requested;
  else if (envHostedKey !== "") kind = "hosted";
  else if (localKey !== "") kind = "local";
  else if (savedKey !== "") kind = "hosted";
  else return { config: null, problem: MISSING_KEY_HELP, missingKey: true };

  const source: GatewayKeySource = kind === "hosted" && envHostedKey === "" ? "saved" : "env";
  const apiKey = kind === "hosted" ? envHostedKey || savedKey : localKey;
  if (apiKey === "") {
    const variable = kind === "hosted" ? GATEWAY_ENV.hostedKey : GATEWAY_ENV.localKey;
    return { config: null, problem: `${GATEWAY_ENV.kind}=${kind} needs ${variable}. ${MISSING_KEY_HELP}`, missingKey: true };
  }
  const override = env[GATEWAY_ENV.baseUrl]?.trim() ?? "";
  const savedBase = source === "saved" ? (saved?.baseUrl?.trim() ?? "") : "";
  const fallback = kind === "hosted" ? HOSTED_GATEWAY_BASE_URL : LOCAL_GATEWAY_BASE_URL;
  const baseUrl = trimBaseUrl(override !== "" ? override : savedBase !== "" ? savedBase : fallback);
  return { config: { kind, baseUrl, apiKey }, source };
}
