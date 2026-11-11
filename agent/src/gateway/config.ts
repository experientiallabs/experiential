/**
 * Where exp-code sends model calls: the hosted Experiential gateway or a local `exp` gateway.
 *
 * The selection is read from the environment only, so the same binary works when launched by
 * `exp code`, from a shell, or inside a sandbox. Keys never travel through command-line arguments.
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
}

/** The outcome of reading the environment: a config, or the reason there is none. */
export type GatewayResolution = { config: GatewayConfig; problem?: undefined } | { config: null; problem: string };

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
  "No Experiential gateway key found. Set EXP_API_KEY to an xpl_ key for the hosted gateway, " +
  "set EXP_GATEWAY_KEY for a local `exp` gateway, or launch through `exp code`, which uses your `exp login` credential.";

/** Remove trailing slashes so path joins stay predictable. */
export function trimBaseUrl(url: string): string {
  return url.trim().replace(/\/+$/, "");
}

/**
 * Resolve the gateway from environment variables.
 *
 * An explicit `EXP_CODE_GATEWAY` wins. Otherwise a hosted key selects the hosted gateway and a
 * local key selects the local one; with neither, there is no gateway and the reason says how to
 * provide one.
 */
export function resolveGateway(env: NodeJS.ProcessEnv): GatewayResolution {
  const hostedKey = env[GATEWAY_ENV.hostedKey]?.trim() ?? "";
  const localKey = env[GATEWAY_ENV.localKey]?.trim() ?? "";
  const requested = env[GATEWAY_ENV.kind]?.trim().toLowerCase() ?? "";
  if (requested !== "" && requested !== "hosted" && requested !== "local") {
    return { config: null, problem: `${GATEWAY_ENV.kind} must be "hosted" or "local", not "${requested}".` };
  }
  let kind: GatewayKind;
  if (requested === "hosted" || requested === "local") kind = requested;
  else if (hostedKey !== "") kind = "hosted";
  else if (localKey !== "") kind = "local";
  else return { config: null, problem: MISSING_KEY_HELP };

  const apiKey = kind === "hosted" ? hostedKey : localKey;
  if (apiKey === "") {
    const variable = kind === "hosted" ? GATEWAY_ENV.hostedKey : GATEWAY_ENV.localKey;
    return { config: null, problem: `${GATEWAY_ENV.kind}=${kind} needs ${variable}. ${MISSING_KEY_HELP}` };
  }
  const override = env[GATEWAY_ENV.baseUrl]?.trim() ?? "";
  const baseUrl = trimBaseUrl(override !== "" ? override : kind === "hosted" ? HOSTED_GATEWAY_BASE_URL : LOCAL_GATEWAY_BASE_URL);
  return { config: { kind, baseUrl, apiKey } };
}
