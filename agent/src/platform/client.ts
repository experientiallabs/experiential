/**
 * A small read-only client for the Experiential platform API, used by the platform tools.
 *
 * The platform API lives on the same origin as the hosted gateway (`/api/...` beside `/v1`) and
 * accepts the same organization key (`Authorization: Bearer xpl_...`). The client only issues
 * GETs, resolves the key's organization once per session (`GET /api/whoami`), and turns every
 * refusal into a message the model can act on. The key never appears in a message.
 */

import { type GatewayConfig, trimTrailingSlashes } from "../gateway/config.js";

/** How long one platform call may take before it is abandoned. */
export const PLATFORM_TIMEOUT_MS = 30_000;

/** The organization the key acts for (`GET /api/whoami`). */
export interface Whoami {
  org_id: string;
  org_slug: string;
  org_name: string;
  identity_id: string | null;
  customer_plan_identity_id: string | null;
  is_provisioning: boolean;
}

/** A refused or failed platform call, with a message safe to show the model. */
export class PlatformError extends Error {
  constructor(
    message: string,
    /** HTTP status, or 0 when no response arrived. */
    readonly status: number,
    /** The server's stable reason code, when it sent one. */
    readonly code?: string
  ) {
    super(message);
    this.name = "PlatformError";
  }
}

export type QueryValue = string | number | boolean | undefined | null;

export interface PlatformClientOptions {
  /** The platform origin, without a trailing slash (`https://api.experientiallabs.ai`). */
  origin: string;
  apiKey: string;
  fetch?: typeof globalThis.fetch;
  timeoutMs?: number;
}

/**
 * The platform origin for a hosted gateway: its base URL without the trailing `/v1`. Undefined
 * for a local `exp` gateway (it serves no platform API) or a base URL that does not end in `/v1`.
 */
export function platformOrigin(gateway: GatewayConfig): string | undefined {
  if (gateway.kind !== "hosted") return undefined;
  const base = trimTrailingSlashes(gateway.baseUrl);
  if (!base.endsWith("/v1")) return undefined;
  const origin = base.slice(0, -3);
  return /^https?:\/\/[^/]+/.test(origin) ? origin : undefined;
}

/** Replace anything shaped like an Experiential key, so a server message can never echo one. */
export function redactKeys(text: string): string {
  return text.replace(/\bxpl[a-z]*_[A-Za-z0-9_-]+/g, "[redacted key]");
}

/** The server's error text from a JSON or plain body: `{"error": "..."}`, `{"error": {"message"}}` or `{"detail"}`. */
function serverMessage(raw: string): { message: string; code?: string } {
  try {
    const body = JSON.parse(raw) as { error?: unknown; detail?: unknown; code?: unknown; message?: unknown };
    const code = typeof body.code === "string" ? body.code : undefined;
    if (typeof body.error === "string") return { message: body.error, code };
    const nested = body.error as { message?: unknown; code?: unknown } | undefined;
    if (nested && typeof nested.message === "string") return { message: nested.message, code: typeof nested.code === "string" ? nested.code : code };
    if (typeof body.detail === "string") return { message: body.detail, code };
    if (Array.isArray(body.detail)) {
      const parts = body.detail.map((entry) => (entry as { msg?: unknown }).msg).filter((msg): msg is string => typeof msg === "string");
      if (parts.length > 0) return { message: parts.join("; "), code };
    }
    if (typeof body.message === "string") return { message: body.message, code };
  } catch {
    // Not JSON; use the text.
  }
  return { message: raw.trim().slice(0, 300) };
}

/** Turn an HTTP refusal into a message that says what happened and what to do. */
export function describeRefusal(status: number, raw: string, retryAfter: string | null): PlatformError {
  const { message, code } = serverMessage(raw);
  const said = redactKeys(message).slice(0, 500);
  const detail = said === "" ? "" : `: ${said}`;
  let text: string;
  if (status === 401) {
    text = "The Experiential API did not accept this key (401). Run `exp-code login` again, or check EXP_API_KEY.";
  } else if (status === 403) {
    text = `This key is not allowed to read that (403)${detail}. Some reads (per-key usage, evaluations) need the organization's provisioning key.`;
  } else if (status === 404) {
    text = `Not found (404)${detail}`;
  } else if (status === 429) {
    const wait = retryAfter !== null && /^\d+$/.test(retryAfter) ? ` Retry in about ${retryAfter} s.` : " Retry shortly.";
    text = `The Experiential API is busy or rate limited this key (429)${detail}.${wait}`;
  } else if (status >= 500) {
    text = `The Experiential API had a problem (${status})${detail}. Try again shortly.`;
  } else {
    text = `The Experiential API refused the request (${status})${detail}`;
  }
  return new PlatformError(text, status, code);
}

/** Build `path?query` from defined query values. */
export function withQuery(path: string, query: Record<string, QueryValue> = {}): string {
  const params = new URLSearchParams();
  for (const [name, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === "") continue;
    params.set(name, String(value));
  }
  const text = params.toString();
  return text === "" ? path : `${path}?${text}`;
}

export class PlatformClient {
  readonly origin: string;
  private readonly apiKey: string;
  private readonly fetchImpl: typeof globalThis.fetch;
  private readonly timeoutMs: number;
  private whoamiPromise: Promise<Whoami> | undefined;

  constructor(options: PlatformClientOptions) {
    this.origin = trimTrailingSlashes(options.origin);
    this.apiKey = options.apiKey;
    this.fetchImpl = options.fetch ?? globalThis.fetch;
    this.timeoutMs = options.timeoutMs ?? PLATFORM_TIMEOUT_MS;
  }

  /** GET a JSON document. Throws PlatformError on any refusal, timeout, or unreadable body. */
  async get<T>(path: string, query: Record<string, QueryValue> = {}, signal?: AbortSignal): Promise<T> {
    const timeout = AbortSignal.timeout(this.timeoutMs);
    const combined = signal === undefined ? timeout : AbortSignal.any([signal, timeout]);
    let response: Response;
    try {
      response = await this.fetchImpl(`${this.origin}${withQuery(path, query)}`, {
        method: "GET",
        signal: combined,
        headers: { accept: "application/json", authorization: `Bearer ${this.apiKey}` }
      });
    } catch (error) {
      if (signal?.aborted === true) throw new PlatformError("The request was cancelled.", 0);
      if (timeout.aborted) throw new PlatformError(`The Experiential API did not answer within ${Math.round(this.timeoutMs / 1000)} s. Try again shortly.`, 0);
      throw new PlatformError(`Could not reach the Experiential API at ${this.origin}: ${redactKeys((error as Error).message)}`, 0);
    }
    const raw = await response.text();
    if (!response.ok) throw describeRefusal(response.status, raw, response.headers.get("retry-after"));
    try {
      return JSON.parse(raw) as T;
    } catch {
      throw new PlatformError(`The Experiential API sent an unreadable answer (${response.status}).`, response.status);
    }
  }

  /** The key's organization, read once per session; a failed read is retried on the next call. */
  whoami(signal?: AbortSignal): Promise<Whoami> {
    if (this.whoamiPromise === undefined) {
      const pending = this.get<Whoami>("/api/whoami", {}, signal).then((body) => {
        if (typeof body?.org_id !== "string" || body.org_id === "") throw new PlatformError("The Experiential API did not say which organization this key belongs to.", 200);
        return body;
      });
      this.whoamiPromise = pending;
      pending.catch(() => {
        if (this.whoamiPromise === pending) this.whoamiPromise = undefined;
      });
    }
    return this.whoamiPromise;
  }

  /** `/api/orgs/{org}` + `suffix` for the key's organization. */
  async orgPath(suffix: string, signal?: AbortSignal): Promise<string> {
    const { org_id } = await this.whoami(signal);
    return `/api/orgs/${encodeURIComponent(org_id)}${suffix}`;
  }
}
