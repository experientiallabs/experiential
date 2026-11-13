/**
 * The loopback half of `exp-code login`: a one-time listener on 127.0.0.1 that receives the key.
 *
 * The CLI binds an ephemeral port, sends the member to the Experiential platform's approval page
 * with that port, a one-time `state` and the key name, and waits. After approval the browser is
 * sent to `http://127.0.0.1:<port>/callback?token=<key>&state=<state>`. Only that path, with the
 * matching state (compared in constant time) and an `xpl_` key, is accepted, and only once; any
 * other path is a 404 and nothing is logged, so the key never reaches a terminal or a log.
 */

import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import { createServer, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { trimTrailingSlashes } from "../gateway/config.js";

/** The platform the approval page lives on. */
export const PLATFORM_DEFAULT_URL = "https://platform.experientiallabs.ai";
/** Overrides the platform origin (staging, a preview). */
export const PLATFORM_URL_ENV = "EXP_PLATFORM_URL";
/** How long a login waits for the browser. */
export const LOGIN_TIMEOUT_MS = 300_000;
/** The platform keeps the first 80 characters of a requested key name. */
export const KEY_NAME_MAX_LENGTH = 80;

/** The platform origin, without a trailing slash. */
export function platformUrl(env: NodeJS.ProcessEnv): string {
  const configured = trimTrailingSlashes(env[PLATFORM_URL_ENV]?.trim() ?? "");
  return configured !== "" ? configured : PLATFORM_DEFAULT_URL;
}

/** The key name exp-code asks for: `exp-code (<hostname>)`, within the platform's limit. */
export function defaultKeyName(hostname: string): string {
  const host = hostname.trim();
  const name = host === "" ? "exp-code" : `exp-code (${host})`;
  if (name.length <= KEY_NAME_MAX_LENGTH) return name;
  return `${name.slice(0, KEY_NAME_MAX_LENGTH - 1)})`;
}

/** A one-time state, the same size as Python's `secrets.token_urlsafe(24)`. */
export function newState(): string {
  return randomBytes(24).toString("base64url");
}

/** The approval URL. Without a port the page shows the key for pasting instead of handing it back. */
export function authorizeUrl(platform: string, options: { state: string; port?: number; keyName: string }): string {
  const url = new URL(`${trimTrailingSlashes(platform)}/cli/auth`);
  url.searchParams.set("state", options.state);
  if (options.port !== undefined) url.searchParams.set("port", String(options.port));
  url.searchParams.set("name", options.keyName);
  return url.toString();
}

function sameState(received: string, expected: string): boolean {
  // Hash first so the comparison is constant time whatever the lengths.
  const a = createHash("sha256").update(received).digest();
  const b = createHash("sha256").update(expected).digest();
  return timingSafeEqual(a, b) && received.length === expected.length;
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] ?? c);
}

const FAILURE_PAGE = `<!doctype html>
<html><body style="font-family: system-ui; padding: 48px; color: #171717;">
<h1 style="font-size: 18px;">That did not match</h1>
<p>This callback was not for the login attempt waiting in your terminal.
Run <code>exp-code login</code> again and use the new URL.</p>
</body></html>`;

function successPage(platform: string): string {
  const href = escapeHtml(platform);
  return `<!doctype html>
<html><head><meta http-equiv="refresh" content="1;url=${href}"></head>
<body style="font-family: system-ui; padding: 48px; color: #171717;">
<h1 style="font-size: 18px;">exp-code is logged in</h1>
<p>The key was handed to your terminal. You can close this tab or return to
<a href="${href}">the Experiential platform</a>.</p>
</body></html>`;
}

function respond(response: ServerResponse, status: number, body: string): void {
  response.writeHead(status, { "content-type": "text/html; charset=utf-8", "content-length": Buffer.byteLength(body), "cache-control": "no-store" });
  response.end(body);
}

/** One login attempt's listener. */
export class LoopbackLogin {
  readonly state: string;
  private readonly platform: string;
  private server: Server | null = null;
  private key: string | null = null;
  private waiters: ((key: string | null) => void)[] = [];

  constructor(platform: string, state: string = newState()) {
    this.platform = trimTrailingSlashes(platform);
    this.state = state;
  }

  /** The bound port. */
  get port(): number {
    if (this.server === null) throw new Error("the login listener is not running");
    return (this.server.address() as AddressInfo).port;
  }

  /** Bind 127.0.0.1 on an ephemeral port. */
  async start(): Promise<number> {
    if (this.server !== null) throw new Error("the login listener is already running");
    const server = createServer((request, response) => {
      const url = new URL(request.url ?? "/", "http://127.0.0.1");
      if (request.method !== "GET" || url.pathname !== "/callback") {
        respond(response, 404, "Not found");
        return;
      }
      const token = url.searchParams.get("token") ?? "";
      const state = url.searchParams.get("state") ?? "";
      if (!token.startsWith("xpl_") || !sameState(state, this.state) || this.key !== null) {
        respond(response, 400, FAILURE_PAGE);
        return;
      }
      this.key = token;
      respond(response, 200, successPage(this.platform));
      for (const resolve of this.waiters.splice(0)) resolve(token);
    });
    this.server = server;
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", () => {
        server.off("error", reject);
        resolve();
      });
    });
    return this.port;
  }

  /** The approval URL for this attempt. */
  url(keyName: string): string {
    return authorizeUrl(this.platform, { state: this.state, port: this.port, keyName });
  }

  /** The key the callback delivered, if any yet. */
  received(): string | null {
    return this.key;
  }

  /** Wait for the key; `null` after the timeout or when `signal` aborts. */
  wait(timeoutMs: number = LOGIN_TIMEOUT_MS, signal?: AbortSignal): Promise<string | null> {
    if (this.key !== null) return Promise.resolve(this.key);
    return new Promise((resolve) => {
      let settled = false;
      const finish = (value: string | null) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
        resolve(value);
      };
      const onAbort = () => finish(null);
      const timer = setTimeout(() => finish(null), timeoutMs);
      signal?.addEventListener("abort", onAbort, { once: true });
      if (signal?.aborted) finish(null);
      this.waiters.push((key) => finish(key));
    });
  }

  /** Stop listening; repeated calls are safe. */
  async close(): Promise<void> {
    const server = this.server;
    // Settle pending waits so no timer outlives the listener.
    for (const resolve of this.waiters.splice(0)) resolve(null);
    if (server === null) return;
    this.server = null;
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}
