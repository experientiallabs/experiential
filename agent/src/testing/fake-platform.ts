/**
 * A scripted Experiential platform API on loopback for the platform tool tests.
 *
 * Routes are `"GET /api/whoami"`-style keys mapped to a reply or a function of the request. Every
 * request is recorded with its path, query and authorization header. Unknown routes answer 404.
 */

import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";

export interface PlatformRequest {
  method: string;
  path: string;
  query: Record<string, string>;
  authorization: string | undefined;
}

export interface PlatformReply {
  status?: number;
  body?: unknown;
  headers?: Record<string, string>;
  /** Wait this long before answering. */
  delayMs?: number;
}

export type PlatformRoute = PlatformReply | ((request: PlatformRequest) => PlatformReply);

export interface FakePlatform {
  /** The origin, `http://127.0.0.1:<port>`. */
  origin: string;
  /** The gateway base URL on the same origin (`<origin>/v1`). */
  baseUrl: string;
  requests: PlatformRequest[];
  /** Requests to one path, in order. */
  calls: (path: string) => PlatformRequest[];
  route: (key: string, reply: PlatformRoute) => void;
  close: () => Promise<void>;
}

export const FAKE_ORG_ID = "org-1111";

export const WHOAMI = {
  org_id: FAKE_ORG_ID,
  org_slug: "acme",
  org_name: "Acme",
  identity_id: null,
  customer_plan_identity_id: null,
  is_provisioning: false
};

export async function startFakePlatform(routes: Record<string, PlatformRoute> = {}): Promise<FakePlatform> {
  const table = new Map<string, PlatformRoute>(Object.entries({ "GET /api/whoami": { body: WHOAMI }, ...routes }));
  const requests: PlatformRequest[] = [];
  const server: Server = createServer((incoming, response) => {
    const url = new URL(incoming.url ?? "/", "http://fake");
    const request: PlatformRequest = {
      method: incoming.method ?? "GET",
      path: url.pathname,
      query: Object.fromEntries(url.searchParams.entries()),
      authorization: incoming.headers.authorization
    };
    requests.push(request);
    const route = table.get(`${request.method} ${request.path}`);
    const reply: PlatformReply = route === undefined ? { status: 404, body: { error: `no route ${request.path}` } } : typeof route === "function" ? route(request) : route;
    const send = () => {
      const body = typeof reply.body === "string" ? reply.body : JSON.stringify(reply.body ?? {});
      response.writeHead(reply.status ?? 200, { "content-type": "application/json", ...reply.headers });
      response.end(body);
    };
    if (reply.delayMs) setTimeout(send, reply.delayMs);
    else send();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  const origin = `http://127.0.0.1:${port}`;
  return {
    origin,
    baseUrl: `${origin}/v1`,
    requests,
    calls: (path) => requests.filter((request) => request.path === path),
    route: (key, reply) => table.set(key, reply),
    close: () =>
      new Promise((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      })
  };
}
