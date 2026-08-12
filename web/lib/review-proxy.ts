const LOOPBACK_HOSTS = new Set(["127.0.0.1", "::1", "[::1]", "localhost"]);
const RUBRIC_ACTIONS = new Set([
  "accept",
  "reject",
  "edit",
  "add",
  "replace_all",
  "order",
  "finalize"
]);

export class LocalProxyError extends Error {}

export function configuredWebPort(): number {
  return configuredPort("WMO_REVIEW_WEB_PORT", 3000);
}

export function validateLocalWebRequest(request: Request, expectedPort: number): void {
  const host = request.headers.get("host");
  if (!host) {
    throw new LocalProxyError("Local review requires a Host header.");
  }
  const expected = parseOrigin(`http://${host}`, "Host");
  if (!LOOPBACK_HOSTS.has(expected.hostname) || expected.port !== expectedPort) {
    throw new LocalProxyError("Local review Host must be loopback with the expected port.");
  }
  for (const headerName of ["origin", "referer"] as const) {
    const value = request.headers.get(headerName);
    if (value && !sameOrigin(parseOrigin(value, headerName), expected)) {
      throw new LocalProxyError(`Local review ${headerName} must match the Host origin.`);
    }
  }
}

export async function proxyLocalReview(
  request: Request,
  path: string[],
  method: "GET" | "POST"
): Promise<Response> {
  const webPort = configuredWebPort();
  validateLocalWebRequest(request, webPort);
  const apiUrl = configuredApiUrl();
  const targetPath = allowedTargetPath(path, method);
  const body = method === "POST" ? await request.text() : undefined;
  const response = await fetch(new URL(targetPath, apiUrl), {
    method,
    body,
    cache: "no-store",
    headers: method === "POST" ? { "Content-Type": "application/json" } : undefined
  });
  return new Response(response.body, {
    status: response.status,
    headers: { "Content-Type": response.headers.get("content-type") ?? "application/json" }
  });
}

function configuredApiUrl(): URL {
  const value = process.env.WMO_REVIEW_API_URL ?? "http://127.0.0.1:8017";
  const url = new URL(value);
  if (
    url.protocol !== "http:" ||
    !LOOPBACK_HOSTS.has(url.hostname) ||
    !url.port ||
    url.username ||
    url.password ||
    url.pathname !== "/" ||
    url.search ||
    url.hash
  ) {
    throw new LocalProxyError("WMO_REVIEW_API_URL must be an explicit loopback HTTP origin.");
  }
  return url;
}

function configuredPort(name: string, fallback: number): number {
  const value = process.env[name];
  const port = value === undefined ? fallback : Number(value);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new LocalProxyError(`${name} must be an integer from 1 through 65535.`);
  }
  return port;
}

function allowedTargetPath(path: string[], method: "GET" | "POST"): string {
  if (method === "GET" && path.join("/") === "api/review") {
    return "/api/review";
  }
  if (method === "POST" && path.join("/") === "api/review/score") {
    return "/api/review/score";
  }
  if (
    method === "POST" &&
    path.length === 4 &&
    path[0] === "api" &&
    path[1] === "review" &&
    path[2] === "rubric" &&
    RUBRIC_ACTIONS.has(path[3])
  ) {
    return `/api/review/rubric/${path[3]}`;
  }
  if (
    method === "POST" &&
    path.length === 5 &&
    path[0] === "api" &&
    path[1] === "review" &&
    path[2] === "calibration" &&
    /^[a-z0-9][a-z0-9._-]*$/.test(path[3]) &&
    path[4] === "approve"
  ) {
    return `/api/review/calibration/${path[3]}/approve`;
  }
  throw new LocalProxyError("This local review proxy path is not allowed.");
}

function parseOrigin(value: string, label: string): { hostname: string; port: number; protocol: string } {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new LocalProxyError(`Local review ${label} is not a valid URL.`);
  }
  if (url.protocol !== "http:" || !url.hostname || !url.port || url.username || url.password) {
    throw new LocalProxyError(`Local review ${label} must be an explicit HTTP origin.`);
  }
  return { hostname: url.hostname.toLowerCase(), port: Number(url.port), protocol: url.protocol };
}

function sameOrigin(
  left: { hostname: string; port: number; protocol: string },
  right: { hostname: string; port: number; protocol: string }
): boolean {
  return (
    left.protocol === right.protocol && left.hostname === right.hostname && left.port === right.port
  );
}
