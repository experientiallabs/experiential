/**
 * Typed client for a locally running `wmh serve` (FastAPI, wmh/serving/server.py).
 *
 * Plain fetch wrappers, one per endpoint. Unexpected statuses throw with the server's `detail`
 * message; callers branch on thrown ApiError.status for expected failures (404, 409, 503).
 * The base URL comes from NEXT_PUBLIC_WMH_API (default: http://localhost:8000) — never a secret.
 */

import type {
  Action,
  BuildSnapshot,
  EnvState,
  ModelsResponse,
  Observation,
  RunRecord,
  Session,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_WMH_API ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    detail: string,
  ) {
    super(detail);
  }
}

/** Turn a FastAPI error body into a readable string. 422 `detail` is a LIST of error objects. */
function errorDetail(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((e) =>
          e && typeof e === "object" && "msg" in e
            ? `${Array.isArray((e as { loc?: unknown[] }).loc) ? (e as { loc: unknown[] }).loc.slice(1).join(".") + ": " : ""}${(e as { msg: string }).msg}`
            : String(e),
        )
        .join("; ");
    }
  }
  return fallback;
}

/** Parse an error response into an ApiError, tolerating non-JSON bodies. */
async function toApiError(res: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON error body (proxy HTML, plain text): fall back to statusText
  }
  return new ApiError(res.status, errorDetail(body, res.statusText));
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
  });
  if (!res.ok) throw await toApiError(res);
  return res.json() as Promise<T>;
}

/** True when a wmh serve backend answers on API_BASE. */
export async function isServeUp(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/healthz`, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}

export function listModels(): Promise<ModelsResponse> {
  return request("/world_models");
}

export function createSession(
  model: string,
  task: string | null,
): Promise<{ session_id: string; state: EnvState }> {
  return request(`/world_models/${encodeURIComponent(model)}/sessions`, {
    method: "POST",
    body: JSON.stringify({ task }),
  });
}

export function getSession(model: string, sessionId: string): Promise<Session> {
  return request(
    `/world_models/${encodeURIComponent(model)}/sessions/${sessionId}`,
  );
}

/** Step returns both the observation and the post-step env state (no follow-up session GET). */
export function step(
  model: string,
  sessionId: string,
  action: Action,
): Promise<{ observation: Observation; state: EnvState }> {
  return request(
    `/world_models/${encodeURIComponent(model)}/sessions/${sessionId}/step`,
    { method: "POST", body: JSON.stringify({ action }) },
  );
}

export function sessionUsage(
  model: string,
  sessionId: string,
): Promise<RunRecord> {
  return request(
    `/world_models/${encodeURIComponent(model)}/sessions/${sessionId}/usage`,
  );
}

export type BuildRequest = {
  name: string;
  file: string;
  title?: string;
  description?: string;
  tags?: string[];
  provider?: string;
  model?: string;
  region?: string | null;
  gepa_budget?: number;
  train_split?: number;
};

export async function startBuild(req: BuildRequest): Promise<string> {
  const res = await request<{ build_id: string }>("/world_models/builds", {
    method: "POST",
    body: JSON.stringify(req),
  });
  return res.build_id;
}

export function buildSnapshot(buildId: string): Promise<BuildSnapshot> {
  return request(`/world_models/builds/${buildId}`);
}

/** EventSource over the build's SSE stream; caller closes it on terminal event. */
export function openBuildEvents(buildId: string): EventSource {
  return new EventSource(`${API_BASE}/world_models/builds/${buildId}/events`);
}

export async function uploadTraces(file: File): Promise<string> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API_BASE}/world_models/builds/uploads`, {
    method: "POST",
    body,
  });
  if (!res.ok) throw await toApiError(res);
  return (await res.json()).path as string;
}
