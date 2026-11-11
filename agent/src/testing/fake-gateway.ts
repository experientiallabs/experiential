/**
 * A scripted OpenAI-compatible gateway on loopback for tests.
 *
 * It serves `GET /v1/models` and streaming `POST /v1/chat/completions` from a queue of turns
 * (text, tool calls, reasoning, or an HTTP error), records every request body, and accepts any
 * other POST through a pluggable handler (event push and completion callback endpoints).
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";

export type FakeTurn =
  | { text: string; reasoning?: string; delayMs?: number; cost?: number }
  | { calls: { name: string; arguments: Record<string, unknown> }[]; delayMs?: number }
  | { status: number; error: string };

export interface RecordedRequest {
  method: string;
  path: string;
  headers: IncomingMessage["headers"];
  body: unknown;
}

export interface FakeGateway {
  /** OpenAI-compatible root, `http://127.0.0.1:<port>/v1`. */
  baseUrl: string;
  /** Origin without `/v1`. */
  origin: string;
  /** Every request received, in order. */
  requests: RecordedRequest[];
  /** Bodies of chat completion requests, in order. */
  completions: () => Record<string, unknown>[];
  /** Replace the queued turns. */
  script: (turns: FakeTurn[]) => void;
  /** Handle POSTs to any path other than chat completions. */
  onPost: (handler: (request: RecordedRequest) => { status: number; body?: unknown }) => void;
  close: () => Promise<void>;
}

async function readBody(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) chunks.push(chunk as Buffer);
  const text = Buffer.concat(chunks).toString("utf8");
  if (text === "") return undefined;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function sse(response: ServerResponse, chunks: unknown[]): void {
  response.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
  for (const chunk of chunks) response.write(`data: ${JSON.stringify(chunk)}\n\n`);
  response.end("data: [DONE]\n\n");
}

function turnChunks(turn: Exclude<FakeTurn, { status: number }>, index: number): unknown[] {
  const base = { id: `chatcmpl-${index}`, object: "chat.completion.chunk", created: 0, model: "fake" };
  const chunks: unknown[] = [];
  if ("text" in turn) {
    if (turn.reasoning) chunks.push({ ...base, choices: [{ index: 0, delta: { role: "assistant", reasoning: turn.reasoning } }] });
    chunks.push({ ...base, choices: [{ index: 0, delta: { role: "assistant", content: turn.text } }] });
    chunks.push({ ...base, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] });
  } else {
    turn.calls.forEach((call, i) => {
      chunks.push({
        ...base,
        choices: [
          {
            index: 0,
            delta: { role: "assistant", tool_calls: [{ index: i, id: `call_${index}_${i}`, type: "function", function: { name: call.name, arguments: JSON.stringify(call.arguments) } }] }
          }
        ]
      });
    });
    chunks.push({ ...base, choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] });
  }
  const cost = "cost" in turn && turn.cost !== undefined ? turn.cost : 0.001;
  chunks.push({ ...base, choices: [], usage: { prompt_tokens: 50, completion_tokens: 5, total_tokens: 55, cost } });
  return chunks;
}

/** Start a fake gateway serving the given turns (more can be queued with `script`). */
export async function startFakeGateway(initial: FakeTurn[] = [], models: unknown[] = [{ id: "fake-model", object: "model", created: 0, owned_by: "exp" }]): Promise<FakeGateway> {
  let turns = [...initial];
  let served = 0;
  const requests: RecordedRequest[] = [];
  let postHandler: (request: RecordedRequest) => { status: number; body?: unknown } = () => ({ status: 200, body: {} });
  const sockets = new Set<import("node:net").Socket>();

  const server: Server = createServer(async (request, response) => {
    const path = new URL(request.url ?? "/", "http://localhost").pathname;
    const recorded: RecordedRequest = { method: request.method ?? "GET", path, headers: request.headers, body: await readBody(request) };
    requests.push(recorded);
    if (recorded.method === "GET" && path === "/v1/models") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ object: "list", data: models }));
      return;
    }
    if (recorded.method === "POST" && path === "/v1/chat/completions") {
      served += 1;
      const turn = turns.shift() ?? { text: "Done." };
      if ("delayMs" in turn && turn.delayMs) {
        const closed = await new Promise<boolean>((resolve) => {
          const timer = setTimeout(() => resolve(false), turn.delayMs);
          request.socket.once("close", () => {
            clearTimeout(timer);
            resolve(true);
          });
        });
        if (closed) return;
      }
      if ("status" in turn) {
        response.writeHead(turn.status, { "content-type": "application/json" });
        response.end(JSON.stringify({ error: { message: turn.error, type: "invalid_request_error" } }));
        return;
      }
      sse(response, turnChunks(turn, served));
      return;
    }
    const answer = postHandler(recorded);
    response.writeHead(answer.status, { "content-type": "application/json" });
    response.end(JSON.stringify(answer.body ?? {}));
  });
  server.on("connection", (socket) => {
    sockets.add(socket);
    socket.on("close", () => sockets.delete(socket));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  const origin = `http://127.0.0.1:${port}`;
  return {
    baseUrl: `${origin}/v1`,
    origin,
    requests,
    completions: () => requests.filter((r) => r.path === "/v1/chat/completions").map((r) => r.body as Record<string, unknown>),
    script: (next) => {
      turns = [...next];
    },
    onPost: (handler) => {
      postHandler = handler;
    },
    close: async () => {
      for (const socket of sockets) socket.destroy();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  };
}
