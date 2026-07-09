/**
 * Stdio pi runner: the in-sandbox peer of wmh/harness/pi_e2b.py (E2BStdioChannel).
 *
 * Same per-episode contract as runner_service.ts — materialize episode_start.files into
 * ./ep-<episode_id>/, bridge pi's LLM calls through an ephemeral localhost SSE server that frames
 * them as `llm_request` (model credentials stay on the host), route env tools as `tool_request`,
 * finish with `done`/`episode_error` — but the transport is this process's own stdin/stdout:
 * one base64(JSON) frame per line, because the E2B command channel is a text stream (the TCP
 * length-prefixed framing of runner_frames.ts does not apply here).
 *
 * stdout is the frame stream and NOTHING else may write to it: the first statements below rebind
 * console.log/info/warn/debug to stderr before any agent code can load. The host collects stderr
 * for diagnostics.
 *
 * Run (inside the sandbox, workdir holding node_modules + package.json {"type":"module"}):
 *   cd /home/user/pi-run && node --experimental-strip-types runner_stdio.ts
 *
 * Unlike runner_service.ts there is NO static ./src/agent.ts fallback — a fresh sandbox has no
 * checkout — so an episode_start without files is answered with episode_error. Episode logic is
 * deliberately a small controlled mirror of runner_service.ts (that file boots a TCP client at
 * import time, so its episode helpers cannot be imported without dialing out).
 */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import util from "node:util";
import { pathToFileURL } from "node:url";
import type { AddressInfo } from "node:net";

// CRITICAL — FIRST statements: every console channel that defaults to stdout is rebound to stderr
// before anything else runs (especially dynamically imported agent code), so stray prints can
// never corrupt the frame stream.
const toStderr = (...args: unknown[]): void => {
	process.stderr.write(util.format(...args) + "\n");
};
console.log = toStderr;
console.info = toStderr;
console.warn = toStderr;
console.debug = toStderr;

const AGENT_MODEL = process.env.PI_AGENT_MODEL ?? "worker";
const MAX_TURNS = Number(process.env.PI_MAX_TURNS ?? "20");

type Frame = Record<string, any>;

/** Encode one frame as the base64(JSON) + "\n" line pi_e2b.py's reader decodes. */
function encodeFrame(frame: Frame): string {
	return Buffer.from(JSON.stringify(frame), "utf8").toString("base64") + "\n";
}

/** Decode one base64(JSON) line into a frame; null for blank or undecodable lines. */
function decodeFrame(line: string): Frame | null {
	const text = line.trim();
	if (!text) return null;
	try {
		const frame = JSON.parse(Buffer.from(text, "base64").toString("utf8"));
		return frame && typeof frame === "object" && !Array.isArray(frame) ? (frame as Frame) : null;
	} catch {
		return null;
	}
}

/**
 * The stdio twin of runner_frames.FrameConn (same waiter/handler semantics, different wire):
 * `request` sends a frame with a fresh req_id and resolves on the matching response frame;
 * host-pushed frames (episode_start, shutdown) fire registered handlers.
 */
class StdioConn {
	private buf = "";
	private waiters = new Map<number, (f: Frame) => void>();
	private handlers = new Map<string, (f: Frame) => void>();
	private reqSeq = 0;

	constructor() {
		process.stdin.setEncoding("utf8");
		process.stdin.on("data", (chunk: string) => this.onData(chunk));
		// stdin EOF = the host side is gone; nobody is left to answer llm/tool requests.
		process.stdin.on("end", () => process.exit(0));
	}

	on(type: string, handler: (f: Frame) => void): void {
		this.handlers.set(type, handler);
	}

	send(frame: Frame): void {
		process.stdout.write(encodeFrame(frame));
	}

	request(type: string, payload: Frame): Promise<Frame> {
		const req_id = ++this.reqSeq;
		return new Promise((resolve) => {
			this.waiters.set(req_id, resolve);
			this.send({ type, req_id, ...payload });
		});
	}

	private onData(chunk: string): void {
		this.buf += chunk;
		let nl = this.buf.indexOf("\n");
		while (nl >= 0) {
			const line = this.buf.slice(0, nl);
			this.buf = this.buf.slice(nl + 1);
			const frame = decodeFrame(line);
			if (frame) this.dispatch(frame);
			nl = this.buf.indexOf("\n");
		}
	}

	private dispatch(frame: Frame): void {
		const rid = frame.req_id;
		if (typeof rid === "number" && this.waiters.has(rid)) {
			const resolve = this.waiters.get(rid);
			this.waiters.delete(rid);
			resolve?.(frame);
			return;
		}
		const handler = this.handlers.get(frame.type);
		if (handler) handler(frame);
	}
}

function assistantText(msg: any): string {
	if (!msg || msg.role !== "assistant" || !Array.isArray(msg.content)) return "";
	return msg.content
		.filter((c: any) => c?.type === "text")
		.map((c: any) => String(c.text ?? ""))
		.join("")
		.trim();
}

interface Bridge {
	url: string;
	close: () => void;
}

/** Localhost HTTP endpoint pi's openai-completions transport POSTs to; frames each request to the
 *  host and streams the returned completion object back as the SSE pi's parser expects.
 *  (Mirror of runner_service.ts's bridge, over StdioConn.) */
function startLlmBridge(conn: StdioConn): Promise<Bridge> {
	return new Promise((resolve) => {
		const server = http.createServer((req, res) => {
			const chunks: Buffer[] = [];
			req.on("data", (c: Buffer) => chunks.push(c));
			req.on("end", async () => {
				res.writeHead(200, { "Content-Type": "text/event-stream", Connection: "close" });
				try {
					const body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
					const reply = await conn.request("llm_request", { openai_body: body });
					if (reply.error) {
						res.end(`data: ${JSON.stringify({ error: { message: reply.error } })}\n\ndata: [DONE]\n\n`);
						return;
					}
					const choice = reply.completion?.choices?.[0] ?? {};
					const msg = choice.message ?? {};
					const delta: any = { role: "assistant", content: msg.content ?? "" };
					if (msg.tool_calls) {
						// Keep `function` explicitly nested (the streaming OpenAI shape the pi parser
						// expects); index each call.
						delta.tool_calls = msg.tool_calls.map((tc: any, i: number) => ({
							index: i,
							id: tc.id,
							type: tc.type ?? "function",
							function: tc.function ?? {},
						}));
					}
					const first = { choices: [{ index: 0, delta, finish_reason: null }] };
					const last = { choices: [{ index: 0, delta: {}, finish_reason: choice.finish_reason ?? "stop" }] };
					res.write(`data: ${JSON.stringify(first)}\n\n`);
					res.write(`data: ${JSON.stringify(last)}\n\n`);
					res.end("data: [DONE]\n\n");
				} catch (e) {
					res.end(`data: ${JSON.stringify({ error: { message: String(e) } })}\n\ndata: [DONE]\n\n`);
				}
			});
		});
		server.listen(0, "127.0.0.1", () => {
			const addr = server.address() as AddressInfo;
			resolve({ url: `http://127.0.0.1:${addr.port}/v1`, close: () => server.close() });
		});
	});
}

/**
 * Materialize episode_start.files (the doc's code surfaces) into a fresh per-episode dir under
 * cwd (so node_modules resolves upward from the workdir) and dynamically import the Agent from
 * there — a distinct module URL per episode, so a searched code mutation actually takes effect.
 * Returns [AgentCtor, cleanup].
 */
async function loadAgent(start: Frame): Promise<[any, () => void]> {
	const files: Record<string, string> = start.files ?? {};
	if (!files["src/agent.ts"]) {
		throw new Error("episode_start carried no src/agent.ts (the stdio runner has no static fallback)");
	}
	const base = path.join(process.cwd(), `ep-${start.episode_id}`);
	for (const [rel, content] of Object.entries(files)) {
		const dst = path.join(base, rel);
		fs.mkdirSync(path.dirname(dst), { recursive: true });
		fs.writeFileSync(dst, content);
	}
	const mod = await import(pathToFileURL(path.join(base, "src/agent.ts")).href);
	return [mod.Agent, () => fs.rmSync(base, { recursive: true, force: true })];
}

async function runEpisode(conn: StdioConn, start: Frame): Promise<void> {
	const episodeId = start.episode_id;
	let doneSent = false;
	let lastAssistantText = "";
	let bridge: Bridge | null = null;
	let cleanupSrc: () => void = () => {};

	try {
		const [AgentCtor, cleanup] = await loadAgent(start);
		cleanupSrc = cleanup;
		bridge = await startLlmBridge(conn);

		const model = {
			id: AGENT_MODEL,
			name: AGENT_MODEL,
			api: "openai-completions",
			provider: "link", // non-builtin -> uses model.baseUrl (our localhost bridge) directly
			baseUrl: bridge.url,
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 128000,
			maxTokens: 4096,
		};

		const envTools: any[] = (start.tools ?? [])
			.filter((t: any) => t.name !== "submit")
			.map((t: any) => ({
				name: t.name,
				label: t.name,
				description: t.description,
				parameters: t.parameters,
				execute: async (_id: string, params: any) => {
					const r = await conn.request("tool_request", { name: t.name, arguments: params });
					return {
						content: [{ type: "text", text: String(r.content ?? "") }],
						details: r,
						terminate: false,
					};
				},
			}));
		const submit = {
			name: "submit",
			label: "submit",
			description: "Submit the final answer and finish the task.",
			parameters: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
			execute: async (_id: string, params: { answer: string }) => {
				doneSent = true;
				conn.send({ type: "done", episode_id: episodeId, answer: params.answer ?? "" });
				return { content: [{ type: "text", text: "submitted" }], details: {}, terminate: true };
			},
		};

		const agent = new AgentCtor({
			initialState: { systemPrompt: start.system ?? "", model, tools: [...envTools, submit] },
			getApiKey: () => "x",
		});
		let turns = 0;
		agent.subscribe((event: any) => {
			if (event.type === "turn_end" || event.type === "message_end") {
				const t = assistantText(event.message);
				if (t) lastAssistantText = t;
			}
			if (event.type === "turn_end") {
				turns += 1;
				if (turns >= MAX_TURNS) agent.abort();
			}
		});

		await agent.prompt(start.instruction);
		if (!doneSent) conn.send({ type: "done", episode_id: episodeId, answer: lastAssistantText });
	} catch (e) {
		if (!doneSent) conn.send({ type: "episode_error", episode_id: episodeId, note: String(e) });
	} finally {
		bridge?.close();
		cleanupSrc();
	}
}

function main(): void {
	const conn = new StdioConn();
	conn.on("shutdown", () => process.exit(0));
	conn.on("episode_start", (start) => {
		runEpisode(conn, start).catch((e) => process.stderr.write(`[runner-stdio] episode fatal ${e}\n`));
	});
	conn.send({
		type: "hello",
		node_version: process.version,
		pi_version: "0.80.3",
		max_concurrent: 1,
		transport: "stdio",
	});
	process.stderr.write("[runner-stdio] ready\n");
}

main();
