/**
 * Live-session pi runner: the in-sandbox peer of wmh/harness/live_session.py (LiveSession).
 *
 * Where runner_stdio.ts runs ONE fire-and-forget episode, this runner hosts ONE long-lived pi
 * Agent for an interactive multi-turn session: the host sends `session_start` once (materializing
 * the champion's code surfaces), then `user_message` / `abort` / `ping` frames over the session's
 * life, and the runner drives the same Agent across every turn on one accumulating transcript.
 * The transport, the localhost LLM bridge (credentials stay host-side), and the env-tools-as-
 * `tool_request` contract are identical to runner_stdio.ts — deliberately re-mirrored rather than
 * imported, because runner_stdio boots per-episode helpers and this runner's lifecycle differs.
 *
 * stdout is the frame stream and NOTHING else may write to it: console.* is rebound to stderr
 * before any agent code loads. One base64(JSON) frame per line.
 *
 * Run (inside the sandbox, workdir holding node_modules + package.json {"type":"module"}):
 *   cd /home/user/pi-run && node --experimental-strip-types runner_live.ts
 */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import util from "node:util";
import { pathToFileURL } from "node:url";
import type { AddressInfo } from "node:net";

// CRITICAL — FIRST statements: rebind every stdout-defaulting console channel to stderr before
// anything else runs, so stray prints can never corrupt the frame stream.
const toStderr = (...args: unknown[]): void => {
	process.stderr.write(util.format(...args) + "\n");
};
console.log = toStderr;
console.info = toStderr;
console.warn = toStderr;
console.debug = toStderr;

const AGENT_MODEL = process.env.PI_AGENT_MODEL ?? "worker";

type Frame = Record<string, any>;

function encodeFrame(frame: Frame): string {
	return Buffer.from(JSON.stringify(frame), "utf8").toString("base64") + "\n";
}

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

/** The stdio twin of runner_frames.FrameConn: `request` awaits a matching response by req_id;
 *  host-pushed frames (session_start, user_message, abort, ping, shutdown) fire handlers. */
class StdioConn {
	private buf = "";
	private waiters = new Map<number, (f: Frame) => void>();
	private handlers = new Map<string, (f: Frame) => void>();
	private reqSeq = 0;

	constructor() {
		process.stdin.setEncoding("utf8");
		process.stdin.on("data", (chunk: string) => this.onData(chunk));
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

/** Localhost endpoint pi's openai-completions transport POSTs to; frames each request to the host
 *  and renders the returned completion back as the SSE pi's parser expects. Creds stay host-side. */
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

/** Materialize session_start.files (the champion's code surfaces) into ./live/ once and import the
 *  Agent. Unlike the episode runner there is no per-run dir — the session reuses one Agent. */
async function loadAgent(start: Frame): Promise<any> {
	const files: Record<string, string> = start.files ?? {};
	if (!files["src/agent.ts"]) {
		throw new Error("session_start carried no src/agent.ts (the live runner has no static fallback)");
	}
	const base = path.join(process.cwd(), "live");
	const basePrefix = base.endsWith(path.sep) ? base : base + path.sep;
	for (const [rel, content] of Object.entries(files)) {
		// Keep every materialized file under ./live: `path.join`/`resolve` let a
		// `../` or absolute manifest key escape and overwrite arbitrary sandbox
		// files, so reject anything that resolves outside the base.
		const dst = path.resolve(base, rel);
		if (dst !== base && !dst.startsWith(basePrefix)) {
			throw new Error(`session_start file path escapes the live directory: ${rel}`);
		}
		fs.mkdirSync(path.dirname(dst), { recursive: true });
		fs.writeFileSync(dst, content);
	}
	const mod = await import(pathToFileURL(path.join(base, "src/agent.ts")).href);
	if (typeof mod.Agent !== "function") {
		throw new Error("champion src/agent.ts does not export an Agent class");
	}
	return mod.Agent;
}

const REQUIRED_AGENT_METHODS = ["prompt", "steer", "abort", "subscribe"] as const;

function assertInteractive(AgentCtor: any): void {
	for (const method of REQUIRED_AGENT_METHODS) {
		if (typeof AgentCtor.prototype?.[method] !== "function") {
			throw new Error(
				`champion Agent is not steerable: missing ${method}() — this harness cannot run a live session`,
			);
		}
	}
}

function userMessage(text: string): any {
	return { role: "user", content: [{ type: "text", text }], timestamp: Date.now() };
}

/**
 * After an aborted run, the transcript tail can hold an assistant message whose toolCalls never got
 * results (the loop returned on the abort signal before executing/finishing them). Left as-is, the
 * NEXT provider request is a 400 (OpenAI) / validation error (Bedrock Converse): an assistant
 * tool-call with no matching tool result. Append a synthetic "cancelled by user" result for every
 * orphaned toolCall so the next turn is well-formed. (Vendored eval runners never hit this — they
 * only ever run one prompt; interactivity is what surfaces it.)
 */
function repairOrphanedToolCalls(agent: any): void {
	const messages: any[] = agent.state?.messages ?? [];
	const resolved = new Set<string>();
	for (const m of messages) {
		if (m?.role === "toolResult" && typeof m.toolCallId === "string") resolved.add(m.toolCallId);
	}
	const repairs: any[] = [];
	for (const m of messages) {
		if (m?.role !== "assistant" || !Array.isArray(m.content)) continue;
		for (const block of m.content) {
			if (block?.type === "toolCall" && typeof block.id === "string" && !resolved.has(block.id)) {
				resolved.add(block.id);
				repairs.push({
					role: "toolResult",
					toolCallId: block.id,
					toolName: block.name ?? "",
					content: [{ type: "text", text: "cancelled by user" }],
					details: {},
					isError: true,
					timestamp: Date.now(),
				});
			}
		}
	}
	if (repairs.length > 0) agent.state.messages = [...messages, ...repairs];
}

/** Owns the single Agent and serializes prompt/steer/abort across the session's frames. */
class Session {
	// NOTE: fields declared explicitly (not via constructor parameter properties) — node's
	// --experimental-strip-types rejects `constructor(private x)` in strip-only mode.
	private readonly conn: StdioConn;
	private agent: any = null;
	private bridge: Bridge | null = null;
	private running = false;
	private turns = 0;
	private turnCap = 60;
	private interrupted = false;
	private hitTurnCap = false;

	constructor(conn: StdioConn) {
		this.conn = conn;
	}

	async start(frame: Frame): Promise<void> {
		this.turnCap = Number(frame.turn_cap ?? 60);
		const AgentCtor = await loadAgent(frame);
		assertInteractive(AgentCtor);
		this.bridge = await startLlmBridge(this.conn);

		const model = {
			id: AGENT_MODEL,
			name: AGENT_MODEL,
			api: "openai-completions",
			provider: "link", // non-builtin -> uses model.baseUrl (our localhost bridge) directly
			baseUrl: this.bridge.url,
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 128000,
			maxTokens: 4096,
		};

		const tools = this.buildTools(frame.tools ?? []);
		this.agent = new AgentCtor({
			initialState: { systemPrompt: frame.system ?? "", model, tools },
			getApiKey: () => "x",
			// Real filesystem tools may race if run in parallel; keep them ordered. Steers drain
			// as a batch ("all") so a burst of user messages doesn't each cost an extra turn.
			toolExecution: "sequential",
			steeringMode: "all",
		});
		this.agent.subscribe((event: any) => {
			if (event.type === "turn_end") {
				this.turns += 1;
				if (this.turns >= this.turnCap && this.running) {
					this.hitTurnCap = true;
					this.agent.abort();
				}
			}
		});
		this.sendState("idle");
	}

	private buildTools(specs: any[]): any[] {
		const envTools = specs
			.filter((t: any) => t.name !== "submit")
			.map((t: any) => ({
				name: t.name,
				label: t.name,
				description: t.description,
				parameters: t.parameters,
				execute: async (_id: string, params: any, signal?: AbortSignal) => {
					if (signal?.aborted) {
						return { content: [{ type: "text", text: "interrupted" }], details: {}, terminate: false };
					}
					const r = await this.conn.request("tool_request", { name: t.name, arguments: params });
					// A host-side failure (budget exhausted, unknown tool, executor error) must reach
					// the agent AS a failure: throw so pi records an error tool result, rather than
					// letting the model reason from a failed action as if it succeeded.
					if (r.is_error) {
						throw new Error(String(r.content ?? "tool failed"));
					}
					return { content: [{ type: "text", text: String(r.content ?? "") }], details: r, terminate: false };
				},
			}));
		const submit = {
			name: "submit",
			label: "submit",
			description: "Finish the task and submit your answer/result summary. This ends the run.",
			parameters: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
			execute: async (_id: string, params: { answer?: string }, signal?: AbortSignal) => {
				// Honor an interrupt racing a submit: don't emit a final submit for an aborted turn.
				if (signal?.aborted) {
					return { content: [{ type: "text", text: "interrupted" }], details: {}, terminate: false };
				}
				// The host emits the submit event; the run then ends (but the SESSION stays alive
				// awaiting the next user message).
				await this.conn.request("tool_request", { name: "submit", arguments: { answer: params.answer ?? "" } });
				// Re-check after the round-trip: if the interrupt landed while the request was in
				// flight, end via the abort path rather than terminating the run as a clean submit.
				if (signal?.aborted) {
					return { content: [{ type: "text", text: "interrupted" }], details: {}, terminate: false };
				}
				return { content: [{ type: "text", text: "submitted" }], details: {}, terminate: true };
			},
		};
		return [...envTools, submit];
	}

	handleUserMessage(frame: Frame): void {
		const text = String(frame.text ?? "");
		if (!this.agent) return;
		if (this.running) {
			this.agent.steer(userMessage(text));
			return;
		}
		void this.runPrompt(text);
	}

	private async runPrompt(text: string): Promise<void> {
		this.running = true;
		this.turns = 0;
		this.interrupted = false;
		this.hitTurnCap = false;
		this.sendState("running");
		try {
			await this.agent.prompt(text);
		} catch (e) {
			this.conn.send({ type: "episode_error", note: String(e) });
		} finally {
			this.running = false;
			repairOrphanedToolCalls(this.agent);
			const reason = this.hitTurnCap ? "turn_limit" : this.interrupted ? "aborted" : "completed";
			this.sendState("idle", reason);
		}
	}

	handleAbort(_frame: Frame): void {
		if (this.agent && this.running) {
			this.interrupted = true;
			// Interrupt cancels the WHOLE pending turn: clear any messages the user queued (via
			// steer) before pressing Stop, so a follow-up typed just before the interrupt is not
			// silently drained into the next prompt. New messages after this start a fresh turn.
			this.agent.clearAllQueues?.();
			this.agent.abort();
		}
	}

	handlePing(frame: Frame): void {
		this.conn.send({ type: "pong", nonce: frame.nonce });
	}

	private sendState(status: "idle" | "running", reason?: string): void {
		const frame: Frame = { type: "state", status, turns: this.turns };
		if (reason) frame.reason = reason;
		this.conn.send(frame);
	}
}

function main(): void {
	const conn = new StdioConn();
	const session = new Session(conn);
	conn.on("shutdown", () => process.exit(0));
	conn.on("session_start", (start) => {
		session.start(start).catch((e) => {
			conn.send({ type: "episode_error", note: String(e) });
			process.stderr.write(`[runner-live] start fatal ${e}\n`);
		});
	});
	conn.on("user_message", (f) => session.handleUserMessage(f));
	conn.on("abort", (f) => session.handleAbort(f));
	conn.on("ping", (f) => session.handlePing(f));
	conn.send({
		type: "hello",
		node_version: process.version,
		pi_version: "0.80.3",
		mode: "session",
		transport: "stdio",
	});
	process.stderr.write("[runner-live] ready\n");
}

main();
