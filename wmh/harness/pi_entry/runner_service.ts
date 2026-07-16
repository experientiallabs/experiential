/**
 * Persistent pi runner for the RunnerLink transport.
 *
 * Dials the control-plane host once (outbound — no inbound port, no reverse tunnel), sends `hello`,
 * then runs one pi episode per `episode_start` frame the host pushes. The worker LLM and the
 * environment tools never touch the network directly:
 *   - the worker LLM goes through a tiny in-process localhost bridge that pi's openai-completions
 *     transport POSTs to; the bridge frames the request as `llm_request` and renders the host's
 *     returned completion object back as SSE (so pi-ai's own codec does the OpenAI translation and
 *     no model credentials ever reach this process),
 *   - each env tool `execute` sends a `tool_request` frame and awaits the observation,
 *   - `submit` (or, failing that, pi's last assistant message) sends `done`.
 *
 * Run: PI_LINK_ADDR=host:port node --experimental-strip-types runner_service.ts
 *
 * NOTE (step 3): pi source is loaded from ./src next to this file (statically imported). Per-episode
 * source materialization from episode_start.files + child-process isolation per episode is a later
 * migration step; today one runner serves episodes sequentially over the one connection.
 */
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import path from "node:path";
import { pathToFileURL } from "node:url";
import type { Model } from "@earendil-works/pi-ai";
import { Agent as StaticAgent } from "./src/agent.ts";
import type { AgentTool, AgentToolResult } from "./src/types.ts";
import { FrameConn, type Frame } from "./runner_frames.ts";

const [HOST, PORT] = (process.env.PI_LINK_ADDR ?? "127.0.0.1:8900").split(":");
const AGENT_MODEL = process.env.PI_AGENT_MODEL ?? "worker";
const configuredMaxTurns = Number(process.env.PI_MAX_TURNS ?? "20");
const DEFAULT_MAX_TURNS =
	Number.isInteger(configuredMaxTurns) && configuredMaxTurns >= 1 ? configuredMaxTurns : 20;

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
 *  host and streams the returned completion object back as the SSE pi's parser expects. */
function startLlmBridge(conn: FrameConn): Promise<Bridge> {
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
					if (msg.reasoning_details) delta.reasoning_details = msg.reasoning_details;
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
					const last = {
						choices: [{ index: 0, delta: {}, finish_reason: choice.finish_reason ?? "stop" }],
						// Pi uses the latest assistant usage to estimate occupied context. Without this,
						// it falls back to chars/4 and can prematurely clamp the next output budget.
						usage: reply.completion?.usage,
					};
					res.write(`data: ${JSON.stringify(first)}\n\n`);
					res.write(`data: ${JSON.stringify(last)}\n\n`);
					res.end("data: [DONE]\n\n");
				} catch (e) {
					res.end(`data: ${JSON.stringify({ error: { message: String(e) } })}\n\ndata: [DONE]\n\n`);
				}
			});
		});
		server.listen(0, "127.0.0.1", () => {
			const addr = server.address() as net.AddressInfo;
			resolve({ url: `http://127.0.0.1:${addr.port}/v1`, close: () => server.close() });
		});
	});
}

/**
 * Load pi's Agent for this episode. If episode_start carries `files` (the doc's code surfaces), they
 * are materialized into a fresh per-episode dir under cwd (~/pi-run, so node_modules resolves
 * upward) and the Agent is dynamically imported from there — a distinct module URL per episode, so a
 * searched code mutation actually takes effect. With no files, the statically-imported Agent runs
 * (dev / prompt-only searches). Returns [AgentCtor, cleanup].
 */
async function loadAgent(start: Frame): Promise<[any, () => void]> {
	const files: Record<string, string> = start.files ?? {};
	if (Object.keys(files).length === 0) return [StaticAgent, () => {}];
	const base = path.join(process.cwd(), `ep-${start.episode_id}`);
	for (const [rel, content] of Object.entries(files)) {
		const dst = path.join(base, rel);
		fs.mkdirSync(path.dirname(dst), { recursive: true });
		fs.writeFileSync(dst, content);
	}
	const mod = await import(pathToFileURL(path.join(base, "src/agent.ts")).href);
	return [mod.Agent, () => fs.rmSync(base, { recursive: true, force: true })];
}

async function runEpisode(conn: FrameConn, start: Frame): Promise<void> {
	const bridge = await startLlmBridge(conn);
	const [AgentCtor, cleanupSrc] = await loadAgent(start);
	const episodeId = start.episode_id;
	const maxTurns =
		Number.isInteger(start.max_turns) && start.max_turns >= 1
			? start.max_turns
			: DEFAULT_MAX_TURNS;
	const maxOutputTokens =
		Number.isInteger(start.max_output_tokens) && start.max_output_tokens >= 1
			? start.max_output_tokens
			: 4096;
	let doneSent = false;
	let lastAssistantText = "";

	const model: Model<"openai-completions"> = {
		id: AGENT_MODEL,
		name: AGENT_MODEL,
		api: "openai-completions",
		provider: "link", // non-builtin -> uses model.baseUrl (our localhost bridge) directly
		baseUrl: bridge.url,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: maxOutputTokens,
	};

	const envTools: AgentTool<any>[] = (start.tools ?? [])
		.filter((t: any) => t.name !== "submit")
		.map(
			(t: any): AgentTool<any> => ({
				name: t.name,
				label: t.name,
				description: t.description,
				parameters: t.parameters,
				execute: async (_id, params): Promise<AgentToolResult<any>> => {
					const r = await conn.request("tool_request", { name: t.name, arguments: params });
					return {
						content: [{ type: "text", text: String(r.content ?? "") }],
						details: r,
						terminate: false,
					};
				},
			}),
		);
	const submit: AgentTool<any> = {
		name: "submit",
		label: "submit",
		description: "Submit the final answer and finish the task.",
		parameters: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
		execute: async (_id, params: { answer: string }): Promise<AgentToolResult<any>> => {
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
			if (turns >= maxTurns) agent.abort();
		}
	});

	try {
		await agent.prompt(start.instruction);
		if (!doneSent) conn.send({ type: "done", episode_id: episodeId, answer: lastAssistantText });
	} catch (e) {
		if (!doneSent) conn.send({ type: "episode_error", episode_id: episodeId, note: String(e) });
	} finally {
		bridge.close();
		cleanupSrc();
	}
}

function main(): void {
	const sock = net.connect(Number(PORT), HOST);
	const conn = new FrameConn(sock);
	sock.on("connect", () => {
		conn.send({ type: "hello", node_version: process.version, pi_version: "0.80.3", max_concurrent: 1 });
		process.stderr.write(`[runner] connected ${HOST}:${PORT}\n`);
	});
	conn.on("episode_start", (start) => {
		runEpisode(conn, start).catch((e) => process.stderr.write(`[runner] episode fatal ${e}\n`));
	});
	conn.on("hello_ack", () => {});
	sock.on("close", () => process.exit(0));
	sock.on("error", (e: Error) => {
		process.stderr.write(`[runner] socket error ${e}\n`);
		process.exit(1);
	});
}

main();
