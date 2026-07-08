/**
 * Headless pi-agent entrypoint for the E2B stdio-broker transport (no host-inbound, no tunnel).
 *
 * The world model stays on the control host; this process reaches it only through E2B's own
 * channel, host-driven:
 *   - task is read from PI_TASK_FILE (the host uploads it),
 *   - each tool call is emitted on STDOUT as `__WMH_TOOL__<json>` and the process then polls
 *     PI_RESP_DIR/<id>.json, which the host writes back into the sandbox after answering it with
 *     the in-process world model,
 *   - `submit` emits `__WMH_DONE__<json>` and ends the loop.
 * The agent's own LLM calls go straight out to the model provider (PI_AGENT_BASE_URL) with the
 * user's key — normal outbound; no secret ever comes from the host.
 *
 * Lives next to src/ when materialized (import "./src/agent.ts").
 */
import { readFileSync, existsSync, rmSync } from "node:fs";
import { Agent } from "./src/agent.ts";
import type { AgentTool, AgentToolResult } from "./src/types.ts";
import type { Model } from "@earendil-works/pi-ai";

const TASK_FILE = process.env.PI_TASK_FILE ?? "/home/user/harness/wm_task.json";
const RESP_DIR = process.env.PI_RESP_DIR ?? "/home/user/harness/resp";
const AGENT_BASE_URL = process.env.PI_AGENT_BASE_URL ?? "https://api.deepseek.com/v1";
const AGENT_MODEL = process.env.PI_AGENT_MODEL ?? "deepseek-chat";
const AGENT_KEY = process.env.PI_AGENT_KEY ?? "";
const MAX_TURNS = Number(process.env.PI_MAX_TURNS ?? "20");

interface TaskTool {
	name: string;
	description: string;
	parameters: any;
}
interface Task {
	instruction: string;
	system?: string;
	tools: TaskTool[];
}

function emit(tag: string, obj: unknown): void {
	process.stdout.write(`${tag}${JSON.stringify(obj)}\n`);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Block until the host writes the response file for this tool-call id, then consume it.
async function awaitResponse(id: string): Promise<{ content: string; is_error?: boolean }> {
	const path = `${RESP_DIR}/${id}.json`;
	for (let i = 0; i < 3000; i++) {
		// up to ~5 min at 100ms
		if (existsSync(path)) {
			const body = readFileSync(path, "utf8");
			rmSync(path, { force: true });
			return JSON.parse(body);
		}
		await sleep(100);
	}
	return { content: "tool response timed out", is_error: true };
}

let toolSeq = 0;
let doneSent = false;

// pi often ends by writing its final answer as a normal assistant message rather than calling
// `submit`; capture the latest assistant text to use as the answer if the loop exits without one.
let lastAssistantText = "";
function assistantText(msg: any): string {
	if (!msg || msg.role !== "assistant" || !Array.isArray(msg.content)) return "";
	return msg.content
		.filter((c: any) => c?.type === "text")
		.map((c: any) => String(c.text ?? ""))
		.join("")
		.trim();
}

function makeEnvTool(t: TaskTool): AgentTool<any> {
	return {
		name: t.name,
		label: t.name,
		description: t.description,
		parameters: t.parameters,
		execute: async (_id, params): Promise<AgentToolResult<any>> => {
			const id = `t${toolSeq++}`;
			emit("__WMH_TOOL__", { id, name: t.name, arguments: params });
			const r = await awaitResponse(id);
			return {
				content: [{ type: "text", text: String(r.content ?? "") }],
				details: r,
				terminate: false,
			};
		},
	};
}

function makeSubmitTool(): AgentTool<any> {
	return {
		name: "submit",
		label: "submit",
		description: "Submit the final answer and finish the task.",
		parameters: {
			type: "object",
			properties: { answer: { type: "string" } },
			required: ["answer"],
		},
		execute: async (_id, params: { answer: string }): Promise<AgentToolResult<any>> => {
			doneSent = true;
			emit("__WMH_DONE__", { answer: params.answer ?? "" });
			return {
				content: [{ type: "text", text: "submitted" }],
				details: { answer: params.answer },
				terminate: true,
			};
		},
	};
}

async function main(): Promise<void> {
	const task = JSON.parse(readFileSync(TASK_FILE, "utf8")) as Task;
	const model: Model<"openai-completions"> = {
		id: AGENT_MODEL,
		name: AGENT_MODEL,
		api: "openai-completions",
		provider: "agent", // non-builtin -> uses model.baseUrl directly (the real provider)
		baseUrl: AGENT_BASE_URL,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 4096,
	};
	const envTools = task.tools.filter((t) => t.name !== "submit");
	const tools: AgentTool<any>[] = [...envTools.map(makeEnvTool), makeSubmitTool()];

	const agent = new Agent({
		initialState: { systemPrompt: task.system ?? "", model, tools },
		getApiKey: () => AGENT_KEY || "x",
	});
	let turns = 0;
	agent.subscribe((event) => {
		if (event.type === "turn_end" || event.type === "message_end") {
			const t = assistantText((event as any).message);
			if (t) lastAssistantText = t;
		}
		if (event.type === "turn_end") {
			turns += 1;
			if (turns >= MAX_TURNS) agent.abort();
		}
	});

	await agent.prompt(task.instruction);
	if (!doneSent) emit("__WMH_DONE__", { answer: lastAssistantText });
	process.stderr.write(`[entry_e2b] turns=${turns} done=${doneSent}\n`);
	process.exit(0);
}

main().catch((e) => {
	process.stderr.write(`[entry_e2b] fatal ${e}\n`);
	emit("__WMH_DONE__", { answer: "", error: String(e) });
	process.exit(1);
});
