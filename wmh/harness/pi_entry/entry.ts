/**
 * Headless pi-agent entrypoint driven by the Python world-model shim.
 *
 * Run: PI_SHIM_URL=http://127.0.0.1:$PORT node --experimental-strip-types entry.ts
 *
 * Flow:
 *   1. GET  $PI_SHIM_URL/task              -> {instruction, system, tools[]}
 *   2. Build a pi Agent whose Model.baseUrl = $PI_SHIM_URL + "/v1" and
 *      api = "openai-completions" (so streamSimple hits the shim's SSE endpoint).
 *   3. Register each task tool as an AgentTool whose execute() POSTs /tool.
 *   4. Register a `submit` tool whose execute() POSTs /done {answer} and
 *      terminates the loop (AgentToolResult.terminate = true).
 *   5. agent.prompt(instruction); after the loop, POST /done if not already sent.
 *
 * Lives next to src/ when materialized on the runner (import path "./src/agent.ts").
 */
import { Agent } from "./src/agent.ts";
import type { AgentTool, AgentToolResult } from "./src/types.ts";
import type { Model } from "@earendil-works/pi-ai";

const SHIM = process.env.PI_SHIM_URL;
if (!SHIM) {
	console.error("PI_SHIM_URL not set");
	process.exit(2);
}
const BASE = SHIM.replace(/\/$/, "");
const MAX_TURNS = 20;

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

async function getJson<T>(path: string): Promise<T> {
	const res = await fetch(BASE + path);
	if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
	return (await res.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
	const res = await fetch(BASE + path, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
	});
	if (!res.ok) throw new Error(`POST ${path} -> ${res.status}`);
	return (await res.json()) as T;
}

let doneSent = false;
async function sendDone(answer: string | null): Promise<void> {
	if (doneSent) return;
	doneSent = true;
	await postJson("/done", { answer });
}

// pi often ends by writing its final answer as a normal assistant message rather than calling
// `submit`. Capture the latest assistant text so we can use it as the answer if the loop exits
// without a submit call (otherwise the answer would be empty).
let lastAssistantText = "";
function assistantText(msg: any): string {
	if (!msg || msg.role !== "assistant" || !Array.isArray(msg.content)) return "";
	return msg.content
		.filter((c: any) => c?.type === "text")
		.map((c: any) => String(c.text ?? ""))
		.join("")
		.trim();
}

function makeShimTool(t: TaskTool): AgentTool<any> {
	return {
		name: t.name,
		label: t.name,
		description: t.description,
		parameters: t.parameters,
		execute: async (_id, params): Promise<AgentToolResult<any>> => {
			const r = await postJson<{ content: string; is_error?: boolean }>("/tool", {
				name: t.name,
				arguments: params,
			});
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
			await sendDone(params.answer ?? "");
			return {
				content: [{ type: "text", text: "submitted" }],
				details: { answer: params.answer },
				terminate: true, // stop the agent loop after this tool batch
			};
		},
	};
}

async function main(): Promise<void> {
	const task = await getJson<Task>("/task");

	const model: Model<"openai-completions"> = {
		id: "stub-model",
		name: "stub-model",
		api: "openai-completions",
		provider: "shim", // non-builtin provider -> uses model.baseUrl directly
		baseUrl: BASE + "/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 4096,
	};

	// `submit` is provided by entry.ts (it drives /done + loop termination); drop any
	// task-supplied `submit` so the tool list pi sends the model has unique names.
	const envTools = task.tools.filter((t) => t.name !== "submit");
	const tools: AgentTool<any>[] = [...envTools.map(makeShimTool), makeSubmitTool()];

	const agent = new Agent({
		initialState: {
			systemPrompt: task.system ?? "",
			model,
			tools,
		},
		// apiKey passed via stream options; shim ignores it but SDK requires non-empty.
		getApiKey: () => "x",
	});

	// Hard turn cap: abort after MAX_TURNS assistant turns to avoid runaway loops.
	let turnCount = 0;
	agent.subscribe((event) => {
		if (event.type === "turn_end" || event.type === "message_end") {
			const t = assistantText((event as any).message);
			if (t) lastAssistantText = t;
		}
		if (event.type === "turn_end") {
			turnCount += 1;
			if (turnCount >= MAX_TURNS) agent.abort();
		}
	});

	await agent.prompt(task.instruction);

	// After the loop, ensure /done was sent. If pi never called submit, fall back to its last
	// assistant message text (its de-facto answer) rather than reporting empty.
	if (!doneSent) {
		const err = agent.state.errorMessage;
		await sendDone(err ? null : lastAssistantText);
	}

	console.error(`[entry] done sent=${doneSent} turns=${turnCount} err=${agent.state.errorMessage ?? ""}`);
	process.exit(0);
}

main().catch((e) => {
	console.error("[entry] fatal", e);
	process.exit(1);
});
