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
 *   4. Register a `submit` tool whose execute() POSTs /done {answer, reason:"submit"} and
 *      terminates the loop (AgentToolResult.terminate = true).
 *   5. agent.prompt(instruction); a turn WITHOUT tool calls is not a completion, so read the
 *      shim's GET /signal (the host's view of the last completion) and nudge, bounded, before
 *      POSTing /done with the classified reason. See runner_termination.ts.
 *
 * Lives next to src/ when materialized on the runner (import paths "./src/agent.ts",
 * "./runner_termination.ts").
 */
import { Agent } from "./src/agent.ts";
import type { AgentTool, AgentToolResult } from "./src/types.ts";
import type { Model } from "@earendil-works/pi-ai";
import {
	classifyEnd,
	newTurnSignal,
	nudgeFor,
	shouldNudge,
	type DoneReason,
} from "./runner_termination.ts";

const SHIM = process.env.PI_SHIM_URL;
if (!SHIM) {
	console.error("PI_SHIM_URL not set");
	process.exit(2);
}
const BASE = SHIM.replace(/\/$/, "");
const DEFAULT_MAX_TURNS = 20;
// Last-resort model context window when /task carries none. The host resolves the REAL served
// window (provider/SDK model info) and reports it as context_window; never assume a size here.
const DEFAULT_CONTEXT_WINDOW = 128000;

interface TaskTool {
	name: string;
	description: string;
	parameters: any;
}
interface Task {
	instruction: string;
	system?: string;
	tools: TaskTool[];
	max_turns?: number;
	max_output_tokens?: number;
	context_window?: number;
}
/** The shim's host-side view of the most recent worker completion. */
interface ShimSignal {
	finish_reason?: string;
	unparsed_tool_calls?: string[];
	provider_error?: string;
	tool_call_turns?: number;
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
async function sendDone(answer: string | null, reason: DoneReason): Promise<void> {
	if (doneSent) return;
	doneSent = true;
	await postJson("/done", { answer, reason });
}

/** Pull the host's view of the last completion into the shared TurnSignal shape. */
async function readSignal(toolCallTurns: number): Promise<ReturnType<typeof newTurnSignal>> {
	const signal = newTurnSignal();
	signal.toolCallTurns = toolCallTurns;
	try {
		const shim = await getJson<ShimSignal>("/signal");
		signal.finishReason = String(shim.finish_reason ?? "");
		signal.unparsedToolCallErrors = Array.isArray(shim.unparsed_tool_calls)
			? shim.unparsed_tool_calls.map((e) => String(e))
			: [];
		signal.providerError = String(shim.provider_error ?? "");
		signal.toolCallTurns = Number.isInteger(shim.tool_call_turns)
			? Number(shim.tool_call_turns)
			: toolCallTurns;
	} catch (e) {
		// An older shim has no /signal endpoint. Classification then degrades to "no_tool_call",
		// which is still honest (it is never reported as a submission).
		console.error(`[entry] /signal unavailable: ${e}`);
	}
	return signal;
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
			await sendDone(params.answer ?? "", "submit");
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
	const configuredMaxTurns = task.max_turns;
	const maxTurns =
		configuredMaxTurns !== undefined &&
		Number.isInteger(configuredMaxTurns) &&
		configuredMaxTurns >= 1
			? configuredMaxTurns
			: DEFAULT_MAX_TURNS;
	const configuredMaxOutputTokens = task.max_output_tokens;
	const maxOutputTokens =
		configuredMaxOutputTokens !== undefined &&
		Number.isInteger(configuredMaxOutputTokens) &&
		configuredMaxOutputTokens >= 1
			? configuredMaxOutputTokens
			: 4096;
	const configuredContextWindow = task.context_window;
	const contextWindow =
		configuredContextWindow !== undefined &&
		Number.isInteger(configuredContextWindow) &&
		configuredContextWindow >= 1024
			? configuredContextWindow
			: DEFAULT_CONTEXT_WINDOW;

	const model: Model<"openai-completions"> = {
		id: "stub-model",
		name: "stub-model",
		api: "openai-completions",
		provider: "shim", // non-builtin provider -> uses model.baseUrl directly
		baseUrl: BASE + "/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow,
		maxTokens: maxOutputTokens,
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

	// Hard turn cap: use the harness document's per-episode value.
	let turnCount = 0;
	let hitTurnCap = false;
	agent.subscribe((event) => {
		if (event.type === "turn_end" || event.type === "message_end") {
			const t = assistantText((event as any).message);
			if (t) lastAssistantText = t;
		}
		if (event.type === "turn_end") {
			turnCount += 1;
			if (turnCount >= maxTurns) {
				hitTurnCap = true;
				agent.abort();
			}
		}
	});

	await agent.prompt(task.instruction);

	// A turn without tool calls is NOT a completion. Nudge (bounded) before ending, then report
	// exactly why this episode stopped so the host never records it as a submission.
	let signal = await readSignal(0);
	let reason: DoneReason = classifyEnd(signal, {
		hitTurnCap,
		agentError: String(agent.state.errorMessage ?? ""),
	});
	let consecutiveNonAction = 1;
	while (!doneSent && shouldNudge(reason, consecutiveNonAction, turnCount, maxTurns)) {
		const before = signal.toolCallTurns;
		await agent.prompt(nudgeFor(reason, signal, maxOutputTokens));
		signal = await readSignal(before);
		consecutiveNonAction = signal.toolCallTurns > before ? 1 : consecutiveNonAction + 1;
		reason = classifyEnd(signal, {
			hitTurnCap,
			agentError: String(agent.state.errorMessage ?? ""),
		});
	}

	// Ensure /done was sent. If pi never called submit, fall back to its last assistant message
	// text (its de-facto answer) rather than reporting empty.
	if (!doneSent) {
		const err = agent.state.errorMessage;
		await sendDone(err ? null : lastAssistantText, reason);
	}

	console.error(
		`[entry] done sent=${doneSent} reason=${reason} turns=${turnCount} err=${agent.state.errorMessage ?? ""}`,
	);
	process.exit(0);
}

main().catch((e) => {
	console.error("[entry] fatal", e);
	process.exit(1);
});
