/**
 * Why a pi episode finished, and what to do about it before giving up.
 *
 * pi's `agent.prompt()` resolves as soon as an assistant turn carries no tool calls. That single
 * event covers four completely different things: the model called `submit`, it emitted prose, its
 * tool call was cut off at the output-token cap, or the renderer could not parse the tool call it
 * did emit. The episode runners used to report all four as one `done` frame, which the host mapped
 * to `submitted`, so every one of them scored as a clean completion with reward 0.
 *
 * This module is the shared classifier + nudge policy the three episode runners
 * (`runner_stdio.ts`, `runner_service.ts`, `entry.ts`) drive:
 *
 *   - `TurnSignal` is updated by each runner's LLM bridge from the host's completion frame, so the
 *     runner can see the finish_reason, whether tool calls came back, and any tool-call parse
 *     errors the host's renderer reported (`wmh_unparsed_tool_calls`).
 *   - `classifyEnd` turns that plus the runner's own flags into a `DoneReason`.
 *   - `nudgeFor` is the observation fed back to the model instead of ending the episode, modeled
 *     on the reference terminus-2 agent's behavior (report the parser's complaint, tell the model
 *     to re-issue in smaller chunks, and ask it to either act or submit).
 *   - `shouldNudge` bounds that to MAX_NONACTION_TURNS consecutive non-action turns.
 *
 * `DoneReason` values are the wire vocabulary `wmh/harness/runner_link.py` maps onto distinct
 * `StopReason`s, and MAX_NONACTION_TURNS mirrors `wmh.harness.runtime.MAX_NONACTION_TURNS`.
 */

/** The `done` frame's `reason`: exactly why this episode stopped. */
export type DoneReason =
	| "submit"
	| "no_tool_call"
	| "output_truncated"
	| "unparsed_tool_call"
	| "provider_error"
	| "max_turns";

/** Consecutive non-action turns a runner nudges through before reporting `done`. */
export const MAX_NONACTION_TURNS = 3;

/** What the last host completion frame carried, as the bridge observed it. */
export interface TurnSignal {
	/** finish_reason of the most recent completion ("length" means the output cap was hit). */
	finishReason: string;
	/** Tool-call parse errors the host's renderer reported for the most recent completion. */
	unparsedToolCallErrors: string[];
	/** The most recent host-side worker error (context overflow, outage), or "". */
	providerError: string;
	/** Cumulative count of completions that carried at least one tool call. */
	toolCallTurns: number;
}

/** A fresh signal, before any completion has come back. */
export function newTurnSignal(): TurnSignal {
	return { finishReason: "", unparsedToolCallErrors: [], providerError: "", toolCallTurns: 0 };
}

/**
 * Record one host completion frame into `signal`.
 *
 * Called by each runner's LLM bridge with the raw `llm_request` reply, so classification sees the
 * host's own view of the turn rather than re-deriving it from pi's message log.
 */
export function observeCompletion(signal: TurnSignal, reply: Record<string, any>): void {
	if (reply.error) {
		signal.providerError = String(reply.error);
		return;
	}
	const choice = reply.completion?.choices?.[0] ?? {};
	const message = choice.message ?? {};
	signal.providerError = "";
	signal.finishReason = String(choice.finish_reason ?? "stop");
	const unparsed = choice.wmh_unparsed_tool_calls;
	signal.unparsedToolCallErrors = Array.isArray(unparsed) ? unparsed.map((e: unknown) => String(e)) : [];
	if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
		signal.toolCallTurns += 1;
	}
}

/** Runner-owned flags `classifyEnd` needs beyond the last completion. */
export interface EndContext {
	/** The runner aborted the agent because the turn cap was reached. */
	hitTurnCap: boolean;
	/** pi's own terminal error message, if any (`agent.state.errorMessage`). */
	agentError: string;
}

/**
 * Why `agent.prompt()` returned, for an episode where `submit` was NOT called.
 *
 * Order matters: a dead provider explains everything downstream of it, an explicit parse error is
 * more specific than "no tool call", and truncation at the cap is more specific still.
 */
export function classifyEnd(signal: TurnSignal, context: EndContext): DoneReason {
	if (signal.providerError) return "provider_error";
	if (signal.unparsedToolCallErrors.length > 0) return "unparsed_tool_call";
	if (signal.finishReason === "length") return "output_truncated";
	if (context.hitTurnCap) return "max_turns";
	if (context.agentError) return "provider_error";
	return "no_tool_call";
}

/**
 * Whether the runner should nudge instead of reporting `done`.
 *
 * A provider that is failing every call and a turn cap that has already fired are terminal: more
 * prompts would only re-pay for the same failure. Everything else gets up to MAX_NONACTION_TURNS
 * consecutive attempts to act or submit.
 */
export function shouldNudge(
	reason: DoneReason,
	consecutiveNonActionTurns: number,
	turns: number,
	maxTurns: number,
): boolean {
	if (reason === "provider_error" || reason === "max_turns") return false;
	if (consecutiveNonActionTurns >= MAX_NONACTION_TURNS) return false;
	return turns < maxTurns;
}

const ACT_OR_SUBMIT =
	"Continue the task: either call exactly one tool now, or call `submit` with your final answer " +
	"if the task is already complete. Do not reply with prose alone.";

/** The observation fed back to the model in place of ending the episode. */
export function nudgeFor(reason: DoneReason, signal: TurnSignal, maxOutputTokens: number): string {
	if (reason === "output_truncated") {
		return (
			`[ERROR] NONE of the actions you just requested were performed: your reply exceeded ` +
			`${maxOutputTokens} output tokens and was cut off mid-emission. Re-issue the request, ` +
			`breaking it into chunks each of which is well under ${maxOutputTokens} tokens. ` +
			ACT_OR_SUBMIT
		);
	}
	if (reason === "unparsed_tool_call") {
		const warnings = signal.unparsedToolCallErrors.join("; ");
		return (
			`[ERROR] your tool call could not be parsed, so NOTHING was executed. Parser ` +
			`warnings from your last reply: ${warnings}. Emit the call again in exactly the ` +
			`documented format, closing every block you open. ` +
			ACT_OR_SUBMIT
		);
	}
	return `[ERROR] your last reply contained no tool call, so nothing happened. ${ACT_OR_SUBMIT}`;
}
