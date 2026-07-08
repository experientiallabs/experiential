"""A minimal LLM agent for rollouts: one tool call (or DONE) per turn, JSON-formatted.

This is the reusable counterpart of the throwaway agent inside `wmh demo`: it implements the
`Agent` protocol so scenario verification and research runs can roll real episodes against a
world model without every caller re-writing the same prompt-and-parse loop. It is deliberately
simple — no planning scaffold — because its role is "a competent baseline agent", not SOTA.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.render import render_agent_messages
from wmh.core.types import Action, ActionKind, EnvState, HarnessContext, JsonObject, Step
from wmh.env.episode import DONE_SIGNAL
from wmh.providers.base import Message, Provider

# The reply protocol is separate from the persona so a captured harness system prompt can replace
# the persona while the parseable contract stays intact (`act` parses exactly this shape).
REPLY_PROTOCOL = """Each turn, respond with ONLY a JSON object, no prose around it — one of:
{"tool": "<tool name>", "arguments": {...}}         to act,
{"done": true, "summary": "<what you achieved>"}    when the task is complete or impossible.

Choose tool names and arguments consistent with the environment's responses so far. Work
efficiently: no redundant calls, finish as soon as the task is done."""

AGENT_SYSTEM = f"""You are an agent operating in a tool environment to complete a task.

{REPLY_PROTOCOL}"""

_MAX_HISTORY_CHARS = 500


class _AgentReply(BaseModel):
    """Lenient view of the agent's JSON reply."""

    tool: str | None = None
    arguments: JsonObject = {}
    done: bool = False
    summary: str = ""


class LLMAgent:
    """`Agent`-protocol adapter around a provider: history in, one JSON tool call out.

    Pass `harness` (e.g. `world_model.harness`) to run the agent under the system prompt and tool
    definitions the corpus' real agent had — token-realistic rollouts — with the JSON reply
    protocol appended so replies stay parseable.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        temperature: float = 0.0,
        harness: HarnessContext | None = None,
    ) -> None:
        self._provider = provider
        self._temperature = temperature
        self._system = _agent_system(harness)

    def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action:
        prompt = _render_turn(task, state, history)
        completion = self._provider.complete(
            self._system,
            [Message(role="user", content=prompt)],
            temperature=self._temperature,
            max_tokens=1024,
        )
        raw = extract_json_object(completion.text)
        if raw is not None:
            try:
                reply = _AgentReply.model_validate_json(raw)
            except ValidationError:
                reply = None
            if reply is not None:
                if reply.done or reply.tool is None:
                    return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL)
                return Action(kind=ActionKind.TOOL_CALL, name=reply.tool, arguments=reply.arguments)
        # Unparseable reply: surface it as a message action; the env will answer and the episode
        # continues rather than crashing the batch.
        return Action(kind=ActionKind.MESSAGE, content=completion.text.strip()[:_MAX_HISTORY_CHARS])


def _agent_system(harness: HarnessContext | None) -> str:
    """The captured harness system prompt (+ tools) with the reply protocol, or the baseline."""
    if harness is None or not harness:
        return AGENT_SYSTEM
    realistic, _task = render_agent_messages(harness, "")
    return f"{realistic}\n\n{REPLY_PROTOCOL}"


def _render_turn(task: str | None, state: EnvState, history: list[Step]) -> str:
    lines = [f"TASK: {task or '(none)'}"]
    if state.scratchpad:
        lines.append(f"ENVIRONMENT NOTES: {state.scratchpad}")
    if history:
        lines.append("EPISODE SO FAR:")
        for index, step in enumerate(history):
            action = step.action
            if action.kind is ActionKind.TOOL_CALL:
                call = f"{action.name}({json.dumps(action.arguments, default=str)})"
            else:
                call = f"message: {action.content}"
            observation = step.observation.content[:_MAX_HISTORY_CHARS]
            error_mark = " [ERROR]" if step.observation.is_error else ""
            lines.append(f"{index}. {call} -> {observation}{error_mark}")
    lines.append("Your next move (JSON only):")
    return "\n".join(lines)
