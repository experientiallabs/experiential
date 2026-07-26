"""A minimal LLM agent for rollouts: one tool call (or DONE) per turn, JSON-formatted.

This is the reusable counterpart of the throwaway agent inside `wmh demo`: it implements the
`Agent` protocol so scenario verification and research runs can roll real episodes against a
world model without every caller re-writing the same prompt-and-parse loop. It is deliberately
simple — no planning scaffold — because its role is "a competent baseline agent", not SOTA.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.types import Action, ActionKind, EnvState, JsonObject, Step
from wmh.env.episode import DONE_SIGNAL
from wmh.providers.base import Message, Provider

AGENT_SYSTEM = """You are an agent operating in a tool environment to complete a task.

Each turn, respond with ONLY a JSON object, no prose around it — one of:
{"tool": "<tool name>", "arguments": {...}}         to act,
{"done": true, "summary": "<what you achieved>"}    when the task is complete or impossible.

Choose tool names and arguments consistent with the environment's responses so far. Work
efficiently: no redundant calls, finish as soon as the task is done."""

# Observation/history truncation. 500 chars starved tool-heavy environments (tau-bench
# get_user_details payloads exceed it several-fold), producing verbatim re-fetch loops;
# callers with big observations can raise it further via `history_chars`.
_MAX_HISTORY_CHARS = 2000


class _AgentReply(BaseModel):
    """Lenient view of the agent's JSON reply."""

    tool: str | None = None
    arguments: JsonObject = {}
    done: bool = False
    summary: str = ""


_EMPTY_RETRIES = 2
_CALL_HEAD = re.compile(r"([A-Za-z_]\w*)\s*\(\s*\{")


def _parse_bare_call(text: str) -> tuple[str, JsonObject] | None:
    """Parse `tool_name({...})` syntax, bare or fused with prose.

    Some models emit function-call syntax instead of the JSON envelope; a strict parser
    would extract the inner arguments object and misread the turn entirely. Returns the
    FIRST well-formed call (sequential execution order when a reply stacks several).
    """
    decoder = json.JSONDecoder()
    for match in _CALL_HEAD.finditer(text):
        start = text.index("{", match.end() - 1)
        try:
            arguments, end = decoder.raw_decode(text, start)
        except ValueError:
            continue
        rest = text[end:].lstrip()
        if isinstance(arguments, dict) and rest.startswith(")"):
            return match.group(1), arguments
    return None


class LLMAgent:
    """`Agent`-protocol adapter around a provider: history in, one JSON tool call out."""

    def __init__(
        self,
        provider: Provider,
        *,
        temperature: float = 0.0,
        tools_hint: str | None = None,
        history_chars: int = _MAX_HISTORY_CHARS,
    ) -> None:
        self._provider = provider
        self._temperature = temperature
        self._history_chars = history_chars
        # Corpus-derived tool surface (names + argument keys observed in the traces). Without
        # it, capable models honestly refuse to invent tools while weaker ones hallucinate
        # them, and closed-loop rewards measure affordance-guessing instead of capability.
        self._system = AGENT_SYSTEM if not tools_hint else f"{AGENT_SYSTEM}\n\n{tools_hint}"

    def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action:
        prompt = _render_turn(task, state, history, self._history_chars)
        completion = self._complete_retrying_empty(prompt)
        raw = extract_json_object(completion.text)
        # Only a JSON object that speaks the envelope's language is the envelope; a bare
        # arguments object (the inner {...} of function-call syntax) must fall through,
        # not read as "done" (that mistake silently ended episodes at step 1).
        if raw is not None and _speaks_envelope(raw):
            try:
                reply = _AgentReply.model_validate_json(raw)
            except ValidationError:
                reply = None
            if reply is not None:
                if reply.done or reply.tool is None:
                    return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL)
                return Action(kind=ActionKind.TOOL_CALL, name=reply.tool, arguments=reply.arguments)
        bare = _parse_bare_call(completion.text)
        if bare is not None:
            return Action(kind=ActionKind.TOOL_CALL, name=bare[0], arguments=bare[1])
        # Unparseable reply: surface it as a message action; the env will answer and the episode
        # continues rather than crashing the batch.
        content = completion.text.strip()[: self._history_chars]
        return Action(kind=ActionKind.MESSAGE, content=content)

    def _complete_retrying_empty(self, prompt: str):  # noqa: ANN202 - provider Completion
        """Retry blank completions: an empty reply otherwise burns a step doing nothing.

        Some providers intermittently return empty text (observed at 24% of replies for
        one pool model); without the retry each blank consumes a step as an empty message
        until the budget dies.
        """
        completion = None
        for _attempt in range(1 + _EMPTY_RETRIES):
            completion = self._provider.complete(
                self._system,
                [Message(role="user", content=prompt)],
                temperature=self._temperature,
                max_tokens=1024,
            )
            if completion.text.strip():
                break
        return completion


def _speaks_envelope(raw: str) -> bool:
    try:
        data = json.loads(raw)
    except ValueError:
        return False
    return isinstance(data, dict) and ("tool" in data or "done" in data)


def _render_turn(
    task: str | None, state: EnvState, history: list[Step], history_chars: int = _MAX_HISTORY_CHARS
) -> str:
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
            observation = step.observation.content[:history_chars]
            error_mark = " [ERROR]" if step.observation.is_error else ""
            lines.append(f"{index}. {call} -> {observation}{error_mark}")
    lines.append("Your next move (JSON only):")
    return "\n".join(lines)
