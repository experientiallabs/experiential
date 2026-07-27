"""A minimal LLM agent for rollouts: one tool call (or DONE) per turn, JSON-formatted.

This is the reusable counterpart of the throwaway agent inside `wmo demo`: it implements the
`Agent` protocol so scenario verification and research runs can roll real episodes against a
world model without every caller re-writing the same prompt-and-parse loop. It is deliberately
simple — no planning scaffold — because its role is "a competent baseline agent", not SOTA.
"""

from __future__ import annotations

import json
import re
from typing import NamedTuple

from pydantic import BaseModel, ValidationError

from wmo.core.parsing import extract_json_object
from wmo.core.types import Action, ActionKind, EnvState, JsonObject, Step
from wmo.env.episode import DONE_SIGNAL
from wmo.providers.base import Completion, Message, Provider

AGENT_SYSTEM = """You are an agent operating in a tool environment to complete a task.

Each turn, respond with ONLY a JSON object, no prose around it — one of:
{"tool": "<tool name>", "arguments": {...}}         to act,
{"done": true, "summary": "<what you achieved>"}    when the task is complete or impossible.

Choose tool names and arguments consistent with the environment's responses so far. Work
efficiently: no redundant calls, finish as soon as the task is done."""

# Observation/history truncation. 500 chars starved tool-heavy environments (a tau-bench
# `get_user_details` payload is several times that), so the agent never saw the answer it had
# just fetched and re-fetched it verbatim until the step budget died. Callers with even bigger
# observations can raise it per instance via `history_chars`.
_MAX_HISTORY_CHARS = 2000

# How many extra completions to buy when a provider returns blank text. Measured at 24% of
# replies for one pool model; without the retry every blank burns a step as an empty message.
_EMPTY_RETRIES = 2


class _AgentReply(BaseModel):
    """Lenient view of the agent's JSON reply."""

    tool: str | None = None
    arguments: JsonObject = {}
    done: bool = False
    summary: str = ""


class _BareCall(NamedTuple):
    """A `tool_name({...})` call recovered from prose, and where its argument object starts."""

    name: str
    arguments: JsonObject
    arguments_at: int


_CALL_HEAD = re.compile(r"([A-Za-z_]\w*)\s*\(\s*\{")


def _parse_bare_call(text: str) -> _BareCall | None:
    """Parse `tool_name({...})` syntax, bare or fused with prose.

    Some models emit function-call syntax instead of the JSON envelope the system prompt asks
    for. Without this the turn is wasted: the call never executes and the argument object gets
    read as an envelope or echoed back as a message. Returns the FIRST well-formed call, which
    is the one a sequential executor would have run when a reply stacks several.

    Args:
        text: The raw completion text.

    Returns:
        The recovered call, or None when the text holds no well-formed one.
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
            return _BareCall(name=match.group(1), arguments=arguments, arguments_at=start)
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
        bare = _parse_bare_call(completion.text)
        raw = extract_json_object(completion.text)
        # An object that a bare call opens is that call's ARGUMENTS, never the envelope, even
        # when its keys happen to collide with the envelope's ("tool", "done"). Reading
        # `book_trip({"tool": "train"})` as an envelope would run a tool named "train".
        if raw is not None and not (
            bare is not None and _text_starts_object(completion.text, bare)
        ):
            try:
                reply = _AgentReply.model_validate_json(raw)
            except ValidationError:
                reply = None
            if reply is not None:
                if reply.done:
                    return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL)
                if reply.tool is not None:
                    return Action(
                        kind=ActionKind.TOOL_CALL, name=reply.tool, arguments=reply.arguments
                    )
                # A JSON object with neither `done` nor `tool` is NOT a finish signal: models
                # that write prose-style calls (`get_user(...)` around a bare argument object)
                # land here, and treating it as done ended their episodes unexecuted at step 1
                # and scored them 0 (measured: 34% of glm-5.2 wm episodes, 0% for other pool
                # models). Fall through so the call is recovered or the env answers.
        if bare is not None:
            return Action(kind=ActionKind.TOOL_CALL, name=bare.name, arguments=bare.arguments)
        # Unparseable reply: surface it as a message action; the env will answer and the episode
        # continues rather than crashing the batch.
        return Action(
            kind=ActionKind.MESSAGE, content=completion.text.strip()[: self._history_chars]
        )

    def _complete_retrying_empty(self, prompt: str) -> Completion:
        """Buy up to `_EMPTY_RETRIES` extra completions while the provider returns blank text.

        A blank reply is indistinguishable from an unparseable one downstream, so it costs a
        step doing nothing; retrying converts an intermittent provider hiccup into a real turn
        instead of a slow death by step budget. Returns the last completion either way, so a
        persistently blank provider still yields an (empty) message action rather than raising.
        """
        completion = self._provider.complete(
            self._system,
            [Message(role="user", content=prompt)],
            temperature=self._temperature,
            max_tokens=1024,
        )
        for _attempt in range(_EMPTY_RETRIES):
            if completion.text.strip():
                break
            completion = self._provider.complete(
                self._system,
                [Message(role="user", content=prompt)],
                temperature=self._temperature,
                max_tokens=1024,
            )
        return completion


def _text_starts_object(text: str, bare: _BareCall) -> bool:
    """True when the first JSON object in `text` is the one `bare`'s call opens."""
    return text.find("{") == bare.arguments_at


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
