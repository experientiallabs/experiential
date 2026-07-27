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
    """A `tool_name({...})` call recovered from prose."""

    name: str
    arguments: JsonObject


# `tool_name({` with no space before the paren: real call syntax never has one, while prose
# ("use the search tool ({...})") always does, and allowing it made the reply's own envelope
# parse as a call to a tool named "tool". The lookbehind stops a hyphenated or dotted name from
# matching only its last segment, which turned `get-user-details({...})` into a call to
# `details`; an unknown-but-whole name is a tool error the env can report honestly.
_CALL_HEAD = re.compile(r"(?<![\w.\-])([A-Za-z_][\w\-]*)\(\s*\{")


def _parse_bare_call(text: str) -> _BareCall | None:
    """Parse `tool_name({...})` syntax, bare or fused with prose.

    Some models emit function-call syntax instead of the JSON envelope the system prompt asks
    for. Without this the turn is wasted: the call never executes and the argument object gets
    echoed back as a message. Returns the FIRST well-formed call, which is the one a sequential
    executor would have run when a reply stacks several.

    Args:
        text: The raw completion text.

    Returns:
        The recovered call, or None when the text holds no well-formed one.
    """
    decoder = json.JSONDecoder()
    for match in _CALL_HEAD.finditer(text):
        start = match.end() - 1
        try:
            arguments, end = decoder.raw_decode(text, start)
        except ValueError:
            continue
        rest = text[end:].lstrip()
        if isinstance(arguments, dict) and rest.startswith(")"):
            return _BareCall(name=match.group(1), arguments=arguments)
    return None


def _parse_envelope(text: str) -> _AgentReply | None:
    """The first object in `text` that speaks the envelope's language, or None.

    Scans EVERY balanced object rather than only the first, because a reasoning model states its
    conclusion after its deliberation: the leading object is often an example or a rejected
    hypothesis, and the envelope it actually chose comes later in the reply.
    """
    offset = 0
    while offset < len(text):
        relative = text[offset:].find("{")
        if relative == -1:
            return None
        start = offset + relative
        raw = extract_json_object(text[start:])
        if raw is None:
            return None
        offset = start + len(raw)
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        # An object with neither key is not the envelope: a bare argument payload validates
        # against `_AgentReply`'s all-optional fields and would read as a silent finish.
        if not (isinstance(data, dict) and ("tool" in data or "done" in data)):
            continue
        try:
            return _AgentReply.model_validate(data)
        except ValidationError:
            continue
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
        empty_retries: int = _EMPTY_RETRIES,
    ) -> None:
        self._provider = provider
        self._temperature = temperature
        self._history_chars = history_chars
        # Callers measuring per-call latency can set this to 0: each retry is one more timed
        # provider call at the metering boundary (see `wmo.optimize.report`).
        self._empty_retries = empty_retries
        # Corpus-derived tool surface (names + argument keys observed in the traces). Without
        # it, capable models honestly refuse to invent tools while weaker ones hallucinate
        # them, and closed-loop rewards measure affordance-guessing instead of capability.
        self._system = AGENT_SYSTEM if not tools_hint else f"{AGENT_SYSTEM}\n\n{tools_hint}"

    def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action:
        prompt = _render_turn(task, state, history, self._history_chars)
        completion = self._complete_retrying_empty(prompt)
        # Precedence: an envelope anywhere in the reply is the model's DECISION and always wins.
        # A call written in prose is only ever a fallback, because prose that mentions a call is
        # routinely deliberation about one ("the schema is update_order({...}), not applicable
        # here") rather than an instruction to run it. Letting such a mention outrank a later
        # envelope executed a hypothesis the model had just rejected, and turned
        # `finish({"done": true})` into a tool call that never terminated the episode.
        reply = _parse_envelope(completion.text)
        if reply is not None:
            if reply.done:
                return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL)
            if reply.tool is not None:
                return Action(kind=ActionKind.TOOL_CALL, name=reply.tool, arguments=reply.arguments)
        # No envelope. A JSON object with neither `done` nor `tool` is NOT a finish signal:
        # models that write prose-style calls (`get_user(...)` around a bare argument object)
        # land here, and treating it as done ended their episodes unexecuted at step 1 and
        # scored them 0 (measured: 34% of glm-5.2 wm episodes, 0% for other pool models).
        bare = _parse_bare_call(completion.text)
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
        for _attempt in range(self._empty_retries):
            if completion.text.strip():
                break
            completion = self._provider.complete(
                self._system,
                [Message(role="user", content=prompt)],
                temperature=self._temperature,
                max_tokens=1024,
            )
        return completion


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
