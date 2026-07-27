"""A minimal LLM agent for rollouts: one tool call (or DONE) per turn, JSON-formatted.

This is the reusable counterpart of the throwaway agent inside `wmo demo`: it implements the
`Agent` protocol so scenario verification and research runs can roll real episodes against a
world model without every caller re-writing the same prompt-and-parse loop. It is deliberately
simple — no planning scaffold — because its role is "a competent baseline agent", not SOTA.
"""

from __future__ import annotations

import json
import re
from typing import cast

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

# Observation/history truncation. 500 starved tool-heavy environments (tau-bench
# `get_user_details` payloads exceed it several-fold), and an agent that cannot see what it just
# fetched re-fetches it: verbatim re-fetch loops appeared in 12-18 of 30 sim2real episodes.
# Callers with bigger observations raise it further through `history_chars`.
DEFAULT_HISTORY_CHARS = 2000

# Extra attempts when a provider returns a blank completion. Measured on kimi-k2.6, where 24% of
# replies were empty strings: without a retry each blank becomes an empty MESSAGE action that
# burns a step, and the episode loops to its step cap having done nothing. Excluding the affected
# episodes closed 71% (s80) and 100% (25-cohort) of that model's sim-to-real gap, so the blanks
# were the defect, not the model.
_EMPTY_RETRIES = 2

# The head of `tool_name({`, the function-call syntax some models emit instead of the JSON
# envelope. Deliberately not anchored: models fuse the call onto the end of prose.
_CALL_HEAD = re.compile(r"([A-Za-z_]\w*)\s*\(\s*\{")


class _AgentReply(BaseModel):
    """Lenient view of the agent's JSON reply."""

    tool: str | None = None
    arguments: JsonObject = {}
    done: bool = False
    summary: str = ""


def _parse_bare_call(text: str) -> tuple[str, JsonObject] | None:
    """Parse `tool_name({...})` syntax, bare or fused with prose; None when there is none.

    Some models answer with function-call syntax rather than the envelope this agent asks for.
    That reply is a perfectly clear intention to call a tool, and executing it is strictly better
    than the alternative: measured on glm-5.2, 56% of its real tau-bench retail episodes ended at
    step 1-3 because the harness could not read a bare or prose-fused call (mean reward 0.14
    against 0.47 for the episodes that survived). Real tau-bench uses native structured tool
    calling, so the model pays nothing there and the whole gap was ours.

    Returns the FIRST well-formed call, which is sequential execution order when a reply stacks
    several. The closing `)` is required, so an arguments object that merely follows a word is
    not mistaken for a call.
    """
    decoder = json.JSONDecoder()
    for match in _CALL_HEAD.finditer(text):
        start = text.index("{", match.end() - 1)
        try:
            arguments, end = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(arguments, dict) and text[end:].lstrip().startswith(")"):
            return match.group(1), cast("JsonObject", arguments)
    return None


class LLMAgent:
    """`Agent`-protocol adapter around a provider: history in, one JSON tool call out."""

    def __init__(
        self,
        provider: Provider,
        *,
        temperature: float = 0.0,
        tools_hint: str | None = None,
        history_chars: int = DEFAULT_HISTORY_CHARS,
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
        if raw is not None:
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
                # models). Fall through to the message path so the env can answer and the
                # episode recovers.
        # Not the envelope, but possibly still a clear call: models that write `tool_name({...})`
        # land here, and executing that beats answering it with prose (see `_parse_bare_call`).
        bare = _parse_bare_call(completion.text)
        if bare is not None:
            return Action(kind=ActionKind.TOOL_CALL, name=bare[0], arguments=bare[1])
        # Unparseable reply: surface it as a message action; the env will answer and the episode
        # continues rather than crashing the batch.
        return Action(
            kind=ActionKind.MESSAGE, content=completion.text.strip()[: self._history_chars]
        )

    def _complete_retrying_empty(self, prompt: str) -> Completion:
        """One turn's completion, re-asking while the provider answers with nothing.

        A blank reply is not an answer, but the parse path has to treat it as one, so it becomes
        an empty MESSAGE action that spends a step and teaches the agent nothing. Re-asking is
        cheap next to losing the episode (see `_EMPTY_RETRIES` for what that cost). After the
        last retry the blank is returned and handled as before, so a provider that only ever
        answers empty still terminates.

        LIMITATION, stated because the retry depends on it and it is not verified: the retry
        sends the IDENTICAL prompt at the agent's temperature, which is 0.0 by default, so it
        only helps if the blanks are provider-side nondeterminism rather than a deterministic
        response to this prompt. The 24% blank rate that motivated this was measured on the
        original transcripts, and the retry's effect on that rate was NOT re-measured; the
        diagnosis that excluding blank-affected episodes closed the model's gap is what is
        evidenced, not that re-asking recovers them. If a future capture shows blanks surviving
        all three attempts at the same rate, the fix is a nonzero retry temperature or a
        provider-side report, not more retries.
        """
        completion = self._complete_once(prompt)
        for _retry in range(_EMPTY_RETRIES):
            if completion.text.strip():
                break
            completion = self._complete_once(prompt)
        return completion

    def _complete_once(self, prompt: str) -> Completion:
        return self._provider.complete(
            self._system,
            [Message(role="user", content=prompt)],
            temperature=self._temperature,
            max_tokens=1024,
        )


def _render_turn(
    task: str | None,
    state: EnvState,
    history: list[Step],
    history_chars: int = DEFAULT_HISTORY_CHARS,
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
