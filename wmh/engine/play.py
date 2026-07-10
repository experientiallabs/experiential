"""`wmh play`: a human drives the agent inside the reconstructed environment.

This is the interactive sibling of `wmh demo`. Instead of replaying a recorded trajectory,
a *person* types actions — `tool_name {json args}` or a free-text message — and the world model
returns the observation, advancing the session (history + scratchpad "database") exactly as a real
agent would experience it.

The engine here is UI-agnostic: it parses a typed line into an `Action` and steps the world model,
returning the observation alongside the exact env prompt that produced it. The REPL loop, prompts,
and rendering live in the CLI (`wmh.cli.ui`); this module is what the CLI and its tests call.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, JsonValue

from wmh.core.parsing import extract_json_object
from wmh.core.types import Action, ActionKind, JsonObject, Observation
from wmh.engine.world_model import WorldModel


class PlayTurn(BaseModel):
    """The outcome of one human turn: the action taken, the observation, and the env prompt sent."""

    action: Action
    observation: Observation
    env_prompt: str


def parse_action(line: str) -> Action:
    """Parse a typed REPL line into an `Action`.

    Grammar (forgiving, human-first):
      - `get_user {"id": "u1"}`  -> tool call `get_user` with JSON arguments
      - `list_flights`           -> tool call `list_flights` with no arguments
      - `say hello there`        -> free-text message "hello there"
      - anything else            -> a free-text message (so the env can react to plain prose)

    A bare first token is treated as a tool name when it looks like an identifier; otherwise the
    whole line is a message. The `say ` prefix forces a message even when it looks tool-like.
    """
    text = line.strip()
    if not text:
        raise ValueError("empty action")

    if text.startswith("say "):
        return Action(kind=ActionKind.MESSAGE, content=text[4:].strip())

    head, _, rest = text.partition(" ")
    rest = rest.strip()
    # A tool call is a bare identifier (`list_flights`) or `name {json args}`. A trailing tail not
    # starting with a JSON bracket means prose (e.g. "what is the weather?") -> treat as a message.
    if _looks_like_tool_name(head) and (not rest or rest.startswith(("{", "["))):
        return Action(kind=ActionKind.TOOL_CALL, name=head, arguments=_parse_arguments(rest))
    return Action(kind=ActionKind.MESSAGE, content=text)


def play_turn(world_model: WorldModel, session_id: str, action: Action) -> PlayTurn:
    """Render the env prompt for `action`, step the world model, and return the full turn."""
    env_prompt = world_model.render_step_prompt(session_id, action)
    observation = world_model.step(session_id, action)
    return PlayTurn(action=action, observation=observation, env_prompt=env_prompt)


def _looks_like_tool_name(token: str) -> bool:
    """A tool name is an ASCII identifier-ish token: [A-Za-z][A-Za-z0-9._-]*.

    ASCII-only on purpose: `str.isalpha()`/`isalnum()` accept Unicode letters/digits, so without
    this a non-ASCII first word (e.g. "café ..." or "日本 ...") would be misread as a tool call
    instead of a prose message.
    """
    if not token or not ("a" <= token[0].lower() <= "z"):
        return False
    return all(c.isascii() and (c.isalnum() or c in "._-") for c in token)


def _parse_arguments(rest: str) -> JsonObject:
    """Parse the argument tail into a JSON object; empty tail -> no arguments."""
    if not rest:
        return {}
    raw = extract_json_object(rest)
    if raw is None:
        raise ValueError(
            f"could not read tool arguments from {rest!r}; pass a JSON object like "
            '{"id": "u1"} (or omit it for no arguments)'
        )
    parsed: JsonValue = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError('tool arguments must be a JSON object, e.g. {"id": "u1"}')
    return parsed
