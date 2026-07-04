"""Robust parsing of model completions into structured values.

Two concerns live here because both the serving engine and the optimizer need them, and `wmh.core`
has no dependencies (so neither imports the other):

- `extract_json_object`: pull the first complete JSON object out of a noisy LLM reply.
- `parse_observation`: turn a world-model completion into a structured `Observation`.

The world-model output contract (see `wmh.core.render.build_env_prompt`) asks the model to reply
with a JSON object ``{"output": str, "is_error": bool, "state_note": str}``. `parse_observation`
is lenient: a reply that is not JSON is treated as a plain-text observation, so a model that ignores
the contract still produces a usable (non-error) observation rather than crashing the step.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from wmh.core.types import JsonObject, Observation


def extract_json_object(text: str) -> str | None:
    """Return the first complete JSON object substring in `text`, or None if there is none.

    Scans from the first ``{`` to its balanced closing ``}``, tracking string literals and escapes.
    This tolerates ```json fences, surrounding prose, nested objects, and multiple objects (the
    first is returned) — cases a greedy/lazy regex gets wrong.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class _RawObservation(BaseModel):
    """Lenient view of the world-model JSON contract before normalization."""

    output: str = ""
    is_error: bool = False
    state_note: str = ""


# The keys that mark a reply as following the observation contract (any one present is enough). Used
# to tell a real — possibly empty — contract response apart from arbitrary JSON that happens to
# validate against `_RawObservation`'s all-defaulted fields.
_CONTRACT_KEYS = frozenset({"output", "is_error", "state_note"})


def parse_observation(text: str) -> Observation:
    """Parse a world-model completion into a structured Observation.

    Prefers the JSON contract ``{"output", "is_error", "state_note"}``; the ``state_note`` (a
    one-line fact the env wants to remember) is carried in ``metadata`` for the session scratchpad.
    Falls back to treating the whole reply as plain observation text when it is not the expected
    JSON, so an off-contract model still yields a usable observation.
    """
    raw = extract_json_object(text)
    if raw is not None:
        try:
            obj: object = json.loads(raw)
        except json.JSONDecodeError:
            obj = None
        # Recognize the contract by the PRESENCE of its keys, not by truthy values.
        # `_RawObservation` defaults every field, so arbitrary JSON like `{"foo": 1}` would validate
        # to an all-empty observation; requiring a contract key keeps that falling through to raw
        # text. But a legitimate silent success `{"output": "", "is_error": false, ...}` (many shell
        # writes/redirects print nothing) MUST be honored as an empty observation, not re-serialized
        # as visible JSON text — closed-loop rollouts would otherwise show spurious output.
        if isinstance(obj, dict) and _CONTRACT_KEYS.intersection(obj):
            try:
                parsed = _RawObservation.model_validate(obj)
            except ValidationError:
                parsed = None
            if parsed is not None:
                metadata: JsonObject = {}
                if parsed.state_note:
                    metadata["state_note"] = parsed.state_note
                return Observation(
                    content=parsed.output, is_error=parsed.is_error, metadata=metadata
                )
    return Observation(content=text.strip())


def dumps_observation_contract(observation: Observation) -> str:
    """Render an Observation back into the JSON output contract (used to seed/demo the format)."""
    payload: JsonObject = {"output": observation.content, "is_error": observation.is_error}
    note = observation.metadata.get("state_note")
    if isinstance(note, str) and note:
        payload["state_note"] = note
    return json.dumps(payload)
