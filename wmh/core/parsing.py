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
import re

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
    """Lenient view of the world-model JSON contract before normalization.

    The reasoning-mode fields (`reasoning`, `kb_note`, `ground_query` — see
    `wmh.core.render.output_contract`) default to empty so base-contract replies parse unchanged.
    """

    reasoning: str = ""
    output: str = ""
    is_error: bool = False
    state_note: str = ""
    kb_note: str = ""
    ground_query: str = ""
    state_update: str = ""


# The keys that mark a reply as following the observation contract (any one present is enough).
# Used to tell a real — possibly empty — contract response apart from arbitrary JSON that happens
# to validate against `_RawObservation`'s all-defaulted fields. Deliberately ONLY the core keys:
# every complete contract reply (base or reasoning mode) carries `output`/`is_error`, while a
# reasoning-mode superset key alone (e.g. off-contract JSON with a "reasoning" field but no
# "output") must fall through to the plain-text fallback, not become an empty observation.
_CONTRACT_KEYS = frozenset({"output", "is_error", "state_note"})


def parse_observation(text: str) -> Observation:
    """Parse a world-model completion into a structured Observation.

    Prefers the JSON contract ``{"output", "is_error", "state_note"}`` and its reasoning-mode
    superset (``reasoning``/``kb_note``/``ground_query``). ``output`` becomes the observation the
    agent sees; every other populated field is carried in ``metadata`` (``state_note`` feeds the
    session scratchpad, ``kb_note`` the cross-session knowledge base, ``ground_query`` the
    grounder, ``reasoning`` is kept for inspection only). Falls back to treating the whole reply
    as plain observation text when it is not the expected JSON, so an off-contract model still
    yields a usable observation.
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
                for key, value in (
                    ("state_note", parsed.state_note),
                    ("reasoning", parsed.reasoning),
                    ("kb_note", parsed.kb_note),
                    ("ground_query", parsed.ground_query),
                    ("state_update", parsed.state_update),
                ):
                    if value:
                        metadata[key] = value
                return Observation(
                    content=parsed.output, is_error=parsed.is_error, metadata=metadata
                )
    # A contract reply cut off mid-generation is not valid JSON at all: salvage the fields the
    # text already contains rather than surfacing the raw truncated JSON as the observation.
    salvaged = _salvage_truncated_contract(text)
    if salvaged is not None:
        return salvaged
    return Observation(content=text.strip())


def _salvage_truncated_contract(text: str) -> Observation | None:
    """Recover a contract reply whose JSON never closed (token-budget truncation).

    Long deliberations plus long escaped observations can blow the completion budget mid-string;
    without this, the ENTIRE raw contract text (reasoning included) becomes the observation the
    agent sees — observed live as a catastrophic 0.26-fidelity step. Conservative trigger: the
    text must look like a contract object (starts with ``{`` and names an ``"output"`` key) and
    must NOT have parsed as complete JSON (callers try that first). Recovered string fields are
    unescaped up to the truncation point.
    """
    stripped = text.strip()
    if not stripped.startswith("{") or '"output"' not in stripped:
        return None
    output = _string_field_value(stripped, "output")
    if output is None:
        return None
    metadata: JsonObject = {}
    # Recover every metadata-carried contract field the truncated text still contains —
    # dropping state_note/state_update here would silently stall the scratchpad and belief
    # profile for the rest of the session.
    for key in ("reasoning", "state_note", "kb_note", "ground_query", "state_update"):
        value = _string_field_value(stripped, key)
        if value:
            metadata[key] = value
    is_error = re.search(r'"is_error"\s*:\s*true', stripped) is not None
    return Observation(content=output, is_error=is_error, metadata=metadata)


def _string_field_value(text: str, key: str) -> str | None:
    """Extract `key`'s JSON string value from possibly-truncated JSON, unescaping as we go."""
    marker = f'"{key}"'
    at = text.find(marker)
    if at == -1:
        return None
    i = at + len(marker)
    while i < len(text) and text[i] in ": \t\n":
        i += 1
    if i >= len(text) or text[i] != '"':
        return None
    i += 1
    out: list[str] = []
    escaped = False
    while i < len(text):
        ch = text[i]
        if escaped:
            if ch == "u" and i + 4 < len(text):
                # \uXXXX escape: decode the four hex digits (accents/box-drawing chars are
                # common in terminal corpora; dropping the backslash rendered 'u00e9' garbage).
                hex_digits = text[i + 1 : i + 5]
                try:
                    out.append(chr(int(hex_digits, 16)))
                    i += 4
                except ValueError:
                    out.append(ch)
            else:
                out.append(_UNESCAPE.get(ch, ch))
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            break  # properly terminated string
        else:
            out.append(ch)
        i += 1
    return "".join(out)


_UNESCAPE = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


def dumps_observation_contract(observation: Observation) -> str:
    """Render an Observation back into the JSON output contract (used to seed/demo the format)."""
    payload: JsonObject = {"output": observation.content, "is_error": observation.is_error}
    note = observation.metadata.get("state_note")
    if isinstance(note, str) and note:
        payload["state_note"] = note
    return json.dumps(payload)
