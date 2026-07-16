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

from pydantic import BaseModel, ValidationError, field_validator

from wmh.core.types import JsonObject, JsonValue, Observation


def accepted_confidence(value: float | int | str | bool) -> float | None:
    """The one definition of a usable stated confidence: a finite number in [0, 1], else None.

    Gates and calibration both consume this, so the acceptance rule must not fork: booleans are
    not confidences (JSON `true` is not 1.0), NaN/inf are garbage, and OUT-OF-RANGE numerics
    degrade to "not stated" rather than clamping — a model answering 85 (percent) or 7 (out of
    10) has violated the 0.0-1.0 contract, and clamping such a reply to 1.0 would record maximal
    certainty on exactly the steps where the model is off the rails. Missing conservatively
    gates as LOW; malformed must never gate as certain.
    """
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not (0.0 <= parsed <= 1.0):  # also rejects NaN (all comparisons false) and +/-inf
        return None
    return parsed


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
    # Verbalized confidence (WS-A6): None when the model didn't state one. Lenient like the rest
    # of the contract — an off-contract value degrades to "no stated confidence", never a crash.
    confidence: float | None = None
    confidence_why: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _lenient_confidence(cls, value: JsonValue) -> float | None:
        """Coerce the raw JSON field through `accepted_confidence` (off-contract -> None)."""
        if isinstance(value, int | float | str):
            return accepted_confidence(value)
        return None


# The keys that mark a reply as following the observation contract (any one present is enough).
# Used to tell a real — possibly empty — contract response apart from arbitrary JSON that happens
# to validate against `_RawObservation`'s all-defaulted fields. Deliberately ONLY the core keys:
# every complete contract reply (base or reasoning mode) carries `output`/`is_error`, while a
# reasoning-mode superset key alone (e.g. off-contract JSON with a "reasoning" field but no
# "output") must fall through to the plain-text fallback, not become an empty observation.
# Confidence-mode keys are deliberately excluded too: an arbitrary API payload with its own
# "confidence" field must not be mistaken for a contract reply.
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
                    ("confidence_why", parsed.confidence_why),
                ):
                    if value:
                        metadata[key] = value
                # Separate from the truthiness loop: a stated confidence of 0.0 must survive.
                if parsed.confidence is not None:
                    metadata["confidence"] = parsed.confidence
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
    salvage_keys = (
        "reasoning",
        "state_note",
        "kb_note",
        "ground_query",
        "state_update",
        "confidence_why",
    )
    for key in salvage_keys:
        value = _string_field_value(stripped, key)
        if value:
            metadata[key] = value
    # Salvage a stated confidence too: truncation correlates with HARD steps, so silently
    # dropping their confidences would bias any calibration analysis toward the easy ones.
    # Same acceptance rule as the validator — the two paths must not fork.
    match = _CONFIDENCE_VALUE.search(stripped)
    if match is not None:
        confidence = accepted_confidence(match.group(1))
        if confidence is not None:
            metadata["confidence"] = confidence
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

# The numeric confidence value in possibly-truncated contract text (salvage path only; complete
# JSON goes through `_RawObservation`).
_CONFIDENCE_VALUE = re.compile(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)')


def dumps_observation_contract(observation: Observation) -> str:
    """Render an Observation back into the JSON output contract (used to seed/demo the format).

    Carries the confidence fields when present (key order mirroring the contract:
    justification before the number, both after `is_error`) — the verify pass embeds this as
    the draft, and a draft missing the field the contract demands invites the reviser to drop
    it too, thinning stated confidence exactly on the verified (low-confidence) population.
    """
    payload: JsonObject = {"output": observation.content, "is_error": observation.is_error}
    why = observation.metadata.get("confidence_why")
    if isinstance(why, str) and why:
        payload["confidence_why"] = why
    confidence = observation.metadata.get("confidence")
    if isinstance(confidence, int | float) and not isinstance(confidence, bool):
        payload["confidence"] = float(confidence)
    note = observation.metadata.get("state_note")
    if isinstance(note, str) and note:
        payload["state_note"] = note
    return json.dumps(payload)
