"""Cloud Code Assist request envelope for Antigravity plan dispatch.

An Antigravity plan reaches Gemini models through Google's Cloud Code Assist backend
(``v1internal:streamGenerateContent``), which takes the ordinary Gemini ``generateContent`` body
nested under ``request`` beside the account's project and the plan's own model id, and answers
with each Gemini response nested under ``response``. This module owns both directions of that
wrapping plus the request shaping the backend requires beyond the public Gemini API:

* ``safetySettings`` is not accepted and is dropped.
* Tool schemas are read as OpenAPI-style ``parameters``: ``parametersJsonSchema`` moves there,
  local ``$ref`` pointers are inlined, ``const`` becomes a one-value ``enum``, and the keywords
  the backend refuses (``$schema``, ``$defs``, ``definitions``, ``default``, ``examples``) are
  removed.
* Function names must be at most 64 characters, start with a letter or underscore, and contain
  no ``/``. A name outside that shape is refused before dispatch, never renamed, because a
  renamed tool call would come back under a name the caller never declared.
* ``maxOutputTokens`` must exceed ``thinkingConfig.thinkingBudget``; a fixed budget at or above
  the caller's ceiling is lowered to fit under it.

``thoughtSignature`` parts in the history are part of the Gemini body and pass through
unchanged, so the backend sees the signatures it issued.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.errors import ProviderCapabilityError, ProviderResponseError

CODE_ASSIST_BASE_URL = "https://daily-cloudcode-pa.googleapis.com"
"""The consumer Cloud Code Assist origin; plan traffic is sent nowhere else."""

CODE_ASSIST_STREAM_PATH = "v1internal:streamGenerateContent?alt=sse"
CODE_ASSIST_GENERATE_PATH = "v1internal:generateContent"
CODE_ASSIST_MODELS_PATH = "v1internal:fetchAvailableModels"

CODE_ASSIST_USER_AGENT_FIELD = "antigravity"
"""The envelope's ``userAgent`` field value (distinct from the HTTP header)."""

CODE_ASSIST_REQUEST_TYPE = "agent"

TOOL_NAME_CAPABILITY = "code_assist_tool_name"
"""The capability a refused tool name names on its error."""

MAXIMUM_TOOL_NAME_CHARACTERS = 64

_TOOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,63}")
_DROPPED_SCHEMA_KEYS = frozenset({"$schema", "$defs", "definitions", "default", "examples", "$id"})
_LOCAL_REFERENCE = re.compile(r"#/(?:\$defs|definitions)/([^/]+)")
# Inlining is bounded so a self-referential schema cannot expand without limit; past the bound
# the reference collapses to an unconstrained object.
_MAXIMUM_REFERENCE_DEPTH = 8


@dataclass(frozen=True, slots=True)
class CodeAssistEnvelope:
    """The per-connection facts a Code Assist dispatch wraps its Gemini body with.

    Attributes:
        project: The account's Cloud Code Assist project id.
        model: The plan's own model id the catalog model translates to.
    """

    project: str
    model: str


def wrap_code_assist_request(
    payload: JsonObject,
    envelope: CodeAssistEnvelope,
    *,
    session_id: str | None = None,
    request_id: str | None = None,
) -> JsonObject:
    """Nest one Gemini ``generateContent`` body in the Code Assist envelope.

    Args:
        payload: The complete Gemini body the dialect built for this rung.
        envelope: The connection's project and the translated plan model id.
        session_id: Stable conversation id, when one is derivable; a fresh one otherwise.
        request_id: Explicit request id for deterministic tests; a fresh one otherwise.

    Returns:
        The envelope, with the shaped Gemini body under ``request``.

    Raises:
        ProviderCapabilityError: A declared tool name does not fit the backend's name rules.
    """
    request = shape_code_assist_body(payload)
    request["sessionId"] = session_id or f"session-{uuid4()}"
    return {
        "project": envelope.project,
        "model": envelope.model,
        "userAgent": CODE_ASSIST_USER_AGENT_FIELD,
        "requestType": CODE_ASSIST_REQUEST_TYPE,
        "requestId": request_id or f"agent-{uuid4()}",
        "request": request,
    }


def shape_code_assist_body(payload: JsonObject) -> JsonObject:
    """Return a copy of a Gemini body shaped to the Code Assist backend's rules.

    Args:
        payload: The Gemini ``generateContent`` body.

    Returns:
        The body without ``safetySettings``, with sanitized tool schemas and a thinking budget
        below the output ceiling.

    Raises:
        ProviderCapabilityError: A declared tool name does not fit the backend's name rules.
    """
    body = deepcopy(payload)
    body.pop("safetySettings", None)
    body.pop("model", None)
    tools = body.get("tools")
    if isinstance(tools, list):
        body["tools"] = [_shape_tool(tool) for tool in tools]
    generation = body.get("generationConfig")
    if isinstance(generation, dict):
        body["generationConfig"] = _fit_thinking_budget(generation)
    return body


def unwrap_code_assist_response(payload: JsonObject) -> JsonObject:
    """Return the Gemini response nested in one Code Assist answer.

    Args:
        payload: A decoded Code Assist ``generateContent`` answer.

    Returns:
        The inner Gemini response object.

    Raises:
        ProviderResponseError: The answer carries no ``response`` object.
    """
    inner = payload.get("response")
    if not isinstance(inner, dict):
        raise ProviderResponseError("Code Assist answer has no response object")
    return inner


def _shape_tool(tool: JsonValue) -> JsonValue:
    """Sanitize every function declaration of one Gemini tool entry."""
    if not isinstance(tool, dict):
        return tool
    declarations = tool.get("functionDeclarations")
    if not isinstance(declarations, list):
        return tool
    return {**tool, "functionDeclarations": [_shape_declaration(d) for d in declarations]}


def _shape_declaration(declaration: JsonValue) -> JsonValue:
    """Check one declaration's name and move its schema onto sanitized ``parameters``.

    Raises:
        ProviderCapabilityError: The declaration's name does not fit the backend's rules.
    """
    if not isinstance(declaration, dict):
        return declaration
    name = declaration.get("name")
    if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
        raise ProviderCapabilityError(
            capability=TOOL_NAME_CAPABILITY,
            detail=(
                f"The tool name {str(name)[:80]!r} does not fit this plan's function-name rules: "
                f"at most {MAXIMUM_TOOL_NAME_CHARACTERS} characters, starting with a letter or "
                "underscore, using only letters, digits, '_', '.', ':' and '-'. Rename the tool."
            ),
        )
    shaped = dict(declaration)
    schema = shaped.pop("parametersJsonSchema", None)
    if schema is None:
        schema = shaped.get("parameters")
    if isinstance(schema, dict):
        shaped["parameters"] = sanitize_tool_schema(schema)
    response_schema = shaped.pop("responseJsonSchema", None)
    if isinstance(response_schema, dict):
        shaped["response"] = sanitize_tool_schema(response_schema)
    return shaped


def sanitize_tool_schema(schema: JsonObject) -> JsonObject:
    """Return a tool schema without the keywords the Code Assist backend refuses.

    Local ``#/$defs/...`` and ``#/definitions/...`` references are inlined from the schema's own
    definitions; ``const`` becomes a one-value ``enum``; ``$schema``, ``$defs``, ``definitions``,
    ``$id``, ``default`` and ``examples`` are removed. Property NAMES are never touched, so a
    property literally called ``default`` survives.

    Args:
        schema: One JSON Schema object.

    Returns:
        The sanitized schema.
    """
    definitions: dict[str, JsonValue] = {}
    for key in ("$defs", "definitions"):
        raw = schema.get(key)
        if isinstance(raw, dict):
            definitions.update(raw)
    cleaned = _sanitize(schema, definitions, depth=0)
    return cleaned if isinstance(cleaned, dict) else {"type": "object"}


def _sanitize(value: JsonValue, definitions: dict[str, JsonValue], *, depth: int) -> JsonValue:
    """Sanitize one schema node recursively.

    Args:
        value: The schema node.
        definitions: Local definitions a ``$ref`` may name.
        depth: Current reference-inlining depth.

    Returns:
        The sanitized node.
    """
    if isinstance(value, list):
        return [_sanitize(item, definitions, depth=depth) for item in value]
    if not isinstance(value, dict):
        return value
    reference = value.get("$ref")
    if isinstance(reference, str):
        match = _LOCAL_REFERENCE.fullmatch(reference)
        target = definitions.get(match.group(1)) if match is not None else None
        if not isinstance(target, dict) or depth >= _MAXIMUM_REFERENCE_DEPTH:
            # An unresolvable or runaway reference collapses to an open object, keeping only
            # the description the caller wrote beside it.
            fallback: JsonObject = {"type": "object"}
            if isinstance(value.get("description"), str):
                fallback["description"] = value["description"]
            return fallback
        merged = {**target, **{k: v for k, v in value.items() if k != "$ref"}}
        return _sanitize(merged, definitions, depth=depth + 1)
    result: JsonObject = {}
    for key, item in value.items():
        if key in _DROPPED_SCHEMA_KEYS:
            continue
        if key == "const":
            if "enum" not in value:
                result["enum"] = [item]
            continue
        if key == "properties" and isinstance(item, dict):
            # Keys here are property names, not keywords: keep every one.
            result[key] = {
                name: _sanitize(child, definitions, depth=depth) for name, child in item.items()
            }
            continue
        result[key] = _sanitize(item, definitions, depth=depth)
    return result


def _fit_thinking_budget(generation: JsonObject) -> JsonObject:
    """Lower a fixed thinking budget that does not fit under the output ceiling.

    The backend refuses a request whose ``maxOutputTokens`` does not exceed the thinking
    budget. The caller's ceiling is authority, so the budget moves, never the ceiling. A
    dynamic budget (``-1``) or an absent ceiling is left as it is.

    Args:
        generation: The body's ``generationConfig``.

    Returns:
        The configuration with a budget strictly below the ceiling.
    """
    ceiling = generation.get("maxOutputTokens")
    thinking = generation.get("thinkingConfig")
    if isinstance(ceiling, bool) or not isinstance(ceiling, int) or not isinstance(thinking, dict):
        return generation
    budget = thinking.get("thinkingBudget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < ceiling or budget < 0:
        return generation
    lowered = dict(thinking)
    lowered["thinkingBudget"] = max(ceiling - 1, 0)
    if lowered["thinkingBudget"] == 0:
        lowered.pop("includeThoughts", None)
    return {**generation, "thinkingConfig": lowered}
