"""Tests for the Cloud Code Assist envelope and its request shaping."""

from __future__ import annotations

import json

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.code_assist_envelope import (
    TOOL_NAME_CAPABILITY,
    CodeAssistEnvelope,
    sanitize_tool_schema,
    shape_code_assist_body,
    unwrap_code_assist_response,
    wrap_code_assist_request,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError, ProviderResponseError

_ENVELOPE = CodeAssistEnvelope(project="proud-sprite-a1b2", model="gemini-3-flash")


def _body(**extra: object) -> JsonObject:
    """Return a minimal Gemini body with ``extra`` top-level fields."""
    return json.loads(
        json.dumps({"contents": [{"role": "user", "parts": [{"text": "hi"}]}], **extra})
    )


def _get(value: JsonValue, *path: str | int) -> JsonValue:
    """Walk ``path`` through nested JSON objects and arrays."""
    for key in path:
        if isinstance(key, int):
            assert isinstance(value, list)
            value = value[key]
        else:
            assert isinstance(value, dict)
            value = value[key]
    return value


def test_the_envelope_nests_the_body_beside_project_model_and_ids() -> None:
    """The Gemini body rides under ``request`` with the session; ids are fresh when absent."""
    wrapped = wrap_code_assist_request(
        _body(systemInstruction={"parts": [{"text": "be brief"}]}),
        _ENVELOPE,
        session_id="conversation-1",
        request_id="agent-fixed",
    )
    assert wrapped == {
        "project": "proud-sprite-a1b2",
        "model": "gemini-3-flash",
        "userAgent": "antigravity",
        "requestType": "agent",
        "requestId": "agent-fixed",
        "request": {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "systemInstruction": {"parts": [{"text": "be brief"}]},
            "sessionId": "conversation-1",
        },
    }
    first = wrap_code_assist_request(_body(), _ENVELOPE)
    second = wrap_code_assist_request(_body(), _ENVELOPE)
    assert first["requestId"] != second["requestId"]
    assert str(first["requestId"]).startswith("agent-")
    assert _get(first, "request", "sessionId") != _get(second, "request", "sessionId")


def test_safety_settings_are_dropped_and_the_input_is_not_mutated() -> None:
    """``safetySettings`` never reaches the backend; the caller's payload stays intact."""
    body = _body(safetySettings=[{"category": "HARM_CATEGORY_HATE_SPEECH"}])
    shaped = shape_code_assist_body(body)
    assert "safetySettings" not in shaped
    assert "safetySettings" in body


def test_thought_signatures_in_history_pass_through() -> None:
    """Replayed signed parts are part of the body and are not altered."""
    contents = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "sig=="}],
        },
    ]
    assert shape_code_assist_body({"contents": contents})["contents"] == contents


def test_tool_schemas_move_to_parameters_and_lose_refused_keywords() -> None:
    """``parametersJsonSchema`` becomes sanitized ``parameters``; property names survive."""
    schema: JsonObject = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "mode": {"const": "fast", "description": "m"},
            "default": {"type": "string", "default": "x", "examples": ["y"]},
            "target": {"$ref": "#/$defs/Target"},
        },
        "required": ["mode"],
        "$defs": {"Target": {"type": "object", "properties": {"id": {"type": "string"}}}},
    }
    shaped = shape_code_assist_body(
        _body(tools=[{"functionDeclarations": [{"name": "run", "parametersJsonSchema": schema}]}])
    )
    declaration = _get(shaped, "tools", 0, "functionDeclarations", 0)
    assert isinstance(declaration, dict)
    assert "parametersJsonSchema" not in declaration
    assert _get(declaration, "parameters") == {
        "type": "object",
        "properties": {
            "mode": {"enum": ["fast"], "description": "m"},
            "default": {"type": "string"},
            "target": {"type": "object", "properties": {"id": {"type": "string"}}},
        },
        "required": ["mode"],
    }


def test_unresolvable_and_recursive_references_collapse_to_open_objects() -> None:
    """A foreign or runaway ``$ref`` never expands without bound."""
    assert sanitize_tool_schema(
        {"type": "object", "properties": {"x": {"$ref": "https://e.test/s", "description": "d"}}}
    ) == {"type": "object", "properties": {"x": {"type": "object", "description": "d"}}}
    recursive = sanitize_tool_schema(
        {
            "$ref": "#/definitions/Node",
            "definitions": {
                "Node": {"type": "object", "properties": {"next": {"$ref": "#/definitions/Node"}}}
            },
        }
    )
    depth = 0
    node: JsonValue = recursive
    while isinstance(node, dict) and "properties" in node:
        node = _get(node, "properties", "next")
        depth += 1
    assert 0 < depth <= 10
    assert node == {"type": "object"}


@pytest.mark.parametrize("name", ["mcp/fs_read", "1tool", "t" * 65, ""])
def test_a_tool_name_outside_the_backends_rules_is_refused_not_renamed(name: str) -> None:
    """A renamed call would come back under a name the caller never declared."""
    with pytest.raises(ProviderCapabilityError) as raised:
        shape_code_assist_body(_body(tools=[{"functionDeclarations": [{"name": name}]}]))
    assert raised.value.capability == TOOL_NAME_CAPABILITY


def test_accepted_tool_names_pass() -> None:
    """Letters, digits and ``_.:-`` up to 64 characters are kept unchanged."""
    for name in ["mcp__fs__read", "_private", "a.b:c-d", "t" * 64]:
        shaped = shape_code_assist_body(_body(tools=[{"functionDeclarations": [{"name": name}]}]))
        assert _get(shaped, "tools", 0, "functionDeclarations", 0, "name") == name


@pytest.mark.parametrize(
    ("ceiling", "budget", "expected"),
    [(1_000, 4_096, 999), (1_000, 1_000, 999), (8_000, 4_096, 4_096), (1_000, -1, -1)],
)
def test_a_fixed_thinking_budget_fits_under_the_output_ceiling(
    ceiling: int, budget: int, expected: int
) -> None:
    """The caller's ceiling is authority: only the budget moves, and dynamic stays dynamic."""
    shaped = shape_code_assist_body(
        _body(
            generationConfig={
                "maxOutputTokens": ceiling,
                "thinkingConfig": {"thinkingBudget": budget, "includeThoughts": True},
            }
        )
    )
    assert _get(shaped, "generationConfig", "thinkingConfig", "thinkingBudget") == expected
    assert _get(shaped, "generationConfig", "thinkingConfig", "includeThoughts") is True


def test_a_ceiling_of_one_turns_thinking_off() -> None:
    """A zero budget drops ``includeThoughts``, which the backend refuses without thinking."""
    shaped = shape_code_assist_body(
        _body(
            generationConfig={
                "maxOutputTokens": 1,
                "thinkingConfig": {"thinkingBudget": 64, "includeThoughts": True},
            }
        )
    )
    assert _get(shaped, "generationConfig", "thinkingConfig") == {"thinkingBudget": 0}


def test_answers_unwrap_and_a_missing_response_is_malformed() -> None:
    """The Gemini response is the ``response`` object; anything else is refused."""
    inner: JsonObject = {"candidates": [], "modelVersion": "gemini-3-flash"}
    assert unwrap_code_assist_response({"response": inner, "traceId": "t"}) == inner
    with pytest.raises(ProviderResponseError):
        unwrap_code_assist_response({"traceId": "t"})
