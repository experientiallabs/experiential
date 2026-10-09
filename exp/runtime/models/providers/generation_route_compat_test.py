"""Tests for per-rung generation compatibility ranking across a gateway route."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.generation_route_compat import (
    compatible_generation_parameter_profile_indexes,
)
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses

_AZURE_CHAT = GatewayWireProfile(
    dialect="openai_compatible", url="https://azure.test/openai/v1", model_id="gpt-5-nano"
)
_OPENROUTER = GatewayWireProfile(
    dialect="openai_compatible", url="https://openrouter.test/api/v1", model_id="openai/gpt-5-nano"
)
_OPENAI_RESPONSES = GatewayWireProfile(
    dialect="openai_responses",
    url="https://api.openai.test/v1",
    model_id="gpt-5-nano",
    supports_reasoning=True,
    reasoning_wire_format="openai_responses",
)
_CUSTOM_TOOL: JsonObject = {
    "type": "custom",
    "custom": {
        "name": "code_exec",
        "format": {"type": "grammar", "grammar": {"syntax": "lark", "definition": "start: x"}},
    },
}


def _chat(tools: list[JsonObject]) -> GatewayRequest:
    """One Chat request declaring ``tools``."""
    return decode_chat(
        {
            "model": "gpt-5-nano",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": list(tools),
        }
    ).request


def test_a_native_responses_rung_outranks_translation_for_a_custom_grammar() -> None:
    """Only a native Responses rung enforces the grammar, so it serves a grammar tool first."""
    profiles = (_AZURE_CHAT, _OPENROUTER, _OPENAI_RESPONSES)

    assert compatible_generation_parameter_profile_indexes(profiles, _chat([_CUSTOM_TOOL])) == (2,)


def test_a_custom_tool_without_a_grammar_keeps_every_rung() -> None:
    """Free-text custom input translates losslessly, so the route keeps its fallbacks."""
    profiles = (_AZURE_CHAT, _OPENROUTER, _OPENAI_RESPONSES)
    text_tool: JsonObject = {
        "type": "custom",
        "custom": {"name": "code_exec", "format": {"type": "text"}},
    }

    assert compatible_generation_parameter_profile_indexes(profiles, _chat([text_tool])) == (
        0,
        1,
        2,
    )


def test_function_tools_alone_keep_every_rung() -> None:
    """Without a custom tool the route's ranking is unchanged."""
    profiles = (_AZURE_CHAT, _OPENROUTER, _OPENAI_RESPONSES)
    function: JsonObject = {"type": "function", "function": {"name": "f", "parameters": {}}}

    assert compatible_generation_parameter_profile_indexes(profiles, _chat([function])) == (
        0,
        1,
        2,
    )


def test_a_route_without_a_native_rung_serves_custom_tools_by_translation() -> None:
    """No native rung leaves the whole route eligible; route shaping then translates."""
    profiles = (_AZURE_CHAT, _OPENROUTER)

    assert compatible_generation_parameter_profile_indexes(profiles, _chat([_CUSTOM_TOOL])) == (
        0,
        1,
    )


def test_summary_and_custom_preferences_combine() -> None:
    """A rung carrying both a summary and custom tools natively ranks above all others."""
    request = decode_responses(
        {
            "model": "gpt-5-nano",
            "input": "hi",
            "reasoning": {"summary": "auto"},
            "tools": [
                {
                    "type": "custom",
                    "name": "code_exec",
                    "format": {"type": "grammar", "syntax": "regex", "definition": "^[0-9]+$"},
                }
            ],
        }
    ).request

    assert compatible_generation_parameter_profile_indexes(
        (_AZURE_CHAT, _OPENAI_RESPONSES), request
    ) == (1,)
