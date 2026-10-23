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


def test_a_namespaced_custom_grammar_also_prefers_the_native_rung() -> None:
    """A grammar inside a namespace is translated off the native wire too, so it narrows."""
    request = decode_responses(
        {
            "model": "gpt-5-nano",
            "input": "hi",
            "tools": [
                {
                    "type": "namespace",
                    "name": "editor",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "apply_patch",
                            "format": {
                                "type": "grammar",
                                "syntax": "lark",
                                "definition": "start: x",
                            },
                        }
                    ],
                }
            ],
        }
    ).request

    assert compatible_generation_parameter_profile_indexes(
        (_AZURE_CHAT, _OPENAI_RESPONSES), request
    ) == (1,)


def _bedrock_claude(*, reasoning: bool) -> GatewayWireProfile:
    """A Bedrock Claude Opus 4.8 rung, with or without the catalog's reasoning claim."""
    return GatewayWireProfile(
        dialect="bedrock_converse_stream",
        url="https://bedrock-runtime.us-east-1.amazonaws.test/model/x/converse-stream",
        model_id="us.anthropic.claude-opus-4-8",
        supports_reasoning=reasoning,
        reasoning_wire_format="anthropic_adaptive" if reasoning else "none",
        signs_request_body=True,
    )


def test_effort_request_keeps_a_reasoning_bedrock_claude_rung() -> None:
    """A Chat effort narrows past a plain relay to the reasoning-declared Bedrock rung.

    The claude-opus-4.8 chain leads with a compatible relay that declares no
    reasoning; before Bedrock carried adaptive thinking the effort narrowed
    past Bedrock too, leaving the direct Anthropic lane as the only reasoning
    rung of the model.
    """
    relay = GatewayWireProfile(
        dialect="openai_compatible", url="https://relay.test/v1", model_id="pa/claude-opus-4-8"
    )
    request = decode_chat(
        {
            "model": "claude-opus-4.8",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
        }
    ).request
    profiles = (relay, _bedrock_claude(reasoning=False), _bedrock_claude(reasoning=True))
    assert compatible_generation_parameter_profile_indexes(profiles, request) == (2,)
