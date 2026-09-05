"""Anthropic builder tool-selection and strict-schema preflight; the rest of the
Messages-family builders are exercised in streaming_requests_test.py."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.messages_payloads import anthropic_messages_stream_payload

_LOOKUP = GatewayToolDefinition(
    name="lookup",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
)


def _tool_request(**overrides: object) -> GatewayRequest:
    """Build one streaming Messages request carrying the lookup tool."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="weather in Paris"),),
        tools=(_LOOKUP,),
        stream=True,
        include_usage=True,
    )
    return request.model_copy(update=dict(overrides))


def _capability(exc_info: pytest.ExceptionInfo[ProviderCapabilityError]) -> str:
    """Return the capability literal of one raised preflight rejection."""
    return exc_info.value.capability


@pytest.mark.parametrize("choice", ("required", GatewayNamedToolChoice(name="lookup")))
def test_fable_5_1_declines_a_forced_tool_choice_before_dispatch(
    choice: str | GatewayNamedToolChoice,
) -> None:
    """The release rejects ``any``/``tool`` by name (live 2026-09-05, with or
    without a thinking config), so the rung declines at build time and route
    admission can prefer another rung or relax to ``auto`` with disclosure."""
    with pytest.raises(ProviderCapabilityError) as raised:
        anthropic_messages_stream_payload(
            "claude-fable-5-1",
            _tool_request(tool_choice=choice),
            supports_reasoning=True,
            reasoning_effort="medium",
        )
    assert _capability(raised) == "forced_tool_choice"
    # auto and none stay servable on the same model.
    for open_choice in ("auto", "none", None):
        payload = anthropic_messages_stream_payload(
            "claude-fable-5-1", _tool_request(tool_choice=open_choice)
        )
        assert payload.get("tool_choice") == (
            {"type": open_choice} if open_choice is not None else None
        )


def test_other_releases_force_tools_verbatim_under_adaptive_thinking() -> None:
    """opus-5 (and every non-5.1 release probed live) carries ``any``/``tool``
    verbatim, including beside the adaptive thinking config the rung emits."""
    payload = anthropic_messages_stream_payload(
        "claude-opus-5",
        _tool_request(tool_choice="required", reasoning_effort="high"),
        supports_reasoning=True,
    )
    assert payload["tool_choice"] == {"type": "any"}
    assert payload["thinking"] == {"type": "adaptive"}
    named = anthropic_messages_stream_payload(
        "claude-sonnet-4-6",
        _tool_request(tool_choice=GatewayNamedToolChoice(name="lookup")),
    )
    assert named["tool_choice"] == {"type": "tool", "name": "lookup"}


def test_budgeted_thinking_cannot_ride_beside_a_forced_choice() -> None:
    """Every model answers a forced choice plus ``thinking: enabled`` with
    "Thinking may not be enabled when tool_choice forces tool use" (live
    2026-09-05), whether the caller sent the budget or the rung derived one
    from an effort on a budgeted-only model."""
    with pytest.raises(ProviderCapabilityError) as caller_budget:
        anthropic_messages_stream_payload(
            "claude-sonnet-4-6",
            _tool_request(
                tool_choice="required",
                provider_thinking_config={"type": "enabled", "budget_tokens": 1024},
                maximum_output_tokens=4096,
            ),
            supports_reasoning=True,
        )
    assert _capability(caller_budget) == "forced_tool_choice"
    with pytest.raises(ProviderCapabilityError) as derived_budget:
        anthropic_messages_stream_payload(
            "claude-haiku-4-5",
            _tool_request(tool_choice="required", reasoning_effort="medium"),
            supports_reasoning=True,
        )
    assert _capability(derived_budget) == "forced_tool_choice"
    # A disabled config forces fine on a model that accepts it.
    payload = anthropic_messages_stream_payload(
        "claude-sonnet-4-6",
        _tool_request(tool_choice="required", provider_thinking_config={"type": "disabled"}),
        supports_reasoning=True,
    )
    assert payload["tool_choice"] == {"type": "any"}
    assert payload["thinking"] == {"type": "disabled"}


_ARRAY_WITH_MAX_ITEMS: JsonObject = {
    "type": "object",
    "properties": {"cities": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
    "required": ["cities"],
    "additionalProperties": False,
}


def test_strict_tools_with_unsupported_keywords_decline_as_strict_tools() -> None:
    """A strict schema the strict validator cannot compile (``maxItems``, live
    2026-09-05 on every current model) is declined as ``strict_tools`` with the
    schema untouched, so admission drops only ``strict`` when no rung can honor it."""
    strict = GatewayToolDefinition(name="list", parameters=_ARRAY_WITH_MAX_ITEMS, strict=True)
    with pytest.raises(ProviderCapabilityError) as raised:
        anthropic_messages_stream_payload("claude-fable-5-1", _tool_request(tools=(strict,)))
    assert _capability(raised) == "strict_tools"

    # The same schema without strict forwards verbatim, and a supported strict
    # schema keeps its strict flag on the wire.
    loose = strict.model_copy(update={"strict": False})
    payload = anthropic_messages_stream_payload("claude-fable-5-1", _tool_request(tools=(loose,)))
    tools = payload["tools"]
    assert isinstance(tools, list)
    assert tools[0] == {"name": "list", "input_schema": _ARRAY_WITH_MAX_ITEMS}
    clean = _LOOKUP.model_copy(update={"strict": True})
    payload = anthropic_messages_stream_payload("claude-fable-5-1", _tool_request(tools=(clean,)))
    tools = payload["tools"]
    assert isinstance(tools, list)
    assert tools[0]["strict"] is True


def test_system_prompts_with_no_readable_text_are_omitted() -> None:
    """The wire rejects an empty system text block and an all-whitespace
    system prompt ("system: text content blocks must be non-empty" / "must
    contain non-whitespace text", live 2026-09-05); a prompt with nothing to
    read is omitted, and a readable one keeps its exact bytes."""

    def payload_for(*system: GatewayMessage) -> JsonObject:
        """Build the Anthropic payload for ``system`` turns plus one user turn."""
        request = GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(*system, GatewayMessage(role="user", content="hi")),
            stream=True,
            include_usage=True,
        )
        return anthropic_messages_stream_payload("claude-fable-5-1", request)

    assert "system" not in payload_for(GatewayMessage(role="system", content=""))
    assert "system" not in payload_for(GatewayMessage(role="system", content="  \n"))
    assert "system" not in payload_for(
        GatewayMessage(role="system", content="  "), GatewayMessage(role="system", content="")
    )
    # Whitespace beside real instructions is accepted, so the joined bytes stay exact.
    assert (
        payload_for(
            GatewayMessage(role="system", content="  "),
            GatewayMessage(role="system", content="rules"),
        )["system"]
        == "  \n\nrules"
    )
    # A cache-marked run drops its empty blocks with the breakpoint migrated.
    marked = payload_for(
        GatewayMessage(
            role="system",
            content="rules",
            provider_text_blocks=(
                {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "rules"},
            ),
        )
    )
    assert marked["system"] == [
        {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
    ]
    assert "system" not in payload_for(
        GatewayMessage(
            role="system",
            content="",
            provider_text_blocks=(
                {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
            ),
        )
    )


def test_a_marker_on_a_collapsed_assistant_turn_moves_to_a_neighboring_block() -> None:
    """A whitespace-only assistant run that carried a cache breakpoint has no
    block of its own once it dispatches as an empty array, so the marker lands
    on the closest retained block before it, or on the first block after it
    when nothing precedes, and never on a thinking block."""
    marked_blank: tuple[JsonObject, ...] = (
        {"type": "text", "text": " \n", "cache_control": {"type": "ephemeral"}},
    )

    def payload_for(*messages: GatewayMessage) -> list[JsonObject]:
        """Return the emitted Anthropic messages for ``messages``."""
        request = GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=messages,
            stream=True,
            include_usage=True,
        )
        built = anthropic_messages_stream_payload("claude-fable-5-1", request)["messages"]
        assert isinstance(built, list)
        return built

    before = payload_for(
        GatewayMessage(role="user", content="hello"),
        GatewayMessage(role="assistant", content=" \n", provider_text_blocks=marked_blank),
        GatewayMessage(role="user", content="continue"),
    )
    assert before[0]["content"] == [
        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
    ]
    assert before[1] == {"role": "assistant", "content": []}
    assert before[2]["content"] == [{"type": "text", "text": "continue"}]

    after = payload_for(
        GatewayMessage(role="assistant", content=" \n", provider_text_blocks=marked_blank),
        GatewayMessage(role="user", content="continue"),
    )
    assert after[0] == {"role": "assistant", "content": []}
    assert after[1]["content"] == [
        {"type": "text", "text": "continue", "cache_control": {"type": "ephemeral"}}
    ]

    # Consecutive collapsed turns sit on one boundary: both markers land on
    # the same earlier block, none defers forward to a later, larger prefix.
    consecutive = payload_for(
        GatewayMessage(role="user", content="hello"),
        GatewayMessage(role="assistant", content=" \n", provider_text_blocks=marked_blank),
        GatewayMessage(role="assistant", content=" \n", provider_text_blocks=marked_blank),
        GatewayMessage(role="user", content="continue"),
    )
    assert consecutive == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}],
        },
        {"role": "assistant", "content": []},
        {"role": "user", "content": [{"type": "text", "text": "continue"}]},
    ]

    # A block that already carries a marker keeps its own (one per boundary),
    # and a whitespace run beside a tool call is not collapsed at all, so its
    # marker stays where the caller put it.
    already = payload_for(
        GatewayMessage(
            role="user",
            content="hello",
            provider_text_blocks=(
                {
                    "type": "text",
                    "text": "hello",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
            ),
        ),
        GatewayMessage(role="assistant", content=" \n", provider_text_blocks=marked_blank),
        GatewayMessage(role="user", content="continue"),
    )
    assert already[0]["content"] == [
        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]
    from exp.runtime.gateway.contracts import ToolCall

    beside_tool = payload_for(
        GatewayMessage(role="user", content="hello"),
        GatewayMessage(
            role="assistant",
            content=" \n",
            provider_text_blocks=marked_blank,
            tool_calls=(
                ToolCall(call_id="call-1", name="lookup", arguments={}, raw_arguments="{}"),
            ),
        ),
        GatewayMessage(role="tool", content="found", tool_call_id="call-1"),
    )
    assert beside_tool[1]["content"] == [
        {"type": "text", "text": " \n", "cache_control": {"type": "ephemeral"}},
        {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {}},
    ]
