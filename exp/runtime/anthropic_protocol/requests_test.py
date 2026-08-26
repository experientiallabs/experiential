"""Round-trip and rejection tests for the Anthropic Messages decoder."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayNamedToolChoice
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


def _body(**overrides: JsonValue) -> JsonObject:
    """Return one minimal valid Messages body with overrides applied."""
    payload: JsonObject = {
        "model": "coding",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(overrides)
    return payload


def test_decode_full_request_is_lossless() -> None:
    """Every supported field lands on the canonical request."""
    decoded = decode_messages(
        _body(
            system=[{"type": "text", "text": "be terse"}, {"type": "text", "text": "and kind"}],
            temperature=0.5,
            top_p=0.9,
            stop_sequences=["STOP", "STOP", "END"],
            stream=True,
            tools=[
                {
                    "name": "search",
                    "description": "look things up",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice={"type": "tool", "name": "search", "disable_parallel_tool_use": True},
            metadata={"user_id": "user-1"},
        )
    )
    request = decoded.request
    assert decoded.alias == "coding"
    assert request.surface == GatewayApiSurface.MESSAGES
    assert request.messages[0].role == "system"
    assert request.messages[0].content == "be terse\n\nand kind"
    assert request.messages[1].role == "user"
    assert request.maximum_output_tokens == 128
    assert request.temperature == 0.5
    assert request.top_p == 0.9
    assert request.stop == ("STOP", "END")
    assert request.stream is True
    assert request.include_usage is True
    assert request.tools[0].name == "search"
    assert request.tools[0].parameters == {"type": "object"}
    assert request.tool_choice == GatewayNamedToolChoice(name="search")
    assert request.parallel_tool_calls is False
    assert request.metadata == {"user_id": "user-1"}
    assert request.idempotency_key is None
    assert request.client_request_id is None


def test_decode_splits_tool_results_and_keeps_assistant_tool_calls() -> None:
    """A mixed history turn splits into ordered canonical messages."""
    decoded = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "run the tool"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "on it"},
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "search",
                            "input": {"q": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": [{"type": "text", "text": "found it"}],
                        },
                        {"type": "text", "text": "now answer"},
                    ],
                },
            ]
        )
    )
    roles = [message.role for message in decoded.request.messages]
    assert roles == ["user", "assistant", "tool", "user"]
    assistant = decoded.request.messages[1]
    assert assistant.content == "on it"
    assert assistant.tool_calls[0].call_id == "call-1"
    assert assistant.tool_calls[0].raw_arguments == '{"q":"x"}'
    tool = decoded.request.messages[2]
    assert tool.tool_call_id == "call-1"
    assert tool.content == "found it"
    assert decoded.request.messages[3].content == "now answer"


def test_decode_drops_only_nonsemantic_cache_control() -> None:
    """A cache hint may be omitted without changing the requested model behavior."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                        }
                    ],
                }
            ],
        )
    )
    assert decoded.request.messages[0].content == "hi"
    assert decoded.request.metadata == {}


@pytest.mark.parametrize(
    "field",
    [
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        {"output_config": {"effort": "high"}},
    ],
)
def test_reasoning_controls_are_rejected_instead_of_silently_dropped(field: JsonObject) -> None:
    """Native Messages reasoning controls fail until their semantics can be preserved."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(**field))
    assert excinfo.value.detail.param in field


def test_thinking_history_blocks_are_rejected_instead_of_dropped() -> None:
    """Assistant reasoning history cannot disappear during canonical translation."""
    with pytest.raises(OpenAIProtocolError, match="thinking history blocks are not supported"):
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "assistant",
                        "content": [{"type": "thinking", "thinking": "private"}],
                    }
                ]
            )
        )


@pytest.mark.parametrize(
    ("overrides", "param_fragment"),
    [
        ({"service_tier": "auto"}, "service_tier"),
        ({"container": "c"}, "container"),
        ({"unknown_field": 1}, "unknown_field"),
    ],
)
def test_unsupported_and_unknown_top_level_fields_are_rejected(
    overrides: JsonObject, param_fragment: str
) -> None:
    """Unsupported and unknown fields answer a loud field-specific 400."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(**overrides))
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail.param == param_fragment


def test_top_k_is_preserved_for_route_specific_validation() -> None:
    """The official Messages top-k field reaches the shared route contract."""
    decoded = decode_messages(_body(top_k=5))
    assert decoded.request.top_k == 5


def test_missing_max_tokens_is_rejected_with_its_field() -> None:
    """max_tokens is required by the Anthropic protocol."""
    payload = _body()
    del payload["max_tokens"]
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(payload)
    assert excinfo.value.detail.param == "max_tokens"


def test_image_blocks_are_rejected_with_a_targeted_hint() -> None:
    """A known-but-unsupported block gets its own explanation."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}],
                    }
                ]
            )
        )
    assert "image blocks are not supported" in excinfo.value.detail.message


def test_document_block_inside_tool_result_is_rejected() -> None:
    """Nested unsupported blocks inside tool results are rejected loudly."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-1",
                                "content": [{"type": "document", "source": {}}],
                            }
                        ],
                    }
                ]
            )
        )
    assert "document blocks are not supported" in excinfo.value.detail.message


def test_role_misplaced_blocks_and_empty_content_are_rejected() -> None:
    """Blocks are validated against their legal roles and non-empty turns."""
    with pytest.raises(OpenAIProtocolError, match="only valid in assistant messages"):
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "tool_use", "id": "call-1", "name": "n", "input": {}}],
                    }
                ]
            )
        )
    with pytest.raises(OpenAIProtocolError, match="only valid in user messages"):
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_result", "tool_use_id": "call-1"}],
                    }
                ]
            )
        )
    with pytest.raises(OpenAIProtocolError, match="must not be empty"):
        decode_messages(_body(messages=[{"role": "user", "content": ""}]))
    with pytest.raises(OpenAIProtocolError, match="must contain text"):
        decode_messages(_body(messages=[{"role": "assistant", "content": []}]))


def test_tool_choice_forms_and_stop_sequence_validation() -> None:
    """Every tool-choice form normalizes; bad stop sequences are rejected."""
    assert decode_messages(_body(tool_choice={"type": "auto"})).request.tool_choice == "auto"
    assert decode_messages(_body(tool_choice={"type": "none"})).request.tool_choice == "none"
    required = decode_messages(
        _body(
            tool_choice={"type": "any"},
            tools=[{"name": "search", "input_schema": {}}],
        )
    )
    assert required.request.tool_choice == "required"
    with pytest.raises(OpenAIProtocolError, match="requires a name"):
        decode_messages(_body(tool_choice={"type": "tool"}))
    with pytest.raises(OpenAIProtocolError, match="non-empty"):
        decode_messages(_body(stop_sequences=[""]))


def test_invalid_json_shape_errors_carry_a_dotted_field_path() -> None:
    """Wire validation errors name the offending field path."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(max_tokens=0))
    assert excinfo.value.detail.param == "max_tokens"
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(messages=[]))
    assert excinfo.value.detail.param == "messages"


def test_tool_result_error_state_is_preserved_on_the_canonical_message() -> None:
    """is_error travels on the canonical tool message without touching digests."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": "boom",
                            "is_error": True,
                        }
                    ],
                }
            ]
        )
    )
    tool = decoded.request.messages[0]
    assert tool.role == "tool"
    assert tool.tool_is_error is True
    plain = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1"}]}
            ]
        )
    )
    assert plain.request.messages[0].tool_is_error is False
