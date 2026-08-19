"""Tests for separate Chat Completions and Responses request decoding."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from wmo.common.core.artifacts import JsonObject
from wmo.runtime.gateway.contracts import GatewayApiSurface, GatewayNamedToolChoice
from wmo.runtime.gateway.openai.errors import OpenAIProtocolError
from wmo.runtime.gateway.openai.requests import (
    DecodedGatewayRequest,
    decode_chat,
    decode_responses,
)


def test_chat_decoder_preserves_every_supported_semantic_field() -> None:
    """Chat conversion retains roles, raw tools, strict schema, controls, usage, and metadata."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "developer", "content": "Follow policy."},
                {"role": "user", "content": [{"type": "text", "text": "Call weather."}]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{ "city" : "Zürich" }',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-one", "content": "sunny"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Read weather",
                        "parameters": {"type": "object"},
                        "strict": True,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "weather"}},
            "parallel_tool_calls": True,
            "max_completion_tokens": 123,
            "stop": ["END", "STOP"],
            "temperature": 0.2,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
            "stream": True,
            "stream_options": {"include_usage": True},
            "metadata": {"cohort": "test"},
        },
        idempotency_key="operation-one",
        client_request_id="operation-one",
    )

    request = decoded.request
    assert decoded.alias == "coding"
    assert request.surface == GatewayApiSurface.CHAT_COMPLETIONS
    assert tuple(message.role for message in request.messages) == (
        "developer",
        "user",
        "assistant",
        "tool",
    )
    assert request.messages[2].tool_calls[0].raw_arguments == '{ "city" : "Zürich" }'
    assert request.tools[0].strict
    assert isinstance(request.tool_choice, GatewayNamedToolChoice)
    assert request.tool_choice.name == "weather"
    assert request.maximum_output_tokens == 123
    assert request.stop == ("END", "STOP")
    assert request.structured_text is not None and request.structured_text.strict
    assert request.include_usage
    assert request.metadata == {"cohort": "test"}


def test_responses_decoder_preserves_continuation_and_distinct_wire_shapes() -> None:
    """Responses conversion keeps instructions, item history, named tools, and structured text."""
    decoded = decode_responses(
        {
            "model": "coding",
            "instructions": "Use tools.",
            "input": [
                {"type": "message", "role": "user", "content": "Weather?"},
                {
                    "type": "function_call",
                    "call_id": "call-one",
                    "name": "weather",
                    "arguments": '{"city":"Paris"}',
                },
                {"type": "function_call_output", "call_id": "call-one", "output": "sunny"},
            ],
            "previous_response_id": "resp_previous",
            "tools": [
                {
                    "type": "function",
                    "name": "weather",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "function", "name": "weather"},
            "parallel_tool_calls": False,
            "max_output_tokens": 321,
            "temperature": 0.4,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            },
            "stream": True,
            "metadata": {"cohort": "test"},
        },
        client_request_id="operation-two",
    )

    request = decoded.request
    assert request.surface == GatewayApiSurface.RESPONSES
    assert tuple(message.role for message in request.messages) == (
        "developer",
        "user",
        "assistant",
        "tool",
    )
    assert request.previous_response_id == "resp_previous"
    assert request.messages[2].tool_calls[0].raw_arguments == '{"city":"Paris"}'
    assert request.parallel_tool_calls is False
    assert request.maximum_output_tokens == 321
    assert request.structured_text is not None
    assert request.client_request_id == "operation-two"


@pytest.mark.parametrize(
    ("decoder", "payload", "param"),
    (
        (
            decode_chat,
            {"model": "coding", "messages": [{"role": "user", "content": "x"}], "n": 2},
            "n",
        ),
        (
            decode_responses,
            {"model": "coding", "input": "x", "background": True},
            "background",
        ),
        (
            decode_chat,
            {"model": "coding", "messages": [{"role": "user", "content": "x"}], "future": 1},
            "future",
        ),
    ),
)
def test_unknown_and_excluded_fields_fail_with_exact_param(
    decoder: Callable[[JsonObject], DecodedGatewayRequest], payload: JsonObject, param: str
) -> None:
    """Closed manifests reject excluded and future SDK fields before dispatch."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decoder(payload)
    assert captured.value.detail.code == "unsupported_parameter"
    assert captured.value.detail.param == param


def test_invalid_tool_arguments_and_conflicting_operation_headers_are_specific() -> None:
    """Malformed history and mismatched dedup headers identify the exact public field."""
    with pytest.raises(OpenAIProtocolError) as arguments:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-one",
                                "type": "function",
                                "function": {"name": "tool", "arguments": "{"},
                            }
                        ],
                    }
                ],
            }
        )
    assert arguments.value.detail.param == "messages.0.tool_calls.0.function.arguments"

    with pytest.raises(OpenAIProtocolError) as operation:
        decode_responses(
            {"model": "coding", "input": "x"},
            idempotency_key="one",
            client_request_id="two",
        )
    assert operation.value.detail.code == "idempotency_conflict"
    assert operation.value.detail.param == "Idempotency-Key"


def test_empty_responses_input_is_a_public_protocol_error() -> None:
    """Canonical validation failures do not leak internal Pydantic exceptions."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_responses({"model": "coding", "input": []})
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == "messages"
