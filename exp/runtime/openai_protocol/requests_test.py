"""Tests for shared Chat Completions and Responses request decoding."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayNamedToolChoice
from exp.runtime.models.providers.streaming_requests import openai_responses_stream_payload
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import (
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
            "top_p": 1,
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
    assert request.temperature == 0.2
    assert request.top_p == 1.0
    assert request.structured_text is not None and request.structured_text.strict
    assert request.include_usage
    assert request.metadata == {"cohort": "test"}


def test_chat_legacy_max_tokens_reaches_native_responses_as_max_output_tokens() -> None:
    """A Chat request using legacy max_tokens serves a native Responses max_output_tokens.

    Chat clients (playground and agents) commonly send the legacy max_tokens field. On a
    direct OpenAI deployment the native Responses API rejects max_tokens and wants
    max_output_tokens, so the canonical request must translate the field and the native
    payload must never carry max_tokens.
    """
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 256,
            "stream": True,
        }
    )

    assert decoded.request.maximum_output_tokens == 256
    payload = openai_responses_stream_payload(
        "gpt-fixture",
        decoded.request,
        supports_temperature=True,
        reasoning_effort=None,
    )
    assert payload["max_output_tokens"] == 256
    assert "max_tokens" not in payload


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


def test_chat_decoder_accepts_echoed_assistant_message_with_empty_sdk_fields() -> None:
    """Assistant messages echoed verbatim from a prior gateway response must decode.

    The gateway's own Chat responses and official SDK message dumps carry
    refusal, annotations, audio, function_call, and a possibly null tool_calls
    key; a tool-call continuation sends that message back unchanged.
    """
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "refusal": None,
                    "annotations": [],
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-one", "content": "sunny"},
                {
                    "role": "assistant",
                    "content": "It is sunny.",
                    "refusal": None,
                    "tool_calls": None,
                },
                {"role": "user", "content": "Thanks."},
            ],
        }
    )
    assert decoded.request.messages[1].tool_calls[0].name == "weather"
    assert decoded.request.messages[3].content == "It is sunny."
    assert decoded.request.messages[3].tool_calls == ()


def test_chat_decoder_still_rejects_populated_unsupported_message_fields() -> None:
    """A populated refusal or annotation in request history stays rejected."""
    for extra in ({"refusal": "no"}, {"annotations": [{"type": "url_citation"}]}):
        with pytest.raises(OpenAIProtocolError) as captured:
            decode_chat(
                {
                    "model": "coding",
                    "messages": [{"role": "assistant", "content": "x", **extra}],
                }
            )
        param = captured.value.detail.param
        assert param is not None and param.startswith("messages.0.")


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


def test_chat_decoder_accepts_opencode_nucleus_and_usage_stream_shape() -> None:
    """OpenCode Chat Completions send top_p=1 with streamed usage and must decode losslessly."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hello"}],
            "top_p": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )

    assert decoded.request.top_p == 1.0
    assert decoded.request.stream
    assert decoded.request.include_usage


def test_chat_decoder_rejects_out_of_range_top_p() -> None:
    """Nucleus sampling stays inside the official [0, 1] interval."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hello"}],
                "top_p": 1.5,
            }
        )
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == "top_p"


def test_responses_decoder_still_rejects_top_p() -> None:
    """Responses keeps top_p excluded so Chat support does not widen that surface."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_responses({"model": "coding", "input": "hello", "top_p": 1})
    assert captured.value.detail.code == "unsupported_parameter"
    assert captured.value.detail.param == "top_p"


def test_empty_responses_input_is_a_public_protocol_error() -> None:
    """Canonical validation failures do not leak internal Pydantic exceptions."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_responses({"model": "coding", "input": []})
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == "messages"
