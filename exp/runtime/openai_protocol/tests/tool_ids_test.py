"""Opaque tool IDs survive the public protocol and provider replay boundaries."""

from __future__ import annotations

import base64
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.streaming_requests import openai_compatible_stream_payload
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses


@pytest.mark.parametrize("delimiter", ["~sig1:", "_sig1_"])
def test_signed_tool_ids_replay_verbatim_on_chat_and_responses(delimiter: str) -> None:
    """Opaque upstream signatures survive both public decoders and Chat dispatch."""
    call_id = "toolu_bdrk_synthetic" + delimiter + base64.b64encode(bytes(range(256)) * 3).decode()
    chat = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "use the tool"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "content": "ok"},
            ],
        }
    )
    responses = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "terminal",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": call_id, "output": "ok"},
            ],
        }
    )
    for decoded in (chat, responses):
        assert decoded.request.messages[-2].tool_calls[0].call_id == call_id
        assert decoded.request.messages[-1].tool_call_id == call_id
        payload = openai_compatible_stream_payload("upstream", decoded.request)
        messages = cast(list[JsonObject], payload["messages"])
        assert messages[-2]["tool_calls"] == [
            {"id": call_id, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
        ]
        assert messages[-1]["tool_call_id"] == call_id


@pytest.mark.parametrize("length", [1, 256, 257, 65_536, 0, 65_537])
@pytest.mark.parametrize(
    "field", ["chat_call", "chat_result", "responses_call", "responses_result"]
)
def test_tool_id_length_boundary_is_field_scoped(length: int, field: str) -> None:
    """Both call and result fields accept the bounded opaque ID and reject excess."""
    call_id = "x" * length
    payload: JsonObject
    if field == "chat_call":
        payload = {
            "model": "coding",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"},
                        }
                    ],
                }
            ],
        }
        param = "messages.0.tool_calls.0.id"
    elif field == "chat_result":
        payload = {
            "model": "coding",
            "messages": [{"role": "tool", "tool_call_id": call_id, "content": "ok"}],
        }
        param = "messages.0.tool_call_id"
    elif field == "responses_call":
        payload = {
            "model": "coding",
            "input": [
                {"type": "function_call", "call_id": call_id, "name": "terminal", "arguments": "{}"}
            ],
        }
        param = "input.0.call_id"
    else:
        payload = {
            "model": "coding",
            "input": [{"type": "function_call_output", "call_id": call_id, "output": "ok"}],
        }
        param = "input.0.call_id"
    decoder = decode_chat if field.startswith("chat") else decode_responses
    if length in (0, 65_537):
        with pytest.raises(OpenAIProtocolError) as error:
            decoder(payload)
        assert error.value.detail.param == param
        assert error.value.detail.code == "invalid_parameter"
    else:
        decoded = decoder(payload)
        message = decoded.request.messages[0]
        assert (
            message.tool_calls[0].call_id if field.endswith("call") else message.tool_call_id
        ) == call_id
