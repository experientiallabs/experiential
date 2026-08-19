"""Tests for incremental Chat and Responses SSE state machines."""

from __future__ import annotations

import json
from typing import cast

import pytest

from wmo.common.core.artifacts import JsonObject
from wmo.common.models.model import ToolCall
from wmo.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from wmo.runtime.gateway.openai.errors import OpenAIProtocolError
from wmo.runtime.gateway.openai.streaming import (
    ChatSseEncoder,
    ResponsesSseEncoder,
    encode_chat_events,
    encode_responses_events,
)

_RAW_ARGUMENTS = '{ "city" : "Zürich" }'


def _tool_events() -> tuple[GatewayEvent, ...]:
    """Create one arbitrarily fragmented Unicode tool call and terminal usage."""
    fragments = ('{ "ci', 'ty" : "Zü', 'rich" }')
    call = ToolCall(
        call_id="call-one",
        name="weather",
        arguments={"city": "Zürich"},
        raw_arguments=_RAW_ARGUMENTS,
    )
    return (
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=0,
            tool_call_index=0,
            tool_call_id="call-one",
            tool_name="weather",
        ),
        *tuple(
            GatewayEvent(
                kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
                sequence_number=index + 1,
                tool_call_index=0,
                raw_arguments_delta=fragment,
            )
            for index, fragment in enumerate(fragments)
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=4,
            tool_call_index=0,
            tool_call=call,
        ),
        GatewayEvent(
            kind=GatewayEventKind.USAGE,
            sequence_number=5,
            usage=GatewayUsage(
                input_tokens=10,
                cached_input_tokens=2,
                output_tokens=4,
                reasoning_tokens=1,
            ),
        ),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=6),
    )


def _chat_payload(frame: str) -> JsonObject:
    """Decode one non-sentinel Chat SSE data frame."""
    return json.loads(frame.removeprefix("data: ").strip())


def _responses_payload(frame: str) -> JsonObject:
    """Decode one named Responses SSE data frame."""
    data_line = next(line for line in frame.splitlines() if line.startswith("data: "))
    return json.loads(data_line.removeprefix("data: "))


def test_chat_sse_preserves_raw_arguments_stable_ids_usage_and_one_terminal() -> None:
    """Chat chunks retain exact fragments, one stable ID, one finish, and one done sentinel."""
    encoder = ChatSseEncoder(
        request_id="request-one",
        model="coding",
        created_at=123,
        include_usage=True,
    )
    frames = encode_chat_events(encoder, _tool_events())
    payloads = tuple(_chat_payload(frame) for frame in frames if frame != "data: [DONE]\n\n")
    argument_parts: list[str] = []
    finish_reasons: list[str] = []
    for payload in payloads:
        for choice in cast(list[JsonObject], payload["choices"]):
            delta = cast(JsonObject, choice["delta"])
            for tool in cast(list[JsonObject], delta.get("tool_calls", [])):
                function = cast(JsonObject, tool["function"])
                if "arguments" in function:
                    argument_parts.append(str(function["arguments"]))
            finish_reason = choice["finish_reason"]
            if isinstance(finish_reason, str):
                finish_reasons.append(finish_reason)
    arguments = "".join(argument_parts)
    assert arguments == _RAW_ARGUMENTS
    assert len({payload["id"] for payload in payloads}) == 1
    assert finish_reasons == ["tool_calls"]
    assert sum(frame == "data: [DONE]\n\n" for frame in frames) == 1
    assert payloads[-1]["choices"] == []
    usage = cast(JsonObject, payloads[-1]["usage"])
    prompt_details = cast(JsonObject, usage["prompt_tokens_details"])
    assert prompt_details["cached_tokens"] == 2

    with pytest.raises(OpenAIProtocolError, match="after its terminal"):
        encoder.feed(GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=7))


def test_responses_sse_emits_full_lifecycle_monotonic_sequence_and_exact_arguments() -> None:
    """Responses emits created through completed with exact raw tool bytes and one terminal."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="weather"),),
        stream=True,
    )
    encoder = ResponsesSseEncoder(
        request_id="request-one",
        model="coding",
        created_at=123.0,
        request=request,
    )
    frames = encode_responses_events(encoder, _tool_events())
    payloads = tuple(_responses_payload(frame) for frame in frames)
    event_types = tuple(str(payload["type"]) for payload in payloads)
    deltas = tuple(
        str(payload["delta"])
        for payload in payloads
        if payload["type"] == "response.function_call_arguments.delta"
    )
    assert event_types[:2] == ("response.created", "response.in_progress")
    assert event_types.count("response.completed") == 1
    assert not any(name in {"response.failed", "response.incomplete"} for name in event_types)
    assert "".join(deltas) == _RAW_ARGUMENTS
    assert [payload["sequence_number"] for payload in payloads] == list(range(len(payloads)))
    terminal = cast(JsonObject, payloads[-1]["response"])
    assert terminal["status"] == "completed"
    output = cast(list[JsonObject], terminal["output"])
    assert output[0]["arguments"] == _RAW_ARGUMENTS
    usage = cast(JsonObject, terminal["usage"])
    output_details = cast(JsonObject, usage["output_tokens_details"])
    assert output_details["reasoning_tokens"] == 1

    with pytest.raises(OpenAIProtocolError, match="after its terminal"):
        encoder.feed(GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=7))


def test_responses_failure_closes_visible_content_then_emits_one_failed_terminal() -> None:
    """A post-output failure never produces completed output or a second terminal."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hello"),),
        stream=True,
    )
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
        safe_message="Provider stream failed.",
    )
    frames = encode_responses_events(
        ResponsesSseEncoder(
            request_id="request-two",
            model="coding",
            created_at=123.0,
            request=request,
        ),
        (
            GatewayEvent(
                kind=GatewayEventKind.TEXT_DELTA,
                sequence_number=0,
                text_delta="partial",
            ),
            GatewayEvent(
                kind=GatewayEventKind.FAILED,
                sequence_number=1,
                failure=failure,
            ),
        ),
    )
    payloads = tuple(_responses_payload(frame) for frame in frames)
    terminals = [
        payload
        for payload in payloads
        if payload["type"] in {"response.completed", "response.failed", "response.incomplete"}
    ]
    assert len(terminals) == 1
    assert terminals[0]["type"] == "response.failed"
    response = cast(JsonObject, terminals[0]["response"])
    error = cast(JsonObject, response["error"])
    assert error["message"] == "Provider stream failed."
    assert "response.output_item.done" in {payload["type"] for payload in payloads}
