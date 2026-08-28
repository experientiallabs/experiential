"""Tests for incremental Chat and Responses SSE state machines."""

from __future__ import annotations

import json
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ToolCall
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.streaming import (
    ChatSseEncoder,
    ResponsesSseEncoder,
    encode_chat_events,
    encode_responses_events,
)

_RAW_ARGUMENTS = '{ "city" : "Zürich" }'
_FIREWORKS_MODELS = (
    "accounts/fireworks/models/deepseek-v4-flash-0731",
    "accounts/fireworks/models/glm-5p2",
    "accounts/fireworks/models/kimi-k2p7-code",
)
_FIREWORKS_ROUTE = "a" * 64


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


def test_chat_sse_exposes_ignored_compatibility_parameters() -> None:
    """Accepted lossy controls are visible to clients instead of being silently dropped."""
    encoder = ChatSseEncoder(
        request_id="request-ignored",
        model="coding",
        created_at=123,
        include_usage=False,
        ignored_parameters=("logprobs",),
    )

    payload = _chat_payload(encoder.start()[0])

    assert payload["x-experiential-ignored-parameters"] == ["logprobs"]


@pytest.mark.parametrize("model", _FIREWORKS_MODELS)
def test_chat_sse_emits_one_opaque_fireworks_carrier_for_completed_tool_calls(
    model: str,
) -> None:
    """DeepSeek, GLM, and Kimi reasoning fragments become one replay carrier."""
    events = (
        GatewayEvent(
            kind=GatewayEventKind.REASONING_CONTENT_DELTA,
            sequence_number=0,
            text_delta="first private ",
            reasoning_content_route_sha256=_FIREWORKS_ROUTE,
        ),
        GatewayEvent(
            kind=GatewayEventKind.REASONING_CONTENT_DELTA,
            sequence_number=1,
            text_delta="reasoning:\nZürich",
            reasoning_content_route_sha256=_FIREWORKS_ROUTE,
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=2,
            tool_call_index=0,
            tool_call_id="call-one",
            tool_name="weather",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=3,
            tool_call_index=0,
            raw_arguments_delta="{}",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=4,
            tool_call_index=0,
            tool_call=ToolCall(
                call_id="call-one",
                name="weather",
                arguments={},
                raw_arguments="{}",
            ),
        ),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=5),
    )

    frames = encode_chat_events(
        ChatSseEncoder(
            request_id="request-reasoning",
            model=model,
            created_at=123,
            include_usage=False,
            reasoning_content_carrier="authenticated-carrier-v2",
        ),
        events,
    )
    payloads = tuple(_chat_payload(frame) for frame in frames if frame != "data: [DONE]\n\n")
    carriers = [
        delta["reasoning_content"]
        for payload in payloads
        for choice in cast("list[JsonObject]", payload["choices"])
        if "reasoning_content" in (delta := cast("JsonObject", choice["delta"]))
    ]

    assert len(carriers) == 1
    assert carriers == ["authenticated-carrier-v2"]
    carrier_index = next(
        index
        for index, payload in enumerate(payloads)
        if any(
            "reasoning_content" in cast("JsonObject", choice["delta"])
            for choice in cast("list[JsonObject]", payload["choices"])
        )
    )
    finish_index = next(
        index
        for index, payload in enumerate(payloads)
        if any(
            choice["finish_reason"] == "tool_calls"
            for choice in cast("list[JsonObject]", payload["choices"])
        )
    )
    assert carrier_index < finish_index


@pytest.mark.parametrize("terminal", (GatewayEventKind.COMPLETED, GatewayEventKind.INCOMPLETE))
def test_chat_sse_never_exposes_reasoning_without_a_completed_tool_action(
    terminal: GatewayEventKind,
) -> None:
    """Text-only and incomplete outputs discard provider-private reasoning history."""
    events = (
        GatewayEvent(
            kind=GatewayEventKind.REASONING_CONTENT_DELTA,
            sequence_number=0,
            text_delta="provider private",
            reasoning_content_route_sha256=_FIREWORKS_ROUTE,
        ),
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=1, text_delta="answer"),
        GatewayEvent(kind=terminal, sequence_number=2),
    )

    frames = encode_chat_events(
        ChatSseEncoder(
            request_id="request-no-carrier",
            model="coding",
            created_at=123,
            include_usage=False,
        ),
        events,
    )
    assert not any("reasoning_content" in frame for frame in frames)


def test_chat_sse_rejects_reasoning_that_changes_route_mid_stream() -> None:
    """A malformed upstream cannot combine reasoning from two provider routes."""
    encoder = ChatSseEncoder(
        request_id="request-route-change",
        model="coding",
        created_at=123,
        include_usage=False,
    )
    encoder.start()
    encoder.feed(
        GatewayEvent(
            kind=GatewayEventKind.REASONING_CONTENT_DELTA,
            sequence_number=0,
            text_delta="first",
            reasoning_content_route_sha256=_FIREWORKS_ROUTE,
        )
    )

    with pytest.raises(OpenAIProtocolError, match="changed provider route"):
        encoder.feed(
            GatewayEvent(
                kind=GatewayEventKind.REASONING_CONTENT_DELTA,
                sequence_number=1,
                text_delta="second",
                reasoning_content_route_sha256="b" * 64,
            )
        )


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


def test_responses_sse_exposes_ignored_compatibility_parameters() -> None:
    """Responses advertises accepted lossy controls in every response envelope."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="weather"),),
        stream=True,
        ignored_parameters=("reasoning.summary",),
    )
    encoder = ResponsesSseEncoder(
        request_id="request-ignored",
        model="coding",
        created_at=123.0,
        request=request,
    )

    payload = _responses_payload(encoder.start()[0])
    response = cast(JsonObject, payload["response"])

    assert response["x-experiential-ignored-parameters"] == ["reasoning.summary"]


def test_responses_sse_preserves_reasoning_summary_items() -> None:
    """Reasoning summary deltas become official streaming and terminal output items."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="weather"),),
        stream=True,
        reasoning_effort="high",
        reasoning_summary="concise",
        reasoning_summary_parameters=("reasoning.summary",),
    )
    frames = encode_responses_events(
        ResponsesSseEncoder(
            request_id="request-reasoning",
            model="coding",
            created_at=123.0,
            request=request,
        ),
        (
            GatewayEvent(
                kind=GatewayEventKind.REASONING_SUMMARY_DELTA,
                sequence_number=0,
                reasoning_summary_output_index=0,
                reasoning_summary_index=0,
                reasoning_item_id="rs-provider",
                text_delta="Checked ",
            ),
            GatewayEvent(
                kind=GatewayEventKind.REASONING_SUMMARY_DELTA,
                sequence_number=1,
                reasoning_summary_output_index=0,
                reasoning_summary_index=0,
                reasoning_item_id="rs-provider",
                text_delta="the forecast.",
            ),
            GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=2),
        ),
    )
    payloads = tuple(_responses_payload(frame) for frame in frames)
    event_types = tuple(str(payload["type"]) for payload in payloads)

    assert event_types.count("response.reasoning_summary_part.added") == 1
    assert event_types.count("response.reasoning_summary_text.delta") == 2
    assert event_types.count("response.reasoning_summary_text.done") == 1
    terminal = cast(JsonObject, payloads[-1]["response"])
    assert terminal["reasoning"] == {"effort": "high", "summary": "concise"}
    output = cast(list[JsonObject], terminal["output"])
    assert output == [
        {
            "id": "rs-provider",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Checked the forecast."}],
            "status": "completed",
        }
    ]


def test_responses_sse_round_trips_fireworks_carrier_without_plaintext() -> None:
    """Responses emits only the sealed carrier on the completed reasoning item."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="weather"),),
        stream=True,
        include_encrypted_reasoning=True,
    )
    encoder = ResponsesSseEncoder(
        request_id="request-fireworks",
        model="coding",
        created_at=123.0,
        request=request,
    )
    encoder.set_reasoning_content_carrier("authenticated-carrier-v2")
    frames = list(encoder.start())
    events = (
        GatewayEvent(
            kind=GatewayEventKind.REASONING_CONTENT_DELTA,
            sequence_number=0,
            text_delta="hidden provider reasoning",
            reasoning_content_route_sha256=_FIREWORKS_ROUTE,
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=1,
            tool_call_index=0,
            tool_call_id="call-one",
            tool_name="weather",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=2,
            tool_call_index=0,
            raw_arguments_delta="{}",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=3,
            tool_call_index=0,
            tool_call=ToolCall(call_id="call-one", name="weather", arguments={}),
        ),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=4),
    )
    for event in events:
        frames.extend(encoder.feed(event))

    assert "hidden provider reasoning" not in "".join(frames)
    terminal = cast("JsonObject", _responses_payload(frames[-1])["response"])
    output = cast("list[JsonObject]", terminal["output"])
    reasoning = next(item for item in output if item["type"] == "reasoning")
    assert reasoning["encrypted_content"] == "authenticated-carrier-v2"


@pytest.mark.parametrize("include_encrypted_reasoning", (False, True))
def test_responses_sse_exposes_provider_encrypted_reasoning_only_when_requested(
    include_encrypted_reasoning: bool,
) -> None:
    """Internal continuation state is not automatically part of the public response."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="plan"),),
        stream=True,
        include_encrypted_reasoning=include_encrypted_reasoning,
    )
    frames = encode_responses_events(
        ResponsesSseEncoder(
            request_id="request-provider-encrypted",
            model="coding",
            created_at=123.0,
            request=request,
        ),
        (
            GatewayEvent(
                kind=GatewayEventKind.ENCRYPTED_REASONING,
                sequence_number=0,
                reasoning_block_index=0,
                reasoning_item_id="rs-provider",
                encrypted_content="provider-opaque",
            ),
            GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1),
        ),
    )

    payloads = tuple(_responses_payload(frame) for frame in frames)
    done = next(payload for payload in payloads if payload["type"] == "response.output_item.done")
    terminal = cast("JsonObject", payloads[-1]["response"])
    items = (cast("JsonObject", done["item"]), cast("list[JsonObject]", terminal["output"])[0])
    for reasoning in items:
        if include_encrypted_reasoning:
            assert cast("str", reasoning["encrypted_content"]).encode() == b"provider-opaque"
        else:
            assert "encrypted_content" not in reasoning


def test_responses_sse_fails_closed_when_fireworks_carrier_is_not_sealed() -> None:
    """A completed Fireworks tool turn cannot terminate without its carrier."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="weather"),),
        stream=True,
        include_encrypted_reasoning=True,
    )
    encoder = ResponsesSseEncoder(
        request_id="request-fireworks-missing",
        model="coding",
        created_at=123.0,
        request=request,
    )
    encoder.start()
    for event in (
        GatewayEvent(
            kind=GatewayEventKind.REASONING_CONTENT_DELTA,
            sequence_number=0,
            text_delta="hidden",
            reasoning_content_route_sha256=_FIREWORKS_ROUTE,
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=1,
            tool_call_index=0,
            tool_call_id="call-one",
            tool_name="weather",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=2,
            tool_call_index=0,
            raw_arguments_delta="{}",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=3,
            tool_call_index=0,
            tool_call=ToolCall(call_id="call-one", name="weather", arguments={}),
        ),
    ):
        encoder.feed(event)

    with pytest.raises(OpenAIProtocolError, match="not sealed"):
        encoder.feed(
            GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=4,
            )
        )


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
