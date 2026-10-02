"""Cache TTL presence survives the native and Python-compatible wire encoders."""

import json

import pytest
from openai.types.responses.response_usage import ResponseUsage

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.native_responses import responses_envelope
from exp.runtime.openai_protocol.response import chat_usage
from exp.runtime.openai_protocol.streaming import ChatSseEncoder, ResponsesSseEncoder
from exp.runtime.openai_protocol.streaming_support import encode_events


@pytest.mark.parametrize(
    ("hour", "unreported"), [(None, False), (0, False), (60, False), (None, True)]
)
@pytest.mark.parametrize("surface", ["chat", "responses"])
def test_native_and_python_cache_ttl_presence_matches(
    hour: int | None, unreported: bool, surface: str
) -> None:
    """Unknown, known-zero and positive TTL subsets have identical wire evidence."""
    native = pytest.importorskip("exp_gateway_native")
    usage = GatewayUsage(
        input_tokens=120,
        output_tokens=10,
        cached_input_tokens=None if unreported else 10,
        cache_creation_input_tokens=None if unreported else 100,
        cache_creation_1h_input_tokens=hour,
        reasoning_tokens=None if unreported else 5,
    )
    events = (
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta="ok"),
        GatewayEvent(kind=GatewayEventKind.USAGE, sequence_number=1, usage=usage),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=2),
    )
    fixture = json.dumps(
        [
            {"kind": "text_delta", "text": "ok"},
            {"kind": "usage", **usage.model_dump(mode="json")},
            {"kind": "completed"},
        ]
    )
    if surface == "chat":
        encoder = ChatSseEncoder(
            request_id="cache-request", model="model", created_at=123, include_usage=True
        )
        expected = encode_events(encoder, events)
        actual = native.encode_chat_fixture("cache-request", "model", 123, True, fixture)
        wire = chat_usage(usage)
        assert wire is not None
        details = wire["prompt_tokens_details"]
        assert details == (
            None
            if unreported
            else {
                "cached_tokens": 10,
                "cache_write_tokens": 100,
                **({"cache_write_1h_tokens": hour} if hour is not None else {}),
            }
        )
    else:
        request = GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=(GatewayMessage(role="user", content="hello"),),
            stream=True,
        )
        responses = ResponsesSseEncoder(
            request_id="cache-request", model="model", created_at=123, request=request
        )
        expected = encode_events(responses, events)
        actual = native.encode_responses_fixture(
            "cache-request",
            "model",
            123,
            json.dumps(responses_envelope(request)),
            fixture,
        )
    assert list(actual) == list(expected)
    terminal = "".join(actual)
    assert ('"cache_write_1h_tokens"' in terminal) == (hour is not None)
    assert ('"unreported_token_details"' in terminal) == unreported
    if surface == "responses":
        payloads = [
            json.loads(line.removeprefix("data: "))
            for frame in actual
            for line in frame.splitlines()
            if line.startswith("data: ")
        ]
        completed = next(
            item["response"] for item in payloads if item["type"] == "response.completed"
        )
        # The official SDK still accepts required integer details while the
        # additive marker keeps their actual observability explicit.
        parsed = ResponseUsage.model_validate(completed["usage"])
        assert parsed.input_tokens_details.cache_write_tokens == (0 if unreported else 100)
