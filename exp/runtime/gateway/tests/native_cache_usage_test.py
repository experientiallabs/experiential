"""Cache and reasoning usage survive native and Python-compatible wire encoders."""

import json
from typing import cast

import pytest
from openai.types.responses.response_usage import ResponseUsage

from exp.common.core.artifacts import JsonObject
from exp.common.models import Usage
from exp.common.models.catalog_prices import GatewayTokenPrices
from exp.common.models.token_cost import schedule_usage_cost_nano_usd
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.native_responses import responses_envelope
from exp.runtime.gateway.tests.native_dialect_parity_test import _native_normalized, _sse
from exp.runtime.openai_protocol.response import chat_usage
from exp.runtime.openai_protocol.streaming import ChatSseEncoder, ResponsesSseEncoder
from exp.runtime.openai_protocol.streaming_support import encode_events


@pytest.mark.parametrize("additive", [False, True])
@pytest.mark.parametrize("surface", ["chat", "responses"])
@pytest.mark.parametrize("evidence", ["established", "late", "provisional"])
def test_sparse_reasoning_growth_stays_priceable_after_two_native_hops(
    additive: bool, surface: str, evidence: str
) -> None:
    """Coherent samples govern final usage, including corrected split inferences."""
    native = pytest.importorskip("exp_gateway_native")
    reports: tuple[JsonObject, ...] = (
        {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 115 if additive else 110,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
        {"completion_tokens": 15},
        {"completion_tokens": 20},
        {"completion_tokens_details": {"reasoning_tokens": 8}},
    )
    expected_output = 28 if additive else 20
    expected_reasoning = 8
    expected_cost = 180_000 if additive else 164_000
    if evidence == "late":
        reports = (
            {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 115,
                "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            },
            {"completion_tokens": 15, "completion_tokens_details": {"reasoning_tokens": 5}},
            {"completion_tokens": 20},
            {"total_tokens": 120},
            {
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 8},
                "total_tokens": 128 if additive else 120,
            },
        )
    elif evidence == "provisional":
        reports = (
            {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110 if additive else 115,
                "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            },
            {"completion_tokens_details": {"reasoning_tokens": 5}},
            {
                "prompt_tokens": 100 if additive else 105,
                "completion_tokens": 10,
                "completion_tokens_details": {"reasoning_tokens": 5},
                "total_tokens": 115,
            },
        )
        expected_output = 15 if additive else 10
        expected_reasoning = 5
        expected_cost = 145_000 if additive else 140_000
    normalized = _native_normalized(
        "openai_compatible",
        [
            _sse({"choices": [{"index": 0, "delta": {"content": "done"}}]}),
            *[_sse({"choices": [], "usage": report}) for report in reports],
            _sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            b"data: [DONE]\n\n",
        ],
    )
    assert normalized["failure"] is None
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hello"),),
        stream=True,
    )
    prices = GatewayTokenPrices(
        input_nano_usd_per_million_tokens=1_000_000_000,
        cached_input_nano_usd_per_million_tokens=100_000_000,
        cache_creation_input_nano_usd_per_million_tokens=1_250_000_000,
        cache_creation_1h_input_nano_usd_per_million_tokens=2_000_000_000,
        output_nano_usd_per_million_tokens=2_000_000_000,
        reasoning_nano_usd_per_million_tokens=5_000_000_000,
    )
    for _ in range(2):
        fixture = json.dumps(normalized["events"])
        if surface == "chat":
            wire = native.encode_chat_fixture("request", "model", 123, True, fixture)
        else:
            wire = native.encode_responses_fixture(
                "request", "model", 123, json.dumps(responses_envelope(request)), fixture
            )
        normalized = json.loads(
            native.normalize_stream_fixture(
                "openai_compatible" if surface == "chat" else "openai_responses",
                json.dumps(wire),
            )
        )
        assert normalized["failure"] is None
        observed = [event for event in normalized["events"] if event["kind"] == "usage"][-1]
        assert observed["output_tokens"] == expected_output
        assert observed["reasoning_tokens"] == expected_reasoning
        assert (
            schedule_usage_cost_nano_usd(
                prices,
                Usage(
                    input_tokens=observed["input_tokens"],
                    output_tokens=observed["output_tokens"],
                    cached_input_tokens=observed["cached_input_tokens"],
                    cache_write_input_tokens=observed["cache_creation_input_tokens"],
                    cache_write_1h_input_tokens=observed.get("cache_creation_1h_input_tokens"),
                    reasoning_tokens=observed["reasoning_tokens"],
                ),
            )
            == expected_cost
        )


@pytest.mark.parametrize("reported_write", [None, 0])
@pytest.mark.parametrize("surface", ["chat", "responses"])
def test_anthropic_zero_writes_remain_priceable_after_two_native_hops(
    reported_write: int | None, surface: str
) -> None:
    """Zero writes in an input-bearing report survive relays and surcharge pricing."""
    native = pytest.importorskip("exp_gateway_native")
    usage: JsonObject = {"input_tokens": 100, "output_tokens": 1}
    if reported_write is not None:
        usage["cache_creation_input_tokens"] = reported_write
    normalized = _native_normalized(
        "anthropic_messages",
        [
            _sse({"type": "message_start", "message": {"usage": usage}}),
            _sse(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": "done"},
                }
            ),
            _sse({"type": "content_block_stop", "index": 0}),
            _sse(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 20},
                }
            ),
            _sse({"type": "message_stop"}),
        ],
    )
    assert normalized["failure"] is None
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hello"),),
        stream=True,
    )
    for _ in range(2):
        events = cast(list[JsonObject], normalized["events"])
        # Chat/Responses omit the Anthropic-only text-block boundary event.
        fixture = json.dumps([event for event in events if event["kind"] != "text_block_started"])
        if surface == "chat":
            wire = native.encode_chat_fixture("request", "model", 123, True, fixture)
        else:
            wire = native.encode_responses_fixture(
                "request", "model", 123, json.dumps(responses_envelope(request)), fixture
            )
        normalized = json.loads(
            native.normalize_stream_fixture(
                "openai_compatible" if surface == "chat" else "openai_responses",
                json.dumps(wire),
            )
        )
        assert normalized["failure"] is None
        observed = [event for event in normalized["events"] if event["kind"] == "usage"][-1]
        assert observed["cache_creation_input_tokens"] == 0
        prices = GatewayTokenPrices(
            input_nano_usd_per_million_tokens=1_000_000_000,
            cached_input_nano_usd_per_million_tokens=100_000_000,
            cache_creation_input_nano_usd_per_million_tokens=1_250_000_000,
            cache_creation_1h_input_nano_usd_per_million_tokens=2_000_000_000,
            output_nano_usd_per_million_tokens=5_000_000_000,
            reasoning_nano_usd_per_million_tokens=5_000_000_000,
        )
        assert (
            schedule_usage_cost_nano_usd(
                prices,
                Usage(
                    input_tokens=observed["input_tokens"],
                    output_tokens=observed["output_tokens"],
                    cached_input_tokens=observed["cached_input_tokens"],
                    cache_write_input_tokens=observed["cache_creation_input_tokens"],
                    cache_write_1h_input_tokens=observed.get("cache_creation_1h_input_tokens"),
                    reasoning_tokens=observed["reasoning_tokens"],
                ),
            )
            == 200_000
        )


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
