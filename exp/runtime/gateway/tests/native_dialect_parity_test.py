"""Cross-engine provider-dialect parity over committed golden stream fixtures.

The golden fixtures are the contract: raw provider stream bytes in, the exact
canonical event sequence out. The native (Rust) normalizer is checked against
the goldens through ``normalize_stream_fixture``; the python event mappers
(deprecated, scheduled for removal with the python data plane) are checked
against the same goldens as a secondary assertion while they still serve
traffic.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import BillingSource, ModelSnapshot
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.models.providers.async_transport import (
    HttpxAsyncJsonTransport,
    RequestDeadline,
)
from exp.runtime.models.providers.gemini import GeminiClient
from exp.runtime.models.providers.transport import RetryPolicy


def _sse(payload: JsonObject) -> bytes:
    """Encode one JSON object as a complete SSE data event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


GEMINI_GOLDEN_CHUNKS: tuple[bytes, ...] = (
    _sse({"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}),
    _sse(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "hidden reasoning"},
                            {"text": "lo"},
                        ]
                    }
                }
            ]
        }
    ),
    _sse(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "call-1",
                                    "name": "lookup",
                                    "args": {"city": "Zürich", "count": 2},
                                }
                            }
                        ]
                    }
                }
            ]
        }
    ),
    _sse(
        {
            "candidates": [{"finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 5,
                "cachedContentTokenCount": 2,
                "thoughtsTokenCount": 3,
            },
        }
    ),
)

_GEMINI_RAW_ARGUMENTS = '{"city":"Zürich","count":2}'

GEMINI_GOLDEN_EVENTS: tuple[JsonObject, ...] = (
    {"kind": "text_delta", "text": "Hel"},
    {"kind": "text_delta", "text": "lo"},
    {"kind": "tool_call_started", "index": 0, "call_id": "call-1", "name": "lookup"},
    {"kind": "tool_arguments_delta", "index": 0, "text": _GEMINI_RAW_ARGUMENTS},
    {
        "kind": "tool_call_completed",
        "index": 0,
        "call_id": "call-1",
        "name": "lookup",
        "raw_arguments": _GEMINI_RAW_ARGUMENTS,
    },
    {
        "kind": "usage",
        "input_tokens": 11,
        "output_tokens": 5,
        "cached_input_tokens": 2,
        "reasoning_tokens": 3,
    },
    {"kind": "completed"},
)

GEMINI_REFUSAL_CHUNKS: tuple[bytes, ...] = (_sse({"candidates": [{"finishReason": "SAFETY"}]}),)

GEMINI_INCOMPLETE_CHUNKS: tuple[bytes, ...] = (
    _sse({"candidates": [{"content": {"parts": [{"text": "cut"}]}}]}),
    _sse(
        {
            "candidates": [{"finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 9},
        }
    ),
)

GEMINI_INCOMPLETE_EVENTS: tuple[JsonObject, ...] = (
    {"kind": "text_delta", "text": "cut"},
    {
        "kind": "usage",
        "input_tokens": 4,
        "output_tokens": 9,
        "cached_input_tokens": 0,
        "reasoning_tokens": None,
    },
    {"kind": "incomplete"},
)


def _native_normalized(dialect: str, chunks: Sequence[bytes]) -> JsonObject:
    """Run raw chunks through the Rust frame decoder and dialect normalizer.

    Args:
        dialect: Native dialect identifier from the wire profile.
        chunks: Raw provider stream bytes in arrival order.

    Returns:
        The decoded ``{"events": [...], "failure": ...}`` fixture result.
    """
    native = pytest.importorskip("exp_gateway_native")
    argument = json.dumps([chunk.decode("latin-1") for chunk in chunks])
    return json.loads(native.normalize_stream_fixture(dialect, argument))


def _simplified(event: GatewayEvent) -> JsonObject:
    """Project one python gateway event onto the golden fixture vocabulary.

    Args:
        event: Normalized event from a python provider mapper.

    Returns:
        The content-bearing fields in the shared fixture shape.
    """
    if event.kind is GatewayEventKind.TEXT_DELTA:
        return {"kind": "text_delta", "text": event.text_delta}
    if event.kind is GatewayEventKind.REFUSAL_DELTA:
        return {"kind": "refusal_delta", "text": event.text_delta}
    if event.kind is GatewayEventKind.TOOL_CALL_STARTED:
        return {
            "kind": "tool_call_started",
            "index": event.tool_call_index,
            "call_id": event.tool_call_id,
            "name": event.tool_name,
        }
    if event.kind is GatewayEventKind.TOOL_ARGUMENTS_DELTA:
        return {
            "kind": "tool_arguments_delta",
            "index": event.tool_call_index,
            "text": event.raw_arguments_delta,
        }
    if event.kind is GatewayEventKind.TOOL_CALL_COMPLETED:
        call = event.tool_call
        assert call is not None
        return {
            "kind": "tool_call_completed",
            "index": event.tool_call_index,
            "call_id": call.call_id,
            "name": call.name,
            "raw_arguments": call.raw_arguments,
        }
    if event.kind is GatewayEventKind.USAGE:
        usage = event.usage
        assert usage is not None
        return {
            "kind": "usage",
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
        }
    if event.kind is GatewayEventKind.COMPLETED:
        return {"kind": "completed"}
    if event.kind is GatewayEventKind.INCOMPLETE:
        return {"kind": "incomplete"}
    assert event.failure is not None
    return {
        "kind": "failed",
        "failure_class": event.failure.failure_class.value,
        "safe_message": event.failure.safe_message,
    }


class _ChunkStream(httpx.AsyncByteStream):
    """Yield exact provider stream chunks for the python mapper."""

    def __init__(self, chunks: Sequence[bytes]) -> None:
        """Store response chunks in provider order."""
        self._chunks = tuple(chunks)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield every configured chunk once."""
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        """Provider response closure needs no bookkeeping here."""


def _python_gemini_events(chunks: Sequence[bytes]) -> list[JsonObject]:
    """Run raw chunks through the deprecated python Gemini stream mapper.

    Args:
        chunks: Raw provider SSE bytes in arrival order.

    Returns:
        Simplified canonical events in the shared fixture shape.
    """

    async def scenario() -> list[JsonObject]:
        """Consume one scripted native Gemini streaming response."""

        def handler(_request: httpx.Request) -> httpx.Response:
            """Return the scripted SSE response."""
            return httpx.Response(200, stream=_ChunkStream(chunks))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = GeminiClient(
                model=ModelSnapshot(
                    provider="gemini",
                    model_id="gemini-2.5-pro",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    capabilities_sha256="a" * 64,
                    connection_sha256="b" * 64,
                ),
                api_key="fixture-key",
                base_url="https://gemini.test/v1beta",
                transport=HttpxAsyncJsonTransport(http_client),
            )
            stream = await client.stream(
                GatewayRequest(
                    surface=GatewayApiSurface.CHAT_COMPLETIONS,
                    messages=(GatewayMessage(role="user", content="hello"),),
                    stream=True,
                    include_usage=True,
                ),
                deadline=RequestDeadline.after(10),
                idempotency_key="parity-operation",
                retry_policy=RetryPolicy(1, 0, 0),
            )
            return [_simplified(event) async for event in stream]

    return asyncio.run(scenario())


def test_native_gemini_normalizer_matches_the_golden_fixture() -> None:
    """The Rust normalizer reproduces the committed canonical event sequence."""
    result = _native_normalized("gemini_generate_content", GEMINI_GOLDEN_CHUNKS)
    assert result["failure"] is None
    assert result["events"] == list(GEMINI_GOLDEN_EVENTS)

    incomplete = _native_normalized("gemini_generate_content", GEMINI_INCOMPLETE_CHUNKS)
    assert incomplete["failure"] is None
    assert incomplete["events"] == list(GEMINI_INCOMPLETE_EVENTS)

    refusal = _native_normalized("gemini_generate_content", GEMINI_REFUSAL_CHUNKS)
    assert refusal["failure"] is None
    assert refusal["events"] == [
        {
            "kind": "failed",
            "failure_class": "refusal",
            "safe_message": "provider refused the request",
        }
    ]


def test_native_gemini_normalizer_fails_streams_without_a_terminal() -> None:
    """A stream that closes before its terminal candidate fails as malformed."""
    result = _native_normalized("gemini_generate_content", GEMINI_GOLDEN_CHUNKS[:1])
    assert result["events"] == [{"kind": "text_delta", "text": "Hel"}]
    failure = result["failure"]
    assert isinstance(failure, dict)
    assert failure["failure_class"] == "malformed_response"


def test_python_gemini_mapper_matches_the_same_goldens() -> None:
    """The deprecated python mapper agrees with the committed goldens."""
    events = _python_gemini_events(GEMINI_GOLDEN_CHUNKS)
    # Known python-mapper quirk: it omits tool_call_index on
    # TOOL_CALL_COMPLETED (its consumers key on the started and delta
    # events). The golden carries the index, so align that one field.
    assert events[4]["kind"] == "tool_call_completed"
    assert events[4]["index"] is None
    events[4]["index"] = 0
    assert events == list(GEMINI_GOLDEN_EVENTS)
    assert _python_gemini_events(GEMINI_INCOMPLETE_CHUNKS) == list(GEMINI_INCOMPLETE_EVENTS)
    refusal = _python_gemini_events(GEMINI_REFUSAL_CHUNKS)
    assert len(refusal) == 1
    assert refusal[0]["kind"] == "failed"
    assert refusal[0]["failure_class"] == "refusal"
    assert refusal[0]["safe_message"] == "provider refused the request"
