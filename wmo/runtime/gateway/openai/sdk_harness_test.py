"""OpenAI Python SDK 3.0.0 sync and async protocol conformance harness."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import httpx2
import openai
from openai import AsyncOpenAI, OpenAI

from wmo.common.core.artifacts import JsonObject
from wmo.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayUsage,
)
from wmo.runtime.gateway.openai.errors import OpenAIProtocolError
from wmo.runtime.gateway.openai.requests import decode_chat, decode_responses
from wmo.runtime.gateway.openai.streaming import (
    ChatSseEncoder,
    ResponsesSseEncoder,
    encode_chat_events,
    encode_responses_events,
)


class _SdkHarness:
    """Deterministic in-memory HTTP gateway exercised only through official SDK clients."""

    def __init__(self, *, fail_first: bool = False) -> None:
        """Initialize responder accounting and optional first-attempt transport failure."""
        self.fail_first = fail_first
        self.requests = 0
        self.responder_calls = 0
        self.operation_headers: list[tuple[str | None, str | None]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        """Decode one SDK request and return protocol-encoded deterministic SSE."""
        self.requests += 1
        self.operation_headers.append(
            (
                request.headers.get("idempotency-key"),
                request.headers.get("x-client-request-id"),
            )
        )
        if self.fail_first and self.requests == 1:
            return httpx2.Response(
                500,
                json={
                    "error": {
                        "message": "retryable test failure",
                        "type": "api_error",
                        "param": None,
                        "code": "server_error",
                    }
                },
            )
        payload = cast(JsonObject, json.loads(request.content))
        try:
            if request.url.path.endswith("/chat/completions"):
                decoded = decode_chat(
                    payload,
                    idempotency_key=request.headers.get("idempotency-key"),
                    client_request_id=request.headers.get("x-client-request-id"),
                )
                self.responder_calls += 1
                frames = encode_chat_events(
                    ChatSseEncoder(
                        request_id=f"request-{self.requests}",
                        model=decoded.alias,
                        created_at=123,
                        include_usage=decoded.request.include_usage,
                    ),
                    _text_events(),
                )
            else:
                decoded = decode_responses(
                    payload,
                    idempotency_key=request.headers.get("idempotency-key"),
                    client_request_id=request.headers.get("x-client-request-id"),
                )
                self.responder_calls += 1
                frames = encode_responses_events(
                    ResponsesSseEncoder(
                        request_id=f"request-{self.requests}",
                        model=decoded.alias,
                        created_at=123.0,
                        request=decoded.request,
                    ),
                    _text_events(),
                )
        except OpenAIProtocolError as exc:
            return httpx2.Response(exc.status_code, json=exc.json_body())
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(frames).encode(),
        )


def _text_events() -> tuple[GatewayEvent, ...]:
    """Create one incremental text, usage, and terminal provider stream."""
    return (
        GatewayEvent(
            kind=GatewayEventKind.TEXT_DELTA,
            sequence_number=0,
            text_delta="hello ",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TEXT_DELTA,
            sequence_number=1,
            text_delta="world",
        ),
        GatewayEvent(
            kind=GatewayEventKind.USAGE,
            sequence_number=2,
            usage=GatewayUsage(input_tokens=2, output_tokens=2),
        ),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=3),
    )


def test_sync_openai_300_parses_both_advertised_streaming_surfaces() -> None:
    """Synchronous OpenAI 3.0.0 consumes Chat chunks and the Responses lifecycle."""
    assert openai.__version__ == "3.0.0"
    harness = _SdkHarness()
    http = httpx2.Client(transport=httpx2.MockTransport(harness))
    with OpenAI(api_key="sdk-test", base_url="http://gateway.test/v1", http_client=http) as client:
        chat = list(
            client.chat.completions.create(
                model="coding",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
                stream_options={"include_usage": True},
            )
        )
        responses = list(client.responses.create(model="coding", input="hello", stream=True))

    assert (
        "".join(chunk.choices[0].delta.content or "" for chunk in chat if chunk.choices)
        == "hello world"
    )
    assert chat[-1].usage is not None and chat[-1].usage.total_tokens == 4
    assert responses[0].type == "response.created"
    assert responses[1].type == "response.in_progress"
    assert responses[-1].type == "response.completed"
    assert harness.responder_calls == 2


def test_async_openai_300_parses_both_advertised_streaming_surfaces() -> None:
    """Asynchronous AsyncOpenAI 3.0.0 consumes both protocol state machines."""

    async def scenario() -> None:
        """Run both async SDK resources against the same deterministic harness."""
        harness = _SdkHarness()
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(harness))
        async with AsyncOpenAI(
            api_key="sdk-test",
            base_url="http://gateway.test/v1",
            http_client=http,
        ) as client:
            chat_stream = await client.chat.completions.create(
                model="coding",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            )
            chat = [chunk async for chunk in chat_stream]
            response_stream = await client.responses.create(
                model="coding",
                input="hello",
                stream=True,
            )
            responses = [event async for event in response_stream]
        assert (
            "".join(chunk.choices[0].delta.content or "" for chunk in chat if chunk.choices)
            == "hello world"
        )
        assert responses[-1].type == "response.completed"
        assert harness.responder_calls == 2

    asyncio.run(scenario())


def test_default_sdk_post_retry_is_measured_as_unkeyed_second_dispatch() -> None:
    """OpenAI's default retry sends no dedup header and can create a second billable call."""
    harness = _SdkHarness(fail_first=True)
    http = httpx2.Client(transport=httpx2.MockTransport(harness))
    with OpenAI(api_key="sdk-test", base_url="http://gateway.test/v1", http_client=http) as client:
        chunks = list(
            client.chat.completions.create(
                model="coding",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            )
        )
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert harness.requests == 2
    assert harness.responder_calls == 1
    assert harness.operation_headers == [(None, None), (None, None)]


def test_unsupported_field_reaches_zero_responder_calls() -> None:
    """Closed-profile rejection occurs before the injected responder boundary."""
    harness = _SdkHarness()
    request = httpx2.Request(
        "POST",
        "http://gateway.test/v1/chat/completions",
        json={
            "model": "coding",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "n": 2,
        },
    )
    response = harness(request)
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "n"
    assert harness.responder_calls == 0
