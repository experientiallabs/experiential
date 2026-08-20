"""Native Gemini streaming, usage, tool, refusal, and cancellation certification."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence

import httpx

from exp.common.models import BillingSource, ModelSnapshot
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEventKind,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.models.providers.async_transport import (
    HttpxAsyncJsonTransport,
    RequestDeadline,
)
from exp.runtime.models.providers.gemini import GeminiClient
from exp.runtime.models.providers.transport import RetryPolicy


class _ChunkStream(httpx.AsyncByteStream):
    """Yield exact Gemini SSE chunks and record transport closure."""

    def __init__(self, chunks: Sequence[bytes]) -> None:
        """Store response chunks in provider order."""
        self._chunks = tuple(chunks)
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield every configured chunk once."""
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        """Record closure of the provider response."""
        self.closed = True


class _HangingStream(httpx.AsyncByteStream):
    """Block Gemini first-byte delivery until cancellation closes the response."""

    def __init__(self) -> None:
        """Initialize first-byte and closure synchronization events."""
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Wait for response closure without yielding a byte."""
        self.started.set()
        await self.closed.wait()
        if False:
            yield b""

    async def aclose(self) -> None:
        """Release the pending first-byte wait."""
        self.closed.set()


def test_gemini_stream_normalizes_text_usage_and_terminal_state() -> None:
    """Gemini SSE text and provider-specific usage become ordered gateway events."""

    async def scenario() -> None:
        """Consume one successful native Gemini response."""
        upstream = _ChunkStream(
            (
                _sse({"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}),
                _sse(
                    {
                        "candidates": [{"finishReason": "STOP"}],
                        "usageMetadata": {
                            "promptTokenCount": 5,
                            "candidatesTokenCount": 3,
                            "cachedContentTokenCount": 2,
                            "thoughtsTokenCount": 1,
                        },
                    }
                ),
            )
        )
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Return the scripted SSE response and capture its request."""
            captured.append(request)
            return httpx.Response(200, stream=upstream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            stream = await _client(http_client).stream(
                _request(),
                deadline=RequestDeadline.after(10),
                idempotency_key="deployment-operation",
                retry_policy=RetryPolicy(1, 0, 0),
            )
            events = [event async for event in stream]

        assert [event.kind for event in events] == [
            GatewayEventKind.TEXT_DELTA,
            GatewayEventKind.USAGE,
            GatewayEventKind.COMPLETED,
        ]
        assert events[0].text_delta == "hello"
        assert events[1].usage is not None
        assert events[1].usage.input_tokens == 5
        assert events[1].usage.cached_input_tokens == 2
        assert events[1].usage.reasoning_tokens == 1
        assert captured[0].url.path.endswith("/models/gemini-2.5-pro:streamGenerateContent")
        assert captured[0].url.params["alt"] == "sse"
        assert captured[0].headers["x-goog-api-key"] == "fixture-key"
        assert upstream.closed

    asyncio.run(scenario())


def test_gemini_stream_preserves_complete_tool_arguments_and_typed_refusal() -> None:
    """Gemini function calls retain canonical JSON and safety stops stay content-free."""

    async def scenario() -> None:
        """Consume independent tool-completion and refusal streams."""
        tool_upstream = _ChunkStream(
            (
                _sse(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "id": "call-one",
                                                "name": "lookup",
                                                "args": {"city": "Zürich"},
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ),
                _sse({"candidates": [{"finishReason": "STOP"}]}),
            )
        )
        refusal_upstream = _ChunkStream((_sse({"candidates": [{"finishReason": "SAFETY"}]}),))
        responses = iter((tool_upstream, refusal_upstream))

        def handler(_request: httpx.Request) -> httpx.Response:
            """Return each scripted provider stream in order."""
            return httpx.Response(200, stream=next(responses))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = _client(http_client)
            first = await client.stream(
                _request(),
                deadline=RequestDeadline.after(10),
                idempotency_key="tool-operation",
                retry_policy=RetryPolicy(1, 0, 0),
            )
            tool_events = [event async for event in first]
            second = await client.stream(
                _request(),
                deadline=RequestDeadline.after(10),
                idempotency_key="refusal-operation",
                retry_policy=RetryPolicy(1, 0, 0),
            )
            refusal_events = [event async for event in second]

        assert [event.kind for event in tool_events] == [
            GatewayEventKind.TOOL_CALL_STARTED,
            GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            GatewayEventKind.TOOL_CALL_COMPLETED,
            GatewayEventKind.COMPLETED,
        ]
        assert tool_events[1].raw_arguments_delta == '{"city":"Zürich"}'
        assert tool_events[2].tool_call is not None
        assert tool_events[2].tool_call.raw_arguments == '{"city":"Zürich"}'
        assert refusal_events[-1].kind is GatewayEventKind.FAILED
        assert refusal_events[-1].failure is not None
        assert refusal_events[-1].failure.failure_class is GatewayFailureClass.REFUSAL
        assert refusal_events[-1].failure.safe_details == {"signal": "safety"}

    asyncio.run(scenario())


def test_gemini_stream_cancellation_closes_the_active_http_response() -> None:
    """Cancelling Gemini first-byte wait closes its provider-owned response promptly."""

    async def scenario() -> None:
        """Open a hanging Gemini response and cancel it before semantic output."""
        upstream = _HangingStream()

        def handler(_request: httpx.Request) -> httpx.Response:
            """Return one response that never produces its first byte."""
            return httpx.Response(200, stream=upstream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            stream = await _client(http_client).stream(
                _request(),
                deadline=RequestDeadline.after(10),
                idempotency_key="cancel-operation",
                retry_policy=RetryPolicy(1, 0, 0),
            )
            pending = asyncio.create_task(anext(stream))
            await upstream.started.wait()
            await stream.cancel()
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

        assert upstream.closed.is_set()

    asyncio.run(scenario())


def test_gemini_stream_skips_reasoning_parts_and_thought_signatures() -> None:
    """Thought parts and bare thought signatures are skipped, not rejected as malformed."""

    async def scenario() -> None:
        """Consume a stream that interleaves reasoning parts with visible text."""
        upstream = _ChunkStream(
            (
                _sse(
                    {
                        "candidates": [
                            {"content": {"parts": [{"text": "pondering", "thought": True}]}}
                        ]
                    }
                ),
                _sse({"candidates": [{"content": {"parts": [{"thoughtSignature": "sig-abc"}]}}]}),
                _sse({"candidates": [{"content": {"parts": [{"text": ""}]}}]}),
                _sse({"candidates": [{"content": {"parts": [{"text": "answer"}]}}]}),
                _sse(
                    {
                        "candidates": [{"finishReason": "STOP"}],
                        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
                    }
                ),
            )
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            """Return the scripted reasoning-and-text SSE response."""
            return httpx.Response(200, stream=upstream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            stream = await _client(http_client).stream(
                _request(),
                deadline=RequestDeadline.after(10),
                idempotency_key="reasoning-operation",
                retry_policy=RetryPolicy(1, 0, 0),
            )
            events = [event async for event in stream]

        assert [event.kind for event in events] == [
            GatewayEventKind.TEXT_DELTA,
            GatewayEventKind.USAGE,
            GatewayEventKind.COMPLETED,
        ]
        assert events[0].text_delta == "answer"

    asyncio.run(scenario())


def _client(http_client: httpx.AsyncClient) -> GeminiClient:
    """Construct one native Gemini client over the injected async transport."""
    return GeminiClient(
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


def _request() -> GatewayRequest:
    """Build one canonical streaming Gemini request."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
        stream=True,
        include_usage=True,
    )


def _sse(payload: dict[str, object]) -> bytes:
    """Encode one JSON object as a complete SSE data event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
