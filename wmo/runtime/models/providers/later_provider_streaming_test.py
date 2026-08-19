"""Azure and OpenRouter inherited streaming certification fixtures."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from wmo.common.models import BillingSource, ModelSnapshot
from wmo.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEventKind,
    GatewayMessage,
    GatewayRequest,
)
from wmo.runtime.models.providers.async_transport import (
    HttpxAsyncJsonTransport,
    RequestDeadline,
)
from wmo.runtime.models.providers.azure import AzureClient
from wmo.runtime.models.providers.openai_compatible import (
    OpenAICompatibleClient,
    OpenRouterClient,
)
from wmo.runtime.models.providers.transport import RetryPolicy


class _ChunkStream(httpx.AsyncByteStream):
    """Yield one complete compatible stream and record closure."""

    def __init__(self, content: bytes) -> None:
        """Retain the exact SSE bytes for one response."""
        self._content = content
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield the provider response once."""
        yield self._content

    async def aclose(self) -> None:
        """Record response closure."""
        self.closed = True


@pytest.mark.parametrize("provider", ["azure", "openrouter"])
def test_later_compatible_providers_stream_with_native_auth_and_usage(provider: str) -> None:
    """Azure and OpenRouter preserve their auth while inheriting normalized Chat streaming."""

    async def scenario() -> None:
        """Consume one provider fixture and inspect its wire request."""
        upstream = _ChunkStream(
            b"".join(
                (
                    _sse(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": "hello"},
                                    "finish_reason": "stop",
                                }
                            ]
                        }
                    ),
                    _sse(
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 4,
                                "completion_tokens": 2,
                            },
                        }
                    ),
                    b"data: [DONE]\n\n",
                )
            )
        )
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Capture one request and return the compatible SSE fixture."""
            captured.append(request)
            return httpx.Response(200, stream=upstream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = _client(provider, http_client)
            stream = await client.stream(
                _request(),
                deadline=RequestDeadline.after(10),
                idempotency_key=f"{provider}-operation",
                retry_policy=RetryPolicy(1, 0, 0),
            )
            events = [event async for event in stream]

        assert [event.kind for event in events] == [
            GatewayEventKind.TEXT_DELTA,
            GatewayEventKind.USAGE,
            GatewayEventKind.COMPLETED,
        ]
        assert events[1].usage is not None
        assert events[1].usage.input_tokens == 4
        assert events[1].usage.output_tokens == 2
        assert captured[0].headers["idempotency-key"] == f"{provider}-operation"
        if provider == "azure":
            assert captured[0].headers["api-key"] == "fixture-key"
            assert "authorization" not in captured[0].headers
            assert captured[0].url.path == "/openai/v1/chat/completions"
        else:
            assert captured[0].headers["authorization"] == "Bearer fixture-key"
            assert captured[0].headers["http-referer"]
            assert captured[0].headers["x-title"]
            assert captured[0].url.path == "/api/v1/chat/completions"
        assert upstream.closed

    asyncio.run(scenario())


def _client(
    provider: str,
    http_client: httpx.AsyncClient,
) -> OpenAICompatibleClient:
    """Construct one later compatible provider over the injected transport."""
    transport = HttpxAsyncJsonTransport(http_client)
    snapshot = ModelSnapshot(
        provider=provider,
        model_id="deployment-one" if provider == "azure" else "vendor/model-one",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )
    builders: dict[str, Callable[[], OpenAICompatibleClient]] = {
        "azure": lambda: AzureClient(
            model=snapshot,
            endpoint="https://azure.test",
            api_key="fixture-key",
            api_version="v1",
            transport=transport,
        ),
        "openrouter": lambda: OpenRouterClient(
            model=snapshot,
            api_key="fixture-key",
            base_url="https://openrouter.test/api/v1",
            transport=transport,
        ),
    }
    return builders[provider]()


def _request() -> GatewayRequest:
    """Build one minimal compatible streaming request."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
        stream=True,
        include_usage=True,
    )


def _sse(payload: dict[str, object]) -> bytes:
    """Encode one compatible JSON chunk as an SSE data event."""
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
