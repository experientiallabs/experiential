"""Deterministic launch-provider streaming, usage, deadline, and cancellation tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence

import httpx
import pytest

from wmo.common.models import BillingSource, ModelSnapshot
from wmo.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from wmo.runtime.models.providers.anthropic import AnthropicClient
from wmo.runtime.models.providers.async_transport import (
    HttpxAsyncJsonTransport,
    RequestDeadline,
)
from wmo.runtime.models.providers.errors import ProviderResponseError
from wmo.runtime.models.providers.openai import OpenAIClient
from wmo.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from wmo.runtime.models.providers.streaming import (
    _MAXIMUM_SSE_EVENT_BYTES,
    NormalizedProviderStream,
    _SseDecoder,
)
from wmo.runtime.models.providers.transport import RetryPolicy


class _ChunkStream(httpx.AsyncByteStream):
    """HTTPX fixture stream that yields exact byte fragments and records closure."""

    def __init__(self, chunks: Sequence[bytes]) -> None:
        """Store provider-order response chunks.

        Args:
            chunks: Exact byte fragments exposed to the transport.
        """
        self._chunks = tuple(chunks)
        self.closed = False

    @property
    def status_code(self) -> int:
        """Expose a successful status when used directly as the decoder byte seam."""
        return 200

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield every scripted response fragment in order."""
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        """Record response closure."""
        self.closed = True


class _HangingStream(httpx.AsyncByteStream):
    """HTTPX fixture stream that blocks until response cancellation closes it."""

    def __init__(self) -> None:
        """Create one initially open blocking stream."""
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Block the first-byte phase until closure."""
        self.started.set()
        await self.closed.wait()
        if False:
            yield b""

    async def aclose(self) -> None:
        """Release the blocked iterator and record cancellation."""
        self.closed.set()


class _ReadFailureStream(httpx.AsyncByteStream):
    """HTTPX fixture stream that raises one read-time transport failure."""

    def __init__(self) -> None:
        """Create one initially open failing stream."""
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Raise the same HTTPX error a disconnected response body produces."""
        if False:
            yield b""
        raise httpx.ReadError("private-read-error-canary")

    async def aclose(self) -> None:
        """Record response closure after the read error."""
        self.closed = True


def _snapshot(provider: str) -> ModelSnapshot:
    """Build one immutable provider identity fixture."""
    return ModelSnapshot(
        provider=provider,
        model_id="exact-model",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


def _request(surface: GatewayApiSurface) -> GatewayRequest:
    """Build one canonical streaming request with a strict function tool."""
    return GatewayRequest(
        surface=surface,
        messages=(
            GatewayMessage(role="developer", content="Be precise."),
            GatewayMessage(role="user", content="Use the tool."),
        ),
        tools=(
            GatewayToolDefinition(
                name="lookup",
                parameters={"type": "object"},
                strict=True,
            ),
        ),
        parallel_tool_calls=True,
        stream=True,
        include_usage=True,
    )


def _provider_client(
    provider: str,
    http_client: httpx.AsyncClient,
) -> OpenAIClient | AnthropicClient | OpenAICompatibleClient:
    """Construct one launch adapter over a caller-owned mock HTTP client.

    Args:
        provider: Stable launch provider family.
        http_client: Mock-backed HTTPX client.

    Returns:
        The corresponding launch adapter.
    """
    transport = HttpxAsyncJsonTransport(http_client)
    if provider == "openai":
        return OpenAIClient(
            model=_snapshot(provider),
            api_key="fixture-key",
            transport=transport,
        )
    if provider == "anthropic":
        return AnthropicClient(
            model=_snapshot(provider),
            api_key="fixture-key",
            base_url="https://anthropic.test/v1",
            transport=transport,
        )
    if provider == "openai-compatible":
        return OpenAICompatibleClient(
            model=_snapshot(provider),
            api_key="fixture-key",
            base_url="https://compatible.test/v1",
            transport=transport,
        )
    raise AssertionError(f"unknown provider fixture {provider!r}")


def _provider_surface(provider: str) -> GatewayApiSurface:
    """Return the canonical public surface used by one provider fixture."""
    return (
        GatewayApiSurface.RESPONSES if provider == "openai" else GatewayApiSurface.CHAT_COMPLETIONS
    )


def _sse(payload: Mapping[str, object], *, event: str | None = None) -> bytes:
    """Encode one provider fixture as a complete SSE event."""
    prefix = b"" if event is None else f"event: {event}\n".encode()
    return (
        prefix
        + b"data: "
        + json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        + b"\n\n"
    )


async def _collect(stream: NormalizedProviderStream) -> list[GatewayEvent]:
    """Consume one normalized stream through its terminal event."""
    return [event async for event in stream]


def test_openai_responses_stream_preserves_raw_tools_usage_and_commitment() -> None:
    """Responses streaming keeps raw tool order and provider-specific token subsets."""
    raw_arguments = '{ "city": "Zürich" }'
    frames = b"".join(
        (
            _sse(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc-1",
                        "call_id": "call-1",
                        "name": "lookup",
                        "arguments": "",
                    },
                },
                event="response.output_item.added",
            ),
            _sse(
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "delta": '{ "city": ',
                }
            ),
            _sse(
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "delta": '"Zürich" }',
                }
            ),
            _sse(
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": 0,
                    "arguments": raw_arguments,
                }
            ),
            _sse({"type": "response.output_text.delta", "delta": "done"}),
            _sse(
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 7,
                            "input_tokens_details": {"cached_tokens": 4},
                            "output_tokens_details": {"reasoning_tokens": 3},
                        },
                    },
                }
            ),
        )
    )
    unicode_offset = frames.index("ü".encode())
    upstream = _ChunkStream((frames[: unicode_offset + 1], frames[unicode_offset + 1 :]))
    observed_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """Record the streaming request and return fragmented native events."""
        observed_payloads.append(json.loads(request.content))
        return httpx.Response(200, stream=upstream)

    async def scenario() -> list[GatewayEvent]:
        """Open the Responses stream and read events beginning with its first."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAIClient(
                model=_snapshot("openai"),
                api_key="fixture-key",
                transport=HttpxAsyncJsonTransport(http_client),
            )
            stream = await client.stream(
                _request(GatewayApiSurface.RESPONSES),
                deadline=RequestDeadline.after(2),
                idempotency_key="attempt-1",
            )
            first = await anext(stream)
            assert first.kind is GatewayEventKind.TOOL_CALL_STARTED
            return [first, *await _collect(stream)]

    events = asyncio.run(scenario())

    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert [event.kind for event in events] == [
        GatewayEventKind.TOOL_CALL_STARTED,
        GatewayEventKind.TOOL_ARGUMENTS_DELTA,
        GatewayEventKind.TOOL_ARGUMENTS_DELTA,
        GatewayEventKind.TOOL_CALL_COMPLETED,
        GatewayEventKind.TEXT_DELTA,
        GatewayEventKind.USAGE,
        GatewayEventKind.COMPLETED,
    ]
    assert events[3].tool_call is not None
    assert events[3].tool_call_index == 0
    assert events[3].tool_call.arguments_json() == raw_arguments
    assert events[5].usage is not None
    assert events[5].usage.cached_input_tokens == 4
    assert events[5].usage.reasoning_tokens == 3
    assert observed_payloads[0]["stream"] is True
    assert observed_payloads[0]["store"] is False
    assert observed_payloads[0]["instructions"] == "Be precise."
    assert upstream.closed is True


@pytest.mark.parametrize(
    ("reason", "terminal_kind", "failure_class"),
    [
        ("max_output_tokens", GatewayEventKind.INCOMPLETE, None),
        ("content_filter", GatewayEventKind.FAILED, GatewayFailureClass.REFUSAL),
        ("unsupported_reason", GatewayEventKind.FAILED, GatewayFailureClass.PROVIDER_INTERNAL),
    ],
)
def test_openai_incomplete_reasons_do_not_conflate_filtering_with_length(
    reason: str,
    terminal_kind: GatewayEventKind,
    failure_class: GatewayFailureClass | None,
) -> None:
    """Only output-budget exhaustion maps to the public incomplete terminal."""
    frames = _sse(
        {
            "type": "response.incomplete",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": reason},
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        }
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return one native incomplete terminal reason."""
        del request
        return httpx.Response(200, stream=_ChunkStream((frames,)))

    async def scenario() -> list[GatewayEvent]:
        """Consume usage and the classified terminal event."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = _provider_client("openai", http_client)
            stream = await client.stream(
                _request(GatewayApiSurface.RESPONSES),
                deadline=RequestDeadline.after(1),
                idempotency_key=f"incomplete-{reason}",
            )
            return await _collect(stream)

    events = asyncio.run(scenario())

    assert [event.kind for event in events] == [GatewayEventKind.USAGE, terminal_kind]
    if failure_class is None:
        assert events[-1].failure is None
    else:
        assert events[-1].failure is not None
        assert events[-1].failure.failure_class is failure_class


def test_anthropic_stream_normalizes_cache_usage_and_tool_fragments() -> None:
    """Messages streaming adds cache units to total input and keeps raw input JSON."""
    frames = b"".join(
        (
            _sse(
                {
                    "type": "message_start",
                    "message": {
                        "usage": {
                            "input_tokens": 10,
                            "cache_read_input_tokens": 3,
                            "cache_creation_input_tokens": 2,
                        }
                    },
                },
                event="message_start",
            ),
            _sse(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "lookup",
                        "input": {},
                    },
                },
                event="content_block_start",
            ),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"x":'},
                },
                event="content_block_delta",
            ),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": "1}"},
                },
                event="content_block_delta",
            ),
            _sse({"type": "content_block_stop", "index": 0}, event="content_block_stop"),
            _sse(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 5},
                },
                event="message_delta",
            ),
            _sse({"type": "message_stop"}, event="message_stop"),
        )
    )
    upstream = _ChunkStream((frames,))

    async def handler(request: httpx.Request) -> httpx.Response:
        """Require native streaming and return the Anthropic fixture."""
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, stream=upstream)

    async def scenario() -> list[GatewayEvent]:
        """Consume one native Messages stream."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = AnthropicClient(
                model=_snapshot("anthropic"),
                api_key="fixture-key",
                base_url="https://anthropic.test/v1",
                transport=HttpxAsyncJsonTransport(http_client),
            )
            stream = await client.stream(
                _request(GatewayApiSurface.CHAT_COMPLETIONS),
                deadline=RequestDeadline.after(2),
                idempotency_key="attempt-2",
            )
            return await _collect(stream)

    events = asyncio.run(scenario())

    assert [event.kind for event in events] == [
        GatewayEventKind.TOOL_CALL_STARTED,
        GatewayEventKind.TOOL_ARGUMENTS_DELTA,
        GatewayEventKind.TOOL_ARGUMENTS_DELTA,
        GatewayEventKind.TOOL_CALL_COMPLETED,
        GatewayEventKind.USAGE,
        GatewayEventKind.COMPLETED,
    ]
    assert events[3].tool_call is not None
    assert events[3].tool_call_index == 0
    assert events[3].tool_call.arguments_json() == '{"x":1}'
    assert events[4].usage is not None
    assert events[4].usage.input_tokens == 15
    assert events[4].usage.output_tokens == 5
    assert events[4].usage.cached_input_tokens == 3


def test_anthropic_text_stream_emits_text_usage_and_completion() -> None:
    """Native Messages text streaming commits and emits one complete lifecycle."""
    frames = b"".join(
        (
            _sse(
                {
                    "type": "message_start",
                    "message": {
                        "usage": {
                            "input_tokens": 3,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        }
                    },
                },
                event="message_start",
            ),
            _sse(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                event="content_block_start",
            ),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hello"},
                },
                event="content_block_delta",
            ),
            _sse({"type": "content_block_stop", "index": 0}, event="content_block_stop"),
            _sse(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 2},
                },
                event="message_delta",
            ),
            _sse({"type": "message_stop"}, event="message_stop"),
        )
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        """Require native streaming and return the Anthropic text fixture."""
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, stream=_ChunkStream((frames,)))

    async def scenario() -> list[GatewayEvent]:
        """Consume one native Messages text stream."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = AnthropicClient(
                model=_snapshot("anthropic"),
                api_key="fixture-key",
                base_url="https://anthropic.test/v1",
                transport=HttpxAsyncJsonTransport(http_client),
            )
            stream = await client.stream(
                _request(GatewayApiSurface.CHAT_COMPLETIONS),
                deadline=RequestDeadline.after(2),
                idempotency_key="anthropic-text-attempt",
            )
            return await _collect(stream)

    events = asyncio.run(scenario())

    assert [event.kind for event in events] == [
        GatewayEventKind.TEXT_DELTA,
        GatewayEventKind.USAGE,
        GatewayEventKind.COMPLETED,
    ]
    assert events[0].text_delta == "hello"
    assert events[1].usage is not None
    assert events[1].usage.input_tokens == 3
    assert events[1].usage.output_tokens == 2


def test_openai_compatible_stream_emits_refusal_reasoning_usage_and_terminal() -> None:
    """Generic Chat streaming retains refusal semantics and reported reasoning units."""
    frames = b"".join(
        (
            _sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"refusal": "cannot comply"},
                            "finish_reason": "content_filter",
                        }
                    ]
                }
            ),
            _sse(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 5},
                        "completion_tokens_details": {"reasoning_tokens": 1},
                    },
                }
            ),
            b"data: [DONE]\n\n",
        )
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        """Verify usage was requested and return a refusal stream."""
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        return httpx.Response(200, stream=_ChunkStream((frames,)))

    async def scenario() -> list[GatewayEvent]:
        """Consume one generic compatible refusal stream."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleClient(
                model=_snapshot("openai-compatible"),
                api_key="fixture-key",
                base_url="https://compatible.test/v1",
                transport=HttpxAsyncJsonTransport(http_client),
            )
            stream = await client.stream(
                _request(GatewayApiSurface.CHAT_COMPLETIONS),
                deadline=RequestDeadline.after(2),
                idempotency_key="attempt-3",
            )
            return await _collect(stream)

    events = asyncio.run(scenario())

    assert [event.kind for event in events] == [
        GatewayEventKind.REFUSAL_DELTA,
        GatewayEventKind.USAGE,
        GatewayEventKind.FAILED,
    ]
    assert events[1].usage is not None
    assert events[1].usage.cached_input_tokens == 5
    assert events[1].usage.reasoning_tokens == 1
    assert events[2].failure is not None
    assert events[2].failure.failure_class is GatewayFailureClass.REFUSAL


def test_openai_compatible_stream_preserves_provider_order_tool_arguments() -> None:
    """Generic Chat tool fragments remain byte-for-byte ordered through completion."""
    raw_arguments = '{"b": 2, "a": 1}'
    frames = b"".join(
        (
            _sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"b":',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            _sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": ' 2, "a": 1}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
            b"data: [DONE]\n\n",
        )
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return incrementally fragmented generic Chat tool arguments."""
        del request
        return httpx.Response(200, stream=_ChunkStream((frames,)))

    async def scenario() -> list[GatewayEvent]:
        """Consume one compatible tool stream."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = _provider_client("openai-compatible", http_client)
            stream = await client.stream(
                _request(GatewayApiSurface.CHAT_COMPLETIONS),
                deadline=RequestDeadline.after(1),
                idempotency_key="compatible-tool",
            )
            return await _collect(stream)

    events = asyncio.run(scenario())

    assert [event.kind for event in events] == [
        GatewayEventKind.TOOL_CALL_STARTED,
        GatewayEventKind.TOOL_ARGUMENTS_DELTA,
        GatewayEventKind.TOOL_ARGUMENTS_DELTA,
        GatewayEventKind.TOOL_CALL_COMPLETED,
        GatewayEventKind.COMPLETED,
    ]
    assert events[3].tool_call is not None
    assert events[3].tool_call_index == 0
    assert events[3].tool_call.arguments_json() == raw_arguments


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openai-compatible"])
def test_launch_adapters_complete_empty_argument_tool_calls(provider: str) -> None:
    """A tool call with no streamed argument fragments completes with empty-object JSON."""
    if provider == "openai":
        frames = b"".join(
            (
                _sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "noargs",
                            "arguments": "",
                        },
                    }
                ),
                _sse(
                    {
                        "type": "response.function_call_arguments.done",
                        "output_index": 0,
                        "arguments": "",
                    }
                ),
                _sse(
                    {
                        "type": "response.completed",
                        "response": {"status": "completed", "usage": None},
                    }
                ),
            )
        )
    elif provider == "anthropic":
        frames = b"".join(
            (
                _sse(
                    {
                        "type": "message_start",
                        "message": {
                            "usage": {
                                "input_tokens": 2,
                                "cache_read_input_tokens": 0,
                                "cache_creation_input_tokens": 0,
                            }
                        },
                    }
                ),
                _sse(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "noargs",
                            "input": {},
                        },
                    }
                ),
                _sse({"type": "content_block_stop", "index": 0}),
                _sse(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use"},
                        "usage": {"output_tokens": 1},
                    }
                ),
                _sse({"type": "message_stop"}),
            )
        )
    else:
        frames = b"".join(
            (
                _sse(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call-1",
                                            "function": {"name": "noargs"},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                ),
                b"data: [DONE]\n\n",
            )
        )

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return the selected provider's no-argument tool-call fixture."""
        del request
        return httpx.Response(200, stream=_ChunkStream((frames,)))

    async def scenario() -> list[GatewayEvent]:
        """Consume one empty-argument tool-call stream."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = _provider_client(provider, http_client)
            stream = await client.stream(
                _request(_provider_surface(provider)),
                deadline=RequestDeadline.after(1),
                idempotency_key=f"empty-args-{provider}",
            )
            return await _collect(stream)

    events = asyncio.run(scenario())

    assert [event.kind for event in events[:3]] == [
        GatewayEventKind.TOOL_CALL_STARTED,
        GatewayEventKind.TOOL_ARGUMENTS_DELTA,
        GatewayEventKind.TOOL_CALL_COMPLETED,
    ]
    assert events[1].tool_call_index == 0
    assert events[1].raw_arguments_delta == "{}"
    assert events[2].tool_call_index == 0
    assert events[2].tool_call is not None
    assert events[2].tool_call.raw_arguments == "{}"
    assert events[2].tool_call.arguments == {}
    assert events[-1].kind is GatewayEventKind.COMPLETED


def test_stream_open_retry_reuses_stable_idempotency_before_commit() -> None:
    """Safe same-endpoint opening retries keep one identity before semantic output."""
    calls = 0
    identities: list[str] = []
    failed_stream = _ChunkStream(())
    success_stream = _ChunkStream(
        (
            b"".join(
                (
                    _sse(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": "ok"},
                                    "finish_reason": "stop",
                                }
                            ]
                        }
                    ),
                    b"data: [DONE]\n\n",
                )
            ),
        )
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        """Fail one opening attempt, then return a valid streaming response."""
        nonlocal calls
        calls += 1
        identities.append(request.headers["Idempotency-Key"])
        if calls == 1:
            return httpx.Response(503, stream=failed_stream)
        return httpx.Response(200, stream=success_stream)

    async def scenario() -> list[GatewayEvent]:
        """Consume the stream opened by the safe retry."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleClient(
                model=_snapshot("openai-compatible"),
                api_key="fixture-key",
                base_url="https://compatible.test/v1",
                transport=HttpxAsyncJsonTransport(http_client),
                retry_policy=RetryPolicy(
                    maximum_attempts=2,
                    initial_delay_seconds=0,
                    maximum_delay_seconds=0,
                ),
            )
            stream = await client.stream(
                _request(GatewayApiSurface.CHAT_COMPLETIONS),
                deadline=RequestDeadline.after(1),
                idempotency_key="stable-attempt",
            )
            return await _collect(stream)

    events = asyncio.run(scenario())

    assert calls == 2
    assert identities == ["stable-attempt", "stable-attempt"]
    assert [event.kind for event in events] == [
        GatewayEventKind.TEXT_DELTA,
        GatewayEventKind.COMPLETED,
    ]
    assert failed_stream.closed is True
    assert success_stream.closed is True


@pytest.mark.parametrize("failure_mode", ["first_byte", "read_disconnect", "malformed"])
def test_presemantic_body_failures_share_the_opening_retry_budget(failure_mode: str) -> None:
    """First-byte, read, and malformed pre-semantic failures safely reopen one endpoint."""
    if failure_mode == "first_byte":
        first_stream: _HangingStream | _ReadFailureStream | _ChunkStream = _HangingStream()
    elif failure_mode == "read_disconnect":
        first_stream = _ReadFailureStream()
    else:
        first_stream = _ChunkStream((b"data: not-json\n\n",))
    success_stream = _ChunkStream(
        (
            b"".join(
                (
                    _sse(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": "recovered"},
                                    "finish_reason": "stop",
                                }
                            ]
                        }
                    ),
                    b"data: [DONE]\n\n",
                )
            ),
        )
    )
    calls = 0
    identities: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return one pre-semantic failure followed by a valid stream."""
        nonlocal calls
        calls += 1
        identities.append(request.headers["Idempotency-Key"])
        return httpx.Response(200, stream=first_stream if calls == 1 else success_stream)

    async def scenario() -> list[GatewayEvent]:
        """Consume the successful retry without observing the failed attempt."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleClient(
                model=_snapshot("openai-compatible"),
                api_key="fixture-key",
                base_url="https://compatible.test/v1",
                transport=HttpxAsyncJsonTransport(http_client),
                retry_policy=RetryPolicy(
                    maximum_attempts=2,
                    initial_delay_seconds=0,
                    maximum_delay_seconds=0,
                ),
                timeout_seconds=0.02,
            )
            stream = await client.stream(
                _request(GatewayApiSurface.CHAT_COMPLETIONS),
                deadline=RequestDeadline.after(1),
                idempotency_key="stable-body-retry",
            )
            return await _collect(stream)

    events = asyncio.run(scenario())

    assert calls == 2
    assert identities == ["stable-body-retry", "stable-body-retry"]
    assert [event.kind for event in events] == [
        GatewayEventKind.TEXT_DELTA,
        GatewayEventKind.COMPLETED,
    ]
    if isinstance(first_stream, _HangingStream):
        assert first_stream.closed.is_set()
    else:
        assert first_stream.closed is True
    assert success_stream.closed is True


def test_exhausted_read_disconnect_is_transport_failure_and_closes_response() -> None:
    """A final body-read disconnect is sanitized as retryable transport, not internal."""
    upstream = _ReadFailureStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return one response whose body disconnects on the first read."""
        del request
        return httpx.Response(200, stream=upstream)

    async def scenario() -> GatewayEvent:
        """Consume the only allowed attempt through its normalized failure."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleClient(
                model=_snapshot("openai-compatible"),
                api_key="fixture-key",
                base_url="https://compatible.test/v1",
                transport=HttpxAsyncJsonTransport(http_client),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            stream = await client.stream(
                _request(GatewayApiSurface.CHAT_COMPLETIONS),
                deadline=RequestDeadline.after(1),
                idempotency_key="read-failure",
            )
            return await anext(stream)

    event = asyncio.run(scenario())

    assert event.kind is GatewayEventKind.FAILED
    assert event.failure is not None
    assert event.failure.failure_class is GatewayFailureClass.TRANSPORT
    assert event.failure.retryable_same_deployment is True
    assert "canary" not in event.failure.safe_message
    assert upstream.closed is True


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openai-compatible"])
def test_launch_adapters_emit_refusal_as_semantic_commit(provider: str) -> None:
    """Every launch adapter commits on its native refusal signal without leaking an error body."""
    if provider == "openai":
        frames = b"".join(
            (
                _sse({"type": "response.refusal.delta", "delta": "declined"}),
                _sse(
                    {
                        "type": "response.completed",
                        "response": {"status": "completed", "usage": None},
                    }
                ),
            )
        )
    elif provider == "anthropic":
        frames = b"".join(
            (
                _sse(
                    {
                        "type": "message_start",
                        "message": {
                            "usage": {
                                "input_tokens": 1,
                                "cache_read_input_tokens": 0,
                                "cache_creation_input_tokens": 0,
                            }
                        },
                    }
                ),
                _sse(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "refusal", "refusal": "declined"},
                    }
                ),
                _sse(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "refusal"},
                        "usage": {"output_tokens": 1},
                    }
                ),
                _sse({"type": "message_stop"}),
            )
        )
    else:
        frames = b"".join(
            (
                _sse(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"refusal": "declined"},
                                "finish_reason": "content_filter",
                            }
                        ]
                    }
                ),
                b"data: [DONE]\n\n",
            )
        )

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return the selected provider's native refusal fixture."""
        del request
        return httpx.Response(200, stream=_ChunkStream((frames,)))

    async def scenario() -> list[GatewayEvent]:
        """Read the refusal event before consuming its terminal continuation."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = _provider_client(provider, http_client)
            stream = await client.stream(
                _request(_provider_surface(provider)),
                deadline=RequestDeadline.after(1),
                idempotency_key=f"refusal-{provider}",
            )
            first = await anext(stream)
            return [first, *await _collect(stream)]

    events = asyncio.run(scenario())

    assert events[0].kind is GatewayEventKind.REFUSAL_DELTA
    assert events[-1].kind is GatewayEventKind.FAILED
    assert events[-1].failure is not None
    assert events[-1].failure.failure_class is GatewayFailureClass.REFUSAL


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openai-compatible"])
def test_launch_adapters_sanitize_native_stream_errors(provider: str) -> None:
    """Every launch adapter converts native error events without exposing provider content."""
    canary = "private-provider-error-canary"
    if provider == "openai":
        frames = _sse(
            {
                "type": "response.failed",
                "response": {"error": {"message": canary}},
            }
        )
    elif provider == "anthropic":
        frames = _sse(
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": canary},
            }
        )
    else:
        frames = _sse({"error": {"message": canary}})

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return the selected provider's native error fixture."""
        del request
        return httpx.Response(200, stream=_ChunkStream((frames,)))

    async def scenario() -> GatewayEvent:
        """Consume one sanitized provider failure."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = _provider_client(provider, http_client)
            stream = await client.stream(
                _request(_provider_surface(provider)),
                deadline=RequestDeadline.after(1),
                idempotency_key=f"error-{provider}",
            )
            return await anext(stream)

    event = asyncio.run(scenario())

    assert event.kind is GatewayEventKind.FAILED
    assert event.failure is not None
    assert event.failure.failure_class is GatewayFailureClass.PROVIDER_INTERNAL
    assert canary not in event.failure.safe_message


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openai-compatible"])
def test_launch_adapters_reject_duplicate_or_conflicting_tool_starts(provider: str) -> None:
    """Tool accumulator identity cannot be overwritten or spliced after commitment."""
    if provider == "openai":
        start = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": "",
            },
        }
        frames = _sse(start) + _sse(start)
    elif provider == "anthropic":
        start = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call-1",
                "name": "lookup",
                "input": {},
            },
        }
        frames = _sse(start) + _sse(start)
    else:
        frames = b"".join(
            (
                _sse(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call-1",
                                            "function": {
                                                "name": "lookup",
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
                _sse(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call-conflict",
                                            "function": {"name": "lookup"},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
            )
        )
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return malformed repeated tool metadata from one provider attempt."""
        nonlocal calls
        del request
        calls += 1
        return httpx.Response(200, stream=_ChunkStream((frames,)))

    async def scenario() -> list[GatewayEvent]:
        """Consume the tool start and its terminal malformed-response failure."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = _provider_client(provider, http_client)
            stream = await client.stream(
                _request(_provider_surface(provider)),
                deadline=RequestDeadline.after(1),
                idempotency_key=f"duplicate-{provider}",
            )
            return await _collect(stream)

    events = asyncio.run(scenario())

    assert calls == 1
    assert [event.kind for event in events] == [
        GatewayEventKind.TOOL_CALL_STARTED,
        GatewayEventKind.FAILED,
    ]
    assert events[-1].failure is not None
    assert events[-1].failure.failure_class is GatewayFailureClass.MALFORMED_RESPONSE


def test_first_byte_deadline_returns_sanitized_timeout_and_closes_upstream() -> None:
    """A spent first-byte phase fails before commitment and closes active HTTP work."""
    upstream = _HangingStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return a response whose first body byte never arrives."""
        del request
        return httpx.Response(200, stream=upstream)

    async def scenario() -> GatewayEvent:
        """Wait through one bounded first-byte phase."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleClient(
                model=_snapshot("openai-compatible"),
                api_key="fixture-key",
                base_url="https://compatible.test/v1",
                transport=HttpxAsyncJsonTransport(http_client),
                timeout_seconds=0.02,
            )
            stream = await client.stream(
                _request(GatewayApiSurface.CHAT_COMPLETIONS),
                deadline=RequestDeadline.after(0.02),
                idempotency_key="attempt-timeout",
            )
            return await anext(stream)

    event = asyncio.run(scenario())

    assert event.kind is GatewayEventKind.FAILED
    assert event.failure is not None
    assert event.failure.failure_class is GatewayFailureClass.TIMEOUT
    assert upstream.closed.is_set()


def test_sse_size_limit_tracks_one_multiline_event_across_chunks() -> None:
    """Accumulated data lines cannot evade the per-event size limit through fragmentation."""
    half = _MAXIMUM_SSE_EVENT_BYTES // 2 + 16
    upstream = _ChunkStream(
        (
            b"data: " + b"a" * half + b"\n",
            b"data: " + b"b" * half + b"\n\n",
        )
    )

    async def scenario() -> None:
        """Consume the oversized logical event and require focused rejection."""
        decoder = _SseDecoder(
            upstream,
            deadline=RequestDeadline.after(1),
            phase_timeout_seconds=1,
        )
        with pytest.raises(ProviderResponseError, match="size limit"):
            async for _event in decoder.events():
                pass

    asyncio.run(scenario())


def test_sse_size_limit_allows_one_large_chunk_of_small_events() -> None:
    """Socket batching does not reject many independently bounded keepalive frames."""
    keepalive = b": " + b"x" * (_MAXIMUM_SSE_EVENT_BYTES // 4) + b"\n\n"
    upstream = _ChunkStream((keepalive * 5 + b"data: {}\n\n",))

    async def scenario() -> list[str]:
        """Decode the only data-bearing event after the large keepalive batch."""
        decoder = _SseDecoder(
            upstream,
            deadline=RequestDeadline.after(5),
            phase_timeout_seconds=5,
        )
        return [event.data async for event in decoder.events()]

    assert asyncio.run(scenario()) == ["{}"]


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openai-compatible"])
def test_explicit_cancellation_closes_each_launch_stream_within_bound(provider: str) -> None:
    """Client disconnect cancellation closes each launch provider response immediately."""
    upstream = _HangingStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return one open response that can only finish through cancellation."""
        del request
        return httpx.Response(200, stream=upstream)

    async def scenario() -> None:
        """Cancel before reading and prove the response lifecycle is closed."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = _provider_client(provider, http_client)
            stream = await client.stream(
                _request(_provider_surface(provider)),
                deadline=RequestDeadline.after(1),
                idempotency_key=f"cancel-{provider}",
            )
            await stream.cancel()

    asyncio.run(scenario())

    assert upstream.closed.is_set()
