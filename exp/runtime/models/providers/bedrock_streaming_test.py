"""Native Bedrock EventStream normalization and bounded-worker certification."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator, Mapping

import pytest

from exp.common.models import BillingSource, ModelSnapshot
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEventKind,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.models.providers.async_transport import RequestDeadline
from exp.runtime.models.providers.bedrock import BedrockClient, BoundedBedrockClient
from exp.runtime.models.providers.transport import ProviderTransportError, RetryPolicy


class _EventStream:
    """Expose scripted synchronous Bedrock events and a close method."""

    def __init__(self, events: tuple[Mapping[str, object], ...]) -> None:
        """Store events in provider order."""
        self._events = iter(events)
        self.closed = False

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        """Return this one-pass synchronous iterator."""
        return self

    def __next__(self) -> Mapping[str, object]:
        """Return the next scripted provider event."""
        return next(self._events)

    def close(self) -> None:
        """Record closure of the EventStream response."""
        self.closed = True


class _HangingEventStream:
    """Block one EventStream read until close interrupts it."""

    def __init__(self) -> None:
        """Initialize thread-safe start and close signals."""
        self.started = threading.Event()
        self.closed = threading.Event()

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        """Return this blocking iterator."""
        return self

    def __next__(self) -> Mapping[str, object]:
        """Wait for close, then end the provider stream."""
        self.started.set()
        self.closed.wait()
        raise StopIteration

    def close(self) -> None:
        """Release the pending EventStream read."""
        self.closed.set()


class _Runtime:
    """Return injected EventStreams without contacting AWS."""

    def __init__(self, streams: list[_EventStream | _HangingEventStream | BaseException]) -> None:
        """Retain one stream or opening failure per expected request."""
        self._streams = streams
        self.stream_calls: list[Mapping[str, object]] = []

    def converse(self, **request: object) -> Mapping[str, object]:
        """Reject non-streaming completion in this fixture."""
        del request
        raise AssertionError("test made an unexpected Converse request")

    def converse_stream(self, **request: object) -> Mapping[str, object]:
        """Return or raise the next scripted EventStream opening result."""
        self.stream_calls.append(request)
        outcome = self._streams.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return {"stream": outcome}

    def invoke_model(self, **request: object) -> Mapping[str, object]:
        """Reject embeddings in this fixture."""
        del request
        raise AssertionError("test made an unexpected InvokeModel request")


def test_bedrock_stream_normalizes_text_tools_usage_and_terminal_state() -> None:
    """Bedrock events preserve raw tool fragments and cache-aware usage."""

    async def scenario() -> None:
        """Consume one complete native Bedrock stream."""
        upstream = _EventStream(
            (
                {"messageStart": {"role": "assistant"}},
                {
                    "contentBlockStart": {
                        "contentBlockIndex": 0,
                        "start": {},
                    }
                },
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": "hello"},
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {
                    "contentBlockStart": {
                        "contentBlockIndex": 1,
                        "start": {
                            "toolUse": {
                                "toolUseId": "tool-one",
                                "name": "lookup",
                            }
                        },
                    }
                },
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 1,
                        "delta": {"toolUse": {"input": '{"city":'}},
                    }
                },
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 1,
                        "delta": {"toolUse": {"input": '"Zürich"}'}},
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 1}},
                {"messageStop": {"stopReason": "tool_use"}},
                {
                    "metadata": {
                        "usage": {
                            "inputTokens": 5,
                            "outputTokens": 3,
                            "cacheReadInputTokens": 2,
                            "cacheWriteInputTokens": 1,
                        }
                    }
                },
            )
        )
        runtime = _Runtime([upstream])
        client = _client(runtime)
        stream = await client.stream(
            _request(),
            deadline=RequestDeadline.after(10),
            idempotency_key="deployment-operation",
            retry_policy=RetryPolicy(1, 0, 0),
        )
        events = [event async for event in stream]

        assert [event.kind for event in events] == [
            GatewayEventKind.TEXT_DELTA,
            GatewayEventKind.TOOL_CALL_STARTED,
            GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            GatewayEventKind.TOOL_CALL_COMPLETED,
            GatewayEventKind.USAGE,
            GatewayEventKind.COMPLETED,
        ]
        assert events[2].raw_arguments_delta == '{"city":'
        assert events[4].tool_call_index == 1
        assert events[4].tool_call is not None
        assert events[4].tool_call.raw_arguments == '{"city":"Zürich"}'
        assert events[5].usage is not None
        assert events[5].usage.input_tokens == 8
        assert events[5].usage.cached_input_tokens == 2
        assert runtime.stream_calls[0]["modelId"] == "exact-model"
        assert upstream.closed

    asyncio.run(scenario())


def test_bedrock_empty_tool_input_delta_is_skipped() -> None:
    """A leading empty toolUse input fragment produces no delta event and no failure."""

    async def scenario() -> None:
        """Consume one tool-call stream whose first input fragment is empty."""
        upstream = _EventStream(
            (
                {"messageStart": {"role": "assistant"}},
                {
                    "contentBlockStart": {
                        "contentBlockIndex": 0,
                        "start": {
                            "toolUse": {
                                "toolUseId": "tool-empty",
                                "name": "write",
                            }
                        },
                    }
                },
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"toolUse": {"input": ""}},
                    }
                },
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"toolUse": {"input": '{"path":"fib.py"}'}},
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "tool_use"}},
                {
                    "metadata": {
                        "usage": {
                            "inputTokens": 4,
                            "outputTokens": 2,
                            "cacheReadInputTokens": 0,
                            "cacheWriteInputTokens": 0,
                        }
                    }
                },
            )
        )
        stream = await _client(_Runtime([upstream])).stream(
            _request(),
            deadline=RequestDeadline.after(10),
            idempotency_key="empty-fragment-operation",
            retry_policy=RetryPolicy(1, 0, 0),
        )
        events = [event async for event in stream]

        assert [event.kind for event in events] == [
            GatewayEventKind.TOOL_CALL_STARTED,
            GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            GatewayEventKind.TOOL_CALL_COMPLETED,
            GatewayEventKind.USAGE,
            GatewayEventKind.COMPLETED,
        ]
        assert events[1].raw_arguments_delta == '{"path":"fib.py"}'
        assert events[2].tool_call_index == 0
        assert events[2].tool_call is not None
        assert events[2].tool_call.raw_arguments == '{"path":"fib.py"}'

    asyncio.run(scenario())


def test_bedrock_guardrail_stop_is_typed_and_precommit() -> None:
    """A content-free guardrail stop stays eligible only for explicit outer policy."""

    async def scenario() -> None:
        """Consume one refusal without semantic output."""
        upstream = _EventStream(
            (
                {"messageStop": {"stopReason": "guardrail_intervened"}},
                {
                    "metadata": {
                        "usage": {
                            "inputTokens": 1,
                            "outputTokens": 0,
                            "cacheReadInputTokens": 0,
                            "cacheWriteInputTokens": 0,
                        }
                    }
                },
            )
        )
        stream = await _client(_Runtime([upstream])).stream(
            _request(),
            deadline=RequestDeadline.after(10),
            idempotency_key="refusal-operation",
            retry_policy=RetryPolicy(1, 0, 0),
        )
        events = [event async for event in stream]

        assert [event.kind for event in events] == [
            GatewayEventKind.USAGE,
            GatewayEventKind.FAILED,
        ]
        assert events[-1].failure is not None
        assert events[-1].failure.failure_class is GatewayFailureClass.REFUSAL
        assert not stream.committed

    asyncio.run(scenario())


def test_bedrock_cancel_closes_blocking_read_before_releasing_worker() -> None:
    """Cancellation closes EventStream and preserves the bounded-worker permit contract."""

    async def scenario() -> None:
        """Cancel one blocked read and wait for its worker to exit."""
        upstream = _HangingEventStream()
        client = _client(_Runtime([upstream]))
        stream = await client.stream(
            _request(),
            deadline=RequestDeadline.after(10),
            idempotency_key="cancel-operation",
            retry_policy=RetryPolicy(1, 0, 0),
        )
        pending = asyncio.create_task(anext(stream))
        assert await asyncio.to_thread(upstream.started.wait, 1)
        await stream.cancel()
        result = (await asyncio.gather(pending, return_exceptions=True))[0]
        await asyncio.sleep(0)

        assert isinstance(result, StopAsyncIteration)
        assert upstream.closed.is_set()
        assert client._permits._value == 1  # noqa: SLF001 - worker-bound regression

    asyncio.run(scenario())


def test_bedrock_single_dispatch_disables_internal_opening_retry() -> None:
    """A caller-owned one-attempt policy prevents a hidden boto opening retry."""

    async def scenario() -> None:
        """Fail one stream opening and prove no second physical call occurs."""
        runtime = _Runtime(
            [
                ProviderTransportError("provider unavailable", status_code=503),
                _EventStream(()),
            ]
        )
        client = _client(runtime)

        with pytest.raises(ProviderTransportError):
            await client.stream(
                _request(),
                deadline=RequestDeadline.after(10),
                idempotency_key="single-dispatch-operation",
                retry_policy=RetryPolicy(1, 0, 0),
            )
        await asyncio.sleep(0)

        assert len(runtime.stream_calls) == 1
        assert client._permits._value == 1  # noqa: SLF001 - opening-failure regression

    asyncio.run(scenario())


def test_bedrock_cancelled_open_closes_late_response_before_releasing_worker() -> None:
    """A cancelled opening worker cannot leak its eventual EventStream response."""

    class _DelayedRuntime(_Runtime):
        """Block ConverseStream opening until the caller has cancelled its wait."""

        def __init__(self, upstream: _EventStream) -> None:
            """Retain the late response and thread-safe coordination signals."""
            super().__init__([upstream])
            self.started = threading.Event()
            self.released = threading.Event()

        def converse_stream(self, **request: object) -> Mapping[str, object]:
            """Wait for test release, then return the scripted late response."""
            self.started.set()
            self.released.wait()
            return super().converse_stream(**request)

    async def scenario() -> None:
        """Cancel opening, release its worker, and observe close-before-permit behavior."""
        upstream = _EventStream(())
        runtime = _DelayedRuntime(upstream)
        client = _client(runtime)
        opening = asyncio.create_task(
            client.stream(
                _request(),
                deadline=RequestDeadline.after(10),
                idempotency_key="cancelled-open-operation",
                retry_policy=RetryPolicy(1, 0, 0),
            )
        )
        assert await asyncio.to_thread(runtime.started.wait, 1)
        opening.cancel()
        await asyncio.gather(opening, return_exceptions=True)
        assert client._permits._value == 0  # noqa: SLF001 - detached-worker regression
        runtime.released.set()
        for _ in range(100):
            if upstream.closed and client._permits._value == 1:  # noqa: SLF001
                break
            await asyncio.sleep(0.01)

        assert upstream.closed
        assert client._permits._value == 1  # noqa: SLF001 - cleanup completion regression

    asyncio.run(scenario())


def _client(runtime: _Runtime) -> BoundedBedrockClient:
    """Construct one bounded native Bedrock client over the injected runtime."""
    return BoundedBedrockClient(
        BedrockClient(
            model=ModelSnapshot(
                provider="bedrock",
                model_id="exact-model",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                capabilities_sha256="a" * 64,
                connection_sha256="b" * 64,
            ),
            region="us-east-1",
            environment={},
            runtime_factory=lambda *, region_name: runtime,
        ),
        maximum_outstanding_calls=1,
    )


def _request() -> GatewayRequest:
    """Build one canonical Bedrock streaming request."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
        stream=True,
        include_usage=True,
    )
