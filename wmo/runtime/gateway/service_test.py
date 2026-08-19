"""Gateway service regressions for keyed replay boundaries and Responses instructions."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import cast

import pytest
from fastapi.responses import StreamingResponse

from wmo.runtime.gateway.aggregation import BoundedGatewayEvents
from wmo.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from wmo.runtime.gateway.tests.data_plane_test import (
    _BlockingStream,
    _ControlStore,
    _EventStream,
    _Provider,
    _service,
)
from wmo.runtime.openai_protocol.errors import OpenAIProtocolError
from wmo.runtime.openai_protocol.requests import decode_chat, decode_responses


def _completed_events() -> _EventStream:
    """Return one short successful provider event stream."""
    return _EventStream(
        (
            GatewayEvent(
                kind=GatewayEventKind.TEXT_DELTA,
                sequence_number=0,
                text_delta="hello",
            ),
            GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1),
        )
    )


def _failed_events() -> _EventStream:
    """Return one provider stream that fails after visible committed text."""
    return _EventStream(
        (
            GatewayEvent(
                kind=GatewayEventKind.TEXT_DELTA,
                sequence_number=0,
                text_delta="partial",
            ),
            GatewayEvent(
                kind=GatewayEventKind.FAILED,
                sequence_number=1,
                failure=GatewayFailure(
                    failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
                    safe_message="provider failed mid-stream",
                ),
            ),
        )
    )


async def _consume(response: StreamingResponse) -> bytes:
    """Consume one public streaming response into its exact emitted bytes."""
    frames: list[bytes] = []
    async for frame in cast(AsyncIterator[bytes], response.body_iterator):
        frames.append(frame)
    return b"".join(frames)


class _SleepingControlStore:
    """Delegate authority calls after a blocking sleep that would stall a shared loop."""

    def __init__(self, inner: _ControlStore, *, sleep_seconds: float) -> None:
        """Wrap one deterministic control store with a synchronous delay."""
        self._inner = inner
        self._sleep_seconds = sleep_seconds

    def authenticate_key(self, *, raw_key: str) -> None:
        """Block synchronously, then delegate authentication."""
        time.sleep(self._sleep_seconds)
        self._inner.authenticate_key(raw_key=raw_key)

    def granted_aliases(self, *, raw_key: str) -> tuple[str, ...]:
        """Block synchronously, then delegate alias discovery."""
        time.sleep(self._sleep_seconds)
        return self._inner.granted_aliases(raw_key=raw_key)

    def authorize_request(
        self,
        *,
        raw_key: str,
        alias: str,
        request: GatewayRequest,
        deadline_monotonic: float,
    ) -> AuthorizationSnapshot:
        """Block synchronously, then delegate authorization."""
        time.sleep(self._sleep_seconds)
        return self._inner.authorize_request(
            raw_key=raw_key,
            alias=alias,
            request=request,
            deadline_monotonic=deadline_monotonic,
        )


def test_blocking_store_calls_run_off_the_event_loop() -> None:
    """Synchronous authority and ledger work cannot stall concurrent loop progress."""

    async def scenario() -> None:
        """Heartbeat a sibling coroutine while one request blocks in the control store."""
        provider = _Provider(_completed_events)
        service, control, _ledger, _proof = _service(provider)
        service._control = _SleepingControlStore(  # noqa: SLF001 - inject the blocking seam
            control,
            sleep_seconds=0.2,
        )
        ticks = 0

        async def heartbeat() -> None:
            """Count loop turns available while the store call blocks its thread."""
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        beat = asyncio.create_task(heartbeat())
        try:
            response = await service.complete(
                raw_key="caller-secret",
                decoded=decode_chat(
                    {
                        "model": "public-model",
                        "messages": [{"role": "user", "content": "hello"}],
                    }
                ),
            )
        finally:
            beat.cancel()
            await asyncio.gather(beat, return_exceptions=True)

        assert response.status_code == 200
        assert ticks >= 5

    asyncio.run(scenario())


def test_streaming_chat_retains_no_events_and_cannot_trip_aggregation_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat streams pass events through without retention, so bounded retention never trips."""

    class _ExplodingEvents(BoundedGatewayEvents):
        """Fail the test on any retention attempt during a chat stream."""

        def append(self, event: GatewayEvent) -> None:
            """Reject retention outright."""
            del event
            raise AssertionError("chat streams must not retain events")

    monkeypatch.setattr("wmo.runtime.gateway.service.BoundedGatewayEvents", _ExplodingEvents)

    async def scenario() -> None:
        """Stream one chat completion and prove every frame is emitted unretained."""
        provider = _Provider(
            lambda: _EventStream(
                (
                    *(
                        GatewayEvent(
                            kind=GatewayEventKind.TEXT_DELTA,
                            sequence_number=index,
                            text_delta="delta-payload",
                        )
                        for index in range(16)
                    ),
                    GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=16),
                )
            )
        )
        service, _control, _ledger, _proof = _service(provider)
        response = await service.complete(
            raw_key="caller-secret",
            decoded=decode_chat(
                {
                    "model": "public-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                }
            ),
        )
        assert isinstance(response, StreamingResponse)
        emitted = bytearray()
        async for frame in cast(AsyncIterator[bytes], response.body_iterator):
            emitted.extend(frame)

        assert emitted.count(b"delta-payload") == 16
        assert b"[DONE]" in emitted

    asyncio.run(scenario())


def test_keyed_failed_stream_is_not_cached_and_a_retry_redispatches() -> None:
    """A mid-stream failure abandons the keyed lease instead of replaying the failure."""

    async def scenario() -> None:
        """Fail one keyed stream, then prove the retry executes and replay caches success."""
        outcomes = iter((_failed_events, _completed_events, _completed_events))
        provider = _Provider(lambda: next(outcomes)())
        service, _control, _ledger, _proof = _service(provider)
        decoded = decode_chat(
            {
                "model": "public-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            idempotency_key="op-failed-stream",
        )

        failed = await service.complete(raw_key="caller-secret", decoded=decoded)
        assert isinstance(failed, StreamingResponse)
        emitted = await _consume(failed)
        assert b'"error"' in emitted
        assert len(provider.streams) == 1

        retried = await service.complete(raw_key="caller-secret", decoded=decoded)
        assert isinstance(retried, StreamingResponse)
        retried_body = await _consume(retried)
        assert b"hello" in retried_body
        assert b'"error"' not in retried_body
        assert len(provider.streams) == 2

        replay = await service.complete(raw_key="caller-secret", decoded=decoded)
        assert replay.status_code == 200
        assert b"hello" in replay.body
        assert b'"error"' not in replay.body
        assert len(provider.streams) == 2

    asyncio.run(scenario())


def test_keyed_joiner_wait_is_bounded_by_the_request_deadline() -> None:
    """A duplicate keyed request cannot outlive its budget waiting on a stalled owner."""

    async def scenario() -> None:
        """Stall the keyed owner and prove the joiner fails with the replay 409."""
        provider = _Provider(lambda: _BlockingStream())
        service, _control, _ledger, _proof = _service(provider, request_timeout_seconds=0.05)
        decoded = decode_chat(
            {
                "model": "public-model",
                "messages": [{"role": "user", "content": "hold"}],
                "stream": True,
            },
            idempotency_key="op-stalled-owner",
        )

        owner = await service.complete(raw_key="caller-secret", decoded=decoded)
        assert isinstance(owner, StreamingResponse)
        with pytest.raises(OpenAIProtocolError) as captured:
            await service.complete(raw_key="caller-secret", decoded=decoded)
        assert captured.value.status_code == 409
        assert captured.value.detail.code == "idempotency_replay_unavailable"
        assert len(provider.streams) == 1
        await cast(AsyncGenerator[bytes, None], owner.body_iterator).aclose()

    asyncio.run(scenario())


def test_client_request_id_alone_never_replays_or_deduplicates() -> None:
    """X-Client-Request-Id is correlation only, so identical requests each dispatch."""

    async def scenario() -> None:
        """Send one correlation id twice and prove both requests execute."""
        provider = _Provider(_completed_events)
        service, _control, _ledger, _proof = _service(provider)
        decoded = decode_chat(
            {
                "model": "public-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
            client_request_id="observability-run-42",
        )

        first = await service.complete(raw_key="caller-secret", decoded=decoded)
        second = await service.complete(raw_key="caller-secret", decoded=decoded)

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.headers["x-client-request-id"] == "observability-run-42"
        assert second.headers["x-client-request-id"] == "observability-run-42"
        assert len(provider.streams) == 2

    asyncio.run(scenario())


def test_responses_instructions_are_per_turn_and_echoed_without_inheritance() -> None:
    """Each turn's instructions lead the provider context once and never accumulate."""

    async def scenario() -> None:
        """Run three continuation turns with changing then absent instructions."""
        provider = _Provider(_completed_events)
        service, _control, _ledger, _proof = _service(provider)

        first = await service.complete(
            raw_key="caller-secret",
            decoded=decode_responses(
                {"model": "public-model", "input": "hi", "instructions": "Be terse."}
            ),
        )
        first_body = json.loads(bytes(first.body))
        assert first_body["instructions"] == "Be terse."
        assert tuple(message.role for message in provider.requests[0].messages) == (
            "developer",
            "user",
        )
        assert provider.requests[0].messages[0].content == "Be terse."

        second = await service.complete(
            raw_key="caller-secret",
            decoded=decode_responses(
                {
                    "model": "public-model",
                    "input": "again",
                    "instructions": "Be verbose.",
                    "previous_response_id": first_body["id"],
                }
            ),
        )
        second_body = json.loads(bytes(second.body))
        assert second_body["instructions"] == "Be verbose."
        second_messages = provider.requests[1].messages
        assert tuple(message.role for message in second_messages) == (
            "developer",
            "user",
            "assistant",
            "user",
        )
        assert second_messages[0].content == "Be verbose."
        assert all(message.content != "Be terse." for message in second_messages)

        third = await service.complete(
            raw_key="caller-secret",
            decoded=decode_responses(
                {
                    "model": "public-model",
                    "input": "more",
                    "previous_response_id": second_body["id"],
                }
            ),
        )
        third_body = json.loads(bytes(third.body))
        assert third_body["instructions"] is None
        third_messages = provider.requests[2].messages
        assert tuple(message.role for message in third_messages) == (
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        )
        assert all(message.role != "developer" for message in third_messages)

    asyncio.run(scenario())


def test_client_disconnect_still_records_a_cancelled_terminal() -> None:
    """Repeated caller cancellation never loses the durable cancelled terminal write.

    ASGI servers redeliver cancellation at every await point while tearing down a
    disconnected response task, so the settlement path must complete its ledger
    write and upstream cancellation in a detached task.
    """

    async def scenario() -> None:
        """Cancel a parked streaming consumer until it dies, then check accounting."""
        provider = _Provider(_BlockingStream)
        service, _control, ledger, _proof = _service(provider)
        response = await service.complete(
            raw_key="caller-secret",
            decoded=decode_chat(
                {
                    "model": "public-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                }
            ),
        )
        assert isinstance(response, StreamingResponse)
        iterator = cast(AsyncIterator[bytes], response.body_iterator)

        async def consume() -> None:
            """Read frames until the caller is torn down."""
            async for _frame in iterator:
                pass

        consumer = asyncio.create_task(consume())
        stream = None
        while stream is None:
            await asyncio.sleep(0)
            stream = provider.streams[0] if provider.streams else None
        assert isinstance(stream, _BlockingStream)
        await stream.entered.wait()
        assert not ledger.finished
        while not consumer.done():
            consumer.cancel()
            await asyncio.sleep(0)
        assert consumer.cancelled()
        for _ in range(200):
            if ledger.finished and stream.cancelled:
                break
            await asyncio.sleep(0.005)
        assert stream.cancelled
        terminal, failure = ledger.finished[-1]
        assert failure is not None
        assert failure.failure_class is GatewayFailureClass.CANCELLED
        assert terminal is not None
        assert terminal.kind is GatewayEventKind.FAILED

    asyncio.run(scenario())
