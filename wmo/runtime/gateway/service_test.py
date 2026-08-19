"""Gateway service regressions for keyed replay boundaries and Responses instructions."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import cast

import pytest
from fastapi.responses import StreamingResponse
from wmo.runtime.gateway.data_plane_test import (
    _BlockingStream,
    _EventStream,
    _Provider,
    _service,
)

from wmo.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
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
