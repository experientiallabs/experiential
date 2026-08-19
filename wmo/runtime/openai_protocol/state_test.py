"""Tests for bounded, namespaced, opt-in protocol state."""

from __future__ import annotations

import asyncio

import pytest

from wmo.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage
from wmo.runtime.openai_protocol.errors import OpenAIProtocolError
from wmo.runtime.openai_protocol.state import (
    BoundedContinuationStore,
    BoundedReplayStore,
    CachedResponse,
    ContinuationState,
    ProtocolNamespace,
    ReplayClaimKind,
    ReplayKey,
    episode_namespace,
    replay_key,
)

_REQUEST_DIGEST = "a" * 64


def _namespace(identity: str = "identity-one", revision: str = "revision-one") -> ProtocolNamespace:
    """Create one tenant and immutable alias-revision state namespace."""
    return ProtocolNamespace(
        organization_id="org-one",
        identity_id=identity,
        alias_revision_id=revision,
    )


def test_opt_in_replay_joins_inflight_replays_completion_and_rejects_body_drift() -> None:
    """One keyed operation has one owner, exact replay, and body-conflict rejection."""

    async def scenario() -> None:
        """Exercise owner, join, replay, and conflict claims in one event loop."""
        store = BoundedReplayStore(capacity=4, byte_cap=1_024, ttl_seconds=60)
        key = replay_key(
            namespace=_namespace(),
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            caller_operation="operation-one",
            canonical_request_sha256=_REQUEST_DIGEST,
        )
        assert key is not None
        owner = await store.claim(key)
        join = await store.claim(key)
        assert owner.kind == ReplayClaimKind.OWNER
        assert join.kind == ReplayClaimKind.JOIN
        response = CachedResponse(
            status_code=200,
            media_type="application/json",
            body=b'{"id":"response-one"}',
        )
        await owner.complete(response)
        assert await join.result() == response
        await owner.abandon()
        replay = await store.claim(key)
        assert replay.kind == ReplayClaimKind.REPLAY
        assert await replay.result() == response

        conflict = replay_key(
            namespace=_namespace(),
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            caller_operation="operation-one",
            canonical_request_sha256="b" * 64,
        )
        assert conflict is not None
        with pytest.raises(OpenAIProtocolError) as captured:
            await store.claim(conflict)
        assert captured.value.detail.code == "idempotency_conflict"

    asyncio.run(scenario())


def test_unkeyed_requests_do_not_deduplicate_and_episode_keys_are_namespaced() -> None:
    """Absent opt-in yields no replay key and equal raw keys cannot cross identities or aliases."""
    assert (
        replay_key(
            namespace=_namespace(),
            surface=GatewayApiSurface.RESPONSES,
            caller_operation=None,
            canonical_request_sha256=_REQUEST_DIGEST,
        )
        is None
    )
    first = episode_namespace(
        namespace=_namespace(), caller_episode_key="same", request_id="request-one"
    )
    other_identity = episode_namespace(
        namespace=_namespace(identity="identity-two"),
        caller_episode_key="same",
        request_id="request-two",
    )
    other_revision = episode_namespace(
        namespace=_namespace(revision="revision-two"),
        caller_episode_key="same",
        request_id="request-three",
    )
    assert first != other_identity
    assert first != other_revision
    assert first[-1] == other_identity[-1] == other_revision[-1]


def test_joiner_result_is_bounded_and_owner_abandonment_maps_to_replay_unavailable() -> None:
    """A joiner neither waits past its bound nor surfaces owner abandonment as cancellation."""

    async def scenario() -> None:
        """Time out one stalled join, then abandon the owner under a live join."""
        store = BoundedReplayStore(capacity=4, byte_cap=1_024, ttl_seconds=60)
        key = replay_key(
            namespace=_namespace(),
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            caller_operation="operation-slow",
            canonical_request_sha256=_REQUEST_DIGEST,
        )
        assert key is not None
        owner = await store.claim(key)
        join = await store.claim(key)
        with pytest.raises(OpenAIProtocolError) as timed_out:
            await join.result(timeout_seconds=0.01)
        assert timed_out.value.detail.code == "idempotency_replay_unavailable"
        assert timed_out.value.status_code == 409

        waiter = asyncio.create_task(join.result(timeout_seconds=5.0))
        await asyncio.sleep(0)
        await owner.abandon()
        with pytest.raises(OpenAIProtocolError) as abandoned:
            await waiter
        assert abandoned.value.detail.code == "idempotency_replay_unavailable"
        assert abandoned.value.status_code == 409

    asyncio.run(scenario())


def test_replay_capacity_rejects_new_work_without_evicting_inflight_owners() -> None:
    """A full in-flight window fails closed instead of exceeding its count bound."""

    async def scenario() -> None:
        """Hold the only slot in flight and prove a distinct operation is rejected."""
        store = BoundedReplayStore(capacity=1, byte_cap=1_024, ttl_seconds=60)
        first = replay_key(
            namespace=_namespace(),
            surface=GatewayApiSurface.RESPONSES,
            caller_operation="operation-one",
            canonical_request_sha256=_REQUEST_DIGEST,
        )
        second = replay_key(
            namespace=_namespace(),
            surface=GatewayApiSurface.RESPONSES,
            caller_operation="operation-two",
            canonical_request_sha256="b" * 64,
        )
        assert first is not None and second is not None
        owner = await store.claim(first)
        with pytest.raises(OpenAIProtocolError) as captured:
            await store.claim(second)
        assert captured.value.detail.code == "gateway_overloaded"
        await owner.abandon()

    asyncio.run(scenario())


def test_cancelled_replay_joiner_does_not_cancel_shared_ownership() -> None:
    """Cancelling one waiter preserves owner publication and later exact replay."""

    async def scenario() -> None:
        """Cancel a join, publish through the owner, and replay the result."""
        store = BoundedReplayStore(capacity=2, byte_cap=1_024, ttl_seconds=60)
        key = replay_key(
            namespace=_namespace(),
            surface=GatewayApiSurface.RESPONSES,
            caller_operation="operation-one",
            canonical_request_sha256=_REQUEST_DIGEST,
        )
        assert key is not None
        owner = await store.claim(key)
        joiner = await store.claim(key)
        waiting = asyncio.create_task(joiner.result())
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        response = CachedResponse(
            status_code=200,
            media_type="application/json",
            body=b'{"id":"response-one"}',
        )
        await owner.complete(response)
        replay = await store.claim(key)
        assert replay.kind == ReplayClaimKind.REPLAY
        assert await replay.result() == response

    asyncio.run(scenario())


def test_abandoned_replay_owner_wakes_joiners_with_unavailable_error() -> None:
    """Owner cancellation releases joiners through a defined fail-closed result."""

    async def scenario() -> None:
        """Abandon one owner and prove its joiner can fail without cancellation."""
        store = BoundedReplayStore(capacity=2, byte_cap=1_024, ttl_seconds=60)
        key = replay_key(
            namespace=_namespace(),
            surface=GatewayApiSurface.RESPONSES,
            caller_operation="operation-one",
            canonical_request_sha256=_REQUEST_DIGEST,
        )
        assert key is not None
        owner = await store.claim(key)
        joiner = await store.claim(key)
        waiting = asyncio.create_task(joiner.result())
        await asyncio.sleep(0)
        await owner.abandon()

        with pytest.raises(OpenAIProtocolError) as captured:
            await waiting
        assert captured.value.detail.code == "idempotency_replay_unavailable"
        replacement = await store.claim(key)
        assert replacement.kind == ReplayClaimKind.OWNER
        await replacement.abandon()

    asyncio.run(scenario())


def test_replay_rejects_unretainable_response_and_releases_ownership() -> None:
    """An oversized result fails closed and leaves no false completed replay."""

    async def scenario() -> None:
        """Reject one publication, then prove the operation can be owned again."""
        store = BoundedReplayStore(capacity=1, byte_cap=8, ttl_seconds=60)
        key = replay_key(
            namespace=_namespace(),
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            caller_operation="operation-one",
            canonical_request_sha256=_REQUEST_DIGEST,
        )
        assert key is not None
        owner = await store.claim(key)
        with pytest.raises(OpenAIProtocolError) as captured:
            await owner.complete(
                CachedResponse(
                    status_code=200,
                    media_type="application/json",
                    body=b"response exceeds the bound",
                )
            )
        assert captured.value.detail.code == "idempotency_replay_unavailable"
        replacement = await store.claim(key)
        assert replacement.kind == ReplayClaimKind.OWNER
        await replacement.abandon()

    asyncio.run(scenario())


def test_continuation_is_bounded_namespaced_and_restart_unavailable() -> None:
    """Responses history obeys identity, alias revision, capacity, TTL, and restart boundaries."""

    async def scenario() -> None:
        """Store and resolve one continuation while proving every isolation boundary."""
        now = [100.0]

        def clock() -> float:
            """Return the controllable monotonic test time."""
            return now[0]

        store = BoundedContinuationStore(
            capacity=1,
            byte_cap=4_096,
            ttl_seconds=10,
            clock=clock,
        )
        state = ContinuationState(
            episode_key="c" * 64,
            messages=(GatewayMessage(role="user", content="retained only in memory"),),
        )
        await store.remember(namespace=_namespace(), response_id="resp_one", state=state)
        assert await store.resolve(namespace=_namespace(), previous_response_id="resp_one") == state

        with pytest.raises(OpenAIProtocolError, match="unavailable or expired"):
            await store.resolve(
                namespace=_namespace(identity="identity-two"),
                previous_response_id="resp_one",
            )
        await store.remember(namespace=_namespace(), response_id="resp_two", state=state)
        with pytest.raises(OpenAIProtocolError, match="unavailable or expired"):
            await store.resolve(namespace=_namespace(), previous_response_id="resp_one")
        restarted = BoundedContinuationStore()
        with pytest.raises(OpenAIProtocolError, match="unavailable or expired"):
            await restarted.resolve(namespace=_namespace(), previous_response_id="resp_one")

        now[0] = 111.0
        with pytest.raises(OpenAIProtocolError, match="unavailable or expired"):
            await store.resolve(namespace=_namespace(), previous_response_id="resp_two")

    asyncio.run(scenario())


def test_replay_operation_index_stays_aligned_across_expiry_and_abandonment() -> None:
    """Conflicts last exactly as long as a live entry, and both maps stay aligned."""

    async def scenario() -> None:
        """Expire and abandon keyed entries, then re-claim with changed bodies."""
        now = [100.0]
        store = BoundedReplayStore(capacity=4, byte_cap=4_096, ttl_seconds=10, clock=lambda: now[0])

        def keyed(operation: str, canonical: str) -> ReplayKey:
            """Return one non-optional replay key for this test namespace."""
            key = replay_key(
                namespace=_namespace(),
                surface=GatewayApiSurface.CHAT_COMPLETIONS,
                caller_operation=operation,
                canonical_request_sha256=canonical,
            )
            assert key is not None
            return key

        completed = keyed("op-completed", "a" * 64)
        owner = await store.claim(completed)
        await owner.complete(
            CachedResponse(status_code=200, media_type="application/json", headers=(), body=b"{}")
        )
        with pytest.raises(OpenAIProtocolError, match="different request body"):
            await store.claim(keyed("op-completed", "b" * 64))
        now[0] += 11
        retry = await store.claim(keyed("op-completed", "b" * 64))
        assert retry.kind == ReplayClaimKind.OWNER
        await retry.abandon()
        reclaimed = await store.claim(keyed("op-completed", "c" * 64))
        assert reclaimed.kind == ReplayClaimKind.OWNER
        await reclaimed.abandon()
        assert len(store._operations) == len(store._entries)  # noqa: SLF001 - invariant probe.

    asyncio.run(scenario())
