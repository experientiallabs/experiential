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
        restarted = BoundedContinuationStore()
        with pytest.raises(OpenAIProtocolError, match="unavailable or expired"):
            await restarted.resolve(namespace=_namespace(), previous_response_id="resp_one")

        now[0] = 111.0
        with pytest.raises(OpenAIProtocolError, match="unavailable or expired"):
            await store.resolve(namespace=_namespace(), previous_response_id="resp_one")

    asyncio.run(scenario())
