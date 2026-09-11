"""Bounded process-local continuation, replay, and episode identity contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from enum import StrEnum
from typing import Protocol

from pydantic import Field

from exp.common.core.artifacts import ArtifactId, ContractModel, Sha256
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


class ProtocolNamespace(ContractModel):
    """Tenant and immutable alias boundary for retained process-local state."""

    organization_id: str = Field(min_length=1, max_length=128)
    identity_id: str = Field(min_length=1, max_length=128)
    alias_revision_id: str = Field(min_length=1, max_length=128)


class ReplayKey(ContractModel):
    """Content-free opt-in operation identity for one canonical request."""

    namespace: ProtocolNamespace
    surface: GatewayApiSurface
    caller_operation_sha256: Sha256
    canonical_request_sha256: Sha256


class CachedResponse(ContractModel):
    """Exact bounded HTTP result retained only for in-process replay."""

    status_code: int = Field(ge=100, le=599)
    media_type: str = Field(min_length=1, max_length=256)
    headers: tuple[tuple[str, str], ...] = ()
    body: bytes

    @property
    def size_bytes(self) -> int:
        """Return retained body and metadata bytes for capacity accounting."""
        metadata = len(self.media_type.encode()) + sum(
            len(name.encode()) + len(value.encode()) for name, value in self.headers
        )
        return len(self.body) + metadata


class ReplayClaimKind(StrEnum):
    """Whether a keyed caller owns work, joins it, or replays completion."""

    OWNER = "owner"
    JOIN = "join"
    REPLAY = "replay"


class _ReplayEntry:
    """One in-flight or completed response signal with bounded retention metadata."""

    def __init__(self, expires_at: float) -> None:
        """Create an unpublished replay entry."""
        self.published = asyncio.Event()
        self.response: CachedResponse | None = None
        self.expires_at = expires_at
        self.size_bytes = 0


class ReplayLease(Protocol):
    """Ownership-scoped operations for one claimed replay key.

    Implementations must make publication conditional on ownership established by
    :meth:`ResponseReplayStore.claim`. Cancelling a caller waiting on :meth:`result` must not
    cancel shared work.
    """

    @property
    def kind(self) -> ReplayClaimKind:
        """Return whether this caller owns, joins, or replays the operation."""
        ...

    async def result(self) -> CachedResponse:
        """Join in-flight work or return the already completed exact response."""
        ...

    async def complete(self, response: CachedResponse) -> None:
        """Publish one exact successful response from the unique owner."""
        ...

    async def abandon(self) -> None:
        """Release this claim only when it still owns unpublished work."""
        ...


class ResponseReplayStore(Protocol):
    """Atomic completed-response ownership, joining, and replay operations.

    A claim is scoped by the complete :class:`ReplayKey`. Reusing a caller operation within its
    namespace and surface for another canonical body must fail closed. Implementations must
    return exactly one owner while work is unpublished, join matching concurrent claims, and
    replay only a response that its owner successfully published. Cancellation before a claim
    returns must not strand ownership. A shared implementation must expire ownership left by
    worker loss within a finite implementation-defined lease so joiners cannot wait forever.
    """

    async def claim(self, key: ReplayKey) -> ReplayLease:
        """Claim original work, join an in-flight duplicate, or replay completion."""
        ...


class _BoundedReplayLease:
    """One local caller's ownership or join handle for a keyed response."""

    def __init__(
        self,
        *,
        store: BoundedReplayStore,
        key: ReplayKey,
        entry: _ReplayEntry,
        kind: ReplayClaimKind,
    ) -> None:
        """Bind one replay claim to its store entry."""
        self._store = store
        self._key = key
        self._entry = entry
        self.kind = kind

    async def result(self) -> CachedResponse:
        """Join in-flight work or return the already completed exact response."""
        await asyncio.shield(self._entry.published.wait())
        response = self._entry.response
        if response is None:
            raise OpenAIProtocolError(
                status_code=409,
                code="idempotency_replay_unavailable",
                message="The original keyed request ended before publishing a replayable result.",
                error_type="api_error",
                param="Idempotency-Key",
            )
        return response

    async def complete(self, response: CachedResponse) -> None:
        """Publish one exact response from the unique owner.

        Args:
            response: Completed non-streaming or fully captured SSE result.

        Raises:
            OpenAIProtocolError: This lease does not own the operation.
        """
        if self.kind != ReplayClaimKind.OWNER:
            raise OpenAIProtocolError(
                status_code=409,
                code="idempotency_conflict",
                message="Only the original keyed request may publish its result.",
                param="Idempotency-Key",
            )
        await self._store._complete(self._key, self._entry, response)

    async def abandon(self) -> None:
        """Remove failed owner work so no joiner receives invented response content."""
        if self.kind == ReplayClaimKind.OWNER:
            await self._store._abandon(self._key, self._entry)


class BoundedReplayStore:
    """Single-process duplicate joining and exact completed-response replay."""

    def __init__(
        self,
        *,
        capacity: int = 4_096,
        byte_cap: int = 64 * 1024 * 1024,
        ttl_seconds: float = 24 * 60 * 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize finite replay state.

        Args:
            capacity: Maximum completed and in-flight operation count.
            byte_cap: Maximum bytes retained across completed results.
            ttl_seconds: Completed result retention lifetime.
            clock: Injectable monotonic clock.
        """
        if capacity <= 0 or byte_cap <= 0 or ttl_seconds <= 0:
            raise ValueError("replay bounds must be positive")
        self._capacity = capacity
        self._byte_cap = byte_cap
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = asyncio.Lock()
        self._entries: OrderedDict[ReplayKey, _ReplayEntry] = OrderedDict()
        self._response_bytes = 0

    async def claim(self, key: ReplayKey) -> ReplayLease:
        """Claim original work, join an in-flight duplicate, or replay completion.

        Args:
            key: Fully namespaced, hashed caller operation and canonical request.

        Returns:
            Lease identifying the caller's safe action.
        """
        async with self._lock:
            self._expire(self._clock())
            for existing_key in self._entries:
                if (
                    existing_key.namespace == key.namespace
                    and existing_key.surface == key.surface
                    and existing_key.caller_operation_sha256 == key.caller_operation_sha256
                    and existing_key.canonical_request_sha256 != key.canonical_request_sha256
                ):
                    raise OpenAIProtocolError(
                        status_code=409,
                        code="idempotency_conflict",
                        message="The caller operation was reused with a different request body.",
                        param="Idempotency-Key",
                    )
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                kind = (
                    ReplayClaimKind.REPLAY if entry.response is not None else ReplayClaimKind.JOIN
                )
                return _BoundedReplayLease(store=self, key=key, entry=entry, kind=kind)
            self._make_capacity()
            entry = _ReplayEntry(self._clock() + self._ttl_seconds)
            self._entries[key] = entry
            self._evict_completed()
            return _BoundedReplayLease(
                store=self,
                key=key,
                entry=entry,
                kind=ReplayClaimKind.OWNER,
            )

    async def _complete(
        self,
        key: ReplayKey,
        claimed_entry: _ReplayEntry,
        response: CachedResponse,
    ) -> None:
        """Atomically publish an owner result and apply retention bounds."""
        if response.size_bytes > self._byte_cap:
            await self._abandon(key, claimed_entry)
            raise OpenAIProtocolError(
                status_code=500,
                code="idempotency_replay_unavailable",
                message="The completed response exceeds the bounded replay cache.",
                error_type="api_error",
            )
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry is not claimed_entry or entry.published.is_set():
                raise OpenAIProtocolError(
                    status_code=409,
                    code="idempotency_conflict",
                    message="The keyed operation no longer belongs to this request.",
                    param="Idempotency-Key",
                )
            entry.size_bytes = response.size_bytes
            entry.expires_at = self._clock() + self._ttl_seconds
            self._response_bytes += entry.size_bytes
            entry.response = response
            entry.published.set()
            self._entries.move_to_end(key)
            self._evict_completed()

    async def _abandon(self, key: ReplayKey, claimed_entry: _ReplayEntry) -> None:
        """Remove matching in-flight work without erasing a published result."""
        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry is claimed_entry and not entry.published.is_set():
                self._entries.pop(key)
                entry.published.set()

    def _expire(self, now: float) -> None:
        """Drop completed expired entries without evicting active work."""
        for key, entry in tuple(self._entries.items()):
            if entry.response is not None and entry.expires_at <= now:
                self._entries.pop(key)
                self._response_bytes -= entry.size_bytes

    def _evict_completed(self) -> None:
        """Evict oldest completed entries until count and byte bounds hold."""
        while len(self._entries) > self._capacity or self._response_bytes > self._byte_cap:
            completed = next(
                (
                    (key, entry)
                    for key, entry in self._entries.items()
                    if entry.response is not None
                ),
                None,
            )
            if completed is None:
                return
            key, entry = completed
            self._entries.pop(key)
            self._response_bytes -= entry.size_bytes

    def _make_capacity(self) -> None:
        """Evict completed work or reject when every bounded slot is in flight."""
        while len(self._entries) >= self._capacity:
            completed = next(
                (
                    (key, entry)
                    for key, entry in self._entries.items()
                    if entry.response is not None
                ),
                None,
            )
            if completed is None:
                raise OpenAIProtocolError(
                    status_code=429,
                    code="gateway_overloaded",
                    message="The bounded in-process replay window is full.",
                    error_type="api_error",
                )
            key, entry = completed
            self._entries.pop(key)
            self._response_bytes -= entry.size_bytes


class ContinuationRouteBinding(ContractModel):
    """Secret-free authority binding for retained encrypted reasoning."""

    deployment_id: ArtifactId
    connection_sha256: Sha256
    wire_authority_sha256: Sha256


class ContinuationState(ContractModel):
    """Bounded content-bearing Responses continuation retained only in memory."""

    episode_key: Sha256
    messages: tuple[GatewayMessage, ...]
    route_binding: ContinuationRouteBinding | None = None

    @property
    def size_bytes(self) -> int:
        """Return serialized bytes including provider replay fields excluded from artifacts."""
        size = len(self.model_dump_json().encode())
        replay_authority: list[dict[str, object]] = []
        for message_index, message in enumerate(self.messages):
            authority: dict[str, object] = {"message_index": message_index}
            if message.provider_item_id is not None:
                authority["provider_item_id"] = message.provider_item_id
                authority["provider_output_index"] = message.provider_output_index
                authority["provider_status"] = message.provider_status
                authority["provider_phase"] = message.provider_phase
            if message.provider_reasoning:
                blocks: list[dict[str, object]] = []
                for block in message.provider_reasoning:
                    serialized = block.model_dump(mode="json")
                    if block.kind == "encrypted_reasoning":
                        serialized["output_index"] = block.output_index
                        serialized["status"] = block.status
                    blocks.append(serialized)
                authority["provider_reasoning"] = blocks
            retained_calls: list[dict[str, object]] = []
            for call in message.tool_calls:
                if (
                    call.raw_arguments is None
                    and call.provider_item_id is None
                    and call.provider_output_index is None
                    and call.provider_status is None
                    and call.provider_namespace is None
                    and call.provider_caller is None
                ):
                    continue
                retained_call: dict[str, object] = {
                    "call_id": call.call_id,
                    "raw_arguments": call.raw_arguments,
                    "provider_item_id": call.provider_item_id,
                    "provider_output_index": call.provider_output_index,
                    "provider_status": call.provider_status,
                }
                if call.provider_namespace is not None:
                    retained_call["provider_namespace"] = call.provider_namespace
                if call.provider_caller is not None:
                    retained_call["provider_caller"] = call.provider_caller
                retained_calls.append(retained_call)
            if retained_calls:
                authority["tool_calls"] = retained_calls
            if message.tool_is_error:
                authority["tool_is_error"] = True
            if message.provider_tool_name is not None:
                authority["provider_tool_name"] = message.provider_tool_name
            if message.provider_tool_namespace is not None:
                authority["provider_tool_namespace"] = message.provider_tool_namespace
            if message.provider_tool_caller is not None:
                authority["provider_tool_caller"] = message.provider_tool_caller
            if len(authority) > 1:
                replay_authority.append(authority)
        if replay_authority:
            size += len(
                json.dumps(
                    replay_authority,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            )
        return size


class _ContinuationEntry:
    """Namespaced continuation plus its finite expiry."""

    def __init__(self, state: ContinuationState, expires_at: float) -> None:
        """Bind continuation content to its monotonic expiry."""
        self.state = state
        self.expires_at = expires_at


class BoundedContinuationStore:
    """Tenant-isolated Responses continuation with count, byte, and TTL limits.

    Guarded by one non-async lock held only for in-memory bookkeeping, so the
    same instance is shared safely between async callers and the native data
    plane's control-plane worker threads.
    """

    def __init__(
        self,
        *,
        capacity: int = 4_096,
        byte_cap: int = 64 * 1024 * 1024,
        ttl_seconds: float = 24 * 60 * 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize finite continuation state."""
        if capacity <= 0 or byte_cap <= 0 or ttl_seconds <= 0:
            raise ValueError("continuation bounds must be positive")
        self._capacity = capacity
        self._byte_cap = byte_cap
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[ProtocolNamespace, str], _ContinuationEntry] = (
            OrderedDict()
        )
        self._content_bytes = 0

    async def remember(
        self,
        *,
        namespace: ProtocolNamespace,
        response_id: str,
        state: ContinuationState,
    ) -> None:
        """Retain one completed Responses continuation within strict bounds.

        Args:
            namespace: Tenant, identity, and alias-revision boundary.
            response_id: Public completed response identity.
            state: Canonical history and hashed episode identity.

        Raises:
            OpenAIProtocolError: One continuation exceeds the total byte ceiling.
        """
        self.remember_now(namespace=namespace, response_id=response_id, state=state)

    async def resolve(
        self, *, namespace: ProtocolNamespace, previous_response_id: str
    ) -> ContinuationState:
        """Resolve an exact namespaced continuation or fail closed.

        Args:
            namespace: Current caller and immutable alias-revision boundary.
            previous_response_id: Public response identity to continue.

        Returns:
            Retained canonical history.

        Raises:
            OpenAIProtocolError: State expired, was evicted, crossed namespace, or restarted.
        """
        return self.resolve_now(namespace=namespace, previous_response_id=previous_response_id)

    def remember_now(
        self,
        *,
        namespace: ProtocolNamespace,
        response_id: str,
        state: ContinuationState,
    ) -> None:
        """Retain one completed Responses continuation within strict bounds.

        Args:
            namespace: Tenant, identity, and alias-revision boundary.
            response_id: Public completed response identity.
            state: Canonical history and hashed episode identity.

        Raises:
            OpenAIProtocolError: One continuation exceeds the total byte ceiling.
        """
        if state.size_bytes > self._byte_cap:
            raise OpenAIProtocolError(
                status_code=400,
                code="continuation_unavailable",
                message=(
                    "The response is too large for bounded local continuation. "
                    "Resend the full conversation history in this request instead of "
                    "previous_response_id."
                ),
                param="previous_response_id",
            )
        key = (namespace, response_id)
        with self._lock:
            self._expire(self._clock())
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._content_bytes -= previous.state.size_bytes
            self._entries[key] = _ContinuationEntry(
                state,
                self._clock() + self._ttl_seconds,
            )
            self._content_bytes += state.size_bytes
            self._evict()

    def resolve_now(
        self, *, namespace: ProtocolNamespace, previous_response_id: str
    ) -> ContinuationState:
        """Resolve an exact namespaced continuation or fail closed.

        Args:
            namespace: Current caller and immutable alias-revision boundary.
            previous_response_id: Public response identity to continue.

        Returns:
            Retained canonical history.

        Raises:
            OpenAIProtocolError: State expired, was evicted, crossed namespace, or restarted.
        """
        key = (namespace, previous_response_id)
        with self._lock:
            self._expire(self._clock())
            entry = self._entries.get(key)
            if entry is None:
                raise OpenAIProtocolError(
                    status_code=400,
                    # api.openai.com's code for an unknown previous_response_id;
                    # the Codex client auto-recovers on exactly this string by
                    # resending the full conversation.
                    code="previous_response_not_found",
                    message=(
                        "previous_response_id is unavailable or expired in this namespace. "
                        "Resend the full conversation history in this request."
                    ),
                    param="previous_response_id",
                )
            self._entries.move_to_end(key)
            return entry.state

    def _expire(self, now: float) -> None:
        """Remove every expired continuation before reads and writes."""
        for key, entry in tuple(self._entries.items()):
            if entry.expires_at <= now:
                self._entries.pop(key)
                self._content_bytes -= entry.state.size_bytes

    def _evict(self) -> None:
        """Evict least-recent continuations until both bounds hold."""
        while self._entries and (
            len(self._entries) > self._capacity or self._content_bytes > self._byte_cap
        ):
            _, entry = self._entries.popitem(last=False)
            self._content_bytes -= entry.state.size_bytes


def replay_key(
    *,
    namespace: ProtocolNamespace,
    surface: GatewayApiSurface,
    caller_operation: str | None,
    canonical_request_sha256: Sha256,
) -> ReplayKey | None:
    """Build an opt-in content-free replay key without retaining the raw caller value.

    Args:
        namespace: Tenant and immutable alias-revision scope.
        surface: Chat Completions or Responses.
        caller_operation: Explicit caller key, or ``None`` for unkeyed work.
        canonical_request_sha256: Canonical body digest.

    Returns:
        Fully scoped replay key, or ``None`` when deduplication was not requested.
    """
    if caller_operation is None:
        return None
    return ReplayKey(
        namespace=namespace,
        surface=surface,
        caller_operation_sha256=hashlib.sha256(caller_operation.encode()).hexdigest(),
        canonical_request_sha256=canonical_request_sha256,
    )


def episode_namespace(
    *,
    namespace: ProtocolNamespace,
    caller_episode_key: str | None,
    request_id: str,
) -> tuple[str, str, str, str]:
    """Create a tenant-isolated affinity namespace without retaining raw caller keys.

    Args:
        namespace: Organization, identity, and alias revision.
        caller_episode_key: Explicit sticky episode key when supplied.
        request_id: Request-local fallback for unkeyed calls.

    Returns:
        Four-part namespace safe to pass to router selection.
    """
    material = caller_episode_key or request_id
    digest = hashlib.sha256(material.encode()).hexdigest()
    return (
        namespace.organization_id,
        namespace.identity_id,
        namespace.alias_revision_id,
        digest,
    )
