"""Bounded process-local continuation, replay, and episode identity contracts."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from enum import StrEnum

from pydantic import Field

from wmo.common.core.artifacts import ContractModel, Sha256
from wmo.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage
from wmo.runtime.openai_protocol.errors import OpenAIProtocolError


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
    """One in-flight or completed response future with bounded retention metadata."""

    def __init__(self, future: asyncio.Future[CachedResponse], expires_at: float) -> None:
        """Create an empty replay entry around one event-loop future."""
        self.future = future
        self.expires_at = expires_at
        self.size_bytes = 0


class ReplayLease:
    """One caller's ownership or join handle for a keyed response."""

    def __init__(
        self,
        *,
        store: BoundedReplayStore,
        key: ReplayKey,
        future: asyncio.Future[CachedResponse],
        kind: ReplayClaimKind,
    ) -> None:
        """Bind one replay claim to its store entry."""
        self._store = store
        self._key = key
        self._future = future
        self.kind = kind

    async def result(self) -> CachedResponse:
        """Join in-flight work or return the already completed exact response."""
        return await asyncio.shield(self._future)

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
        await self._store._complete(self._key, self._future, response)

    async def abandon(self) -> None:
        """Remove failed owner work so no joiner receives invented response content."""
        if self.kind == ReplayClaimKind.OWNER:
            await self._store._abandon(self._key, self._future)


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
                kind = ReplayClaimKind.REPLAY if entry.future.done() else ReplayClaimKind.JOIN
                return ReplayLease(store=self, key=key, future=entry.future, kind=kind)
            self._make_capacity()
            future = asyncio.get_running_loop().create_future()
            entry = _ReplayEntry(future, self._clock() + self._ttl_seconds)
            self._entries[key] = entry
            self._evict_completed()
            return ReplayLease(
                store=self,
                key=key,
                future=future,
                kind=ReplayClaimKind.OWNER,
            )

    async def _complete(
        self,
        key: ReplayKey,
        future: asyncio.Future[CachedResponse],
        response: CachedResponse,
    ) -> None:
        """Atomically publish an owner result and apply retention bounds."""
        if response.size_bytes > self._byte_cap:
            await self._abandon(key, future)
            raise OpenAIProtocolError(
                status_code=500,
                code="idempotency_replay_unavailable",
                message="The completed response exceeds the bounded replay cache.",
                error_type="api_error",
            )
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.future is not future or future.done():
                raise OpenAIProtocolError(
                    status_code=409,
                    code="idempotency_conflict",
                    message="The keyed operation no longer belongs to this request.",
                    param="Idempotency-Key",
                )
            entry.size_bytes = response.size_bytes
            entry.expires_at = self._clock() + self._ttl_seconds
            self._response_bytes += entry.size_bytes
            future.set_result(response)
            self._entries.move_to_end(key)
            self._evict_completed()

    async def _abandon(self, key: ReplayKey, future: asyncio.Future[CachedResponse]) -> None:
        """Remove matching in-flight work without erasing a published result."""
        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.future is future and not future.done():
                self._entries.pop(key)
                future.cancel()

    def _expire(self, now: float) -> None:
        """Drop completed expired entries without evicting active work."""
        for key, entry in tuple(self._entries.items()):
            if entry.future.done() and entry.expires_at <= now:
                self._entries.pop(key)
                self._response_bytes -= entry.size_bytes

    def _evict_completed(self) -> None:
        """Evict oldest completed entries until count and byte bounds hold."""
        while len(self._entries) > self._capacity or self._response_bytes > self._byte_cap:
            completed = next(
                ((key, entry) for key, entry in self._entries.items() if entry.future.done()),
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
                ((key, entry) for key, entry in self._entries.items() if entry.future.done()),
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


class ContinuationState(ContractModel):
    """Bounded content-bearing Responses continuation retained only in memory."""

    episode_key: Sha256
    messages: tuple[GatewayMessage, ...]

    @property
    def size_bytes(self) -> int:
        """Return deterministic serialized bytes used for retention accounting."""
        return len(self.model_dump_json().encode())


class _ContinuationEntry:
    """Namespaced continuation plus its finite expiry."""

    def __init__(self, state: ContinuationState, expires_at: float) -> None:
        """Bind continuation content to its monotonic expiry."""
        self.state = state
        self.expires_at = expires_at


class BoundedContinuationStore:
    """Tenant-isolated Responses continuation with count, byte, and TTL limits."""

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
        self._lock = asyncio.Lock()
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
        if state.size_bytes > self._byte_cap:
            raise OpenAIProtocolError(
                status_code=400,
                code="continuation_unavailable",
                message="The response is too large for bounded local continuation.",
                param="previous_response_id",
            )
        key = (namespace, response_id)
        async with self._lock:
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
        key = (namespace, previous_response_id)
        async with self._lock:
            self._expire(self._clock())
            entry = self._entries.get(key)
            if entry is None:
                raise OpenAIProtocolError(
                    status_code=400,
                    code="continuation_unavailable",
                    message="previous_response_id is unavailable or expired in this namespace.",
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
