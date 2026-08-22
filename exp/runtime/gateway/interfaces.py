"""Injected gateway service interfaces with no storage or provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol

from exp.common.core.artifacts import ArtifactId
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AttemptId,
    AuthorizationSnapshot,
    ExecutionSnapshot,
    GatewayEvent,
    GatewayFailure,
    GatewayRequest,
    GatewayTarget,
    ProjectSelection,
)


class GatewayControlStore(Protocol):
    """Persistence seam for identities, grants, aliases, and immutable revisions."""

    def authenticate_key(self, *, raw_key: str) -> None:
        """Validate a virtual key before parsing a full content-bearing request."""
        ...

    def authenticated_identity(self, *, raw_key: str) -> tuple[str, str]:
        """Return the organization and identity IDs owning one valid key."""
        ...

    def authorize_request(
        self,
        *,
        raw_key: str,
        alias: str,
        request: GatewayRequest,
        deadline_monotonic: float,
    ) -> AuthorizationSnapshot:
        """Authenticate, authorize, and freeze authority before route selection."""
        ...

    def granted_aliases(self, *, raw_key: str) -> tuple[str, ...]:
        """Return active aliases explicitly granted to the key-derived identity."""
        ...

    def granted_alias_authorities(self, *, raw_key: str) -> tuple[tuple[str, str, str], ...]:
        """Return granted alias, active revision, and catalog digest triples."""
        ...


class SecretResolver(Protocol):
    """Late-bound resolver for opaque provider credential references."""

    def resolve(self, reference: str) -> str:
        """Resolve one configured reference without logging or persisting its value."""
        ...


class AttemptLedger(Protocol):
    """Content-free persistence seam for request and provider-attempt accounting.

    Every method resolves only after its write is durable, so callers keep
    write-through semantics regardless of how an implementation batches or
    schedules the underlying commits.
    """

    async def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Persist one accepted request before selection or provider dispatch."""
        ...

    async def start_attempt(
        self,
        *,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_micro_usd: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
    ) -> AttemptId:
        """Atomically reserve cost and persist one attempt before provider dispatch."""
        ...

    async def finish_attempt(
        self,
        *,
        attempt_id: AttemptId,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
    ) -> None:
        """Settle one physical attempt and optionally finalize its parent request."""
        ...

    async def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Terminalize accepted work that failed before a provider dispatch existed."""
        ...


class ProviderStream(Protocol):
    """True provider stream yielding normalized events in provider order."""

    @property
    def committed(self) -> bool:
        """Return whether an outward semantic event has committed this provider route."""
        ...

    def __aiter__(self) -> AsyncIterator[GatewayEvent]:
        """Iterate normalized provider events until one terminal event."""
        ...

    async def cancel(self) -> None:
        """Cancel active upstream work within the adapter's declared bound."""
        ...


class ProjectTargetResolver(Protocol):
    """Runtime seam that consumes learned selection without executing a provider."""

    async def select(
        self,
        *,
        target: GatewayTarget,
        request: GatewayRequest,
        episode_namespace: tuple[ArtifactId, ArtifactId, ArtifactId, str],
        deadline_monotonic: float,
    ) -> ProjectSelection:
        """Resolve one direct or project target to a frozen exact logical model."""
        ...

    def select_blocking(
        self,
        *,
        target: GatewayTarget,
        request: GatewayRequest,
        episode_namespace: tuple[ArtifactId, ArtifactId, ArtifactId, str],
        deadline_monotonic: float,
    ) -> ProjectSelection:
        """Resolve one project target synchronously for callers without an event loop."""
        ...


class GatewayClock(Protocol):
    """Injectable wall and monotonic clock for deadlines and persisted timestamps."""

    def now(self) -> datetime:
        """Return the current timezone-aware wall-clock time."""
        ...

    def monotonic(self) -> float:
        """Return the process-local monotonic time in seconds."""
        ...
