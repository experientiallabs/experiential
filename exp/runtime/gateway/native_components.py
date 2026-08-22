"""Structural component contracts for the native gateway control plane."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AttemptId,
    AuthorizationSnapshot,
    ExecutionSnapshot,
    GatewayEvent,
    GatewayFailure,
)
from exp.runtime.gateway.execution import GatewayExecutor
from exp.runtime.gateway.group_commit import GroupCommitAttemptLedger
from exp.runtime.gateway.interfaces import GatewayControlStore
from exp.runtime.gateway.routing import CatalogRouteResolver
from exp.runtime.models import RuntimeModelCatalog


class SyncWriteLedger(Protocol):
    """Synchronous durable write surface the native data plane settles through.

    Methods are called from data-plane worker threads with no event loop and
    must return only after their write is durable. The local engine satisfies
    this with the group-commit facade; a hosted store satisfies it with its
    own thread-safe synchronous ledger (for example one SQL call per method
    on a pooled connection).
    """

    def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Durably accept one authorized request."""
        ...

    def start_attempt(
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
        """Durably mark one provider dispatch before network work."""
        ...

    def finish_attempt(
        self,
        *,
        attempt_id: AttemptId,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
    ) -> None:
        """Durably settle one attempt exactly once."""
        ...

    def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Durably finalize one request that produced no billable attempt."""
        ...


class NativeGatewayComponents(Protocol):
    """Engine-neutral components required by the native control plane."""

    @property
    def store(self) -> GatewayControlStore:
        """Return the authority store."""
        ...

    @property
    def ledger(self) -> SyncWriteLedger:
        """Return the synchronous durable ledger.

        The control plane reads content-free reports through this object and,
        when :attr:`write_ledger` is ``None``, also settles through it, so a
        hosted implementation must be thread-safe for data-plane callers.
        """
        ...

    @property
    def write_ledger(self) -> GroupCommitAttemptLedger | None:
        """Return the local engine's shared group-commit writer, if any.

        The local composition provides the batching writer so both engines
        share fsync batches. Hosted compositions over their own stores return
        ``None`` (or omit the attribute); the control plane then settles
        directly through :attr:`ledger`.
        """
        ...

    @property
    def routes(self) -> CatalogRouteResolver:
        """Return the direct-route resolver."""
        ...

    @property
    def executor(self) -> GatewayExecutor:
        """Return the shared accounting-health latch."""
        ...

    @property
    def reconciled_expired_requests(self) -> int:
        """Return startup-reconciled request count."""
        ...

    @property
    def reconciled_unknown_attempts(self) -> int:
        """Return startup-reconciled attempt count."""
        ...

    @property
    def runtime_catalogs(self) -> Mapping[tuple[str, str], RuntimeModelCatalog]:
        """Return runtime catalogs keyed by alias revision and digest."""
        ...

    @property
    def organization_id(self) -> str:
        """Return the organization used by the local usage endpoint."""
        ...
