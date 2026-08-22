"""Structural component contracts for the native gateway control plane."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from exp.runtime.gateway.execution import GatewayExecutor
from exp.runtime.gateway.group_commit import GroupCommitAttemptLedger
from exp.runtime.gateway.interfaces import GatewayControlStore
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.routing import CatalogRouteResolver
from exp.runtime.models import RuntimeModelCatalog


class NativeGatewayComponents(Protocol):
    """Engine-neutral components required by the native control plane."""

    @property
    def store(self) -> GatewayControlStore:
        """Return the authority store."""
        ...

    @property
    def ledger(self) -> SQLiteAttemptLedger:
        """Return the raw ledger used for content-free reporting reads."""
        ...

    @property
    def write_ledger(self) -> GroupCommitAttemptLedger:
        """Return the shared durable group-commit writer."""
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
