# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Loaded component bundle for the local gateway lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exp.runtime.gateway.contracts import ExecutionSnapshot
    from exp.runtime.gateway.group_commit import GroupCommitAttemptLedger
    from exp.runtime.gateway.interfaces import GatewayControlStore
    from exp.runtime.gateway.ledger import SQLiteAttemptLedger
    from exp.runtime.gateway.lifecycle import _AliasAuthorityReloader
    from exp.runtime.gateway.management import GatewayManagement
    from exp.runtime.gateway.routing import CatalogRouteResolver, SelectionWorkerPool
    from exp.runtime.models import RuntimeModelCatalog


@dataclass(frozen=True)
class LocalGatewayComponents:
    """Loaded authority, accounting, and routing for the native gateway engine.

    The native engine's control-plane bridge uses these directly for
    admission and settlement over shared SQLite state and hot-reloadable
    authority generations.
    """

    manager: GatewayManagement
    store: GatewayControlStore
    ledger: SQLiteAttemptLedger
    write_ledger: GroupCommitAttemptLedger
    routes: CatalogRouteResolver
    reloader: _AliasAuthorityReloader
    selection_workers: SelectionWorkerPool
    reconciled_expired_requests: int
    reconciled_unknown_attempts: int
    # The local launch serves no asynchronous batch lane; hosted compositions
    # supply a BatchControlPlane here to enable /v1/batches.
    batches: object | None = None

    @property
    def runtime_catalogs(self) -> Mapping[tuple[str, str], RuntimeModelCatalog]:
        """Return the current generation's runtime catalogs."""
        return self.reloader.state.runtime_catalogs

    @property
    def accounting_healthy(self) -> bool:
        """Return whether the shared group-commit writer can still land writes.

        The native bridge's readiness callback reads this composition health
        surface; per-settlement losses latch in the bridge's own registry.
        """
        return not self.write_ledger.closed

    @property
    def readiness(self) -> tuple[ExecutionSnapshot, ...]:
        """Return the current generation's credential-free route proof."""
        return (self.reloader.state.proof,)

    @property
    def unavailable_aliases(self) -> tuple[tuple[str, str], ...]:
        """Return the current generation's failed aliases with their exact reasons."""
        return self.reloader.state.unavailable_aliases

    @property
    def organization_id(self) -> str:
        """Return the single local organization identity."""
        return self.manager.organization_id
