"""Request-local continuation after an authoritative destination-only budget refusal."""

from __future__ import annotations

from exp.runtime.gateway.budgets import BudgetReservationRejected, BudgetScopeKind
from exp.runtime.gateway.contracts import ExecutionSnapshot


def denied_destination_pool(
    rejection: BudgetReservationRejected,
    snapshot: ExecutionSnapshot,
    depth: int,
) -> str | None:
    """Return only the current nonroot pool proved refused by the atomic ledger decision."""
    binding = rejection.binding
    stage = snapshot.stage_for_depth(depth)
    auth = snapshot.authorization
    if (
        rejection.scope_kind != BudgetScopeKind.POOL
        or binding is None
        or binding.scope.kind != BudgetScopeKind.POOL
        or binding.application != "destination"
        or binding.request_id != auth.request_id
        or binding.organization_id != auth.organization_id
        or binding.alias_revision_id != auth.alias_revision_id
        or binding.scope.pool_id != stage.pool_id
        or stage.pool_id == snapshot.pool_id
    ):
        return None
    return stage.pool_id
