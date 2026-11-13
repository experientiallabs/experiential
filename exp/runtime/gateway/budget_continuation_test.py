"""Initial admission accepts only exact destination budget refusal bindings."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.budget_continuation import denied_destination_pool
from exp.runtime.gateway.budgets import (
    BudgetRefusalBinding,
    BudgetReservationRejected,
    BudgetScope,
    BudgetScopeKind,
)
from exp.runtime.gateway.model_plan import model_execution_snapshot
from exp.runtime.gateway.model_plan_test import catalog as model_catalog
from exp.runtime.gateway.native_execution_test import _route


@pytest.mark.parametrize(
    "mismatch",
    ["request_id", "organization_id", "alias_revision_id", "root", "shared", "unknown", "pool"],
)
def test_only_exact_destination_refusal_is_skippable(mismatch: str) -> None:
    """Unknown or stale refusal authority cannot bypass request-wide budget gates."""
    catalog = model_catalog()
    authorization = _route().snapshot.authorization
    snapshot = model_execution_snapshot(catalog, authorization, catalog.pools[0])
    binding = BudgetRefusalBinding(
        request_id=snapshot.authorization.request_id,
        organization_id=authorization.organization_id,
        alias_revision_id=snapshot.authorization.alias_revision_id,
        scope=BudgetScope(
            kind=BudgetScopeKind.POOL,
            alias_id="coding",
            pool_id=snapshot.stage_for_depth(1).pool_id,
        ),
        application="destination",
    )
    assert (
        denied_destination_pool(
            BudgetReservationRejected(
                scope_kind=BudgetScopeKind.POOL, reason="denied", binding=binding
            ),
            snapshot,
            1,
        )
        == snapshot.stage_for_depth(1).pool_id
    )
    if mismatch in ("request_id", "organization_id", "alias_revision_id"):
        binding = binding.model_copy(update={mismatch: "different"})
    elif mismatch in ("root", "shared"):
        binding = binding.model_copy(update={"application": mismatch})
    elif mismatch == "pool":
        binding = binding.model_copy(
            update={
                "scope": BudgetScope(
                    kind=BudgetScopeKind.POOL, alias_id="coding", pool_id=snapshot.pool_id
                )
            }
        )
    rejection = BudgetReservationRejected(
        scope_kind=BudgetScopeKind.POOL,
        reason="denied",
        binding=None if mismatch == "unknown" else binding,
    )
    assert denied_destination_pool(rejection, snapshot, 1) is None
