"""Tests for hot-reloadable granted alias serving authority."""

from __future__ import annotations

from exp.runtime.gateway.alias_authority import (
    UNSERVED_AUTHORITY_RELOAD_BACKOFF_SECONDS,
    _AliasAuthorityReloader,
    _AliasAuthorityState,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
)
from exp.runtime.gateway.execution_snapshot import ExecutionSnapshot
from exp.runtime.gateway.routing import CatalogRouteResolver, SelectionWorkerPool


def _state() -> _AliasAuthorityState:
    """Build one served generation with a single direct alias authority.

    Returns:
        A generation serving ``("coding", "revision-one", "a" * 64)`` only.
    """
    proof = ExecutionSnapshot(
        authorization=AuthorizationSnapshot(
            request_id="readiness-probe",
            organization_id="org",
            identity_id="identity",
            virtual_key_id="key",
            alias="coding",
            alias_revision_id="revision-one",
            target=DirectTarget(pool_id="coding"),
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            catalog_sha256="a" * 64,
            canonical_request_sha256="c" * 64,
            refusal_failover=False,
            deadline_monotonic=1.0,
        ),
        exact_model_id="exact",
        pool_id="coding",
        deployment_ids=("deployment",),
    )
    return _AliasAuthorityState(
        authorities=frozenset({("coding", "revision-one", "a" * 64)}),
        normalized_catalogs={},
        runtime_catalogs={},
        activations={},
        exact_models={},
        listing_pools={},
        proof=proof,
        unavailable_aliases=(("broken", "pins an unservable snapshot"),),
    )


def test_unserved_authority_reloads_at_most_once_per_backoff_window() -> None:
    """An alias that loads unavailable with no fallback does not reload on every call."""
    now = [100.0]
    loads = [0]

    def loader() -> _AliasAuthorityState:
        """Count one full reload that never serves the drifted authority."""
        loads[0] += 1
        return _state()

    workers = SelectionWorkerPool()
    try:
        reloader = _AliasAuthorityReloader(
            loader=loader,
            state=_state(),
            routes=CatalogRouteResolver({}),
            selection_workers=workers,
            monotonic=lambda: now[0],
        )
        unserved = ("broken", "revision-dead", "b" * 64)
        reloader.refresh_if_drifted(unserved)
        reloader.refresh_if_drifted(unserved)
        assert loads[0] == 1
        now[0] += UNSERVED_AUTHORITY_RELOAD_BACKOFF_SECONDS + 0.1
        reloader.refresh_if_drifted(unserved)
        assert loads[0] == 2
    finally:
        workers.shutdown()
