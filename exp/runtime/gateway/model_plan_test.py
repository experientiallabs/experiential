"""Frozen model stages preserve exact identity, segment order and destination policy."""

from datetime import UTC, datetime

import pytest

from exp.common.models.dispatch_policy import GatewayThrottleRedialPolicy
from exp.common.models.gateway_catalog import ExactModelPool, NormalizedGatewayCatalog
from exp.common.models.gateway_chains import ModelStagePolicy
from exp.common.models.gateway_chains_test import chain
from exp.common.models.gateway_pools import GatewayEquivalenceCertification
from exp.runtime.gateway.contracts import DirectTarget, GatewayApiSurface
from exp.runtime.gateway.model_plan import model_execution_snapshot, project_stage_selection
from exp.runtime.gateway.native_execution import reorder_route_deployments
from exp.runtime.gateway.native_execution_test import _route
from exp.runtime.gateway.routing import CatalogRouteResolver


def catalog() -> NormalizedGatewayCatalog:
    """Build a reciprocal graph with independent exact certification and policies."""
    route = _route()
    a1, b1, a2 = route.deployments
    a1 = a1.model_copy(update={"exact_model_id": "a", "deployment_id": "a1"})
    a2 = a2.model_copy(update={"exact_model_id": "a", "deployment_id": "a2"})
    b1 = b1.model_copy(update={"exact_model_id": "b", "deployment_id": "b1"})
    return NormalizedGatewayCatalog(
        deployments=(a1, a2, b1),
        pools=(
            ExactModelPool(
                pool_id="pool-a",
                exact_model_id="a",
                deployment_ids=("a1", "a2"),
                equivalence=GatewayEquivalenceCertification(
                    certification_id="test",
                    provenance="verified",
                    evidence_sha256="e" * 64,
                    certified_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
                failover_mode="maximize_cache",
                throttle_cache_threshold=0.5,
            ),
            ExactModelPool(
                pool_id="pool-b",
                exact_model_id="b",
                deployment_ids=("b1",),
                failover_mode="maximize_cache_affinity",
                throttle_cache_threshold=0.8,
            ),
        ),
        model_chains=(chain("a", "a1", ">b", "a2"), chain("b", "b1", ">a")),
    )


@pytest.mark.parametrize("threshold", [None, 0.8])
def test_singleton_chain_policy_overrides_pool_type_defaults(threshold: float | None) -> None:
    """A singleton child keeps its model policy without asserting equivalence."""
    normalized = catalog()
    child = normalized.model_chains[1].model_copy(
        update={
            "policy": ModelStagePolicy(
                failover_mode="maximize_cache", throttle_cache_threshold=threshold
            )
        }
    )
    normalized = normalized.model_copy(update={"model_chains": (normalized.model_chains[0], child)})
    snapshot = model_execution_snapshot(
        normalized, _route().snapshot.authorization, normalized.pools[0]
    )
    stage = snapshot.stage_for_depth(1)
    assert stage.failover_mode == "maximize_cache"
    assert stage.throttle_cache_threshold == threshold
    assert normalized.pools[1].equivalence is None


def test_root_child_redial_schedules_are_frozen_and_stage_local() -> None:
    """Root and child redials survive snapshot, budgets, wire projection and digesting."""
    from exp.runtime.gateway.native_execution import deployment_wire_entry
    from exp.runtime.gateway.native_rung_policy import throttle_redial_budgets
    from exp.runtime.gateway.rung_admission import RungLoadRegistry
    from exp.runtime.models.providers.base import GatewayWireProfile

    normalized = catalog()
    root_redial = GatewayThrottleRedialPolicy(max_attempts=1, base_delay_ms=10, max_delay_ms=100)
    child_redial = GatewayThrottleRedialPolicy(max_attempts=4, base_delay_ms=50, max_delay_ms=900)
    chains = tuple(
        c.model_copy(
            update={
                "policy": ModelStagePolicy(
                    failover_mode="maximize_cache",
                    throttle_redial=root_redial if c.model_id == "a" else child_redial,
                )
            }
        )
        for c in normalized.model_chains
    )
    updated = normalized.model_copy(update={"model_chains": chains})
    assert updated.identity_sha256() != normalized.identity_sha256()
    updated = NormalizedGatewayCatalog.model_validate_json(updated.model_dump_json())
    auth = _route().snapshot.authorization.model_copy(
        update={
            "catalog_sha256": updated.identity_sha256(),
            "target": DirectTarget(pool_id="pool-a"),
        }
    )
    route = CatalogRouteResolver(
        {(auth.alias_revision_id, auth.catalog_sha256): updated}
    ).resolve_direct(auth)
    assert [route.snapshot.stage_for_depth(i).throttle_redial for i in range(3)] == [
        root_redial,
        child_redial,
        root_redial,
    ]
    assert throttle_redial_budgets(RungLoadRegistry(), route, auth.organization_id) == (1, 4, 1)
    wire = deployment_wire_entry(
        route,
        route.deployments[1],
        GatewayWireProfile(dialect="openai_compatible", url="https://provider.test"),
        {},
        throttle_redial_budget=4,
    )
    assert wire["throttle_redial"] == child_redial.model_dump(mode="json")


def test_snapshot_policy_and_sticky_suffix_keep_root_identity() -> None:
    """Starting on B cannot revisit A's primary, but retains A's authored suffix."""
    normalized = catalog()
    snapshot = model_execution_snapshot(
        normalized, _route().snapshot.authorization, normalized.pools[0]
    )
    assert snapshot.exact_model_id == "a"
    assert snapshot.deployment_ids == ("a1", "b1", "a2")
    assert [
        (s.exact_model_id, s.failover_mode, s.throttle_cache_threshold)
        for s in snapshot.model_stages
    ] == [
        ("a", "maximize_cache", 0.5),
        ("b", "maximize_cache_affinity", 0.8),
        ("a", "maximize_cache", 0.5),
    ]
    sticky = project_stage_selection(snapshot, (1, 2))
    assert sticky.exact_model_id == "a"
    assert sticky.deployment_ids == ("b1", "a2")
    assert sticky.stage_for_depth(0).exact_model_id == "b"
    assert sticky.stage_for_depth(0).ancestry == ("a", "b")
    assert sticky.stage_for_depth(1).pool_id == "pool-a"


def test_global_reorder_is_rejected_and_hint_reachability_is_authorized() -> None:
    """Scheduling cannot cross a model boundary; continuation can reach its issuer."""
    normalized = catalog()
    authorization = _route().snapshot.authorization.model_copy(
        update={
            "catalog_sha256": normalized.identity_sha256(),
            "target": DirectTarget(pool_id="pool-a"),
        }
    )
    authorization = type(authorization).model_validate(authorization.model_dump())
    resolver = CatalogRouteResolver(
        {(authorization.alias_revision_id, authorization.catalog_sha256): normalized}
    )
    route = resolver.resolve_direct(authorization)
    with pytest.raises(ValueError, match="boundary"):
        reorder_route_deployments(route, (1, 0, 2))
    hinted = resolver.resolve_deployment_hint(authorization, "b1")
    assert hinted.deployment.exact_model_id == "b"
    assert hinted.snapshot.exact_model_id == "a"
    assert hinted.snapshot.pool_id == "pool-a"
    assert hinted.snapshot.stage_for_depth(0).exact_model_id == "b"
    assert hinted.snapshot.deployment_ids == ("b1", "a2")
    assert hinted.reasoning_pinned_deployment_id == "b1"
    assert hinted.requires_reasoning_strip(hinted.fallback_deployments[0])


@pytest.mark.parametrize("surface", [GatewayApiSurface.EMBEDDINGS, GatewayApiSurface.IMAGES])
def test_nonconversational_surfaces_stay_direct(surface: GatewayApiSurface) -> None:
    """Image and embedding semantics never gain implicit cross-model fallbacks."""
    normalized = catalog()
    auth = _route().snapshot.authorization.model_copy(update={"surface": surface})
    plan = model_execution_snapshot(normalized, auth, normalized.pools[0])
    assert plan.deployment_ids == ("a1", "a2")
    assert not plan.model_stages
