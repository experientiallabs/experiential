"""Frozen model stages preserve exact identity, segment order and destination policy."""

from datetime import UTC, datetime
from itertools import permutations
from unittest.mock import patch

import pytest

from exp.common.models.dispatch_policy import GatewayThrottleRedialPolicy
from exp.common.models.gateway_catalog import (
    ExactModelPool,
    NormalizedGatewayCatalog,
    normalize_gateway_catalog,
)
from exp.common.models.gateway_catalog_test import unavailable_child_catalog
from exp.common.models.gateway_chains import ModelStagePolicy
from exp.common.models.gateway_chains_test import chain
from exp.common.models.gateway_pools import GatewayEquivalenceCertification
from exp.runtime.gateway.contracts import (
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    ProjectTarget,
)
from exp.runtime.gateway.health import DeploymentHealthRegistry
from exp.runtime.gateway.model_plan import model_execution_snapshot, project_stage_selection
from exp.runtime.gateway.native_execution import (
    MAXIMUM_TOTAL_ATTEMPTS,
    InflightRequest,
    deployment_health_key,
    deployment_wire_entry,
    reorder_route_deployments,
)
from exp.runtime.gateway.native_execution_test import _route
from exp.runtime.gateway.native_rung_policy import (
    failed_dispatch_candidate,
    throttle_redial_budgets,
)
from exp.runtime.gateway.routing import CatalogRouteResolver, GatewayRoutingError
from exp.runtime.gateway.rung_admission import RungLoadRegistry
from exp.runtime.models.providers.base import GatewayWireProfile


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


def test_unavailable_poolless_child_never_becomes_a_stage_or_root_route() -> None:
    """A retired reference keeps parent placements and unrelated routing without child authority."""
    normalized = normalize_gateway_catalog(unavailable_child_catalog())
    authorization = _route().snapshot.authorization.model_copy(
        update={
            "catalog_sha256": normalized.identity_sha256(),
            "target": DirectTarget(pool_id="pool-a"),
        }
    )
    resolver = CatalogRouteResolver(
        {(authorization.alias_revision_id, authorization.catalog_sha256): normalized}
    )
    route = resolver.resolve_direct(authorization)
    assert tuple(d.deployment_id for d in route.deployments) == ("a1", "a2")
    assert [s.deployment_ids for s in route.snapshot.model_stages] == [("a1",), ("a2",)]
    assert all(
        s.exact_model_id == "a" and s.pool_id == "pool-a" for s in route.snapshot.model_stages
    )
    assert route.snapshot.authorization == authorization
    assert resolver.resolve_direct(
        authorization.model_copy(
            update={
                "target": DirectTarget(pool_id="c1"),
            }
        )
    ).snapshot.deployment_ids == ("c1",)
    with pytest.raises(GatewayRoutingError, match="pool"):
        resolver.resolve_direct(
            authorization.model_copy(update={"target": DirectTarget(pool_id="retired-b")})
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


def test_stage_retry_floors_and_shared_attempt_cap() -> None:
    """Sticky, pinned and final floors use their own stage schedule and one global cap."""
    normalized = catalog()
    root_redial = GatewayThrottleRedialPolicy(max_attempts=1, base_delay_ms=10, max_delay_ms=100)
    child_redial = GatewayThrottleRedialPolicy(max_attempts=4, base_delay_ms=50, max_delay_ms=900)
    chains = tuple(
        c.model_copy(
            update={
                "policy": ModelStagePolicy(
                    failover_mode="maximize_cache",
                    throttle_cache_threshold=0.5,
                    throttle_redial=root_redial if c.model_id == "a" else child_redial,
                )
            }
        )
        for c in normalized.model_chains
    )
    normalized = normalized.model_copy(update={"model_chains": chains})
    auth = _route().snapshot.authorization.model_copy(
        update={
            "catalog_sha256": normalized.identity_sha256(),
            "target": DirectTarget(pool_id="pool-a"),
        }
    )
    route = CatalogRouteResolver(
        {(auth.alias_revision_id, auth.catalog_sha256): normalized}
    ).resolve_direct(auth)
    loads = RungLoadRegistry()
    assert throttle_redial_budgets(loads, route, auth.organization_id) == (0, 0, 1)
    assert throttle_redial_budgets(
        loads, route, auth.organization_id, sticky_deployment_id="b1"
    ) == (0, 4, 1)
    pinned = route.model_copy(update={"reasoning_pinned_deployment_id": "b1"})
    assert throttle_redial_budgets(loads, pinned, auth.organization_id) == (0, 4, 1)
    request = GatewayRequest(
        surface=auth.surface, messages=(GatewayMessage(role="user", content="hello"),)
    )
    entry = InflightRequest(authorization=auth, route=route, request=request, deadline_monotonic=10)
    assert entry.throttle_redial_budgets == (1, 4, 1)
    entry.total_attempts = MAXIMUM_TOTAL_ATTEMPTS
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.THROTTLED, safe_message="limited", failover_eligible=True
    )
    candidate, _ = failed_dispatch_candidate(
        health=DeploymentHealthRegistry(),
        loads=loads,
        keys=tuple(deployment_health_key(auth, d) for d in route.deployments),
        entry=entry,
        failure=failure,
        current_depth=1,
        throttle_backoff=True,
    )
    assert candidate is None


def test_frozen_stage_policy_survives_another_catalog_generation() -> None:
    """A newer child policy cannot change a previously accepted route's pool or schedule."""
    old = catalog()
    child_policy = ModelStagePolicy(
        failover_mode="maximize_availability", throttle_cache_threshold=None
    )
    child = old.model_chains[1].model_copy(update={"revision": "new-child", "policy": child_policy})
    new = old.model_copy(update={"model_chains": (old.model_chains[0], child)})
    old_auth = _route().snapshot.authorization.model_copy(
        update={"catalog_sha256": old.identity_sha256(), "target": DirectTarget(pool_id="pool-a")}
    )
    new_auth = old_auth.model_copy(
        update={"alias_revision_id": "new-revision", "catalog_sha256": new.identity_sha256()}
    )
    resolver = CatalogRouteResolver(
        {
            (old_auth.alias_revision_id, old_auth.catalog_sha256): old,
            (new_auth.alias_revision_id, new_auth.catalog_sha256): new,
        }
    )
    assert (
        resolver.resolve_direct(new_auth).snapshot.stage_for_depth(1).failover_mode
        == "maximize_availability"
    )
    old_stage = resolver.resolve_direct(old_auth).snapshot.stage_for_depth(1)
    assert (
        old_stage.exact_model_id,
        old_stage.pool_id,
        old_stage.failover_mode,
        old_stage.throttle_cache_threshold,
    ) == (
        "b",
        "pool-b",
        "maximize_cache_affinity",
        0.8,
    )


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
    with pytest.raises(ValueError, match="requires explicit authorization"):
        resolver.resolve_deployment_hint(authorization, "b1")
    hinted = resolver.resolve_deployment_hint(
        authorization.model_copy(update={"descendant_start_authorized": True}), "b1"
    )
    assert hinted.deployment.exact_model_id == "b"
    assert hinted.snapshot.exact_model_id == "a"
    assert hinted.snapshot.pool_id == "pool-a"
    assert hinted.snapshot.stage_for_depth(0).exact_model_id == "b"
    assert hinted.snapshot.deployment_ids == ("b1", "a2")
    assert hinted.reasoning_pinned_deployment_id == "b1"
    assert hinted.requires_reasoning_strip(hinted.fallback_deployments[0])


@pytest.mark.parametrize(
    "surface",
    [GatewayApiSurface.CHAT_COMPLETIONS, GatewayApiSurface.RESPONSES, GatewayApiSurface.MESSAGES],
)
def test_project_selection_never_enters_authored_model_references(
    surface: GatewayApiSurface,
) -> None:
    """Selecting exact A through a project cannot authorize A's direct-only A-to-B chain."""
    normalized = catalog()
    digest = normalized.identity_sha256()
    auth = _route().snapshot.authorization.model_copy(
        update={
            "surface": surface,
            "catalog_sha256": digest,
            "target": ProjectTarget(
                project_ref="project-one", activation_ref="activation-one", catalog_sha256=digest
            ),
        }
    )
    with patch.object(
        NormalizedGatewayCatalog, "chains_by_model", wraps=normalized.chains_by_model
    ) as chains:
        plan = model_execution_snapshot(normalized, auth, normalized.pools[0])
    chains.assert_not_called()
    assert plan.deployment_ids == ("a1", "a2")
    assert plan.exact_model_id == "a"
    assert plan.pool_id == "pool-a"
    assert plan.failover_mode == "maximize_cache"
    assert plan.throttle_cache_threshold == 0.5
    assert plan.model_stages == ()
    assert plan.traversal_events == ()


def test_unstaged_projection_only_updates_selected_ids_and_stage_field() -> None:
    """All direct-route subsets and permutations avoid temporary execution stages."""
    snapshot = _route().snapshot
    assert "model_stages" not in snapshot.model_fields_set
    with patch.object(ExecutionSnapshot, "stage_for_depth", autospec=True) as stage_for_depth:
        for size in range(len(snapshot.deployment_ids) + 1):
            for indexes in permutations(range(len(snapshot.deployment_ids)), size):
                projected = project_stage_selection(snapshot, indexes)
                expected = snapshot.model_copy(
                    update={
                        "deployment_ids": tuple(snapshot.deployment_ids[i] for i in indexes),
                        "model_stages": (),
                    }
                )
                assert projected == expected
                assert projected.model_fields_set == expected.model_fields_set
    stage_for_depth.assert_not_called()


@pytest.mark.parametrize("indexes", [(-1,), (3,), (0, -1), (0, 3)])
def test_unstaged_projection_rejects_out_of_range_depths(indexes: tuple[int, ...]) -> None:
    """The direct fast path keeps the cursor's ValueError guard, including negative depths."""
    with pytest.raises(ValueError, match="execution route depth is outside the authorized plan"):
        project_stage_selection(_route().snapshot, indexes)


@pytest.mark.parametrize("surface", [GatewayApiSurface.EMBEDDINGS, GatewayApiSurface.IMAGES])
def test_nonconversational_surfaces_stay_direct(surface: GatewayApiSurface) -> None:
    """Image and embedding semantics never gain implicit cross-model fallbacks."""
    normalized = catalog()
    auth = _route().snapshot.authorization.model_copy(update={"surface": surface})
    plan = model_execution_snapshot(normalized, auth, normalized.pools[0])
    assert plan.deployment_ids == ("a1", "a2")
    assert not plan.model_stages
