"""Tests for the per-reservation rung-policy decisions the accounting bridge applies."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from exp.common.models.catalog import (
    BillingSource,
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    GatewayRungDispatchPolicy,
    ModelCatalog,
    ModelRecord,
)
from exp.common.models.dispatch_policy import GatewayThrottleRedialPolicy
from exp.common.models.gateway_catalog import (
    ExactModelDeployment,
    FailoverMode,
    normalize_gateway_catalog,
)
from exp.common.models.gateway_pools import GatewayEquivalenceCertification, GatewayPoolRecord
from exp.runtime.gateway import native_rung_policy
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.health import DeploymentHealthRegistry
from exp.runtime.gateway.native_execution import InflightRequest, deployment_health_key
from exp.runtime.gateway.native_rung_policy import (
    failed_dispatch_candidate,
    reserve_rung_slot,
    shed_keeps_pin,
    shed_keeps_rung,
    throttle_redial_budgets,
)
from exp.runtime.gateway.routing import CatalogRouteResolver, GatewayRoute
from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed
from exp.runtime.gateway.sticky_affinity import StickySpillRegistry


def _deployment(
    deployment_id: str,
    *,
    connection_sha256: str,
    dispatch: GatewayRungDispatchPolicy | None = None,
) -> ExactModelDeployment:
    """Build one deployment in the shared certified exact-model pool."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai",
        provider_model="provider-model",
        connection_sha256=connection_sha256,
        capabilities_sha256="d" * 64,
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
            dispatch=dispatch,
        ),
    )


def _entry(
    deployments: tuple[ExactModelDeployment, ...],
    *,
    failover_mode: FailoverMode = "maximize_availability",
    throttle_cache_threshold: float | None = None,
    throttle_redial: GatewayThrottleRedialPolicy | None = None,
    affinity_fingerprint: bytes | None = None,
) -> InflightRequest:
    """Build one admitted request over the given rung ladder."""
    authorization = AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256="a" * 64,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )
    route = GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
            failover_mode=failover_mode,
            throttle_cache_threshold=throttle_cache_threshold,
            throttle_redial=throttle_redial,
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )
    return InflightRequest(
        authorization=authorization,
        route=route,
        request=GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="hello"),),
        ),
        deadline_monotonic=1.0,
        attempt_counts=[1, 0],
        total_attempts=1,
        affinity_fingerprint=affinity_fingerprint,
    )


_THROTTLE = GatewayFailure(
    failure_class=GatewayFailureClass.THROTTLED,
    safe_message="provider throttled the request",
    failover_eligible=True,
)


def test_reserve_rung_slot_is_inert_without_an_admission_policy() -> None:
    """A rung authoring no bound or rate window reserves nothing and sheds nothing."""
    loads = RungLoadRegistry()
    deployment = _deployment("deployment-a", connection_sha256="b" * 64)
    entry = _entry((deployment,))
    assert (
        reserve_rung_slot(
            loads, StickySpillRegistry(), entry, deployment, reserved_tokens=10, force=False
        )
        is None
    )
    assert loads.inflight(("deployment-a", "b" * 64)) == 0


def test_reserve_rung_slot_sheds_fresh_sessions_early_only_with_warm_standing_absent() -> None:
    """Under affinity a fingerprint without a live binding sheds at the early threshold."""
    loads = RungLoadRegistry()
    sticky = StickySpillRegistry()
    deployment = _deployment(
        "deployment-a",
        connection_sha256="b" * 64,
        dispatch=GatewayRungDispatchPolicy(
            concurrency_bound=4, fresh_session_spill_fraction=0.5, sticky_spill_seconds=60
        ),
    )
    warm = _entry(
        (deployment,), failover_mode="maximize_cache_affinity", affinity_fingerprint=b"warm"
    )
    sticky.bind(b"warm", "deployment-a", ttl_seconds=60.0)
    fresh = _entry(
        (deployment,), failover_mode="maximize_cache_affinity", affinity_fingerprint=b"fresh"
    )
    for _ in range(2):
        assert isinstance(
            reserve_rung_slot(loads, sticky, warm, deployment, reserved_tokens=0, force=False),
            str,
        )
    shed = reserve_rung_slot(loads, sticky, fresh, deployment, reserved_tokens=0, force=False)
    assert isinstance(shed, RungShed) and shed.reason == "fresh_session_spill"
    assert isinstance(
        reserve_rung_slot(loads, sticky, warm, deployment, reserved_tokens=0, force=False), str
    )


def test_failed_dispatch_candidate_reads_the_organizations_cache_on_the_failed_rung() -> None:
    """The disposition follows the requesting organization's EWMA on the throttled rung."""
    deployments = (
        _deployment("deployment-a", connection_sha256="b" * 64),
        _deployment("deployment-b", connection_sha256="c" * 64),
    )
    entry = _entry(deployments, throttle_cache_threshold=0.5)
    health = DeploymentHealthRegistry()
    keys = tuple(deployment_health_key(entry.authorization, item) for item in deployments)
    loads = RungLoadRegistry()

    # No evidence: fail over cold.
    assert failed_dispatch_candidate(
        health=health, loads=loads, keys=keys, entry=entry, failure=_THROTTLE, current_depth=0
    ) == (1, "throttle_failover_cold")
    # Another organization's warm cache on the rung does not count.
    loads.record_settle(
        ("deployment-a", "b" * 64), "organization-other", cached_tokens=1, input_tokens=1
    )
    assert failed_dispatch_candidate(
        health=health, loads=loads, keys=keys, entry=entry, failure=_THROTTLE, current_depth=0
    ) == (1, "throttle_failover_cold")
    # The requesting organization's own warm cache surfaces the throttle.
    loads.record_settle(
        ("deployment-a", "b" * 64), "organization-one", cached_tokens=9, input_tokens=10
    )
    assert failed_dispatch_candidate(
        health=health, loads=loads, keys=keys, entry=entry, failure=_THROTTLE, current_depth=0
    ) == (None, "throttle_surfaced_cache_preserving")
    # Without a threshold the same warm cache is inert and the mode rules.
    plain = _entry(deployments, failover_mode="maximize_cache")
    assert failed_dispatch_candidate(
        health=health, loads=loads, keys=keys, entry=plain, failure=_THROTTLE, current_depth=0
    ) == (None, None)


def test_authored_record_threshold_is_the_one_next_route_candidate_receives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every hop carries the authored value: record, normalized pool, route, decision.

    The platform authors ``GatewayPoolRecord.throttle_cache_threshold``; the
    engine normalizes it onto ``ExactModelPool``, the resolver stamps it onto
    the route's ``ExecutionSnapshot``, and the failed-dispatch decision hands
    exactly that value (with the organization's live cached fraction) to the
    frozen candidate policy. A drop at any hop would leave the platform
    authoring a control the waterfall silently ignores.
    """
    certification = GatewayEquivalenceCertification(
        certification_id="certification-threshold",
        provenance="operator comparison run 2026-09-10",
        evidence_sha256="e" * 64,
        certified_at=datetime(2026, 9, 10, tzinfo=UTC),
    )
    authored = ModelCatalog(
        connections={"openai": ConnectionConfig(provider="openai")},
        models={
            "route-a": ModelRecord(
                connection="openai",
                model="m-a",
                billing_source=BillingSource.HOST_MANAGED,
                gateway=GatewayDeploymentMetadata(exact_model_id="exact-threshold"),
            ),
            "route-b": ModelRecord(
                connection="openai",
                model="m-b",
                billing_source=BillingSource.HOST_MANAGED,
                gateway=GatewayDeploymentMetadata(exact_model_id="exact-threshold"),
            ),
        },
        gateway_pools={
            "threshold-pool": GatewayPoolRecord(
                exact_model_id="exact-threshold",
                deployment_aliases=("route-a", "route-b"),
                equivalence=certification,
                failover_mode="maximize_cache",
                throttle_cache_threshold=0.5,
            )
        },
    )
    normalized = normalize_gateway_catalog(authored)
    digest = normalized.identity_sha256()
    authorization = AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="threshold-pool"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256=digest,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )
    route = CatalogRouteResolver({("revision-one", digest): normalized}).resolve_direct(
        authorization
    )
    assert route.snapshot.throttle_cache_threshold == 0.5
    entry = InflightRequest(
        authorization=authorization,
        route=route,
        request=GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="hello"),),
        ),
        deadline_monotonic=1.0,
        attempt_counts=[1, 0],
        total_attempts=1,
    )
    loads = RungLoadRegistry()
    lead = route.deployments[0]
    loads.record_settle(
        (lead.deployment_id, lead.connection_sha256),
        "organization-one",
        cached_tokens=3,
        input_tokens=4,
    )
    received: dict[str, object] = {}

    def _capture(**kwargs: object) -> int | None:
        """Record the candidate policy's inputs instead of deciding."""
        received.update(kwargs)
        return None

    monkeypatch.setattr(native_rung_policy, "next_route_candidate", _capture)
    keys = tuple(deployment_health_key(authorization, item) for item in route.deployments)
    candidate, disposition = failed_dispatch_candidate(
        health=DeploymentHealthRegistry(),
        loads=loads,
        keys=keys,
        entry=entry,
        failure=_THROTTLE,
        current_depth=0,
    )
    assert candidate is None
    assert received["throttle_cache_threshold"] == 0.5
    assert received["failover_mode"] == "maximize_cache"
    assert received["cached_fraction"] == pytest.approx(0.75)
    assert received["current_depth"] == 0
    # The disclosure is computed from the same two inputs the policy received.
    assert disposition == "throttle_surfaced_cache_preserving"


_REDIAL = GatewayThrottleRedialPolicy(max_attempts=2, base_delay_ms=100, max_delay_ms=2_000)


def test_failed_dispatch_candidate_names_backoff_then_cold_under_a_redial_schedule() -> None:
    """With a schedule a throttle redials (disclosed), then advances cold, never surfaces."""
    deployments = (
        _deployment("deployment-a", connection_sha256="b" * 64),
        _deployment("deployment-b", connection_sha256="c" * 64),
    )
    entry = _entry(deployments, throttle_redial=_REDIAL)
    health = DeploymentHealthRegistry(throttle_seconds=30.0)
    keys = tuple(deployment_health_key(entry.authorization, item) for item in deployments)
    loads = RungLoadRegistry()
    health.failed(keys[0], _THROTTLE)

    # The data plane waited the backoff: the same rung, disclosed as the redial.
    assert failed_dispatch_candidate(
        health=health,
        loads=loads,
        keys=keys,
        entry=entry,
        failure=_THROTTLE,
        current_depth=0,
        throttle_backoff=True,
    ) == (0, "throttle_backoff")
    # The redial budget is spent: the cold advance names the bypassed warm rung.
    entry.attempt_counts = [3, 0]
    entry.throttle_redials = [2, 0]
    entry.total_attempts = 3
    assert failed_dispatch_candidate(
        health=health,
        loads=loads,
        keys=keys,
        entry=entry,
        failure=_THROTTLE,
        current_depth=0,
        throttle_backoff=True,
    ) == (1, "throttle_failover_cold")
    # Warm cache above an authored threshold no longer surfaces once a
    # schedule is authored: without the data plane's wait it advances cold.
    warm = _entry(deployments, throttle_cache_threshold=0.5, throttle_redial=_REDIAL)
    loads.record_settle(
        ("deployment-a", "b" * 64), "organization-one", cached_tokens=9, input_tokens=10
    )
    assert failed_dispatch_candidate(
        health=health, loads=loads, keys=keys, entry=warm, failure=_THROTTLE, current_depth=0
    ) == (1, "throttle_failover_cold")
    # A single rung past its redials is a plain exhausted throttle.
    single = _entry(deployments[:1], throttle_redial=_REDIAL)
    single.attempt_counts = [3]
    single.throttle_redials = [2]
    single.total_attempts = 3
    assert failed_dispatch_candidate(
        health=health,
        loads=loads,
        keys=keys[:1],
        entry=single,
        failure=_THROTTLE,
        current_depth=0,
        throttle_backoff=True,
    ) == (None, None)


def test_throttle_redial_budgets_scale_with_the_schedule_and_the_cache_at_stake() -> None:
    """No schedule: zero. Schedule alone: the full cap. Plus threshold: scaled by warm cache.

    The proportional rule is read on rungs that still have a cold alternative
    after them; the last rung of the admitted route always gets the full
    budget (there is nowhere to fail over), so a three-rung ladder shows both.
    """
    deployments = (
        _deployment("deployment-a", connection_sha256="b" * 64),
        _deployment("deployment-b", connection_sha256="c" * 64),
        _deployment("deployment-c", connection_sha256="e" * 64),
    )
    loads = RungLoadRegistry()
    plain = _entry(deployments, failover_mode="maximize_cache")
    assert throttle_redial_budgets(loads, plain.route, "organization-one") == (0, 0, 0)
    scheduled = _entry(deployments, throttle_redial=_REDIAL)
    assert throttle_redial_budgets(loads, scheduled.route, "organization-one") == (2, 2, 2)
    gated = _entry(deployments, throttle_cache_threshold=0.5, throttle_redial=_REDIAL)
    # No cache evidence: nothing to wait for where a colder rung follows, so
    # the first two fail over at once; the last rung waits the whole schedule.
    assert throttle_redial_budgets(loads, gated.route, "organization-one") == (0, 0, 2)
    # Another organization's warm cache on the rung does not count...
    loads.record_settle(
        ("deployment-a", "b" * 64), "organization-other", cached_tokens=9, input_tokens=10
    )
    assert throttle_redial_budgets(loads, gated.route, "organization-one") == (0, 0, 2)
    # ...the requesting organization's own does, rung by rung: a fraction at
    # or above the threshold earns the whole budget.
    loads.record_settle(
        ("deployment-a", "b" * 64), "organization-one", cached_tokens=9, input_tokens=10
    )
    assert throttle_redial_budgets(loads, gated.route, "organization-one") == (2, 0, 2)
    # Below the threshold the budget is the proportional share, so a request
    # with little cache at stake fails over sooner rather than never waiting.
    loads.record_settle(
        ("deployment-b", "c" * 64), "organization-one", cached_tokens=3, input_tokens=10
    )
    assert throttle_redial_budgets(loads, gated.route, "organization-one") == (2, 1, 2)
    # A zero threshold means every cache reading meets it: the full budget.
    free = _entry(deployments, throttle_cache_threshold=0.0, throttle_redial=_REDIAL)
    assert throttle_redial_budgets(loads, free.route, "organization-one") == (2, 2, 2)
    # Entries built without the admission step default to the full budget.
    assert scheduled.throttle_redial_budgets == (2, 2, 2)
    assert plain.throttle_redial_budgets == (0, 0, 0)


def test_throttle_redial_budget_is_the_full_schedule_where_no_cold_alternative_follows() -> None:
    """A rung with nothing to fail over to waits the whole schedule without cache evidence.

    The worker-local EWMA reads zero for an organization with no settled
    sample on this worker even when its conversation is warm at the
    provider; on the only (or last) live rung a zero budget would surface the
    throttle at once with a bounded wait still able to serve.
    """
    single = (_deployment("deployment-a", connection_sha256="b" * 64),)
    loads = RungLoadRegistry()
    gated = _entry(single, throttle_cache_threshold=0.5, throttle_redial=_REDIAL)
    assert loads.cached_fraction(("deployment-a", "b" * 64), "organization-one") == 0.0
    assert throttle_redial_budgets(loads, gated.route, "organization-one") == (2,)
    # Two live rungs: the first still fails over cold at once (proportional
    # rule, no evidence), the last waits the schedule.
    pair = (
        _deployment("deployment-a", connection_sha256="b" * 64),
        _deployment("deployment-b", connection_sha256="c" * 64),
    )
    gated = _entry(pair, throttle_cache_threshold=0.5, throttle_redial=_REDIAL)
    assert throttle_redial_budgets(loads, gated.route, "organization-one") == (0, 2)
    # Without a schedule the rule is inert: the last rung stays failover-only.
    plain = _entry(single, failover_mode="maximize_cache", throttle_cache_threshold=0.5)
    assert throttle_redial_budgets(loads, plain.route, "organization-one") == (0,)


def test_throttle_redial_budget_is_the_full_schedule_on_the_reasoning_pinned_rung() -> None:
    """The issuing rung of a reasoning continuation waits the whole schedule.

    Its fallbacks dispatch without the request's thinking, so a throttle there
    is worth every redial the pool authored before the ladder advances, whatever
    the worker-local cache EWMA reads; the fallbacks keep the proportional rule.
    """
    deployments = (
        _deployment("deployment-a", connection_sha256="b" * 64),
        _deployment("deployment-b", connection_sha256="c" * 64),
        _deployment("deployment-c", connection_sha256="e" * 64),
    )
    loads = RungLoadRegistry()
    gated = _entry(deployments, throttle_cache_threshold=0.5, throttle_redial=_REDIAL)
    assert throttle_redial_budgets(loads, gated.route, "organization-one") == (0, 0, 2)
    pinned = gated.route.model_copy(
        update={
            "route_reason": "reasoning_continuation",
            "reasoning_pinned_deployment_id": "deployment-a",
        }
    )
    assert throttle_redial_budgets(loads, pinned, "organization-one") == (2, 0, 2)


def test_throttle_redial_budget_is_the_full_schedule_on_the_warm_sticky_rung() -> None:
    """A live sticky binding on a rung is cache evidence for the whole schedule there.

    The binding says the conversation's provider cache lives on that rung,
    so the missing worker-local EWMA sample cannot zero its budget; a rung the
    binding does not name keeps the proportional rule, and a binding to a
    rung outside the route changes nothing.
    """
    deployments = (
        _deployment("deployment-a", connection_sha256="b" * 64),
        _deployment("deployment-b", connection_sha256="c" * 64),
        _deployment("deployment-c", connection_sha256="e" * 64),
    )
    loads = RungLoadRegistry()
    gated = _entry(
        deployments,
        failover_mode="maximize_cache_affinity",
        throttle_cache_threshold=0.5,
        throttle_redial=_REDIAL,
    )
    assert throttle_redial_budgets(
        loads, gated.route, "organization-one", sticky_deployment_id="deployment-a"
    ) == (2, 0, 2)
    assert throttle_redial_budgets(
        loads, gated.route, "organization-one", sticky_deployment_id="deployment-b"
    ) == (0, 2, 2)
    assert throttle_redial_budgets(
        loads, gated.route, "organization-one", sticky_deployment_id="deployment-elsewhere"
    ) == (0, 0, 2)
    # The binding does not lower a budget the EWMA already earned elsewhere.
    loads.record_settle(
        ("deployment-a", "b" * 64), "organization-one", cached_tokens=9, input_tokens=10
    )
    assert throttle_redial_budgets(
        loads, gated.route, "organization-one", sticky_deployment_id="deployment-b"
    ) == (2, 2, 2)


def test_shed_keeps_rung_force_admits_the_redialed_rung_and_the_first_pinned_dispatch() -> None:
    """A backoff redial keeps its rung through a shed; a pinned issuing rung only pre-failure."""
    deployments = (
        _deployment("deployment-a", connection_sha256="b" * 64),
        _deployment("deployment-b", connection_sha256="c" * 64),
    )
    route = _entry(deployments).route
    # The redialed rung is kept whatever the failure history or pin state.
    assert shed_keeps_rung(route, 0, 0, _THROTTLE) is True
    assert shed_keeps_rung(route, 0, 0, None) is True
    # A shed on any other rung of a failed ladder spills sideways.
    assert shed_keeps_rung(route, 1, 0, _THROTTLE) is False
    assert shed_keeps_rung(route, 1, None, _THROTTLE) is False
    # Without a pin, a first-dispatch shed spills too.
    assert shed_keeps_rung(route, 0, None, None) is False
    assert shed_keeps_pin(route, 0) is False
    pinned = route.model_copy(update={"reasoning_pinned_deployment_id": "deployment-a"})
    assert shed_keeps_pin(pinned, 0) is True
    assert shed_keeps_pin(pinned, 1) is False
    # The pinned issuing rung is kept on its first dispatch, not after a real failure on it.
    assert shed_keeps_rung(pinned, 0, None, None) is True
    assert shed_keeps_rung(pinned, 0, None, _THROTTLE) is False
    assert shed_keeps_rung(pinned, 1, None, None) is False
