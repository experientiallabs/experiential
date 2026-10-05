"""Tests for the default lane bound and the refuse-instead-of-overflow rule."""

from __future__ import annotations

import pytest

from exp.common.models.catalog import GatewayDeploymentCapabilities, GatewayDeploymentMetadata
from exp.common.models.dispatch_policy import GatewayRungDispatchPolicy
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
)
from exp.runtime.gateway.lane_saturation import (
    DEFAULT_BOUND_OVERFLOW_FACTORS,
    DEFAULT_LANE_SHARE,
    LANE_SATURATED_RETRY_AFTER_SECONDS,
    PRIORITY_OVERFLOW_FACTORS,
    default_lane_bound,
    lane_saturated_failure,
    overflow_target,
    priority_overflow_ceiling,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.rung_admission import RungShed
from exp.runtime.gateway.stream_contracts import GatewayFailureClass


def _deployment(
    deployment_id: str, dispatch: GatewayRungDispatchPolicy | None
) -> ExactModelDeployment:
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai",
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="d" * 64,
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
            dispatch=dispatch,
        ),
    )


def _route(*deployments: ExactModelDeployment, priority_admission: int = 0) -> GatewayRoute:
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
        priority_admission=priority_admission,
    )
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


def test_default_lane_bound_is_a_share_of_the_workers_permits_rounded_up() -> None:
    """Half of 64 permits is 32; odd counts round up; the floor is one."""
    assert DEFAULT_LANE_SHARE == 0.5
    assert default_lane_bound(64) == 32
    assert default_lane_bound(7) == 4
    assert default_lane_bound(1) == 1
    assert default_lane_bound(64, share=0.25) == 16
    assert default_lane_bound(64, share=1.0) == 64


@pytest.mark.parametrize(("permits", "share"), [(0, 0.5), (64, 0.0), (64, 1.5), (64, -0.1)])
def test_default_lane_bound_refuses_meaningless_inputs(permits: int, share: float) -> None:
    """A non-positive permit count or a share outside (0, 1] is a programming error."""
    with pytest.raises(ValueError):
        default_lane_bound(permits, share=share)


def test_lane_saturated_failure_is_a_retryable_throttle_with_the_wait_it_states() -> None:
    """Nothing is down: the pool is full here, so the caller gets a 429 with Retry-After."""
    failure = lane_saturated_failure()
    assert failure.failure_class is GatewayFailureClass.THROTTLED
    assert failure.retry_after_seconds == LANE_SATURATED_RETRY_AFTER_SECONDS
    assert failure.safe_message == (
        "This model is at capacity right now. Please retry in a few seconds. "
        "Pro subscribers get priority access when models are busy."
    )
    assert failure.failover_eligible is False


def test_overflow_target_keeps_the_historical_overflow_for_an_authored_bound() -> None:
    """An authored bound with the default saturation still force-admits its first shed rung."""
    route = _route(
        _deployment("a", GatewayRungDispatchPolicy(concurrency_bound=1)),
        _deployment("b", GatewayRungDispatchPolicy(concurrency_bound=1)),
    )
    sheds = {0: RungShed("queue_bound"), 1: RungShed("queue_bound")}
    assert overflow_target(route, [(0, "queue_bound"), (1, "queue_bound")], sheds) == 0


def test_overflow_target_refuses_when_the_first_shed_rung_authors_refuse() -> None:
    """``saturation="refuse"`` on the bypassed rung turns the overflow into a fast refusal."""
    route = _route(
        _deployment("a", GatewayRungDispatchPolicy(concurrency_bound=1, saturation="refuse")),
        _deployment("b", GatewayRungDispatchPolicy(concurrency_bound=1)),
    )
    sheds = {0: RungShed("queue_bound"), 1: RungShed("queue_bound")}
    assert overflow_target(route, [(0, "queue_bound"), (1, "queue_bound")], sheds) is None


def test_overflow_target_overflows_a_refusing_rung_for_a_priority_caller() -> None:
    """A priority caller overflows both refusing bounds: an authored refuse and the default."""
    refusing = (
        _deployment("a", GatewayRungDispatchPolicy(concurrency_bound=1, saturation="refuse")),
        _deployment("b", GatewayRungDispatchPolicy(concurrency_bound=1)),
    )
    sheds = {0: RungShed("queue_bound"), 1: RungShed("queue_bound")}
    shed_order = [(0, "queue_bound"), (1, "queue_bound")]
    assert overflow_target(_route(*refusing, priority_admission=2), shed_order, sheds) == 0
    # The worker's default bound overflows for a priority caller too (the
    # reservation caps it at its level's ceiling).
    unauthored = _route(_deployment("a", None), _deployment("b", None), priority_admission=2)
    default_shed = {0: RungShed("queue_bound", default_bound=True)}
    assert overflow_target(unauthored, [(0, "queue_bound")], default_shed) == 0


def test_overflow_target_never_force_admits_past_the_default_lane_bound() -> None:
    """A shed by the worker's default share refuses: overflowing it would protect nothing."""
    route = _route(_deployment("a", None), _deployment("b", None))
    sheds = {0: RungShed("queue_bound", default_bound=True)}
    assert overflow_target(route, [(0, "queue_bound")], sheds) is None


def test_overflow_target_keeps_the_overflow_for_a_cold_throttle_bypass() -> None:
    """A bypass that was not a registry shed (a cold throttle failover) overflows as before."""
    route = _route(_deployment("a", None), _deployment("b", None))
    assert overflow_target(route, [(0, "throttle_failover_cold")], {}) == 0


def test_overflow_target_has_nothing_to_overflow_without_a_shed() -> None:
    """No policy bypass means no overflow target (the caller reads the exhaustion elsewhere)."""
    route = _route(_deployment("a", None))
    assert overflow_target(route, [], {}) is None


def test_priority_overflow_ceiling_scales_the_bound_by_level() -> None:
    """Free callers get no overflow; paying callers 1.25x the bound; Pro callers 1.5x."""
    assert priority_overflow_ceiling(4, 0, default_bound=False) is None
    assert priority_overflow_ceiling(4, 1, default_bound=False) == 5.0
    assert priority_overflow_ceiling(4, 2, default_bound=False) == 6.0
    assert PRIORITY_OVERFLOW_FACTORS[1] < PRIORITY_OVERFLOW_FACTORS[2] <= 1.5
    assert priority_overflow_ceiling(None, 2, default_bound=False) is None
    # The default bound is half the worker's permits: Pro stays below all of them.
    assert priority_overflow_ceiling(32, 1, default_bound=True) == 40.0
    assert priority_overflow_ceiling(32, 2, default_bound=True) == 48.0
    assert DEFAULT_BOUND_OVERFLOW_FACTORS[2] < 1 / DEFAULT_LANE_SHARE


def test_overflow_target_refuses_a_shed_at_the_priority_ceiling() -> None:
    """A forced priority overflow that hit its level's ceiling is refused, never retried."""
    route = _route(_deployment("a", None), priority_admission=2)
    capped = {0: RungShed("queue_bound", overflow_ceiling=True)}
    assert overflow_target(route, [(0, "queue_bound")], capped) is None


def test_overflow_target_moves_past_a_capped_rung_to_the_next_bypassed_rung() -> None:
    """A priority caller capped on the first rung overflows the next one still below its cap."""
    route = _route(_deployment("a", None), _deployment("b", None), priority_admission=2)
    sheds = {
        0: RungShed("queue_bound", overflow_ceiling=True),
        1: RungShed("queue_bound", default_bound=True),
    }
    assert overflow_target(route, [(0, "queue_bound"), (1, "queue_bound")], sheds) == 1
    sheds[1] = RungShed("queue_bound", overflow_ceiling=True)
    assert overflow_target(route, [(0, "queue_bound"), (1, "queue_bound")], sheds) is None


def test_a_rate_window_shed_never_earns_the_priority_exception() -> None:
    """Priority overflows capacity only: a rate shed keeps the free caller's rule."""
    refusing = _route(
        _deployment("a", GatewayRungDispatchPolicy(requests_per_minute=1, saturation="refuse")),
        priority_admission=2,
    )
    rate = {0: RungShed("rate_limit")}
    assert overflow_target(refusing, [(0, "rate_limit")], rate) is None
    soft = _route(
        _deployment("a", GatewayRungDispatchPolicy(requests_per_minute=1)), priority_admission=2
    )
    assert overflow_target(soft, [(0, "rate_limit")], rate) == 0
    selected = soft.model_copy(update={"resolved_route_id": "route_" + "a" * 64})
    assert overflow_target(selected, [(0, "rate_limit")], rate) is None
