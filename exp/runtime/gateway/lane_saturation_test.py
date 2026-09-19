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
    DEFAULT_LANE_SHARE,
    LANE_SATURATED_RETRY_AFTER_SECONDS,
    default_lane_bound,
    lane_saturated_failure,
    overflow_target,
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


def _route(*deployments: ExactModelDeployment) -> GatewayRoute:
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
    assert f"retry in {LANE_SATURATED_RETRY_AFTER_SECONDS} seconds" in failure.safe_message
    assert "in-flight bound" in failure.safe_message
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
