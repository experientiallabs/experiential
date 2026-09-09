"""Tests for deployment health circuit classification of provider failures."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry

_KEY: DeploymentHealthKey = ("catalog", "deployment", "connection")


def _failure(failure_class: GatewayFailureClass) -> GatewayFailure:
    """Build one sanitized failure of the given class for circuit accounting.

    Args:
        failure_class: Failure class applied to the health registry.

    Returns:
        A minimal sanitized failure carrying only the class under test.
    """
    return GatewayFailure(failure_class=failure_class, safe_message="scripted failure")


def test_caller_invalid_request_bursts_never_open_the_circuit() -> None:
    """A storm of caller-fault rejections keeps the deployment fully admissible."""
    registry = DeploymentHealthRegistry(failure_threshold=2, clock=lambda: 100.0)

    for _ in range(50):
        registry.failed(_KEY, _failure(GatewayFailureClass.INVALID_REQUEST))

    assert registry.claim(_KEY)


def test_unsupported_capability_rejections_never_open_the_circuit() -> None:
    """Preflight capability rejections are caller-corrected and never suppress routes."""
    registry = DeploymentHealthRegistry(failure_threshold=2, clock=lambda: 100.0)

    for _ in range(50):
        registry.failed(_KEY, _failure(GatewayFailureClass.UNSUPPORTED_CAPABILITY))

    assert registry.claim(_KEY)


def test_caller_invalid_requests_do_not_reset_operational_failure_progress() -> None:
    """Interleaved caller faults neither add to nor clear genuine failure counts."""
    registry = DeploymentHealthRegistry(failure_threshold=2, clock=lambda: 100.0)

    registry.failed(_KEY, _failure(GatewayFailureClass.TRANSPORT))
    registry.failed(_KEY, _failure(GatewayFailureClass.INVALID_REQUEST))
    assert registry.claim(_KEY)
    registry.failed(_KEY, _failure(GatewayFailureClass.TRANSPORT))

    assert not registry.claim(_KEY)


def test_operational_failures_still_open_the_circuit() -> None:
    """Genuine provider health failures keep opening the circuit at the threshold."""
    for failure_class in (
        GatewayFailureClass.TRANSPORT,
        GatewayFailureClass.TIMEOUT,
        GatewayFailureClass.MALFORMED_RESPONSE,
        GatewayFailureClass.PROVIDER_INTERNAL,
    ):
        registry = DeploymentHealthRegistry(failure_threshold=2, clock=lambda: 100.0)

        registry.failed(_KEY, _failure(failure_class))
        registry.failed(_KEY, _failure(failure_class))

        assert not registry.claim(_KEY)


def test_provider_quota_opens_the_circuit_immediately_like_other_hard_deadness() -> None:
    """An unfunded provider account (402) suppresses the rung after ONE failure.

    Without this, the billing-dead deployment stays first-in-line and burns one
    wasted attempt per request before every failover; with it, the circuit opens
    at once and the half-open probe rediscovers the rung when the operator
    funds or enables the account.
    """
    registry = DeploymentHealthRegistry(failure_threshold=2, clock=lambda: 100.0)

    registry.failed(_KEY, _failure(GatewayFailureClass.PROVIDER_QUOTA))

    assert not registry.claim(_KEY)


def test_throttle_storms_still_suppress_the_deployment() -> None:
    """Provider throttling keeps its authoritative suppression window."""
    registry = DeploymentHealthRegistry(throttle_seconds=30.0, clock=lambda: 100.0)

    registry.failed(_KEY, _failure(GatewayFailureClass.THROTTLED))

    assert not registry.claim(_KEY)


def test_forced_claim_admits_every_request_through_an_open_circuit() -> None:
    """Forced claims stay available for concurrent traffic once probes are taken."""
    registry = DeploymentHealthRegistry(failure_threshold=1, clock=lambda: 100.0)

    registry.failed(_KEY, _failure(GatewayFailureClass.TRANSPORT))
    assert not registry.claim(_KEY)
    assert registry.claim_last_resort(_KEY)
    assert not registry.claim_last_resort(_KEY)

    assert registry.claim_forced(_KEY)
    assert registry.claim_forced(_KEY)


def test_forced_claim_still_respects_the_throttle_window() -> None:
    """A provider-requested backoff window refuses even forced dispatch."""
    now = [100.0]
    registry = DeploymentHealthRegistry(throttle_seconds=30.0, clock=lambda: now[0])

    registry.failed(_KEY, _failure(GatewayFailureClass.THROTTLED))

    assert not registry.claim_forced(_KEY)
    now[0] += 31
    assert registry.claim_forced(_KEY)


@pytest.mark.parametrize(
    ("retry_after", "expected_window"),
    [(2, 5.0), (7_200, 7_200.0), (999_999, 21_600.0)],
)
def test_retry_after_sizes_the_throttle_window_clamped(
    retry_after: int, expected_window: float
) -> None:
    """A provider-stated wait sizes the window inside the [5s, 6h] clamp.

    A short or degenerate wait floors at five seconds, an in-range wait (an
    hourly quota reset) suppresses for exactly what the provider asked, and an
    absurd wait ceilings at six hours; a throttle carrying no wait keeps the
    fixed default window.
    """
    now = [100.0]
    registry = DeploymentHealthRegistry(throttle_seconds=30.0, clock=lambda: now[0])
    registry.failed(
        _KEY,
        GatewayFailure(
            failure_class=GatewayFailureClass.THROTTLED,
            safe_message="scripted throttle",
            retry_after_seconds=retry_after,
        ),
    )
    now[0] = 100.0 + expected_window - 0.5
    assert not registry.claim(_KEY)
    now[0] = 100.0 + expected_window + 0.5
    assert registry.claim(_KEY)


def test_throttle_without_retry_after_keeps_the_default_window() -> None:
    """An absent wait falls back to the registry's fixed throttle window."""
    now = [100.0]
    registry = DeploymentHealthRegistry(throttle_seconds=30.0, clock=lambda: now[0])
    registry.failed(_KEY, _failure(GatewayFailureClass.THROTTLED))
    now[0] = 129.0
    assert not registry.claim(_KEY)
    now[0] = 131.0
    assert registry.claim(_KEY)


def test_suppressed_is_a_read_only_probe() -> None:
    """``suppressed`` reports throttle and circuit windows without claiming.

    Unlike ``claim`` it must not consume the half-open probe: a sticky-affinity
    lookup that peeked would otherwise steal the one probe real dispatch needs.
    """
    now = [100.0]
    registry = DeploymentHealthRegistry(
        failure_threshold=1, open_seconds=30.0, throttle_seconds=30.0, clock=lambda: now[0]
    )
    assert not registry.suppressed(_KEY)
    registry.failed(_KEY, _failure(GatewayFailureClass.THROTTLED))
    assert registry.suppressed(_KEY)
    now[0] = 131.0
    assert not registry.suppressed(_KEY)
    registry.failed(_KEY, _failure(GatewayFailureClass.TRANSPORT))
    assert registry.suppressed(_KEY)
    for _ in range(3):
        assert registry.suppressed(_KEY)
    now[0] = 162.0
    assert not registry.suppressed(_KEY)
    # The half-open probe is still available to the first real claim.
    assert registry.claim(_KEY)


def test_throttled_remaining_seconds_names_a_fully_throttled_route() -> None:
    """The longest remaining window is reported only when EVERY key is inside
    a throttle window; any dispatchable deployment (or an empty route) yields
    None so the caller-facing throttled class is never invented."""
    now = [100.0]
    registry = DeploymentHealthRegistry(throttle_seconds=30.0, clock=lambda: now[0])
    first: DeploymentHealthKey = ("catalog", "deployment-one", "connection")
    second: DeploymentHealthKey = ("catalog", "deployment-two", "connection")

    assert registry.throttled_remaining_seconds(()) is None
    assert registry.throttled_remaining_seconds((first,)) is None

    registry.failed(first, _failure(GatewayFailureClass.THROTTLED))
    now[0] += 10.0
    registry.failed(second, _failure(GatewayFailureClass.THROTTLED))
    remaining = registry.throttled_remaining_seconds((first, second))
    assert remaining == 30.0

    # One key outside its window makes the route dispatchable again.
    now[0] += 21.0
    assert registry.throttled_remaining_seconds((first, second)) is None


def test_exhausted_plan_window_suppresses_the_rung_for_the_stated_reset() -> None:
    """A reported 100 percent window throttles the deployment until the provider's reset."""
    now = [1_000.0]
    registry = DeploymentHealthRegistry(clock=lambda: now[0])
    key = ("catalog", "plan-a", "connection")

    registry.exhausted(key, 3_600)

    assert registry.suppressed(key)
    assert not registry.claim(key)
    assert not registry.claim_forced(key)
    now[0] += 3_601
    assert not registry.suppressed(key)
    assert registry.claim(key)


def test_exhaustion_windows_are_clamped_and_never_shortened() -> None:
    """A week-long reset is clamped to the ceiling; a shorter later report keeps the longer wait."""
    now = [0.0]
    registry = DeploymentHealthRegistry(clock=lambda: now[0])
    key = ("catalog", "plan-a", "connection")

    registry.exhausted(key, 7 * 24 * 3_600)
    now[0] = 6 * 3_600 - 1
    assert registry.suppressed(key)
    now[0] = 6 * 3_600 + 1
    assert not registry.suppressed(key)

    registry.exhausted(key, 1_000)
    registry.exhausted(key, 1)
    now[0] += 900
    assert registry.suppressed(key)
