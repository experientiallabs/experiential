"""Tests for deployment health circuit classification of provider failures."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.health import (
    DeploymentHealthKey,
    DeploymentHealthRegistry,
    health_failure_cause,
)

_KEY: DeploymentHealthKey = ("catalog", "deployment", "connection")


def _failure(failure_class: GatewayFailureClass) -> GatewayFailure:
    """Build one sanitized failure of the given class for circuit accounting.

    Args:
        failure_class: Failure class applied to the health registry.

    Returns:
        A minimal sanitized failure carrying only the class under test.
    """
    return GatewayFailure(failure_class=failure_class, safe_message="scripted failure")


@pytest.mark.parametrize("failure_class", list(GatewayFailureClass))
@pytest.mark.parametrize("customer_owned", [False, True])
def test_shared_failure_categories_preserve_every_circuit_policy(
    failure_class: GatewayFailureClass, customer_owned: bool
) -> None:
    """Every failure keeps its threshold, cooldown and refusal accounting semantics."""
    operational = {
        GatewayFailureClass.TRANSPORT,
        GatewayFailureClass.TIMEOUT,
        GatewayFailureClass.MALFORMED_RESPONSE,
        GatewayFailureClass.PROVIDER_INTERNAL,
    }
    hard = {
        GatewayFailureClass.PROVIDER_AUTHENTICATION,
        GatewayFailureClass.PROVIDER_NOT_FOUND,
        GatewayFailureClass.PROVIDER_QUOTA,
    }
    expected = (
        "transport"
        if failure_class in operational
        else "credential"
        if failure_class in hard
        else "throttle"
        if failure_class == GatewayFailureClass.THROTTLED
        else None
    )
    assert health_failure_cause(failure_class) == expected
    now = [100.0]
    registry = DeploymentHealthRegistry(
        failure_threshold=2, open_seconds=30, throttle_seconds=20, clock=lambda: now[0]
    )
    failure = _failure(failure_class).model_copy(update={"customer_owned": customer_owned})
    registry.failed(_KEY, failure)
    assert registry.suppressed(_KEY) is (expected in ("credential", "throttle"))
    registry.failed(_KEY, failure)
    assert registry.suppressed(_KEY) is (expected is not None)
    state = registry._states[_KEY]  # noqa: SLF001 - preserve refusal-only bookkeeping too.
    assert state.refusal_count == (2 if failure_class == GatewayFailureClass.REFUSAL else 0)
    now[0] += 21
    assert registry.suppressed(_KEY) is (expected in ("credential", "transport"))
    now[0] += 10
    assert registry.claim(_KEY)


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


def test_empty_completions_never_open_the_circuit_but_release_probe_state() -> None:
    """An empty completion is the model's answer to the content, not rung deadness.

    2026-09-15: one Claude Code session re-sent the same prompt every minute
    and OpenAI answered each with a 4-token empty message; as a
    ``provider_internal`` those failures would have opened the luna rung's
    circuit on that worker for every other caller. The class never counts
    toward the threshold and never resets genuine progress, while a probe
    that answered empty is still consumed like any other outcome.
    """
    registry = DeploymentHealthRegistry(failure_threshold=2, clock=lambda: 100.0)

    for _ in range(50):
        registry.failed(_KEY, _failure(GatewayFailureClass.EMPTY_COMPLETION))
    assert registry.claim(_KEY)

    registry.failed(_KEY, _failure(GatewayFailureClass.TRANSPORT))
    registry.failed(_KEY, _failure(GatewayFailureClass.EMPTY_COMPLETION))
    assert registry.claim(_KEY)
    registry.failed(_KEY, _failure(GatewayFailureClass.TRANSPORT))
    assert not registry.claim(_KEY)

    state = registry._states[_KEY]  # noqa: SLF001 - probe bookkeeping is the assertion.
    state.half_open_probe = True
    registry.failed(_KEY, _failure(GatewayFailureClass.EMPTY_COMPLETION))
    assert not state.half_open_probe


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


def test_throttle_redial_claim_passes_the_window_but_not_an_open_circuit() -> None:
    """A post-backoff redial re-enters the throttled rung; a dead rung still refuses it."""
    now = [100.0]
    registry = DeploymentHealthRegistry(
        failure_threshold=1, throttle_seconds=30.0, clock=lambda: now[0]
    )

    registry.failed(_KEY, _failure(GatewayFailureClass.THROTTLED))
    # Every other claim honors the window the provider asked for...
    assert not registry.claim(_KEY)
    assert not registry.claim_last_resort(_KEY)
    assert not registry.claim_forced(_KEY)
    # ...while the request that waited the backoff is the one probing back.
    assert registry.claim_throttle_redial(_KEY)

    # Operational deadness marked meanwhile opens the circuit: no redial.
    registry.failed(_KEY, _failure(GatewayFailureClass.TRANSPORT))
    assert not registry.claim_throttle_redial(_KEY)
