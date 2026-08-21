"""Tests for deployment health circuit classification of provider failures."""

from __future__ import annotations

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


def test_throttle_storms_still_suppress_the_deployment() -> None:
    """Provider throttling keeps its authoritative suppression window."""
    registry = DeploymentHealthRegistry(throttle_seconds=30.0, clock=lambda: 100.0)

    registry.failed(_KEY, _failure(GatewayFailureClass.THROTTLED))

    assert not registry.claim(_KEY)
