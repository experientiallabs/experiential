"""Tests for provider refusal signals and sanitized gateway failure classification."""

from __future__ import annotations

import asyncio

import pytest

from wmo.runtime.gateway.contracts import GatewayFailureClass
from wmo.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from wmo.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderRefusalError,
    ProviderRefusalSignal,
    ProviderResponseError,
    normalized_provider_failure,
)
from wmo.runtime.models.providers.transport import ProviderTransportError


@pytest.mark.parametrize(
    ("exception", "failure_class", "retryable", "failover"),
    [
        (
            ProviderTransportError("raw auth canary", status_code=401),
            GatewayFailureClass.PROVIDER_AUTHENTICATION,
            False,
            True,
        ),
        (
            ProviderTransportError("raw throttle canary", status_code=429),
            GatewayFailureClass.THROTTLED,
            False,
            True,
        ),
        (
            ProviderTransportError("raw server canary", status_code=503),
            GatewayFailureClass.PROVIDER_INTERNAL,
            True,
            True,
        ),
        (
            ProviderTransportError("raw body canary"),
            GatewayFailureClass.TRANSPORT,
            True,
            True,
        ),
        (
            ProviderDeadlineExceeded("raw deadline canary"),
            GatewayFailureClass.TIMEOUT,
            False,
            True,
        ),
        (
            ProviderResponseError("raw response canary"),
            GatewayFailureClass.MALFORMED_RESPONSE,
            False,
            True,
        ),
    ],
)
def test_failures_are_typed_without_raw_provider_content(
    exception: BaseException,
    failure_class: GatewayFailureClass,
    retryable: bool,
    failover: bool,
) -> None:
    """Normalized failures retain policy but discard raw exception messages."""
    failure = normalized_provider_failure(exception)

    assert failure.failure_class is failure_class
    assert failure.retryable_same_deployment is retryable
    assert failure.failover_eligible is failover
    assert "canary" not in failure.model_dump_json()


def test_refusal_and_capability_failures_keep_only_safe_signals() -> None:
    """Refusal and preflight failures expose categories without model-generated text."""
    refusal = normalized_provider_failure(
        ProviderRefusalError(
            provider="fixture",
            signal=ProviderRefusalSignal.SAFETY,
        )
    )
    capability = normalized_provider_failure(ProviderCapabilityError(capability="strict_tools"))

    assert refusal.failure_class is GatewayFailureClass.REFUSAL
    assert refusal.safe_details == {"signal": "safety"}
    assert capability.failure_class is GatewayFailureClass.UNSUPPORTED_CAPABILITY
    assert capability.safe_details == {"capability": "strict_tools"}


def test_cancellation_has_a_dedicated_nonretryable_failure() -> None:
    """Client disconnect cancellation must not be retried or made failover eligible."""
    failure = normalized_provider_failure(asyncio.CancelledError())

    assert failure.failure_class is GatewayFailureClass.CANCELLED
    assert failure.retryable_same_deployment is False
    assert failure.failover_eligible is False
