"""Tests for provider refusal signals and sanitized gateway failure classification."""

from __future__ import annotations

import asyncio

import pytest

from exp.runtime.gateway.contracts import GatewayFailureClass
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderRefusalError,
    ProviderRefusalSignal,
    ProviderResponseError,
    normalized_provider_failure,
)
from exp.runtime.models.providers.transport import ProviderTransportError


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
            ProviderTransportError("raw invalid canary", status_code=400),
            GatewayFailureClass.INVALID_REQUEST,
            False,
            False,
        ),
        (
            ProviderTransportError("raw reject canary", status_code=422),
            GatewayFailureClass.INVALID_REQUEST,
            False,
            False,
        ),
        (
            ProviderTransportError("raw conflict canary", status_code=409),
            GatewayFailureClass.PROVIDER_INTERNAL,
            True,
            True,
        ),
        (
            ProviderTransportError("raw early canary", status_code=425),
            GatewayFailureClass.PROVIDER_INTERNAL,
            True,
            True,
        ),
        (
            ProviderTransportError("raw redirect canary", status_code=302),
            GatewayFailureClass.PROVIDER_INTERNAL,
            False,
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


@pytest.mark.parametrize(
    ("exception", "safe_message"),
    [
        (
            ProviderTransportError("raw auth canary", status_code=401),
            "provider authentication failed; ask the gateway operator to verify "
            "the provider connection credential",
        ),
        (
            ProviderTransportError("raw missing canary", status_code=404),
            "provider deployment was not found; ask the gateway operator to verify "
            "the deployment model ID in the catalog",
        ),
        (
            ProviderTransportError("raw throttle canary", status_code=429),
            "provider throttled the request; retry after the delay in the Retry-After header",
        ),
        (
            ProviderTransportError("raw slow canary", status_code=408),
            "provider request timed out; retry the request",
        ),
        (
            ProviderTransportError("raw server canary", status_code=503),
            "provider service failed; retry after a short delay",
        ),
        (
            ProviderTransportError("raw reject canary", status_code=422),
            "provider rejected the request; verify the request fields against "
            "the model alias capabilities",
        ),
        (
            ProviderTransportError("raw conflict canary", status_code=409),
            "provider reported a transient conflict; retry the request",
        ),
        (
            ProviderTransportError("raw redirect canary", status_code=302),
            "provider returned an unexpected status; retry the request",
        ),
        (
            ProviderTransportError("raw body canary"),
            "provider transport failed; retry the request",
        ),
        (
            ProviderDeadlineExceeded("raw deadline canary"),
            "provider request deadline exceeded; retry with a shorter prompt "
            "or a smaller max_tokens value",
        ),
        (
            ProviderResponseError("raw response canary"),
            "provider returned a malformed response; retry the request",
        ),
        (
            RuntimeError("raw internal canary"),
            "provider execution failed; retry the request",
        ),
    ],
)
def test_safe_messages_state_the_failure_and_the_next_action(
    exception: BaseException,
    safe_message: str,
) -> None:
    """Every public message says what failed and exactly what to do next."""
    failure = normalized_provider_failure(exception)

    assert failure.safe_message == safe_message
    assert "canary" not in failure.model_dump_json()


def test_cancellation_has_a_dedicated_nonretryable_failure() -> None:
    """Client disconnect cancellation must not be retried or made failover eligible."""
    failure = normalized_provider_failure(asyncio.CancelledError())

    assert failure.failure_class is GatewayFailureClass.CANCELLED
    assert failure.retryable_same_deployment is False
    assert failure.failover_eligible is False
