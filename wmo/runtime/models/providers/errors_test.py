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
from wmo.runtime.openai_protocol.errors import public_failure_error


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
            ProviderTransportError("raw bad-request canary", status_code=400),
            GatewayFailureClass.INVALID_REQUEST,
            False,
            True,
        ),
        (
            ProviderTransportError("raw unprocessable canary", status_code=422),
            GatewayFailureClass.INVALID_REQUEST,
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


def test_client_4xx_surfaces_as_an_actionable_400_not_all_routes_failed() -> None:
    """A deterministic provider 4xx reaches the caller as a correctable 400.

    Reasoning models reject a non-default ``temperature`` with a provider 400,
    and an oversized prompt or bad parameter does the same. Before, every such
    rejection collapsed into the generic 502 ``all_routes_failed`` (an outage
    signal telling an agent to retry with backoff). It must instead surface as
    an ``invalid_request_error`` 400 with the offending status retained for
    telemetry, so the caller self-corrects rather than retrying a doomed call.
    """
    failure = normalized_provider_failure(
        ProviderTransportError("raw temperature canary", status_code=400)
    )
    assert failure.failure_class is GatewayFailureClass.INVALID_REQUEST
    assert failure.safe_details == {"status_code": 400}

    error = public_failure_error(failure)
    assert error.status_code == 400
    assert error.detail.code == "invalid_request"
    assert error.detail.type == "invalid_request_error"

    server = normalized_provider_failure(
        ProviderTransportError("raw outage canary", status_code=503)
    )
    # A genuine multi-provider outage keeps its all-routes-failed 502 semantics.
    assert public_failure_error(server).status_code == 502


def test_cancellation_has_a_dedicated_nonretryable_failure() -> None:
    """Client disconnect cancellation must not be retried or made failover eligible."""
    failure = normalized_provider_failure(asyncio.CancelledError())

    assert failure.failure_class is GatewayFailureClass.CANCELLED
    assert failure.retryable_same_deployment is False
    assert failure.failover_eligible is False
