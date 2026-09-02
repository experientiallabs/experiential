"""Tests for neutral OpenAI-compatible protocol errors."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.models.providers.errors import (
    ProviderParameterError,
    UnsupportedReasoningEffortError,
    normalized_provider_failure,
)
from exp.runtime.openai_protocol.errors import (
    THROTTLED_RETRY_AFTER_SECONDS,
    UNAVAILABLE_RETRY_AFTER_SECONDS,
    OpenAIProtocolError,
    public_failure_error,
)


def test_monthly_quota_failure_uses_openai_insufficient_quota_shape() -> None:
    """Hard gateway exhaustion returns the standard HTTP and envelope semantics."""
    error = public_failure_error(
        GatewayFailure(
            failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
            safe_message="monthly gateway allocation is exhausted",
        ),
        now=datetime(2026, 8, 19, 12, 0, tzinfo=UTC),
    )

    assert error.status_code == 429
    assert error.json_body() == {
        "error": {
            "message": (
                "monthly gateway allocation is exhausted. The allocation resets at "
                "2026-09-01T00:00:00Z; retry after that time or ask the gateway "
                "operator to raise the monthly budget."
            ),
            "type": "insufficient_quota",
            "param": None,
            "code": "insufficient_quota",
        }
    }


def test_quota_retry_after_counts_down_to_the_next_utc_month_boundary() -> None:
    """The advertised wait is the exact seconds until the UTC month rolls over."""
    now = datetime(2026, 12, 31, 23, 59, 0, tzinfo=UTC)
    error = public_failure_error(
        GatewayFailure(
            failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
            safe_message="monthly gateway allocation is exhausted",
        ),
        now=now,
    )

    assert error.retry_after_seconds == 60
    assert error.headers() == {"Retry-After": "60"}
    assert "2027-01-01T00:00:00Z" in error.detail.message


def test_throttled_failure_advertises_a_default_retry_after() -> None:
    """Provider throttling keeps its frozen code and adds a short bounded wait."""
    error = public_failure_error(
        GatewayFailure(
            failure_class=GatewayFailureClass.THROTTLED,
            safe_message="provider throttled the request",
        )
    )

    assert error.status_code == 429
    assert error.detail.code == "unavailable_route"
    assert error.retry_after_seconds == THROTTLED_RETRY_AFTER_SECONDS
    assert error.headers() == {"Retry-After": str(THROTTLED_RETRY_AFTER_SECONDS)}


def test_unavailable_failure_is_a_retryable_503() -> None:
    """A transient roll condition is a retryable 503, never a closed 5xx."""
    error = public_failure_error(
        GatewayFailure(
            failure_class=GatewayFailureClass.UNAVAILABLE,
            safe_message="the gateway is updating; retry the request",
        )
    )

    assert error.status_code == 503
    assert error.detail.code == "gateway_unavailable"
    assert error.detail.type == "api_error"
    assert error.retry_after_seconds == UNAVAILABLE_RETRY_AFTER_SECONDS
    assert error.headers() == {"Retry-After": str(UNAVAILABLE_RETRY_AFTER_SECONDS)}


def test_guardrail_failure_uses_content_filter_shape() -> None:
    """A guardrail block is a sanitized 400 with no request content."""
    error = public_failure_error(
        GatewayFailure(
            failure_class=GatewayFailureClass.GUARDRAIL,
            safe_message="The request was blocked by a gateway guardrail.",
        )
    )

    assert error.status_code == 400
    assert error.detail.code == "content_filter"
    assert error.detail.type == "invalid_request_error"
    assert error.retry_after_seconds is None
    assert "prompt" not in error.detail.message


def test_non_retryable_failures_carry_no_retry_after_header() -> None:
    """Only quota and throttling errors advertise a wait."""
    error = public_failure_error(
        GatewayFailure(
            failure_class=GatewayFailureClass.AUTHENTICATION,
            safe_message="the key is not valid",
        )
    )

    assert error.retry_after_seconds is None
    assert error.headers() == {}


def test_unsupported_reasoning_effort_preserves_field_specific_error() -> None:
    """Normalized gateway accounting retains the corrective public error contract."""
    failure = normalized_provider_failure(
        UnsupportedReasoningEffortError(
            effort="minimal",
            supported_efforts=("high",),
            param="reasoning.effort",
        )
    )

    error = public_failure_error(failure)

    assert failure.failure_class is GatewayFailureClass.INVALID_REQUEST
    assert error.status_code == 400
    assert error.detail.code == "unsupported_parameter"
    assert error.detail.param == "reasoning.effort"
    assert "Supported values: 'high'" in error.detail.message


def test_invalid_route_parameter_preserves_field_specific_error() -> None:
    """Model-specific range failures use the caller field and invalid code."""
    failure = normalized_provider_failure(
        ProviderParameterError(
            message="The value 65000 exceeds the route maximum.",
            param="max_completion_tokens",
            code="invalid_parameter",
        )
    )

    error = public_failure_error(failure)

    assert error.status_code == 400
    assert error.detail.code == "invalid_parameter"
    assert error.detail.param == "max_completion_tokens"


def test_provider_explanation_replaces_the_generic_client_error_advice() -> None:
    """A relayed provider sentence names the refusal the caller must fix."""
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.INVALID_REQUEST,
        safe_message=(
            "provider rejected the request; verify the request fields against "
            "the model alias capabilities"
        ),
        rejected_parameter="top_p",
        provider_detail="`top_p` is deprecated for this model.",
    )

    error = public_failure_error(failure)

    assert error.status_code == 400
    assert error.detail.param == "top_p"
    assert error.detail.message == (
        "provider rejected the request: `top_p` is deprecated for this model."
    )


def test_provider_explanation_is_ignored_outside_the_client_error_class() -> None:
    """Only a client error is caller-actionable, so only it relays a sentence."""
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
        safe_message="provider service failed; retry after a short delay",
        provider_detail="deployment gpt-internal-7 is unhealthy",
    )

    error = public_failure_error(failure)

    assert error.detail.message == "provider service failed; retry after a short delay"


@pytest.mark.parametrize("param", (None, "tools"))
def test_unsupported_capability_never_exposes_internal_identifiers(
    param: str | None,
) -> None:
    """Generic public mapping ignores internal capability names and messages."""
    error = public_failure_error(
        GatewayFailure(
            failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
            safe_message="internal tinker_gateway_execution capability failed",
            safe_details={"capability": "tinker_gateway_execution"},
        ),
        param=param,
    )

    assert error.status_code == 400
    assert error.detail.code == "unsupported_capability"
    assert error.detail.param == param
    assert "tinker_gateway_execution" not in error.detail.message
    assert "internal" not in error.detail.message
    if param is not None:
        assert param in error.detail.message


def test_retry_after_must_be_positive() -> None:
    """A non-positive advertised wait is a programming error, not a response."""
    with pytest.raises(ValueError, match="retry_after_seconds must be positive"):
        OpenAIProtocolError(
            status_code=429,
            code="insufficient_quota",
            message="quota exhausted",
            retry_after_seconds=0,
        )
