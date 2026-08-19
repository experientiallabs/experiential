"""Tests for sanitized HTTP boundary error responses of the gateway service."""

from __future__ import annotations

import json

from fastapi import Response

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.execution import GatewayExecutionError
from exp.runtime.gateway.service import GatewayDrainingError, _exception_response
from exp.runtime.gateway.sqlite.store import AliasNotGrantedError, InvalidVirtualKeyError


def _decoded_error(response: Response) -> dict[str, str | None]:
    """Return the OpenAI error object carried by one boundary response.

    Args:
        response: Boundary JSON response under test.

    Returns:
        Decoded ``error`` object.
    """
    body = json.loads(bytes(response.body))
    return body["error"]


def test_ungranted_alias_points_the_caller_at_the_models_route() -> None:
    """A denied alias states the failure and the exact discovery request."""
    response = _exception_response(AliasNotGrantedError("alias is not granted"))

    error = _decoded_error(response)
    assert response.status_code == 403
    assert error["code"] == "model_not_granted"
    assert error["message"] is not None
    assert "GET /v1/models lists the model aliases available to this key." in error["message"]
    assert "Retry-After" not in response.headers


def test_draining_gateway_advertises_a_retry_after_wait() -> None:
    """Draining is a temporary condition and says exactly how to proceed."""
    response = _exception_response(GatewayDrainingError("gateway is draining"))

    error = _decoded_error(response)
    assert response.status_code == 503
    assert error["code"] == "gateway_draining"
    assert response.headers["Retry-After"] == "10"
    assert error["message"] is not None
    assert "Retry after the delay in the Retry-After header." in error["message"]


def test_throttled_execution_failure_carries_a_retry_after_header() -> None:
    """A provider 429 surfaces its frozen code plus a bounded advertised wait."""
    response = _exception_response(
        GatewayExecutionError(
            GatewayFailure(
                failure_class=GatewayFailureClass.THROTTLED,
                safe_message=(
                    "provider throttled the request; retry after the delay in the "
                    "Retry-After header"
                ),
            )
        )
    )

    error = _decoded_error(response)
    assert response.status_code == 429
    assert error["code"] == "unavailable_route"
    assert response.headers["Retry-After"] == "5"


def test_quota_exhaustion_reports_the_utc_reset_boundary() -> None:
    """Monthly exhaustion includes the reset time and a Retry-After wait."""
    response = _exception_response(
        GatewayExecutionError(
            GatewayFailure(
                failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
                safe_message="monthly gateway allocation is exhausted",
            )
        )
    )

    error = _decoded_error(response)
    assert response.status_code == 429
    assert error["code"] == "insufficient_quota"
    assert error["message"] is not None
    assert "The allocation resets at " in error["message"]
    assert int(response.headers["Retry-After"]) >= 1


def test_invalid_key_states_the_recovery_path_without_key_material() -> None:
    """An invalid key failure explains issuance without echoing any credential."""
    response = _exception_response(InvalidVirtualKeyError("wmo_vk_canary rejected"))

    error = _decoded_error(response)
    assert response.status_code == 401
    assert error["code"] == "invalid_key"
    assert error["message"] is not None
    assert "issue a new virtual key" in error["message"]
    assert "wmo_vk_canary" not in json.dumps(_decoded_error(response))
