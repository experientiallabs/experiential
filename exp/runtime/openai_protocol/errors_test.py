"""Tests for neutral OpenAI-compatible protocol errors."""

from __future__ import annotations

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.openai_protocol.errors import public_failure_error


def test_monthly_quota_failure_uses_openai_insufficient_quota_shape() -> None:
    """Hard gateway exhaustion returns the standard HTTP and envelope semantics."""
    error = public_failure_error(
        GatewayFailure(
            failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
            safe_message="monthly gateway allocation is exhausted",
        )
    )

    assert error.status_code == 429
    assert error.json_body() == {
        "error": {
            "message": "monthly gateway allocation is exhausted",
            "type": "insufficient_quota",
            "param": None,
            "code": "insufficient_quota",
        }
    }
