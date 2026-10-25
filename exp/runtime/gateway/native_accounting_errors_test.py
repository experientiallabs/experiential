"""Native accounting errors expose only the public protocol boundary fields."""

import json

from exp.runtime.gateway.native_accounting_errors import (
    NativeBridgeError,
    accounting_recovering_error,
    authority_error,
    internal_protocol_error,
)


def test_internal_error_encoding_is_the_exact_public_shape() -> None:
    """The production boundary exception preserves only public status and fields."""
    assert json.loads(NativeBridgeError(internal_protocol_error()).public_error_json) == {
        "status_code": 500,
        "code": "internal_error",
        "message": "The gateway request failed.",
        "error_type": "api_error",
        "param": None,
        "retry_after_seconds": None,
    }


def test_existing_accounting_refusal_retains_public_status_and_identity() -> None:
    """Moving tenant recovery inside authority must not turn its 503 into a 500."""
    refusal = NativeBridgeError(accounting_recovering_error())
    assert authority_error(refusal) is refusal
    assert json.loads(refusal.public_error_json)["status_code"] == 503
