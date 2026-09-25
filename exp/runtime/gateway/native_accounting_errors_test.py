"""Native accounting errors expose only the public protocol boundary fields."""

import json

from exp.runtime.gateway.native_accounting_errors import NativeBridgeError, internal_protocol_error


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
