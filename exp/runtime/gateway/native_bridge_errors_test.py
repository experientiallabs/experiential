"""Native public error encoding excludes internal exception and credential material."""

import json

from exp.runtime.gateway.native_bridge_errors import encoded_public_error, internal_protocol_error


def test_internal_error_encoding_is_the_exact_public_shape() -> None:
    """The helper extraction preserves native-boundary status and fields."""
    assert json.loads(encoded_public_error(internal_protocol_error())) == {
        "status_code": 500,
        "code": "internal_error",
        "message": "The gateway request failed.",
        "error_type": "api_error",
        "param": None,
        "retry_after_seconds": None,
    }
