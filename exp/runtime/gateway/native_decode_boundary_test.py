"""Preserve typed decoder behavior through the native error boundary."""

import json

import pytest

from exp.runtime.gateway.native_accounting_errors import NativeBridgeError
from exp.runtime.gateway.native_decode_boundary import NativeDecodeMixin


def test_decode_mixin_preserves_public_request() -> None:
    """Successful Chat decoding keeps model and visible text unchanged."""
    result = NativeDecodeMixin()._decode_body(
        json.dumps({"model": "synthetic", "messages": [{"role": "user", "content": "ok"}]})
    )
    assert result.alias == "synthetic"
    assert result.request.messages[0].content == "ok"


def test_decode_mixin_wraps_invalid_json() -> None:
    """The native boundary carries the existing sanitized protocol error."""
    with pytest.raises(NativeBridgeError) as rejected:
        NativeDecodeMixin()._decode_body("not json")
    error = json.loads(rejected.value.public_error_json)
    assert error["status_code"] == 400
    assert error["code"] == "invalid_json"
