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


def test_decode_mixin_logs_and_sanitizes_an_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A decoder bug answers the bridge's 500, now with its type and frame in the log."""

    def broken(*_args: object, **_kwargs: object) -> None:
        message = "decoder bug"
        raise RuntimeError(message)

    monkeypatch.setattr("exp.runtime.gateway.native_decode_boundary.decode_native_body", broken)
    with caplog.at_level("ERROR", logger="exp.runtime.gateway.boundary"):
        with pytest.raises(NativeBridgeError) as rejected:
            NativeDecodeMixin()._decode_body("{}")
    error = json.loads(rejected.value.public_error_json)
    assert error["status_code"] == 500
    assert error["code"] == "internal_error"
    assert "decoder bug" not in rejected.value.public_error_json
    assert "decoder bug" not in caplog.text
    assert "unclassified RuntimeError raised at native_decode_boundary_test.py:" in caplog.text
