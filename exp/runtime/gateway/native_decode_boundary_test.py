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


@pytest.mark.parametrize(
    ("message", "index", "role"),
    [
        ({"role": "tool", "tool_call_id": "call_1", "content": None}, 2, "tool"),
        ({"role": "tool", "tool_call_id": "call_1"}, 2, "tool"),
        ({"role": "user", "content": None}, 2, "user"),
        ({"role": "system"}, 2, "system"),
    ],
)
def test_decode_mixin_answers_a_contentless_chat_message_as_a_typed_400(
    message: dict[str, object], index: int, role: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A Chat message with null or omitted content is the caller's 400, never a logged 500.

    Go clients drop an empty tool result under ``omitempty``; the canonical
    message refused it in its own validator, which escaped decode as an
    unclassified pydantic ValidationError and answered 500 internal_error
    before admission (2026-10-24, 36 requests from one org in 90 seconds).
    """
    body = {
        "model": "synthetic",
        "messages": [
            {"role": "user", "content": "run it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run", "arguments": "{}"},
                    }
                ],
            },
            message,
        ],
    }
    with caplog.at_level("ERROR", logger="exp.runtime.gateway.boundary"):
        with pytest.raises(NativeBridgeError) as rejected:
            NativeDecodeMixin()._decode_body(json.dumps(body))
    error = json.loads(rejected.value.public_error_json)
    assert error["status_code"] == 400
    assert error["code"] == "invalid_parameter"
    assert error["param"] == f"messages.{index}.content"
    assert f"a {role} message needs content" in error["message"]
    assert "unclassified" not in caplog.text


def test_decode_mixin_keeps_an_empty_tool_result_valid() -> None:
    """An empty-string tool result is content, so it still decodes unchanged."""
    result = NativeDecodeMixin()._decode_body(
        json.dumps(
            {
                "model": "synthetic",
                "messages": [
                    {"role": "user", "content": "run it"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "run", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": ""},
                ],
            }
        )
    )
    assert result.request.messages[2].content == ""
