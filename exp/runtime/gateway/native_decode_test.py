"""Tests for shared native request-body decoding."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.native_decode import NativeDecodeError, decode_native_body


def test_decode_native_body_accepts_chat_and_responses() -> None:
    """Both public surfaces use the shared protocol decoders."""
    chat = decode_native_body(
        '{"model":"public-model","messages":[{"role":"user","content":"hi"}]}'
    )
    responses = decode_native_body(
        '{"model":"public-model","input":"hi"}',
        surface="responses",
    )

    assert chat.alias == "public-model"
    assert responses.request.messages[0].content == "hi"


def test_invalid_json_is_a_native_decode_error() -> None:
    """Malformed bodies keep the shared public invalid-JSON shape."""
    with pytest.raises(NativeDecodeError) as raised:
        decode_native_body("{not json")

    assert raised.value.error.status_code == 400
    assert raised.value.error.detail.code == "invalid_json"


def test_messages_surface_threads_the_caller_beta_header() -> None:
    """The Messages decode receives the caller anthropic-beta header so
    allowlisted tokens (the 1M context window) survive to dispatch."""
    decoded = decode_native_body(
        '{"model":"public-model","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}',
        surface="messages",
        anthropic_beta="claude-code-20250219,context-1m-2025-08-07",
    )
    assert decoded.request.provider_beta_tokens == ("context-1m-2025-08-07",)
    assert decoded.request.ignored_parameters == ("anthropic-beta.claude-code-20250219",)
