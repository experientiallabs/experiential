"""Tests for the Anthropic error-envelope translation."""

from __future__ import annotations

import pytest

from exp.runtime.anthropic_protocol.errors import anthropic_error_body, anthropic_error_type
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


@pytest.mark.parametrize(
    ("status_code", "openai_type", "expected"),
    [
        (401, "authentication_error", "authentication_error"),
        (403, "permission_error", "permission_error"),
        (404, "invalid_request_error", "not_found_error"),
        (413, "invalid_request_error", "request_too_large"),
        (429, "insufficient_quota", "rate_limit_error"),
        (429, "api_error", "rate_limit_error"),
        (503, "api_error", "overloaded_error"),
        (400, "invalid_request_error", "invalid_request_error"),
        (409, "invalid_request_error", "invalid_request_error"),
        (499, "api_error", "api_error"),
        (500, "api_error", "api_error"),
        (502, "api_error", "api_error"),
        (504, "api_error", "api_error"),
    ],
)
def test_status_decides_the_anthropic_type_before_the_openai_type(
    status_code: int, openai_type: str, expected: str
) -> None:
    """The translation branches on status first, then the envelope type."""
    assert anthropic_error_type(status_code, openai_type) == expected


def test_error_body_folds_the_param_pointer_into_the_message() -> None:
    """The OpenAI param pointer has no Anthropic field, so it joins the text."""
    error = OpenAIProtocolError(
        status_code=400,
        code="unsupported_parameter",
        message="The parameter 'top_k' is not supported.",
        param="top_k",
    )
    assert anthropic_error_body(error) == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "The parameter 'top_k' is not supported. (param: top_k)",
        },
    }


def test_error_body_without_param_keeps_the_message_verbatim() -> None:
    """A param-free error translates its message unchanged."""
    error = OpenAIProtocolError(
        status_code=401,
        code="invalid_key",
        message="The gateway key is invalid.",
        error_type="authentication_error",
    )
    assert anthropic_error_body(error) == {
        "type": "error",
        "error": {"type": "authentication_error", "message": "The gateway key is invalid."},
    }
