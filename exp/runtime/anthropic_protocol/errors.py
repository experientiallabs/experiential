"""Anthropic-shaped public error envelopes for the Messages surface.

Every gateway failure is first mapped to its stable OpenAI-shaped protocol
error by the shared boundary authority (``boundary_protocol_error`` and
``public_failure_error``); this module owns the one translation from that
error to the Anthropic error envelope
``{"type": "error", "error": {"type": ..., "message": ...}}``, so the two
data planes cannot answer the same failure differently on
``POST /v1/messages``. The Rust data plane mirrors this table byte for byte
(``anthropic_error_fixture`` proves parity).
"""

from __future__ import annotations

from typing import Literal

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

AnthropicErrorType = Literal[
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "request_too_large",
    "rate_limit_error",
    "api_error",
    "overloaded_error",
]


def anthropic_error_type(status_code: int, openai_type: str) -> AnthropicErrorType:
    """Select the Anthropic error type for one sanitized gateway failure.

    Branches on HTTP status first because several distinct gateway codes share
    a status, then falls back on the OpenAI envelope type: any remaining
    caller mistake stays ``invalid_request_error`` and everything else is an
    ``api_error``.

    Args:
        status_code: HTTP status of the OpenAI-shaped gateway error.
        openai_type: OpenAI envelope ``type`` of the same error.

    Returns:
        The Anthropic envelope ``error.type``.
    """
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 413:
        return "request_too_large"
    if status_code == 429:
        return "rate_limit_error"
    if status_code == 503:
        return "overloaded_error"
    if openai_type == "invalid_request_error":
        return "invalid_request_error"
    return "api_error"


def anthropic_error_body(error: OpenAIProtocolError) -> JsonObject:
    """Render one sanitized protocol error as the Anthropic error envelope.

    The OpenAI ``param`` pointer has no Anthropic field, so a present param is
    folded into the message text; the caller still receives the exact field
    path responsible for the failure.

    Args:
        error: Sanitized OpenAI-shaped protocol error.

    Returns:
        The Anthropic wire envelope ``{"type": "error", "error": {...}}``.
    """
    message = error.detail.message
    if error.detail.param:
        message = f"{message} (param: {error.detail.param})"
    return {
        "type": "error",
        "error": {
            "type": anthropic_error_type(error.status_code, error.detail.type),
            "message": message,
        },
    }
