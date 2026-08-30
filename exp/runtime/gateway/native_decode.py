"""Shared native-bridge request-body decoding."""

from __future__ import annotations

import json

from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import (
    DecodedGatewayRequest,
    decode_chat,
    decode_responses,
)


class NativeDecodeError(Exception):
    """A request body failed shared protocol validation."""

    def __init__(self, error: OpenAIProtocolError) -> None:
        """Retain the public protocol error for the native boundary.

        Args:
            error: Sanitized OpenAI-shaped decode failure.
        """
        super().__init__(error.detail.message)
        self.error = error


def decode_native_body(
    body: str,
    *,
    surface: str = "chat",
    idempotency_key: str | None = None,
    client_request_id: str | None = None,
    anthropic_beta: str | None = None,
) -> DecodedGatewayRequest:
    """Decode one raw request body with the shared surface decoder.

    Args:
        body: Raw request body text.
        surface: Public surface, ``chat``, ``responses``, or ``messages``.
        idempotency_key: Optional raw ``Idempotency-Key`` header value.
            Ignored on the Anthropic Messages surface, which defines no
            idempotency header.
        client_request_id: Optional raw ``X-Client-Request-Id`` header value.
            Ignored on the Anthropic Messages surface.
        anthropic_beta: Optional raw caller ``anthropic-beta`` header value.
            Meaningful only on the Messages surface, where allowlisted
            tokens are retained for Anthropic dispatch.

    Returns:
        The public alias and canonical request.

    Raises:
        NativeDecodeError: The body is not JSON, not an object, or fails
            shared protocol validation.
    """
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise NativeDecodeError(
            OpenAIProtocolError(
                status_code=400,
                code="invalid_json",
                message=("Request body must contain valid JSON. Re-encode the payload and resend."),
            )
        ) from exc
    if not isinstance(payload, dict):
        raise NativeDecodeError(
            OpenAIProtocolError(
                status_code=400,
                code="invalid_request",
                message="Request body must be a JSON object. Re-encode the payload and resend.",
            )
        )
    try:
        if surface == "messages":
            return decode_messages(payload, anthropic_beta=anthropic_beta)
        decoder = decode_responses if surface == "responses" else decode_chat
        return decoder(
            payload,
            idempotency_key=idempotency_key,
            client_request_id=client_request_id,
        )
    except OpenAIProtocolError as exc:
        raise NativeDecodeError(exc) from exc
