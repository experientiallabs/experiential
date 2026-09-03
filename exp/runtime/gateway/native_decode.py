"""Shared native-bridge request-body decoding."""

from __future__ import annotations

import json

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import (
    DecodedEmbeddingsRequest,
    DecodedGatewayRequest,
    DecodedImagesRequest,
    decode_chat,
    decode_embeddings,
    decode_images,
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
    payload = _load_object_body(body)
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


def decode_native_embeddings_body(body: str) -> DecodedEmbeddingsRequest:
    """Decode one raw ``/embeddings`` body with the shared embeddings decoder.

    The embeddings surface is message-less and non-streaming, so it decodes
    through its own entrypoint rather than the chat/responses/messages
    dispatch above. Keyed replay is a future add, so no idempotency header is
    threaded here.

    Args:
        body: Raw request body text.

    Returns:
        The public alias and canonical embeddings request.

    Raises:
        NativeDecodeError: The body is not JSON, not an object, or fails shared
            protocol validation.
    """
    payload = _load_object_body(body)
    try:
        return decode_embeddings(payload)
    except OpenAIProtocolError as exc:
        raise NativeDecodeError(exc) from exc


def _load_object_body(body: str) -> JsonObject:
    """Parse one raw request body into a JSON object or raise the public error."""
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
    return payload


def decode_native_images_body(body: str) -> DecodedImagesRequest:
    """Decode one raw ``/images/generations`` body with the shared images decoder.

    Args:
        body: Raw request body text.

    Returns:
        The public alias and canonical image-generation request.

    Raises:
        NativeDecodeError: The body is not JSON, not an object, or fails shared
            protocol validation.
    """
    payload = _load_object_body(body)
    try:
        return decode_images(payload)
    except OpenAIProtocolError as exc:
        raise NativeDecodeError(exc) from exc
