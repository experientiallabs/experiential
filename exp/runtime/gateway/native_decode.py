"""Shared native-bridge request-body decoding."""

from __future__ import annotations

import json

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.requests import decode_messages, decode_messages_count_tokens
from exp.runtime.gateway.lone_surrogates import (
    LoneSurrogateRefusal,
    may_hold_lone_surrogate,
    refuse_lone_surrogates,
    replace_lone_surrogates,
)
from exp.runtime.google_protocol.requests import (
    decode_generate_content,
    decode_generate_content_count_tokens,
)
from exp.runtime.models.providers.json_prefix import incomplete_json_object_prefix
from exp.runtime.openai_protocol.audio_requests import (
    DecodedSpeechRequest,
    DecodedTranscriptionRequest,
    decode_speech,
    decode_transcription,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, invalid_field
from exp.runtime.openai_protocol.image_edits_requests import decode_image_edits
from exp.runtime.openai_protocol.images_requests import DecodedImagesRequest, decode_images
from exp.runtime.openai_protocol.requests import (
    DecodedEmbeddingsRequest,
    DecodedGatewayRequest,
    decode_chat,
    decode_embeddings,
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
    model: str | None = None,
) -> DecodedGatewayRequest:
    """Decode one raw request body with the shared surface decoder.

    Args:
        body: Raw request body text.
        surface: Public surface, ``chat``, ``responses``, ``messages``,
            ``generate_content`` or ``stream_generate_content``.
        idempotency_key: Optional raw ``Idempotency-Key`` header value.
            Ignored on the Anthropic Messages surface, which defines no
            idempotency header.
        client_request_id: Optional raw ``X-Client-Request-Id`` header value.
            Ignored on the Anthropic Messages surface.
        anthropic_beta: Optional raw caller ``anthropic-beta`` header value.
            Meaningful only on the Messages surface, where allowlisted
            tokens are retained for Anthropic dispatch.
        model: The alias named in the request path, on the Gemini surfaces
            (their body carries no model field).

    Returns:
        The public alias and canonical request.

    Raises:
        NativeDecodeError: The body is not JSON, not an object, or fails
            shared protocol validation.
    """
    payload, repairs = _load_repaired_body(body)
    try:
        if surface == "messages":
            decoded = decode_messages(payload, anthropic_beta=anthropic_beta)
        elif surface in {"generate_content", "stream_generate_content"}:
            decoded = decode_generate_content(
                payload, model=model or "", stream=surface == "stream_generate_content"
            )
        else:
            decoder = decode_responses if surface == "responses" else decode_chat
            decoded = decoder(
                payload,
                idempotency_key=idempotency_key,
                client_request_id=client_request_id,
            )
    except OpenAIProtocolError as exc:
        raise NativeDecodeError(exc) from exc
    return _disclosed(decoded, repairs)


def decode_native_count_tokens_body(
    body: str,
    *,
    anthropic_beta: str | None = None,
    surface: str = "messages",
    model: str | None = None,
) -> DecodedGatewayRequest:
    """Decode one raw count body for prompt counting.

    Anthropic's count body (``/v1/messages/count_tokens``) carries no
    ``max_tokens``, and Google's (``:countTokens``) only ``contents`` or a
    wrapped ``generateContentRequest``, so each decodes through its
    count-specific entrypoint rather than the generation one.

    Args:
        body: Raw request body text.
        anthropic_beta: Optional raw caller ``anthropic-beta`` header value.
        surface: ``messages`` (default) or ``generate_content``.
        model: The alias named in the request path on the Gemini surface.

    Returns:
        The public alias and the canonical request to count.

    Raises:
        NativeDecodeError: The body is not JSON, not an object, or fails
            shared protocol validation.
    """
    payload, repairs = _load_repaired_body(body)
    try:
        if surface == "generate_content":
            decoded = decode_generate_content_count_tokens(payload, model=model or "")
        else:
            decoded = decode_messages_count_tokens(payload, anthropic_beta=anthropic_beta)
    except OpenAIProtocolError as exc:
        raise NativeDecodeError(exc) from exc
    return _disclosed(decoded, repairs)


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


def _load_repaired_body(body: str) -> tuple[JsonObject, tuple[str, ...]]:
    """Parse a text-generation body, replacing lone surrogates in its content.

    The one place every Chat, Responses, and Messages body (and a Messages
    count) is repaired, before any decoder reads a string; see
    :mod:`exp.runtime.gateway.lone_surrogates`.

    Returns:
        The repaired object and its ``->replaced(lone_surrogate)`` disclosures.

    Raises:
        NativeDecodeError: The body is not a JSON object, or a name holds a
            lone surrogate.
    """
    payload = _parse_object_body(body)
    if not may_hold_lone_surrogate(body):
        return payload, ()
    try:
        return payload, replace_lone_surrogates(
            payload, object_prefix=incomplete_json_object_prefix
        )
    except LoneSurrogateRefusal as exc:
        raise NativeDecodeError(invalid_field(exc.param, exc.message)) from exc


def _disclosed(decoded: DecodedGatewayRequest, repairs: tuple[str, ...]) -> DecodedGatewayRequest:
    """Append the decode-time repairs to the request's disclosed parameters."""
    if not repairs:
        return decoded
    request = decoded.request
    disclosures = tuple(dict.fromkeys((*request.ignored_parameters, *repairs)))
    return decoded.model_copy(
        update={"request": request.model_copy(update={"ignored_parameters": disclosures})}
    )


def _load_object_body(body: str) -> JsonObject:
    """Parse a body for a surface with no disclosure channel; refuse lone surrogates.

    Embeddings, images, and audio cannot announce a rewrite, so a lone
    surrogate there is a field-specific 400 rather than a silent U+FFFD.
    """
    payload = _parse_object_body(body)
    if may_hold_lone_surrogate(body):
        try:
            refuse_lone_surrogates(payload)
        except LoneSurrogateRefusal as exc:
            raise NativeDecodeError(invalid_field(exc.param, exc.message)) from exc
    return payload


def _parse_object_body(body: str) -> JsonObject:
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


def decode_native_image_edits_upload(upload: str) -> DecodedImagesRequest:
    """Decode the data plane's parsed ``/images/edits`` multipart upload.

    The data plane parses the multipart body itself; ``upload`` is its JSON
    object of the text ``fields`` plus the base64 image ``files``.

    Args:
        upload: JSON text of the parsed upload.

    Returns:
        The public alias and canonical image-edit request.

    Raises:
        NativeDecodeError: The upload fails shared protocol validation.
    """
    payload = _load_object_body(upload)
    try:
        return decode_image_edits(payload)
    except OpenAIProtocolError as exc:
        raise NativeDecodeError(exc) from exc


def decode_native_speech_body(body: str) -> DecodedSpeechRequest:
    """Decode one raw ``/audio/speech`` body with the shared speech decoder.

    Args:
        body: Raw request body text.

    Returns:
        The public alias and canonical speech request.

    Raises:
        NativeDecodeError: The body is not JSON, not an object, or fails shared
            protocol validation.
    """
    payload = _load_object_body(body)
    try:
        return decode_speech(payload)
    except OpenAIProtocolError as exc:
        raise NativeDecodeError(exc) from exc


def decode_native_transcription_upload(upload: str) -> DecodedTranscriptionRequest:
    """Decode the data plane's parsed ``/audio/transcriptions`` upload.

    The data plane parses the multipart (or JSON ``input_audio``) body itself
    and keeps the audio bytes; ``upload`` is its JSON object of the text fields
    plus the measured ``file`` facts.

    Args:
        upload: JSON text of the parsed upload.

    Returns:
        The public alias and canonical transcription request.

    Raises:
        NativeDecodeError: The upload fails shared protocol validation.
    """
    payload = _load_object_body(upload)
    try:
        return decode_transcription(payload)
    except OpenAIProtocolError as exc:
        raise NativeDecodeError(exc) from exc
