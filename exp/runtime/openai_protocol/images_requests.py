"""Decode the OpenAI Images generation body into the canonical images surface.

Lives beside :mod:`exp.runtime.openai_protocol.requests` (which owns the
line-budgeted chat, Responses, and embeddings decoders) and reuses its
validation helpers, so every public surface decodes through one manifest,
one official-SDK shape check, and one closed wire model.
"""

from __future__ import annotations

from typing import Literal

from openai.types.image_generate_params import ImageGenerateParamsNonStreaming
from pydantic import Field, TypeAdapter, ValidationError

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.gateway.images_contracts import ImagesRequest
from exp.runtime.openai_protocol.manifest import IMAGES_MANIFEST
from exp.runtime.openai_protocol.requests import (
    _validate_manifest,
    _validate_official,
    _validate_wire,
)
from exp.runtime.openai_protocol.validation_errors import validation_protocol_error
from exp.runtime.openai_protocol.wire_models import _WireModel


class _ImagesRequest(_WireModel):
    """Closed gateway image-generation request profile (OpenAI Images API)."""

    model: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=32_000)
    n: int | None = Field(default=None, ge=1, le=10)
    size: (
        Literal[
            "auto",
            "256x256",
            "512x512",
            "1024x1024",
            "1536x1024",
            "1024x1536",
            "1792x1024",
            "1024x1792",
        ]
        | None
    ) = None
    quality: Literal["standard", "hd", "low", "medium", "high", "auto"] | None = None
    background: Literal["transparent", "opaque", "auto"] | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None
    output_compression: int | None = Field(default=None, ge=0, le=100)
    moderation: Literal["low", "auto"] | None = None
    response_format: Literal["url", "b64_json"] | None = None
    style: Literal["vivid", "natural"] | None = None
    user: str | None = Field(default=None, max_length=1024)


_IMAGES_OFFICIAL: TypeAdapter[object] = TypeAdapter[object](ImageGenerateParamsNonStreaming)


class DecodedImagesRequest(ContractModel):
    """Public alias plus its canonical image-generation request."""

    alias: str = Field(min_length=1, max_length=256)
    request: ImagesRequest


def decode_images(payload: JsonObject) -> DecodedImagesRequest:
    """Decode one Images generation body into the canonical images surface.

    The surface is buffered and never streamed, so ``stream`` / ``partial_images``
    are refused by the manifest rather than silently answered whole.

    Args:
        payload: Parsed JSON request body.

    Returns:
        Public alias and canonical image-generation request.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
    """
    _validate_manifest(payload, IMAGES_MANIFEST)
    _validate_official(_IMAGES_OFFICIAL, payload)
    request = _validate_wire(_ImagesRequest, payload)
    try:
        canonical = ImagesRequest(
            prompt=request.prompt,
            n=1 if request.n is None else request.n,
            size=request.size,
            quality=request.quality,
            background=request.background,
            output_format=request.output_format,
            output_compression=request.output_compression,
            moderation=request.moderation,
            response_format=request.response_format,
            style=request.style,
            user=request.user,
        )
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc
    return DecodedImagesRequest(alias=request.model, request=canonical)
