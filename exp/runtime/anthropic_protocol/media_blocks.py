"""Anthropic Messages image and document blocks and their canonical decoding.

An ``image`` or PDF ``document`` block names its bytes through a ``source``:
inline ``base64``, a remote ``url``, or a ``file`` the caller already
uploaded to the Anthropic Files API. A ``url`` source may also carry an
``s3://`` or ``gs://`` object URI, which is a Bedrock or Vertex handle rather
than a fetchable URL; each form becomes the matching carrier of the canonical
content part so a route that declares the capability forwards it and every
other route refuses before dispatch.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from exp.common.models.content import (
    DOCUMENT_MEDIA_TYPES,
    IMAGE_MEDIA_TYPES,
    MAXIMUM_DOCUMENT_BASE64_BYTES,
    MAXIMUM_DOCUMENT_NAME_CHARACTERS,
    MAXIMUM_IMAGE_BASE64_BYTES,
    DocumentContentPart,
    ImageContentPart,
    MediaHandle,
    image_part_from_url,
    media_handle_from_uri,
)
from exp.runtime.openai_protocol.errors import invalid_field

_MAXIMUM_FILE_ID_CHARACTERS = 512
"""Longest Anthropic Files handle accepted on the wire."""


class AnthropicWireModel(BaseModel):
    """Strict private Anthropic wire model rejecting unknown nested fields."""

    model_config = ConfigDict(extra="forbid")


class CacheControl(AnthropicWireModel):
    """Anthropic prompt-caching annotation, validated and then dropped."""

    type: Literal["ephemeral"]
    ttl: Literal["5m", "1h"] | None = None


class ImageSource(AnthropicWireModel):
    """Where one image block's bytes come from: base64, a URL, or a file id."""

    type: Literal["base64", "url", "file"]
    media_type: str | None = Field(default=None, max_length=64)
    data: str | None = Field(default=None, max_length=MAXIMUM_IMAGE_BASE64_BYTES)
    url: str | None = Field(default=None, max_length=8_192)
    file_id: str | None = Field(default=None, max_length=_MAXIMUM_FILE_ID_CHARACTERS)


class ImageBlock(AnthropicWireModel):
    """One caller image content block."""

    type: Literal["image"]
    source: ImageSource
    cache_control: CacheControl | None = None


class DocumentSource(AnthropicWireModel):
    """Where one document block's bytes come from: base64, a URL, or a file id.

    ``text`` and ``content`` sources carry non-PDF documents no other wire
    accepts, so only the carriers every declared route can serve are
    accepted.
    """

    type: Literal["base64", "url", "file"]
    media_type: str | None = Field(default=None, max_length=64)
    data: str | None = Field(default=None, max_length=MAXIMUM_DOCUMENT_BASE64_BYTES)
    url: str | None = Field(default=None, max_length=8_192)
    file_id: str | None = Field(default=None, max_length=_MAXIMUM_FILE_ID_CHARACTERS)


class DocumentCitations(AnthropicWireModel):
    """Per-document citations toggle; only the disabled form is servable."""

    enabled: bool


class DocumentBlock(AnthropicWireModel):
    """One caller PDF document content block."""

    type: Literal["document"]
    source: DocumentSource
    title: str | None = Field(default=None, max_length=MAXIMUM_DOCUMENT_NAME_CHARACTERS)
    citations: DocumentCitations | None = None
    cache_control: CacheControl | None = None


def _anthropic_handle(file_id: str | None) -> MediaHandle:
    """Wrap one Anthropic Files id as a provider-scoped handle.

    Args:
        file_id: Caller ``source.file_id`` value.

    Returns:
        The Anthropic-scoped handle.

    Raises:
        ValueError: The id is missing or does not have the ``file_...`` shape.
    """
    if file_id is None:
        raise ValueError("a file source needs its file_id")
    return MediaHandle(provider="anthropic", reference=file_id)


def image_part_from_block(block: ImageBlock, param: str) -> ImageContentPart:
    """Convert one Anthropic image block into the canonical image part.

    Args:
        block: Validated caller image block.
        param: Public parameter path used to report an invalid image.

    Returns:
        The canonical image part carrying the caller's bytes, URL, or handle.

    Raises:
        OpenAIProtocolError: The source is not a supported image.
    """
    source = block.source
    marker = (
        block.cache_control.model_dump(mode="json", exclude_none=True)
        if block.cache_control is not None
        else None
    )
    try:
        if source.type == "file":
            return ImageContentPart(handle=_anthropic_handle(source.file_id), cache_control=marker)
        if source.type == "url":
            part = image_part_from_url(source.url or "")
            return part.model_copy(update={"cache_control": marker})
        return ImageContentPart(
            media_type=IMAGE_MEDIA_TYPES[source.media_type or ""],
            data=source.data,
            cache_control=marker,
        )
    except (KeyError, ValueError) as exc:
        raise invalid_field(
            f"{param}.source",
            f"'{param}.source' must carry an http(s) URL or base64 data for a PNG, "
            "JPEG, GIF, or WebP image, an Anthropic Files id (file_...), or an "
            "s3:// or gs:// URI of an uploaded image whose suffix states its format.",
        ) from exc


def document_part_from_block(block: DocumentBlock, param: str) -> DocumentContentPart:
    """Convert one Anthropic document block into the canonical document part.

    Args:
        block: Validated caller document block.
        param: Public parameter path used to report an invalid document.

    Returns:
        The canonical document part carrying the caller's bytes, URL, or
        handle.

    Raises:
        OpenAIProtocolError: The source is not a PDF this gateway forwards,
            or the block enables citations.
    """
    if block.citations is not None and block.citations.enabled:
        raise invalid_field(
            f"{param}.citations",
            "document citations are not supported over this gateway; "
            "send citations.enabled as false or omit the field.",
        )
    source = block.source
    marker = (
        block.cache_control.model_dump(mode="json", exclude_none=True)
        if block.cache_control is not None
        else None
    )
    try:
        if source.type == "file":
            return DocumentContentPart(
                handle=_anthropic_handle(source.file_id), name=block.title, cache_control=marker
            )
        if source.type == "url":
            handle = media_handle_from_uri(source.url or "")
            if handle is not None:
                return DocumentContentPart(handle=handle, name=block.title, cache_control=marker)
            return DocumentContentPart(url=source.url, name=block.title, cache_control=marker)
        return DocumentContentPart(
            media_type=DOCUMENT_MEDIA_TYPES[source.media_type or ""],
            data=source.data,
            name=block.title,
            cache_control=marker,
        )
    except (KeyError, ValueError) as exc:
        raise invalid_field(
            f"{param}.source",
            f"'{param}.source' must carry an http(s) URL or base64 data for a PDF "
            "(media_type application/pdf), an Anthropic Files id (file_...), or an "
            "s3:// or gs:// URI of an uploaded PDF.",
        ) from exc
