"""Anthropic Messages image and document blocks and their canonical decoding.

An ``image`` or PDF ``document`` block names its bytes through a ``source``:
inline ``base64``, a remote ``url``, or a ``file`` the caller already
uploaded to the Anthropic Files API. A ``url`` source may also carry an
``s3://`` or ``gs://`` object URI, which is a Bedrock or Vertex handle rather
than a fetchable URL; each form becomes the matching carrier of the canonical
content part so a route that declares the capability forwards it and every
other route refuses before dispatch. A ``tool_result`` carries the same image
and document blocks (tool screenshots and tool-returned files), decoded here
into the tool message's ordered parts, plus ``tool_reference`` blocks (a
client-side tool search naming the deferred tools it loaded), decoded into the
tool message's ordered reference names.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import (
    DOCUMENT_MEDIA_TYPES,
    IMAGE_MEDIA_TYPES,
    MAXIMUM_DOCUMENT_BASE64_BYTES,
    MAXIMUM_DOCUMENT_NAME_CHARACTERS,
    MAXIMUM_IMAGE_BASE64_BYTES,
    DocumentContentPart,
    ImageContentPart,
    MediaHandle,
    MessageContentPart,
    TextContentPart,
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


class MediaSource(AnthropicWireModel):
    """Where one media block's bytes come from: base64, a URL, or a file id.

    The ``type`` selects the carrier, and only that carrier's fields may be
    present, so a block never names two sources at once.
    """

    type: Literal["base64", "url", "file"]
    media_type: str | None = Field(default=None, max_length=64)
    data: str | None = None
    url: str | None = Field(default=None, max_length=8_192)
    file_id: str | None = Field(default=None, max_length=_MAXIMUM_FILE_ID_CHARACTERS)

    @model_validator(mode="after")
    def _require_matching_carrier(self) -> MediaSource:
        """Reject carrier fields that belong to a different source type."""
        expected = {
            "base64": ("data", "media_type"),
            "text": ("data", "media_type"),
            "url": ("url",),
            "file": ("file_id",),
        }[self.type]
        present = {
            name
            for name, value in (
                ("data", self.data),
                ("media_type", self.media_type),
                ("url", self.url),
                ("file_id", self.file_id),
            )
            if value is not None
        }
        stray = sorted(present - set(expected))
        if stray:
            raise ValueError(
                f"a {self.type} source accepts only {', '.join(expected)}; "
                f"remove {', '.join(stray)}"
            )
        return self


class ImageSource(MediaSource):
    """Where one image block's bytes come from: base64, a URL, or a file id."""

    data: str | None = Field(default=None, max_length=MAXIMUM_IMAGE_BASE64_BYTES)


class ImageBlock(AnthropicWireModel):
    """One caller image content block."""

    type: Literal["image"]
    source: ImageSource
    cache_control: CacheControl | None = None


class DocumentSource(MediaSource):
    """Where one document block's bytes come from: base64, text, a URL, or a file id.

    A ``text`` source (Anthropic's plain-text document, ``media_type``
    ``text/plain``) is decoded as the caller's text rather than a document,
    so every route serves it. ``content`` sources stay refused.
    """

    type: Literal["base64", "text", "url", "file"]
    data: str | None = Field(default=None, max_length=MAXIMUM_DOCUMENT_BASE64_BYTES)

    @model_validator(mode="after")
    def _require_plain_text(self) -> DocumentSource:
        """A text source carries its data as ``text/plain``, as Anthropic defines it."""
        if self.type == "text" and (self.data is None or self.media_type != "text/plain"):
            raise ValueError("a text source needs data and media_type 'text/plain'")
        return self


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


def text_document_text(block: DocumentBlock) -> str | None:
    """Return a plain-text document's text, or None for any other source.

    Anthropic renders a text document to the model as its title and body; the
    same text reaches every other wire as an ordinary text part, so a title
    leads the data on its own line.

    Args:
        block: Validated caller document block.

    Returns:
        The document's text with its title, or None when the source is not text.
    """
    if block.source.type != "text" or block.source.data is None:
        return None
    if block.title:
        return f"{block.title}\n\n{block.source.data}"
    return block.source.data


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


class TextBlockLike(Protocol):
    """The fields a tool-result text sub-block contributes to the tool message."""

    @property
    def text(self) -> str:
        """The block's text."""
        ...

    @property
    def cache_control(self) -> CacheControl | None:
        """The block's caller cache marker."""
        ...


class ToolReferenceBlock(AnthropicWireModel):
    """One ``tool_reference`` sub-block inside a ``tool_result``.

    A client-side tool search (Claude Code's ToolSearch) answers with one
    reference per deferred tool it loads; the provider expands each into the
    named tool's full definition. Validity of the name (a tool the request
    declares) is the provider's own rule on Anthropic rungs.

    Attributes:
        type: Always ``tool_reference``.
        tool_name: The loaded tool's name, as declared in ``tools``.
        cache_control: Optional caller cache marker; it re-emits on the
            enclosing tool result (the same prompt boundary).
    """

    type: Literal["tool_reference"]
    tool_name: str = Field(min_length=1, max_length=256)
    cache_control: CacheControl | None = None


ToolResultContent = (
    str | Sequence[TextBlockLike | ImageBlock | DocumentBlock | ToolReferenceBlock] | None
)
"""A validated ``tool_result.content``: a string, or its ordered sub-blocks."""


def tool_result_references(content: ToolResultContent) -> tuple[str, ...]:
    """Return the deferred tools a tool result loads, in caller order.

    Args:
        content: Validated ``tool_result.content``.

    Returns:
        The referenced tool names, ``()`` when the result loads none.
    """
    if content is None or isinstance(content, str):
        return ()
    return tuple(part.tool_name for part in content if isinstance(part, ToolReferenceBlock))


def tool_reference_cache_marker(content: ToolResultContent) -> CacheControl | None:
    """Return the cache marker the caller put on a tool result's last marked reference.

    A marker on a reference sub-block bounds the same prompt prefix as one on
    the enclosing tool result, so it re-emits there.

    Args:
        content: Validated ``tool_result.content``.

    Returns:
        The marker, or ``None`` when no reference carries one.
    """
    if content is None or isinstance(content, str):
        return None
    markers = [
        part.cache_control
        for part in content
        if isinstance(part, ToolReferenceBlock) and part.cache_control is not None
    ]
    return markers[-1] if markers else None


def cache_marker(marker: CacheControl | None) -> JsonObject | None:
    """Dump one caller cache annotation for a retained content part."""
    return None if marker is None else marker.model_dump(mode="json", exclude_none=True)


def plain_document_text(block: DocumentBlock, param: str) -> str | None:
    """Return a plain-text document's text, refusing enabled citations.

    Args:
        block: Validated caller document block.
        param: Public parameter path of the block.

    Returns:
        The text with its title, or None when the source is not text.

    Raises:
        OpenAIProtocolError: The text document enables citations.
    """
    text = text_document_text(block)
    if text is not None and block.citations is not None and block.citations.enabled:
        raise invalid_field(
            f"{param}.citations",
            "document citations are not supported over this gateway; "
            "send citations.enabled as false or omit the field.",
        )
    return text


def tool_result_text(content: ToolResultContent, param: str) -> str:
    """Flatten one tool result into the canonical tool-message text.

    A plain-text document is the caller's text, exactly as on a user turn;
    image, PDF document, and tool-reference sub-blocks contribute no text.

    Args:
        content: Validated ``tool_result.content``.
        param: Public parameter path of the tool result.

    Returns:
        The concatenated text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for index, part in enumerate(content):
        if isinstance(part, DocumentBlock):
            text = plain_document_text(part, f"{param}.content.{index}")
            if text is not None:
                texts.append(text)
        elif not isinstance(part, (ImageBlock, ToolReferenceBlock)):
            texts.append(part.text)
    return "".join(texts)


def tool_result_parts(content: ToolResultContent, param: str) -> tuple[MessageContentPart, ...]:
    """Retain a tool result's ordered parts when it carries an attachment.

    Text-only results keep the flattened ``content`` string and no parts, so
    existing requests serialize and digest exactly as before. An image
    sub-block (a tool screenshot) or a PDF document sub-block (a tool-returned
    file) makes the result multimodal: every part is retained in caller order
    so an attachment-capable rung re-emits the exact block run. Empty text
    adds nothing to the flattened content and cannot ride a multimodal turn,
    mirroring the user-message path.

    Args:
        content: Validated ``tool_result.content``.
        param: Public parameter path of the tool result.

    Returns:
        The ordered canonical parts, or ``()`` for a text-only result.

    Raises:
        OpenAIProtocolError: An attachment's source is invalid or a
            document enables citations.
    """
    if content is None or isinstance(content, str):
        return ()
    if not any(
        isinstance(part, ImageBlock)
        or (isinstance(part, DocumentBlock) and text_document_text(part) is None)
        for part in content
    ):
        return ()
    parts: list[MessageContentPart] = []
    for index, part in enumerate(content):
        path = f"{param}.content.{index}"
        if isinstance(part, ImageBlock):
            parts.append(image_part_from_block(part, path))
        elif isinstance(part, DocumentBlock):
            text = plain_document_text(part, path)
            if text is None:
                parts.append(document_part_from_block(part, path))
            elif text:
                parts.append(
                    TextContentPart(text=text, cache_control=cache_marker(part.cache_control))
                )
        elif isinstance(part, ToolReferenceBlock):
            # A reference is not content: it rides the tool message's
            # ``tool_references`` carrier (see ``tool_result_references``).
            continue
        elif part.text:
            parts.append(
                TextContentPart(text=part.text, cache_control=cache_marker(part.cache_control))
            )
    return tuple(parts)
