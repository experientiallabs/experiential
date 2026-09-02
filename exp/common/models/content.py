"""Multimodal message content parts shared by the gateway and model clients.

A caller message is canonically one flattened text string. When the caller
also sends images, videos, or documents, the ordered parts that produced that
string are retained here so every provider wire can re-emit the caller's exact
interleaving. The text parts always flatten to the message's canonical
content, so a text-only route sees exactly what it saw before attachments
existed.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject

ImageMediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]

IMAGE_MEDIA_TYPES: dict[str, ImageMediaType] = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/gif": "image/gif",
    "image/webp": "image/webp",
}
"""Media types every image-capable provider wire in this gateway accepts."""

MAXIMUM_IMAGE_BASE64_BYTES = 5 * 1024 * 1024
"""Largest encoded image this gateway forwards (the narrowest provider cap)."""

MAXIMUM_IMAGES_PER_REQUEST = 20
"""Largest number of images one request may carry across all its messages."""

VideoMediaType = Literal[
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/webm",
    "video/x-flv",
    "video/3gpp",
    "video/x-ms-wmv",
]

VIDEO_MEDIA_TYPES: dict[str, VideoMediaType] = {
    "video/mp4": "video/mp4",
    "video/mpeg": "video/mpeg",
    "video/quicktime": "video/quicktime",
    "video/mov": "video/quicktime",
    "video/webm": "video/webm",
    "video/x-flv": "video/x-flv",
    "video/flv": "video/x-flv",
    "video/3gpp": "video/3gpp",
    "video/x-ms-wmv": "video/x-ms-wmv",
    "video/wmv": "video/x-ms-wmv",
}
"""Media types every video-capable provider wire in this gateway accepts.

The set is the intersection of the Gemini and Bedrock Converse format lists,
so an admitted video never fails on a format one native wire lacks. Common
non-standard spellings map to their canonical type."""

MAXIMUM_VIDEO_BASE64_BYTES = 20 * 1024 * 1024
"""Largest encoded video this gateway forwards inline.

Bedrock Converse caps the whole inline request payload at 25 MB, the
narrowest inline limit across the video-capable wires; the ceiling leaves
room for the surrounding request. Larger videos travel by URL."""

MAXIMUM_VIDEOS_PER_REQUEST = 10
"""Largest number of videos one request may carry across all its messages.

Gemini accepts at most ten videos per request; Bedrock Nova accepts one, and
that narrower model limit surfaces as a provider error rather than a gateway
ceiling."""

DocumentMediaType = Literal["application/pdf"]

DOCUMENT_MEDIA_TYPES: dict[str, DocumentMediaType] = {"application/pdf": "application/pdf"}
"""Media types every document-capable provider wire in this gateway accepts.

PDF is the one document format that every wire carries natively (OpenAI
``file`` and ``input_file``, Anthropic ``document``, Gemini ``inline_data``,
Bedrock ``document``); other office formats differ per provider and are not
forwarded."""

MAXIMUM_DOCUMENT_BASE64_BYTES = 6 * 1024 * 1024
"""Largest encoded document this gateway forwards.

Bedrock Converse caps one document at 4.5 MB of raw bytes, the narrowest
provider cap (OpenAI and Gemini accept 50 MB, Anthropic a 32 MB request);
6 MiB of base64 is that many raw bytes, so every declared rung can carry
every admitted document."""

MAXIMUM_DOCUMENTS_PER_REQUEST = 5
"""Largest number of documents one request may carry across all its messages.

Bedrock Converse accepts at most five document blocks per request, the
narrowest provider cap."""

MAXIMUM_DOCUMENT_NAME_CHARACTERS = 200
"""Longest document name forwarded; Bedrock caps its ``name`` at 200."""

MediaHandleProvider = Literal["openai", "anthropic", "gemini", "vertex", "bedrock"]
"""Providers whose wire accepts a reference to media the caller already
uploaded to that provider. A handle is scoped to exactly one of them."""

MEDIA_HANDLE_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "anthropic", "gemini", "vertex", "bedrock"}
)
"""Every provider a media handle can name."""

MEDIA_TYPE_BOUND_HANDLE_PROVIDERS: frozenset[str] = frozenset({"vertex", "bedrock"})
"""Handle providers whose wire needs the media type next to the reference:
Vertex ``file_data`` requires ``mime_type`` for a ``gs://`` object and
Bedrock Converse requires ``format`` next to an ``s3Location``."""

GEMINI_FILE_URI_PREFIX = "https://generativelanguage.googleapis.com/v1beta/files/"
"""Prefix of every URI the Gemini Files API mints."""

_OPENAI_FILE_ID = re.compile(r"^file-[A-Za-z0-9_-]{1,256}$")
_ANTHROPIC_FILE_ID = re.compile(r"^file_[A-Za-z0-9_-]{1,256}$")
_GEMINI_FILE_NAME = re.compile(r"^[a-z0-9-]{1,128}$")
_BUCKET_OBJECT_URI = re.compile(r"^(?P<bucket>[a-z0-9][a-z0-9._-]{1,254})/(?P<key>.+)$", re.DOTALL)
_AWS_ACCOUNT_ID = re.compile(r"^[0-9]{12}$")

_MEDIA_HANDLE_SCHEMES: dict[str, MediaHandleProvider] = {
    "s3://": "bedrock",
    "gs://": "vertex",
    GEMINI_FILE_URI_PREFIX: "gemini",
}
"""URI prefixes a caller URL field may carry that name a provider handle."""


class MediaHandle(ContractModel):
    """One reference to media the caller already uploaded to a provider.

    Handles are provider scoped and never portable: an OpenAI ``file_id``
    means nothing to Gemini, and an ``s3://`` object is readable only by the
    caller's Bedrock identity. The gateway forwards a handle verbatim to a
    route on the same provider and never uploads, fetches, or mints one.
    """

    provider: MediaHandleProvider
    reference: str = Field(min_length=1, max_length=8_192)
    """The provider's own identifier: an OpenAI or Anthropic file id, a Gemini
    Files API URI, a ``gs://bucket/object`` URI, or an ``s3://bucket/key``
    URI."""
    bucket_owner: str | None = None
    """AWS account id that owns a cross-account S3 bucket (Bedrock only)."""

    @model_validator(mode="after")
    def _validate_shape(self) -> MediaHandle:
        """Require the reference to have the named provider's handle shape.

        Returns:
            The validated handle.

        Raises:
            ValueError: The reference does not match the provider's handle
                form, or a bucket owner accompanies a non-Bedrock handle.
        """
        if self.bucket_owner is not None:
            if self.provider != "bedrock":
                raise ValueError("bucket_owner applies only to bedrock s3 handles")
            if _AWS_ACCOUNT_ID.match(self.bucket_owner) is None:
                raise ValueError("bucket_owner must be a 12 digit AWS account id")
        reference = self.reference
        if self.provider == "openai":
            if _OPENAI_FILE_ID.match(reference) is None:
                raise ValueError("an openai file handle looks like file-...")
        elif self.provider == "anthropic":
            if _ANTHROPIC_FILE_ID.match(reference) is None:
                raise ValueError("an anthropic file handle looks like file_...")
        elif self.provider == "gemini":
            if not reference.startswith(GEMINI_FILE_URI_PREFIX) or (
                _GEMINI_FILE_NAME.match(reference.removeprefix(GEMINI_FILE_URI_PREFIX)) is None
            ):
                raise ValueError(f"a gemini file handle looks like {GEMINI_FILE_URI_PREFIX}<name>")
        elif self.provider == "vertex":
            if not reference.startswith("gs://") or (
                _BUCKET_OBJECT_URI.match(reference.removeprefix("gs://")) is None
            ):
                raise ValueError("a vertex handle looks like gs://bucket/object")
        elif not reference.startswith("s3://") or (
            _BUCKET_OBJECT_URI.match(reference.removeprefix("s3://")) is None
        ):
            raise ValueError("a bedrock handle looks like s3://bucket/key")
        return self


def media_handle_from_uri(uri: str, *, bucket_owner: str | None = None) -> MediaHandle | None:
    """Recognize a provider handle written into a caller URL field.

    Args:
        uri: Caller value from a URL field such as ``image_url.url``.
        bucket_owner: Optional AWS account id for a cross-account bucket.

    Returns:
        The handle for an ``s3://``, ``gs://``, or Gemini Files URI, or
        ``None`` for a URL that is not a handle.

    Raises:
        ValueError: The URI has a handle scheme but a malformed body.
    """
    for prefix, provider in _MEDIA_HANDLE_SCHEMES.items():
        if uri.startswith(prefix):
            return MediaHandle(provider=provider, reference=uri, bucket_owner=bucket_owner)
    return None


def _validate_handle_media_type(handle: MediaHandle, media_type: str | None, noun: str) -> None:
    """Require a media type where the handle's wire cannot do without one.

    Args:
        handle: The provider handle carried by the part.
        media_type: The part's declared media type, if any.
        noun: Media kind for the error message.

    Raises:
        ValueError: The handle's provider needs a media type and none is set.
    """
    if media_type is None and handle.provider in MEDIA_TYPE_BOUND_HANDLE_PROVIDERS:
        raise ValueError(f"a {handle.provider} {noun} handle needs its media type")


_DATA_URL = re.compile(
    r"^data:(?P<media_type>[\w.+-]+/[\w.+-]+)(?P<parameters>;[^,]*)?,(?P<data>.*)$",
    re.DOTALL,
)

_VIDEO_URL_EXTENSIONS: dict[str, VideoMediaType] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".flv": "video/x-flv",
    ".3gp": "video/3gpp",
    ".wmv": "video/x-ms-wmv",
}
"""Container suffixes whose media type a remote URL states on its own. Gemini
requires a MIME type alongside a fetched HTTP URL, so the suffix is recorded
when it is unambiguous; other URLs carry no media type."""


def _carrier_count(*carriers: str | MediaHandle | None) -> int:
    """Count the carriers a part actually sets."""
    return sum(carrier is not None for carrier in carriers)


class TextContentPart(ContractModel):
    """One text run of a message that also carries images or videos."""

    kind: Literal["text"] = "text"
    text: str


class ImageContentPart(ContractModel):
    """One caller-supplied image, either inline bytes or a remote URL.

    Exactly one carrier is present. Inline images hold standard base64 of
    the image bytes with their declared media type; remote images hold the
    caller's URL and are forwarded only on wires that fetch it themselves.
    """

    kind: Literal["image"] = "image"
    media_type: ImageMediaType | None = None
    data: str | None = Field(default=None, max_length=MAXIMUM_IMAGE_BASE64_BYTES)
    url: str | None = Field(default=None, max_length=8_192)
    handle: MediaHandle | None = None
    detail: Literal["auto", "low", "high"] | None = None
    cache_control: JsonObject | None = Field(default=None, exclude=True)
    """Prompt-cache breakpoint the caller placed on this image, re-emitted
    verbatim on wires that cache a marked block natively and dropped
    elsewhere: a cache hint changes cost, not semantics. Like the other
    cache carriers it joins neither serialization nor replay identity, so
    two otherwise identical requests differing only here are one request."""

    @model_validator(mode="after")
    def _require_one_carrier(self) -> ImageContentPart:
        """Require exactly one well-formed image carrier.

        Returns:
            The validated image part.

        Raises:
            ValueError: Not exactly one carrier is present, the inline
                payload is not base64, a remote URL is not http(s), or a
                handle whose wire needs a media type has none.
        """
        if _carrier_count(self.data, self.url, self.handle) != 1:
            raise ValueError("an image needs exactly one of inline data, a URL, or a handle")
        if self.data is not None:
            if self.media_type is None:
                raise ValueError("inline image data needs its media type")
            try:
                base64.b64decode(self.data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("inline image data must be base64") from exc
        elif self.url is not None and not self.url.startswith(("http://", "https://")):
            raise ValueError("an image URL must be an http(s) URL")
        elif self.handle is not None:
            _validate_handle_media_type(self.handle, self.media_type, "image")
        return self

    def data_url(self) -> str:
        """Return this image as one wire value for the OpenAI image field."""
        if self.data is None:
            return self.url or ""
        return f"data:{self.media_type};base64,{self.data}"

    def image_format(self) -> str:
        """Return the bare format name Bedrock and Gemini style wires use."""
        return (self.media_type or "image/png").removeprefix("image/")


class VideoContentPart(ContractModel):
    """One caller-supplied video, either inline bytes or a remote URL.

    Exactly one carrier is present. Inline videos hold standard base64 of
    the video bytes with their declared media type; remote videos hold the
    caller's http(s) URL and are forwarded only on wires whose provider
    fetches it itself. Unlike images, no wire caches a video block, so the
    part carries no cache marker.
    """

    kind: Literal["video"] = "video"
    media_type: VideoMediaType | None = None
    data: str | None = Field(default=None, max_length=MAXIMUM_VIDEO_BASE64_BYTES)
    url: str | None = Field(default=None, max_length=8_192)
    handle: MediaHandle | None = None

    @model_validator(mode="after")
    def _require_one_carrier(self) -> VideoContentPart:
        """Require exactly one well-formed video carrier.

        Returns:
            The validated video part.

        Raises:
            ValueError: Not exactly one carrier is present, the inline
                payload is not base64, a remote URL is not http(s), or a
                handle whose wire needs a media type has none.
        """
        if _carrier_count(self.data, self.url, self.handle) != 1:
            raise ValueError("a video needs exactly one of inline data, a URL, or a handle")
        if self.data is not None:
            if self.media_type is None:
                raise ValueError("inline video data needs its media type")
            try:
                base64.b64decode(self.data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("inline video data must be base64") from exc
        elif self.url is not None and not self.url.startswith(("http://", "https://")):
            raise ValueError("a video URL must be an http(s) URL")
        elif self.handle is not None:
            _validate_handle_media_type(self.handle, self.media_type, "video")
        return self

    def data_url(self) -> str:
        """Return this video as one wire value for an OpenAI-style URL field."""
        if self.data is None:
            return self.url or ""
        return f"data:{self.media_type};base64,{self.data}"


class DocumentContentPart(ContractModel):
    """One caller-supplied PDF document, either inline bytes or a remote URL.

    Exactly one carrier is present. Inline documents hold standard base64 of
    the PDF bytes; remote documents hold the caller's URL and are forwarded
    only on wires that fetch it themselves. The optional name is the
    caller's filename or title: providers show it to the model, so it is
    part of what the model sees and joins request identity.
    """

    kind: Literal["document"] = "document"
    media_type: DocumentMediaType = "application/pdf"
    data: str | None = Field(default=None, max_length=MAXIMUM_DOCUMENT_BASE64_BYTES)
    url: str | None = Field(default=None, max_length=8_192)
    handle: MediaHandle | None = None
    name: str | None = Field(
        default=None, min_length=1, max_length=MAXIMUM_DOCUMENT_NAME_CHARACTERS
    )
    cache_control: JsonObject | None = Field(default=None, exclude=True)
    """Prompt-cache breakpoint the caller placed on this document, re-emitted
    verbatim on wires that cache a marked block natively and dropped
    elsewhere. Cost, not semantics: never in serialization or replay
    identity."""

    @model_validator(mode="after")
    def _require_one_carrier(self) -> DocumentContentPart:
        """Require exactly one well-formed document carrier.

        Returns:
            The validated document part.

        Raises:
            ValueError: Not exactly one carrier is present, the inline
                payload is not base64, or a remote URL is not http(s).
        """
        if _carrier_count(self.data, self.url, self.handle) != 1:
            raise ValueError("a document needs exactly one of inline data, a URL, or a handle")
        if self.data is not None:
            try:
                base64.b64decode(self.data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("inline document data must be base64") from exc
        elif self.url is not None and not self.url.startswith(("http://", "https://")):
            raise ValueError("a document URL must be an http(s) URL")
        return self

    def data_url(self) -> str:
        """Return the inline bytes as one OpenAI ``file_data`` value."""
        return f"data:{self.media_type};base64,{self.data or ''}"


MessageContentPart = Annotated[
    TextContentPart | ImageContentPart | VideoContentPart | DocumentContentPart,
    Field(discriminator="kind"),
]


def image_part_from_url(
    url: str,
    *,
    detail: Literal["auto", "low", "high"] | None = None,
) -> ImageContentPart:
    """Build one image part from a caller URL, inlining a data URL's bytes.

    Args:
        url: Caller value from an OpenAI image field: either a data URL
            carrying base64 image bytes or a remote http(s) URL.
        detail: Caller detail hint preserved for the wires that accept it.

    Returns:
        The canonical image part for that value.

    Raises:
        ValueError: The data URL is malformed, is not base64, or names a
            media type this gateway does not forward.
    """
    match = _DATA_URL.match(url)
    if match is None:
        handle = media_handle_from_uri(url)
        if handle is not None:
            return ImageContentPart(
                handle=handle, media_type=_image_media_type_from_url(url), detail=detail
            )
        return ImageContentPart(url=url, detail=detail)
    media_type = IMAGE_MEDIA_TYPES.get(match["media_type"].lower())
    if media_type is None:
        raise ValueError(f"unsupported image media type {match['media_type']!r}")
    if ";base64" not in (match["parameters"] or ""):
        raise ValueError("inline images must be base64 encoded")
    data = match["data"].strip()
    if len(data) > MAXIMUM_IMAGE_BASE64_BYTES:
        raise ValueError("inline image exceeds the maximum encoded size")
    return ImageContentPart(media_type=media_type, data=data, detail=detail)


_IMAGE_URL_EXTENSIONS: dict[str, ImageMediaType] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
"""Image suffixes whose media type an object URI states on its own; Bedrock
and Vertex need it next to a bucket handle."""


def _image_media_type_from_url(url: str) -> ImageMediaType | None:
    """Return the image media type a URI's path suffix states, if any."""
    path = urlsplit(url).path.lower()
    for suffix, media_type in _IMAGE_URL_EXTENSIONS.items():
        if path.endswith(suffix):
            return media_type
    return None


def _media_type_from_url(url: str) -> VideoMediaType | None:
    """Return the media type a remote video URL's path suffix states, if any.

    Args:
        url: Remote http(s) URL of a video.

    Returns:
        The container media type for a known suffix, otherwise ``None``.
    """
    path = urlsplit(url).path.lower()
    for suffix, media_type in _VIDEO_URL_EXTENSIONS.items():
        if path.endswith(suffix):
            return media_type
    return None


def video_part_from_url(url: str) -> VideoContentPart:
    """Build one video part from a caller URL, inlining a data URL's bytes.

    Args:
        url: Caller value from an OpenAI-style ``video_url`` field: either a
            data URL carrying base64 video bytes or a remote http(s) URL.

    Returns:
        The canonical video part for that value.

    Raises:
        ValueError: The data URL is malformed, is not base64, or names a
            media type this gateway does not forward.
    """
    match = _DATA_URL.match(url)
    if match is None:
        handle = media_handle_from_uri(url)
        if handle is not None:
            return VideoContentPart(handle=handle, media_type=_media_type_from_url(url))
        return VideoContentPart(url=url, media_type=_media_type_from_url(url))
    media_type = VIDEO_MEDIA_TYPES.get(match["media_type"].lower())
    if media_type is None:
        raise ValueError(f"unsupported video media type {match['media_type']!r}")
    if ";base64" not in (match["parameters"] or ""):
        raise ValueError("inline videos must be base64 encoded")
    data = match["data"].strip()
    if len(data) > MAXIMUM_VIDEO_BASE64_BYTES:
        raise ValueError("inline video exceeds the maximum encoded size")
    return VideoContentPart(media_type=media_type, data=data)


def document_part_from_file_data(
    file_data: str,
    *,
    name: str | None = None,
) -> DocumentContentPart:
    """Build one inline document part from an OpenAI ``file_data`` value.

    Args:
        file_data: Caller value from a ``file`` or ``input_file`` part:
            either a ``data:application/pdf;base64,...`` URL or the bare
            base64 of the PDF bytes (the official documentation shows both).
        name: Caller filename preserved for the wires that show it.

    Returns:
        The canonical document part for that value.

    Raises:
        ValueError: The data URL is malformed, is not base64, names a media
            type this gateway does not forward, or exceeds the size ceiling.
    """
    match = _DATA_URL.match(file_data)
    if match is None:
        data = file_data.strip()
    else:
        if DOCUMENT_MEDIA_TYPES.get(match["media_type"].lower()) is None:
            raise ValueError(f"unsupported document media type {match['media_type']!r}")
        if ";base64" not in (match["parameters"] or ""):
            raise ValueError("inline documents must be base64 encoded")
        data = match["data"].strip()
    if len(data) > MAXIMUM_DOCUMENT_BASE64_BYTES:
        raise ValueError("inline document exceeds the maximum encoded size")
    return DocumentContentPart(data=data, name=name or None)
