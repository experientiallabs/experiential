"""Multimodal message content parts shared by the gateway and model clients.

A caller message is canonically one flattened text string. When the caller
also sends images or videos, the ordered parts that produced that string are
retained here so every provider wire can re-emit the caller's exact
interleaving. The text parts always flatten to the message's canonical
content, so a text-only route sees exactly what it saw before media existed.
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
            ValueError: Both or neither carrier is present, the inline
                payload is not base64, or a remote URL is not http(s).
        """
        if (self.data is None) == (self.url is None):
            raise ValueError("an image needs either inline data or a URL")
        if self.data is not None:
            if self.media_type is None:
                raise ValueError("inline image data needs its media type")
            try:
                base64.b64decode(self.data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("inline image data must be base64") from exc
        elif self.url is not None and not self.url.startswith(("http://", "https://")):
            raise ValueError("an image URL must be an http(s) URL")
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

    @model_validator(mode="after")
    def _require_one_carrier(self) -> VideoContentPart:
        """Require exactly one well-formed video carrier.

        Returns:
            The validated video part.

        Raises:
            ValueError: Both or neither carrier is present, the inline
                payload is not base64, or a remote URL is not http(s).
        """
        if (self.data is None) == (self.url is None):
            raise ValueError("a video needs either inline data or a URL")
        if self.data is not None:
            if self.media_type is None:
                raise ValueError("inline video data needs its media type")
            try:
                base64.b64decode(self.data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("inline video data must be base64") from exc
        elif self.url is not None and not self.url.startswith(("http://", "https://")):
            raise ValueError("a video URL must be an http(s) URL")
        return self

    def data_url(self) -> str:
        """Return this video as one wire value for an OpenAI-style URL field."""
        if self.data is None:
            return self.url or ""
        return f"data:{self.media_type};base64,{self.data}"


MessageContentPart = Annotated[
    TextContentPart | ImageContentPart | VideoContentPart,
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
