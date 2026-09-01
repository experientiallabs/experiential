"""Multimodal message content parts shared by the gateway and model clients.

A caller message is canonically one flattened text string. When the caller
also sends images, the ordered parts that produced that string are retained
here so every provider wire can re-emit the caller's exact interleaving. The
text parts always flatten to the message's canonical content, so a text-only
route sees exactly what it saw before images existed.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Annotated, Literal

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

_DATA_URL = re.compile(
    r"^data:(?P<media_type>[\w.+-]+/[\w.+-]+)(?P<parameters>;[^,]*)?,(?P<data>.*)$",
    re.DOTALL,
)


class TextContentPart(ContractModel):
    """One text run of a message that also carries images."""

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
    cache_control: JsonObject | None = None
    """Prompt-cache breakpoint the caller placed on this image, re-emitted
    verbatim on wires that cache a marked block natively and dropped
    elsewhere: a cache hint changes cost, not semantics."""

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


MessageContentPart = Annotated[TextContentPart | ImageContentPart, Field(discriminator="kind")]


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
