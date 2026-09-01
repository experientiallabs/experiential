"""Validation tests for canonical multimodal content parts."""

from __future__ import annotations

import pytest

from exp.common.models.content import (
    MAXIMUM_IMAGE_BASE64_BYTES,
    ImageContentPart,
    image_part_from_url,
)

_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
"""One valid single-pixel PNG, base64 encoded."""


@pytest.mark.parametrize(
    "media_type",
    ["image/png", "image/jpeg", "image/gif", "image/webp"],
)
def test_every_supported_media_type_inlines(media_type: str) -> None:
    """Each forwarded media type round-trips through a data URL."""
    part = image_part_from_url(f"data:{media_type};base64,{_PNG_BASE64}")
    assert part.media_type == media_type
    assert part.data == _PNG_BASE64
    assert part.data_url() == f"data:{media_type};base64,{_PNG_BASE64}"
    assert part.image_format() == media_type.removeprefix("image/")


def test_remote_url_is_kept_verbatim() -> None:
    """An http(s) image URL rides as a URL carrier with no inline bytes."""
    part = image_part_from_url("https://example.com/cat.png", detail="low")
    assert part.url == "https://example.com/cat.png"
    assert part.data is None
    assert part.detail == "low"
    assert part.data_url() == "https://example.com/cat.png"


def test_unsupported_media_type_is_rejected() -> None:
    """A media type no provider wire accepts fails closed."""
    with pytest.raises(ValueError, match="unsupported image media type"):
        image_part_from_url(f"data:image/tiff;base64,{_PNG_BASE64}")


def test_non_base64_data_url_is_rejected() -> None:
    """A data URL must carry base64, not percent-encoded text."""
    with pytest.raises(ValueError, match="base64"):
        image_part_from_url("data:image/png,not-base64")


def test_malformed_base64_is_rejected() -> None:
    """Inline bytes that are not base64 never reach a provider."""
    with pytest.raises(ValueError, match="base64"):
        ImageContentPart(media_type="image/png", data="not base64 !!")


def test_oversized_inline_image_is_rejected() -> None:
    """An inline image beyond the narrowest provider cap fails closed."""
    with pytest.raises(ValueError, match="maximum encoded size"):
        image_part_from_url(f"data:image/png;base64,{'A' * (MAXIMUM_IMAGE_BASE64_BYTES + 4)}")


def test_exactly_one_carrier_is_required() -> None:
    """An image carries either inline bytes or a URL, never both or neither."""
    with pytest.raises(ValueError, match="either inline data or a URL"):
        ImageContentPart(media_type="image/png", data=_PNG_BASE64, url="https://example.com/a.png")
    with pytest.raises(ValueError, match="either inline data or a URL"):
        ImageContentPart()


def test_inline_data_requires_its_media_type() -> None:
    """Bytes without a declared media type cannot be encoded for any wire."""
    with pytest.raises(ValueError, match="media type"):
        ImageContentPart(data=_PNG_BASE64)


def test_non_http_url_is_rejected() -> None:
    """A non-http(s) URL is never forwarded to a provider."""
    with pytest.raises(ValueError, match="http"):
        ImageContentPart(url="file:///etc/passwd")
