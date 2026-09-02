"""Validation tests for canonical multimodal content parts."""

from __future__ import annotations

import pytest

from exp.common.models.content import (
    MAXIMUM_DOCUMENT_BASE64_BYTES,
    MAXIMUM_IMAGE_BASE64_BYTES,
    DocumentContentPart,
    ImageContentPart,
    document_part_from_file_data,
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


def test_a_cache_marker_never_changes_the_serialized_image() -> None:
    """A cost-only cache hint leaves replay identity untouched."""
    plain = ImageContentPart(media_type="image/png", data=_PNG_BASE64)
    marked = ImageContentPart(
        media_type="image/png",
        data=_PNG_BASE64,
        cache_control={"type": "ephemeral"},
    )
    assert marked.cache_control == {"type": "ephemeral"}
    assert plain.model_dump(mode="json") == marked.model_dump(mode="json")


_PDF_BASE64 = "JVBERi0xLjQKJSBtaW5pbWFsIHBkZgo="
"""One short PDF header, base64 encoded."""


def test_document_inlines_from_a_pdf_data_url() -> None:
    """An OpenAI ``file_data`` data URL decodes to one inline PDF part."""
    part = document_part_from_file_data(
        f"data:application/pdf;base64,{_PDF_BASE64}", name="brief.pdf"
    )
    assert part.kind == "document"
    assert part.media_type == "application/pdf"
    assert part.data == _PDF_BASE64
    assert part.name == "brief.pdf"
    assert part.url is None
    assert part.data_url() == f"data:application/pdf;base64,{_PDF_BASE64}"


def test_document_inlines_from_bare_base64() -> None:
    """Bare base64 ``file_data`` (the documented alternative form) is accepted."""
    part = document_part_from_file_data(_PDF_BASE64)
    assert part.data == _PDF_BASE64
    assert part.name is None


def test_document_data_url_must_name_a_pdf() -> None:
    """A data URL for any other media type fails closed."""
    with pytest.raises(ValueError, match="unsupported document media type"):
        document_part_from_file_data(f"data:text/plain;base64,{_PDF_BASE64}")
    with pytest.raises(ValueError, match="base64"):
        document_part_from_file_data("data:application/pdf,plain-text")


def test_malformed_document_base64_is_rejected() -> None:
    """Inline bytes that are not base64 never reach a provider."""
    with pytest.raises(ValueError, match="base64"):
        DocumentContentPart(data="not base64 !!")
    with pytest.raises(ValueError, match="base64"):
        document_part_from_file_data("not base64 !!")


def test_oversized_inline_document_is_rejected() -> None:
    """An inline document beyond the gateway ceiling fails closed."""
    with pytest.raises(ValueError, match="maximum encoded size"):
        document_part_from_file_data(
            f"data:application/pdf;base64,{'A' * (MAXIMUM_DOCUMENT_BASE64_BYTES + 4)}"
        )
    with pytest.raises(ValueError):
        DocumentContentPart(data="A" * (MAXIMUM_DOCUMENT_BASE64_BYTES + 4))


def test_document_requires_exactly_one_carrier() -> None:
    """A document carries either inline bytes or a URL, never both or neither."""
    with pytest.raises(ValueError, match="either inline data or a URL"):
        DocumentContentPart(data=_PDF_BASE64, url="https://example.com/a.pdf")
    with pytest.raises(ValueError, match="either inline data or a URL"):
        DocumentContentPart()


def test_document_url_must_be_http() -> None:
    """A non-http(s) document URL is never forwarded to a provider."""
    with pytest.raises(ValueError, match="http"):
        DocumentContentPart(url="file:///etc/passwd")
    remote = DocumentContentPart(url="https://example.com/a.pdf")
    assert remote.data is None
    assert remote.url == "https://example.com/a.pdf"


def test_document_media_type_is_pdf_only() -> None:
    """The canonical part accepts no media type other than PDF."""
    with pytest.raises(ValueError):
        DocumentContentPart.model_validate({"data": _PDF_BASE64, "media_type": "text/plain"})


def test_document_name_joins_serialization_but_a_cache_marker_does_not() -> None:
    """The name is model-visible and semantic; a cache hint is cost-only."""
    plain = DocumentContentPart(data=_PDF_BASE64)
    named = DocumentContentPart(data=_PDF_BASE64, name="brief.pdf")
    marked = DocumentContentPart(data=_PDF_BASE64, cache_control={"type": "ephemeral"})
    assert plain.model_dump(mode="json") != named.model_dump(mode="json")
    assert marked.cache_control == {"type": "ephemeral"}
    assert plain.model_dump(mode="json") == marked.model_dump(mode="json")
