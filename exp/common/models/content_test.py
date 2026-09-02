"""Validation tests for canonical multimodal content parts."""

from __future__ import annotations

import pytest

from exp.common.models.content import (
    GEMINI_FILE_URI_PREFIX,
    MAXIMUM_DOCUMENT_BASE64_BYTES,
    MAXIMUM_IMAGE_BASE64_BYTES,
    MAXIMUM_VIDEO_BASE64_BYTES,
    DocumentContentPart,
    ImageContentPart,
    MediaHandle,
    VideoContentPart,
    document_part_from_file_data,
    image_part_from_url,
    media_handle_from_uri,
    video_part_from_url,
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
    """An image carries inline bytes, a URL, or a handle, never two or none."""
    with pytest.raises(ValueError, match="exactly one of inline data, a URL, or a handle"):
        ImageContentPart(media_type="image/png", data=_PNG_BASE64, url="https://example.com/a.png")
    with pytest.raises(ValueError, match="exactly one of inline data, a URL, or a handle"):
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


_MP4_BASE64 = "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDE="
"""A base64 prefix of an MP4 ``ftyp`` box, enough for a carrier fixture."""


@pytest.mark.parametrize(
    ("wire_type", "media_type"),
    [
        ("video/mp4", "video/mp4"),
        ("video/mpeg", "video/mpeg"),
        ("video/quicktime", "video/quicktime"),
        ("video/mov", "video/quicktime"),
        ("video/webm", "video/webm"),
        ("video/x-flv", "video/x-flv"),
        ("video/3gpp", "video/3gpp"),
        ("video/x-ms-wmv", "video/x-ms-wmv"),
    ],
)
def test_every_supported_video_media_type_inlines(wire_type: str, media_type: str) -> None:
    """Each documented video container round-trips through a data URL."""
    part = video_part_from_url(f"data:{wire_type};base64,{_MP4_BASE64}")
    assert part.kind == "video"
    assert part.media_type == media_type
    assert part.data == _MP4_BASE64
    assert part.data_url() == f"data:{media_type};base64,{_MP4_BASE64}"


def test_unsupported_video_media_type_is_rejected() -> None:
    """A container no provider wire documents fails closed."""
    with pytest.raises(ValueError, match="unsupported video media type"):
        video_part_from_url(f"data:video/x-matroska;base64,{_MP4_BASE64}")
    with pytest.raises(ValueError, match="unsupported video media type"):
        video_part_from_url(f"data:image/png;base64,{_MP4_BASE64}")


def test_video_remote_url_keeps_no_bytes() -> None:
    """An http(s) video URL is retained for wires whose provider fetches it."""
    part = video_part_from_url("https://example.com/clip.mp4")
    assert part.url == "https://example.com/clip.mp4"
    assert part.data is None
    assert part.media_type == "video/mp4"
    assert part.data_url() == "https://example.com/clip.mp4"
    assert video_part_from_url("https://example.com/clip?id=1").media_type is None


def test_video_inline_requires_strict_base64() -> None:
    """Inline video bytes that are not base64 never reach a provider."""
    with pytest.raises(ValueError, match="base64"):
        VideoContentPart(media_type="video/mp4", data="not base64 !!")
    with pytest.raises(ValueError, match="base64 encoded"):
        video_part_from_url("data:video/mp4,plain")


def test_oversized_inline_video_is_rejected() -> None:
    """An inline video beyond the narrowest provider cap fails closed."""
    with pytest.raises(ValueError, match="maximum encoded size"):
        video_part_from_url(f"data:video/mp4;base64,{'A' * (MAXIMUM_VIDEO_BASE64_BYTES + 4)}")


def test_video_needs_exactly_one_carrier() -> None:
    """A video carries either inline bytes or a URL, never both or neither."""
    with pytest.raises(ValueError, match="exactly one of inline data, a URL, or a handle"):
        VideoContentPart(media_type="video/mp4", data=_MP4_BASE64, url="https://example.com/a.mp4")
    with pytest.raises(ValueError, match="exactly one of inline data, a URL, or a handle"):
        VideoContentPart()
    with pytest.raises(ValueError, match="media type"):
        VideoContentPart(data=_MP4_BASE64)
    with pytest.raises(ValueError, match="http"):
        VideoContentPart(url="gs://bucket/clip.mp4")


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
    with pytest.raises(ValueError, match="exactly one of inline data, a URL, or a handle"):
        DocumentContentPart(data=_PDF_BASE64, url="https://example.com/a.pdf")
    with pytest.raises(ValueError, match="exactly one of inline data, a URL, or a handle"):
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


_OPENAI_HANDLE = MediaHandle(provider="openai", reference="file-abc123")
"""One OpenAI Files handle."""


@pytest.mark.parametrize(
    ("provider", "reference"),
    [
        ("openai", "file-abc123"),
        ("anthropic", "file_011CNha8iCJcU1wXNR6q4V8w"),
        ("gemini", f"{GEMINI_FILE_URI_PREFIX}abc-123"),
        ("vertex", "gs://my-bucket/path/to/object.png"),
        ("bedrock", "s3://my-bucket/path/to/object.png"),
    ],
)
def test_each_provider_handle_shape_is_accepted(provider: str, reference: str) -> None:
    """Every provider's documented handle form validates under its own provider."""
    handle = MediaHandle.model_validate({"provider": provider, "reference": reference})
    assert handle.reference == reference


@pytest.mark.parametrize(
    ("provider", "reference"),
    [
        ("openai", "file_abc123"),
        ("openai", "abc123"),
        ("anthropic", "file-abc123"),
        ("gemini", "https://example.com/files/abc"),
        ("gemini", f"{GEMINI_FILE_URI_PREFIX}Has/Slash"),
        ("vertex", "s3://bucket/object"),
        ("vertex", "gs://bucket"),
        ("bedrock", "gs://bucket/object"),
        ("bedrock", "s3://bucket"),
    ],
)
def test_a_handle_in_another_providers_shape_is_rejected(provider: str, reference: str) -> None:
    """A reference that does not match the named provider's form fails closed."""
    with pytest.raises(ValueError, match="handle looks like"):
        MediaHandle.model_validate({"provider": provider, "reference": reference})


def test_bucket_owner_is_bedrock_only_and_must_be_an_account_id() -> None:
    """Only an s3 handle carries a bucket owner, and only a 12 digit account id."""
    owned = MediaHandle(provider="bedrock", reference="s3://bkt/k", bucket_owner="123456789012")
    assert owned.bucket_owner == "123456789012"
    with pytest.raises(ValueError, match="12 digit"):
        MediaHandle(provider="bedrock", reference="s3://bkt/k", bucket_owner="12345")
    with pytest.raises(ValueError, match="only to bedrock"):
        MediaHandle(provider="vertex", reference="gs://bkt/k", bucket_owner="123456789012")


def test_a_handle_is_a_third_exclusive_carrier() -> None:
    """A handle never rides beside inline bytes or a URL on any media part."""
    assert ImageContentPart(handle=_OPENAI_HANDLE).handle == _OPENAI_HANDLE
    with pytest.raises(ValueError, match="exactly one of inline data, a URL, or a handle"):
        ImageContentPart(media_type="image/png", data=_PNG_BASE64, handle=_OPENAI_HANDLE)
    with pytest.raises(ValueError, match="exactly one of inline data, a URL, or a handle"):
        VideoContentPart(url="https://example.com/a.mp4", handle=_OPENAI_HANDLE)
    with pytest.raises(ValueError, match="exactly one of inline data, a URL, or a handle"):
        DocumentContentPart(data="JVBERi0=", handle=_OPENAI_HANDLE)


def test_bucket_handles_need_a_media_type_for_images_and_videos() -> None:
    """Vertex ``mime_type`` and Bedrock ``format`` come from the media type."""
    s3 = MediaHandle(provider="bedrock", reference="s3://bucket/clip")
    gs = MediaHandle(provider="vertex", reference="gs://bucket/frame")
    with pytest.raises(ValueError, match="bedrock video handle needs its media type"):
        VideoContentPart(handle=s3)
    with pytest.raises(ValueError, match="vertex image handle needs its media type"):
        ImageContentPart(handle=gs)
    assert VideoContentPart(handle=s3, media_type="video/mp4").media_type == "video/mp4"
    assert ImageContentPart(handle=gs, media_type="image/png").media_type == "image/png"
    assert DocumentContentPart(handle=s3).media_type == "application/pdf"


def test_openai_and_gemini_handles_need_no_media_type() -> None:
    """OpenAI, Anthropic, and Gemini Files carry their own MIME type."""
    assert ImageContentPart(handle=_OPENAI_HANDLE).media_type is None
    gemini = MediaHandle(provider="gemini", reference=f"{GEMINI_FILE_URI_PREFIX}xyz")
    assert VideoContentPart(handle=gemini).media_type is None


def test_provider_uris_in_a_url_field_become_handles() -> None:
    """``s3://``, ``gs://``, and Gemini Files URIs are handles, not fetch URLs."""
    image = image_part_from_url("s3://bucket/photo.JPG")
    assert image.handle == MediaHandle(provider="bedrock", reference="s3://bucket/photo.JPG")
    assert image.media_type == "image/jpeg" and image.url is None
    video = video_part_from_url("gs://bucket/clip.mov")
    assert video.handle is not None and video.handle.provider == "vertex"
    assert video.media_type == "video/quicktime"
    gemini = video_part_from_url(f"{GEMINI_FILE_URI_PREFIX}abc")
    assert gemini.handle is not None and gemini.handle.provider == "gemini"
    assert media_handle_from_uri("https://example.com/a.png") is None


def test_a_bucket_uri_without_a_known_suffix_is_refused() -> None:
    """Bedrock cannot derive ``format`` from an extensionless key."""
    with pytest.raises(ValueError, match="needs its media type"):
        image_part_from_url("s3://bucket/photo")
    with pytest.raises(ValueError, match="needs its media type"):
        video_part_from_url("gs://bucket/clip")
