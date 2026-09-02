"""Tests for per-dialect image encoding."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import (
    GEMINI_FILE_URI_PREFIX,
    ImageContentPart,
    MediaHandle,
    image_part_from_url,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.images import (
    anthropic_image_block,
    bedrock_image_block,
    gemini_image_part,
    openai_chat_image_part,
    responses_image_part,
)

_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
"""One valid single-pixel PNG, base64 encoded."""

_INLINE = ImageContentPart(media_type="image/png", data=_PNG_BASE64, detail="high")
"""One inline image with a caller-declared detail hint."""

_REMOTE = image_part_from_url("https://example.com/cat.jpeg")
"""One image the provider would have to fetch itself."""


def test_openai_chat_inlines_as_a_data_url() -> None:
    """Chat Completions carries inline bytes in the ``image_url`` field."""
    assert openai_chat_image_part(_INLINE) == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{_PNG_BASE64}", "detail": "high"},
    }


def test_openai_chat_forwards_a_remote_url_verbatim() -> None:
    """A URL image rides the Chat wire unchanged, with no detail invented."""
    assert openai_chat_image_part(_REMOTE) == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/cat.jpeg"},
    }


def test_responses_uses_input_image() -> None:
    """The Responses wire names the part ``input_image`` with a flat URL."""
    assert responses_image_part(_INLINE) == {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{_PNG_BASE64}",
        "detail": "high",
    }


def test_anthropic_splits_base64_and_url_sources() -> None:
    """Anthropic declares the carrier and the media type on the source."""
    assert anthropic_image_block(_INLINE) == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _PNG_BASE64},
    }
    assert anthropic_image_block(_REMOTE) == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/cat.jpeg"},
    }


def test_anthropic_re_emits_a_cache_marker_placed_on_the_image() -> None:
    """A breakpoint on the image survives to the wire that honors it."""
    marked = _INLINE.model_copy(update={"cache_control": {"type": "ephemeral"}})
    assert anthropic_image_block(marked) == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _PNG_BASE64},
        "cache_control": {"type": "ephemeral"},
    }


def test_gemini_uses_inline_data() -> None:
    """Gemini carries the bytes as an ``inline_data`` part."""
    assert gemini_image_part(_INLINE) == {
        "inline_data": {"mime_type": "image/png", "data": _PNG_BASE64}
    }


def test_bedrock_uses_a_format_and_byte_source() -> None:
    """Bedrock Converse names the bare format and carries encoded bytes."""
    assert bedrock_image_block(_INLINE) == {
        "image": {"format": "png", "source": {"bytes": _PNG_BASE64}}
    }


@pytest.mark.parametrize("encode", [gemini_image_part, bedrock_image_block])
def test_inline_only_wires_reject_a_remote_url_as_a_capability(
    encode: Callable[[ImageContentPart], JsonObject],
) -> None:
    """A wire without a URL carrier fails as a capability, so a waterfall can narrow."""
    with pytest.raises(ProviderCapabilityError) as error:
        encode(_REMOTE)
    assert error.value.capability == "image_url_input"


_OPENAI_HANDLE = ImageContentPart(
    handle=MediaHandle(provider="openai", reference="file-abc"), detail="low"
)
_ANTHROPIC_HANDLE = ImageContentPart(handle=MediaHandle(provider="anthropic", reference="file_abc"))
_GEMINI_HANDLE = ImageContentPart(
    handle=MediaHandle(provider="gemini", reference=f"{GEMINI_FILE_URI_PREFIX}abc")
)
_VERTEX_HANDLE = ImageContentPart(
    handle=MediaHandle(provider="vertex", reference="gs://bkt/cat.png"), media_type="image/png"
)
_BEDROCK_HANDLE = ImageContentPart(
    handle=MediaHandle(
        provider="bedrock", reference="s3://bkt/cat.jpg", bucket_owner="123456789012"
    ),
    media_type="image/jpeg",
)


def test_responses_carries_an_openai_handle_as_file_id() -> None:
    """An OpenAI Files handle rides ``file_id`` with the detail hint preserved."""
    assert responses_image_part(_OPENAI_HANDLE) == {
        "type": "input_image",
        "file_id": "file-abc",
        "detail": "low",
    }


def test_openai_chat_defines_no_image_handle() -> None:
    """Chat ``image_url`` has no ``file_id``, so a handle is a capability refusal."""
    with pytest.raises(ProviderCapabilityError, match="media_handle_input"):
        openai_chat_image_part(_OPENAI_HANDLE)


def test_anthropic_carries_a_files_api_handle_as_a_file_source() -> None:
    """An Anthropic Files handle becomes ``source.type: file``."""
    assert anthropic_image_block(_ANTHROPIC_HANDLE) == {
        "type": "image",
        "source": {"type": "file", "file_id": "file_abc"},
    }


def test_gemini_carries_files_and_gcs_handles_as_file_data() -> None:
    """Gemini Files URIs ride bare; a ``gs://`` object carries its MIME type."""
    assert gemini_image_part(_GEMINI_HANDLE) == {
        "file_data": {"file_uri": f"{GEMINI_FILE_URI_PREFIX}abc"}
    }
    assert gemini_image_part(_VERTEX_HANDLE) == {
        "file_data": {"file_uri": "gs://bkt/cat.png", "mime_type": "image/png"}
    }


def test_bedrock_carries_an_s3_handle_with_its_owner() -> None:
    """A Bedrock handle becomes ``s3Location`` beside the derived format."""
    assert bedrock_image_block(_BEDROCK_HANDLE) == {
        "image": {
            "format": "jpeg",
            "source": {"s3Location": {"uri": "s3://bkt/cat.jpg", "bucketOwner": "123456789012"}},
        }
    }


@pytest.mark.parametrize(
    ("encode", "image"),
    [
        (responses_image_part, _ANTHROPIC_HANDLE),
        (anthropic_image_block, _OPENAI_HANDLE),
        (gemini_image_part, _BEDROCK_HANDLE),
        (bedrock_image_block, _VERTEX_HANDLE),
    ],
)
def test_a_handle_from_another_provider_never_reaches_a_wire(
    encode: Callable[[ImageContentPart], JsonObject], image: ImageContentPart
) -> None:
    """Every encoder refuses a foreign handle as a provider capability error."""
    with pytest.raises(ProviderCapabilityError, match="media_handle_provider") as error:
        encode(image)
    assert error.value.detail is not None
    assert "uploaded to" in error.value.detail
