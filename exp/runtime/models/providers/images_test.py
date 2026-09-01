"""Tests for per-dialect image encoding."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import ImageContentPart, image_part_from_url
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
