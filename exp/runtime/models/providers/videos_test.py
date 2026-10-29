"""Tests for per-dialect video encoding."""

from __future__ import annotations

import pytest

from exp.common.models.content import (
    GEMINI_FILE_URI_PREFIX,
    MediaHandle,
    VideoContentPart,
    video_part_from_url,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.videos import (
    VIDEO_DIALECTS,
    VIDEO_URL_DIALECTS,
    bedrock_video_block,
    gemini_video_part,
    openai_chat_video_part,
    reject_video_part,
)

_MP4_BASE64 = "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDE="
"""A base64 prefix of an MP4 ``ftyp`` box, enough for a carrier fixture."""

_INLINE = VideoContentPart(media_type="video/mp4", data=_MP4_BASE64)
"""One inline video with its declared media type."""

_REMOTE = video_part_from_url("https://example.com/clip.webm")
"""One video the provider would have to fetch itself."""


def test_openai_chat_inlines_as_a_data_url() -> None:
    """The OpenAI-compatible wire carries inline bytes in ``video_url``."""
    assert openai_chat_video_part(_INLINE) == {
        "type": "video_url",
        "video_url": {"url": f"data:video/mp4;base64,{_MP4_BASE64}"},
    }


def test_openai_chat_forwards_a_remote_url_verbatim() -> None:
    """A URL video rides the OpenAI-compatible wire unchanged."""
    assert openai_chat_video_part(_REMOTE) == {
        "type": "video_url",
        "video_url": {"url": "https://example.com/clip.webm"},
    }


def test_gemini_splits_inline_data_and_file_data() -> None:
    """Gemini carries bytes as ``inline_data`` and a URI as ``file_data``."""
    assert gemini_video_part(_INLINE) == {
        "inline_data": {"mime_type": "video/mp4", "data": _MP4_BASE64}
    }
    assert gemini_video_part(_REMOTE) == {
        "file_data": {"file_uri": "https://example.com/clip.webm", "mime_type": "video/webm"}
    }
    assert gemini_video_part(VideoContentPart(url="https://example.com/watch?v=abc")) == {
        "file_data": {"file_uri": "https://example.com/watch?v=abc"}
    }


def test_bedrock_declares_the_converse_format_enum() -> None:
    """Bedrock names the container by its enum value, not by MIME type."""
    assert bedrock_video_block(_INLINE) == {
        "video": {"format": "mp4", "source": {"bytes": _MP4_BASE64}}
    }
    quicktime = VideoContentPart(media_type="video/quicktime", data=_MP4_BASE64)
    assert bedrock_video_block(quicktime) == {
        "video": {"format": "mov", "source": {"bytes": _MP4_BASE64}}
    }
    three_gp = VideoContentPart(media_type="video/3gpp", data=_MP4_BASE64)
    assert bedrock_video_block(three_gp) == {
        "video": {"format": "three_gp", "source": {"bytes": _MP4_BASE64}}
    }


def test_bedrock_refuses_a_remote_url() -> None:
    """Converse cannot fetch a caller URL, so the rung declines the video."""
    with pytest.raises(ProviderCapabilityError, match="video_url_input"):
        bedrock_video_block(_REMOTE)


def test_wires_without_a_video_carrier_reject_every_video() -> None:
    """Responses and Anthropic define no video content, so nothing is dropped silently."""
    with pytest.raises(ProviderCapabilityError, match="video_input"):
        reject_video_part(_INLINE)
    with pytest.raises(ProviderCapabilityError, match="video_input"):
        reject_video_part(_REMOTE)


def test_video_dialects_exclude_the_wires_without_a_carrier() -> None:
    """Only the three documented video wires are video dialects."""
    assert "openai_responses" not in VIDEO_DIALECTS
    assert "anthropic_messages" not in VIDEO_DIALECTS
    assert VIDEO_URL_DIALECTS < VIDEO_DIALECTS
    assert "bedrock_converse_stream" not in VIDEO_URL_DIALECTS


_GEMINI_HANDLE = VideoContentPart(
    handle=MediaHandle(provider="gemini", reference=f"{GEMINI_FILE_URI_PREFIX}clip")
)
_VERTEX_HANDLE = VideoContentPart(
    handle=MediaHandle(provider="vertex", reference="gs://bkt/clip.mp4"), media_type="video/mp4"
)
_BEDROCK_HANDLE = VideoContentPart(
    handle=MediaHandle(provider="bedrock", reference="s3://bkt/clip.webm"), media_type="video/webm"
)


def test_gemini_carries_video_handles_as_file_data() -> None:
    """A Gemini Files URI rides bare; a ``gs://`` object carries its MIME type."""
    assert gemini_video_part(_GEMINI_HANDLE) == {
        "file_data": {"file_uri": f"{GEMINI_FILE_URI_PREFIX}clip"}
    }
    assert gemini_video_part(_VERTEX_HANDLE) == {
        "file_data": {"file_uri": "gs://bkt/clip.mp4", "mime_type": "video/mp4"}
    }
    with pytest.raises(ProviderCapabilityError, match="media_handle_provider"):
        gemini_video_part(_BEDROCK_HANDLE)


def test_bedrock_carries_an_s3_video_handle() -> None:
    """A Bedrock handle becomes ``s3Location`` beside the Converse format enum."""
    assert bedrock_video_block(_BEDROCK_HANDLE) == {
        "video": {"format": "webm", "source": {"s3Location": {"uri": "s3://bkt/clip.webm"}}}
    }
    with pytest.raises(ProviderCapabilityError, match="media_handle_provider"):
        bedrock_video_block(_VERTEX_HANDLE)


def test_openai_compatible_chat_defines_no_video_handle() -> None:
    """``video_url`` carries a URL only, so a handle is a capability refusal."""
    with pytest.raises(ProviderCapabilityError, match="media_handle_input"):
        openai_chat_video_part(_GEMINI_HANDLE)
