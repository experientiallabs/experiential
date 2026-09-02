"""Tests for per-dialect audio encoding."""

from __future__ import annotations

import pytest

from exp.common.models.content import AudioContentPart
from exp.runtime.models.providers.audios import (
    AUDIO_DIALECTS,
    gemini_audio_part,
    openai_chat_audio_part,
    reject_audio_part,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError

_WAV_BASE64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA="
"""A 44-byte WAV header with an empty data chunk, base64 encoded."""

_WAV = AudioContentPart(media_type="audio/wav", data=_WAV_BASE64)
_MP3 = AudioContentPart(media_type="audio/mpeg", data="SUQzBAAAAAAAAA==")


def test_openai_chat_carries_input_audio_with_its_format_name() -> None:
    """The Chat wire needs the ``format`` name the media type maps to."""
    assert openai_chat_audio_part(_WAV) == {
        "type": "input_audio",
        "input_audio": {"data": _WAV_BASE64, "format": "wav"},
    }
    assert openai_chat_audio_part(_MP3) == {
        "type": "input_audio",
        "input_audio": {"data": "SUQzBAAAAAAAAA==", "format": "mp3"},
    }


def test_gemini_carries_inline_data_with_the_mime_type() -> None:
    """Gemini carries bytes as ``inline_data`` under the registered MIME type."""
    assert gemini_audio_part(_WAV) == {
        "inline_data": {"mime_type": "audio/wav", "data": _WAV_BASE64}
    }
    assert gemini_audio_part(_MP3) == {
        "inline_data": {"mime_type": "audio/mpeg", "data": "SUQzBAAAAAAAAA=="}
    }


def test_wires_without_an_audio_carrier_reject_every_clip() -> None:
    """A wire with no servable audio part refuses the clip instead of dropping it."""
    with pytest.raises(ProviderCapabilityError, match="audio_input"):
        reject_audio_part(_WAV)


def test_audio_dialects_name_only_the_wires_a_model_serves() -> None:
    """Responses, Anthropic, and Bedrock Converse are outside the audio dialects."""
    assert AUDIO_DIALECTS == {"openai_compatible", "gemini_generate_content"}
    assert "openai_responses" not in AUDIO_DIALECTS
    assert "anthropic_messages" not in AUDIO_DIALECTS
    assert "bedrock_converse_stream" not in AUDIO_DIALECTS
