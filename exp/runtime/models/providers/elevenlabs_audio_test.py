"""Tests for translating OpenAI audio requests onto ElevenLabs' wire."""

import pytest

from exp.runtime.gateway.audio_contracts import SpeechRequest, TranscriptionRequest
from exp.runtime.models.providers.elevenlabs_audio import (
    ELEVENLABS_OPENAI_VOICES,
    elevenlabs_speech_target,
    elevenlabs_transcription_fields,
    elevenlabs_voice_id,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

_ROOT = "https://api.elevenlabs.io/v1/text-to-speech"
_SHA = "a" * 64


def _upload(**overrides: object) -> TranscriptionRequest:
    """Build one valid transcription request with field overrides."""
    fields: dict[str, object] = {
        "audio_bytes": 64_000,
        "audio_filename": "clip.wav",
        "audio_sha256": _SHA,
        "audio_duration_milli": 2_000,
    }
    fields.update(overrides)
    return TranscriptionRequest.model_validate(fields)


def test_openai_voice_names_map_and_voice_ids_pass_through() -> None:
    """The six classic names resolve; a twenty-character id is used as given."""
    assert set(ELEVENLABS_OPENAI_VOICES) == {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
    assert all(len(voice_id) == 20 for voice_id in ELEVENLABS_OPENAI_VOICES.values())
    assert elevenlabs_voice_id("Nova") == ELEVENLABS_OPENAI_VOICES["nova"]
    assert elevenlabs_voice_id("JBFqnCBsd6RMkjVDRZzb") == "JBFqnCBsd6RMkjVDRZzb"
    assert elevenlabs_voice_id("coral") is None


def test_speech_target_renders_voice_format_and_speed() -> None:
    """The voice rides the path, the format the query, the speed voice_settings."""
    request = SpeechRequest(input="Hello", voice="fable", response_format="wav", speed=0.8)
    target = elevenlabs_speech_target(_ROOT, "eleven_v3", request)
    assert not isinstance(target, OpenAIProtocolError)
    url, body = target
    assert url == f"{_ROOT}/{ELEVENLABS_OPENAI_VOICES['fable']}?output_format=wav_24000"
    assert body == {"text": "Hello", "model_id": "eleven_v3", "voice_settings": {"speed": 0.8}}
    default = elevenlabs_speech_target(_ROOT, "eleven_v3", SpeechRequest(input="Hi", voice="alloy"))
    assert not isinstance(default, OpenAIProtocolError)
    assert default[0].endswith("?output_format=mp3_44100_128")
    assert default[1] == {"text": "Hi", "model_id": "eleven_v3"}


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        ({"voice": "coral"}, "voice"),
        ({"response_format": "flac"}, "response_format"),
        ({"response_format": "aac"}, "response_format"),
        ({"speed": 0.5}, "speed"),
        ({"speed": 1.5}, "speed"),
        ({"instructions": "calm"}, "instructions"),
    ],
)
def test_speech_options_elevenlabs_cannot_carry_are_refused(
    fields: dict[str, object], param: str
) -> None:
    """Each unsupported option names its own field."""
    request = SpeechRequest.model_validate({"input": "Hi", "voice": "alloy", **fields})
    refused = elevenlabs_speech_target(_ROOT, "eleven_v3", request)
    assert isinstance(refused, OpenAIProtocolError)
    assert (refused.status_code, refused.detail.param) == (400, param)


def test_transcription_fields_use_elevenlabs_names() -> None:
    """The upload carries ``model_id`` and ``language_code``, nothing OpenAI-only."""
    assert elevenlabs_transcription_fields("scribe_v2", _upload(language="en")) == {
        "model_id": "scribe_v2",
        "language_code": "en",
    }
    verbose = _upload(response_format="verbose_json", timestamp_granularities=("word",))
    assert elevenlabs_transcription_fields("scribe_v2", verbose) == {"model_id": "scribe_v2"}


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        ({"prompt": "names"}, "prompt"),
        ({"temperature": 0.2}, "temperature"),
        (
            {"response_format": "verbose_json", "timestamp_granularities": ("segment",)},
            "timestamp_granularities",
        ),
    ],
)
def test_transcription_options_elevenlabs_cannot_carry_are_refused(
    fields: dict[str, object], param: str
) -> None:
    """A prompt, temperature, or segment timestamps are refused on their field."""
    refused = elevenlabs_transcription_fields("scribe_v2", _upload(**fields))
    assert isinstance(refused, OpenAIProtocolError)
    assert (refused.status_code, refused.detail.param) == (400, param)
