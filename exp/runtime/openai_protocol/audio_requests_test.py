"""Tests for the speech and transcription public-wire decoders."""

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.audio_requests import decode_speech, decode_transcription
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

_FILE = {"bytes": 64_000, "filename": "clip.wav", "sha256": "b" * 64, "duration_milli": 2_000}


def test_speech_decodes_the_official_body_into_the_canonical_request() -> None:
    """Every supported field reaches the canonical request; the alias is the model."""
    decoded = decode_speech(
        {
            "model": "tts-1",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "wav",
            "speed": 1.25,
            "instructions": "calm",
        }
    )
    assert decoded.alias == "tts-1"
    assert (decoded.request.voice, decoded.request.response_format) == ("alloy", "wav")
    assert (decoded.request.speed, decoded.request.instructions) == (1.25, "calm")


@pytest.mark.parametrize(
    ("body", "param"),
    [
        (
            {"model": "tts-1", "input": "hi", "voice": "alloy", "stream_format": "sse"},
            "stream_format",
        ),
        ({"model": "tts-1", "input": "hi", "voice": "alloy", "speed": 9}, "speed"),
        ({"model": "tts-1", "input": "hi"}, "voice"),
    ],
)
def test_speech_refuses_streaming_out_of_range_or_missing_fields(
    body: JsonObject, param: str
) -> None:
    """Each refusal is a 400 naming the field the caller must fix."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_speech(body)
    assert raised.value.status_code == 400
    assert raised.value.detail.param == param


def test_transcription_decodes_the_parsed_upload_and_its_audio_facts() -> None:
    """The data plane's parsed upload becomes the canonical request, digest included."""
    decoded = decode_transcription(
        {
            "model": "whisper-1",
            "file": _FILE,
            "language": "en",
            "temperature": 0.2,
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        }
    )
    request = decoded.request
    assert decoded.alias == "whisper-1"
    assert (request.audio_bytes, request.audio_sha256) == (64_000, "b" * 64)
    assert (request.audio_duration_milli, request.timestamp_granularities) == (2_000, ("word",))


def test_speech_too_large_for_the_output_buffer_is_refused_before_dispatch() -> None:
    """A long, slow, uncompressed request is refused on input instead of overflowing later."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_speech(
            {
                "model": "tts-1",
                "input": "x" * 4_096,
                "voice": "alloy",
                "response_format": "pcm",
                "speed": 0.25,
            }
        )
    assert raised.value.detail.param == "input"
    assert "Split the text" in raised.value.detail.message


def test_speech_accepts_a_custom_voice_reference() -> None:
    """OpenAI's ``voice: {"id": ...}`` form decodes to a canonical voice reference."""
    decoded = decode_speech({"model": "gpt-4o-mini-tts", "input": "hi", "voice": {"id": "voice_1"}})
    assert not isinstance(decoded.request.voice, str)
    assert decoded.request.voice.id == "voice_1"


def test_timestamps_without_verbose_json_are_refused_locally() -> None:
    """The provider only timestamps verbose_json, so other formats fail before upload."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_transcription(
            {"model": "whisper-1", "file": _FILE, "timestamp_granularities": ["word"]}
        )
    assert raised.value.detail.param == "timestamp_granularities"


@pytest.mark.parametrize("granularities", [["word", "word"], ["word", "segment", "word"]])
def test_timestamp_granularities_are_bounded_and_distinct(granularities: list[str]) -> None:
    """A flood of repeated granularity parts is refused on that field."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_transcription(
            {
                "model": "whisper-1",
                "file": _FILE,
                "response_format": "verbose_json",
                "timestamp_granularities": granularities,
            }
        )
    assert raised.value.detail.param == "timestamp_granularities"


@pytest.mark.parametrize("field", ["stream", "chunking_strategy", "include", "known_speaker_names"])
def test_transcription_refuses_unsupported_fields_by_name(field: str) -> None:
    """Streaming, server chunking, logprob includes, and diarization are refused, never dropped."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_transcription({"model": "whisper-1", "file": _FILE, field: True})
    assert raised.value.detail.param == field


@pytest.mark.parametrize("temperature", ["NaN", "inf", "-Infinity", "1e999", "warm"])
def test_non_finite_multipart_temperature_is_refused_on_its_field(temperature: str) -> None:
    """A form temperature the data plane kept as text never becomes the provider default."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_transcription({"model": "whisper-1", "file": _FILE, "temperature": temperature})
    assert raised.value.detail.param == "temperature"
