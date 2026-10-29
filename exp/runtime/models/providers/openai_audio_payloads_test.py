"""Tests for the OpenAI-wire speech body and transcription upload fields."""

from exp.runtime.gateway.audio_contracts import (
    SpeechRequest,
    SpeechVoiceReference,
    TranscriptionRequest,
)
from exp.runtime.models.providers.openai_audio_payloads import (
    openai_speech_request,
    openai_transcription_fields,
)


def test_speech_body_omits_absent_controls_and_meters_only_on_request() -> None:
    """Absent controls take the provider default; SSE is asked for only on token lanes."""
    request = SpeechRequest(input="Hi", voice="coral", speed=1.5)
    assert openai_speech_request("tts-1", request, metered_stream=False) == {
        "model": "tts-1",
        "input": "Hi",
        "voice": "coral",
        "speed": 1.5,
    }
    assert (
        openai_speech_request("gpt-4o-mini-tts", request, metered_stream=True)["stream_format"]
        == "sse"
    )


def test_custom_voice_reference_is_forwarded_as_an_id_object() -> None:
    """A custom voice reaches the provider in its documented ``{"id": ...}`` form."""
    request = SpeechRequest(input="Hi", voice=SpeechVoiceReference(id="voice_1"))
    body = openai_speech_request("gpt-4o-mini-tts", request, metered_stream=True)
    assert body["voice"] == {"id": "voice_1"}


def test_transcription_fields_serve_text_from_json_and_repeat_list_fields() -> None:
    """``text`` is requested as ``json`` (it carries the meter); granularities repeat."""
    request = TranscriptionRequest(
        audio_bytes=10,
        audio_filename="a.wav",
        audio_sha256="c" * 64,
        audio_duration_milli=1_000,
        response_format="text",
        temperature=0.0,
        timestamp_granularities=("word", "segment"),
    )
    assert openai_transcription_fields("whisper-1", request) == {
        "model": "whisper-1",
        "response_format": "json",
        "temperature": "0.0",
        "timestamp_granularities[]": ["word", "segment"],
    }
    verbose = request.model_copy(update={"response_format": "verbose_json"})
    assert openai_transcription_fields("whisper-1", verbose)["response_format"] == "verbose_json"
