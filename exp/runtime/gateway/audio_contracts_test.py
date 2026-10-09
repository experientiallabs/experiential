"""Tests for the speech and transcription contracts and their reservation ceilings."""

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.audio_contracts import (
    SpeechRequest,
    TranscriptionRequest,
    speech_ceiling_nano_usd,
    transcription_ceiling_nano_usd,
)

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


def test_speech_reserves_its_characters_or_its_input_and_declared_output_ceiling() -> None:
    """A per-character lane reserves what it will bill; a token lane its output cap."""
    request = SpeechRequest(input="Hello there", voice="alloy")
    assert (
        speech_ceiling_nano_usd(
            request,
            unit_rate=15_000,
            input_tokens=4,
            input_rate=None,
            output_rate=None,
            maximum_output_tokens=None,
        )
        == 11 * 15_000
    )
    token_ceiling = speech_ceiling_nano_usd(
        request,
        unit_rate=None,
        input_tokens=4,
        input_rate=1_000_000,
        output_rate=2_000_000,
        maximum_output_tokens=2_000,
    )
    assert token_ceiling == (4 * 1_000_000 + 2_000 * 2_000_000 + 999_999) // 1_000_000
    # Generated audio is not bounded by the text: no declared cap, no price.
    for output_rate, maximum in ((2_000_000, None), (None, 2_000)):
        assert (
            speech_ceiling_nano_usd(
                request,
                unit_rate=None,
                input_tokens=4,
                input_rate=1,
                output_rate=output_rate,
                maximum_output_tokens=maximum,
            )
            is None
        )


def test_transcription_reserves_seconds_or_its_input_and_declared_output_ceiling() -> None:
    """Units reserve the measured seconds; tokens reserve input plus the lane's output cap."""
    assert _upload(audio_duration_milli=2_000).maximum_audio_seconds == 3
    assert _upload(audio_duration_milli=2_001).maximum_audio_seconds == 4
    request = _upload(audio_duration_milli=9_000)
    unit_ceiling = transcription_ceiling_nano_usd(
        request,
        unit_rate=100_000,
        input_tokens=257,
        input_rate=None,
        output_rate=None,
        maximum_output_tokens=None,
    )
    assert unit_ceiling == 10 * 100_000
    token_ceiling = transcription_ceiling_nano_usd(
        request,
        unit_rate=None,
        input_tokens=257,
        input_rate=1_000_000,
        output_rate=2_000_000,
        maximum_output_tokens=2_000,
    )
    assert token_ceiling == (257 * 1_000_000 + 2_000 * 2_000_000 + 999_999) // 1_000_000
    undeclared = transcription_ceiling_nano_usd(
        request,
        unit_rate=None,
        input_tokens=257,
        input_rate=1_000_000,
        output_rate=2_000_000,
        maximum_output_tokens=None,
    )
    assert undeclared is None


def test_speech_worst_case_output_grows_with_length_slowness_and_format() -> None:
    """Slower speech and uncompressed formats need more retained bytes."""
    text = "x" * 1_000
    base = SpeechRequest(input=text, voice="v").worst_case_output_bytes
    assert SpeechRequest(input=text, voice="v", speed=0.25).worst_case_output_bytes > 3 * base
    assert (
        SpeechRequest(input=text, voice="v", response_format="pcm").worst_case_output_bytes > base
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"audio_bytes": 0},
        {"audio_bytes": 25 * 1024 * 1024 + 1},
        {"audio_sha256": "not-a-digest"},
        {"temperature": 1.5},
        {"audio_duration_milli": 0},
        {"audio_duration_milli": 4 * 60 * 60 * 1000 + 1},
        {"response_format": "srt"},
    ],
)
def test_transcription_contract_refuses_out_of_range_uploads(overrides: dict[str, object]) -> None:
    """Empty or oversized audio, a malformed digest, or an unsupported format never admit."""
    with pytest.raises(ValidationError):
        _upload(**overrides)


def test_audio_requests_carry_no_attribution_label() -> None:
    """Neither OpenAI audio wire has a ``user`` field, so neither surface attributes one."""
    assert SpeechRequest(input="x", voice="v").attribution_label is None
    assert _upload().attribution_label is None
