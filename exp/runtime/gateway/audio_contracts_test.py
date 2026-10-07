"""Tests for the speech and transcription contracts and their reservation ceilings."""

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.audio_contracts import (
    SPEECH_AUDIO_TOKENS_PER_SECOND,
    SPEECH_MINIMUM_CHARACTERS_PER_SECOND,
    SpeechRequest,
    TranscriptionRequest,
    speech_ceiling_nano_usd,
    speech_output_token_bound,
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


def test_speech_reserves_exactly_its_billed_characters_on_a_unit_lane() -> None:
    """A per-character lane reserves what it will bill; a token lane bounds the audio."""
    request = SpeechRequest(input="Hello there", voice="alloy")
    assert (
        speech_ceiling_nano_usd(
            request, unit_rate=15_000, input_tokens=4, input_rate=None, output_rate=None
        )
        == 11 * 15_000
    )
    token_ceiling = speech_ceiling_nano_usd(
        request, unit_rate=None, input_tokens=4, input_rate=1_000_000, output_rate=2_000_000
    )
    output_tokens = speech_output_token_bound(request)
    expected = 4 * 1_000_000 + output_tokens * 2_000_000 + 999_999
    assert token_ceiling == expected // 1_000_000
    # 11 characters at the 4/s floor last 2.75 s, plus 5 s padding; 25 tokens/s, plus one.
    assert output_tokens == int((11 / SPEECH_MINIMUM_CHARACTERS_PER_SECOND + 5) * 25) + 1
    assert (
        speech_ceiling_nano_usd(
            request, unit_rate=None, input_tokens=4, input_rate=1, output_rate=None
        )
        is None
    )


def test_speech_token_bound_scales_with_slowness_not_with_the_script() -> None:
    """The bound follows the slowest plausible reading at the requested speed."""
    text = "\u4e2d" * 400  # 400 CJK characters, read about as slowly as the floor.
    normal = speech_output_token_bound(SpeechRequest(input=text, voice="v"))
    slow = speech_output_token_bound(SpeechRequest(input=text, voice="v", speed=0.25))
    reading = 400 / SPEECH_MINIMUM_CHARACTERS_PER_SECOND
    assert normal == int((reading + 5) * SPEECH_AUDIO_TOKENS_PER_SECOND) + 1
    assert slow == int((4 * reading + 5) * SPEECH_AUDIO_TOKENS_PER_SECOND) + 1


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
