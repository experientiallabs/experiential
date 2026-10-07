"""Canonical speech-synthesis and transcription request contracts.

Parallel to the chat ``GatewayRequest`` like the embeddings and images
contracts: never streamed, never keyed for replay. A speech request carries
its whole text. A transcription request carries only the facts of its audio
upload (size, name, type, digest, and the duration the data plane measures by
summing codec frame lengths, never from container timing the uploader controls):
the audio bytes stay in the native data plane, which rebuilds the provider's
multipart body itself, so a multi-megabyte upload never crosses the bridge.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.ledger_valuation import require_representable_nano_usd

MAXIMUM_SPEECH_INPUT_CHARACTERS = 4_096
"""The OpenAI speech wire's input cap; every speech rung today speaks that wire."""
MAXIMUM_TRANSCRIPTION_AUDIO_BYTES = 25 * 1024 * 1024
"""The OpenAI transcription upload cap, which every OpenAI-wire provider shares."""

MAXIMUM_TRANSCRIPTION_AUDIO_MILLI = 4 * 60 * 60 * 1000
"""The longest transcription admitted (4 hours), so a crafted container timeline
can never size a reservation past it. The data plane refuses longer uploads."""

SPEECH_PADDING_SECONDS = 5
"""Audio a speech model adds beyond the reading itself (leading and trailing
silence, a short input's minimum length), added to every duration bound."""

AUDIO_INPUT_TOKEN_SLACK = 64
"""Input tokens beyond the gateway's own estimate that an audio request's
reservation (and its settle ceiling) covers: a provider tokenizer and its
framing tokens can count slightly above the gateway's count."""

MAXIMUM_SPEECH_OUTPUT_BYTES = 64 * 1024 * 1024
"""The data plane's retained-output cap; a speech answer above it is refused."""

SPEECH_MINIMUM_CHARACTERS_PER_SECOND = 4
"""Language-independent slow-speech floor at speed 1.0: slow English reads about
10 characters per second and Mandarin about 4 to 5, so no voice reads fewer.
Duration bounds divide the input length by this floor times the requested
speed, never assuming a faster script."""

SPEECH_BYTES_PER_SECOND: dict[str, int] = {
    "pcm": 48_000,
    "wav": 48_000,
    "flac": 52_000,
    "mp3": 24_000,
    "opus": 24_000,
    "aac": 24_000,
}
"""Upper bound on audio bytes per second for each speech format: 24 kHz 16-bit
mono uncompressed; FLAC at that rate plus frame overhead (lossless audio is
not guaranteed to compress); generous constant bitrates for the lossy ones."""

SSE_INFLATION_NUMERATOR, SSE_INFLATION_DENOMINATOR = 4, 3
"""Base64 inflation of audio carried in a token lane's SSE events."""

SSE_ENVELOPE_ALLOWANCE_BYTES = 1024 * 1024
"""Headroom for the SSE event framing around the base64 audio (``data:`` lines,
event JSON, per-chunk padding), a small fraction of a 64 MiB answer."""

TRANSCRIPTION_AUDIO_TOKENS_PER_SECOND = 25
"""Reservation bound on audio input tokens per second for token-priced
transcription models (gpt-4o-transcribe meters about 17 per second)."""


class SpeechVoiceReference(ContractModel):
    """A custom voice named by id (OpenAI's ``voice: {"id": ...}`` form).

    Attributes:
        id: The provider's custom voice id.
    """

    id: str = Field(min_length=1, max_length=128)


class SpeechRequest(ContractModel):
    """Canonical, provider-neutral text-to-speech request (OpenAI ``/audio/speech``).

    Attributes:
        surface: Always the speech surface.
        input: The text to speak, 1 to 4,096 characters; billed per character
            on a per-character lane.
        voice: A built-in voice name or a custom voice reference.
        response_format: Audio format of the answer; the provider default
            (``mp3``) when None.
        speed: Playback speed from 0.25 to 4.0; the provider default when None.
        instructions: Delivery instructions for instruction-following voices.
    """

    surface: Literal[GatewayApiSurface.SPEECH] = GatewayApiSurface.SPEECH
    input: str = Field(min_length=1, max_length=MAXIMUM_SPEECH_INPUT_CHARACTERS)
    voice: str | SpeechVoiceReference
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] | None = None
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    instructions: str | None = Field(default=None, max_length=4_096)

    @property
    def attribution_label(self) -> str | None:
        """Speech carries no end-user attribution field."""
        return None

    @property
    def input_characters(self) -> int:
        """Billed characters: the provider meters the input text by code point."""
        return len(self.input)

    @property
    def maximum_audio_seconds(self) -> float:
        """Seconds the slowest plausible reading of the input lasts at the requested speed."""
        speed = self.speed if self.speed is not None else 1.0
        reading = self.input_characters / (SPEECH_MINIMUM_CHARACTERS_PER_SECOND * speed)
        return reading + SPEECH_PADDING_SECONDS

    @property
    def worst_case_output_bytes(self) -> int:
        """Bytes the data plane may have to retain for this request's audio.

        The slowest plausible reading of the input at the requested speed, in
        the requested format's largest byte rate, inflated by the base64
        framing a token lane's SSE answer carries.
        """
        rate = SPEECH_BYTES_PER_SECOND[self.response_format or "mp3"]
        encoded = self.maximum_audio_seconds * rate * SSE_INFLATION_NUMERATOR
        return int(encoded / SSE_INFLATION_DENOMINATOR) + SSE_ENVELOPE_ALLOWANCE_BYTES


class TranscriptionRequest(ContractModel):
    """Canonical, provider-neutral speech-to-text request (OpenAI ``/audio/transcriptions``).

    ``audio_sha256`` binds the canonical request digest to the uploaded bytes,
    so two different recordings with the same parameters never share one.
    ``audio_duration_milli`` is measured by the data plane as the sum of each
    coded frame's codec-defined length (Opus TOC bytes, AAC frame counts, MP3
    and FLAC frame headers, PCM sample counts), never from container timestamps
    or headers, which the uploader controls. It sizes the reservation;
    settlement bills the provider's own metered seconds or tokens.

    Attributes:
        surface: Always the transcription surface.
        audio_bytes: Size of the uploaded audio, 1 byte to 25 MB.
        audio_filename: The upload's filename, forwarded to the provider.
        audio_content_type: The upload's content type, when it named one.
        audio_sha256: Lowercase hex SHA-256 of the audio bytes.
        audio_duration_milli: Measured duration in milliseconds, at most 4 hours.
        language: ISO-639-1 language of the audio, when the caller knows it.
        prompt: Optional text guiding the transcript's style or vocabulary.
        response_format: ``json``, ``verbose_json``, or ``text``; ``json`` when None.
        temperature: Sampling temperature from 0 to 1; the provider default when None.
        timestamp_granularities: ``word`` and/or ``segment`` timestamps for
            ``verbose_json``.
    """

    surface: Literal[GatewayApiSurface.TRANSCRIPTION] = GatewayApiSurface.TRANSCRIPTION
    audio_bytes: int = Field(ge=1, le=MAXIMUM_TRANSCRIPTION_AUDIO_BYTES)
    audio_filename: str = Field(min_length=1, max_length=256)
    audio_content_type: str | None = Field(default=None, max_length=128)
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audio_duration_milli: int = Field(ge=1, le=MAXIMUM_TRANSCRIPTION_AUDIO_MILLI)
    language: str | None = Field(default=None, min_length=2, max_length=16)
    prompt: str | None = Field(default=None, max_length=4_096)
    response_format: Literal["json", "verbose_json", "text"] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    timestamp_granularities: tuple[Literal["word", "segment"], ...] = ()

    @property
    def attribution_label(self) -> str | None:
        """Transcription carries no end-user attribution field."""
        return None

    @property
    def maximum_audio_seconds(self) -> int:
        """Whole seconds that bound the measured duration (plus one), for the reservation."""
        return (self.audio_duration_milli + 999) // 1000 + 1


def speech_ceiling_nano_usd(
    request: SpeechRequest,
    *,
    unit_rate: int | None,
    input_tokens: int,
    input_rate: int | None,
    output_rate: int | None,
    maximum_output_tokens: int | None,
) -> int | None:
    """Return the conservative reservation ceiling for one speech call.

    A per-character lane (``tts-1``) reserves exactly what it will bill: every
    input character at the unit rate. A token-priced lane (``gpt-4o-mini-tts``)
    reserves its byte-bounded input tokens plus the lane's provider-enforced
    output ceiling: generated audio's length is not bounded by the text (a
    delivery instruction can add pauses no reading rate predicts), so a token
    lane without a declared ceiling is unpriced, as is a lane with neither price.

    Args:
        request: Canonical speech request.
        unit_rate: Nano-USD per character from the lane's unit card, or None.
        input_tokens: The text's estimated input-token reservation.
        input_rate: Nano-USD per million input tokens, or None.
        output_rate: Nano-USD per million audio output tokens, or None.
        maximum_output_tokens: The lane's provider-enforced output ceiling, or None.

    Returns:
        Nano-USD ceiling, or None when the lane is unpriced.
    """
    if unit_rate is not None:
        return require_representable_nano_usd(
            request.input_characters * unit_rate, what="speech reservation ceiling"
        )
    if input_rate is None or output_rate is None or maximum_output_tokens is None:
        return None
    total = input_tokens * input_rate + maximum_output_tokens * output_rate
    return require_representable_nano_usd(
        (total + 999_999) // 1_000_000, what="speech reservation ceiling"
    )


def transcription_ceiling_nano_usd(
    request: TranscriptionRequest,
    *,
    unit_rate: int | None,
    input_tokens: int,
    input_rate: int | None,
    output_rate: int | None,
    maximum_output_tokens: int | None,
) -> int | None:
    """Return the conservative reservation ceiling for one transcription call.

    A per-second lane (``whisper-1``) reserves the bounded duration at the unit
    rate. A token-priced lane (``gpt-4o-transcribe``) reserves its input tokens
    (the prompt plus the per-second audio bound, as :func:`worst_case_input_tokens`
    counts them) and the lane's provider-enforced output ceiling: transcript
    length is not bounded by the recording (a model can emit text no speaking
    rate predicts), so a token lane without a declared ceiling is unpriced.

    Args:
        request: Canonical transcription request.
        unit_rate: Nano-USD per audio second from the lane's unit card, or None.
        input_tokens: The request's input-token reservation (prompt plus audio).
        input_rate: Nano-USD per million audio input tokens, or None.
        output_rate: Nano-USD per million transcript tokens, or None.
        maximum_output_tokens: The lane's provider-enforced output ceiling, or None.

    Returns:
        Nano-USD ceiling, or None when the lane is unpriced.
    """
    if unit_rate is not None:
        return require_representable_nano_usd(
            request.maximum_audio_seconds * unit_rate, what="transcription reservation ceiling"
        )
    if input_rate is None or output_rate is None or maximum_output_tokens is None:
        return None
    total = input_tokens * input_rate + maximum_output_tokens * output_rate
    return require_representable_nano_usd(
        (total + 999_999) // 1_000_000, what="transcription reservation ceiling"
    )
