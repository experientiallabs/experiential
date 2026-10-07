"""Decode the OpenAI audio bodies into the canonical speech and transcription surfaces.

Speech arrives as JSON and decodes like the images surface: one manifest, one
official-SDK shape check, and one closed wire model. Transcription arrives as
a multipart upload the native data plane parses itself; it hands this module
the upload's text fields plus the facts of the audio part (size, name, type,
digest, measured duration), never the audio bytes.
"""

from __future__ import annotations

from typing import Literal

from openai.types.audio.speech_create_params import SpeechCreateParams
from pydantic import Field, TypeAdapter, ValidationError, field_validator

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.gateway.audio_contracts import (
    MAXIMUM_SPEECH_INPUT_CHARACTERS,
    MAXIMUM_SPEECH_OUTPUT_BYTES,
    MAXIMUM_TRANSCRIPTION_AUDIO_BYTES,
    MAXIMUM_TRANSCRIPTION_AUDIO_MILLI,
    SpeechRequest,
    SpeechVoiceReference,
    TranscriptionRequest,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.manifest import (
    SPEECH_MANIFEST,
    TRANSCRIPTION_MANIFEST,
    validate_manifest,
)
from exp.runtime.openai_protocol.requests import _validate_official, _validate_wire
from exp.runtime.openai_protocol.validation_errors import validation_protocol_error
from exp.runtime.openai_protocol.wire_models import _WireModel


class _VoiceReference(_WireModel):
    """Closed custom-voice reference (OpenAI ``voice: {"id": ...}``).

    Attributes:
        id: The provider's custom voice id.
    """

    id: str = Field(min_length=1, max_length=128)


class _SpeechRequest(_WireModel):
    """Closed gateway speech request profile (OpenAI ``/audio/speech``).

    Attributes:
        model: The public alias to route.
        input: The text to speak.
        voice: A built-in voice name or a custom voice reference.
        instructions: Optional delivery instructions.
        response_format: Optional audio format.
        speed: Optional playback speed.
    """

    model: str = Field(min_length=1, max_length=256)
    input: str = Field(min_length=1, max_length=MAXIMUM_SPEECH_INPUT_CHARACTERS)
    voice: str | _VoiceReference
    instructions: str | None = Field(default=None, max_length=4_096)
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] | None = None
    speed: float | None = Field(default=None, ge=0.25, le=4.0)


class _AudioPart(_WireModel):
    """The facts of one uploaded audio part, measured by the native data plane.

    Attributes:
        bytes: Size of the audio in bytes.
        filename: The upload's filename.
        content_type: The upload's content type, when it named one.
        sha256: Lowercase hex SHA-256 of the audio bytes.
        duration_milli: Duration measured from the demuxed packets.
    """

    bytes: int = Field(ge=1, le=MAXIMUM_TRANSCRIPTION_AUDIO_BYTES)
    filename: str = Field(min_length=1, max_length=256)
    content_type: str | None = Field(default=None, max_length=128)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_milli: int = Field(ge=1, le=MAXIMUM_TRANSCRIPTION_AUDIO_MILLI)


class _TranscriptionRequest(_WireModel):
    """Closed gateway transcription profile: the upload's fields plus the audio facts.

    Attributes:
        model: The public alias to route.
        file: The measured facts of the audio part.
        language: Optional ISO-639-1 language.
        prompt: Optional guiding text.
        response_format: Optional ``json``, ``verbose_json``, or ``text``.
        temperature: Optional sampling temperature.
        timestamp_granularities: Optional ``word`` / ``segment`` timestamps.
    """

    model: str = Field(min_length=1, max_length=256)
    file: _AudioPart
    language: str | None = Field(default=None, min_length=2, max_length=16)
    prompt: str | None = Field(default=None, max_length=4_096)
    response_format: Literal["json", "verbose_json", "text"] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    timestamp_granularities: tuple[Literal["word", "segment"], ...] = Field(
        default=(), max_length=2
    )

    @field_validator("timestamp_granularities")
    @classmethod
    def _distinct_granularities(
        cls, value: tuple[Literal["word", "segment"], ...]
    ) -> tuple[Literal["word", "segment"], ...]:
        """Refuse a repeated granularity: only ``word`` and ``segment`` exist."""
        if len(set(value)) != len(value):
            raise ValueError("timestamp_granularities must not repeat a value")
        return value


_SPEECH_OFFICIAL: TypeAdapter[object] = TypeAdapter[object](SpeechCreateParams)


class DecodedSpeechRequest(ContractModel):
    """Public alias plus its canonical speech request.

    Attributes:
        alias: The public model alias the caller named.
        request: The canonical speech request.
    """

    alias: str = Field(min_length=1, max_length=256)
    request: SpeechRequest


class DecodedTranscriptionRequest(ContractModel):
    """Public alias plus its canonical transcription request.

    Attributes:
        alias: The public model alias the caller named.
        request: The canonical transcription request.
    """

    alias: str = Field(min_length=1, max_length=256)
    request: TranscriptionRequest


def decode_speech(payload: JsonObject) -> DecodedSpeechRequest:
    """Decode one ``/audio/speech`` body into the canonical speech surface.

    Args:
        payload: Parsed JSON request body.

    Returns:
        Public alias and canonical speech request.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
    """
    validate_manifest(payload, SPEECH_MANIFEST)
    _validate_official(_SPEECH_OFFICIAL, payload)
    request = _validate_wire(_SpeechRequest, payload)
    try:
        canonical = SpeechRequest(
            input=request.input,
            voice=(
                request.voice
                if isinstance(request.voice, str)
                else SpeechVoiceReference(id=request.voice.id)
            ),
            response_format=request.response_format,
            speed=request.speed,
            instructions=request.instructions,
        )
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc
    if canonical.worst_case_output_bytes > MAXIMUM_SPEECH_OUTPUT_BYTES:
        # Refused before any provider call: an answer this large would be
        # paid for and then dropped at the data plane's retained-output cap.
        raise OpenAIProtocolError(
            status_code=400,
            code="invalid_request",
            message=(
                "This input is too long for one speech request at this speed and format. "
                "Split the text, raise speed, or choose a compressed response_format."
            ),
            param="input",
        )
    return DecodedSpeechRequest(alias=request.model, request=canonical)


def decode_transcription(payload: JsonObject) -> DecodedTranscriptionRequest:
    """Decode one parsed ``/audio/transcriptions`` upload into the canonical surface.

    Args:
        payload: The upload's text fields and its measured ``file`` facts.

    Returns:
        Public alias and canonical transcription request.

    Raises:
        OpenAIProtocolError: The upload is invalid, unknown, or unsupported.
    """
    validate_manifest(payload, TRANSCRIPTION_MANIFEST)
    request = _validate_wire(_TranscriptionRequest, payload)
    if request.timestamp_granularities and request.response_format != "verbose_json":
        # The provider rejects timestamps on any other format after the
        # upload; refusing here costs the caller nothing.
        raise OpenAIProtocolError(
            status_code=400,
            code="invalid_request",
            message="timestamp_granularities requires response_format=verbose_json.",
            param="timestamp_granularities",
        )
    try:
        canonical = TranscriptionRequest(
            audio_bytes=request.file.bytes,
            audio_filename=request.file.filename,
            audio_content_type=request.file.content_type,
            audio_sha256=request.file.sha256,
            audio_duration_milli=request.file.duration_milli,
            language=request.language,
            prompt=request.prompt,
            response_format=request.response_format,
            temperature=request.temperature,
            timestamp_granularities=request.timestamp_granularities,
        )
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc
    return DecodedTranscriptionRequest(alias=request.model, request=canonical)
