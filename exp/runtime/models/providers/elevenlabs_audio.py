"""ElevenLabs request translation for the gateway's OpenAI audio surfaces.

Callers speak the OpenAI audio wire (``/v1/audio/speech`` and
``/v1/audio/transcriptions``); an ElevenLabs rung is dispatched on ElevenLabs'
own wire. This module is the whole translation, kept pure so admission can
build each rung's endpoint and body, or name the field that rung cannot honor:

- Speech posts ``{text, model_id, voice_settings?}`` as JSON to
  ``/text-to-speech/{voice_id}?output_format=...`` and answers with the audio
  bytes. The voice is an ElevenLabs voice id, or one of the six classic OpenAI
  voice names mapped onto a premade ElevenLabs voice.
- Transcription uploads the audio as multipart ``file`` beside ``model_id``
  (and ``language_code``); the answer's ``audio_duration_secs`` is the meter.

A request field the ElevenLabs wire cannot carry (a format it does not
produce, a speed outside its range, delivery instructions, a transcription
prompt, temperature, or segment timestamps) is refused on that field rather
than silently dropped.
"""

from __future__ import annotations

import re

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.audio_contracts import SpeechRequest, TranscriptionRequest
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, unsupported_field

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
"""ElevenLabs API root; speech and transcription routes hang below it."""

ELEVENLABS_AUDIO_DIALECT = "elevenlabs_audio"
"""Wire dialect naming an ElevenLabs audio rung to admission and the data plane."""

ELEVENLABS_OPENAI_VOICES: dict[str, str] = {
    "alloy": "SAz9YHcvj6GT2YYXdXww",  # River: neutral, relaxed, informative.
    "echo": "cjVigY5qzO86Huf0OWal",  # Eric: smooth, trustworthy male.
    "fable": "JBFqnCBsd6RMkjVDRZzb",  # George: warm British storyteller.
    "onyx": "nPczCjzI2devNBz1zQrb",  # Brian: deep, resonant male.
    "nova": "hpp4J3VqNfWAUOO0d1Us",  # Bella: bright, warm, professional female.
    "shimmer": "EXAVITQu4vr4xnSDxMaL",  # Sarah: soft, reassuring female.
}
"""The classic OpenAI voice names mapped onto premade ElevenLabs voices
(``GET /v1/voices``, category ``premade``), so an OpenAI SDK caller's voice
works on an ElevenLabs rung unchanged."""

_VOICE_ID = re.compile(r"^[A-Za-z0-9]{20}$")
"""The shape of an ElevenLabs voice id (twenty alphanumerics)."""

ELEVENLABS_OUTPUT_FORMATS: dict[str, str] = {
    "mp3": "mp3_44100_128",
    "opus": "opus_48000_64",
    "pcm": "pcm_24000",
    "wav": "wav_24000",
}
"""OpenAI speech formats ElevenLabs produces, as its ``output_format`` values.
``aac`` and ``flac`` have no ElevenLabs equivalent."""

MINIMUM_ELEVENLABS_SPEED, MAXIMUM_ELEVENLABS_SPEED = 0.7, 1.2
"""The speech speed range ElevenLabs' ``voice_settings.speed`` accepts."""


def _refusal(param: str, message: str) -> OpenAIProtocolError:
    """Build one field-specific 400 for a request an ElevenLabs rung cannot serve."""
    return unsupported_field(param, capability=True, message=message)


def elevenlabs_voice_id(voice: str) -> str | None:
    """Resolve a caller's voice to an ElevenLabs voice id, or None when it names none."""
    mapped = ELEVENLABS_OPENAI_VOICES.get(voice.lower())
    if mapped is not None:
        return mapped
    return voice if _VOICE_ID.fullmatch(voice) else None


def elevenlabs_speech_target(
    speech_url: str, model_id: str, request: SpeechRequest
) -> tuple[str, JsonObject] | OpenAIProtocolError:
    """Build one ElevenLabs rung's speech endpoint and JSON body.

    Args:
        speech_url: The connection's ``/text-to-speech`` root.
        model_id: Served ElevenLabs model id (for example ``eleven_flash_v2_5``).
        request: Canonical speech request.

    Returns:
        The full endpoint (voice in the path, output format in the query) and
        the JSON body, or the field-specific 400 naming what this rung cannot
        honor.
    """
    voice = request.voice if isinstance(request.voice, str) else request.voice.id
    voice_id = elevenlabs_voice_id(voice)
    if voice_id is None:
        return _refusal(
            "voice",
            "This model takes an ElevenLabs voice id or one of alloy, echo, fable, onyx, "
            "nova, or shimmer.",
        )
    output_format = ELEVENLABS_OUTPUT_FORMATS.get(request.response_format or "mp3")
    if output_format is None:
        return _refusal("response_format", "This model produces mp3, opus, pcm, or wav audio.")
    if request.instructions is not None:
        return _refusal("instructions", "This model does not take delivery instructions.")
    body: JsonObject = {"text": request.input, "model_id": model_id}
    if request.speed is not None:
        if not MINIMUM_ELEVENLABS_SPEED <= request.speed <= MAXIMUM_ELEVENLABS_SPEED:
            return _refusal(
                "speed",
                f"This model speaks at a speed from {MINIMUM_ELEVENLABS_SPEED} to "
                f"{MAXIMUM_ELEVENLABS_SPEED}.",
            )
        body["voice_settings"] = {"speed": request.speed}
    return f"{speech_url.rstrip('/')}/{voice_id}?output_format={output_format}", body


def elevenlabs_transcription_fields(
    model_id: str, request: TranscriptionRequest
) -> JsonObject | OpenAIProtocolError:
    """Build the text fields of one ElevenLabs ``/speech-to-text`` upload.

    The audio is attached by the data plane as the ``file`` part. ElevenLabs
    answers one JSON shape for every caller format (the data plane renders
    ``json``, ``text``, and ``verbose_json`` from it), so no format field is
    sent, and a caller URL, webhook, or prompt is never forwarded.

    Args:
        model_id: Served ElevenLabs transcription model id (``scribe_v2``).
        request: Canonical transcription request.

    Returns:
        Ordered multipart text fields, or the field-specific 400 naming what
        this rung cannot honor.
    """
    if request.prompt is not None:
        return _refusal("prompt", "This model does not take a transcription prompt.")
    if request.temperature is not None:
        return _refusal("temperature", "This model does not take a sampling temperature.")
    if "segment" in request.timestamp_granularities:
        return _refusal(
            "timestamp_granularities",
            "This model returns word timestamps only; request granularity word.",
        )
    fields: JsonObject = {"model_id": model_id}
    if request.language is not None:
        fields["language_code"] = request.language
    return fields
