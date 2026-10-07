"""OpenAI-wire request bodies for the speech and transcription surfaces.

Speech is a JSON body. Transcription is a multipart upload whose audio part
the native data plane attaches itself; this module builds only its text
fields, as the ordered name/value pairs the multipart encoder writes.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.audio_contracts import SpeechRequest, TranscriptionRequest


def openai_speech_request(
    model_id: str, request: SpeechRequest, *, metered_stream: bool
) -> JsonObject:
    """Convert one canonical speech request into the OpenAI ``/audio/speech`` body.

    Args:
        model_id: Served speech model id.
        request: Canonical speech request.
        metered_stream: Ask for the provider's SSE format, whose final event
            reports token usage; token-priced lanes need it to bill, and the
            data plane reassembles the audio from the event stream.

    Returns:
        The OpenAI-compatible ``/audio/speech`` request body.
    """
    voice: str | JsonObject = (
        request.voice if isinstance(request.voice, str) else {"id": request.voice.id}
    )
    body: JsonObject = {"model": model_id, "input": request.input, "voice": voice}
    for field in ("response_format", "speed", "instructions"):
        value = getattr(request, field)
        if value is not None:
            body[field] = value
    if metered_stream:
        body["stream_format"] = "sse"
    return body


def openai_transcription_fields(model_id: str, request: TranscriptionRequest) -> JsonObject:
    """Build the text fields of one OpenAI ``/audio/transcriptions`` upload.

    A caller asking for ``text`` is served from the provider's ``json`` answer
    (the same transcript plus the usage the surface bills on); ``json`` and
    ``verbose_json`` pass through. Multi-valued fields are lists, written as
    repeated parts under their ``[]`` name.

    Args:
        model_id: Served transcription model id.
        request: Canonical transcription request.

    Returns:
        Ordered multipart text fields (the audio part is attached by the data plane).
    """
    upstream_format = "verbose_json" if request.response_format == "verbose_json" else "json"
    fields: JsonObject = {"model": model_id, "response_format": upstream_format}
    if request.language is not None:
        fields["language"] = request.language
    if request.prompt is not None:
        fields["prompt"] = request.prompt
    if request.temperature is not None:
        fields["temperature"] = str(request.temperature)
    if request.timestamp_granularities:
        fields["timestamp_granularities[]"] = list(request.timestamp_granularities)
    return fields
