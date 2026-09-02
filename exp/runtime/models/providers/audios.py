"""Per-dialect encoders for caller audio parts.

Audio input rides exactly two provider wires in this gateway: the OpenAI
Chat Completions ``input_audio`` part (OpenAI, Azure OpenAI, OpenRouter, and
other Chat-compatible endpoints) and the Gemini ``inline_data`` part. The
OpenAI Responses API refuses audio input on every model, the Anthropic
Messages wire defines no audio block, and no Bedrock Converse model accepts
the ``audio`` content block its schema lists, so those wires reject audio as
a capability the rung cannot preserve and a waterfall narrows past them
instead of dropping the caller's clip.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import AudioContentPart
from exp.runtime.models.providers.errors import ProviderCapabilityError

AUDIO_DIALECTS = frozenset({"openai_compatible", "gemini_generate_content"})
"""Dialects whose wire defines an inline audio content part a model serves."""

AUDIO_CAPABILITY = "audio_input"
"""Capability literal naming inline audio input on a route."""


def openai_chat_audio_part(audio: AudioContentPart) -> JsonObject:
    """Encode one clip as an OpenAI-compatible Chat ``input_audio`` content part."""
    return {
        "type": "input_audio",
        "input_audio": {"data": audio.data, "format": audio.audio_format()},
    }


def gemini_audio_part(audio: AudioContentPart) -> JsonObject:
    """Encode one clip as a Gemini ``inline_data`` part."""
    return {"inline_data": {"mime_type": audio.media_type, "data": audio.data}}


def reject_audio_part(audio: AudioContentPart) -> JsonObject:
    """Refuse one clip on a wire that defines no audio carrier a model serves."""
    del audio
    raise ProviderCapabilityError(capability=AUDIO_CAPABILITY)
