"""Per-dialect encoders for caller video parts.

Video is narrower than images: only three provider wires define a video
carrier. Gemini ``generateContent`` takes ``inline_data`` bytes or a
``file_data`` URI it fetches itself; Bedrock Converse takes a ``video``
block with inline bytes (an S3 location is a provider-side resource the
gateway does not author); and the OpenAI-compatible Chat wire served by
OpenRouter and Fireworks takes a ``video_url`` content part holding a data
URL or an http(s) URL. The OpenAI Responses and Anthropic Messages wires
define no video content at all, so a video part reaching them is a
capability the rung cannot preserve and route selection narrows past it.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import VideoContentPart, VideoMediaType
from exp.runtime.models.providers.errors import ProviderCapabilityError

VIDEO_DIALECTS = frozenset(
    {"openai_compatible", "gemini_generate_content", "bedrock_converse_stream"}
)
"""Dialects whose wire defines a caller video carrier at all."""

VIDEO_URL_DIALECTS = frozenset({"openai_compatible", "gemini_generate_content"})
"""Dialects whose provider fetches a caller video URL on the gateway's behalf."""

VIDEO_CAPABILITY = "video_input"
"""Capability literal naming any caller video on a wire without a carrier."""

VIDEO_URL_CAPABILITY = "video_url_input"
"""Capability literal naming a provider-side fetch of a caller video URL."""

_BEDROCK_VIDEO_FORMATS: dict[VideoMediaType, str] = {
    "video/mp4": "mp4",
    "video/mpeg": "mpeg",
    "video/quicktime": "mov",
    "video/webm": "webm",
    "video/x-flv": "flv",
    "video/3gpp": "three_gp",
    "video/x-ms-wmv": "wmv",
}
"""Bedrock Converse ``VideoBlock.format`` enum value for each canonical media type."""


def openai_chat_video_part(video: VideoContentPart) -> JsonObject:
    """Encode one video as an OpenAI-compatible Chat ``video_url`` content part.

    OpenRouter and Fireworks document this exact shape; OpenAI itself does
    not define it, so route declaration keeps it off OpenAI routes.
    """
    return {"type": "video_url", "video_url": {"url": video.data_url()}}


def gemini_video_part(video: VideoContentPart) -> JsonObject:
    """Encode one video as a Gemini ``inline_data`` or ``file_data`` part.

    Args:
        video: Canonical video part from the caller's message.

    Returns:
        The native Gemini part carrying the video bytes or its URI.
    """
    if video.data is None:
        return {"file_data": {"file_uri": video.url}}
    return {"inline_data": {"mime_type": video.media_type, "data": video.data}}


def bedrock_video_block(video: VideoContentPart) -> JsonObject:
    """Encode one video as a Bedrock Converse ``video`` block.

    Args:
        video: Canonical video part from the caller's message.

    Returns:
        The native Converse block carrying the video bytes.

    Raises:
        ProviderCapabilityError: The video is a remote URL, which this wire
            cannot fetch on the caller's behalf.
    """
    if video.data is None or video.media_type is None:
        raise ProviderCapabilityError(capability=VIDEO_URL_CAPABILITY)
    return {
        "video": {
            "format": _BEDROCK_VIDEO_FORMATS[video.media_type],
            "source": {"bytes": video.data},
        }
    }


def reject_video_part(video: VideoContentPart) -> JsonObject:
    """Refuse one video on a wire that defines no video carrier.

    Args:
        video: Canonical video part from the caller's message.

    Raises:
        ProviderCapabilityError: Always; the rung cannot preserve the video.
    """
    del video
    raise ProviderCapabilityError(capability=VIDEO_CAPABILITY)
