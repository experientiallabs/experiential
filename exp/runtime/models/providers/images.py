"""Per-dialect encoders for caller image parts.

One canonical image part translates differently on every provider wire, so
each dialect's exact shape lives here and the message translators stay
about roles and ordering. Inline base64 rides every image-capable wire; a
remote URL is a provider-side fetch, so a wire without one rejects it as a
capability the rung cannot preserve, which lets a waterfall narrow past it
instead of dropping the caller's picture. A provider handle (an uploaded
file id, a Gemini Files URI, a ``gs://`` or ``s3://`` object) rides only the
wire of the provider that minted it; the Chat Completions ``image_url`` part
defines no handle at all, so that wire refuses every image handle.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import ImageContentPart
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.media_handles import (
    ANTHROPIC_HANDLE_PROVIDERS,
    BEDROCK_HANDLE_PROVIDERS,
    GEMINI_HANDLE_PROVIDERS,
    MEDIA_HANDLE_CAPABILITY,
    OPENAI_HANDLE_PROVIDERS,
    bedrock_s3_location,
    require_handle_provider,
)

IMAGE_URL_DIALECTS = frozenset({"openai_compatible", "openai_responses", "anthropic_messages"})
"""Dialects whose provider fetches a caller image URL on the gateway's behalf."""

IMAGE_URL_CAPABILITY = "image_url_input"
"""Capability literal naming a provider-side fetch of a caller image URL."""


def openai_chat_image_part(image: ImageContentPart) -> JsonObject:
    """Encode one image as a Chat Completions ``image_url`` content part.

    Raises:
        ProviderCapabilityError: The image is a provider handle; the Chat
            ``image_url`` part carries a URL or data URL only.
    """
    if image.handle is not None:
        raise ProviderCapabilityError(capability=MEDIA_HANDLE_CAPABILITY)
    image_url: JsonObject = {"url": image.data_url()}
    if image.detail is not None:
        image_url["detail"] = image.detail
    return {"type": "image_url", "image_url": image_url}


def responses_image_part(image: ImageContentPart) -> JsonObject:
    """Encode one image as a Responses ``input_image`` content part.

    An OpenAI Files handle rides ``file_id``; every other carrier rides
    ``image_url``.
    """
    part: JsonObject = {"type": "input_image"}
    if image.handle is not None:
        require_handle_provider(image.handle, OPENAI_HANDLE_PROVIDERS)
        part["file_id"] = image.handle.reference
    else:
        part["image_url"] = image.data_url()
    if image.detail is not None:
        part["detail"] = image.detail
    return part


def anthropic_image_block(image: ImageContentPart) -> JsonObject:
    """Encode one image as an Anthropic ``image`` content block.

    A caller cache breakpoint re-emits on the block it was placed on: this
    wire caches a marked image, and a lost marker silently returns the whole
    prefix to full input billing.
    """
    source: JsonObject
    if image.handle is not None:
        require_handle_provider(image.handle, ANTHROPIC_HANDLE_PROVIDERS)
        source = {"type": "file", "file_id": image.handle.reference}
    elif image.data is None:
        source = {"type": "url", "url": image.url or ""}
    else:
        source = {"type": "base64", "media_type": image.media_type, "data": image.data}
    block: JsonObject = {"type": "image", "source": source}
    if image.cache_control is not None:
        block["cache_control"] = image.cache_control
    return block


def gemini_image_part(image: ImageContentPart) -> JsonObject:
    """Encode one image as a Gemini ``inline_data`` or ``file_data`` part.

    Args:
        image: Canonical image part from the caller's message.

    Returns:
        The native Gemini part carrying the image bytes, or the ``file_data``
        reference for a Gemini Files or ``gs://`` handle.

    Raises:
        ProviderCapabilityError: The image is a remote URL, which this wire
            cannot fetch on the caller's behalf, or a handle from another
            provider.
    """
    if image.handle is not None:
        require_handle_provider(image.handle, GEMINI_HANDLE_PROVIDERS)
        file_data: JsonObject = {"file_uri": image.handle.reference}
        if image.media_type is not None:
            file_data["mime_type"] = image.media_type
        return {"file_data": file_data}
    if image.data is None:
        raise ProviderCapabilityError(capability=IMAGE_URL_CAPABILITY)
    return {"inline_data": {"mime_type": image.media_type, "data": image.data}}


def bedrock_image_block(image: ImageContentPart) -> JsonObject:
    """Encode one image as a Bedrock Converse ``image`` block.

    The Converse REST body carries the bytes base64 encoded, which is
    exactly the canonical inline form.

    Args:
        image: Canonical image part from the caller's message.

    Returns:
        The native Converse block carrying the image bytes, or its
        ``s3Location`` for a Bedrock handle.

    Raises:
        ProviderCapabilityError: The image is a remote URL, which this wire
            cannot fetch on the caller's behalf, or a handle from another
            provider.
    """
    if image.handle is not None:
        require_handle_provider(image.handle, BEDROCK_HANDLE_PROVIDERS)
        return {
            "image": {
                "format": image.image_format(),
                "source": {"s3Location": bedrock_s3_location(image.handle)},
            }
        }
    if image.data is None:
        raise ProviderCapabilityError(capability=IMAGE_URL_CAPABILITY)
    return {
        "image": {
            "format": image.image_format(),
            "source": {"bytes": image.data},
        }
    }
