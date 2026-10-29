"""Chat image-output admission and provider payload shaping."""

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelCapabilities
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile, ProviderCapabilityError
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload


def image_aware_stream_payload(
    profile: GatewayWireProfile,
    request: GatewayRequest,
    capabilities: ModelCapabilities | None,
    provider: str,
) -> JsonObject:
    """Build a streaming payload whose declared image output the public surface preserves.

    Args:
        profile: Frozen provider wire profile.
        request: The admitted canonical request.
        capabilities: Exact model capabilities, including its output modalities.
        provider: Provider identity from the admitted deployment.

    Returns:
        Provider payload, requesting text and images on image-emitting chat lanes.

    Raises:
        ProviderCapabilityError: The selected surface or wire cannot carry generated images.
    """
    if profile.dialect not in {"gemini_generate_content", "openai_compatible"} and any(
        message.role == "assistant" and message.images for message in request.messages
    ):
        raise ProviderCapabilityError(
            capability="assistant_image_history",
            detail=(
                "This provider wire cannot preserve assistant images. "
                "Choose a Gemini or Chat-compatible route."
            ),
        )
    emits_images = capabilities is not None and capabilities.emits_images
    if emits_images and (
        request.surface != GatewayApiSurface.CHAT_COMPLETIONS
        or profile.dialect not in {"gemini_generate_content", "openai_compatible"}
    ):
        raise ProviderCapabilityError(
            capability="image_output",
            detail=(
                "Use /v1/chat/completions or a supported Images API route for this image model."
            ),
        )
    payload = dialect_stream_payload(profile, request)
    if emits_images and profile.dialect == "gemini_generate_content":
        generation = payload.get("generationConfig")
        if not isinstance(generation, dict):
            raise ValueError("Gemini payload requires generationConfig")
        payload["generationConfig"] = {**generation, "responseModalities": ["TEXT", "IMAGE"]}
    elif emits_images and provider == "openrouter":
        payload["modalities"] = ["text", "image"]
    return payload
