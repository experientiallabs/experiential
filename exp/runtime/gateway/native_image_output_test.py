"""Admission and payload contracts for mixed text/image chat output."""

import pytest

from exp.common.models import ModelCapabilities
from exp.common.models.content import ImageContentPart
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.native_image_output import image_aware_stream_payload
from exp.runtime.models.providers.base import GatewayWireProfile, ProviderCapabilityError


@pytest.mark.parametrize(
    ("dialect", "provider", "field", "modalities"),
    [
        ("gemini_generate_content", "google", "generationConfig", ["TEXT", "IMAGE"]),
        ("openai_compatible", "openrouter", "modalities", ["text", "image"]),
    ],
)
def test_image_chat_requests_both_modalities(
    dialect: str, provider: str, field: str, modalities: list[str]
) -> None:
    """An image-emitting model explicitly requests text and image output."""
    profile = GatewayWireProfile(dialect=dialect, url="https://example.test", model_id="image")
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="Draw a cat"),),
    )
    payload = image_aware_stream_payload(
        profile, request, ModelCapabilities(emits_images=True), provider
    )
    if field == "generationConfig":
        config = payload[field]
        assert isinstance(config, dict)
        assert config["responseModalities"] == modalities
    else:
        assert payload[field] == modalities


@pytest.mark.parametrize("surface", [GatewayApiSurface.RESPONSES, GatewayApiSurface.MESSAGES])
def test_image_chat_refuses_surfaces_without_image_delivery(surface: GatewayApiSurface) -> None:
    """Unsupported output surfaces fail before a paid provider dispatch."""
    profile = GatewayWireProfile(dialect="openai_compatible", url="https://example.test")
    request = GatewayRequest(
        surface=surface, messages=(GatewayMessage(role="user", content="Draw"),)
    )
    with pytest.raises(ProviderCapabilityError, match="image_output"):
        image_aware_stream_payload(
            profile, request, ModelCapabilities(emits_images=True), "openrouter"
        )


def test_text_lane_does_not_request_images() -> None:
    """Existing text model payloads keep their ordinary generation behavior."""
    profile = GatewayWireProfile(dialect="openai_compatible", url="https://example.test")
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="Hi"),),
    )
    assert "modalities" not in image_aware_stream_payload(profile, request, None, "openrouter")


@pytest.mark.parametrize(
    "dialect", ["anthropic_messages", "openai_responses", "bedrock_converse_stream"]
)
def test_assistant_images_reject_non_preserving_fallback_wires(dialect: str) -> None:
    """Image-input support alone does not prove assistant-history preservation."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(
                role="assistant",
                content="",
                content_parts=(ImageContentPart(media_type="image/png", data="iVBORw0KGgo="),),
            ),
        ),
    )
    profile = GatewayWireProfile(dialect=dialect, url="https://example.test")
    with pytest.raises(ProviderCapabilityError, match="assistant_image_history"):
        image_aware_stream_payload(profile, request, None, "provider")
