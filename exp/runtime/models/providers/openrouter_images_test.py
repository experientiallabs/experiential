"""Dedicated Images API request conversion regressions."""

import pytest

from exp.runtime.gateway.images_contracts import ImagesRequest
from exp.runtime.models.providers.openrouter_images import openrouter_images_request
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


def test_image_request_maps_controls_without_chat_fields() -> None:
    """OpenRouter receives its own shape, no unsupported OpenAI presentation fields."""
    body = openrouter_images_request(
        "openai/gpt-5.4-image-2",
        ImagesRequest(
            prompt="a cat",
            n=2,
            size="auto",
            quality="low",
            moderation="auto",
            response_format="b64_json",
            output_format="png",
            user="private-user",
        ),
    )
    assert body == {
        "model": "openai/gpt-5.4-image-2",
        "prompt": "a cat",
        "n": 2,
        "aspect_ratio": "auto",
        "quality": "low",
        "output_format": "png",
        "provider": {"allow_fallbacks": False, "options": {"openai": {"moderation": "auto"}}},
    }


@pytest.mark.parametrize(
    "values", [{"response_format": "url"}, {"style": "vivid"}, {"quality": "hd"}]
)
def test_unsupported_controls_fail_before_generation(values: dict[str, str]) -> None:
    """An unsupported requested mode must never silently create a paid image."""
    with pytest.raises(OpenAIProtocolError):
        openrouter_images_request(
            "openai/gpt-5.4-image-2", ImagesRequest.model_validate({"prompt": "cat", **values})
        )
