"""OpenAI prompt-cache option origins and payloads."""

from __future__ import annotations

import pytest

from exp.runtime.models.providers.prompt_cache_options import (
    openai_prompt_cache_origin,
    prompt_cache_options_payload,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    (
        ("https://api.openai.com/v1/responses", True),
        ("https://openrouter.ai/api/v1/chat/completions", True),
        ("https://acme.openai.azure.com/openai/v1/responses", True),
        ("https://acme.cognitiveservices.azure.com/openai/deployments/x", True),
        ("https://acme.services.ai.azure.com/models", False),
        ("https://api.novita.ai/v3/openai/responses", False),
        ("http://localhost:8000/v1/chat/completions", False),
    ),
)
def test_only_documented_origins_carry_the_options(url: str, expected: bool) -> None:
    """OpenAI, OpenRouter and Azure OpenAI hosts carry the fields; strict servers do not."""
    assert openai_prompt_cache_origin(url) is expected


def test_payload_sends_only_non_default_values() -> None:
    """The defaults are never sent; explicit mode needs a carrying origin."""
    assert prompt_cache_options_payload(mode=None, ttl=None, forwards_options=True) is None
    assert prompt_cache_options_payload(mode="explicit", ttl=None, forwards_options=False) is None
    assert prompt_cache_options_payload(mode="explicit", ttl=None, forwards_options=True) == {
        "mode": "explicit"
    }
    assert prompt_cache_options_payload(mode="explicit", ttl="12h", forwards_options=True) == {
        "mode": "explicit",
        "ttl": "12h",
    }
    assert prompt_cache_options_payload(mode=None, ttl="12h", forwards_options=False) == {
        "ttl": "12h"
    }
