# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""DeepSeek first-party origin detection."""

from __future__ import annotations

import pytest

from exp.runtime.models.providers.deepseek import is_deepseek_base_url


@pytest.mark.parametrize(
    "base_url",
    [
        # The ``/v1`` alias the platform's house lane dispatches to.
        "https://api.deepseek.com/v1",
        "https://api.deepseek.com/v1/",
        # DeepSeek's documented bare root, equivalent to ``/v1``.
        "https://api.deepseek.com",
        "https://api.deepseek.com/",
    ],
)
def test_matches_deepseeks_documented_roots(base_url: str) -> None:
    """Both documented OpenAI-compatible roots are DeepSeek's origin."""
    assert is_deepseek_base_url(base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        # Plaintext, credential-bearing, ported, or path-shifted variants.
        "http://api.deepseek.com/v1",
        "https://user:pass@api.deepseek.com/v1",
        "https://api.deepseek.com:8443/v1",
        "https://api.deepseek.com/v1/chat/completions",
        "https://api.deepseek.com/v1?x=1",
        # The beta feature root is a different surface.
        "https://api.deepseek.com/beta",
        # A look-alike host must not match.
        "https://api.deepseek.com.evil.test/v1",
        # Third-party hosts serving DeepSeek weights enforce no replay rule.
        "https://openrouter.ai/api/v1",
        "https://api.fireworks.ai/inference/v1",
        "https://example.services.ai.azure.com/openai/v1",
    ],
)
def test_rejects_non_canonical_endpoints(base_url: str) -> None:
    """Only the exact scheme/host/root with no extras is DeepSeek's origin."""
    assert not is_deepseek_base_url(base_url)
