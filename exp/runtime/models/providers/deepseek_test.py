# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""DeepSeek first-party origin detection."""

from __future__ import annotations

import pytest

from exp.runtime.models.providers.deepseek import (
    is_deepseek_base_url,
    is_deepseek_model_id,
)


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


@pytest.mark.parametrize(
    "model_id",
    ["deepseek/deepseek-v4-flash", "DeepSeek-V4-Flash", "deepseek-chat", "accounts/x/deepseek-v4"],
)
def test_deepseek_model_ids_match_on_the_family_token(model_id: str) -> None:
    assert is_deepseek_model_id(model_id)


@pytest.mark.parametrize("model_id", ["gpt-5.6-luna", "tencent/hy4-preview", "qwen3.8-27b", ""])
def test_other_model_ids_do_not_match(model_id: str) -> None:
    assert not is_deepseek_model_id(model_id)
