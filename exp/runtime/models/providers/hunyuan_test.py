# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Hunyuan OpenAI-compatible endpoint detection."""

from __future__ import annotations

import pytest

from exp.runtime.models.providers.hunyuan import is_hunyuan_base_url


def test_matches_exact_public_hunyuan_root() -> None:
    """The public OpenAI-compatible inference root is recognized."""
    assert is_hunyuan_base_url("https://api.hunyuan.cloud.tencent.com/v1")
    assert is_hunyuan_base_url("https://api.hunyuan.cloud.tencent.com/v1/")


@pytest.mark.parametrize(
    "base_url",
    [
        # The TC3-HMAC-signed API host is a different protocol entirely.
        "https://hunyuan.tencentcloudapi.com/v1",
        # Plaintext, credential-bearing, ported, or path-shifted variants.
        "http://api.hunyuan.cloud.tencent.com/v1",
        "https://user:pass@api.hunyuan.cloud.tencent.com/v1",
        "https://api.hunyuan.cloud.tencent.com:8443/v1",
        "https://api.hunyuan.cloud.tencent.com/v1/chat/completions",
        "https://api.hunyuan.cloud.tencent.com/openai/v1",
        # A look-alike host must not match.
        "https://api.hunyuan.cloud.tencent.com.evil.test/v1",
    ],
)
def test_rejects_non_canonical_endpoints(base_url: str) -> None:
    """Only the exact scheme/host/path with no extras is a Hunyuan route."""
    assert not is_hunyuan_base_url(base_url)
