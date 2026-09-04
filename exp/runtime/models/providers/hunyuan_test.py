# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Hunyuan OpenAI-compatible endpoint detection."""

from __future__ import annotations

import pytest

from exp.runtime.models.providers.hunyuan import is_hunyuan_base_url


@pytest.mark.parametrize(
    "base_url",
    [
        # The mainland Hunyuan OpenAI-compatible inference root.
        "https://api.hunyuan.cloud.tencent.com/v1",
        "https://api.hunyuan.cloud.tencent.com/v1/",
        # The TokenHub international MaaS origin the platform's Tencent lane
        # dispatches through — the endpoint that must resolve a carrier route.
        "https://tokenhub-intl.tencentcloudmaas.com/v1",
        "https://tokenhub-intl.tencentcloudmaas.com/v1/",
    ],
)
def test_matches_the_hunyuan_reasoning_roots(base_url: str) -> None:
    """Both OpenAI-compatible reasoning origins are recognized."""
    assert is_hunyuan_base_url(base_url)


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
        # Site-bound TokenHub siblings are not served today and must not match
        # until a lane dispatches through them.
        "https://tokenhub-us.tencentcloudmaas.com/v1",
        "https://tokenhub.tencentcloudmaas.com/v1",
        # A TokenHub look-alike host must not match.
        "https://tokenhub-intl.tencentcloudmaas.com.evil.test/v1",
        "http://tokenhub-intl.tencentcloudmaas.com/v1",
        "https://tokenhub-intl.tencentcloudmaas.com/openai/v1",
    ],
)
def test_rejects_non_canonical_endpoints(base_url: str) -> None:
    """Only the exact scheme/host/path with no extras is a Hunyuan route."""
    assert not is_hunyuan_base_url(base_url)
