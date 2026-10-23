"""Tests for xAI origin detection and its video and image endpoints."""

from __future__ import annotations

import pytest

from exp.runtime.models.providers.xai import is_xai_base_url, is_xai_images_url, xai_videos_url


@pytest.mark.parametrize(
    "base_url", ["https://api.x.ai/v1", "https://api.x.ai/v1/", "https://api.x.ai:443/v1"]
)
def test_xais_own_root_is_recognized(base_url: str) -> None:
    """The documented root, with or without a trailing slash or default port, matches."""
    assert is_xai_base_url(base_url)
    assert xai_videos_url(base_url) == f"{base_url.rstrip('/')}/videos/generations"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.x.ai/v1",
        "https://api.x.ai",
        "https://openrouter.ai/api/v1",
        "https://user:pass@api.x.ai/v1",
        "https://api.x.ai/v1?x=1",
        "https://api.x.ai.evil.com/v1",
    ],
)
def test_every_other_root_is_not_xai(base_url: str) -> None:
    """Resellers, other paths, credentials, queries, and look-alike hosts never match."""
    assert not is_xai_base_url(base_url)
    assert xai_videos_url(base_url) is None


@pytest.mark.parametrize(
    ("images_url", "expected"),
    [
        ("https://api.x.ai/v1/images/generations", True),
        ("https://api.x.ai/v1/images/generations/", True),
        ("https://openrouter.ai/api/v1/images/generations", False),
        ("https://api.openai.com/v1/images/generations", False),
        ("https://api.x.ai/v1/images/edits", False),
        ("https://api.x.ai.evil.com/v1/images/generations", False),
    ],
)
def test_only_xais_own_image_endpoint_is_recognized(images_url: str, expected: bool) -> None:
    """xAI's ``/images/generations`` matches; resellers, other paths, and look-alikes do not."""
    assert is_xai_images_url(images_url) is expected
