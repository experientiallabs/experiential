"""Tests for the Anthropic official-origin check and its safeguards gate."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.anthropic import (
    anthropic_official_origin,
    safeguards_for_upstream,
)


@pytest.mark.parametrize(
    ("url", "official"),
    [
        ("https://api.anthropic.com/v1/messages", True),
        ("https://api.anthropic.com:443/v1/messages", True),
        ("https://x.services.ai.azure.com/anthropic/v1/messages", False),
        ("https://claude-proxy.example.com/v1/messages", False),
        ("http://api.anthropic.com/v1/messages", False),
        ("https://api.anthropic.com.evil.test/v1/messages", False),
        ("https://user@api.anthropic.com/v1/messages", False),
        ("https://api.anthropic.com:8443/v1/messages", False),
        ("https://api.anthropic.com/v10/messages", False),
    ],
)
def test_only_anthropics_own_api_is_the_official_origin(url: str, official: bool) -> None:
    """Exact scheme, host, port and path prefix; no lookalike passes."""
    assert anthropic_official_origin(url) is official


def test_safeguards_survive_only_on_the_official_origin() -> None:
    """The gate clears the field elsewhere and leaves other requests untouched."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
        safeguards=({"type": "dangerous_tool_use"},),
    )
    official = safeguards_for_upstream("https://api.anthropic.com/v1/messages", request)
    assert official is request
    azure = safeguards_for_upstream(
        "https://x.services.ai.azure.com/anthropic/v1/messages", request
    )
    assert azure.safeguards is None
    plain = request.model_copy(update={"safeguards": None})
    assert safeguards_for_upstream("https://proxy.test/v1/messages", plain) is plain
