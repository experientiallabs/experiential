"""Inline tests for the dialect dispatch seam.

The request-shaping behaviour is exercised through ``streaming_requests_test``
and every payload-builder suite; this module pins the disclosure wording that
callers read off the wire.
"""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import (
    CACHE_CONTROL_NOT_FORWARDED_SUFFIX,
    dialect_stream_payload,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError


def test_cache_control_disclosure_names_where_cache_reads_show_up() -> None:
    """The unforwarded-marker disclosure is a stable wire string that never reads as "ignored".

    It travels in ``x-experiential-ignored-parameters`` beside a billed
    ``cache_read_input_tokens`` on OpenAI-compatible routes, so it has to say
    that caching is the provider's decision and where any reads are reported.
    """
    assert CACHE_CONTROL_NOT_FORWARDED_SUFFIX == (
        "->not_forwarded(provider_decides_caching;"
        " cache reads reported in usage.cache_read_input_tokens)"
    )
    assert "ignored" not in CACHE_CONTROL_NOT_FORWARDED_SUFFIX
    assert "usage.cache_read_input_tokens" in CACHE_CONTROL_NOT_FORWARDED_SUFFIX


@pytest.mark.parametrize("surface", list(GatewayApiSurface))
@pytest.mark.parametrize("stream", [False, True])
def test_connection_us_geography_overrides_caller_on_every_surface(
    surface: GatewayApiSurface, stream: bool
) -> None:
    """The trusted connection wins after surface translation, including a conflicting caller."""
    request = GatewayRequest(
        surface=surface,
        stream=stream,
        messages=(GatewayMessage(role="user", content="hello"),),
        inference_geo="global" if surface == GatewayApiSurface.MESSAGES else None,
    )
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id="claude-sonnet-4-6",
        maximum_output_tokens=128_000,
        inference_geo="us",
    )
    assert dialect_stream_payload(profile, request)["inference_geo"] == "us"
    assert request.inference_geo == ("global" if surface == GatewayApiSurface.MESSAGES else None)


@pytest.mark.parametrize("caller", [None, "us", "global"])
def test_unrestricted_connection_preserves_caller_geography(caller: str | None) -> None:
    """No operator setting leaves the existing request contract unchanged."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hello"),),
        inference_geo=caller,
    )
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id="claude-sonnet-4-6",
        maximum_output_tokens=128_000,
    )
    assert dialect_stream_payload(profile, request).get("inference_geo") == caller


def test_non_anthropic_dialect_refuses_geography_constraint() -> None:
    """An incorrectly constructed profile cannot silently drop the constraint."""
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://example.test/v1/chat/completions",
        inference_geo="us",
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
    )
    with pytest.raises(ProviderCapabilityError, match="inference_geo"):
        dialect_stream_payload(profile, request)
