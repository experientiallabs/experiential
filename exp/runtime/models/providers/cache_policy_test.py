"""Provider payloads preserve cache durations for gateway price reservation."""

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import ImageContentPart, TextContentPart
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.cache_policy import (
    multimodal_text_cache_blocks,
    retain_multimodal_cache_boundaries,
)
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload


def test_empty_markers_preserve_complete_boundary_without_mutating_input() -> None:
    """Absent text carries no checkpoint; repeated empty hints share one boundary."""
    marker: JsonObject = {"type": "ephemeral"}
    image = ImageContentPart(media_type="image/png", data="aGk=")
    parts = (
        TextContentPart(text="prefix"),
        image,
        TextContentPart(text="", cache_control=marker),
        TextContentPart(text="", cache_control=marker),
    )
    retained, unsupported = retain_multimodal_cache_boundaries(parts)
    assert not unsupported
    assert len(retained) == 2 and retained[-1].kind == "image"
    assert retained[-1].cache_control == marker
    assert image.cache_control is None
    assert multimodal_text_cache_blocks(retained) == ()
    plain, unsupported = retain_multimodal_cache_boundaries(
        (parts[0], image, TextContentPart(text=""))
    )
    assert not unsupported and plain == parts[:2]
    assert plain[-1] is image


@pytest.mark.parametrize("reverse", [False, True])
def test_colocated_marker_comparison_uses_documented_default_ttl(reverse: bool) -> None:
    """Omitted TTL and explicit five minutes are the same published ephemeral contract."""
    markers: list[JsonObject] = [{"type": "ephemeral"}, {"type": "ephemeral", "ttl": "5m"}]
    if reverse:
        markers.reverse()
    parts = (
        TextContentPart(text="prefix", cache_control=markers[0]),
        TextContentPart(text="", cache_control=markers[1]),
    )
    retained, unsupported = retain_multimodal_cache_boundaries(parts)
    assert not unsupported
    assert len(retained) == 1 and retained[0].kind == "text"
    assert retained[0].cache_control == {"type": "ephemeral", "ttl": "5m"}


@pytest.mark.parametrize("customer_managed", [True, False])
def test_one_hour_cache_reaches_provider_for_both_billing_modes(customer_managed: bool) -> None:
    """Gateway reservation prices the TTL; serialization preserves it."""
    request = decode_messages(
        {
            "model": "coding",
            "max_tokens": 32,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).request
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://example.invalid",
        billing_customer_managed=customer_managed,
    )
    payload = dialect_stream_payload(profile, request)
    assert payload["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.parametrize("customer_managed", [True, False])
@pytest.mark.parametrize("ttl", ["5m", "1h"])
def test_server_tool_cache_duration_is_preserved_for_settlement(
    customer_managed: bool, ttl: str
) -> None:
    """Server-tool TTL survives serialization for both billing modes."""
    marker = {"type": "ephemeral", "ttl": ttl}
    request = decode_messages(
        {
            "model": "coding",
            "max_tokens": 32,
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "cache_control": marker,
                }
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).request
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://example.invalid",
        billing_customer_managed=customer_managed,
    )
    payload = dialect_stream_payload(profile, request)
    assert payload["tools"] == [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "cache_control": marker,
        }
    ]
