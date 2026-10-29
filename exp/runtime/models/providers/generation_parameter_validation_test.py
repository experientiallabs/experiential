# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Declared per-rung output bounds and omission-preserving generation policy."""

import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.generation_parameter_validation import bounded_output_request


def _request(limit: int | None = None) -> GatewayRequest:
    """Build a chat request with an optional caller ceiling."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
        maximum_output_tokens=limit,
    )


@pytest.mark.parametrize("maximum", (512, 2_048, 8_192, 128_000))
def test_required_output_cap_uses_the_exact_declared_bound(maximum: int) -> None:
    """Neither small nor large model maxima are replaced with an arbitrary default."""
    request = _request()
    profile = GatewayWireProfile(
        dialect="anthropic_messages", url="https://p.test", maximum_output_tokens=maximum
    )
    shaped, reservation = bounded_output_request(profile, request)
    assert request.maximum_output_tokens is None
    assert shaped.maximum_output_tokens == reservation == maximum


@pytest.mark.parametrize(
    "dialect",
    ("openai_compatible", "openai_responses", "gemini_generate_content", "bedrock_converse_stream"),
)
def test_optional_output_cap_stays_omitted_but_fully_reserved(dialect: str) -> None:
    """Known metadata bounds the money decision without changing optional wire semantics."""
    request = _request()
    profile = GatewayWireProfile(
        dialect=dialect, url="https://p.test", maximum_output_tokens=128_000
    )
    shaped, reservation = bounded_output_request(profile, request)
    assert shaped is request
    assert shaped.maximum_output_tokens is None
    assert reservation == 128_000


@pytest.mark.parametrize("dialect", ("anthropic_messages", "openai_compatible"))
def test_unknown_bounds_require_a_named_explicit_cap(dialect: str) -> None:
    """An unbounded omission cannot dispatch on either required or optional wires."""
    profile = GatewayWireProfile(dialect=dialect, url="https://p.test")
    with pytest.raises(ProviderParameterError, match="Supply an explicit max_tokens") as raised:
        bounded_output_request(profile, _request())
    assert raised.value.param == "max_tokens"
    assert raised.value.code == "invalid_parameter"
    shaped, reservation = bounded_output_request(profile, _request(12))
    assert shaped.maximum_output_tokens == reservation == 12


def test_context_only_metadata_cannot_invent_a_required_wire_maximum() -> None:
    """A full context window is a safe reservation, not a legal Anthropic max_tokens claim."""
    profile = GatewayWireProfile(dialect="anthropic_messages", url="https://p.test")
    with pytest.raises(ProviderParameterError, match="no declared output maximum"):
        bounded_output_request(profile, _request(), context_window_tokens=200_000)
    optional = GatewayWireProfile(dialect="openai_compatible", url="https://p.test")
    shaped, reservation = bounded_output_request(
        optional, _request(), context_window_tokens=200_000
    )
    assert shaped.maximum_output_tokens is None
    assert reservation == 200_000


@pytest.mark.parametrize("caller", (None, 128, 8_192))
def test_required_cap_respects_catalog_and_window_without_guessing_input(
    caller: int | None,
) -> None:
    """A total window bounds output, but no approximate tokenizer pretends it is exact room."""
    profile = GatewayWireProfile(
        dialect="anthropic_messages", url="https://p.test", maximum_output_tokens=128_000
    )
    shaped, reservation = bounded_output_request(
        profile, _request(caller), model_maximum_output_tokens=64_000, context_window_tokens=8_192
    )
    assert shaped.maximum_output_tokens == reservation == (caller or 8_192)


@pytest.mark.parametrize("bound_key", ("model_maximum_output_tokens", "context_window_tokens"))
def test_explicit_cap_above_a_declared_bound_is_rejected(bound_key: str) -> None:
    """Explicit values cannot be silently clamped to satisfy the catalog."""
    profile = GatewayWireProfile(dialect="anthropic_messages", url="https://p.test")
    with pytest.raises(ProviderParameterError, match="declared bound of 2048"):
        bounded_output_request(profile, _request(8_192), **{bound_key: 2_048})
