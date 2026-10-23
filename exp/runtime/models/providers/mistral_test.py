"""Tests for Mistral's greedy-sampling ``top_p`` rule."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.mistral import MISTRAL_GREEDY_TOP_P, is_mistral_model_id
from exp.runtime.models.providers.streaming_requests import (
    dialect_stream_payload,
    route_generation_parameter_requests,
)


def _request(*, temperature: float | None, top_p: float | None) -> GatewayRequest:
    """Build one streaming Chat request carrying the sampling controls."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
        temperature=temperature,
        top_p=top_p,
        stream=True,
        include_usage=True,
    )


def _profile(model_id: str) -> GatewayWireProfile:
    """Return one OpenAI-compatible rung serving ``model_id`` that takes ``top_p``."""
    return GatewayWireProfile(
        dialect="openai_compatible",
        url="https://api.mistral.test/v1",
        model_id=model_id,
        supports_top_p=True,
    )


@pytest.mark.parametrize(
    "model_id",
    ("mistral-large-4", "mistralai/mistral-large-4", "Mistral-Large-4", "devstral-2", "magistral"),
)
def test_mistral_model_ids_match_on_every_host_spelling(model_id: str) -> None:
    """The family token is matched case-insensitively wherever the host puts it."""
    assert is_mistral_model_id(model_id)


def test_non_mistral_model_ids_do_not_match() -> None:
    """Other families never trigger the rule."""
    assert not is_mistral_model_id("deepseek-v4-flash")


def test_greedy_mistral_request_sends_top_p_one_with_disclosure() -> None:
    """Temperature 0 with top_p 0.95 dispatches top_p 1 instead of a Mistral 400."""
    profile = _profile("mistral-large-4")
    public, provider = route_generation_parameter_requests(
        (profile,), _request(temperature=0, top_p=0.95)
    )

    assert MISTRAL_GREEDY_TOP_P in public.ignored_parameters
    assert public.top_p == 0.95
    payload = dialect_stream_payload(profile, provider)
    assert payload["temperature"] == 0
    assert payload["top_p"] == 1.0


@pytest.mark.parametrize(
    ("model_id", "temperature", "top_p"),
    (
        ("mistral-large-4", 0.3, 0.95),
        ("mistral-large-4", 0, 1.0),
        ("mistral-large-4", 0, None),
        ("deepseek-v4-flash", 0, 0.95),
    ),
)
def test_rule_leaves_other_requests_verbatim(
    model_id: str, temperature: float, top_p: float | None
) -> None:
    """Sampled requests, top_p already 1, absent top_p, and other families are untouched."""
    profile = _profile(model_id)
    public, provider = route_generation_parameter_requests(
        (profile,), _request(temperature=temperature, top_p=top_p)
    )

    assert MISTRAL_GREEDY_TOP_P not in public.ignored_parameters
    assert provider.top_p == top_p
