"""Tests for the numeric thinking budget to effort translation."""

from __future__ import annotations

import pytest

from exp.common.models.model import ReasoningEffort
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.capability_policy import coerce_generation_parameters
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.generation_route_compat import (
    compatible_generation_parameter_profile_indexes,
)
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests
from exp.runtime.models.providers.thinking_budget_coercion import budget_effort_tier


def _effort_lane(efforts: tuple[ReasoningEffort, ...]) -> GatewayWireProfile:
    """One Experiential Cloud style vLLM lane that reasons through an effort ladder."""
    return GatewayWireProfile(
        dialect="openai_compatible",
        url="https://cloud.experiential.test/v1",
        model_id="qwen3.8-flash-next-abliterated",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
        supported_reasoning_efforts=efforts,
    )


def _serves(profiles: tuple[GatewayWireProfile, ...], request: GatewayRequest) -> bool:
    """Whether a candidate survives narrowing and full route construction."""
    try:
        indexes = compatible_generation_parameter_profile_indexes(profiles, request)
        route_generation_parameter_requests(tuple(profiles[i] for i in indexes), request)
    except ProviderParameterError:
        return False
    return True


def _messages(budget: int) -> GatewayRequest:
    """A Messages request carrying Anthropic's enabled config with a numeric budget."""
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
        provider_thinking_config={"type": "enabled", "budget_tokens": budget},
        maximum_output_tokens=32_000,
    )


@pytest.mark.parametrize(
    ("budget", "tier"),
    ((1_024, "low"), (4_096, "medium"), (16_000, "high"), (31_999, "high")),
)
def test_a_numeric_budget_serves_at_the_nearest_route_tier(
    budget: int, tier: ReasoningEffort
) -> None:
    """qwen3.8-flash-next-uncensored refused every budget by name (27 in two
    hours, production 2026-10-19); the budget now snaps onto the ladder."""
    profiles = (_effort_lane(("low", "medium", "high")),)
    request = _messages(budget)
    assert not _serves(profiles, request)
    coercion = coerce_generation_parameters(
        profiles, request, admits=lambda candidate: _serves(profiles, candidate)
    )
    assert coercion is not None
    assert coercion.request.reasoning_effort == tier
    assert coercion.request.provider_thinking_config is None
    assert coercion.disclosures == (
        f"thinking.budget_tokens->reasoning_effort:{tier}(route_effort_ladder)",
    )


def test_a_chat_thinking_budget_names_its_own_field() -> None:
    """The Chat ``thinking_budget`` control is disclosed under its own name."""
    profiles = (_effort_lane(("low", "high", "max")),)
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="go"),),
        thinking_budget=4_096,
    )
    coercion = coerce_generation_parameters(
        profiles, request, admits=lambda candidate: _serves(profiles, candidate)
    )
    assert coercion is not None
    assert coercion.request.thinking_budget is None
    assert coercion.request.reasoning_effort == "low"
    assert coercion.disclosures == ("thinking_budget->reasoning_effort:low(route_effort_ladder)",)


def test_a_route_without_reasoning_keeps_the_budget_refusal() -> None:
    """No tier to snap to: the named refusal stands."""
    plain = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://plain.test/v1",
        model_id="plain-model",
        supports_reasoning=False,
    )
    request = _messages(4_096)
    coercion = coerce_generation_parameters(
        (plain,), request, admits=lambda candidate: _serves((plain,), candidate)
    )
    assert coercion is None or coercion.request.provider_thinking_config is None
    assert coercion is None or not any(
        "route_effort_ladder" in disclosure for disclosure in coercion.disclosures
    )


def test_budget_tiers_and_special_values() -> None:
    """Zero disables thinking; dynamic thinking takes the lane default depth."""
    lane = _effort_lane(("none", "low", "high"))
    assert budget_effort_tier(0, (lane,)) == "none"
    assert budget_effort_tier(-1, (lane,)) == "medium"
    assert budget_effort_tier(100_000, (lane,)) == "xhigh"
