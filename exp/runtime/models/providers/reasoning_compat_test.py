"""Tests for model-specific reasoning effort normalization."""

import pytest

from exp.common.models.known_models import known_model_metadata
from exp.runtime.models.providers.errors import (
    ProviderParameterError,
    UnsupportedReasoningEffortError,
)
from exp.runtime.models.providers.reasoning_compat import (
    anthropic_adaptive_only_thinking,
    anthropic_reasoning_effort,
    default_reasoning_effort,
    gemini_thinking_level,
    openai_reasoning_effort,
    require_sampling_reasoning_compatibility,
    supported_reasoning_efforts,
)


def test_anthropic_adaptive_efforts_are_never_silently_clamped() -> None:
    """Anthropic receives an exact supported value or a local error."""
    with pytest.raises(UnsupportedReasoningEffortError):
        anthropic_reasoning_effort("claude-opus-5", "minimal")
    assert anthropic_reasoning_effort("claude-opus-5", "xhigh") == "xhigh"
    assert anthropic_reasoning_effort("claude-opus-5", "max") == "max"
    with pytest.raises(UnsupportedReasoningEffortError):
        anthropic_reasoning_effort("claude-sonnet-4-6", "xhigh")


def test_gemini_thinking_levels_follow_exact_model_tables() -> None:
    """Gemini receives only levels its exact current family accepts."""
    with pytest.raises(UnsupportedReasoningEffortError):
        gemini_thinking_level("gemini-3.7-flash", "minimal")
    with pytest.raises(UnsupportedReasoningEffortError):
        gemini_thinking_level("gemini-3.1-pro-preview", "xhigh")
    with pytest.raises(UnsupportedReasoningEffortError):
        gemini_thinking_level("gemini-3-pro-preview", "medium")
    with pytest.raises(UnsupportedReasoningEffortError):
        gemini_thinking_level("gemini-3.1-flash-lite-image", "low")
    assert gemini_thinking_level("gemini-3.6-flash", "minimal") == "minimal"
    with pytest.raises(UnsupportedReasoningEffortError):
        gemini_thinking_level("gemini-2.5-pro", "minimal")
    with pytest.raises(UnsupportedReasoningEffortError):
        gemini_thinking_level("gemini-3.99-unknown", "low")
    assert (
        gemini_thinking_level(
            "publishers/google/models/gemini-2.5-pro",
            "medium",
        )
        == "medium"
    )


def test_openai_reasoning_efforts_follow_exact_model_tables() -> None:
    """Every maintained OpenAI family receives only an accepted effort value."""
    with pytest.raises(UnsupportedReasoningEffortError):
        openai_reasoning_effort("gpt-5-pro", "minimal")
    with pytest.raises(UnsupportedReasoningEffortError):
        openai_reasoning_effort("gpt-5.2-pro", "low")
    assert openai_reasoning_effort("gpt-5.4-pro-2026-03-05", "xhigh") == "xhigh"
    # Provider-verified 2026-08-28: the gpt-5.6 family accepts the full
    # seven-effort ladder (and rejects "ultra" by name).
    assert openai_reasoning_effort("gpt-5.6-sol", "minimal") == "minimal"
    assert openai_reasoning_effort("gpt-5.6-sol", "max") == "max"
    with pytest.raises(UnsupportedReasoningEffortError):
        openai_reasoning_effort("gpt-5.6-sol", "ultra")
    with pytest.raises(UnsupportedReasoningEffortError):
        openai_reasoning_effort("gpt-5.5", "minimal")
    with pytest.raises(UnsupportedReasoningEffortError):
        openai_reasoning_effort("gpt-5.1-2025-11-13", "xhigh")
    assert openai_reasoning_effort("gpt-5.1", "none") == "none"
    assert openai_reasoning_effort("gpt-5", "minimal") == "minimal"
    with pytest.raises(UnsupportedReasoningEffortError):
        openai_reasoning_effort("gpt-5-mini", "xhigh")
    with pytest.raises(UnsupportedReasoningEffortError):
        openai_reasoning_effort("o3", "minimal")
    with pytest.raises(UnsupportedReasoningEffortError):
        openai_reasoning_effort("o4-mini", "xhigh")
    with pytest.raises(UnsupportedReasoningEffortError):
        openai_reasoning_effort("openai/gpt-5-pro", "low")
    assert openai_reasoning_effort("gpt-5.6-sol", "xhigh") == "xhigh"
    assert openai_reasoning_effort("third-party-reasoner", "minimal") == "minimal"


@pytest.mark.parametrize(
    "model_id",
    [
        "gpt-5.2",
        "gpt-5.2-2025-12-11",
        "gpt-5.4",
        "gpt-5.4-2026-03-05",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.5",
        "gpt-5.5-2026-04-23",
    ],
)
@pytest.mark.parametrize("wire_format", ["openai_responses", "reasoning_effort"])
def test_gpt_5x_ladders_reach_the_none_sampling_hatch(model_id: str, wire_format: str) -> None:
    """The documented ladder for gpt-5.2/5.4/5.5 starts at "none" on both OpenAI wires.

    Provider-verified 2026-09-03 on direct OpenAI (every model here) and on
    Azure OpenAI (gpt-5.4, which shares the reasoning_effort wire): "none"
    returns zero reasoning tokens and is the only effort at which temperature
    and top_p are honored. The maintained metadata declares that hatch through
    sampling_requires_reasoning_none, so the ladder must contain "none" or the
    declared sampling support is unreachable.

    Args:
        model_id: Pointer or dated snapshot id of one affected model.
        wire_format: Direct OpenAI or Azure OpenAI reasoning wire.
    """
    assert supported_reasoning_efforts(model_id, wire_format) == (
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
    )
    assert openai_reasoning_effort(model_id, "none") == "none"
    with pytest.raises(UnsupportedReasoningEffortError):
        openai_reasoning_effort(model_id, "minimal")
    known = known_model_metadata("openai", model_id)
    assert known is not None
    assert known.supports_temperature is True
    assert known.supports_top_p is True
    assert known.sampling_requires_reasoning_none is True


@pytest.mark.parametrize("model_id", ["gpt-5.2-pro", "gpt-5.4-pro", "gpt-5.5-pro", "gpt-5"])
def test_gpt_5x_siblings_without_a_documented_none_keep_their_ladders(model_id: str) -> None:
    """The pro tiers and gpt-5 document no "none" effort and expose no sampling hatch.

    Args:
        model_id: One neighbor of the gpt-5.2/5.4/5.5 base models.
    """
    assert "none" not in supported_reasoning_efforts(model_id, "reasoning_effort")
    known = known_model_metadata("openai", model_id)
    assert known is not None
    assert known.supports_temperature is False
    assert known.sampling_requires_reasoning_none is False


def test_sampling_hatch_still_rejects_temperature_at_every_effort_but_none() -> None:
    """Declaring the hatch admits sampling only at exact effort "none".

    Mirrors the provider's live 400 ("'temperature' does not support 0.3 with
    this model") at low and high, and its 200 at none, for gpt-5.2/5.4/5.5.
    """
    for effort in ("low", "medium", "high", "xhigh", None):
        with pytest.raises(ProviderParameterError) as excinfo:
            require_sampling_reasoning_compatibility(
                reasoning_effort=effort,
                sampling_requires_reasoning_none=True,
                temperature_requested=True,
                top_p_requested=False,
            )
        assert excinfo.value.param == "temperature"
    with pytest.raises(ProviderParameterError) as excinfo:
        require_sampling_reasoning_compatibility(
            reasoning_effort="high",
            sampling_requires_reasoning_none=True,
            temperature_requested=False,
            top_p_requested=True,
        )
    assert excinfo.value.param == "top_p"
    require_sampling_reasoning_compatibility(
        reasoning_effort="none",
        sampling_requires_reasoning_none=True,
        temperature_requested=True,
        top_p_requested=True,
    )


def test_exact_effort_support_covers_each_reasoning_wire_family() -> None:
    """Admission sees only values each provider can transmit without clamping."""
    assert supported_reasoning_efforts("gpt-5-pro", "openai_responses") == ("high",)
    assert supported_reasoning_efforts("gpt-5.2-pro", "reasoning_effort") == (
        "medium",
        "high",
        "xhigh",
    )
    assert supported_reasoning_efforts("claude-sonnet-4-6", "anthropic_adaptive") == (
        "low",
        "medium",
        "high",
        "max",
    )
    assert supported_reasoning_efforts("gemini-3-pro-preview", "gemini_thinking") == (
        "low",
        "high",
    )
    assert supported_reasoning_efforts("openai/gpt-5-pro", "reasoning") == ("high",)
    assert supported_reasoning_efforts("anthropic/claude-opus-5", "reasoning") == (
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )
    assert supported_reasoning_efforts("google/gemini-3.6-flash", "reasoning") == (
        "minimal",
        "low",
        "medium",
        "high",
    )


def test_unknown_compatible_model_exposes_only_its_catalog_pin() -> None:
    """Unknown upstream shims cannot silently normalize arbitrary caller efforts."""
    assert supported_reasoning_efforts(
        "vendor/reasoner",
        "reasoning",
        configured_effort="medium",
    ) == ("medium",)
    assert supported_reasoning_efforts("vendor/reasoner", "reasoning") == ()


def test_default_effort_is_always_valid_for_the_exact_model() -> None:
    """Catalog defaults prefer medium, but never pin a level the model rejects."""
    assert default_reasoning_effort("gpt-5.6-sol", "openai_responses") == "medium"
    assert default_reasoning_effort("gpt-5-pro", "openai_responses") == "high"
    assert default_reasoning_effort("gemini-3-pro-preview", "gemini_thinking") == "high"
    assert default_reasoning_effort("vendor/reasoner", "reasoning") == "medium"


def test_adaptive_only_thinking_families_match_the_live_api_boundary() -> None:
    """The adaptive-only set was verified against the live API on 2026-08-28:
    the xhigh generation rejects thinking.type.enabled while the 4.6/4.5 line
    still honors budgeted thinking verbatim."""
    from exp.runtime.models.providers.reasoning_compat import anthropic_adaptive_only_thinking

    for model in (
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4.7",
        "claude-sonnet-5",
    ):
        assert anthropic_adaptive_only_thinking(model), model
    for model in ("claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5", "claude-haiku-4.5"):
        assert not anthropic_adaptive_only_thinking(model), model


def test_anthropic_point_releases_inherit_their_generation_effort_contract() -> None:
    """Generation prefixes match point releases by construction.

    claude-fable-5-1 (launched 2026-09-01, verified live: adaptive-only
    thinking, efforts low through max with ultra rejected by name) must
    resolve claude-fable-5's family without a table edit, and so must the
    next minor.
    """
    for model_id in ("claude-fable-5-1", "claude-fable-5.1", "claude-sonnet-5-2"):
        assert anthropic_adaptive_only_thinking(model_id) is True, model_id
        assert anthropic_reasoning_effort(model_id, "xhigh") == "xhigh"
        assert anthropic_reasoning_effort(model_id, "max") == "max"
    with pytest.raises(UnsupportedReasoningEffortError):
        anthropic_reasoning_effort("claude-fable-5-1", "ultra")
    # Pre-adaptive families stay budgeted.
    assert anthropic_adaptive_only_thinking("claude-haiku-4-5") is False


def test_thinking_budget_maps_onto_the_documented_effort_tiers() -> None:
    """Every band of the thinking-to-effort table maps as documented."""
    from exp.runtime.models.providers.reasoning_compat import thinking_config_reasoning_effort

    assert thinking_config_reasoning_effort({"type": "disabled"}) == "none"
    assert thinking_config_reasoning_effort({"type": "adaptive"}) == "medium"
    assert thinking_config_reasoning_effort({"type": "enabled"}) == "medium"
    assert thinking_config_reasoning_effort({"type": "enabled", "budget_tokens": 1024}) == "low"
    assert thinking_config_reasoning_effort({"type": "enabled", "budget_tokens": 4096}) == "low"
    assert thinking_config_reasoning_effort({"type": "enabled", "budget_tokens": 4097}) == "medium"
    assert thinking_config_reasoning_effort({"type": "enabled", "budget_tokens": 16384}) == "medium"
    assert thinking_config_reasoning_effort({"type": "enabled", "budget_tokens": 16385}) == "high"
    # A boolean is not a budget; the config falls back to the default depth.
    assert thinking_config_reasoning_effort({"type": "enabled", "budget_tokens": True}) == "medium"
