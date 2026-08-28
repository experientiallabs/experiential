"""Tests for model-specific reasoning effort normalization."""

import pytest

from exp.runtime.models.providers.errors import UnsupportedReasoningEffortError
from exp.runtime.models.providers.reasoning_compat import (
    anthropic_reasoning_effort,
    default_reasoning_effort,
    gemini_thinking_level,
    openai_reasoning_effort,
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
    with pytest.raises(UnsupportedReasoningEffortError):
        openai_reasoning_effort("gpt-5.6-sol", "minimal")
    with pytest.raises(UnsupportedReasoningEffortError):
        openai_reasoning_effort("gpt-5.5", "minimal")
    assert openai_reasoning_effort("gpt-5.5", "none") == "none"
    assert openai_reasoning_effort("gpt-5.6-sol", "max") == "max"
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
    with pytest.raises(UnsupportedReasoningEffortError):
        openai_reasoning_effort("gpt-5.6-pro", "max")
    assert openai_reasoning_effort("third-party-reasoner", "minimal") == "minimal"


def test_exact_effort_support_covers_each_reasoning_wire_family() -> None:
    """Admission sees only values each provider can transmit without clamping."""
    assert supported_reasoning_efforts("gpt-5-pro", "openai_responses") == ("high",)
    assert supported_reasoning_efforts("gpt-5.2-pro", "reasoning_effort") == (
        "medium",
        "high",
        "xhigh",
    )
    assert supported_reasoning_efforts("gpt-5.5", "openai_responses") == (
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
    )
    assert supported_reasoning_efforts("gpt-5.6-luna", "openai_responses") == (
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
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
