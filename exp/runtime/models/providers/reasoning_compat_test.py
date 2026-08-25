"""Tests for model-specific reasoning effort normalization."""

from exp.runtime.models.providers.reasoning_compat import (
    anthropic_reasoning_effort,
    gemini_thinking_level,
    openai_reasoning_effort,
    supported_reasoning_efforts,
)


def test_anthropic_adaptive_efforts_clamp_only_unsupported_extremes() -> None:
    """Anthropic never receives minimal or 4.6-only xhigh values it rejects."""
    assert anthropic_reasoning_effort("claude-opus-5", "minimal") == "low"
    assert anthropic_reasoning_effort("claude-opus-5", "xhigh") == "xhigh"
    assert anthropic_reasoning_effort("claude-sonnet-4-6", "xhigh") == "high"
    assert anthropic_reasoning_effort("claude-mythos-preview", "xhigh") == "high"


def test_gemini_thinking_levels_follow_exact_model_tables() -> None:
    """Gemini receives the nearest supported level for each current family."""
    assert gemini_thinking_level("gemini-3.7-flash", "minimal") == "low"
    assert gemini_thinking_level("gemini-3.1-pro-preview", "xhigh") == "high"
    assert gemini_thinking_level("gemini-3-pro-preview", "medium") == "high"
    assert gemini_thinking_level("gemini-3.1-flash-lite-image", "low") == "minimal"
    assert gemini_thinking_level("gemini-3.6-flash", "minimal") == "minimal"
    assert gemini_thinking_level("gemini-2.5-pro", "minimal") == "low"
    assert gemini_thinking_level("gemini-3.99-unknown", "low") == "high"


def test_openai_reasoning_efforts_follow_exact_model_tables() -> None:
    """Every maintained OpenAI family receives only an accepted effort value."""
    assert openai_reasoning_effort("gpt-5-pro", "minimal") == "high"
    assert openai_reasoning_effort("gpt-5.2-pro", "low") == "medium"
    assert openai_reasoning_effort("gpt-5.4-pro-2026-03-05", "xhigh") == "xhigh"
    assert openai_reasoning_effort("gpt-5.6-sol", "minimal") == "low"
    assert openai_reasoning_effort("gpt-5.5", "minimal") == "low"
    assert openai_reasoning_effort("gpt-5.1-2025-11-13", "xhigh") == "high"
    assert openai_reasoning_effort("gpt-5", "minimal") == "minimal"
    assert openai_reasoning_effort("gpt-5-mini", "xhigh") == "high"
    assert openai_reasoning_effort("o3", "minimal") == "low"
    assert openai_reasoning_effort("o4-mini", "xhigh") == "high"
    assert openai_reasoning_effort("openai/gpt-5-pro", "low") == "high"
    assert openai_reasoning_effort("gpt-5.6-sol", "xhigh") == "xhigh"
    assert openai_reasoning_effort("third-party-reasoner", "minimal") == "minimal"


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
