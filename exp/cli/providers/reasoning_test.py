"""Interactive reasoning choices respect the selected model's actual effort contract."""

from dataclasses import replace

import pytest

from exp.cli.providers.model_picker_test import _CHAT, _model
from exp.cli.providers.reasoning import model_reasoning_efforts
from exp.common.models import DiscoveredModel, ModelCapabilities


@pytest.mark.parametrize("provider", ["openai-compatible", "openrouter"])
def test_deepseek_shows_distinct_native_levels_without_compatibility_aliases(provider: str) -> None:
    """A wide Cloud effort contract never presents aliases as additional thinking depths."""
    item = replace(
        _CHAT,
        provider=provider,
        model="deepseek/deepseek-v4.1-flash",
        capabilities=ModelCapabilities(supports_reasoning=True, reasoning_effort="high"),
        supported_reasoning_efforts=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    )

    assert model_reasoning_efforts(item) == ("none", "low", "high", "max")
    assert model_reasoning_efforts(replace(item, supported_reasoning_efforts=("low", "high"))) == (
        "low",
        "high",
    )


def test_published_deployment_choices_survive_without_a_default_effort() -> None:
    """Discovery may advertise a control even when it does not publish a default."""
    item = replace(
        _CHAT,
        model="custom-reasoner",
        provider="openai-compatible",
        capabilities=ModelCapabilities(supports_reasoning=True),
        published=DiscoveredModel(
            provider="openai-compatible",
            model="custom-reasoner",
            supported_reasoning_efforts=("max", "high", "ultra"),
        ),
    )

    assert model_reasoning_efforts(item) == ("high", "max", "ultra")
    assert model_reasoning_efforts(replace(item, supported_reasoning_efforts=())) == ()


def test_known_provider_families_do_not_invent_extra_levels() -> None:
    """OpenAI, Anthropic and Gemini expose their own maintained contracts."""
    assert "ultra" not in model_reasoning_efforts(_CHAT)
    assert model_reasoning_efforts(_model("gpt-5.1")) == ("none", "low", "medium", "high")
    assert model_reasoning_efforts(_model("gemini-3.6-flash", provider="gemini")) == (
        "minimal",
        "low",
        "medium",
        "high",
    )
    assert model_reasoning_efforts(_model("claude-opus-4-6", provider="anthropic")) == (
        "low",
        "medium",
        "high",
        "max",
    )


def test_unknown_provider_exposes_only_its_pin_and_nonreasoners_expose_nothing() -> None:
    """An unknown family does not inherit the global enum as its support contract."""
    item = replace(_CHAT, model="private-model", provider="openai-compatible")
    assert model_reasoning_efforts(item) == ("medium",)
    assert model_reasoning_efforts(replace(item, capabilities=None)) == ()
    assert model_reasoning_efforts(replace(item, capabilities=ModelCapabilities())) == ()
