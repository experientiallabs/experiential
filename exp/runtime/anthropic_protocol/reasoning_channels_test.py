"""Tests for the Messages reasoning-channel resolution."""

from __future__ import annotations

import pytest

from exp.runtime.anthropic_protocol.reasoning_channels import (
    REASONING_EXCLUDE_DISCLOSURE,
    REASONING_SUPERSEDES_OUTPUT_CONFIG_DISCLOSURE,
    REASONING_SUPERSEDES_THINKING_DISCLOSURE,
    ReasoningConfig,
    output_config_effort,
    resolve_reasoning_channels,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


def test_absent_reasoning_keeps_the_anthropic_channels_verbatim() -> None:
    """Without the OpenRouter object the surface is exactly Anthropic's."""
    channels = resolve_reasoning_channels(
        None,
        max_tokens=128,
        thinking={"type": "enabled", "budget_tokens": 1024},
        output_config={"effort": "low"},
    )
    assert channels.effort == "low"
    assert channels.effort_parameter is None
    assert channels.thinking_config == {"type": "enabled", "budget_tokens": 1024}
    assert channels.output_config == {"effort": "low"}
    assert channels.disclosures == ()
    assert output_config_effort({"effort": "hyperdrive"}) is None


def test_budget_form_becomes_a_budgeted_thinking_config_by_tier() -> None:
    """A token budget forwards as ``enabled`` thinking and maps to the nearest tier."""
    channels = resolve_reasoning_channels(
        ReasoningConfig(max_tokens=4096), max_tokens=8192, thinking=None, output_config=None
    )
    assert channels.thinking_config == {"type": "enabled", "budget_tokens": 4096}
    assert channels.effort == "low"
    with pytest.raises(OpenAIProtocolError) as oversized:
        resolve_reasoning_channels(
            ReasoningConfig(max_tokens=8192), max_tokens=8192, thinking=None, output_config=None
        )
    assert oversized.value.detail.param == "reasoning.max_tokens"


def test_explicit_effort_supersedes_the_anthropic_channels_with_disclosure() -> None:
    """Both channels present: the OpenRouter effort wins and every drop is disclosed."""
    channels = resolve_reasoning_channels(
        ReasoningConfig(effort="high", exclude=True),
        max_tokens=128,
        thinking={"type": "adaptive"},
        output_config={"effort": "low", "format": {"type": "text"}},
    )
    assert channels.effort == "high"
    assert channels.effort_parameter == "reasoning.effort"
    assert channels.thinking_config is None
    assert channels.output_config == {"format": {"type": "text"}}
    assert channels.disclosures == (
        REASONING_SUPERSEDES_THINKING_DISCLOSURE,
        REASONING_SUPERSEDES_OUTPUT_CONFIG_DISCLOSURE,
        REASONING_EXCLUDE_DISCLOSURE,
    )
    # An output_config reduced to its effort alone drops entirely.
    alone = resolve_reasoning_channels(
        ReasoningConfig(effort="high"),
        max_tokens=128,
        thinking=None,
        output_config={"effort": "low"},
    )
    assert alone.output_config is None
