"""Tests for the per-deployment capability-parity export."""

from __future__ import annotations

from exp.common.models.catalog import GatewayDeploymentCapabilities
from exp.runtime.models.providers.capability_parity import (
    CAPABILITY_PARITY_SCHEMA_VERSION,
    deployment_capability_parity,
)


def test_parity_row_joins_declaration_with_engine_ground_truth() -> None:
    """Declared gateway capabilities merge with family effort ground truth."""
    row = deployment_capability_parity(
        provider="openai",
        model_id="gpt-5.6-sol",
        dialect="openai_responses",
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_developer_messages=True,
            supports_strict_tools=True,
            supports_structured_text=True,
        ),
        reasoning_wire_format="openai_responses",
    )
    assert row.schema_version == CAPABILITY_PARITY_SCHEMA_VERSION
    assert row.supports_strict_tools is True
    assert row.supports_stop_sequences is False
    # Provider-verified gpt-5.6 ladder, from the engine's family table.
    assert row.reasoning_efforts == (
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )
    assert row.thinking_config_support == "none"


def test_parity_row_reports_thinking_generations_and_declared_ladders() -> None:
    """Anthropic rows carry the thinking generation; declared ladders win."""
    adaptive = deployment_capability_parity(
        provider="anthropic",
        model_id="claude-fable-5",
        dialect="anthropic_messages",
        capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        reasoning_wire_format="anthropic_adaptive",
    )
    assert adaptive.thinking_config_support == "adaptive"
    assert "max" in adaptive.reasoning_efforts

    budgeted = deployment_capability_parity(
        provider="anthropic",
        model_id="claude-haiku-4-5",
        dialect="anthropic_messages",
        capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        reasoning_wire_format="anthropic_adaptive",
    )
    assert budgeted.thinking_config_support == "enabled"

    declared = deployment_capability_parity(
        provider="openrouter",
        model_id="vendor/custom-model",
        dialect="openai_compatible",
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supported_reasoning_efforts=("low", "high"),
        ),
        reasoning_wire_format="reasoning",
    )
    # An explicit catalog declaration overrides family ground truth.
    assert declared.reasoning_efforts == ("low", "high")

    unknown = deployment_capability_parity(
        provider="local",
        model_id="mystery-model",
        dialect="openai_compatible",
        capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        reasoning_wire_format="reasoning_effort",
    )
    # No declaration and no family ground truth: the row shows the gap.
    assert unknown.reasoning_efforts == ()


def test_parity_row_forwards_pdf_urls_only_on_a_fetching_dialect() -> None:
    """A PDF URL declaration counts only where the wire itself fetches the URL."""
    declared = GatewayDeploymentCapabilities(supports_pdf_input=True, supports_pdf_url_input=True)
    fetching = deployment_capability_parity(
        provider="anthropic",
        model_id="claude-fixture",
        dialect="anthropic_messages",
        capabilities=declared,
        reasoning_wire_format="anthropic_thinking",
    )
    inline_only = deployment_capability_parity(
        provider="gemini",
        model_id="gemini-fixture",
        dialect="gemini_generate_content",
        capabilities=declared,
        reasoning_wire_format="none",
    )
    text_only = deployment_capability_parity(
        provider="openai",
        model_id="gpt-fixture",
        dialect="openai_responses",
        capabilities=GatewayDeploymentCapabilities(supports_pdf_url_input=True),
        reasoning_wire_format="openai_responses",
    )
    assert (fetching.supports_pdf_input, fetching.forwards_pdf_urls) == (True, True)
    assert (inline_only.supports_pdf_input, inline_only.forwards_pdf_urls) == (True, False)
    assert (text_only.supports_pdf_input, text_only.forwards_pdf_urls) == (False, False)
