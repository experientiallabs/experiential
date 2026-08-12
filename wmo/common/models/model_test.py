"""Tests for canonical model data contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmo.common.models import (
    AssistantAction,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    ToolCall,
    Usage,
)

_CAPABILITIES_DIGEST = "a" * 64


def _snapshot() -> ModelSnapshot:
    return ModelSnapshot(
        provider="openai",
        model_id="gpt-5.4",
        revision="2026-08-11",
        capabilities_sha256=_CAPABILITIES_DIGEST,
    )


def test_model_response_round_trip_preserves_tool_action_and_economics() -> None:
    """A complete model result round trips through its deterministic Pydantic boundary."""
    response = ModelResponse(
        output=AssistantAction(
            content="I created the ticket.",
            tool_calls=(
                ToolCall(call_id="call-1", name="create_ticket", arguments={"priority": "high"}),
            ),
        ),
        model=_snapshot(),
        economics=OperationEconomics(
            usage=Usage(input_tokens=30, output_tokens=12),
            cost_usd=NumericMeasurement(value=0.004, provenance="estimated"),
            latency_seconds=NumericMeasurement(value=1.2, provenance="observed"),
        ),
    )

    assert ModelResponse.model_validate_json(response.model_dump_json()) == response


def test_actions_need_payload_and_measurements_are_finite() -> None:
    """Invalid empty actions and non-finite economics fail at the shared boundary."""
    with pytest.raises(ValidationError, match="content or at least one tool"):
        AssistantAction()
    with pytest.raises(ValidationError, match="finite"):
        NumericMeasurement(value=float("inf"), provenance="observed")
