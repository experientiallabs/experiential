"""Tests for canonical model data contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmo.common.models import (
    AssistantAction,
    Embedding,
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    NumericMeasurement,
    ToolChoice,
)
from wmo.common.tasks import ToolSchema

_CAPABILITIES_DIGEST = "a" * 64


def test_actions_need_payload_and_measurements_are_finite() -> None:
    """Invalid empty actions and non-finite economics fail at the shared boundary."""
    with pytest.raises(ValidationError, match="content or at least one tool"):
        AssistantAction()
    with pytest.raises(ValidationError, match="finite"):
        NumericMeasurement(value=float("inf"), provenance="observed")


def test_model_request_keeps_tool_contract_and_capabilities_deterministic() -> None:
    """A request retains typed tool behavior and rejects incoherent choices.

    The regression also verifies deterministic capability identity hashing.
    """
    tool = ToolSchema(
        name="create_ticket",
        description="Create one support ticket.",
        input_schema={"type": "object"},
    )
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="Please help."),),
        tools=(tool,),
        tool_choice=ToolChoice(name="create_ticket"),
        maximum_output_tokens=512,
    )

    assert request.tools == (tool,)
    assert ModelCapabilities(supports_tools=True).model_dump(mode="json") == {
        "supports_tools": True,
        "supports_embeddings": False,
        "supports_structured_output": False,
        "context_window_tokens": None,
        "maximum_output_tokens": None,
        "input_cost_per_million_tokens_usd": None,
    }
    with pytest.raises(ValidationError, match="named tool_choice"):
        ModelRequest(
            messages=(ModelMessage(role="user", content="Please help."),),
            tool_choice=ToolChoice(name="missing"),
        )
    with pytest.raises(ValidationError, match="required tool_choice"):
        ModelRequest(
            messages=(ModelMessage(role="user", content="Please help."),),
            tool_choice="required",
        )


def test_model_messages_reject_tool_and_assistant_fields_on_the_wrong_roles() -> None:
    """Request-visible messages retain tool linkage without role-crossing payloads."""
    with pytest.raises(ValidationError, match="assistant_action is valid only"):
        ModelMessage(role="user", assistant_action=AssistantAction(content="wrong"))
    with pytest.raises(ValidationError, match="tool messages require tool_call_id"):
        ModelMessage(role="tool", content="missing linkage")


def test_embeddings_require_nonempty_finite_vectors() -> None:
    """Embedding output cannot hide empty or non-finite provider data."""
    assert Embedding(values=(1.0, 0.0)).values == (1.0, 0.0)
    with pytest.raises(ValidationError, match="at least 1 item"):
        Embedding(values=())
    with pytest.raises(ValidationError, match="finite"):
        Embedding(values=(float("nan"),))
    with pytest.raises(ValidationError, match="unit norm"):
        Embedding(values=(0.1, -0.2))
