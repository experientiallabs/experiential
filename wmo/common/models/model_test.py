"""Tests for canonical model data contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import sha256_json
from wmo.common.models import (
    AssistantAction,
    Embedding,
    ModelCapabilities,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    RoutedCandidateSnapshot,
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


def test_completed_factory_prefers_served_identity_and_maps_the_length_limit() -> None:
    """The shared factory keeps served identity, observed latency, and the finish reason."""
    configured = ModelSnapshot(
        provider="openai",
        model_id="configured-model",
        capabilities_sha256=_CAPABILITIES_DIGEST,
        connection_sha256="b" * 64,
    )
    action = AssistantAction(content="Done.")

    served = ModelResponse.completed(
        output=action,
        configured_model=configured,
        served_model_id="served-model",
        usage=None,
        latency_seconds=0.25,
        hit_length_limit=True,
    )
    kept = ModelResponse.completed(
        output=action,
        configured_model=configured,
        served_model_id="",
        usage=None,
        latency_seconds=0.25,
    )

    assert served.model.model_id == "served-model"
    assert served.finish_reason is ModelFinishReason.LENGTH
    assert kept.model.model_id == "configured-model"
    assert kept.finish_reason is ModelFinishReason.COMPLETED
    assert kept.economics.latency_seconds == NumericMeasurement(value=0.25, provenance="observed")


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
        "supports_completions": None,
        "context_window_tokens": None,
        "maximum_output_tokens": None,
        "input_cost_per_million_tokens_usd": None,
        "output_cost_per_million_tokens_usd": None,
        "cached_input_cost_per_million_tokens_usd": None,
        "cache_write_cost_per_million_tokens_usd": None,
    }
    with pytest.raises(ValidationError, match="named tool_choice"):
        ModelRequest(
            messages=(ModelMessage(role="user", content="Please help."),),
            tool_choice=ToolChoice(name="missing"),
        )


def test_completion_support_preserves_provider_identity_for_existing_traces() -> None:
    """Completion eligibility is frozen separately without orphaning old trace snapshots."""
    legacy_payload = {
        "supports_tools": False,
        "supports_embeddings": False,
        "context_window_tokens": None,
        "maximum_output_tokens": None,
    }
    unknown = ModelCapabilities()
    supported = ModelCapabilities(supports_completions=True)
    unsupported = ModelCapabilities(supports_completions=False)

    assert unknown.identity_sha256() == sha256_json(legacy_payload)
    assert supported.identity_sha256() == unsupported.identity_sha256()
    assert supported.identity_sha256() == unknown.identity_sha256()
    with pytest.raises(ValidationError, match="required tool_choice"):
        ModelRequest(
            messages=(ModelMessage(role="user", content="Please help."),),
            tool_choice="required",
        )


def test_legacy_routed_candidate_payload_reserializes_byte_for_byte() -> None:
    """Automatic capability freezing cannot add fields to the v1 candidate contract."""
    payload = {
        "alias": "candidate-a",
        "model": {
            "provider": "openai",
            "model_id": "model-a",
            "revision": None,
            "capabilities_sha256": "a" * 64,
            "connection_sha256": "b" * 64,
        },
    }

    candidate = RoutedCandidateSnapshot.model_validate(payload)

    assert candidate.model_dump(mode="json") == payload
    assert isinstance(candidate.model, ModelSnapshot)


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
