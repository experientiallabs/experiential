"""Tests for the judge-visible rollout evidence projection and output-token budgets."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from exp.common.core.artifacts import SourceIdentity
from exp.common.judging.evidence import (
    DEFAULT_JUDGE_OUTPUT_TOKENS,
    visible_rollout_evidence,
)
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
)
from exp.common.rollouts import (
    ProductionSimulatorSnapshot,
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    SimulationMode,
    StopReason,
)

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 11, tzinfo=UTC)


def _model() -> ModelSnapshot:
    """Return one deterministic candidate model snapshot."""
    return ModelSnapshot(
        provider="fake",
        model_id="candidate-model",
        billing_source=BillingSource.HOST_MANAGED,
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )


def _span(span_id: str, kind: RolloutEventKind, payload: dict[str, object]) -> RolloutSpan:
    """Return one rollout span with fixed timestamps and the given payload."""
    return RolloutSpan.model_validate(
        {
            "span_id": span_id,
            "kind": kind.value,
            "started_at": _TIME.isoformat(),
            "ended_at": _TIME.isoformat(),
            "payload": payload,
        }
    )


def _rollout(spans: tuple[RolloutSpan, ...]) -> RolloutArtifact:
    """Return one verified rollout artifact wrapping the given spans."""
    return RolloutArtifact(
        schema_version=1,
        created_at=_TIME,
        code_revision="rollout-revision",
        artifact_id="artifact-rollout-1",
        simulation_id="simulation-1",
        cell_id="cell-1",
        mode=SimulationMode.WORLD_MODEL,
        rollout_id="rollout-1",
        trace_id="trace-1",
        evidence_source="production",
        source_run_id="production-run-1",
        task_id="task-1",
        candidate=_model(),
        agent_id="support-agent",
        simulator=ProductionSimulatorSnapshot(
            source=SourceIdentity(kind="production", source_id="trace-source", sha256=_DIGEST)
        ),
        spans=spans,
        repeat=0,
        final_output=AssistantAction(
            content="The refund is complete.",
            tool_calls=(ToolCall(call_id="call-2", name="notify", arguments={"ok": True}),),
        ),
        stop_reason=StopReason.COMPLETED,
        candidate_economics=OperationEconomics(),
    )


def _multi_turn_rollout() -> RolloutArtifact:
    """Return a rollout whose spans carry request histories, reasoning, and tool evidence."""
    first_request = {
        "messages": [
            {"role": "system", "content": "You are a refund agent."},
            {"role": "user", "content": "Refund order 42."},
        ],
    }
    second_request = {
        "messages": [
            {"role": "system", "content": "You are a refund agent."},
            {"role": "user", "content": "Refund order 42."},
            {"role": "assistant", "content": "Looking up order 42."},
            {"role": "tool", "content": "order 42 found"},
        ],
    }
    return _rollout(
        (
            _span(
                "span-1",
                RolloutEventKind.AGENT_MODEL_CALL,
                {
                    "request": first_request,
                    "response": {
                        "output": {
                            "content": "Looking up order 42.",
                            "tool_calls": [
                                {"call_id": "call-1", "name": "lookup", "arguments": {"id": 42}}
                            ],
                        },
                        "finish_reason": "tool_call",
                        "reasoning": "secret chain of thought",
                    },
                },
            ),
            _span(
                "span-2",
                RolloutEventKind.OBSERVATION,
                {"content": "order 42 found", "is_error": False, "metadata": {}},
            ),
            _span(
                "span-3",
                RolloutEventKind.AGENT_MODEL_CALL,
                {
                    "request": second_request,
                    "response": {
                        "output": {"content": "The refund is complete.", "tool_calls": []},
                        "finish_reason": "stop",
                    },
                },
            ),
            _span(
                "span-4",
                RolloutEventKind.MESSAGE,
                {
                    "name": "chat production-model",
                    "attributes": {
                        "gen_ai.prompt": "Refund order 42.",
                        "gen_ai.input.messages": [{"role": "user", "content": "Refund order 42."}],
                        "gen_ai.completion": "The refund is complete.",
                    },
                },
            ),
        )
    )


def test_default_budget_is_sixteen_k() -> None:
    """The shared judge output-token budget is 16384."""
    assert DEFAULT_JUDGE_OUTPUT_TOKENS == 16_384


def test_visible_evidence_excludes_requests_and_reasoning() -> None:
    """No provider request history or reasoning content reaches the judge payload."""
    rendered = json.dumps(
        visible_rollout_evidence(_multi_turn_rollout()), ensure_ascii=False, sort_keys=True
    )
    assert '"request"' not in rendered
    assert '"gen_ai.input.messages"' not in rendered
    assert '"reasoning"' not in rendered
    assert "secret chain of thought" not in rendered
    assert "Looking up order 42." in rendered
    assert rendered.count("You are a refund agent.") == 1


def test_visible_evidence_retains_visible_outputs_tools_and_final_output() -> None:
    """Task framing, visible outputs, tool calls, tool responses, and final output remain."""
    evidence = visible_rollout_evidence(_multi_turn_rollout())
    assert evidence["rollout_id"] == "rollout-1"
    assert evidence["task_id"] == "task-1"
    assert evidence["stop_reason"] == StopReason.COMPLETED.value
    assert evidence["task_context"] == [
        {"role": "system", "content": "You are a refund agent."},
        {"role": "user", "content": "Refund order 42."},
    ]
    final_output = evidence["final_output"]
    assert isinstance(final_output, dict)
    assert final_output["content"] == "The refund is complete."
    spans = evidence["spans"]
    assert isinstance(spans, list)
    first = spans[0]
    assert isinstance(first, dict)
    first_payload = first["payload"]
    assert isinstance(first_payload, dict)
    response = first_payload["response"]
    assert isinstance(response, dict)
    output = response["output"]
    assert isinstance(output, dict)
    assert output["tool_calls"] == [
        {"call_id": "call-1", "name": "lookup", "arguments": {"id": 42}}
    ]
    second = spans[1]
    assert isinstance(second, dict)
    assert second["payload"] == {"content": "order 42 found", "is_error": False, "metadata": {}}
    production = spans[3]
    assert isinstance(production, dict)
    production_payload = production["payload"]
    assert isinstance(production_payload, dict)
    attributes = production_payload["attributes"]
    assert isinstance(attributes, dict)
    assert attributes["gen_ai.prompt"] == "Refund order 42."
    assert attributes["gen_ai.completion"] == "The refund is complete."


def test_visible_evidence_renders_deterministically() -> None:
    """Repeated projections of one immutable rollout serialize identically."""
    first = json.dumps(
        visible_rollout_evidence(_multi_turn_rollout()), ensure_ascii=False, sort_keys=True
    )
    second = json.dumps(
        visible_rollout_evidence(_multi_turn_rollout()), ensure_ascii=False, sort_keys=True
    )
    assert first == second


def test_task_context_is_empty_without_recorded_requests() -> None:
    """Rollouts without provider request payloads yield an empty task framing list."""
    rollout = _rollout(
        (
            _span(
                "span-1",
                RolloutEventKind.MESSAGE,
                {"text": "Agent addressed the refund request."},
            ),
        )
    )
    assert visible_rollout_evidence(rollout)["task_context"] == []
