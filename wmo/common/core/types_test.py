"""Tests for the core data types."""

from __future__ import annotations

from wmo.common.core.types import (
    Action,
    ActionKind,
    EnvState,
    Observation,
    Session,
    Step,
    StepAttribution,
    Trace,
)


def test_types_instantiate() -> None:
    action = Action(kind=ActionKind.TOOL_CALL, name="cd", arguments={"path": "/tmp"})
    obs = Observation(content="", is_error=False)
    step = Step(action=action, observation=obs, state_before=EnvState(), task="poke around")
    trace = Trace(trace_id="t1", steps=[step], source="file:demo.jsonl")
    session = Session(id="s1", task="poke around")
    assert trace.steps[0].action.name == "cd"
    assert session.history == []


def test_w05_production_trace_fixture_preserves_behavior_and_provenance() -> None:
    """Map current `Trace` and `Step` to the approved production trace contract.

    This freezes externally meaningful task, action, observation, source, span, model, and cost
    evidence. It does not introduce the target artifact envelope or a second trace representation.
    """
    trace = Trace(
        trace_id="prod-trace-w05-001",
        source="otel:production",
        metadata={"capture": "otel-genai", "domain": "orders", "outcome": "success"},
        steps=[
            Step(
                action=Action(
                    kind=ActionKind.TOOL_CALL,
                    name="lookup_order",
                    arguments={"order_id": "A-42"},
                ),
                observation=Observation(
                    content='{"status":"refunded"}', metadata={"status_code": 200}
                ),
                state_before=EnvState(structured={"account_id": "acct-demo"}),
                task="Refund order A-42",
                raw_span_ids=["span-action-001", "span-tool-001"],
                attribution=StepAttribution(
                    model="candidate-incumbent",
                    provider="openai",
                    input_tokens=12,
                    output_tokens=8,
                    cost_usd=0.0012,
                    latency_ms=240.0,
                    provenance="otel-genai-v1",
                ),
            )
        ],
    )

    assert trace.model_dump(mode="json") == {
        "trace_id": "prod-trace-w05-001",
        "steps": [
            {
                "action": {
                    "kind": "tool_call",
                    "name": "lookup_order",
                    "arguments": {"order_id": "A-42"},
                    "content": None,
                },
                "observation": {
                    "content": '{"status":"refunded"}',
                    "is_error": False,
                    "reward": None,
                    "metadata": {"status_code": 200},
                },
                "state_before": {"structured": {"account_id": "acct-demo"}, "scratchpad": ""},
                "task": "Refund order A-42",
                "raw_span_ids": ["span-action-001", "span-tool-001"],
                "attribution": {
                    "model": "candidate-incumbent",
                    "provider": "openai",
                    "config": {},
                    "input_tokens": 12,
                    "output_tokens": 8,
                    "cost_usd": 0.0012,
                    "latency_ms": 240.0,
                    "error_class": None,
                    "provenance": "otel-genai-v1",
                },
            }
        ],
        "source": "otel:production",
        "metadata": {"capture": "otel-genai", "domain": "orders", "outcome": "success"},
    }
    assert Trace.model_validate_json(trace.model_dump_json()) == trace
