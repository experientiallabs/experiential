"""Tests for the core data types."""

from __future__ import annotations

import wmo
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


def test_every_name_the_package_root_promises_is_importable() -> None:
    """`wmo.__all__` is the documented import path for these types (README quickstart).

    `wmo/__init__.py` has no suite of its own (AGENTS.md rule 2 exempts it), and the failure mode
    lives here: a type renamed or moved in this module leaves a name in `__all__` that
    `from wmo import X` raises AttributeError on, while `import wmo` still succeeds.
    """
    missing = [name for name in wmo.__all__ if not hasattr(wmo, name)]

    assert not missing, f"names in wmo.__all__ that no longer resolve: {missing}"
    assert wmo.ActionKind is ActionKind


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

    step = trace.steps[0]
    assert trace.trace_id == "prod-trace-w05-001"
    assert trace.source == "otel:production"
    assert trace.metadata == {"capture": "otel-genai", "domain": "orders", "outcome": "success"}
    assert step.task == "Refund order A-42"
    assert step.action.kind is ActionKind.TOOL_CALL
    assert step.action.name == "lookup_order"
    assert step.action.arguments == {"order_id": "A-42"}
    assert step.observation.content == '{"status":"refunded"}'
    assert step.observation.is_error is False
    assert step.observation.metadata == {"status_code": 200}
    assert step.state_before.structured == {"account_id": "acct-demo"}
    assert step.raw_span_ids == ["span-action-001", "span-tool-001"]
    assert step.attribution is not None
    assert step.attribution.model == "candidate-incumbent"
    assert step.attribution.provider == "openai"
    assert step.attribution.input_tokens == 12
    assert step.attribution.output_tokens == 8
    assert step.attribution.cost_usd == 0.0012
    assert step.attribution.latency_ms == 240.0
    assert step.attribution.provenance == "otel-genai-v1"
    assert Trace.model_validate_json(trace.model_dump_json()) == trace
