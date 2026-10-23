"""The investigation budget bounds purchases, never whether a trajectory is learned."""

import json
from pathlib import Path

import pytest

from exp.common.evaluations.reconstruction import ReferenceStep
from exp.common.models import ToolCall
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import (
    FidelitySpec,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.fidelity_memory import saved_memory_receipt
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    RoutedTransport,
    investigation,
    judged,
    memory_project,
    predicted,
    trajectory,
)

BUDGET = 24  # the default online-research investigation ceiling


def test_a_trace_with_more_misses_than_the_budget_is_still_learned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Miss 25 is recorded without an investigation; the trajectory is consumed."""
    project, base = memory_project(tmp_path)
    actions = tuple(
        ToolCall(call_id=f"long-{i}", name="inspect", arguments={"key": str(i)})
        for i in range(BUDGET + 1)
    )
    data = trajectory(base, "long-trace", "long", "learn").model_dump(mode="json")
    data["maximum_physical_attempts"] = 200
    data["workload"]["actions"] = [action.model_dump(mode="json") for action in actions]
    data["references"]["steps"] = [
        ReferenceStep(
            step=i,
            action={"name": action.name, "arguments": action.arguments},
            reply_available=True,
            reply=f"actual-{i}",
        ).model_dump(mode="json")
        for i, action in enumerate(actions)
    ]
    spec = FidelitySpec.model_validate(data)
    responses = []
    for i in range(BUDGET + 1):
        responses.append(predicted(f"long-{i}", "wrong"))
        if i < BUDGET:
            responses.append(investigation(str(i)))
    responses.append(judged(spec))
    wire = RoutedTransport(responses)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    plan = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    execute_fidelity_evaluation(project, plan)
    receipt = saved_memory_receipt(project, plan)
    assert receipt is not None and receipt.get("reason") == "consumed", receipt
    assert receipt.get("published") is True
    investigations = sum(
        "hypothesis" in json.dumps(request.payload.get("tools")) for request in wire.requests
    )
    assert investigations == BUDGET, "no investigation is bought past the budget"
