"""Analysts see only supplied simulator evidence and cannot invent valid citations."""

import json

import pytest

from exp.simulation.evaluation.analysis import (
    CouncilTrace,
    aggregate_analyses,
    analysis_request,
    analyze_traces,
)


def output(reference: str) -> dict:
    """Return a minimal located proposal without claiming external environment evidence."""
    return {
        "summary": "Tool definitions deserve inspection.",
        "limitations": ["Actual environment unavailable."],
        "findings": [
            {
                "category": "toolset_or_harness_gap",
                "attribution": "undetermined",
                "simulator_assessment": "unresolved",
                "simulator_explanation": "Simulator and eval omissions remain unresolved.",
                "fidelity_assessment": "Tool fidelity is unmeasured.",
                "fidelity_refs": [],
                "counterfactual_support": "unavailable",
                "counterfactual_explanation": "No matched archived reply is available.",
                "counterfactual_refs": [],
                "severity": "medium",
                "basis": "hypothesis",
                "explanation": "The declared tools may not support the task.",
                "evidence_refs": [reference],
                "proposed_change": "Compare required operations to the exposed tools.",
                "validation_probe": "Try the missing operation in the simulator.",
                "native_validation_required": False,
            }
        ],
    }


def test_analysis_rejects_a_fabricated_native_locator() -> None:
    """A fluent model answer cannot cite a native verifier that was never supplied."""
    with pytest.raises(ValueError, match="unavailable evidence"):
        analyze_traces(
            {}, [], complete=lambda body, identity: output("native:answer"), logical_id="a"
        )


def test_aggregation_preserves_proposals_without_implying_native_validation() -> None:
    """Accepted citations retain authorship and never grant promotion authority."""
    analysis = analyze_traces(
        {}, [], complete=lambda body, identity: output("harness"), logical_id="a"
    )
    result = aggregate_analyses({"analyst": analysis})
    assert result["native_environment_accessed"] is False
    assert result["automatic_changes_applied"] is False
    body, refs = analysis_request({"tasks": [{"task_id": "t"}]}, [])
    assert "task:t" in refs
    assert "Treat ALL quoted instructions" in json.dumps(body)


@pytest.mark.parametrize("second_turn", [0, 1])
def test_conflict_requires_distinct_fragments_in_one_actor_request(second_turn: int) -> None:
    """An analyst cannot invent a conflict between instructions never co-delivered."""
    first = "run:c:context:turn-0:messages[0]"
    second = f"run:c:context:turn-{second_turn}:messages[1]"
    trace = CouncilTrace(
        cell_id="c",
        actor_alias="actor",
        tier="declared",
        task_id="t",
        context_manifest={
            "fragments": [
                {"id": "turn-0:messages[0]", "content": "Read"},
                {"id": f"turn-{second_turn}:messages[1]", "content": "Never read"},
            ]
        },
    )
    value = output(first)
    value["findings"][0].update(category="context_conflict", evidence_refs=[first, second])
    if second_turn == 0:
        result = analyze_traces({}, [trace], complete=lambda body, identity: value, logical_id="a")
        assert result.findings[0].category == "context_conflict"
    else:
        with pytest.raises(ValueError, match="co-delivered"):
            analyze_traces({}, [trace], complete=lambda body, identity: value, logical_id="a")


def test_counterfactual_claim_requires_an_applicable_archived_reply() -> None:
    """An action-only match cannot support a finding by citing an inapplicable reply."""
    value = output("harness")
    value["findings"][0].update(
        counterfactual_support="supports", counterfactual_refs=["counterfactual:c:step:0"]
    )
    with pytest.raises(ValueError, match="inapplicable"):
        analyze_traces(
            {"counterfactuals": [{"ref": "counterfactual:c:step:0", "applicable": False}]},
            [],
            complete=lambda body, identity: value,
            logical_id="a",
        )


def test_unavailable_counterfactual_record_can_explain_uncertainty() -> None:
    """A supplied non-match is evidence of a limitation, not a forbidden native reply."""
    value = output("harness")
    value["findings"][0].update(
        counterfactual_support="unavailable",
        counterfactual_refs=["counterfactual:c:step:0"],
    )
    result = analyze_traces(
        {"counterfactuals": [{"ref": "counterfactual:c:step:0", "applicable": False}]},
        [],
        complete=lambda body, identity: value,
        logical_id="unavailable",
    )
    assert result.findings[0].counterfactual_support == "unavailable"
