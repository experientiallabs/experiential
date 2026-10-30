"""Specification probes expose false passes without fabricating native execution."""

from exp.common.core.artifacts import JsonObject
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier
from exp.simulation.evaluation.verifier_probes import probe_verifier


def test_header_and_extra_effect_counterexamples_survive_as_synthetic_evidence() -> None:
    """A value-only verifier cannot certify preserved headers, types or object inventory."""
    observation: JsonObject = {
        "steps": [],
        "stop_reason": "completed",
        "final_answer": "done",
        "final_state": {"rows": [["key"], [1]]},
    }
    weak = TaskVerifier(
        description="Weak row goal",
        outcome_schema={},
        rules=(OutcomeRule(path="/final_state/rows/1/0", operator="equals", value=1),),
    )
    strong = TaskVerifier(
        description="Exact goal",
        outcome_schema={},
        rules=(
            OutcomeRule(path="/final_state", operator="equals", value=observation["final_state"]),
        ),
    )
    report = probe_verifier(weak, strong, observation)
    assert report.authority == "synthetic_specification_probe"
    assert any(row.path == ("rows", 0, 0) for row in report.false_passes)
    assert any(row.path == () for row in report.false_passes)
    assert not probe_verifier(strong, strong, observation).false_passes
    assert probe_verifier(weak, strong, observation, maximum=1).truncated


def test_strict_schema_const_rejects_a_numeric_type_counterexample() -> None:
    """A schema const on 5 no longer accepts 5.0, so the type probe finds no false pass."""
    state: JsonObject = {"amount": 5}
    observation: JsonObject = {
        "steps": [],
        "stop_reason": "completed",
        "final_answer": "done",
        "final_state": state,
    }
    candidate = TaskVerifier(
        description="Numeric const",
        outcome_schema={
            "type": "object",
            "required": ["final_state"],
            "properties": {"final_state": {"const": state}},
        },
    )
    independent = TaskVerifier(
        description="Typed goal",
        outcome_schema={},
        rules=(OutcomeRule(path="/final_state", operator="equals", value=state),),
    )
    report = probe_verifier(candidate, independent, observation)
    assert not any(
        row.path == ("amount",) and type(row.replacement) is float for row in report.false_passes
    )
