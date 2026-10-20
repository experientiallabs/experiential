"""Specification probes expose false passes without fabricating native execution."""

from exp.common.core.artifacts import JsonObject
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier
from exp.simulation.verifier_probes import probe_verifier


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
