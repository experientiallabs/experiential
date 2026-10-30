"""Missing state and matching mistakes cannot turn into positive competence evidence."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.evaluations.actor_equivalence import RUBRIC_DIMENSIONS, ActorEquivalenceReport
from exp.common.evaluations.reconstruction import AxisOutcome
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.admission import ContinuationAdmission
from exp.optimize.evaluation.continuations.admission_test import admitted_workbook_study
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.equivalence.evidence import (
    PairMeasurement,
    equivalence_metrics,
    measure_pair,
)
from exp.simulation.continuations import load_continuation_checkpoint


@pytest.mark.parametrize(
    "fault",
    [
        "missing_state",
        "changed_initial",
        "changed_request",
        "matching_failure",
        "interrupted",
        "candidate_failed",
    ],
)
def test_material_evidence_cannot_be_inferred_from_matching_final_prose(
    tmp_path: Path, fault: str
) -> None:
    """Alter saved evidence locally and check that a common 'done' answer cannot rescue it."""
    project = ProjectStore(tmp_path, "case")
    ref = admitted_workbook_study(project)
    admission = ContinuationAdmission.model_validate_json(
        project.artifacts.read_bytes(ref.artifact_id, "admission.json")
    )
    study = load_study(project.artifacts, admission.study)
    checkpoint = load_continuation_checkpoint(project.artifacts, study.spec.checkpoint)
    reference = study.branches[0]
    outcome = reference.model_dump(mode="json")["outcome"]
    if fault == "missing_state":
        outcome.pop("final_state")
    elif fault == "changed_initial":
        outcome["initial_state"]["input_digest"] = "different"
    elif fault == "changed_request":
        outcome["model_calls"][0]["request"]["messages"][0]["content"] = "different task"
    elif fault == "matching_failure":
        outcome["final_state"]["rows"] = 4
        reference = reference.model_copy(update={"outcome": outcome})
    candidate = reference.model_copy(update={"outcome": outcome})
    if fault == "interrupted":
        candidate = candidate.model_copy(update={"status": "outcome_unknown"})
    if fault == "candidate_failed":
        crashed = {"steps": [], "stop_reason": "failed"}
        candidate = candidate.model_copy(update={"status": "failed", "outcome": crashed})
    measured = measure_pair(
        reference,
        candidate,
        checkpoint=checkpoint,
        actor_sha256=sha256_json(checkpoint.actor),
        verifier=study.spec.verifier,
        initial_paths=("/input_digest",),
        final_paths=("/rows",),
        case=0,
        repeat=0,
        condition="candidate",
        identity="test-pair",
    )
    if fault == "matching_failure":
        assert measured.material_agreement == "match"
        assert measured.comparison.reference_assessment.task_result == "fail"
        metrics = equivalence_metrics((measured,))["candidate"]
        assert isinstance(metrics, dict) and metrics["both_task_pass"] == 0
        pooled = equivalence_metrics(
            (
                measured,
                measured.model_copy(update={"repeat": 1}),
                measured.model_copy(update={"case": 1, "material_agreement": "different"}),
            )
        )
        cases = pooled["by_case"]
        assert isinstance(cases, dict)
        assert cases["0"]["candidate"]["agreement_bounds"] == [1, 1]
        assert cases["1"]["candidate"]["agreement_bounds"] == [0, 0]
        headline = pooled["candidate"]
        assert isinstance(headline, dict) and headline["agreement_bounds"] == [0.5, 0.5]
        assert headline["agreement"]["estimate"] == 0.5
    elif fault == "candidate_failed":
        assert measured.material_agreement == "different"
        assert "different:completed_execution" in measured.differences
        metrics = equivalence_metrics((measured,))["candidate"]
        assert isinstance(metrics, dict)
        assert metrics["material_differences"] == 1 and metrics["unassessable"] == 0
        assert metrics["agreement_bounds"] == [0, 0]
    else:
        assert measured.material_agreement == "unassessable"
        assert measured.differences
    assert measured.behavior_assessable == (fault in {"missing_state", "matching_failure"})
    if fault == "interrupted":
        assert measured.comparison.candidate_assessment.task_result == "unassessable"
        assert measured.comparison.candidate.stop_reason == "outcome_unknown"


def _pair(case: int, repeat: int, condition: str, agreement: str) -> PairMeasurement:
    """Build a measurement row with only the fields the aggregate reads."""
    return PairMeasurement.model_construct(
        case=case,
        repeat=repeat,
        condition=condition,
        material_agreement=agreement,
        comparison=_COMPARISON,
        behavior_assessable=False,
        differences=(),
    )


class _Assessment:
    """Task result stub for the pair matrix."""

    task_result = "pass"


class _Comparison:
    """Comparison stub carrying both assessments."""

    reference_assessment = _Assessment()
    candidate_assessment = _Assessment()


_COMPARISON = _Comparison()


def test_cases_weigh_equally_and_control_agreement_is_subtracted() -> None:
    """Many repeats of one case cannot outvote another case, and control noise is removed."""
    rows = (
        *(_pair(0, repeat, "candidate", "match") for repeat in range(8)),
        _pair(1, 0, "candidate", "different"),
        _pair(1, 1, "candidate", "unassessable"),
        *(_pair(0, repeat, "reference_control", "match") for repeat in range(8)),
        _pair(1, 0, "reference_control", "match"),
        _pair(1, 1, "reference_control", "different"),
    )
    metrics = equivalence_metrics(rows)
    candidate = metrics["candidate"]
    assert isinstance(candidate, dict)
    assert candidate["planned_pairs"] == 10
    assert candidate["agreement"]["estimate"] == 0.5
    assert candidate["agreement"]["independent_units"] == 2
    assert candidate["agreement_bounds"] == [0.5, 0.75]
    difference = metrics["candidate_minus_control"]
    assert isinstance(difference, dict)
    assert difference["unit_differences"] == {"0": 0.0, "1": -0.5}
    assert difference["estimate"] == -0.25 and difference["independent_units"] == 2
    low, high = difference["wilson_95"]
    assert -1 <= low < -0.25 < high <= 1
    cases = metrics["by_case"]
    assert isinstance(cases, dict) and cases["1"]["candidate_minus_control"] == -0.5


def test_rubric_ratings_average_repeats_within_a_case_first() -> None:
    """A case with more repeats cannot dominate a criterion's mean rating."""

    def judged(case: int, repeat: int, score: int | None) -> PairMeasurement:
        """One pair whose judge rated every criterion, or left it unassessable."""
        axes = tuple(
            AxisOutcome(
                dimension_id=name,
                status="assessed" if score is not None else "unassessable",
                score=score,
                rationale="fixture",
                evidence_refs=("A:final",) if score is not None else (),
            )
            for name in RUBRIC_DIMENSIONS
        )
        report = ActorEquivalenceReport(
            comparison_sha256="a" * 64,
            judge_sha256="b" * 64,
            status="completed",
            axes=axes,
            score_100=None,
        )
        pair = _pair(case, repeat, "candidate", "match")
        return pair.model_copy(update={"judgment": report, "behavior_assessable": True})

    rows = (judged(0, 0, 5), judged(0, 1, 5), judged(0, 2, 5), judged(1, 0, 1), judged(2, 0, None))
    metrics = equivalence_metrics(rows)["candidate"]
    assert isinstance(metrics, dict)
    goal = metrics["rubric_dimensions"]["goal"]
    assert goal["case_mean_score_1_to_5_not_accuracy"] == 3
    assert (goal["assessed"], goal["unassessable"], goal["cases_assessed"]) == (4, 1, 2)
    assert metrics["behavior_assessable_pairs"] == 5
