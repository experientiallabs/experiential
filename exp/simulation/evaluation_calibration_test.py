"""Calibration cannot turn synthetic labels or leaked cases into expert qualification."""

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.evaluations.actor_rubric import actor_judge
from exp.common.evaluations.reconstruction import AxisOutcome, WorkflowEvidence, WorkflowStep
from exp.simulation.evaluation_calibration import (
    CalibrationCase,
    CalibrationJudgment,
    CalibrationPack,
    blinded_case,
    calibrate,
    task_decision,
)
from exp.simulation.evaluation_tasks import FieldRelation, TaskVerifier, VerifierChallenge


def pack() -> CalibrationPack:
    """Declare four independent synthetic specification cases for software checks only."""
    witness = ArtifactInput(artifact_id="declared-specification", sha256="1" * 64)
    cases = []
    for partition in ("development", "confirmation"):
        for correct in (True, False):
            value: JsonObject = {
                "source_instance": partition,
                "final_answer": {"status": "pending" if correct else "delivered"},
                "observed": {"status": "pending"},
                "stop_reason": "completed",
            }
            cases.append(
                CalibrationCase(
                    challenge=VerifierChallenge(
                        id=f"{partition}-{correct}",
                        outcome=value,
                        expected="pass" if correct else "fail",
                        authority="synthetic_specification",
                        approved=True,
                        rationale="Declared report must agree with its observation",
                        source_refs=(witness,),
                    ),
                    lineage=partition,
                    partition=partition,
                )
            )
    return CalibrationPack(
        task=ArtifactInput(artifact_id="evaluation-task", sha256="2" * 64),
        parent=TaskVerifier(description="Weak", outcome_schema={"type": "object"}),
        candidate=TaskVerifier(
            description="Observed report equality",
            outcome_schema={"type": "object"},
            relations=(FieldRelation(left="/final_answer/status", right="/observed/status"),),
        ),
        cases=tuple(cases),
    )


def test_synthetic_specification_does_not_self_certify_the_judge() -> None:
    """Mechanical agreement is useful but not evidence of expert rubric agreement."""
    report = calibrate(pack())
    assert report["development_status"] == "passed"
    assert report["confirmation_status"] == "passed"
    assert report["rubric_status"] == "provisional"
    assert report["qualification"] == "diagnostic_only"
    assert report["false_acceptances"] == 0
    assert report["physical_attempts"] == 0


def test_wrong_development_candidate_keeps_confirmation_closed() -> None:
    """An exposed failure cannot be rescued by opening reserved labels."""
    original = pack()
    report = calibrate(original.model_copy(update={"candidate": original.parent}))
    assert report["confirmation_status"] == "not_run"
    rows = report["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        if row["partition"] == "confirmation":
            assert row["candidate"] is None and row["expected"] is None


def test_related_variants_cannot_cross_partitions() -> None:
    """Different IDs and values do not make the same source family held out."""
    original = pack()
    cases = original.cases[:2] + tuple(
        case.model_copy(update={"lineage": "development"}) for case in original.cases[2:]
    )
    with pytest.raises(ValidationError, match="related source"):
        CalibrationPack.model_validate({**original.model_dump(), "cases": cases})


def test_blinded_inputs_hide_label_authority_and_expected_verdict() -> None:
    """The teacher receives an outcome, not the human label or its source origin."""
    value = blinded_case(pack().cases[0])
    assert set(value) == {"case_id", "outcome"}
    assert "expected" not in value and "authority" not in value


def test_model_or_synthetic_labels_cannot_be_expert_rubric_scores() -> None:
    """A good software fixture is not a human calibration dataset."""
    case = pack().cases[0]
    with pytest.raises(ValidationError, match="human witnesses"):
        CalibrationCase.model_validate({**case.model_dump(), "expert_scores": {"correctness": 5}})


@pytest.mark.parametrize("score", [True, "5", 6])
def test_expert_scores_cannot_coerce_boolean_text_or_out_of_range_values(score: int | str) -> None:
    """Malformed labels must not become trusted ordinal evidence."""
    original = pack().cases[0]
    data = original.model_dump()
    data["challenge"] = original.challenge.model_copy(update={"authority": "human"})
    data["expert_scores"] = {"outcome": score}
    with pytest.raises(ValidationError):
        CalibrationCase.model_validate(data)


def task_pack() -> CalibrationPack:
    """Separate component validity from task truth in declared software fixtures."""
    original = pack()
    cases = []
    for case in original.cases:
        data = case.model_dump()
        data["expected_task"] = case.challenge.expected
        data["challenge"] = case.challenge.model_copy(update={"expected": "pass"})
        cases.append(CalibrationCase.model_validate(data))
    return CalibrationPack.model_validate(
        {
            **original.model_dump(),
            "candidate": original.parent,
            "cases": tuple(cases),
            "judge": actor_judge("fixture-judge"),
            "required_dimensions": ("outcome",),
            "minimum_scores": {"outcome": 4},
        }
    )


def task_judgments(
    original: CalibrationPack, *, biased: bool = False
) -> tuple[CalibrationJudgment, ...]:
    """Supply explicit model stand-ins, never claim independent expert calibration."""
    assert original.judge is not None
    return tuple(
        CalibrationJudgment(
            case_id=case.challenge.id,
            outcome_sha256=sha256_json(case.challenge.outcome),
            judge_sha256=sha256_json(original.judge),
            judge_completion_sha256=original.judge_completion_sha256,
            report=ArtifactInput(artifact_id="fixture-report", sha256="3" * 64),
            axes=tuple(
                AxisOutcome(
                    dimension_id=dimension.dimension_id,
                    status="assessed" if dimension.dimension_id == "outcome" else "unassessable",
                    score=(5 if biased or case.expected_task == "pass" else 1)
                    if dimension.dimension_id == "outcome"
                    else None,
                    rationale="Declared fixture decision",
                    evidence_refs=("workflow-0",),
                )
                for dimension in original.judge.definition.dimensions
            ),
        )
        for case in original.cases
    )


def test_task_labels_calibrate_semantics_without_manual_numeric_ratings() -> None:
    """Valid envelopes can receive different task decisions without changing their format."""
    original = task_pack()
    report = calibrate(original, task_judgments(original))
    assert report["development_status"] == "passed"
    assert report["confirmation_status"] == "passed"
    assert report["task_false_acceptances"] == 0
    assert report["rubric_status"] == "task_decision_agreement"
    assert report["qualification"] == "diagnostic_only"
    rows = report["rows"]
    assert isinstance(rows, list)
    assert all(isinstance(row, dict) and row["candidate"]["status"] == "pass" for row in rows)
    assert any(isinstance(row, dict) and row["task_decision"] == "fail" for row in rows)


def test_semantic_false_acceptance_keeps_confirmation_closed() -> None:
    """A judge rewarding contradicted answers cannot unlock the reserved panel."""
    original = task_pack()
    report = calibrate(original, task_judgments(original, biased=True))
    assert report["development_status"] == "incomplete_or_incorrect"
    assert report["confirmation_status"] == "not_run"
    assert report["task_false_acceptances"] == 1


def test_missing_judgments_are_not_task_passes() -> None:
    """An available mechanical check cannot stand in for required semantic assessment."""
    report = calibrate(task_pack())
    assert report["development_status"] == "incomplete_or_incorrect"
    rows = report["rows"]
    assert isinstance(rows, list)
    assert rows[0]["task_decision"] == "unassessable"


def test_task_thresholds_must_name_exactly_the_required_criteria() -> None:
    """Neither an omitted criterion nor an extra efficiency reward changes correctness."""
    original = task_pack()
    with pytest.raises(ValidationError, match="thresholds"):
        CalibrationPack.model_validate({**original.model_dump(), "minimum_scores": {"quality": 4}})


def test_efficiency_cannot_compensate_for_incorrectness_or_critical_errors() -> None:
    """A high quality score cannot offset a failed correctness gate or a critical flag."""
    axes = (
        AxisOutcome(
            dimension_id="outcome",
            status="assessed",
            score=1,
            rationale="Wrong observed status",
            evidence_refs=("step-0",),
        ),
        AxisOutcome(
            dimension_id="quality",
            status="assessed",
            score=5,
            rationale="Short execution",
            evidence_refs=("step-0",),
        ),
    )
    assert task_decision("pass", axes, {"outcome": 4}) == "fail"
    corrected = (axes[0].model_copy(update={"score": 5}), axes[1])
    assert task_decision("pass", corrected, {"outcome": 4}) == "pass"
    flagged = (corrected[0], corrected[1].model_copy(update={"critical_flags": ("unsafe_action",)}))
    assert task_decision("pass", flagged, {"outcome": 4}) == "fail"


def test_saved_judge_result_cannot_change_case_or_prompt_binding() -> None:
    """Source labels do not authorize using a score for another case or rubric."""
    original = task_pack()
    judgments = task_judgments(original)
    stale = judgments[0].model_copy(update={"judge_sha256": "9" * 64})
    with pytest.raises(ValueError, match="frozen case or judge binding"):
        calibrate(original, (stale,) + judgments[1:])


def test_selection_failure_keeps_final_confirmation_closed() -> None:
    """Authoring success cannot override an independent selection failure."""
    original = task_pack()
    selection = []
    for case in original.cases[:2]:
        outcome = dict(case.challenge.outcome)
        outcome["source_instance"] = "selection"
        selection.append(
            CalibrationCase.model_validate(
                {
                    **case.model_dump(),
                    "partition": "selection",
                    "lineage": "selection",
                    "challenge": {
                        **case.challenge.model_dump(),
                        "id": "selection-" + case.challenge.id,
                        "outcome": outcome,
                    },
                }
            )
        )
    expanded = CalibrationPack.model_validate(
        {**original.model_dump(), "cases": original.cases + tuple(selection)}
    )
    judgments = task_judgments(expanded)
    assert expanded.judge is not None
    biased = tuple(
        judgment.model_copy(
            update={
                "axes": tuple(
                    axis.model_copy(update={"score": 5}) if axis.dimension_id == "outcome" else axis
                    for axis in judgment.axes
                )
            }
        )
        if judgment.case_id.startswith("selection-")
        else judgment
        for judgment in judgments
    )
    result = calibrate(expanded, biased)
    assert result["development_status"] == "passed"
    assert result["selection_status"] == "incomplete_or_incorrect"
    assert result["confirmation_status"] == "not_run"
    rows = result["rows"]
    assert isinstance(rows, list)
    assert all(row["candidate"] is None for row in rows if row["partition"] == "confirmation")


def test_source_outcome_labels_remain_explicit() -> None:
    """Published reward agreement cannot silently become complete task-correctness evidence."""
    original = task_pack()
    scoped = original.model_copy(
        update={
            "cases": tuple(
                case.model_copy(update={"task_label_scope": "source_outcome"})
                for case in original.cases
            )
        }
    )
    result = calibrate(scoped, task_judgments(scoped))
    assert result["qualification"] == "diagnostic_only"
    rows = result["rows"]
    assert isinstance(rows, list)
    assert all(
        isinstance(row, dict) and row["task_label_scope"] == "source_outcome" for row in rows
    )


def test_source_reward_scope_prevents_an_otherwise_qualified_fixture() -> None:
    """Exercise qualification mechanics with fixtures, not claimed native reference evidence."""
    original = task_pack()
    cases = []
    for case in original.cases:
        assert isinstance(case.challenge.outcome, dict)
        evidence = WorkflowEvidence(
            workflow_id=case.challenge.id,
            task="Report the observed status",
            initial_state={},
            steps=(
                WorkflowStep(
                    step=0, action={"name": "read"}, status="valid", reply={"status": "pending"}
                ),
            ),
            final_answer=case.challenge.outcome["final_answer"],
            sources={"frozen_profile_sha256": "4" * 64},
            stop_reason="completed",
        )
        cases.append(
            case.model_copy(
                update={
                    "challenge": case.challenge.model_copy(
                        update={
                            "outcome": evidence,
                            "authority": "recorded_reference",
                        }
                    )
                }
            )
        )
    bound = original.model_copy(
        update={
            "cases": tuple(cases),
            "frozen_profile_sha256": "4" * 64,
            "judge_completion_sha256": "7" * 64,
        }
    )
    assert calibrate(bound, task_judgments(bound))["qualification"] == "qualified_in_scope"
    unbound = bound.model_copy(update={"judge_completion_sha256": None})
    assert calibrate(unbound, task_judgments(unbound))["qualification"] == "diagnostic_only"
    source_only = bound.model_copy(
        update={
            "cases": tuple(
                case.model_copy(update={"task_label_scope": "source_outcome"})
                for case in bound.cases
            )
        }
    )
    assert calibrate(source_only, task_judgments(source_only))["qualification"] == "diagnostic_only"


def test_unavailable_judge_and_withheld_cases_remain_in_task_denominator() -> None:
    """Zero false decisions must not hide a missing judgment or closed confirmation."""
    original = task_pack()
    result = calibrate(original, task_judgments(original)[1:])
    assert result["task_planned"] == 4
    assert result["task_assessed"] == 1
    assert result["task_unassessable"] == 1
    assert result["task_withheld"] == 2
    assert result["task_false_acceptances"] == 0
    assert result["task_false_rejections"] == 0


def test_an_invented_rating_for_an_unassessable_criterion_blocks_confirmation() -> None:
    """A correct outcome cannot hide an unsupported task-quality rating."""
    original = task_pack()
    labeled = CalibrationPack.model_validate(
        {
            **original.model_dump(),
            "cases": tuple(
                {**case.model_dump(), "expected_unassessable": ("quality",)}
                for case in original.cases
            ),
        }
    )
    judgments = task_judgments(labeled)
    assert calibrate(labeled, judgments)["development_status"] == "passed"
    invented = judgments[0].model_copy(
        update={
            "axes": tuple(
                axis.model_copy(update={"status": "assessed", "score": 5})
                if axis.dimension_id == "quality"
                else axis
                for axis in judgments[0].axes
            )
        }
    )
    report = calibrate(labeled, (invented,) + judgments[1:])
    assert report["development_status"] == "incomplete_or_incorrect"
    assert report["confirmation_status"] == "not_run"
    rows = report["rows"]
    assert isinstance(rows, list)
    assert rows[0]["judge"]["assessment_errors"] == {"quality": 5}


def test_wrong_expert_anchor_keeps_confirmation_closed_even_when_task_passes() -> None:
    """Opening reserved ratings requires development agreement, not only binary task success."""
    original = task_pack()
    labeled = CalibrationPack.model_validate(
        {
            **original.model_dump(),
            "cases": tuple(
                {
                    **case.model_dump(),
                    "challenge": {**case.challenge.model_dump(), "authority": "human"},
                    "expert_scores": {"outcome": 5 if case.expected_task == "pass" else 1},
                }
                for case in original.cases
            ),
        }
    )
    judgments = task_judgments(labeled)
    inaccurate = judgments[0].model_copy(
        update={
            "axes": tuple(
                axis.model_copy(update={"score": 4}) if axis.dimension_id == "outcome" else axis
                for axis in judgments[0].axes
            )
        }
    )
    report = calibrate(labeled, (inaccurate,) + judgments[1:])
    assert report["development_status"] == "incomplete_or_incorrect"
    assert report["confirmation_status"] == "not_run"


@pytest.mark.parametrize("criteria", [("quality", "quality"), ("not-a-criterion",)])
def test_unassessable_labels_require_a_unique_registered_criterion(
    criteria: tuple[str, ...],
) -> None:
    """Assessment-status labels cannot silently name extra criteria."""
    original = task_pack()
    with pytest.raises(ValidationError):
        CalibrationPack.model_validate(
            {
                **original.model_dump(),
                "cases": ({**original.cases[0].model_dump(), "expected_unassessable": criteria},),
            }
        )


def test_bound_judge_cannot_reuse_scores_from_another_completion_profile() -> None:
    """Identical rubric text does not make JSON and tool output configurations equivalent."""
    original = task_pack()
    bound = CalibrationPack.model_validate(
        {**original.model_dump(), "judge_completion_sha256": "7" * 64}
    )
    with pytest.raises(ValueError, match="completion profile"):
        calibrate(bound, task_judgments(original))
