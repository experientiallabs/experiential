"""Calibration cannot turn synthetic labels or leaked cases into expert qualification."""

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput, JsonObject, canonical_json_bytes, sha256_json
from exp.common.evaluations.actor_rubric import actor_judge
from exp.common.evaluations.reconstruction import AxisOutcome, WorkflowEvidence, WorkflowStep
from exp.simulation.evaluation.admission import AdmissionRefusal, EvidenceAdmission
from exp.simulation.evaluation.calibration import (
    CalibrationCase,
    CalibrationJudgment,
    CalibrationPack,
    blinded_case,
    calibrate,
    task_decision,
)
from exp.simulation.evaluation.tasks import FieldRelation, TaskVerifier, VerifierChallenge


def pack(*, variants: int = 1) -> CalibrationPack:
    """Declare attested recorded-reference cases for software checks only."""
    witness = ArtifactInput(artifact_id="declared-specification", sha256="1" * 64)
    cases = []
    for partition in ("development", "confirmation"):
        for correct in (True, False):
            for variant in range(variants):
                value: JsonObject = {
                    "source_instance": partition,
                    "final_answer": {"status": "pending" if correct else "delivered"},
                    "observed": {"status": "pending"},
                    "stop_reason": "completed",
                }
                if variant:
                    value["variant"] = variant
                cases.append(
                    CalibrationCase(
                        challenge=VerifierChallenge(
                            id=f"{partition}-{correct}" + (f"-{variant}" if variant else ""),
                            outcome=value,
                            expected="pass" if correct else "fail",
                            authority="recorded_reference",
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


def test_mechanical_agreement_does_not_self_certify_the_judge() -> None:
    """Mechanical agreement is useful but not evidence of expert rubric agreement."""
    report = calibrate(pack(), admission=EvidenceAdmission())
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


def task_pack(*, variants: int = 1) -> CalibrationPack:
    """Separate component validity from task truth in declared software fixtures."""
    original = pack(variants=variants)
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
    original = task_pack(variants=2)
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
    admitted = EvidenceAdmission()
    assert (
        calibrate(bound, task_judgments(bound), admission=admitted)["qualification"]
        == "qualified_in_scope"
    )
    unbound = bound.model_copy(update={"judge_completion_sha256": None})
    assert (
        calibrate(unbound, task_judgments(unbound), admission=admitted)["qualification"]
        == "diagnostic_only"
    )
    source_only = bound.model_copy(
        update={
            "cases": tuple(
                case.model_copy(update={"task_label_scope": "source_outcome"})
                for case in bound.cases
            )
        }
    )
    assert (
        calibrate(source_only, task_judgments(source_only), admission=admitted)["qualification"]
        == "diagnostic_only"
    )


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


def bound_pack(*, witnessed: bool = False, variants: int = 2) -> CalibrationPack:
    """A profile-bound task pack whose labels are attested or carry saved witnesses."""
    original = task_pack(variants=variants)
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
        challenge = case.challenge.model_dump(mode="json") | {
            "outcome": evidence.model_dump(mode="json"),
            "authority": "recorded_reference",
        }
        if witnessed:
            challenge |= {
                "authority": "synthetic_specification",
                "approved": False,
                "source_refs": [],
                "expected": case.expected_task,
                "witness": {
                    "artifact_id": "witness-" + case.challenge.id.lower(),
                    "sha256": "5" * 64,
                },
            }
        cases.append({**case.model_dump(mode="json"), "challenge": challenge})
    # JSON parsing, as the CLI does, keeps each outcome a typed WorkflowEvidence.
    return CalibrationPack.model_validate_json(
        canonical_json_bytes(
            {
                **original.model_dump(mode="json"),
                "cases": cases,
                "candidate": TaskVerifier(
                    description="Answer agrees with the observed reply",
                    outcome_schema={"type": "object"},
                    relations=(
                        FieldRelation(left="/final_answer/status", right="/steps/0/reply/status"),
                    ),
                ).model_dump(mode="json"),
                "frozen_profile_sha256": "4" * 64,
                "judge_completion_sha256": "7" * 64,
            }
        )
    )


def verified(pack: CalibrationPack, *refusals: AdmissionRefusal) -> EvidenceAdmission:
    """Stand in for store admission that re-verified every witnessed label."""
    return EvidenceAdmission(
        verified={
            case.challenge.id: sha256_json(case.challenge)
            for case in pack.cases
            if case.challenge.witness is not None
        },
        refusals=refusals,
    )


def refusals(report: JsonObject) -> list[AdmissionRefusal]:
    """The report's named qualification refusals."""
    rows = report["qualification_refusals"]
    assert isinstance(rows, list)
    return [AdmissionRefusal.model_validate(row) for row in rows]


def test_witnessed_labels_qualify_without_any_approval_step() -> None:
    """Verified specification witnesses replace the human approval dependency entirely."""
    pack = bound_pack(witnessed=True)
    assert not any(case.challenge.approved for case in pack.cases)
    report = calibrate(pack, task_judgments(pack), admission=verified(pack))
    assert report["witnessed_verified"] == len(pack.cases)
    assert report["qualification_refusals"] == []
    assert report["qualification"] == "qualified_in_scope"


def test_unverified_witnesses_admit_nothing_and_name_inadequate_evidence() -> None:
    """A witness reference that admission did not re-derive is not a label."""
    pack = bound_pack(witnessed=True)
    report = calibrate(pack, task_judgments(pack), admission=EvidenceAdmission())
    assert report["independent"] == 0
    assert report["development_status"] == "incomplete_or_incorrect"
    assert report["qualification"] == "diagnostic_only"
    assert {row.reason for row in refusals(report)} == {"inadequate_evidence"}


def test_too_few_witnessed_cases_per_partition_block_qualification() -> None:
    """Correct decisions on one case per side are inadequate evidence, not qualification."""
    pack = bound_pack(witnessed=True, variants=1)
    report = calibrate(pack, task_judgments(pack), admission=verified(pack))
    assert report["development_status"] == "passed"
    assert report["confirmation_status"] == "passed"
    assert report["qualification"] == "diagnostic_only"
    subjects = {row.subject for row in refusals(report) if row.reason == "inadequate_evidence"}
    assert subjects == {
        "development",
        "development on axis outcome",
        "confirmation",
        "confirmation on axis outcome",
    }


@pytest.mark.parametrize(
    "reason", ["contaminated_lineage", "sealed_partition_breach", "self_approval"]
)
def test_any_admission_refusal_blocks_an_otherwise_qualified_pack(reason: str) -> None:
    """Contamination, partition breaches and self-approval each block qualification by name."""
    pack = bound_pack(witnessed=True)
    refusal = AdmissionRefusal.model_validate(
        {"reason": reason, "subject": "lineage", "detail": "named by store admission"}
    )
    report = calibrate(pack, task_judgments(pack), admission=verified(pack, refusal))
    assert report["qualification"] == "diagnostic_only"
    assert report["qualification_refusals"] == [refusal.model_dump(mode="json")]


def test_qualification_needs_store_admission_even_for_attested_labels() -> None:
    """Without store admission, contamination is undecided, so nothing qualifies."""
    pack = bound_pack()
    report = calibrate(pack, task_judgments(pack))
    assert report["qualification"] == "diagnostic_only"
    assert [row.subject for row in refusals(report)] == ["evidence admission"]


def test_a_witness_establishes_one_verdict() -> None:
    """A witnessed case cannot carry a task label its witness does not establish."""
    case = bound_pack(witnessed=True).cases[0]
    data = case.model_dump(mode="json")
    data["expected_task"] = "fail" if case.challenge.expected == "pass" else "pass"
    with pytest.raises(ValidationError, match="one verdict"):
        CalibrationCase.model_validate(data)


def stratified_pack() -> CalibrationPack:
    """Human-anchored task labels split into representative and failure-enriched samples."""
    original = task_pack(variants=2)
    return CalibrationPack.model_validate(
        {
            **original.model_dump(),
            "cases": tuple(
                {
                    **case.model_dump(),
                    "challenge": {**case.challenge.model_dump(), "authority": "human"},
                    "expert_scores": {"outcome": 5 if case.expected_task == "pass" else 1},
                    "category": "wrong_cell",
                    "sample": "failure_enriched"
                    if case.challenge.id.endswith("-1")
                    else "representative",
                }
                for case in original.cases
            ),
        }
    )


def with_outcome(
    judgment: CalibrationJudgment, status: str, score: int | None
) -> CalibrationJudgment:
    """Replace the judged outcome criterion while keeping the exact case binding."""
    return judgment.model_copy(
        update={
            "axes": tuple(
                axis.model_copy(update={"status": status, "score": score})
                if axis.dimension_id == "outcome"
                else axis
                for axis in judgment.axes
            )
        }
    )


def validation(report: JsonObject) -> JsonObject:
    """Return the per-category judge validation section of a calibration report."""
    section = report["judge_validation"]
    assert isinstance(section, dict)
    return section


def test_judge_errors_are_reported_per_sample_and_category_without_pooling() -> None:
    """A failure-enriched false acceptance never dilutes into representative error rates."""
    original = stratified_pack()
    judgments = list(task_judgments(original))
    ids = [case.challenge.id for case in original.cases]
    # The enriched defective variant is passed; a representative valid case scores 3 against an
    # expert 5 at threshold 4 (a one-point crossing), so it is falsely rejected.
    judgments[ids.index("development-False-1")] = with_outcome(
        judgments[ids.index("development-False-1")], "assessed", 5
    )
    judgments[ids.index("development-True")] = with_outcome(
        judgments[ids.index("development-True")], "assessed", 3
    )
    report = calibrate(original, tuple(judgments))
    section = validation(report)
    assert section["task"] == "evaluation-task"
    assert section["withheld_confirmation_cases"] == 4, "confirmation stays closed"
    floor = section["admission_floor"]
    assert isinstance(floor, dict) and floor["minimum_admitted_per_verdict"] == 2
    assert "not a statistical certification" in str(floor["interpretation"])
    samples = section["samples"]
    assert isinstance(samples, dict) and set(samples) == {"failure_enriched", "representative"}
    enriched = samples["failure_enriched"]["categories"]["wrong_cell"]
    representative = samples["representative"]["categories"]["wrong_cell"]
    assert enriched["task_decision"]["false_acceptances"] == 1
    assert enriched["task_decision"]["false_rejections"] == 0
    assert representative["task_decision"]["false_acceptances"] == 0
    assert representative["task_decision"]["false_rejections"] == 1
    assert enriched["task_decision"]["defect_sensitivity"]["estimate"] == 0.0
    assert "prevalence-dependent" in enriched["task_decision"]["defect_precision"]
    precision = representative["task_decision"]["defect_precision"]
    assert precision["successes"] == 1 and precision["trials"] == 2
    assert precision["independent_units"] == 1, "one development lineage, not two cases"
    crossings = representative["criteria"]["threshold_crossings"]
    assert crossings["pessimistic"] == 1 and crossings["optimistic"] == 0
    assert enriched["criteria"]["threshold_crossings"]["optimistic"] == 1
    assert samples["representative"]["overall"]["cases"] == 2


def test_unnecessary_abstention_and_unsupported_confidence_are_counted_separately() -> None:
    """Declining a judgeable case and judging an unjudgeable one are different errors."""
    original = stratified_pack()
    cases = list(original.cases)
    ids = [case.challenge.id for case in cases]
    index = ids.index("development-True-1")
    unsupported = cases[index].model_dump()
    unsupported.update(
        {
            "expected_task": "unassessable",
            "expert_scores": {},
            "expected_unassessable": ("quality",),
        }
    )
    cases[index] = CalibrationCase.model_validate(unsupported)
    labeled = CalibrationPack.model_validate({**original.model_dump(), "cases": tuple(cases)})
    judgments = list(task_judgments(labeled))
    judgments[index] = judgments[index].model_copy(
        update={
            "axes": tuple(
                axis.model_copy(update={"status": "assessed", "score": 5})
                if axis.dimension_id in ("outcome", "quality")
                else axis
                for axis in judgments[index].axes
            )
        }
    )
    declined = ids.index("development-False")
    judgments[declined] = with_outcome(judgments[declined], "unassessable", None)
    report = calibrate(labeled, tuple(judgments))
    samples = validation(report)["samples"]
    assert isinstance(samples, dict)
    enriched = samples["failure_enriched"]["categories"]["wrong_cell"]
    representative = samples["representative"]["categories"]["wrong_cell"]
    assert enriched["task_decision"]["confident_without_evidence"] == 1
    assert enriched["criteria"]["confident_without_evidence"] == 1
    assert representative["task_decision"]["unnecessary_abstentions"] == 1
    assert representative["criteria"]["unnecessary_abstentions"] == 1
    assert representative["task_decision"]["false_acceptances"] == 0


def test_uncategorized_cases_without_judgments_stay_visible() -> None:
    """Legacy packs report under explicit defaults and unjudged task cases are not decisions."""
    report = calibrate(task_pack())
    samples = validation(report)["samples"]
    assert isinstance(samples, dict) and set(samples) == {"unspecified"}
    category = samples["unspecified"]["categories"]["uncategorized"]
    assert category["task_decision"]["not_judged"] == 2
    assert category["task_decision"]["labeled"] == 0
    assert category["verifier"]["labeled"] == 2
    assert "prevalence-dependent" in category["verifier"]["defect_precision"]


def test_rejected_judge_outputs_are_failures_not_abstentions() -> None:
    """A judge output the evaluator rejected is an operational failure, not a judge decision."""
    original = stratified_pack()
    judgments = list(task_judgments(original))
    ids = [case.challenge.id for case in original.cases]
    rejected = ids.index("development-False")
    judgments[rejected] = judgments[rejected].model_copy(
        update={
            "axes": tuple(
                axis.model_copy(
                    update={
                        "status": "judge_failure",
                        "score": None,
                        "rationale": "ReturnedOutputRejected",
                        "evidence_refs": (),
                    }
                )
                for axis in judgments[rejected].axes
            )
        }
    )
    report = calibrate(original, tuple(judgments))
    samples = validation(report)["samples"]
    assert isinstance(samples, dict)
    representative = samples["representative"]["categories"]["wrong_cell"]["task_decision"]
    assert representative["judge_failures"] == 1
    assert representative["unnecessary_abstentions"] == 0
    assert representative["false_acceptances"] == 0 and representative["false_rejections"] == 0


@pytest.mark.parametrize("disagreement", ["none", "threshold", "second-threshold", "critical"])
def test_judge_only_qualification_needs_matching_threshold_decisions(disagreement: str) -> None:
    """Scores within tolerance still disqualify a judge whose pass/fail decision flips.

    Two required criteria carry different minimums; a crossing on either one disqualifies.
    """
    bound = bound_pack(variants=2)
    required = {"outcome": 5, "evidence": 4}
    judgments = tuple(
        judgment.model_copy(
            update={
                "axes": tuple(
                    axis.model_copy(update={"status": "assessed", "score": 4})
                    if axis.dimension_id == "evidence"
                    else axis
                    for axis in judgment.axes
                )
            }
        )
        for judgment in task_judgments(bound)
    )
    labeled = CalibrationPack.model_validate(
        {
            **bound.model_dump(),
            "score_tolerance": 1,
            "required_dimensions": tuple(required),
            "minimum_scores": required,
            "cases": tuple(
                {
                    **case.model_dump(),
                    "expected_task": None,
                    "challenge": {
                        **case.challenge.model_dump(),
                        "authority": "human",
                        "expected": case.expected_task,
                    },
                    "expert_scores": {
                        "outcome": 5 if case.expected_task == "pass" else 1,
                        "evidence": 4,
                    },
                }
                for case in bound.cases
            ),
        }
    )
    position = next(i for i, case in enumerate(labeled.cases) if case.challenge.expected == "pass")
    changes = {
        "threshold": ("outcome", {"score": 4}),
        "second-threshold": ("evidence", {"score": 3}),
        "critical": ("outcome", {"critical_flags": ("unsupported_claim",)}),
    }
    if disagreement in changes:
        criterion, update = changes[disagreement]
        changed = judgments[position].model_copy(
            update={
                "axes": tuple(
                    axis.model_copy(update=update) if axis.dimension_id == criterion else axis
                    for axis in judgments[position].axes
                )
            }
        )
        judgments = judgments[:position] + (changed,) + judgments[position + 1 :]
    report = calibrate(labeled, judgments, admission=EvidenceAdmission())
    assert report["rubric_status"] == "supported_on_labeled_cases"
    assert report["qualification"] == (
        "qualified_in_scope" if disagreement == "none" else "diagnostic_only"
    )
