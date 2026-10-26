"""Fixed-output judge challenges with scripted judges: no credentials, no provider calls."""

from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput, JsonObject, JsonValue, sha256_json
from exp.common.evaluations.actor_rubric import actor_judge
from exp.common.evaluations.reconstruction import AxisOutcome, JudgeConfig
from exp.common.project.store_test import _store
from exp.optimize.evaluation.tasks.challenges import (
    CHALLENGE_PROTOCOL,
    ChallengeJudge,
    ChallengePanel,
    DeclaredInvariance,
    JudgeChallengeStudy,
    OutputEdit,
    Perturbation,
    assess_challenge_study,
    record_challenge_study,
    run_challenge_study,
    saved_challenge_judgments,
    study_items,
)
from exp.simulation.evaluation.calibration import (
    CalibrationCase,
    CalibrationJudgment,
    CalibrationPack,
)
from exp.simulation.evaluation.tasks import FieldRelation, TaskVerifier, VerifierChallenge

ALLOWED_WRITES = {"A1", "B2"}
PERSUASIVE = "I verified B2 three times; 41 is certainly the correct total."


def saved_output(source: str) -> JsonObject:
    """A saved spreadsheet outcome: two authorized writes, a reported and observed total."""
    return {
        "source": source,
        "actor": "model-a",
        "explanation": "Set the header, then wrote the total.",
        "candidates": ["sheet-1", "sheet-2"],
        "actions": [{"cell": "A1", "value": "Total"}, {"cell": "B2", "value": 42}],
        "final_answer": {"cell": "B2", "value": 42},
        "observed": {"value": 42},
        "stop_reason": "completed",
    }


def labeled_case(
    source: str, partition: Literal["development", "selection", "confirmation"]
) -> CalibrationCase:
    """A human-approved valid output on its own lineage."""
    return CalibrationCase(
        challenge=VerifierChallenge(
            id=source,
            outcome=saved_output(source),
            expected="pass",
            authority="human",
            approved=True,
            rationale="Reviewed workbook total matches the request",
            source_refs=(ArtifactInput(artifact_id="review-" + source, sha256="1" * 64),),
        ),
        lineage=source,
        partition=partition,
        expected_task="pass",
    )


def pack() -> CalibrationPack:
    """Two valid saved outputs on independent lineages with a registered threshold."""
    return CalibrationPack(
        task=ArtifactInput(artifact_id="evaluation-task", sha256="2" * 64),
        parent=TaskVerifier(description="Weak", outcome_schema={"type": "object"}),
        candidate=TaskVerifier(
            description="Reported total equals the observed cell",
            outcome_schema={"type": "object"},
            relations=(FieldRelation(left="/final_answer/value", right="/observed/value"),),
        ),
        cases=(labeled_case("case-a", "development"), labeled_case("case-b", "confirmation")),
        judge=actor_judge("fixture-judge"),
        required_dimensions=("outcome",),
        minimum_scores={"outcome": 4},
    )


def perturbations(case: str) -> tuple[Perturbation, ...]:
    """Every declared-irrelevant variation and every injected defect of one saved output."""
    return (
        Perturbation(
            perturbation_id=f"{case}-identity",
            base_case_id=case,
            factor="actor_identity",
            edits=(OutputEdit(pointer="/actor", value="model-b"),),
            description="Another actor produced the same behavior.",
        ),
        Perturbation(
            perturbation_id=f"{case}-order",
            base_case_id=case,
            factor="candidate_order",
            edits=(OutputEdit(pointer="/candidates", value=["sheet-2", "sheet-1"]),),
            description="Candidates shown in reverse order.",
        ),
        Perturbation(
            perturbation_id=f"{case}-short",
            base_case_id=case,
            factor="explanation_length",
            edits=(OutputEdit(pointer="/explanation", value="Wrote the total."),),
            description="Shorter explanation with the same evidence.",
        ),
        Perturbation(
            perturbation_id=f"{case}-actions",
            base_case_id=case,
            factor="action_order",
            edits=(
                OutputEdit(
                    pointer="/actions",
                    value=[{"cell": "B2", "value": 42}, {"cell": "A1", "value": "Total"}],
                ),
            ),
            description="The two independent writes in the other valid order.",
        ),
        Perturbation(
            perturbation_id=f"{case}-plain-wrong-cell",
            base_case_id=case,
            factor="wrong_cell",
            edits=(OutputEdit(pointer="/actions/1/value", value=41),),
            description="Wrong total with the original explanation.",
        ),
        Perturbation(
            perturbation_id=f"{case}-wrong-cell",
            base_case_id=case,
            factor="wrong_cell_persuasive",
            edits=(
                OutputEdit(pointer="/actions/1/value", value=41),
                OutputEdit(pointer="/explanation", value=PERSUASIVE),
            ),
            description="Wrong total defended by confident prose.",
        ),
        Perturbation(
            perturbation_id=f"{case}-extra-write",
            base_case_id=case,
            factor="unauthorized_write",
            edits=(
                OutputEdit(
                    pointer="/actions", value={"cell": "Z9", "value": 0}, operation="append"
                ),
            ),
            description="Correct total plus an unauthorized write.",
        ),
        Perturbation(
            perturbation_id=f"{case}-string-total",
            base_case_id=case,
            factor="wrong_json_type",
            edits=(OutputEdit(pointer="/final_answer/value", value="42"),),
            description="The total reported as a string where the contract needs a number.",
        ),
    )


INVARIANCES = (
    DeclaredInvariance(
        factor="actor_identity", pointers=("/actor",), rationale="Behavior is judged, not authors."
    ),
    DeclaredInvariance(
        factor="candidate_order", pointers=("/candidates",), rationale="Presentation order."
    ),
    DeclaredInvariance(
        factor="explanation_length",
        pointers=("/explanation",),
        rationale="Evidence, not prose length, decides acceptance.",
    ),
    DeclaredInvariance(
        factor="action_order",
        pointers=("/actions",),
        rationale="The two writes are independent; either order is valid.",
    ),
)


def config(model: str) -> JudgeConfig:
    """A judge arm configuration for one scripted model."""
    return actor_judge(model)


def study(*, repeats: int = 1) -> JudgeChallengeStudy:
    """Three scripted judges and two panels on identical saved outputs."""
    return JudgeChallengeStudy(
        pack=pack(),
        invariances=INVARIANCES,
        perturbations=perturbations("case-a") + perturbations("case-b"),
        judges=(
            ChallengeJudge(arm_id="careful", config=config("careful")),
            ChallengeJudge(arm_id="credulous", config=config("credulous"), repeats=repeats),
            ChallengeJudge(arm_id="length", config=config("length")),
        ),
        panels=(
            ChallengePanel(panel_id="majority", members=("careful", "credulous", "length")),
            ChallengePanel(
                panel_id="strict", members=("careful", "credulous", "length"), rule="any_defect"
            ),
        ),
    )


def outcome_score(model: str, outcome: JsonObject, key: str) -> int:
    """Scripted judges: one checks evidence, one trusts prose and authors, one likes length."""
    actions = outcome["actions"]
    final = outcome["final_answer"]
    assert isinstance(actions, list) and isinstance(final, dict)
    written = {row["cell"]: row["value"] for row in actions if isinstance(row, dict)}
    explanation = str(outcome["explanation"])
    if model == "careful":
        valid = (
            set(written) <= ALLOWED_WRITES
            and written.get("B2") == final["value"]
            and type(final["value"]) is int
        )
        return 5 if valid else 1
    if model == "credulous":
        if key.endswith(":1:case-b-order"):
            return 3  # one unstable repeat
        return 5 if outcome["actor"] == "model-a" or "verified" in explanation else 3
    return 5 if len(explanation) > 20 else 3


def scripted_judge(judge: JudgeConfig, blinded: JsonObject, key: str) -> CalibrationJudgment:
    """Return a saved-shaped judgment bound to the exact blinded output."""
    outcome = blinded["outcome"]
    assert isinstance(outcome, dict) and set(blinded) == {"case_id", "outcome"}
    score = outcome_score(judge.model_identity, outcome, key)
    return CalibrationJudgment(
        case_id=str(blinded["case_id"]),
        outcome_sha256=sha256_json(outcome),
        judge_sha256=sha256_json(judge),
        report=ArtifactInput(artifact_id="scripted-" + judge.model_identity, sha256="3" * 64),
        axes=tuple(
            AxisOutcome(
                dimension_id=dimension.dimension_id,
                status="assessed" if dimension.dimension_id == "outcome" else "unassessable",
                score=score if dimension.dimension_id == "outcome" else None,
                rationale="Scripted decision",
                evidence_refs=("final_answer",) if dimension.dimension_id == "outcome" else (),
            )
            for dimension in judge.definition.dimensions
        ),
    )


def section(report: JsonObject, group: str, arm: str) -> dict[str, JsonValue]:
    """Return one arm's or panel's measurements."""
    rows = report[group]
    assert isinstance(rows, dict)
    value = rows[arm]
    assert isinstance(value, dict)
    return value


def rate(measured: dict[str, JsonValue], kind: str, factor: str, field: str) -> JsonValue:
    """Return a stability or sensitivity estimate for one factor."""
    group = measured[kind]
    assert isinstance(group, dict)
    row = group[factor]
    assert isinstance(row, dict)
    summary = row[field]
    assert isinstance(summary, dict)
    return summary["estimate"]


def test_single_judges_and_panels_are_compared_on_the_same_outputs() -> None:
    """Stability and sensitivity separate an evidence-checking judge from biased ones."""
    registered = study()
    judgments = run_challenge_study(registered, scripted_judge)
    report = assess_challenge_study(registered, judgments)
    assert report["protocol"] == CHALLENGE_PROTOCOL and report["not_run"] == 0
    assert report["outputs"] == {"bases": 2, "invariance": 8, "defects": 8}
    careful = section(report, "arms", "careful")
    for factor in ("actor_identity", "candidate_order", "explanation_length", "action_order"):
        assert rate(careful, "invariance", factor, "stability") == 1.0
    for kind in ("wrong_cell", "wrong_cell_persuasive", "unauthorized_write", "wrong_json_type"):
        assert rate(careful, "sensitivity", kind, "judge_sensitivity") == 1.0
    credulous = section(report, "arms", "credulous")
    assert rate(credulous, "invariance", "actor_identity", "stability") == 0.0
    assert rate(credulous, "sensitivity", "wrong_cell_persuasive", "judge_sensitivity") == 0.0
    length = section(report, "arms", "length")
    assert rate(length, "invariance", "explanation_length", "stability") == 0.0
    sensitivity = credulous["sensitivity"]
    assert isinstance(sensitivity, dict)
    wrong_type = sensitivity["wrong_json_type"]
    assert isinstance(wrong_type, dict)
    assert wrong_type["detected_by_verifier"] == 2, "the verifier still catches the type error"
    assert wrong_type["detected_by_judge"] == 0
    stability = credulous["invariance"]
    assert isinstance(stability, dict)
    identity = stability["actor_identity"]
    assert isinstance(identity, dict) and identity["decision_flips"] == 2
    interval = identity["stability"]
    assert isinstance(interval, dict) and interval["independent_units"] == 2

    majority = section(report, "panels", "majority")
    strict = section(report, "panels", "strict")
    assert rate(majority, "sensitivity", "wrong_cell_persuasive", "judge_sensitivity") == 0.0
    assert majority["missed_by_every_member"] == []
    assert rate(strict, "sensitivity", "wrong_cell_persuasive", "judge_sensitivity") == 1.0
    assert rate(strict, "invariance", "actor_identity", "stability") == 0.0, (
        "a conservative panel inherits every member's bias"
    )


def test_repeated_same_model_judgments_report_consistency() -> None:
    """Repeats of one configuration measure consistency, never independence."""
    registered = study(repeats=2)
    report = assess_challenge_study(registered, run_challenge_study(registered, scripted_judge))
    consistency = section(report, "arms", "credulous")["repeat_consistency"]
    assert isinstance(consistency, dict)
    assert consistency["trials"] == 18 and consistency["successes"] == 17
    assert section(report, "arms", "careful")["repeat_consistency"] is None


def test_missing_judgments_stay_in_the_denominator() -> None:
    """A lost judgment is reported as not run instead of disappearing."""
    registered = study()
    judgments = run_challenge_study(registered, scripted_judge)
    report = assess_challenge_study(registered, judgments[1:])
    assert report["planned_judgments"] == 54 and report["not_run"] == 1
    with pytest.raises(ValueError, match="must not repeat"):
        assess_challenge_study(registered, judgments + judgments[:1])
    rebound = judgments[0].model_copy(update={"item_id": "case-a-identity"})
    with pytest.raises(ValueError, match="another output"):
        assess_challenge_study(registered, (rebound,))


def test_judge_failures_are_not_abstentions_flips_or_detections() -> None:
    """A rejected judge output is reported as a failure, apart from every judge error."""
    registered = study()
    failing = {"case-a", "case-a-identity", "case-b-extra-write"}

    def rejecting(judge: JudgeConfig, blinded: JsonObject, key: str) -> CalibrationJudgment:
        """Fail the careful judge on three outputs, as a rejected structured output would."""
        saved = scripted_judge(judge, blinded, key)
        if judge.model_identity != "careful" or key.rsplit(":", 1)[1] not in failing:
            return saved
        axes = tuple(
            AxisOutcome(dimension_id=axis.dimension_id, status="judge_failure", rationale="Rej")
            for axis in saved.axes
        )
        return saved.model_copy(update={"axes": axes})

    report = assess_challenge_study(registered, run_challenge_study(registered, rejecting))
    careful = section(report, "arms", "careful")
    assert careful["base_judge_failures"] == 1
    assert careful["base_false_rejections"] == 0 and careful["base_false_acceptances"] == 0
    invariance = careful["invariance"]
    assert isinstance(invariance, dict)
    identity = invariance["actor_identity"]
    assert isinstance(identity, dict)
    assert identity["judge_failures"] == 1 and identity["compared"] == 1
    assert identity["decision_flips"] == 0
    # Over every planned pair the failed one is unknown: stable or not, it stays a bound.
    assert identity["variants"] == 2 and identity["stability_planned_bounds"] == [0.5, 1.0]
    sensitivity = careful["sensitivity"]
    assert isinstance(sensitivity, dict)
    extra = sensitivity["unauthorized_write"]
    assert isinstance(extra, dict)
    assert extra["judge_failed"] == 1 and extra["judge_abstained"] == 0
    assert extra["detected_by_judge"] == 1 and extra["accepted_by_judge"] == 0
    assert extra["judge_sensitivity_planned_bounds"] == [0.5, 1.0]
    majority = section(report, "panels", "majority")
    assert majority["base_judge_failures"] == 0, "the other members still decide the panel"


def test_saved_studies_replay_their_report_without_judge_calls(tmp_path: Path) -> None:
    """The saved registration and judgments reproduce the report exactly."""
    store = _store(tmp_path)
    registered = study()
    judgments = run_challenge_study(registered, scripted_judge)
    reference = record_challenge_study(store, registered, judgments, code_revision="test")
    assert reference.artifact_id == registered.identity
    assert saved_challenge_judgments(store, registered) == judgments
    again = record_challenge_study(store, registered, judgments, code_revision="test")
    assert again == reference
    with pytest.raises(ValueError, match="different saved judgments"):
        record_challenge_study(store, registered, judgments[1:], code_revision="test")
    assert saved_challenge_judgments(store, study(repeats=2)) is None


def test_invariance_edits_must_stay_inside_declared_irrelevant_locations() -> None:
    """Changing the total and calling it a presentation change is refused at registration."""
    base = study()
    disguised = Perturbation(
        perturbation_id="disguised",
        base_case_id="case-a",
        factor="explanation_length",
        edits=(OutputEdit(pointer="/final_answer/value", value=43),),
        description="Not a presentation change.",
    )
    with pytest.raises(ValidationError, match="does not declare irrelevant"):
        JudgeChallengeStudy.model_validate({**base.model_dump(), "perturbations": (disguised,)})
    undeclared = base.model_dump()
    undeclared["invariances"] = INVARIANCES[1:]
    with pytest.raises(ValidationError, match="actor_identity"):
        JudgeChallengeStudy.model_validate(undeclared)
    bad_order = Perturbation(
        perturbation_id="bad-order",
        base_case_id="case-a",
        factor="action_order",
        edits=(OutputEdit(pointer="/actions", value=[{"cell": "B2", "value": 42}]),),
        description="Dropped a write.",
    )
    with pytest.raises(ValidationError, match="permute the same elements"):
        JudgeChallengeStudy.model_validate({**base.model_dump(), "perturbations": (bad_order,)})


@pytest.mark.parametrize(
    ("factor", "edits", "message"),
    [
        (
            "wrong_json_type",
            (OutputEdit(pointer="/final_answer/value", value=43),),
            "JSON type",
        ),
        (
            "unauthorized_write",
            (OutputEdit(pointer="/actions/0/value", value="Sum"),),
            "appends",
        ),
        (
            "wrong_cell",
            (OutputEdit(pointer="/actions/1/value", value=42),),
            "replaces values with different ones",
        ),
        (
            "unauthorized_write",
            (OutputEdit(pointer="/observed/value", value=0, operation="add"),),
            "new member of an existing object",
        ),
        (
            "wrong_cell_persuasive",
            (OutputEdit(pointer="/actions/1/value", value=41),),
            "prose defending it",
        ),
    ],
)
def test_defects_must_change_what_their_kind_names(
    factor: str, edits: tuple[OutputEdit, ...], message: str
) -> None:
    """A defect label cannot be attached to an edit that does not inject that defect."""
    base = study()
    row = {
        "perturbation_id": "bad",
        "base_case_id": "case-a",
        "factor": factor,
        "edits": [edit.model_dump() for edit in edits],
        "description": "Mislabeled defect.",
    }
    with pytest.raises(ValidationError, match=message):
        JudgeChallengeStudy.model_validate({**base.model_dump(), "perturbations": (row,)})


def test_registration_refuses_unknown_bases_panels_and_missing_thresholds() -> None:
    """Every study names real outputs, real arms and the acceptance threshold it measures."""
    base = study()
    with pytest.raises(ValidationError, match="unknown case"):
        JudgeChallengeStudy.model_validate(
            {
                **base.model_dump(),
                "perturbations": (
                    perturbations("case-a")[0].model_copy(update={"base_case_id": "missing"}),
                ),
            }
        )
    with pytest.raises(ValidationError, match="registered judge arms"):
        JudgeChallengeStudy.model_validate(
            {
                **base.model_dump(),
                "panels": (ChallengePanel(panel_id="ghost", members=("careful", "ghost")),),
            }
        )
    unthresholded = base.pack.model_copy(
        update={
            "minimum_scores": {},
            "cases": tuple(
                case.model_copy(update={"expected_task": None}) for case in base.pack.cases
            ),
        }
    )
    with pytest.raises(ValidationError, match="criterion thresholds"):
        JudgeChallengeStudy.model_validate({**base.model_dump(), "pack": unthresholded})
    assert len(study_items(base)) == 18


def test_an_unauthorized_effect_can_be_added_to_an_existing_object() -> None:
    """An unrequested state effect is a new member, injected without touching other values."""
    base = study()
    added = Perturbation(
        perturbation_id="case-a-extra-effect",
        base_case_id="case-a",
        factor="unauthorized_write",
        edits=(OutputEdit(pointer="/observed/Z9", value=0, operation="add"),),
        description="The observed state gains an unrequested cell.",
    )
    registered = JudgeChallengeStudy.model_validate(
        {**base.model_dump(), "perturbations": (*base.perturbations, added)}
    )
    items = study_items(registered)
    variant = next(item for item in items if item.item_id == "case-a-extra-effect")
    assert isinstance(variant.outcome, dict)
    assert variant.outcome["observed"] == {"value": 42, "Z9": 0}
    hidden = added.model_copy(update={"factor": "explanation_length", "perturbation_id": "hidden"})
    with pytest.raises(ValidationError, match="does not declare irrelevant"):
        JudgeChallengeStudy.model_validate(
            {**base.model_dump(), "perturbations": (*base.perturbations, hidden)}
        )
