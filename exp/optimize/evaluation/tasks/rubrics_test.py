"""Rubric development protects expert labels and never optimizes on confirmation."""

import json
from typing import NoReturn

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.evaluations.actor_rubric import actor_judge
from exp.common.evaluations.reconstruction import AxisOutcome, JudgeConfig
from exp.optimize.evaluation.tasks.rubrics import RubricEdit, refine_rubric
from exp.simulation.evaluation.calibration import (
    CalibrationCase,
    CalibrationJudgment,
    CalibrationPack,
)
from exp.simulation.evaluation.calibration_test import pack


@pytest.mark.parametrize("task_labels", [False, True])
@pytest.mark.parametrize("selection", ["repaired", "regressed", "already-perfect"])
def test_blinded_revision_improves_without_opening_confirmation_during_selection(
    task_labels: bool,
    selection: str,
) -> None:
    """Use explicit software-fixture human labels, never claim real expert calibration.

    The selection partition is either repaired with development, regressed by the overfit
    revision, or already judged perfectly so it can only stay unchanged.
    """
    selection_overfit = selection == "regressed"
    original = pack()
    config = actor_judge("fixture-judge")
    cases = tuple(
        case.model_copy(
            update={
                "challenge": case.challenge.model_copy(update={"authority": "human"}),
                "expert_scores": {"outcome": 5 if case.challenge.expected == "pass" else 1},
            }
        )
        for case in original.cases
    )
    calibrated = original.model_copy(
        update={
            "judge": config,
            "cases": cases,
            "required_dimensions": ("outcome",),
            "frozen_profile_sha256": "4" * 64,
        }
    )
    if task_labels:
        cases = tuple(
            case.model_copy(
                update={
                    "expected_task": case.challenge.expected,
                    "expert_scores": {},
                    "challenge": case.challenge.model_copy(update={"expected": "pass"}),
                }
            )
            for case in original.cases
        )
        calibrated = calibrated.model_copy(
            update={
                "cases": cases,
                "candidate": original.parent,
                "minimum_scores": {"outcome": 4},
            }
        )
    selections = []
    for case in calibrated.cases[:2]:
        assert isinstance(case.challenge.outcome, dict)
        selections.append(
            CalibrationCase.model_validate(
                {
                    **case.model_dump(),
                    "partition": "selection",
                    "lineage": "selection",
                    "challenge": {
                        **case.challenge.model_dump(),
                        "id": "selection-" + case.challenge.id,
                        "outcome": {**case.challenge.outcome, "source_instance": "selection"},
                    },
                }
            )
        )
    calibrated = CalibrationPack.model_validate(
        {**calibrated.model_dump(), "cases": calibrated.cases + tuple(selections)}
    )
    calls: list[str] = []
    proposals: list[str] = []

    def judge(current: JudgeConfig, visible: JsonObject, identity: str) -> CalibrationJudgment:
        """Simulate a biased baseline and a repaired rubric for lifecycle tests only."""
        calls.append(identity)
        value = visible["outcome"]
        assert isinstance(value, dict)
        assert "expected" not in visible and "authority" not in visible
        answer, observed = value["final_answer"], value["observed"]
        score = (5 if answer == observed else 1) if current.prompt == "fixture-corrected" else 5
        if value.get("source_instance") == "selection" and selection != "repaired":
            score = 5 if answer == observed else 1
            if selection_overfit and current.prompt != config.prompt:
                score = 6 - score
        return CalibrationJudgment(
            case_id=str(visible["case_id"]),
            outcome_sha256=sha256_json(value),
            judge_sha256=sha256_json(current),
            report=ArtifactInput(artifact_id="fixture-judge-report", sha256="3" * 64),
            axes=tuple(
                AxisOutcome(
                    dimension_id=dimension.dimension_id,
                    status="assessed" if dimension.dimension_id == "outcome" else "unassessable",
                    score=score if dimension.dimension_id == "outcome" else None,
                    rationale="Declared software fixture",
                    evidence_refs=("workflow-0",),
                )
                for dimension in current.definition.dimensions
            ),
        )

    def propose(payload: JsonObject, identity: str) -> RubricEdit:
        """Check that no held-out observations or labels reach rubric construction."""
        proposals.append(identity)
        assert "confirmation-True" not in json.dumps(payload)
        assert "confirmation-False" not in json.dumps(payload)
        assert "selection-development-" not in json.dumps(payload)
        assert '"source_instance": "selection"' not in json.dumps(payload)
        return RubricEdit(
            definition=config.definition,
            prompt="fixture-corrected",
            evidence_refs=(cases[0].challenge.id,),
            reason="Fix the observed false acceptance",
        )

    result = refine_rubric(
        calibrated,
        propose=propose,
        judge=judge,
        maximum_revisions=1 if selection_overfit else 2,
    )
    assert len(proposals) == 1
    assert len(calls) == (8 if selection_overfit else 10)
    assert result["development_targets_met"] is not selection_overfit
    assert result["selection_targets_met"] is True
    history = result["history"]
    assert isinstance(history, list) and len(history) == 1
    assert history[0]["selected"] is not selection_overfit
    final = result["final"]
    assert isinstance(final, dict)
    assert final["confirmation_status"] == ("not_run" if selection_overfit else "passed")
    assert final["qualification"] == "diagnostic_only"


def test_missing_human_labels_cannot_start_rubric_optimization() -> None:
    """A benchmark success flag does not supply five expert rubric judgments."""

    def unused(*args: object) -> NoReturn:
        """No inference callback may execute before required labels exist."""
        raise AssertionError("unexpected callback")

    with pytest.raises(ValueError, match="exact initial judge"):
        refine_rubric(pack(), propose=unused, judge=unused, maximum_revisions=1)


def test_published_reward_cannot_train_full_task_rubric_without_reviewed_labels() -> None:
    """An imported reward flag is not permission to optimize away policy or grounding errors."""
    original = pack()
    config = actor_judge("fixture-judge")
    scoped = original.model_copy(
        update={
            "judge": config,
            "required_dimensions": ("outcome",),
            "minimum_scores": {"outcome": 4},
            "cases": tuple(
                case.model_copy(
                    update={
                        "expected_task": case.challenge.expected,
                        "task_label_scope": "source_outcome",
                    }
                )
                for case in original.cases
            ),
        }
    )

    def unused(*args: object) -> NoReturn:
        """No model may see unreviewed source labels as full task truth."""
        raise AssertionError("unexpected inference")

    with pytest.raises(ValueError, match="admitted task labels"):
        refine_rubric(scoped, propose=unused, judge=unused, maximum_revisions=1)


def test_refinement_learns_when_not_to_rate_without_inventing_expert_scores() -> None:
    """A declared absent quality criterion supplies a status label, not a human rating."""
    original = pack()
    config = actor_judge("fixture-judge")
    cases = [
        CalibrationCase.model_validate({**case.model_dump(), "expected_unassessable": ("quality",)})
        for case in original.cases
    ]
    for case in tuple(cases[:2]):
        assert isinstance(case.challenge.outcome, dict)
        cases.append(
            CalibrationCase.model_validate(
                {
                    **case.model_dump(),
                    "partition": "selection",
                    "lineage": "selection",
                    "challenge": {
                        **case.challenge.model_dump(),
                        "id": "selection-" + case.challenge.id,
                        "outcome": {**case.challenge.outcome, "source_instance": "selection"},
                    },
                }
            )
        )
    calibrated = CalibrationPack.model_validate(
        {
            **original.model_dump(),
            "judge": config,
            "cases": tuple(cases),
            "required_dimensions": ("quality",),
        }
    )

    def judge(current: JudgeConfig, visible: JsonObject, identity: str) -> CalibrationJudgment:
        """Supply fixture responses that either invent quality or correctly abstain."""
        fixed = current.prompt == "fixture-no-invented-quality"
        return CalibrationJudgment(
            case_id=str(visible["case_id"]),
            outcome_sha256=sha256_json(visible["outcome"]),
            judge_sha256=sha256_json(current),
            report=ArtifactInput(artifact_id="fixture-" + sha256_json(identity), sha256="3" * 64),
            axes=tuple(
                AxisOutcome(
                    dimension_id=dimension.dimension_id,
                    status="assessed"
                    if dimension.dimension_id == "quality" and not fixed
                    else "unassessable",
                    score=5 if dimension.dimension_id == "quality" and not fixed else None,
                    rationale="Declared fixture",
                    evidence_refs=("workflow-0",),
                )
                for dimension in current.definition.dimensions
            ),
        )

    def propose(payload: JsonObject, identity: str) -> RubricEdit:
        """The author must receive the actual status discrepancy, without reserved labels."""
        development = payload["development"]
        assert isinstance(development, list)
        assert all(row["expected_unassessable"] == ["quality"] for row in development)
        assert "confirmation-" not in json.dumps(payload)
        assert "selection-development" not in json.dumps(payload)
        return RubricEdit(
            definition=config.definition,
            prompt="fixture-no-invented-quality",
            evidence_refs=(cases[0].challenge.id,),
            reason="The task declares no quality criterion",
        )

    result = refine_rubric(calibrated, propose=propose, judge=judge, maximum_revisions=1)
    assert result["development_targets_met"] is True
    assert result["selection_targets_met"] is True
    final = result["final"]
    assert isinstance(final, dict)
    assert final["confirmation_status"] == "passed"
    assert final["qualification"] == "diagnostic_only"
