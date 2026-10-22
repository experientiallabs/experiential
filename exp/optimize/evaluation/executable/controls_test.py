"""Declared controls are witnessed and re-verified with the checks qualification runs."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.controls import (
    ControlWitnessSpec,
    DeclaredControl,
    SpecificationRegistration,
    preflight_controls,
    preflight_specification,
    register_specification,
    witness_controls,
)
from exp.optimize.evaluation.executable.qualification import (
    ExecutableQualificationSpec,
    assess_executable_qualification,
)
from exp.optimize.evaluation.executable.qualification_test import qualified_fixture
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.simulation.evaluation.admission import EvaluationSpecification, EvidenceAdmissionError

NOW = datetime.now(UTC)


def declared(spec: ExecutableQualificationSpec) -> tuple[DeclaredControl, ...]:
    """The fixture's witnessed cases restated as literal declared controls."""
    controls = []
    for case in spec.cases:
        outcome = case.challenge.outcome
        assert isinstance(outcome, dict)
        controls.append(
            DeclaredControl(
                id=case.challenge.id,
                lineage=case.lineage,
                partition=case.partition,
                expected=case.challenge.expected,
                outcome=outcome,
                rationale=case.challenge.rationale,
            )
        )
    return tuple(controls)


def controls_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, ControlWitnessSpec]:
    """A registered task, its feasible study, and its controls declared as outcomes."""
    project, task, spec = qualified_fixture(tmp_path, monkeypatch)
    return project, ControlWitnessSpec(
        task=task, feasibility=spec.feasibility, controls=declared(spec)
    )


def test_declared_controls_become_a_qualifying_witnessed_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Witnesses use the job's specification by default, and the result qualifies."""
    project, spec = controls_fixture(tmp_path, monkeypatch)
    planned = preflight_controls(project, spec)
    assert all(case.challenge.witness is None for case in planned.cases)
    qualification = witness_controls(project, spec, created_at=NOW, code_revision="test")
    assert all(case.challenge.witness is not None for case in qualification.cases)
    report = assess_executable_qualification(project, qualification)
    assert report.qualified, report.refusals
    assert witness_controls(project, spec, created_at=NOW, code_revision="test") == qualification


def test_branch_controls_apply_edits_to_the_saved_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A branch-based control edits a copy of the saved outcome; bad bases are refused."""
    project, spec = controls_fixture(tmp_path, monkeypatch)
    edited = spec.controls[2].model_copy(
        update={
            "outcome": None,
            "branch": 0,
            "edits": ({"op": "set", "path": ["final_state", "rows"], "value": 7},),
        }
    )
    changed = spec.model_copy(update={"controls": (*spec.controls[:2], edited, *spec.controls[3:])})
    planned = preflight_controls(project, changed)
    outcome, saved = planned.cases[2].challenge.outcome, spec.controls[2].outcome
    assert isinstance(outcome, dict) and saved is not None
    final, original = outcome["final_state"], saved["final_state"]
    assert isinstance(final, dict) and isinstance(original, dict)
    assert final == {**original, "rows": 7}
    missing = edited.model_copy(update={"branch": 9})
    with pytest.raises(ValueError, match="names branch 9, but the feasibility study has"):
        preflight_controls(project, changed.model_copy(update={"controls": (missing,)}))
    broken = edited.model_copy(update={"edits": ({"op": "set", "path": ["absent", "x"]},)})
    with pytest.raises(ValueError, match="edit does not apply"):
        preflight_controls(project, changed.model_copy(update={"controls": (broken,)}))
    with pytest.raises(ValueError, match="exactly one of branch or outcome"):
        DeclaredControl.model_validate(edited.model_dump(mode="json") | {"outcome": {}})


def test_controls_refuse_what_qualification_would_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A contradicted label and a specification that is the task's own verifier both fail."""
    project, spec = controls_fixture(tmp_path, monkeypatch)
    wrong = spec.controls[2].model_copy(update={"expected": "pass"})
    mislabeled = spec.model_copy(update={"controls": (*spec.controls[:2], wrong)})
    with pytest.raises(EvidenceAdmissionError, match="establishes fail, but the label declares"):
        witness_controls(project, mislabeled, created_at=NOW, code_revision="test")
    verifier = load_executable_task(project, spec.task).job.study.verifier
    own = register_specification(
        project,
        SpecificationRegistration(
            specification=EvaluationSpecification(
                description="The task's own verifier.", predicate=verifier, subject="outcome"
            )
        ),
        created_at=NOW,
        code_revision="test",
    )
    with pytest.raises(EvidenceAdmissionError, match="self_approval"):
        witness_controls(
            project,
            spec.model_copy(update={"specification": own}),
            created_at=NOW,
            code_revision="test",
        )


def test_specification_registration_is_idempotent_and_checks_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical content saves once; a source that differs from its digest is refused."""
    project, spec = controls_fixture(tmp_path, monkeypatch)
    verifier = load_executable_task(project, spec.task).job.study.verifier
    registration = SpecificationRegistration(
        specification=EvaluationSpecification(
            description="Independent check.", predicate=verifier, subject="native_observation"
        ),
        sources=(spec.feasibility,),
    )
    preflight_specification(project, registration)
    first = register_specification(project, registration, created_at=NOW, code_revision="test")
    assert register_specification(project, registration, created_at=NOW, code_revision="a") == first
    changed = ArtifactInput(artifact_id=spec.feasibility.artifact_id, sha256="0" * 64)
    stale = registration.model_copy(update={"sources": (changed,)})
    with pytest.raises(ValueError, match="differs from its digest"):
        preflight_specification(project, stale)
    with pytest.raises(ValueError, match="native-observation specifications are witnessed"):
        preflight_controls(project, spec.model_copy(update={"specification": first}))
