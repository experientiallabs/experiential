"""Objective qualification tests use independently authored predicates and actual XLSX execution."""

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import execute_continuation_job
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.executable.qualification import (
    ExecutableQualificationSpec,
    assess_executable_qualification,
    load_executable_qualification,
    qualify_executable_task,
)
from exp.optimize.evaluation.executable.task import (
    ExecutableTask,
    load_executable_task,
    persist_executable_task,
)
from exp.simulation.evaluation.admission import record_label_witness
from exp.simulation.evaluation.calibration import CalibrationCase
from exp.simulation.evaluation.tasks import OutcomeRule, VerifierChallenge


def qualified_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ProjectStore, ArtifactInput, ExecutableQualificationSpec]:
    """Run a real feasible workbook branch and make explicit synthetic verifier controls."""
    project, job, _ = job_fixture(tmp_path, monkeypatch, pin_policy=True)
    result = execute_continuation_job(project, job, code_revision="test")
    study_id = next(
        key
        for key in project.artifacts.list_ids(artifact_type="continuation-study")
        if key.endswith("-result")
        and project.artifacts.read(key).manifest.code_revision == job.implementation("test")
    )
    feasibility = artifact_input(project.artifacts.read(study_id).manifest)
    task = ExecutableTask(
        intent="Delete one data row while retaining the input.",
        job=job,
        output_paths=("output.xlsx",),
    )
    reference = persist_executable_task(
        project, task, created_at=datetime.now(UTC), code_revision="test"
    )
    cases = []
    assert job.study.training_specification is not None
    for partition, offset in (("development", 0), ("confirmation", 10)):
        for index, kind in enumerate(
            ("valid-one", "valid-two", "goal-miss", "side-effect", "unknown")
        ):
            outcome = deepcopy(result.branches[0].outcome)
            # These are specification witnesses over explicit counterexamples, not execution labels.
            final_state = outcome["final_state"]
            steps = outcome["steps"]
            assert isinstance(final_state, dict) and isinstance(steps, list)
            if kind == "goal-miss":
                final_state["rows"] = 7 if partition == "development" else 8
            elif kind == "side-effect":
                final_state["input_digest"] = "modified " + partition
            elif kind == "unknown":
                if partition == "development":
                    outcome["steps"] = None
                else:
                    outcome["stop_reason"] = "failure"
            else:
                step = steps[0]
                assert isinstance(step, dict)
                action = step["action"]
                assert isinstance(action, dict)
                arguments = action["arguments"]
                assert isinstance(arguments, dict)
                arguments["deletions"] = [
                    {"sheet": "Orders", "start": 2 + index + (2 if offset else 0), "count": 1}
                ]
            expected = (
                "pass"
                if kind.startswith("valid")
                else "unassessable"
                if kind == "unknown"
                else "fail"
            )
            witness = record_label_witness(
                project.artifacts,
                specification=job.study.training_specification,
                outcome=outcome,
                created_at=datetime.now(UTC),
                code_revision="test",
            )
            cases.append(
                CalibrationCase(
                    challenge=VerifierChallenge(
                        id=f"{partition}-{kind}",
                        outcome=outcome,
                        expected=expected,
                        authority="synthetic_specification",
                        witness=witness,
                        rationale="Authored observation-level verifier challenge.",
                    ),
                    lineage=f"authored-control-{offset + index}",
                    partition=partition,
                )
            )
    return (
        project,
        reference,
        ExecutableQualificationSpec(task=reference, feasibility=feasibility, cases=tuple(cases)),
    )


def test_witnessed_controls_qualify_only_observable_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A feasible run plus correctly handled positive, negative and unknown controls is admitted."""
    project, _, spec = qualified_fixture(tmp_path, monkeypatch)
    report = assess_executable_qualification(project, spec)
    assert report.qualified, report.refusals
    reference = qualify_executable_task(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    assert load_executable_qualification(project, reference) == report
    assert "containment" in report.scope and "tier-separation" in report.scope


def test_feasibility_from_another_runtime_cannot_qualify_the_same_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matching task and verifier settings do not prove execution under the new runtime file."""
    project, reference, spec = qualified_fixture(tmp_path, monkeypatch)
    task = load_executable_task(project, reference)
    from exp.common.core.artifacts import sha256_bytes

    runtime_file = tmp_path / "different_runtime.py"
    runtime_file.write_bytes(task.job.runtime_file.read_bytes() + b"\n# different runtime\n")
    changed = task.model_copy(
        update={
            "job": task.job.model_copy(
                update={
                    "runtime_file": runtime_file,
                    "runtime_sha256": sha256_bytes(runtime_file.read_bytes()),
                }
            )
        }
    )
    new_reference = persist_executable_task(
        project, changed, created_at=datetime.now(UTC), code_revision="test"
    )
    report = assess_executable_qualification(
        project, spec.model_copy(update={"task": new_reference})
    )
    assert not report.qualified
    assert any("runtime" in reason for reason in report.refusals)


@pytest.mark.parametrize("check", ["arguments_only", "full_output"])
def test_argument_rules_alone_cannot_stand_in_for_a_state_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check: str
) -> None:
    """Rules over named call arguments leave unrequested writes unobserved."""
    project, reference, spec = qualified_fixture(tmp_path, monkeypatch)
    task = load_executable_task(project, reference)
    verifier = task.job.study.verifier
    pinned = OutcomeRule(path="/final_state/files/output.xlsx", operator="equals", value="0" * 64)
    weakened = verifier.model_copy(
        update={
            "preserved_paths": (),
            "argument_rules": {
                "edit_workbook": (
                    OutcomeRule(path="/destination", operator="equals", value="output.xlsx"),
                )
            },
            "rules": (*verifier.rules, pinned) if check == "full_output" else verifier.rules,
        }
    )
    job = task.job.model_copy(
        update={"study": task.job.study.model_copy(update={"verifier": weakened})}
    )
    changed = persist_executable_task(
        project,
        task.model_copy(update={"job": job}),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    report = assess_executable_qualification(project, spec.model_copy(update={"task": changed}))
    assert not report.qualified
    refused = any("state-preservation or full-output" in reason for reason in report.refusals)
    assert refused == (check == "arguments_only")


def test_removing_collateral_damage_control_blocks_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outcome-only challenges cannot certify a task that needs a side-effect policy."""
    project, _, spec = qualified_fixture(tmp_path, monkeypatch)
    weakened = spec.model_copy(
        update={
            "cases": tuple(case for case in spec.cases if "side-effect" not in case.challenge.id)
        }
    )
    report = assess_executable_qualification(project, weakened)
    assert not report.qualified
    assert any("prohibited-effect" in refusal for refusal in report.refusals)


def test_claimed_approval_without_witness_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Human-free qualification requires re-derived source witnesses rather than booleans."""
    project, _, spec = qualified_fixture(tmp_path, monkeypatch)
    cases = tuple(
        case.model_copy(
            update={
                "challenge": case.challenge.model_copy(
                    update={
                        "witness": None,
                        "authority": "human",
                        "approved": True,
                    }
                )
            }
        )
        for case in spec.cases
    )
    assert not assess_executable_qualification(
        project, spec.model_copy(update={"cases": cases})
    ).qualified


def test_native_record_replay_is_explicit_and_checks_runtime_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit replay refuses simulated domains and checks the source pin before importing."""
    from exp.optimize.evaluation.executable.qualification import assess_executable_qualification
    from exp.optimize.evaluation.executable.task import load_executable_task

    project, task_ref, spec = qualified_fixture(tmp_path, monkeypatch)
    native = spec.model_copy(update={"replay_native_records": True})
    with pytest.raises(ValueError, match="wholly native"):
        assess_executable_qualification(project, native)
    task = load_executable_task(project, task_ref)
    task.job.runtime_file.write_text(task.job.runtime_file.read_text() + "\n# changed runtime\n")
    with pytest.raises(ValueError, match="runtime source changed"):
        assess_executable_qualification(project, native)
    assert assess_executable_qualification(project, spec).qualified
