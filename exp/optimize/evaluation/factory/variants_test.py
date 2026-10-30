"""Candidates preserve native restore identity without inheriting approval or training labels."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.intervention import (
    ActionInterventionSpec,
    capture_action_intervention,
)
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.executable.package import export_executable_harbor
from exp.optimize.evaluation.executable.task import (
    ExecutableTask,
    load_executable_task,
    persist_executable_task,
)
from exp.optimize.evaluation.executable.trial import open_executable_harbor_trial
from exp.optimize.evaluation.factory.variants import (
    TaskVariantSpec,
    load_task_variant,
    publish_task_variant,
)
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.simulation.capability_briefs import persist_capability_brief
from exp.simulation.capability_briefs_test import brief
from exp.simulation.continuations import load_continuation_checkpoint


def variant_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, TaskVariantSpec]:
    """Register an executable base and a visibly unresolved capability hypothesis."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    now = datetime.now(UTC)
    intent = "Delete one data row while retaining the input."
    base = persist_executable_task(
        project,
        ExecutableTask(intent=intent, job=job, output_paths=("output.xlsx",)),
        created_at=now,
        code_revision="test",
    )
    capability = persist_capability_brief(
        project.artifacts,
        brief().model_copy(update={"outcome": intent}),
        created_at=now,
        code_revision="test",
    )
    return project, TaskVariantSpec(
        base=base,
        kind="rephrasing",
        intent=intent,
        prompt="Remove the first data row. Keep the source workbook intact.",
        verifier=job.study.verifier,
        brief=capability,
    )


def test_rephrased_candidate_runs_native_world_with_original_restore_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actor wording can change without rebuilding native state or inheriting old approval."""
    project, spec = variant_fixture(tmp_path, monkeypatch)
    reference = publish_task_variant(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    result = load_task_variant(project, reference)
    base = load_executable_task(project, spec.base)
    parent = load_continuation_checkpoint(project.artifacts, base.job.study.checkpoint)
    point = load_continuation_checkpoint(project.artifacts, result.checkpoint)
    assert point.world == parent.world and point.world_task == parent.task
    assert point.task.task_id != parent.task.task_id
    assert point.task.lineage_group_id == parent.task.lineage_group_id == result.lineage
    assert point.task.source_trace_ids == parent.task.source_trace_ids
    assert point.actor.messages[0].content == spec.prompt
    task = load_executable_task(project, result.task)
    assert task.job.study.training_specification is None
    assert task.job.study.critic_prediction is None and task.job.study.selection_plan is None
    assert task.job.maximum_model_calls == base.job.maximum_model_calls
    with pytest.raises(ValueError, match="unqualified"):
        export_executable_harbor(project, result.task, tmp_path / "refused", name="test/variant")
    package = export_executable_harbor(
        project, result.task, tmp_path / "diagnostic", name="test/variant", diagnostic=True
    )
    assert spec.prompt in (package / "instruction.md").read_text()
    with open_executable_harbor_trial(
        project, result.task, trial_id="variant", code_revision="test"
    ) as trial:
        trial.call(workbook_case().first)
        trial.submit({"answer": "done"})
        outcome = trial.result()
        assert trial.runtime.probe(trial.session.checkpoint())["rows"] == 6
        mechanical = outcome["mechanical"]
        assert isinstance(mechanical, dict) and mechanical["status"] == "pass"
        assert outcome["reward"] is None
    base.job.runtime_file.unlink()
    assert (
        publish_task_variant(project, spec, created_at=datetime.now(UTC), code_revision="test")
        == reference
    )


def test_state_variant_uses_executed_receipt_and_preserves_its_public_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """State variation is a saved world transition, not an authored claim about changed files."""
    project, spec = variant_fixture(tmp_path, monkeypatch)
    base = load_executable_task(project, spec.base)
    intervention = ActionInterventionSpec(
        base=base.job, action=workbook_case().first, rationale="Start after one native deletion."
    )
    applied = capture_action_intervention(project, intervention, code_revision="test")
    receipt = artifact_input(
        project.artifacts.read(intervention.identity("test") + "-result").manifest
    )
    spec = spec.model_copy(update={"kind": "state_variation", "intervention": receipt})
    reference = publish_task_variant(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    point = load_continuation_checkpoint(
        project.artifacts, load_task_variant(project, reference).checkpoint
    )
    parent = load_continuation_checkpoint(project.artifacts, applied.checkpoint)
    assert point.world == parent.world and point.actor.messages[-1] == parent.actor.messages[-1]
    assert point.actor.calls == parent.actor.calls
    with pytest.raises(ValueError, match="only state_variation"):
        publish_task_variant(
            project,
            spec.model_copy(update={"kind": "drill"}),
            created_at=datetime.now(UTC),
            code_revision="test",
        )


def test_rephrasing_cannot_silently_weaken_the_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing grading criteria creates a different task treatment, despite similar wording."""
    project, spec = variant_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(update={"verifier": spec.verifier.model_copy(update={"rules": ()})})
    with pytest.raises(ValueError, match="same intent and verifier"):
        publish_task_variant(project, spec, created_at=datetime.now(UTC), code_revision="test")
    assert not project.artifacts.exists(spec.identity("test"))


def test_contrast_treatment_can_seed_variant_but_changed_actor_policy_cannot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repetition metadata may differ; runtime, task and sampling identity still must match."""
    from exp.optimize.evaluation.continuations.action_contrast import (
        ActionContrastSpec,
        run_action_contrast,
    )

    project, variant = variant_fixture(tmp_path, monkeypatch)
    base = load_executable_task(project, variant.base)
    action = workbook_case().first
    contrast = ActionContrastSpec(
        base=base.job,
        baseline=action,
        alternative=action.model_copy(
            update={
                "arguments": {
                    **action.arguments,
                    "deletions": [{"sheet": "Orders", "start": 2, "count": 2}],
                }
            }
        ),
        hypothesis="Test a native decision before making a drill.",
    )
    report = run_action_contrast(project, contrast, code_revision="test")
    receipt = report.cells[0].intervention
    assert receipt is not None
    spec = variant.model_copy(update={"kind": "state_variation", "intervention": receipt})
    reference = publish_task_variant(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    candidate = load_executable_task(project, load_task_variant(project, reference).task)
    assert candidate.job.study.training_specification is None
    altered = base.model_copy(
        update={
            "job": base.job.model_copy(
                update={"study": base.job.study.model_copy(update={"temperature": 0.01})}
            )
        }
    )
    changed_base = persist_executable_task(
        project, altered, created_at=datetime.now(UTC), code_revision="test"
    )
    with pytest.raises(ValueError, match="exact base task"):
        publish_task_variant(
            project,
            spec.model_copy(update={"base": changed_base}),
            created_at=datetime.now(UTC),
            code_revision="test",
        )
