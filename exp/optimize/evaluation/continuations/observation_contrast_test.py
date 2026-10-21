"""Observation experiments preserve native worlds, original policy failures and missing draws."""

from pathlib import Path
from typing import Never

import pytest
from filelock import Timeout

from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations import observation_contrast as owner
from exp.optimize.evaluation.continuations.intervention import (
    ActionIntervention,
    ActionInterventionSpec,
    capture_action_intervention,
)
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.job_test import OutageTransport, job_fixture
from exp.optimize.evaluation.continuations.observation_contrast import (
    ObservationContrastSpec,
    preflight_observation_contrast,
    run_observation_contrast,
)
from exp.optimize.evaluation.continuations.study import ContinuationStudy
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import JsonHttpResponse
from exp.simulation.continuations import load_continuation_checkpoint


def observation_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    forbidden: bool = False,
) -> tuple[ProjectStore, ObservationContrastSpec, OutageTransport]:
    """Execute a real workbook edit once and compare its acknowledgement with an authored error."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    if forbidden:
        verifier = job.study.verifier.model_copy(update={"allowed_tools": ()})
        job = job.model_copy(update={"study": job.study.model_copy(update={"verifier": verifier})})
    spec = ActionInterventionSpec(base=job, action=workbook_case().first, rationale="Native edit.")
    capture_action_intervention(project, spec, code_revision="test")
    effect = artifact_input(project.artifacts.read(spec.identity("test") + "-result").manifest)
    transport = OutageTransport(
        tuple(
            JsonHttpResponse(
                status_code=200,
                body={
                    "id": f"response-{index}",
                    "model": "world-v1",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "done"},
                        }
                    ],
                },
            )
            for index in range(2)
        )
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: transport
    )
    return (
        project,
        ObservationContrastSpec(
            intervention=effect,
            alternative='{"error":"write failed"}',
            hypothesis="Does a misleading acknowledgement change the subsequent workflow?",
            pairs=2,
        ),
        transport,
    )


def test_only_one_reply_changes_and_unknown_draws_remain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual workbook state is identical, calls are equal, approvals clear and unknowns persist."""
    project, spec, transport = observation_fixture(tmp_path, monkeypatch)
    report = run_observation_contrast(project, spec, code_revision="test")
    assert [cell.arm for cell in report.cells] == [
        "recorded",
        "alternative",
        "alternative",
        "recorded",
    ]
    points = [
        load_continuation_checkpoint(project.artifacts, cell.job.study.checkpoint)
        for cell in report.cells
    ]
    effect = ActionIntervention.model_validate_json(
        project.artifacts.read_bytes(spec.intervention.artifact_id, "intervention.json")
    )
    original = load_continuation_checkpoint(project.artifacts, effect.checkpoint)
    for cell, point in zip(report.cells, points, strict=True):
        assert point.world == original.world and point.task == original.task
        assert point.environment_task == original.environment_task
        assert point.actor.calls == original.actor.calls
        assert point.actor.messages[:-1] == original.actor.messages[:-1]
        assert point.actor.messages[-1].tool_call_id == original.actor.messages[-1].tool_call_id
        assert point.actor.messages[-1].content == (
            spec.alternative if cell.arm == "alternative" else original.actor.messages[-1].content
        )
        assert cell.job.study.training_specification is None
        assert cell.job.study.critic_prediction is None and cell.job.study.selection_plan is None
    metrics = report.metrics()
    assert (
        metrics["recorded"]
        == metrics["alternative"]
        == {
            "planned": 2,
            "assessable": 1,
            "successes": 1,
            "unknown": 1,
            "success_bounds": [0.5, 1.0],
        }
    )
    assert metrics["pairs"] == [{"pair": 0, "difference": 0}, {"pair": 1, "difference": None}]
    attempts = len(transport.requests)
    report.cells[0].job.runtime_file.unlink()
    project.model_catalog_path.unlink()
    assert run_observation_contrast(project, spec, code_revision="test") == report
    assert len(transport.requests) == attempts


def test_substituting_reply_cannot_erase_forbidden_prefix_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both final workbooks satisfy the goal, but an imposed prohibited edit still fails policy."""
    project, spec, _ = observation_fixture(tmp_path, monkeypatch, forbidden=True)
    report = run_observation_contrast(
        project, spec.model_copy(update={"pairs": 1}), code_revision="test"
    )
    for cell in report.cells:
        assert cell.assessment is not None
        assert cell.assessment.goal.status == "pass"
        assert cell.assessment.policy.status == "fail"
        assert cell.assessment.status == "fail"


def test_unknown_cell_dispatch_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process interruption after an intent does not buy the same continuation again."""
    project, spec, transport = observation_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(update={"pairs": 1})
    execute = owner.execute_continuation_job

    def interrupted(project: ProjectStore, job: ContinuationJob, *, code_revision: str) -> Never:
        """Interrupt after the parent's durable intent and before a recoverable child result."""
        raise KeyboardInterrupt

    monkeypatch.setattr(owner, "execute_continuation_job", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_observation_contrast(project, spec, code_revision="test")
    monkeypatch.setattr(owner, "execute_continuation_job", execute)
    report = run_observation_contrast(project, spec, code_revision="test")
    assert report.cells[0].failure == "UnfinishedDispatchIntent"
    assert report.cells[1].assessment is not None
    assert len(transport.requests) == 1


def test_identical_or_unbound_reply_sources_refuse_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arbitrary artifacts are not executed observations; identical replies have no contrast."""
    project, spec, transport = observation_fixture(tmp_path, monkeypatch)
    effect = ActionIntervention.model_validate_json(
        project.artifacts.read_bytes(spec.intervention.artifact_id, "intervention.json")
    )
    with pytest.raises(ValueError, match="different replies"):
        preflight_observation_contrast(
            project, spec.model_copy(update={"alternative": effect.observation.content})
        )
    with pytest.raises(ValueError, match="exact saved action"):
        preflight_observation_contrast(
            project, spec.model_copy(update={"intervention": effect.checkpoint})
        )
    assert not transport.requests


def test_completed_child_recovers_without_redrawing_after_parent_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved suffix remains its first measurement even when the parent lost its result."""
    project, spec, transport = observation_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(update={"pairs": 1})
    execute = owner.execute_continuation_job

    def save_then_interrupt(
        project: ProjectStore, job: ContinuationJob, *, code_revision: str
    ) -> ContinuationStudy:
        """Lose the parent's result only after the complete child was committed."""
        execute(project, job, code_revision=code_revision)
        raise KeyboardInterrupt

    monkeypatch.setattr(owner, "execute_continuation_job", save_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_observation_contrast(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    monkeypatch.setattr(owner, "execute_continuation_job", execute)
    report = run_observation_contrast(project, spec, code_revision="test")
    assert len(transport.requests) == 2
    assert all(
        cell.assessment is not None and cell.assessment.status == "pass" for cell in report.cells
    )


def test_deferred_child_resumes_the_same_registered_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known contention stays pending and retry reuses the child's original execution identity."""
    project, spec, transport = observation_fixture(tmp_path, monkeypatch)
    original = owner.execute_continuation_job
    blocked = []

    def busy_once(
        project: ProjectStore, job: ContinuationJob, *, code_revision: str
    ) -> ContinuationStudy:
        """Inject one child lease refusal before delegating to durable native execution."""
        if not blocked:
            blocked.append(job)
            raise Timeout("child.lock")
        return original(project, job, code_revision=code_revision)

    monkeypatch.setattr(owner, "execute_continuation_job", busy_once)
    with pytest.raises(Timeout):
        run_observation_contrast(project, spec, code_revision="test")
    assert not project.artifacts.exists(spec.identity("test") + "-cell-0")
    report = run_observation_contrast(project, spec, code_revision="test")
    assert report.cells[0].result is not None and report.cells[0].failure is None
    assert run_observation_contrast(project, spec, code_revision="test") == report
