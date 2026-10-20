"""Paired workbook decisions cannot erase policy violations or missing draws."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from filelock import Timeout
from pydantic import BaseModel

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonValue
from exp.common.project import ArtifactManifest, ArtifactStore, ProjectStore
from exp.optimize.evaluation.continuations import action_contrast
from exp.optimize.evaluation.continuations.action_contrast import (
    ActionContrastSpec,
    ContrastCell,
    run_action_contrast,
)
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import ContinuationStudy
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport


def contrast_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, ActionContrastSpec, ScriptedJsonTransport]:
    """Use one native start, delete-one/delete-two treatments and four finish-only suffixes."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    baseline = workbook_case().first
    alternative = baseline.model_copy(
        update={
            "arguments": {
                **baseline.arguments,
                "deletions": [{"sheet": "Orders", "start": 2, "count": 2}],
            }
        }
    )
    response = JsonHttpResponse(
        status_code=200,
        body={
            "id": "finish",
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
    transport = ScriptedJsonTransport((response,) * 4)
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: transport
    )
    return (
        project,
        ActionContrastSpec(
            base=job,
            baseline=baseline,
            alternative=alternative,
            hypothesis="Deleting another row changes goal attainment.",
        ),
        transport,
    )


def test_paired_draws_keep_same_start_and_budget_with_complete_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AB/BA fresh draws produce the expected real workbook contrast and exact saved replay."""
    project, spec, transport = contrast_fixture(tmp_path, monkeypatch)
    report = run_action_contrast(project, spec, code_revision="test")
    assert [(cell.pair, cell.arm) for cell in report.cells] == [
        (0, "baseline"),
        (0, "alternative"),
        (1, "alternative"),
        (1, "baseline"),
    ]
    assert [cell.assessment.status for cell in report.cells if cell.assessment] == [
        "pass",
        "fail",
        "fail",
        "pass",
    ]
    assert report.metrics["alternative_minus_baseline_bounds"] == [-1.0, -1.0]
    assert len(transport.requests) == 4
    for cell in report.cells:
        assert cell.study is not None and cell.intervention is not None
        study = load_study(project.artifacts, cell.study)
        assert study.spec.training_specification is None and study.spec.critic_prediction is None
        assert study.spec.maximum_output_tokens == spec.base.study.maximum_output_tokens
        assert study.spec.repeats == 1
    spec.base.runtime_file.unlink()
    assert run_action_contrast(project, spec, code_revision="test") == report
    assert len(transport.requests) == 4


def test_imposed_policy_violation_is_not_erased_by_resuming_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finish-only suffixes satisfy the read-only policy alone, but their imposed edits do not."""
    project, spec, transport = contrast_fixture(tmp_path, monkeypatch)
    verifier = spec.base.study.verifier.model_copy(update={"allowed_tools": ()})
    job = spec.base.model_copy(
        update={"study": spec.base.study.model_copy(update={"verifier": verifier})}
    )
    spec = spec.model_copy(update={"base": job})
    report = run_action_contrast(project, spec, code_revision="test")
    assert len(transport.requests) == 4
    for cell in report.cells:
        assert cell.assessment is not None and cell.assessment.policy is not None
        assert cell.assessment.policy.status == "fail" and cell.assessment.status == "fail"
        assert cell.study is not None
        suffix = load_study(project.artifacts, cell.study)
        assessment = suffix.branches[0].assessment
        assert assessment is not None and assessment.policy is not None
        assert assessment.policy.status == "pass"


def test_unavailable_suffix_keeps_the_planned_denominator_and_cannot_redraw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One operationally unknown arm widens bounds instead of being treated as a miss or dropped."""
    from exp.optimize.evaluation.continuations import action_contrast

    project, spec, transport = contrast_fixture(tmp_path, monkeypatch)
    original = action_contrast.execute_continuation_job
    calls = []

    def fail_once(
        project: ProjectStore, job: ContinuationJob, *, code_revision: str
    ) -> ContinuationStudy:
        """Interrupt one treatment's suffix before returning a result."""
        calls.append(job.study.checkpoint)
        if len(calls) == 1:
            raise RuntimeError("unknown provider outcome")
        return original(project, job, code_revision=code_revision)

    monkeypatch.setattr(action_contrast, "execute_continuation_job", fail_once)
    report = run_action_contrast(project, spec, code_revision="test")
    assert report.cells[0].study is None and report.cells[0].failure == "RuntimeError"
    assert report.metrics["baseline"] == {
        "planned": 2,
        "assessable": 1,
        "successes": 1,
        "unknown": 1,
        "success_bounds": [0.5, 1.0],
        "goal_failures": 0,
        "policy_failures": 0,
    }
    assert report.metrics["alternative_minus_baseline_bounds"] == [-1.0, -0.5]
    assert len(transport.requests) == 3
    assert run_action_contrast(project, spec, code_revision="test") == report
    assert len(transport.requests) == 3 and len(calls) == 4


def test_parent_interruption_adopts_saved_action_and_suffix_without_new_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child measurement survives a failed parent write and keeps its original identity."""
    project, spec, transport = contrast_fixture(tmp_path, monkeypatch)
    write = project.artifacts.write_json
    lost = []

    def lose_parent(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Fail after the first suffix is complete, before the parent's receipt commits."""
        if artifact_type == "action-contrast-cell" and not lost:
            cell = files["cell.json"]
            assert isinstance(cell, ContrastCell)
            lost.append(cell.study)
            raise OSError("lost parent write")
        return write(
            artifact_id=artifact_id, artifact_type=artifact_type, envelope=envelope, files=files
        )

    monkeypatch.setattr(project.artifacts, "write_json", lose_parent)
    with pytest.raises(OSError, match="lost parent"):
        run_action_contrast(project, spec, code_revision="test")
    monkeypatch.setattr(project.artifacts, "write_json", write)
    report = run_action_contrast(project, spec, code_revision="test")
    assert report.cells[0].study == lost[0] and report.cells[0].failure is None
    assert len(transport.requests) == 4


@pytest.mark.parametrize("status", ["failed", "outcome_unknown"])
def test_interrupted_suffix_cannot_pass_from_partial_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """Partial goal-satisfying state stays unknown when its execution never completed."""
    project, spec, _ = contrast_fixture(tmp_path, monkeypatch)
    original = action_contrast.load_study

    def interrupted_study(store: ArtifactStore, reference: ArtifactInput) -> ContinuationStudy:
        """Retain observed state while simulating a later operational interruption."""
        study = original(store, reference)
        return study.model_copy(
            update={
                "branches": tuple(
                    branch.model_copy(update={"status": status}) for branch in study.branches
                )
            }
        )

    monkeypatch.setattr(action_contrast, "load_study", interrupted_study)
    report = run_action_contrast(project, spec, code_revision="test")
    for cell in report.cells:
        assert cell.assessment is not None
        assert not cell.assessment.complete
        assert cell.assessment.status == "unassessable"
    metrics = report.metrics["baseline"]
    assert isinstance(metrics, dict)
    assert metrics["success_bounds"] == [0.0, 1.0]


def test_deferred_child_resumes_the_same_registered_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known contention stays pending and retry reuses the child's original execution identity."""
    project, spec, transport = contrast_fixture(tmp_path, monkeypatch)
    original = action_contrast.execute_continuation_job
    blocked = []

    def busy_once(
        project: ProjectStore, job: ContinuationJob, *, code_revision: str
    ) -> ContinuationStudy:
        """Inject one child lease refusal before delegating to durable native execution."""
        if not blocked:
            blocked.append(job)
            raise Timeout("child.lock")
        return original(project, job, code_revision=code_revision)

    monkeypatch.setattr(action_contrast, "execute_continuation_job", busy_once)
    with pytest.raises(Timeout):
        run_action_contrast(project, spec, code_revision="test")
    assert not project.artifacts.exists(spec.identity("test") + "-cell-0")
    report = run_action_contrast(project, spec, code_revision="test")
    assert report.cells[0].study is not None and report.cells[0].failure is None
    assert run_action_contrast(project, spec, code_revision="test") == report
