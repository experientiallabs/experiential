"""Matched experiments execute real workbook continuations and preserve saved child measurements."""

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
from filelock import Timeout
from pydantic import BaseModel

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonValue, sha256_json
from exp.common.project import ArtifactManifest, ProjectStore
from exp.optimize.evaluation.continuations import experiment as owner
from exp.optimize.evaluation.continuations.critic import CriticSpec
from exp.optimize.evaluation.continuations.experiment import (
    CheckpointExperimentSpec,
    ExperimentCell,
    run_checkpoint_experiment,
)
from exp.optimize.evaluation.continuations.inference import experiment_inference
from exp.optimize.evaluation.continuations.job import ContinuationJob, continuation_result_reference
from exp.optimize.evaluation.continuations.selection import persist_selection_plan
from exp.optimize.evaluation.continuations.selection_test import selection_fixture
from exp.optimize.evaluation.continuations.study import ContinuationStudy
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.simulation.continuations import load_continuation_checkpoint


def test_selected_and_random_points_execute_same_budget_and_replay_without_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matched-arm misses and unavailable branches stay visible, with no causal interpretation."""
    project, selection = selection_fixture(tmp_path, monkeypatch)
    reference = persist_selection_plan(
        project, selection, created_at=datetime.now(UTC), code_revision="test"
    )
    spec = CheckpointExperimentSpec(selection=reference)
    report = run_checkpoint_experiment(project, spec, code_revision="test")
    assert len(report.cells) == 2 and report.unmatched == 0
    assert [cell.arm for cell in report.cells] == ["proposed", "random"]
    for cell in report.cells:
        capsule = load_continuation_checkpoint(project.artifacts, cell.job.study.checkpoint)
        assert capsule.actor.calls == 1
        assert cell.job.maximum_model_calls == 4
        assert cell.job.study.selection_plan == reference
        assert cell.result == continuation_result_reference(project, cell.job, code_revision="test")
    random = report.metrics["random"]
    assert isinstance(random, dict) and random["planned_branches"] == 2
    assert random["unknown"] == 2
    # Unavailable provider replies are unknown, not an actor that spent its output allowance.
    assert (random["actor_budget_exhausted"], random["actor_budget_exhausted_share"]) == (0, 0.0)
    selection.base.runtime_file.unlink()
    assert run_checkpoint_experiment(project, spec, code_revision="test") == report


def test_parent_crash_adopts_complete_child_instead_of_losing_or_redrawing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A measured child remains the same measurement if its parent receipt failed to publish."""
    project, selection = selection_fixture(tmp_path, monkeypatch)
    reference = persist_selection_plan(
        project, selection, created_at=datetime.now(UTC), code_revision="test"
    )
    spec = CheckpointExperimentSpec(selection=reference)
    write = project.artifacts.write_json
    lost: list[ArtifactInput | None] = []

    def lose_parent(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Fail once after a complete continuation result already exists."""
        if artifact_type == "checkpoint-experiment-cell" and not lost:
            cell = files["cell.json"]
            assert isinstance(cell, ExperimentCell)
            lost.append(cell.result)
            raise OSError("lost parent publication")
        return write(
            artifact_id=artifact_id, artifact_type=artifact_type, envelope=envelope, files=files
        )

    monkeypatch.setattr(project.artifacts, "write_json", lose_parent)
    with pytest.raises(OSError, match="lost parent"):
        run_checkpoint_experiment(project, spec, code_revision="test")
    monkeypatch.setattr(project.artifacts, "write_json", write)
    report = run_checkpoint_experiment(project, spec, code_revision="test")
    assert isinstance(lost[0], ArtifactInput)
    assert report.cells[0].result == lost[0]
    assert report.cells[0].failure is None


def test_matched_experiment_records_critic_before_each_fresh_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prospective probabilities bind fresh branches; unknown branches do not acquire labels."""
    from exp.common.core.artifacts import sha256_json
    from exp.optimize.evaluation.continuations.critic import CriticSpec, load_critic_prediction
    from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport

    project, selection = selection_fixture(tmp_path, monkeypatch)
    job = selection.base
    critic = CriticSpec(
        checkpoint=job.study.checkpoint,
        actor_model=job.study.actor_model,
        verifier_sha256=sha256_json(job.study.verifier),
        probe_sha256=job.study.probe_sha256,
        maximum_model_calls=job.maximum_model_calls,
        system_prompt=job.system_prompt,
        temperature=job.study.temperature,
        actor_output_tokens=job.study.maximum_output_tokens,
        critic_alias=job.actor_alias,
        critic_model=job.study.actor_model,
        catalog_sha256=job.catalog_sha256,
        deadline=job.deadline,
    )
    response = JsonHttpResponse(
        status_code=200,
        body={
            "id": "estimate",
            "model": "world-v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "estimate",
                                "type": "function",
                                "function": {
                                    "name": "estimate_success",
                                    "arguments": (
                                        '{"probability": 0.75, "rationale": "Visible prefix."}'
                                    ),
                                },
                            }
                        ],
                    },
                }
            ],
        },
    )
    transport = ScriptedJsonTransport((response, response))
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.critic.HttpxJsonTransport", lambda: transport
    )
    reference = persist_selection_plan(
        project, selection, created_at=datetime.now(UTC), code_revision="test"
    )
    spec = CheckpointExperimentSpec(selection=reference, critic=critic)
    report = run_checkpoint_experiment(project, spec, code_revision="test")
    for cell in report.cells:
        prediction_ref = cell.job.study.critic_prediction
        assert prediction_ref is not None and cell.result is not None
        prediction = load_critic_prediction(project.artifacts, prediction_ref)
        assert prediction.spec.checkpoint == cell.job.study.checkpoint
        assert project.artifacts.read(prediction_ref.artifact_id).manifest.created_at < (
            project.artifacts.read(cell.result.artifact_id).manifest.created_at
        )
    random = report.metrics["random"]
    assert isinstance(random, dict)
    assert random["mean_checkpoint_brier"] is None and random["unknown"] == 2
    job.runtime_file.unlink()
    assert run_checkpoint_experiment(project, spec, code_revision="test") == report
    assert len(transport.requests) == 2


def test_deferred_child_resumes_the_same_registered_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known contention stays pending and retry reuses the child's original execution identity."""
    project, selection = selection_fixture(tmp_path, monkeypatch)
    reference = persist_selection_plan(
        project, selection, created_at=datetime.now(UTC), code_revision="test"
    )
    spec = CheckpointExperimentSpec(selection=reference)
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
        run_checkpoint_experiment(project, spec, code_revision="test")
    assert not project.artifacts.exists(spec.identity("test") + "-cell-0")
    report = run_checkpoint_experiment(project, spec, code_revision="test")
    assert report.cells[0].result is not None and report.cells[0].failure is None
    assert run_checkpoint_experiment(project, spec, code_revision="test") == report


def _truncated_critic(
    selection_job: ContinuationJob, monkeypatch: pytest.MonkeyPatch
) -> tuple[CriticSpec, ScriptedJsonTransport]:
    """A critic whose every draw ends at its output limit while reasoning, with no estimate."""
    critic = CriticSpec(
        checkpoint=selection_job.study.checkpoint,
        actor_model=selection_job.study.actor_model,
        verifier_sha256=sha256_json(selection_job.study.verifier),
        probe_sha256=selection_job.study.probe_sha256,
        maximum_model_calls=selection_job.maximum_model_calls,
        system_prompt=selection_job.system_prompt,
        temperature=selection_job.study.temperature,
        actor_output_tokens=selection_job.study.maximum_output_tokens,
        critic_alias=selection_job.actor_alias,
        critic_model=selection_job.study.actor_model,
        catalog_sha256=selection_job.catalog_sha256,
        deadline=selection_job.deadline,
    )
    truncated = JsonHttpResponse(
        status_code=200,
        body={
            "id": "truncated",
            "model": "world-v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "Weighing the visible prefix"},
                }
            ],
        },
    )
    transport = ScriptedJsonTransport((truncated, truncated))
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.critic.HttpxJsonTransport", lambda: transport
    )
    return critic, transport


def test_a_failed_critic_still_measures_its_cell_and_counts_as_a_critic_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An estimate cut off at its output limit leaves the arms measured and the miss counted."""
    project, selection = selection_fixture(tmp_path, monkeypatch)
    critic, transport = _truncated_critic(selection.base, monkeypatch)
    reference = persist_selection_plan(
        project, selection, created_at=datetime.now(UTC), code_revision="test"
    )
    spec = CheckpointExperimentSpec(selection=reference, critic=critic)
    report = run_checkpoint_experiment(project, spec, code_revision="test")
    for cell in report.cells:
        assert cell.result is not None and cell.failure is None
        assert cell.critic_failure == "finish_reason:length"
        assert cell.job.study.critic_prediction is None
        assert cell.job.study.predicted_success is None
    for arm in ("proposed", "random"):
        metrics = report.metrics[arm]
        assert isinstance(metrics, dict)
        assert metrics["critic_failures"] == 1
        assert metrics["critic_failure_kinds"] == {"finish_reason:length": 1}
        assert metrics["critic_assessable_checkpoints"] == 0
        arms = experiment_inference(project, report)["arms"]
        assert isinstance(arms, dict) and isinstance(arms[arm], dict)
        assert arms[arm]["critic"] == {
            "forecast_checkpoints": 0,
            "critic_failures": 1,
            "status": "no forecast with known outcomes",
        }
    assert len(transport.requests) == 2
    selection.base.runtime_file.unlink()
    assert run_checkpoint_experiment(project, spec, code_revision="test") == report
    assert len(transport.requests) == 2


def test_a_saved_critic_failure_cannot_be_dropped_from_a_measured_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay refuses a report whose measured cell no longer carries its critic failure."""
    project, selection = selection_fixture(tmp_path, monkeypatch)
    critic, _ = _truncated_critic(selection.base, monkeypatch)
    reference = persist_selection_plan(
        project, selection, created_at=datetime.now(UTC), code_revision="test"
    )
    spec = CheckpointExperimentSpec(selection=reference, critic=critic)
    report = run_checkpoint_experiment(project, spec, code_revision="test")
    erased_cell = report.cells[0].model_copy(update={"critic_failure": None})
    erased = report.model_copy(update={"cells": (erased_cell, *report.cells[1:])})
    identity = spec.identity("test")
    forged = {
        (identity + "-result", "report.json"): erased.model_dump_json().encode(),
        (identity + "-cell-0", "cell.json"): erased_cell.model_dump_json().encode(),
    }
    read_bytes = project.artifacts.read_bytes
    monkeypatch.setattr(
        project.artifacts,
        "read_bytes",
        lambda artifact_id, path: forged.get((artifact_id, path)) or read_bytes(artifact_id, path),
    )
    with pytest.raises(ValueError, match="without its critic estimate or failure"):
        owner.saved_checkpoint_experiment(project, spec, code_revision="test")


def test_parent_crash_after_a_critic_failure_keeps_the_failure_without_redrawing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A measured child whose critic failed is adopted with the same saved failure reason."""
    project, selection = selection_fixture(tmp_path, monkeypatch)
    critic, transport = _truncated_critic(selection.base, monkeypatch)
    reference = persist_selection_plan(
        project, selection, created_at=datetime.now(UTC), code_revision="test"
    )
    spec = CheckpointExperimentSpec(selection=reference, critic=critic)
    write = project.artifacts.write_json
    lost: list[ExperimentCell] = []

    def lose_parent(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Fail once after a complete continuation result already exists."""
        if artifact_type == "checkpoint-experiment-cell" and not lost:
            cell = files["cell.json"]
            assert isinstance(cell, ExperimentCell)
            lost.append(cell)
            raise OSError("lost parent publication")
        return write(
            artifact_id=artifact_id, artifact_type=artifact_type, envelope=envelope, files=files
        )

    monkeypatch.setattr(project.artifacts, "write_json", lose_parent)
    with pytest.raises(OSError, match="lost parent"):
        run_checkpoint_experiment(project, spec, code_revision="test")
    monkeypatch.setattr(project.artifacts, "write_json", write)
    report = run_checkpoint_experiment(project, spec, code_revision="test")
    assert report.cells[0] == lost[0]
    assert report.cells[0].critic_failure == "finish_reason:length"
    assert report.cells[0].result is not None and report.cells[0].failure is None
    assert len(transport.requests) == 2
    assert run_checkpoint_experiment(project, spec, code_revision="test") == report
