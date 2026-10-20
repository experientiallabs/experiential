"""Explicit repair retains original scores and recovers a publication without redrawing."""

import json
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import ArtifactEnvelope, JsonValue
from exp.common.project import ProjectStore, artifact_input
from exp.common.project.manifests import ArtifactManifest
from exp.optimize.evaluation import learning_repair
from exp.optimize.evaluation.audit.panel_test import REVISION, build_project, episode
from exp.optimize.evaluation.fidelity.memory import saved_memory_receipt, trajectory_record
from exp.optimize.evaluation.fidelity.runner import PreparedFidelity
from exp.optimize.evaluation.learning.traces import (
    _saved_attempts,
    learn_from_traces,
    learning_cases,
    learning_curve,
    learning_plan,
)
from exp.optimize.evaluation.learning_repair import LearningRepairSpec, repair_learning
from exp.optimize.evaluation.tests.replay_retirement_test import RedrawnPrediction, now
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    memory_catalog,
    memory_head,
    memory_version_id,
    publish_trajectory,
)


def failed_run(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, PreparedFidelity, RedrawnPrediction, LearningRepairSpec]:
    """Leave a saved malformed investigation while unrelated learning advances memory."""
    episodes = [episode(i) for i in range(4)]
    provider = RedrawnPrediction(episodes)
    provider.failures = 1
    _, project, _ = build_project(root, monkeypatch, provider, episodes=episodes)
    outcome = learn_from_traces(
        project, recall_alias="jev", maximum_attempts=10_000, code_revision=REVISION, clock=now
    )
    assert outcome.incomplete == 1 and outcome.head_version == 3
    case = learning_cases(project, "all")[0]
    prepared = _saved_attempts(project, learning_plan(project), case)[0][0]
    head = memory_head(project.artifacts, outcome.plan.lineage)
    assert head is not None
    spec = LearningRepairSpec(
        run=artifact_input(project.artifacts.read(prepared.id).manifest),
        report=artifact_input(project.artifacts.read(prepared.id + "-report").manifest),
        parent=artifact_input(
            project.artifacts.read(memory_version_id(head.lineage, head.version)).manifest
        ),
        reason="Retain disclosed observations; abstain from the failed hypothesis.",
    )
    return project, prepared, provider, spec


def test_repair_advances_named_head_without_changing_any_prediction_or_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original miss remains scored while the unconsumed trace joins memory v4."""
    project, prepared, provider, spec = failed_run(tmp_path, monkeypatch)
    store = project.artifacts
    original = store.read_bytes(prepared.id + "-report", "report.json")
    first_receipt = store.read_bytes(prepared.id + "-memory", "memory.json")
    before_calls = len(provider.requests)
    before_curve = learning_curve(project)
    before_ids = set(store.list_ids())
    assert (
        repair_learning(project, spec, code_revision=REVISION, dry_run=True)["provider_calls"] == 0
    )
    assert set(store.list_ids()) == before_ids
    result = repair_learning(project, spec, code_revision=REVISION)
    assert result["published"] is True and result["version"] == 4
    assert result["original_parent_version"] == 0 and result["parent_version"] == 3
    assert repair_learning(project, spec, code_revision=REVISION) == result
    receipt = saved_memory_receipt(project, prepared)
    assert receipt is not None and receipt["repair"] == result["repair"]
    assert len(provider.requests) == before_calls and provider.draws == 1
    assert store.read_bytes(prepared.id + "-report", "report.json") == original
    assert store.read_bytes(prepared.id + "-memory", "memory.json") == first_receipt
    curve = learning_curve(project)
    assert curve is not None and before_curve is not None
    assert [(p.exact, p.steps) for p in curve.points] == [
        (p.exact, p.steps) for p in before_curve.points
    ]
    assert curve.incomplete == 0
    with pytest.raises(ValueError, match="different disposition"):
        repair_learning(
            project, spec.model_copy(update={"reason": "Different reason"}), code_revision=REVISION
        )


def test_stale_parent_refuses_before_writing_and_explicit_repair_keeps_unknowns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old empty parent cannot overwrite the advanced bank or silently rebase."""
    project, _, provider, spec = failed_run(tmp_path, monkeypatch)
    before = set(project.artifacts.list_ids())
    calls = len(provider.requests)
    with pytest.raises(ValueError, match="memory advanced"):
        repair_learning(project, spec.model_copy(update={"parent": None}), code_revision=REVISION)
    assert set(project.artifacts.list_ids()) == before and len(provider.requests) == calls


def test_crash_after_publication_recovers_same_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing repair receipt does not publish a second version after restart."""
    project, prepared, provider, spec = failed_run(tmp_path, monkeypatch)
    original_write = project.artifacts.write_json

    def crash(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Simulate process loss precisely after the memory owner published."""
        if artifact_type == "fidelity-learning-repair-result":
            raise KeyboardInterrupt("loss before receipt")
        return original_write(
            artifact_id=artifact_id, artifact_type=artifact_type, envelope=envelope, files=files
        )

    with monkeypatch.context() as patch:
        patch.setattr(project.artifacts, "write_json", crash)
        with pytest.raises(KeyboardInterrupt):
            repair_learning(project, spec, code_revision=REVISION)
    calls = len(provider.requests)
    result = repair_learning(project, spec, code_revision=REVISION)
    assert result["version"] == 4 and result["published"] is True
    assert len(provider.requests) == calls and provider.draws == 1
    receipt = saved_memory_receipt(project, prepared)
    assert receipt is not None and receipt["version"] == 4


def test_public_cli_repairs_then_replays_without_catalog_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The explicit operation publishes no replacement score and needs no provider setup."""
    project, prepared, provider, spec = failed_run(tmp_path, monkeypatch)
    command = tmp_path / "repair.json"
    command.write_text(
        ContinuationCommand(action="repair-learning", learning_repair=spec).model_dump_json()
    )
    args = [
        "eval",
        project.paths.project_id,
        "--root",
        str(project.paths.root),
        "--continuations",
        str(command),
    ]
    calls = len(provider.requests)
    for suffix in (["--dry-run"], ["--yes", "--non-interactive"], []):
        result = CliRunner().invoke(app, args + suffix)
        assert result.exit_code == 0, (result.output, result.exception)
    assert len(provider.requests) == calls
    receipt = saved_memory_receipt(project, prepared)
    assert receipt is not None and receipt["version"] == 4


def test_an_advanced_head_after_interruption_needs_a_new_explicit_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unpublished attempt can be superseded without overwriting its registered intent."""
    project, prepared, provider, spec = failed_run(tmp_path, monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(
            learning_repair, "publish_trajectory", Mock(side_effect=ValueError("head raced"))
        )
        with pytest.raises(ValueError, match="head raced"):
            repair_learning(project, spec, code_revision=REVISION)
    assert prepared.spec.memory is not None
    lineage = prepared.spec.memory.lineage
    parent = memory_head(project.artifacts, lineage)
    different = {
        **trajectory_record(prepared.spec),
        "task_lineage": "unrelated-source",
        "task": "Another observed task",
    }
    head, consumed = publish_trajectory(
        project.artifacts,
        lineage=lineage,
        parent=parent,
        world_model=prepared.spec.world_model,
        selection_policy_sha256=str(prepared.spec.memory_selection_identity),
        trajectory=different,
        research=(),
        created_at=now(),
        code_revision=REVISION,
        publisher="other-run",
    )
    assert consumed and head.version == 4
    with pytest.raises(ValueError, match="memory advanced"):
        repair_learning(project, spec, code_revision=REVISION)
    current = artifact_input(project.artifacts.read(memory_version_id(lineage, 4)).manifest)
    calls = len(provider.requests)
    result = repair_learning(
        project, spec.model_copy(update={"parent": current}), code_revision=REVISION
    )
    assert result["version"] == 5 and len(provider.requests) == calls
    assert len(project.artifacts.list_ids(artifact_type="fidelity-learning-repair")) == 2


def test_observation_only_repair_never_admits_candidate_text_as_original_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate-only token cannot bypass a hidden source through fabricated research."""
    project, prepared, _, spec = failed_run(tmp_path, monkeypatch)
    report = project.artifacts.read_bytes(prepared.id + "-report", "report.json")
    assert b"first draw missed" in project.artifacts.read_bytes(
        prepared.id + "-report-intent", "evidence.json"
    )
    repair_learning(project, spec, code_revision=REVISION)
    assert prepared.spec.memory is not None
    head = memory_head(project.artifacts, prepared.spec.memory.lineage)
    records, _ = memory_catalog(
        project.artifacts,
        head,
        eligibility=Eligibility(hidden_task_lineage="hidden-prediction-source", strict=True),
    )
    assert "first draw missed" not in json.dumps(records)
    observations = [row for row in records.values() if row.get("kind") == "original-observation"]
    assert len(observations) == 12
    assert project.artifacts.read_bytes(prepared.id + "-report", "report.json") == report
