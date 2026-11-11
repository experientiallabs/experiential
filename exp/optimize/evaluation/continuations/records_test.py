"""Regression checks for complete immutable continuation inventories."""

from collections import Counter
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import AssistantAction
from exp.common.project import ArtifactStore, ProjectStore
from exp.optimize.evaluation.continuations import records
from exp.optimize.evaluation.continuations.admission import ContinuationAdmission
from exp.optimize.evaluation.continuations.admission_test import admitted_workbook_study
from exp.optimize.evaluation.continuations.study import ContinuationStudy
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.simulation.continuations import ContinuationCheckpoint


def _saved_study(tmp_path: Path) -> tuple[ProjectStore, ArtifactInput, ContinuationStudy]:
    """Build two native saved branches before any loader instrumentation is installed."""
    project = ProjectStore(tmp_path, "case")
    call = workbook_case().first
    actions = tuple(
        action
        for _ in range(2)
        for action in (
            AssistantAction(tool_calls=(call,)),
            AssistantAction(content="done"),
        )
    )
    admission_ref = admitted_workbook_study(project, actor_actions=actions)
    admission = ContinuationAdmission.model_validate_json(
        project.artifacts.read_bytes(admission_ref.artifact_id, "admission.json")
    )
    reference = admission.study
    study = records.load_study(project.artifacts, reference)
    assert len(study.branches) == 2 and all(branch.checkpoints for branch in study.branches)
    return project, reference, study


def test_a_shared_parent_is_loaded_once_while_every_branch_checkpoint_is_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading a complete study reuses its parent without losing any branch's checks."""
    project, reference, saved = _saved_study(tmp_path)
    load_checkpoint = records.load_continuation_checkpoint
    loaded: list[ArtifactInput] = []

    def observe(store: ArtifactStore, point: ArtifactInput) -> ContinuationCheckpoint:
        """Count actual checkpoint reconstructions while retaining the real artifact loader."""
        loaded.append(point)
        return load_checkpoint(store, point)

    monkeypatch.setattr(records, "load_continuation_checkpoint", observe)

    assert records.load_study(project.artifacts, reference) == saved

    expected = Counter([saved.spec.checkpoint])
    expected.update(point for branch in saved.branches for point in branch.checkpoints)
    assert Counter(loaded) == expected


@pytest.mark.parametrize("branch_index", [0, 1])
def test_a_loaded_branch_with_a_corrupted_actor_prefix_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, branch_index: int
) -> None:
    """Every branch must preserve the common actor prefix, including a later branch."""
    project, reference, saved = _saved_study(tmp_path)
    load_checkpoint = records.load_continuation_checkpoint
    corrupted = saved.branches[branch_index].checkpoints[0]

    def corrupt_prefix(store: ArtifactStore, point: ArtifactInput) -> ContinuationCheckpoint:
        """Inject one inconsistent loaded point after verifying its real stored artifact."""
        checkpoint = load_checkpoint(store, point)
        if point != corrupted:
            return checkpoint
        first, *rest = checkpoint.actor.messages
        messages = (first.model_copy(update={"content": "a different inherited prefix"}), *rest)
        return checkpoint.model_copy(
            update={"actor": checkpoint.actor.model_copy(update={"messages": messages})}
        )

    monkeypatch.setattr(records, "load_continuation_checkpoint", corrupt_prefix)

    with pytest.raises(ValueError, match="exact prefix or branch provenance"):
        records.load_study(project.artifacts, reference)
