"""Summary forgetting: the decision, its pinned readers and the open trajectory-count study."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope
from exp.common.project import ProjectStore
from exp.optimize.evaluation.learning.forgetting import SummaryForget, forget_memory_summary
from exp.optimize.evaluation.learning.sample_count import (
    REGISTRATION_TYPE,
    SampleCountRegistration,
)
from exp.optimize.evaluation.tests.memory_forget_test import LINEAGE, NOW, Compressor, Memory


def registered(project: ProjectStore, grid: tuple[int, ...]) -> None:
    """Save a trajectory-count registration on the fixture memory."""
    registration = SampleCountRegistration(
        project=project.paths.project_id,
        memory_lineage=LINEAGE,
        trace_order_sha256="a" * 64,
        grid=grid,
        threshold=0.5,
        criterion="point_estimate",
        development_lineages=("dev",),
        registered_at=NOW,
        receipts_before=0,
    )
    project.artifacts.write_json(
        artifact_id=registration.identity,
        artifact_type=REGISTRATION_TYPE,
        envelope=ArtifactEnvelope(schema_version=1, created_at=NOW, code_revision="x"),
        files={"registration.json": registration.model_dump(mode="json")},
    )


def test_forgetting_reports_ancestors_and_waits_for_an_open_study(tmp_path: Path) -> None:
    """The decision names every raised block; a study still learning this memory refuses."""
    memory = Memory(tmp_path)
    memory.learn("alpha", "[unresolved] alpha claim")
    memory.learn("beta", "[unresolved] beta BAD-SUMMARY")
    memory.maintain(Compressor())
    request = SummaryForget(lineage=LINEAGE, block="#2-3", reason="bad summary")
    assert request.bounds == (2, 4)
    registered(memory.project, (1, 2))  # a study whose checkpoints are all published
    dry = forget_memory_summary(
        memory.project, request, created_at=NOW, code_revision="x", dry_run=True
    )
    assert dry.record is None and dry.forgets == ("#2-3", "#0-3")
    registered(memory.project, (2, 8))
    with pytest.raises(ValueError, match="up to 8 trajectories"):
        forget_memory_summary(memory.project, request, created_at=NOW, code_revision="x")
    assert not memory.store.list_ids(artifact_type="memory-forget")


def test_a_forget_applies_from_the_next_version(tmp_path: Path) -> None:
    """The head stays as published; the record is folded into the next version."""
    memory = Memory(tmp_path)
    memory.learn("alpha", "[unresolved] alpha claim")
    memory.learn("beta", "[unresolved] beta BAD-SUMMARY")
    memory.maintain(Compressor())
    request = SummaryForget(lineage=LINEAGE, block="#2-3", reason="bad summary")
    decision = forget_memory_summary(memory.project, request, created_at=NOW, code_revision="x")
    assert (decision.head_version, decision.takes_effect_at) == (2, 3)
    assert decision.forgets == ("#2-3", "#0-3") and decision.pinned_audits == ()
    assert decision.record is not None and memory.store.exists(decision.record)
    assert forget_memory_summary(memory.project, request, created_at=NOW, code_revision="x") == (
        decision
    )
    assert memory.learn("gamma", "[unresolved] gamma claim").forgotten == ("2-4", "0-4")
