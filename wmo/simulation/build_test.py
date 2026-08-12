"""Tests for the direct immutable trace-dataset to task-set composition path."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from wmo.common.core.artifacts import SourceIdentity
from wmo.common.project import ArtifactStore, ProjectConfig, ProjectStore, artifact_input
from wmo.common.project.paths import ProjectPaths
from wmo.common.traces import Trace, TraceSource, TraceSpan
from wmo.simulation.build import build_project, build_task_set
from wmo.simulation.ingest.otlp import TraceNormalizationIssue, TraceNormalizationResult
from wmo.simulation.mining.service import MiningSpec


def _trace(index: int) -> Trace:
    """Build one distinct canonical source trace for deterministic representative selection."""
    started_at = datetime(2026, 8, 11, tzinfo=UTC) + timedelta(minutes=index)
    return Trace(
        trace_id=f"trace-{index}",
        conversation_id=f"conversation-{index}",
        task=f"Resolve a distinct support case {index}",
        spans=(
            TraceSpan(
                span_id=f"span-{index}",
                name="agent.model_call",
                started_at=started_at,
                ended_at=started_at + timedelta(seconds=1),
            ),
        ),
        source=TraceSource(
            identity=SourceIdentity(
                kind="otlp",
                source_id="fixture.otlp",
                sha256="a" * 64,
            ),
            semantic_convention_version="1.37.0",
        ),
    )


def test_build_task_set_uses_only_the_persisted_trace_dataset_as_task_set_input(
    tmp_path: Path,
) -> None:
    """The task-set manifest has exactly one immutable trace-dataset dependency."""
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    normalized = TraceNormalizationResult(
        traces=(_trace(2), _trace(1)),
        issues=(TraceNormalizationIssue("line-7", "invalid record"),),
    )

    built = build_task_set(
        normalized,
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
        mining_spec=MiningSpec(fit_task_budget=1, held_out_task_budget=1),
    )

    dataset_input = artifact_input(built.trace_dataset.manifest)
    assert built.task_set.inputs == (dataset_input,)
    assert built.task_set.source is None
    assert built.trace_dataset.traces == (_trace(1), _trace(2))
    assert built.task_set.task_ids
    assert store.read(built.task_set.task_set_id).manifest.inputs == (dataset_input,)


def test_build_project_resumes_and_records_provider_free_review_readiness(tmp_path: Path) -> None:
    """A repeated local build reuses immutable IDs and never invents rubric proposals."""
    store = ProjectStore(tmp_path, "project-a")
    store.initialize(ProjectConfig(project_id="project-a"))
    normalized = TraceNormalizationResult(traces=(_trace(1), _trace(2)), issues=())
    first = build_project(
        normalized,
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
        mining_spec=MiningSpec(fit_task_budget=1, held_out_task_budget=1),
    )
    replay = build_project(
        normalized,
        store,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="test-revision",
        mining_spec=MiningSpec(fit_task_budget=1, held_out_task_budget=1),
    )

    assert replay == first
    assert replay.review.status == "proposals_pending"
    assert replay.review.paid_calls_made == 0
    assert store.read_review() == {"build_review": replay.review.model_dump(mode="json")}
    assert len(store.artifacts.list_ids()) == 2
