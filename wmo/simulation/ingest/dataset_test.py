"""Tests for immutable canonical trace-dataset persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wmo.common.core.artifacts import SourceIdentity
from wmo.common.project import ArtifactAlreadyExistsError, ArtifactStore, artifact_input
from wmo.common.project.paths import ProjectPaths
from wmo.common.traces import Trace, TraceDataset, TraceSource, TraceSpan
from wmo.simulation.ingest.dataset import persist_trace_dataset
from wmo.simulation.ingest.otlp import TraceNormalizationIssue, TraceNormalizationResult

_SOURCE_DIGEST = "a" * 64


def _store(tmp_path: Path, project_id: str) -> ArtifactStore:
    """Return an isolated project-local artifact store for one persistence test."""
    return ArtifactStore(ProjectPaths(root=tmp_path, project_id=project_id))


def _trace(index: int, *, source_id: str = "fixture.otlp") -> Trace:
    """Build one normalized trace with fixed provenance and deterministic ordering."""
    started_at = datetime(2026, 8, 11, tzinfo=UTC) + timedelta(minutes=index)
    return Trace(
        trace_id=f"trace-{index}",
        task=f"Resolve support case {index}",
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
                source_id=source_id,
                sha256=_SOURCE_DIGEST,
            ),
            semantic_convention_version="1.37.0",
        ),
    )


def test_persist_trace_dataset_writes_trace_and_issue_evidence(tmp_path: Path) -> None:
    """One normalized result becomes a digest-addressed trace artifact with its exclusions."""
    trace = _trace(1)
    result = TraceNormalizationResult(
        traces=(trace,),
        issues=(TraceNormalizationIssue("line-4", "invalid JSONL record"),),
    )
    store = _store(tmp_path, "project-a")

    persisted = persist_trace_dataset(
        result,
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )

    dataset = persisted.dataset
    assert dataset.source == trace.source.identity
    assert dataset.semantic_convention_version == "1.37.0"
    assert dataset.trace_ids == (trace.trace_id,)
    assert dataset.invalid_trace_count == 1
    assert persisted.manifest == store.read(dataset.dataset_id).manifest
    assert artifact_input(persisted.manifest).artifact_id == dataset.dataset_id
    assert (
        TraceDataset.model_validate_json(store.read_bytes(dataset.dataset_id, "trace-dataset.json"))
        == dataset
    )
    assert (
        Trace.model_validate_json(
            store.read_bytes(dataset.dataset_id, dataset.traces_path).decode("utf-8")
        )
        == trace
    )
    assert json.loads(store.read_bytes(dataset.dataset_id, "normalization-issues.json")) == {
        "invalid_trace_count": 1,
        "issues": [{"message": "invalid JSONL record", "source_record": "line-4"}],
    }


def test_persist_trace_dataset_is_content_addressed_despite_input_order(tmp_path: Path) -> None:
    """Canonical ordering makes equivalent normalized results yield one deterministic identity."""
    first = _trace(1)
    second = _trace(2)
    result = TraceNormalizationResult(traces=(second, first), issues=())
    reversed_result = TraceNormalizationResult(traces=(first, second), issues=())

    first_persisted = persist_trace_dataset(
        result,
        _store(tmp_path / "first", "project-a"),
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )
    second_persisted = persist_trace_dataset(
        reversed_result,
        _store(tmp_path / "second", "project-a"),
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )

    assert first_persisted.dataset == second_persisted.dataset
    assert first_persisted.manifest == second_persisted.manifest


def test_persist_trace_dataset_refuses_to_overwrite_completed_evidence(tmp_path: Path) -> None:
    """A completed trace dataset cannot be rewritten under its stable content identity."""
    result = TraceNormalizationResult(traces=(_trace(1),), issues=())
    store = _store(tmp_path, "project-a")
    first = persist_trace_dataset(
        result,
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )

    with pytest.raises(ArtifactAlreadyExistsError, match="immutable"):
        persist_trace_dataset(
            result,
            store,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="test-revision",
        )

    assert store.read(first.dataset.dataset_id).manifest == first.manifest


def test_persist_trace_dataset_rejects_mixed_raw_source_provenance(tmp_path: Path) -> None:
    """A trace artifact cannot hide a second raw source behind one immutable manifest."""
    result = TraceNormalizationResult(
        traces=(_trace(1), _trace(2, source_id="other.otlp")),
        issues=(),
    )
    store = _store(tmp_path, "project-a")

    with pytest.raises(ValueError, match="exactly one trace source"):
        persist_trace_dataset(
            result,
            store,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="test-revision",
        )

    paths = ProjectPaths(root=tmp_path, project_id="project-a")
    assert not paths.artifacts_directory.exists()
