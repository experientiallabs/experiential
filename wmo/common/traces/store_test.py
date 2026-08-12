"""Tests for verified immutable trace-dataset reads."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from wmo.common.core.artifacts import SourceIdentity
from wmo.common.project import ArtifactCorruptionError, ArtifactStore, ProjectPaths
from wmo.common.traces import Trace, TraceSource, TraceSpan
from wmo.common.traces.store import load_trace_dataset
from wmo.simulation.ingest.dataset import persist_trace_dataset
from wmo.simulation.ingest.otlp import TraceNormalizationResult


def _store(tmp_path: Path) -> ArtifactStore:
    """Create one isolated artifact store for trace-dataset read coverage."""
    return ArtifactStore(ProjectPaths(root=tmp_path, project_id="trace-load"))


def _trace(trace_id: str) -> Trace:
    """Create one exact canonical trace fixture."""
    return Trace(
        trace_id=trace_id,
        task="Resolve support request.",
        spans=(
            TraceSpan(
                span_id="span-1",
                name="agent.model_call",
                started_at=datetime(2026, 8, 12, tzinfo=UTC),
                ended_at=datetime(2026, 8, 12, 0, 0, 1, tzinfo=UTC),
            ),
        ),
        source=TraceSource(
            identity=SourceIdentity(kind="otlp", source_id="fixture", sha256="a" * 64),
            semantic_convention_version="1.37.0",
        ),
    )


def test_load_trace_dataset_returns_verified_envelope_and_records(tmp_path: Path) -> None:
    """The loader returns only records from a digest-verified canonical payload."""
    store = _store(tmp_path)
    persisted = persist_trace_dataset(
        TraceNormalizationResult(traces=(_trace("trace-1"),), issues=()),
        store,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="test",
    )

    loaded = load_trace_dataset(store, persisted.dataset.dataset_id)

    assert loaded.dataset == persisted.dataset
    assert loaded.traces == persisted.traces


def test_load_trace_dataset_rejects_payload_corruption(tmp_path: Path) -> None:
    """A changed source record cannot be hydrated into an SFT build."""
    store = _store(tmp_path)
    persisted = persist_trace_dataset(
        TraceNormalizationResult(traces=(_trace("trace-1"),), issues=()),
        store,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="test",
    )
    path = store.read(persisted.dataset.dataset_id).directory / persisted.dataset.traces_path
    path.write_text('{"trace_id":"forged"}\n', encoding="utf-8")

    with pytest.raises(ArtifactCorruptionError, match="data file digest mismatch"):
        load_trace_dataset(store, persisted.dataset.dataset_id)
