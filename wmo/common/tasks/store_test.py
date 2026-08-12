"""Tests for verified immutable task-set reads."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from wmo.common.core.artifacts import SourceIdentity
from wmo.common.project import ArtifactCorruptionError, ArtifactStore, ProjectPaths
from wmo.common.tasks import TaskCase, TaskSet, ToolSchema
from wmo.common.tasks.store import load_task_set, resolve_task_set
from wmo.common.traces import Trace, TraceSource, TraceSpan
from wmo.simulation.mining.descriptors import HashingDescriptorEmbedder
from wmo.simulation.mining.service import MiningSpec, mine_tasks, persist_task_set


def _trace(index: int) -> Trace:
    """Return one deterministic canonical trace for a task-set artifact fixture."""
    source = TraceSource(
        identity=SourceIdentity(kind="production", source_id="fixture", sha256="0" * 64),
        semantic_convention_version="1.37.0",
    )
    return Trace(
        trace_id=f"trace-{index}",
        conversation_id=f"conversation-{index}",
        task=f"Handle support request {index}",
        initial_context={"channel": "email"},
        tools=(
            ToolSchema(
                name="lookup_order",
                description="Look up one customer order.",
                input_schema={"type": "object"},
            ),
        ),
        spans=(
            TraceSpan(
                span_id=f"span-{index}",
                name="agent.model_call",
                started_at=datetime(2026, 8, 11, 0, index, tzinfo=UTC),
                ended_at=datetime(2026, 8, 11, 0, index, 1, tzinfo=UTC),
                attributes={"gen_ai.tool.name": "lookup_order"},
            ),
        ),
        source=source,
    )


def _task_set_store(
    tmp_path: Path,
    task_set_id: str = "task-set-fixture",
) -> tuple[ArtifactStore, TaskSet]:
    """Create one persisted representative task set for loader checks."""
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    result = mine_tasks(
        (_trace(1), _trace(2)),
        MiningSpec(fit_task_budget=1, held_out_task_budget=1),
        embedder=HashingDescriptorEmbedder(),
    )
    task_set = persist_task_set(
        result,
        store,
        task_set_id=task_set_id,
        created_at=datetime.now(UTC),
        code_revision="test",
        source=SourceIdentity(kind="production", source_id="fixture", sha256="0" * 64),
    )
    return store, task_set


def test_load_task_set_returns_the_artifact_envelope_and_exact_ordered_tasks(tmp_path) -> None:  # noqa: ANN001
    """The task-set read boundary is typed, digest-verified, and source-read free."""
    store, task_set = _task_set_store(tmp_path)

    loaded = load_task_set(store, task_set.task_set_id)

    assert loaded.task_set == task_set
    assert tuple(task.task_id for task in loaded.tasks) == task_set.task_ids
    assert all(isinstance(task, TaskCase) for task in loaded.tasks)


def test_load_task_set_rejects_a_typed_envelope_with_wrong_task_ids(tmp_path) -> None:  # noqa: ANN001
    """A manifest cannot silently point to task records that disagree with its envelope."""
    store, task_set = _task_set_store(tmp_path)
    artifact = store.read(task_set.task_set_id)
    task_set_path = artifact.directory / "task-set.json"
    payload = task_set.model_copy(update={"task_ids": tuple(reversed(task_set.task_ids))})
    task_set_path.write_text(payload.model_dump_json(), encoding="utf-8")

    with pytest.raises(ArtifactCorruptionError, match="data file digest mismatch"):
        load_task_set(store, task_set.task_set_id)


def test_resolve_task_set_uses_the_only_immutable_task_set(tmp_path) -> None:  # noqa: ANN001
    """A caller cannot accidentally pick an unspecified task set from a project."""
    store, task_set = _task_set_store(tmp_path)

    assert resolve_task_set(store).task_set == task_set


def test_resolve_task_set_rejects_an_absent_or_ambiguous_default(tmp_path) -> None:  # noqa: ANN001
    """Unnamed consumers get deterministic directions instead of an arbitrary artifact."""
    empty = ArtifactStore(ProjectPaths(root=tmp_path / "empty", project_id="support"))

    with pytest.raises(ArtifactCorruptionError, match="no immutable task set"):
        resolve_task_set(empty)

    store, _ = _task_set_store(tmp_path / "multiple")
    _task_set_store(tmp_path / "multiple", task_set_id="task-set-second")

    with pytest.raises(
        ArtifactCorruptionError,
        match="multiple task sets; pass --task-set with one of: task-set-fixture, task-set-second",
    ):
        resolve_task_set(store)
