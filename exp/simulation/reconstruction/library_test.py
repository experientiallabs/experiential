"""Growing libraries retain original bytes and do not activate unaccepted hypotheses."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_bytes
from exp.common.project import ProjectConfig, ProjectStore, artifact_input
from exp.simulation.reconstruction.library import (
    LibraryEntry,
    extend_library,
    remember_trajectories,
    retrieve_library,
)
from exp.simulation.world_model.runtime_test import _trace


def test_library_grows_without_changing_pinned_parent_or_promoting_unqualified_rules(
    tmp_path: Path,
) -> None:
    """A new environment queries actual originals, while rejected rules stay research."""
    now = datetime(2026, 10, 14, tzinfo=UTC)
    store = ProjectStore(tmp_path, "library")
    store.initialize(ProjectConfig(project_id="library"))
    payload = json.dumps({"original": {"request": "read", "reply": "observed"}}).encode()
    manifest = store.artifacts.write(
        artifact_id="source",
        artifact_type="native-observation",
        envelope=ArtifactEnvelope(schema_version=1, created_at=now, code_revision="test"),
        files={"original.json": payload},
    )
    original = LibraryEntry(
        entry_id="trace",
        kind="trajectory",
        artifact=artifact_input(manifest),
        path="original.json",
        payload_sha256=sha256_bytes(payload),
        disposition="source",
    )
    parent = extend_library(store.artifacts, (original,), created_at=now, code_revision="test")
    rule = original.model_copy(
        update={"entry_id": "rule", "kind": "rule", "disposition": "unresolved"}
    )
    child = extend_library(
        store.artifacts, (rule,), parent=parent, created_at=now, code_revision="test"
    )
    assert len(parent.entries) == 1 and len(child.entries) == 2

    def select(query: JsonObject, entries: tuple[LibraryEntry, ...]) -> tuple[str, ...]:
        """Inspect the exact raw-target query and retain eligible source IDs."""
        assert "target_traces" in query and "environment_name" not in query
        assert [entry.entry_id for entry in entries] == ["trace"]
        return ("trace",)

    context = retrieve_library(store.artifacts, child, (_trace(now),), select=select, top_k=1)
    assert context["library_id"] == child.library_id
    selected = context["selected"]
    assert isinstance(selected, list) and isinstance(selected[0], dict)
    assert selected[0]["payload"] == json.loads(payload)
    with pytest.raises(ValueError, match="immutable"):
        extend_library(
            store.artifacts, (original,), parent=child, created_at=now, code_revision="test"
        )


def test_rejected_skill_cannot_supersede_an_active_one() -> None:
    """Failed library edits remain records, never replacement behavior."""
    with pytest.raises(ValueError, match="cannot supersede"):
        LibraryEntry.model_validate(
            {
                "entry_id": "bad",
                "kind": "skill",
                "artifact": {"artifact_id": "x", "sha256": "a" * 64},
                "path": "rule.json",
                "payload_sha256": "b" * 64,
                "disposition": "rejected",
                "supersedes": "old",
            }
        )


def test_source_readmission_returns_persisted_snapshot_timestamp(tmp_path: Path) -> None:
    """Reopening the same original inventory can safely seed another library child."""
    now = datetime(2026, 10, 14, tzinfo=UTC)
    project = ProjectStore(tmp_path, "replay")
    project.initialize(ProjectConfig(project_id="replay"))
    trace = _trace(now)
    first = remember_trajectories(
        project.artifacts, (trace,), parent=None, created_at=now, code_revision="test"
    )
    reopened = remember_trajectories(
        project.artifacts,
        (trace,),
        parent=None,
        created_at=now + timedelta(seconds=1),
        code_revision="test",
    )
    assert reopened == first
    research = first.entries[0].model_copy(
        update={"entry_id": "investigation", "kind": "research", "disposition": "rejected"}
    )
    child = extend_library(
        project.artifacts,
        (research,),
        parent=reopened,
        created_at=now + timedelta(seconds=2),
        code_revision="test",
    )
    assert len(child.entries) == 2


def test_source_snapshot_versions_publication_code_without_rewriting_originals(
    tmp_path: Path,
) -> None:
    """An upgraded publisher preserves source provenance without a manifest collision."""
    now = datetime(2026, 10, 14, tzinfo=UTC)
    project = ProjectStore(tmp_path, "upgrade")
    project.initialize(ProjectConfig(project_id="upgrade"))
    trace = _trace(now)
    old = remember_trajectories(
        project.artifacts, (trace,), parent=None, created_at=now, code_revision="old"
    )
    new = remember_trajectories(
        project.artifacts,
        (trace,),
        parent=None,
        created_at=now + timedelta(seconds=1),
        code_revision="new",
    )
    assert new.library_id != old.library_id
    assert new.entries == old.entries
    assert json.loads(
        project.artifacts.read_bytes(old.library_id, "library.json")
    ) == old.model_dump(mode="json")
