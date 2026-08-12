"""Tests for atomic immutable artifacts, corruption detection, and mutable review state."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

import wmo.common.project.store as project_store_module
from wmo.common.core.artifacts import ArtifactEnvelope
from wmo.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactCorruptionError,
    ArtifactStoreError,
    ProjectConfig,
    ProjectStore,
    ProjectStoreError,
    artifact_input,
)


def _envelope() -> ArtifactEnvelope:
    return ArtifactEnvelope(
        schema_version=1,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="e7aad17",
    )


def _store(tmp_path: Path) -> ProjectStore:
    store = ProjectStore(tmp_path / ".wmo", "support-project")
    store.initialize(ProjectConfig(project_id="support-project"))
    return store


def test_artifact_round_trip_is_digest_verified_and_immutable(tmp_path: Path) -> None:
    """Completed JSON artifacts round trip, reference by manifest digest, and never overwrite."""
    store = _store(tmp_path)
    manifest = store.artifacts.write_json(
        artifact_id="task-set-v1",
        artifact_type="task-set",
        envelope=_envelope(),
        files={"tasks.json": {"task_ids": ["task-1"]}},
    )

    stored = store.artifacts.read("task-set-v1")

    assert stored.manifest == manifest
    assert artifact_input(manifest).artifact_id == "task-set-v1"
    assert store.artifacts.read_bytes("task-set-v1", "tasks.json") == b'{"task_ids":["task-1"]}'
    jsonl_manifest = store.artifacts.write_jsonl(
        artifact_id="traces-v1",
        artifact_type="trace-set",
        envelope=_envelope(),
        files={"traces.jsonl": ({"trace_id": "trace-1"}, {"trace_id": "trace-2"})},
    )
    assert jsonl_manifest.files[0].path == "traces.jsonl"
    assert store.artifacts.read_bytes("traces-v1", "traces.jsonl") == (
        b'{"trace_id":"trace-1"}\n{"trace_id":"trace-2"}\n'
    )
    with pytest.raises(ArtifactAlreadyExistsError, match="immutable"):
        store.artifacts.write_json(
            artifact_id="task-set-v1",
            artifact_type="task-set",
            envelope=_envelope(),
            files={"tasks.json": {"task_ids": ["task-2"]}},
        )
    assert store.artifacts.read("task-set-v1").manifest == manifest


def test_corruption_and_crash_do_not_create_valid_partial_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Digest mutation fails reads, and a failed rename leaves no valid partial artifact."""
    store = _store(tmp_path)
    store.artifacts.write_json(
        artifact_id="task-set-v1",
        artifact_type="task-set",
        envelope=_envelope(),
        files={"tasks.json": {"task_ids": ["task-1"]}},
    )
    store.paths.artifact_file("task-set-v1", "tasks.json").write_text(
        '{"task_ids":["task-2"]}', encoding="utf-8"
    )
    with pytest.raises(ArtifactCorruptionError, match="digest mismatch"):
        store.artifacts.read("task-set-v1")

    def fail_rename(source: Path, destination: Path) -> None:
        raise OSError("simulated crash before promotion")

    monkeypatch.setattr(project_store_module.os, "rename", fail_rename)
    with pytest.raises(OSError, match="simulated crash"):
        store.artifacts.write_json(
            artifact_id="task-set-v2",
            artifact_type="task-set",
            envelope=_envelope(),
            files={"tasks.json": {"task_ids": ["task-2"]}},
        )
    assert not store.paths.artifact_directory("task-set-v2").exists()
    assert not tuple(store.paths.artifacts_directory.glob(".task-set-v2.*.partial"))


def test_read_bytes_rechecks_the_exact_file_snapshot_after_full_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement after `read` cannot make `read_bytes` return unchecked data."""
    store = _store(tmp_path)
    store.artifacts.write_json(
        artifact_id="task-set-v1",
        artifact_type="task-set",
        envelope=_envelope(),
        files={"tasks.json": {"task_ids": ["task-1"]}},
    )
    target = store.paths.artifact_file("task-set-v1", "tasks.json")
    verified_read = store.artifacts.read

    def replace_after_verification(artifact_id: str) -> project_store_module.StoredArtifact:
        stored = verified_read(artifact_id)
        target.write_bytes(b'{"task_ids":["task-replaced"]}')
        return stored

    monkeypatch.setattr(store.artifacts, "read", replace_after_verification)

    with pytest.raises(ArtifactCorruptionError, match="digest mismatch"):
        store.artifacts.read_bytes("task-set-v1", "tasks.json")


def test_read_bytes_rejects_a_symlink_swapped_after_full_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement symlink cannot escape an artifact after `read` verified it."""
    store = _store(tmp_path)
    store.artifacts.write_json(
        artifact_id="task-set-v1",
        artifact_type="task-set",
        envelope=_envelope(),
        files={"tasks.json": {"task_ids": ["task-1"]}},
    )
    target = store.paths.artifact_file("task-set-v1", "tasks.json")
    outside = tmp_path / "outside.json"
    outside.write_bytes(b'{"task_ids":["outside"]}')
    verified_read = store.artifacts.read

    def replace_after_verification(artifact_id: str) -> project_store_module.StoredArtifact:
        stored = verified_read(artifact_id)
        target.unlink()
        target.symlink_to(outside)
        return stored

    monkeypatch.setattr(store.artifacts, "read", replace_after_verification)

    with pytest.raises(ArtifactCorruptionError, match="unsafe or unreadable"):
        store.artifacts.read_bytes("task-set-v1", "tasks.json")


def test_secret_boundary_review_draft_and_write_once_model_config_binding(tmp_path: Path) -> None:
    """Artifacts reject credentials; review is mutable and the SFT config pointer is write-once."""
    store = _store(tmp_path)
    with pytest.raises(ArtifactStoreError, match="api_key_env"):
        store.artifacts.write_json(
            artifact_id="unsafe-v1",
            artifact_type="task-set",
            envelope=_envelope(),
            files={"tasks.json": {"api_key_env": "OPENAI_API_KEY"}},
        )
    with pytest.raises(ArtifactStoreError, match="credential environment name"):
        store.artifacts.write_json(
            artifact_id="unsafe-variable-v1",
            artifact_type="task-set",
            envelope=_envelope(),
            files={"tasks.json": {"connection_hint": "OPENAI_API_KEY"}},
        )
    with pytest.raises(ArtifactStoreError, match="relative POSIX"):
        store.artifacts.write_json(
            artifact_id="unsafe-path-v1",
            artifact_type="task-set",
            envelope=_envelope(),
            files={"../tasks.json": {"task_ids": ["task-1"]}},
        )
    with pytest.raises(ArtifactStoreError, match="secret boundary"):
        store.artifacts.write(
            artifact_id="unsafe-text-v1",
            artifact_type="task-set",
            envelope=_envelope(),
            files={"notes.md": b"Use api-key environment configuration."},
        )

    store.write_review({"rubric_status": "draft-one"})
    store.write_review({"rubric_status": "draft-two"})

    assert store.read_review() == {"rubric_status": "draft-two"}
    with pytest.raises(ProjectStoreError, match="different immutable"):
        store.initialize(
            ProjectConfig(project_id="support-project", redacted_field_names=("email",))
        )
    assert store.set_model_optimization_config_id("sft-model-optimization-config-a") == (
        ProjectConfig(
            project_id="support-project",
            model_optimization_config_id="sft-model-optimization-config-a",
        )
    )
    assert store.set_model_optimization_config_id("sft-model-optimization-config-a").project_id == (
        "support-project"
    )
    with pytest.raises(ProjectStoreError, match="different immutable model optimization config"):
        store.set_model_optimization_config_id("sft-model-optimization-config-b")
