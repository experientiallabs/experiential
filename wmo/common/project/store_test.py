"""Tests for atomic immutable artifacts, corruption detection, and mutable review state."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

import wmo.common.project.store as project_store_module
from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    SourceIdentity,
    canonical_json_bytes,
)
from wmo.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactCorruptionError,
    ArtifactStoreError,
    ProjectConfig,
    ProjectProviderFreeStage,
    ProjectStore,
    ProjectStoreError,
    ProjectTracePreparationSettings,
    artifact_input,
)


def _envelope() -> ArtifactEnvelope:
    return ArtifactEnvelope(
        schema_version=1,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="e7aad17",
    )


def _store(tmp_path: Path) -> ProjectStore:
    """Return one initialized ordinary Project store."""
    store = ProjectStore(tmp_path / ".wmo", "support-project")
    store.initialize(ProjectConfig(project_id="support-project"))
    return store


def _provider_free_store(tmp_path: Path) -> ProjectStore:
    """Return one initialized trace-first Project store."""
    store = ProjectStore(tmp_path / ".wmo", "support-project")
    store.initialize(
        ProjectConfig(
            project_id="support-project",
            trace_preparation=ProjectTracePreparationSettings(source_kind="chat-json"),
            retrieval=None,
            budgets=None,
        )
    )
    return store


def _write_provider_free_stage(
    store: ProjectStore,
    *,
    suffix: str,
    source_id: str = "platform-source:upload",
    source_sha256: str | None = "a" * 64,
    trace_revision: str = "producer-revision",
    task_revision: str = "producer-revision",
    bind_trace: bool = True,
) -> ProjectProviderFreeStage:
    """Write one candidate trace/task graph and return its exact pointer stage.

    Args:
        store: Project store receiving the immutable fixtures.
        suffix: Safe suffix distinguishing fixture artifact IDs.
        source_id: Manifest-owned durable source label.
        source_sha256: Optional source byte digest.
        trace_revision: Trace manifest producer revision.
        task_revision: Task manifest producer revision.
        bind_trace: Whether the task manifest binds the exact trace input.

    Returns:
        Minimal exact trace and task pointer graph.
    """
    trace_manifest = store.artifacts.write_json(
        artifact_id=f"trace-stage-{suffix}",
        artifact_type="trace-dataset",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision=trace_revision,
            source=SourceIdentity(
                kind="file",
                source_id=source_id,
                sha256=source_sha256,
            ),
        ),
        files={"trace-dataset.json": {"dataset_id": f"trace-stage-{suffix}"}},
    )
    trace_input = artifact_input(trace_manifest)
    task_manifest = store.artifacts.write_json(
        artifact_id=f"task-stage-{suffix}",
        artifact_type="task-set",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            inputs=(trace_input,) if bind_trace else (),
            code_revision=task_revision,
        ),
        files={"task-set.json": {"task_set_id": f"task-stage-{suffix}"}},
    )
    return ProjectProviderFreeStage(
        trace_dataset=trace_input,
        task_set=artifact_input(task_manifest),
    )


def test_provider_free_stage_binding_is_verified_write_once_and_replayable(
    tmp_path: Path,
) -> None:
    """ProjectStore owns exact verification, idempotent replay, and conflicting rejection."""
    store = _provider_free_store(tmp_path)
    selected = _write_provider_free_stage(store, suffix="selected")

    first = store.bind_provider_free_stage(selected)
    replay = store.bind_provider_free_stage(selected)

    assert first.provider_free_stage == selected
    assert replay == first
    assert store.load_project().provider_free_stage == selected
    conflicting = _write_provider_free_stage(store, suffix="conflicting")
    with pytest.raises(ProjectStoreError, match="different provider-free stage"):
        store.bind_provider_free_stage(conflicting)
    assert store.load_project().provider_free_stage == selected


@pytest.mark.parametrize(
    (
        "source_id",
        "source_sha256",
        "trace_revision",
        "task_revision",
        "bind_trace",
        "message",
    ),
    [
        ("tmp/upload.json", "a" * 64, "revision", "revision", True, "worker-local"),
        ("C:upload.json", "a" * 64, "revision", "revision", True, "worker-local"),
        ("platform-source:upload", None, "revision", "revision", True, "byte digest"),
        ("platform-source:upload", "a" * 64, "trace", "task", True, "revision"),
        (
            "platform-source:upload",
            "a" * 64,
            "revision",
            "revision",
            False,
            "does not bind",
        ),
    ],
)
def test_provider_free_stage_binding_derives_and_validates_manifest_provenance(
    source_id: str,
    source_sha256: str | None,
    trace_revision: str,
    task_revision: str,
    bind_trace: bool,
    message: str,
    tmp_path: Path,
) -> None:
    """Source, revision, and task lineage are verified from manifests, not stage duplicates.

    Args:
        source_id: Candidate manifest-owned source label.
        source_sha256: Candidate manifest-owned source digest.
        trace_revision: Trace manifest producer revision.
        task_revision: Task manifest producer revision.
        bind_trace: Whether the task names the exact trace manifest input.
        message: Expected verification failure fragment.
        tmp_path: Isolated Project root.
    """
    store = _provider_free_store(tmp_path)
    stage = _write_provider_free_stage(
        store,
        suffix="invalid",
        source_id=source_id,
        source_sha256=source_sha256,
        trace_revision=trace_revision,
        task_revision=task_revision,
        bind_trace=bind_trace,
    )

    with pytest.raises(ProjectStoreError, match=message):
        store.bind_provider_free_stage(stage)

    assert store.load_project().provider_free_stage is None


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
    with pytest.raises(ArtifactAlreadyExistsError, match="immutable"):
        store.artifacts.write_json(
            artifact_id="task-set-v1",
            artifact_type="task-set",
            envelope=_envelope(),
            files={"tasks.json": {"task_ids": ["task-2"]}},
        )
    assert store.artifacts.read("task-set-v1").manifest == manifest


def test_write_or_replay_adopts_exact_replays_and_rejects_any_drift(tmp_path: Path) -> None:
    """Idempotent writes adopt the original materialization time and reject drifted evidence."""
    store = _store(tmp_path)
    envelope = _envelope()
    files = {
        "envelope.json": canonical_json_bytes(envelope),
        "records.jsonl": b'{"row":1}\n',
    }
    written, first_manifest = store.artifacts.write_or_replay(
        artifact_id="replay-v1",
        artifact_type="task-set",
        envelope=envelope,
        envelope_path="envelope.json",
        envelope_type=ArtifactEnvelope,
        files=files,
    )
    assert written == envelope
    later = envelope.model_copy(update={"created_at": datetime(2026, 8, 12, tzinfo=UTC)})
    replayed, manifest = store.artifacts.write_or_replay(
        artifact_id="replay-v1",
        artifact_type="task-set",
        envelope=later,
        envelope_path="envelope.json",
        envelope_type=ArtifactEnvelope,
        files={**files, "envelope.json": canonical_json_bytes(later)},
    )
    assert replayed == envelope
    assert manifest == first_manifest
    drifted_cases = {
        "payload differs from exact replay": {**files, "records.jsonl": b'{"row":2}\n'},
        "file set differs from exact replay": {"envelope.json": files["envelope.json"]},
    }
    for message, drifted_files in drifted_cases.items():
        with pytest.raises(ValueError, match=message):
            store.artifacts.write_or_replay(
                artifact_id="replay-v1",
                artifact_type="task-set",
                envelope=later,
                envelope_path="envelope.json",
                envelope_type=ArtifactEnvelope,
                files=drifted_files,
            )
    with pytest.raises(ValueError, match="envelope differs from exact replay"):
        store.artifacts.write_or_replay(
            artifact_id="replay-v1",
            artifact_type="task-set",
            envelope=later.model_copy(update={"code_revision": "forged"}),
            envelope_path="envelope.json",
            envelope_type=ArtifactEnvelope,
            files=files,
        )


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
    (store.paths.artifact_directory("task-set-v1") / "tasks.json").write_text(
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
    target = store.paths.artifact_directory("task-set-v1") / "tasks.json"
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
    target = store.paths.artifact_directory("task-set-v1") / "tasks.json"
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
    manifest = store.artifacts.write_json(
        artifact_id="sft-model-optimization-config-a",
        artifact_type="sft-model-optimization-config",
        envelope=_envelope(),
        files={"config.json": {"config_id": "sft-model-optimization-config-a"}},
    )
    config_input = artifact_input(manifest)
    assert store.bind_model_optimization_config(
        config_input, artifact_type="sft-model-optimization-config"
    ) == (
        ProjectConfig(
            project_id="support-project",
            model_optimization_config=config_input,
        )
    )
    assert store.bind_model_optimization_config(
        config_input, artifact_type="sft-model-optimization-config"
    ).project_id == ("support-project")
    different_manifest = store.artifacts.write_json(
        artifact_id="sft-model-optimization-config-b",
        artifact_type="sft-model-optimization-config",
        envelope=_envelope(),
        files={"config.json": {"config_id": "sft-model-optimization-config-b"}},
    )
    with pytest.raises(ProjectStoreError, match="different immutable model optimization config"):
        store.bind_model_optimization_config(
            artifact_input(different_manifest), artifact_type="sft-model-optimization-config"
        )
