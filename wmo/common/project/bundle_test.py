"""Tests for deterministic, verified, secret-free Project bundles."""

from __future__ import annotations

import os
import stat
import warnings
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

import wmo.common.project.bundle as bundle_module
from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    SourceIdentity,
    canonical_json_bytes,
    sha256_bytes,
)
from wmo.common.models import ModelCapabilities, ModelSnapshot
from wmo.common.project import (
    ProjectConfig,
    ProjectModelConfiguration,
    ProjectProviderFreeStage,
    ProjectStore,
    ProjectTracePreparationSettings,
    artifact_input,
    export_project_bundle,
    restore_project_bundle,
)
from wmo.common.project.bundle import ProjectBundleError
from wmo.common.project.catalog import (
    ProjectCatalogModel,
    ProjectModelCatalog,
    persist_project_model_catalog,
)
from wmo.common.project.events import ProjectStage
from wmo.common.project.manifests import ArtifactManifest, file_digest
from wmo.common.project.project import write_project_config

_CREATED_AT = datetime(2026, 8, 17, tzinfo=UTC)
_REVISION = "producer-revision"


def _catalog_model(alias: str) -> ProjectCatalogModel:
    """Build one portable model snapshot for a catalog fixture."""
    capabilities = ModelCapabilities(
        supports_completions=True,
        context_window_tokens=8_192,
        maximum_output_tokens=1_024,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=2.0,
    )
    return ProjectCatalogModel(
        alias=alias,
        model=ModelSnapshot(
            provider="openai",
            model_id=f"model-{alias}",
            capabilities_sha256=capabilities.identity_sha256(),
            connection_sha256="c" * 64,
        ),
        capabilities=capabilities,
    )


def _project_store(tmp_path: Path, *, with_catalog: bool = False) -> ProjectStore:
    """Create a selected provider-free graph plus one unrelated sibling artifact."""
    store = ProjectStore(tmp_path / "source-root", "portable-project")
    trace = store.artifacts.write_json(
        artifact_id="trace-selected",
        artifact_type="trace-dataset",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=_CREATED_AT,
            code_revision=_REVISION,
            source=SourceIdentity(
                kind="file",
                source_id="platform-source:upload-1",
                sha256="a" * 64,
            ),
        ),
        files={"nested/trace.json": {"trace_id": "trace-1"}},
    )
    trace_input = artifact_input(trace)
    task = store.artifacts.write_json(
        artifact_id="tasks-selected",
        artifact_type="task-set",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=_CREATED_AT,
            code_revision=_REVISION,
            inputs=(trace_input,),
        ),
        files={"tasks.json": {"task_ids": ["task-1"]}},
    )
    store.artifacts.write_json(
        artifact_id="tasks-unselected",
        artifact_type="task-set",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=_CREATED_AT,
            code_revision=_REVISION,
        ),
        files={"tasks.json": {"task_ids": ["unselected"]}},
    )
    catalog_pointer = None
    models = None
    if with_catalog:
        catalog = ProjectModelCatalog(
            project_id="portable-project",
            models=(
                _catalog_model("baseline"),
                _catalog_model("candidate-a"),
                _catalog_model("candidate-b"),
                _catalog_model("embedder"),
                _catalog_model("judge"),
                _catalog_model("world-model"),
            ),
        )
        catalog_pointer = persist_project_model_catalog(
            store.artifacts,
            catalog,
            created_at=_CREATED_AT,
            code_revision=_REVISION,
        )
        models = ProjectModelConfiguration(
            world_model="world-model",
            judge="judge",
            embedder="embedder",
            candidates=("baseline", "candidate-a", "candidate-b"),
        )
    stage = ProjectProviderFreeStage(
        trace_dataset=trace_input,
        task_set=artifact_input(task),
    )
    store.initialize(
        ProjectConfig(
            project_id="portable-project",
            trace_preparation=ProjectTracePreparationSettings(source_kind="otlp"),
            provider_free_stage=stage,
            models=models,
            model_catalog=catalog_pointer,
            retrieval=None,
            budgets=None,
        )
    )
    store.paths.runtime_directory.mkdir()
    store.paths.runtime_journal.write_text('{"mutable":true}\n', encoding="utf-8")
    return store


def _canonical_info(name: str, *, mode: int = stat.S_IFREG | 0o600) -> zipfile.ZipInfo:
    """Return one deterministic Unix ZIP member descriptor for adversarial fixtures."""
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = mode << 16
    return info


def _rewrite_bundle(
    source: Path,
    destination: Path,
    transform: Callable[[list[tuple[zipfile.ZipInfo, bytes]]], None],
) -> Path:
    """Copy a bundle after applying one deliberate archive-level corruption."""
    with zipfile.ZipFile(source, "r") as archive:
        entries = [(info, archive.read(info)) for info in archive.infolist()]
    transform(entries)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
        with zipfile.ZipFile(destination, "w") as archive:
            for info, payload in entries:
                archive.writestr(info, payload)
    return destination


def test_provider_free_bundle_is_deterministic_minimal_and_portable(tmp_path: Path) -> None:
    """Only the selected closure round-trips under a different root with stable identities."""
    store = _project_store(tmp_path)
    first = export_project_bundle(
        store,
        tmp_path / "first.wmo-project",
        producer_revision=_REVISION,
    )
    second = export_project_bundle(
        store,
        tmp_path / "second.wmo-project",
        producer_revision=_REVISION,
    )

    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    assert tuple(item.artifact_id for item in first.manifest.selected_artifacts) == (
        "tasks-selected",
        "trace-selected",
    )
    assert first.manifest.completed_stages == (ProjectStage.PREPARING_TRACES,)
    assert first.manifest.runtime_state == "excluded"
    with zipfile.ZipFile(first.path, "r") as archive:
        names = archive.namelist()
    assert "artifacts/tasks-unselected/manifest.json" not in names
    assert "runtime/interactions.jsonl" not in names
    assert all("\\" not in name and not name.startswith("/") for name in names)

    restored = restore_project_bundle(
        first.path,
        root=tmp_path / "different-absolute-root",
        expected_sha256=first.sha256,
    )

    assert restored.load_project() == store.load_project()
    assert restored.artifacts.list_ids() == ("tasks-selected", "trace-selected")
    assert not restored.paths.runtime_directory.exists()
    for artifact_id in restored.artifacts.list_ids():
        assert artifact_input(restored.artifacts.read(artifact_id).manifest) == artifact_input(
            store.artifacts.read(artifact_id).manifest
        )


def test_catalog_binding_round_trips_without_ambient_models_toml(tmp_path: Path) -> None:
    """An explicit project catalog is in closure while root-global catalog bytes are irrelevant."""
    store = _project_store(tmp_path, with_catalog=True)
    store.model_catalog_path.write_text("ambient = 'first'\n", encoding="utf-8")
    first = export_project_bundle(
        store,
        tmp_path / "catalog-first.wmo-project",
        producer_revision=_REVISION,
    )
    store.model_catalog_path.write_text("ambient = 'changed'\n", encoding="utf-8")
    second = export_project_bundle(
        store,
        tmp_path / "catalog-second.wmo-project",
        producer_revision=_REVISION,
    )

    assert first.sha256 == second.sha256
    catalog_pointer = store.load_project().model_catalog
    assert catalog_pointer is not None
    assert catalog_pointer in first.manifest.selected_artifacts
    restored = restore_project_bundle(
        first.path,
        root=tmp_path / "restored-catalog-root",
        expected_sha256=first.sha256,
    )
    assert restored.load_project().model_catalog == catalog_pointer
    assert not restored.model_catalog_path.exists()


@pytest.mark.parametrize(
    ("name", "transform", "message"),
    [
        (
            "missing",
            lambda entries: entries.pop(0),
            "member set",
        ),
        (
            "extra",
            lambda entries: entries.append((_canonical_info("extra.txt"), b"extra")),
            "member set",
        ),
        (
            "duplicate",
            lambda entries: entries.append(entries[0]),
            "duplicate",
        ),
        (
            "case-collision",
            lambda entries: entries.append((_canonical_info("PROJECT.JSON"), b"{}")),
            "colliding member",
        ),
        (
            "unicode-collision",
            lambda entries: entries.extend(
                (
                    (_canonical_info("caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt"), b"first"),
                    (_canonical_info("cafe\N{COMBINING ACUTE ACCENT}.txt"), b"second"),
                )
            ),
            "colliding member",
        ),
        (
            "traversal",
            lambda entries: entries.append((_canonical_info("../escape"), b"escape")),
            "relative POSIX",
        ),
        (
            "absolute",
            lambda entries: entries.append((_canonical_info("/escape"), b"escape")),
            "relative POSIX",
        ),
        (
            "backslash",
            lambda entries: entries.append((_canonical_info(r"artifacts\\escape"), b"escape")),
            "relative POSIX",
        ),
        (
            "symlink",
            lambda entries: entries.append(
                ((_canonical_info("unsafe-link", mode=stat.S_IFLNK | 0o777)), b"project.json")
            ),
            "regular files",
        ),
        (
            "fifo",
            lambda entries: entries.append(
                ((_canonical_info("unsafe-fifo", mode=stat.S_IFIFO | 0o600)), b"pipe")
            ),
            "regular files",
        ),
    ],
)
def test_restore_rejects_unsafe_archive_members(
    tmp_path: Path,
    name: str,
    transform: Callable[[list[tuple[zipfile.ZipInfo, bytes]]], None],
    message: str,
) -> None:
    """Archive membership attacks fail before the destination Project becomes visible."""
    exported = export_project_bundle(
        _project_store(tmp_path),
        tmp_path / "valid.wmo-project",
        producer_revision=_REVISION,
    )
    corrupted = _rewrite_bundle(exported.path, tmp_path / f"{name}.zip", transform)
    digest = bundle_module._sha256_path(corrupted)
    root = tmp_path / f"restore-{name}"

    with pytest.raises(ProjectBundleError, match=message):
        restore_project_bundle(corrupted, root=root, expected_sha256=digest)

    assert not (root / "projects" / "portable-project").exists()


def test_restore_rejects_corruption_schema_drift_and_compression(tmp_path: Path) -> None:
    """Payload drift, unsupported schemas, and compressed expansion inputs fail closed."""
    exported = export_project_bundle(
        _project_store(tmp_path),
        tmp_path / "valid.wmo-project",
        producer_revision=_REVISION,
    )

    def corrupt_payload(entries: list[tuple[zipfile.ZipInfo, bytes]]) -> None:
        """Change a selected payload without updating its bundle digest record."""
        index = next(
            index
            for index, (info, _payload) in enumerate(entries)
            if info.filename.endswith("tasks.json")
        )
        info, _payload = entries[index]
        entries[index] = (info, b'{"task_ids":["changed"]}')

    corrupt = _rewrite_bundle(
        exported.path,
        tmp_path / "corrupt.zip",
        corrupt_payload,
    )
    with pytest.raises(ProjectBundleError, match="digest"):
        restore_project_bundle(
            corrupt,
            root=tmp_path / "corrupt-root",
            expected_sha256=bundle_module._sha256_path(corrupt),
        )

    def change_schema(entries: list[tuple[zipfile.ZipInfo, bytes]]) -> None:
        """Replace the bundle schema with an unsupported future value."""
        index = next(
            index
            for index, (info, _payload) in enumerate(entries)
            if info.filename == "bundle.json"
        )
        info, payload = entries[index]
        raw = bundle_module._JSON_OBJECT_ADAPTER.validate_json(payload)
        assert isinstance(raw, dict)
        raw["schema_version"] = 999
        entries[index] = (info, canonical_json_bytes(raw))

    schema = _rewrite_bundle(exported.path, tmp_path / "schema.zip", change_schema)
    with pytest.raises(ProjectBundleError, match="manifest"):
        restore_project_bundle(
            schema,
            root=tmp_path / "schema-root",
            expected_sha256=bundle_module._sha256_path(schema),
        )

    def compress_member(entries: list[tuple[zipfile.ZipInfo, bytes]]) -> None:
        """Mark one member as compressed so expansion is never attempted."""
        info, payload = entries[0]
        replacement = _canonical_info(info.filename)
        replacement.compress_type = zipfile.ZIP_DEFLATED
        entries[0] = (replacement, payload)

    compressed = _rewrite_bundle(exported.path, tmp_path / "compressed.zip", compress_member)
    with pytest.raises(ProjectBundleError, match="uncompressed"):
        restore_project_bundle(
            compressed,
            root=tmp_path / "compressed-root",
            expected_sha256=bundle_module._sha256_path(compressed),
        )


def test_bundle_rejects_unsupported_project_schema_and_truncation(tmp_path: Path) -> None:
    """Future Project schemas and incomplete archives fail before destination visibility."""
    unsupported = _project_store(tmp_path / "unsupported")
    write_project_config(
        unsupported.paths.project_toml,
        unsupported.load_project().model_copy(update={"schema_version": 999}),
    )
    with pytest.raises(ProjectBundleError, match="schema version is unsupported"):
        export_project_bundle(
            unsupported,
            tmp_path / "unsupported.wmo-project",
            producer_revision=_REVISION,
        )

    exported = export_project_bundle(
        _project_store(tmp_path / "truncated"),
        tmp_path / "valid-for-truncation.wmo-project",
        producer_revision=_REVISION,
    )
    truncated = tmp_path / "truncated.wmo-project"
    truncated.write_bytes(exported.path.read_bytes()[:32])
    root = tmp_path / "truncated-root"
    with pytest.raises(ProjectBundleError, match="readable archive"):
        restore_project_bundle(
            truncated,
            root=root,
            expected_sha256=bundle_module._sha256_path(truncated),
        )
    assert not (root / "projects" / "portable-project").exists()


def test_restore_requires_exact_digest_absent_destination_and_hard_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restore verifies content identity, destination absence, and expansion before writing."""
    exported = export_project_bundle(
        _project_store(tmp_path),
        tmp_path / "valid.wmo-project",
        producer_revision=_REVISION,
    )
    with pytest.raises(ProjectBundleError, match="content digest"):
        restore_project_bundle(
            exported.path,
            root=tmp_path / "wrong-digest-root",
            expected_sha256="0" * 64,
        )

    existing_root = tmp_path / "existing-root"
    existing = existing_root / "projects" / "portable-project"
    existing.mkdir(parents=True)
    sentinel = existing / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ProjectBundleError, match="destination must be absent"):
        restore_project_bundle(
            exported.path,
            root=existing_root,
            expected_sha256=exported.sha256,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    monkeypatch.setattr(bundle_module, "_MAX_EXPANDED_BYTES", 1)
    bounded_root = tmp_path / "bounded-root"
    with pytest.raises(ProjectBundleError, match="expanded-byte limit"):
        restore_project_bundle(
            exported.path,
            root=bounded_root,
            expected_sha256=exported.sha256,
        )
    assert not (bounded_root / "projects" / "portable-project").exists()


def test_bundle_manifest_does_not_duplicate_artifact_provenance() -> None:
    """Stage selection stays small while artifact manifests own source and lineage fields."""
    assert {
        "source",
        "source_id",
        "created_at",
        "inputs",
        "artifact_producer_revision",
    }.isdisjoint(bundle_module.ProjectBundleManifest.model_fields)


def test_export_rejects_secret_bearing_and_absolute_source_provenance(tmp_path: Path) -> None:
    """A self-consistent forged graph still fails the bundle's secret and path boundaries."""
    secret_store = _project_store(tmp_path / "secret")
    _rewrite_trace_graph(
        secret_store,
        payload={"authorization": "Bearer abcdefghijklmnopqrstuvwxyz"},
    )
    with pytest.raises(ProjectBundleError, match="secret boundary"):
        export_project_bundle(
            secret_store,
            tmp_path / "secret.wmo-project",
            producer_revision=_REVISION,
        )

    path_store = _project_store(tmp_path / "path")
    _rewrite_trace_graph(path_store, source_id="/tmp/worker-upload.json")
    with pytest.raises(ProjectBundleError, match="worker-local"):
        export_project_bundle(
            path_store,
            tmp_path / "path.wmo-project",
            producer_revision=_REVISION,
        )


def test_restore_rejects_a_digest_consistent_secret_bearing_artifact(tmp_path: Path) -> None:
    """Recomputed archive and provenance digests cannot bypass artifact secret validation."""
    exported = export_project_bundle(
        _project_store(tmp_path),
        tmp_path / "valid.wmo-project",
        producer_revision=_REVISION,
    )

    def inject_secret(entries: list[tuple[zipfile.ZipInfo, bytes]]) -> None:
        """Rebind every affected digest around one malicious selected trace payload."""
        by_name = {info.filename: index for index, (info, _payload) in enumerate(entries)}
        changed: dict[str, bytes] = {}
        trace_data_path = "artifacts/trace-selected/nested/trace.json"
        changed[trace_data_path] = canonical_json_bytes(
            {"authorization": "Bearer abcdefghijklmnopqrstuvwxyz"}
        )
        trace_manifest_path = "artifacts/trace-selected/manifest.json"
        trace_manifest = ArtifactManifest.model_validate_json(
            entries[by_name[trace_manifest_path]][1]
        ).model_copy(
            update={"files": (file_digest("nested/trace.json", changed[trace_data_path]),)}
        )
        changed[trace_manifest_path] = canonical_json_bytes(trace_manifest)
        trace_pointer = artifact_input(trace_manifest)

        task_manifest_path = "artifacts/tasks-selected/manifest.json"
        task_manifest = ArtifactManifest.model_validate_json(
            entries[by_name[task_manifest_path]][1]
        ).model_copy(update={"inputs": (trace_pointer,)})
        changed[task_manifest_path] = canonical_json_bytes(task_manifest)
        task_pointer = artifact_input(task_manifest)

        project = ProjectConfig.model_validate_json(entries[by_name["project.json"]][1])
        project = project.model_copy(
            update={
                "provider_free_stage": ProjectProviderFreeStage(
                    trace_dataset=trace_pointer,
                    task_set=task_pointer,
                )
            }
        )
        changed["project.json"] = canonical_json_bytes(project)

        bundle_manifest = bundle_module.ProjectBundleManifest.model_validate_json(
            entries[by_name["bundle.json"]][1]
        )
        members = tuple(
            member.model_copy(
                update={
                    "sha256": sha256_bytes(
                        changed.get(member.path, entries[by_name[member.path]][1])
                    ),
                    "size_bytes": len(changed.get(member.path, entries[by_name[member.path]][1])),
                }
            )
            for member in bundle_manifest.members
        )
        bundle_manifest = bundle_manifest.model_copy(
            update={
                "selected_artifacts": tuple(
                    sorted((task_pointer, trace_pointer), key=lambda item: item.artifact_id)
                ),
                "members": members,
                "expanded_size_bytes": sum(member.size_bytes for member in members),
            }
        )
        changed["bundle.json"] = canonical_json_bytes(bundle_manifest)
        for member_path, payload in changed.items():
            index = by_name[member_path]
            info, _old_payload = entries[index]
            entries[index] = (info, payload)

    poisoned = _rewrite_bundle(
        exported.path,
        tmp_path / "secret-bearing.wmo-project",
        inject_secret,
    )
    root = tmp_path / "secret-restore-root"
    with pytest.raises(ProjectBundleError, match="secret boundary"):
        restore_project_bundle(
            poisoned,
            root=root,
            expected_sha256=bundle_module._sha256_path(poisoned),
        )
    assert not (root / "projects" / "portable-project").exists()


def _rewrite_trace_graph(
    store: ProjectStore,
    *,
    payload: dict[str, str] | None = None,
    source_id: str = "platform-source:upload-1",
) -> None:
    """Forge a digest-consistent selected graph for boundary-failure tests."""
    trace_directory = store.paths.artifact_directory("trace-selected")
    trace_manifest = ArtifactManifest.model_validate_json(
        (trace_directory / "manifest.json").read_bytes()
    )
    trace_payload = canonical_json_bytes(payload or {"trace_id": "trace-1"})
    (trace_directory / "nested/trace.json").write_bytes(trace_payload)
    source = trace_manifest.source
    assert source is not None
    trace_manifest = trace_manifest.model_copy(
        update={
            "source": source.model_copy(update={"source_id": source_id}),
            "files": (file_digest("nested/trace.json", trace_payload),),
        }
    )
    (trace_directory / "manifest.json").write_bytes(canonical_json_bytes(trace_manifest))
    trace_pointer = artifact_input(trace_manifest)

    task_directory = store.paths.artifact_directory("tasks-selected")
    task_manifest = ArtifactManifest.model_validate_json(
        (task_directory / "manifest.json").read_bytes()
    ).model_copy(update={"inputs": (trace_pointer,)})
    (task_directory / "manifest.json").write_bytes(canonical_json_bytes(task_manifest))
    project = store.load_project()
    write_project_config(
        store.paths.project_toml,
        project.model_copy(
            update={
                "provider_free_stage": ProjectProviderFreeStage(
                    trace_dataset=trace_pointer,
                    task_set=artifact_input(task_manifest),
                )
            }
        ),
    )


def test_restore_failure_never_exposes_a_partial_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure at the final atomic rename leaves no selected destination Project."""
    exported = export_project_bundle(
        _project_store(tmp_path),
        tmp_path / "valid.wmo-project",
        producer_revision=_REVISION,
    )
    root = tmp_path / "restore-root"

    def fail_rename(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        """Simulate a crash immediately before destination visibility."""
        raise OSError(f"cannot rename {source!s} to {destination!s}")

    monkeypatch.setattr(bundle_module.os, "rename", fail_rename)
    with pytest.raises(ProjectBundleError, match="atomically restore"):
        restore_project_bundle(
            exported.path,
            root=root,
            expected_sha256=exported.sha256,
        )

    assert not (root / "projects" / "portable-project").exists()
    assert not tuple(root.glob(".restore-*.partial"))


def test_export_failure_never_replaces_the_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed final export promotion removes its temporary file and leaves no target."""
    store = _project_store(tmp_path)
    destination = tmp_path / "failed-export.wmo-project"

    def fail_replace(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        """Simulate a crash immediately before bundle publication."""
        raise OSError(f"cannot replace {source!s} with {target!s}")

    monkeypatch.setattr(bundle_module.os, "replace", fail_replace)
    with pytest.raises(ProjectBundleError, match="atomically export"):
        export_project_bundle(
            store,
            destination,
            producer_revision=_REVISION,
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".failed-export.wmo-project.*.partial"))
