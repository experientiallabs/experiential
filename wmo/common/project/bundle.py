"""Deterministic export and atomic restore for one selected WMO Project state.

The immutable bundle intentionally excludes ``ProjectPaths.runtime_directory``. Serving restores
the verified build state first, then attaches its separately owned mutable runtime journal.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    SecretBoundaryError,
    Sha256,
    assert_secret_free,
    canonical_json_bytes,
    sha256_bytes,
    validate_artifact_file_path,
)
from wmo.common.core.files import fsync_directory_best_effort
from wmo.common.core.locks import file_write_lock
from wmo.common.project.catalog import (
    ProjectModelCatalog,
    ProjectModelCatalogError,
    load_project_model_catalog,
)
from wmo.common.project.events import ProjectStage
from wmo.common.project.manifests import ArtifactManifest, artifact_input
from wmo.common.project.paths import ProjectPaths, validate_local_id
from wmo.common.project.project import (
    ProjectConfig,
    require_durable_source_id,
)
from wmo.common.project.store import ProjectStore

_BUNDLE_MANIFEST_PATH = "bundle.json"
_PROJECT_CONFIG_PATH = "project.json"
_MAX_BUNDLE_MEMBERS = 20_000
_MAX_EXPANDED_BYTES = 1_073_741_824
_MAX_ARCHIVE_BYTES = 2 * _MAX_EXPANDED_BYTES
_READ_CHUNK_BYTES = 1024 * 1024
_SUPPORTED_PROJECT_SCHEMA_VERSIONS = frozenset({1, 2, 3})
_JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)
_SHA256_ADAPTER = TypeAdapter(Sha256)
_STAGE_ORDER = {
    ProjectStage.PREPARING_TRACES: 0,
    ProjectStage.BUILDING_WORLD_MODEL: 1,
    ProjectStage.OPTIMIZING_ROUTER: 2,
    ProjectStage.COMPLETING_REPORT: 3,
}


class ProjectBundleError(RuntimeError):
    """A Project could not be exported or restored without weakening bundle guarantees."""


class ProjectBundleMember(ContractModel):
    """Digest and expanded size of one canonical regular file inside a bundle."""

    path: str = Field(min_length=1, max_length=4_096)
    sha256: Sha256
    size_bytes: int = Field(ge=0, le=_MAX_EXPANDED_BYTES)

    @field_validator("path")
    @classmethod
    def _require_safe_member_path(cls, value: str) -> str:
        """Return one normalized portable relative member path."""
        return validate_artifact_file_path(value).as_posix()


class ProjectBundleManifest(ContractModel):
    """Versioned binding for one Project config and its selected artifact closure."""

    schema_version: Literal[1] = 1
    project_id: ArtifactId
    project_schema_version: int = Field(ge=1)
    producer_revision: str = Field(min_length=1, max_length=256)
    selected_artifacts: tuple[ArtifactInput, ...]
    completed_stages: tuple[ProjectStage, ...]
    members: tuple[ProjectBundleMember, ...] = Field(
        min_length=1,
        max_length=_MAX_BUNDLE_MEMBERS,
    )
    expanded_file_count: int = Field(ge=1, le=_MAX_BUNDLE_MEMBERS)
    expanded_size_bytes: int = Field(ge=0, le=_MAX_EXPANDED_BYTES)
    runtime_state: Literal["excluded"] = "excluded"

    @field_validator("producer_revision")
    @classmethod
    def _require_canonical_revision(cls, value: str) -> str:
        """Reject blank or padded producer identities."""
        if value != value.strip():
            raise ValueError("bundle producer_revision must not have surrounding whitespace")
        return value

    @field_validator("selected_artifacts")
    @classmethod
    def _require_sorted_unique_artifacts(
        cls, value: tuple[ArtifactInput, ...]
    ) -> tuple[ArtifactInput, ...]:
        """Require one canonical pointer per selected artifact ID."""
        artifact_ids = tuple(item.artifact_id for item in value)
        if not artifact_ids:
            raise ValueError("project bundle needs at least one selected artifact")
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("project bundle selected artifacts must not repeat")
        if artifact_ids != tuple(sorted(artifact_ids)):
            raise ValueError("project bundle selected artifacts must be sorted")
        return value

    @field_validator("completed_stages")
    @classmethod
    def _require_ordered_unique_stages(
        cls, value: tuple[ProjectStage, ...]
    ) -> tuple[ProjectStage, ...]:
        """Require completed durable stages in the canonical workflow order."""
        if not value:
            raise ValueError("project bundle needs at least one completed durable stage")
        if len(set(value)) != len(value):
            raise ValueError("completed project stages must not repeat")
        if value != tuple(sorted(value, key=_STAGE_ORDER.__getitem__)):
            raise ValueError("completed project stages must follow workflow order")
        return value

    @field_validator("members")
    @classmethod
    def _require_sorted_unique_members(
        cls, value: tuple[ProjectBundleMember, ...]
    ) -> tuple[ProjectBundleMember, ...]:
        """Require canonical member order without exact or case-folding collisions."""
        paths = tuple(item.path for item in value)
        if paths != tuple(sorted(paths)):
            raise ValueError("project bundle members must be sorted by path")
        _require_unique_portable_names(paths)
        return value

    @model_validator(mode="after")
    def _require_exact_expansion_totals(self) -> ProjectBundleManifest:
        """Bind the declared expansion bounds to the complete member list."""
        if self.expanded_file_count != len(self.members):
            raise ValueError("bundle expanded_file_count differs from its member list")
        if self.expanded_size_bytes != sum(item.size_bytes for item in self.members):
            raise ValueError("bundle expanded_size_bytes differs from its member list")
        if _PROJECT_CONFIG_PATH not in {item.path for item in self.members}:
            raise ValueError("project bundle manifest does not bind project.json")
        return self


@dataclass(frozen=True)
class ExportedProjectBundle:
    """Published bundle path, content identity, size, and parsed manifest."""

    path: Path
    sha256: str
    size_bytes: int
    manifest: ProjectBundleManifest


@dataclass(frozen=True)
class _LoadedBundle:
    """Fully verified archive content used only before atomic restore visibility."""

    manifest: ProjectBundleManifest
    project: ProjectConfig
    artifacts: Mapping[str, ArtifactManifest]
    payloads: Mapping[str, bytes]


def export_project_bundle(
    store: ProjectStore,
    destination: Path,
    *,
    producer_revision: str,
) -> ExportedProjectBundle:
    """Export one selected Project state as a deterministic verified bundle.

    Args:
        store: Project-local store whose selected immutable state is exported.
        destination: Final bundle file, published through an atomic replacement.
        producer_revision: Exact WMO revision or installed distribution identity.

    Returns:
        Final path, content digest, size, and parsed versioned manifest.

    Raises:
        ProjectBundleError: The selected graph is unsafe, incomplete, oversized, or cannot be
            published atomically.
    """
    try:
        project = store.load_project()
        _validate_project_config(project)

        def resolve(artifact_id: str) -> ArtifactManifest:
            """Read one fully verified selected artifact manifest."""
            return store.artifacts.read(artifact_id).manifest

        selected_artifacts, manifests = _collect_selected_artifacts(project, resolve)
        _validate_catalog_from_store(store, project, manifests)
        completed_stages = _completed_stages(project)
        payloads = _export_payloads(store, project, manifests)
        members = _bundle_members(payloads)
        manifest = ProjectBundleManifest(
            project_id=project.project_id,
            project_schema_version=project.schema_version,
            producer_revision=producer_revision,
            selected_artifacts=selected_artifacts,
            completed_stages=completed_stages,
            members=members,
            expanded_file_count=len(members),
            expanded_size_bytes=sum(item.size_bytes for item in members),
        )
        assert_secret_free(manifest)
        if manifest.expanded_size_bytes + len(canonical_json_bytes(manifest)) > _MAX_EXPANDED_BYTES:
            raise ProjectBundleError(
                f"Project bundle exceeds the {_MAX_EXPANDED_BYTES} total expanded-byte limit"
            )
    except ProjectBundleError:
        raise
    except (OSError, RuntimeError, SecretBoundaryError, ValidationError, ValueError) as exc:
        raise ProjectBundleError(f"cannot export selected Project state: {exc}") from exc
    final_path = Path(destination)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists() and final_path.is_dir():
        raise ProjectBundleError(f"bundle destination is a directory: {final_path}")
    staging = final_path.parent / f".{final_path.name}.{uuid4().hex}.partial"
    try:
        with file_write_lock(final_path, what="project bundle export"):
            _write_bundle_archive(staging, payloads, manifest)
            digest = _sha256_path(staging)
            size_bytes = staging.stat().st_size
            os.replace(staging, final_path)
            fsync_directory_best_effort(final_path.parent)
    except ProjectBundleError:
        staging.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        staging.unlink(missing_ok=True)
        raise ProjectBundleError(f"cannot atomically export Project bundle: {exc}") from exc
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return ExportedProjectBundle(
        path=final_path,
        sha256=digest,
        size_bytes=size_bytes,
        manifest=manifest,
    )


def restore_project_bundle(
    bundle_path: Path,
    *,
    root: Path,
    expected_sha256: str,
) -> ProjectStore:
    """Verify and atomically restore one Project into an absent scoped destination.

    Args:
        bundle_path: Downloaded canonical bundle file.
        root: Local WMO root that will own ``projects/<project_id>``.
        expected_sha256: Content address supplied by the bundle storage owner.

    Returns:
        Project store addressing the newly visible restored state.

    Raises:
        ProjectBundleError: The content address, archive, graph, or destination is unsafe.
    """
    try:
        expected = _SHA256_ADAPTER.validate_python(expected_sha256)
    except ValidationError as exc:
        raise ProjectBundleError("expected bundle digest is not a lowercase SHA-256 value") from exc
    source = Path(bundle_path)
    with _open_regular_file(source) as source_handle:
        with tempfile.TemporaryFile(mode="w+b") as snapshot:
            actual = _copy_bundle_snapshot(source_handle, snapshot)
            if actual != expected:
                raise ProjectBundleError(
                    "Project bundle content digest does not match expected_sha256"
                )
            loaded = _load_and_verify_bundle(snapshot)
    destination_root = Path(root)
    _require_safe_restore_root(destination_root)
    paths = ProjectPaths(destination_root, loaded.project.project_id)
    destination = paths.project_directory
    if destination.exists() or destination.is_symlink():
        raise ProjectBundleError(
            f"restore destination must be absent and scoped to one Project: {destination}"
        )
    paths.projects_directory.mkdir(parents=True, exist_ok=True)
    staging_root = destination_root / f".restore-{uuid4().hex}.partial"
    staged = ProjectStore(staging_root, loaded.project.project_id)
    try:
        _materialize_loaded_bundle(staged, loaded)
        _verify_restored_project(staged, loaded)
        os.rename(staged.paths.project_directory, destination)
        fsync_directory_best_effort(paths.projects_directory)
    except ProjectBundleError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProjectBundleError(f"cannot atomically restore Project bundle: {exc}") from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return ProjectStore(destination_root, loaded.project.project_id)


def _export_payloads(
    store: ProjectStore,
    project: ProjectConfig,
    manifests: Mapping[str, ArtifactManifest],
) -> dict[str, bytes]:
    """Read the canonical Project config and selected artifact closure into bundle members."""
    payloads = {_PROJECT_CONFIG_PATH: canonical_json_bytes(project)}
    for artifact_id in sorted(manifests):
        manifest = manifests[artifact_id]
        prefix = f"artifacts/{artifact_id}"
        payloads[f"{prefix}/manifest.json"] = canonical_json_bytes(manifest)
        for entry in manifest.files:
            payloads[f"{prefix}/{entry.path}"] = store.artifacts.read_bytes(
                artifact_id,
                entry.path,
            )
    return payloads


def _bundle_members(payloads: Mapping[str, bytes]) -> tuple[ProjectBundleMember, ...]:
    """Build bounded digest records for every member except the manifest itself."""
    if len(payloads) > _MAX_BUNDLE_MEMBERS:
        raise ProjectBundleError(
            f"Project bundle exceeds the {_MAX_BUNDLE_MEMBERS} expanded-file limit"
        )
    members = tuple(
        ProjectBundleMember(
            path=path,
            sha256=sha256_bytes(payload),
            size_bytes=len(payload),
        )
        for path, payload in sorted(payloads.items())
    )
    expanded = sum(item.size_bytes for item in members)
    if expanded > _MAX_EXPANDED_BYTES:
        raise ProjectBundleError(
            f"Project bundle exceeds the {_MAX_EXPANDED_BYTES} expanded-byte limit"
        )
    return members


def _write_bundle_archive(
    path: Path,
    payloads: Mapping[str, bytes],
    manifest: ProjectBundleManifest,
) -> None:
    """Write one canonical uncompressed ZIP archive to a new private path."""
    with path.open("xb") as handle:
        with zipfile.ZipFile(handle, mode="w", allowZip64=True) as archive:
            for member_path in sorted(payloads):
                archive.writestr(_canonical_zip_info(member_path), payloads[member_path])
            archive.writestr(
                _canonical_zip_info(_BUNDLE_MANIFEST_PATH),
                canonical_json_bytes(manifest),
            )
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_zip_info(path: str) -> zipfile.ZipInfo:
    """Return deterministic regular-file metadata for one canonical archive member."""
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    return info


def _load_and_verify_bundle(handle: BinaryIO) -> _LoadedBundle:
    """Verify every archive member through the descriptor authenticated by the caller."""
    try:
        with zipfile.ZipFile(handle, mode="r") as archive:
            payloads = _read_archive_payloads(archive)
    except ProjectBundleError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ProjectBundleError(
            f"Project bundle is not a complete readable archive: {exc}"
        ) from exc
    manifest_payload = payloads.get(_BUNDLE_MANIFEST_PATH)
    if manifest_payload is None:
        raise ProjectBundleError("Project bundle has no bundle.json manifest")
    try:
        manifest = ProjectBundleManifest.model_validate_json(manifest_payload)
        assert_secret_free(manifest)
    except (SecretBoundaryError, ValidationError, ValueError) as exc:
        raise ProjectBundleError("Project bundle manifest is invalid or unsupported") from exc
    expected_names = {_BUNDLE_MANIFEST_PATH, *(item.path for item in manifest.members)}
    if set(payloads) != expected_names:
        raise ProjectBundleError("Project bundle member set differs from its manifest")
    for member in manifest.members:
        payload = payloads[member.path]
        if len(payload) != member.size_bytes or sha256_bytes(payload) != member.sha256:
            raise ProjectBundleError(f"Project bundle member digest changed: {member.path}")
    project_payload = payloads.get(_PROJECT_CONFIG_PATH)
    if project_payload is None:
        raise ProjectBundleError("Project bundle has no project.json")
    try:
        project = ProjectConfig.model_validate_json(project_payload)
        assert_secret_free(project)
        _validate_project_config(project)
    except (SecretBoundaryError, ValidationError, ValueError) as exc:
        raise ProjectBundleError(f"bundled Project configuration is invalid: {exc}") from exc
    if (
        project.project_id != manifest.project_id
        or project.schema_version != manifest.project_schema_version
    ):
        raise ProjectBundleError("bundled Project identity or schema differs from bundle.json")
    artifact_manifests = _parse_artifact_members(payloads)

    def resolve(artifact_id: str) -> ArtifactManifest:
        """Resolve one manifest already bounded by the archive member set."""
        try:
            return artifact_manifests[artifact_id]
        except KeyError as exc:
            raise ProjectBundleError(f"selected artifact is missing: {artifact_id}") from exc

    selected, closure = _collect_selected_artifacts(project, resolve)
    if selected != manifest.selected_artifacts:
        raise ProjectBundleError("selected artifact closure differs from bundle.json")
    if set(artifact_manifests) != set(closure):
        raise ProjectBundleError("bundle contains an unselected artifact directory")
    if _completed_stages(project) != manifest.completed_stages:
        raise ProjectBundleError("completed durable stage selection differs from bundle.json")
    _validate_catalog_from_payloads(project, closure, payloads)
    return _LoadedBundle(
        manifest=manifest,
        project=project,
        artifacts=closure,
        payloads=payloads,
    )


def _read_archive_payloads(archive: zipfile.ZipFile) -> dict[str, bytes]:
    """Reject noncanonical ZIP metadata and read every member within hard expansion bounds."""
    infos = archive.infolist()
    if not infos or len(infos) > _MAX_BUNDLE_MEMBERS + 1:
        raise ProjectBundleError("Project bundle archive has an invalid member count")
    names = []
    declared_size = 0
    for info in infos:
        if info.orig_filename != info.filename:
            raise ProjectBundleError("Project bundle member names cannot contain NUL bytes")
        try:
            name = validate_artifact_file_path(info.filename).as_posix()
        except ValueError as exc:
            raise ProjectBundleError(
                f"Project bundle member path is not relative POSIX: {exc}"
            ) from exc
        if info.is_dir():
            raise ProjectBundleError("Project bundle members must be regular files")
        unix_mode = info.external_attr >> 16
        if info.create_system != 3 or stat.S_IFMT(unix_mode) != stat.S_IFREG:
            raise ProjectBundleError("Project bundle members must be canonical Unix regular files")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ProjectBundleError("Project bundle members must be uncompressed")
        if info.flag_bits & 0x1:
            raise ProjectBundleError("Project bundle members must not be encrypted")
        declared_size += info.file_size
        if declared_size > _MAX_EXPANDED_BYTES:
            raise ProjectBundleError("Project bundle exceeds its hard expanded-byte limit")
        names.append(name)
    _require_unique_portable_names(names)
    payloads: dict[str, bytes] = {}
    actual_size = 0
    for info, name in zip(infos, names, strict=True):
        payload = _read_bounded_member(archive, info)
        actual_size += len(payload)
        if actual_size > _MAX_EXPANDED_BYTES:
            raise ProjectBundleError("Project bundle expansion exceeds its hard byte limit")
        payloads[name] = payload
    return payloads


def _read_bounded_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """Read one stored member without trusting its declared size as an allocation bound."""
    chunks: list[bytes] = []
    total = 0
    with archive.open(info, mode="r") as member:
        while chunk := member.read(_READ_CHUNK_BYTES):
            total += len(chunk)
            if total > info.file_size or total > _MAX_EXPANDED_BYTES:
                raise ProjectBundleError(
                    f"Project bundle member expands beyond bounds: {info.filename}"
                )
            chunks.append(chunk)
    if total != info.file_size:
        raise ProjectBundleError(f"Project bundle member is truncated: {info.filename}")
    return b"".join(chunks)


def _parse_artifact_members(
    payloads: Mapping[str, bytes],
) -> dict[str, ArtifactManifest]:
    """Parse every artifact manifest and bind its complete declared member set."""
    artifact_files: dict[str, set[str]] = {}
    for path in payloads:
        if not path.startswith("artifacts/"):
            continue
        parts = PurePosixPath(path).parts
        if len(parts) < 3:
            raise ProjectBundleError(f"artifact bundle member has an incomplete path: {path}")
        try:
            artifact_id = validate_local_id(parts[1], label="artifact ID")
        except ValueError as exc:
            raise ProjectBundleError(f"artifact bundle member has an invalid ID: {path}") from exc
        artifact_files.setdefault(artifact_id, set()).add("/".join(parts[2:]))
    manifests: dict[str, ArtifactManifest] = {}
    for artifact_id, relative_files in artifact_files.items():
        manifest_path = f"artifacts/{artifact_id}/manifest.json"
        payload = payloads.get(manifest_path)
        if payload is None:
            raise ProjectBundleError(f"bundled artifact has no manifest.json: {artifact_id}")
        try:
            manifest = ArtifactManifest.model_validate_json(payload)
            assert_secret_free(manifest)
        except (SecretBoundaryError, ValidationError, ValueError) as exc:
            raise ProjectBundleError(
                f"bundled artifact manifest is invalid: {artifact_id}"
            ) from exc
        if manifest.artifact_id != artifact_id:
            raise ProjectBundleError(
                f"bundled artifact directory differs from its manifest: {artifact_id}"
            )
        expected = {"manifest.json", *(item.path for item in manifest.files)}
        if relative_files != expected:
            raise ProjectBundleError(
                f"bundled artifact file set differs from manifest: {artifact_id}"
            )
        for entry in manifest.files:
            data = payloads[f"artifacts/{artifact_id}/{entry.path}"]
            if len(data) != entry.size_bytes or sha256_bytes(data) != entry.sha256:
                raise ProjectBundleError(
                    f"bundled artifact data digest changed: {artifact_id}/{entry.path}"
                )
        manifests[artifact_id] = manifest
    return manifests


def _collect_selected_artifacts(
    project: ProjectConfig,
    resolve: Callable[[str], ArtifactManifest],
) -> tuple[tuple[ArtifactInput, ...], dict[str, ArtifactManifest]]:
    """Resolve the exact transitive closure of every selected Project artifact pointer."""
    direct = _project_artifact_inputs(project)
    if not direct:
        raise ProjectBundleError("Project has no completed durable artifact selection to export")
    expected: dict[str, ArtifactInput] = {}
    manifests: dict[str, ArtifactManifest] = {}
    visited: set[str] = set()
    visiting: set[str] = set()
    stack: list[tuple[ArtifactInput, bool]] = [(pointer, False) for pointer in reversed(direct)]
    while stack:
        pointer, exiting = stack.pop()
        previous = expected.get(pointer.artifact_id)
        if previous is not None and previous != pointer:
            raise ProjectBundleError(
                f"artifact {pointer.artifact_id} has conflicting selected manifest digests"
            )
        expected[pointer.artifact_id] = pointer
        if exiting:
            visiting.remove(pointer.artifact_id)
            visited.add(pointer.artifact_id)
            continue
        if pointer.artifact_id in visited:
            continue
        if pointer.artifact_id in visiting:
            raise ProjectBundleError("selected artifact graph contains a provenance cycle")
        manifest = resolve(pointer.artifact_id)
        if artifact_input(manifest) != pointer:
            raise ProjectBundleError(
                f"selected artifact manifest digest changed: {pointer.artifact_id}"
            )
        _validate_manifest_source(manifest)
        manifests[pointer.artifact_id] = manifest
        visiting.add(pointer.artifact_id)
        stack.append((pointer, True))
        stack.extend((dependency, False) for dependency in reversed(manifest.inputs))
    selected = tuple(expected[artifact_id] for artifact_id in sorted(expected))
    return selected, manifests


def _project_artifact_inputs(project: ProjectConfig) -> tuple[ArtifactInput, ...]:
    """Return every explicitly selected ArtifactInput nested in Project configuration."""
    pointers: list[ArtifactInput] = []

    def collect(value: object) -> None:
        """Walk typed Project fields without interpreting arbitrary serialized dictionaries."""
        if isinstance(value, ArtifactInput):
            pointers.append(value)
            return
        if isinstance(value, BaseModel):
            for field_name in type(value).model_fields:
                collect(getattr(value, field_name))
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                collect(item)

    collect(project)
    by_id: dict[str, ArtifactInput] = {}
    for pointer in pointers:
        previous = by_id.get(pointer.artifact_id)
        if previous is not None and previous != pointer:
            raise ProjectBundleError(
                f"Project selects conflicting digests for artifact {pointer.artifact_id}"
            )
        by_id[pointer.artifact_id] = pointer
    return tuple(by_id[artifact_id] for artifact_id in sorted(by_id))


def _completed_stages(project: ProjectConfig) -> tuple[ProjectStage, ...]:
    """Derive completed WMO stage vocabulary only from durable selected pointers."""
    stages = []
    if project.provider_free_stage is not None or project.build is not None:
        stages.append(ProjectStage.PREPARING_TRACES)
    if project.build is not None:
        stages.append(ProjectStage.BUILDING_WORLD_MODEL)
    return tuple(stages)


def _validate_project_config(project: ProjectConfig) -> None:
    """Enforce portable config state without reading ambient root-global files."""
    assert_secret_free(project)
    if project.schema_version not in _SUPPORTED_PROJECT_SCHEMA_VERSIONS:
        raise ProjectBundleError(
            f"Project schema version is unsupported by this bundle reader: {project.schema_version}"
        )
    _reject_absolute_path_values(project.model_dump(mode="json"), path="project")
    if (project.models is None) != (project.model_catalog is None):
        raise ProjectBundleError(
            "Project model roles and project-scoped catalog pointer must be selected together"
        )


def _validate_manifest_source(manifest: ArtifactManifest) -> None:
    """Reject worker-local source provenance while retaining durable URI labels."""
    if manifest.source is None:
        return
    try:
        require_durable_source_id(manifest.source.source_id)
    except ValueError as exc:
        raise ProjectBundleError(
            f"artifact {manifest.artifact_id} has worker-local source provenance: {exc}"
        ) from exc


def _validate_catalog_from_store(
    store: ProjectStore,
    project: ProjectConfig,
    manifests: Mapping[str, ArtifactManifest],
) -> None:
    """Verify an optional catalog artifact from its explicit Project pointer only."""
    pointer = project.model_catalog
    if pointer is None:
        return
    if pointer.artifact_id not in manifests:
        raise ProjectBundleError("Project catalog pointer is absent from selected closure")
    try:
        catalog = load_project_model_catalog(store.artifacts, pointer)
    except ProjectModelCatalogError as exc:
        raise ProjectBundleError(str(exc)) from exc
    _validate_catalog_roles(project, catalog)


def _validate_catalog_from_payloads(
    project: ProjectConfig,
    manifests: Mapping[str, ArtifactManifest],
    payloads: Mapping[str, bytes],
) -> None:
    """Verify an optional catalog artifact from already bounded archive payloads."""
    pointer = project.model_catalog
    if pointer is None:
        return
    manifest = manifests.get(pointer.artifact_id)
    if manifest is None or artifact_input(manifest) != pointer:
        raise ProjectBundleError("bundled Project catalog pointer is stale or missing")
    if manifest.artifact_type != "project-model-catalog":
        raise ProjectBundleError("bundled Project catalog has the wrong artifact type")
    if tuple(item.path for item in manifest.files) != ("catalog.json",):
        raise ProjectBundleError("bundled Project catalog must contain only catalog.json")
    try:
        catalog = ProjectModelCatalog.model_validate_json(
            payloads[f"artifacts/{pointer.artifact_id}/catalog.json"]
        )
    except (KeyError, ValidationError, ValueError) as exc:
        raise ProjectBundleError("bundled Project catalog payload is invalid") from exc
    _validate_catalog_roles(project, catalog)


def _validate_catalog_roles(project: ProjectConfig, catalog: ProjectModelCatalog) -> None:
    """Bind Project role aliases to one complete project-scoped catalog snapshot."""
    if catalog.project_id != project.project_id:
        raise ProjectBundleError("Project catalog belongs to a different Project")
    models = project.models
    if models is None:
        raise ProjectBundleError("Project catalog cannot be selected without model roles")
    available = {item.alias for item in catalog.models}
    required = {
        models.world_model,
        models.judge,
        models.embedder,
        *models.candidates,
    }
    missing = sorted(required - available)
    if missing:
        raise ProjectBundleError(f"Project catalog is missing selected model aliases: {missing}")


def _materialize_loaded_bundle(store: ProjectStore, loaded: _LoadedBundle) -> None:
    """Write every verified immutable artifact and the config into a private staging root."""
    for artifact_id in sorted(loaded.artifacts):
        manifest = loaded.artifacts[artifact_id]
        files = {
            entry.path: loaded.payloads[f"artifacts/{artifact_id}/{entry.path}"]
            for entry in manifest.files
        }
        written = store.artifacts.write(
            artifact_id=artifact_id,
            artifact_type=manifest.artifact_type,
            envelope=ArtifactEnvelope(
                schema_version=manifest.schema_version,
                created_at=manifest.created_at,
                inputs=manifest.inputs,
                code_revision=manifest.code_revision,
                source=manifest.source,
            ),
            files=files,
        )
        if written != manifest:
            raise ProjectBundleError(f"restored artifact manifest changed: {artifact_id}")
    store.initialize(loaded.project)


def _verify_restored_project(store: ProjectStore, loaded: _LoadedBundle) -> None:
    """Reopen staged state and verify stage, closure, catalog, and runtime exclusion contracts."""
    project = store.load_project()
    if project != loaded.project:
        raise ProjectBundleError("restored Project configuration changed")
    if project.provider_free_stage is not None:
        store.bind_provider_free_stage(project.provider_free_stage)
    if project.build is not None:
        store.bind_completed_build(project.build)

    def resolve(artifact_id: str) -> ArtifactManifest:
        """Resolve one fully verified staged artifact manifest."""
        return store.artifacts.read(artifact_id).manifest

    selected, manifests = _collect_selected_artifacts(project, resolve)
    if selected != loaded.manifest.selected_artifacts or set(manifests) != set(loaded.artifacts):
        raise ProjectBundleError("restored selected artifact closure changed")
    _validate_catalog_from_store(store, project, manifests)
    if store.paths.runtime_directory.exists():
        raise ProjectBundleError("immutable bundle restore must not create mutable runtime state")


def _require_safe_restore_root(root: Path) -> None:
    """Reject a symlinked or non-directory restore root before creating staged state."""
    if root.is_symlink():
        raise ProjectBundleError(f"restore root must not be a symlink: {root}")
    if root.exists() and not root.is_dir():
        raise ProjectBundleError(f"restore root must be a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    projects = root / "projects"
    if projects.is_symlink():
        raise ProjectBundleError(f"restore projects directory must not be a symlink: {projects}")
    if projects.exists() and not projects.is_dir():
        raise ProjectBundleError(f"restore projects path must be a directory: {projects}")


def _reject_absolute_path_values(value: JsonValue, *, path: str) -> None:
    """Reject local absolute filesystem strings from portable Project metadata."""
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_absolute_path_values(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_path_values(item, path=f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    parsed = urlsplit(value)
    windows = PureWindowsPath(value)
    if (
        PurePosixPath(value).is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or parsed.scheme.casefold() == "file"
    ):
        raise ProjectBundleError(f"portable Project metadata contains an absolute path at {path}")


def _require_unique_portable_names(names: Sequence[str]) -> None:
    """Reject duplicate, case-colliding, and Unicode-normalization-colliding member names."""
    exact: set[str] = set()
    portable: set[str] = set()
    for name in names:
        if name in exact:
            raise ProjectBundleError(f"Project bundle contains duplicate member {name}")
        exact.add(name)
        normalized = unicodedata.normalize("NFC", name).casefold()
        if normalized in portable:
            raise ProjectBundleError(
                f"Project bundle contains a case- or Unicode-colliding member {name}"
            )
        portable.add(normalized)


@contextmanager
def _open_regular_file(path: Path) -> Iterator[BinaryIO]:
    """Open one regular file descriptor without following a final symlink."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise ProjectBundleError("secure bundle reads require O_NOFOLLOW")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ProjectBundleError(f"Project bundle is missing or unsafe: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ProjectBundleError(f"Project bundle is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle
    finally:
        os.close(descriptor)


def _sha256_path(path: Path) -> str:
    """Hash one regular non-symlink file through a stable descriptor."""
    with _open_regular_file(Path(path)) as handle:
        return _sha256_file(handle)


def _sha256_file(handle: BinaryIO) -> str:
    """Hash and rewind one open file so its authenticated bytes can be consumed next."""
    handle.seek(0)
    digest = hashlib.sha256(usedforsecurity=False)
    while chunk := handle.read(_READ_CHUNK_BYTES):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _copy_bundle_snapshot(source: BinaryIO, snapshot: BinaryIO) -> str:
    """Hash one bounded source read into the private snapshot used for restore."""
    source.seek(0)
    digest = hashlib.sha256(usedforsecurity=False)
    size_bytes = 0
    while chunk := source.read(_READ_CHUNK_BYTES):
        size_bytes += len(chunk)
        if size_bytes > _MAX_ARCHIVE_BYTES:
            raise ProjectBundleError(
                f"Project bundle archive exceeds the {_MAX_ARCHIVE_BYTES} byte limit"
            )
        digest.update(chunk)
        snapshot.write(chunk)
    snapshot.flush()
    snapshot.seek(0)
    return digest.hexdigest()
