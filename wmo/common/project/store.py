"""Atomic immutable artifact storage in the canonical project-local `.wmo` layout."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    SecretBoundaryError,
    assert_secret_free,
    assert_text_secret_free,
    canonical_json_bytes,
    validate_artifact_file_path,
)
from wmo.common.core.files import write_bytes_atomic
from wmo.common.core.locks import file_write_lock
from wmo.common.project.manifests import ArtifactFile, ArtifactManifest, file_digest
from wmo.common.project.paths import ProjectPaths, validate_local_id
from wmo.common.project.project import ProjectConfig, load_project_config, write_project_config

logger = logging.getLogger(__name__)

_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)


class ArtifactStoreError(RuntimeError):
    """Base error for immutable local artifact storage failures."""


class ArtifactAlreadyExistsError(ArtifactStoreError):
    """A completed artifact ID was reused instead of creating a new artifact."""


class ArtifactCorruptionError(ArtifactStoreError):
    """A completed artifact no longer matches its immutable manifest."""


class ProjectStoreError(RuntimeError):
    """Project initialization or mutable review-draft persistence failed."""


@dataclass(frozen=True)
class StoredArtifact:
    """A digest-verified immutable artifact directory and its parsed manifest."""

    directory: Path
    manifest: ArtifactManifest


class ArtifactStore:
    """Writes and reads immutable, digest-verified artifact directories for one project."""

    def __init__(self, paths: ProjectPaths) -> None:
        """Create a store rooted at one project's canonical artifact directory."""
        self._paths = paths

    @property
    def project_directory(self) -> Path:
        """Return the project-owned local state directory for durable coordination records.

        Immutable artifacts remain under ``artifacts``. Callers that need a mutable, local-only
        coordination record, such as an in-flight paid-work lease, use this project directory so
        independent processes addressing the same project contend on one durable location.
        """
        return self._paths.project_directory

    def write(
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, bytes],
    ) -> ArtifactManifest:
        """Atomically write one completed immutable artifact directory.

        Args:
            artifact_id: New stable artifact identifier. Existing IDs cannot be overwritten.
            artifact_type: Stable domain type, such as ``task-set`` or ``evaluation``.
            envelope: Shared provenance for the completed artifact.
            files: Complete data-file payloads keyed by safe relative POSIX path.

        Returns:
            The immutable manifest that names and digests every data file.

        Raises:
            ArtifactAlreadyExistsError: A completed artifact already has this ID.
            ArtifactStoreError: A path, secret boundary, or atomic write requirement failed.
        """
        validated_artifact_id = validate_local_id(artifact_id, label="artifact ID")
        validated_artifact_type = validate_local_id(artifact_type, label="artifact type")
        if not files:
            raise ArtifactStoreError("completed artifacts need at least one data file")
        normalized_files = []
        seen_paths: set[str] = set()
        manifest_files = []
        for relative_path, payload in sorted(files.items()):
            try:
                validated_path = validate_artifact_file_path(relative_path).as_posix()
            except ValueError as exc:
                raise ArtifactStoreError(str(exc)) from exc
            if validated_path == "manifest.json":
                raise ArtifactStoreError("artifact data files cannot replace manifest.json")
            if validated_path in seen_paths:
                raise ArtifactStoreError(
                    f"artifact data files repeat normalized path {validated_path}"
                )
            _assert_payload_secret_free(validated_path, payload)
            seen_paths.add(validated_path)
            normalized_files.append((validated_path, payload))
            manifest_files.append(file_digest(validated_path, payload))
        manifest = ArtifactManifest(
            artifact_id=validated_artifact_id,
            artifact_type=validated_artifact_type,
            schema_version=envelope.schema_version,
            created_at=envelope.created_at,
            inputs=envelope.inputs,
            code_revision=envelope.code_revision,
            source=envelope.source,
            files=tuple(manifest_files),
        )
        try:
            assert_secret_free(manifest)
        except SecretBoundaryError as exc:
            raise ArtifactStoreError(str(exc)) from exc
        self._paths.artifacts_directory.mkdir(parents=True, exist_ok=True)
        destination = self._paths.artifact_directory(validated_artifact_id)
        with file_write_lock(destination, what=f"artifact {validated_artifact_id}"):
            if destination.exists():
                raise ArtifactAlreadyExistsError(
                    f"completed artifact already exists and is immutable: {validated_artifact_id}"
                )
            staging = self._paths.artifacts_directory / (
                f".{validated_artifact_id}.{uuid4().hex}.partial"
            )
            try:
                staging.mkdir(mode=0o700)
                for relative_path, payload in normalized_files:
                    _write_staged_file(staging / relative_path, payload)
                _write_staged_file(staging / "manifest.json", canonical_json_bytes(manifest))
                _fsync_staging_tree(staging)
                if destination.exists():
                    raise ArtifactAlreadyExistsError(
                        "completed artifact already exists and is immutable: "
                        f"{validated_artifact_id}"
                    )
                os.rename(staging, destination)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        _fsync_directory_best_effort(self._paths.artifacts_directory)
        return manifest

    def write_json(
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Persist deterministic secret-free JSON files as one immutable artifact.

        Args:
            artifact_id: New stable artifact identifier.
            artifact_type: Stable domain type for the manifest.
            envelope: Shared immutable provenance.
            files: Structured records keyed by their JSON file paths.

        Returns:
            The completed digest-verified manifest.

        Raises:
            ArtifactStoreError: If a path or record violates the artifact contract.
        """
        serialized_files: dict[str, bytes] = {}
        for relative_path, value in files.items():
            if Path(relative_path).suffix != ".json":
                raise ArtifactStoreError("write_json requires .json data-file paths")
            try:
                assert_secret_free(value)
            except SecretBoundaryError as exc:
                raise ArtifactStoreError(str(exc)) from exc
            serialized_files[relative_path] = canonical_json_bytes(value)
        return self.write(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            envelope=envelope,
            files=serialized_files,
        )

    def write_jsonl(
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, Sequence[BaseModel | JsonValue]],
    ) -> ArtifactManifest:
        """Persist deterministic secret-free JSONL records as one immutable artifact.

        Args:
            artifact_id: New stable artifact identifier.
            artifact_type: Stable domain type for the manifest.
            envelope: Shared immutable provenance.
            files: Ordered structured records keyed by their JSONL file paths.

        Returns:
            The completed digest-verified manifest.

        Raises:
            ArtifactStoreError: A file is not JSONL or one record violates the secret boundary.
        """
        serialized_files: dict[str, bytes] = {}
        for relative_path, records in files.items():
            if Path(relative_path).suffix != ".jsonl":
                raise ArtifactStoreError("write_jsonl requires .jsonl data-file paths")
            serialized_records = []
            for record in records:
                try:
                    assert_secret_free(record)
                except SecretBoundaryError as exc:
                    raise ArtifactStoreError(str(exc)) from exc
                serialized_records.append(canonical_json_bytes(record))
            payload = b"\n".join(serialized_records)
            if payload:
                payload += b"\n"
            serialized_files[relative_path] = payload
        return self.write(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            envelope=envelope,
            files=serialized_files,
        )

    def read(self, artifact_id: str) -> StoredArtifact:
        """Read and fully verify one completed immutable artifact.

        Args:
            artifact_id: Stable ID of the artifact to verify and open.

        Returns:
            A parsed manifest and its verified directory.

        Raises:
            ArtifactCorruptionError: The directory, manifest, file list, or digest is invalid.
        """
        directory = self._paths.artifact_directory(artifact_id)
        if not directory.is_dir() or directory.is_symlink():
            raise ArtifactCorruptionError(f"completed artifact is missing or unsafe: {artifact_id}")
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ArtifactCorruptionError(f"artifact {artifact_id} has no safe manifest.json")
        try:
            manifest = ArtifactManifest.model_validate_json(manifest_path.read_bytes())
            assert_secret_free(manifest)
        except (OSError, ValidationError, ValueError, SecretBoundaryError) as exc:
            raise ArtifactCorruptionError(
                f"artifact {artifact_id} has an invalid manifest"
            ) from exc
        if manifest.artifact_id != artifact_id:
            raise ArtifactCorruptionError(
                f"artifact directory {artifact_id} does not match manifest ID "
                f"{manifest.artifact_id}"
            )
        expected_files = {entry.path: entry for entry in manifest.files}
        actual_files = _artifact_data_files(directory)
        if set(actual_files) != set(expected_files):
            raise ArtifactCorruptionError(
                f"artifact {artifact_id} data files do not match its manifest"
            )
        for entry in expected_files.values():
            _read_verified_artifact_file(directory, artifact_id, entry)
        return StoredArtifact(directory=directory, manifest=manifest)

    def read_bytes(self, artifact_id: str, relative_path: str) -> bytes:
        """Read a named data file after verifying the complete immutable artifact.

        Args:
            artifact_id: Stable ID of the artifact to open.
            relative_path: Safe artifact-relative data-file path.

        Returns:
            Complete verified file bytes.

        Raises:
            ArtifactCorruptionError: If the artifact or requested file is invalid.
            ValueError: If the relative path is not a safe artifact file path.
        """
        stored = self.read(artifact_id)
        validated_path = validate_artifact_file_path(relative_path).as_posix()
        expected_files = {entry.path: entry for entry in stored.manifest.files}
        entry = expected_files.get(validated_path)
        if entry is None:
            raise ArtifactCorruptionError(
                f"artifact {artifact_id} does not own data file {validated_path}"
            )
        return _read_verified_artifact_file(stored.directory, artifact_id, entry)

    def list_ids(self) -> tuple[str, ...]:
        """Return completed artifact IDs, excluding partial directories and lock files.

        Returns:
            Sorted IDs for completed artifact directories that pass local ID validation.
        """
        directory = self._paths.artifacts_directory
        if not directory.is_dir():
            return ()
        artifact_ids = []
        for candidate in directory.iterdir():
            if candidate.name.startswith(".") or not candidate.is_dir() or candidate.is_symlink():
                continue
            try:
                artifact_ids.append(validate_local_id(candidate.name, label="artifact ID"))
            except ValueError:
                continue
        return tuple(sorted(artifact_ids))


class ProjectStore:
    """Owns project configuration, the sole mutable review draft, and immutable artifacts."""

    def __init__(self, root: Path, project_id: str) -> None:
        """Create a project-local store without writing state until an explicit method is called."""
        self.paths = ProjectPaths(root=root, project_id=project_id)
        self.artifacts = ArtifactStore(self.paths)

    @property
    def model_catalog_path(self) -> Path:
        """Return this `.wmo` root's local `models.toml` path."""
        return self.paths.root / "models.toml"

    def initialize(self, config: ProjectConfig) -> None:
        """Create an immutable project configuration, or verify the existing identical one.

        Args:
            config: Configuration whose project ID matches this store.

        Raises:
            ProjectStoreError: The project ID differs or existing configuration is not identical.
        """
        if config.project_id != self.paths.project_id:
            raise ProjectStoreError("project configuration ID does not match the store project ID")
        self.paths.project_directory.mkdir(parents=True, exist_ok=True)
        with file_write_lock(self.paths.project_toml, what="project configuration"):
            if self.paths.project_toml.exists():
                try:
                    existing = load_project_config(self.paths.project_toml)
                except ValueError as exc:
                    raise ProjectStoreError(str(exc)) from exc
                if existing != config:
                    raise ProjectStoreError(
                        "project.toml already exists with different immutable config"
                    )
                return
            try:
                write_project_config(self.paths.project_toml, config)
            except ValueError as exc:
                raise ProjectStoreError(str(exc)) from exc

    def load_project(self) -> ProjectConfig:
        """Load this store's typed immutable project configuration.

        Returns:
            The parsed project configuration.

        Raises:
            ProjectStoreError: If the configuration is missing or invalid.
        """
        try:
            return load_project_config(self.paths.project_toml)
        except ValueError as exc:
            raise ProjectStoreError(str(exc)) from exc

    def write_review(self, review: BaseModel | JsonValue) -> None:
        """Lock and atomically replace the sole mutable local review draft.

        Args:
            review: Structured draft task, rubric, or calibration review state.

        Raises:
            ProjectStoreError: The draft crosses the local no-secret boundary.
        """
        with file_write_lock(self.paths.review_json, what="project review"):
            self._write_review_unlocked(review)

    def update_review(
        self,
        update: Callable[[JsonValue | None], BaseModel | JsonValue],
    ) -> JsonValue:
        """Lock the complete review read-modify-write cycle and return the stored value.

        Args:
            update: Transition that receives the latest draft while the project lock is held.

        Returns:
            The validated JSON value written atomically by this transaction.

        Raises:
            ProjectStoreError: The current draft is corrupt or the replacement crosses the
                no-secret boundary.
        """
        with file_write_lock(self.paths.review_json, what="project review"):
            current = self._read_review_unlocked()
            replacement = update(current)
            payload = canonical_json_bytes(replacement)
            try:
                stored = _JSON_VALUE_ADAPTER.validate_json(payload)
            except ValidationError as exc:
                raise ProjectStoreError("review update did not produce valid JSON") from exc
            self._write_review_unlocked(stored)
            return stored

    def read_review(self) -> JsonValue | None:
        """Load the mutable review draft, returning ``None`` before any draft exists.

        Returns:
            The parsed review JSON, or ``None`` when no draft exists.

        Raises:
            ProjectStoreError: If the draft cannot be read or is not valid JSON.
        """
        return self._read_review_unlocked()

    def _write_review_unlocked(self, review: BaseModel | JsonValue) -> None:
        """Atomically write a review value while the caller owns the project review lock."""
        try:
            assert_secret_free(review)
        except SecretBoundaryError as exc:
            raise ProjectStoreError(str(exc)) from exc
        write_bytes_atomic(self.paths.review_json, canonical_json_bytes(review))

    def _read_review_unlocked(self) -> JsonValue | None:
        """Read the review value without taking a lock for callers that already hold it."""
        if not self.paths.review_json.exists():
            return None
        try:
            return _JSON_VALUE_ADAPTER.validate_json(self.paths.review_json.read_bytes())
        except (OSError, ValidationError) as exc:
            raise ProjectStoreError("review.json is not valid JSON") from exc


def _assert_payload_secret_free(relative_path: str, payload: bytes) -> None:
    """Apply secret checks to structured JSON and UTF-8 text artifact data."""
    suffix = Path(relative_path).suffix
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return
    try:
        if suffix == ".json":
            assert_secret_free(_JSON_VALUE_ADAPTER.validate_json(payload))
        elif suffix == ".jsonl":
            for _line_number, line in enumerate(text.splitlines(), start=1):
                if line:
                    assert_secret_free(_JSON_VALUE_ADAPTER.validate_json(line))
        else:
            assert_text_secret_free(text)
    except (SecretBoundaryError, ValidationError) as exc:
        raise ArtifactStoreError(
            f"artifact data file {relative_path} violates the secret boundary"
        ) from exc


def _read_verified_artifact_file(
    directory: Path,
    artifact_id: str,
    entry: ArtifactFile,
) -> bytes:
    """Read one artifact file from a stable descriptor and verify those exact bytes."""
    try:
        payload = _read_artifact_file_snapshot(directory, entry.path)
    except OSError as exc:
        raise ArtifactCorruptionError(
            f"artifact {artifact_id} has an unsafe or unreadable data file: {entry.path}"
        ) from exc
    actual_digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != entry.size_bytes or actual_digest != entry.sha256:
        raise ArtifactCorruptionError(
            f"artifact {artifact_id} data file digest mismatch: {entry.path}"
        )
    try:
        _assert_payload_secret_free(entry.path, payload)
    except ArtifactStoreError as exc:
        raise ArtifactCorruptionError(
            f"artifact {artifact_id} data file violates the secret boundary: {entry.path}"
        ) from exc
    return payload


def _read_artifact_file_snapshot(directory: Path, relative_path: str) -> bytes:
    """Read a regular descendant without following a replaced path component."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("secure artifact reads require O_NOFOLLOW")
    directory_fd = os.open(directory, os.O_RDONLY | os.O_NOFOLLOW)
    current_fd = directory_fd
    try:
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise OSError("artifact directory is not a directory")
        parts = PurePosixPath(relative_path).parts
        for component in parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current_fd)
            try:
                if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                    raise OSError("artifact data path has a non-directory component")
            except OSError:
                os.close(next_fd)
                raise
            os.close(current_fd)
            current_fd = next_fd
        file_descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current_fd)
        try:
            if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                raise OSError("artifact data path is not a regular file")
            return _read_file_descriptor(file_descriptor)
        finally:
            os.close(file_descriptor)
    finally:
        os.close(current_fd)


def _read_file_descriptor(file_descriptor: int) -> bytes:
    """Read all available bytes from one already-open regular file descriptor."""
    chunks: list[bytes] = []
    while chunk := os.read(file_descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _artifact_data_files(directory: Path) -> tuple[str, ...]:
    """Return all regular artifact data paths, rejecting symlinks and unexpected files."""
    data_files = []
    for candidate in directory.rglob("*"):
        if candidate.is_symlink():
            raise ArtifactCorruptionError(f"artifact contains unsupported symlink: {candidate}")
        if candidate.is_file():
            relative_path = candidate.relative_to(directory).as_posix()
            if relative_path != "manifest.json":
                data_files.append(relative_path)
    return tuple(sorted(data_files))


def _write_staged_file(path: Path, payload: bytes) -> None:
    """Write and fsync one new file inside a private artifact staging directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory_strict(directory: Path) -> None:
    """Persist a staged directory before atomically exposing it to readers."""
    file_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _fsync_staging_tree(directory: Path) -> None:
    """Persist nested staged directory entries before their top-level atomic rename."""
    child_directories = [path for path in directory.rglob("*") if path.is_dir()]
    for child_directory in sorted(
        child_directories, key=lambda path: len(path.parts), reverse=True
    ):
        _fsync_directory_strict(child_directory)
    _fsync_directory_strict(directory)


def _fsync_directory_best_effort(directory: Path) -> None:
    """Attempt to persist a completed directory rename without misreporting a landed write."""
    try:
        _fsync_directory_strict(directory)
    except OSError as exc:
        logger.warning(
            "artifact rename at %s landed but directory fsync failed: %s", directory, exc
        )
