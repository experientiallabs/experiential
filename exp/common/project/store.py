"""Transactional project configuration and immutable digest-verified artifact storage."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    SecretBoundaryError,
    assert_secret_free,
    assert_text_secret_free,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    envelope_matches_manifest,
    validate_artifact_file_path,
)
from exp.common.core.locks import file_write_lock
from exp.common.project.artifact_sqlite import (
    StoredArtifact,
    artifact_exists,
    artifact_ids,
    read_artifact,
    write_artifact,
)
from exp.common.project.config_sqlite import config_exists, require_current_layout
from exp.common.project.database import project_connection
from exp.common.project.errors import (
    ArtifactAlreadyExistsError,
    ArtifactCorruptionError,
    ArtifactStoreError,
    ProjectStoreError,
)
from exp.common.project.hosted_state import HostedProjectStoreMixin
from exp.common.project.manifests import ArtifactManifest, artifact_input, file_digest
from exp.common.project.paths import ProjectPaths, validate_local_id
from exp.common.project.project import (
    ProjectBuildArtifacts,
    ProjectConfig,
    ProjectProviderFreeStage,
    load_project_config,
    require_durable_source_id,
    write_project_config,
)
from exp.common.project.records import ProjectRecords

_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)
_ACTIVE_COMPLETED_BUILD_COORDINATION: ContextVar[str | None] = ContextVar(
    "active_completed_build_coordination",
    default=None,
)


class ArtifactStore:
    """Writes and reads immutable, digest-verified artifact records for one project."""

    def __init__(self, paths: ProjectPaths) -> None:
        """Bind one project namespace without creating any metadata."""
        require_current_layout(paths)
        self._paths = paths

    @property
    def project_directory(self) -> Path:
        """Return the project-owned local state directory for durable coordination records.

        Artifact metadata lives in SQLite and large payloads remain in ``blobs``.
        Callers that need a mutable, local-only
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
        write_artifact(self._paths, manifest, dict(normalized_files))
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
            for record in records:
                try:
                    assert_secret_free(record)
                except SecretBoundaryError as exc:
                    raise ArtifactStoreError(str(exc)) from exc
            serialized_files[relative_path] = canonical_jsonl_bytes(records)
        return self.write(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            envelope=envelope,
            files=serialized_files,
        )

    def write_or_verify_exact(
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, bytes],
    ) -> ArtifactManifest:
        """Write a deterministic artifact or verify an existing exact replay.

        Args:
            artifact_id: Stable content-derived artifact identity.
            artifact_type: Expected immutable artifact type.
            envelope: Exact expected artifact provenance envelope.
            files: Exact expected serialized payloads.

        Returns:
            Newly written or byte-for-byte verified artifact manifest.

        Raises:
            ValueError: An existing artifact has different manifest fields or payload bytes.
            ArtifactStoreError: The existing artifact is corrupt or the write fails.
        """
        if not self.exists(artifact_id):
            # The existence probe only skips write() work that is doomed to AlreadyExists on a
            # replay; losing the creation race still lands in the verify branch below.
            try:
                return self.write(
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    envelope=envelope,
                    files=files,
                )
            except ArtifactAlreadyExistsError:
                pass
        manifest = self.read(artifact_id).manifest
        if manifest.artifact_type != artifact_type or not envelope_matches_manifest(
            envelope, manifest
        ):
            raise ValueError(f"existing artifact {artifact_id} manifest differs from exact replay")
        self._verify_replay_files(artifact_id, manifest, files)
        return manifest

    def write_or_replay[EnvelopeT: ArtifactEnvelope](
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: EnvelopeT,
        envelope_path: str,
        envelope_type: type[EnvelopeT],
        files: Mapping[str, bytes],
    ) -> tuple[EnvelopeT, ArtifactManifest]:
        """Write a deterministic artifact or adopt its exact existing replay.

        An existing artifact matches when it repeats the expected type, file set, and payload
        bytes, and its persisted canonical envelope equals the expected envelope after adopting
        the original materialization time.

        Args:
            artifact_id: Stable content-derived artifact identity.
            artifact_type: Expected immutable artifact type.
            envelope: Exact expected artifact provenance envelope.
            envelope_path: Data-file path holding the canonical serialized envelope.
            envelope_type: Concrete envelope model used to parse the stored payload.
            files: Exact expected serialized payloads, including the envelope file.

        Returns:
            The newly written or existing verified envelope and its artifact manifest.

        Raises:
            ValueError: An existing artifact differs from the expected exact replay.
            ArtifactStoreError: The existing artifact is corrupt or the write fails.
        """
        if not self.exists(artifact_id):
            # The existence probe only skips write() work that is doomed to AlreadyExists on a
            # replay; losing the creation race still lands in the verify branch below.
            try:
                manifest = self.write(
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    envelope=envelope,
                    files=files,
                )
            except ArtifactAlreadyExistsError:
                pass
            else:
                return envelope, manifest
        stored = self.read(artifact_id)
        manifest = stored.manifest
        if manifest.artifact_type != artifact_type:
            raise ValueError(f"existing artifact {artifact_id} manifest differs from exact replay")
        stored_envelope = self._stored_file_bytes(stored, envelope_path)
        try:
            existing = envelope_type.model_validate_json(stored_envelope)
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"existing artifact {artifact_id} has an invalid envelope") from exc
        adopted = envelope.model_copy(update={"created_at": existing.created_at})
        if stored_envelope != canonical_json_bytes(adopted):
            raise ValueError(f"existing artifact {artifact_id} envelope differs from exact replay")
        if not envelope_matches_manifest(existing, manifest):
            raise ValueError(f"existing artifact {artifact_id} manifest differs from exact replay")
        self._verify_replay_files(artifact_id, manifest, files, skip=envelope_path)
        return existing, manifest

    def _verify_replay_files(
        self,
        artifact_id: str,
        manifest: ArtifactManifest,
        files: Mapping[str, bytes],
        *,
        skip: str | None = None,
    ) -> None:
        """Require an existing artifact to repeat the expected file set and payload bytes.

        The caller has already fully verified the artifact with `read`, which proves every stored
        payload matches its manifest digest, so digesting the EXPECTED payloads against the same
        manifest entries proves byte equality without re-reading any stored file.

        Args:
            artifact_id: Existing verified artifact identity.
            manifest: Parsed manifest of the existing artifact, already verified by `read`.
            files: Exact expected serialized payloads.
            skip: Optional path whose bytes the caller compares separately.

        Raises:
            ValueError: The stored file set or any payload differs from the expected replay.
        """
        if tuple(sorted(files)) != tuple(item.path for item in manifest.files):
            raise ValueError(f"existing artifact {artifact_id} file set differs from exact replay")
        entries = {item.path: item for item in manifest.files}
        for relative_path, expected_payload in files.items():
            if relative_path == skip:
                continue
            if file_digest(relative_path, expected_payload) != entries[relative_path]:
                raise ValueError(
                    f"existing artifact {artifact_id} payload differs from exact replay"
                )

    @staticmethod
    def _stored_file_bytes(stored: StoredArtifact, relative_path: str) -> bytes:
        """Read one data file from an artifact the caller already fully verified with `read`.

        Args:
            stored: Verified artifact returned by `read`.
            relative_path: Safe artifact-relative data-file path.

        Returns:
            Complete verified file bytes.

        Raises:
            ArtifactCorruptionError: The artifact does not own the file or its bytes changed.
            ValueError: The relative path is not a safe artifact file path.
        """
        validated_path = validate_artifact_file_path(relative_path).as_posix()
        entry = next((item for item in stored.manifest.files if item.path == validated_path), None)
        if entry is None:
            raise ArtifactCorruptionError(
                f"artifact {stored.manifest.artifact_id} does not own data file {validated_path}"
            )
        return stored.payloads[validated_path]

    def read(self, artifact_id: str) -> StoredArtifact:
        """Read and fully verify one completed immutable artifact.

        Args:
            artifact_id: Stable ID of the artifact to verify and open.

        Returns:
            A parsed manifest and its verified directory.

        Raises:
            ArtifactCorruptionError: The directory, manifest, file list, or digest is invalid.
        """
        validate_local_id(artifact_id, label="artifact ID")
        stored = read_artifact(self._paths, artifact_id)
        assert_secret_free(stored.manifest)
        for path, payload in stored.payloads.items():
            _assert_payload_secret_free(path, payload)
        return stored

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
        return stored.payloads[validated_path]

    def list_ids(self, *, artifact_type: str | None = None) -> tuple[str, ...]:
        """Return committed artifact IDs, optionally filtered by indexed domain type."""
        return artifact_ids(self._paths, artifact_type=artifact_type)

    def exists(self, artifact_id: str) -> bool:
        """Check a committed identity without consulting loose files or directories."""
        validate_local_id(artifact_id, label="artifact ID")
        return artifact_exists(self._paths, artifact_id)


class ProjectStore(HostedProjectStoreMixin):
    """Own project configuration, immutable pointer bindings, review draft, and artifacts."""

    def __init__(self, root: Path, project_id: str, *, config_sha256: str | None = None) -> None:
        """Create a project-local store without writing state until an explicit method is called."""
        self.paths = ProjectPaths(root=root, project_id=project_id)
        self._config_sha256 = config_sha256
        self.artifacts = ArtifactStore(self.paths)
        self.records = ProjectRecords(root, project_id, "project")

    def snapshot(self, sha256: str) -> ProjectStore:
        """Bind exact configuration for reproducible execution while sharing immutable evidence."""
        selected = ProjectStore(self.paths.root, self.paths.project_id, config_sha256=sha256)
        selected.load_project()
        return selected

    def _require_mutable_config(self) -> None:
        """Refuse configuration-head changes from an execution snapshot."""
        if self._config_sha256 is not None:
            raise ProjectStoreError("frozen project snapshots cannot change configuration")

    def exists(self) -> bool:
        """Return whether this project has committed configuration."""
        return config_exists(self.paths)

    @property
    def model_catalog_path(self) -> Path:
        """Return this `.exp` root's local `models.toml` path."""
        return self.paths.root / "models.toml"

    def initialize(self, config: ProjectConfig) -> None:
        """Create the initial project configuration, or verify the existing identical one.

        Args:
            config: Configuration whose project ID matches this store.

        Raises:
            ProjectStoreError: The project ID differs or existing configuration is not identical.
        """
        if config.project_id != self.paths.project_id:
            raise ProjectStoreError("project configuration ID does not match the store project ID")
        self._require_mutable_config()
        require_current_layout(self.paths)
        self.paths.project_directory.mkdir(parents=True, exist_ok=True)
        with project_connection(self.paths.root, write=True):
            if self.exists():
                try:
                    existing = load_project_config(self.paths)
                except ValueError as exc:
                    raise ProjectStoreError(str(exc)) from exc
                if existing != config:
                    raise ProjectStoreError(
                        "project already exists with different immutable config"
                    )
                return
            try:
                write_project_config(self.paths, config)
            except ValueError as exc:
                raise ProjectStoreError(str(exc)) from exc

    def load_project(self) -> ProjectConfig:
        """Load this store's typed project configuration.

        Returns:
            The parsed project configuration.

        Raises:
            ProjectStoreError: If the configuration is missing or invalid.
        """
        try:
            return load_project_config(self.paths, sha256=self._config_sha256)
        except ValueError as exc:
            raise ProjectStoreError(str(exc)) from exc

    def bind_provider_free_stage(self, stage: ProjectProviderFreeStage) -> ProjectConfig:
        """Atomically select one verified provider-free graph or accept its exact replay.

        Args:
            stage: Exact trace and task manifest pointers to select.

        Returns:
            Existing or newly updated Project configuration naming the stage.

        Raises:
            ProjectStoreError: The graph is invalid, settings are absent, or another stage won.
        """
        self._require_mutable_config()
        with project_connection(self.paths.root, write=True):
            try:
                self._verify_provider_free_stage(stage)
                existing = load_project_config(self.paths)
                if existing.trace_preparation is None:
                    raise ValueError("Project has no provider-free trace preparation settings")
                if existing.provider_free_stage == stage:
                    return existing
                if existing.provider_free_stage is not None:
                    raise ValueError("project already selects a different provider-free stage")
                updated = existing.model_copy(update={"provider_free_stage": stage})
                write_project_config(self.paths, updated)
            except (ArtifactStoreError, ValueError) as exc:
                raise ProjectStoreError(f"cannot bind provider-free stage: {exc}") from exc
            return updated

    def _verify_provider_free_stage(self, stage: ProjectProviderFreeStage) -> None:
        """Verify exact manifests, derived provenance, and lineage behind one stage pointer."""
        trace = self.artifacts.read(stage.trace_dataset.artifact_id).manifest
        task = self.artifacts.read(stage.task_set.artifact_id).manifest
        if trace.artifact_type != "trace-dataset":
            raise ValueError("provider-free trace pointer does not name a trace-dataset artifact")
        if task.artifact_type != "task-set":
            raise ValueError("provider-free task pointer does not name a task-set artifact")
        if artifact_input(trace) != stage.trace_dataset:
            raise ValueError("provider-free trace manifest digest changed")
        if artifact_input(task) != stage.task_set:
            raise ValueError("provider-free task manifest digest changed")
        if trace.inputs:
            raise ValueError("provider-free trace dataset must not have artifact inputs")
        if task.inputs != (stage.trace_dataset,):
            raise ValueError("provider-free task set does not bind the selected trace dataset")
        source = trace.source
        if source is None or source.sha256 is None:
            raise ValueError("provider-free trace manifest source requires a byte digest")
        require_durable_source_id(source.source_id)
        if task.code_revision != trace.code_revision:
            raise ValueError("provider-free trace and task manifest revisions differ")

    def bind_completed_build(
        self, build: ProjectBuildArtifacts, *, trace_import_id: ArtifactId | None = None
    ) -> ProjectConfig:
        """Atomically select a fully verified immutable build for future project workflows.

        Args:
            build: Exact trace, task, RAG, and world-model artifact manifest references.
            trace_import_id: Optional exact stored corpus backing the selected graph.

        Returns:
            Updated project configuration naming the completed build.

        Raises:
            ProjectStoreError: A pointer is stale, has the wrong type, or configuration is invalid.
        """
        expected_types = {
            "trace_dataset": "trace-dataset",
            "task_set": "task-set",
            "serving_rag": "trace-rag-index",
            "fit_rag": "trace-rag-index",
            "world_model": "grounded-world-model",
        }
        self._require_mutable_config()
        with project_connection(self.paths.root, write=True):
            try:
                manifests: dict[str, ArtifactManifest] = {}
                for field_name, artifact_type in expected_types.items():
                    pointer = getattr(build, field_name)
                    stored = self.artifacts.read(pointer.artifact_id)
                    if stored.manifest.artifact_type != artifact_type:
                        raise ValueError(
                            f"{field_name} artifact is {stored.manifest.artifact_type!r}, "
                            f"not {artifact_type!r}"
                        )
                    if artifact_input(stored.manifest) != pointer:
                        raise ValueError(f"{field_name} artifact manifest digest changed")
                    manifests[field_name] = stored.manifest
                expected_inputs = {
                    "task_set": (build.trace_dataset,),
                    "serving_rag": (build.trace_dataset,),
                    "fit_rag": (build.trace_dataset,),
                    "world_model": (build.serving_rag,),
                }
                for field_name, inputs in expected_inputs.items():
                    if manifests[field_name].inputs != inputs:
                        raise ValueError(
                            f"{field_name} artifact does not bind the completed build graph"
                        )
                existing = load_project_config(self.paths)
                updated = existing.model_copy(
                    update={"build": build, "trace_import_id": trace_import_id}
                )
                write_project_config(self.paths, updated)
            except (ArtifactCorruptionError, ValueError) as exc:
                raise ProjectStoreError(f"cannot bind completed build: {exc}") from exc
            return updated

    def bind_model_optimization_config(
        self, config: ArtifactInput, *, artifact_type: str
    ) -> ProjectConfig:
        """Atomically bind one verified immutable SFT config artifact to this project.

        The pointer is deliberately write-once.  The referenced config remains an immutable
        artifact, while this narrow project-level binding makes ``exp optimize model <project>``
        unambiguous after the W12 dataset has been persisted.

        Args:
            config: Exact manifest input for the persisted local config artifact.
            artifact_type: Expected domain-specific immutable artifact type.

        Returns:
            The existing or newly bound project configuration.

        Raises:
            ProjectStoreError: The project is missing, invalid, or already names another config.
        """
        try:
            validated_type = validate_local_id(
                artifact_type, label="model optimization artifact type"
            )
            self._verify_model_optimization_config_input(config, artifact_type=validated_type)
        except (ArtifactCorruptionError, ValueError) as exc:
            raise ProjectStoreError(
                f"model optimization config binding is not a verified immutable artifact: {exc}"
            ) from exc
        self._require_mutable_config()
        with project_connection(self.paths.root, write=True):
            try:
                self._verify_model_optimization_config_input(config, artifact_type=validated_type)
            except (ArtifactCorruptionError, ValueError) as exc:
                raise ProjectStoreError(
                    f"model optimization config binding changed before commit: {exc}"
                ) from exc
            try:
                existing = load_project_config(self.paths)
            except ValueError as exc:
                raise ProjectStoreError(str(exc)) from exc
            current_config = existing.model_optimization_config
            if current_config == config:
                return existing
            if current_config is not None:
                raise ProjectStoreError(
                    "project already names a different immutable model optimization config"
                )
            updated = existing.model_copy(update={"model_optimization_config": config})
            try:
                write_project_config(self.paths, updated)
            except ValueError as exc:
                raise ProjectStoreError(str(exc)) from exc
            return updated

    def _verify_model_optimization_config_input(
        self, config: ArtifactInput, *, artifact_type: str
    ) -> None:
        """Require a project pointer to name this exact completed artifact manifest."""
        stored = self.artifacts.read(config.artifact_id)
        if stored.manifest.artifact_type != artifact_type:
            raise ValueError(
                f"artifact {config.artifact_id} is {stored.manifest.artifact_type!r}, "
                f"not {artifact_type!r}"
            )
        if artifact_input(stored.manifest) != config:
            raise ValueError("artifact manifest digest differs from the requested project binding")

    def write_review(self, review: BaseModel | JsonValue) -> None:
        """Lock and atomically replace the sole mutable local review draft.

        Args:
            review: Structured draft task, rubric, or calibration review state.

        Raises:
            ProjectStoreError: The draft crosses the local no-secret boundary.
        """
        with self.records.transaction():
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
        with self.records.transaction():
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
        self.records.write("review", canonical_json_bytes(review))

    def _read_review_unlocked(self) -> JsonValue | None:
        """Read the review value without taking a lock for callers that already hold it."""
        payload = self.records.read("review")
        if payload is None:
            return None
        try:
            return _JSON_VALUE_ADAPTER.validate_json(payload)
        except (OSError, ValidationError) as exc:
            raise ProjectStoreError("project review is not valid JSON") from exc


@contextmanager
def coordinate_completed_build_selection(
    store: ProjectStore,
    *,
    task_set_id: ArtifactId | None = None,
) -> Iterator[None]:
    """Serialize build selection with mutable review writers for the selected task set.

    The coordination lock is reentrant only for the same project inside one execution context.
    This lets a workflow coordinate common review services without reacquiring
    the non-reentrant filesystem lock. A task-set-bound writer verifies the current selection
    after acquiring the lock, so a service opened for an older build cannot restore its namespace
    after a replacement commits.

    Args:
        store: Project whose completed build and review namespaces must change atomically.
        task_set_id: Optional immutable task-set identity required by the review mutation.

    Yields:
        None while completed-build replacement is excluded.

    Raises:
        ProjectStoreError: Nested coordination targets another project, or the selected build uses
            a different task set from the review writer.
    """
    project_key = str(store.paths.project_directory.absolute())
    active = _ACTIVE_COMPLETED_BUILD_COORDINATION.get()
    if active is not None:
        if active != project_key:
            raise ProjectStoreError(
                "nested completed-build coordination cannot target another project"
            )
        _require_selected_task_set(store, task_set_id)
        yield
        return
    coordination_path = store.paths.project_directory / "completed-build-selection"
    with file_write_lock(coordination_path, what="completed build and review state"):
        token = _ACTIVE_COMPLETED_BUILD_COORDINATION.set(project_key)
        try:
            _require_selected_task_set(store, task_set_id)
            yield
        finally:
            _ACTIVE_COMPLETED_BUILD_COORDINATION.reset(token)


def _require_selected_task_set(
    store: ProjectStore,
    task_set_id: ArtifactId | None,
) -> None:
    """Reject a build-scoped writer whose task set is no longer selected.

    Args:
        store: Project containing the optional completed-build selection.
        task_set_id: Task-set identity required by the writer, or None for the selector itself.

    Raises:
        ProjectStoreError: A completed build selects a different task set.
    """
    if task_set_id is None:
        return
    completed = store.load_project().build
    if completed is not None and completed.task_set.artifact_id != task_set_id:
        raise ProjectStoreError(
            "review evidence differs from the selected completed build; reopen the review from "
            "the current project build"
        )


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
