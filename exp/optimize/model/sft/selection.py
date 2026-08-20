"""Verified mutable selection of the latest immutable model-optimization config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, model_validator

from exp.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    assert_secret_free,
    canonical_json_bytes,
)
from exp.common.core.files import write_bytes_atomic
from exp.common.core.locks import file_write_lock
from exp.common.project import (
    ArtifactCorruptionError,
    ArtifactManifest,
    ProjectStore,
    ProjectStoreError,
    artifact_input,
)
from exp.runtime.router.snapshot import load_runtime_trace_snapshot

_CONFIG_ARTIFACT_TYPE = "sft-model-optimization-config"
_DATASET_ARTIFACT_TYPE = "sft-dataset"
_SNAPSHOT_ARTIFACT_TYPE = "runtime-trace-snapshot"
_LATEST_DIRECTORY = "model-optimization"
_LATEST_FILE = "latest.json"


class SFTModelOptimizationSelectionError(ValueError):
    """The selected model-optimization artifact graph is missing or inconsistent."""


class LatestSFTModelOptimization(ContractModel):
    """Mutable coordination pointer to one verified immutable runtime SFT graph."""

    schema_version: Literal[1] = 1
    project_id: ArtifactId
    config: ArtifactInput
    dataset: ArtifactInput
    runtime_snapshot: ArtifactInput
    model_alias_prefix: ArtifactId
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def _require_distinct_artifacts(self) -> LatestSFTModelOptimization:
        """Require the pointer to name three distinct artifact identities."""
        artifact_ids = {
            self.config.artifact_id,
            self.dataset.artifact_id,
            self.runtime_snapshot.artifact_id,
        }
        if len(artifact_ids) != 3:
            raise ValueError("latest model-optimization artifacts must have distinct IDs")
        return self


def latest_sft_model_optimization_path(store: ProjectStore) -> Path:
    """Return the project-local coordination path for the latest verified config.

    Args:
        store: Project whose selected model-optimization graph is addressed.

    Returns:
        Canonical mutable pointer path outside the immutable artifact directory.
    """
    return store.paths.project_directory / _LATEST_DIRECTORY / _LATEST_FILE


def load_latest_sft_model_optimization(
    store: ProjectStore,
) -> LatestSFTModelOptimization | None:
    """Load and recursively verify the latest coordination pointer when present.

    Args:
        store: Project whose pointer and immutable artifacts are verified.

    Returns:
        The verified pointer, or ``None`` before automatic runtime SFT materialization.

    Raises:
        SFTModelOptimizationSelectionError: The pointer is unsafe, malformed, secret-bearing,
            belongs to another project, or names an inconsistent artifact graph.
    """
    path = latest_sft_model_optimization_path(store)
    _require_safe_coordination_path(store, path)
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise SFTModelOptimizationSelectionError(
            f"latest model-optimization pointer is not a safe file: {path}"
        )
    try:
        pointer = LatestSFTModelOptimization.model_validate_json(path.read_bytes())
        assert_secret_free(pointer)
    except (OSError, ValueError) as exc:
        raise SFTModelOptimizationSelectionError(
            f"latest model-optimization pointer is invalid: {path}"
        ) from exc
    if pointer.project_id != store.paths.project_id:
        raise SFTModelOptimizationSelectionError(
            "latest model-optimization pointer belongs to another project"
        )
    _verify_pointer_artifacts(store, pointer)
    return pointer


def selected_sft_model_optimization_config_input(store: ProjectStore) -> ArtifactInput:
    """Return the latest verified config input or the project's bootstrap binding.

    Args:
        store: Project containing immutable model-optimization state.

    Returns:
        Exact manifest input selected for the next preflight or replay.

    Raises:
        SFTModelOptimizationSelectionError: Neither a valid latest pointer nor a bootstrap
            project binding is available.
    """
    latest = load_latest_sft_model_optimization(store)
    if latest is not None:
        return latest.config
    try:
        bootstrap = store.load_project().model_optimization_config
    except ProjectStoreError as exc:
        raise SFTModelOptimizationSelectionError(str(exc)) from exc
    if bootstrap is None:
        raise SFTModelOptimizationSelectionError(
            "project has no model-optimization settings; configure a bounded Tinker SFT "
            "template before running `exp optimize model`"
        )
    _require_artifact_input(store, bootstrap, artifact_type=_CONFIG_ARTIFACT_TYPE)
    return bootstrap


def require_selected_sft_model_optimization_config(
    store: ProjectStore, config: ArtifactInput
) -> None:
    """Require an exact config manifest to be the project's current verified selection.

    Args:
        store: Project containing the selection state.
        config: Config manifest input requested by a caller.

    Raises:
        SFTModelOptimizationSelectionError: Another config is selected or the selection is
            invalid.
    """
    if selected_sft_model_optimization_config_input(store) != config:
        raise SFTModelOptimizationSelectionError(
            "requested SFT model-optimization config is not the latest verified selection"
        )


def write_latest_sft_model_optimization(
    store: ProjectStore,
    pointer: LatestSFTModelOptimization,
    *,
    expected_current: ArtifactInput | None,
) -> LatestSFTModelOptimization:
    """Atomically advance the verified latest pointer with compare-and-swap semantics.

    Args:
        store: Project receiving the coordination update.
        pointer: New pointer whose complete immutable graph already exists.
        expected_current: Exact config input observed before creating the new graph, or ``None``
            when no model-optimization graph has ever been selected.

    Returns:
        The newly stored pointer, or a byte-equivalent pointer written by a concurrent replay.

    Raises:
        SFTModelOptimizationSelectionError: The proposed graph is invalid or another process
            selected a different config before commit.
    """
    if pointer.project_id != store.paths.project_id:
        raise SFTModelOptimizationSelectionError(
            "cannot select a model-optimization graph for another project"
        )
    _verify_pointer_artifacts(store, pointer)
    path = latest_sft_model_optimization_path(store)
    _require_safe_coordination_path(store, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_safe_coordination_path(store, path)
    with file_write_lock(path, what="latest model-optimization selection"):
        _require_safe_coordination_path(store, path)
        existing = load_latest_sft_model_optimization(store)
        if existing is not None and _same_selected_graph(existing, pointer):
            return existing
        current = existing.config if existing is not None else _bootstrap_config_input(store)
        if current != expected_current:
            raise SFTModelOptimizationSelectionError(
                "latest model-optimization selection changed before commit"
            )
        try:
            write_bytes_atomic(path, canonical_json_bytes(pointer))
        except OSError as exc:
            raise SFTModelOptimizationSelectionError(
                f"cannot persist latest model-optimization pointer: {path}"
            ) from exc
    stored = load_latest_sft_model_optimization(store)
    if stored is None or not _same_selected_graph(stored, pointer):
        raise SFTModelOptimizationSelectionError(
            "latest model-optimization pointer did not preserve the selected graph"
        )
    return stored


def _verify_pointer_artifacts(store: ProjectStore, pointer: LatestSFTModelOptimization) -> None:
    """Verify pointer manifests and their exact config-to-dataset-to-snapshot edges.

    Args:
        store: Project owning every selected immutable artifact.
        pointer: Coordination record whose complete graph is checked.

    Raises:
        SFTModelOptimizationSelectionError: An artifact, digest, edge, or project identity differs.
    """
    config_manifest = _require_artifact_input(
        store, pointer.config, artifact_type=_CONFIG_ARTIFACT_TYPE
    )
    dataset_manifest = _require_artifact_input(
        store, pointer.dataset, artifact_type=_DATASET_ARTIFACT_TYPE
    )
    _require_artifact_input(store, pointer.runtime_snapshot, artifact_type=_SNAPSHOT_ARTIFACT_TYPE)
    if config_manifest.inputs != (pointer.dataset,):
        raise SFTModelOptimizationSelectionError(
            "selected model-optimization config does not bind the selected dataset"
        )
    try:
        config_payload = json.loads(
            store.artifacts.read_bytes(pointer.config.artifact_id, "config.json")
        )
        config_alias = config_payload["model_alias"]
    except (ArtifactCorruptionError, KeyError, TypeError, ValueError) as exc:
        raise SFTModelOptimizationSelectionError(
            "selected model-optimization config has no verified model alias"
        ) from exc
    if not isinstance(config_alias, str):
        raise SFTModelOptimizationSelectionError(
            "selected model-optimization config has no verified model alias"
        )
    expected_alias = versioned_sft_model_alias(
        pointer.model_alias_prefix,
        pointer.dataset.artifact_id,
    )
    if config_alias != expected_alias:
        raise SFTModelOptimizationSelectionError(
            "latest model alias prefix does not derive the selected immutable config alias"
        )
    if pointer.runtime_snapshot not in dataset_manifest.inputs:
        raise SFTModelOptimizationSelectionError(
            "selected SFT dataset does not bind the selected runtime snapshot"
        )
    try:
        snapshot = load_runtime_trace_snapshot(
            store.artifacts, pointer.runtime_snapshot.artifact_id
        ).snapshot
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SFTModelOptimizationSelectionError(
            "selected runtime snapshot cannot be recursively verified"
        ) from exc
    if snapshot.project_id != store.paths.project_id:
        raise SFTModelOptimizationSelectionError(
            "selected runtime snapshot belongs to another project"
        )


def _bootstrap_config_input(store: ProjectStore) -> ArtifactInput | None:
    """Return and verify the optional project-bound bootstrap config input.

    Args:
        store: Project that may contain a compatibility bootstrap binding.

    Returns:
        Exact verified config manifest input, or ``None`` when absent.

    Raises:
        SFTModelOptimizationSelectionError: Project configuration or its artifact is invalid.
    """
    try:
        bootstrap = store.load_project().model_optimization_config
    except ProjectStoreError as exc:
        raise SFTModelOptimizationSelectionError(str(exc)) from exc
    if bootstrap is not None:
        _require_artifact_input(store, bootstrap, artifact_type=_CONFIG_ARTIFACT_TYPE)
    return bootstrap


def _require_artifact_input(
    store: ProjectStore, expected: ArtifactInput, *, artifact_type: str
) -> ArtifactManifest:
    """Return a verified manifest after checking its exact input and domain type.

    Args:
        store: Project containing the expected immutable artifact.
        expected: Exact artifact ID and manifest digest required by the pointer.
        artifact_type: Required domain type for the artifact manifest.

    Returns:
        Fully verified immutable manifest.

    Raises:
        SFTModelOptimizationSelectionError: The artifact is unavailable, wrong-typed, or changed.
    """
    try:
        stored = store.artifacts.read(expected.artifact_id)
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SFTModelOptimizationSelectionError(
            f"selected {artifact_type} artifact is unavailable: {expected.artifact_id}"
        ) from exc
    if stored.manifest.artifact_type != artifact_type:
        raise SFTModelOptimizationSelectionError(
            f"selected artifact {expected.artifact_id} is not {artifact_type}"
        )
    if artifact_input(stored.manifest) != expected:
        raise SFTModelOptimizationSelectionError(
            f"selected {artifact_type} manifest digest changed"
        )
    return stored.manifest


def _same_selected_graph(
    left: LatestSFTModelOptimization, right: LatestSFTModelOptimization
) -> bool:
    """Return whether two coordination records select the same immutable graph."""
    return left.model_copy(update={"updated_at": right.updated_at}) == right


def versioned_sft_model_alias(prefix: str, dataset_id: str) -> ArtifactId:
    """Derive the immutable-dataset-specific catalog alias from its selected prefix.

    Args:
        prefix: Persisted user-selected alias prefix.
        dataset_id: Immutable W12 dataset identity for this trained version.

    Returns:
        Artifact-safe local alias no longer than the catalog contract permits.

    Raises:
        SFTModelOptimizationSelectionError: The prefix cannot retain a valid leading identifier.
    """
    suffix = dataset_id.rsplit("-", maxsplit=1)[-1][:12]
    maximum_prefix_length = 128 - len(suffix) - 1
    normalized_prefix = prefix[:maximum_prefix_length].rstrip("._-")
    if not normalized_prefix:
        raise SFTModelOptimizationSelectionError(
            "configured SFT model alias cannot form a versioned runtime alias"
        )
    return f"{normalized_prefix}-{suffix}"


def _require_safe_coordination_path(store: ProjectStore, path: Path) -> None:
    """Reject symlinked coordination directories before reading, locking, or writing.

    Args:
        store: Project that owns the coordination path.
        path: Expected latest pointer below the project directory.

    Raises:
        SFTModelOptimizationSelectionError: The path escapes the project or an ancestor is a
            symlink.
    """
    project_directory = store.paths.project_directory
    try:
        path.relative_to(project_directory)
    except ValueError as exc:
        raise SFTModelOptimizationSelectionError(
            "latest model-optimization pointer escapes its project directory"
        ) from exc
    current = path.parent
    while True:
        if current.is_symlink():
            raise SFTModelOptimizationSelectionError(
                f"latest model-optimization coordination directory is not safe: {current}"
            )
        if current == project_directory:
            break
        current = current.parent
