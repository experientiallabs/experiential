"""Public W14M composition from a persisted W12 dataset through W13 Tinker SFT.

This module deliberately composes only existing immutable artifacts. It never builds an SFT
dataset, launches a teacher rollout, changes routing roles, or creates a provider client. The
caller supplies the narrow W13 backend seam, while the completed sampling handle is registered
only after a second recursive W13 verification pass.

Examples:
    Production traces are first accepted and frozen through W12, then passed here by ID::

        dataset = build_sft_dataset(
            store=store,
            production_sources=(accepted_production_source,),
            teacher_sources=(),
            spec=build_spec,
            created_at=created_at,
            code_revision=revision,
        )
        write_sft_dataset(store, dataset)
        config = create_sft_model_optimization_config(
            store,
            dataset_id=dataset.dataset.dataset_id,
            model_alias="support-sft",
            tinker_connection="tinker",
            base_model_alias="base",
            training=training_spec,
            created_at=created_at,
            code_revision=revision,
        )
        write_sft_model_optimization_config(store, config)
        result = run_sft_model_optimization(
            store,
            config.config_id,
            TinkerTrainerBackend(caller_owned_tinker_service),
            created_at=created_at,
            code_revision=revision,
        )

    Teacher simulation remains an explicit earlier application step. Its already-persisted
    ``accepted_teacher_source`` must reference accepted rollout, judgment, calibration, score-rule,
    and fidelity artifacts before W12 will accept it::

        teacher_dataset = build_sft_dataset(
            store=store,
            production_sources=(),
            teacher_sources=(accepted_teacher_source,),
            spec=build_spec,
            created_at=created_at,
            code_revision=revision,
        )
        write_sft_dataset(store, teacher_dataset)
        # Configure and call run_sft_model_optimization exactly as above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Never

from pydantic import model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    Sha256,
    canonical_json_bytes,
    sha256_json,
    stable_id,
)
from wmo.common.core.locks import file_write_lock
from wmo.common.models import (
    ModelCatalog,
    ModelCatalogError,
    ModelRecord,
    ModelSnapshot,
    NumericMeasurement,
    SFTModelProvenance,
    load_model_catalog,
    write_model_catalog,
)
from wmo.common.project import (
    ArtifactCorruptionError,
    ProjectStore,
    ProjectStoreError,
    StoredArtifact,
    artifact_input,
)
from wmo.optimize.model.sft.builder import SFTBuildError, load_verified_sft_dataset
from wmo.optimize.model.sft.contracts import PartitionedSFTExample
from wmo.optimize.model.sft.provider_resources import validate_provider_resource_id
from wmo.optimize.model.sft.selection import (
    SFTModelOptimizationSelectionError,
    require_selected_sft_model_optimization_config,
)
from wmo.optimize.model.sft.training import TinkerSFTOptimizer, train_tinker_sft
from wmo.optimize.model.sft.training_contracts import (
    TinkerSFTError,
    TinkerSFTModelArtifact,
    TinkerSFTResult,
    TinkerSFTSpec,
    TrainerBackend,
)
from wmo.optimize.model.sft.training_runtime import _expected_schedule

_CONFIG_ARTIFACT_TYPE = "sft-model-optimization-config"
_MODEL_FILE = "model.json"
_RESULT_FILE = "result.json"
_RUNS_DIRECTORY = "sft-runs"


class SFTModelOptimizationError(RuntimeError):
    """A persisted SFT model optimization configuration cannot be used safely."""


class SFTModelOptimizationPreflightError(SFTModelOptimizationError):
    """Read-only validation found a condition that must block managed SFT dispatch."""


def _required_maximum_cost_usd(training: TinkerSFTSpec) -> float:
    """Return the immutable cap when it is a finite positive amount."""
    maximum_cost_usd = training.maximum_cost_usd
    if maximum_cost_usd is None or not math.isfinite(maximum_cost_usd) or maximum_cost_usd <= 0:
        raise ValueError(
            "SFT model optimization configs require a finite maximum_cost_usd greater than zero"
        )
    return maximum_cost_usd


class SFTModelOptimizationConfig(ArtifactEnvelope):
    """Immutable W14M intent naming one verified W12 dataset and local trained-model alias."""

    config_id: ArtifactId
    dataset: ArtifactInput
    model_alias: ArtifactId
    tinker_connection: ArtifactId
    base_model_alias: ArtifactId
    base_model: ModelSnapshot
    connection_config_sha256: Sha256
    training: TinkerSFTSpec

    @model_validator(mode="after")
    def _require_exact_bindings(self) -> SFTModelOptimizationConfig:
        """Require content-addressed intent and exactly one frozen W12 input."""
        if self.inputs != (self.dataset,):
            raise ValueError("SFT model optimization configs must name exactly their W12 dataset")
        if self.training.base_model != self.base_model.model_id:
            raise ValueError("SFT training base_model must match the frozen base model snapshot")
        _required_maximum_cost_usd(self.training)
        expected_config_id = _config_id(
            dataset=self.dataset,
            model_alias=self.model_alias,
            tinker_connection=self.tinker_connection,
            base_model_alias=self.base_model_alias,
            base_model=self.base_model,
            connection_config_sha256=self.connection_config_sha256,
            training=self.training,
        )
        if self.config_id != expected_config_id:
            raise ValueError("SFT model optimization config ID is not content-addressed")
        return self


@dataclass(frozen=True)
class SFTModelOptimizationPreflight:
    """Read-only facts required before one W13 SFT dispatch.

    Attributes:
        config: Immutable config selected by the project pointer.
        output_dir: Stable local W13 append-only run directory for the config.
        completed_result: Recursively verified W13 result when a prior run already completed.
        completed_model: Matching verified W13 model artifact when a prior run already completed.
    """

    config: SFTModelOptimizationConfig
    config_input: ArtifactInput
    output_dir: Path
    completed_result: TinkerSFTResult | None
    completed_model: TinkerSFTModelArtifact | None
    planned_batch_counts: tuple[int, ...]
    conservative_schedule_cost_usd: NumericMeasurement | None

    def __post_init__(self) -> None:
        """Require terminal result and model artifacts to appear as one verified pair."""
        if (self.completed_result is None) != (self.completed_model is None):
            raise ValueError("completed W13 result and model must be present together")


@dataclass(frozen=True)
class SFTModelOptimizationResult:
    """Completed W13 provenance and its idempotent local model-catalog registration.

    Attributes:
        config_id: Immutable W14M config that selected this run.
        training_result: Recursively verified terminal W13 result.
        model: Recursively verified W13 model artifact with an opaque sampling handle.
        model_record: Local catalog record registered for the configured alias.
        output_dir: Stable append-only local W13 run directory.
        catalog_updated: Whether this call added the alias instead of confirming the same record.
    """

    config_id: ArtifactId
    training_result: TinkerSFTResult
    model: TinkerSFTModelArtifact
    model_record: ModelRecord
    output_dir: Path
    catalog_updated: bool


class _VerificationBackend:
    """Backend that proves a completed W13 run cannot dispatch any provider operation."""

    def conservative_step_cost(self, spec: TinkerSFTSpec, *, batch_example_count: int) -> None:
        """Return no estimate because completed-run verification must not plan a new step."""
        del spec, batch_example_count
        return None

    def open(self, spec: TinkerSFTSpec, resume_state_path: str | None) -> Never:
        """Fail closed if a claimed completed result would require provider dispatch."""
        del spec, resume_state_path
        raise TinkerSFTError(
            "a claimed completed Tinker SFT result required provider dispatch during verification"
        )


def create_sft_model_optimization_config(
    store: ProjectStore,
    *,
    dataset_id: ArtifactId,
    model_alias: ArtifactId,
    tinker_connection: ArtifactId,
    base_model_alias: ArtifactId,
    training: TinkerSFTSpec,
    created_at: datetime,
    code_revision: str,
) -> SFTModelOptimizationConfig:
    """Create immutable W14M intent from one already-persisted and verified W12 dataset.

    This is intentionally separate from W12 construction.  Advanced Python callers may build a
    production-trace dataset or an explicitly accepted teacher-rollout dataset first, then bind
    that already-persisted dataset here without causing a simulation or rollout.

    Args:
        store: Project store that owns the existing W12 dataset.
        dataset_id: Persisted accepted W12 SFT dataset ID.
        model_alias: New local ``models.toml`` alias for the completed sampling handle.
        tinker_connection: Existing local connection name whose provider is ``tinker``.
        base_model_alias: Existing catalog alias frozen as W13's exact base model.
        training: Frozen W13 Tinker SFT settings with a finite maximum cost cap.
        created_at: Time the immutable config is created.
        code_revision: Exact revision that created the config.

    Returns:
        Immutable config ready for ``write_sft_model_optimization_config``.

    Raises:
        SFTModelOptimizationError: The named dataset is not a verified accepted W12 artifact.
    """
    dataset_input = _verified_dataset_input(store, dataset_id)
    try:
        _required_maximum_cost_usd(training)
        resolved_base_model = _resolve_base_model(
            _load_tinker_catalog(store, tinker_connection),
            tinker_connection=tinker_connection,
            base_model_alias=base_model_alias,
            training=training,
        )
        config_id = _config_id(
            dataset=dataset_input,
            model_alias=model_alias,
            tinker_connection=tinker_connection,
            base_model_alias=base_model_alias,
            base_model=resolved_base_model.snapshot,
            connection_config_sha256=resolved_base_model.connection_config_sha256,
            training=training,
        )
        return SFTModelOptimizationConfig(
            schema_version=1,
            created_at=created_at,
            inputs=(dataset_input,),
            code_revision=code_revision,
            config_id=config_id,
            dataset=dataset_input,
            model_alias=model_alias,
            tinker_connection=tinker_connection,
            base_model_alias=base_model_alias,
            base_model=resolved_base_model.snapshot,
            connection_config_sha256=resolved_base_model.connection_config_sha256,
            training=training,
        )
    except (SFTModelOptimizationPreflightError, ValueError) as exc:
        raise SFTModelOptimizationError(f"SFT model optimization config is invalid: {exc}") from exc


def write_sft_model_optimization_config(
    store: ProjectStore,
    config: SFTModelOptimizationConfig,
    *,
    bind_project: bool = True,
) -> SFTModelOptimizationConfig:
    """Persist W14M intent and optionally establish the bootstrap project binding.

    Args:
        store: Project store that owns the config artifact and project pointer.
        config: Immutable config produced by ``create_sft_model_optimization_config``.
        bind_project: Whether to establish or verify the project's bootstrap config binding.
            Automatic journal composition persists later immutable configs with this disabled,
            then advances the separately verified latest pointer.

    Returns:
        The persisted config.

    Raises:
        SFTModelOptimizationError: The artifact conflicts, is corrupted, or project is bound
            to another immutable config.
    """
    try:
        config, config_manifest = store.artifacts.write_or_replay(
            artifact_id=config.config_id,
            artifact_type=_CONFIG_ARTIFACT_TYPE,
            envelope=config,
            envelope_path="config.json",
            envelope_type=SFTModelOptimizationConfig,
            files={"config.json": canonical_json_bytes(config)},
        )
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SFTModelOptimizationError(
            f"cannot persist SFT model optimization config {config.config_id}: {exc}"
        ) from exc
    if bind_project:
        try:
            store.bind_model_optimization_config(
                artifact_input(config_manifest), artifact_type=_CONFIG_ARTIFACT_TYPE
            )
        except ProjectStoreError as exc:
            raise SFTModelOptimizationError(
                f"cannot bind SFT model optimization config to the project: {exc}"
            ) from exc
    return config


def load_sft_model_optimization_config(
    store: ProjectStore, config_id: ArtifactId
) -> SFTModelOptimizationConfig:
    """Load one digest-verified immutable W14M config artifact.

    Args:
        store: Project store that owns the config artifact.
        config_id: Expected content-addressed config ID.

    Returns:
        Typed immutable W14M intent.

    Raises:
        SFTModelOptimizationError: The artifact is missing, malformed, or mismatches its manifest.
    """
    config, manifest = _load_sft_model_optimization_config_artifact(store, config_id)
    try:
        require_selected_sft_model_optimization_config(store, artifact_input(manifest.manifest))
    except SFTModelOptimizationSelectionError as exc:
        raise SFTModelOptimizationError(str(exc)) from exc
    return config


def sft_model_optimization_output_dir(store: ProjectStore, config_id: ArtifactId) -> Path:
    """Return the stable append-only W13 run directory for one immutable config.

    Args:
        store: Project store that owns the local coordination directory.
        config_id: Project-bound immutable W14M config that fixes the directory name.

    Returns:
        Local path that W13 initializes only when execution begins.
    """
    load_sft_model_optimization_config(store, config_id)
    return store.artifacts.project_directory / _RUNS_DIRECTORY / config_id


def preflight_sft_model_optimization(
    store: ProjectStore,
    config_id: ArtifactId,
    backend: TrainerBackend,
    *,
    code_revision: str,
) -> SFTModelOptimizationPreflight:
    """Perform all read-only checks before a caller can consent to managed SFT dispatch.

    The preflight validates the persisted W12 dataset, local Tinker connection, target alias,
    finite maximum-cost estimator, and any claimed completed W13 result. It never calls
    ``backend.open``.

    Args:
        store: Project store holding the W12 dataset and local model catalog.
        config_id: Project-bound immutable W14M config selected by the project.
        backend: Concrete Tinker adapter or deterministic injected fake.
        code_revision: Exact revision to require from an existing W13 run.

    Returns:
        Read-only facts required by ``run_sft_model_optimization``.

    Raises:
        SFTModelOptimizationPreflightError: A validation, budget, catalog, or resume condition
            makes dispatch unsafe.
    """
    config = load_sft_model_optimization_config(store, config_id)
    config_input = _bound_config_input(store, config_id)
    dataset = _verified_dataset(store, config.dataset)
    output_dir = sft_model_optimization_output_dir(store, config_id)
    completed = _verify_completed_run_if_present(
        store,
        config,
        output_dir,
        code_revision=code_revision,
    )
    catalog = _load_tinker_catalog(store, config.tinker_connection)
    _require_frozen_base_model(catalog, config)
    existing = catalog.models.get(config.model_alias)
    if existing is not None:
        if completed is None:
            raise SFTModelOptimizationPreflightError(
                f"model alias {config.model_alias!r} already exists without a recursively verified "
                "completed W13 result; refusing to risk a conflicting managed run"
            )
        expected = _model_record(config, config_input, completed.result, completed.model)
        if existing != expected:
            raise SFTModelOptimizationPreflightError(
                f"model alias {config.model_alias!r} already names a different model record"
            )
    planned_batch_counts = _planned_batch_counts(dataset.rows, config.training)
    conservative_schedule_cost: NumericMeasurement | None = None
    try:
        maximum_cost_usd = _required_maximum_cost_usd(config.training)
    except ValueError as exc:
        raise SFTModelOptimizationPreflightError(f"{exc} before Tinker dispatch") from exc
    if completed is None:
        conservative_schedule_cost = _full_schedule_cost_upper_bound(
            backend,
            config.training,
            planned_batch_counts,
        )
        if conservative_schedule_cost.value > maximum_cost_usd:
            raise SFTModelOptimizationPreflightError(
                "full Tinker SFT schedule conservative cost upper bound "
                f"${conservative_schedule_cost.value:.2f} exceeds maximum_cost_usd "
                f"${maximum_cost_usd:.2f}; no provider call was dispatched"
            )
    return SFTModelOptimizationPreflight(
        config=config,
        config_input=config_input,
        output_dir=output_dir,
        completed_result=None if completed is None else completed.result,
        completed_model=None if completed is None else completed.model,
        planned_batch_counts=planned_batch_counts,
        conservative_schedule_cost_usd=conservative_schedule_cost,
    )


def run_sft_model_optimization(
    store: ProjectStore,
    config_id: ArtifactId,
    backend: TrainerBackend,
    *,
    created_at: datetime,
    code_revision: str,
    preflight: SFTModelOptimizationPreflight | None = None,
) -> SFTModelOptimizationResult:
    """Run only W13 Tinker SFT, then register a recursively verified opaque sampling handle.

    Args:
        store: Project store holding all immutable inputs and ``models.toml``.
        config_id: Project-bound immutable W14M configuration naming the existing W12 dataset.
        backend: Concrete Tinker adapter or deterministic injected fake.
        created_at: Time W13 records if it initializes a new append-only run directory.
        code_revision: Exact revision W13 records or requires for a resumed directory.
        preflight: Optional matching pre-consent preflight. Execution always reruns the
            authoritative preflight immediately before dispatch, so this object cannot bypass a
            budget, resume, or drift gate.

    Returns:
        Completed W13 artifacts and the idempotent model-catalog registration result.

    Raises:
        SFTModelOptimizationError: Preflight, W13, verification, or catalog registration failed.
    """
    config = load_sft_model_optimization_config(store, config_id)
    config_input = _bound_config_input(store, config_id)
    if preflight is not None:
        if preflight.config != config or preflight.config_input != config_input:
            raise SFTModelOptimizationError("supplied preflight belongs to a different SFT config")
        if preflight.output_dir != sft_model_optimization_output_dir(store, config_id):
            raise SFTModelOptimizationError(
                "supplied preflight uses a different W13 output directory"
            )
    effective_preflight = preflight_sft_model_optimization(
        store,
        config_id,
        backend,
        code_revision=code_revision,
    )
    if effective_preflight.completed_result is not None:
        verified = _verify_completed_run_if_present(
            store,
            config,
            effective_preflight.output_dir,
            code_revision=code_revision,
        )
        if verified is None:
            raise SFTModelOptimizationError(
                "completed preflight no longer has a terminal W13 result"
            )
        if (
            effective_preflight.completed_model is None
            or verified.result != effective_preflight.completed_result
            or verified.model != effective_preflight.completed_model
        ):
            raise SFTModelOptimizationError(
                "completed preflight differs from the recursive W13 verification result"
            )
        return _register_verified_model(
            store,
            config,
            config_input,
            verified.result,
            verified.model,
            effective_preflight.output_dir,
        )
    try:
        _require_frozen_base_model(_load_tinker_catalog(store, config.tinker_connection), config)
        result = TinkerSFTOptimizer(backend).optimize(
            store=store,
            dataset_id=config.dataset.artifact_id,
            spec=config.training,
            output_dir=effective_preflight.output_dir,
            created_at=created_at,
            code_revision=code_revision,
        )
    except TinkerSFTError as exc:
        raise SFTModelOptimizationError(f"W13 Tinker SFT did not complete safely: {exc}") from exc
    completed = _verify_completed_run_if_present(
        store,
        config,
        effective_preflight.output_dir,
        code_revision=code_revision,
    )
    if completed is None:
        raise SFTModelOptimizationError("W13 returned without a completed result artifact")
    if completed.result != result:
        raise SFTModelOptimizationError(
            "W13 returned a result that differs from its recursive completed-run verification"
        )
    return _register_verified_model(
        store,
        config,
        config_input,
        completed.result,
        completed.model,
        effective_preflight.output_dir,
    )


@dataclass(frozen=True)
class _VerifiedCompletedRun:
    """Private pair produced only after W13 recursively validates the terminal artifact graph."""

    result: TinkerSFTResult
    model: TinkerSFTModelArtifact


def _verified_dataset_input(store: ProjectStore, dataset_id: ArtifactId) -> ArtifactInput:
    """Return the manifest input of a recursively verified accepted W12 dataset."""
    expected = ArtifactInput(artifact_id=dataset_id, sha256=_dataset_sha256(store, dataset_id))
    return _verified_dataset(store, expected).dataset_input


def _dataset_sha256(store: ProjectStore, dataset_id: ArtifactId) -> str:
    """Read one artifact manifest digest while retaining a clear W14M error boundary."""
    try:
        return artifact_input(store.artifacts.read(dataset_id).manifest).sha256
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SFTModelOptimizationError(
            f"W12 dataset {dataset_id} is not a verified artifact: {exc}"
        ) from exc


@dataclass(frozen=True)
class _VerifiedDataset:
    """Private verified W12 artifact and its manifest-derived immutable input."""

    dataset_input: ArtifactInput
    rows: tuple[PartitionedSFTExample, ...]


def _verified_dataset(store: ProjectStore, expected: ArtifactInput) -> _VerifiedDataset:
    """Recursively validate W12 and require the config to pin its exact manifest digest."""
    try:
        stored = store.artifacts.read(expected.artifact_id)
        actual = artifact_input(stored.manifest)
        if actual != expected:
            raise SFTModelOptimizationPreflightError(
                "SFT model optimization config pins a different W12 dataset manifest digest"
            )
        dataset = load_verified_sft_dataset(store, expected.artifact_id)
    except SFTModelOptimizationPreflightError:
        raise
    except (ArtifactCorruptionError, SFTBuildError, ValueError) as exc:
        raise SFTModelOptimizationPreflightError(
            f"W12 dataset {expected.artifact_id} is not safe for model optimization: {exc}"
        ) from exc
    if dataset.dataset.dataset_id != expected.artifact_id:
        raise SFTModelOptimizationPreflightError(
            "verified W12 dataset ID differs from its configured artifact input"
        )
    return _VerifiedDataset(dataset_input=actual, rows=tuple(dataset.rows))


def _require_config_matches_manifest(
    config: SFTModelOptimizationConfig, stored: StoredArtifact
) -> None:
    """Reject config JSON whose provenance does not exactly match its verified manifest."""
    manifest = stored.manifest
    if (
        config.schema_version != manifest.schema_version
        or config.created_at != manifest.created_at
        or config.inputs != manifest.inputs
        or config.code_revision != manifest.code_revision
        or config.source != manifest.source
    ):
        raise SFTModelOptimizationError(
            "SFT model optimization config provenance does not match its artifact manifest"
        )


def _load_sft_model_optimization_config_artifact(
    store: ProjectStore, config_id: ArtifactId
) -> tuple[SFTModelOptimizationConfig, StoredArtifact]:
    """Load one immutable config artifact before checking the project binding."""
    try:
        stored = store.artifacts.read(config_id)
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SFTModelOptimizationError(
            f"SFT model optimization config {config_id} is not a verified artifact: {exc}"
        ) from exc
    if stored.manifest.artifact_type != _CONFIG_ARTIFACT_TYPE:
        raise SFTModelOptimizationError(
            f"artifact {config_id} is {stored.manifest.artifact_type!r}, not an SFT model config"
        )
    try:
        config = SFTModelOptimizationConfig.model_validate_json(
            store.artifacts.read_bytes(config_id, "config.json")
        )
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SFTModelOptimizationError(
            f"SFT model optimization config {config_id} has an invalid config.json: {exc}"
        ) from exc
    _require_config_matches_manifest(config, stored)
    if config.config_id != config_id:
        raise SFTModelOptimizationError("SFT model optimization config ID differs from its path")
    return config, stored


def _bound_config_input(store: ProjectStore, config_id: ArtifactId) -> ArtifactInput:
    """Return the exact manifest input after public loading verifies the project binding."""
    load_sft_model_optimization_config(store, config_id)
    _config, stored = _load_sft_model_optimization_config_artifact(store, config_id)
    return artifact_input(stored.manifest)


@dataclass(frozen=True)
class _ResolvedBaseModel:
    """Catalog facts frozen at configuration and checked again immediately before dispatch."""

    snapshot: ModelSnapshot
    connection_config_sha256: Sha256


def _config_id(
    *,
    dataset: ArtifactInput,
    model_alias: ArtifactId,
    tinker_connection: ArtifactId,
    base_model_alias: ArtifactId,
    base_model: ModelSnapshot,
    connection_config_sha256: Sha256,
    training: TinkerSFTSpec,
) -> ArtifactId:
    """Compute the content ID for all immutable dispatch-relevant W14M bindings."""
    return stable_id(
        "sft-model-optimization-config",
        {
            "dataset": dataset.model_dump(mode="json"),
            "model_alias": model_alias,
            "tinker_connection": tinker_connection,
            "base_model_alias": base_model_alias,
            "base_model": base_model.model_dump(mode="json"),
            "connection_config_sha256": connection_config_sha256,
            "training": training.model_dump(mode="json"),
        },
    )


def _load_tinker_catalog(store: ProjectStore, connection_name: ArtifactId) -> ModelCatalog:
    """Load a local catalog and require its configured connection to be native Tinker."""
    try:
        catalog = load_model_catalog(store.model_catalog_path)
    except ModelCatalogError as exc:
        raise SFTModelOptimizationPreflightError(
            f"cannot load local models.toml before Tinker SFT: {exc}"
        ) from exc
    connection = catalog.connections.get(connection_name)
    if connection is None:
        raise SFTModelOptimizationPreflightError(
            f"SFT model optimization config names unknown connection {connection_name!r}"
        )
    if connection.provider != "tinker":
        raise SFTModelOptimizationPreflightError(
            "SFT model optimization config requires a tinker connection, not "
            f"{connection.provider!r}"
        )
    return catalog


def _resolve_base_model(
    catalog: ModelCatalog,
    *,
    tinker_connection: ArtifactId,
    base_model_alias: ArtifactId,
    training: TinkerSFTSpec,
) -> _ResolvedBaseModel:
    """Resolve exactly the catalog target W13 will train from without reading credentials."""
    connection = catalog.connections[tinker_connection]
    record = catalog.models.get(base_model_alias)
    if record is None:
        raise SFTModelOptimizationPreflightError(
            f"SFT model optimization config names unknown base model alias {base_model_alias!r}"
        )
    if record.connection != tinker_connection:
        raise SFTModelOptimizationPreflightError(
            f"base model alias {base_model_alias!r} does not use configured Tinker connection "
            f"{tinker_connection!r}"
        )
    if record.model != training.base_model:
        raise SFTModelOptimizationPreflightError(
            f"base model alias {base_model_alias!r} resolves to {record.model!r}, not frozen "
            f"W13 base_model {training.base_model!r}"
        )
    capabilities = (
        record.capabilities.model_dump(mode="json") if record.capabilities is not None else None
    )
    return _ResolvedBaseModel(
        snapshot=ModelSnapshot(
            provider=connection.provider,
            model_id=record.model,
            revision=record.revision,
            capabilities_sha256=sha256_json(capabilities),
            connection_sha256=connection.identity_sha256(),
        ),
        connection_config_sha256=sha256_json(
            {
                "provider": connection.provider,
                "base_url": connection.base_url,
                "api_key_env": connection.api_key_env,
            }
        ),
    )


def _require_frozen_base_model(catalog: ModelCatalog, config: SFTModelOptimizationConfig) -> None:
    """Fail closed when any base model, capability, endpoint, or key reference drifted."""
    current = _resolve_base_model(
        catalog,
        tinker_connection=config.tinker_connection,
        base_model_alias=config.base_model_alias,
        training=config.training,
    )
    if current.snapshot != config.base_model:
        raise SFTModelOptimizationPreflightError(
            "configured Tinker base model snapshot drifted after immutable config creation"
        )
    if current.connection_config_sha256 != config.connection_config_sha256:
        raise SFTModelOptimizationPreflightError(
            "configured Tinker connection metadata drifted after immutable config creation"
        )


def _planned_batch_counts(
    rows: tuple[PartitionedSFTExample, ...], spec: TinkerSFTSpec
) -> tuple[int, ...]:
    """Derive W13's exact frozen train schedule before asking for any cost estimate."""
    try:
        schedule = _expected_schedule(
            tuple(row.example for row in rows if row.partition == "train"), spec
        )
    except TinkerSFTError as exc:
        raise SFTModelOptimizationPreflightError(
            f"cannot derive the exact W13 training schedule: {exc}"
        ) from exc
    return tuple(len(batch.example_ids) for batch in schedule)


def _full_schedule_cost_upper_bound(
    backend: TrainerBackend, spec: TinkerSFTSpec, batch_counts: tuple[int, ...]
) -> NumericMeasurement:
    """Require a finite conservative estimate for every exact W13 batch before dispatch."""
    total = 0.0
    for batch_number, batch_example_count in enumerate(batch_counts, start=1):
        try:
            estimate = backend.conservative_step_cost(spec, batch_example_count=batch_example_count)
        except Exception as exc:
            raise SFTModelOptimizationPreflightError(
                f"cannot obtain conservative Tinker SFT cost for planned batch {batch_number}"
            ) from exc
        if estimate is None:
            raise SFTModelOptimizationPreflightError(
                "Tinker has no supported conservative cost estimate for every planned batch; "
                "no provider call was dispatched"
            )
        if estimate.value < 0:
            raise SFTModelOptimizationPreflightError(
                f"conservative Tinker SFT cost for planned batch {batch_number} is negative"
            )
        total += estimate.value
    return NumericMeasurement(value=total, provenance="estimated")


def _verify_completed_run_if_present(
    store: ProjectStore,
    config: SFTModelOptimizationConfig,
    output_dir: Path,
    *,
    code_revision: str,
) -> _VerifiedCompletedRun | None:
    """Recursively verify a claimed W13 terminal result without allowing provider dispatch."""
    result_path = output_dir / _RESULT_FILE
    model_path = output_dir / _MODEL_FILE
    if not result_path.exists() and not model_path.exists():
        return None
    try:
        result = train_tinker_sft(
            store,
            config.dataset.artifact_id,
            config.training,
            output_dir,
            backend=_VerificationBackend(),
            created_at=datetime.now(UTC),
            code_revision=code_revision,
        )
        model = TinkerSFTModelArtifact.model_validate_json((output_dir / _MODEL_FILE).read_bytes())
        validate_provider_resource_id(model.sampling_handle, label="sampling handle")
    except (OSError, TinkerSFTError, ValueError) as exc:
        raise SFTModelOptimizationPreflightError(
            "claimed completed W13 result is not recursively verified; "
            "refusing model registration: "
            f"{exc}"
        ) from exc
    if result.model_id != model.model_id or result.model_sha256 != sha256_json(model):
        raise SFTModelOptimizationPreflightError(
            "claimed completed W13 result does not match its verified model artifact"
        )
    return _VerifiedCompletedRun(result=result, model=model)


def _model_record(
    config: SFTModelOptimizationConfig,
    config_input: ArtifactInput,
    result: TinkerSFTResult,
    model: TinkerSFTModelArtifact,
) -> ModelRecord:
    """Build the only catalog record permitted for one verified opaque W13 handle."""
    try:
        return ModelRecord(
            connection=config.tinker_connection,
            model=model.sampling_handle,
            sft_provenance=SFTModelProvenance(
                source_dataset=config.dataset,
                optimization_config=config_input,
                training_spec_sha256=sha256_json(config.training),
                run_id=result.run_id,
                model_id=model.model_id,
                model_sha256=sha256_json(model),
                result_id=result.result_id,
                result_sha256=sha256_json(result),
                base_model=config.base_model,
                connection_config_sha256=config.connection_config_sha256,
                sampling_handle_sha256=sha256_json({"sampling_handle": model.sampling_handle}),
            ),
        )
    except ValueError as exc:
        raise SFTModelOptimizationError(
            "verified W13 sampling handle cannot be represented in the local model catalog"
        ) from exc


def _register_verified_model(
    store: ProjectStore,
    config: SFTModelOptimizationConfig,
    config_input: ArtifactInput,
    result: TinkerSFTResult,
    model: TinkerSFTModelArtifact,
    output_dir: Path,
) -> SFTModelOptimizationResult:
    """Idempotently register only a completed recursively verified model artifact.

    The lock covers the catalog read-modify-write cycle, but never the managed training call.
    A racing incompatible alias therefore fails closed instead of replacing another record.
    """
    try:
        validate_provider_resource_id(model.sampling_handle, label="sampling handle")
    except TinkerSFTError as exc:
        raise SFTModelOptimizationError(
            "refusing to register an unsafe Tinker sampling handle"
        ) from exc
    if result.model_id != model.model_id or result.model_sha256 != sha256_json(model):
        raise SFTModelOptimizationError(
            "refusing to register a W13 result that does not match its model artifact"
        )
    desired = _model_record(config, config_input, result, model)
    try:
        with file_write_lock(store.model_catalog_path, what="the local model catalog"):
            catalog = _load_tinker_catalog(store, config.tinker_connection)
            _require_frozen_base_model(catalog, config)
            existing = catalog.models.get(config.model_alias)
            if existing is not None:
                if existing != desired:
                    raise SFTModelOptimizationError(
                        f"model alias {config.model_alias!r} already names a different model record"
                    )
                updated = False
            else:
                models = dict(catalog.models)
                models[config.model_alias] = desired
                updated_catalog = catalog.model_copy(update={"models": models})
                write_model_catalog(store.model_catalog_path, updated_catalog)
                updated = True
    except ModelCatalogError as exc:
        raise SFTModelOptimizationError(f"cannot register verified Tinker model: {exc}") from exc
    return SFTModelOptimizationResult(
        config_id=config.config_id,
        training_result=result,
        model=model,
        model_record=desired,
        output_dir=output_dir,
        catalog_updated=updated,
    )
