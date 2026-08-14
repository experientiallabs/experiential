"""Automatic immutable runtime-dataset preparation for managed model optimization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from wmo.common.core.artifacts import ArtifactId, ArtifactInput, ContractModel, canonical_json_bytes
from wmo.common.models import ModelCatalogError, load_model_catalog
from wmo.common.project import (
    ArtifactCorruptionError,
    ProjectStore,
    ProjectStoreError,
    artifact_input,
)
from wmo.optimize.model.sft.builder import (
    SFTBuildError,
    build_sft_dataset,
    load_verified_sft_dataset,
    write_sft_dataset,
)
from wmo.optimize.model.sft.composition import (
    SFTModelOptimizationConfig,
    SFTModelOptimizationError,
    create_sft_model_optimization_config,
    load_sft_model_optimization_config,
    write_sft_model_optimization_config,
)
from wmo.optimize.model.sft.contracts import RuntimeSFTSource, SFTBuildSpec, SFTDatasetArtifact
from wmo.optimize.model.sft.selection import (
    LatestSFTModelOptimization,
    SFTModelOptimizationSelectionError,
    load_latest_sft_model_optimization,
    versioned_sft_model_alias,
    write_latest_sft_model_optimization,
)
from wmo.optimize.model.sft.training_contracts import TinkerSFTSpec
from wmo.runtime.router.journal import (
    RuntimeCompletedEvent,
    RuntimeInteractionJournal,
    RuntimeJournalError,
    RuntimeJournalEvent,
)
from wmo.runtime.router.snapshot import (
    RuntimeTraceSnapshot,
    RuntimeTraceSnapshotError,
    load_runtime_trace_snapshot,
    seal_runtime_trace_snapshot,
)


class AutomaticSFTPreparationError(ValueError):
    """Runtime evidence cannot safely select an immutable SFT dataset and config."""


class InitialSFTModelOptimizationSettings(ContractModel):
    """Confirmed first-run Tinker selections and bounded training schedule."""

    model_alias_prefix: ArtifactId
    tinker_connection: ArtifactId
    base_model_alias: ArtifactId
    training: TinkerSFTSpec


@dataclass(frozen=True)
class AutomaticSFTPreparation:
    """Verified artifacts selected before consent or trainer-backend composition.

    Attributes:
        snapshot: Exact sealed runtime journal prefix used as the SFT source.
        dataset: Recursively verified immutable W12 runtime dataset.
        config: Latest immutable bounded Tinker model-optimization config.
        created: Whether this call advanced the selected immutable graph.
    """

    snapshot: RuntimeTraceSnapshot
    dataset: SFTDatasetArtifact
    config: SFTModelOptimizationConfig
    created: bool


def prepare_runtime_sft_model_optimization(
    store: ProjectStore,
    *,
    created_at: datetime,
    code_revision: str,
    initial_settings: InitialSFTModelOptimizationSettings | None = None,
) -> AutomaticSFTPreparation:
    """Seal completed routed interactions and select their deterministic W12 and config.

    A latest config supplies previously confirmed Tinker settings. The first run instead requires
    explicit confirmed settings from the caller. Runtime journal content supplies every example.
    The function performs no trainer-backend construction, consent, credential read, or provider
    call.

    Args:
        store: Initialized project containing the runtime journal and local model catalog.
        created_at: Time recorded by newly materialized immutable artifacts and latest pointer.
        code_revision: Exact revision recorded by newly materialized artifacts.
        initial_settings: Confirmed settings required only when no prior config is selected.

    Returns:
        The reused or newly selected snapshot, W12 dataset, and bounded optimization config.

    Raises:
        AutomaticSFTPreparationError: The journal has no completed interaction, diverges from the
            selected prefix, or any source, dataset, config, catalog, or pointer cannot be
            verified without dispatch.
    """
    journal = RuntimeInteractionJournal(store.paths)
    events = _completed_runtime_events(journal)
    try:
        latest = load_latest_sft_model_optimization(store)
        if latest is not None:
            selected_input = latest.config
            selected_config = load_sft_model_optimization_config(store, selected_input.artifact_id)
            selected_dataset = load_verified_sft_dataset(store, selected_config.dataset.artifact_id)
            reused = _reuse_or_require_append(
                store,
                events=events,
                latest=latest,
                config=selected_config,
                dataset=selected_dataset,
            )
            if reused is not None:
                return reused
            settings = InitialSFTModelOptimizationSettings(
                model_alias_prefix=latest.model_alias_prefix,
                tinker_connection=selected_config.tinker_connection,
                base_model_alias=selected_config.base_model_alias,
                training=selected_config.training,
            )
        else:
            selected_input, bootstrap = _optional_bootstrap_config(store)
            if bootstrap is not None:
                settings = InitialSFTModelOptimizationSettings(
                    model_alias_prefix=bootstrap.model_alias,
                    tinker_connection=bootstrap.tinker_connection,
                    base_model_alias=bootstrap.base_model_alias,
                    training=bootstrap.training,
                )
            elif initial_settings is not None:
                settings = initial_settings
            else:
                raise AutomaticSFTPreparationError(
                    "first model optimization needs confirmed Tinker connection, base model, "
                    "model alias, and bounded training settings"
                )
        return _materialize_appended_prefix(
            store,
            journal=journal,
            selected_input=selected_input,
            settings=settings,
            created_at=created_at,
            code_revision=code_revision,
        )
    except AutomaticSFTPreparationError:
        raise
    except (
        ArtifactCorruptionError,
        ModelCatalogError,
        ProjectStoreError,
        RuntimeTraceSnapshotError,
        SFTBuildError,
        SFTModelOptimizationError,
        SFTModelOptimizationSelectionError,
        ValueError,
    ) as exc:
        raise AutomaticSFTPreparationError(
            f"cannot prepare routed interactions for model optimization: {exc}"
        ) from exc


def require_completed_runtime_interactions(store: ProjectStore) -> None:
    """Validate that a project has at least one completed routed SFT target.

    Args:
        store: Project whose journal is checked before any setup, consent, or backend work.

    Raises:
        AutomaticSFTPreparationError: The journal is corrupt, empty, or has no completion.
    """
    _completed_runtime_events(RuntimeInteractionJournal(store.paths))


def _completed_runtime_events(
    journal: RuntimeInteractionJournal,
) -> tuple[RuntimeJournalEvent, ...]:
    """Return a valid journal containing at least one completed routed interaction.

    Args:
        journal: Project journal read through its complete event validation boundary.

    Returns:
        Every current durable event in global journal order.

    Raises:
        AutomaticSFTPreparationError: The journal is corrupt, empty, or contains no completion.
    """
    try:
        events = journal.read_events()
    except RuntimeJournalError as exc:
        raise AutomaticSFTPreparationError(f"runtime journal is invalid: {exc}") from exc
    if not events:
        raise AutomaticSFTPreparationError(
            "runtime journal has no interactions; run the router and complete at least one "
            "request before `wmo optimize model`"
        )
    if not any(isinstance(event, RuntimeCompletedEvent) for event in events):
        raise AutomaticSFTPreparationError(
            "runtime journal has no completed routed interactions; failed or disconnected "
            "requests are not SFT targets"
        )
    return events


def _optional_bootstrap_config(
    store: ProjectStore,
) -> tuple[ArtifactInput | None, SFTModelOptimizationConfig | None]:
    """Load an optional project-bound config as compatible first-run settings.

    Args:
        store: Project that may retain an explicit bootstrap config binding.

    Returns:
        The exact config manifest input and verified config, or two ``None`` values.
    """
    bound = store.load_project().model_optimization_config
    if bound is None:
        return None, None
    return bound, load_sft_model_optimization_config(store, bound.artifact_id)


def _reuse_or_require_append(
    store: ProjectStore,
    *,
    events: tuple[RuntimeJournalEvent, ...],
    latest: LatestSFTModelOptimization,
    config: SFTModelOptimizationConfig,
    dataset: SFTDatasetArtifact,
) -> AutomaticSFTPreparation | None:
    """Reuse an unchanged selected prefix or require a strict append-only continuation.

    Args:
        store: Project containing the selected snapshot.
        events: Complete currently validated runtime journal.
        latest: Recursively verified latest graph pointer.
        config: Config selected by the pointer.
        dataset: Dataset selected by the config.

    Returns:
        A reuse result when the journal is unchanged, otherwise ``None`` for a valid append.

    Raises:
        AutomaticSFTPreparationError: The current journal removed or changed selected events.
    """
    loaded = load_runtime_trace_snapshot(store.artifacts, latest.runtime_snapshot.artifact_id)
    snapshot = loaded.snapshot
    if len(events) < snapshot.last_ordinal:
        raise AutomaticSFTPreparationError(
            "runtime journal is shorter than the latest selected immutable prefix"
        )
    selected_prefix = events[: snapshot.last_ordinal]
    if _events_sha256(selected_prefix) != snapshot.prefix_sha256:
        raise AutomaticSFTPreparationError(
            "runtime journal changed inside the latest selected immutable prefix"
        )
    if len(events) == snapshot.last_ordinal:
        return AutomaticSFTPreparation(
            snapshot=snapshot,
            dataset=dataset,
            config=config,
            created=False,
        )
    return None


def _materialize_appended_prefix(
    store: ProjectStore,
    *,
    journal: RuntimeInteractionJournal,
    selected_input: ArtifactInput | None,
    settings: InitialSFTModelOptimizationSettings,
    created_at: datetime,
    code_revision: str,
) -> AutomaticSFTPreparation:
    """Persist a new full journal prefix, W12 dataset, config, and latest pointer.

    Args:
        store: Project receiving immutable artifacts and the coordination update.
        journal: Validated project runtime journal.
        selected_input: Exact prior config selection, or ``None`` on a first run.
        settings: Confirmed Tinker selections and bounded deterministic schedule.
        created_at: Time recorded on new artifacts and coordination state.
        code_revision: Exact revision recorded on new artifacts.

    Returns:
        Newly selected recursively verified artifacts.

    Raises:
        AutomaticSFTPreparationError: The deterministic alias selection collides.
    """
    exported = seal_runtime_trace_snapshot(
        journal,
        store.artifacts,
        created_at=created_at,
        code_revision=code_revision,
    )
    dataset = write_sft_dataset(
        store,
        build_sft_dataset(
            store=store,
            production_sources=(),
            teacher_sources=(),
            runtime_sources=(RuntimeSFTSource(snapshot_id=exported.snapshot.snapshot_id),),
            spec=SFTBuildSpec(held_out_fraction=0.0),
            created_at=created_at,
            code_revision=code_revision,
        ),
    )
    alias_prefix = settings.model_alias_prefix
    model_alias = versioned_sft_model_alias(alias_prefix, dataset.dataset.dataset_id)
    catalog = load_model_catalog(store.model_catalog_path)
    if model_alias in catalog.models:
        raise AutomaticSFTPreparationError(
            f"automatic SFT model alias {model_alias!r} already names another catalog record"
        )
    config = write_sft_model_optimization_config(
        store,
        create_sft_model_optimization_config(
            store,
            dataset_id=dataset.dataset.dataset_id,
            model_alias=model_alias,
            tinker_connection=settings.tinker_connection,
            base_model_alias=settings.base_model_alias,
            training=settings.training,
            created_at=created_at,
            code_revision=code_revision,
        ),
        bind_project=False,
    )
    snapshot_input = artifact_input(exported.snapshot_manifest)
    dataset_input = artifact_input(store.artifacts.read(dataset.dataset.dataset_id).manifest)
    config_input = artifact_input(store.artifacts.read(config.config_id).manifest)
    pointer = write_latest_sft_model_optimization(
        store,
        LatestSFTModelOptimization(
            project_id=store.paths.project_id,
            config=config_input,
            dataset=dataset_input,
            runtime_snapshot=snapshot_input,
            model_alias_prefix=alias_prefix,
            updated_at=created_at,
        ),
        expected_current=selected_input,
    )
    selected = load_sft_model_optimization_config(store, pointer.config.artifact_id)
    verified_dataset = load_verified_sft_dataset(store, pointer.dataset.artifact_id)
    verified_snapshot = load_runtime_trace_snapshot(
        store.artifacts, pointer.runtime_snapshot.artifact_id
    ).snapshot
    return AutomaticSFTPreparation(
        snapshot=verified_snapshot,
        dataset=verified_dataset,
        config=selected,
        created=True,
    )


def _events_sha256(events: tuple[RuntimeJournalEvent, ...]) -> str:
    """Return the canonical newline-framed digest for a validated journal prefix.

    Args:
        events: Ordered validated journal events in the selected prefix.

    Returns:
        SHA-256 digest matching the runtime snapshot prefix contract.
    """
    payload = b"\n".join(canonical_json_bytes(event) for event in events)
    if payload:
        payload += b"\n"
    return hashlib.sha256(payload).hexdigest()
