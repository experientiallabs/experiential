"""Automatic immutable runtime-dataset preparation for managed model optimization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from exp.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    canonical_jsonl_bytes,
    sha256_bytes,
    sha256_json,
)
from exp.common.core.locks import file_write_lock
from exp.common.models import ModelCatalogError, load_model_catalog
from exp.common.project import (
    ArtifactCorruptionError,
    ProjectStore,
    ProjectStoreError,
    artifact_input,
)
from exp.optimize.model.sft.builder import (
    SFTBuildError,
    build_sft_dataset,
    load_verified_sft_dataset,
    write_sft_dataset,
)
from exp.optimize.model.sft.composition import (
    SFTModelOptimizationConfig,
    SFTModelOptimizationError,
    create_sft_model_optimization_config,
    load_sft_model_optimization_config,
    sft_model_optimization_output_dir,
    write_sft_model_optimization_config,
)
from exp.optimize.model.sft.contracts import RuntimeSFTSource, SFTBuildSpec, SFTDatasetArtifact
from exp.optimize.model.sft.run_manifest import (
    AutomaticSFTRunAcceptance,
    AutomaticSFTRunAcceptanceSelection,
    initialize_automatic_sft_acceptance,
    initialize_tinker_sft_run,
    load_automatic_sft_acceptance,
    load_automatic_sft_acceptance_selection,
    load_tinker_sft_run,
    require_automatic_sft_acceptance_binding,
    write_automatic_sft_acceptance_selection_unlocked,
)
from exp.optimize.model.sft.selection import (
    LatestSFTModelOptimization,
    SFTModelOptimizationSelectionError,
    latest_sft_model_optimization_path,
    load_latest_sft_model_optimization,
    versioned_sft_model_alias,
    write_latest_sft_model_optimization,
)
from exp.optimize.model.sft.training_contracts import (
    TinkerSFTError,
    TinkerSFTRunManifest,
    TinkerSFTSpec,
)
from exp.runtime.router.journal import (
    RuntimeCompletedEvent,
    RuntimeInteractionJournal,
    RuntimeJournalError,
    RuntimeJournalEvent,
)
from exp.runtime.router.journal_handoff import commit_runtime_journal_prefix
from exp.runtime.router.snapshot import (
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
        accepted: Whether explicit consent already durably accepted this graph for W13.
    """

    snapshot: RuntimeTraceSnapshot
    dataset: SFTDatasetArtifact
    config: SFTModelOptimizationConfig
    created: bool
    accepted: bool


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
                code_revision=code_revision,
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
        TinkerSFTError,
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


def accept_runtime_sft_model_optimization(
    store: ProjectStore,
    preparation: AutomaticSFTPreparation,
    *,
    created_at: datetime,
    code_revision: str,
) -> TinkerSFTRunManifest:
    """Atomically bind the current journal prefix to one durable local W13 run.

    The exact prepared prefix is checked under the runtime journal append lock. The callback then
    initializes only the local W13 run manifest before the lock is released. Credential reads,
    trainer construction, and provider work remain outside the journal lock.

    Args:
        store: Project containing the prepared graph and runtime journal.
        preparation: Exact snapshot, W12 dataset, and config approved for managed execution.
        created_at: Time recorded if the W13 run is first accepted.
        code_revision: Exact release revision recorded or required by the W13 run.

    Returns:
        The new or exactly matching durable W13 run manifest.

    Raises:
        AutomaticSFTPreparationError: The selected graph or runtime journal changed before local
            W13 acceptance, or the run manifest cannot be initialized safely.
    """
    try:
        _require_preparation_matches_latest(store, preparation)

        def initialize_run() -> TinkerSFTRunManifest:
            """CAS-select config-bound acceptance while journal appends are excluded.

            Returns:
                The new or exactly matching W13 run manifest.
            """
            coordination_path = latest_sft_model_optimization_path(store)
            with file_write_lock(
                coordination_path,
                what="model optimization and automatic SFT acceptance",
            ):
                latest = _require_preparation_matches_latest(store, preparation)
                selected = load_automatic_sft_acceptance_selection(store)
                previous_acceptance: ArtifactInput | None = None
                if selected is not None:
                    prior, prior_input = _load_selected_acceptance(store, selected)
                    if prior.config == latest.config:
                        manifest = _load_selected_run_manifest(
                            store,
                            preparation=preparation,
                            code_revision=code_revision,
                        )
                        _require_current_acceptance_binding(
                            store,
                            prior,
                            manifest=manifest,
                            latest=latest,
                            preparation=preparation,
                            code_revision=code_revision,
                        )
                        return manifest
                    _require_terminal_registered_acceptance(store, prior)
                    previous_acceptance = prior_input
                manifest = initialize_tinker_sft_run(
                    store,
                    preparation.dataset.dataset.dataset_id,
                    preparation.config.training,
                    sft_model_optimization_output_dir(store, preparation.config.config_id),
                    created_at=created_at,
                    code_revision=code_revision,
                )
                _receipt, acceptance_input = initialize_automatic_sft_acceptance(
                    store,
                    manifest=manifest,
                    previous_acceptance=previous_acceptance,
                    config=latest.config,
                    dataset=latest.dataset,
                    runtime_snapshot=latest.runtime_snapshot,
                    model_alias=preparation.config.model_alias,
                    tinker_connection=preparation.config.tinker_connection,
                    base_model=preparation.config.base_model,
                    connection_config_sha256=preparation.config.connection_config_sha256,
                    training_spec_sha256=sha256_json(preparation.config.training),
                    runtime_last_ordinal=preparation.snapshot.last_ordinal,
                    runtime_prefix_sha256=preparation.snapshot.prefix_sha256,
                    created_at=created_at,
                    code_revision=code_revision,
                )
                write_automatic_sft_acceptance_selection_unlocked(
                    store,
                    AutomaticSFTRunAcceptanceSelection(
                        schema_version=1,
                        project_id=store.paths.project_id,
                        acceptance=acceptance_input,
                        previous_acceptance=previous_acceptance,
                        config=latest.config,
                        updated_at=created_at,
                    ),
                    expected_current=previous_acceptance,
                )
                return manifest

        return commit_runtime_journal_prefix(
            RuntimeInteractionJournal(store.paths),
            last_ordinal=preparation.snapshot.last_ordinal,
            prefix_sha256=preparation.snapshot.prefix_sha256,
            commit=initialize_run,
        )
    except AutomaticSFTPreparationError:
        raise
    except RuntimeJournalError as exc:
        raise AutomaticSFTPreparationError(
            "runtime journal changed before managed training acceptance; rerun "
            "`exp optimize model` so every durable completion is included in the immutable "
            "schedule and spend consent"
        ) from exc
    except (
        ArtifactCorruptionError,
        ProjectStoreError,
        RuntimeTraceSnapshotError,
        SFTBuildError,
        SFTModelOptimizationError,
        SFTModelOptimizationSelectionError,
        TinkerSFTError,
        ValueError,
    ) as exc:
        raise AutomaticSFTPreparationError(
            f"cannot accept routed interactions for managed training: {exc}"
        ) from exc


def _require_preparation_matches_latest(
    store: ProjectStore,
    preparation: AutomaticSFTPreparation,
) -> LatestSFTModelOptimization:
    """Recursively require one preparation to equal the selected immutable graph.

    Args:
        store: Project containing the selected graph.
        preparation: Exact snapshot, W12 dataset, and config proposed for acceptance.

    Returns:
        Recursively verified latest pointer matching every preparation component.

    Raises:
        AutomaticSFTPreparationError: Selection or any immutable artifact differs.
    """
    latest = load_latest_sft_model_optimization(store)
    if latest is None or (
        latest.config.artifact_id != preparation.config.config_id
        or latest.dataset.artifact_id != preparation.dataset.dataset.dataset_id
        or latest.runtime_snapshot.artifact_id != preparation.snapshot.snapshot_id
    ):
        raise AutomaticSFTPreparationError(
            "selected model-optimization graph changed before managed training acceptance"
        )
    selected_config = load_sft_model_optimization_config(store, preparation.config.config_id)
    selected_dataset = load_verified_sft_dataset(store, preparation.dataset.dataset.dataset_id)
    selected_snapshot = load_runtime_trace_snapshot(
        store.artifacts, preparation.snapshot.snapshot_id
    ).snapshot
    if (
        selected_config != preparation.config
        or selected_dataset != preparation.dataset
        or selected_snapshot != preparation.snapshot
    ):
        raise AutomaticSFTPreparationError(
            "prepared model-optimization graph differs from recursive verification"
        )
    return latest


def _load_selected_acceptance(
    store: ProjectStore,
    selection: AutomaticSFTRunAcceptanceSelection,
) -> tuple[AutomaticSFTRunAcceptance, ArtifactInput]:
    """Load the selected receipt and verify its prior chain and pointer bindings.

    Args:
        store: Project owning the selection and immutable acceptance artifacts.
        selection: Canonical mutable selection pointer.

    Returns:
        Selected receipt and exact immutable artifact input.

    Raises:
        AutomaticSFTPreparationError: Pointer fields, prior chain, or terminal provenance differ.
    """
    receipt, receipt_input = load_automatic_sft_acceptance(store, selection.acceptance.artifact_id)
    if (
        receipt_input != selection.acceptance
        or receipt.config != selection.config
        or receipt.previous_acceptance != selection.previous_acceptance
    ):
        raise AutomaticSFTPreparationError(
            "automatic SFT acceptance pointer differs from its immutable receipt"
        )
    previous = receipt.previous_acceptance
    while previous is not None:
        prior, prior_input = load_automatic_sft_acceptance(store, previous.artifact_id)
        if prior_input != previous:
            raise AutomaticSFTPreparationError(
                "automatic SFT acceptance prior input differs from its manifest"
            )
        _require_terminal_registered_acceptance(store, prior)
        previous = prior.previous_acceptance
    return receipt, receipt_input


def _require_terminal_registered_acceptance(
    store: ProjectStore,
    acceptance: AutomaticSFTRunAcceptance,
) -> None:
    """Require an earlier selected acceptance to have exact registered terminal provenance.

    Args:
        store: Project owning the local model catalog.
        acceptance: Earlier immutable receipt that must be terminal before a successor.

    Raises:
        AutomaticSFTPreparationError: Registration is absent or differs from the accepted graph.
    """
    catalog = load_model_catalog(store.model_catalog_path)
    record = catalog.models.get(acceptance.model_alias)
    provenance = None if record is None else record.sft_provenance
    if (
        record is None
        or record.connection != acceptance.tinker_connection
        or provenance is None
        or provenance.source_dataset != acceptance.dataset
        or provenance.optimization_config != acceptance.config
        or provenance.training_spec_sha256 != acceptance.training_spec_sha256
        or provenance.run_id != acceptance.tinker_run_id
        or provenance.base_model != acceptance.base_model
        or provenance.connection_config_sha256 != acceptance.connection_config_sha256
    ):
        raise AutomaticSFTPreparationError(
            "a newer automatic SFT acceptance cannot replace an incomplete or differently "
            "registered prior accepted run"
        )


def _load_selected_run_manifest(
    store: ProjectStore,
    *,
    preparation: AutomaticSFTPreparation,
    code_revision: str,
) -> TinkerSFTRunManifest:
    """Load the exact generic W13 manifest required by a selected automatic receipt.

    Args:
        store: Project owning the W12 dataset and local W13 run.
        preparation: Current recursively verified selected graph.
        code_revision: Exact release revision required from the run.

    Returns:
        Recursively verified existing generic W13 manifest.

    Raises:
        AutomaticSFTPreparationError: The selected receipt has no matching W13 manifest.
    """
    manifest = load_tinker_sft_run(
        store,
        preparation.dataset.dataset.dataset_id,
        preparation.config.training,
        sft_model_optimization_output_dir(store, preparation.config.config_id),
        code_revision=code_revision,
    )
    if manifest is None:
        raise AutomaticSFTPreparationError(
            "selected automatic SFT acceptance has no recursively verified W13 manifest"
        )
    return manifest


def _require_current_acceptance_binding(
    store: ProjectStore,
    acceptance: AutomaticSFTRunAcceptance,
    *,
    manifest: TinkerSFTRunManifest,
    latest: LatestSFTModelOptimization,
    preparation: AutomaticSFTPreparation,
    code_revision: str,
) -> None:
    """Require one selected receipt to bind every current immutable graph component.

    Args:
        store: Project owning the selected graph.
        acceptance: Selected immutable automatic acceptance receipt.
        manifest: Recursively verified generic W13 run manifest.
        latest: Current recursively verified graph pointer.
        preparation: Current exact snapshot, W12 dataset, and config.
        code_revision: Exact release revision required by the receipt.

    Raises:
        TinkerSFTError: Any receipt or graph field differs.
    """
    require_automatic_sft_acceptance_binding(
        acceptance,
        manifest=manifest,
        project_id=store.paths.project_id,
        previous_acceptance=acceptance.previous_acceptance,
        config=latest.config,
        dataset=latest.dataset,
        runtime_snapshot=latest.runtime_snapshot,
        model_alias=preparation.config.model_alias,
        tinker_connection=preparation.config.tinker_connection,
        base_model=preparation.config.base_model,
        connection_config_sha256=preparation.config.connection_config_sha256,
        training_spec_sha256=sha256_json(preparation.config.training),
        runtime_last_ordinal=preparation.snapshot.last_ordinal,
        runtime_prefix_sha256=preparation.snapshot.prefix_sha256,
        code_revision=code_revision,
    )


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
            "request before `exp optimize model`"
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
    code_revision: str,
) -> AutomaticSFTPreparation | None:
    """Reuse an unchanged selected prefix or require a strict append-only continuation.

    Args:
        store: Project containing the selected snapshot.
        events: Complete currently validated runtime journal.
        latest: Recursively verified latest graph pointer.
        config: Config selected by the pointer.
        dataset: Dataset selected by the config.
        code_revision: Exact release revision required by any accepted W13 run.

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
    if sha256_bytes(canonical_jsonl_bytes(selected_prefix)) != snapshot.prefix_sha256:
        raise AutomaticSFTPreparationError(
            "runtime journal changed inside the latest selected immutable prefix"
        )
    manifest = load_tinker_sft_run(
        store,
        dataset.dataset.dataset_id,
        config.training,
        sft_model_optimization_output_dir(store, config.config_id),
        code_revision=code_revision,
    )
    automatic_acceptance: AutomaticSFTRunAcceptance | None = None
    selected = load_automatic_sft_acceptance_selection(store)
    if selected is not None:
        selected_acceptance, _selected_input = _load_selected_acceptance(store, selected)
        if selected_acceptance.config == latest.config:
            if manifest is None:
                raise AutomaticSFTPreparationError(
                    "selected automatic SFT acceptance has no recursively verified W13 manifest"
                )
            preparation = AutomaticSFTPreparation(
                snapshot=snapshot,
                dataset=dataset,
                config=config,
                created=False,
                accepted=True,
            )
            _require_current_acceptance_binding(
                store,
                selected_acceptance,
                manifest=manifest,
                latest=latest,
                preparation=preparation,
                code_revision=code_revision,
            )
            automatic_acceptance = selected_acceptance
        else:
            _require_terminal_registered_acceptance(store, selected_acceptance)
    accepted_incomplete = automatic_acceptance is not None and _accepted_run_needs_resume(
        store, latest=latest, config=config
    )
    if accepted_incomplete or len(events) == snapshot.last_ordinal:
        return AutomaticSFTPreparation(
            snapshot=snapshot,
            dataset=dataset,
            config=config,
            created=False,
            accepted=accepted_incomplete,
        )
    return None


def _accepted_run_needs_resume(
    store: ProjectStore,
    *,
    latest: LatestSFTModelOptimization,
    config: SFTModelOptimizationConfig,
) -> bool:
    """Return whether a durably accepted W13 run still lacks catalog registration.

    A registered alias is the existing durable terminal boundary: composition writes it only
    after recursively verifying the completed W13 result and sampling handle. Any alias that does
    not bind the selected W12/config inputs fails before a newer journal prefix is materialized.

    Args:
        store: Project owning the selected graph and local model catalog.
        latest: Exact selected immutable graph pointer.
        config: Recursively verified config selected by the pointer.

    Returns:
        ``True`` while an accepted run must resume its original prefix, otherwise ``False``.

    Raises:
        AutomaticSFTPreparationError: The registered alias has missing or conflicting provenance.
    """
    catalog = load_model_catalog(store.model_catalog_path)
    record = catalog.models.get(config.model_alias)
    if record is None:
        return True
    provenance = record.sft_provenance
    if (
        record.connection != config.tinker_connection
        or provenance is None
        or provenance.source_dataset != latest.dataset
        or provenance.optimization_config != latest.config
        or provenance.training_spec_sha256 != sha256_json(config.training)
        or provenance.base_model != config.base_model
        or provenance.connection_config_sha256 != config.connection_config_sha256
    ):
        raise AutomaticSFTPreparationError(
            f"registered automatic SFT alias {config.model_alias!r} does not bind the selected "
            "immutable W12/W13 graph"
        )
    return False


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
        accepted=False,
    )
