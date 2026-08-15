"""Local acceptance and recursive verification for one managed W13 run manifest."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    assert_secret_free,
    canonical_json_bytes,
    sha256_json,
    stable_id,
)
from wmo.common.core.files import write_bytes_atomic
from wmo.common.core.locks import file_write_lock
from wmo.common.models import ModelSnapshot
from wmo.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactCorruptionError,
    ArtifactStoreError,
    ProjectStore,
    artifact_input,
)
from wmo.optimize.model.sft.builder import SFTBuildError, load_verified_sft_dataset
from wmo.optimize.model.sft.contracts import SFTBuildSpec, SFTDatasetArtifact
from wmo.optimize.model.sft.rendering import partitioned_rows_sha256
from wmo.optimize.model.sft.training_contracts import (
    TinkerSFTError,
    TinkerSFTResumeError,
    TinkerSFTRunManifest,
    TinkerSFTSpec,
)

MANIFEST_FILE = "manifest.json"
AUTOMATIC_ACCEPTANCE_FILE = "acceptance.json"
AUTOMATIC_ACCEPTANCE_TYPE = "automatic-sft-acceptance"
AUTOMATIC_ACCEPTANCE_POINTER_FILE = "automatic-acceptance.json"


class AutomaticSFTRunAcceptance(ArtifactEnvelope):
    """Exact automatic W12 selection durably authorized for one local W13 run."""

    schema_version: Literal[1] = 1
    acceptance_id: ArtifactId
    project_id: ArtifactId
    previous_acceptance: ArtifactInput | None = None
    config: ArtifactInput
    dataset: ArtifactInput
    runtime_snapshot: ArtifactInput
    model_alias: ArtifactId
    tinker_connection: ArtifactId
    base_model: ModelSnapshot
    connection_config_sha256: Sha256
    training_spec_sha256: Sha256
    runtime_last_ordinal: int = Field(ge=1)
    runtime_prefix_sha256: Sha256
    tinker_run_id: str = Field(min_length=1, max_length=128)
    tinker_manifest_sha256: Sha256


class AutomaticSFTRunAcceptanceSelection(ContractModel):
    """Mutable CAS pointer selecting the only automatic acceptance authorized to replay."""

    schema_version: Literal[1] = 1
    project_id: ArtifactId
    acceptance: ArtifactInput
    previous_acceptance: ArtifactInput | None = None
    config: ArtifactInput
    updated_at: AwareDatetime


def automatic_sft_acceptance_path(store: ProjectStore) -> Path:
    """Return the canonical project-local automatic-acceptance selection path.

    Args:
        store: Project whose managed run acceptance is coordinated.

    Returns:
        Mutable pointer path sharing the model-optimization coordination directory.
    """
    return store.paths.project_directory / "model-optimization" / AUTOMATIC_ACCEPTANCE_POINTER_FILE


def initialize_automatic_sft_acceptance(
    store: ProjectStore,
    *,
    manifest: TinkerSFTRunManifest,
    previous_acceptance: ArtifactInput | None,
    config: ArtifactInput,
    dataset: ArtifactInput,
    runtime_snapshot: ArtifactInput,
    model_alias: ArtifactId,
    tinker_connection: ArtifactId,
    base_model: ModelSnapshot,
    connection_config_sha256: Sha256,
    training_spec_sha256: Sha256,
    runtime_last_ordinal: int,
    runtime_prefix_sha256: Sha256,
    created_at: datetime,
    code_revision: str,
) -> tuple[AutomaticSFTRunAcceptance, ArtifactInput]:
    """Persist or verify automatic consent bound to an exact selected W12/W13 graph.

    Args:
        store: Project receiving the immutable acceptance artifact.
        manifest: Recursively verified generic W13 run manifest created under the journal lock.
        previous_acceptance: Exact prior selected acceptance, or ``None`` at genesis.
        config: Exact immutable model-optimization config artifact input.
        dataset: Exact immutable W12 dataset artifact input.
        runtime_snapshot: Exact immutable runtime-prefix snapshot artifact input.
        model_alias: Versioned trained-model alias frozen by the selected config.
        tinker_connection: Secret-free local Tinker connection name.
        base_model: Exact provider model snapshot used as the training base.
        connection_config_sha256: Exact secret-free Tinker connection digest.
        training_spec_sha256: Exact immutable training-spec digest.
        runtime_last_ordinal: Final journal ordinal authorized by consent.
        runtime_prefix_sha256: Canonical digest of every authorized journal event.
        created_at: Time recorded only when acceptance is first persisted.
        code_revision: Exact release revision authorizing the automatic run.

    Returns:
        New or exactly matching immutable receipt and its exact artifact input.

    Raises:
        TinkerSFTError: The proposed binding conflicts with the W13 manifest or prior receipt.
    """
    acceptance = _automatic_acceptance(
        manifest=manifest,
        project_id=store.paths.project_id,
        previous_acceptance=previous_acceptance,
        config=config,
        dataset=dataset,
        runtime_snapshot=runtime_snapshot,
        model_alias=model_alias,
        tinker_connection=tinker_connection,
        base_model=base_model,
        connection_config_sha256=connection_config_sha256,
        training_spec_sha256=training_spec_sha256,
        runtime_last_ordinal=runtime_last_ordinal,
        runtime_prefix_sha256=runtime_prefix_sha256,
        created_at=created_at,
        code_revision=code_revision,
    )
    try:
        manifest_record = store.artifacts.write_json(
            artifact_id=acceptance.acceptance_id,
            artifact_type=AUTOMATIC_ACCEPTANCE_TYPE,
            envelope=acceptance,
            files={AUTOMATIC_ACCEPTANCE_FILE: acceptance},
        )
    except ArtifactAlreadyExistsError:
        existing, existing_input = load_automatic_sft_acceptance(store, acceptance.acceptance_id)
        _validate_automatic_acceptance(
            existing,
            expected=acceptance.model_copy(update={"created_at": existing.created_at}),
        )
        return existing, existing_input
    except (ArtifactCorruptionError, ArtifactStoreError, ValueError) as exc:
        raise TinkerSFTError(f"cannot persist automatic SFT acceptance: {exc}") from exc
    return acceptance, artifact_input(manifest_record)


def load_automatic_sft_acceptance(
    store: ProjectStore,
    acceptance_id: ArtifactId,
) -> tuple[AutomaticSFTRunAcceptance, ArtifactInput]:
    """Load one immutable acceptance artifact with exact manifest and canonical-byte checks.

    Args:
        store: Project owning the immutable acceptance artifact.
        acceptance_id: Exact content-addressed acceptance identity.

    Returns:
        Verified receipt and exact artifact-manifest input.

    Raises:
        TinkerSFTResumeError: The artifact type, files, envelope, bytes, or identity differ.
    """
    try:
        stored = store.artifacts.read(acceptance_id)
        if stored.manifest.artifact_type != AUTOMATIC_ACCEPTANCE_TYPE:
            raise TinkerSFTResumeError("automatic SFT acceptance has the wrong artifact type")
        if tuple(file.path for file in stored.manifest.files) != (AUTOMATIC_ACCEPTANCE_FILE,):
            raise TinkerSFTResumeError("automatic SFT acceptance has unexpected artifact files")
        payload = store.artifacts.read_bytes(acceptance_id, AUTOMATIC_ACCEPTANCE_FILE)
        receipt = AutomaticSFTRunAcceptance.model_validate_json(payload)
    except (ArtifactCorruptionError, OSError, ValueError) as exc:
        raise TinkerSFTResumeError(f"cannot verify automatic SFT acceptance: {exc}") from exc
    if payload != canonical_json_bytes(receipt):
        raise TinkerSFTResumeError("automatic SFT acceptance payload is not canonical")
    if (
        receipt.acceptance_id != acceptance_id
        or receipt.acceptance_id != _automatic_acceptance_id(receipt)
        or receipt.project_id != store.paths.project_id
        or receipt.source is not None
        or stored.manifest.schema_version != receipt.schema_version
        or stored.manifest.created_at != receipt.created_at
        or stored.manifest.inputs != receipt.inputs
        or stored.manifest.code_revision != receipt.code_revision
        or stored.manifest.source is not None
    ):
        raise TinkerSFTResumeError("automatic SFT acceptance manifest differs from its receipt")
    return receipt, artifact_input(stored.manifest)


def require_automatic_sft_acceptance_binding(
    acceptance: AutomaticSFTRunAcceptance,
    *,
    manifest: TinkerSFTRunManifest,
    project_id: ArtifactId,
    previous_acceptance: ArtifactInput | None,
    config: ArtifactInput,
    dataset: ArtifactInput,
    runtime_snapshot: ArtifactInput,
    model_alias: ArtifactId,
    tinker_connection: ArtifactId,
    base_model: ModelSnapshot,
    connection_config_sha256: Sha256,
    training_spec_sha256: Sha256,
    runtime_last_ordinal: int,
    runtime_prefix_sha256: Sha256,
    code_revision: str,
) -> None:
    """Require a selected receipt to match the exact recursively verified current graph.

    Args:
        acceptance: Immutable receipt selected by project coordination.
        manifest: Recursively verified generic W13 run manifest.
        project_id: Project authorized to replay the managed run.
        previous_acceptance: Exact prior selected acceptance, or ``None`` at genesis.
        config: Exact immutable model-optimization config artifact input.
        dataset: Exact immutable W12 dataset artifact input.
        runtime_snapshot: Exact immutable runtime-prefix snapshot artifact input.
        model_alias: Versioned trained-model alias frozen by the selected config.
        tinker_connection: Secret-free local Tinker connection name.
        base_model: Exact provider model snapshot used as the training base.
        connection_config_sha256: Exact secret-free Tinker connection digest.
        training_spec_sha256: Exact immutable training-spec digest.
        runtime_last_ordinal: Final journal ordinal authorized by consent.
        runtime_prefix_sha256: Canonical digest of every authorized journal event.
        code_revision: Exact release revision required by the receipt.

    Raises:
        TinkerSFTResumeError: Any receipt identity or selected-graph binding differs.
    """
    expected = _automatic_acceptance(
        manifest=manifest,
        project_id=project_id,
        previous_acceptance=previous_acceptance,
        config=config,
        dataset=dataset,
        runtime_snapshot=runtime_snapshot,
        model_alias=model_alias,
        tinker_connection=tinker_connection,
        base_model=base_model,
        connection_config_sha256=connection_config_sha256,
        training_spec_sha256=training_spec_sha256,
        runtime_last_ordinal=runtime_last_ordinal,
        runtime_prefix_sha256=runtime_prefix_sha256,
        created_at=acceptance.created_at,
        code_revision=code_revision,
    )
    _validate_automatic_acceptance(acceptance, expected=expected)


def load_automatic_sft_acceptance_selection(
    store: ProjectStore,
) -> AutomaticSFTRunAcceptanceSelection | None:
    """Read the selected automatic acceptance without acquiring its external coordination lock.

    Callers that mutate the pointer must already hold the model-optimization coordination lock.
    Read-only preparation relies on atomic replacement and verifies the immutable target.

    Args:
        store: Project whose selected automatic acceptance is loaded.

    Returns:
        Canonical selected acceptance pointer, or ``None`` before the first consent commit.

    Raises:
        TinkerSFTResumeError: The pointer is malformed, noncanonical, or belongs elsewhere.
    """
    path = automatic_sft_acceptance_path(store)
    _require_safe_automatic_acceptance_path(store, path)
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise TinkerSFTResumeError("automatic SFT acceptance pointer is not a safe file")
    try:
        payload = path.read_bytes()
        selection = AutomaticSFTRunAcceptanceSelection.model_validate_json(payload)
    except (OSError, ValueError) as exc:
        raise TinkerSFTResumeError(f"cannot read automatic SFT acceptance pointer: {exc}") from exc
    if payload != canonical_json_bytes(selection):
        raise TinkerSFTResumeError("automatic SFT acceptance pointer is not canonical")
    if selection.project_id != store.paths.project_id:
        raise TinkerSFTResumeError("automatic SFT acceptance pointer belongs to another project")
    return selection


def write_automatic_sft_acceptance_selection_unlocked(
    store: ProjectStore,
    selection: AutomaticSFTRunAcceptanceSelection,
    *,
    expected_current: ArtifactInput | None,
) -> AutomaticSFTRunAcceptanceSelection:
    """CAS-select one immutable acceptance while the caller owns the coordination lock.

    Args:
        store: Project receiving the selected acceptance pointer.
        selection: New exact acceptance and prior-selection binding.
        expected_current: Exact acceptance selected before consent, or ``None`` at genesis.

    Returns:
        Stored selection or a byte-equivalent concurrent replay.

    Raises:
        TinkerSFTResumeError: Project identity, prior selection, or stored bytes conflict.
    """
    if selection.project_id != store.paths.project_id:
        raise TinkerSFTResumeError("cannot select automatic SFT acceptance for another project")
    current = load_automatic_sft_acceptance_selection(store)
    if current is not None and current == selection:
        return current
    current_input = None if current is None else current.acceptance
    if current_input != expected_current or selection.previous_acceptance != expected_current:
        raise TinkerSFTResumeError(
            "automatic SFT acceptance selection changed before consent commit"
        )
    path = automatic_sft_acceptance_path(store)
    _require_safe_automatic_acceptance_path(store, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_safe_automatic_acceptance_path(store, path)
    write_bytes_atomic(path, canonical_json_bytes(selection), follow_symlinks=False)
    stored = load_automatic_sft_acceptance_selection(store)
    if stored is None or stored != selection:
        raise TinkerSFTResumeError("automatic SFT acceptance pointer did not preserve selection")
    return stored


def initialize_tinker_sft_run(
    store: ProjectStore,
    dataset_id: ArtifactId,
    spec: TinkerSFTSpec,
    output_dir: Path,
    *,
    created_at: datetime,
    code_revision: str,
) -> TinkerSFTRunManifest:
    """Durably accept one verified W12 dataset without composing a trainer backend.

    Args:
        store: Project store owning the immutable W12 dataset.
        dataset_id: Persisted W12 dataset selected for the managed run.
        spec: Frozen base model, schedule, and spend settings.
        output_dir: Stable append-only W13 run directory for the selected config.
        created_at: Time recorded only when the run is first accepted.
        code_revision: Exact release revision recorded or required by the run.

    Returns:
        The new or exactly matching immutable W13 run manifest.

    Raises:
        TinkerSFTError: Dataset verification or run-manifest acceptance is unsafe.
    """
    dataset, dataset_input = verified_training_inputs(store, dataset_id)
    validate_run_inputs(dataset, created_at=created_at, code_revision=code_revision)
    manifest_path = output_dir / MANIFEST_FILE
    with file_write_lock(manifest_path, what="the Tinker SFT run"):
        return load_or_create_manifest(
            dataset=dataset,
            dataset_input=dataset_input,
            spec=spec,
            output_dir=output_dir,
            created_at=created_at,
            code_revision=code_revision,
        )


def load_tinker_sft_run(
    store: ProjectStore,
    dataset_id: ArtifactId,
    spec: TinkerSFTSpec,
    output_dir: Path,
    *,
    code_revision: str,
) -> TinkerSFTRunManifest | None:
    """Load an accepted W13 run manifest without creating local state.

    Args:
        store: Project store owning the immutable W12 dataset.
        dataset_id: Persisted W12 dataset selected for the managed run.
        spec: Frozen base model, schedule, and spend settings.
        output_dir: Stable append-only W13 run directory for the selected config.
        code_revision: Exact release revision required by the existing run.

    Returns:
        The recursively verified run manifest, or ``None`` when no run was accepted.

    Raises:
        TinkerSFTError: The dataset or existing manifest cannot be verified exactly.
    """
    dataset, dataset_input = verified_training_inputs(store, dataset_id)
    path = output_dir / MANIFEST_FILE
    if not path.exists():
        return None
    with file_write_lock(path, what="the Tinker SFT run"):
        if not path.exists():
            return None
        manifest = _read_model(path, TinkerSFTRunManifest, "Tinker SFT manifest")
        validate_manifest(
            manifest,
            dataset=dataset,
            dataset_input=dataset_input,
            spec=spec,
            code_revision=code_revision,
        )
        return manifest


def verified_training_inputs(
    store: ProjectStore,
    dataset_id: ArtifactId,
    *,
    legacy_build_spec: SFTBuildSpec | None = None,
) -> tuple[SFTDatasetArtifact, ArtifactInput]:
    """Load one recursively verified W12 dataset and its exact manifest input.

    Args:
        store: Project store owning the immutable dataset.
        dataset_id: Persisted W12 dataset identity.
        legacy_build_spec: Explicit settings required only for a legacy dataset.

    Returns:
        The verified dataset and exact manifest input used by W13.

    Raises:
        TinkerSFTError: The W12 artifact graph cannot be verified for training.
    """
    try:
        dataset = load_verified_sft_dataset(
            store,
            dataset_id,
            legacy_build_spec=legacy_build_spec,
        )
        dataset_input = artifact_input(store.artifacts.read(dataset_id).manifest)
    except (ArtifactCorruptionError, SFTBuildError) as exc:
        raise TinkerSFTError(f"W12 dataset {dataset_id} is not safe for training: {exc}") from exc
    return dataset, dataset_input


def validate_run_inputs(
    dataset: SFTDatasetArtifact, *, created_at: datetime, code_revision: str
) -> None:
    """Validate immutable W12 rows and local producer facts before W13 acceptance.

    Args:
        dataset: Recursively verified W12 dataset artifact.
        created_at: Time proposed for a new run manifest.
        code_revision: Exact release revision proposed or required for the run.

    Raises:
        TinkerSFTError: Dataset rows or producer facts are not safe for managed training.
    """
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise TinkerSFTError("Tinker SFT creation time must include a timezone")
    if not code_revision:
        raise TinkerSFTError("Tinker SFT code_revision must be non-empty")
    if dataset.dataset.status != "accepted":
        raise TinkerSFTError("Tinker SFT only accepts an accepted frozen W12 dataset")
    if dataset.dataset.examples_sha256 != partitioned_rows_sha256(dataset.rows):
        raise TinkerSFTError("Tinker SFT dataset rows do not match the W12 examples digest")
    train_rows = tuple(row for row in dataset.rows if row.partition == "train")
    held_out_rows = tuple(row for row in dataset.rows if row.partition == "held_out")
    if not train_rows:
        raise TinkerSFTError("an accepted W12 dataset needs at least one train example")
    if dataset.dataset.train_example_ids != tuple(
        sorted(row.example.example_id for row in train_rows)
    ):
        raise TinkerSFTError("Tinker SFT train rows do not match the W12 manifest")
    if dataset.dataset.held_out_example_ids != tuple(
        sorted(row.example.example_id for row in held_out_rows)
    ):
        raise TinkerSFTError("Tinker SFT held-out rows do not match the W12 manifest")


def load_or_create_manifest(
    *,
    dataset: SFTDatasetArtifact,
    dataset_input: ArtifactInput,
    spec: TinkerSFTSpec,
    output_dir: Path,
    created_at: datetime,
    code_revision: str,
) -> TinkerSFTRunManifest:
    """Load an exact run manifest or create it once from verified immutable inputs.

    Args:
        dataset: Recursively verified W12 dataset artifact.
        dataset_input: Exact W12 artifact manifest input.
        spec: Frozen base model, schedule, and spend settings.
        output_dir: Stable append-only W13 run directory.
        created_at: Time recorded only for a new manifest.
        code_revision: Exact release revision recorded or required by the run.

    Returns:
        The new or recursively verified existing manifest.

    Raises:
        TinkerSFTError: The dataset input or existing manifest differs from the run contract.
    """
    path = output_dir / MANIFEST_FILE
    if path.exists():
        manifest = _read_model(path, TinkerSFTRunManifest, "Tinker SFT manifest")
        validate_manifest(
            manifest,
            dataset=dataset,
            dataset_input=dataset_input,
            spec=spec,
            code_revision=code_revision,
        )
        return manifest
    if dataset_input.artifact_id != dataset.dataset.dataset_id:
        raise TinkerSFTError("canonical W12 manifest names a different dataset")
    spec_sha256 = sha256_json(spec)
    run_id = stable_id(
        "tinker-sft-run",
        {
            "dataset_id": dataset.dataset.dataset_id,
            "dataset_manifest_sha256": dataset_input.sha256,
            "dataset_build_sha256": dataset.dataset.build_sha256,
            "dataset_examples_sha256": dataset.dataset.examples_sha256,
            "spec_sha256": spec_sha256,
        },
    )
    manifest = TinkerSFTRunManifest(
        schema_version=1,
        created_at=created_at,
        inputs=(dataset_input,),
        code_revision=code_revision,
        run_id=run_id,
        dataset_id=dataset.dataset.dataset_id,
        dataset_manifest_sha256=dataset_input.sha256,
        dataset_build_sha256=dataset.dataset.build_sha256,
        dataset_examples_sha256=dataset.dataset.examples_sha256,
        spec=spec,
        spec_sha256=spec_sha256,
    )
    _write_new_json(path, manifest, "Tinker SFT manifest")
    return manifest


def validate_manifest(
    manifest: TinkerSFTRunManifest,
    *,
    dataset: SFTDatasetArtifact,
    dataset_input: ArtifactInput,
    spec: TinkerSFTSpec,
    code_revision: str,
) -> None:
    """Require one existing run manifest to match every immutable W12 input.

    Args:
        manifest: Existing local W13 run manifest.
        dataset: Recursively verified W12 dataset artifact.
        dataset_input: Exact W12 artifact manifest input.
        spec: Frozen base model, schedule, and spend settings.
        code_revision: Exact release revision required by the run.

    Raises:
        TinkerSFTResumeError: Any existing run field differs from the selected W12 graph.
    """
    if manifest.dataset_id != dataset.dataset.dataset_id:
        raise TinkerSFTResumeError("existing Tinker SFT run belongs to a different W12 dataset")
    if manifest.inputs != (dataset_input,):
        raise TinkerSFTResumeError(
            "existing Tinker SFT run does not name the canonical W12 artifact manifest"
        )
    if manifest.dataset_manifest_sha256 != dataset_input.sha256:
        raise TinkerSFTResumeError(
            "existing Tinker SFT run has a different W12 artifact manifest digest"
        )
    if manifest.dataset_build_sha256 != dataset.dataset.build_sha256:
        raise TinkerSFTResumeError("existing Tinker SFT run has a different W12 dataset build")
    if manifest.dataset_examples_sha256 != dataset.dataset.examples_sha256:
        raise TinkerSFTResumeError("existing Tinker SFT run has different W12 example rows")
    if manifest.spec != spec:
        raise TinkerSFTResumeError("existing Tinker SFT run has a different training spec")
    if manifest.code_revision != code_revision:
        raise TinkerSFTResumeError("existing Tinker SFT run has a different code revision")


def _automatic_acceptance(
    *,
    manifest: TinkerSFTRunManifest,
    project_id: ArtifactId,
    previous_acceptance: ArtifactInput | None,
    config: ArtifactInput,
    dataset: ArtifactInput,
    runtime_snapshot: ArtifactInput,
    model_alias: ArtifactId,
    tinker_connection: ArtifactId,
    base_model: ModelSnapshot,
    connection_config_sha256: Sha256,
    training_spec_sha256: Sha256,
    runtime_last_ordinal: int,
    runtime_prefix_sha256: Sha256,
    created_at: datetime,
    code_revision: str,
) -> AutomaticSFTRunAcceptance:
    """Build the content-addressed automatic acceptance for one selected graph.

    Args:
        manifest: Recursively verified generic W13 run manifest.
        project_id: Project authorized to replay the managed run.
        previous_acceptance: Exact prior selected acceptance, or ``None`` at genesis.
        config: Exact immutable model-optimization config artifact input.
        dataset: Exact immutable W12 dataset artifact input.
        runtime_snapshot: Exact immutable runtime-prefix snapshot artifact input.
        model_alias: Versioned trained-model alias frozen by the selected config.
        tinker_connection: Secret-free local Tinker connection name.
        base_model: Exact provider model snapshot used as the training base.
        connection_config_sha256: Exact secret-free Tinker connection digest.
        training_spec_sha256: Exact immutable training-spec digest.
        runtime_last_ordinal: Final journal ordinal authorized by consent.
        runtime_prefix_sha256: Canonical digest of every authorized journal event.
        created_at: Materialization time for a new receipt or existing receipt verification.
        code_revision: Exact release revision authorizing the automatic run.

    Returns:
        Fully bound automatic acceptance receipt.
    """
    inputs = tuple(
        sorted(
            (
                config,
                dataset,
                runtime_snapshot,
                *((previous_acceptance,) if previous_acceptance is not None else ()),
            ),
            key=lambda item: (item.artifact_id, item.sha256),
        )
    )
    manifest_sha256 = sha256_json(manifest)
    material = {
        "schema_version": 1,
        "code_revision": code_revision,
        "project_id": project_id,
        "previous_acceptance": (
            None if previous_acceptance is None else previous_acceptance.model_dump(mode="json")
        ),
        "inputs": tuple(item.model_dump(mode="json") for item in inputs),
        "config": config.model_dump(mode="json"),
        "dataset": dataset.model_dump(mode="json"),
        "runtime_snapshot": runtime_snapshot.model_dump(mode="json"),
        "model_alias": model_alias,
        "tinker_connection": tinker_connection,
        "base_model": base_model.model_dump(mode="json"),
        "connection_config_sha256": connection_config_sha256,
        "training_spec_sha256": training_spec_sha256,
        "runtime_last_ordinal": runtime_last_ordinal,
        "runtime_prefix_sha256": runtime_prefix_sha256,
        "tinker_run_id": manifest.run_id,
        "tinker_manifest_sha256": manifest_sha256,
    }
    return AutomaticSFTRunAcceptance(
        schema_version=1,
        created_at=created_at,
        code_revision=code_revision,
        source=None,
        acceptance_id=stable_id("automatic-sft-acceptance", material),
        project_id=project_id,
        previous_acceptance=previous_acceptance,
        inputs=inputs,
        config=config,
        dataset=dataset,
        runtime_snapshot=runtime_snapshot,
        model_alias=model_alias,
        tinker_connection=tinker_connection,
        base_model=base_model,
        connection_config_sha256=connection_config_sha256,
        training_spec_sha256=training_spec_sha256,
        runtime_last_ordinal=runtime_last_ordinal,
        runtime_prefix_sha256=runtime_prefix_sha256,
        tinker_run_id=manifest.run_id,
        tinker_manifest_sha256=manifest_sha256,
    )


def _automatic_acceptance_id(acceptance: AutomaticSFTRunAcceptance) -> ArtifactId:
    """Recompute one receipt's content identity from every semantic authorization field.

    Args:
        acceptance: Immutable automatic acceptance receipt to verify.

    Returns:
        Stable identity excluding only materialization time and redundant stored identity.
    """
    material = {
        "schema_version": acceptance.schema_version,
        "code_revision": acceptance.code_revision,
        "project_id": acceptance.project_id,
        "previous_acceptance": (
            None
            if acceptance.previous_acceptance is None
            else acceptance.previous_acceptance.model_dump(mode="json")
        ),
        "inputs": tuple(item.model_dump(mode="json") for item in acceptance.inputs),
        "config": acceptance.config.model_dump(mode="json"),
        "dataset": acceptance.dataset.model_dump(mode="json"),
        "runtime_snapshot": acceptance.runtime_snapshot.model_dump(mode="json"),
        "model_alias": acceptance.model_alias,
        "tinker_connection": acceptance.tinker_connection,
        "base_model": acceptance.base_model.model_dump(mode="json"),
        "connection_config_sha256": acceptance.connection_config_sha256,
        "training_spec_sha256": acceptance.training_spec_sha256,
        "runtime_last_ordinal": acceptance.runtime_last_ordinal,
        "runtime_prefix_sha256": acceptance.runtime_prefix_sha256,
        "tinker_run_id": acceptance.tinker_run_id,
        "tinker_manifest_sha256": acceptance.tinker_manifest_sha256,
    }
    return stable_id("automatic-sft-acceptance", material)


def _validate_automatic_acceptance(
    acceptance: AutomaticSFTRunAcceptance,
    *,
    expected: AutomaticSFTRunAcceptance,
) -> None:
    """Require every semantic field of an existing acceptance to match exactly.

    Args:
        acceptance: Existing canonical automatic acceptance receipt.
        expected: Receipt rebuilt from the recursively verified selected graph.

    Raises:
        TinkerSFTResumeError: Existing acceptance belongs to any different graph or producer.
    """
    if acceptance != expected:
        raise TinkerSFTResumeError(
            "existing automatic SFT acceptance does not bind the selected config, runtime "
            "snapshot, W12 dataset, connection, alias, and W13 manifest"
        )


def _write_new_json(path: Path, value: BaseModel, label: str) -> None:
    """Persist one secret-free canonical JSON contract without replacement.

    Args:
        path: New append-only file path.
        value: Validated contract to serialize canonically.
        label: User-facing artifact label used by failures.

    Raises:
        TinkerSFTResumeError: The append-only path already exists.
    """
    if path.exists():
        raise TinkerSFTResumeError(
            f"{label} already exists at {path}; append-only runs do not replace it"
        )
    assert_secret_free(value)
    write_bytes_atomic(path, canonical_json_bytes(value) + b"\n")


def _read_model[ModelT: BaseModel](path: Path, model_type: type[ModelT], label: str) -> ModelT:
    """Read one validated local contract with a contextual resume failure.

    Args:
        path: Existing local contract path.
        model_type: Pydantic model class used for exact validation.
        label: User-facing artifact label used by failures.

    Returns:
        The validated contract instance.

    Raises:
        TinkerSFTResumeError: The file is absent, unreadable, or malformed.
    """
    try:
        return model_type.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise TinkerSFTResumeError(f"cannot read {label} at {path}: {exc}") from exc


def _require_safe_automatic_acceptance_path(store: ProjectStore, path: Path) -> None:
    """Reject a pointer path escaping through any project-relative symlinked ancestor.

    Args:
        store: Project that owns the coordination path.
        path: Expected automatic-acceptance pointer below the project directory.

    Raises:
        TinkerSFTResumeError: The path escapes the project or an ancestor is a symlink.
    """
    project_directory = store.paths.project_directory
    try:
        path.relative_to(project_directory)
    except ValueError as exc:
        raise TinkerSFTResumeError(
            "automatic SFT acceptance pointer escapes its project directory"
        ) from exc
    if path.is_symlink():
        raise TinkerSFTResumeError("automatic SFT acceptance pointer is not a safe file")
    current = path.parent
    while True:
        if current.is_symlink():
            raise TinkerSFTResumeError(
                f"automatic SFT acceptance coordination directory is not safe: {current}"
            )
        if current == project_directory:
            break
        current = current.parent
