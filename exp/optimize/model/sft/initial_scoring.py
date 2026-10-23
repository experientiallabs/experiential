"""Budgeted initialization scoring from a verified dataset and a fresh training session."""

from datetime import datetime

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, sha256_json, stable_id
from exp.common.core.locks import CoordinationLock
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.model.sft.builder import load_verified_sft_dataset
from exp.optimize.model.sft.tail_sft import LossScorer, load_initial_losses, record_initial_losses
from exp.optimize.model.sft.training_contracts import TinkerSFTError, TinkerSFTSpec, TrainerBackend


def scoring_identity(dataset: ArtifactInput, spec: TinkerSFTSpec, code_revision: str) -> str:
    """Bind a fresh scoring pass without including the subsequent optimizer schedule."""
    return stable_id(
        "sft-initial-scoring",
        {
            "dataset": dataset.model_dump(mode="json"),
            "initialization": spec.initialization_identity(),
            "batch_size": spec.batch_size,
            "code_revision": code_revision,
        },
    )


def saved_initial_scoring(
    store: ProjectStore, dataset: ArtifactInput, spec: TinkerSFTSpec, *, code_revision: str
) -> ArtifactInput | None:
    """Verify a completed reference without constructing any provider client."""
    identity = scoring_identity(dataset, spec, code_revision)
    if not store.artifacts.exists(identity):
        return None
    reference = ArtifactInput.model_validate_json(
        store.artifacts.read_bytes(identity, "reference.json")
    )
    initial = load_initial_losses(store.artifacts, reference)
    if initial.dataset != dataset or initial.model_configuration != spec.initialization_identity():
        raise TinkerSFTError("saved initial scoring identity differs")
    return reference


def score_initial_training_losses(
    store: ProjectStore,
    dataset: ArtifactInput,
    spec: TinkerSFTSpec,
    backend: TrainerBackend,
    *,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Open only an untrained session and score the complete verified training partition.

    Args:
        store: Dataset owner, never an actor-controlled directory.
        dataset: Exact accepted persisted dataset.
        spec: Explicit finite price, input ceiling and total scoring budget. Tail filtering must
            be absent because this pass defines its reference.
        backend: Authorized backend; open always receives no resume handle.
        created_at: Receipt timestamp.
        code_revision: Implementation identity shared by replay.

    Returns:
        Immutable initial losses; saved work replays without opening a session.
    """
    if spec.tail_sft is not None:
        raise TinkerSFTError("initial scoring cannot use an existing TailSFT reference")
    artifact = load_verified_sft_dataset(store, dataset.artifact_id)
    if artifact_input(store.artifacts.read(dataset.artifact_id).manifest) != dataset:
        raise TinkerSFTError("initial scoring dataset digest differs")
    examples = tuple(row.example for row in artifact.rows if row.partition == "train")
    identity = scoring_identity(dataset, spec, code_revision)
    with CoordinationLock(store.artifacts.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_initial_scoring(store, dataset, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        bounds = tuple(
            backend.conservative_step_cost(
                spec, batch_example_count=len(examples[start : start + spec.batch_size])
            )
            for start in range(0, len(examples), spec.batch_size)
        )
        if not bounds or any(bound is None for bound in bounds) or spec.maximum_cost_usd is None:
            raise TinkerSFTError("initial scoring requires a finite conservative price and budget")
        estimate = sum(bound.value for bound in bounds if bound is not None)
        if estimate > spec.maximum_cost_usd:
            raise TinkerSFTError("initial scoring exceeds its frozen budget before dispatch")
        session = backend.open(spec, None)
        if not isinstance(session, LossScorer):
            raise TinkerSFTError("selected backend cannot score read-only initial losses")
        datums = session.render_examples(examples)
        if tuple(d.example_id for d in datums) != tuple(e.example_id for e in examples):
            raise TinkerSFTError("initial renderer changed the training partition")
        reference = record_initial_losses(
            store.artifacts,
            dataset,
            spec.initialization_identity(),
            datums,
            session,
            batch_size=spec.batch_size,
            created_at=created_at,
            code_revision=code_revision,
        )
        store.artifacts.write_json(
            artifact_id=identity,
            artifact_type="sft-initial-scoring",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=created_at,
                code_revision=code_revision,
                inputs=(dataset, reference),
            ),
            files={
                "reference.json": reference,
                "accounting.json": {
                    "conservative_cost_usd": estimate,
                    "provenance": "estimated",
                    "spec_sha256": sha256_json(spec),
                    "training_examples": len(examples),
                    "scoring_batches": len(bounds),
                },
            },
        )
        return reference
