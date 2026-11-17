"""Frozen initialization losses and per-selection-batch TailSFT filtering.

Filtering ranks mean masked target loss minus its frozen initialization value. It removes
the most improved examples from this update only. Selected targets use token-averaged CE;
this algorithm does not select checkpoints or establish task correctness.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.project import ArtifactStore, artifact_input


class TailSFTPolicy(ContractModel):
    """Immutable initialization reference and a static per-batch drop fraction.

    Attributes:
        reference: Initialization losses over this exact dataset and rendered target masks.
        drop_fraction: Fraction rounded to nearest integer, ties to even; batches that would
            retain no examples are refused before dispatch.
    """

    reference: ArtifactInput
    drop_fraction: float = Field(ge=0, lt=1, allow_inf_nan=False)


class TargetLoss(ContractModel):
    """One length-normalized loss over an exact rendered input and target mask.

    Attributes:
        example_id: Exact immutable source example.
        datum_sha256: Tokenizer output, model input, target tokens and loss weights.
        target_tokens: Number of supervised tokens, excluding context.
        mean_loss: Mean negative target log probability; finite and nonnegative.
    """

    example_id: str
    datum_sha256: Sha256
    target_tokens: int = Field(gt=0)
    mean_loss: float = Field(ge=0, allow_inf_nan=False)


class InitialLosses(ContractModel):
    """All initialization losses, recorded before updates under one model and data identity.

    Attributes:
        dataset: Exact accepted training dataset.
        model_configuration: Frozen initial weights and rendering configuration identity.
        losses: Every initial target loss, bound to its rendered tokens and mask.
    """

    dataset: ArtifactInput
    model_configuration: Sha256
    losses: tuple[TargetLoss, ...]


class LossDatum(Protocol):
    """Rendered identity required by the filtering owner."""

    @property
    def example_id(self) -> str:
        """Return the source example identity."""
        ...

    @property
    def supervised_token_count(self) -> int:
        """Return the target-only token count."""
        ...


@runtime_checkable
class LossScorer(Protocol):
    """An explicit read-only scoring capability of the selected training backend."""

    def score_batch(self, datums: Sequence[LossDatum]) -> tuple[TargetLoss, ...]:
        """Score exactly these masks without a backward pass or optimizer update."""
        ...

    def datum_identity(self, datum: LossDatum) -> Sha256:
        """Hash exact rendered model inputs, target tokens and weights."""
        ...


class TailSelection(ContractModel):
    """Complete comparison over a selection batch, including excluded examples.

    Attributes:
        initial: Initial losses used for this selection batch.
        current: Read-only current losses for every candidate example.
        retained: Chosen example IDs in their original batch order.
        margins: Signed current-minus-initial loss for every candidate.
    """

    initial: tuple[TargetLoss, ...]
    current: tuple[TargetLoss, ...]
    retained: tuple[str, ...]
    margins: dict[str, float]


def select_tail(
    initial: Sequence[TargetLoss], current: Sequence[TargetLoss], fraction: float
) -> TailSelection:
    """Rank signed loss margins across the whole supplied selection batch.

    Args:
        initial: Frozen per-example initialization losses with the same target masks.
        current: Current per-example losses before this optimizer update.
        fraction: Fraction to drop, rounded to nearest integer with ties to even.

    Returns:
        Retained IDs in original batch order, plus both losses and every margin.
    """
    if not 0 <= fraction < 1 or not current:
        raise ValueError("TailSFT requires a nonempty batch and 0 <= drop_fraction < 1")
    baseline = {row.example_id: row for row in initial}
    if len(baseline) != len(initial) or len({row.example_id for row in current}) != len(current):
        raise ValueError("TailSFT loss identities must be unique")
    margins = {}
    for row in current:
        first = baseline.get(row.example_id)
        if first is None or (first.datum_sha256, first.target_tokens) != (
            row.datum_sha256,
            row.target_tokens,
        ):
            raise ValueError(
                "TailSFT initial reference or target mask changed; start a new registered run"
            )
        margins[row.example_id] = row.mean_loss - first.mean_loss
    count = round(len(current) * fraction)
    if count >= len(current):
        raise ValueError(
            "TailSFT would remove the whole batch; lower the fraction or enlarge the batch"
        )
    removed = set(sorted(margins, key=lambda key: (margins[key], key))[:count])
    return TailSelection(
        initial=tuple(baseline[row.example_id] for row in current),
        current=tuple(current),
        retained=tuple(row.example_id for row in current if row.example_id not in removed),
        margins=margins,
    )


def load_initial_losses(store: ArtifactStore, reference: ArtifactInput) -> InitialLosses:
    """Verify the saved immutable initialization result and dataset binding."""
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != "sft-initial-losses":
        raise ValueError("TailSFT reference differs from its saved initialization result")
    value = InitialLosses.model_validate_json(
        store.read_bytes(reference.artifact_id, "losses.json")
    )
    if value.dataset not in manifest.inputs:
        raise ValueError("TailSFT initialization lost its dataset dependency")
    for reference_input in manifest.inputs:
        if artifact_input(store.read(reference_input.artifact_id).manifest) != reference_input:
            raise ValueError("TailSFT initialization input changed")
    if artifact_input(store.read(value.dataset.artifact_id).manifest) != value.dataset:
        raise ValueError("TailSFT initialization dataset changed")
    if len({row.example_id for row in value.losses}) != len(value.losses):
        raise ValueError("TailSFT initialization contains duplicate examples")
    return value


def record_initial_losses(
    store: ArtifactStore,
    dataset: ArtifactInput,
    model_configuration: Sha256,
    datums: Sequence[LossDatum],
    scorer: LossScorer,
    *,
    batch_size: int,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Score every training example once before updates, retaining durable dispatch intents.

    Args:
        store: Trusted artifact owner, under the enclosing training job's budget.
        dataset: Exact accepted dataset; the caller supplies only its training partition.
        model_configuration: Frozen base model, LoRA initialization and renderer configuration.
        datums: Exact rendered training inputs, never held-out examples.
        scorer: Fresh untrained session with read-only loss scoring.
        batch_size: Finite selection batch ceiling.
        created_at: Publication time.
        code_revision: Producing implementation identity.

    Returns:
        Exact reference. A saved result replays; an unknown scoring outcome refuses retry.
    """
    if batch_size < 1 or not datums:
        raise ValueError("initial scoring needs training data and a positive batch size")
    binding = {
        "dataset": dataset.model_dump(mode="json"),
        "model": model_configuration,
        "datums": [
            {"example_id": row.example_id, "sha256": scorer.datum_identity(row)} for row in datums
        ],
        "batch_size": batch_size,
        "code": code_revision,
    }
    identity = stable_id("sft-initial-losses", binding)
    envelope = ArtifactEnvelope(
        schema_version=1, created_at=created_at, code_revision=code_revision, inputs=(dataset,)
    )
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        if store.exists(identity):
            return artifact_input(store.read(identity).manifest)
        losses = []
        dependencies = [dataset]
        for offset in range(0, len(datums), batch_size):
            batch = datums[offset : offset + batch_size]
            key = f"{identity}-batch-{offset}"
            if store.exists(key):
                recorded = InitialLosses.model_validate_json(store.read_bytes(key, "losses.json"))
                if (
                    recorded.dataset != dataset
                    or recorded.model_configuration != model_configuration
                ):
                    raise ValueError("saved initial scoring batch has a different initialization")
                rows = recorded.losses
            else:
                if store.exists(key + "-intent"):
                    raise ValueError(
                        "initial scoring outcome unknown; inspect the saved intent "
                        "before registering fresh work"
                    )
                store.write_json(
                    artifact_id=key + "-intent",
                    artifact_type="sft-scoring-intent",
                    envelope=envelope,
                    files={"intent.json": binding},
                )
                rows = scorer.score_batch(batch)
                store.write_json(
                    artifact_id=key,
                    artifact_type="sft-initial-losses",
                    envelope=envelope,
                    files={
                        "losses.json": InitialLosses(
                            dataset=dataset, model_configuration=model_configuration, losses=rows
                        )
                    },
                )
            if tuple(row.example_id for row in rows) != tuple(row.example_id for row in batch):
                raise ValueError("initial scorer returned a different example inventory")
            if any(
                row.datum_sha256 != scorer.datum_identity(datum)
                or row.target_tokens != datum.supervised_token_count
                for row, datum in zip(rows, batch, strict=True)
            ):
                raise ValueError("initial scorer changed the rendered target mask")
            dependencies.append(artifact_input(store.read(key).manifest))
            losses.extend(rows)
        manifest = store.write_json(
            artifact_id=identity,
            artifact_type="sft-initial-losses",
            envelope=envelope.model_copy(
                update={"inputs": tuple(sorted(dependencies, key=lambda ref: ref.artifact_id))}
            ),
            files={
                "losses.json": InitialLosses(
                    dataset=dataset, model_configuration=model_configuration, losses=tuple(losses)
                )
            },
        )
        return artifact_input(manifest)
