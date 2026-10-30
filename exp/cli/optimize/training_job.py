"""Explicit persisted-dataset scoring and candidate training through the existing SFT owner."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from rich.console import Console

from exp.cli.shared.consent import require_spend_consent
from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256, stable_id
from exp.common.project import ProjectStore
from exp.optimize.model.sft.builder import load_verified_sft_dataset
from exp.optimize.model.sft.composition import create_sft_model_optimization_config
from exp.optimize.model.sft.initial_scoring import (
    saved_initial_scoring,
    score_initial_training_losses,
)
from exp.optimize.model.sft.run_manifest import sft_run_records
from exp.optimize.model.sft.tail_sft import load_initial_losses
from exp.optimize.model.sft.training import train_tinker_sft
from exp.optimize.model.sft.training_contracts import TinkerSFTSpec, TrainerBackend
from exp.optimize.model.sft.training_runtime import _schedule_batch_counts


class TrainingJob(ContractModel):
    """Explicit training inputs, without teacher generation or automatic serving activation.

    Attributes:
        action: Initial read-only losses or optimizer updates.
        dataset: Exact admitted dataset already present in this project.
        connection: Existing native Tinker connection.
        base_model_alias: Catalog snapshot matching the training base model.
        training: Frozen initialization, schedule, target budget and optional TailSFT reference.
    """

    action: Literal["score-initial", "train"]
    dataset: ArtifactInput
    connection: str
    base_model_alias: str
    training: TinkerSFTSpec


def run_training_job(
    store: ProjectStore,
    path: Path,
    *,
    console: Console,
    yes: bool,
    non_interactive: bool,
    dry_run: bool,
    code_revision: str,
    local_backend: TrainerBackend,
    compose_backend: Callable[[ProjectStore, str, Sha256], TrainerBackend],
) -> None:
    """Validate and price before credentials, then reuse the existing durable SFT execution.

    The resulting sampling handle is a candidate artifact. This path neither selects a serving
    alias nor treats training loss as evidence of task quality.
    """
    job = TrainingJob.model_validate_json(path.read_bytes())
    created_at = datetime.now(UTC)
    dataset = load_verified_sft_dataset(store, job.dataset.artifact_id)
    config = create_sft_model_optimization_config(
        store,
        dataset_id=job.dataset.artifact_id,
        model_alias="explicit-training-candidate",
        tinker_connection=job.connection,
        base_model_alias=job.base_model_alias,
        training=job.training,
        created_at=created_at,
        code_revision=code_revision,
    )
    if config.dataset != job.dataset:
        raise ValueError("training job dataset digest differs from its immutable manifest")
    train_examples = tuple(row.example for row in dataset.rows if row.partition == "train")
    identity = stable_id(
        "explicit-sft",
        {
            "job": job.model_dump(mode="json"),
            "base_model": config.base_model.model_dump(mode="json"),
            "connection": config.connection_config_sha256,
            "code": code_revision,
        },
    )
    output_dir = store.artifacts.project_directory / "sft-candidates" / identity
    saved = None
    if job.action == "score-initial":
        if job.training.tail_sft is not None:
            raise ValueError("initial scoring must not contain a TailSFT reference")
        saved = saved_initial_scoring(store, job.dataset, job.training, code_revision=code_revision)
        counts = tuple(
            len(train_examples[start : start + job.training.batch_size])
            for start in range(0, len(train_examples), job.training.batch_size)
        )
        complete = saved is not None
    else:
        counts = _schedule_batch_counts(len(train_examples), job.training)
        if job.training.tail_sft is not None:
            initial = load_initial_losses(store.artifacts, job.training.tail_sft.reference)
            if (
                initial.dataset != job.dataset
                or initial.model_configuration != job.training.initialization_identity()
                or {row.example_id for row in initial.losses}
                != {row.example_id for row in train_examples}
            ):
                raise ValueError("TailSFT initialization differs from this training job")
            if any(round(count * job.training.tail_sft.drop_fraction) >= count for count in counts):
                raise ValueError("TailSFT would remove a whole scheduled batch")
        complete = sft_run_records(store, output_dir).read("result.json") is not None
    costs = tuple(
        local_backend.conservative_step_cost(job.training, batch_example_count=n) for n in counts
    )
    if any(cost is None for cost in costs):
        raise ValueError("provide a conservative token price and maximum datum tokens")
    estimate = 0.0 if complete else sum(cost.value for cost in costs if cost is not None)
    if job.training.maximum_cost_usd is None or estimate > job.training.maximum_cost_usd:
        raise ValueError("scheduled work exceeds its frozen budget before credentials")
    console.print(f"{job.action}: {len(train_examples)} training examples; ceiling ${estimate:.6f}")
    if dry_run:
        console.print("Validated without credentials or provider calls.")
        return
    if not complete and not require_spend_consent(
        console,
        root=store.paths.root,
        yes=yes,
        estimated_cost_usd=estimate,
        command=f"exp optimize model {store.paths.project_id} --training-job {path}",
        non_interactive=non_interactive,
    ):
        return
    backend = (
        local_backend
        if complete
        else compose_backend(store, job.connection, config.connection_config_sha256)
    )
    if job.action == "score-initial":
        reference = saved or score_initial_training_losses(
            store,
            job.dataset,
            job.training,
            backend,
            created_at=created_at,
            code_revision=code_revision,
        )
        console.print_json(data=reference.model_dump(mode="json"))
        return
    result = train_tinker_sft(
        store,
        job.dataset.artifact_id,
        job.training,
        output_dir,
        backend=backend,
        created_at=created_at,
        code_revision=code_revision,
    )
    console.print_json(data=result.model_dump(mode="json"))
    console.print("Candidate saved. Task evaluation and promotion are separate operations.")
