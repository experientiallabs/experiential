"""Append-only offline Tinker SFT orchestration over immutable W12 datasets."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

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
from wmo.common.models import NumericMeasurement, Usage
from wmo.optimize.model.sft.contracts import SFTDatasetArtifact, SFTExample
from wmo.optimize.model.sft.rendering import partitioned_rows_sha256

_MANIFEST_FILE = "manifest.json"
_EVENTS_FILE = "events.jsonl"
_MODEL_FILE = "model.json"
_MODEL_INTENT_FILE = "model-intent.json"
_RESULT_FILE = "result.json"


class TinkerSFTError(RuntimeError):
    """The frozen dataset or append-only local SFT run cannot proceed safely."""


class TinkerSFTResumeError(TinkerSFTError):
    """An existing run directory cannot be resumed without mixing immutable state."""


class TinkerSFTBudgetExceeded(TinkerSFTError):
    """Observed or required spend accounting prevents another managed training step."""


class TinkerSFTSpec(ContractModel):
    """Frozen base-model, LoRA, batch, checkpoint, and managed-spend settings."""

    base_model: str = Field(min_length=1, max_length=512)
    lora_rank: int = Field(default=32, gt=0, le=4096)
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    checkpoint_every_steps: int = Field(gt=0)
    maximum_steps: int | None = Field(default=None, gt=0)
    maximum_datum_tokens: int | None = Field(default=None, gt=0)
    maximum_cost_usd: float | None = Field(default=None, gt=0)

    @field_validator("learning_rate", "maximum_cost_usd")
    @classmethod
    def _require_finite_float(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Tinker SFT numeric settings must be finite")
        return value


class TrainerBatchResult(ContractModel):
    """Metrics returned by one backend-managed cross-entropy update, without fabrication."""

    loss: float | None = None
    gradient_norm: float | None = None
    usage: Usage | None = None
    cost_usd: NumericMeasurement | None = None

    @field_validator("loss", "gradient_norm")
    @classmethod
    def _require_finite_metric(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Tinker SFT backend metrics must be finite")
        return value


class TrainerDatum(Protocol):
    """The tiny datum identity slice retained after a backend renders frozen examples."""

    @property
    def example_id(self) -> str:
        """Return the exact frozen W12 example identifier rendered into this datum."""
        ...

    @property
    def supervised_token_count(self) -> int:
        """Return the non-zero cross-entropy token count after renderer truncation."""
        ...


class TrainerSession(Protocol):
    """One managed training client already created or restored by its backend."""

    def render_examples(self, examples: Sequence[SFTExample]) -> tuple[TrainerDatum, ...]:
        """Render complete frozen assistant-action targets into cross-entropy datums."""
        ...

    def train_batch(
        self, datums: Sequence[TrainerDatum], *, learning_rate: float
    ) -> TrainerBatchResult:
        """Accumulate and apply one managed cross-entropy batch."""
        ...

    def save_state(self, checkpoint_name: str) -> str:
        """Persist a resumable managed state and return its non-secret provider handle."""
        ...

    def save_sampling_handle(self, model_name: str) -> str:
        """Persist completed weights for later sampling and return the non-secret handle."""
        ...


class TrainerBackend(Protocol):
    """The injection boundary for the concrete Tinker SDK adapter or a test fake."""

    def open(self, spec: TinkerSFTSpec, resume_state_path: str | None) -> TrainerSession:
        """Create an uninitialized session, restoring state before renderer work when set."""
        ...


class TinkerSFTRunManifest(ArtifactEnvelope):
    """Immutable local identity for one append-only run directory."""

    run_id: ArtifactId
    dataset_id: ArtifactId
    dataset_build_sha256: Sha256
    dataset_examples_sha256: Sha256
    spec: TinkerSFTSpec
    spec_sha256: Sha256

    @model_validator(mode="after")
    def _require_exact_dataset_and_spec_bindings(self) -> TinkerSFTRunManifest:
        expected_inputs = (
            ArtifactInput(artifact_id=self.dataset_id, sha256=self.dataset_build_sha256),
        )
        if self.inputs != expected_inputs:
            raise ValueError("Tinker SFT manifests must name exactly their frozen dataset build")
        if self.spec_sha256 != sha256_json(self.spec):
            raise ValueError("Tinker SFT manifest spec digest does not match settings")
        return self


class TinkerSFTMetric(ContractModel):
    """One observed managed training batch, without a task-behavior or quality claim."""

    record_type: Literal["metric"] = "metric"
    run_id: ArtifactId
    attempt_id: int = Field(ge=1)
    step: int = Field(ge=1)
    epoch: int = Field(ge=1)
    batch_index: int = Field(ge=1)
    batch_example_count: int = Field(gt=0)
    supervised_token_count: int = Field(gt=0)
    loss: float | None = None
    gradient_norm: float | None = None
    usage: Usage | None = None
    cost_usd: NumericMeasurement | None = None
    cumulative_cost_usd: NumericMeasurement | None = None

    @field_validator("loss", "gradient_norm")
    @classmethod
    def _require_finite_metric(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Tinker SFT metrics must be finite")
        return value


class TinkerSFTCheckpoint(ContractModel):
    """One durable provider state tied to a completed prefix of frozen training batches."""

    record_type: Literal["checkpoint"] = "checkpoint"
    checkpoint_id: ArtifactId
    run_id: ArtifactId
    attempt_id: int = Field(ge=1)
    step: int = Field(ge=1)
    state_path: str = Field(min_length=1, max_length=2048)
    metric_count: int = Field(gt=0)
    cumulative_cost_usd: NumericMeasurement | None = None


class TinkerSFTResumeEvent(ContractModel):
    """An append-only record that a new attempt starts from a durable earlier checkpoint."""

    record_type: Literal["resume"] = "resume"
    run_id: ArtifactId
    attempt_id: int = Field(ge=2)
    checkpoint_id: ArtifactId
    resumed_from_step: int = Field(ge=1)


class TinkerSFTCheckpointIntent(ContractModel):
    """A durable pre-call marker that prevents an ambiguous checkpoint save from being repeated."""

    run_id: ArtifactId
    checkpoint_id: ArtifactId
    attempt_id: int = Field(ge=1)
    step: int = Field(ge=1)
    checkpoint_name: str = Field(min_length=1, max_length=512)


class TinkerSFTModelIntent(ContractModel):
    """A durable pre-call marker for the non-idempotent terminal sampling-handle save."""

    run_id: ArtifactId
    model_name: str = Field(min_length=1, max_length=512)
    checkpoint_id: ArtifactId


class TinkerSFTModelArtifact(ArtifactEnvelope):
    """Completed sampling handle plus the frozen dataset and durable checkpoint that produced it."""

    model_id: ArtifactId
    run_id: ArtifactId
    dataset_id: ArtifactId
    dataset_build_sha256: Sha256
    final_checkpoint_id: ArtifactId
    final_checkpoint_state_path: str = Field(min_length=1, max_length=2048)
    sampling_handle: str = Field(min_length=1, max_length=2048)
    training_step_count: int = Field(gt=0)
    training_metric_count: int = Field(gt=0)
    total_cost_usd: NumericMeasurement | None = None

    @model_validator(mode="after")
    def _require_exact_dataset_binding(self) -> TinkerSFTModelArtifact:
        expected_inputs = (
            ArtifactInput(artifact_id=self.dataset_id, sha256=self.dataset_build_sha256),
        )
        if self.inputs != expected_inputs:
            raise ValueError("Tinker SFT models must name exactly their frozen dataset build")
        return self


class TinkerSFTResult(ArtifactEnvelope):
    """Terminal local result that names the completed handle artifact and training facts."""

    result_id: ArtifactId
    run_id: ArtifactId
    dataset_id: ArtifactId
    dataset_build_sha256: Sha256
    model_id: ArtifactId
    model_sha256: Sha256
    training_step_count: int = Field(gt=0)
    training_metric_count: int = Field(gt=0)
    checkpoint_count: int = Field(gt=0)
    total_cost_usd: NumericMeasurement | None = None

    @model_validator(mode="after")
    def _require_exact_terminal_inputs(self) -> TinkerSFTResult:
        expected_inputs = tuple(
            sorted(
                (
                    ArtifactInput(artifact_id=self.dataset_id, sha256=self.dataset_build_sha256),
                    ArtifactInput(artifact_id=self.model_id, sha256=self.model_sha256),
                ),
                key=lambda item: item.artifact_id,
            )
        )
        if self.inputs != expected_inputs:
            raise ValueError("Tinker SFT results must name their dataset and terminal model")
        return self


type TinkerSFTEvent = Annotated[
    TinkerSFTMetric | TinkerSFTCheckpoint | TinkerSFTResumeEvent,
    Field(discriminator="record_type"),
]
_EVENT_ADAPTER = TypeAdapter(TinkerSFTEvent)


class TinkerSFTOptimizer:
    """Concrete offline SFT optimizer composed with one injected Tinker-capable backend."""

    def __init__(self, backend: TrainerBackend) -> None:
        """Bind one caller-composed concrete adapter or deterministic test fake."""
        self._backend = backend

    def optimize(
        self,
        *,
        dataset: SFTDatasetArtifact,
        spec: TinkerSFTSpec,
        output_dir: Path,
        created_at: datetime,
        code_revision: str,
    ) -> TinkerSFTResult:
        """Train from one frozen dataset without changing router, catalog, or serving state."""
        return train_tinker_sft(
            dataset,
            spec,
            output_dir,
            backend=self._backend,
            created_at=created_at,
            code_revision=code_revision,
        )


def train_tinker_sft(
    dataset: SFTDatasetArtifact,
    spec: TinkerSFTSpec,
    output_dir: Path,
    *,
    backend: TrainerBackend,
    created_at: datetime,
    code_revision: str,
) -> TinkerSFTResult:
    """Train a managed LoRA from frozen W12 examples with append-only local provenance.

    Args:
        dataset: W12 immutable artifact with accepted train and held-out example rows.
        spec: Base model, LoRA, optimization, budget, and checkpoint settings.
        output_dir: Append-only local run directory.
        backend: Concrete Tinker SDK adapter or a deterministic injected fake.
        created_at: Time recorded when this output directory is first initialized.
        code_revision: Exact revision recorded when this output directory is first initialized.

    Returns:
        Completed model-handle and result artifacts containing only training and checkpoint facts.

    Raises:
        TinkerSFTError: Dataset, budget, or append-only resume state is unsafe to continue.
    """
    _validate_run_inputs(dataset, created_at=created_at, code_revision=code_revision)
    manifest_path = output_dir / _MANIFEST_FILE
    with file_write_lock(manifest_path, what="the Tinker SFT run"):
        return _train_locked(
            dataset=dataset,
            spec=spec,
            output_dir=output_dir,
            backend=backend,
            created_at=created_at,
            code_revision=code_revision,
        )


def _train_locked(
    *,
    dataset: SFTDatasetArtifact,
    spec: TinkerSFTSpec,
    output_dir: Path,
    backend: TrainerBackend,
    created_at: datetime,
    code_revision: str,
) -> TinkerSFTResult:
    manifest = _load_or_create_manifest(
        dataset=dataset,
        spec=spec,
        output_dir=output_dir,
        created_at=created_at,
        code_revision=code_revision,
    )
    completed = _load_completed_result(output_dir, manifest)
    if completed is not None:
        return completed
    events = _read_events(output_dir, manifest.run_id)
    _assert_no_ambiguous_checkpoint_intent(output_dir, manifest.run_id, events)
    existing_model = _load_model(output_dir, manifest)
    if existing_model is not None:
        return _write_terminal_result(
            output_dir=output_dir,
            manifest=manifest,
            model=existing_model,
            events=events,
        )
    latest_checkpoint = _latest_checkpoint(events)
    durable_metrics = _metrics_at_checkpoint(events, latest_checkpoint)
    durable_cost = _total_cost(durable_metrics)
    _require_budget_can_continue(spec, durable_metrics, durable_cost)

    train_examples = tuple(row.example for row in dataset.rows if row.partition == "train")
    session = backend.open(
        spec,
        latest_checkpoint.state_path if latest_checkpoint is not None else None,
    )
    attempt_id = _next_attempt_id(events)
    if latest_checkpoint is not None:
        _append_event(
            output_dir,
            TinkerSFTResumeEvent(
                run_id=manifest.run_id,
                attempt_id=attempt_id,
                checkpoint_id=latest_checkpoint.checkpoint_id,
                resumed_from_step=latest_checkpoint.step,
            ),
        )
        events = (*events, _read_last_event(output_dir))
    datums = session.render_examples(train_examples)
    _validate_rendered_datums(datums, train_examples)
    schedule = _build_schedule(datums, spec)
    start_step = latest_checkpoint.step if latest_checkpoint is not None else 0
    if start_step > len(schedule):
        raise TinkerSFTResumeError(
            "the latest Tinker SFT checkpoint is beyond this frozen training schedule"
        )
    current_cost = durable_cost
    for step, batch in enumerate(schedule[start_step:], start=start_step + 1):
        batch_result = session.train_batch(batch.datums, learning_rate=spec.learning_rate)
        current_cost = _add_cost(current_cost, batch_result.cost_usd, has_prior_metrics=step > 1)
        metric = TinkerSFTMetric(
            run_id=manifest.run_id,
            attempt_id=attempt_id,
            step=step,
            epoch=batch.epoch,
            batch_index=batch.batch_index,
            batch_example_count=len(batch.datums),
            supervised_token_count=sum(datum.supervised_token_count for datum in batch.datums),
            loss=batch_result.loss,
            gradient_norm=batch_result.gradient_norm,
            usage=batch_result.usage,
            cost_usd=batch_result.cost_usd,
            cumulative_cost_usd=current_cost,
        )
        _append_event(output_dir, metric)
        events = (*events, metric)
        must_checkpoint = step % spec.checkpoint_every_steps == 0 or step == len(schedule)
        if spec.maximum_cost_usd is not None:
            if current_cost is None:
                checkpoint = _save_checkpoint(
                    session=session,
                    output_dir=output_dir,
                    manifest=manifest,
                    events=events,
                    attempt_id=attempt_id,
                    step=step,
                    cumulative_cost_usd=current_cost,
                )
                events = (*events, checkpoint)
                raise TinkerSFTBudgetExceeded(
                    "maximum_cost_usd requires every completed Tinker SFT batch to report cost"
                )
            if current_cost.value >= spec.maximum_cost_usd:
                checkpoint = _save_checkpoint(
                    session=session,
                    output_dir=output_dir,
                    manifest=manifest,
                    events=events,
                    attempt_id=attempt_id,
                    step=step,
                    cumulative_cost_usd=current_cost,
                )
                events = (*events, checkpoint)
                raise TinkerSFTBudgetExceeded(
                    "observed Tinker SFT spend reached maximum_cost_usd; resume requires a larger "
                    "explicit budget"
                )
        if must_checkpoint:
            checkpoint = _save_checkpoint(
                session=session,
                output_dir=output_dir,
                manifest=manifest,
                events=events,
                attempt_id=attempt_id,
                step=step,
                cumulative_cost_usd=current_cost,
            )
            events = (*events, checkpoint)
    final_checkpoint = _latest_checkpoint(events)
    if final_checkpoint is None or final_checkpoint.step != len(schedule):
        raise TinkerSFTError("completed Tinker SFT training has no final durable checkpoint")
    model = _save_or_load_terminal_model(
        session=session,
        output_dir=output_dir,
        manifest=manifest,
        final_checkpoint=final_checkpoint,
        events=events,
    )
    return _write_terminal_result(
        output_dir=output_dir,
        manifest=manifest,
        model=model,
        events=events,
    )


def _validate_run_inputs(
    dataset: SFTDatasetArtifact, *, created_at: datetime, code_revision: str
) -> None:
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


def _load_or_create_manifest(
    *,
    dataset: SFTDatasetArtifact,
    spec: TinkerSFTSpec,
    output_dir: Path,
    created_at: datetime,
    code_revision: str,
) -> TinkerSFTRunManifest:
    path = output_dir / _MANIFEST_FILE
    if path.exists():
        manifest = _read_model(path, TinkerSFTRunManifest, "Tinker SFT manifest")
        _validate_manifest(manifest, dataset=dataset, spec=spec, code_revision=code_revision)
        return manifest
    dataset_input = ArtifactInput(
        artifact_id=dataset.dataset.dataset_id,
        sha256=dataset.dataset.build_sha256,
    )
    spec_sha256 = sha256_json(spec)
    run_id = stable_id(
        "tinker-sft-run",
        {
            "dataset_id": dataset.dataset.dataset_id,
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
        dataset_build_sha256=dataset.dataset.build_sha256,
        dataset_examples_sha256=dataset.dataset.examples_sha256,
        spec=spec,
        spec_sha256=spec_sha256,
    )
    _write_new_json(path, manifest, "Tinker SFT manifest")
    return manifest


def _validate_manifest(
    manifest: TinkerSFTRunManifest,
    *,
    dataset: SFTDatasetArtifact,
    spec: TinkerSFTSpec,
    code_revision: str,
) -> None:
    if manifest.dataset_id != dataset.dataset.dataset_id:
        raise TinkerSFTResumeError("existing Tinker SFT run belongs to a different W12 dataset")
    if manifest.dataset_build_sha256 != dataset.dataset.build_sha256:
        raise TinkerSFTResumeError("existing Tinker SFT run has a different W12 dataset build")
    if manifest.dataset_examples_sha256 != dataset.dataset.examples_sha256:
        raise TinkerSFTResumeError("existing Tinker SFT run has different W12 example rows")
    if manifest.spec != spec:
        raise TinkerSFTResumeError("existing Tinker SFT run has a different training spec")
    if manifest.code_revision != code_revision:
        raise TinkerSFTResumeError("existing Tinker SFT run has a different code revision")


def _read_events(output_dir: Path, run_id: ArtifactId) -> tuple[TinkerSFTEvent, ...]:
    path = output_dir / _EVENTS_FILE
    if not path.exists():
        return ()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TinkerSFTResumeError(f"cannot read Tinker SFT events at {path}: {exc}") from exc
    events: list[TinkerSFTEvent] = []
    for number, line in enumerate(lines, start=1):
        if not line:
            raise TinkerSFTResumeError(f"Tinker SFT event log has an empty line at {number}")
        try:
            event = _EVENT_ADAPTER.validate_json(line)
        except ValueError as exc:
            raise TinkerSFTResumeError(
                f"Tinker SFT event log has an invalid record at line {number}: {exc}"
            ) from exc
        if event.run_id != run_id:
            raise TinkerSFTResumeError("Tinker SFT event log names a different run")
        events.append(event)
    _validate_event_history(tuple(events))
    return tuple(events)


def _validate_event_history(events: tuple[TinkerSFTEvent, ...]) -> None:
    checkpoints: dict[str, TinkerSFTCheckpoint] = {}
    seen_attempts: set[int] = set()
    for event in events:
        if isinstance(event, TinkerSFTCheckpoint):
            if event.checkpoint_id in checkpoints:
                raise TinkerSFTResumeError("Tinker SFT event log repeats a checkpoint identifier")
            checkpoints[event.checkpoint_id] = event
            seen_attempts.add(event.attempt_id)
        elif isinstance(event, TinkerSFTResumeEvent):
            checkpoint = checkpoints.get(event.checkpoint_id)
            if checkpoint is None:
                raise TinkerSFTResumeError("Tinker SFT resume event names an unknown checkpoint")
            if checkpoint.step != event.resumed_from_step:
                raise TinkerSFTResumeError("Tinker SFT resume event has the wrong checkpoint step")
            if event.attempt_id in seen_attempts:
                raise TinkerSFTResumeError("Tinker SFT event log reuses an attempt identifier")
            seen_attempts.add(event.attempt_id)
        else:
            seen_attempts.add(event.attempt_id)


def _latest_checkpoint(events: Sequence[TinkerSFTEvent]) -> TinkerSFTCheckpoint | None:
    checkpoints = [event for event in events if isinstance(event, TinkerSFTCheckpoint)]
    return checkpoints[-1] if checkpoints else None


def _metrics_at_checkpoint(
    events: Sequence[TinkerSFTEvent], checkpoint: TinkerSFTCheckpoint | None
) -> tuple[TinkerSFTMetric, ...]:
    if checkpoint is None:
        return ()
    checkpoints = {
        event.checkpoint_id: event for event in events if isinstance(event, TinkerSFTCheckpoint)
    }
    resume_events = {
        event.attempt_id: event for event in events if isinstance(event, TinkerSFTResumeEvent)
    }
    metrics_by_attempt: dict[int, list[TinkerSFTMetric]] = {}
    for event in events:
        if isinstance(event, TinkerSFTMetric):
            metrics_by_attempt.setdefault(event.attempt_id, []).append(event)

    def collect(attempt_id: int, through_step: int) -> tuple[TinkerSFTMetric, ...]:
        resume = resume_events.get(attempt_id)
        start_step = 0
        inherited: tuple[TinkerSFTMetric, ...] = ()
        if resume is not None:
            parent = checkpoints[resume.checkpoint_id]
            start_step = parent.step
            inherited = collect(parent.attempt_id, parent.step)
        local = tuple(
            metric
            for metric in metrics_by_attempt.get(attempt_id, [])
            if start_step < metric.step <= through_step
        )
        expected_steps = tuple(range(start_step + 1, through_step + 1))
        if tuple(metric.step for metric in local) != expected_steps:
            raise TinkerSFTResumeError(
                "Tinker SFT checkpoint lacks a complete metric lineage for its durable state"
            )
        return (*inherited, *local)

    return collect(checkpoint.attempt_id, checkpoint.step)


def _next_attempt_id(events: Sequence[TinkerSFTEvent]) -> int:
    return max((event.attempt_id for event in events), default=0) + 1


def _build_schedule(
    datums: tuple[TrainerDatum, ...], spec: TinkerSFTSpec
) -> tuple[_ScheduledBatch, ...]:
    one_epoch = tuple(
        datums[index : index + spec.batch_size] for index in range(0, len(datums), spec.batch_size)
    )
    schedule = tuple(
        _ScheduledBatch(epoch=epoch, batch_index=batch_index, datums=batch)
        for epoch in range(1, spec.epochs + 1)
        for batch_index, batch in enumerate(one_epoch, start=1)
    )
    if spec.maximum_steps is not None:
        schedule = schedule[: spec.maximum_steps]
    if not schedule:
        raise TinkerSFTError("Tinker SFT schedule has no managed training batches")
    return schedule


class _ScheduledBatch:
    def __init__(self, *, epoch: int, batch_index: int, datums: Sequence[TrainerDatum]) -> None:
        self.epoch = epoch
        self.batch_index = batch_index
        self.datums = tuple(datums)


def _validate_rendered_datums(
    datums: tuple[TrainerDatum, ...], examples: Sequence[SFTExample]
) -> None:
    example_ids = tuple(example.example_id for example in examples)
    datum_ids = tuple(datum.example_id for datum in datums)
    if datum_ids != example_ids:
        raise TinkerSFTError("Tinker SFT backend rendered a different ordered train-example set")
    if any(datum.supervised_token_count <= 0 for datum in datums):
        raise TinkerSFTError("Tinker SFT backend produced a datum without supervised target tokens")


def _add_cost(
    current: NumericMeasurement | None,
    added: NumericMeasurement | None,
    *,
    has_prior_metrics: bool,
) -> NumericMeasurement | None:
    if added is None:
        return None
    if current is None:
        if has_prior_metrics:
            return None
        return added
    provenance: Literal["observed", "estimated"] = (
        "observed"
        if current.provenance == "observed" and added.provenance == "observed"
        else "estimated"
    )
    return NumericMeasurement(value=current.value + added.value, provenance=provenance)


def _total_cost(metrics: Sequence[TinkerSFTMetric]) -> NumericMeasurement | None:
    if not metrics:
        return None
    total: NumericMeasurement | None = None
    for index, metric in enumerate(metrics):
        total = _add_cost(total, metric.cost_usd, has_prior_metrics=index > 0)
        if total is None:
            return None
    return total


def _require_budget_can_continue(
    spec: TinkerSFTSpec,
    metrics: Sequence[TinkerSFTMetric],
    cost: NumericMeasurement | None,
) -> None:
    if spec.maximum_cost_usd is None:
        return
    if metrics and cost is None:
        raise TinkerSFTBudgetExceeded(
            "maximum_cost_usd cannot resume because a prior completed Tinker SFT batch lacked cost"
        )
    if cost is not None and cost.value >= spec.maximum_cost_usd:
        raise TinkerSFTBudgetExceeded(
            "observed Tinker SFT spend already reached maximum_cost_usd; no further provider call "
            "was made"
        )


def _save_checkpoint(
    *,
    session: TrainerSession,
    output_dir: Path,
    manifest: TinkerSFTRunManifest,
    events: Sequence[TinkerSFTEvent],
    attempt_id: int,
    step: int,
    cumulative_cost_usd: NumericMeasurement | None,
) -> TinkerSFTCheckpoint:
    checkpoint_id = stable_id(
        "tinker-sft-checkpoint",
        {"run_id": manifest.run_id, "attempt_id": attempt_id, "step": step},
    )
    checkpoint_name = f"{manifest.run_id}-attempt-{attempt_id}-step-{step}"
    intent = TinkerSFTCheckpointIntent(
        run_id=manifest.run_id,
        checkpoint_id=checkpoint_id,
        attempt_id=attempt_id,
        step=step,
        checkpoint_name=checkpoint_name,
    )
    _write_new_json(
        _checkpoint_intent_path(output_dir, checkpoint_id), intent, "Tinker SFT checkpoint intent"
    )
    state_path = session.save_state(checkpoint_name)
    checkpoint = TinkerSFTCheckpoint(
        checkpoint_id=checkpoint_id,
        run_id=manifest.run_id,
        attempt_id=attempt_id,
        step=step,
        state_path=state_path,
        metric_count=step,
        cumulative_cost_usd=cumulative_cost_usd,
    )
    _append_event(output_dir, checkpoint)
    return checkpoint


def _save_or_load_terminal_model(
    *,
    session: TrainerSession,
    output_dir: Path,
    manifest: TinkerSFTRunManifest,
    final_checkpoint: TinkerSFTCheckpoint,
    events: Sequence[TinkerSFTEvent],
) -> TinkerSFTModelArtifact:
    existing_model = _load_model(output_dir, manifest)
    if existing_model is not None:
        return existing_model
    intent_path = output_dir / _MODEL_INTENT_FILE
    if intent_path.exists():
        intent = _read_model(intent_path, TinkerSFTModelIntent, "Tinker SFT model intent")
        if intent.run_id != manifest.run_id:
            raise TinkerSFTResumeError("Tinker SFT model intent names a different run")
        raise TinkerSFTResumeError(
            "a prior terminal Tinker sampling-handle save may have completed without a local model "
            "artifact; inspect the provider before retrying"
        )
    model_name = f"{manifest.run_id}-final"
    _write_new_json(
        intent_path,
        TinkerSFTModelIntent(
            run_id=manifest.run_id,
            model_name=model_name,
            checkpoint_id=final_checkpoint.checkpoint_id,
        ),
        "Tinker SFT model intent",
    )
    sampling_handle = session.save_sampling_handle(model_name)
    effective_metrics = _metrics_at_checkpoint(events, final_checkpoint)
    model_id = stable_id(
        "tinker-sft-model",
        {
            "run_id": manifest.run_id,
            "checkpoint_id": final_checkpoint.checkpoint_id,
            "sampling_handle": sampling_handle,
        },
    )
    model = TinkerSFTModelArtifact(
        schema_version=1,
        created_at=manifest.created_at,
        inputs=(
            ArtifactInput(
                artifact_id=manifest.dataset_id,
                sha256=manifest.dataset_build_sha256,
            ),
        ),
        code_revision=manifest.code_revision,
        model_id=model_id,
        run_id=manifest.run_id,
        dataset_id=manifest.dataset_id,
        dataset_build_sha256=manifest.dataset_build_sha256,
        final_checkpoint_id=final_checkpoint.checkpoint_id,
        final_checkpoint_state_path=final_checkpoint.state_path,
        sampling_handle=sampling_handle,
        training_step_count=final_checkpoint.step,
        training_metric_count=len(effective_metrics),
        total_cost_usd=_total_cost(effective_metrics),
    )
    _write_new_json(output_dir / _MODEL_FILE, model, "Tinker SFT model artifact")
    return model


def _write_terminal_result(
    *,
    output_dir: Path,
    manifest: TinkerSFTRunManifest,
    model: TinkerSFTModelArtifact,
    events: Sequence[TinkerSFTEvent],
) -> TinkerSFTResult:
    existing = _load_completed_result(output_dir, manifest)
    if existing is not None:
        return existing
    final_checkpoint = _latest_checkpoint(events)
    if final_checkpoint is None:
        raise TinkerSFTResumeError("Tinker SFT model artifact has no checkpoint event")
    effective_metrics = _metrics_at_checkpoint(events, final_checkpoint)
    model_sha256 = sha256_json(model)
    result = TinkerSFTResult(
        schema_version=1,
        created_at=manifest.created_at,
        inputs=tuple(
            sorted(
                (
                    ArtifactInput(
                        artifact_id=manifest.dataset_id,
                        sha256=manifest.dataset_build_sha256,
                    ),
                    ArtifactInput(artifact_id=model.model_id, sha256=model_sha256),
                ),
                key=lambda item: item.artifact_id,
            )
        ),
        code_revision=manifest.code_revision,
        result_id=stable_id(
            "tinker-sft-result",
            {"run_id": manifest.run_id, "model_sha256": model_sha256},
        ),
        run_id=manifest.run_id,
        dataset_id=manifest.dataset_id,
        dataset_build_sha256=manifest.dataset_build_sha256,
        model_id=model.model_id,
        model_sha256=model_sha256,
        training_step_count=final_checkpoint.step,
        training_metric_count=len(effective_metrics),
        checkpoint_count=sum(isinstance(event, TinkerSFTCheckpoint) for event in events),
        total_cost_usd=_total_cost(effective_metrics),
    )
    _write_new_json(output_dir / _RESULT_FILE, result, "Tinker SFT result")
    return result


def _load_completed_result(
    output_dir: Path, manifest: TinkerSFTRunManifest
) -> TinkerSFTResult | None:
    path = output_dir / _RESULT_FILE
    if not path.exists():
        return None
    result = _read_model(path, TinkerSFTResult, "Tinker SFT result")
    if result.run_id != manifest.run_id:
        raise TinkerSFTResumeError("Tinker SFT result names a different run")
    if result.dataset_id != manifest.dataset_id:
        raise TinkerSFTResumeError("Tinker SFT result names a different dataset")
    model = _load_model(output_dir, manifest)
    if model is None:
        raise TinkerSFTResumeError("Tinker SFT result exists without its model artifact")
    if result.model_id != model.model_id or result.model_sha256 != sha256_json(model):
        raise TinkerSFTResumeError("Tinker SFT result does not match its model artifact")
    return result


def _load_model(output_dir: Path, manifest: TinkerSFTRunManifest) -> TinkerSFTModelArtifact | None:
    path = output_dir / _MODEL_FILE
    if not path.exists():
        return None
    model = _read_model(path, TinkerSFTModelArtifact, "Tinker SFT model artifact")
    if model.run_id != manifest.run_id:
        raise TinkerSFTResumeError("Tinker SFT model artifact names a different run")
    if model.dataset_id != manifest.dataset_id:
        raise TinkerSFTResumeError("Tinker SFT model artifact names a different dataset")
    if model.dataset_build_sha256 != manifest.dataset_build_sha256:
        raise TinkerSFTResumeError("Tinker SFT model artifact has a different dataset build")
    return model


def _assert_no_ambiguous_checkpoint_intent(
    output_dir: Path, run_id: ArtifactId, events: Sequence[TinkerSFTEvent]
) -> None:
    checkpoint_ids = {
        event.checkpoint_id for event in events if isinstance(event, TinkerSFTCheckpoint)
    }
    for path in output_dir.glob("*.checkpoint-intent.json"):
        intent = _read_model(path, TinkerSFTCheckpointIntent, "Tinker SFT checkpoint intent")
        if intent.run_id != run_id:
            raise TinkerSFTResumeError("Tinker SFT checkpoint intent names a different run")
        if intent.checkpoint_id not in checkpoint_ids:
            raise TinkerSFTResumeError(
                "a prior Tinker checkpoint save may have completed without an event record; "
                "inspect the provider before retrying"
            )


def _checkpoint_intent_path(output_dir: Path, checkpoint_id: ArtifactId) -> Path:
    return output_dir / f"{checkpoint_id}.checkpoint-intent.json"


def _append_event(output_dir: Path, event: TinkerSFTEvent) -> None:
    assert_secret_free(event)
    path = output_dir / _EVENTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(event) + b"\n"
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _read_last_event(output_dir: Path) -> TinkerSFTEvent:
    events = _read_events(
        output_dir,
        _read_model(
            output_dir / _MANIFEST_FILE, TinkerSFTRunManifest, "Tinker SFT manifest"
        ).run_id,
    )
    if not events:
        raise TinkerSFTResumeError("Tinker SFT resume event was not persisted")
    return events[-1]


def _write_new_json(path: Path, value: BaseModel, label: str) -> None:
    if path.exists():
        raise TinkerSFTResumeError(
            f"{label} already exists at {path}; append-only runs do not replace it"
        )
    assert_secret_free(value)
    write_bytes_atomic(path, canonical_json_bytes(value) + b"\n")


def _read_model[ModelT: BaseModel](path: Path, model_type: type[ModelT], label: str) -> ModelT:
    try:
        return model_type.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise TinkerSFTResumeError(f"cannot read {label} at {path}: {exc}") from exc
