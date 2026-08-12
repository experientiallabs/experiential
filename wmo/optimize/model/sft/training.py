"""Append-only offline Tinker SFT orchestration over immutable W12 datasets."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    assert_secret_free,
    canonical_json_bytes,
    sha256_json,
    stable_id,
)
from wmo.common.core.files import write_bytes_atomic
from wmo.common.core.locks import file_write_lock
from wmo.common.models import NumericMeasurement
from wmo.common.project import ProjectStore
from wmo.optimize.model.sft.builder import SFTBuildError, load_verified_sft_dataset
from wmo.optimize.model.sft.contracts import SFTDatasetArtifact, SFTExample
from wmo.optimize.model.sft.provider_resources import validate_provider_resource_id
from wmo.optimize.model.sft.rendering import partitioned_rows_sha256
from wmo.optimize.model.sft.training_contracts import (
    TINKER_SFT_EVENT_ADAPTER,
    TinkerSFTAmbiguousStepError,
    TinkerSFTBudgetExceeded,
    TinkerSFTCheckpoint,
    TinkerSFTCheckpointIntent,
    TinkerSFTError,
    TinkerSFTEvent,
    TinkerSFTMetric,
    TinkerSFTModelArtifact,
    TinkerSFTModelIntent,
    TinkerSFTResult,
    TinkerSFTResumeError,
    TinkerSFTResumeEvent,
    TinkerSFTRunManifest,
    TinkerSFTSpec,
    TinkerSFTStepIntent,
    TrainerBackend,
    TrainerDatum,
    TrainerSession,
)

_MANIFEST_FILE = "manifest.json"
_EVENTS_FILE = "events.jsonl"
_MODEL_FILE = "model.json"
_MODEL_INTENT_FILE = "model-intent.json"
_RESULT_FILE = "result.json"


class TinkerSFTOptimizer:
    """Concrete offline SFT optimizer composed with one injected Tinker-capable backend."""

    def __init__(self, backend: TrainerBackend) -> None:
        """Bind one caller-composed concrete adapter or deterministic test fake."""
        self._backend = backend

    def optimize(
        self,
        *,
        store: ProjectStore,
        dataset_id: ArtifactId,
        spec: TinkerSFTSpec,
        output_dir: Path,
        created_at: datetime,
        code_revision: str,
    ) -> TinkerSFTResult:
        """Train from one persisted dataset without changing catalog, router, or serving state."""
        return train_tinker_sft(
            store,
            dataset_id,
            spec,
            output_dir,
            backend=self._backend,
            created_at=created_at,
            code_revision=code_revision,
        )


def train_tinker_sft(
    store: ProjectStore,
    dataset_id: ArtifactId,
    spec: TinkerSFTSpec,
    output_dir: Path,
    *,
    backend: TrainerBackend,
    created_at: datetime,
    code_revision: str,
) -> TinkerSFTResult:
    """Train a managed LoRA from frozen W12 examples with append-only local provenance.

    Args:
        store: Project store owning the dataset and every transitive acceptance artifact.
        dataset_id: Persisted W12 dataset ID. Caller-constructed rows are never authoritative.
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
    try:
        dataset = load_verified_sft_dataset(store, dataset_id)
    except SFTBuildError as exc:
        raise TinkerSFTError(f"W12 dataset {dataset_id} is not safe for training: {exc}") from exc
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
    events = _read_events(output_dir, manifest.run_id)
    _assert_no_ambiguous_step_intent(output_dir, manifest.run_id, events)
    _assert_no_ambiguous_checkpoint_intent(output_dir, manifest.run_id, events)
    _assert_no_uncheckpointed_completed_steps(events)
    completed = _load_completed_result(output_dir, manifest, events)
    if completed is not None:
        return completed
    existing_model = _load_model(output_dir, manifest)
    if existing_model is not None:
        _validate_model_lineage(output_dir, existing_model, events)
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
    schedule_counts = _schedule_batch_counts(len(train_examples), spec)
    start_step = latest_checkpoint.step if latest_checkpoint is not None else 0
    if start_step > len(schedule_counts):
        raise TinkerSFTResumeError(
            "the latest Tinker SFT checkpoint is beyond this frozen training schedule"
        )
    if start_step < len(schedule_counts):
        _require_step_budget(
            backend=backend,
            spec=spec,
            batch_example_count=schedule_counts[start_step],
            current_cost=durable_cost,
        )
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
    current_cost = durable_cost
    for step, batch in enumerate(schedule[start_step:], start=start_step + 1):
        upper_bound = _require_step_budget(
            backend=backend,
            spec=spec,
            batch_example_count=len(batch.datums),
            current_cost=current_cost,
        )
        _write_new_json(
            _step_intent_path(output_dir, attempt_id, step),
            TinkerSFTStepIntent(
                run_id=manifest.run_id,
                attempt_id=attempt_id,
                step=step,
                epoch=batch.epoch,
                batch_index=batch.batch_index,
                example_ids=tuple(datum.example_id for datum in batch.datums),
                conservative_cost_upper_bound_usd=upper_bound,
            ),
            "Tinker SFT optimizer-step intent",
        )
        batch_result = session.train_batch(batch.datums, learning_rate=spec.learning_rate)
        current_cost = _add_cost(current_cost, batch_result.cost_usd, has_prior_metrics=step > 1)
        metric = TinkerSFTMetric(
            run_id=manifest.run_id,
            attempt_id=attempt_id,
            step=step,
            epoch=batch.epoch,
            batch_index=batch.batch_index,
            batch_example_count=len(batch.datums),
            example_ids=tuple(datum.example_id for datum in batch.datums),
            supervised_token_count=sum(datum.supervised_token_count for datum in batch.datums),
            loss=batch_result.loss,
            gradient_norm=batch_result.gradient_norm,
            usage=batch_result.usage,
            cost_usd=batch_result.cost_usd,
            cumulative_cost_usd=current_cost,
        )
        _append_event(output_dir, metric)
        events = (*events, metric)
        if upper_bound is not None and batch_result.cost_usd is not None:
            if batch_result.cost_usd.value > upper_bound.value:
                raise TinkerSFTBudgetExceeded(
                    "backend-reported step cost exceeded its conservative pre-dispatch bound; "
                    "the completed step is not replayable and requires manual review"
                )
        must_checkpoint = step % spec.checkpoint_every_steps == 0 or step == len(schedule)
        if spec.maximum_cost_usd is not None and current_cost is None:
            raise TinkerSFTBudgetExceeded(
                "a completed Tinker SFT step did not report exact cost; no later step may dispatch"
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
            event = TINKER_SFT_EVENT_ADAPTER.validate_json(line)
        except ValueError as exc:
            raise TinkerSFTResumeError(
                f"Tinker SFT event log has an invalid record at line {number}: {exc}"
            ) from exc
        if event.run_id != run_id:
            raise TinkerSFTResumeError("Tinker SFT event log names a different run")
        if isinstance(event, TinkerSFTCheckpoint):
            try:
                validate_provider_resource_id(event.state_path, label="checkpoint state")
            except TinkerSFTError as exc:
                raise TinkerSFTResumeError(
                    "Tinker SFT checkpoint has an unsafe resource ID"
                ) from exc
        events.append(event)
    _validate_event_history(tuple(events))
    return tuple(events)


def _validate_event_history(events: tuple[TinkerSFTEvent, ...]) -> None:
    checkpoints: dict[str, TinkerSFTCheckpoint] = {}
    metric_keys: set[tuple[int, int]] = set()
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
            metric_key = (event.attempt_id, event.step)
            if metric_key in metric_keys:
                raise TinkerSFTResumeError("Tinker SFT event log repeats a completed step")
            if event.batch_example_count != len(event.example_ids):
                raise TinkerSFTResumeError("Tinker SFT metric has an inconsistent batch size")
            metric_keys.add(metric_key)
            seen_attempts.add(event.attempt_id)


def _latest_checkpoint(events: Sequence[TinkerSFTEvent]) -> TinkerSFTCheckpoint | None:
    checkpoints = [event for event in events if isinstance(event, TinkerSFTCheckpoint)]
    return checkpoints[-1] if checkpoints else None


def _assert_no_uncheckpointed_completed_steps(events: Sequence[TinkerSFTEvent]) -> None:
    checkpoint = _latest_checkpoint(events)
    durable_step = 0 if checkpoint is None else checkpoint.step
    if any(isinstance(event, TinkerSFTMetric) and event.step > durable_step for event in events):
        raise TinkerSFTAmbiguousStepError(
            "a completed optimizer step has no durable checkpoint state; replay would duplicate "
            "remote work and spend, so manual recovery or a new run is required"
        )


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
    schedule_items: list[_ScheduledBatch] = []
    for epoch in range(1, spec.epochs + 1):
        shuffled = tuple(
            sorted(
                datums,
                key=lambda datum: hashlib.sha256(
                    f"{spec.seed}:{epoch}:{datum.example_id}".encode()
                ).hexdigest(),
            )
        )
        batches = tuple(
            shuffled[index : index + spec.batch_size]
            for index in range(0, len(shuffled), spec.batch_size)
        )
        schedule_items.extend(
            _ScheduledBatch(epoch=epoch, batch_index=batch_index, datums=batch)
            for batch_index, batch in enumerate(batches, start=1)
        )
    schedule = tuple(schedule_items)
    if spec.maximum_steps is not None:
        schedule = schedule[: spec.maximum_steps]
    if not schedule:
        raise TinkerSFTError("Tinker SFT schedule has no managed training batches")
    return schedule


def _schedule_batch_counts(example_count: int, spec: TinkerSFTSpec) -> tuple[int, ...]:
    one_epoch = tuple(
        min(spec.batch_size, example_count - index)
        for index in range(0, example_count, spec.batch_size)
    )
    counts = one_epoch * spec.epochs
    return counts[: spec.maximum_steps] if spec.maximum_steps is not None else counts


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


def _require_step_budget(
    *,
    backend: TrainerBackend,
    spec: TinkerSFTSpec,
    batch_example_count: int,
    current_cost: NumericMeasurement | None,
) -> NumericMeasurement | None:
    upper_bound = backend.conservative_step_cost(
        spec,
        batch_example_count=batch_example_count,
    )
    if upper_bound is not None and upper_bound.value < 0:
        raise TinkerSFTBudgetExceeded("backend cost upper bounds must be nonnegative")
    if spec.maximum_cost_usd is None:
        return upper_bound
    if upper_bound is None:
        raise TinkerSFTBudgetExceeded(
            "maximum_cost_usd requires a backend-proven conservative step cost before dispatch"
        )
    spent = 0.0 if current_cost is None else current_cost.value
    if spent + upper_bound.value > spec.maximum_cost_usd:
        raise TinkerSFTBudgetExceeded(
            "the next Tinker SFT step's conservative cost bound exceeds the remaining budget; "
            "no optimizer call was dispatched"
        )
    return upper_bound


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
    state_path = validate_provider_resource_id(
        session.save_state(checkpoint_name), label="checkpoint state"
    )
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
    sampling_handle = validate_provider_resource_id(
        session.save_sampling_handle(model_name), label="sampling handle"
    )
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
        events_sha256=_events_sha256(output_dir),
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
    existing = _load_completed_result(output_dir, manifest, events)
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
        events_sha256=_events_sha256(output_dir),
        final_checkpoint_id=final_checkpoint.checkpoint_id,
        training_step_count=final_checkpoint.step,
        training_metric_count=len(effective_metrics),
        checkpoint_count=sum(isinstance(event, TinkerSFTCheckpoint) for event in events),
        total_cost_usd=_total_cost(effective_metrics),
    )
    _write_new_json(output_dir / _RESULT_FILE, result, "Tinker SFT result")
    return result


def _load_completed_result(
    output_dir: Path,
    manifest: TinkerSFTRunManifest,
    events: Sequence[TinkerSFTEvent],
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
    _validate_model_lineage(output_dir, model, events)
    final_checkpoint = _latest_checkpoint(events)
    if final_checkpoint is None or result.final_checkpoint_id != final_checkpoint.checkpoint_id:
        raise TinkerSFTResumeError("Tinker SFT result does not match its final checkpoint event")
    effective_metrics = _metrics_at_checkpoint(events, final_checkpoint)
    if result.events_sha256 != _events_sha256(output_dir):
        raise TinkerSFTResumeError("Tinker SFT result does not hash the current event log")
    if result.events_sha256 != model.events_sha256:
        raise TinkerSFTResumeError("Tinker SFT result and model name different event logs")
    if result.training_step_count != final_checkpoint.step:
        raise TinkerSFTResumeError("Tinker SFT result has the wrong durable step count")
    if result.training_metric_count != len(effective_metrics):
        raise TinkerSFTResumeError("Tinker SFT result has the wrong durable metric count")
    if result.total_cost_usd != _total_cost(effective_metrics):
        raise TinkerSFTResumeError("Tinker SFT result has the wrong durable cost facts")
    if result.checkpoint_count != sum(isinstance(event, TinkerSFTCheckpoint) for event in events):
        raise TinkerSFTResumeError("Tinker SFT result has the wrong checkpoint count")
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
    try:
        validate_provider_resource_id(model.sampling_handle, label="sampling handle")
        validate_provider_resource_id(model.final_checkpoint_state_path, label="checkpoint state")
    except TinkerSFTError as exc:
        raise TinkerSFTResumeError("Tinker SFT model artifact has an unsafe resource ID") from exc
    return model


def _validate_model_lineage(
    output_dir: Path,
    model: TinkerSFTModelArtifact,
    events: Sequence[TinkerSFTEvent],
) -> None:
    final_checkpoint = _latest_checkpoint(events)
    if final_checkpoint is None:
        raise TinkerSFTResumeError("Tinker SFT model artifact has no durable checkpoint event")
    if model.events_sha256 != _events_sha256(output_dir):
        raise TinkerSFTResumeError("Tinker SFT model artifact does not hash the current event log")
    if model.final_checkpoint_id != final_checkpoint.checkpoint_id:
        raise TinkerSFTResumeError("Tinker SFT model artifact names a different final checkpoint")
    if model.final_checkpoint_state_path != final_checkpoint.state_path:
        raise TinkerSFTResumeError("Tinker SFT model artifact has a different checkpoint resource")
    metrics = _metrics_at_checkpoint(events, final_checkpoint)
    if model.training_step_count != final_checkpoint.step or model.training_metric_count != len(
        metrics
    ):
        raise TinkerSFTResumeError("Tinker SFT model artifact has inconsistent training counts")
    if model.total_cost_usd != _total_cost(metrics):
        raise TinkerSFTResumeError("Tinker SFT model artifact has inconsistent durable cost facts")


def _events_sha256(output_dir: Path) -> str:
    path = output_dir / _EVENTS_FILE
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TinkerSFTResumeError(f"Tinker SFT event log is required at {path}: {exc}") from exc
    if not payload:
        raise TinkerSFTResumeError("Tinker SFT event log is empty")
    return hashlib.sha256(payload).hexdigest()


def _assert_no_ambiguous_checkpoint_intent(
    output_dir: Path, run_id: ArtifactId, events: Sequence[TinkerSFTEvent]
) -> None:
    checkpoint_ids = {
        event.checkpoint_id for event in events if isinstance(event, TinkerSFTCheckpoint)
    }
    intent_ids: set[ArtifactId] = set()
    for path in output_dir.glob("*.checkpoint-intent.json"):
        intent = _read_model(path, TinkerSFTCheckpointIntent, "Tinker SFT checkpoint intent")
        if intent.run_id != run_id:
            raise TinkerSFTResumeError("Tinker SFT checkpoint intent names a different run")
        if intent.checkpoint_id not in checkpoint_ids:
            raise TinkerSFTResumeError(
                "a prior Tinker checkpoint save may have completed without an event record; "
                "inspect the provider before retrying"
            )
        intent_ids.add(intent.checkpoint_id)
    if checkpoint_ids != intent_ids:
        raise TinkerSFTResumeError("Tinker SFT checkpoint event is missing its pre-call intent")


def _assert_no_ambiguous_step_intent(
    output_dir: Path, run_id: ArtifactId, events: Sequence[TinkerSFTEvent]
) -> None:
    metrics = {
        (event.attempt_id, event.step): event
        for event in events
        if isinstance(event, TinkerSFTMetric)
    }
    intent_keys: set[tuple[int, int]] = set()
    for path in output_dir.glob("*.step-intent.json"):
        intent = _read_model(path, TinkerSFTStepIntent, "Tinker SFT optimizer-step intent")
        if intent.run_id != run_id:
            raise TinkerSFTResumeError("Tinker SFT optimizer-step intent names a different run")
        metric = metrics.get((intent.attempt_id, intent.step))
        if metric is None:
            raise TinkerSFTAmbiguousStepError(
                "a prior Tinker optimizer step was durably marked before dispatch but has no "
                "completion event; remote completion and spend are unknown, so the step will "
                "not be replayed. Manual provider inspection or a new run is required"
            )
        if metric.example_ids != intent.example_ids:
            raise TinkerSFTResumeError(
                "Tinker SFT optimizer-step intent does not match its completed metric batch"
            )
        intent_keys.add((intent.attempt_id, intent.step))
    if set(metrics) != intent_keys:
        raise TinkerSFTResumeError("Tinker SFT completed metric is missing its pre-dispatch intent")


def _step_intent_path(output_dir: Path, attempt_id: int, step: int) -> Path:
    return output_dir / f"attempt-{attempt_id}-step-{step}.step-intent.json"


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
