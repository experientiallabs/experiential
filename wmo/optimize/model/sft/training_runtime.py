"""Deterministic scheduling and conservative cost helpers for offline SFT."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Literal

from wmo.common.core.artifacts import ArtifactId, stable_id
from wmo.common.models import NumericMeasurement
from wmo.optimize.model.sft.contracts import SFTExample
from wmo.optimize.model.sft.training_contracts import (
    TinkerSFTBudgetExceeded,
    TinkerSFTCheckpoint,
    TinkerSFTError,
    TinkerSFTEvent,
    TinkerSFTMetric,
    TinkerSFTResumeError,
    TinkerSFTSpec,
    TrainerBackend,
    TrainerDatum,
)


class _ScheduledBatch:
    def __init__(self, *, epoch: int, batch_index: int, datums: Sequence[TrainerDatum]) -> None:
        self.epoch = epoch
        self.batch_index = batch_index
        self.datums = tuple(datums)


class _ExpectedBatch:
    def __init__(self, *, epoch: int, batch_index: int, example_ids: Sequence[ArtifactId]) -> None:
        self.epoch = epoch
        self.batch_index = batch_index
        self.example_ids = tuple(example_ids)


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


def _expected_schedule(
    examples: Sequence[SFTExample], spec: TinkerSFTSpec
) -> tuple[_ExpectedBatch, ...]:
    schedule: list[_ExpectedBatch] = []
    for epoch in range(1, spec.epochs + 1):
        shuffled_ids = tuple(
            sorted(
                (example.example_id for example in examples),
                key=lambda example_id: hashlib.sha256(
                    f"{spec.seed}:{epoch}:{example_id}".encode()
                ).hexdigest(),
            )
        )
        batches = tuple(
            shuffled_ids[index : index + spec.batch_size]
            for index in range(0, len(shuffled_ids), spec.batch_size)
        )
        schedule.extend(
            _ExpectedBatch(epoch=epoch, batch_index=batch_index, example_ids=batch)
            for batch_index, batch in enumerate(batches, start=1)
        )
    expected = tuple(schedule)
    if spec.maximum_steps is not None:
        expected = expected[: spec.maximum_steps]
    if not expected:
        raise TinkerSFTError("Tinker SFT schedule has no managed training batches")
    return expected


def _validate_event_schedule(
    events: Sequence[TinkerSFTEvent], expected_schedule: Sequence[_ExpectedBatch]
) -> None:
    for event in events:
        if isinstance(event, TinkerSFTMetric):
            if event.step > len(expected_schedule):
                raise TinkerSFTResumeError("Tinker SFT metric is beyond the frozen schedule")
            expected = expected_schedule[event.step - 1]
            if (
                event.epoch != expected.epoch
                or event.batch_index != expected.batch_index
                or event.example_ids != expected.example_ids
                or event.batch_example_count != len(expected.example_ids)
            ):
                raise TinkerSFTResumeError(
                    "Tinker SFT metric does not match its frozen scheduled batch"
                )
        elif isinstance(event, TinkerSFTCheckpoint):
            expected_checkpoint_id = stable_id(
                "tinker-sft-checkpoint",
                {"run_id": event.run_id, "attempt_id": event.attempt_id, "step": event.step},
            )
            if event.checkpoint_id != expected_checkpoint_id:
                raise TinkerSFTResumeError("Tinker SFT checkpoint ID is not content-addressed")
            if event.step > len(expected_schedule) or event.metric_count != event.step:
                raise TinkerSFTResumeError(
                    "Tinker SFT checkpoint does not match the frozen schedule prefix"
                )


def _schedule_batch_counts(example_count: int, spec: TinkerSFTSpec) -> tuple[int, ...]:
    one_epoch = tuple(
        min(spec.batch_size, example_count - index)
        for index in range(0, example_count, spec.batch_size)
    )
    counts = one_epoch * spec.epochs
    return counts[: spec.maximum_steps] if spec.maximum_steps is not None else counts


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
