"""Candidate-only cost and matched-denominator aggregation tests."""

from __future__ import annotations

import pytest

from exp.common.evaluations.dataset import EvaluationRow
from exp.common.evaluations.model_report import (
    ModelEvaluationMetrics,
    _comparable,
    _dominates,
    _metrics,
)
from exp.common.models import (
    BillingSource,
    ModelSnapshot,
    NumericMeasurement,
    RoutedCandidateSnapshot,
)


def _candidate(alias: str) -> RoutedCandidateSnapshot:
    """Construct an explicit isolated worker identity."""
    return RoutedCandidateSnapshot(
        alias=alias,
        model=ModelSnapshot(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            provider="test",
            model_id=alias,
            capabilities_sha256="a" * 64,
            connection_sha256="b" * 64,
        ),
    )


def _row(task: str, repeat: int, quality: float, cost: float) -> EvaluationRow:
    """Make one measured worker row with deliberately expensive simulation overhead."""
    return EvaluationRow(
        cell_id=f"cell-{task}-{repeat}",
        task_id=task,
        candidate_alias="worker",
        repeat=repeat,
        purpose="fit",
        protocol_id="protocol",
        source_run_id="run",
        status="completed",
        rollout_id=f"rollout-{task}-{repeat}",
        judgment_id=f"judgment-{task}-{repeat}",
        score=quality,
        candidate_cost_usd=NumericMeasurement(value=cost, provenance="observed"),
        world_model_cost_usd=NumericMeasurement(value=100, provenance="observed"),
    )


def test_weighted_metrics_exclude_environment_cost_and_do_not_overweight_repeats() -> None:
    """The plotted cost is worker operation cost, not the expense of creating the report."""
    rows = (_row("a", 0, 1, 0.1), _row("a", 1, 1, 0.1), _row("b", 0, 0, 0.01))
    indexed: dict[tuple[str, int, str], EvaluationRow] = {
        (row.task_id, row.repeat, row.purpose): row for row in rows
    }
    result = _metrics(_candidate("worker"), indexed, tuple(indexed), {"a": 0.5, "b": 0.5})
    assert result.quality == pytest.approx(0.5)
    assert result.operating_cost_usd == pytest.approx(0.055)
    assert result.latency_seconds is None


def test_empty_comparison_does_not_invent_zero_quality_or_cost() -> None:
    """An all-excluded worker has no plotted metrics."""
    row = _row("a", 0, 1, 0.1)
    result = _metrics(_candidate("worker"), {("a", 0, "fit"): row}, (), {"a": 1})
    assert result.quality is None
    assert result.operating_cost_usd is None
    assert not _dominates(result, result)


def test_pareto_dominance_preserves_ties_and_cost_quality_tradeoffs() -> None:
    """Only a strict improvement without any regression dominates another worker."""
    base = ModelEvaluationMetrics(
        candidate=_candidate("worker"),
        planned_cells=1,
        scored_cells=1,
        failed_cells=0,
        not_run_cells=0,
        compared_cells=1,
        quality=0.8,
        operating_cost_usd=0.1,
    )
    assert not _dominates(base, base)
    assert _dominates(base.model_copy(update={"operating_cost_usd": 0.05}), base)
    assert not _dominates(base.model_copy(update={"quality": 0.9, "operating_cost_usd": 0.2}), base)
    assert not _comparable(_row("a", 0, 1, 0.1).model_copy(update={"candidate_cost_usd": None}))
