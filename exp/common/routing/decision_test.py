"""Tests for pure fit-evidence helpers used by cache-aware sticky routing."""

from __future__ import annotations

import numpy as np
import pytest

from exp.common.routing.bank import KnnEvidenceBank
from exp.common.routing.decision import fitted_quality_gain


def _bank(scores: tuple[tuple[float, float], ...]) -> KnnEvidenceBank:
    """Build one minimal two-candidate bank around the given sparse score cells.

    Args:
        scores: Per-task ``(baseline, cheap)`` scores, using NaN for unscored cells.

    Returns:
        A validated bank with workload weights ``1, 1, 3`` over three fit tasks.
    """
    return KnnEvidenceBank(
        task_ids=("task-1", "task-2", "task-3"),
        candidate_aliases=("baseline", "cheap"),
        embeddings=np.asarray(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)), dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        candidate_costs=np.asarray(((0.5, 0.1),) * 3, dtype=np.float64),
        score_counts=np.ones((3, 2), dtype=np.int32),
        cost_counts=np.ones((3, 2), dtype=np.int32),
        workload_weights=np.asarray((1.0, 1.0, 3.0), dtype=np.float64),
        novelty_floor=0.0,
    )


def test_fitted_quality_gain_weights_jointly_scored_fit_tasks() -> None:
    """Only jointly scored tasks contribute, weighted by their workload weights."""
    bank = _bank(((0.2, 0.8), (0.5, float("nan")), (0.4, 0.9)))

    gain = fitted_quality_gain(bank, incumbent_alias="baseline", challenger_alias="cheap")

    assert gain == pytest.approx((1.0 * 0.6 + 3.0 * 0.5) / 4.0, abs=1e-6)
    inverse = fitted_quality_gain(bank, incumbent_alias="cheap", challenger_alias="baseline")
    assert inverse == pytest.approx(-(1.0 * 0.6 + 3.0 * 0.5) / 4.0, abs=1e-6)


def test_fitted_quality_gain_is_none_without_a_jointly_scored_task() -> None:
    """Disjoint score coverage yields no gain estimate instead of a silent zero."""
    bank = _bank(
        (
            (0.2, float("nan")),
            (float("nan"), 0.8),
            (0.3, float("nan")),
        )
    )

    assert fitted_quality_gain(bank, incumbent_alias="baseline", challenger_alias="cheap") is None
