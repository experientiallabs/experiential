"""Tests for the evaluator-neutral scoring contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmh.harness.doc import HarnessDoc
from wmh.harness.scoring import (
    RewardMode,
    ScoreCell,
    ScoreReport,
    ScoreRequest,
    reward_passed,
)


def _cell(task_id: str, attempt: int, reward: float, mode: RewardMode = "raw") -> ScoreCell:
    return ScoreCell(
        task_id=task_id,
        attempt=attempt,
        reward=reward,
        passed=reward_passed(reward, mode),
        artifact_dir=f"/jobs/{task_id}__x{attempt}",
    )


def _report(cells: tuple[ScoreCell, ...], *, tasks: tuple[str, ...], attempts: int) -> ScoreReport:
    return ScoreReport(
        doc_hash=HarnessDoc.baseline().doc_hash,
        request=ScoreRequest(task_ids=tasks, attempts=attempts),
        reward_mode="raw",
        cells=cells,
    )


def test_reward_modes_follow_the_frozen_selection_protocol() -> None:
    # raw: passed iff exactly 1.0; positive-binary: passed iff strictly positive.
    assert reward_passed(1.0, "raw")
    assert not reward_passed(0.99, "raw")
    assert not reward_passed(0.0, "raw")
    assert reward_passed(0.01, "positive-binary")
    assert reward_passed(1.0, "positive-binary")
    assert not reward_passed(0.0, "positive-binary")


def test_score_weights_tasks_equally_and_keeps_raw_rewards() -> None:
    report = _report(
        (
            _cell("a", 1, 1.0),
            _cell("a", 2, 1.0),
            _cell("b", 1, 0.25),
            _cell("b", 2, 1.0),
        ),
        tasks=("a", "b"),
        attempts=2,
    )
    assert report.score == pytest.approx(0.75)  # mean of per-task means: (1.0 + 0.5) / 2
    assert report.pass_rate == pytest.approx(0.75)
    assert [cell.reward for cell in report.by_task()["b"]] == [0.25, 1.0]
    assert report.by_task()["b"][0].artifact_dir == "/jobs/b__x1"


def test_report_rejects_missing_duplicate_and_extra_cells() -> None:
    with pytest.raises(ValidationError, match="missing"):
        _report((_cell("a", 1, 1.0),), tasks=("a", "b"), attempts=1)
    with pytest.raises(ValidationError, match="duplicate"):
        _report((_cell("a", 1, 1.0), _cell("a", 1, 0.0)), tasks=("a",), attempts=1)
    with pytest.raises(ValidationError, match="extra"):
        _report((_cell("a", 1, 1.0), _cell("a", 2, 1.0)), tasks=("a",), attempts=1)


def test_cells_canonicalize_and_reject_invalid_rewards() -> None:
    report = _report(
        (_cell("b", 1, 0.0), _cell("a", 1, 1.0)),
        tasks=("a", "b"),
        attempts=1,
    )
    assert [cell.task_id for cell in report.cells] == ["a", "b"]
    with pytest.raises(ValidationError):
        _cell("a", 1, 1.5)
    with pytest.raises(ValidationError):
        _cell("a", 1, float("nan"))
    with pytest.raises(ValidationError, match="not boolean"):
        ScoreCell(task_id="a", attempt=1, reward=True, passed=True)  # type: ignore[arg-type]


def test_request_rejects_empty_duplicate_and_boolean_inputs() -> None:
    with pytest.raises(ValidationError, match="nonempty"):
        ScoreRequest(task_ids=(), attempts=1)
    with pytest.raises(ValidationError, match="unique"):
        ScoreRequest(task_ids=("a", "a"), attempts=1)
    with pytest.raises(ValidationError, match="not boolean"):
        ScoreRequest(task_ids=("a",), attempts=True)  # type: ignore[arg-type]
