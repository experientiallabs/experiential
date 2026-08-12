"""Tests for the evaluator-neutral scoring contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmo.runtime.harness.doc import HarnessDoc
from wmo.runtime.harness.scoring import (
    GradedTests,
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


def test_graded_tests_score_over_resolved_tests_only() -> None:
    assert GradedTests(passed=1, resolved=2).score == 0.5
    assert GradedTests(passed=6, resolved=6).score == 1.0
    assert GradedTests(passed=0, resolved=1).score == 0.0
    # Unresolved tests (skipped/pending/other) are carried, never put in the denominator: a suite
    # that passes everything it ran scores 1.0, matching the binary verdict it earned.
    assert GradedTests(passed=2, resolved=2, unresolved=3).score == 1.0
    with pytest.raises(ValidationError, match="cannot exceed resolved"):
        GradedTests(passed=3, resolved=2)
    with pytest.raises(ValidationError):
        GradedTests(passed=0, resolved=0)  # no verdict is no score, not a 0.0


def test_a_cell_without_a_test_report_has_no_graded_score() -> None:
    # The default, and what every cell deserialized from an artifact written before this field
    # carries: a graded rate must exclude it rather than average in a fabricated 0.0.
    cell = _cell("a", 1, 0.0)
    assert cell.tests is None
    assert cell.graded_score is None
    restored = ScoreCell.model_validate_json(
        '{"task_id": "a", "attempt": 1, "reward": 0.0, "passed": false, '
        '"artifact_dir": "/jobs/a__x1", "note": "completed", "infra_failed": false}'
    )
    assert restored.graded_score is None

    graded = cell.model_copy(update={"tests": GradedTests(passed=1, resolved=2)})
    assert graded.graded_score == 0.5
    assert graded.reward == 0.0  # the binary reward is untouched beside it


def test_request_rejects_empty_duplicate_and_boolean_inputs() -> None:
    with pytest.raises(ValidationError, match="nonempty"):
        ScoreRequest(task_ids=(), attempts=1)
    with pytest.raises(ValidationError, match="unique"):
        ScoreRequest(task_ids=("a", "a"), attempts=1)
    with pytest.raises(ValidationError, match="not boolean"):
        ScoreRequest(task_ids=("a",), attempts=True)  # type: ignore[arg-type]
