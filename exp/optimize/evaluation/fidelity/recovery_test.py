"""Recorded re-dispatches are idempotent, and a report with any is qualified in its headline."""

from datetime import UTC, datetime
from pathlib import Path

from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity.recovery import (
    JUDGE_OPERATION,
    RecoveryLog,
    UnknownOutcome,
    qualify,
    recorded_outcomes,
)


def test_a_repeated_record_counts_once_and_rows_number_steps(tmp_path: Path) -> None:
    """A second resumption re-recording the same re-dispatch never inflates its attempts."""
    store = ProjectStore(tmp_path, "demo")
    log = RecoveryLog(store, "fidelity-run", code_revision="r", clock=lambda: datetime.now(UTC))
    actions = tuple(
        ToolCall(call_id=f"task-{index}", name="read", arguments={}) for index in range(3)
    )
    log.record("prediction", "task-2", "scope-retry-1")
    log.record("prediction", "task-2", "scope-retry-1")
    log.record("prediction", "task-2", "scope-retry-2")
    log.record("judge", JUDGE_OPERATION, "judge-retry-1")
    log.record("investigation", "task-0", "note-investigate-retry-1")
    rows = recorded_outcomes(store, "fidelity-run", actions)
    assert [(row.stage, row.step, row.attempts) for row in rows] == [
        ("investigation", 0, 2),
        ("prediction", 2, 3),
        ("judge", None, 2),
    ]
    assert (
        rows[1].line() == "step 3 re-dispatched after an unknown outcome (prediction, 3 attempts)"
    )
    assert recorded_outcomes(store, "another-run", actions) == ()


def test_qualification_counts_stages_and_shows_both_sensitivity_figures() -> None:
    """Counts by stage, affected traces, and each subset's score with its denominator."""
    retry = UnknownOutcome(stage="prediction", step=1, attempts=2)
    lost = UnknownOutcome(stage="retrieval", step=0, attempts=2)
    assert qualify([((), 3, 3), ((), 1, 3)]) is None
    qualification = qualify([((retry, lost), 1, 3), ((), 3, 3), ((), 2, 3)])
    assert qualification is not None
    stages, unaffected, affected = qualification.lines()
    assert stages == (
        "  re-dispatched after an unknown outcome  prediction 1 · judge 0 · investigation 0 · "
        "retrieval 1 · 1 of 3 traces affected (a lost response may have been seen)"
    )
    assert unaffected == "  Unaffected traces only  5/6 replies exact (83%) · 2/2 traces"
    assert affected == "  Re-attempted traces only  1/3 replies exact (33%) · 1/1 traces"


def test_consolidation_retry_has_no_invented_prediction_step(tmp_path: Path) -> None:
    """Run-level learning retries remain visible without inventing a measured tool step."""
    store = ProjectStore(tmp_path, "demo")
    log = RecoveryLog(store, "run", code_revision="r", clock=lambda: datetime.now(UTC))
    log.record("investigation", "dream:reader-a", "reader-retry-1")
    log.record("investigation", "dream:reader-b", "reader-retry-2")
    rows = recorded_outcomes(store, "run", ())
    assert rows == (UnknownOutcome(stage="investigation", step=None, attempts=2),) * 2
    assert rows[0].line().startswith("memory consolidation re-dispatched")
