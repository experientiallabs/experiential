"""The front door's phase and run machines refuse every move outside their tables."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.state_machine import IllegalTransition, TransitionJournal
from exp.optimize.evaluation.variance.frontdoor_states import (
    RUN_MACHINE,
    STAGE_MACHINE,
    RunState,
    StageState,
    Tracker,
    statecharts,
)


def test_a_phase_cannot_run_without_approval_or_leave_done(tmp_path: Path) -> None:
    """Planned never jumps to running, and a done phase never moves again."""
    tracker = Tracker(TransitionJournal(tmp_path / "t.jsonl"), "run")
    tracker.stage("pipeline a", "planned")
    with pytest.raises(IllegalTransition):
        tracker.stage("pipeline a", "started")
    tracker.stage("pipeline a", "approved")
    tracker.stage("pipeline a", "started")
    tracker.stage("pipeline a", "finished")
    with pytest.raises(IllegalTransition):
        tracker.stage("pipeline a", "started")
    rows = tracker.journal.rows()
    assert [row["refused"] for row in rows].count(True) == 2
    assert tracker.stages["pipeline a"] == StageState.DONE


def test_a_capped_phase_is_refused_before_it_starts_and_can_be_replanned() -> None:
    """stopped_at_cap is reached only from planned or running, and returns to planned."""
    assert STAGE_MACHINE.target(StageState.PLANNED, "refused") == StageState.STOPPED_AT_CAP
    assert STAGE_MACHINE.target(StageState.STOPPED_AT_CAP, "replanned") == StageState.PLANNED
    with pytest.raises(IllegalTransition):
        STAGE_MACHINE.target(StageState.APPROVED, "refused")
    with pytest.raises(IllegalTransition):
        STAGE_MACHINE.target(StageState.BLOCKED, "approved")


def test_a_run_consents_before_it_starts_and_ends_once() -> None:
    """No run starts unconsented, and a finished run is final."""
    with pytest.raises(IllegalTransition):
        RUN_MACHINE.target(RunState.PLANNED, "started")
    with pytest.raises(IllegalTransition):
        RUN_MACHINE.target(RunState.REPLAYING, "stopped")
    assert RunState.FINISHED in RUN_MACHINE.terminal
    assert "planned --> consented: consented" in statecharts()
