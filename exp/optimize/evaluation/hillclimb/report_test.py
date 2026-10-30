"""A report covers a fixed number of rounds of one climb."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.optimize.evaluation.hillclimb import stages
from exp.optimize.evaluation.hillclimb.prompts import PromptEdit
from exp.optimize.evaluation.hillclimb.report import (
    ClimbReportSpec,
    preflight_climb_report,
    run_climb_report,
)
from exp.optimize.evaluation.hillclimb.rounds import ClimbRound, run_climb_round
from exp.optimize.evaluation.hillclimb.tests.climb_test import HELPFUL, climb_fixture, deadline


def test_report_identity_ignores_its_deadline_and_needs_decided_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without kept edits no final comparison is planned; undecided rounds refuse."""
    project, spec, actor = climb_fixture(tmp_path, monkeypatch, "records")
    now = datetime.now(UTC)
    first = ClimbReportSpec(climb=spec, rounds=0, deadline=now)
    assert (
        first.identity()
        == first.model_copy(update={"deadline": now + timedelta(days=1)}).identity()
    )
    assert preflight_climb_report(project, first) is False
    with pytest.raises(ValueError, match="round 1 of this climb has no saved decision"):
        preflight_climb_report(project, ClimbReportSpec(climb=spec, rounds=1, deadline=now))
    assert not actor.requests


def test_an_unchanged_best_prompt_replays_its_final_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reverted later rounds cannot buy a second final test of the same best prompt."""
    project, spec, actor = climb_fixture(tmp_path, monkeypatch, "records")
    helpful = PromptEdit(kind="append", text=HELPFUL, change_note="Re-read before editing.")
    noise = PromptEdit(kind="append", text="Be concise.", change_note="Shorter replies.")
    run_climb_round(
        project,
        ClimbRound(climb=spec, round=1, edit=helpful, deadline=deadline()),
        code_revision="test",
    )
    first = run_climb_report(
        project, ClimbReportSpec(climb=spec, rounds=1, deadline=deadline()), code_revision="test"
    )
    # The kept prompt passes every train case; lift the saturation refusal to decide a reverted
    # second round on the same incumbent.
    monkeypatch.setattr(stages, "require_headroom", lambda outcomes: None)
    reverted = run_climb_round(
        project,
        ClimbRound(climb=spec, round=2, edit=noise, deadline=deadline()),
        code_revision="test",
    )
    assert not reverted.kept
    calls = len(actor.requests)
    later = ClimbReportSpec(climb=spec, rounds=2, deadline=deadline() + timedelta(hours=1))
    with pytest.raises(ValueError, match="registered under code revision test"):
        run_climb_report(project, later, code_revision="other")
    second = run_climb_report(project, later, code_revision="test")
    assert second.best == first.best and second.final == first.final
    assert first.verdict == "merge" and "not an independent holdout" in first.scope
    assert len(actor.requests) == calls
    assert run_climb_report(project, later, code_revision="other") == second
    assert len(actor.requests) == calls
