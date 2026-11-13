"""Step mode: a card before and after each stage, replayed against ran-now, evidence paths."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from exp.cli.shared.theme import EXP_THEME
from exp.cli.variance import step
from exp.cli.variance.fixtures_test import RECORDED, write_journal
from exp.cli.variance.step import Stepper, evidence, one_line, stage_key
from exp.cli.variance.view import FrontDoorView
from exp.common.progress import ProgressEvent
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.frontdoor import FrontDoorPlan
from exp.optimize.evaluation.variance.frontdoor_contracts import FrontDoorStage
from exp.optimize.evaluation.variance.frontdoor_states import StageState
from exp.optimize.evaluation.variance.pipeline_spec import CostLine

PLAN = FrontDoorPlan(
    spec_sha256="a" * 64,
    code_revision="r",
    stages=(
        FrontDoorStage(stage="project-build", status=StageState.DONE),
        FrontDoorStage(stage="slot tag-banks", status=StageState.DONE),
        FrontDoorStage(
            stage="slot curriculum",
            status=StageState.PLANNED,
            costs=(CostLine(name="glm", attempts=40),),
            lines=("Order every start's admitted tasks by strength.",),
        ),
    ),
)


def _stepper(tmp_path: Path, waits: list[str]) -> tuple[Stepper, io.StringIO]:
    """A stepper over the recorded journal whose waits are recorded, not blocking."""
    stream = io.StringIO()
    console = Console(file=stream, width=160, theme=EXP_THEME)
    stepper = Stepper(
        console=console,
        store=ProjectStore(tmp_path, "demo"),
        view=FrontDoorView(title="t", project="demo"),
        plan=PLAN,
        journal=write_journal(tmp_path / "transitions.jsonl", RECORDED),
        pause=False,
    )
    stepper.wait = waits.append  # ty: ignore[invalid-assignment]
    return stepper, stream


def test_a_replayed_phase_says_when_it_was_saved_and_where_its_evidence_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replayed, not ran now: the saved time comes from the run that finished it."""
    monkeypatch.setattr(step, "WIKI", tmp_path / "wiki")
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "slot-tag-banks.html").write_text("<html></html>")
    waits: list[str] = []
    stepper, stream = _stepper(tmp_path, waits)
    stepper.observe(ProgressEvent("slot tag-banks", detail="replayed"))
    out = stream.getvalue()
    assert "▸ slot tag-banks" in out
    assert "reads      each environment family's traces and decision tags" in out
    assert "✓ slot tag-banks  replayed (saved 2026-11-12 09:31 UTC) · no model call" in out
    assert f"evidence   {tmp_path / 'wiki' / 'slot-tag-banks.html'}" in out
    assert waits == ["continue"]


def test_a_phase_that_runs_gets_a_card_before_and_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before: what it does, reads and its bound, then a pause; after: ran now and evidence."""
    monkeypatch.setattr(step, "WIKI", tmp_path / "wiki")
    waits: list[str] = []
    stepper, stream = _stepper(tmp_path, waits)
    stepper.observe(ProgressEvent("slot curriculum", detail="running"))
    assert "does       Order every start's admitted tasks by strength." in stream.getvalue()
    assert "bound      at most 40 physical requests" in stream.getvalue()
    assert waits == ["run"]
    stepper.observe(ProgressEvent("slot curriculum", detail="finished"))
    out = stream.getvalue()
    assert "✓ slot curriculum  ran now in under 1s · 0 model attempts · finished" in out
    assert "slot-curriculum.html (not built yet)" in out
    assert waits == ["run", "continue"]


def test_a_pipeline_stage_that_runs_shows_its_summary(tmp_path: Path) -> None:
    """The reporting executors' results become the stage's one-line outputs."""
    waits: list[str] = []
    stepper, stream = _stepper(tmp_path, waits)
    stepper.before_stage("task-1", "traceml")
    stepper.after_stage("task-1", "traceml", {"error_points": {"a": 1}, "rubric": "r-1"})
    out = stream.getvalue()
    assert "▸ traceml task-1" in out and "actor groups ×5 tiers" in out
    assert "outputs  error_points 1 · rubric r-1" in out
    assert "pipeline-task-1-traceml.html" in out
    assert waits == ["run", "continue"]


def test_stage_keys_and_summaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wiki page keys are the stage names, hyphenated; long summaries are cut."""
    monkeypatch.setattr(step, "WIKI", tmp_path / "wiki")
    assert stage_key("slot tag-banks") == "slot-tag-banks"
    assert stage_key("pipeline task-1 traceml") == "pipeline-task-1-traceml"
    assert evidence("bootstrap-preparation").endswith("(not built yet)[/dim]")
    assert one_line({"a": "x" * 200}, limit=20) == "a " + "x" * 17 + "…"


def test_from_begins_at_a_named_stage_and_earlier_cards_pass_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--from slot curriculum: earlier phases replay without a card or a pause."""
    monkeypatch.setattr(step, "WIKI", tmp_path / "wiki")
    stream = io.StringIO()
    console = Console(file=stream, width=160, theme=EXP_THEME)
    stepper = Stepper(
        console=console,
        store=ProjectStore(tmp_path, "demo"),
        view=FrontDoorView(title="t", project="demo"),
        plan=PLAN,
        journal=write_journal(tmp_path / "transitions.jsonl", RECORDED),
        pause=False,
        start_at="curriculum",
    )
    waits: list[str] = []
    stepper.wait = waits.append  # ty: ignore[invalid-assignment]
    stepper.observe(ProgressEvent("project-build", detail="replayed"))
    stepper.observe(ProgressEvent("slot tag-banks", detail="replayed"))
    stepper.observe(ProgressEvent("slot curriculum", detail="running"))
    out = stream.getvalue()
    assert "· 1. project-build (skipped, replayed)" in out
    assert "· 2. slot tag-banks (skipped, replayed)" in out
    assert "▸ slot tag-banks" not in out and "▸ slot curriculum" in out
    assert waits == ["run"]


def test_a_number_typed_at_a_pause_jumps_forward(tmp_path: Path) -> None:
    """Typing 3 at the first pause skips straight to the third card."""
    answers = iter(["3"])
    stream = io.StringIO()
    console = Console(file=stream, width=160, theme=EXP_THEME)
    console.input = lambda prompt="": next(answers)  # ty: ignore[invalid-assignment]
    stepper = Stepper(
        console=console,
        store=ProjectStore(tmp_path, "demo"),
        view=FrontDoorView(title="t", project="demo"),
        plan=PLAN,
        journal=write_journal(tmp_path / "transitions.jsonl", RECORDED),
        pause=True,
    )
    stepper.observe(ProgressEvent("project-build", detail="replayed"))
    stepper.observe(ProgressEvent("slot tag-banks", detail="replayed"))
    out = stream.getvalue()
    assert "▸ project-build" in out
    assert "· 2. slot tag-banks (skipped, replayed)" in out


def test_an_unknown_from_is_refused(tmp_path: Path) -> None:
    """--from must name a stage of the plan."""
    with pytest.raises(ValueError, match="names no stage"):
        Stepper(
            console=Console(file=io.StringIO()),
            store=ProjectStore(tmp_path, "demo"),
            view=FrontDoorView(title="t", project="demo"),
            plan=PLAN,
            journal=None,
            start_at="no-such-stage",
        )
