"""The front-door live view: journal parsing, rows, panels, stable lines and the rich region."""

from __future__ import annotations

import io
from collections import Counter
from datetime import timedelta
from pathlib import Path

from rich.console import Console

from exp.cli.shared.theme import EXP_THEME
from exp.cli.variance.fixtures_test import (
    RECORDED,
    START,
    TASK,
    write_journal,
    write_ledger,
)
from exp.cli.variance.sources import (
    Outcome,
    StartStages,
    Traffic,
    journal_segments,
    segment_after,
)
from exp.cli.variance.view import FrontDoorView, LinePrinter
from exp.common.progress import ProgressEvent

NOW = (START + timedelta(minutes=75)).timestamp()


def _view(tmp_path: Path) -> FrontDoorView:
    """A view over the recorded journal and a small ledger, frozen at minute 75."""
    journal = write_journal(tmp_path / "transitions.jsonl", RECORDED)
    ledger = write_ledger(
        tmp_path / "ledger.sqlite",
        [
            ("clef", "settled", 2),
            ("clef", "settled", 3),
            ("glm-flash", "settled", 40),
            ("actor-gemma4", "settled", 62),
            ("actor-gemma4", "dispatched", 74.5),
            ("actor-kimi", "unknown", 70),
        ],
    )
    traffic = Traffic(ledger, None, "demo")
    traffic.refresh()
    view = FrontDoorView(title="front door 0123456789ab", project="demo")
    view.started = START.timestamp()
    view.update(segment=segment_after(journal_segments(journal), None), traffic=traffic)
    view.update(
        starts={
            TASK: StartStages(
                TASK,
                run="decision-variance-run-x",
                saved={"variance-build": "a-stage", "rules": "b-stage"},
                saved_at={
                    "variance-build": START + timedelta(minutes=61),
                    "rules": START + timedelta(minutes=62),
                },
            )
        }
    )
    return view


def test_the_journal_splits_into_runs_and_tracks_each_phase(tmp_path: Path) -> None:
    """Each phase's state, start and end come from the checked transitions."""
    rows = [*RECORDED, (90, "run", "front-door-x", "running", "raised", "failed")]
    rows += [(100, "run", "front-door-x", None, "planned", "planned")]
    segments = journal_segments(write_journal(tmp_path / "t.jsonl", rows))
    assert len(segments) == 2
    first = segments[0]
    assert first.state == "failed" and first.ended is not None
    assert first.phases["project-build"].replayed
    tags = first.phases["slot tag-banks"]
    assert tags.state == "done" and tags.started is not None and tags.ended is not None
    assert (tags.ended - tags.started).total_seconds() == 30 * 60
    assert segment_after(segments, START + timedelta(minutes=95)) is segments[1]
    assert segment_after(segments, None) is segments[1]


def test_rows_show_phases_starts_and_pipeline_stages(tmp_path: Path) -> None:
    """The tree: replayed and finished phases, the running start and its six stages."""
    rows = {row.key: row for row in _view(tmp_path).rows(now=NOW)}
    assert rows["project-build"].state == "done" and rows["project-build"].note == "saved"
    assert rows["slot tag-banks"].elapsed == 30 * 60
    pipeline = rows[f"pipeline {TASK}"]
    assert pipeline.state == "running" and pipeline.elapsed == 14 * 60
    assert pipeline.note == "3 calls, 1 in flight"
    assert rows[f"pipeline {TASK}/rules"].state == "done"
    traceml = rows[f"pipeline {TASK}/traceml"]
    assert traceml.state == "running" and "actor groups ×5 tiers" in traceml.note
    assert rows[f"pipeline {TASK}/records"].state == "pending"
    assert rows["pipelines"].note == "1 starts, 0 complete"
    order = list(rows)
    assert order.index(f"pipeline {TASK}") < order.index("slot dense-hybrid")


def test_progress_events_drive_rows_without_a_journal() -> None:
    """In process, phase and pipeline-stage events show before any source is read."""
    view = FrontDoorView(title="t", project="demo")
    view.observe(ProgressEvent("slot tag-banks", detail="running"))
    view.observe(ProgressEvent("pipelines", completed=0, total=2, detail=TASK))
    view.observe(ProgressEvent(f"pipeline {TASK}", detail="running"))
    view.observe(ProgressEvent(f"pipeline {TASK}", completed=2, total=6, detail="traceml"))
    rows = {row.key: row for row in view.rows()}
    assert rows["slot tag-banks"].state == "running"
    assert rows["pipelines"].note == "0/2 starts"
    assert rows[f"pipeline {TASK}/traceml"].state == "running"
    assert view.run_state() == "running"
    view.observe(ProgressEvent(f"pipeline {TASK}", completed=3, total=6, detail="traceml"))
    assert {row.key: row for row in view.rows()}[f"pipeline {TASK}/traceml"].state == "done"


def test_calls_cap_and_running_panels(tmp_path: Path) -> None:
    """Calls per role and tier, in flight, timeouts, the cap and the running stage's calls."""
    view = _view(tmp_path)
    calls = view.calls_lines()
    assert calls[0].startswith("clef: 2 calls, 2 done")
    assert any(line.startswith("tier gemma4: 2 calls, 1 done, 1 in flight") for line in calls)
    assert any(
        line.startswith("tier kimi: 1 calls, 0 done") and "1 timeouts" in line for line in calls
    )
    assert view.cap_line().startswith("6 / 1,000 real attempts (0.6%)")
    view.interrupted = NOW
    running = view.running()
    assert running == [
        f"pipeline {TASK} · 14m00s · tier gemma4 1 done +1 in flight, tier kimi 0 done"
    ]
    assert view.run_state() == "interrupted"


def test_the_summary_names_points_verdicts_admissions_and_packages() -> None:
    """At the end: qualified points, judged tiers, families, admitted tasks, Harbor paths."""
    view = FrontDoorView(title="t", project="demo")
    view.update(
        outcome=Outcome(
            qualified_points=Counter({"qualified": 3, "rejected": 5}),
            error_points=Counter({"actor-gemma4": 2}),
            judged={TASK: ["actor-gemma4", "actor-kimi"]},
            families=Counter({"admitted": 1, "refused": 1}),
            admitted=1,
            harbor=["/tmp/harbor/task-a"],
            diagnostic=["/tmp/harbor/task-b"],
        )
    )
    summary = view.summary_lines()
    assert summary[0] == "qualified points  3  (rejected 5, qualified 3)"
    assert summary[1] == "error points      2  (tier gemma4 2)"
    assert summary[2] == f"judged            {TASK}: 2 tiers"
    assert "1 admitted, 1 refused" in summary[3]
    assert summary[4] == "admitted tasks    1"
    assert summary[-2:] == ["  /tmp/harbor/task-a", "  /tmp/harbor/task-b (diagnostic)"]


def test_a_log_gets_the_tree_once_then_only_changes(tmp_path: Path) -> None:
    """Non-interactive output is stable lines: nothing repeats until a row changes state."""
    view = _view(tmp_path)
    stream = io.StringIO()
    printer = LinePrinter(Console(file=stream, width=200, theme=EXP_THEME))
    printer(view)
    first = stream.getvalue().splitlines()
    assert first[0].startswith("front door 0123456789ab · demo · running")
    assert any("✓ slot tag-banks  done  30m00s" in line for line in first)
    printer(view)
    assert stream.getvalue().splitlines() == first
    view.observe(ProgressEvent(f"pipeline {TASK}", completed=3, total=6, detail="traceml"))
    printer(view)
    added = stream.getvalue().splitlines()[len(first) :]
    assert len(added) == 2
    assert "✓ traceml  done" in added[0] and "● error-mining  running" in added[1]


def test_the_interactive_region_renders_tree_calls_cap_and_preview(tmp_path: Path) -> None:
    """The rich region carries the tree, the calls table, the cap and the last preview."""
    view = _view(tmp_path)
    view.announce(("Before phase slot dense-hybrid:", "   Completed work: project-build."))
    console = Console(record=True, width=160, theme=EXP_THEME, force_terminal=True)
    console.print(view.render())
    text = console.export_text()
    assert "front door 0123456789ab · demo · running" in text
    assert "slot tag-banks" in text and "30m00s" in text
    assert "tier gemma4" in text and "in flight" in text
    assert "Cap  6 / 1,000 real attempts" in text
    assert "Before phase slot dense-hybrid:" in text
