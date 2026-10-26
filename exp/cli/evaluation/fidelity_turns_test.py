"""The public fidelity command shows each turn as a chat with its own measurements."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    RoutedTransport,
    investigation,
    memory_project,
    predicted,
    trajectory,
)
from exp.simulation.world_model.session_research_test import structured_reply


def test_default_progress_is_minimal_and_verbose_lists_each_judged_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default output is stages plus a result block; --verbose lists each judged turn."""
    _, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    spec = trajectory(base, "traj-a", "alpha", "learn")
    spec = spec.model_copy(update={"judge": spec.judge.model_copy(update={"per_turn": True})})
    ids = [d.dimension_id for d in spec.judge.definition.dimensions]
    axis: JsonObject = {
        "status": "assessed",
        "score": 3,
        "rationale": "Fixture",
        "evidence_refs": ["step-0"],
        "critical_flags": [],
    }
    judged = structured_reply(
        {
            "axes": {key: axis for key in ids},
            "turns": {
                "step-0": {"scores": dict.fromkeys(ids, 1), "rationale": "Wrong reply."},
                "step-1": {"scores": dict.fromkeys(ids, 5), "rationale": "Exact reply."},
            },
        }
    )
    wire = RoutedTransport(
        [
            predicted("traj-a-0", "wrong"),
            investigation("alpha"),
            predicted("traj-a-1", "alpha-reply-1"),
            judged,
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    path = tmp_path / "traj-a.json"
    path.write_text(spec.model_dump_json())
    run = CliRunner().invoke(
        app,
        ["eval", "fidelity-test", "--root", str(tmp_path), "--fidelity", str(path)]
        + ["--yes", "--non-interactive"],
        env={"COLUMNS": "120"},
    )
    assert run.exit_code == 0, run.output
    assert ". Simulating turns 2/2" in run.output and ". Judging" in run.output
    assert "Replies exact  1/2   ✗ ✓" in run.output
    assert "Memory         v0 → v1 · 1 new note" in run.output
    assert "turn 1" not in run.output and "Physical attempts" not in run.output

    shown = CliRunner().invoke(
        app,
        ["eval", "fidelity-test", "--root", str(tmp_path), "--fidelity", str(path)]
        + ["--non-interactive", "--verbose"],
        env={"COLUMNS": "120"},
    )
    assert shown.exit_code == 0, shown.output
    first, second = shown.output.split("turn 1  ")[1].split("turn 2  ")
    assert first.startswith('inspect(key="alpha")')
    assert '(reply): "wrong" · recorded "alpha-reply-0"' in first
    assert "divergence 1.00" in first and "response 1" in first and "note written" in first
    assert "divergence 0.00" in second and "response 5" in second
    assert "whole run only" not in shown.output


def test_a_hard_stop_shows_failed_turns_as_not_reproduced_and_progress_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed first turn blocks the rest: marks are ✗, not ·, and progress never says 2/2."""
    _, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    spec = trajectory(base, "traj-x", "alpha", "learn")
    spec = type(spec).model_validate(
        spec.model_dump(mode="json", exclude={"memory", "online_research", "retrieval"})
    )
    axis: JsonObject = {
        "status": "assessed",
        "score": 1,
        "rationale": "Fixture",
        "evidence_refs": ["step-0"],
        "critical_flags": [],
    }
    ids = [d.dimension_id for d in spec.judge.definition.dimensions]
    wire = RoutedTransport(
        [
            structured_reply({"not": "a transition"}),
            structured_reply({"axes": {key: axis for key in ids}}),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    path = tmp_path / "traj-x.json"
    path.write_text(spec.model_dump_json())
    run = CliRunner().invoke(
        app,
        ["eval", "fidelity-test", "--root", str(tmp_path), "--fidelity", str(path)]
        + ["--yes", "--non-interactive", "--verbose"],
        env={"COLUMNS": "120"},
    )
    assert run.exit_code == 0, run.output
    assert "Replies exact  0/2   ✗ ✗" in run.output
    assert "Simulating turns 2/2" not in run.output
    assert "Simulating turns (stopped) 1/2" in run.output
    assert "no reply (blocked)" in run.output and "no recorded reply" not in run.output
    assert "memory " not in run.output.split("turn 1  ")[1]
