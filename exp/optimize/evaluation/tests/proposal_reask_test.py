"""A refused simulator proposal is asked again with its reason, never repaired."""

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.evaluations.reconstruction import DualReport
from exp.common.project import ProjectStore
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import (
    FidelitySpec,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    RoutedTransport,
    judged,
    memory_project,
    predicted,
    trajectory,
)
from exp.runtime.sessions.store import SessionStore
from exp.simulation.world_model.session_research_test import structured_reply


def plain_spec(tmp_path: Path, reasks: int) -> tuple[ProjectStore, FidelitySpec]:
    """A project and a two-turn spec with no memory and the given re-ask allowance."""
    project, base = memory_project(tmp_path)
    spec = trajectory(base, "traj-r", "alpha", "learn")
    value = spec.model_dump(mode="json", exclude={"memory", "online_research", "retrieval"})
    return project, FidelitySpec.model_validate({**value, "proposal_reasks": reasks})


def steps(report: DualReport) -> list[JsonObject]:
    """The report's per-step mechanical rows."""
    rows = report.mechanical["steps"]
    assert isinstance(rows, list)
    return [row for row in rows if isinstance(row, dict)]


def test_a_refused_proposal_is_asked_again_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second answer commits; the refusal and its reason stay on the transition."""
    project, spec = plain_spec(tmp_path, 2)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    wire = RoutedTransport(
        [
            structured_reply({"not": "a transition"}),
            predicted("traj-r-0", "alpha-reply-0"),
            predicted("traj-r-1", "alpha-reply-1"),
            judged(spec),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    report = execute_fidelity_evaluation(project, prepared)
    assert report.mechanical["valid_steps"] == 2
    assert [row["trajectory_reply_exact"] for row in steps(report)] == [True, True]
    second = wire.requests[1].payload
    assert "That proposal was refused" in json.dumps(second)
    path = project.artifacts.project_directory / (prepared.id + "-sessions.sqlite")
    transitions = SessionStore(path).read_session(prepared.id).state["transitions"]
    assert isinstance(transitions, list)
    refused = transitions[0]["evidence"]["refused_proposals"]
    assert len(refused) == 1 and "reason" in refused[0]


def test_reasks_are_bounded_and_the_turn_then_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the allowance the turn fails visibly; nothing is coerced into a reply."""
    project, spec = plain_spec(tmp_path, 2)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    bad = structured_reply({"not": "a transition"})
    wire = RoutedTransport([bad, bad, bad, judged(spec)])
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    report = execute_fidelity_evaluation(project, prepared)
    statuses = [row["status"] for row in steps(report)]
    assert statuses == ["failure", "blocked"]
    assert (
        len([r for r in wire.requests if "That proposal was refused" in json.dumps(r.payload)]) == 2
    )
