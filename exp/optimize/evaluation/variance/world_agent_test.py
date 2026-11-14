"""Offline tests for the agentic world author over the ticket packet and scripted replies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.optimize.evaluation.variance import world_agent as _world_agent_module

from exp.common.core.artifacts import JsonObject
from exp.common.project import artifact_input
from exp.optimize.evaluation.variance.authoring_packet_test import ticket_packet, tool_reply, wire
from exp.optimize.evaluation.variance.tests.grounded_world_test import grounded_fixture
from exp.optimize.evaluation.variance.world_agent import (
    PLAN_TOOL,
    PROMPT_SETS,
    author_world_agentic,
    load_world_authoring,
    world_agent_protocol_sha256,
)
from exp.optimize.evaluation.variance.world_author import WORLD_TOOL, load_world_draft
from exp.optimize.evaluation.variance.world_author_test import CLOSE, research_answer, world_answer
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.simulation.induction.sandbox import ProgramSandbox

WRONG_CLOSE = CLOSE.replace('"error": "ticket is held', '"error": "locked')
"""A close program whose refusal text misreplays the held fixture."""


def _plan() -> JsonObject:
    """A plan citing supplied locators."""
    return {
        "state_machine": "tickets: open -> escalated | closed; close refuses unless lock is none",
        "programs": ["read_ticket", "check_sla", "escalate", "close"],
        "failure_modes": ["held ticket refused", "missing lock refused"],
        "fixtures": ["held close", "escalate high"],
        "supported_by": ["rule:tickets.lock"],
    }


def _users(transport: ScriptedJsonTransport) -> list[JsonObject]:
    """Decode the user payload of every request served."""
    found: list[JsonObject] = []
    for request in transport.requests:
        messages = request.payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[-1], dict)
        found.append(json.loads(str(messages[-1]["content"])))
    return found


def _systems(transport: ScriptedJsonTransport) -> list[str]:
    """Return the system prompt of every request served."""
    found: list[str] = []
    for request in transport.requests:
        messages = request.payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[0], dict)
        found.append(str(messages[0]["content"]))
    return found


def test_a_repair_round_fixes_a_failing_program_from_its_concrete_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sandbox rejection is quoted back, the repair is accepted and everything replays."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = ticket_packet(fixture)
    transport = wire(
        monkeypatch,
        tool_reply(PLAN_TOOL, _plan()),
        tool_reply(WORLD_TOOL, world_answer(packet, close=WRONG_CLOSE)),
        tool_reply(WORLD_TOOL, world_answer(packet)),
    )
    with ProgramSandbox() as sandbox:
        authoring = author_world_agentic(
            fixture.project,
            fixture.settings,
            reference,
            sandbox=sandbox,
            identity="agentic-repair",
            code_revision="test",
            prompt_sets=("failure-mode-first",),
        )
        replay = author_world_agentic(
            fixture.project,
            fixture.settings,
            reference,
            sandbox=sandbox,
            identity="agentic-repair",
            code_revision="test",
            prompt_sets=("failure-mode-first",),
        )

    assert replay == authoring and len(transport.requests) == 3
    (trajectory,) = authoring.trajectories
    assert [(row.phase, row.status) for row in trajectory.rounds] == [
        ("plan", "planned"),
        ("draft", "refused"),
        ("repair", "ready"),
    ]
    refused = trajectory.rounds[1]
    assert refused.failure is not None and "induction validation" in refused.failure
    assert refused.validation is not None
    plan_user, draft_user, repair_user = _users(transport)
    assert set(plan_user) == {"authoring_packet", "citable_locators"}
    assert draft_user["your_plan"] == _plan() | {"predicted": [], "research": []}
    assert repair_user["host_failures"] == [refused.failure]
    failing = repair_user["failing_programs"]
    assert isinstance(failing, list) and len(failing) == 1
    assert isinstance(failing[0], dict) and failing[0]["tool"] == "close"
    assert '"actual": {"error": "locked by another agent"}' in json.dumps(failing[0]["failures"])
    assert all("rubric" not in key for user in _users(transport) for key in user)
    assert ["PHASE plan" in row for row in _systems(transport)] == [True, False, False]
    assert "PHASE repair" in _systems(transport)[2]
    assert authoring.selected == "failure-mode-first" and authoring.draft == trajectory.saved
    assert trajectory.saved is not None
    draft = load_world_draft(fixture.project, trajectory.saved)
    assert draft.status == "ready" and draft.world is not None
    saved = artifact_input(fixture.project.artifacts.read("agentic-repair-authoring").manifest)
    assert load_world_authoring(fixture.project, saved) == authoring


def test_each_prompt_set_runs_its_own_trajectory_and_the_first_ready_one_is_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exhausted repairs stay refused, research stays blocked, and a ready draft is selected."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = ticket_packet(fixture)
    transport = wire(
        monkeypatch,
        tool_reply(PLAN_TOOL, _plan()),
        tool_reply(WORLD_TOOL, world_answer(packet, close=WRONG_CLOSE)),
        tool_reply(WORLD_TOOL, world_answer(packet, close=WRONG_CLOSE)),
        tool_reply(PLAN_TOOL, _plan()),
        tool_reply(WORLD_TOOL, research_answer("question:0")),
        tool_reply(WORLD_TOOL, research_answer("question:0")),
        tool_reply(PLAN_TOOL, _plan()),
        tool_reply(WORLD_TOOL, world_answer(packet)),
    )
    with ProgramSandbox() as sandbox:
        authoring = author_world_agentic(
            fixture.project,
            fixture.settings,
            reference,
            sandbox=sandbox,
            identity="agentic-sets",
            code_revision="test",
            maximum_repairs=1,
        )

    assert len(transport.requests) == 8
    assert [(row.prompt_set, row.draft.status) for row in authoring.trajectories] == [
        ("contract-first", "refused"),
        ("state-machine-first", "needs_research"),
        ("failure-mode-first", "ready"),
    ]
    assert [len(row.rounds) for row in authoring.trajectories] == [3, 3, 2]
    assert authoring.selected == "failure-mode-first"
    systems = _systems(transport)
    for index, name in (
        (0, "contract-first"),
        (3, "state-machine-first"),
        (6, "failure-mode-first"),
    ):
        assert PROMPT_SETS[name].version in systems[index]
    assert len({row.prompt_sha256 for row in authoring.trajectories}) == 3
    assert len(world_agent_protocol_sha256()) == 64
    with pytest.raises(ValueError, match="distinct prompt sets"):
        author_world_agentic(
            fixture.project,
            fixture.settings,
            reference,
            sandbox=ProgramSandbox(),
            identity="agentic-bad",
            code_revision="test",
            prompt_sets=("contract-first", "contract-first"),
        )


@pytest.fixture(autouse=True)
def _sequential_prompt_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordered fake transport needs the prompt sets drawn one after another."""
    monkeypatch.setattr(_world_agent_module, "PARALLEL_PROMPT_SETS", False)
