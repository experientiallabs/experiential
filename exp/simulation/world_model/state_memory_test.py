"""Per-task state descriptions: drafted per variant, refreshed on change, kept in budget."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelRequest
from exp.common.project.paths import ProjectPaths
from exp.common.state_machine import IllegalTransition
from exp.simulation.mining.tagging import DecisionCache
from exp.simulation.world_model.state_memory import (
    MEMORY_MACHINE,
    STATE_MEMORY_GUIDANCE,
    MemoryState,
    StateMemory,
    StateMemoryPolicy,
    StateMemoryPrompt,
    clip_utf8,
)

WORKBOOK: JsonObject = {"workbook": {"sheets": {"Budget": {"B2": 10, "B3": "=B2*2"}}}}


class Author:
    """A scripted author: drafts echo their stance, the reconcile writes a fixed description."""

    def __init__(self, final: str = "Sheet Budget: B2 holds 10; B3 doubles it.") -> None:
        """Start with no requests."""
        self.requests: list[ModelRequest] = []
        self.final = final

    def __call__(self, request: ModelRequest) -> str:
        """Answer one request."""
        self.requests.append(request)
        prompt = request.messages[-1].content
        assert isinstance(prompt, str)
        if "DRAFTS" in prompt:
            return self.final
        if "inconsistent, empty" in prompt:
            return "   "
        return "Draft: " + prompt.split("\n")[1][:60]


def memory(
    tmp_path: Path,
    author: Author,
    policy: StateMemoryPolicy | None = None,
    moves: list[JsonObject] | None = None,
) -> StateMemory:
    """Bind a state memory over the project decision cache, recording lifecycle moves."""
    moves = [] if moves is None else moves
    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    return StateMemory(
        policy or StateMemoryPolicy(author_alias="author"),
        complete=author,
        model={"model": "open-weight-author"},
        purchase=cache.purchase,
        record=moves.append,
    )


def test_description_is_drafted_per_variant_reconciled_and_reused_until_the_state_changes(
    tmp_path: Path,
) -> None:
    """Three drafts and a reconcile per new state; an unchanged state costs nothing."""
    author = Author()
    states = memory(tmp_path, author)
    first = states.describe("Double the budget", WORKBOOK)
    assert first.state is MemoryState.CURRENT and first.refreshed
    assert first.text == "Sheet Budget: B2 holds 10; B3 doubles it."
    assert len(author.requests) == StateMemoryPolicy(author_alias="a").calls_per_refresh == 4
    drafts = {draft.variant_id: draft for draft in first.drafts}
    assert drafts["constraints"].error is not None, "an empty draft is recorded, not redrawn"
    assert "B3" in str(author.requests[0].messages[-1].content)

    same = states.describe("Double the budget", WORKBOOK)
    assert not same.refreshed and same.text == first.text and len(author.requests) == 4

    changed: JsonObject = {"workbook": {"sheets": {"Budget": {"B2": 11, "B3": "=B2*2"}}}}
    refreshed = states.describe("Double the budget", changed)
    assert refreshed.refreshed and refreshed.state_sha256 != first.state_sha256
    assert len(author.requests) == 8


def test_description_stays_inside_its_byte_budget(tmp_path: Path) -> None:
    """An overlong reconcile is clipped on a character boundary and the clip is recorded."""
    author = Author(final="é" * 400)
    states = memory(tmp_path, author, StateMemoryPolicy(author_alias="author", byte_budget=257))
    result = states.describe("Double the budget", WORKBOOK)
    assert result.clipped and result.injected_bytes <= 257
    assert result.text is not None and len(result.text.encode()) == result.injected_bytes
    assert clip_utf8("abc", 10) == ("abc", False)
    assert "257 bytes" in str(author.requests[-1].messages[-1].content)


def test_a_replayed_description_makes_no_call_and_the_identity_tracks_the_policy(
    tmp_path: Path,
) -> None:
    """A fresh memory over the same cache replays; policy changes move the identity."""
    author = Author()
    memory(tmp_path, author).describe("Double the budget", WORKBOOK)
    calls = len(author.requests)
    replayed = memory(tmp_path, author).describe("Double the budget", WORKBOOK)
    assert replayed.refreshed and len(author.requests) == calls, "replay buys nothing"

    base = StateMemoryPolicy(author_alias="author")
    narrower = base.model_copy(update={"byte_budget": 512})
    other = base.model_copy(update={"prompts": (StateMemoryPrompt(variant_id="x", stance="y"),)})
    assert len({base.identity, narrower.identity, other.identity}) == 3
    assert memory(tmp_path, author, narrower).identity != memory(tmp_path, author).identity
    assert "authoritative" in STATE_MEMORY_GUIDANCE


def test_unusable_reconcile_leaves_the_step_without_a_description(tmp_path: Path) -> None:
    """A blank reconcile is recorded as failed and nothing is injected."""
    result = memory(tmp_path, Author(final="")).describe("Double the budget", WORKBOOK)
    assert result.state is MemoryState.FAILED and result.text is None and result.injected_bytes == 0
    assert result.drafts[-1].variant_id == "reconcile"


def test_lifecycle_moves_are_recorded_and_illegal_moves_refused(tmp_path: Path) -> None:
    """Authored, reused, stale and refreshed are explicit moves; an illegal one is refused."""
    moves: list[JsonObject] = []
    states = memory(tmp_path, Author(), moves=moves)
    first = states.describe("Double the budget", WORKBOOK)
    assert [(row.source, row.event, row.target) for row in first.transitions] == [
        ("absent", "author", "authoring"),
        ("authoring", "described", "current"),
    ]
    reused = states.describe("Double the budget", WORKBOOK)
    assert [row.event for row in reused.transitions] == ["reuse"]
    changed: JsonObject = {"workbook": {"sheets": {}}}
    moved = states.describe("Double the budget", changed)
    assert [row.event for row in moved.transitions] == ["state_changed", "refresh", "described"]
    assert len(moves) == 6 and all("transition" in row for row in moves)

    run = states.run("Double the budget")
    with pytest.raises(IllegalTransition, match="cannot apply 'refresh' in state current"):
        run.fire("refresh")
    assert run.state is MemoryState.CURRENT
    refused = moves[-1]["transition"]
    assert isinstance(refused, dict) and refused["accepted"] is False
    chart = MEMORY_MACHINE.mermaid()
    assert chart.startswith("stateDiagram-v2\n    [*] --> absent")
    assert "    stale --> authoring: refresh" in chart
