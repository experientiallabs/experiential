"""Declared state machines refuse and record illegal moves and render as Mermaid."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.state_machine import (
    IllegalTransition,
    MachineRun,
    StateMachine,
    TransitionJournal,
    target_table,
)


class Light(StrEnum):
    """A two-state fixture."""

    OFF = "off"
    ON = "on"
    BROKEN = "broken"


MACHINE = StateMachine(
    name="light",
    states=Light,
    initial=frozenset({Light.OFF}),
    terminal=frozenset({Light.BROKEN}),
    transitions=(
        (Light.OFF, "switch", Light.ON),
        (Light.ON, "switch", Light.OFF),
        (Light.ON, "burn", Light.BROKEN),
    ),
)


def test_legal_moves_follow_the_table_and_are_persisted(tmp_path: Path) -> None:
    """Every accepted move is one journal line."""
    journal = TransitionJournal(tmp_path / "moves.jsonl")
    state = MACHINE.move("lamp", None, "off", journal)
    state = MACHINE.move("lamp", state, "switch", journal)
    assert state == Light.ON
    assert [row["to"] for row in journal.rows()] == ["off", "on"]


def test_an_illegal_move_is_refused_and_recorded(tmp_path: Path) -> None:
    """A move outside the table raises after it is recorded as refused."""
    journal = TransitionJournal(tmp_path / "moves.jsonl")
    with pytest.raises(IllegalTransition, match="burn is not legal from off"):
        MACHINE.move("lamp", Light.OFF, "burn", journal)
    with pytest.raises(IllegalTransition, match="cannot start"):
        MACHINE.move("lamp", None, "on", journal)
    assert [row["refused"] for row in journal.rows()] == [True, True]


def test_a_terminal_state_may_not_be_left() -> None:
    """The table itself is checked."""
    with pytest.raises(ValueError, match="terminal state"):
        StateMachine(
            name="bad",
            states=Light,
            initial=frozenset({Light.OFF}),
            terminal=frozenset({Light.BROKEN}),
            transitions=((Light.BROKEN, "fix", Light.OFF),),
        )


def test_the_table_renders_as_a_mermaid_statechart() -> None:
    """The review surface lists every move."""
    chart = MACHINE.mermaid()
    assert chart.startswith("stateDiagram-v2")
    assert "    on --> broken: burn" in chart and "    broken --> [*]" in chart


class Door(StrEnum):
    """A two-state fixture for one-subject runs and multi-subject lifecycles."""

    SHUT = "shut"
    OPEN = "open"


DOOR = StateMachine(
    name="door",
    states=Door,
    initial=frozenset({Door.SHUT}),
    transitions=((Door.SHUT, "open", Door.OPEN), (Door.OPEN, "close", Door.SHUT)),
)


def test_a_run_records_every_attempt_and_refuses_illegal_moves() -> None:
    """Every attempt is recorded; a refused one leaves the state unchanged."""
    moves: list[JsonObject] = []
    run = MachineRun(DOOR, "front", moves.append)
    assert run.fire("open") is Door.OPEN
    with pytest.raises(IllegalTransition, match="cannot apply 'open' in state open"):
        run.fire("open")
    assert run.state is Door.OPEN
    assert [(row.event, row.accepted, row.target) for row in run.history] == [
        ("open", True, "open"),
        ("open", False, None),
    ]
    assert len(moves) == 2 and [row.sequence for row in run.history] == [0, 1]


def test_unreachable_states_and_duplicate_rows_are_refused_and_identity_tracks_the_table() -> None:
    """The table is checked; the identity moves with any row."""
    with pytest.raises(ValueError, match="unreachable"):
        StateMachine(name="door", states=Door, initial=frozenset({Door.SHUT}), transitions=())
    with pytest.raises(ValueError, match="twice"):
        StateMachine(
            name="door",
            states=Door,
            initial=frozenset({Door.SHUT}),
            transitions=((Door.SHUT, "open", Door.OPEN),) * 2,
        )
    other = StateMachine(
        name="door",
        states=Door,
        initial=frozenset({Door.SHUT}),
        transitions=((Door.SHUT, "open", Door.OPEN), (Door.OPEN, "slam", Door.SHUT)),
    )
    assert other.identity != DOOR.identity


def test_a_choice_event_needs_a_named_target_that_is_a_row() -> None:
    """One event may lead to several targets; the caller names one and the row must exist."""
    choice = StateMachine(
        name="verdict",
        states=Light,
        initial=frozenset({Light.OFF}),
        transitions=((Light.OFF, "judge", Light.ON), (Light.OFF, "judge", Light.BROKEN)),
    )
    with pytest.raises(IllegalTransition, match="named target"):
        choice.move("x", Light.OFF, "judge")
    assert choice.move("x", Light.OFF, "judge", target=Light.BROKEN) is Light.BROKEN
    with pytest.raises(IllegalTransition, match="not legal from"):
        choice.move("x", Light.ON, "judge", target=Light.BROKEN)


def test_a_lifecycle_keeps_moves_and_refusals_per_subject() -> None:
    """A refused move is recorded and leaves the subject where it was."""
    rows = target_table({Door.SHUT: frozenset({Door.OPEN}), Door.OPEN: frozenset({Door.SHUT})})
    machine = StateMachine(
        name="door", states=Door, initial=frozenset({Door.SHUT}), transitions=rows
    )
    states = machine.lifecycle().start("a").start("b")
    assert states.move("a", Door.OPEN, "push")
    assert not states.move("b", Door.SHUT, "already shut")
    assert states.state("a") is Door.OPEN and states.state("b") is Door.SHUT
    assert [(t.subject, t.target) for t in states.transitions] == [("a", "open")]
    assert [(r.subject, r.target) for r in states.refusals] == [("b", "shut")]
