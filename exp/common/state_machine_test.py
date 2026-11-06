"""Declared state machines refuse and record illegal moves and render as Mermaid."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pytest

from exp.common.state_machine import IllegalTransition, StateMachine, TransitionJournal


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
