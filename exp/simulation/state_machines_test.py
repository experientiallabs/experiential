"""Declared state machines refuse illegal moves, record every attempt and render their table."""

from enum import StrEnum

import pytest

from exp.common.core.artifacts import JsonObject
from exp.simulation.state_machines import IllegalTransition, MachineRun, StateMachine


class Door(StrEnum):
    """Fixture states."""

    SHUT = "shut"
    OPEN = "open"


DOOR = StateMachine(
    "door", Door, Door.SHUT, {(Door.SHUT, "open"): Door.OPEN, (Door.OPEN, "close"): Door.SHUT}
)


def test_legal_moves_advance_and_illegal_moves_are_refused_and_recorded() -> None:
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


def test_tables_are_checked_and_render_as_mermaid() -> None:
    """Unreachable or foreign states are refused; the chart lists every move."""
    with pytest.raises(ValueError, match="unreachable"):
        StateMachine("door", Door, Door.SHUT, {})
    assert DOOR.mermaid() == (
        "stateDiagram-v2\n    [*] --> shut\n    open --> shut: close\n    shut --> open: open\n"
    )
    other = StateMachine(
        "door", Door, Door.SHUT, {(Door.SHUT, "open"): Door.OPEN, (Door.OPEN, "slam"): Door.SHUT}
    )
    assert other.identity != DOOR.identity
