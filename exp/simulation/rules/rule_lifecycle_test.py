"""Offline tests for the declared rule lifecycle: legal moves pass, illegal ones are refused."""

from __future__ import annotations

import pytest

from exp.simulation.rules.rule_lifecycle import (
    TERMINAL,
    TRANSITIONS,
    IllegalTransitionError,
    RuleLifecycle,
    RuleState,
    RuleTransition,
    move,
    replay,
    statechart,
)

S = RuleState


def test_admitted_path_is_recorded_and_replays() -> None:
    """Drafted to withdrawn through every admission stage; the history replays to the same state."""
    lifecycle = RuleLifecycle(rule_id="r")
    for event, target in (
        ("compile", S.COMPILED),
        ("tag", S.TAGGED),
        ("admit", S.ADMITTED),
        ("consolidate", S.CONSOLIDATED),
        ("govern", S.WITHDRAWN),
    ):
        lifecycle = move(lifecycle, event, target, "why")
    assert lifecycle.state == S.WITHDRAWN and len(lifecycle.history) == 5
    assert replay("r", lifecycle.history) == lifecycle
    restored = RuleLifecycle.model_validate_json(lifecycle.model_dump_json())
    assert restored == lifecycle


@pytest.mark.parametrize(
    ("source", "event", "target"),
    [
        (S.DRAFTED, "admit", S.ADMITTED),
        (S.COMPILED, "admit", S.ADMITTED),
        (S.COMPILE_FAILED, "admit", S.ADMITTED),
        (S.TAGGED, "consolidate", S.CONSOLIDATED),
        (S.UNCERTAIN, "consolidate", S.CONSOLIDATED),
        (S.CONTRADICTED, "admit", S.ADMITTED),
        (S.UNCERTAIN, "govern", S.WITHDRAWN),
        (S.ADMITTED, "admit", S.ADMITTED),
        (S.WITHDRAWN, "admit", S.ADMITTED),
        (S.TAGGED, "tag", S.ADMITTED),
    ],
)
def test_illegal_transitions_are_refused(source: S, event: str, target: S) -> None:
    """A move the table does not declare raises and leaves no history."""
    lifecycle = RuleLifecycle(rule_id="r", state=source)
    with pytest.raises(IllegalTransitionError, match="declares no such move"):
        move(lifecycle, event, target)


def test_replay_refuses_a_tampered_history() -> None:
    """A persisted history that skips a state or declares an illegal move is refused."""
    skipped = (RuleTransition(source=S.TAGGED, event="admit", target=S.ADMITTED),)
    with pytest.raises(IllegalTransitionError, match="while in drafted"):
        replay("r", skipped)
    forged = (RuleTransition(source=S.DRAFTED, event="admit", target=S.ADMITTED),)
    with pytest.raises(IllegalTransitionError):
        replay("r", forged)


def test_statechart_renders_the_one_table() -> None:
    """Every table row and every terminal state appears in the Mermaid diagram."""
    chart = statechart()
    assert chart.startswith("stateDiagram-v2\n    [*] --> drafted\n")
    for source, event, target in TRANSITIONS:
        assert f"    {source.value} --> {target.value}: {event}" in chart
    assert TERMINAL == {S.CONTRADICTED, S.UNCERTAIN, S.WITHDRAWN}
    for state in TERMINAL:
        assert f"    {state.value} --> [*]" in chart
