"""The boundary machine: its chart is its table, and its terminal and forced paths hold."""

from __future__ import annotations

import pytest

from exp.common.state_machine import IllegalTransition
from exp.optimize.evaluation.variance.frontdoor_states import all_statecharts
from exp.optimize.evaluation.variance.rubric_evo_states import (
    BOUNDARY_MACHINE,
    FINAL_STATES,
    BoundaryState,
    statechart,
)

B = BoundaryState


def test_the_statechart_is_generated_from_the_table_and_reviewed_with_the_others() -> None:
    """Every row is an edge of the chart, and the front door lists the machine."""
    chart = statechart()
    for source, event, target in BOUNDARY_MACHINE.transitions:
        assert f"{source.value} --> {target.value}: {event}" in chart
    assert all_statecharts()[BOUNDARY_MACHINE.name] == chart


def test_terminal_states_and_the_forced_path() -> None:
    """Refusals and outcomes are final; a forced audit cannot exit as no divergence."""
    assert {B.INGEST_REFUSED, B.BUDGET_REFUSED} | FINAL_STATES == set(BOUNDARY_MACHINE.final)
    assert BOUNDARY_MACHINE.targets(B.AUDITED_FORCED, "no_divergence") == ()
    assert BOUNDARY_MACHINE.target(B.AUDITED_FORCED, "escalated") == B.ESCALATED
    with pytest.raises(IllegalTransition):
        BOUNDARY_MACHINE.target(B.RECALL_CHECKED, "unchanged")
    with pytest.raises(IllegalTransition):
        BOUNDARY_MACHINE.target(B.PUBLISHED, "recall_checked")
    assert BOUNDARY_MACHINE.target(B.RETIREMENT_REVERTED, "recall_checked") == B.RECALL_CHECKED
