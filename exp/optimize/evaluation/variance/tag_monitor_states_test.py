"""The monitor's machines: tables, terminals and charts."""

from __future__ import annotations

from exp.optimize.evaluation.variance.tag_monitor_states import (
    CHECKPOINT_MACHINE,
    HALT_MACHINE,
    PROPOSAL_MACHINE,
    CheckpointState,
    HaltState,
    ProposalState,
    statechart,
)


def test_terminal_states_and_charts_follow_the_tables() -> None:
    """Terminals are derived from the tables, and each chart renders every row."""
    assert CHECKPOINT_MACHINE.final == {
        CheckpointState.INPUTS_MISSING,
        CheckpointState.BUDGET_REFUSED,
        CheckpointState.PASSED,
        CheckpointState.HALTED,
    }
    assert HALT_MACHINE.final == {HaltState.ACKNOWLEDGED, HaltState.REAUTHORED}
    assert PROPOSAL_MACHINE.final == {
        ProposalState.ACCEPTED,
        ProposalState.DECLINED,
        ProposalState.SUPERSEDED,
    }
    chart = statechart()
    for machine in (CHECKPOINT_MACHINE, HALT_MACHINE, PROPOSAL_MACHINE):
        assert machine.mermaid() in chart
    assert not CHECKPOINT_MACHINE.allows(CheckpointState.SAMPLED, "passed", CheckpointState.PASSED)
