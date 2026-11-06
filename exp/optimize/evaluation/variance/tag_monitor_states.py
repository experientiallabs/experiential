"""The tag monitor's declared state machines: one checkpoint run, one halt, one proposal.

Each is a ``StrEnum`` plus one transition table, checked by
:class:`~exp.common.state_machine.StateMachine`:

- :data:`CHECKPOINT_MACHINE` walks one checkpoint run. Missing inputs end it before anything is
  computed; statistics are computed and evidence sampled for free; a run with no flagged subject
  and no halt skips the review; otherwise Astra's review is reserved on the cumulative ledger (a
  refused reservation ends the run) and either passes the host's checks or is recorded as
  unusable, and the report ends ``passed`` or ``halted``. Every move is journaled; an undeclared
  move is refused and journaled.
- :data:`HALT_MACHINE`: a raised halt blocks the next front-door phase until it is acknowledged
  (by a person, with a note) or re-authored (a new report for the same checkpoint and scope
  replaces the halted one).
- :data:`PROPOSAL_MACHINE`: a proposal is routed to its owning path, which accepts or declines
  it; a newer revision of the same proposal supersedes an unresolved one.
"""

from __future__ import annotations

from enum import StrEnum

from exp.common.state_machine import StateMachine, target_table


class CheckpointState(StrEnum):
    """One checkpoint run's state.

    Attributes:
        OPENED: The run started.
        INPUTS_MISSING: A record the checkpoint reads is not saved yet.
        MEASURED: Statistics and halt conditions are computed.
        SAMPLED: The bounded summary and evidence sample are built.
        REVIEW_SKIPPED: Nothing is flagged and nothing halts, so Astra is not called.
        BUDGET_REFUSED: The review's bound did not fit the cumulative ledger.
        REVIEWED: Astra's answer passed the host's citation and type checks.
        REVIEW_UNUSABLE: No draft passed the checks; the statistics stand alone.
        REPORTED: The report is assembled and its proposals routed.
        PASSED: Saved with no halt.
        HALTED: Saved with at least one halt.
    """

    OPENED = "opened"
    INPUTS_MISSING = "inputs_missing"
    MEASURED = "measured"
    SAMPLED = "sampled"
    REVIEW_SKIPPED = "review_skipped"
    BUDGET_REFUSED = "budget_refused"
    REVIEWED = "reviewed"
    REVIEW_UNUSABLE = "review_unusable"
    REPORTED = "reported"
    PASSED = "passed"
    HALTED = "halted"


C = CheckpointState

CHECKPOINT_MACHINE = StateMachine(
    name="tag-monitor-checkpoint",
    states=CheckpointState,
    initial=frozenset({C.OPENED}),
    transitions=target_table(
        {
            C.OPENED: frozenset({C.INPUTS_MISSING, C.MEASURED}),
            C.MEASURED: frozenset({C.SAMPLED}),
            C.SAMPLED: frozenset(
                {C.REVIEW_SKIPPED, C.BUDGET_REFUSED, C.REVIEWED, C.REVIEW_UNUSABLE}
            ),
            C.REVIEW_SKIPPED: frozenset({C.REPORTED}),
            C.REVIEWED: frozenset({C.REPORTED}),
            C.REVIEW_UNUSABLE: frozenset({C.REPORTED}),
            C.REPORTED: frozenset({C.PASSED, C.HALTED}),
        }
    ),
)
"""Every legal move of one checkpoint run."""


class HaltState(StrEnum):
    """One halt's state.

    Attributes:
        RAISED: It blocks the next front-door phase.
        ACKNOWLEDGED: A person acknowledged it with a note; the next phase may run.
        REAUTHORED: A newer report for the same checkpoint and scope replaced it.
    """

    RAISED = "raised"
    ACKNOWLEDGED = "acknowledged"
    REAUTHORED = "reauthored"


HALT_MACHINE = StateMachine(
    name="tag-monitor-halt",
    states=HaltState,
    initial=frozenset({HaltState.RAISED}),
    transitions=target_table(
        {HaltState.RAISED: frozenset({HaltState.ACKNOWLEDGED, HaltState.REAUTHORED})}
    ),
)
"""Every legal move of one halt."""


class ProposalState(StrEnum):
    """One proposal's state.

    Attributes:
        PROPOSED: Accepted from a review.
        ROUTED: Handed to its owning path.
        ACCEPTED: The owning path took it up.
        DECLINED: The owning path declined it.
        SUPERSEDED: A newer revision of the same proposal replaced it before it was resolved.
    """

    PROPOSED = "proposed"
    ROUTED = "routed"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    SUPERSEDED = "superseded"


P = ProposalState

PROPOSAL_MACHINE = StateMachine(
    name="tag-monitor-proposal",
    states=ProposalState,
    initial=frozenset({P.PROPOSED}),
    transitions=target_table(
        {
            P.PROPOSED: frozenset({P.ROUTED, P.SUPERSEDED}),
            P.ROUTED: frozenset({P.ACCEPTED, P.DECLINED, P.SUPERSEDED}),
        }
    ),
)
"""Every legal move of one proposal."""


def statechart() -> str:
    """Return the three machines as titled Mermaid statecharts."""
    return "\n\n".join(
        f"{machine.name}:\n{machine.mermaid()}"
        for machine in (CHECKPOINT_MACHINE, HALT_MACHINE, PROPOSAL_MACHINE)
    )
