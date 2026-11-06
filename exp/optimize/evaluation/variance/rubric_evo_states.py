"""The explicit state machine of one epoch boundary of the evolving rubric.

The rubric is frozen within a training epoch and revised only at the boundary between epochs.
One boundary is one subject of :data:`BOUNDARY_MACHINE`, a state enum and one transition table
declared with the one implementation, :class:`~exp.common.state_machine.StateMachine`. The loop
(:mod:`~.rubric_evolution`) names every move; the table refuses an undeclared one, and every
accepted and refused move is persisted to the evolution's
:class:`~exp.common.state_machine.TransitionJournal`. :func:`statechart` renders the same table
as Mermaid, the review surface.

The path through a boundary:

1. ``opened`` -> ``ingested`` when the epoch's export is accepted, ``ingest_refused`` when it
   names another epoch or rubric version (epochs never mix), ``budget_refused`` when the
   boundary's priced bound does not fit the cumulative ledger.
2. ``ingested`` -> ``tagged``: the decision model tagged every step.
3. ``tagged`` -> ``audited`` (routine) or ``audited_forced`` (the divergence trigger fired at the
   previous boundary, so the audit sample is the forced size and the boundary cannot end at
   ``no_divergence``; a forced boundary with nothing flagged is ``escalated`` for review).
4. ``audited`` -> ``diffed`` when the audit flagged high-reward rollouts, else ``no_divergence``.
5. ``diffed`` -> ``admitted`` or ``none_admitted``; ``admitted`` -> ``compiled`` (dense question
   and judge criterion per item) -> ``consolidated``.
6. ``consolidated``, ``none_admitted`` or ``no_divergence`` -> ``retirement_checked``.
7. ``retirement_checked`` -> ``unchanged`` (no item added or retired: the version carries over)
   or ``recall_checked`` (held-out known-hack recall measured for the candidate).
8. ``recall_checked`` -> ``published`` when recall did not drop, else ``recall_regressed``;
   ``recall_regressed`` -> ``retirement_reverted`` (retry the candidate without its
   retirements) -> ``recall_checked``, or -> ``kept`` (the current version carries over).

``published``, ``unchanged``, ``kept``, ``escalated``, ``ingest_refused`` and ``budget_refused``
are terminal.
"""

from __future__ import annotations

from enum import StrEnum

from exp.common.state_machine import StateMachine, target_table


class BoundaryState(StrEnum):
    """States of one epoch boundary."""

    OPENED = "opened"
    INGESTED = "ingested"
    INGEST_REFUSED = "ingest_refused"
    BUDGET_REFUSED = "budget_refused"
    TAGGED = "tagged"
    AUDITED = "audited"
    AUDITED_FORCED = "audited_forced"
    ESCALATED = "escalated"
    NO_DIVERGENCE = "no_divergence"
    DIFFED = "diffed"
    ADMITTED = "admitted"
    NONE_ADMITTED = "none_admitted"
    COMPILED = "compiled"
    CONSOLIDATED = "consolidated"
    RETIREMENT_CHECKED = "retirement_checked"
    UNCHANGED = "unchanged"
    RECALL_CHECKED = "recall_checked"
    RECALL_REGRESSED = "recall_regressed"
    RETIREMENT_REVERTED = "retirement_reverted"
    PUBLISHED = "published"
    KEPT = "kept"


B = BoundaryState
BOUNDARY_MACHINE = StateMachine(
    name="rubric-evolution-boundary-v1",
    states=BoundaryState,
    initial=frozenset({B.OPENED}),
    transitions=target_table(
        {
            B.OPENED: frozenset({B.INGESTED, B.INGEST_REFUSED, B.BUDGET_REFUSED}),
            B.INGESTED: frozenset({B.TAGGED}),
            B.TAGGED: frozenset({B.AUDITED, B.AUDITED_FORCED}),
            B.AUDITED: frozenset({B.DIFFED, B.NO_DIVERGENCE}),
            B.AUDITED_FORCED: frozenset({B.DIFFED, B.ESCALATED}),
            B.DIFFED: frozenset({B.ADMITTED, B.NONE_ADMITTED}),
            B.ADMITTED: frozenset({B.COMPILED}),
            B.COMPILED: frozenset({B.CONSOLIDATED}),
            B.CONSOLIDATED: frozenset({B.RETIREMENT_CHECKED}),
            B.NONE_ADMITTED: frozenset({B.RETIREMENT_CHECKED}),
            B.NO_DIVERGENCE: frozenset({B.RETIREMENT_CHECKED}),
            B.RETIREMENT_CHECKED: frozenset({B.UNCHANGED, B.RECALL_CHECKED}),
            B.RECALL_CHECKED: frozenset({B.PUBLISHED, B.RECALL_REGRESSED}),
            B.RECALL_REGRESSED: frozenset({B.RETIREMENT_REVERTED, B.KEPT}),
            B.RETIREMENT_REVERTED: frozenset({B.RECALL_CHECKED}),
        }
    ),
)
"""One epoch boundary of the evolving rubric."""

FINAL_STATES = frozenset({B.PUBLISHED, B.UNCHANGED, B.KEPT, B.ESCALATED})
"""Terminal states of a boundary that ran; the next epoch then trains under a frozen version."""


def statechart() -> str:
    """Return the boundary machine as a Mermaid ``stateDiagram-v2``, generated from its table."""
    return BOUNDARY_MACHINE.mermaid()
