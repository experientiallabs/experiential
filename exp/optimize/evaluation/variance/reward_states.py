"""Explicit state machines for dense findings and adversarial family mining.

Each machine is a state enum and one transition table declared with the one implementation,
:class:`~exp.common.state_machine.StateMachine`; nothing else decides which moves are legal. Each
event names its target state (:func:`~exp.common.state_machine.target_table`). A
:class:`~exp.common.state_machine.Lifecycle` checks every move against the table, records legal
moves as ``Transition`` rows and illegal ones as ``Refusal`` rows (the subject stays where it
was), and both are persisted with the record that owns them. ``StateMachine.mermaid`` generates
the statechart from the same table.

- :data:`FINDING_MACHINE`: a dense candidate is tagged by the decision model, then cleared or
  routed to the judge; the judge (or the cheap path, for absence only) decides; aggregation then
  scores the decided finding or leaves it unscored. ``unscored`` and ``scored`` are terminal, so an
  unscored finding can never become scored, and ``uncertain`` can only become unscored.
- :data:`FAMILY_MACHINE`: a behavior-diff group is classified adversarial or not; an adversarial
  group is mined into divergence points; a linter is drafted and then accepted or refused.
"""

from __future__ import annotations

from enum import StrEnum

from exp.common.state_machine import StateMachine, target_table


class FindingState(StrEnum):
    """States of one dense finding."""

    CANDIDATE = "candidate"
    TAGGED = "tagged"
    CLEARED = "cleared"
    ROUTED = "routed"
    ESTABLISHED = "established"
    ACCEPTABLE = "acceptable"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"
    SCORED = "scored"
    UNSCORED = "unscored"


class FamilyState(StrEnum):
    """States of one behavior-diff group in adversarial mining."""

    GROUP = "group"
    NOT_ADVERSARIAL = "not_adversarial"
    ADVERSARIAL = "adversarial"
    MINED = "mined"
    LINTER_DRAFTED = "linter_drafted"
    LINTER_FAILED = "linter_failed"
    LINTER_ACCEPTED = "linter_accepted"
    LINTER_REFUSED = "linter_refused"


F = FindingState
FINDING_MACHINE = StateMachine(
    name="dense-finding-v1",
    states=FindingState,
    initial=frozenset({F.CANDIDATE}),
    transitions=target_table(
        {
            F.CANDIDATE: frozenset({F.TAGGED, F.ROUTED}),
            F.TAGGED: frozenset({F.CLEARED, F.ROUTED}),
            F.CLEARED: frozenset({F.ABSENT}),
            F.ROUTED: frozenset({F.ESTABLISHED, F.ACCEPTABLE, F.ABSENT, F.UNCERTAIN}),
            F.ESTABLISHED: frozenset({F.SCORED, F.UNSCORED}),
            F.ACCEPTABLE: frozenset({F.SCORED, F.UNSCORED}),
            F.ABSENT: frozenset({F.SCORED, F.UNSCORED}),
            F.UNCERTAIN: frozenset({F.UNSCORED}),
            F.SCORED: frozenset(),
            F.UNSCORED: frozenset(),
        }
    ),
)
"""Dense finding lifecycle. A candidate is routed directly when tagging is unavailable; only a
cleared tag reaches ``absent`` without the judge; uncertain is never scored."""

A = FamilyState
FAMILY_MACHINE = StateMachine(
    name="adversarial-family-v1",
    states=FamilyState,
    initial=frozenset({A.GROUP}),
    transitions=target_table(
        {
            A.GROUP: frozenset({A.NOT_ADVERSARIAL, A.ADVERSARIAL}),
            A.NOT_ADVERSARIAL: frozenset(),
            A.ADVERSARIAL: frozenset({A.MINED}),
            A.MINED: frozenset({A.LINTER_DRAFTED, A.LINTER_FAILED}),
            A.LINTER_DRAFTED: frozenset({A.LINTER_ACCEPTED, A.LINTER_REFUSED}),
            A.LINTER_FAILED: frozenset(),
            A.LINTER_ACCEPTED: frozenset(),
            A.LINTER_REFUSED: frozenset(),
        }
    ),
)
"""Adversarial family lifecycle, from a diff group to a reviewed library linter."""


class PointState(StrEnum):
    """States of one candidate decision point under the qualification bar."""

    CANDIDATE = "candidate"
    TAGGED = "tagged"
    QUALIFIED = "qualified"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    CONTINUED = "continued"
    DEFERRED = "deferred"


P = PointState
POINT_MACHINE = StateMachine(
    name="point-qualification-v1",
    states=PointState,
    initial=frozenset({P.CANDIDATE}),
    transitions=target_table(
        {
            P.CANDIDATE: frozenset({P.TAGGED, P.UNCERTAIN}),
            P.TAGGED: frozenset({P.QUALIFIED, P.REJECTED, P.UNCERTAIN}),
            P.QUALIFIED: frozenset({P.CONTINUED, P.DEFERRED}),
            P.DEFERRED: frozenset({P.CONTINUED}),
            P.REJECTED: frozenset(),
            P.UNCERTAIN: frozenset(),
            P.CONTINUED: frozenset(),
        }
    ),
)
"""Point qualification lifecycle. Only a qualified point is continued; a qualified point that does
not fit the remaining ledger is deferred and may be continued in a later phase, never dropped."""
