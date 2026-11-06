"""Explicit state machines for dense findings and adversarial family mining.

Each machine is a state enum and one transition table; nothing else decides which moves are legal.
A :class:`Lifecycle` checks every move against the table, records legal moves as
:class:`Transition` rows and illegal ones as :class:`Refusal` rows (the subject stays where it
was), and both are persisted with the record that owns them. :meth:`StateMachine.mermaid`
generates the statechart from the same table.

- :data:`FINDING_MACHINE`: a dense candidate is tagged by the decision model, then cleared or
  routed to the judge; the judge (or the cheap path, for absence only) decides; aggregation then
  scores the decided finding or leaves it unscored. ``unscored`` and ``scored`` are terminal, so an
  unscored finding can never become scored, and ``uncertain`` can only become unscored.
- :data:`FAMILY_MACHINE`: a behavior-diff group is classified adversarial or not; an adversarial
  group is mined into divergence points; a linter is drafted and then accepted or refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Self

from pydantic import Field

from exp.common.core.artifacts import ContractModel


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


class Transition(ContractModel):
    """One legal move, as persisted."""

    machine: str
    subject: str
    source: str
    target: str
    reason: str = Field(default="", max_length=500)


class Refusal(ContractModel):
    """One illegal move that was refused; the subject stayed in ``source``."""

    machine: str
    subject: str
    source: str
    target: str
    reason: str = Field(default="", max_length=500)


class StateMachine[S: StrEnum]:
    """One state enum, its initial state and its single transition table."""

    def __init__(
        self, name: str, states: type[S], initial: S, table: Mapping[S, frozenset[S]]
    ) -> None:
        """Bind and check the table: every state appears, every target is a declared state."""
        if set(table) != set(states) or any(
            t not in set(states) for v in table.values() for t in v
        ):
            raise ValueError(f"{name}: the transition table must list exactly the declared states")
        self.name, self.states, self.initial, self.table = name, states, initial, dict(table)

    def allowed(self, source: S, target: S) -> bool:
        """Whether ``source -> target`` is in the table."""
        return target in self.table[source]

    def terminal(self) -> tuple[S, ...]:
        """States with no outgoing move."""
        return tuple(state for state in self.states if not self.table[state])

    def mermaid(self) -> str:
        """The Mermaid statechart generated from the table."""
        lines = ["stateDiagram-v2", f"    [*] --> {self.initial.value}"]
        for source in self.states:
            for target in sorted(self.table[source], key=lambda s: list(self.states).index(s)):
                lines.append(f"    {source.value} --> {target.value}")
        lines.extend(f"    {state.value} --> [*]" for state in self.terminal())
        return "\n".join(lines)

    def lifecycle(self) -> Lifecycle[S]:
        """A fresh lifecycle under this machine."""
        return Lifecycle(self)


class Lifecycle[S: StrEnum]:
    """Current states of several subjects, with every move and refusal recorded."""

    def __init__(self, machine: StateMachine[S]) -> None:
        """Start with no subjects."""
        self.machine = machine
        self.current: dict[str, S] = {}
        self.transitions: list[Transition] = []
        self.refusals: list[Refusal] = []

    def start(self, subject: str) -> Self:
        """Place a new subject in the initial state."""
        if subject in self.current:
            raise ValueError(f"{self.machine.name}: {subject} already started")
        self.current[subject] = self.machine.initial
        return self

    def move(self, subject: str, target: S, reason: str = "") -> bool:
        """Apply a legal move, or refuse and record an illegal one; return whether it applied."""
        source = self.current[subject]
        row = {
            "machine": self.machine.name,
            "subject": subject,
            "source": source.value,
            "target": target.value,
            "reason": reason[:500],
        }
        if not self.machine.allowed(source, target):
            self.refusals.append(Refusal.model_validate(row))
            return False
        self.transitions.append(Transition.model_validate(row))
        self.current[subject] = target
        return True

    def state(self, subject: str) -> S:
        """The subject's current state."""
        return self.current[subject]


F = FindingState
FINDING_MACHINE = StateMachine(
    "dense-finding-v1",
    FindingState,
    F.CANDIDATE,
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
    },
)
"""Dense finding lifecycle. A candidate is routed directly when tagging is unavailable; only a
cleared tag reaches ``absent`` without the judge; uncertain is never scored."""

A = FamilyState
FAMILY_MACHINE = StateMachine(
    "adversarial-family-v1",
    FamilyState,
    A.GROUP,
    {
        A.GROUP: frozenset({A.NOT_ADVERSARIAL, A.ADVERSARIAL}),
        A.NOT_ADVERSARIAL: frozenset(),
        A.ADVERSARIAL: frozenset({A.MINED}),
        A.MINED: frozenset({A.LINTER_DRAFTED, A.LINTER_FAILED}),
        A.LINTER_DRAFTED: frozenset({A.LINTER_ACCEPTED, A.LINTER_REFUSED}),
        A.LINTER_FAILED: frozenset(),
        A.LINTER_ACCEPTED: frozenset(),
        A.LINTER_REFUSED: frozenset(),
    },
)
"""Adversarial family lifecycle, from a diff group to a reviewed library linter."""
