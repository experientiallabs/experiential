"""The explicit lifecycle of one rule on the tag admission path.

A rule moves through admission as a declared state machine, never an implied one. The states are
an enum, every legal move is one row of :data:`TRANSITIONS`, declared as :data:`RULE_MACHINE` with
the one implementation (:class:`~exp.common.state_machine.StateMachine`); :func:`move` refuses
anything else, each move is kept in the rule's persisted history, and :func:`statechart` renders
the same table as a Mermaid state diagram for review. Compile and admit are choice events: the
draws or the verdict name the target, and the table must contain that exact row. The table is
plain data, so a later model-checking specification can be generated from it unchanged.

States:

- ``drafted``: System Two wrote the rule (truism plus reasoning).
- ``compiled`` / ``compile_failed``: compiler draws produced questions, or none did.
- ``tagged``: every question was answered about every family trajectory.
- ``admitted``: at least two independent lineages agree and nothing confidently contradicts.
- ``contradicted``: a trajectory confidently contradicts the rule.
- ``uncertain``: not admitted and not contradicted (too little independent support, or no
  usable questions).
- ``consolidated`` / ``consolidation_failed``: the consolidation draw for an admitted rule.
- ``withdrawn``: governance removed, demoted or rewrote an admitted rule after admission.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.common.state_machine import IllegalTransition, StateMachine


class RuleState(StrEnum):
    """Every state a rule can be in on the admission path."""

    DRAFTED = "drafted"
    COMPILED = "compiled"
    COMPILE_FAILED = "compile_failed"
    TAGGED = "tagged"
    ADMITTED = "admitted"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"
    CONSOLIDATED = "consolidated"
    CONSOLIDATION_FAILED = "consolidation_failed"
    WITHDRAWN = "withdrawn"


INITIAL = RuleState.DRAFTED
"""The state every rule starts in."""

S = RuleState
TRANSITIONS: tuple[tuple[RuleState, str, RuleState], ...] = (
    (S.DRAFTED, "compile", S.COMPILED),
    (S.DRAFTED, "compile", S.COMPILE_FAILED),
    (S.COMPILED, "tag", S.TAGGED),
    (S.COMPILE_FAILED, "admit", S.UNCERTAIN),
    (S.TAGGED, "admit", S.ADMITTED),
    (S.TAGGED, "admit", S.CONTRADICTED),
    (S.TAGGED, "admit", S.UNCERTAIN),
    (S.ADMITTED, "consolidate", S.CONSOLIDATED),
    (S.ADMITTED, "consolidate", S.CONSOLIDATION_FAILED),
    (S.ADMITTED, "govern", S.WITHDRAWN),
    (S.CONSOLIDATED, "govern", S.WITHDRAWN),
    (S.CONSOLIDATION_FAILED, "govern", S.WITHDRAWN),
)
"""The one transition table: ``(source, event, target)``; every other move is illegal."""

RULE_MACHINE = StateMachine(
    name="rule-admission-lifecycle",
    states=RuleState,
    initial=frozenset({INITIAL}),
    transitions=TRANSITIONS,
)
"""The rule lifecycle machine over :data:`TRANSITIONS`."""

TERMINAL: frozenset[RuleState] = RULE_MACHINE.final
"""States with no outgoing move."""


class IllegalTransitionError(IllegalTransition):
    """A rule move that is not a row of the transition table."""


class RuleTransition(ContractModel):
    """One persisted move.

    Attributes:
        source: State before.
        event: The event that moved it.
        target: State after.
        note: Why, in one line.
    """

    source: RuleState
    event: str = Field(min_length=1, max_length=32)
    target: RuleState
    note: str = Field(default="", max_length=600)


class RuleLifecycle(ContractModel):
    """One rule's current state and every move that led there.

    Attributes:
        rule_id: The rule.
        state: Current state.
        history: Every move, in order.
    """

    rule_id: str = Field(min_length=1)
    state: RuleState = INITIAL
    history: tuple[RuleTransition, ...] = ()


def is_legal(source: RuleState, event: str, target: RuleState) -> bool:
    """Whether a move is a row of the transition table."""
    return RULE_MACHINE.allows(source, event, target)


def move(lifecycle: RuleLifecycle, event: str, target: RuleState, note: str = "") -> RuleLifecycle:
    """Apply one move, refusing any move the table does not declare.

    Raises:
        IllegalTransitionError: The move is not in the transition table.
    """
    if not is_legal(lifecycle.state, event, target):
        raise IllegalTransitionError(
            f"rule {lifecycle.rule_id} cannot move from {lifecycle.state} to {target} on "
            f"{event!r}; the transition table declares no such move"
        )
    step = RuleTransition(source=lifecycle.state, event=event, target=target, note=note[:600])
    return lifecycle.model_copy(update={"state": target, "history": (*lifecycle.history, step)})


def replay(rule_id: str, history: Sequence[RuleTransition]) -> RuleLifecycle:
    """Rebuild a lifecycle from a persisted history, checking every move again.

    Raises:
        IllegalTransitionError: A recorded move is illegal or does not follow its predecessor.
    """
    lifecycle = RuleLifecycle(rule_id=rule_id)
    for step in history:
        if step.source != lifecycle.state:
            raise IllegalTransitionError(
                f"rule {rule_id} history moves from {step.source} while in {lifecycle.state}"
            )
        lifecycle = move(lifecycle, step.event, step.target, step.note)
    return lifecycle


def statechart() -> str:
    """The transition table as a Mermaid state diagram."""
    return RULE_MACHINE.mermaid() + "\n"
