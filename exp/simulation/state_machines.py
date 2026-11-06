"""Explicit finite state machines: one state enum, one transition table, checked and recorded moves.

A component with a lifecycle declares it here instead of carrying ad hoc flags. A
:class:`StateMachine` names its states (a ``StrEnum``), its initial state and one table mapping
``(state, event)`` to the next state. A :class:`MachineRun` follows one subject through that
table: every move is checked against the table, an accepted move and a refused one are both
recorded as a :class:`TransitionRecord` through the injected durable sink, and a refused move
raises :class:`IllegalTransition` without changing the state. :meth:`StateMachine.mermaid`
renders the table as a Mermaid statechart, so the diagram is generated from the same table the
runtime enforces and cannot drift from it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, Sha256, sha256_json


class IllegalTransition(ValueError):
    """A move the table does not allow; the state is unchanged and the refusal is recorded."""


class TransitionRecord(ContractModel):
    """One attempted move of one subject.

    Attributes:
        machine: Name of the machine.
        machine_sha256: Identity of the machine's table.
        subject: What the run follows, such as a task or a step.
        sequence: Position of the attempt in the run, from zero.
        source: State before the attempt.
        event: The attempted event.
        target: State after the move, or None when it was refused.
        accepted: Whether the table allowed the move.
    """

    machine: str
    machine_sha256: Sha256
    subject: str
    sequence: int = Field(ge=0)
    source: str
    event: str
    target: str | None = None
    accepted: bool


class StateMachine:
    """One declared state enum, initial state and transition table."""

    def __init__(
        self,
        name: str,
        states: type[StrEnum],
        initial: StrEnum,
        table: Mapping[tuple[StrEnum, str], StrEnum],
    ) -> None:
        """Declare the machine, refusing a table over foreign or unreachable states.

        Args:
            name: Stable machine name.
            states: The state enum.
            initial: Starting state.
            table: ``(state, event)`` to next state; every other move is illegal.

        Raises:
            ValueError: A state in the table or the initial state is not a member of ``states``,
                or a state is unreachable from the initial state.
        """
        members = set(states)
        if initial not in members or any(
            source not in members or target not in members for (source, _), target in table.items()
        ):
            raise ValueError(f"state machine {name} names a state outside its enum")
        reachable, frontier = {initial}, [initial]
        while frontier:
            current = frontier.pop()
            for (source, _), target in table.items():
                if source == current and target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        if reachable != members:
            unreachable = sorted(str(state) for state in members - reachable)
            raise ValueError(f"state machine {name} has unreachable states {unreachable}")
        self.name, self.states, self.initial = name, states, initial
        self.table = dict(table)

    @property
    def identity(self) -> Sha256:
        """Return the SHA-256 identity of the name, initial state and table."""
        return sha256_json(
            {
                "name": self.name,
                "initial": str(self.initial),
                "table": sorted(
                    [str(source), event, str(target)]
                    for (source, event), target in self.table.items()
                ),
            }
        )

    def next(self, state: StrEnum, event: str) -> StrEnum | None:
        """Return the state ``event`` moves ``state`` to, or None when the move is illegal."""
        return self.table.get((state, event))

    def mermaid(self) -> str:
        """Render the table as a Mermaid ``stateDiagram-v2`` statechart."""
        lines = ["stateDiagram-v2", f"    [*] --> {self.initial}"]
        for (source, event), target in sorted(
            self.table.items(), key=lambda row: (str(row[0][0]), row[0][1])
        ):
            lines.append(f"    {source} --> {target}: {event}")
        return "\n".join(lines) + "\n"


class MachineRun:
    """One subject's walk through a machine, every attempt checked and recorded."""

    def __init__(
        self,
        machine: StateMachine,
        subject: str,
        record: Callable[[JsonObject], None],
    ) -> None:
        """Start ``subject`` in the machine's initial state.

        Args:
            machine: The declared machine.
            subject: What the run follows.
            record: Durable sink for every attempted move; it returns only after recording.
        """
        self.machine, self.subject, self._record = machine, subject, record
        self.state: StrEnum = machine.initial
        self.history: list[TransitionRecord] = []

    def fire(self, event: str) -> StrEnum:
        """Attempt one move, recording it whether or not the table allows it.

        Args:
            event: The event to apply.

        Returns:
            The new state.

        Raises:
            IllegalTransition: The table has no move for this state and event; the state is
                unchanged and the refusal is recorded.
        """
        target = self.machine.next(self.state, event)
        entry = TransitionRecord(
            machine=self.machine.name,
            machine_sha256=self.machine.identity,
            subject=self.subject,
            sequence=len(self.history),
            source=str(self.state),
            event=event,
            target=None if target is None else str(target),
            accepted=target is not None,
        )
        self.history.append(entry)
        self._record({"transition": entry.model_dump(mode="json")})
        if target is None:
            raise IllegalTransition(
                f"{self.machine.name} cannot apply {event!r} in state {self.state} for "
                f"{self.subject}; the refusal was recorded and the state is unchanged"
            )
        self.state = target
        return target
