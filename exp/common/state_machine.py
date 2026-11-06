"""Explicit state machines: a state enum, one plain transition table, checked and recorded moves.

A workflow whose progress lives in scattered flags is an implicit state machine nobody can
review. Declare it instead, with the one implementation in this module:

- ``states``: a ``StrEnum`` of every state;
- ``transitions``: one tuple of ``(source, event, target)`` rows, plain data, so a model-checking
  spec can be derived from it later. One event may lead from a state to several targets (a
  choice whose outcome the caller decides, such as an admission verdict); the caller then names
  the target and the table still has to contain that exact row;
- :meth:`StateMachine.move` checks every move against the table and refuses an illegal one with
  :class:`IllegalTransition`; a :class:`MoveSink` (for example a :class:`TransitionJournal`)
  persists each accepted and refused move;
- :class:`MachineRun` follows one subject and records every attempt as a
  :class:`TransitionRecord`; :class:`Lifecycle` follows several subjects and keeps legal moves
  as :class:`Transition` rows and refused ones as :class:`Refusal` rows for the owning record;
- :meth:`StateMachine.mermaid` renders the table as a Mermaid statechart, the review surface,
  so the diagram cannot drift from the table the runtime enforces.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, Self

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, Sha256, sha256_json


class IllegalTransition(ValueError):
    """A move the machine's transition table does not contain."""


class MoveSink(Protocol):
    """Persists one attempted move; a None target records a refused move."""

    def record(
        self,
        machine: str,
        subject: str,
        source: StrEnum | None,
        event: str,
        target: StrEnum | None,
    ) -> None:
        """Persist the move before returning."""
        ...


@dataclass(frozen=True)
class StateMachine[State: StrEnum]:
    """One declared machine.

    Attributes:
        name: Machine name, recorded with every move.
        states: Every state.
        initial: States a new subject may start in. A :meth:`move` from ``None`` names one of
            them as its event; a :class:`MachineRun` or :class:`Lifecycle` starts in the only
            one.
        transitions: Every legal ``(source, event, target)`` move; rows are unique.
        terminal: States no transition leaves. None derives them from the table; a given set
            must contain no state with an outgoing row.
    """

    name: str
    states: type[State]
    initial: frozenset[State]
    transitions: tuple[tuple[State, str, State], ...]
    terminal: frozenset[State] | None = field(default=None)

    def __post_init__(self) -> None:
        """Refuse a malformed table.

        Raises:
            ValueError: A state is foreign to the enum, a row is listed twice, a terminal state
                has a move, or no initial state reaches some state.
        """
        members = set(self.states)
        if not self.initial or not self.initial <= members:
            raise ValueError(f"{self.name}: initial states must be members of {self.states}")
        if len(set(self.transitions)) != len(self.transitions):
            raise ValueError(f"{self.name}: a transition row is listed twice")
        sources: set[State] = set()
        for source, _, target in self.transitions:
            if source not in members or target not in members:
                raise ValueError(f"{self.name}: a transition names a state outside its enum")
            sources.add(source)
        if self.terminal is None:
            derived = frozenset(state for state in self.states if state not in sources)
            object.__setattr__(self, "terminal", derived)
        else:
            for state in self.terminal:
                if state in sources:
                    raise ValueError(f"{self.name}: terminal state {state} has a transition")
        reachable, frontier = set(self.initial), list(self.initial)
        while frontier:
            current = frontier.pop()
            for source, _, target in self.transitions:
                if source == current and target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        if reachable != members:
            unreachable = sorted(str(state) for state in members - reachable)
            raise ValueError(f"{self.name}: unreachable states {unreachable}")

    @property
    def final(self) -> frozenset[State]:
        """Return the terminal states."""
        return self.terminal or frozenset()

    @property
    def identity(self) -> Sha256:
        """Return the SHA-256 identity of the name, initial states and table."""
        return sha256_json(
            {
                "name": self.name,
                "initial": sorted(str(state) for state in self.initial),
                "table": sorted([str(s), event, str(t)] for s, event, t in self.transitions),
            }
        )

    @property
    def start(self) -> State:
        """Return the only initial state.

        Raises:
            ValueError: The machine has several initial states.
        """
        if len(self.initial) != 1:
            raise ValueError(f"{self.name} has several initial states; name one")
        return next(iter(self.initial))

    def targets(self, source: State, event: str) -> tuple[State, ...]:
        """Return every state ``event`` may move ``source`` to, in table order."""
        return tuple(t for s, e, t in self.transitions if s == source and e == event)

    def allows(self, source: State, event: str, target: State) -> bool:
        """Whether ``(source, event, target)`` is a row of the table."""
        return (source, event, target) in self.transitions

    def target(self, source: State | None, event: str) -> State:
        """Return the state an event leads to.

        Args:
            source: The current state, or None for a new subject.
            event: The event.

        Returns:
            The next state.

        Raises:
            IllegalTransition: The table has no such move, or the event is a choice whose
                target the caller has to name.
        """
        if source is None:
            for state in self.initial:
                if state.value == event:
                    return state
            raise IllegalTransition(f"{self.name}: a new subject cannot start with {event}")
        found = self.targets(source, event)
        if not found:
            raise IllegalTransition(f"{self.name}: {event} is not legal from {source}")
        if len(found) > 1:
            raise IllegalTransition(f"{self.name}: {event} from {source} needs a named target")
        return found[0]

    def move(
        self,
        subject: str,
        source: State | None,
        event: str,
        journal: MoveSink | None = None,
        *,
        target: State | None = None,
    ) -> State:
        """Check one move against the table, persist it, and return the new state.

        Args:
            subject: What moves (a stage, task or rule identity).
            source: Its current state, or None when it is new; a new subject's event names its
                initial state.
            event: The event.
            journal: Where the accepted or refused move is persisted.
            target: The chosen target of a choice event; it must be a row of the table.

        Returns:
            The new state.

        Raises:
            IllegalTransition: The move is not in the table; it is recorded as refused first.
        """
        try:
            if target is None or source is None:
                chosen = self.target(source, event)
                if target is not None and chosen != target:
                    raise IllegalTransition(f"{self.name}: {event} does not start in {target}")
            elif self.allows(source, event, target):
                chosen = target
            else:
                raise IllegalTransition(
                    f"{self.name}: {event} is not legal from {source} to {target}"
                )
        except IllegalTransition:
            if journal is not None:
                journal.record(self.name, subject, source, event, None)
            raise
        if journal is not None:
            journal.record(self.name, subject, source, event, chosen)
        return chosen

    def lifecycle(self) -> Lifecycle[State]:
        """Return a fresh multi-subject :class:`Lifecycle` under this machine."""
        return Lifecycle(self)

    def mermaid(self) -> str:
        """Render the table as a Mermaid ``stateDiagram-v2``."""
        lines = ["stateDiagram-v2"]
        lines.extend(f"    [*] --> {state.value}" for state in sorted(self.initial))
        lines.extend(
            f"    {source.value} --> {target.value}: {event}"
            for source, event, target in self.transitions
        )
        lines.extend(f"    {state.value} --> [*]" for state in sorted(self.final))
        return "\n".join(lines)


class TransitionJournal:
    """Appends every move of one run, accepted or refused, as JSON lines.

    Attributes:
        path: The journal file.
    """

    def __init__(self, path: Path) -> None:
        """Bind the journal file; its directory is created on the first move."""
        self.path = path
        self._lock = threading.Lock()

    def record(
        self,
        machine: str,
        subject: str,
        source: StrEnum | None,
        event: str,
        target: StrEnum | None,
    ) -> None:
        """Append one move; a None target records a refused move."""
        row = {
            "at": datetime.now(UTC).isoformat(),
            "machine": machine,
            "subject": subject,
            "from": None if source is None else source.value,
            "event": event,
            "to": None if target is None else target.value,
            "refused": target is None,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")

    def rows(self) -> tuple[dict[str, object], ...]:
        """Read every recorded move in order."""
        if not self.path.exists():
            return ()
        text = self.path.read_text(encoding="utf-8")
        return tuple(json.loads(line) for line in text.splitlines() if line)


class TransitionRecord(ContractModel):
    """One attempted move of one subject followed by a :class:`MachineRun`.

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


class MachineRun[State: StrEnum]:
    """One subject's walk through a machine, every attempt checked and recorded."""

    def __init__(
        self,
        machine: StateMachine[State],
        subject: str,
        record: Callable[[JsonObject], None],
    ) -> None:
        """Start ``subject`` in the machine's only initial state.

        Args:
            machine: The declared machine.
            subject: What the run follows.
            record: Durable sink for every attempted move; it returns only after recording.
        """
        self.machine, self.subject, self._record = machine, subject, record
        self.state: State = machine.start
        self.history: list[TransitionRecord] = []

    def fire(self, event: str) -> State:
        """Attempt one move, recording it whether or not the table allows it.

        Args:
            event: The event to apply; it must lead to exactly one target.

        Returns:
            The new state.

        Raises:
            IllegalTransition: The table has no single move for this state and event; the state
                is unchanged and the refusal is recorded.
        """
        found = self.machine.targets(self.state, event)
        target = found[0] if len(found) == 1 else None
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


class Transition(ContractModel):
    """One legal move a :class:`Lifecycle` applied, as persisted on its owning record.

    Attributes:
        machine: Name of the machine.
        subject: What moved.
        source: State before.
        target: State after.
        reason: Why, in one line.
    """

    machine: str
    subject: str
    source: str
    target: str
    reason: str = Field(default="", max_length=500)


class Refusal(ContractModel):
    """One illegal move a :class:`Lifecycle` refused; the subject stayed in ``source``.

    Attributes:
        machine: Name of the machine.
        subject: What the move named.
        source: State the subject stayed in.
        target: The refused target.
        reason: Why the move was attempted, in one line.
    """

    machine: str
    subject: str
    source: str
    target: str
    reason: str = Field(default="", max_length=500)


class Lifecycle[State: StrEnum]:
    """Current states of several subjects under one machine whose events name their targets.

    A move to ``target`` is the table row ``(source, target.value, target)``: these machines
    declare one event per target state (see :func:`target_table`), so a persisted move is its
    two states.

    Attributes:
        machine: The declared machine.
        current: Each subject's state.
        transitions: Every applied move, in order.
        refusals: Every refused move, in order.
    """

    def __init__(self, machine: StateMachine[State]) -> None:
        """Start with no subjects."""
        self.machine = machine
        self.current: dict[str, State] = {}
        self.transitions: list[Transition] = []
        self.refusals: list[Refusal] = []

    def start(self, subject: str) -> Self:
        """Place a new subject in the machine's only initial state.

        Raises:
            ValueError: The subject already started.
        """
        if subject in self.current:
            raise ValueError(f"{self.machine.name}: {subject} already started")
        self.current[subject] = self.machine.start
        return self

    def move(self, subject: str, target: State, reason: str = "") -> bool:
        """Apply a legal move, or refuse and record an illegal one; return whether it applied."""
        source = self.current[subject]
        row = {
            "machine": self.machine.name,
            "subject": subject,
            "source": source.value,
            "target": target.value,
            "reason": reason[:500],
        }
        if not self.machine.allows(source, target.value, target):
            self.refusals.append(Refusal.model_validate(row))
            return False
        self.transitions.append(Transition.model_validate(row))
        self.current[subject] = target
        return True

    def state(self, subject: str) -> State:
        """Return the subject's current state."""
        return self.current[subject]


def target_table[State: StrEnum](
    edges: dict[State, frozenset[State]],
) -> tuple[tuple[State, str, State], ...]:
    """Return ``(source, target, target)`` rows for a machine whose events name their targets.

    Args:
        edges: Each state's legal targets.

    Returns:
        The rows, in the order the states and their targets are listed.
    """
    rows: list[tuple[State, str, State]] = []
    for source, targets in edges.items():
        order = list(type(source))
        rows.extend((source, t.value, t) for t in sorted(targets, key=order.index))
    return tuple(rows)
