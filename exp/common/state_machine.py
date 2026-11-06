"""Explicit state machines: a state enum, one plain transition table, checked and recorded moves.

A workflow whose progress lives in scattered flags is an implicit state machine nobody can
review. Declare it instead:

- ``states``: a ``StrEnum`` of every state;
- ``transitions``: one tuple of ``(source, event, target)`` rows, plain data, so a model-checking
  spec can be derived from it later;
- :meth:`StateMachine.move` checks every move against the table and refuses an illegal one with
  :class:`IllegalTransition`; a :class:`TransitionJournal` persists each accepted and refused
  move as one JSON line;
- :meth:`StateMachine.mermaid` renders the table as a Mermaid statechart, the review surface.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class IllegalTransition(ValueError):
    """A move the machine's transition table does not contain."""


@dataclass(frozen=True)
class StateMachine[State: StrEnum]:
    """One declared machine.

    Attributes:
        name: Machine name, recorded with every move.
        states: Every state.
        initial: States a new subject may start in.
        terminal: States no transition leaves.
        transitions: Every legal ``(source, event, target)`` move.
    """

    name: str
    states: type[State]
    initial: frozenset[State]
    terminal: frozenset[State]
    transitions: tuple[tuple[State, str, State], ...]

    def __post_init__(self) -> None:
        """Refuse a table that leaves a terminal state or names one event twice from a state."""
        seen: set[tuple[State, str]] = set()
        for source, event, _ in self.transitions:
            if source in self.terminal:
                raise ValueError(f"{self.name}: terminal state {source} has a transition")
            if (source, event) in seen:
                raise ValueError(f"{self.name}: event {event} leaves {source} twice")
            seen.add((source, event))

    def target(self, source: State | None, event: str) -> State:
        """Return the state an event leads to.

        Args:
            source: The current state, or None for a new subject.
            event: The event.

        Returns:
            The next state.

        Raises:
            IllegalTransition: The table has no such move.
        """
        if source is None:
            for state in self.initial:
                if state.value == event:
                    return state
            raise IllegalTransition(f"{self.name}: a new subject cannot start with {event}")
        for row_source, row_event, row_target in self.transitions:
            if row_source == source and row_event == event:
                return row_target
        raise IllegalTransition(f"{self.name}: {event} is not legal from {source}")

    def move(
        self,
        subject: str,
        source: State | None,
        event: str,
        journal: TransitionJournal | None = None,
    ) -> State:
        """Check one move against the table, persist it, and return the new state.

        Args:
            subject: What moves (a stage or task identity).
            source: Its current state, or None when it is new; a new subject's event names its
                initial state.
            event: The event.
            journal: Where the accepted or refused move is persisted.

        Returns:
            The new state.

        Raises:
            IllegalTransition: The move is not in the table; it is recorded as refused first.
        """
        try:
            target = self.target(source, event)
        except IllegalTransition:
            if journal is not None:
                journal.record(self.name, subject, source, event, None)
            raise
        if journal is not None:
            journal.record(self.name, subject, source, event, target)
        return target

    def mermaid(self) -> str:
        """Render the table as a Mermaid ``stateDiagram-v2``."""
        lines = ["stateDiagram-v2"]
        lines.extend(f"    [*] --> {state.value}" for state in sorted(self.initial))
        lines.extend(
            f"    {source.value} --> {target.value}: {event}"
            for source, event, target in self.transitions
        )
        lines.extend(f"    {state.value} --> [*]" for state in sorted(self.terminal))
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
