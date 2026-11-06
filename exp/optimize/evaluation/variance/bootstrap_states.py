"""The declared state machine of one bootstrap task's manifest status.

A task enters the batch manifest as ``registered`` or ``refused`` (saved outcomes, final) or
``failed`` (retried on resume). A failed task fails again or ends registered or refused. A
saved outcome becomes failed only when its pinned trace evidence no longer matches on replay.
Rewriting a saved status with itself on replay is no move. No state is terminal: changed
evidence can always fail a task, and a rerun can always retry it. Every move is
checked against this table and journaled beside the manifest
(``<batch>.transitions.jsonl``); an illegal move is refused and recorded.
"""

from __future__ import annotations

from enum import StrEnum

from exp.common.state_machine import StateMachine


class TaskState(StrEnum):
    """One bootstrap task's manifest status.

    Attributes:
        REGISTERED: A saved, validated registration.
        REFUSED: A saved diagnostic refusal.
        FAILED: An exception stopped it; a rerun retries it.
    """

    REGISTERED = "registered"
    REFUSED = "refused"
    FAILED = "failed"


TASK_MACHINE = StateMachine(
    name="bootstrap-task",
    states=TaskState,
    initial=frozenset(TaskState),
    terminal=frozenset(),
    transitions=(
        (TaskState.REGISTERED, "failed", TaskState.FAILED),
        (TaskState.REFUSED, "failed", TaskState.FAILED),
        (TaskState.FAILED, "failed", TaskState.FAILED),
        (TaskState.FAILED, "registered", TaskState.REGISTERED),
        (TaskState.FAILED, "refused", TaskState.REFUSED),
    ),
)
"""Every legal move of one bootstrap task's status."""
