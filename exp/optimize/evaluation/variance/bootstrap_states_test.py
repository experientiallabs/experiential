"""A bootstrap task's status moves only through its declared table."""

from __future__ import annotations

import pytest

from exp.common.state_machine import IllegalTransition
from exp.optimize.evaluation.variance.bootstrap_states import TASK_MACHINE, TaskState


def test_a_saved_outcome_never_turns_into_another_outcome() -> None:
    """Registered never becomes refused (or back) without failing and retrying first."""
    with pytest.raises(IllegalTransition):
        TASK_MACHINE.target(TaskState.REGISTERED, "refused")
    with pytest.raises(IllegalTransition):
        TASK_MACHINE.target(TaskState.REFUSED, "registered")
    assert TASK_MACHINE.target(TaskState.REGISTERED, "failed") == TaskState.FAILED
    assert TASK_MACHINE.target(TaskState.FAILED, "registered") == TaskState.REGISTERED
    assert "failed --> registered: registered" in TASK_MACHINE.mermaid()
