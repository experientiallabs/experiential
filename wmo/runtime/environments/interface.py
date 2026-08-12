"""The executable-environment boundary owned by a simulator."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from pydantic import Field

from wmo.common.core.artifacts import ContractModel, JsonObject
from wmo.common.models import ToolCall
from wmo.common.tasks import TaskCase


class Observation(ContractModel):
    """The structured result of one executable tool call."""

    content: str
    is_error: bool = False
    metadata: JsonObject = Field(default_factory=dict)


@runtime_checkable
class EnvironmentSession(Protocol):
    """Executes actions inside one environment already opened by a simulator."""

    def execute(self, action: ToolCall) -> Observation:
        """Execute one tool action and return its observation.

        Args:
            action: The complete tool invocation emitted by the customer agent.

        Returns:
            The structured observation produced by the executable environment.
        """


@runtime_checkable
class EnvironmentRuntime(Protocol):
    """Creates isolated environment sessions and owns their context-exit cleanup."""

    def open(self, task: TaskCase) -> AbstractContextManager[EnvironmentSession]:
        """Open a clean environment session for one task.

        Args:
            task: The task whose executable state the session will isolate.

        Returns:
            A context manager that releases the session when it exits.
        """
