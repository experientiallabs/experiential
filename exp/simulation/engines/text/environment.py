"""Tool-free environment capability used exclusively by text world-model episodes."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass

from exp.common.models import ToolCall
from exp.common.tasks import TaskCase
from exp.runtime.environments import EnvironmentSession, Observation


class TextOnlyToolUseError(RuntimeError):
    """A text-mode agent attempted a tool action that requires sandbox simulation."""


@dataclass(frozen=True)
class TextOnlyEnvironmentSession:
    """Execute-only session that rejects every tool invocation in text simulation."""

    def execute(self, action: ToolCall) -> Observation:
        """Reject an executable tool call without silently changing the task.

        Args:
            action: Tool action emitted by the customer agent.

        Raises:
            TextOnlyToolUseError: Text world-model simulation does not execute tools.
        """
        raise TextOnlyToolUseError(
            f"text world-model simulation cannot execute tool {action.name!r}; use sandbox mode"
        )


class TextOnlyEnvironmentRuntime:
    """Open one no-tools session for a text-only candidate episode."""

    def open(self, task: TaskCase) -> AbstractContextManager[EnvironmentSession]:
        """Return a context manager whose session rejects all tools.

        Args:
            task: Task passed through solely to satisfy the shared environment contract.

        Returns:
            A tool-free environment session that closes without side effects.
        """
        del task
        return _TextOnlyEnvironmentContext()


class _TextOnlyEnvironmentContext(AbstractContextManager[EnvironmentSession]):
    """Stateless context manager for a tool-free episode capability."""

    def __enter__(self) -> EnvironmentSession:
        """Return the one stateless session exposed for this episode."""
        return TextOnlyEnvironmentSession()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> bool:
        """Close without suppressing the customer-agent exception, if any."""
        del exception_type, exception, traceback
        return False
