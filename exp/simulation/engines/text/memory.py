"""The seam through which learned memory reaches actor-evaluation world-model turns."""

from collections.abc import Sequence
from typing import Protocol

from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction, ModelMessage
from exp.common.tasks import TaskCase
from exp.simulation.engines.text.prompt import MemoryContext


class TurnMemory(Protocol):
    """Memory one task may read, recalled for each simulated turn of its episodes."""

    def context(
        self,
        visible_messages: Sequence[ModelMessage],
        candidate_response: AssistantAction,
        state: JsonObject,
    ) -> MemoryContext:
        """Recall the memory relevant to one candidate action.

        Args:
            visible_messages: Candidate-visible conversation through the latest request.
            candidate_response: The action the world model is about to answer.
            state: Private environment state carried from the previous turn.

        Returns:
            Selected records and the overview, split by authority.
        """
        ...


class TaskMemory(Protocol):
    """A frozen memory that hands each evaluated task the view it may read."""

    def for_task(self, task: TaskCase) -> TurnMemory:
        """The memory one task may read, never including the task's own evidence.

        Args:
            task: Evaluated task, including its lineage and source traces.

        Returns:
            Per-turn recall over that task's eligible view.
        """
        ...
