"""Capture source-located actor context at the actual canonical request boundary."""

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, sha256_json
from exp.common.models import ModelClient, ModelRequest, ModelResponse
from exp.common.tasks import TaskCase

ContextSource = Literal[
    "system_prompt", "skill_description", "hook", "reminder", "memory", "appended_prompt"
]
_SOURCES = (
    "system_prompt",
    "tool_description",
    "skill_description",
    "hook",
    "reminder",
    "memory",
    "appended_prompt",
)


class ActorContextFragment(ContractModel):
    """An explicit source annotation for an instruction actually delivered to the actor.

    Attributes:
        fragment_id: Unique identity within this council, without evaluator annotations.
        source: Actor-visible origin, never the simulator memory bank.
        role: Instruction channel supported by the maintained chat runtime.
        content: Exact text, including any deliberate conflicting instructions.
    """

    fragment_id: str = Field(min_length=1)
    source: ContextSource
    role: Literal["system", "developer"] = "developer"
    content: str = Field(min_length=1)


def with_actor_context(task: TaskCase, fragments: Sequence[ActorContextFragment]) -> TaskCase:
    """Append source-annotated instructions without changing simulator initialization."""
    original = task.initial_context.get("instruction_messages", [])
    if not isinstance(original, list):
        raise ValueError("actor instruction_messages must be a list before adding context")
    rows = [*original, *[fragment.model_dump(mode="json") for fragment in fragments]]
    return task.model_copy(
        update={"initial_context": {**task.initial_context, "instruction_messages": rows}}
    )


class ContextRecordingClient:
    """Retain actual request fragments without modifying messages or provider ownership."""

    def __init__(self, client: ModelClient, task: TaskCase) -> None:
        """Bind the metered actor client and its public source annotations."""
        self._client, self._task = client, task
        self._requests: list[JsonObject] = []
        self._fragments: list[JsonObject] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Capture context even when the subsequent provider attempt fails."""
        turn = len(self._requests)
        self._requests.append({"turn": turn, "request_sha256": sha256_json(request)})
        original = self._task.initial_context.get("instruction_messages", [])
        annotations = (
            [
                row
                for row in original
                if isinstance(row, dict)
                and row.get("role") in ("system", "developer")
                and isinstance(row.get("content"), str)
            ]
            if isinstance(original, list)
            else []
        )
        for index, message in enumerate(request.messages):
            source, origin = "conversation", f"messages[{index}]"
            if message.role in ("system", "developer"):
                source = "system_prompt" if message.role == "system" else "unclassified"
                if isinstance(annotations, list) and index < len(annotations):
                    row = annotations[index]
                    if (
                        isinstance(row, dict)
                        and row.get("role") == message.role
                        and row.get("content") == message.content
                    ):
                        source = str(row.get("source", source))
                        origin = str(row.get("fragment_id", origin))
            elif message.role == "user":
                source = "task_and_instance_context"
            elif message.role == "tool":
                source = "tool_observation"
            content = message.content if message.content is not None else message.model_dump_json()
            self._add(turn, origin, source, message.role, content, f"messages[{index}]")
        for index, tool in enumerate(request.tools):
            self._add(
                turn,
                tool.name,
                "tool_description",
                "tool",
                tool.model_dump_json(),
                f"tools[{index}]",
            )
        return self._client.complete(request)

    def _add(
        self, turn: int, origin: str, source: str, role: str, content: str, location: str
    ) -> None:
        """Bind a fragment to its delivery position and exact content."""
        self._fragments.append(
            {
                "id": f"turn-{turn}:{location}",
                "origin": origin,
                "source": source,
                "role": role,
                "content": content,
                "content_sha256": sha256_json(content),
                "turn": turn,
                "location": location,
            }
        )

    def manifest(self) -> JsonObject:
        """Return all delivered fragments and explicitly absent source categories."""
        present = {row["source"] for row in self._fragments}
        return {
            "boundary": "canonical actor ModelRequest before output-budget framing",
            "requests": self._requests,
            "fragments": self._fragments,
            "absent_sources": [source for source in _SOURCES if source not in present],
            "simulator_memory_is_actor_context": False,
        }
