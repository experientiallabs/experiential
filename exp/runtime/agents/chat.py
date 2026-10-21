"""Built-in bounded chat agent for standard model and tool request loops."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Literal, cast

from pydantic import Field

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.common.models import (
    ModelClient,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ToolCall,
    Usage,
)
from exp.common.rollouts import RolloutEventKind, RolloutSpan, StopReason
from exp.common.tasks import TaskCase
from exp.runtime.agents.interface import AgentEpisode
from exp.runtime.environments import EnvironmentSession, Observation

_DEFAULT_MAXIMUM_MODEL_CALLS = 100
_MAXIMUM_SYSTEM_PROMPT_CHARACTERS = 20_000


class InvisibleToolCallError(ValueError):
    """A proposed actor call refused before environment dispatch.

    Attributes:
        action: Exact refused call for operator evidence and attempted-action grading.
    """

    def __init__(self, action: ToolCall) -> None:
        """Retain the action without copying its arguments into the exception message."""
        self.action = action
        super().__init__(
            f"candidate requested tool {action.name!r}, which is not visible to the task"
        )


class ChatCheckpoint(ContractModel):
    """Actor-visible state at a completed tool turn, with no hidden model process state.

    Attributes:
        task_sha256: Exact public task including tools and initial context.
        runtime_sha256: Chat policy and original model-call ceiling.
        messages: Complete actor-visible conversation, excluding future outcomes.
        calls: Completed model calls charged against the original ceiling.
    """

    task_sha256: Sha256
    runtime_sha256: Sha256
    messages: tuple[ModelMessage, ...] = Field(min_length=1)
    calls: int = Field(ge=0, strict=True)


class ChatAgentRuntime:
    """Run a bounded assistant and task-tool loop through EXP's canonical model boundary."""

    def __init__(
        self,
        *,
        maximum_model_calls: int = _DEFAULT_MAXIMUM_MODEL_CALLS,
        system_prompt: str | None = None,
    ) -> None:
        """Configure the hard model-call ceiling for each isolated episode.

        Args:
            maximum_model_calls: Positive maximum number of model requests in one episode.
            system_prompt: Optional canonical instruction prepended to every candidate request.

        Raises:
            ValueError: The ceiling or optional system prompt is invalid.
        """
        if maximum_model_calls < 1:
            raise ValueError("maximum_model_calls must be positive")
        self._maximum_model_calls = maximum_model_calls
        self._system_prompt = normalize_chat_system_prompt(system_prompt)

    @property
    def identity(self) -> Sha256:
        """Bind the stateless actor protocol, instructions and original call ceiling."""
        return sha256_json(
            {
                "protocol": "checkpointable-chat-v1",
                "system_prompt": self._system_prompt,
                "maximum_model_calls": self._maximum_model_calls,
            }
        )

    def remaining_calls(self, checkpoint: ChatCheckpoint) -> int:
        """Return the captured policy's remaining allowance, refusing another runtime."""
        if checkpoint.runtime_sha256 != self.identity:
            raise ValueError("actor checkpoint belongs to another runtime policy")
        return max(0, self._maximum_model_calls - checkpoint.calls)

    def resume(
        self,
        task: TaskCase,
        checkpoint: ChatCheckpoint,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
        on_checkpoint: Callable[[ChatCheckpoint], None] | None = None,
    ) -> AgentEpisode:
        """Continue a completed turn without replaying its model or tool calls.

        Args:
            task: Same public task, permissions and tools as the captured prefix.
            checkpoint: Full visible conversation at a quiescent turn boundary.
            model: Fresh sampling client for this continuation's registered policy.
            environment: World already restored to the same boundary by its owner.
            on_checkpoint: Optional trusted observer; never passed to the actor model.

        Returns:
            Only events and usage bought after the checkpoint.
        """
        if (
            checkpoint.task_sha256 != sha256_json(task)
            or checkpoint.runtime_sha256 != self.identity
        ):
            raise ValueError("actor checkpoint differs from task or runtime; use its frozen inputs")
        if not self.remaining_calls(checkpoint):
            raise ValueError(
                "actor checkpoint has no remaining calls; select an earlier checkpoint"
            )
        return self._continue(task, model, environment, checkpoint, on_checkpoint)

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
        on_checkpoint: Callable[[ChatCheckpoint], None] | None = None,
    ) -> AgentEpisode:
        """Execute model actions until a final response or the configured call ceiling.

        Args:
            task: Real task instruction, context, and task-visible tool schemas.
            model: Candidate completion client selected for this evaluation cell.
            environment: Execute-only environment capability owned by the simulator.
            on_checkpoint: Trusted callback after each fully observed tool turn and at startup.

        Returns:
            Ordered model, tool, and observation spans with the terminal assistant action.

        Raises:
            ValueError: The model invokes a tool outside the task-visible schemas.
        """
        checkpoint = self.initial_checkpoint(task)
        if on_checkpoint is not None:
            on_checkpoint(checkpoint)
        return self._continue(task, model, environment, checkpoint, on_checkpoint)

    def initial_checkpoint(self, task: TaskCase) -> ChatCheckpoint:
        """Construct the exact initial actor context without credentials or provider calls."""
        messages = (
            [ModelMessage(role="system", content=self._system_prompt)]
            if self._system_prompt is not None
            else []
        )
        messages.extend(task_messages(task))
        return ChatCheckpoint(
            task_sha256=sha256_json(task),
            runtime_sha256=self.identity,
            messages=tuple(messages),
            calls=0,
        )

    def _continue(
        self,
        task: TaskCase,
        model: ModelClient,
        environment: EnvironmentSession,
        checkpoint: ChatCheckpoint,
        on_checkpoint: Callable[[ChatCheckpoint], None] | None,
    ) -> AgentEpisode:
        """Run the single chat loop from its exact visible prefix and remaining allowance."""
        messages = list(checkpoint.messages)
        events: list[RolloutSpan] = []
        usages: list[Usage] = []
        visible_tools = {tool.name for tool in task.tools}
        final_action = None
        for step in range(checkpoint.calls, self._maximum_model_calls):
            started_at = datetime.now(UTC)
            response = model.complete(
                ModelRequest(
                    messages=tuple(messages),
                    tools=task.tools,
                    tool_choice="auto" if task.tools else None,
                )
            )
            ended_at = datetime.now(UTC)
            usage = response.economics.usage
            if usage is not None:
                usages.append(usage)
            events.append(
                RolloutSpan(
                    span_id=f"chat-model-{step}",
                    kind=RolloutEventKind.AGENT_MODEL_CALL,
                    started_at=started_at,
                    ended_at=ended_at,
                    payload={"step": step},
                    model=response.model,
                    usage=usage,
                )
            )
            final_action = response.output
            messages.append(ModelMessage(role="assistant", assistant_action=response.output))
            if response.finish_reason == ModelFinishReason.LENGTH:
                return AgentEpisode(
                    events=tuple(events),
                    final_action=final_action,
                    stop_reason=StopReason.LENGTH,
                    usage=_combined_usage(usages),
                )
            if not response.output.tool_calls:
                return AgentEpisode(
                    events=tuple(events),
                    final_action=final_action,
                    stop_reason=StopReason.COMPLETED,
                    usage=_combined_usage(usages),
                )
            for index, call in enumerate(response.output.tool_calls):
                if call.name not in visible_tools:
                    raise InvisibleToolCallError(call)
                observation = _execute_and_record(
                    environment,
                    call,
                    step=step,
                    index=index,
                    events=events,
                )
                messages.append(
                    ModelMessage(
                        role="tool",
                        content=observation.content,
                        tool_call_id=call.call_id,
                    )
                )
            if on_checkpoint is not None:
                on_checkpoint(
                    ChatCheckpoint(
                        task_sha256=checkpoint.task_sha256,
                        runtime_sha256=self.identity,
                        messages=tuple(messages),
                        calls=step + 1,
                    )
                )
        return AgentEpisode(
            events=tuple(events),
            final_action=final_action,
            stop_reason=StopReason.MAXIMUM_STEPS,
            usage=_combined_usage(usages),
        )


def normalize_chat_system_prompt(value: str | None) -> str | None:
    """Return one canonical optional built-in instruction or reject an unsafe shape.

    Args:
        value: Optional system prompt supplied by Project setup.

    Returns:
        Trimmed prompt or ``None`` when no hosted system was selected.

    Raises:
        ValueError: The prompt is blank or exceeds the shared Project contract.
    """
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError("system_prompt must not be blank")
    if len(normalized) > _MAXIMUM_SYSTEM_PROMPT_CHARACTERS:
        raise ValueError("system_prompt exceeds 20000 characters")
    return normalized


def task_messages(task: TaskCase) -> tuple[ModelMessage, ...]:
    """Render the same public instructions and initial facts for every maintained actor adapter.

    Args:
        task: Public task contract; reference answers and evaluator metadata are not rendered.

    Returns:
        Captured system/developer instructions followed by the user task and public context.
    """
    messages = []
    captured = task.initial_context.get("instruction_messages", [])
    if isinstance(captured, list):
        for instruction in captured:
            if isinstance(instruction, dict) and instruction.get("role") in ("system", "developer"):
                content = instruction.get("content")
                if isinstance(content, str):
                    messages.append(
                        ModelMessage(
                            role=cast(Literal["system", "developer"], instruction["role"]),
                            content=content,
                        )
                    )
    return (*messages, ModelMessage(role="user", content=_task_prompt(task)))


def _task_prompt(task: TaskCase) -> str:
    """Render the request-visible task instruction and optional initial context.

    Args:
        task: Task containing the instruction and normalized initial context.

    Returns:
        Stable user message presented to every candidate.
    """
    if not task.initial_context:
        return task.instruction
    import json

    context = json.dumps(
        {
            key: value
            for key, value in task.initial_context.items()
            if key != "instruction_messages"
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{task.instruction}\n\nInitial context:\n{context}"


def _execute_and_record(
    environment: EnvironmentSession,
    call: object,
    *,
    step: int,
    index: int,
    events: list[RolloutSpan],
) -> Observation:
    """Execute one typed tool call and append its call and observation spans.

    Args:
        environment: Execute-only environment capability.
        call: Candidate-emitted tool call.
        step: Zero-based model-call index.
        index: Zero-based tool index within the assistant action.
        events: Ordered episode span accumulator.

    Returns:
        Environment observation appended to the next model request.

    Raises:
        TypeError: The supplied action is not a canonical tool call.
    """
    if not isinstance(call, ToolCall):
        raise TypeError("chat agent tool actions must use ToolCall")
    call_started_at = datetime.now(UTC)
    events.append(
        RolloutSpan(
            span_id=f"chat-tool-{step}-{index}",
            kind=RolloutEventKind.TOOL_CALL,
            started_at=call_started_at,
            ended_at=call_started_at,
            payload={"call_id": call.call_id, "arguments": call.arguments},
            tool_name=call.name,
        )
    )
    observation = environment.execute(call)
    observed_at = datetime.now(UTC)
    events.append(
        RolloutSpan(
            span_id=f"chat-observation-{step}-{index}",
            parent_span_id=f"chat-tool-{step}-{index}",
            kind=RolloutEventKind.OBSERVATION,
            started_at=observed_at,
            ended_at=observed_at,
            payload={"is_error": observation.is_error, "metadata": observation.metadata},
            tool_name=call.name,
        )
    )
    return observation


def _combined_usage(values: Sequence[Usage]) -> Usage | None:
    """Sum observed token usage without inventing unknown cached-token values.

    Args:
        values: Per-request provider usage records.

    Returns:
        Episode totals, or ``None`` when no request reported usage.
    """
    if not values:
        return None
    cached_values = tuple(value.cached_input_tokens for value in values)
    cached_total = (
        sum(value for value in cached_values if value is not None)
        if all(value is not None for value in cached_values)
        else None
    )
    return Usage(
        input_tokens=sum(value.input_tokens for value in values),
        output_tokens=sum(value.output_tokens for value in values),
        cached_input_tokens=cached_total,
    )
