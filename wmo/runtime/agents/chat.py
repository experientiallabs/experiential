"""Built-in bounded chat agent for standard model and tool request loops."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from wmo.common.models import ModelClient, ModelFinishReason, ModelMessage, ModelRequest, Usage
from wmo.common.rollouts import RolloutEventKind, RolloutSpan, StopReason
from wmo.common.tasks import TaskCase
from wmo.runtime.agents.interface import AgentEpisode
from wmo.runtime.environments import EnvironmentSession, Observation

_DEFAULT_MAXIMUM_MODEL_CALLS = 50


class ChatAgentRuntime:
    """Run a bounded assistant and task-tool loop through WMO's canonical model boundary."""

    def __init__(self, *, maximum_model_calls: int = _DEFAULT_MAXIMUM_MODEL_CALLS) -> None:
        """Configure the hard model-call ceiling for each isolated episode.

        Args:
            maximum_model_calls: Positive maximum number of model requests in one episode.

        Raises:
            ValueError: The ceiling is not positive.
        """
        if maximum_model_calls <= 0:
            raise ValueError("maximum_model_calls must be positive")
        self._maximum_model_calls = maximum_model_calls

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        """Execute model actions until a final response or the configured call ceiling.

        Args:
            task: Real task instruction, context, and task-visible tool schemas.
            model: Candidate completion client selected for this evaluation cell.
            environment: Execute-only environment capability owned by the simulator.

        Returns:
            Ordered model, tool, and observation spans with the terminal assistant action.

        Raises:
            ValueError: The model invokes a tool outside the task-visible schemas.
        """
        messages = [ModelMessage(role="user", content=_task_prompt(task))]
        events: list[RolloutSpan] = []
        usages: list[Usage] = []
        visible_tools = {tool.name for tool in task.tools}
        final_action = None
        for step in range(self._maximum_model_calls):
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
                    raise ValueError(
                        f"candidate requested tool {call.name!r}, which is not visible to the task"
                    )
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
        return AgentEpisode(
            events=tuple(events),
            final_action=final_action,
            stop_reason=StopReason.MAXIMUM_STEPS,
            usage=_combined_usage(usages),
        )


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

    context = json.dumps(task.initial_context, sort_keys=True, separators=(",", ":"))
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
    from wmo.common.models import ToolCall

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
