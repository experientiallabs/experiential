"""Built-in bounded chat agent tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from wmo.common.models import (
    AssistantAction,
    ModelCapabilities,
    ModelFinishReason,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
    Usage,
)
from wmo.common.rollouts import RolloutEventKind, StopReason
from wmo.common.tasks import TaskCase, ToolSchema
from wmo.runtime.agents.chat import ChatAgentRuntime
from wmo.runtime.environments import Observation


def test_chat_agent_executes_tools_and_returns_final_response() -> None:
    """The built-in agent performs the standard assistant, tool, assistant loop."""
    model = _Model(
        (
            _response(
                AssistantAction(
                    tool_calls=(
                        ToolCall(call_id="call-1", name="lookup", arguments={"query": "reset"}),
                    )
                ),
                usage=Usage(input_tokens=4, output_tokens=2, cached_input_tokens=1),
            ),
            _response(
                AssistantAction(content="Reset link sent."),
                usage=Usage(input_tokens=7, output_tokens=3, cached_input_tokens=2),
            ),
        )
    )
    environment = _Environment()

    episode = ChatAgentRuntime().run(_task(), model=model, environment=environment)

    assert environment.calls == ["lookup"]
    assert episode.final_action == AssistantAction(content="Reset link sent.")
    assert episode.stop_reason == StopReason.COMPLETED
    assert episode.usage == Usage(input_tokens=11, output_tokens=5, cached_input_tokens=3)
    assert tuple(event.kind for event in episode.events) == (
        RolloutEventKind.AGENT_MODEL_CALL,
        RolloutEventKind.TOOL_CALL,
        RolloutEventKind.OBSERVATION,
        RolloutEventKind.AGENT_MODEL_CALL,
    )
    second_request = model.requests[1]
    assert second_request.messages[-1].role == "tool"
    assert second_request.messages[-1].tool_call_id == "call-1"


def test_chat_agent_stops_after_hard_model_call_ceiling() -> None:
    """A tool loop cannot issue an unbounded number of candidate requests."""
    tool_action = AssistantAction(
        tool_calls=(ToolCall(call_id="call-1", name="lookup", arguments={}),)
    )
    model = _Model((_response(tool_action), _response(tool_action)))

    episode = ChatAgentRuntime(maximum_model_calls=2).run(
        _task(), model=model, environment=_Environment()
    )

    assert len(model.requests) == 2
    assert episode.stop_reason == StopReason.MAXIMUM_STEPS


def test_chat_agent_rejects_tools_outside_task_schema() -> None:
    """Candidate tool names cannot escape the real task-visible interface."""
    model = _Model(
        (
            _response(
                AssistantAction(
                    tool_calls=(ToolCall(call_id="call-1", name="delete", arguments={}),)
                )
            ),
        )
    )

    with pytest.raises(ValueError, match="not visible"):
        ChatAgentRuntime().run(_task(), model=model, environment=_Environment())


def test_chat_agent_preserves_length_terminal_reason() -> None:
    """A provider length stop is terminal and does not execute emitted tools."""
    model = _Model(
        (
            _response(
                AssistantAction(content="partial"),
                finish_reason=ModelFinishReason.LENGTH,
            ),
        )
    )

    episode = ChatAgentRuntime().run(_task(), model=model, environment=_Environment())

    assert episode.stop_reason == StopReason.LENGTH
    assert episode.final_action == AssistantAction(content="partial")


def test_chat_agent_omits_tool_choice_when_task_has_no_tools() -> None:
    """Native provider requests do not receive a tool directive without schemas."""
    model = _Model((_response(AssistantAction(content="done")),))
    task = _task().model_copy(update={"tools": ()})

    ChatAgentRuntime().run(task, model=model, environment=_Environment())

    assert model.requests[0].tools == ()
    assert model.requests[0].tool_choice is None


class _Model:
    """Return a finite sequence of deterministic candidate responses."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        """Store response fixtures and initialize request capture.

        Args:
            responses: Responses returned in request order.
        """
        self._responses = iter(responses)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Record one request and return its next fixture response.

        Args:
            request: Canonical candidate request.

        Returns:
            Next configured response.
        """
        self.requests.append(request)
        return next(self._responses)


class _Environment:
    """Record tool names and return one deterministic observation."""

    def __init__(self) -> None:
        """Initialize empty tool-call capture."""
        self.calls: list[str] = []

    def execute(self, action: ToolCall) -> Observation:
        """Record the call and return a stable tool result.

        Args:
            action: Candidate-emitted task tool call.

        Returns:
            Deterministic observation.
        """
        self.calls.append(action.name)
        return Observation(content="customer@example.com")


def _task() -> TaskCase:
    """Return one fit task with a single visible lookup tool."""
    return TaskCase(
        task_id="task-1",
        lineage_group_id="lineage-1",
        partition="fit",
        instruction="Help reset a password",
        tools=(
            ToolSchema(
                name="lookup",
                description="Look up the account",
                input_schema={"type": "object"},
            ),
        ),
        workload_weight=1,
        source_trace_ids=("trace-1",),
    )


def _response(
    action: AssistantAction,
    *,
    usage: Usage | None = None,
    finish_reason: ModelFinishReason = ModelFinishReason.COMPLETED,
) -> ModelResponse:
    """Return one deterministic model response fixture.

    Args:
        action: Assistant output returned by the fake candidate.
        usage: Optional observed token usage.
        finish_reason: Provider terminal reason.

    Returns:
        Canonical model response fixture.
    """
    capabilities = ModelCapabilities(supports_completions=True)
    return ModelResponse(
        output=action,
        model=ModelSnapshot(
            provider="fixture",
            model_id="candidate",
            capabilities_sha256=capabilities.identity_sha256(),
            connection_sha256="a" * 64,
        ),
        economics=OperationEconomics(usage=usage),
        finish_reason=finish_reason,
    )
