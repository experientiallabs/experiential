"""Tests for simulator-owned customer-agent episode lifecycle composition."""

from __future__ import annotations

from datetime import UTC, datetime
from types import TracebackType
from typing import cast

import pytest

from wmo.common.core.artifacts import FailureAttribution, FailureCode, StructuredFailure
from wmo.common.models import AssistantAction, ModelClient, ModelRequest, ModelResponse, ToolCall
from wmo.common.rollouts import RolloutEventKind, RolloutSpan, StopReason
from wmo.common.tasks import TaskCase
from wmo.runtime.agents.interface import AgentAdapterPreflightError, AgentEpisode, AgentRuntime
from wmo.runtime.agents.lifecycle import execute_agent_episode
from wmo.runtime.environments import EnvironmentResetError, EnvironmentSession, Observation

_EVENT_TIME = datetime(2026, 8, 11, tzinfo=UTC)
_LIFECYCLE_EVENT_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def test_complete_episode_injects_model_and_hides_environment_lifecycle() -> None:
    runtime = _EnvironmentRuntime(_TextSession())
    model = _Model()
    agent = _SuccessfulAgent()

    episode = execute_agent_episode(agent, runtime, _task(), model)

    assert episode.stop_reason == StopReason.COMPLETED
    assert episode.final_action == AssistantAction(content="finished")
    assert agent.model is model
    assert agent.observation == Observation(content="text observation")
    assert agent.saw_reset is False
    assert agent.saw_close is False
    assert runtime.open_calls == 1
    assert runtime.close_calls == 1


def test_executable_environment_is_execute_only_to_the_customer_agent() -> None:
    runtime = _EnvironmentRuntime(_ExecutableSession())
    agent = _SuccessfulAgent()

    episode = execute_agent_episode(agent, runtime, _task(), _Model())

    assert episode.stop_reason == StopReason.COMPLETED
    assert agent.observation == Observation(content="executed: echo hello")
    assert runtime.close_calls == 1


def test_execute_only_capability_does_not_expose_the_raw_lifecycle_owner() -> None:
    runtime = _EnvironmentRuntime(_TextSession())
    agent = _AdversarialAgent()

    episode = execute_agent_episode(agent, runtime, _task(), _Model())

    assert episode.stop_reason == StopReason.COMPLETED
    assert agent.environment is not None
    for attribute in ("_environment", "__dict__", "_ExecuteOnlySession__environment"):
        with pytest.raises(AttributeError):
            getattr(agent.environment, attribute)
    capability = agent.environment.execute.__self__
    assert capability is agent.environment
    for lifecycle_method in ("reset", "close"):
        with pytest.raises(AttributeError):
            getattr(capability, lifecycle_method)
    assert runtime.close_calls == 1


def test_timeout_failure_is_retained_and_cleanup_runs_once() -> None:
    runtime = _EnvironmentRuntime(_TextSession())

    episode = execute_agent_episode(_TimeoutAgent(), runtime, _task(), _Model())

    assert episode.stop_reason == StopReason.FAILURE
    assert episode.failure is not None
    assert episode.failure.code == FailureCode.TIMEOUT
    assert episode.failure.attribution == FailureAttribution.AGENT
    assert _lifecycle_phases(episode) == ["agent timeout"]
    assert runtime.close_calls == 1


def test_execute_timeout_retains_timeout_classification() -> None:
    runtime = _EnvironmentRuntime(_TimeoutToolSession())

    episode = execute_agent_episode(_ToolFailureAgent(), runtime, _task(), _Model())

    assert episode.stop_reason == StopReason.FAILURE
    assert episode.failure is not None
    assert episode.failure.code == FailureCode.TIMEOUT
    assert episode.failure.attribution == FailureAttribution.TOOL
    assert episode.failure.exception_type == "TimeoutError"
    assert _lifecycle_phases(episode) == ["tool timeout"]
    assert runtime.close_calls == 1


def test_customer_reported_model_failure_keeps_its_attribution() -> None:
    runtime = _EnvironmentRuntime(_TextSession())
    agent = _ReportedModelFailureAgent()

    episode = execute_agent_episode(agent, runtime, _task(), _Model())

    assert episode.stop_reason == StopReason.FAILURE
    assert episode.failure is not None
    assert episode.failure.attribution == FailureAttribution.MODEL
    assert episode.failure.code == FailureCode.PROVIDER
    assert runtime.close_calls == 1


def test_execute_failure_is_attributed_to_the_tool_boundary() -> None:
    runtime = _EnvironmentRuntime(_FailingToolSession())

    episode = execute_agent_episode(_ToolFailureAgent(), runtime, _task(), _Model())

    assert episode.stop_reason == StopReason.FAILURE
    assert episode.failure is not None
    assert episode.failure.attribution == FailureAttribution.TOOL
    assert episode.failure.exception_type == "OSError"
    assert _lifecycle_phases(episode) == ["tool"]
    assert runtime.close_calls == 1


def test_agent_exception_is_attributed_without_losing_the_episode() -> None:
    runtime = _EnvironmentRuntime(_TextSession())

    episode = execute_agent_episode(_FailingAgent(), runtime, _task(), _Model())

    assert episode.stop_reason == StopReason.FAILURE
    assert episode.failure is not None
    assert episode.failure.attribution == FailureAttribution.AGENT
    assert episode.events[0].payload == {"phase": "agent"}
    assert episode.events[0].started_at == _LIFECYCLE_EVENT_EPOCH
    assert episode.events[0].ended_at == _LIFECYCLE_EVENT_EPOCH
    assert runtime.close_calls == 1


def test_invalid_adapter_result_is_retained_as_an_agent_failure() -> None:
    runtime = _EnvironmentRuntime(_TextSession())

    episode = execute_agent_episode(
        cast(AgentRuntime, _InvalidResultAgent()), runtime, _task(), _Model()
    )

    assert episode.stop_reason == StopReason.FAILURE
    assert episode.failure is not None
    assert episode.failure.attribution == FailureAttribution.AGENT
    assert episode.failure.exception_type == "TypeError"
    assert _lifecycle_phases(episode) == ["agent"]
    assert runtime.close_calls == 1


def test_reset_failure_becomes_a_structured_episode_failure() -> None:
    runtime = _EnvironmentRuntime(_TextSession(), fail_on_reset=True)

    episode = execute_agent_episode(_SuccessfulAgent(), runtime, _task(), _Model())

    assert episode.stop_reason == StopReason.FAILURE
    assert episode.failure is not None
    assert episode.failure.attribution == FailureAttribution.RESET
    assert _lifecycle_phases(episode) == ["reset"]
    assert runtime.close_calls == 1


def test_environment_open_failure_is_not_classified_as_reset() -> None:
    runtime = _EnvironmentRuntime(_TextSession(), fail_on_open=True)

    episode = execute_agent_episode(_SuccessfulAgent(), runtime, _task(), _Model())

    assert episode.stop_reason == StopReason.FAILURE
    assert episode.failure is not None
    assert episode.failure.code == FailureCode.INTERNAL
    assert episode.failure.attribution == FailureAttribution.ENVIRONMENT
    assert episode.failure.exception_type == "OSError"
    assert _lifecycle_phases(episode) == ["environment open"]
    assert runtime.close_calls == 0


def test_cleanup_failure_replaces_terminal_state_and_preserves_ordered_events() -> None:
    runtime = _EnvironmentRuntime(_TextSession(), fail_on_exit=True)

    episode = execute_agent_episode(_SuccessfulAgent(), runtime, _task(), _Model())

    assert episode.stop_reason == StopReason.FAILURE
    assert episode.failure is not None
    assert episode.failure.attribution == FailureAttribution.CLEANUP
    assert tuple(event.span_id for event in episode.events) == (
        "agent-1",
        episode.events[1].span_id,
    )
    assert episode.events[1].kind == RolloutEventKind.LIFECYCLE
    assert episode.events[1].started_at == episode.events[0].ended_at
    assert runtime.close_calls == 1


def test_cleanup_failure_retains_a_prior_structured_failure() -> None:
    runtime = _EnvironmentRuntime(_TextSession(), fail_on_exit=True)

    episode = execute_agent_episode(_ReportedModelFailureAgent(), runtime, _task(), _Model())

    assert episode.failure is not None
    assert episode.failure.attribution == FailureAttribution.CLEANUP
    assert episode.failure.details["prior_failure"] == {
        "code": FailureCode.PROVIDER.value,
        "message": "customer loop failed",
        "retryable": False,
        "exception_type": None,
        "attribution": FailureAttribution.MODEL.value,
        "details": {},
    }
    assert runtime.close_calls == 1


def test_preflight_names_the_required_customer_adapter_change() -> None:
    with pytest.raises(AgentAdapterPreflightError, match="model: ModelClient"):
        execute_agent_episode(
            cast(AgentRuntime, _NoModelInjectionAgent()),
            _EnvironmentRuntime(_TextSession()),
            _task(),
            _Model(),
        )


class _Model:
    """A canonical model client that is never called by these lifecycle fixtures."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Reject unexpected completion calls outside an agent-owned loop."""
        raise AssertionError(f"unexpected model request: {request!r}")


class _SuccessfulAgent:
    """Returns a complete result after one execute-only environment call."""

    def __init__(self) -> None:
        self.model: ModelClient | None = None
        self.observation: Observation | None = None
        self.saw_reset = False
        self.saw_close = False

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        self.model = model
        self.saw_reset = hasattr(environment, "reset")
        self.saw_close = hasattr(environment, "close")
        self.observation = environment.execute(ToolCall(call_id="call-1", name="run", arguments={}))
        return AgentEpisode(
            events=(_span("agent-1"),),
            final_action=AssistantAction(content="finished"),
            stop_reason=StopReason.COMPLETED,
        )


class _AdversarialAgent:
    """Retains the supplied capability and probes for hidden lifecycle ownership."""

    def __init__(self) -> None:
        self.environment: EnvironmentSession | None = None

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        self.environment = environment
        environment.execute(ToolCall(call_id="call-1", name="run", arguments={}))
        return AgentEpisode(stop_reason=StopReason.COMPLETED)


class _TimeoutAgent:
    """Raises the timeout W4 must preserve as an episode result."""

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        raise TimeoutError("agent turn exceeded its deadline")


class _ReportedModelFailureAgent:
    """Returns a model failure discovered inside its own loop."""

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        failure = StructuredFailure(
            code=FailureCode.PROVIDER,
            message="customer loop failed",
            attribution=FailureAttribution.MODEL,
        )
        return AgentEpisode(
            events=(_span("agent-failure"),),
            stop_reason=StopReason.FAILURE,
            failure=failure,
        )


class _FailingAgent:
    """Raises an unclassified customer-agent exception."""

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        raise RuntimeError("customer code failed")


class _InvalidResultAgent:
    """Violates the whole-episode return contract for a lifecycle regression fixture."""

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> str:
        return "not an episode"


class _ToolFailureAgent:
    """Lets an execute failure escape so W4 can attribute the tool boundary."""

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        environment.execute(ToolCall(call_id="call-1", name="run", arguments={}))
        raise AssertionError("the tool fixture should fail before this line")


class _NoModelInjectionAgent:
    """Omits model injection to exercise the actionable preflight failure."""

    def run(self, task: TaskCase, *, environment: EnvironmentSession) -> AgentEpisode:
        return AgentEpisode(stop_reason=StopReason.COMPLETED)


class _TextSession:
    """A fake text environment that exposes an intentionally hidden close method."""

    def execute(self, action: ToolCall) -> Observation:
        return Observation(content="text observation")

    def reset(self) -> None:
        """Represent a reset capability that must not reach the agent."""

    def close(self) -> None:
        """Represent a close capability that must not reach the agent."""


class _ExecutableSession:
    """A fake executable environment with the same execute-only public surface."""

    def execute(self, action: ToolCall) -> Observation:
        return Observation(content="executed: echo hello")


class _FailingToolSession:
    """A fake executable session whose requested tool fails deterministically."""

    def execute(self, action: ToolCall) -> Observation:
        raise OSError("tool process exited unexpectedly")


class _TimeoutToolSession:
    """A fake executable session whose requested tool reaches its own deadline."""

    def execute(self, action: ToolCall) -> Observation:
        raise TimeoutError("tool process exceeded its deadline")


class _EnvironmentRuntime:
    """Creates one deterministic session and records simulator-owned cleanup."""

    def __init__(
        self,
        session: _TextSession | _ExecutableSession | _FailingToolSession | _TimeoutToolSession,
        *,
        fail_on_open: bool = False,
        fail_on_reset: bool = False,
        fail_on_exit: bool = False,
    ) -> None:
        self._session = session
        self._fail_on_open = fail_on_open
        self._fail_on_reset = fail_on_reset
        self._fail_on_exit = fail_on_exit
        self.open_calls = 0
        self.close_calls = 0

    def open(self, task: TaskCase) -> _EnvironmentContext:
        self.open_calls += 1
        if self._fail_on_open:
            raise OSError("environment transport allocation failed")
        return _EnvironmentContext(self)


class _EnvironmentContext:
    """Context manager that makes cleanup ownership observable in a fixture."""

    def __init__(self, runtime: _EnvironmentRuntime) -> None:
        self._runtime = runtime

    def __enter__(
        self,
    ) -> _TextSession | _ExecutableSession | _FailingToolSession | _TimeoutToolSession:
        if self._runtime._fail_on_reset:
            self._runtime.close_calls += 1
            raise EnvironmentResetError("environment reset failed")
        return self._runtime._session

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._runtime.close_calls += 1
        if self._runtime._fail_on_exit:
            raise RuntimeError("environment cleanup failed")
        return False


def _task() -> TaskCase:
    return TaskCase(
        task_id="task-1",
        lineage_group_id="lineage-1",
        partition="held_out",
        instruction="Complete the deterministic fixture.",
        workload_weight=1.0,
        source_trace_ids=("trace-1",),
    )


def _span(span_id: str) -> RolloutSpan:
    return RolloutSpan(
        span_id=span_id,
        kind=RolloutEventKind.MESSAGE,
        started_at=_EVENT_TIME,
        ended_at=_EVENT_TIME,
        payload={"fixture": span_id},
    )


def _lifecycle_phases(episode: AgentEpisode) -> list[str]:
    return [str(event.payload["phase"]) for event in episode.events]
