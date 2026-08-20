"""Simulator-side lifecycle composition for one model-injected customer episode."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from uuid import uuid4

from exp.common.core.artifacts import (
    FailureAttribution,
    FailureCode,
    JsonObject,
    StructuredFailure,
)
from exp.common.models import ModelClient, ToolCall
from exp.common.rollouts import RolloutEventKind, RolloutSpan, StopReason
from exp.common.tasks import TaskCase
from exp.runtime.agents.interface import AgentEpisode, AgentRuntime, preflight_agent_runtime
from exp.runtime.environments import (
    EnvironmentResetError,
    EnvironmentRuntime,
    EnvironmentSession,
    Observation,
)

_ExecuteAction = Callable[[ToolCall], Observation]
_execute_action: ContextVar[_ExecuteAction | None] = ContextVar("exp_execute_action", default=None)
_LIFECYCLE_EVENT_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def execute_agent_episode(
    agent: AgentRuntime,
    environment_runtime: EnvironmentRuntime,
    task: TaskCase,
    model: ModelClient,
) -> AgentEpisode:
    """Run one agent while retaining reset, cleanup, and exception evidence.

    This is the simulator-side composition point. It opens the environment once, gives the agent
    only the execute-capable session, and always returns an episode result, including cleanup
    failures that occur after the customer agent has returned.

    Args:
        agent: Customer-owned whole-episode runtime.
        environment_runtime: Simulator-owned source of per-episode environment sessions.
        task: Task whose already-open session the agent may execute against.
        model: Candidate model injected into the customer agent.

    Returns:
        A complete in-memory episode with ordered events and structured failure attribution.
    """
    preflight_agent_runtime(agent)
    try:
        environment_context = environment_runtime.open(task)
    except EnvironmentResetError as exc:
        return _failed_episode("reset", FailureAttribution.RESET, exc)
    except Exception as exc:  # noqa: BLE001 - environment allocation is episode evidence
        return _failed_episode("environment open", FailureAttribution.ENVIRONMENT, exc)

    entered = False
    episode: AgentEpisode | None = None
    try:
        with environment_context as environment:
            entered = True
            with _execute_only_session(environment) as execute_session:
                episode = _run_agent(agent, task, model, execute_session)
    except EnvironmentResetError as exc:
        if not entered:
            return _failed_episode("reset", FailureAttribution.RESET, exc)
        if episode is None:
            return _failed_episode("agent", FailureAttribution.AGENT, exc)
        return _append_cleanup_failure(episode, exc)
    except Exception as exc:  # noqa: BLE001 - every lifecycle failure becomes episode evidence
        if not entered:
            return _failed_episode("environment open", FailureAttribution.ENVIRONMENT, exc)
        if episode is None:
            return _failed_episode("agent", FailureAttribution.AGENT, exc)
        return _append_cleanup_failure(episode, exc)
    if episode is None:
        return _failed_episode("agent", FailureAttribution.AGENT, RuntimeError("no episode"))
    return episode


def _run_agent(
    agent: AgentRuntime,
    task: TaskCase,
    model: ModelClient,
    environment: EnvironmentSession,
) -> AgentEpisode:
    """Call the customer adapter while preserving its exception as a typed episode failure."""
    try:
        episode = agent.run(task, model=model, environment=environment)
        if not isinstance(episode, AgentEpisode):
            raise TypeError(
                f"customer agent adapters must return AgentEpisode, not {type(episode).__name__}"
            )
        return episode
    except _ToolExecutionError as exc:
        timeout = isinstance(exc.cause, TimeoutError)
        return _failed_episode(
            "tool timeout" if timeout else "tool",
            FailureAttribution.TOOL,
            exc.cause,
            FailureCode.TIMEOUT if timeout else FailureCode.INTERNAL,
        )
    except TimeoutError as exc:
        return _failed_episode("agent timeout", FailureAttribution.AGENT, exc, FailureCode.TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - customer adapter failures are episode evidence
        return _failed_episode("agent", FailureAttribution.AGENT, exc)


def _failed_episode(
    phase: str,
    attribution: FailureAttribution,
    exception: Exception,
    code: FailureCode = FailureCode.INTERNAL,
) -> AgentEpisode:
    """Build an event-preserving terminal result for one boundary failure."""
    failure = StructuredFailure(
        code=code,
        message=f"{phase} failed with {type(exception).__name__}",
        exception_type=type(exception).__name__,
        attribution=attribution,
        details={"phase": phase},
    )
    return AgentEpisode(
        events=(_lifecycle_span(phase, failure),),
        stop_reason=StopReason.FAILURE,
        failure=failure,
    )


def _append_cleanup_failure(episode: AgentEpisode, exception: Exception) -> AgentEpisode:
    """Replace terminal state with cleanup failure while retaining every prior event and output."""
    details: JsonObject = {
        "phase": "cleanup",
        "prior_stop_reason": episode.stop_reason.value,
    }
    if episode.failure is not None:
        details["prior_failure"] = _failure_details(episode.failure)
    failure = StructuredFailure(
        code=FailureCode.INTERNAL,
        message=f"cleanup failed with {type(exception).__name__}",
        exception_type=type(exception).__name__,
        attribution=FailureAttribution.CLEANUP,
        details=details,
    )
    return AgentEpisode(
        events=(*episode.events, _lifecycle_span("cleanup", failure, episode.events)),
        final_action=episode.final_action,
        stop_reason=StopReason.FAILURE,
        usage=episode.usage,
        failure=failure,
    )


def _lifecycle_span(
    phase: str,
    failure: StructuredFailure,
    prior_events: tuple[RolloutSpan, ...] = (),
) -> RolloutSpan:
    """Create one ordered lifecycle span without inventing a wall-clock event time.

    Failure spans follow the most recent event when one exists. A failure before any event uses a
    fixed epoch, which records ordering without claiming a source timestamp that was not observed.
    """
    timestamp = prior_events[-1].ended_at if prior_events else _LIFECYCLE_EVENT_EPOCH
    return RolloutSpan(
        span_id=f"lifecycle-{uuid4().hex}",
        kind=RolloutEventKind.LIFECYCLE,
        started_at=timestamp,
        ended_at=timestamp,
        payload={"phase": phase},
        failure=failure,
    )


@contextmanager
def _execute_only_session(environment: EnvironmentSession) -> Iterator[EnvironmentSession]:
    """Scope one raw session to a capability that exposes only execute.

    The capability itself retains no raw-session attribute. The lifecycle owns the context-local
    executor for the duration of the customer call, so reset and close are not attributes of the
    object passed to the agent. This API restriction does not itself sandbox customer code.
    """
    token = _execute_action.set(environment.execute)
    try:
        yield _ExecuteOnlySession()
    finally:
        _execute_action.reset(token)


class _ExecuteOnlySession:
    """A capability object that exposes only the current episode's execute operation."""

    __slots__ = ()

    def execute(self, action: ToolCall) -> Observation:
        """Execute one action through the lifecycle-owned session capability."""
        executor = _execute_action.get()
        if executor is None:
            raise RuntimeError(
                "the execute-only environment session is unavailable outside its episode"
            )
        try:
            return executor(action)
        except Exception as exc:  # noqa: BLE001 - tool boundary failures become episode evidence
            raise _ToolExecutionError(exc) from exc


class _ToolExecutionError(RuntimeError):
    """Preserve the original environment exception across the execute-only proxy."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


def _failure_details(failure: StructuredFailure) -> JsonObject:
    """Return the complete previously structured failure as JSON-safe cleanup evidence."""
    return {
        "code": failure.code.value,
        "message": failure.message,
        "retryable": failure.retryable,
        "exception_type": failure.exception_type,
        "attribution": failure.attribution.value if failure.attribution is not None else None,
        "details": failure.details,
    }
