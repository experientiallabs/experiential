"""Simulator-side lifecycle composition for one model-injected customer episode."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from wmo.common.core.artifacts import (
    FailureAttribution,
    FailureCode,
    JsonObject,
    StructuredFailure,
)
from wmo.common.models import ToolCall
from wmo.common.rollouts import RolloutEventKind, RolloutSpan, StopReason
from wmo.common.tasks import TaskCase
from wmo.runtime.agents.interface import AgentEpisode, AgentRuntime, preflight_agent_runtime
from wmo.runtime.environments import EnvironmentRuntime, EnvironmentSession, Observation


def execute_agent_episode(
    agent: AgentRuntime,
    environment_runtime: EnvironmentRuntime,
    task: TaskCase,
    model: object,
) -> AgentEpisode:
    """Run one agent while retaining reset, cleanup, and exception evidence.

    This is the simulator-side composition point. It opens the environment once, gives the agent
    only the execute-capable session, and always returns an episode result, including cleanup
    failures that occur after the customer agent has returned.

    Args:
        agent: Customer-owned whole-episode runtime.
        environment_runtime: Simulator-owned source of isolated environment sessions.
        task: Task whose already-open session the agent may execute against.
        model: Candidate model injected into the customer agent.

    Returns:
        A complete in-memory episode with ordered events and structured failure attribution.
    """
    preflight_agent_runtime(agent)
    opened = False
    episode: AgentEpisode | None = None
    try:
        with environment_runtime.open(task) as environment:
            opened = True
            episode = _run_agent(agent, task, model, _ExecuteOnlySession(environment))
    except Exception as exc:  # noqa: BLE001 - every lifecycle failure becomes episode evidence
        if not opened:
            return _failed_episode("reset", FailureAttribution.RESET, exc)
        if episode is None:
            return _failed_episode("agent", FailureAttribution.AGENT, exc)
        return _append_cleanup_failure(episode, exc)
    if episode is None:
        return _failed_episode("agent", FailureAttribution.AGENT, RuntimeError("no episode"))
    return episode


def _run_agent(
    agent: AgentRuntime,
    task: TaskCase,
    model: object,
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
        return _failed_episode("tool", FailureAttribution.TOOL, exc.cause)
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
    """Create one ordered lifecycle span for a failure that prevented normal agent progress."""
    now = datetime.now(UTC)
    if prior_events:
        now = max(now, prior_events[-1].ended_at)
    return RolloutSpan(
        span_id=f"lifecycle-{uuid4().hex}",
        kind=RolloutEventKind.LIFECYCLE,
        started_at=now,
        ended_at=now,
        payload={"phase": phase},
        failure=failure,
    )


class _ExecuteOnlySession:
    """Hide simulator lifecycle methods from the customer agent."""

    __slots__ = ("_environment",)

    def __init__(self, environment: EnvironmentSession) -> None:
        self._environment = environment

    def execute(self, action: ToolCall) -> Observation:
        """Forward one action to the simulator-owned session."""
        try:
            return self._environment.execute(action)
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
