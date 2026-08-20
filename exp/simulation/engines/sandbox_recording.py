"""Bounded model and environment recording for executable sandbox episodes."""

from __future__ import annotations

import os
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from types import TracebackType

from exp.common.core.artifacts import (
    FailureAttribution,
    FailureCode,
    JsonObject,
    StructuredFailure,
)
from exp.common.models import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    ToolCall,
    combine_economics,
)
from exp.common.rollouts import RolloutEventKind, RolloutSpan, StopReason
from exp.common.tasks import TaskCase
from exp.runtime.agents import AgentEpisode, AgentRuntime
from exp.runtime.agents.lifecycle import execute_agent_episode
from exp.runtime.environments import EnvironmentRuntime, EnvironmentSession, Observation
from exp.runtime.environments.harbor import HarborCleanupTimeoutError
from exp.simulation.engines.clock import timestamp, utc_now


class SandboxStepLimitError(RuntimeError):
    """The customer agent attempted another model turn after the exact step limit."""


class SandboxTimeLimitError(TimeoutError):
    """The complete executable episode crossed its configured wall-clock limit."""


class SandboxCostLimitError(RuntimeError):
    """A candidate request could not be admitted under the remaining finite spend ceiling."""


class SandboxUnknownCostError(RuntimeError):
    """A dispatched candidate request returned without the required finite cost evidence."""


@dataclass(frozen=True)
class SandboxExecutionEvidence:
    """Recorded candidate calls, tool transcript, economics, and enforced terminal limit."""

    candidate_spans: tuple[RolloutSpan, ...]
    environment_spans: tuple[RolloutSpan, ...]
    candidate_economics: OperationEconomics
    sandbox_economics: OperationEconomics
    limit_stop_reason: StopReason | None
    limit_failure: StructuredFailure | None


def _enforce_finite_cost_evidence(
    evidence: SandboxExecutionEvidence,
    *,
    remaining_cost_usd: float | None,
    environment_maximum_episode_cost_usd: float | None,
) -> SandboxExecutionEvidence:
    """Fail closed when any dispatched candidate or environment spend is unknown or unsafe."""
    if remaining_cost_usd is None or evidence.limit_stop_reason is not None:
        return evidence
    if environment_maximum_episode_cost_usd is None:  # pragma: no cover - preflight requires it
        raise ValueError("finite sandbox cost enforcement requires an environment reservation")
    made_candidate_call = bool(evidence.candidate_spans)
    candidate_cost = evidence.candidate_economics.cost_usd
    if made_candidate_call and candidate_cost is None:
        failure = _limit_failure(
            FailureCode.BUDGET,
            "candidate dispatch has unknown spend under a finite sandbox budget",
            SandboxUnknownCostError,
            "candidate_cost_unknown",
            provider_dispatch_unknown_spend=True,
        )
        return replace(
            evidence,
            limit_stop_reason=StopReason.MAXIMUM_COST,
            limit_failure=failure,
        )
    sandbox_cost = evidence.sandbox_economics.cost_usd
    if environment_maximum_episode_cost_usd == 0:
        environment_spend = 0.0
        if sandbox_cost is not None and sandbox_cost.value != 0:
            failure = _environment_cost_failure(
                "sandbox environment reported cost inconsistent with its zero-cost proof",
                "environment_cost_bound",
                unknown_spend=sandbox_cost.value < 0,
            )
            return replace(
                evidence,
                limit_stop_reason=StopReason.MAXIMUM_COST,
                limit_failure=failure,
            )
    elif sandbox_cost is None:
        failure = _environment_cost_failure(
            "sandbox environment omitted cost required by a finite run budget",
            "environment_cost_unknown",
            unknown_spend=True,
        )
        return replace(
            evidence,
            limit_stop_reason=StopReason.MAXIMUM_COST,
            limit_failure=failure,
        )
    else:
        environment_spend = sandbox_cost.value
        if environment_spend < 0 or environment_spend > environment_maximum_episode_cost_usd:
            failure = _environment_cost_failure(
                "sandbox environment cost exceeded its proven episode reservation",
                "environment_cost_bound",
                unknown_spend=environment_spend < 0,
            )
            return replace(
                evidence,
                limit_stop_reason=StopReason.MAXIMUM_COST,
                limit_failure=failure,
            )
    total = (candidate_cost.value if candidate_cost is not None else 0.0) + environment_spend
    if total > remaining_cost_usd:
        failure = _environment_cost_failure(
            "candidate and sandbox cost exceeded the remaining run budget",
            "maximum_cost",
        )
        return replace(
            evidence,
            limit_stop_reason=StopReason.MAXIMUM_COST,
            limit_failure=failure,
        )
    return evidence


def execute_bounded_sandbox_episode(
    *,
    agent_factory: Callable[[], AgentRuntime],
    task: TaskCase,
    candidate: ModelClient,
    candidate_snapshot: ModelSnapshot,
    environment_runtime: EnvironmentRuntime,
    environment_maximum_episode_cost_usd: float | None,
    environment_cost_is_observable: bool,
    maximum_steps: int,
    maximum_time_seconds: float,
    remaining_cost_usd: float | None,
    maximum_call_cost_usd: float | None,
    cost_is_observable: bool,
    record_dispatch_intent: Callable[[], None],
    clock: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[AgentEpisode, SandboxExecutionEvidence]:
    """Run one customer episode with hard turn, wall-time, and finite-cost boundaries.

    Args:
        agent_factory: Creates the fresh customer agent inside the hard episode deadline.
        task: Canonical executable task.
        candidate: Resolved candidate model client.
        candidate_snapshot: Exact candidate identity pinned by the evaluation plan.
        environment_runtime: Simulator-owned executable environment factory.
        environment_maximum_episode_cost_usd: Proven whole-episode environment reservation.
        environment_cost_is_observable: Whether nonzero environment spend reaches observations.
        maximum_steps: Maximum candidate model calls admitted for the episode.
        maximum_time_seconds: Hard wall-clock bound for the whole lifecycle and cleanup path.
        remaining_cost_usd: Remaining run budget, or ``None`` for an unbounded run.
        maximum_call_cost_usd: Proven upper bound reserved before each candidate dispatch.
        cost_is_observable: Whether responses must report exact or estimated finite cost.
        record_dispatch_intent: Fsync-backed callback invoked immediately before each external
            candidate or environment dispatch may begin.
        clock: Aware event clock.
        monotonic: Monotonic deadline and latency clock.

    Returns:
        The canonical agent episode plus independently recorded execution evidence.

    Raises:
        ValueError: A finite budget lacks a safe call reservation or observable cost capability.
        SandboxTimeLimitError: The platform cannot install a hard wall timer in this thread.
    """
    if remaining_cost_usd is not None:
        if maximum_call_cost_usd is None or not cost_is_observable:
            raise ValueError(
                "finite sandbox budgets require observable candidate cost and a maximum call cost"
            )
        if environment_maximum_episode_cost_usd is None or not environment_cost_is_observable:
            raise ValueError(
                "finite sandbox budgets require observable environment cost and a maximum "
                "episode cost"
            )
        if environment_maximum_episode_cost_usd > remaining_cost_usd:
            raise SandboxCostLimitError(
                "environment reservation exceeds the remaining sandbox budget"
            )
    candidate_remaining_cost_usd = (
        None
        if remaining_cost_usd is None
        else remaining_cost_usd - (environment_maximum_episode_cost_usd or 0.0)
    )
    event_clock = clock or utc_now
    deadline = _Deadline(maximum_time_seconds, monotonic)
    model = _RecordingCandidateClient(
        candidate,
        candidate_snapshot,
        maximum_steps=maximum_steps,
        deadline=deadline,
        remaining_cost_usd=candidate_remaining_cost_usd,
        maximum_call_cost_usd=maximum_call_cost_usd,
        clock=event_clock,
        monotonic=monotonic,
        record_dispatch_intent=record_dispatch_intent,
    )
    environment = _RecordingEnvironmentRuntime(
        environment_runtime,
        deadline=deadline,
        clock=event_clock,
        monotonic=monotonic,
        record_dispatch_intent=record_dispatch_intent,
    )
    with _hard_wall_timeout(deadline):
        agent = agent_factory()
        episode = execute_agent_episode(agent, environment, task, model)
    evidence = SandboxExecutionEvidence(
        candidate_spans=tuple(model.spans),
        environment_spans=tuple(environment.spans),
        candidate_economics=combine_economics(model.economics),
        sandbox_economics=combine_economics(environment.economics),
        limit_stop_reason=model.limit_stop_reason or environment.limit_stop_reason,
        limit_failure=model.limit_failure or environment.limit_failure,
    )
    return episode, _enforce_finite_cost_evidence(
        evidence,
        remaining_cost_usd=remaining_cost_usd,
        environment_maximum_episode_cost_usd=environment_maximum_episode_cost_usd,
    )


class _Deadline:
    """One monotonic per-episode deadline shared by every runtime boundary."""

    def __init__(self, maximum_time_seconds: float, monotonic: Callable[[], float]) -> None:
        self._monotonic = monotonic
        self._deadline = monotonic() + maximum_time_seconds

    def remaining(self) -> float:
        """Return positive remaining time or raise the terminal time-limit error."""
        remaining = self._deadline - self._monotonic()
        if remaining <= 0:
            raise SandboxTimeLimitError("sandbox episode exceeded maximum_time_seconds")
        return remaining


class _RecordingCandidateClient:
    """Record candidate evidence and stop before unsafe turn or finite-cost dispatch."""

    def __init__(
        self,
        client: ModelClient,
        snapshot: ModelSnapshot,
        *,
        maximum_steps: int,
        deadline: _Deadline,
        remaining_cost_usd: float | None,
        maximum_call_cost_usd: float | None,
        clock: Callable[[], datetime],
        monotonic: Callable[[], float],
        record_dispatch_intent: Callable[[], None],
    ) -> None:
        self._client = client
        self._snapshot = snapshot
        self._maximum_steps = maximum_steps
        self._deadline = deadline
        self._remaining_cost_usd = remaining_cost_usd
        self._maximum_call_cost_usd = maximum_call_cost_usd
        self._clock = clock
        self._monotonic = monotonic
        self._record_dispatch_intent = record_dispatch_intent
        self._calls = 0
        self._spent = 0.0
        self.spans: list[RolloutSpan] = []
        self.economics: list[OperationEconomics] = []
        self.limit_stop_reason: StopReason | None = None
        self.limit_failure: StructuredFailure | None = None

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Dispatch one admitted request and retain its exact response economics."""
        self._deadline.remaining()
        if self._calls >= self._maximum_steps:
            failure = _limit_failure(
                FailureCode.BUDGET,
                "sandbox candidate reached maximum_steps",
                SandboxStepLimitError,
                "maximum_steps",
            )
            self._set_limit(StopReason.MAXIMUM_STEPS, failure)
            raise SandboxStepLimitError(failure.message)
        if self._remaining_cost_usd is not None:
            reservation = self._maximum_call_cost_usd
            if reservation is None or self._spent + reservation > self._remaining_cost_usd:
                failure = _limit_failure(
                    FailureCode.BUDGET,
                    "sandbox candidate request would exceed maximum_cost_usd",
                    SandboxCostLimitError,
                    "maximum_cost",
                )
                self._set_limit(StopReason.MAXIMUM_COST, failure)
                raise SandboxCostLimitError(failure.message)
        call_index = self._calls
        self._calls += 1
        started_at = timestamp(self._clock)
        started = self._monotonic()
        self._record_dispatch_intent()
        try:
            response = self._client.complete(request)
        except Exception as exc:
            ended_at = timestamp(self._clock, not_before=started_at)
            failure = StructuredFailure(
                code=FailureCode.TIMEOUT if isinstance(exc, TimeoutError) else FailureCode.PROVIDER,
                message=f"candidate provider failed with {type(exc).__name__}",
                exception_type=type(exc).__name__,
                attribution=FailureAttribution.MODEL,
                details={
                    "phase": "candidate_dispatch",
                    "provider_dispatch_unknown_spend": True,
                },
            )
            self.spans.append(
                _model_span(call_index, started_at, ended_at, self._snapshot, None, failure)
            )
            if isinstance(exc, SandboxTimeLimitError):
                self._set_limit(StopReason.MAXIMUM_TIME, failure)
            raise
        ended_at = timestamp(self._clock, not_before=started_at)
        if response.model != self._snapshot:
            failure = StructuredFailure(
                code=FailureCode.PROVIDER,
                message="candidate response model identity differs from the evaluation plan",
                attribution=FailureAttribution.MODEL,
                details={"phase": "candidate_identity", "provider_dispatch_unknown_spend": True},
            )
            self.spans.append(
                _model_span(call_index, started_at, ended_at, response.model, response, failure)
            )
            raise SandboxUnknownCostError(failure.message)
        economics = _with_observed_latency(
            response.economics,
            max(0.0, self._monotonic() - started),
        )
        self.economics.append(economics)
        self.spans.append(
            _model_span(call_index, started_at, ended_at, response.model, response, None)
        )
        if self._remaining_cost_usd is not None:
            cost = economics.cost_usd
            if cost is None:
                failure = _limit_failure(
                    FailureCode.BUDGET,
                    "candidate response omitted cost required by a finite sandbox budget",
                    SandboxUnknownCostError,
                    "candidate_cost_unknown",
                    provider_dispatch_unknown_spend=True,
                )
                self._set_limit(StopReason.MAXIMUM_COST, failure)
                raise SandboxUnknownCostError(failure.message)
            if cost.value < 0 or (
                self._maximum_call_cost_usd is not None and cost.value > self._maximum_call_cost_usd
            ):
                failure = _limit_failure(
                    FailureCode.BUDGET,
                    "candidate cost violated its finite call reservation",
                    SandboxUnknownCostError,
                    "candidate_cost_bound",
                    provider_dispatch_unknown_spend=True,
                )
                self._set_limit(StopReason.MAXIMUM_COST, failure)
                raise SandboxUnknownCostError(failure.message)
            self._spent += cost.value
            if self._spent > self._remaining_cost_usd:
                failure = _limit_failure(
                    FailureCode.BUDGET,
                    "candidate cost exceeded its reserved sandbox budget",
                    SandboxCostLimitError,
                    "maximum_cost",
                )
                self._set_limit(StopReason.MAXIMUM_COST, failure)
                raise SandboxCostLimitError(failure.message)
        self._deadline.remaining()
        return response.model_copy(update={"economics": economics})

    def _set_limit(self, reason: StopReason, failure: StructuredFailure) -> None:
        """Retain the first terminal limit without overwriting earlier evidence."""
        if self.limit_stop_reason is None:
            self.limit_stop_reason = reason
            self.limit_failure = failure


class _RecordingEnvironmentRuntime:
    """Wrap one environment context and retain every action, observation, and failure."""

    def __init__(
        self,
        runtime: EnvironmentRuntime,
        *,
        deadline: _Deadline,
        clock: Callable[[], datetime],
        monotonic: Callable[[], float],
        record_dispatch_intent: Callable[[], None],
    ) -> None:
        self._runtime = runtime
        self._deadline = deadline
        self._clock = clock
        self._monotonic = monotonic
        self._record_dispatch_intent = record_dispatch_intent
        self.spans: list[RolloutSpan] = []
        self.economics: list[OperationEconomics] = []
        self.limit_stop_reason: StopReason | None = None
        self.limit_failure: StructuredFailure | None = None

    def open(self, task: TaskCase) -> AbstractContextManager[EnvironmentSession]:
        """Open the underlying session through one evidence-recording context."""
        self._deadline.remaining()
        self._record_dispatch_intent()
        return _RecordingEnvironmentContext(self, self._runtime.open(task))


class _RecordingEnvironmentContext(AbstractContextManager[EnvironmentSession]):
    """Preserve cleanup authority in the original runtime context."""

    def __init__(
        self,
        owner: _RecordingEnvironmentRuntime,
        context: AbstractContextManager[EnvironmentSession],
    ) -> None:
        self._owner = owner
        self._context = context

    def __enter__(self) -> EnvironmentSession:
        """Enter the backing context and expose only a recording execute capability."""
        session = self._context.__enter__()
        try:
            self._owner._deadline.remaining()
        except BaseException as exc:
            self._context.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return _RecordingEnvironmentSession(self._owner, session)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Keep cleanup inside the hard wall bound and retain an honest terminal failure."""
        try:
            suppressed = self._context.__exit__(exception_type, exception, traceback)
            self._owner._deadline.remaining()
        except (HarborCleanupTimeoutError, SandboxTimeLimitError) as exc:
            self._owner.limit_stop_reason = StopReason.MAXIMUM_TIME
            self._owner.limit_failure = StructuredFailure(
                code=FailureCode.TIMEOUT,
                message="sandbox environment cleanup exceeded maximum_time_seconds",
                exception_type=type(exc).__name__,
                attribution=FailureAttribution.CLEANUP,
                details={"phase": "cleanup_timeout"},
            )
            return False
        except Exception as exc:  # noqa: BLE001 - cleanup failures become durable evidence
            self._owner.limit_stop_reason = StopReason.FAILURE
            self._owner.limit_failure = StructuredFailure(
                code=FailureCode.INTERNAL,
                message=f"sandbox environment cleanup failed with {type(exc).__name__}",
                exception_type=type(exc).__name__,
                attribution=FailureAttribution.CLEANUP,
                details={"phase": "cleanup_failure"},
            )
            return False
        return bool(suppressed)


class _RecordingEnvironmentSession:
    """Record one canonical tool transcript without exposing lifecycle methods."""

    def __init__(self, owner: _RecordingEnvironmentRuntime, session: EnvironmentSession) -> None:
        self._owner = owner
        self._session = session
        self._index = 0

    def execute(self, action: ToolCall) -> Observation:
        """Execute one action, retaining partial evidence even when the call raises."""
        self._owner._deadline.remaining()
        index = self._index
        self._index += 1
        started_at = timestamp(self._owner._clock)
        started = self._owner._monotonic()
        self._owner._record_dispatch_intent()
        try:
            observation = self._session.execute(action)
        except Exception as exc:
            ended_at = timestamp(self._owner._clock, not_before=started_at)
            failure = StructuredFailure(
                code=FailureCode.TIMEOUT if isinstance(exc, TimeoutError) else FailureCode.INTERNAL,
                message=f"environment action failed with {type(exc).__name__}",
                exception_type=type(exc).__name__,
                attribution=FailureAttribution.TOOL,
                details={"phase": "environment_execute", "tool": action.name},
            )
            self._owner.spans.extend(
                _tool_spans(index, action, None, started_at, ended_at, failure)
            )
            if isinstance(exc, SandboxTimeLimitError):
                self._owner.limit_stop_reason = StopReason.MAXIMUM_TIME
                self._owner.limit_failure = failure
            raise
        ended_at = timestamp(self._owner._clock, not_before=started_at)
        duration = max(0.0, self._owner._monotonic() - started)
        economics = _observation_economics(observation, duration)
        self._owner.economics.append(economics)
        failure = _observation_failure(action, observation)
        self._owner.spans.extend(
            _tool_spans(index, action, observation, started_at, ended_at, failure)
        )
        self._owner._deadline.remaining()
        return observation


def require_hard_wall_timeout_support() -> tuple[float, float]:
    """Verify that this thread can install the timer required for sandbox execution.

    Returns:
        The inactive prior timer settings that the caller must restore after execution.

    Raises:
        SandboxTimeLimitError: Hard wall-time enforcement is unavailable or already claimed.
    """
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        raise SandboxTimeLimitError(
            "hard sandbox wall-time enforcement requires the POSIX main thread"
        )
    if not hasattr(signal, "setitimer"):
        raise SandboxTimeLimitError("hard sandbox wall-time enforcement is unavailable")
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    if previous_delay > 0 or previous_interval > 0:
        raise SandboxTimeLimitError("another wall timer is already active")
    return previous_delay, previous_interval


@contextmanager
def _hard_wall_timeout(deadline: _Deadline) -> Iterator[None]:
    """Install a POSIX main-thread timer so a silent hung agent cannot evade its limit."""
    previous_delay, previous_interval = require_hard_wall_timeout_support()
    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signum: int, _frame: object) -> None:
        raise SandboxTimeLimitError("sandbox episode exceeded maximum_time_seconds")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, deadline.remaining())
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def merge_sandbox_spans(
    episode: AgentEpisode,
    evidence: SandboxExecutionEvidence,
) -> tuple[RolloutSpan, ...]:
    """Merge recorder spans with nonduplicated customer lifecycle evidence.

    Args:
        episode: Canonical agent episode containing recorder-generated events.
        evidence: Candidate and environment spans captured during sandbox execution.

    Returns:
        All retained spans in deterministic timestamp and ID order.
    """
    retained = tuple(
        span
        for span in episode.events
        if span.kind
        not in {
            RolloutEventKind.AGENT_MODEL_CALL,
            RolloutEventKind.TOOL_CALL,
            RolloutEventKind.OBSERVATION,
        }
    )
    combined = (*retained, *evidence.candidate_spans, *evidence.environment_spans)
    return tuple(sorted(combined, key=lambda span: (span.started_at, span.ended_at, span.span_id)))


def _model_span(
    index: int,
    started_at: datetime,
    ended_at: datetime,
    model: ModelSnapshot,
    response: ModelResponse | None,
    failure: StructuredFailure | None,
) -> RolloutSpan:
    """Build one provider-call span without persisting hidden model reasoning."""
    return RolloutSpan(
        span_id=f"sandbox-model-{index}",
        kind=RolloutEventKind.AGENT_MODEL_CALL,
        started_at=started_at,
        ended_at=ended_at,
        payload={"finish_reason": response.finish_reason.value if response is not None else None},
        model=model,
        usage=response.economics.usage if response is not None else None,
        failure=failure,
    )


def _tool_spans(
    index: int,
    action: ToolCall,
    observation: Observation | None,
    started_at: datetime,
    ended_at: datetime,
    failure: StructuredFailure | None,
) -> tuple[RolloutSpan, RolloutSpan]:
    """Build paired action and observation spans, including partial failure evidence."""
    action_span = RolloutSpan(
        span_id=f"sandbox-tool-{index}",
        kind=RolloutEventKind.TOOL_CALL,
        started_at=started_at,
        ended_at=started_at,
        payload={"call_id": action.call_id, "arguments": action.arguments},
        tool_name=action.name,
    )
    payload: JsonObject = (
        {
            "content": observation.content,
            "is_error": observation.is_error,
            "metadata": observation.metadata,
        }
        if observation is not None
        else {"content": "", "is_error": True}
    )
    observation_span = RolloutSpan(
        span_id=f"sandbox-observation-{index}",
        parent_span_id=action_span.span_id,
        kind=RolloutEventKind.OBSERVATION,
        started_at=ended_at,
        ended_at=ended_at,
        payload=payload,
        tool_name=action.name,
        failure=failure,
    )
    return action_span, observation_span


def _observation_failure(
    action: ToolCall,
    observation: Observation,
) -> StructuredFailure | None:
    """Translate an error observation into structured transcript failure evidence."""
    if not observation.is_error:
        return None
    timed_out = observation.metadata.get("timed_out") is True
    exception_type = observation.metadata.get("exception_type")
    return StructuredFailure(
        code=FailureCode.TIMEOUT if timed_out else FailureCode.INTERNAL,
        message=f"environment tool {action.name!r} returned an error observation",
        retryable=observation.metadata.get("retryable") is True,
        exception_type=exception_type if isinstance(exception_type, str) else None,
        attribution=FailureAttribution.TOOL,
        details={"phase": "environment_observation", "tool": action.name},
    )


def _observation_economics(
    observation: Observation,
    duration_seconds: float,
) -> OperationEconomics:
    """Use supplied environment economics while always retaining observed wall latency."""
    raw = observation.metadata.get("economics")
    try:
        economics = OperationEconomics.model_validate(raw) if isinstance(raw, dict) else None
    except ValueError:
        economics = None
    return _with_observed_latency(economics or OperationEconomics(), duration_seconds)


def _with_observed_latency(
    economics: OperationEconomics,
    duration_seconds: float,
) -> OperationEconomics:
    """Fill only missing latency with a directly observed monotonic duration."""
    if economics.latency_seconds is not None:
        return economics
    return economics.model_copy(
        update={
            "latency_seconds": NumericMeasurement(
                value=duration_seconds,
                provenance="observed",
            )
        }
    )


def _limit_failure(
    code: FailureCode,
    message: str,
    exception_type: type[Exception],
    phase: str,
    *,
    provider_dispatch_unknown_spend: bool = False,
) -> StructuredFailure:
    """Create one explicit limit failure without arbitrary exception text."""
    return StructuredFailure(
        code=code,
        message=message,
        exception_type=exception_type.__name__,
        attribution=FailureAttribution.MODEL,
        details={
            "phase": phase,
            "provider_dispatch_unknown_spend": provider_dispatch_unknown_spend,
        },
    )


def _environment_cost_failure(
    message: str,
    phase: str,
    *,
    unknown_spend: bool = False,
) -> StructuredFailure:
    """Create one environment-attributed finite-budget failure."""
    return StructuredFailure(
        code=FailureCode.BUDGET,
        message=message,
        exception_type=SandboxUnknownCostError.__name__,
        attribution=FailureAttribution.ENVIRONMENT,
        details={
            "phase": phase,
            "environment_dispatch_unknown_spend": unknown_spend,
        },
    )
