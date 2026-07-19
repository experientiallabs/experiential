"""Run one pi harness turn through an injected live-session runner.

The harness document remains the source of the agent prompt, tools, skills, limits, and vendored
pi source. The runner process receives that material over the existing live-session frame
protocol, while worker model calls and task tools stay with the trusted evaluator.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from llm_waterfall import ChatRequest, ChatResponse

from wmh.core.types import JsonObject
from wmh.harness.doc import (
    MAX_OUTPUT_TOKENS_ID,
    RUNTIME_KIND_ID,
    HarnessDoc,
    Surface,
    SurfaceKind,
)
from wmh.harness.live_session import (
    EventSink,
    LiveSession,
    OutputEmitter,
    SessionEvent,
    ToolOutcome,
)
from wmh.harness.pi_vendor import pi_agent_code_surfaces
from wmh.harness.runner_link import Channel, TokenUsage
from wmh.harness.runtime import DEFAULT_MAX_OUTPUT_TOKENS
from wmh.harness.tools import READ_SKILL, ToolSpec, resolve_tools
from wmh.providers.failure_attribution import (
    ProviderFailureOwner,
    classify_provider_failure,
)
from wmh.providers.process_worker import (
    ProviderWorkerDeadlineExceeded,
    ProviderWorkerFailure,
    ProviderWorkerUnavailable,
    RequestDeadlineSource,
)

DEFAULT_PI_TURN_TIMEOUT_S = 300.0
DEFAULT_PROVIDER_CALL_TIMEOUT_S = 240.0
_PUMP_INTERVAL_S = 0.5
_MAX_PI_EVENTS = 4_096
_MAX_PI_EVENT_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class TurnDeadline:
    """Absolute monotonic deadline shared by provider and tool calls in one turn."""

    expires_at: float
    limiting_source: RequestDeadlineSource = RequestDeadlineSource.CALLER_BUDGET

    @classmethod
    def after(cls, timeout_s: float) -> TurnDeadline:
        """Construct a deadline relative to the current process monotonic clock."""
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("turn timeout must be finite and positive")
        return cls(expires_at=time.monotonic() + timeout_s)

    def remaining_s(self) -> float:
        """Return the nonnegative time still available to the whole turn."""
        return max(0.0, self.expires_at - time.monotonic())

    def bounded_by(self, operation_timeout_s: float) -> TurnDeadline:
        """Return the earlier of this caller budget and one operation limit."""
        if not math.isfinite(operation_timeout_s) or operation_timeout_s <= 0:
            raise ValueError("operation timeout must be finite and positive")
        operation_expires_at = time.monotonic() + operation_timeout_s
        if operation_expires_at < self.expires_at:
            return TurnDeadline(
                expires_at=operation_expires_at,
                limiting_source=RequestDeadlineSource.OPERATION_LIMIT,
            )
        return self


class TurnDeadlineExceeded(TimeoutError):
    """A deadline-aware executor exhausted its caller-owned turn budget."""


class DeadlineAwareToolExecutor(Protocol):
    """Execute one task tool without running beyond the caller's absolute deadline."""

    def __call__(
        self,
        name: str,
        arguments: JsonObject,
        emit: OutputEmitter,
        deadline: TurnDeadline,
        /,
    ) -> ToolOutcome: ...


class DeadlineAwareWorker(Protocol):
    """Complete one provider request within the caller-owned turn deadline."""

    def __call__(
        self,
        request: ChatRequest,
        deadline: TurnDeadline,
        /,
    ) -> ChatResponse: ...


class PiRunnerFactory(Protocol):
    """Open one pi runner channel and own its cleanup."""

    def __call__(self) -> AbstractContextManager[Channel]: ...


class PiCandidateChannelError(RuntimeError):
    """Candidate-controlled runner exit or protocol violation reported by a channel."""


class PiOutboundFrameTooLargeError(ValueError):
    """A host-to-runner frame exceeded the isolated transport's fixed byte ceiling."""


class _PiCandidateEventBudgetExceeded(BaseException):
    """Control-flow signal that bypasses LiveSession's ordinary sink-error suppression."""


class PiCandidateFailureStage(StrEnum):
    """The candidate-controlled phase that failed."""

    MATERIALIZATION = "materialization"
    TURN = "turn"


class PiCandidateFailureReason(StrEnum):
    """Bounded reason a candidate-controlled pi execution failed."""

    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    RUNTIME_ERROR = "runtime_error"
    INVALID_REQUEST = "invalid_request"


class PiRunHealth(StrEnum):
    """Whether one run is valid evidence or needs operational handling."""

    VALID = "valid"
    CANDIDATE_DAMAGED = "candidate_damaged"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    AMBIGUOUS = "ambiguous"


class CandidateTaskEnvironmentError(RuntimeError):
    """A candidate action destroyed or exhausted its isolated task environment."""

    def __init__(self, reason: PiCandidateFailureReason) -> None:
        if reason not in {
            PiCandidateFailureReason.RESOURCE_LIMIT,
            PiCandidateFailureReason.RUNTIME_ERROR,
        }:
            raise ValueError("candidate task environment errors need a resource or runtime reason")
        super().__init__("candidate damaged the task environment")
        self.reason = reason


class AmbiguousTaskEnvironmentError(RuntimeError):
    """The task environment disappeared without enough evidence to assign ownership."""


class PiInfrastructureFailureKind(StrEnum):
    """Trusted failure sources that must invalidate an evaluation trial."""

    PROVIDER = "provider"
    PROVIDER_DEADLINE = "provider_deadline"
    TASK_ENVIRONMENT = "task_environment"
    TASK_ENVIRONMENT_CONFIRMATION_REQUIRED = "task_environment_confirmation_required"


class PiInfrastructureError(RuntimeError):
    """Sanitized host-side failure with a stable machine-readable kind."""

    def __init__(
        self,
        kind: PiInfrastructureFailureKind,
        *,
        events: tuple[SessionEvent, ...] = (),
        worker_usage: TokenUsage | None = None,
        run_health: PiRunHealth | None = None,
    ) -> None:
        message = {
            PiInfrastructureFailureKind.PROVIDER: "pi turn worker provider failed",
            PiInfrastructureFailureKind.PROVIDER_DEADLINE: (
                "pi turn worker provider deadline expired"
            ),
            PiInfrastructureFailureKind.TASK_ENVIRONMENT: "pi turn tool executor failed",
            PiInfrastructureFailureKind.TASK_ENVIRONMENT_CONFIRMATION_REQUIRED: (
                "pi turn task environment requires confirmation"
            ),
        }[kind]
        super().__init__(message)
        self.kind = kind
        self.events = events
        self.worker_usage = worker_usage or TokenUsage()
        self.run_health = (
            run_health
            or {
                PiInfrastructureFailureKind.PROVIDER: PiRunHealth.INFRASTRUCTURE_FAILURE,
                PiInfrastructureFailureKind.PROVIDER_DEADLINE: (PiRunHealth.INFRASTRUCTURE_FAILURE),
                PiInfrastructureFailureKind.TASK_ENVIRONMENT: PiRunHealth.AMBIGUOUS,
                PiInfrastructureFailureKind.TASK_ENVIRONMENT_CONFIRMATION_REQUIRED: (
                    PiRunHealth.AMBIGUOUS
                ),
            }[kind]
        )


class PiCandidateError(RuntimeError):
    """A candidate harness failure whose resulting task state remains gradeable."""

    def __init__(
        self,
        message: str,
        *,
        stage: PiCandidateFailureStage,
        reason: PiCandidateFailureReason,
        events: tuple[SessionEvent, ...],
        worker_usage: TokenUsage,
        run_health: PiRunHealth = PiRunHealth.VALID,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.reason = reason
        self.events = events
        self.worker_usage = worker_usage
        self.run_health = run_health


@dataclass(frozen=True)
class PiTurnResult:
    """The host-observed result of one pi user turn."""

    answer: str
    terminal_reason: str
    events: tuple[SessionEvent, ...]
    worker_usage: TokenUsage
    run_health: PiRunHealth = PiRunHealth.VALID


class _EpisodeObservingChannel:
    """Record fatal runner frames while preserving channel transport behavior."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel
        self.episode_error_message: str | None = None
        self.candidate_channel_error_message: str | None = None
        self.candidate_handoff_complete = False

    def send(self, frame: JsonObject) -> None:
        try:
            self._channel.send(frame)
        except PiCandidateChannelError as exc:
            self.candidate_channel_error_message = str(exc)
            raise
        except PiOutboundFrameTooLargeError as exc:
            if frame.get("type") == "session_start":
                self.candidate_channel_error_message = str(exc)
            raise
        if frame.get("type") == "session_start":
            # A successful transport write is the trust boundary after which silence during
            # import or construction belongs to the candidate, not evaluator infrastructure.
            self.candidate_handoff_complete = True

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        try:
            frame = self._channel.recv(timeout)
        except PiCandidateChannelError as exc:
            self.candidate_channel_error_message = str(exc)
            raise
        if frame is not None and frame.get("type") == "episode_error":
            note = frame.get("note")
            self.episode_error_message = note if isinstance(note, str) else "runner error"
        return frame


def assemble_pi_harness(
    doc: HarnessDoc,
) -> tuple[str, list[ToolSpec], dict[str, str], dict[str, str]]:
    """Derive the live-session prompt, tools, source files, and skill bodies from a document."""
    tool_names = doc.tools()
    if doc.skills() and READ_SKILL.name not in tool_names:
        tool_names.append(READ_SKILL.name)
    tool_specs = resolve_tools(tool_names)
    system = doc.assembled_prompt()
    files = {surface.path: surface.content for surface in doc.code_files() if surface.path}
    skill_bodies = {skill.name: skill.body for skill in doc.skills()}
    return system, tool_specs, files, skill_bodies


def pi_node_baseline(name: str = "local-session") -> HarnessDoc:
    """Return the default prompt and tools with the vendored pi agent as runnable source."""
    base = HarnessDoc.baseline(name)
    surfaces = [
        *base.surfaces,
        Surface(
            id=MAX_OUTPUT_TOKENS_ID,
            kind=SurfaceKind.PARAM,
            content=str(DEFAULT_MAX_OUTPUT_TOKENS),
        ),
        *pi_agent_code_surfaces(),
        Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
    ]
    return HarnessDoc(name=name, surfaces=surfaces)


def run_pi_turn(
    doc: HarnessDoc,
    instruction: str,
    *,
    execute_tool: DeadlineAwareToolExecutor,
    worker_fn: DeadlineAwareWorker,
    on_event: EventSink | None = None,
    runner_factory: PiRunnerFactory,
    timeout_s: float = DEFAULT_PI_TURN_TIMEOUT_S,
    provider_call_timeout_s: float = DEFAULT_PROVIDER_CALL_TIMEOUT_S,
) -> PiTurnResult:
    """Run one instruction through the document's pi harness and return its observed event trace.

    The injected factory chooses and owns the isolated runner channel. The worker provider and
    tool executor remain with the evaluator, and only framed requests cross that boundary.
    """
    if doc.runtime_kind() != "pi-node":
        raise ValueError(
            f"pi turn runner requires runtime kind 'pi-node', got {doc.runtime_kind()!r}"
        )
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("pi turn timeout must be finite and positive")
    if not math.isfinite(provider_call_timeout_s) or provider_call_timeout_s <= 0:
        raise ValueError("provider call timeout must be finite and positive")

    system, tools, files, skill_bodies = assemble_pi_harness(doc)
    events: list[SessionEvent] = []
    answer = ""
    terminal_reason = ""
    turn_started = False
    infrastructure_failure: PiInfrastructureError | None = None
    candidate_worker_failure = False
    candidate_task_failure: CandidateTaskEnvironmentError | None = None
    turn_deadline_exceeded = False
    turn_deadline: TurnDeadline | None = None
    event_bytes = 0
    finalizing = False

    def record(event: SessionEvent) -> None:
        nonlocal answer, event_bytes, infrastructure_failure, terminal_reason, turn_started
        if finalizing:
            return
        encoded = json.dumps(
            {"kind": event.kind, "payload": event.payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if len(events) >= _MAX_PI_EVENTS or event_bytes + len(encoded) + 1 > _MAX_PI_EVENT_BYTES:
            if infrastructure_failure is not None:
                return
            raise _PiCandidateEventBudgetExceeded
        events.append(event)
        event_bytes += len(encoded) + 1
        if event.kind == "submit":
            submitted = event.payload.get("answer")
            answer = submitted if isinstance(submitted, str) else ""
        elif event.kind == "state":
            status = event.payload.get("status")
            if status == "running":
                turn_started = True
            elif status == "idle" and turn_started:
                reason = event.payload.get("reason")
                terminal_reason = reason if isinstance(reason, str) else "completed"
        if on_event is not None:
            on_event(event)

    def checked_worker(request: ChatRequest) -> ChatResponse:
        nonlocal candidate_worker_failure, infrastructure_failure, turn_deadline_exceeded
        if turn_deadline is None:
            infrastructure_failure = PiInfrastructureError(PiInfrastructureFailureKind.PROVIDER)
            raise RuntimeError("worker provider unavailable")
        try:
            return worker_fn(request, turn_deadline.bounded_by(provider_call_timeout_s))
        except ProviderWorkerDeadlineExceeded as exc:
            if exc.source is RequestDeadlineSource.CALLER_BUDGET:
                turn_deadline_exceeded = True
                raise RuntimeError("pi turn deadline exhausted during provider execution") from None
            infrastructure_failure = PiInfrastructureError(
                PiInfrastructureFailureKind.PROVIDER_DEADLINE
            )
            raise RuntimeError("worker provider deadline expired") from None
        except ProviderWorkerFailure as exc:
            attribution = exc.attribution
            if attribution.owner is ProviderFailureOwner.CANDIDATE:
                candidate_worker_failure = True
                raise RuntimeError("candidate worker request rejected") from None
            infrastructure_failure = PiInfrastructureError(PiInfrastructureFailureKind.PROVIDER)
            raise RuntimeError("worker provider unavailable") from None
        except ProviderWorkerUnavailable:
            infrastructure_failure = PiInfrastructureError(PiInfrastructureFailureKind.PROVIDER)
            raise RuntimeError("worker provider unavailable") from None
        except Exception as exc:  # noqa: BLE001 - replace trusted error text before it crosses
            attribution = classify_provider_failure(exc)
            if attribution.owner is ProviderFailureOwner.CANDIDATE:
                candidate_worker_failure = True
                raise RuntimeError("candidate worker request rejected") from None
            infrastructure_failure = PiInfrastructureError(PiInfrastructureFailureKind.PROVIDER)
            raise RuntimeError("worker provider unavailable") from None

    def checked_execute_tool(
        name: str,
        args: JsonObject,
        emit: OutputEmitter,
    ) -> ToolOutcome:
        nonlocal candidate_task_failure, infrastructure_failure, turn_deadline_exceeded
        if turn_deadline is None:
            raise RuntimeError("task tools are unavailable before the pi turn starts")
        try:
            return execute_tool(name, args, emit, turn_deadline)
        except CandidateTaskEnvironmentError as exc:
            candidate_task_failure = exc
            raise RuntimeError("candidate damaged the task environment") from None
        except AmbiguousTaskEnvironmentError:
            infrastructure_failure = PiInfrastructureError(
                PiInfrastructureFailureKind.TASK_ENVIRONMENT_CONFIRMATION_REQUIRED,
                run_health=PiRunHealth.AMBIGUOUS,
            )
            raise RuntimeError("task environment health is ambiguous") from None
        except TurnDeadlineExceeded:
            turn_deadline_exceeded = True
            raise RuntimeError("pi turn deadline exhausted during task execution") from None
        except Exception:  # noqa: BLE001 - replace trusted error text before it crosses to pi
            infrastructure_failure = PiInfrastructureError(
                PiInfrastructureFailureKind.TASK_ENVIRONMENT
            )
            raise RuntimeError("task environment unavailable") from None

    def evidenced_infrastructure(error: PiInfrastructureError) -> PiInfrastructureError:
        """Attach bounded partial evidence and usage without exposing the trusted raw failure."""
        return PiInfrastructureError(
            error.kind,
            events=tuple(events),
            worker_usage=session.worker_usage.model_copy(),
            run_health=error.run_health,
        )

    def evidenced_candidate(
        *,
        stage: PiCandidateFailureStage,
    ) -> PiCandidateError | None:
        """Materialize a pending trusted candidate attribution with bounded evidence."""
        if candidate_worker_failure:
            return PiCandidateError(
                f"pi candidate {stage.value} failed: worker request was rejected",
                stage=stage,
                reason=PiCandidateFailureReason.INVALID_REQUEST,
                events=tuple(events),
                worker_usage=session.worker_usage.model_copy(),
            )
        if candidate_task_failure is not None:
            return PiCandidateError(
                f"pi candidate {stage.value} failed: candidate damaged the task environment",
                stage=stage,
                reason=candidate_task_failure.reason,
                events=tuple(events),
                worker_usage=session.worker_usage.model_copy(),
                run_health=PiRunHealth.CANDIDATE_DAMAGED,
            )
        return None

    with runner_factory() as channel:
        observed_channel = _EpisodeObservingChannel(channel)
        session = LiveSession(
            observed_channel,
            tools=tools,
            execute_tool=checked_execute_tool,
            on_event=record,
            files=files,
            system_prompt=system,
            skill_bodies=skill_bodies,
            worker_fn=checked_worker,
            turn_cap=doc.max_turns(),
            max_output_tokens=doc.max_output_tokens(),
            temperature=doc.temperature(),
            conversation_scope="turn",
        )
        try:
            try:
                session.start()
            except _PiCandidateEventBudgetExceeded as exc:
                if infrastructure_failure is not None:
                    raise evidenced_infrastructure(infrastructure_failure) from exc
                pending_candidate = evidenced_candidate(
                    stage=PiCandidateFailureStage.MATERIALIZATION
                )
                if pending_candidate is not None:
                    raise pending_candidate from exc
                raise PiCandidateError(
                    "pi candidate materialization failed: event budget exceeded",
                    stage=PiCandidateFailureStage.MATERIALIZATION,
                    reason=PiCandidateFailureReason.RESOURCE_LIMIT,
                    events=tuple(events),
                    worker_usage=session.worker_usage.model_copy(),
                ) from exc
            except PiOutboundFrameTooLargeError as exc:
                candidate_message = observed_channel.candidate_channel_error_message
                if candidate_message is not None:
                    raise PiCandidateError(
                        f"pi candidate materialization failed: {candidate_message}",
                        stage=PiCandidateFailureStage.MATERIALIZATION,
                        reason=PiCandidateFailureReason.RESOURCE_LIMIT,
                        events=tuple(events),
                        worker_usage=session.worker_usage.model_copy(),
                    ) from exc
                raise
            except RuntimeError as exc:
                if infrastructure_failure is not None:
                    raise evidenced_infrastructure(infrastructure_failure) from exc
                pending_candidate = evidenced_candidate(
                    stage=PiCandidateFailureStage.MATERIALIZATION
                )
                if pending_candidate is not None:
                    raise pending_candidate from exc
                candidate_message = (
                    observed_channel.episode_error_message
                    or observed_channel.candidate_channel_error_message
                )
                if candidate_message is not None:
                    raise PiCandidateError(
                        f"pi candidate materialization failed: {candidate_message}",
                        stage=PiCandidateFailureStage.MATERIALIZATION,
                        reason=PiCandidateFailureReason.RUNTIME_ERROR,
                        events=tuple(events),
                        worker_usage=session.worker_usage.model_copy(),
                    ) from exc
                if observed_channel.candidate_handoff_complete and session.failure_message is None:
                    raise PiCandidateError(
                        "pi candidate materialization failed: runner did not become ready",
                        stage=PiCandidateFailureStage.MATERIALIZATION,
                        reason=PiCandidateFailureReason.TIMEOUT,
                        events=tuple(events),
                        worker_usage=session.worker_usage.model_copy(),
                    ) from exc
                raise
            if infrastructure_failure is not None:
                raise evidenced_infrastructure(infrastructure_failure)
            pending_candidate = evidenced_candidate(stage=PiCandidateFailureStage.MATERIALIZATION)
            if pending_candidate is not None:
                raise pending_candidate
            turn_deadline = TurnDeadline.after(timeout_s)
            session.send_user_message(instruction)
            try:
                while not session.closed and not terminal_reason:
                    remaining = turn_deadline.remaining_s()
                    if remaining <= 0:
                        session.interrupt("pi_turn_timeout")
                        session.flush_pending_intents()
                        raise PiCandidateError(
                            f"pi candidate turn failed: did not finish within {timeout_s:g}s",
                            stage=PiCandidateFailureStage.TURN,
                            reason=PiCandidateFailureReason.TIMEOUT,
                            events=tuple(events),
                            worker_usage=session.worker_usage.model_copy(),
                        )
                    session.pump(timeout=min(_PUMP_INTERVAL_S, remaining))
                    if infrastructure_failure is not None:
                        raise evidenced_infrastructure(infrastructure_failure)
                    pending_candidate = evidenced_candidate(stage=PiCandidateFailureStage.TURN)
                    if pending_candidate is not None:
                        raise pending_candidate
                    if turn_deadline_exceeded:
                        session.interrupt("pi_turn_timeout")
                        session.flush_pending_intents()
                        raise PiCandidateError(
                            f"pi candidate turn failed: did not finish within {timeout_s:g}s",
                            stage=PiCandidateFailureStage.TURN,
                            reason=PiCandidateFailureReason.TIMEOUT,
                            events=tuple(events),
                            worker_usage=session.worker_usage.model_copy(),
                        )
            except _PiCandidateEventBudgetExceeded as exc:
                if infrastructure_failure is not None:
                    raise evidenced_infrastructure(infrastructure_failure) from exc
                pending_candidate = evidenced_candidate(stage=PiCandidateFailureStage.TURN)
                if pending_candidate is not None:
                    raise pending_candidate from exc
                raise PiCandidateError(
                    "pi candidate turn failed: event budget exceeded",
                    stage=PiCandidateFailureStage.TURN,
                    reason=PiCandidateFailureReason.RESOURCE_LIMIT,
                    events=tuple(events),
                    worker_usage=session.worker_usage.model_copy(),
                ) from exc
            if session.failure_message is not None:
                candidate_message = (
                    observed_channel.episode_error_message
                    or observed_channel.candidate_channel_error_message
                )
                if candidate_message is not None:
                    raise PiCandidateError(
                        f"pi candidate turn failed: {candidate_message}",
                        stage=PiCandidateFailureStage.TURN,
                        reason=PiCandidateFailureReason.RUNTIME_ERROR,
                        events=tuple(events),
                        worker_usage=session.worker_usage.model_copy(),
                    )
                raise RuntimeError(f"pi turn runner failed: {session.failure_message}")
            if not terminal_reason:
                raise RuntimeError("pi turn runner ended before the turn reached idle")
            usage = session.worker_usage.model_copy()
        finally:
            finalizing = True
            if not session.closed:
                session.end()
                session.flush_pending_intents()

    return PiTurnResult(
        answer=answer,
        terminal_reason=terminal_reason,
        events=tuple(events),
        worker_usage=usage,
        run_health=PiRunHealth.VALID,
    )
