"""Tests for running one pi harness turn through an injected runner channel."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import cast

import pytest
from llm_waterfall import ChatRequest, ChatResponse

import wmh.harness.pi_runner as mod
from wmh.core.types import JsonObject
from wmh.harness.live_session import ToolOutcome
from wmh.harness.pi_runner import (
    AmbiguousTaskEnvironmentError,
    CandidateTaskEnvironmentError,
    PiCandidateChannelError,
    PiCandidateError,
    PiCandidateFailureReason,
    PiCandidateFailureStage,
    PiInfrastructureError,
    PiInfrastructureFailureKind,
    PiOutboundFrameTooLargeError,
    PiRunHealth,
    TurnDeadline,
    TurnDeadlineExceeded,
    assemble_pi_harness,
    pi_node_baseline,
    run_pi_turn,
)
from wmh.harness.runner_link import TokenUsage
from wmh.providers.process_worker import (
    ProviderWorkerDeadlineExceeded,
    RequestDeadlineSource,
)


class _ScriptedChannel:
    """A closeable runner channel with a fixed inbound frame sequence."""

    def __init__(self) -> None:
        self._inbound: list[JsonObject] = [
            {"type": "state", "status": "idle"},
            {"type": "state", "status": "running"},
            {
                "type": "tool_request",
                "req_id": 1,
                "name": "submit",
                "arguments": {"answer": "finished"},
            },
            {"type": "state", "status": "idle", "reason": "completed", "turns": 1},
        ]
        self.sent: list[JsonObject] = []
        self.closed = False

    def send(self, frame: JsonObject) -> None:
        self.sent.append(frame)

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        _ = timeout
        return self._inbound.pop(0) if self._inbound else None

    def close(self) -> None:
        self.closed = True


def _no_tool(
    name: str,
    args: JsonObject,
    emit: Callable[[str, str], None],
    deadline: TurnDeadline,
) -> ToolOutcome:
    _ = name, args, emit, deadline
    return ToolOutcome(content="unused")


def _unused_worker(
    request: ChatRequest,
    deadline: TurnDeadline,
) -> ChatResponse:
    _ = request, deadline
    raise AssertionError("the scripted turn should not request a worker completion")


def _completion(
    text: str = "ok",
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> ChatResponse:
    return ChatResponse.model_validate(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            },
        }
    )


def test_pi_node_baseline_and_assembly_match_live_session_contract() -> None:
    doc = pi_node_baseline()

    system, tools, files, skill_bodies = assemble_pi_harness(doc)

    assert doc.runtime_kind() == "pi-node"
    assert system == doc.assembled_prompt()
    assert {tool.name for tool in tools} == set(doc.tools())
    assert files
    assert "src/agent.ts" in files
    assert skill_bodies == {}


def test_run_pi_turn_uses_the_injected_runner_factory() -> None:
    channel = _ScriptedChannel()
    lifecycle: list[str] = []

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        lifecycle.append("open")
        try:
            yield channel
        finally:
            lifecycle.append("close")
            channel.close()

    result = run_pi_turn(
        pi_node_baseline(),
        "complete the task",
        execute_tool=_no_tool,
        worker_fn=_unused_worker,
        runner_factory=runner_factory,
    )

    assert result.answer == "finished"
    assert result.terminal_reason == "completed"
    assert lifecycle == ["open", "close"]
    assert any(frame.get("type") == "session_start" for frame in channel.sent)
    assert any(frame.get("type") == "user_message" for frame in channel.sent)
    assert [event.kind for event in result.events].count("submit") == 1


def test_run_pi_turn_surfaces_runner_cleanup_failure() -> None:
    channel = _ScriptedChannel()

    @contextmanager
    def failing_factory() -> Iterator[_ScriptedChannel]:
        try:
            yield channel
        finally:
            channel.close()
            raise RuntimeError("runner cleanup failed")

    with pytest.raises(RuntimeError, match="runner cleanup failed"):
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=failing_factory,
        )

    assert channel.closed is True


@pytest.mark.parametrize(
    ("inbound", "stage"),
    [
        (
            [{"type": "episode_error", "note": "candidate import failed"}],
            PiCandidateFailureStage.MATERIALIZATION,
        ),
        (
            [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
                {"type": "episode_error", "note": "candidate turn failed"},
            ],
            PiCandidateFailureStage.TURN,
        ),
    ],
)
def test_run_pi_turn_types_candidate_runner_failures(
    inbound: list[JsonObject],
    stage: PiCandidateFailureStage,
) -> None:
    channel = _ScriptedChannel()
    channel._inbound = inbound

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(PiCandidateError, match="candidate .* failed") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    assert caught.value.stage is stage
    assert caught.value.reason is PiCandidateFailureReason.RUNTIME_ERROR
    assert caught.value.events[-1].kind == "error"


def test_silent_candidate_materialization_timeout_is_gradeable_after_handoff() -> None:
    """Candidate import silence after session_start is not evaluator infrastructure."""

    class SilentMaterializationChannel(_ScriptedChannel):
        def recv(self, timeout: float | None = None) -> JsonObject | None:
            _ = timeout
            raise TimeoutError

    channel = SilentMaterializationChannel()
    channel._inbound = []

    @contextmanager
    def runner_factory() -> Iterator[SilentMaterializationChannel]:
        yield channel

    with pytest.raises(PiCandidateError, match="materialization.*did not become ready") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    assert caught.value.stage is PiCandidateFailureStage.MATERIALIZATION
    assert caught.value.reason is PiCandidateFailureReason.TIMEOUT
    assert any(frame.get("type") == "session_start" for frame in channel.sent)


def test_materialization_transport_failure_remains_infrastructure() -> None:
    """A typed transport failure is not blamed on candidate readiness silence."""

    class BrokenMaterializationTransport(_ScriptedChannel):
        def recv(self, timeout: float | None = None) -> JsonObject | None:
            _ = timeout
            raise RuntimeError("runner transport unavailable")

    channel = BrokenMaterializationTransport()
    channel._inbound = []

    @contextmanager
    def runner_factory() -> Iterator[BrokenMaterializationTransport]:
        yield channel

    with pytest.raises(RuntimeError, match="transport unavailable") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    assert not isinstance(caught.value, PiCandidateError)


def test_candidate_error_text_cannot_spoof_provider_infrastructure() -> None:
    """Only the trusted provider wrapper, never candidate-authored text, marks infrastructure."""
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "episode_error", "note": "worker LLM error: candidate-controlled note"},
    ]

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(PiCandidateError, match="candidate-controlled note") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    assert caught.value.stage is PiCandidateFailureStage.TURN
    assert caught.value.reason is PiCandidateFailureReason.RUNTIME_ERROR


def test_run_pi_turn_keeps_worker_failures_infrastructure_typed() -> None:
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
    ]

    secret = "provider-secret-sentinel"

    def failing_worker(
        request: ChatRequest,
        deadline: TurnDeadline,
    ) -> ChatResponse:
        _ = request, deadline
        raise RuntimeError(f"provider unavailable: {secret}")

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(RuntimeError, match="pi turn worker provider failed") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=failing_worker,
            runner_factory=runner_factory,
        )

    assert not isinstance(caught.value, PiCandidateError)
    assert isinstance(caught.value, PiInfrastructureError)
    assert caught.value.kind is PiInfrastructureFailureKind.PROVIDER
    assert secret not in str(caught.value)
    assert secret not in str(channel.sent)
    assert "worker provider unavailable" in str(channel.sent)


@pytest.mark.parametrize(
    ("module", "code", "message", "parameter"),
    [
        (
            "openai._exceptions",
            "invalid_function_parameters",
            "tools[0].function.parameters is invalid",
            None,
        ),
        (
            "openai._exceptions",
            "invalid_request_error",
            "candidate max output tokens were rejected",
            "max_completion_tokens",
        ),
        (
            "openai._exceptions",
            "invalid_request_error",
            "candidate temperature was rejected",
            "temperature",
        ),
        (
            "botocore.exceptions",
            "ValidationException",
            "A toolResult block must follow each toolUse block",
            None,
        ),
    ],
)
def test_candidate_owned_provider_rejections_are_gradeable_zeroes(
    module: str,
    code: str,
    message: str,
    parameter: str | None,
) -> None:
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
    ]

    class CandidateRequestError(RuntimeError):
        status_code = 400

        def __init__(self) -> None:
            super().__init__(message)
            self.code = code
            self.body = {"code": code, "message": message, "param": parameter}
            self.response = {
                "Error": {"Code": code, "Message": message},
                "ResponseMetadata": {"HTTPStatusCode": 400},
            }

    CandidateRequestError.__module__ = module

    def rejected_worker(
        request: ChatRequest,
        deadline: TurnDeadline,
    ) -> ChatResponse:
        _ = request, deadline
        raise CandidateRequestError

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(PiCandidateError, match="worker request was rejected") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=rejected_worker,
            runner_factory=runner_factory,
        )

    assert caught.value.stage is PiCandidateFailureStage.TURN
    assert caught.value.reason is PiCandidateFailureReason.INVALID_REQUEST
    assert caught.value.run_health is PiRunHealth.VALID
    assert message not in str(caught.value)
    assert message not in str(channel.sent)


@pytest.mark.parametrize(
    ("status_code", "code"),
    [
        (401, "invalid_api_key"),
        (404, "DeploymentNotFound"),
        (429, "rate_limit_exceeded"),
        (500, "server_error"),
    ],
)
def test_openai_operational_errors_still_invalidate_the_trial(
    status_code: int,
    code: str,
) -> None:
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
    ]

    class OperationalError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("provider operational failure")
            self.status_code = status_code
            self.code = code
            self.body = {"error": {"code": code}}

    OperationalError.__module__ = "openai._exceptions"

    def failing_worker(
        request: ChatRequest,
        deadline: TurnDeadline,
    ) -> ChatResponse:
        _ = request, deadline
        raise OperationalError

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(PiInfrastructureError) as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=failing_worker,
            runner_factory=runner_factory,
        )

    assert caught.value.kind is PiInfrastructureFailureKind.PROVIDER
    assert caught.value.run_health is PiRunHealth.INFRASTRUCTURE_FAILURE


def test_worker_failure_after_success_preserves_bounded_partial_usage_and_events() -> None:
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
        {"type": "llm_request", "req_id": 2, "openai_body": {}},
    ]
    calls = 0
    secret = "provider-secret-sentinel"

    def worker(
        request: ChatRequest,
        deadline: TurnDeadline,
    ) -> ChatResponse:
        nonlocal calls
        _ = request, deadline
        calls += 1
        if calls == 1:
            return _completion("first answer", input_tokens=17, output_tokens=5)
        raise RuntimeError(f"provider failed with {secret}")

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(PiInfrastructureError) as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=worker,
            runner_factory=runner_factory,
        )

    assert caught.value.kind is PiInfrastructureFailureKind.PROVIDER
    assert caught.value.worker_usage == TokenUsage(input_tokens=17, output_tokens=5, calls=1)
    assert any(
        event.kind == "assistant_message" and event.payload.get("text") == "first answer"
        for event in caught.value.events
    )
    assert len(caught.value.events) <= 4_096
    assert secret not in str(caught.value)


def test_operation_provider_deadline_is_typed_infrastructure() -> None:
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
    ]
    observed_deadlines: list[TurnDeadline] = []

    def deadline_worker(
        request: ChatRequest,
        deadline: TurnDeadline,
    ) -> ChatResponse:
        _ = request
        observed_deadlines.append(deadline)
        raise ProviderWorkerDeadlineExceeded(
            "provider detail",
            source=deadline.limiting_source,
        )

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(PiInfrastructureError) as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=deadline_worker,
            runner_factory=runner_factory,
            provider_call_timeout_s=0.25,
        )

    assert len(observed_deadlines) == 1
    assert 0 < observed_deadlines[0].remaining_s() <= 0.25
    assert observed_deadlines[0].limiting_source is RequestDeadlineSource.OPERATION_LIMIT
    assert caught.value.kind is PiInfrastructureFailureKind.PROVIDER_DEADLINE
    assert "provider detail" not in str(caught.value)
    assert "provider detail" not in str(channel.sent)


def test_caller_budget_provider_deadline_remains_candidate_timeout() -> None:
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
    ]

    def deadline_worker(
        request: ChatRequest,
        deadline: TurnDeadline,
    ) -> ChatResponse:
        _ = request
        assert deadline.limiting_source is RequestDeadlineSource.CALLER_BUDGET
        raise ProviderWorkerDeadlineExceeded(
            "provider detail",
            source=deadline.limiting_source,
        )

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(PiCandidateError) as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=deadline_worker,
            runner_factory=runner_factory,
            timeout_s=0.25,
            provider_call_timeout_s=1,
        )

    assert caught.value.stage is PiCandidateFailureStage.TURN
    assert caught.value.reason is PiCandidateFailureReason.TIMEOUT
    assert "provider detail" not in str(caught.value)
    assert "provider detail" not in str(channel.sent)


def test_run_pi_turn_keeps_tool_executor_failures_infrastructure_typed() -> None:
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}},
    ]

    secret = "environment-secret-sentinel"

    def failing_tool(
        name: str,
        args: JsonObject,
        emit: Callable[[str, str], None],
        deadline: TurnDeadline,
    ) -> ToolOutcome:
        _ = name, args, emit, deadline
        raise RuntimeError(f"task container unavailable: {secret}")

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(RuntimeError, match="pi turn tool executor failed") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=failing_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    assert not isinstance(caught.value, PiCandidateError)
    assert isinstance(caught.value, PiInfrastructureError)
    assert caught.value.kind is PiInfrastructureFailureKind.TASK_ENVIRONMENT
    assert secret not in str(caught.value)
    assert secret not in str(channel.sent)
    assert "task environment unavailable" in str(channel.sent)
    assert caught.value.run_health is PiRunHealth.AMBIGUOUS


def test_candidate_task_environment_destruction_is_a_gradeable_resource_failure() -> None:
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}},
    ]

    def destructive_tool(
        name: str,
        args: JsonObject,
        emit: Callable[[str, str], None],
        deadline: TurnDeadline,
    ) -> ToolOutcome:
        _ = name, args, emit, deadline
        raise CandidateTaskEnvironmentError(PiCandidateFailureReason.RESOURCE_LIMIT)

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(PiCandidateError, match="damaged the task environment") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=destructive_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    assert caught.value.stage is PiCandidateFailureStage.TURN
    assert caught.value.reason is PiCandidateFailureReason.RESOURCE_LIMIT
    assert caught.value.run_health is PiRunHealth.CANDIDATE_DAMAGED


def test_candidate_task_damage_precedes_event_budget_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_MAX_PI_EVENTS", 4)
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}},
    ]

    def destructive_tool(
        name: str,
        args: JsonObject,
        emit: Callable[[str, str], None],
        deadline: TurnDeadline,
    ) -> ToolOutcome:
        _ = name, args, emit, deadline
        raise CandidateTaskEnvironmentError(PiCandidateFailureReason.RESOURCE_LIMIT)

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(PiCandidateError) as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=destructive_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    assert caught.value.reason is PiCandidateFailureReason.RESOURCE_LIMIT
    assert caught.value.run_health is PiRunHealth.CANDIDATE_DAMAGED


def test_ambiguous_task_environment_loss_requires_a_retry() -> None:
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}},
    ]

    def ambiguous_tool(
        name: str,
        args: JsonObject,
        emit: Callable[[str, str], None],
        deadline: TurnDeadline,
    ) -> ToolOutcome:
        _ = name, args, emit, deadline
        raise AmbiguousTaskEnvironmentError

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(PiInfrastructureError) as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=ambiguous_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    assert caught.value.kind is PiInfrastructureFailureKind.TASK_ENVIRONMENT_CONFIRMATION_REQUIRED
    assert caught.value.run_health is PiRunHealth.AMBIGUOUS


def test_run_pi_turn_passes_each_tool_the_current_turn_deadline() -> None:
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}},
        {
            "type": "tool_request",
            "req_id": 2,
            "name": "submit",
            "arguments": {"answer": "finished"},
        },
        {"type": "state", "status": "idle", "reason": "completed", "turns": 1},
    ]
    remaining_budgets: list[float] = []

    def record_deadline(
        name: str,
        args: JsonObject,
        emit: Callable[[str, str], None],
        deadline: TurnDeadline,
    ) -> ToolOutcome:
        _ = name, args, emit
        remaining_budgets.append(deadline.remaining_s())
        return ToolOutcome(content="ok")

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    result = run_pi_turn(
        pi_node_baseline(),
        "complete the task",
        execute_tool=record_deadline,
        worker_fn=_unused_worker,
        runner_factory=runner_factory,
        timeout_s=0.25,
    )

    assert result.answer == "finished"
    assert len(remaining_budgets) == 1
    assert 0 < remaining_budgets[0] <= 0.25


def test_tool_deadline_exhaustion_is_a_gradeable_candidate_timeout() -> None:
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}},
    ]

    def expire_tool(
        name: str,
        args: JsonObject,
        emit: Callable[[str, str], None],
        deadline: TurnDeadline,
    ) -> ToolOutcome:
        _ = name, args, emit, deadline
        raise TurnDeadlineExceeded

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(PiCandidateError, match="did not finish") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=expire_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
            timeout_s=42,
        )

    assert caught.value.stage is PiCandidateFailureStage.TURN
    assert caught.value.reason is PiCandidateFailureReason.TIMEOUT


def test_run_pi_turn_keeps_transport_failures_infrastructure_typed() -> None:
    class FailingTransportChannel(_ScriptedChannel):
        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self._inbound:
                return super().recv(timeout)
            raise RuntimeError("runner transport disconnected")

    channel = FailingTransportChannel()
    channel._inbound = [{"type": "state", "status": "idle"}]

    @contextmanager
    def runner_factory() -> Iterator[FailingTransportChannel]:
        yield channel

    with pytest.raises(RuntimeError, match="transport disconnected") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    assert not isinstance(caught.value, PiCandidateError)


def test_run_pi_turn_types_candidate_process_exit_as_gradeable() -> None:
    class ExitedCandidateChannel(_ScriptedChannel):
        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self._inbound:
                return super().recv(timeout)
            raise PiCandidateChannelError("candidate runner exited with status 1")

    channel = ExitedCandidateChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
    ]

    @contextmanager
    def runner_factory() -> Iterator[ExitedCandidateChannel]:
        yield channel

    with pytest.raises(PiCandidateError, match="exited with status 1") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    assert caught.value.stage is PiCandidateFailureStage.TURN
    assert caught.value.reason is PiCandidateFailureReason.RUNTIME_ERROR


def test_oversized_candidate_materialization_is_gradeable() -> None:
    class OversizedMaterializationChannel(_ScriptedChannel):
        def send(self, frame: JsonObject) -> None:
            if frame.get("type") == "session_start":
                raise PiOutboundFrameTooLargeError("outbound materialization limit exceeded")
            super().send(frame)

    channel = OversizedMaterializationChannel()

    @contextmanager
    def runner_factory() -> Iterator[OversizedMaterializationChannel]:
        yield channel

    with pytest.raises(PiCandidateError, match="materialization limit exceeded") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    assert caught.value.stage is PiCandidateFailureStage.MATERIALIZATION
    assert caught.value.reason is PiCandidateFailureReason.RESOURCE_LIMIT


def test_oversized_benchmark_instruction_remains_infrastructure() -> None:
    class OversizedInstructionChannel(_ScriptedChannel):
        def send(self, frame: JsonObject) -> None:
            if frame.get("type") == "user_message":
                raise PiOutboundFrameTooLargeError("outbound instruction limit exceeded")
            super().send(frame)

    channel = OversizedInstructionChannel()
    channel._inbound = [{"type": "state", "status": "idle"}]

    @contextmanager
    def runner_factory() -> Iterator[OversizedInstructionChannel]:
        yield channel

    with pytest.raises(RuntimeError, match="instruction limit exceeded") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "dataset-owned oversized instruction",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    assert not isinstance(caught.value, PiCandidateError)


def test_run_pi_turn_types_candidate_timeout_as_gradeable() -> None:
    class SilentChannel(_ScriptedChannel):
        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self._inbound:
                return super().recv(timeout)
            raise TimeoutError

    channel = SilentChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
    ]

    @contextmanager
    def runner_factory() -> Iterator[SilentChannel]:
        yield channel

    with pytest.raises(PiCandidateError, match="did not finish") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
            timeout_s=0.001,
        )

    assert caught.value.stage is PiCandidateFailureStage.TURN
    assert caught.value.reason is PiCandidateFailureReason.TIMEOUT


@pytest.mark.parametrize("timeout_s", [float("nan"), float("inf"), float("-inf")])
def test_run_pi_turn_rejects_non_finite_timeout(timeout_s: float) -> None:
    channel = _ScriptedChannel()

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(ValueError, match="finite and positive"):
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
            timeout_s=timeout_s,
        )

    assert channel.sent == []


@pytest.mark.parametrize(
    "provider_call_timeout_s",
    [0.0, float("nan"), float("inf"), float("-inf")],
)
def test_run_pi_turn_rejects_invalid_provider_call_timeout(
    provider_call_timeout_s: float,
) -> None:
    channel = _ScriptedChannel()

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(ValueError, match="provider call timeout must be finite and positive"):
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
            provider_call_timeout_s=provider_call_timeout_s,
        )

    assert channel.sent == []


@pytest.mark.parametrize("timeout_s", [float("nan"), float("inf"), float("-inf")])
def test_turn_deadline_rejects_non_finite_timeout(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        TurnDeadline.after(timeout_s)


def test_run_pi_turn_bounds_a_flood_of_individually_valid_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = _ScriptedChannel()
    channel._inbound = [
        cast("JsonObject", {"type": "state", "status": "idle"}),
        cast("JsonObject", {"type": "state", "status": "running"}),
        *(cast("JsonObject", {"type": "state", "status": "running"}) for _index in range(20)),
    ]
    monkeypatch.setattr("wmh.harness.pi_runner._MAX_PI_EVENTS", 5)

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(PiCandidateError, match="event budget exceeded") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    assert caught.value.stage is PiCandidateFailureStage.TURN
    assert caught.value.reason is PiCandidateFailureReason.RESOURCE_LIMIT
    assert len(caught.value.events) == 5


def test_run_pi_turn_bounds_cumulative_serialized_event_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = _ScriptedChannel()
    channel._inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "state", "status": "running", "reason": "x" * 1_000},
    ]
    byte_budget = 512
    monkeypatch.setattr("wmh.harness.pi_runner._MAX_PI_EVENT_BYTES", byte_budget)

    @contextmanager
    def runner_factory() -> Iterator[_ScriptedChannel]:
        yield channel

    with pytest.raises(PiCandidateError, match="event budget exceeded") as caught:
        run_pi_turn(
            pi_node_baseline(),
            "complete the task",
            execute_tool=_no_tool,
            worker_fn=_unused_worker,
            runner_factory=runner_factory,
        )

    serialized = sum(
        len(
            json.dumps(
                {"kind": event.kind, "payload": event.payload},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        + 1
        for event in caught.value.events
    )
    assert caught.value.stage is PiCandidateFailureStage.TURN
    assert caught.value.reason is PiCandidateFailureReason.RESOURCE_LIMIT
    assert serialized <= byte_budget


def test_cleanup_event_near_budget_cannot_escape_or_replace_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CleanupSendFailureChannel(_ScriptedChannel):
        def send(self, frame: JsonObject) -> None:
            if frame.get("type") == "abort":
                raise RuntimeError("cleanup send failed")
            super().send(frame)

    channel = CleanupSendFailureChannel()
    monkeypatch.setattr("wmh.harness.pi_runner._MAX_PI_EVENTS", 5)

    @contextmanager
    def runner_factory() -> Iterator[CleanupSendFailureChannel]:
        yield channel

    result = run_pi_turn(
        pi_node_baseline(),
        "complete the task",
        execute_tool=_no_tool,
        worker_fn=_unused_worker,
        runner_factory=runner_factory,
    )

    assert result.answer == "finished"
    assert result.terminal_reason == "completed"
    assert len(result.events) == 5
