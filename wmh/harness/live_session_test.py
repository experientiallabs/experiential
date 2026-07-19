# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for the LiveSession host engine, driven by a scripted in-process channel peer."""

from __future__ import annotations

import pytest
from llm_waterfall import ChatRequest, ChatResponse

from wmh.core.types import JsonObject
from wmh.harness.live_session import LiveSession, SessionEvent, ToolOutcome
from wmh.harness.tools import BASH, READ_SKILL, SUBMIT


class ScriptedChannel:
    """A `Channel` whose `recv` replays a fixed inbound frame list; captures outbound sends."""

    def __init__(self, inbound: list[JsonObject]) -> None:
        self._inbound = list(inbound)
        self.sent: list[JsonObject] = []

    def send(self, frame: JsonObject) -> None:
        self.sent.append(frame)

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        if self._inbound:
            return self._inbound.pop(0)
        return None  # exhausted = channel closed


def _completion(
    text: str = "",
    tool_calls: list | None = None,
    usage: dict | None = None,
    provider_receipt: dict | None = None,
) -> ChatResponse:
    msg: JsonObject = {"role": "assistant", "content": text}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    choice: JsonObject = {"index": 0, "message": msg, "finish_reason": "stop"}
    completion: JsonObject = {"choices": [choice]}
    if usage is not None:
        completion["usage"] = usage
    if provider_receipt is not None:
        completion["provider_receipt"] = provider_receipt
    return ChatResponse.model_validate(completion)


def _provider_receipt() -> JsonObject:
    return {
        "provider": "bedrock",
        "provider_request_id": "request-1",
        "response_id": None,
        "requested_model": "us.example.model",
        "response_model": None,
        "system_fingerprint": None,
        "request_digest": "sha256:" + "a" * 64,
        "temperature": 0.7,
        "max_tokens": 16_384,
        "max_tokens_field": "inferenceConfig.maxTokens",
        "seed_supplied": False,
        "cache_config_supplied": False,
        "started_at_unix_s": 10.0,
        "finished_at_unix_s": 11.0,
    }


def _drain(session: LiveSession) -> None:
    for _ in range(100):
        if not session.pump(timeout=0):
            return


def test_start_waits_for_first_idle_state() -> None:
    channel = ScriptedChannel([{"type": "state", "status": "idle"}])
    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=lambda e: None,
        max_output_tokens=16384,
        temperature=0.35,
    )
    session.start()
    assert session.status == "idle"
    assert channel.sent[0]["type"] == "session_start"
    assert channel.sent[0]["turn_cap"] == 60
    assert channel.sent[0]["max_output_tokens"] == 16384
    assert channel.sent[0]["temperature"] == 0.35
    assert channel.sent[0]["conversation_scope"] == "session"


def test_start_can_scope_conversation_to_each_outer_turn() -> None:
    channel = ScriptedChannel([{"type": "state", "status": "idle"}])
    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=lambda e: None,
        conversation_scope="turn",
    )

    session.start()

    assert channel.sent[0]["conversation_scope"] == "turn"


def test_rejects_unknown_conversation_scope() -> None:
    channel = ScriptedChannel([])

    with pytest.raises(ValueError, match="conversation_scope"):
        LiveSession(
            channel,
            tools=[],
            execute_tool=_no_tool,
            on_event=lambda e: None,
            conversation_scope="message",  # ty: ignore[invalid-argument-type]
        )


def test_start_surfaces_the_runner_construction_error() -> None:
    channel = ScriptedChannel(
        [{"type": "episode_error", "note": "could not import the agent harness"}]
    )
    session = LiveSession(channel, tools=[], execute_tool=_no_tool, on_event=lambda e: None)

    with pytest.raises(RuntimeError, match="could not import the agent harness"):
        session.start()


def test_full_turn_emits_ordered_events_and_answers_frames() -> None:
    events: list[SessionEvent] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},  # consumed by start()
            {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
            {"type": "tool_request", "req_id": 2, "name": "bash", "arguments": {"command": "ls"}},
            {
                "type": "tool_request",
                "req_id": 3,
                "name": "submit",
                "arguments": {"answer": "done"},
            },
            {"type": "state", "status": "idle", "reason": "completed", "turns": 1},
        ]
    )

    def execute(name: str, args: JsonObject, emit) -> ToolOutcome:  # noqa: ANN001
        emit("stdout", "file-a\n")
        return ToolOutcome(content="file-a\n", is_error=False)

    session = LiveSession(
        channel,
        tools=[BASH, SUBMIT],
        execute_tool=execute,
        on_event=events.append,
        worker_fn=lambda body: _completion(
            text="on it", usage={"prompt_tokens": 5, "completion_tokens": 7}
        ),
    )
    session.start()
    events.clear()  # drop the initial "ready" state event; assert only the turn's events
    session.send_user_message("list the files")
    _drain(session)

    kinds = [e.kind for e in events]
    assert kinds == [
        "user_message",
        "assistant_message",
        "tool_call",
        "tool_output",
        "tool_result",
        "submit",
        "state",
    ]
    assert events[1].payload["text"] == "on it"
    assert events[2].payload["name"] == "bash"
    assert events[4].payload["content"] == "file-a\n"
    assert events[5].payload["answer"] == "done"

    sent_types = [f["type"] for f in channel.sent]
    assert sent_types.count("user_message") == 1
    assert sent_types.count("llm_response") == 1
    assert sent_types.count("tool_response") == 2  # bash + submit
    assert session.worker_usage.calls == 1
    assert session.worker_usage.input_tokens == 5
    assert session.worker_usage.output_tokens == 7


def test_submit_tool_response_is_answered_without_executor() -> None:
    calls: list[str] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "tool_request", "req_id": 1, "name": "submit", "arguments": {"answer": "x"}},
        ]
    )

    def execute(name: str, args: JsonObject, emit) -> ToolOutcome:  # noqa: ANN001
        calls.append(name)
        return ToolOutcome(content="should not run")

    session = LiveSession(channel, tools=[SUBMIT], execute_tool=execute, on_event=lambda e: None)
    session.start()
    session.send_user_message("submit the answer")
    _drain(session)
    assert calls == []  # submit never routes to the real executor
    resp = next(f for f in channel.sent if f["type"] == "tool_response")
    assert resp["content"] == "submitted"
    assert resp["is_error"] is False


def test_interrupt_sends_abort_frame() -> None:
    channel = ScriptedChannel([{"type": "state", "status": "idle"}])
    session = LiveSession(channel, tools=[], execute_tool=_no_tool, on_event=lambda e: None)
    session.start()
    session.interrupt()
    session.pump(timeout=0)
    assert any(f["type"] == "abort" and f["reason"] == "user_interrupt" for f in channel.sent)


def test_end_sends_abort_then_shutdown_and_closes() -> None:
    channel = ScriptedChannel([{"type": "state", "status": "idle"}])
    session = LiveSession(channel, tools=[], execute_tool=_no_tool, on_event=lambda e: None)
    session.start()
    session.end()
    assert session.pump(timeout=0) is False
    tail = [f["type"] for f in channel.sent[-2:]]
    assert tail == ["abort", "shutdown"]
    assert session.closed


def test_action_budget_exhausts_after_cap() -> None:
    events: list[SessionEvent] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {"command": "a"}},
            {"type": "tool_request", "req_id": 2, "name": "bash", "arguments": {"command": "b"}},
        ]
    )
    ran: list[str] = []

    def execute(name: str, args: JsonObject, emit) -> ToolOutcome:  # noqa: ANN001
        ran.append(str(args.get("command")))
        return ToolOutcome(content="ok")

    session = LiveSession(
        channel, tools=[BASH], execute_tool=execute, on_event=events.append, actions_per_turn=1
    )
    session.start()
    session.send_user_message("go")
    _drain(session)
    assert ran == ["a"]  # second call is over budget, never executed
    results = [e for e in events if e.kind == "tool_result"]
    assert results[1].payload["is_error"] is True
    assert "budget exhausted" in str(results[1].payload["content"])


def test_read_skill_answered_from_bodies() -> None:
    events: list[SessionEvent] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {
                "type": "tool_request",
                "req_id": 1,
                "name": "read_skill",
                "arguments": {"name": "deploy"},
            },
            {
                "type": "tool_request",
                "req_id": 2,
                "name": "read_skill",
                "arguments": {"name": "missing"},
            },
        ]
    )
    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=events.append,
        skill_bodies={"deploy": "run ./deploy.sh"},
    )
    session.start()
    session.send_user_message("load the skill")
    raw_tools = channel.sent[0]["tools"]
    assert isinstance(raw_tools, list)
    advertised = {tool["name"] for tool in raw_tools if isinstance(tool, dict) and "name" in tool}
    assert READ_SKILL.name in advertised
    _drain(session)
    results = [e for e in events if e.kind == "tool_result"]
    assert results[0].payload["content"] == "run ./deploy.sh"
    assert results[1].payload["content"] == "no skill named 'missing'"
    assert results[1].payload["is_error"] is True


def test_harness_temperature_overrides_live_runner_request() -> None:
    requests: list[ChatRequest] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {
                "type": "llm_request",
                "req_id": 1,
                "openai_body": {"messages": [], "temperature": 1.75},
            },
        ]
    )

    def worker(request: ChatRequest) -> ChatResponse:
        requests.append(request)
        return _completion()

    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=lambda event: None,
        worker_fn=worker,
        temperature=0.35,
    )
    session.start()
    session.send_user_message("do the task")
    _drain(session)

    assert [request.temperature for request in requests] == [0.35]


def test_provider_receipt_is_emitted_to_trace_but_not_exposed_to_runner() -> None:
    events: list[SessionEvent] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
        ]
    )
    receipt = _provider_receipt()
    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=events.append,
        worker_fn=lambda request: _completion("ok", provider_receipt=receipt),
    )

    session.start()
    session.send_user_message("do the task")
    _drain(session)

    provider_events = [event for event in events if event.kind == "provider_receipt"]
    assert len(provider_events) == 1
    assert provider_events[0].payload == {**receipt, "turn_call_index": 1}
    response = next(frame for frame in channel.sent if frame["type"] == "llm_response")
    completion = response["completion"]
    assert isinstance(completion, dict)
    assert "provider_receipt" not in completion


def test_provider_receipt_indices_count_only_successful_provider_calls() -> None:
    events: list[SessionEvent] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
            {"type": "llm_request", "req_id": 2, "openai_body": {"messages": []}},
        ]
    )
    attempts = 0

    def worker(_request: ChatRequest) -> ChatResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("provider unavailable")
        return _completion("ok", provider_receipt=_provider_receipt())

    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=events.append,
        worker_fn=worker,
    )

    session.start()
    session.send_user_message("do the task")
    _drain(session)

    provider_events = [event for event in events if event.kind == "provider_receipt"]
    assert session.worker_usage.calls == 1
    assert provider_events[0].payload["turn_call_index"] == 1


def test_harness_output_budget_overrides_and_canonicalizes_runner_request() -> None:
    requests: list[ChatRequest] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {
                "type": "llm_request",
                "req_id": 1,
                "openai_body": {
                    "messages": [],
                    "max_tokens": 999_999,
                    "max_completion_tokens": 888_888,
                    "n": 7,
                    "top_p": 0.99,
                    "reasoning_effort": "high",
                    "service_tier": "priority",
                },
            },
        ]
    )

    def worker(request: ChatRequest) -> ChatResponse:
        requests.append(request)
        return _completion()

    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=lambda event: None,
        worker_fn=worker,
        max_output_tokens=2048,
    )
    session.start()
    session.send_user_message("do the task")
    _drain(session)

    assert len(requests) == 1
    request = requests[0]
    assert request.max_tokens is None
    assert request.max_completion_tokens == 2048
    assert request.model_extra == {}
    assert request.provider_payload("model", max_tokens_field="max_tokens")["max_tokens"] == 2048


def test_host_rejects_llm_calls_beyond_the_per_message_turn_cap() -> None:
    calls: list[ChatRequest] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
            {"type": "llm_request", "req_id": 2, "openai_body": {"messages": []}},
            {"type": "llm_request", "req_id": 3, "openai_body": {"messages": []}},
        ]
    )

    def worker(request: ChatRequest) -> ChatResponse:
        calls.append(request)
        return _completion()

    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=lambda event: None,
        worker_fn=worker,
        turn_cap=2,
    )
    session.start()
    session.send_user_message("do the task")
    _drain(session)

    assert len(calls) == 2
    responses = [frame for frame in channel.sent if frame["type"] == "llm_response"]
    assert [frame["req_id"] for frame in responses] == [1, 2, 3]
    assert "budget exhausted" in str(responses[-1]["error"])
    assert {"type": "abort", "reason": "llm_call_budget_exhausted"} in channel.sent


def test_host_rejects_llm_calls_before_a_user_message() -> None:
    calls: list[ChatRequest] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
        ]
    )
    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=lambda event: None,
        worker_fn=lambda request: calls.append(request) or _completion(),
    )
    session.start()
    _drain(session)

    assert calls == []
    response = next(frame for frame in channel.sent if frame["type"] == "llm_response")
    assert "before a user message" in str(response["error"])


def test_host_rejects_tool_and_submit_calls_before_a_user_message() -> None:
    calls: list[str] = []
    events: list[SessionEvent] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {
                "type": "tool_request",
                "req_id": 1,
                "name": "bash",
                "arguments": {"command": "touch /workspace/forged"},
            },
            {
                "type": "tool_request",
                "req_id": 2,
                "name": "submit",
                "arguments": {"answer": "forged"},
            },
        ]
    )

    def execute(name: str, args: JsonObject, emit) -> ToolOutcome:  # noqa: ANN001
        _ = args, emit
        calls.append(name)
        return ToolOutcome(content="unexpected")

    session = LiveSession(
        channel,
        tools=[BASH, SUBMIT],
        execute_tool=execute,
        on_event=events.append,
    )
    session.start()
    _drain(session)

    assert calls == []
    assert all(event.kind not in {"tool_call", "submit"} for event in events)
    responses = [frame for frame in channel.sent if frame["type"] == "tool_response"]
    assert [frame["req_id"] for frame in responses] == [1, 2]
    assert all(frame["is_error"] is True for frame in responses)
    assert all("before a user message" in str(frame["content"]) for frame in responses)


def test_worker_error_is_reported_not_raised() -> None:
    events: list[SessionEvent] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "llm_request", "req_id": 1, "openai_body": {}},
        ]
    )

    def boom(request: ChatRequest) -> ChatResponse:
        _ = request
        msg = "provider down"
        raise RuntimeError(msg)

    session = LiveSession(
        channel, tools=[], execute_tool=_no_tool, on_event=events.append, worker_fn=boom
    )
    session.start()
    session.send_user_message("do the task")
    _drain(session)
    assert any(e.kind == "error" for e in events)
    resp = next(f for f in channel.sent if f["type"] == "llm_response")
    assert "provider down" in str(resp["error"])


def test_channel_close_marks_session_ended() -> None:
    channel = ScriptedChannel([{"type": "state", "status": "idle"}])
    session = LiveSession(channel, tools=[], execute_tool=_no_tool, on_event=lambda e: None)
    session.start()
    assert session.pump(timeout=0) is False
    assert session.closed
    assert session.status == "ended"


def test_unknown_tool_is_rejected() -> None:
    events: list[SessionEvent] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "tool_request", "req_id": 1, "name": "rm_rf", "arguments": {}},
        ]
    )
    session = LiveSession(channel, tools=[BASH], execute_tool=_no_tool, on_event=events.append)
    session.start()
    session.send_user_message("try the tool")
    _drain(session)
    result = next(e for e in events if e.kind == "tool_result")
    assert result.payload["is_error"] is True
    assert "not available" in str(result.payload["content"])


def _no_tool(name: str, args: JsonObject, emit) -> ToolOutcome:  # noqa: ANN001
    return ToolOutcome(content="", is_error=True)


def test_interrupt_suppresses_a_racing_submit_event() -> None:
    """A submit that arrives after an interrupt for the same turn emits no submit event."""
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            # The interrupt is queued (below) before this in-flight submit is processed.
            {"type": "tool_request", "req_id": 1, "name": "submit", "arguments": {"answer": "x"}},
        ]
    )
    events: list[SessionEvent] = []
    session = LiveSession(channel, tools=[SUBMIT], execute_tool=_no_tool, on_event=events.append)
    session.start()
    events.clear()
    session.send_user_message("solve the task")
    session.interrupt()  # user hits Stop while the submit is racing
    _drain(session)
    # The abort was sent; the racing submit is answered but NOT surfaced as a submit event.
    assert not any(e.kind == "submit" for e in events)
    assert any(f["type"] == "abort" for f in channel.sent)
    resp = next(f for f in channel.sent if f["type"] == "tool_response")
    assert resp["content"] == "submitted"  # runner still gets a response (no hang)


def test_submit_after_state_boundary_is_not_suppressed() -> None:
    """A fresh turn's submit is emitted normally after the aborted turn ended (state boundary)."""
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "state", "status": "idle", "reason": "aborted"},  # aborted turn ended
            {"type": "tool_request", "req_id": 1, "name": "submit", "arguments": {"answer": "y"}},
        ]
    )
    events: list[SessionEvent] = []
    session = LiveSession(channel, tools=[SUBMIT], execute_tool=_no_tool, on_event=events.append)
    session.start()
    session.send_user_message("solve the task")
    session.interrupt()
    events.clear()
    _drain(session)
    assert any(e.kind == "submit" for e in events)


def test_stale_submit_after_a_quick_next_message_is_still_suppressed() -> None:
    """A cancelled turn's in-flight submit is suppressed even if the user already sent a new
    message: only the runner's next state frame (the turn boundary) clears the abort gate."""
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            # The cancelled turn's submit arrives AFTER the user's next message was drained.
            {"type": "tool_request", "req_id": 1, "name": "submit", "arguments": {"answer": "x"}},
        ]
    )
    events: list[SessionEvent] = []
    session = LiveSession(channel, tools=[SUBMIT], execute_tool=_no_tool, on_event=events.append)
    session.start()
    session.interrupt()
    session.send_user_message("do the next thing")  # queued before the stale submit is read
    events.clear()
    _drain(session)
    assert not any(e.kind == "submit" for e in events)  # stale submit stays suppressed


def test_running_state_does_not_clear_the_abort_gate() -> None:
    """A `running` frame is a prompt start, not the cancelled turn's boundary: only `idle`
    clears the gate, so a stale submit read after a `running` frame stays suppressed."""
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "state", "status": "running"},  # prompt-start frame, NOT the boundary
            {"type": "tool_request", "req_id": 1, "name": "submit", "arguments": {"answer": "x"}},
        ]
    )
    events: list[SessionEvent] = []
    session = LiveSession(channel, tools=[SUBMIT], execute_tool=_no_tool, on_event=events.append)
    session.start()
    session.interrupt()
    events.clear()
    _drain(session)
    assert not any(e.kind == "submit" for e in events)  # `running` did not re-enable submit


def test_aborting_skips_real_tool_execution() -> None:
    """While a turn is aborting, a side-effecting tool request is answered interrupted, not run."""
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {"command": "rm x"}},
        ]
    )
    ran: list[str] = []

    def execute(name: str, args: JsonObject, emit) -> ToolOutcome:  # noqa: ANN001
        ran.append(name)
        return ToolOutcome(content="ran")

    session = LiveSession(channel, tools=[BASH], execute_tool=execute, on_event=lambda e: None)
    session.start()
    session.send_user_message("modify the task")
    session.interrupt()  # user hits Stop while a bash request is already queued
    _drain(session)
    assert ran == []  # the tool never executed against the live sandbox
    resp = next(f for f in channel.sent if f["type"] == "tool_response")
    assert resp["is_error"] is True
    assert resp["content"] == "interrupted"


def test_tool_executor_exception_becomes_error_result() -> None:
    """A raising executor yields an error tool_result + response instead of crashing pump()."""
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {"command": "ls"}},
        ]
    )

    def boom(name: str, args: JsonObject, emit) -> ToolOutcome:  # noqa: ANN001
        raise RuntimeError("sandbox gone")

    events: list[SessionEvent] = []
    session = LiveSession(channel, tools=[BASH], execute_tool=boom, on_event=events.append)
    session.start()
    session.send_user_message("go")
    events.clear()
    _drain(session)  # must not raise
    result = next(e for e in events if e.kind == "tool_result")
    assert result.payload["is_error"] is True
    assert "failed" in str(result.payload["content"])
    resp = next(f for f in channel.sent if f["type"] == "tool_response")
    assert resp["is_error"] is True
