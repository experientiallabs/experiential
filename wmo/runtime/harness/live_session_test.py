# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for the LiveSession host engine, driven by a scripted in-process channel peer."""

from __future__ import annotations

import threading

import pytest

from wmo.common.core.types import JsonObject
from wmo.common.vendor.waterfall import ChatRequest, ChatResponse
from wmo.runtime.harness.live_session import LiveSession, SessionEvent, ToolOutcome
from wmo.runtime.harness.tools import BASH, READ_SKILL, SUBMIT


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
    tool_calls: list[JsonObject] | None = None,
    usage: JsonObject | None = None,
) -> ChatResponse:
    msg: JsonObject = {"role": "assistant", "content": text}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    choice: JsonObject = {"index": 0, "message": msg, "finish_reason": "stop"}
    completion: JsonObject = {"choices": [choice]}
    if usage is not None:
        completion["usage"] = usage
    return ChatResponse.model_validate(completion)


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


def test_start_carries_the_providers_served_context_window() -> None:
    class ContextProvider:
        def complete_chat(self, request: ChatRequest) -> ChatResponse:
            _ = request
            return _completion()

        def context_window(self) -> int | None:
            return 65_536

    channel = ScriptedChannel([{"type": "state", "status": "idle"}])
    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=lambda event: None,
        provider=ContextProvider(),
    )

    session.start()

    assert channel.sent[0]["context_window"] == 65_536


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


def test_channel_type_error_is_not_retried_as_a_second_receive() -> None:
    class BrokenChannel:
        def __init__(self) -> None:
            self.receives = 0

        def send(self, frame: JsonObject) -> None:
            _ = frame

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            _ = timeout
            self.receives += 1
            raise TypeError("decoder bug")

    channel = BrokenChannel()
    session = LiveSession(channel, tools=[], execute_tool=_no_tool, on_event=lambda event: None)

    with pytest.raises(RuntimeError, match="decoder bug"):
        session.start()

    assert channel.receives == 1


def test_full_turn_emits_ordered_events_and_answers_frames() -> None:
    events: list[SessionEvent] = []
    completed_state: JsonObject = {
        "type": "state",
        "status": "idle",
        "reason": "completed",
        "turns": 1,
    }
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
            completed_state,
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
    message_id = session.send_user_message("list the files")
    completed_state["msg_id"] = message_id
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
    assert session.last_completed_message_id == message_id


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
    _drain(session)
    assert calls == []  # submit never routes to the real executor
    resp = next(f for f in channel.sent if f["type"] == "tool_response")
    assert resp["content"] == "submitted"
    assert resp["is_error"] is False


def test_interrupt_sends_abort_frame() -> None:
    aborted_state: JsonObject = {"type": "state", "status": "idle", "reason": "aborted"}
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            aborted_state,
        ]
    )
    cancelled: list[bool] = []
    resets: list[bool] = []
    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=lambda e: None,
        cancel_active=lambda: cancelled.append(True),
        reset_cancel=lambda: resets.append(True),
    )
    session.start()
    message_id = session.send_user_message("go")
    aborted_state["msg_id"] = message_id
    session.interrupt()
    session.pump(timeout=0)
    assert any(f["type"] == "abort" and f["reason"] == "user_interrupt" for f in channel.sent)
    assert cancelled == [True]
    assert resets == [True, True]  # initial ready state, then the aborted turn boundary


def test_matching_idle_cannot_reset_before_the_cancel_hook_finishes() -> None:
    """A delayed cancel hook must land before its matching terminal reset."""
    idle_delivered = threading.Event()

    class _SignallingChannel(ScriptedChannel):
        def __init__(self, inbound: list[JsonObject]) -> None:
            super().__init__(inbound)
            self._receives = 0

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            self._receives += 1
            frame = super().recv(timeout)
            if self._receives == 2:
                idle_delivered.set()
            return frame

    aborted_state: JsonObject = {"type": "state", "status": "idle", "reason": "aborted"}
    channel = _SignallingChannel([{"type": "state", "status": "idle"}, aborted_state])
    cancel_entered = threading.Event()
    release_cancel = threading.Event()
    reset_called = threading.Event()
    hook_order: list[str] = []
    executor_cancelled = False

    def cancel() -> None:
        nonlocal executor_cancelled
        cancel_entered.set()
        release_cancel.wait(timeout=2)
        executor_cancelled = True
        hook_order.append("cancel")

    def reset() -> None:
        nonlocal executor_cancelled
        executor_cancelled = False
        hook_order.append("reset")
        reset_called.set()

    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=lambda event: None,
        cancel_active=cancel,
        reset_cancel=reset,
    )
    session.start()
    hook_order.clear()
    reset_called.clear()
    message_id = session.send_user_message("go")
    aborted_state["msg_id"] = message_id

    interrupt_thread = threading.Thread(target=session.interrupt)
    interrupt_thread.start()
    assert cancel_entered.wait(timeout=1)
    pump_thread = threading.Thread(target=session.pump, kwargs={"timeout": 0})
    pump_thread.start()
    assert idle_delivered.wait(timeout=1)
    reset_raced_ahead = reset_called.wait(timeout=0.1)
    release_cancel.set()
    interrupt_thread.join(timeout=1)
    pump_thread.join(timeout=1)

    assert not interrupt_thread.is_alive()
    assert not pump_thread.is_alive()
    assert not reset_raced_ahead
    assert hook_order == ["cancel", "reset"]
    assert executor_cancelled is False


def test_idle_callback_can_interrupt_the_deferred_turn() -> None:
    """The next prompt stays active while an aborted prompt crosses its idle boundary."""
    aborted_state: JsonObject = {"type": "state", "status": "idle", "reason": "aborted"}
    channel = ScriptedChannel([{"type": "state", "status": "idle"}, aborted_state])
    holder: dict[str, LiveSession] = {}
    cancellations: list[bool] = []

    def on_event(event: SessionEvent) -> None:
        if event.kind == "state" and event.payload.get("reason") == "aborted":
            holder["session"].interrupt()

    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=on_event,
        cancel_active=lambda: cancellations.append(True),
    )
    holder["session"] = session
    session.start()
    first_message_id = session.send_user_message("first")
    aborted_state["msg_id"] = first_message_id
    session.interrupt()
    session.send_user_message("second")

    session.pump(timeout=0)
    session.flush_pending_intents()

    sent_messages = [frame for frame in channel.sent if frame["type"] == "user_message"]
    aborts = [frame for frame in channel.sent if frame["type"] == "abort"]
    assert [frame["text"] for frame in sent_messages] == ["first", "second"]
    assert len(aborts) == 2
    assert cancellations == [True, True]
    assert session.turn_active


def test_stale_idle_does_not_reset_a_newer_interrupted_message() -> None:
    """An old idle acknowledgement cannot re-enable work for a newer cancelled prompt."""
    running_state: JsonObject = {"type": "state", "status": "running"}
    stale_idle: JsonObject = {"type": "state", "status": "idle", "reason": "completed"}
    interrupted_idle: JsonObject = {"type": "state", "status": "idle", "reason": "aborted"}
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            running_state,
            stale_idle,
            {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
            interrupted_idle,
        ]
    )
    resets: list[bool] = []
    worker_calls: list[ChatRequest] = []

    def worker(request: ChatRequest) -> ChatResponse:
        worker_calls.append(request)
        return _completion(text="must not run")

    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=lambda event: None,
        worker_fn=worker,
        reset_cancel=lambda: resets.append(True),
    )
    session.start()
    resets.clear()
    first_message_id = session.send_user_message("first")
    running_state["msg_id"] = first_message_id
    stale_idle["msg_id"] = first_message_id
    session.pump(timeout=0)

    second_message_id = session.send_user_message("second")
    interrupted_idle["msg_id"] = second_message_id
    session.interrupt()
    session.pump(timeout=0)

    assert session.status == "running"
    assert session.last_completed_message_id == first_message_id
    assert resets == []

    session.pump(timeout=0)
    response = next(frame for frame in channel.sent if frame["type"] == "llm_response")
    assert response["error"] == "interrupted"
    assert worker_calls == []

    session.pump(timeout=0)
    assert session.status == "idle"
    assert session.last_completed_message_id == second_message_id
    assert resets == [True]


def test_stale_idle_does_not_make_the_newer_message_uninterruptible() -> None:
    """A stale pre-Stop idle state cannot make the current prompt appear inactive."""
    first_running: JsonObject = {"type": "state", "status": "running"}
    stale_idle: JsonObject = {"type": "state", "status": "idle", "reason": "completed"}
    interrupted_idle: JsonObject = {"type": "state", "status": "idle", "reason": "aborted"}
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            first_running,
            stale_idle,
            {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}},
            interrupted_idle,
        ]
    )
    cancelled: list[bool] = []
    ran: list[str] = []

    def execute(name: str, args: JsonObject, emit) -> ToolOutcome:  # noqa: ANN001
        ran.append(name)
        return ToolOutcome(content="must not run")

    session = LiveSession(
        channel,
        tools=[BASH],
        execute_tool=execute,
        on_event=lambda event: None,
        cancel_active=lambda: cancelled.append(True),
    )
    session.start()
    first_message_id = session.send_user_message("first")
    first_running["msg_id"] = first_message_id
    stale_idle["msg_id"] = first_message_id
    session.pump(timeout=0)

    second_message_id = session.send_user_message("second")
    interrupted_idle["msg_id"] = second_message_id
    session.pump(timeout=0)
    assert session.status == "running"

    session.interrupt()
    session.pump(timeout=0)

    assert cancelled == [True]
    assert any(frame["type"] == "abort" for frame in channel.sent)
    assert ran == []
    response = next(frame for frame in channel.sent if frame["type"] == "tool_response")
    assert response["content"] == "interrupted"
    assert response["is_error"] is True

    session.pump(timeout=0)
    assert session.status == "idle"


def test_interrupt_while_idle_is_ignored_and_does_not_poison_the_next_turn() -> None:
    running_state: JsonObject = {"type": "state", "status": "running"}
    completed_state: JsonObject = {"type": "state", "status": "idle", "reason": "completed"}
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            running_state,
            {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}},
            completed_state,
        ]
    )
    cancelled: list[bool] = []
    ran: list[str] = []

    def execute(name: str, args: JsonObject, emit) -> ToolOutcome:  # noqa: ANN001
        ran.append(name)
        return ToolOutcome(content="ok")

    session = LiveSession(
        channel,
        tools=[BASH],
        execute_tool=execute,
        on_event=lambda event: None,
        cancel_active=lambda: cancelled.append(True),
    )
    session.start()
    session.interrupt()
    message_id = session.send_user_message("go")
    running_state["msg_id"] = message_id
    completed_state["msg_id"] = message_id
    _drain(session)

    assert cancelled == []
    assert not any(frame["type"] == "abort" for frame in channel.sent)
    assert ran == ["bash"]
    response = next(frame for frame in channel.sent if frame["type"] == "tool_response")
    assert response["content"] == "ok"
    assert response["is_error"] is False


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
    _drain(session)

    assert [request.temperature for request in requests] == [0.35]


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
    _drain(session)
    assert any(e.kind == "error" for e in events)
    resp = next(f for f in channel.sent if f["type"] == "llm_response")
    assert "provider down" in str(resp["error"])


def test_interrupt_releases_a_blocking_worker_request() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    events: list[SessionEvent] = []
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
        ]
    )

    def worker(_request: ChatRequest) -> ChatResponse:
        started.set()
        release.wait(timeout=2)
        finished.set()
        return _completion(text="too late")

    session = LiveSession(
        channel,
        tools=[],
        execute_tool=_no_tool,
        on_event=events.append,
        worker_fn=worker,
    )
    session.start()
    session.send_user_message("go")
    pump = threading.Thread(target=session.pump)
    pump.start()
    assert started.wait(1)

    session.interrupt()
    pump.join(timeout=1)
    session.flush_pending_intents()

    assert not pump.is_alive()
    response = next(frame for frame in channel.sent if frame["type"] == "llm_response")
    assert response["error"] == "interrupted"
    assert any(frame["type"] == "abort" for frame in channel.sent)
    assert not any(event.kind == "assistant_message" for event in events)

    release.set()
    assert finished.wait(1)


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
    session.send_user_message("go")
    events.clear()
    session.interrupt()  # user hits Stop while the submit is racing
    _drain(session)
    # The abort was sent; the racing submit is answered but NOT surfaced as a submit event.
    assert not any(e.kind == "submit" for e in events)
    assert any(f["type"] == "abort" for f in channel.sent)
    resp = next(f for f in channel.sent if f["type"] == "tool_response")
    assert resp["content"] == "submitted"  # runner still gets a response (no hang)


def test_submit_after_state_boundary_is_not_suppressed() -> None:
    """A fresh turn's submit is emitted normally after the aborted turn ended (state boundary)."""
    aborted_state: JsonObject = {"type": "state", "status": "idle", "reason": "aborted"}
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            aborted_state,
            {"type": "tool_request", "req_id": 1, "name": "submit", "arguments": {"answer": "y"}},
        ]
    )
    events: list[SessionEvent] = []
    session = LiveSession(channel, tools=[SUBMIT], execute_tool=_no_tool, on_event=events.append)
    session.start()
    first_message_id = session.send_user_message("first")
    aborted_state["msg_id"] = first_message_id
    session.interrupt()
    events.clear()
    session.pump(timeout=0)  # consume the aborted turn's idle boundary
    session.send_user_message("second")
    _drain(session)
    assert any(e.kind == "submit" for e in events)


def test_stale_submit_after_a_quick_next_message_is_still_suppressed() -> None:
    """Hold the next message until idle while suppressing the cancelled turn's stale submit."""
    aborted_state: JsonObject = {"type": "state", "status": "idle", "reason": "aborted"}
    channel = ScriptedChannel(
        [
            {"type": "state", "status": "idle"},
            {"type": "tool_request", "req_id": 1, "name": "submit", "arguments": {"answer": "x"}},
            aborted_state,
            {"type": "state", "status": "running"},
        ]
    )
    events: list[SessionEvent] = []
    session = LiveSession(channel, tools=[SUBMIT], execute_tool=_no_tool, on_event=events.append)
    session.start()
    first_message_id = session.send_user_message("first")
    aborted_state["msg_id"] = first_message_id
    session.interrupt()
    session.send_user_message("do the next thing")  # queued before the stale submit is read
    events.clear()

    session.pump(timeout=0)  # suppress the stale submit; the next instruction remains held
    sent_messages = [
        str(frame["text"]) for frame in channel.sent if frame["type"] == "user_message"
    ]
    assert sent_messages == ["first"]

    session.pump(timeout=0)  # idle is the boundary that releases the next instruction
    sent_messages = [
        str(frame["text"]) for frame in channel.sent if frame["type"] == "user_message"
    ]
    assert sent_messages == ["first", "do the next thing"]

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
    session.send_user_message("go")
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
    session.send_user_message("go")
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
