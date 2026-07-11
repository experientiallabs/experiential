# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for `wmh session start`: the jailed local executor, the credential state
machine (proxy / local-provider / baseline), and the driver's teardown wiring."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import typer
from llm_waterfall.types import ChatChoice, ChatMessage, ChatRequest, ChatResponse, ChatUsage

import wmh.cli.agent_session as mod
from wmh.harness.live_session import SessionEvent
from wmh.platform.credentials import PlatformCredentials

if TYPE_CHECKING:
    from collections.abc import Callable


def _noop_emit(_stream: str, _chunk: str) -> None:
    """A do-nothing output sink for executor calls that ignore streaming."""


# -- LocalToolExecutor -----------------------------------------------------------------------------


def test_executor_reads_writes_and_jails(tmp_path: Path) -> None:
    """read/write hit the jail; a traversal or absolute path outside it is a clean error."""
    executor = mod.LocalToolExecutor(tmp_path)
    emit = _noop_emit

    wrote = executor("write_file", {"path": "sub/a.txt", "content": "hi"}, emit)
    assert not wrote.is_error
    assert (tmp_path / "sub" / "a.txt").read_text(encoding="utf-8") == "hi"

    read = executor("read_file", {"path": "sub/a.txt"}, emit)
    assert read.content == "hi"

    escaped = executor("read_file", {"path": "../../etc/passwd"}, emit)
    assert escaped.is_error
    assert "escapes" in escaped.content

    absolute = executor("write_file", {"path": "/tmp/evil.txt", "content": "x"}, emit)
    assert absolute.is_error


def test_executor_bash_runs_in_jail_and_reports_exit(tmp_path: Path) -> None:
    """bash runs in the jail root, streams output, and surfaces a non-zero exit."""
    executor = mod.LocalToolExecutor(tmp_path)
    chunks: list[tuple[str, str]] = []

    ok = executor("bash", {"command": "pwd && echo hello"}, lambda s, c: chunks.append((s, c)))
    assert not ok.is_error
    assert str(tmp_path.resolve()) in ok.content
    assert "hello" in ok.content
    assert any(stream == "stdout" for stream, _ in chunks)

    failed = executor("bash", {"command": "exit 3"}, lambda _s, _c: None)
    assert failed.is_error
    assert "[exit 3]" in failed.content


def test_executor_caps_large_output(tmp_path: Path) -> None:
    """A read larger than the cap is truncated with a marker."""
    big = "x" * (mod._TOOL_OUTPUT_CAP + 500)
    (tmp_path / "big.txt").write_text(big, encoding="utf-8")
    executor = mod.LocalToolExecutor(tmp_path)

    result = executor("read_file", {"path": "big.txt"}, lambda _s, _c: None)
    assert result.truncated
    assert "chars truncated" in result.content


def test_unknown_tool_is_an_error(tmp_path: Path) -> None:
    """An unrecognized tool name is a non-crashing error observation."""
    result = mod.LocalToolExecutor(tmp_path)("nope", {}, lambda _s, _c: None)
    assert result.is_error
    assert "not available" in result.content


# -- credential state machine (_build_driver) ------------------------------------------------------


class _FakeProvider:
    """A minimal ToolCallingProvider stand-in."""

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Return an empty response (never actually called in these tests)."""
        _ = request
        return ChatResponse(choices=[])


class _FakeClient:
    """Records proxy calls; serves a baseline champion + a created session."""

    def __init__(self) -> None:
        self.worker_calls: list[ChatRequest] = []
        self.created: list[str] = []

    def fetch_champion_harness(self, agent_id: str) -> object:
        _ = agent_id
        doc = mod.HarnessDoc.baseline("champ").model_dump(mode="json")
        return type("Champ", (), {"doc": doc})()

    def create_local_session(self, agent_id: str, *, title: str | None = None) -> object:
        _ = title
        self.created.append(agent_id)
        return type("Sess", (), {"id": "sess-1"})()

    def complete_worker(
        self, agent_id: str, session_id: str, request: ChatRequest
    ) -> ChatResponse:
        _ = agent_id, session_id
        self.worker_calls.append(request)
        return ChatResponse(
            choices=[ChatChoice(message=ChatMessage(role="assistant", content="ok"))],
            usage=ChatUsage(prompt_tokens=1, completion_tokens=1),
        )


def _patch_local_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "get_provider", lambda _config: _FakeProvider())


def test_build_driver_not_logged_in_runs_baseline_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """No login + no agent: a pi-node baseline runs locally, unrecorded."""
    monkeypatch.setattr(mod, "load_credentials", PlatformCredentials)
    _patch_local_provider(monkeypatch)

    driver = mod._build_driver(
        agent=None,
        jail_root=Path.cwd(),
        local_provider=False,
        provider=None,
        model=None,
        instruction=None,
    )
    assert driver._recorder is None
    assert driver._worker_fn is None
    assert isinstance(driver._provider, _FakeProvider)
    assert driver._doc.runtime_kind() == "pi-node"


def test_build_driver_logged_in_default_uses_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Logged in + agent: the worker is the platform proxy and the session is recorded."""
    creds = PlatformCredentials(api_url="https://api.test", token="xpl_test")
    monkeypatch.setattr(mod, "load_credentials", lambda: creds)
    client = _FakeClient()
    monkeypatch.setattr(mod, "PlatformClient", lambda *_a, **_k: client)

    driver = mod._build_driver(
        agent="a1",
        jail_root=Path.cwd(),
        local_provider=False,
        provider=None,
        model=None,
        instruction=None,
    )
    assert driver._provider is None
    assert driver._worker_fn is not None
    assert driver._recorder is not None
    assert client.created == ["a1"]

    driver._worker_fn(ChatRequest(messages=[ChatMessage(role="user", content="hi")]))
    assert len(client.worker_calls) == 1


def test_build_driver_local_provider_still_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """Logged in + --local-provider: worker runs locally but the session is still recorded."""
    creds = PlatformCredentials(api_url="https://api.test", token="xpl_test")
    monkeypatch.setattr(mod, "load_credentials", lambda: creds)
    monkeypatch.setattr(mod, "PlatformClient", lambda *_a, **_k: _FakeClient())
    _patch_local_provider(monkeypatch)

    driver = mod._build_driver(
        agent="a1",
        jail_root=Path.cwd(),
        local_provider=True,
        provider=None,
        model=None,
        instruction=None,
    )
    assert isinstance(driver._provider, _FakeProvider)
    assert driver._worker_fn is None
    assert driver._recorder is not None


def test_build_driver_agent_without_login_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Naming an agent while logged out is a clear parameter error."""
    monkeypatch.setattr(mod, "load_credentials", PlatformCredentials)
    with pytest.raises(typer.BadParameter):
        mod._build_driver(
            agent="a1",
            jail_root=Path.cwd(),
            local_provider=False,
            provider=None,
            model=None,
            instruction=None,
        )


# -- driver orchestration --------------------------------------------------------------------------


class _FakeSandbox:
    """Records timeout extensions + kills."""

    def __init__(self) -> None:
        self.sandbox_id = "sbx-local"
        self.timeouts: list[int] = []
        self.kills = 0

    def set_timeout(self, timeout: int) -> None:
        self.timeouts.append(timeout)

    def kill(self) -> None:
        self.kills += 1


class _FakeLiveSession:
    """Emits one state event on the first pump, then closes."""

    def __init__(
        self, _channel: object, *, on_event: Callable[[SessionEvent], None], **_: object
    ) -> None:
        self._on_event = on_event
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def start(self, hello_timeout: float = 60.0) -> None:
        _ = hello_timeout

    def send_user_message(self, text: str) -> str:
        _ = text
        return "msg-1"

    def interrupt(self, reason: str = "user_interrupt") -> None:
        _ = reason

    def end(self) -> None:
        self._closed = True

    def pump(self, timeout: float = 0.2) -> bool:
        _ = timeout
        self._on_event(SessionEvent(kind="state", payload={"status": "idle"}))
        self._closed = True
        return False


class _FakeReader:
    """A stdin reader that never touches stdin."""

    def __init__(self, _session: object) -> None:
        pass

    def start(self) -> None:
        pass


def _patch_driver_boundaries(monkeypatch: pytest.MonkeyPatch, sandbox: _FakeSandbox) -> None:
    monkeypatch.setattr(mod, "default_sandbox_factory", lambda **_: lambda: sandbox)
    monkeypatch.setattr(mod, "start_live_runner", lambda *_a, **_k: object())
    monkeypatch.setattr(mod, "LiveSession", _FakeLiveSession)
    monkeypatch.setattr(mod, "StdinCommandReader", _FakeReader)


def test_driver_boots_loops_and_kills_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """The driver boots, runs the pump loop to close, and always kills the sandbox."""
    sandbox = _FakeSandbox()
    _patch_driver_boundaries(monkeypatch, sandbox)

    mod.LocalLiveDriver(
        jail_root=Path.cwd(),
        doc=mod.HarnessDoc.baseline("t"),
        provider=_FakeProvider(),
        worker_fn=None,
        recorder=None,
        instruction=None,
    ).run()

    assert sandbox.kills == 1


def test_driver_reports_finish_to_recorder(monkeypatch: pytest.MonkeyPatch) -> None:
    """When recording, the driver posts a terminal finish on teardown."""
    sandbox = _FakeSandbox()
    _patch_driver_boundaries(monkeypatch, sandbox)
    finished: list[str] = []

    class _Recorder:
        def flush(self) -> None: ...
        def record(self, event: SessionEvent) -> None:
            _ = event

        def finish(self, *, ended_reason: str, sandbox_seconds: int, error: str | None) -> None:
            _ = sandbox_seconds, error
            finished.append(ended_reason)

    mod.LocalLiveDriver(
        jail_root=Path.cwd(),
        doc=mod.HarnessDoc.baseline("t"),
        provider=_FakeProvider(),
        worker_fn=None,
        recorder=cast("mod.SessionRecorder", _Recorder()),
        instruction=None,
    ).run()

    assert finished == ["user_ended"]
    assert sandbox.kills == 1
