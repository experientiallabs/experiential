# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for ``wmh run`` target dispatch and local execution boundaries."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import typer
from llm_waterfall.types import ChatChoice, ChatMessage, ChatRequest, ChatResponse, ChatUsage
from typer.testing import CliRunner

import wmh.cli.agent_session as mod
from wmh.cli import app
from wmh.harness.live_session import SessionEvent
from wmh.harness.workspace_patch import build_workspace_patch
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
    """Records hosted target resolution and built-in Pi proxy calls."""

    def __init__(self) -> None:
        self.worker_calls: list[ChatRequest] = []
        self.target_kind = "agent"
        self.closed = False
        self.local_pi_created: list[str] = []
        self.local_pi_finished: list[str] = []

    def resolve_run_target(self, target_id: str) -> object:
        return type(
            "Target",
            (),
            {"id": target_id, "kind": self.target_kind, "name": "remote-target"},
        )()

    def create_local_pi_run(self, org_id: str) -> object:
        self.local_pi_created.append(org_id)
        return type("Run", (), {"id": "run-1"})()

    def complete_local_pi_worker(
        self, org_id: str, run_id: str, request: ChatRequest
    ) -> ChatResponse:
        _ = org_id, run_id
        self.worker_calls.append(request)
        return ChatResponse(
            choices=[ChatChoice(message=ChatMessage(role="assistant", content="ok"))],
            usage=ChatUsage(prompt_tokens=1, completion_tokens=1),
        )

    def finish_local_pi_run(
        self,
        org_id: str,
        run_id: str,
        *,
        status: str,
        ended_reason: str,
        error: str | None = None,
    ) -> None:
        _ = org_id, status, ended_reason, error
        self.local_pi_finished.append(run_id)

    def close(self) -> None:
        self.closed = True


def _patch_local_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "get_provider", lambda _config: _FakeProvider())


def test_build_driver_not_logged_in_runs_baseline_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """No login + no agent: a pi-node baseline runs locally, unrecorded."""
    monkeypatch.setattr(mod, "load_credentials", PlatformCredentials)
    _patch_local_provider(monkeypatch)

    driver = mod._build_driver(
        target=None,
        jail_root=Path.cwd(),
        provider=None,
        model=None,
        task=None,
    )
    assert isinstance(driver, mod.LocalLiveDriver)
    assert driver._recorder is None
    assert driver._worker_fn is None
    assert isinstance(driver._provider, _FakeProvider)
    assert driver._doc.runtime_kind() == "pi-node"


def test_build_driver_logged_in_agent_uses_hosted_e2b_without_local_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Logged in + agent defaults to platform-owned E2B without local files."""
    creds = PlatformCredentials(api_url="https://api.test", token="xpl_test")
    monkeypatch.setattr(mod, "load_credentials", lambda: creds)
    client = _FakeClient()
    monkeypatch.setattr(mod, "PlatformClient", lambda *_a, **_k: client)

    driver = mod._build_driver(
        target="a1",
        jail_root=None,
        provider=None,
        model=None,
        task=None,
    )
    assert isinstance(driver, mod.RemoteAgentDriver)
    assert driver._target_id == "a1"
    assert driver._jail is None


def test_run_upload_dir_is_explicit_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hosted run receives no local path unless -u/--upload-dir is present."""
    roots: list[Path | None] = []

    class _Driver:
        def run(self) -> None:
            pass

    def build_driver(**kwargs: object) -> _Driver:
        roots.append(cast("Path | None", kwargs["jail_root"]))
        return _Driver()

    monkeypatch.setattr(mod, "_build_driver", build_driver)
    runner = CliRunner()

    plain = runner.invoke(app, ["run", "agent-1"])
    uploaded = runner.invoke(app, ["run", "agent-1", "-u", str(tmp_path)])

    assert plain.exit_code == 0
    assert uploaded.exit_code == 0
    assert roots == [None, tmp_path.resolve()]


def test_build_driver_logged_in_builtin_pi_uses_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Logged-in bare run needs no local provider credentials."""
    creds = PlatformCredentials(api_url="https://api.test", token="xpl_test", default_org="org-1")
    monkeypatch.setattr(mod, "load_credentials", lambda: creds)
    client = _FakeClient()
    monkeypatch.setattr(mod, "PlatformClient", lambda *_a, **_k: client)

    driver = mod._build_driver(
        target=None,
        jail_root=Path.cwd(),
        provider=None,
        model=None,
        task=None,
    )
    assert isinstance(driver, mod.LocalLiveDriver)
    assert driver._provider is None
    assert driver._worker_fn is not None
    assert isinstance(driver._recorder, mod.LocalPiRunRecorder)
    assert client.local_pi_created == ["org-1"]

    driver._worker_fn(ChatRequest(messages=[ChatMessage(role="user", content="hi")]))
    assert len(client.worker_calls) == 1


def test_platform_target_rejects_local_provider_before_creating_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider overrides never create an orphan platform run."""
    creds = PlatformCredentials(api_url="https://api.test", token="xpl_test")
    monkeypatch.setattr(mod, "load_credentials", lambda: creds)
    client = _FakeClient()
    monkeypatch.setattr(mod, "PlatformClient", lambda *_a, **_k: client)

    with pytest.raises(typer.BadParameter, match="platform credentials"):
        mod._build_driver(
            target="a1",
            jail_root=Path.cwd(),
            provider="bedrock",
            model=None,
            task=None,
        )


def test_hosted_agent_does_not_prompt_for_local_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The E2B agent path never presents the bare harness's local-shell warning."""
    creds = PlatformCredentials(api_url="https://api.test", token="xpl_test")
    monkeypatch.setattr(mod, "load_credentials", lambda: creds)
    client = _FakeClient()
    monkeypatch.setattr(mod, "PlatformClient", lambda *_a, **_k: client)

    prompted: list[bool] = []
    driver = mod._build_driver(
        target="a1",
        jail_root=Path.cwd(),
        provider=None,
        model=None,
        task=None,
        confirm_local=lambda: prompted.append(True),
    )
    assert isinstance(driver, mod.RemoteAgentDriver)
    assert prompted == []
    assert not client.closed


def test_build_driver_world_model_uses_hosted_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolved world-model id never boots a local agent process."""
    creds = PlatformCredentials(api_url="https://api.test", token="xpl_test")
    monkeypatch.setattr(mod, "load_credentials", lambda: creds)
    client = _FakeClient()
    client.target_kind = "world_model"
    monkeypatch.setattr(mod, "PlatformClient", lambda *_a, **_k: client)

    driver = mod._build_driver(
        target="wm-1",
        jail_root=None,
        provider=None,
        model=None,
        task="help the customer",
    )
    assert isinstance(driver, mod.RemoteWorldModelDriver)


def test_build_driver_target_without_login_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Naming any platform target while logged out is a clear parameter error."""
    monkeypatch.setattr(mod, "load_credentials", PlatformCredentials)
    with pytest.raises(typer.BadParameter):
        mod._build_driver(
            target="a1",
            jail_root=Path.cwd(),
            provider=None,
            model=None,
            task=None,
        )


# -- driver orchestration --------------------------------------------------------------------------


class _FakeChannel:
    """Records local runner teardown."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


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

    @property
    def status(self) -> str:
        return "ended" if self._closed else "idle"

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


def _patch_driver_boundaries(monkeypatch: pytest.MonkeyPatch, channel: _FakeChannel) -> None:
    monkeypatch.setattr(mod, "start_local_live_runner", lambda: channel)
    monkeypatch.setattr(mod, "LiveSession", _FakeLiveSession)
    monkeypatch.setattr(mod, "StdinCommandReader", _FakeReader)


def test_driver_boots_loops_and_closes_local_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The driver boots, runs the pump loop, and always closes its local process."""
    channel = _FakeChannel()
    _patch_driver_boundaries(monkeypatch, channel)

    mod.LocalLiveDriver(
        jail_root=Path.cwd(),
        doc=mod.HarnessDoc.baseline("t"),
        provider=_FakeProvider(),
        worker_fn=None,
        recorder=None,
        instruction=None,
    ).run()

    assert channel.closed


def test_driver_reports_finish_to_recorder(monkeypatch: pytest.MonkeyPatch) -> None:
    """When recording, the driver posts a terminal finish on teardown."""
    channel = _FakeChannel()
    _patch_driver_boundaries(monkeypatch, channel)
    finished: list[str] = []

    class _Recorder:
        def flush(self) -> None: ...
        def record(self, event: SessionEvent) -> None:
            _ = event

        def finish(self, *, ended_reason: str, error: str | None) -> None:
            _ = error
            finished.append(ended_reason)

    mod.LocalLiveDriver(
        jail_root=Path.cwd(),
        doc=mod.HarnessDoc.baseline("t"),
        provider=_FakeProvider(),
        worker_fn=None,
        recorder=cast("mod.RunRecorder", _Recorder()),
        instruction=None,
    ).run()

    assert finished == ["user_ended"]
    assert channel.closed


def test_stdin_eof_is_reported_without_aborting_the_opening_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closed stdin is a driver signal, not an immediate local or hosted end command."""
    monkeypatch.setattr(mod.sys, "stdin", io.StringIO(""))

    class _Session:
        closed = False
        ended = 0

        def end(self) -> None:
            self.ended += 1

    local_session = _Session()
    local_reader = mod.StdinCommandReader(cast("mod.LiveSession", local_session))
    local_reader.run()
    assert local_reader.eof.is_set()
    assert local_session.ended == 0

    class _Client:
        posted: list[str] = []

        def post_agent_session_command(
            self, _agent_id: str, _session_id: str, kind: str, *, text: str | None = None
        ) -> None:
            _ = text
            self.posted.append(kind)

    client = _Client()
    remote_reader = mod.RemoteAgentCommandReader(
        cast("mod.PlatformClient", client), "agent-1", "session-1"
    )
    remote_reader.run()
    assert remote_reader.eof.is_set()
    assert client.posted == []


def test_local_driver_returns_nonzero_when_the_runner_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner-side failure cannot look like a successful local CLI run."""
    channel = _FakeChannel()

    class _FailedSession(_FakeLiveSession):
        @property
        def status(self) -> str:
            return "failed"

        def pump(self, timeout: float = 0.2) -> bool:
            _ = timeout
            self._closed = True
            return False

    monkeypatch.setattr(mod, "start_local_live_runner", lambda: channel)
    monkeypatch.setattr(mod, "LiveSession", _FailedSession)
    monkeypatch.setattr(mod, "StdinCommandReader", _FakeReader)

    with pytest.raises(typer.Exit) as raised:
        mod.LocalLiveDriver(
            jail_root=Path.cwd(),
            doc=mod.HarnessDoc.baseline("t"),
            provider=_FakeProvider(),
            worker_fn=None,
            recorder=None,
            instruction="do work",
        ).run()

    assert raised.value.exit_code == 1
    assert channel.closed


def test_conflicted_local_patch_does_not_advance_the_synchronized_base(tmp_path: Path) -> None:
    """A rejected same-file edit remains outside the platform-accepted snapshot."""
    path = tmp_path / "answer.txt"
    path.write_text("before", encoding="utf-8")
    initial = mod.snapshot_workspace(tmp_path)
    path.write_text("local", encoding="utf-8")

    class _Client:
        def upload_agent_workspace_patch(self, *_args: object, **_kwargs: object) -> object:
            return type("Result", (), {"conflicts": ("answer.txt",)})()

    driver = mod.RemoteAgentDriver(
        cast("mod.PlatformClient", _Client()), "agent-1", "Agent", tmp_path, "work"
    )

    synchronized = driver._push_local_patch("session-1", initial)

    assert synchronized is initial
    assert driver._live_conflicts == {"answer.txt"}


def test_remote_agent_driver_syncs_final_e2b_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hosted driver downloads, applies, acknowledges, and closes after terminal state."""
    (tmp_path / "answer.txt").write_text("before", encoding="utf-8")
    buffer = io.BytesIO()
    content = b"after"
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("answer.txt")
        info.size = len(content)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(content))

    class _HostedClient:
        def __init__(self) -> None:
            self.acked: list[str] = []
            self.closed = False

        def create_agent_session(
            self, agent_id: str, *, workspace: bytes, instruction: str | None = None
        ) -> object:
            assert agent_id == "agent-1"
            assert workspace
            assert instruction == "fix it"
            return type("Session", (), {"id": "session-1"})()

        def list_agent_session_events(
            self, agent_id: str, session_id: str, *, after: int
        ) -> object:
            _ = agent_id, session_id, after
            return type("Page", (), {"events": [], "last_seq": 0, "status": "ended"})()

        def get_agent_session(self, agent_id: str, session_id: str) -> object:
            _ = agent_id, session_id
            return type("Session", (), {"status": "ended", "error": None})()

        def download_agent_workspace(self, agent_id: str, session_id: str) -> bytes:
            _ = agent_id, session_id
            return buffer.getvalue()

        def acknowledge_agent_workspace(self, agent_id: str, session_id: str) -> None:
            _ = agent_id
            self.acked.append(session_id)

        def close(self) -> None:
            self.closed = True

    class _NoReader:
        def __init__(self, *_args: object) -> None:
            pass

        def start(self) -> None:
            pass

    client = _HostedClient()
    monkeypatch.setattr(mod, "RemoteAgentCommandReader", _NoReader)

    mod.RemoteAgentDriver(
        cast("mod.PlatformClient", client), "agent-1", "Agent", tmp_path, "fix it"
    ).run()

    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "after"
    assert client.acked == ["session-1"]
    assert client.closed


def test_failed_hosted_session_is_not_hidden_by_workspace_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Session failure remains the primary exit even when final sync needs recovery."""
    (tmp_path / "answer.txt").write_text("before", encoding="utf-8")
    final = mod.snapshot_workspace(tmp_path).archive

    class _HostedClient:
        def __init__(self) -> None:
            self.acked = False
            self.closed = False

        def create_agent_session(self, *_args: object, **_kwargs: object) -> object:
            return type("Session", (), {"id": "session-1"})()

        def list_agent_session_events(self, *_args: object, **_kwargs: object) -> object:
            return type("Page", (), {"events": [], "last_seq": 0, "status": "failed"})()

        def get_agent_session(self, *_args: object, **_kwargs: object) -> object:
            return type("Session", (), {"status": "failed", "error": "runner crashed"})()

        def download_agent_workspace(self, *_args: object, **_kwargs: object) -> bytes:
            return final

        def acknowledge_agent_workspace(self, *_args: object, **_kwargs: object) -> None:
            self.acked = True

        def close(self) -> None:
            self.closed = True

    class _NoReader:
        def __init__(self, *_args: object) -> None:
            pass

        def start(self) -> None:
            pass

    client = _HostedClient()
    monkeypatch.setattr(mod, "RemoteAgentCommandReader", _NoReader)
    monkeypatch.setattr(
        mod,
        "sync_workspace",
        lambda *_args, **_kwargs: type(
            "Result", (), {"applied": (), "conflicts": ("answer.txt",)}
        )(),
    )

    with pytest.raises(typer.Exit) as raised:
        mod.RemoteAgentDriver(
            cast("mod.PlatformClient", client), "agent-1", "Agent", tmp_path, "fix it"
        ).run()

    assert raised.value.exit_code == 1
    assert client.acked
    assert client.closed
    assert (tmp_path / ".wmh-conflicts" / "session-1.tar.gz").is_file()


def test_remote_agent_driver_without_upload_never_reads_local_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A default hosted run skips snapshot, upload, patches, and final download."""
    (tmp_path / "private.txt").write_text("do not upload", encoding="utf-8")

    class _HostedClient:
        def __init__(self) -> None:
            self.created_workspaces: list[bytes | None] = []
            self.closed = False

        def create_agent_session(
            self,
            agent_id: str,
            *,
            workspace: bytes | None,
            instruction: str | None = None,
        ) -> object:
            assert agent_id == "agent-1"
            assert instruction == "work remotely"
            self.created_workspaces.append(workspace)
            return type("Session", (), {"id": "session-1"})()

        def list_agent_session_events(self, *_args: object, **_kwargs: object) -> object:
            return type("Page", (), {"events": [], "last_seq": 0, "status": "ended"})()

        def get_agent_session(self, *_args: object) -> object:
            return type("Session", (), {"status": "ended", "error": None})()

        def download_agent_workspace(self, *_args: object) -> bytes:
            pytest.fail("a run without --upload-dir must not download a workspace")

        def close(self) -> None:
            self.closed = True

    class _NoReader:
        def __init__(self, *_args: object) -> None:
            pass

        def start(self) -> None:
            pass

    client = _HostedClient()
    monkeypatch.setattr(mod, "RemoteAgentCommandReader", _NoReader)

    mod.RemoteAgentDriver(
        cast("mod.PlatformClient", client), "agent-1", "Agent", None, "work remotely"
    ).run()

    assert client.created_workspaces == [None]
    assert client.closed


def test_remote_agent_driver_applies_live_workspace_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workspace_patch event updates the local directory before session end."""
    (tmp_path / "answer.txt").write_text("before", encoding="utf-8")
    initial = mod.snapshot_workspace(tmp_path)
    final_buffer = io.BytesIO()
    with tarfile.open(fileobj=final_buffer, mode="w:gz") as archive:
        body = b"during"
        info = tarfile.TarInfo("answer.txt")
        info.size = len(body)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(body))
    patch = build_workspace_patch(initial.archive, final_buffer.getvalue())
    assert patch is not None

    class _HostedClient:
        def __init__(self) -> None:
            self.patch_acks: list[str] = []
            self.closed = False

        def create_agent_session(self, *_args: object, **_kwargs: object) -> object:
            return type("Session", (), {"id": "session-1"})()

        def list_agent_session_events(self, *_args: object, **_kwargs: object) -> object:
            event = type(
                "Event",
                (),
                {"kind": "workspace_patch", "payload": {"revision": "patch-1"}},
            )()
            return type("Page", (), {"events": [event], "last_seq": 1, "status": "ended"})()

        def download_agent_workspace_patch(
            self, _agent_id: str, _session_id: str, revision: str
        ) -> bytes:
            assert revision == "patch-1"
            return patch

        def acknowledge_agent_workspace_patch(
            self, _agent_id: str, _session_id: str, revision: str
        ) -> None:
            self.patch_acks.append(revision)

        def get_agent_session(self, *_args: object) -> object:
            return type("Session", (), {"status": "ended", "error": None})()

        def download_agent_workspace(self, *_args: object) -> bytes:
            return final_buffer.getvalue()

        def acknowledge_agent_workspace(self, *_args: object) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    class _NoReader:
        def __init__(self, *_args: object) -> None:
            pass

        def start(self) -> None:
            pass

    client = _HostedClient()
    monkeypatch.setattr(mod, "RemoteAgentCommandReader", _NoReader)

    mod.RemoteAgentDriver(
        cast("mod.PlatformClient", client), "agent-1", "Agent", tmp_path, None
    ).run()

    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "during"
    assert client.patch_acks == ["patch-1"]
    assert client.closed
