# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for ``wmh run`` target dispatch and local execution boundaries."""

from __future__ import annotations

import io
import tarfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import typer
from llm_waterfall.types import ChatChoice, ChatMessage, ChatRequest, ChatResponse, ChatUsage
from typer.testing import CliRunner

import wmh.cli.agent_session as mod
import wmh.cli.hosted_session as hosted_mod
from wmh.cli import app
from wmh.cli.session_state import DetachedSessionState, SessionStateError, SessionStateStore
from wmh.cli.workspace_sync import snapshot_from_archive
from wmh.config.settings import ModelRole, ModelsSettings, ProjectSettings, save_settings
from wmh.harness.live_session import SessionEvent
from wmh.harness.workspace_patch import build_workspace_patch
from wmh.platform.client import PlatformError
from wmh.platform.credentials import PlatformCredentials
from wmh.providers.base import ProviderKind

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


def test_build_driver_uses_configured_local_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    save_settings(
        ProjectSettings(
            models=ModelsSettings(worker=ModelRole(provider="openai", model="gpt-5.4-mini"))
        ),
        tmp_path / ".wmh",
    )
    monkeypatch.setattr(mod, "load_credentials", PlatformCredentials)
    configs = []

    def get_provider(config: object) -> _FakeProvider:
        configs.append(config)
        return _FakeProvider()

    monkeypatch.setattr(mod, "get_provider", get_provider)

    driver = mod._build_driver(
        target=None,
        jail_root=tmp_path,
        provider=None,
        model=None,
        task=None,
    )

    assert isinstance(driver, mod.LocalLiveDriver)
    [config] = configs
    assert config.kind is ProviderKind.OPENAI
    assert config.model == "gpt-5.4-mini"


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
        cast("mod.PlatformClient", client),
        "agent-1",
        "Agent",
        tmp_path,
        "fix it",
        credentials=PlatformCredentials(api_url="https://api.test", token="xpl_test"),
        state_store=SessionStateStore(tmp_path / ".state"),
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
        hosted_mod,
        "sync_workspace",
        lambda *_args, **_kwargs: type(
            "Result", (), {"applied": (), "conflicts": ("answer.txt",)}
        )(),
    )

    with pytest.raises(typer.Exit) as raised:
        mod.RemoteAgentDriver(
            cast("mod.PlatformClient", client),
            "agent-1",
            "Agent",
            tmp_path,
            "fix it",
            credentials=PlatformCredentials(api_url="https://api.test", token="xpl_test"),
            state_store=SessionStateStore(tmp_path / ".state"),
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
        cast("mod.PlatformClient", client),
        "agent-1",
        "Agent",
        None,
        "work remotely",
        credentials=PlatformCredentials(api_url="https://api.test", token="xpl_test"),
        state_store=SessionStateStore(tmp_path / ".state"),
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
                {"seq": 1, "kind": "workspace_patch", "payload": {"revision": "patch-1"}},
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
        cast("mod.PlatformClient", client),
        "agent-1",
        "Agent",
        tmp_path,
        None,
        credentials=PlatformCredentials(api_url="https://api.test", token="xpl_test"),
        state_store=SessionStateStore(tmp_path / ".state"),
    ).run()

    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "during"
    assert client.patch_acks == ["patch-1"]
    assert client.closed


# -- detached session flags -------------------------------------------------------------------


def test_run_session_flag_combinations_are_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The detached options keep one unambiguous meaning per invocation."""
    monkeypatch.setattr(
        mod, "_build_session_command_driver", lambda **_kw: pytest.fail("must not build")
    )
    monkeypatch.setattr(mod, "_build_driver", lambda **_kw: pytest.fail("must not build"))
    runner = CliRunner()

    exclusive = runner.invoke(app, ["run", "-s", "hi", "-a"])
    assert exclusive.exit_code != 0
    assert "mutually exclusive" in exclusive.output

    dangling_session = runner.invoke(app, ["run", "--session", "sess-1"])
    assert dangling_session.exit_code != 0
    assert "--session" in dangling_session.output

    send_with_target = runner.invoke(app, ["run", "agent-1", "-s", "hi"])
    assert send_with_target.exit_code != 0

    detach_without_target = runner.invoke(app, ["run", "-d"])
    assert detach_without_target.exit_code != 0
    assert "agent id" in detach_without_target.output

    send_with_upload = runner.invoke(app, ["run", "-s", "hi", "-u", "."])
    assert send_with_upload.exit_code != 0

    send_with_task = runner.invoke(app, ["run", "-s", "hi", "--task", "x"])
    assert send_with_task.exit_code != 0

    detach_with_send = runner.invoke(app, ["run", "agent-1", "-d", "-s", "hi"])
    assert detach_with_send.exit_code != 0


def test_run_dispatches_session_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """--send/--attach/--end route to the detached command driver, unified in run."""
    captured: list[dict[str, str | None]] = []

    class _Driver:
        def run(self) -> None:
            pass

    def build(**kwargs: str | None) -> _Driver:
        captured.append(kwargs)
        return _Driver()

    monkeypatch.setattr(mod, "_build_session_command_driver", build)
    runner = CliRunner()

    assert runner.invoke(app, ["run", "--send", "do it"]).exit_code == 0
    assert runner.invoke(app, ["run", "-a", "--session", "sess-2"]).exit_code == 0
    assert runner.invoke(app, ["run", "--end"]).exit_code == 0

    assert captured == [
        {"action": "send", "text": "do it", "session_override": None},
        {"action": "attach", "text": None, "session_override": "sess-2"},
        {"action": "end", "text": None, "session_override": None},
    ]


def test_build_driver_detach_returns_start_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """--detach on an agent id builds the detached start driver, not the streamer."""
    creds = PlatformCredentials(api_url="https://api.test", token="xpl_test")
    monkeypatch.setattr(mod, "load_credentials", lambda: creds)
    client = _FakeClient()
    monkeypatch.setattr(mod, "PlatformClient", lambda *_a, **_k: client)

    driver = mod._build_driver(
        target="a1",
        jail_root=None,
        provider=None,
        model=None,
        task="do it",
        detach=True,
    )

    assert isinstance(driver, hosted_mod.DetachedStartDriver)


def test_build_driver_detach_rejects_world_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """World-model sessions are interactive only; --detach names agents."""
    creds = PlatformCredentials(api_url="https://api.test", token="xpl_test")
    monkeypatch.setattr(mod, "load_credentials", lambda: creds)
    client = _FakeClient()
    client.target_kind = "world_model"
    monkeypatch.setattr(mod, "PlatformClient", lambda *_a, **_k: client)

    with pytest.raises(typer.BadParameter, match="agent"):
        mod._build_driver(
            target="wm-1",
            jail_root=None,
            provider=None,
            model=None,
            task=None,
            detach=True,
        )


# -- plain-run :detach promotion ---------------------------------------------------------------


class _DetachedReader:
    """A reader stub whose user immediately detaches (never EOF, never end)."""

    def __init__(self, *_args: object) -> None:
        self.eof = threading.Event()
        self.detach = threading.Event()
        self.detach.set()

    def start(self) -> None:
        pass


def _plain_driver(
    client: object,
    tmp_path: Path,
    *,
    jail_root: Path | None = None,
    task: str | None = None,
) -> tuple[mod.RemoteAgentDriver, SessionStateStore]:
    """A plain hosted driver wired to an injectable session-state store."""
    store = SessionStateStore(tmp_path / "state")
    driver = mod.RemoteAgentDriver(
        cast("mod.PlatformClient", client),
        "agent-1",
        "Agent",
        jail_root,
        task,
        credentials=PlatformCredentials(
            api_url="https://api.test", web_url="https://platform.test", token="xpl_test"
        ),
        state_store=store,
    )
    return driver, store


def test_reader_detach_skips_eof_end_and_guards_unknown_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """:detach stops reading without ending; unknown :commands never become chat."""

    class _Client:
        def __init__(self) -> None:
            self.posted: list[tuple[str, str | None]] = []

        def post_agent_session_command(
            self, _agent_id: str, _session_id: str, kind: str, *, text: str | None = None
        ) -> None:
            self.posted.append((kind, text))

    monkeypatch.setattr(mod.sys, "stdin", io.StringIO(":detach\n"))
    client = _Client()
    reader = mod.RemoteAgentCommandReader(cast("mod.PlatformClient", client), "a", "s")
    reader.run()
    assert reader.detach.is_set()
    assert not reader.eof.is_set()
    assert client.posted == []

    monkeypatch.setattr(mod.sys, "stdin", io.StringIO(":frob\nhello\n:end\n"))
    client = _Client()
    reader = mod.RemoteAgentCommandReader(cast("mod.PlatformClient", client), "a", "s")
    reader.run()
    assert client.posted == [("user_message", "hello"), ("end", None)]
    assert not reader.detach.is_set()


def test_plain_run_detach_promotes_session_without_ending(tmp_path: Path) -> None:
    """:detach in a plain run persists the current-session reference and exits."""

    class _Client:
        def __init__(self) -> None:
            self.commands: list[str] = []
            self.closed = False

        def create_agent_session(self, *_args: object, **_kwargs: object) -> object:
            return type("Session", (), {"id": "sess-1"})()

        def list_agent_session_events(self, *_args: object, **_kwargs: object) -> object:
            event = type(
                "Event", (), {"seq": 3, "kind": "assistant_message", "payload": {"text": "hi"}}
            )()
            return type("Page", (), {"events": [event], "last_seq": 3, "status": "running"})()

        def post_agent_session_command(
            self, _agent_id: str, _session_id: str, kind: str, *, text: str | None = None
        ) -> None:
            _ = text
            self.commands.append(kind)

        def download_agent_workspace(self, *_args: object) -> bytes:
            pytest.fail("a detach promotion must not run the final workspace sync")

        def close(self) -> None:
            self.closed = True

    client = _Client()
    driver, store = _plain_driver(client, tmp_path)
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(mod, "RemoteAgentCommandReader", _DetachedReader)
        driver.run()

    assert client.commands == []
    assert client.closed
    state = store.load("sess-1")
    assert state is not None
    assert state.api_url == "https://api.test"
    assert state.agent_id == "agent-1"
    assert state.session_id == "sess-1"
    assert state.cursor == 3
    assert state.workspace is None
    assert store.current_session_id() == "sess-1"


def test_plain_run_detach_with_workspace_checkpoints_without_final_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Promotion persists the synced checkpoint; no finalize, patches stay acked."""
    root = tmp_path / "work"
    root.mkdir()
    (root / "answer.txt").write_text("before", encoding="utf-8")
    base = mod.snapshot_workspace(root)
    remote = io.BytesIO()
    with tarfile.open(fileobj=remote, mode="w:gz") as archive:
        body = b"during"
        info = tarfile.TarInfo("answer.txt")
        info.size = len(body)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(body))
    patch = build_workspace_patch(base.archive, remote.getvalue())
    assert patch is not None

    class _Client:
        def __init__(self) -> None:
            self.patch_acks: list[str] = []
            self.closed = False

        def create_agent_session(self, *_args: object, **_kwargs: object) -> object:
            return type("Session", (), {"id": "sess-1"})()

        def list_agent_session_events(self, *_args: object, **_kwargs: object) -> object:
            event = type(
                "Event", (), {"seq": 1, "kind": "workspace_patch", "payload": {"revision": "p1"}}
            )()
            return type("Page", (), {"events": [event], "last_seq": 1, "status": "running"})()

        def download_agent_workspace_patch(
            self, _agent_id: str, _session_id: str, revision: str
        ) -> bytes:
            assert revision == "p1"
            return patch

        def acknowledge_agent_workspace_patch(
            self, _agent_id: str, _session_id: str, revision: str
        ) -> None:
            self.patch_acks.append(revision)

        def download_agent_workspace(self, *_args: object) -> bytes:
            pytest.fail("a detach promotion must not run the final workspace sync")

        def acknowledge_agent_workspace(self, *_args: object) -> None:
            pytest.fail("a detach promotion must not acknowledge the final workspace")

        def close(self) -> None:
            self.closed = True

    client = _Client()
    driver, store = _plain_driver(client, tmp_path, jail_root=root)
    monkeypatch.setattr(mod, "RemoteAgentCommandReader", _DetachedReader)
    driver.run()

    assert client.patch_acks == ["p1"]
    assert (root / "answer.txt").read_text(encoding="utf-8") == "during"
    state = store.load("sess-1")
    assert state is not None
    assert state.cursor == 1
    assert state.workspace is not None
    assert state.workspace.root == str(root)
    checkpoint = snapshot_from_archive(store.load_base_archive(state))
    assert checkpoint.files == mod.snapshot_workspace(root).files


def test_plain_run_detach_state_failure_names_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed promotion save still hands the user the session id."""

    class _FailingStore(SessionStateStore):
        def save(
            self, state: DetachedSessionState, *, base_archive: bytes | None = None
        ) -> DetachedSessionState:
            raise SessionStateError("disk full")

    class _Client:
        def create_agent_session(self, *_args: object, **_kwargs: object) -> object:
            return type("Session", (), {"id": "sess-1"})()

        def list_agent_session_events(self, *_args: object, **_kwargs: object) -> object:
            return type("Page", (), {"events": [], "last_seq": 0, "status": "running"})()

        def close(self) -> None:
            pass

    driver = mod.RemoteAgentDriver(
        cast("mod.PlatformClient", _Client()),
        "agent-1",
        "Agent",
        None,
        None,
        credentials=PlatformCredentials(api_url="https://api.test", token="xpl_test"),
        state_store=_FailingStore(tmp_path / "state"),
    )
    monkeypatch.setattr(mod, "RemoteAgentCommandReader", _DetachedReader)
    with pytest.raises(typer.BadParameter, match="sess-1"):
        driver.run()


def test_world_model_loop_rejects_detach(monkeypatch: pytest.MonkeyPatch) -> None:
    """:detach in a world-model REPL warns instead of stepping the model."""

    class _Client:
        def step_world_model_session(self, *_args: object, **_kwargs: object) -> object:
            pytest.fail(":detach must never reach the world model as an action")

        def close(self) -> None:
            pass

    lines = iter([":detach", ":quit"])
    monkeypatch.setattr(mod._console, "input", lambda *_a, **_k: next(lines))
    driver = mod.RemoteWorldModelDriver(
        cast("mod.PlatformClient", _Client()), "wm-1", "Model", None
    )

    driver._loop("sess-1")


def test_reader_keeps_reading_after_a_failed_steer_post(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A transient post failure warns and keeps the session attached, never ends it."""

    class _FlakyClient:
        def __init__(self) -> None:
            self.posted: list[tuple[str, str | None]] = []
            self.failed_once = False

        def post_agent_session_command(
            self, _agent_id: str, _session_id: str, kind: str, *, text: str | None = None
        ) -> None:
            if kind == "user_message" and not self.failed_once:
                self.failed_once = True
                raise PlatformError("backend unavailable", status_code=503)
            self.posted.append((kind, text))

    monkeypatch.setattr(mod.sys, "stdin", io.StringIO("hello\n:stop\n"))
    client = _FlakyClient()
    reader = mod.RemoteAgentCommandReader(cast("mod.PlatformClient", client), "a", "s")

    reader.run()

    # The failed steer was reported, the reader kept going, and only true
    # stdin EOF set the eof flag; nothing ended or detached the session.
    assert client.posted == [("interrupt", None)]
    assert reader.eof.is_set()
    assert not reader.detach.is_set()
    assert "failed" in capsys.readouterr().out


def test_second_interrupt_during_the_plain_handler_still_ends(tmp_path: Path) -> None:
    """A Ctrl-C landing inside the plain-run interrupt handler escalates to end."""

    class _Client:
        def __init__(self) -> None:
            self.commands: list[str] = []
            self.polls = 0
            self.closed = False

        def create_agent_session(self, *_args: object, **_kwargs: object) -> object:
            return type("Session", (), {"id": "sess-1"})()

        def list_agent_session_events(self, *_args: object, **_kwargs: object) -> object:
            self.polls += 1
            if self.polls == 1:
                raise KeyboardInterrupt
            return type("Page", (), {"events": [], "last_seq": 0, "status": "ended"})()

        def post_agent_session_command(
            self, _agent_id: str, _session_id: str, kind: str, *, text: str | None = None
        ) -> None:
            _ = text
            self.commands.append(kind)
            if kind == "interrupt":
                raise KeyboardInterrupt

        def get_agent_session(self, *_args: object) -> object:
            return type("Session", (), {"status": "ended", "error": None})()

        def close(self) -> None:
            self.closed = True

    class _NoReader:
        def __init__(self, *_args: object) -> None:
            pass

        def start(self) -> None:
            pass

    client = _Client()
    driver, _store = _plain_driver(client, tmp_path)
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(mod, "RemoteAgentCommandReader", _NoReader)
        driver.run()

    assert client.commands == ["interrupt", "end"]
    assert client.closed


def test_plain_run_interrupt_around_a_patch_ack_promotes_a_fresh_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Ctrl-C during the patch ack must not leave a cursor that re-fetches it.

    The resumed loop (and a later :detach promotion) must poll past the
    processed patch event: its object is acknowledged, so re-requesting it
    would 404 and abort the next detached command.
    """
    root = tmp_path / "work"
    root.mkdir()
    (root / "answer.txt").write_text("before", encoding="utf-8")
    base = mod.snapshot_workspace(root)
    remote = io.BytesIO()
    with tarfile.open(fileobj=remote, mode="w:gz") as archive:
        body = b"during"
        info = tarfile.TarInfo("answer.txt")
        info.size = len(body)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(body))
    patch = build_workspace_patch(base.archive, remote.getvalue())
    assert patch is not None

    class _Client:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.acks = 0
            self.closed = False

        def create_agent_session(self, *_args: object, **_kwargs: object) -> object:
            return type("Session", (), {"id": "sess-1"})()

        def list_agent_session_events(
            self, _agent_id: str, _session_id: str, *, after: int
        ) -> object:
            self.calls.append(f"events after={after}")
            if after == 0:
                event = type(
                    "Event",
                    (),
                    {"seq": 1, "kind": "workspace_patch", "payload": {"revision": "p1"}},
                )()
                return type("Page", (), {"events": [event], "last_seq": 1, "status": "running"})()
            return type("Page", (), {"events": [], "last_seq": after, "status": "running"})()

        def download_agent_workspace_patch(
            self, _agent_id: str, _session_id: str, revision: str
        ) -> bytes:
            self.calls.append(f"patch:{revision}")
            if revision != "p1" or self.acks:
                pytest.fail("an acknowledged patch must never be re-requested")
            return patch

        def acknowledge_agent_workspace_patch(
            self, _agent_id: str, _session_id: str, revision: str
        ) -> None:
            _ = revision
            self.acks += 1
            raise KeyboardInterrupt

        def post_agent_session_command(
            self, _agent_id: str, _session_id: str, kind: str, *, text: str | None = None
        ) -> None:
            _ = kind, text

        def close(self) -> None:
            self.closed = True

    class _DetachAfterResume:
        def __init__(self, *_args: object) -> None:
            self.eof = threading.Event()
            self.detach = threading.Event()
            self.detach.set()

        def start(self) -> None:
            pass

    client = _Client()
    driver, store = _plain_driver(client, tmp_path, jail_root=root)
    monkeypatch.setattr(mod, "RemoteAgentCommandReader", _DetachAfterResume)
    driver.run()

    assert client.calls.count("patch:p1") == 1
    assert "events after=1" in client.calls
    state = store.load("sess-1")
    assert state is not None
    assert state.cursor == 1
