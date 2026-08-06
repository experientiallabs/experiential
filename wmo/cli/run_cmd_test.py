# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for wmo run dispatch, local safety boundaries, and hosted world models."""

from __future__ import annotations

import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

import wmo.cli.run_cmd as mod
from wmo.cli.app import app
from wmo.common.config.settings import ModelRole, ModelsSettings, ProjectSettings, save_settings
from wmo.common.core.types import Action
from wmo.common.providers.base import ProviderConfig, ProviderKind
from wmo.common.vendor.waterfall.types import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatUsage,
)
from wmo.runtime.harness.doc import HarnessDoc
from wmo.runtime.harness.live_session import SessionEvent
from wmo.runtime.harness.pi_local import start_local_live_runner
from wmo.runtime.platform.client import PlatformClient
from wmo.runtime.platform.credentials import PlatformCredentials

if TYPE_CHECKING:
    from collections.abc import Callable


def _noop_emit(_stream: str, _chunk: str) -> None:
    """Discard streamed tool output."""


def test_executor_reads_writes_and_jails(tmp_path: Path) -> None:
    executor = mod.LocalToolExecutor(tmp_path)

    wrote = executor("write_file", {"path": "sub/a.txt", "content": "hi"}, _noop_emit)
    assert not wrote.is_error
    assert (tmp_path / "sub" / "a.txt").read_text(encoding="utf-8") == "hi"

    read = executor("read_file", {"path": "sub/a.txt"}, _noop_emit)
    assert read.content == "hi"

    escaped = executor("read_file", {"path": "../../etc/passwd"}, _noop_emit)
    assert escaped.is_error
    assert "escapes" in escaped.content

    absolute = executor("write_file", {"path": "/tmp/evil.txt", "content": "x"}, _noop_emit)
    assert absolute.is_error


def test_executor_cancel_wins_the_race_before_a_file_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = mod.LocalToolExecutor(tmp_path)
    resolved = threading.Event()
    release = threading.Event()
    outcomes: list[mod.ToolOutcome] = []
    original_resolve = executor._resolve

    def pause_after_initial_cancel_check(path: str) -> Path:
        target = original_resolve(path)
        resolved.set()
        release.wait(timeout=2)
        return target

    monkeypatch.setattr(executor, "_resolve", pause_after_initial_cancel_check)
    thread = threading.Thread(
        target=lambda: outcomes.append(
            executor("write_file", {"path": "sub/race.txt", "content": "stale"}, _noop_emit)
        )
    )
    thread.start()
    assert resolved.wait(1)

    executor.cancel()
    release.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    [result] = outcomes
    assert result.is_error
    assert result.content == "interrupted"
    assert not (tmp_path / "sub").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="LocalToolExecutor bash needs a Unix shell")
def test_executor_bash_runs_in_jail_and_reports_exit(tmp_path: Path) -> None:
    executor = mod.LocalToolExecutor(tmp_path)
    chunks: list[tuple[str, str]] = []

    ok = executor("bash", {"command": "pwd && echo hello"}, lambda s, c: chunks.append((s, c)))
    assert not ok.is_error
    assert str(tmp_path.resolve()) in ok.content
    assert "hello" in ok.content
    assert any(stream == "stdout" for stream, _ in chunks)

    failed = executor("bash", {"command": "exit 3"}, _noop_emit)
    assert failed.is_error
    assert "[exit 3]" in failed.content


@pytest.mark.skipif(sys.platform == "win32", reason="LocalToolExecutor bash needs a Unix shell")
def test_executor_bash_streams_chunks_and_retains_bounded_output(tmp_path: Path) -> None:
    chunks: list[tuple[str, str]] = []

    result = mod.LocalToolExecutor(tmp_path)(
        "bash",
        {"command": "printf '%050000d' 0"},
        lambda stream, chunk: chunks.append((stream, chunk)),
    )

    assert not result.is_error
    assert result.truncated
    assert len(result.content) <= mod._TOOL_OUTPUT_CAP
    assert len(chunks) > 1
    assert max(len(chunk) for _stream, chunk in chunks) <= 4096
    assert sum(len(chunk) for _stream, chunk in chunks) == 50_000


@pytest.mark.skipif(sys.platform == "win32", reason="LocalToolExecutor bash needs a Unix shell")
def test_executor_bash_kills_the_process_group_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_BASH_TIMEOUT_S", 0.05)

    started = time.monotonic()
    result = mod.LocalToolExecutor(tmp_path)("bash", {"command": "sleep 5"}, _noop_emit)

    assert time.monotonic() - started < 2
    assert result.is_error
    assert "timed out after 0.05s" in result.content
    assert "[exit 124]" in result.content


@pytest.mark.skipif(sys.platform == "win32", reason="LocalToolExecutor bash needs a Unix shell")
def test_executor_bash_does_not_wait_for_a_background_child_holding_its_pipes(
    tmp_path: Path,
) -> None:
    started = time.monotonic()

    result = mod.LocalToolExecutor(tmp_path)(
        "bash", {"command": "sleep 5 & printf done"}, _noop_emit
    )

    assert time.monotonic() - started < 2
    assert not result.is_error
    assert result.content == "done"


@pytest.mark.skipif(sys.platform == "win32", reason="LocalToolExecutor bash needs a Unix shell")
def test_executor_cancel_stops_an_active_bash_command(tmp_path: Path) -> None:
    executor = mod.LocalToolExecutor(tmp_path)
    started = threading.Event()
    outcomes: list[object] = []

    def emit(_stream: str, chunk: str) -> None:
        if "started" in chunk:
            started.set()

    thread = threading.Thread(
        target=lambda: outcomes.append(
            executor("bash", {"command": "printf started; sleep 5"}, emit)
        )
    )
    thread.start()
    assert started.wait(1)

    executor.cancel()
    thread.join(timeout=2)

    assert not thread.is_alive()
    [result] = outcomes
    assert isinstance(result, mod.ToolOutcome)
    assert result.is_error
    assert result.content == "interrupted"


def test_executor_streams_and_caps_large_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "big.txt").write_text(
        "head" + "x" * (mod._TOOL_OUTPUT_CAP + 500) + "tail",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: pytest.fail("read_file must not materialize the whole file"),
    )

    result = mod.LocalToolExecutor(tmp_path)("read_file", {"path": "big.txt"}, _noop_emit)

    assert result.truncated
    assert "chars truncated" in result.content
    assert result.content.startswith("head")
    assert result.content.endswith("tail")
    assert len(result.content) <= mod._TOOL_OUTPUT_CAP


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are not available on this platform")
def test_executor_rejects_a_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "blocked"
    os.mkfifo(fifo)
    outcomes: list[mod.ToolOutcome] = []
    thread = threading.Thread(
        target=lambda: outcomes.append(
            mod.LocalToolExecutor(tmp_path)("read_file", {"path": "blocked"}, _noop_emit)
        ),
        daemon=True,
    )

    thread.start()
    thread.join(timeout=1)

    assert not thread.is_alive()
    [result] = outcomes
    assert result.is_error
    assert "not a regular file" in result.content


def test_terminal_sink_renders_agent_text_without_rich_markup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(file=io.StringIO(), width=200, color_system=None)
    monkeypatch.setattr(mod, "_console", console)
    text = "Keep list[str], [a link](https://example.test), and [/items] literal."

    mod.TerminalEventSink(recorder=None, on_running=lambda _running: None)(
        SessionEvent(kind="assistant_message", payload={"text": text})
    )

    assert text in cast("io.StringIO", console.file).getvalue()


class _FakeProvider:
    """A minimal tool-calling provider."""

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        _ = request
        return ChatResponse(choices=[])


class _FakeClient:
    """Record target resolution and built-in pi proxy calls."""

    def __init__(self) -> None:
        self.worker_calls: list[ChatRequest] = []
        self.target_kind = "agent"
        self.closed = False
        self.local_pi_created: list[str] = []
        self.local_pi_finished: list[str] = []
        self.local_pi_worker_provider = "bedrock"
        self.local_pi_worker_model = "claude-haiku-4-5"
        self.local_pi_context_window: int | None = None

    def resolve_run_target(self, target_id: str) -> object:
        return type(
            "Target",
            (),
            {"id": target_id, "kind": self.target_kind, "name": "remote-target"},
        )()

    def create_local_pi_run(self, org_id: str) -> object:
        self.local_pi_created.append(org_id)
        return mod.platform_client.LocalPiRunInfo(
            id="run-1",
            org_id=org_id,
            status="running",
            worker_provider=self.local_pi_worker_provider,
            worker_model=self.local_pi_worker_model,
            context_window=self.local_pi_context_window,
        )

    def complete_local_pi_worker(
        self,
        org_id: str,
        run_id: str,
        request: ChatRequest,
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
    monkeypatch.setattr(
        "wmo.common.providers.registry.get_provider",
        lambda _config: _FakeProvider(),
    )


def test_build_driver_logged_out_runs_baseline_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "wmo.runtime.platform.credentials.load_credentials",
        PlatformCredentials,
    )
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    save_settings(
        ProjectSettings(
            models=ModelsSettings(worker=ModelRole(provider="openai", model="gpt-5.4-mini"))
        ),
        tmp_path / ".wmo",
    )
    monkeypatch.setattr(
        "wmo.runtime.platform.credentials.load_credentials",
        PlatformCredentials,
    )
    configs: list[ProviderConfig] = []

    def get_provider(config: ProviderConfig) -> _FakeProvider:
        configs.append(config)
        return _FakeProvider()

    monkeypatch.setattr("wmo.common.providers.registry.get_provider", get_provider)

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


def test_build_driver_provider_override_uses_the_new_providers_default_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    save_settings(
        ProjectSettings(
            models=ModelsSettings(worker=ModelRole(provider="bedrock", model="claude-sonnet-4-6"))
        ),
        tmp_path / ".wmo",
    )
    monkeypatch.setattr(
        "wmo.runtime.platform.credentials.load_credentials",
        PlatformCredentials,
    )
    configs: list[ProviderConfig] = []

    def get_provider(config: ProviderConfig) -> _FakeProvider:
        configs.append(config)
        return _FakeProvider()

    monkeypatch.setattr("wmo.common.providers.registry.get_provider", get_provider)

    driver = mod._build_driver(
        target=None,
        jail_root=tmp_path,
        provider="openai",
        model=None,
        task=None,
    )

    assert isinstance(driver, mod.LocalLiveDriver)
    [config] = configs
    assert config.kind is ProviderKind.OPENAI
    assert config.model == "gpt-5.6-sol"


def test_build_driver_azure_override_supplies_required_azure_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    save_settings(
        ProjectSettings(
            models=ModelsSettings(worker=ModelRole(provider="bedrock", model="claude-sonnet-4-6"))
        ),
        tmp_path / ".wmo",
    )
    monkeypatch.setattr(
        "wmo.runtime.platform.credentials.load_credentials",
        PlatformCredentials,
    )
    configs: list[ProviderConfig] = []

    def get_provider(config: ProviderConfig) -> _FakeProvider:
        configs.append(config)
        return _FakeProvider()

    monkeypatch.setattr("wmo.common.providers.registry.get_provider", get_provider)

    driver = mod._build_driver(
        target=None,
        jail_root=tmp_path,
        provider="azure",
        model=None,
        task=None,
    )

    assert isinstance(driver, mod.LocalLiveDriver)
    [config] = configs
    assert config.kind is ProviderKind.AZURE_OPENAI
    assert config.model == "gpt-5.5"
    assert config.deployment == "gpt-5.5"
    assert config.api_version == mod.DEFAULT_AZURE_API_VERSION


def test_build_driver_azure_model_override_rejects_a_stale_configured_deployment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    save_settings(
        ProjectSettings(
            models=ModelsSettings(
                worker=ModelRole(
                    provider="azure",
                    model="gpt-5.4",
                    endpoint="https://azure.example/v1",
                    deployment="prod-54-canary",
                )
            )
        ),
        tmp_path / ".wmo",
    )
    monkeypatch.setattr(
        "wmo.runtime.platform.credentials.load_credentials",
        PlatformCredentials,
    )
    configs: list[ProviderConfig] = []
    monkeypatch.setattr(
        "wmo.common.providers.registry.get_provider",
        lambda config: configs.append(config) or _FakeProvider(),
    )

    with pytest.raises(typer.BadParameter) as excinfo:
        mod._build_driver(
            target=None,
            jail_root=tmp_path,
            provider=None,
            model="gpt-5.5",
            task=None,
        )

    assert "prod-54-canary" in str(excinfo.value)
    assert "wmo providers set --provider azure --model gpt-5.5 --deployment <deployment>" in str(
        excinfo.value
    )
    assert configs == []


def test_build_driver_preserves_a_configured_tinker_models_base_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    save_settings(
        ProjectSettings(
            models=ModelsSettings(
                worker=ModelRole(
                    provider="tinker",
                    model="tinker://run/weights/42",
                    model_type="Qwen/Qwen3-8B",
                    chat_max_tokens_field="max_tokens",
                )
            )
        ),
        tmp_path / ".wmo",
    )
    monkeypatch.setattr(
        "wmo.runtime.platform.credentials.load_credentials",
        PlatformCredentials,
    )
    configs: list[ProviderConfig] = []
    monkeypatch.setattr(
        "wmo.common.providers.registry.get_provider",
        lambda config: configs.append(config) or _FakeProvider(),
    )

    driver = mod._build_driver(
        target=None,
        jail_root=tmp_path,
        provider=None,
        model=None,
        task=None,
    )

    assert isinstance(driver, mod.LocalLiveDriver)
    [config] = configs
    assert config.kind is ProviderKind.TINKER
    assert config.model == "tinker://run/weights/42"
    assert config.model_type == "Qwen/Qwen3-8B"
    assert config.chat_max_tokens_field == "max_tokens"


def test_build_driver_logged_in_bare_run_uses_platform_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = PlatformCredentials(
        api_url="https://api.test",
        token="xpl_test",
        default_org="org-1",
    )
    monkeypatch.setattr(
        "wmo.runtime.platform.credentials.load_credentials",
        lambda: credentials,
    )
    client = _FakeClient()
    client.local_pi_worker_provider = "tinker"
    client.local_pi_worker_model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:65536"
    monkeypatch.setattr(
        "wmo.runtime.platform.client.PlatformClient",
        lambda *_args, **_kwargs: client,
    )

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
    assert driver._context_window == 65_536
    assert isinstance(driver._recorder, mod.LocalPiRunRecorder)
    assert client.local_pi_created == ["org-1"]

    driver._worker_fn(ChatRequest(messages=[ChatMessage(role="user", content="hi")]))
    assert len(client.worker_calls) == 1

    driver._recorder.finish(ended_reason="user_ended", error=None)
    assert client.local_pi_finished == ["run-1"]
    assert client.closed


def test_build_driver_world_model_uses_hosted_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = PlatformCredentials(api_url="https://api.test", token="xpl_test")
    monkeypatch.setattr(
        "wmo.runtime.platform.credentials.load_credentials",
        lambda: credentials,
    )
    client = _FakeClient()
    client.target_kind = "world_model"
    monkeypatch.setattr(
        "wmo.runtime.platform.client.PlatformClient",
        lambda *_args, **_kwargs: client,
    )

    driver = mod._build_driver(
        target="wm-1",
        jail_root=None,
        provider=None,
        model=None,
        task="help the customer",
    )

    assert isinstance(driver, mod.RemoteWorldModelDriver)
    assert driver._target_id == "wm-1"
    assert not client.closed


def test_build_driver_agent_id_names_removed_platform_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = PlatformCredentials(api_url="https://api.test", token="xpl_test")
    monkeypatch.setattr(
        "wmo.runtime.platform.credentials.load_credentials",
        lambda: credentials,
    )
    client = _FakeClient()
    monkeypatch.setattr(
        "wmo.runtime.platform.client.PlatformClient",
        lambda *_args, **_kwargs: client,
    )

    with pytest.raises(typer.BadParameter, match="hosted agent sessions are unavailable"):
        mod._build_driver(
            target="agent-1",
            jail_root=None,
            provider=None,
            model=None,
            task=None,
        )

    assert client.closed


def test_build_driver_target_without_login_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "wmo.runtime.platform.credentials.load_credentials",
        PlatformCredentials,
    )

    with pytest.raises(typer.BadParameter, match="wmo login"):
        mod._build_driver(
            target="wm-1",
            jail_root=None,
            provider=None,
            model=None,
            task=None,
        )


def test_platform_target_rejects_local_provider_before_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = PlatformCredentials(api_url="https://api.test", token="xpl_test")
    monkeypatch.setattr(
        "wmo.runtime.platform.credentials.load_credentials",
        lambda: credentials,
    )
    client = _FakeClient()
    monkeypatch.setattr(
        "wmo.runtime.platform.client.PlatformClient",
        lambda *_args, **_kwargs: client,
    )

    with pytest.raises(typer.BadParameter, match="platform credentials"):
        mod._build_driver(
            target="wm-1",
            jail_root=None,
            provider="bedrock",
            model=None,
            task=None,
        )


def test_logged_in_bare_run_rejects_local_provider_before_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = PlatformCredentials(api_url="https://api.test", token="xpl_test")
    monkeypatch.setattr(
        "wmo.runtime.platform.credentials.load_credentials",
        lambda: credentials,
    )
    prompted: list[bool] = []

    with pytest.raises(typer.BadParameter, match="logged-in runs use platform credentials"):
        mod._build_driver(
            target=None,
            jail_root=Path.cwd(),
            provider="openai",
            model=None,
            task=None,
            confirm_local=lambda: prompted.append(True),
        )

    assert prompted == []


def test_run_command_dispatches_bare_path_after_consent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Driver:
        def run(self) -> None:
            seen["ran"] = True

    def build_driver(**kwargs: object) -> _Driver:
        callback = cast("Callable[[], None]", kwargs["confirm_local"])
        callback()
        seen.update(kwargs)
        return _Driver()

    monkeypatch.setattr(mod, "_build_driver", build_driver)

    result = CliRunner().invoke(
        app,
        ["run", "--dir", str(tmp_path), "--task", "fix it", "--yes"],
    )

    assert result.exit_code == 0
    assert seen["jail_root"] == tmp_path.resolve()
    assert seen["task"] == "fix it"
    assert seen["ran"] is True
    assert "THIS machine" in result.output


def test_run_command_rejects_a_directory_for_platform_target(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["run", "wm-1", "--dir", str(tmp_path)])

    assert result.exit_code == 2
    assert "--dir is only supported" in result.output


def test_run_command_stops_when_local_execution_is_declined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[bool] = []

    class _Driver:
        def run(self) -> None:
            ran.append(True)

    def build_driver(**kwargs: object) -> _Driver:
        callback = cast("Callable[[], None]", kwargs["confirm_local"])
        callback()
        return _Driver()

    monkeypatch.setattr(mod, "_build_driver", build_driver)

    result = CliRunner().invoke(
        app,
        ["run", "--dir", str(tmp_path)],
        input="n\n",
    )

    assert result.exit_code == 1
    assert ran == []
    assert "continue?" in result.output


class _FakeChannel:
    """Record local runner teardown."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeLiveSession:
    """Emit one idle state and then close."""

    sent: list[str] = []

    def __init__(
        self,
        _channel: object,
        *,
        on_event: Callable[[SessionEvent], None],
        **_kwargs: object,
    ) -> None:
        self._on_event = on_event
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def status(self) -> str:
        return "ended" if self._closed else "idle"

    @property
    def failure_message(self) -> str | None:
        return None

    def start(self, hello_timeout: float = 60.0) -> None:
        _ = hello_timeout

    def send_user_message(self, text: str) -> str:
        self.sent.append(text)
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
        self.last_message_id: str | None = None
        self.eof = threading.Event()

    def start(self) -> None:
        pass


def test_stdin_reader_queues_a_piped_turn_before_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Session:
        closed = False

        def __init__(self) -> None:
            self.messages: list[str] = []

        def send_user_message(self, text: str) -> str:
            self.messages.append(text)
            return "msg-1"

    session = _Session()
    monkeypatch.setattr(mod.sys, "stdin", io.StringIO("fix the tests\n"))
    reader = mod.StdinCommandReader(cast("mod.live_session.LiveSession", session))

    reader.run()

    assert session.messages == ["fix the tests"]
    assert reader.last_message_id == "msg-1"
    assert reader.eof.is_set()


def _patch_driver_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    channel: _FakeChannel,
) -> None:
    monkeypatch.setattr(
        "wmo.runtime.harness.pi_local.start_local_live_runner",
        lambda: channel,
    )
    monkeypatch.setattr(
        "wmo.runtime.harness.live_session.LiveSession",
        _FakeLiveSession,
    )
    monkeypatch.setattr(mod, "StdinCommandReader", _FakeReader)


def test_local_driver_boots_sends_task_and_closes_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = _FakeChannel()
    _FakeLiveSession.sent = []
    _patch_driver_boundaries(monkeypatch, channel)

    mod.LocalLiveDriver(
        jail_root=Path.cwd(),
        doc=HarnessDoc.baseline("test"),
        provider=_FakeProvider(),
        worker_fn=None,
        recorder=None,
        instruction="inspect the repo",
    ).run()

    assert _FakeLiveSession.sent == ["inspect the repo"]
    assert channel.closed


def test_local_driver_treats_an_empty_task_as_no_opening_turn_at_eof() -> None:
    class _IdleSession:
        def __init__(self) -> None:
            self.closed = False
            self.status = "idle"
            self.last_completed_message_id: str | None = None
            self.pumps = 0
            self.ends = 0

        def pump(self, timeout: float = 0.2) -> bool:
            _ = timeout
            self.pumps += 1
            assert self.pumps == 1, "an empty one-shot task must end on its first idle pump"
            return True

        def flush_pending_intents(self) -> None:
            pass

        def end(self) -> None:
            self.ends += 1
            self.closed = True

    driver = mod.LocalLiveDriver(
        jail_root=Path.cwd(),
        doc=HarnessDoc.baseline("test"),
        provider=_FakeProvider(),
        worker_fn=None,
        recorder=None,
        instruction="",
    )
    session = _IdleSession()
    reader = _FakeReader(None)
    reader.eof.set()

    driver._loop(
        cast("mod.live_session.LiveSession", session),
        cast("mod.StdinCommandReader", reader),
        None,
    )

    assert driver._instruction is None
    assert session.ends == 1


def test_local_driver_drains_a_delayed_piped_turn_before_eof_shutdown() -> None:
    reader = _FakeReader(None)
    reader.last_message_id = "second"
    reader.eof.set()

    class _DelayedPipedTurnSession:
        def __init__(self) -> None:
            self.closed = False
            self.status = "running"
            self.last_completed_message_id: str | None = None
            self.pending_messages = 1
            self.pumps = 0
            self.flushes = 0
            self.ends = 0

        def pump(self, timeout: float = 0.2) -> bool:
            _ = timeout
            self.pumps += 1
            assert self.pumps <= 3, "the piped turn must end after returning to idle"
            if self.pumps == 1:
                # This pump drains and sends the second line, then consumes the already-buffered
                # idle frame for the first line. Only the message id distinguishes that stale idle.
                self.pending_messages = 0
                self.status = "idle"
                self.last_completed_message_id = "first"
            elif self.pumps == 2:
                self.status = "running"
            elif self.pumps == 3:
                self.status = "idle"
                self.last_completed_message_id = "second"
            return True

        def flush_pending_intents(self) -> None:
            self.flushes += 1

        def end(self) -> None:
            self.ends += 1
            self.closed = True

    driver = mod.LocalLiveDriver(
        jail_root=Path.cwd(),
        doc=HarnessDoc.baseline("test"),
        provider=_FakeProvider(),
        worker_fn=None,
        recorder=None,
        instruction=None,
    )
    session = _DelayedPipedTurnSession()

    driver._loop(
        cast("mod.live_session.LiveSession", session),
        cast("mod.StdinCommandReader", reader),
        None,
    )

    assert session.pumps == 3
    assert session.flushes == 3
    assert session.ends == 1


def test_ctrl_c_escalation_resets_after_the_turn_returns_idle() -> None:
    class _Session:
        status = "running"

        def __init__(self) -> None:
            self.interrupts = 0
            self.ends = 0

        def interrupt(self) -> None:
            self.interrupts += 1

        def end(self) -> None:
            self.ends += 1

    driver = mod.LocalLiveDriver(
        jail_root=Path.cwd(),
        doc=HarnessDoc.baseline("test"),
        provider=_FakeProvider(),
        worker_fn=None,
        recorder=None,
        instruction=None,
    )
    session = _Session()

    driver._handle_sigint(cast("mod.live_session.LiveSession", session))
    driver._on_running(False)
    session.status = "running"
    driver._handle_sigint(cast("mod.live_session.LiveSession", session))

    assert session.interrupts == 2
    assert session.ends == 0


def test_remote_world_model_driver_creates_and_steps_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WorldModelClient:
        def __init__(self) -> None:
            self.task: str | None = None
            self.actions: list[Action] = []
            self.closed = False

        def create_world_model_session(self, _target: str, *, task: str | None) -> object:
            self.task = task
            return type("Session", (), {"id": "session-1"})()

        def step_world_model_session(self, _session: str, action: Action) -> object:
            self.actions.append(action)
            return type(
                "Observation",
                (),
                {"content": "found list[str] and [/items] literally", "is_error": False},
            )()

        def close(self) -> None:
            self.closed = True

    client = _WorldModelClient()
    lines = iter(['search {"q": "SFO"}', ":quit"])
    console = Console(file=io.StringIO(), width=200)
    monkeypatch.setattr(console, "input", lambda *_args, **_kwargs: next(lines))
    monkeypatch.setattr(mod, "_console", console)

    mod.RemoteWorldModelDriver(
        cast("PlatformClient", client),
        "wm-1",
        "Airline",
        "find a flight",
    ).run()

    assert client.task == "find a flight"
    assert len(client.actions) == 1
    assert client.actions[0].name == "search"
    assert client.closed
    assert "found list[str] and [/items] literally" in cast("io.StringIO", console.file).getvalue()


def test_world_model_loop_rejects_detach(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Client:
        def step_world_model_session(self, *_args: object, **_kwargs: object) -> object:
            pytest.fail(":detach must never reach the world model as an action")

        def close(self) -> None:
            pass

    lines = iter([":detach", ":quit"])
    monkeypatch.setattr(mod._console, "input", lambda *_args, **_kwargs: next(lines))

    mod.RemoteWorldModelDriver(
        cast("PlatformClient", _Client()),
        "wm-1",
        "Model",
        None,
    )._loop("session-1")


@pytest.mark.skipif(
    os.environ.get("WMO_RUN_LIVE_INTEGRATION") != "1",
    reason="set WMO_RUN_LIVE_INTEGRATION=1 to install pi and run the real Node peer",
)
def test_local_pi_runner_and_driver_complete_a_real_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boot the retained Node runner and finish a one-shot task through the real driver."""
    completion = ChatResponse.model_validate(
        {
            "id": "completion-1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "submit",
                                    "arguments": '{"answer":"done"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )
    channel = start_local_live_runner(runtime_dir=tmp_path / "pi-runtime")
    console = Console(file=io.StringIO(), width=200)
    monkeypatch.setattr(
        "wmo.runtime.harness.pi_local.start_local_live_runner",
        lambda: channel,
    )
    monkeypatch.setattr(mod.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(mod, "_console", console)

    mod.LocalLiveDriver(
        jail_root=tmp_path,
        doc=mod._pi_node_baseline(),
        provider=None,
        worker_fn=lambda _request: completion,
        recorder=None,
        instruction="finish successfully",
    ).run()

    output = cast("io.StringIO", console.file).getvalue()
    assert "submitted done" in output
    assert "session ended (user_ended)" in output
