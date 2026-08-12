"""Tests for the narrow optional Harbor and E2B environment adapter."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast

import pytest

from wmo.common.models import ToolCall
from wmo.common.tasks import TaskCase
from wmo.runtime.environments.harbor import (
    HarborCleanupUnprovenError,
    HarborCommandResult,
    HarborEnvironmentRuntime,
    HarborRetryableCommandError,
    HarborTemplateStatusError,
    HarborTranscriptEntry,
    e2b_template_resource_digest,
    qualify_harbor_e2b_template_name,
    resolve_e2b_template_resources,
    retry_template_status,
)
from wmo.runtime.harness.e2b_ledger import SandboxLedger, read_ledger_files


class _TranscriptEnvironmentSession(Protocol):
    """Narrow test-only view of the Harbor session transcript capability."""

    @property
    def partial_transcript(self) -> tuple[HarborTranscriptEntry, ...]:
        """Return every environment observation retained by this Harbor adapter."""
        ...


def test_harbor_runtime_preserves_partial_transcript_and_proven_cleanup(tmp_path: Path) -> None:
    """The environment seam records create, tools, and release without benchmark ownership."""
    ledger = SandboxLedger(tmp_path / "ledger", pid=827)
    session = _Session()
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        ledger=ledger,
        retry_delays_seconds=(),
    )

    with runtime.open(_task()) as environment:
        observation = environment.execute(
            ToolCall(call_id="call-1", name="read_file", arguments={"path": "notes.txt"})
        )
        assert observation.content == "fixture output"
        assert not observation.is_error
        partial = cast(_TranscriptEnvironmentSession, environment).partial_transcript
        assert partial[0].action.name == "read_file"
        assert partial[0].observation == observation

    ledgers = read_ledger_files(tmp_path / "ledger")
    assert len(ledgers) == 1
    assert ledgers[0].held == ()
    assert ledgers[0].released_ids == ("sandbox-1",)
    assert session.closed
    assert len(session.commands) == 1
    assert "pwd -P" in session.commands[0][0]
    assert session.commands[0][1] == {
        "WMO_FILE_PATH": "notes.txt",
        "WMO_SANDBOX_ROOT": "/workspace",
    }


def test_harbor_runtime_retries_only_read_only_transport_failures(tmp_path: Path) -> None:
    """A read can retry, while a mutation makes one exact environment call only."""
    ledger = SandboxLedger(tmp_path / "ledger", pid=828)
    session = _Session(fail_read_once=True)
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        ledger=ledger,
        retry_delays_seconds=(0.0,),
    )

    with runtime.open(_task()) as environment:
        assert (
            environment.execute(
                ToolCall(call_id="call-read", name="read_file", arguments={"path": "notes.txt"})
            ).content
            == "fixture output"
        )
        assert (
            environment.execute(
                ToolCall(
                    call_id="call-write",
                    name="write_file",
                    arguments={"path": "notes.txt", "content": "new content"},
                )
            ).content
            == "wrote notes.txt"
        )

    assert len(session.commands) == 3
    assert "pwd -P" in session.commands[0][0]
    assert session.commands[1][0] == session.commands[0][0]
    assert "WMO_FILE_PATH" in (session.commands[2][1] or {})


def test_harbor_runtime_does_not_mark_unproven_cleanup_released(tmp_path: Path) -> None:
    """A factory cleanup error leaves the durable ledger holding the sandbox for a reaper."""
    ledger = SandboxLedger(tmp_path / "ledger", pid=829)
    session = _Session()
    runtime = HarborEnvironmentRuntime(
        _Factory(session, fail_close=True),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        ledger=ledger,
    )

    with pytest.raises(OSError, match="cleanup"):
        with runtime.open(_task()):
            pass

    ledgers = read_ledger_files(tmp_path / "ledger")
    assert len(ledgers) == 1
    assert tuple(record.sandbox_id for record in ledgers[0].held) == ("sandbox-1",)


def test_harbor_runtime_rejects_false_cleanup_proof_and_keeps_ledger_hold(
    tmp_path: Path,
) -> None:
    """A clean context return is insufficient when the backend reports cleanup as unverified."""
    ledger = SandboxLedger(tmp_path / "ledger", pid=830)
    session = _Session(cleanup_proven=False)
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        ledger=ledger,
    )

    with pytest.raises(HarborCleanupUnprovenError, match="without proof"):
        with runtime.open(_task()):
            pass

    ledgers = read_ledger_files(tmp_path / "ledger")
    assert tuple(record.sandbox_id for record in ledgers[0].held) == ("sandbox-1",)


def test_harbor_file_tools_reject_lexical_and_symlink_path_escapes(tmp_path: Path) -> None:
    """Canonical file actions cannot traverse or follow a link outside the configured root."""
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    session = _ShellSession()
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        ledger=SandboxLedger(tmp_path / "ledger", pid=831),
        retry_delays_seconds=(),
        workspace_root=str(workspace),
    )

    with runtime.open(_task()) as environment:
        lexical = environment.execute(
            ToolCall(call_id="call-lexical", name="read_file", arguments={"path": "../secret"})
        )
        symlink = environment.execute(
            ToolCall(
                call_id="call-symlink",
                name="read_file",
                arguments={"path": "escape/secret.txt"},
            )
        )
        write = environment.execute(
            ToolCall(
                call_id="call-write",
                name="write_file",
                arguments={"path": "escape/new.txt", "content": "blocked"},
            )
        )

    assert lexical.is_error
    assert symlink.is_error and symlink.metadata["exit_code"] == 73
    assert write.is_error and write.metadata["exit_code"] == 73
    assert not (outside / "new.txt").exists()
    assert len(session.commands) == 2


def test_template_identity_includes_resources_and_dependency_versions() -> None:
    """Changing resources or an SDK version cannot silently reuse another template."""
    resources = resolve_e2b_template_resources(cpu_count=None, memory_mb=None)
    first = e2b_template_resource_digest(
        environment_id="a" * 32,
        build_source_kind="docker_image",
        build_source_reference="fixture:image",
        resources=resources,
        harbor_version="0.20.0",
        e2b_sdk_version="2.31.0",
    )
    second = e2b_template_resource_digest(
        environment_id="a" * 32,
        build_source_kind="docker_image",
        build_source_reference="fixture:image",
        resources=resolve_e2b_template_resources(cpu_count=4, memory_mb=None),
        harbor_version="0.20.0",
        e2b_sdk_version="2.31.0",
    )

    assert first != second
    assert (
        qualify_harbor_e2b_template_name(
            "harbor-template-aaaaaaaa",
            environment_id="a" * 32,
            build_source_kind="docker_image",
            build_source_reference="fixture:image",
            resources=resources,
            harbor_version="0.20.0",
            e2b_sdk_version="2.31.0",
            snapshot_hash_length=8,
        )
        == f"wmo-hb-v1-{first}"
    )


def test_template_status_retry_retries_only_the_idempotent_read() -> None:
    """The retry helper is usable for status reads without making submission replayable."""
    attempts = 0

    def read_status() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HarborTemplateStatusError("temporary status transport failure")
        return "ready"

    assert retry_template_status(read_status, retry_delays_seconds=(0.0,)) == "ready"
    assert attempts == 2


class _Factory:
    """Opens one in-memory optional Harbor session for deterministic adapter tests."""

    def __init__(
        self,
        session: _Session | _ShellSession,
        *,
        fail_close: bool = False,
    ) -> None:
        self._session = session
        self._fail_close = fail_close

    def open(
        self,
        task: TaskCase,
        *,
        template_name: str,
    ) -> AbstractContextManager[_Session | _ShellSession]:
        assert task == _task()
        assert template_name == "wmo-hb-v1-fixture"
        return _Context(self._session, fail_close=self._fail_close)


class _Context(AbstractContextManager["_Session | _ShellSession"]):
    """Makes successful and failed cleanup explicit in adapter lifecycle fixtures."""

    def __init__(self, session: _Session | _ShellSession, *, fail_close: bool) -> None:
        self._session = session
        self._fail_close = fail_close

    def __enter__(self) -> _Session | _ShellSession:
        return self._session

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exception_type, exception, traceback
        self._session.closed = True
        if self._fail_close:
            raise OSError("cleanup not proved")
        return False


class _Session:
    """A scriptable session that conforms to the narrow optional Harbor protocol."""

    sandbox_id = "sandbox-1"
    template_id = "wmo-hb-v1-fixture"

    def __init__(
        self,
        *,
        fail_read_once: bool = False,
        cleanup_proven: bool = True,
    ) -> None:
        self._fail_read_once = fail_read_once
        self._cleanup_proven = cleanup_proven
        self.commands: list[tuple[str, Mapping[str, str] | None, int]] = []
        self.closed = False

    def execute_command(
        self,
        command: str,
        *,
        environment: Mapping[str, str] | None,
        timeout_seconds: int,
    ) -> HarborCommandResult:
        self.commands.append((command, environment, timeout_seconds))
        if self._fail_read_once:
            self._fail_read_once = False
            raise HarborRetryableCommandError("temporary transport failure")
        return HarborCommandResult(stdout="fixture output", exit_code=0)

    def cleanup_verified(self) -> bool:
        """Report the scripted cleanup proof only after the context has closed."""
        return self.closed and self._cleanup_proven


class _ShellSession:
    """Execute guarded file fragments locally to exercise real symlink resolution."""

    sandbox_id = "sandbox-shell"
    template_id = "wmo-hb-v1-fixture"

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.closed = False

    def execute_command(
        self,
        command: str,
        *,
        environment: Mapping[str, str] | None,
        timeout_seconds: int,
    ) -> HarborCommandResult:
        """Run one injected fragment with only its explicit variables and a safe system path."""
        self.commands.append(command)
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            check=False,
            capture_output=True,
            env={"PATH": os.defpath, **dict(environment or {})},
            text=True,
            timeout=timeout_seconds,
        )
        return HarborCommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )

    def cleanup_verified(self) -> bool:
        """Return true once the deterministic local context has exited."""
        return self.closed


def _task() -> TaskCase:
    """Build one fixed canonical task for Harbor environment lifecycle fixtures."""
    return TaskCase(
        task_id="task-1",
        lineage_group_id="lineage-1",
        partition="fit",
        instruction="Run the optional environment fixture.",
        workload_weight=1.0,
        source_trace_ids=("trace-1",),
    )
