"""Tests for the narrow optional Harbor and E2B environment adapter."""

from __future__ import annotations

import gc
import multiprocessing
import os
import subprocess
import threading
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol, cast

import pytest

from wmo.common.models import ToolCall
from wmo.common.tasks import TaskCase
from wmo.runtime.environments.harbor import (
    BOUNDED_CLEANUP_CONTRACT,
    HarborCleanupResult,
    HarborCleanupTimeoutError,
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
from wmo.runtime.environments.sandbox_ledger import read_ledger_files


class _TranscriptEnvironmentSession(Protocol):
    """Narrow test-only view of the Harbor session transcript capability."""

    @property
    def partial_transcript(self) -> tuple[HarborTranscriptEntry, ...]:
        """Return every environment observation retained by this Harbor adapter."""
        ...


def test_harbor_runtime_preserves_partial_transcript_and_proven_cleanup(tmp_path: Path) -> None:
    """The environment seam records create, tools, and release without benchmark ownership."""
    session = _Session()
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
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
        session.sandbox_id = "mutated-after-create"

    ledgers = read_ledger_files(tmp_path / "state")
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
    session = _Session(fail_read_once=True)
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
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


def test_harbor_runtime_rejects_false_cleanup_proof_and_keeps_ledger_hold(
    tmp_path: Path,
) -> None:
    """A clean context return is insufficient when the backend reports cleanup as unverified."""
    session = _Session(cleanup_proven=False)
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(HarborCleanupUnprovenError, match="without proof"):
        with runtime.open(_task()):
            pass

    ledgers = read_ledger_files(tmp_path / "state")
    assert tuple(record.sandbox_id for record in ledgers[0].held) == ("sandbox-1",)


def test_blank_template_and_false_proof_hold_the_exact_captured_id(tmp_path: Path) -> None:
    """Bad secondary metadata cannot erase or replace the resource identity already returned."""
    session = _MutatingTemplateSession(cleanup_proven=False)
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(HarborCleanupUnprovenError, match="sandbox-1"):
        runtime.open(_task()).__enter__()

    [ledger] = read_ledger_files(tmp_path / "state")
    assert tuple(record.sandbox_id for record in ledger.held) == ("sandbox-1",)
    assert session.close_ids == ["sandbox-1"]
    assert session.sandbox_id == "corrupt-after-capture"


def test_blank_template_and_throwing_proof_keep_created_held(tmp_path: Path) -> None:
    """A proof exception is failure evidence, never permission to release the captured ID."""
    session = _Session(template_id="", cleanup_error=RuntimeError("proof unavailable"))
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(RuntimeError, match="proof unavailable"):
        runtime.open(_task()).__enter__()

    [ledger] = read_ledger_files(tmp_path / "state")
    assert tuple(record.sandbox_id for record in ledger.held) == ("sandbox-1",)


def test_template_validation_failure_releases_only_after_true_exact_proof(tmp_path: Path) -> None:
    """Invalid metadata still permits a proven exact cleanup to close its durable hold."""
    session = _Session(template_id="", cleanup_proven=True)
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(ValueError, match="template ID"):
        runtime.open(_task()).__enter__()

    [ledger] = read_ledger_files(tmp_path / "state")
    assert ledger.held == ()
    assert ledger.released_ids == ("sandbox-1",)


def test_bounded_close_exception_keeps_created_held(tmp_path: Path) -> None:
    """Adapter cleanup failure remains visible and cannot synthesize a release record."""
    session = _Session(cleanup_error=OSError("cleanup not proved"))
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(OSError, match="cleanup not proved"):
        with runtime.open(_task()):
            pass

    [ledger] = read_ledger_files(tmp_path / "state")
    assert tuple(record.sandbox_id for record in ledger.held) == ("sandbox-1",)
    assert ledger.released_ids == ()


def test_bounded_close_timeout_keeps_created_without_a_wmo_worker(tmp_path: Path) -> None:
    """Adapter-reported timeout returns with the resource held and no WMO cleanup worker."""
    session = _Session(cleanup_timed_out=True)
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
        cleanup_timeout_seconds=0.01,
    )

    before = _wmo_cleanup_workers()
    with pytest.raises(HarborCleanupTimeoutError, match="timed out"):
        with runtime.open(_task()):
            pass

    [ledger] = read_ledger_files(tmp_path / "state")
    assert tuple(record.sandbox_id for record in ledger.held) == ("sandbox-1",)
    assert session.close_ids == ["sandbox-1"]
    assert _wmo_cleanup_workers() == before


def test_twenty_cleanup_timeouts_leave_no_workers_or_session_references(tmp_path: Path) -> None:
    """Repeated bounded timeouts retain exact IDs without accumulating local cleanup state."""
    factory = _TimeoutFactory()
    runtime = HarborEnvironmentRuntime(
        factory,
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
        cleanup_timeout_seconds=0.01,
    )
    before = _wmo_cleanup_workers()
    assert before == ((), ())

    for _ in range(20):
        with pytest.raises(HarborCleanupTimeoutError):
            with runtime.open(_task()):
                pass

    gc.collect()
    [ledger] = read_ledger_files(tmp_path / "state")
    assert tuple(record.sandbox_id for record in ledger.held) == tuple(
        f"sandbox-timeout-{index}" for index in range(20)
    )
    assert all(reference() is None for reference in factory.session_references)
    assert _wmo_cleanup_workers() == before


def test_unbounded_context_factory_is_rejected_before_permanent_hang(tmp_path: Path) -> None:
    """WMO never invokes an arbitrary generic context whose exit can hang permanently."""
    factory = _PermanentHangFactory()
    before = _wmo_cleanup_workers()
    assert before == ((), ())

    with pytest.raises(ValueError, match="bounded-close-v1"):
        HarborEnvironmentRuntime(
            factory,  # ty: ignore[invalid-argument-type]
            environment_id="customer-env",
            template_name="wmo-hb-v1-fixture",
            state_directory=tmp_path / "state",
        )

    assert factory.open_calls == 0
    assert _wmo_cleanup_workers() == before


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
        state_directory=tmp_path / "state",
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

    cleanup_contract: Literal["bounded-close-v1"] = BOUNDED_CLEANUP_CONTRACT

    def __init__(self, session: _Session | _ShellSession) -> None:
        self._session = session

    def open(
        self,
        task: TaskCase,
        *,
        template_name: str,
    ) -> _Session | _ShellSession:
        assert task == _task()
        assert template_name == "wmo-hb-v1-fixture"
        return self._session


class _TimeoutFactory:
    """Create one new deterministically timing-out bounded session per open call."""

    cleanup_contract: Literal["bounded-close-v1"] = BOUNDED_CLEANUP_CONTRACT

    def __init__(self) -> None:
        self.session_references: list[weakref.ReferenceType[_Session]] = []

    def open(
        self,
        task: TaskCase,
        *,
        template_name: str,
    ) -> _Session:
        assert task == _task()
        assert template_name == "wmo-hb-v1-fixture"
        session = _Session(
            sandbox_id=f"sandbox-timeout-{len(self.session_references)}",
            cleanup_timed_out=True,
        )
        self.session_references.append(weakref.ref(session))
        return session


class _PermanentHangFactory:
    """Legacy generic-context shape that WMO must reject without invoking."""

    def __init__(self) -> None:
        self.open_calls = 0

    def open(self, task: TaskCase, *, template_name: str) -> object:
        del task, template_name
        self.open_calls += 1
        return _PermanentHangContext()


class _PermanentHangContext:
    """Adversarial old context whose exit would never return if WMO invoked it."""

    def __enter__(self) -> _Session:
        return _Session()

    def __exit__(self, *_args: object) -> bool:
        threading.Event().wait()
        return False


class _Session:
    """A scriptable session that conforms to the narrow optional Harbor protocol."""

    def __init__(
        self,
        *,
        fail_read_once: bool = False,
        cleanup_proven: bool = True,
        cleanup_timed_out: bool = False,
        cleanup_error: BaseException | None = None,
        sandbox_id: str = "sandbox-1",
        template_id: str = "wmo-hb-v1-fixture",
    ) -> None:
        self._fail_read_once = fail_read_once
        self._cleanup_proven = cleanup_proven
        self._cleanup_timed_out = cleanup_timed_out
        self._cleanup_error = cleanup_error
        self._template_id = template_id
        self.sandbox_id = sandbox_id
        self.commands: list[tuple[str, Mapping[str, str] | None, int]] = []
        self.close_ids: list[str] = []
        self.closed = False

    @property
    def template_id(self) -> str:
        """Return scripted template metadata."""
        return self._template_id

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

    def close(self, *, sandbox_id: str, timeout_seconds: float) -> HarborCleanupResult:
        """Return deterministic adapter-owned cleanup evidence within the supplied bound."""
        assert timeout_seconds > 0
        self.close_ids.append(sandbox_id)
        if self._cleanup_error is not None:
            raise self._cleanup_error
        self.closed = not self._cleanup_timed_out
        return HarborCleanupResult(
            sandbox_id=sandbox_id,
            released=self._cleanup_proven and not self._cleanup_timed_out,
            timed_out=self._cleanup_timed_out,
        )


class _MutatingTemplateSession(_Session):
    """Simulate hostile metadata access attempting to replace the returned sandbox identity."""

    @property
    def template_id(self) -> str:
        """Mutate the live object after capture, then return invalid metadata."""
        self.sandbox_id = "corrupt-after-capture"
        return ""


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

    def close(self, *, sandbox_id: str, timeout_seconds: float) -> HarborCleanupResult:
        """Return exact bounded cleanup proof for the deterministic local session."""
        assert timeout_seconds > 0
        assert sandbox_id == "sandbox-shell"
        self.closed = True
        return HarborCleanupResult(sandbox_id=sandbox_id, released=True)


def _wmo_cleanup_workers() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return WMO-owned cleanup threads and child processes visible to this test process."""
    threads = tuple(
        sorted(thread.name for thread in threading.enumerate() if "wmo-harbor" in thread.name)
    )
    processes = tuple(
        sorted(
            process.name
            for process in multiprocessing.active_children()
            if "wmo-harbor" in process.name
        )
    )
    return threads, processes


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
