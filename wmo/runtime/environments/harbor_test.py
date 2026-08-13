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
from typing import Any, Literal, Never, Protocol, cast

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
    HarborTranscriptEntry,
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


@pytest.mark.parametrize("sandbox_id", ["", "   ", "bad/id", "bad\ncontrol", ".bad"])
def test_malformed_sandbox_id_is_ledgered_then_released_only_after_exact_proof(
    tmp_path: Path, sandbox_id: str
) -> None:
    """Primary identity syntax failure cannot bypass durable bounded cleanup."""
    session = _Session(sandbox_id=sandbox_id, cleanup_proven=True)
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(ValueError, match="valid nonempty sandbox ID"):
        runtime.open(_task()).__enter__()

    [ledger] = read_ledger_files(tmp_path / "state")
    assert ledger.held == ()
    assert ledger.released_ids == (sandbox_id,)
    assert session.close_ids == [sandbox_id]
    assert ledger.path.parent == tmp_path / "state" / "e2b-sandboxes"
    assert tuple(ledger.path.parent.glob("*.jsonl")) == (ledger.path,)


def test_malformed_sandbox_id_keeps_durable_hold_without_cleanup_proof(tmp_path: Path) -> None:
    """Malformed primary identity remains recoverable when bounded close is unproven."""
    session = _Session(sandbox_id="malformed sandbox", cleanup_proven=False)
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(HarborCleanupUnprovenError, match="malformed sandbox"):
        runtime.open(_task()).__enter__()

    [ledger] = read_ledger_files(tmp_path / "state")
    assert tuple(record.sandbox_id for record in ledger.held) == ("malformed sandbox",)
    assert ledger.released_ids == ()
    assert session.close_ids == ["malformed sandbox"]


def test_sandbox_id_is_captured_once_before_hostile_accessor_mutation(tmp_path: Path) -> None:
    """A property that mutates after its first read cannot replace the ledgered close identity."""
    session = _MutatingSandboxIdSession()
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(ValueError, match="valid nonempty sandbox ID"):
        runtime.open(_task()).__enter__()

    [ledger] = read_ledger_files(tmp_path / "state")
    assert ledger.held == ()
    assert ledger.released_ids == ("bad id",)
    assert session.close_ids == ["bad id"]
    assert session.sandbox_reads == 1


@pytest.mark.parametrize(
    ("cleanup_mode", "expected_exception"),
    [
        ("throw", OSError),
        ("timeout", HarborCleanupTimeoutError),
        ("mismatch", HarborCleanupUnprovenError),
    ],
)
def test_malformed_id_cleanup_failure_modes_keep_exact_hold_without_workers(
    tmp_path: Path,
    cleanup_mode: str,
    expected_exception: type[BaseException],
) -> None:
    """Throw, timeout, and mismatched proof never release or spawn WMO cleanup workers."""
    session = {
        "throw": _Session(sandbox_id="bad id", cleanup_error=OSError("close failed")),
        "timeout": _Session(sandbox_id="bad id", cleanup_timed_out=True),
        "mismatch": _MismatchedCleanupSession(sandbox_id="bad id"),
    }[cleanup_mode]
    runtime = HarborEnvironmentRuntime(
        _Factory(session),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )
    before = _wmo_cleanup_workers()

    with pytest.raises(expected_exception):
        runtime.open(_task()).__enter__()

    [ledger] = read_ledger_files(tmp_path / "state")
    assert tuple(record.sandbox_id for record in ledger.held) == ("bad id",)
    assert ledger.released_ids == ()
    assert session.close_ids == ["bad id"]
    assert _wmo_cleanup_workers() == before


@pytest.mark.parametrize(
    "raw_id",
    [
        7,
        True,
        None,
        1.5,
        b"raw/id\x00",
        ["nested", 1],
        {"path": "../../escape"},
    ],
)
def test_non_string_id_is_quarantined_and_mismatched_proof_keeps_hold(
    tmp_path: Path,
    raw_id: object,
) -> None:
    """A JSON-scalar protocol violation remains durable and can never fake exact release."""
    session = _NonStringSandboxIdSession(raw_id=raw_id, cleanup_mode="mismatch")
    runtime = HarborEnvironmentRuntime(
        _Factory(cast(_Session, session)),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(HarborCleanupUnprovenError, match="malformed sandbox identity"):
        runtime.open(_task()).__enter__()

    [ledger] = read_ledger_files(tmp_path / "state")
    assert tuple(tuple(record.sandbox_id.split(":")[:2]) for record in ledger.held) == (
        ("invalid-sandbox-id", _invalid_type_tag(raw_id)),
    )
    assert ledger.released_ids == ()
    assert session.close_ids == [raw_id]


@pytest.mark.parametrize(
    ("cleanup_mode", "expected_exception"),
    [
        ("false", HarborCleanupUnprovenError),
        ("timeout", HarborCleanupTimeoutError),
        ("throw", OSError),
    ],
)
def test_non_string_id_false_timeout_and_throw_keep_quarantine_hold(
    tmp_path: Path,
    cleanup_mode: str,
    expected_exception: type[BaseException],
) -> None:
    """Every non-string cleanup failure retains the type-tagged recovery record."""
    session = _NonStringSandboxIdSession(raw_id=7, cleanup_mode=cleanup_mode)
    runtime = HarborEnvironmentRuntime(
        _Factory(cast(_Session, session)),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(expected_exception):
        runtime.open(_task()).__enter__()

    [ledger] = read_ledger_files(tmp_path / "state")
    assert ledger.held[0].sandbox_id.startswith("invalid-sandbox-id:int:")
    assert ledger.released_ids == ()
    assert session.close_ids == [7]


def test_custom_object_identity_is_safely_hashed_and_never_used_as_a_path(tmp_path: Path) -> None:
    """Complex hostile identity metadata yields one bounded ledger filename and held record."""
    raw_id = _CustomIdentity()
    session = _NonStringSandboxIdSession(raw_id=raw_id, cleanup_mode="false")
    runtime = HarborEnvironmentRuntime(
        _Factory(cast(_Session, session)),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(HarborCleanupUnprovenError):
        runtime.open(_task()).__enter__()

    [ledger] = read_ledger_files(tmp_path / "state")
    assert ledger.held[0].sandbox_id.startswith("invalid-sandbox-id:object:")
    assert ledger.path.parent == tmp_path / "state" / "e2b-sandboxes"
    assert len(ledger.path.name) < 128


def test_quarantine_key_never_invokes_mutating_repr_before_bounded_close(tmp_path: Path) -> None:
    """Opaque invalid objects reach close unchanged without running instance serialization code."""
    raw_id = _MutatingReprIdentity()
    session = _NonStringSandboxIdSession(raw_id=raw_id, cleanup_mode="false")
    runtime = HarborEnvironmentRuntime(
        _Factory(cast(_Session, session)),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(HarborCleanupUnprovenError):
        runtime.open(_task()).__enter__()

    assert raw_id.resource == "sandbox-A"
    assert raw_id.repr_calls == 0
    assert session.close_ids == [raw_id]


def test_quarantine_key_never_invokes_any_hostile_instance_protocol(tmp_path: Path) -> None:
    """Opaque quarantine avoids repr, str, iteration, hashing, equality, and properties."""
    raw_id = _HostileProtocolIdentity()
    session = _NonStringSandboxIdSession(raw_id=raw_id, cleanup_mode="false")
    runtime = HarborEnvironmentRuntime(
        _Factory(cast(_Session, session)),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(HarborCleanupUnprovenError):
        runtime.open(_task()).__enter__()

    assert raw_id.calls == []
    assert session.close_ids[0] is raw_id


def test_distinct_hostile_equal_object_cannot_forge_exact_cleanup_proof(tmp_path: Path) -> None:
    """Literal true from user-defined equality never releases the captured object hold."""
    raw_id = _LiarIdentity()
    session = _NonStringSandboxIdSession(raw_id=raw_id, cleanup_mode="liar")
    runtime = HarborEnvironmentRuntime(
        _Factory(cast(_Session, session)),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(HarborCleanupUnprovenError, match="malformed sandbox identity"):
        runtime.open(_task()).__enter__()

    [ledger] = read_ledger_files(tmp_path / "state")
    assert ledger.held[0].sandbox_id.startswith("invalid-sandbox-id:object:")
    assert ledger.released_ids == ()


def test_non_string_exact_typed_cleanup_proof_releases_quarantine_record(tmp_path: Path) -> None:
    """Only exact typed affirmative evidence closes a non-string quarantine hold."""
    raw_id = ["resource", 7]
    session = _NonStringSandboxIdSession(raw_id=raw_id, cleanup_mode="success")
    runtime = HarborEnvironmentRuntime(
        _Factory(cast(_Session, session)),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(ValueError, match="valid nonempty sandbox ID"):
        runtime.open(_task()).__enter__()

    [ledger] = read_ledger_files(tmp_path / "state")
    assert ledger.held == ()
    assert len(ledger.released_ids) == 1
    assert ledger.released_ids[0].startswith("invalid-sandbox-id:list:")


def test_repeated_invalid_values_receive_distinct_occurrence_holds(tmp_path: Path) -> None:
    """A later successful cleanup cannot release an earlier live invalid resource occurrence."""
    failed = _NonStringSandboxIdSession(raw_id=None, cleanup_mode="false")
    succeeded = _NonStringSandboxIdSession(raw_id=None, cleanup_mode="success")
    runtime = HarborEnvironmentRuntime(
        cast(Any, _SequenceFactory([failed, succeeded])),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )

    with pytest.raises(HarborCleanupUnprovenError):
        runtime.open(_task()).__enter__()
    with pytest.raises(ValueError, match="valid nonempty sandbox ID"):
        runtime.open(_task()).__enter__()

    [ledger] = read_ledger_files(tmp_path / "state")
    assert len(ledger.held) == 1
    assert len(ledger.released_ids) == 1
    assert ledger.held[0].sandbox_id != ledger.released_ids[0]
    assert ledger.held[0].sandbox_id.startswith("invalid-sandbox-id:none:")
    assert ledger.released_ids[0].startswith("invalid-sandbox-id:none:")


def test_quarantine_nonce_failure_happens_before_factory_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallible occurrence allocation cannot happen after a remote resource is created."""
    factory = _CountingFactory()
    runtime = HarborEnvironmentRuntime(
        cast(Any, factory),
        environment_id="customer-env",
        template_name="wmo-hb-v1-fixture",
        state_directory=tmp_path / "state",
    )
    monkeypatch.setattr(
        "wmo.runtime.environments.harbor.uuid4",
        lambda: (_ for _ in ()).throw(OSError("randomness unavailable")),
    )

    with pytest.raises(OSError, match="randomness unavailable"):
        runtime.open(_task()).__enter__()

    assert factory.open_calls == 0
    assert not (tmp_path / "state").exists()


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


class _SequenceFactory:
    """Open a finite sequence of protocol-invalid sessions in occurrence order."""

    cleanup_contract: Literal["bounded-close-v1"] = BOUNDED_CLEANUP_CONTRACT

    def __init__(self, sessions: list[_NonStringSandboxIdSession]) -> None:
        self._sessions = iter(sessions)

    def open(self, task: TaskCase, *, template_name: str) -> _NonStringSandboxIdSession:
        assert task == _task()
        assert template_name == "wmo-hb-v1-fixture"
        return next(self._sessions)


class _CountingFactory:
    """Count opens to prove fallible local preflight precedes remote resource creation."""

    cleanup_contract: Literal["bounded-close-v1"] = BOUNDED_CLEANUP_CONTRACT

    def __init__(self) -> None:
        self.open_calls = 0

    def open(self, task: TaskCase, *, template_name: str) -> _Session:
        del task, template_name
        self.open_calls += 1
        return _Session()


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


class _MismatchedCleanupSession(_Session):
    """Return cleanup evidence for a different resource than the captured raw identity."""

    def close(self, *, sandbox_id: str, timeout_seconds: float) -> HarborCleanupResult:
        self.close_ids.append(sandbox_id)
        return HarborCleanupResult(sandbox_id="different-resource", released=True)


class _MutatingSandboxIdSession(_Session):
    """Expose a malformed resource exactly once, then mutate subsequent metadata reads."""

    def __init__(self) -> None:
        self.sandbox_reads = 0
        self.close_ids: list[str] = []
        self._template_id = "wmo-hb-v1-fixture"
        self._cleanup_error = None
        self._cleanup_timed_out = False
        self._cleanup_proven = True

    @property
    def sandbox_id(self) -> str:
        self.sandbox_reads += 1
        return "bad id" if self.sandbox_reads == 1 else "different-resource"


class _NonStringSandboxIdSession:
    """Return a protocol-invalid scalar identity with scripted bounded cleanup evidence."""

    template_id = "wmo-hb-v1-fixture"

    def __init__(self, *, raw_id: object, cleanup_mode: str) -> None:
        self.sandbox_id = raw_id
        self.cleanup_mode = cleanup_mode
        self.close_ids: list[object] = []

    def close(self, *, sandbox_id: object, timeout_seconds: float) -> HarborCleanupResult:
        assert timeout_seconds > 0
        self.close_ids.append(sandbox_id)
        if self.cleanup_mode == "throw":
            raise OSError("close failed")
        return HarborCleanupResult.model_construct(
            sandbox_id=(
                _LiarIdentity()
                if self.cleanup_mode == "liar"
                else "different-resource"
                if self.cleanup_mode == "mismatch"
                else sandbox_id
            ),
            released=self.cleanup_mode not in {"false", "timeout"},
            timed_out=self.cleanup_mode == "timeout",
            failure=None,
        )


class _CustomIdentity:
    """One non-JSON resource identity with path-shaped hostile representation metadata."""

    def __repr__(self) -> str:
        return "../../customer/secrets\n" * 20


class _LiarIdentity:
    """Claim equality with every distinct instance to attack cleanup proof comparison."""

    def __eq__(self, _other: object) -> bool:
        return True

    def __repr__(self) -> str:
        return "liar-identity"


class _MutatingReprIdentity:
    """Mutate resource state if production ever invokes untrusted representation code."""

    def __init__(self) -> None:
        self.resource = "sandbox-A"
        self.repr_calls = 0

    def __repr__(self) -> str:
        self.repr_calls += 1
        self.resource = "sandbox-B"
        return self.resource


class _HostileProtocolIdentity:
    """Fail if quarantine bookkeeping executes any instance-controlled protocol."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _fail(self, name: str) -> Never:
        self.calls.append(name)
        raise AssertionError(f"unexpected hostile protocol call: {name}")

    def __repr__(self) -> str:
        self._fail("repr")

    def __str__(self) -> str:
        self._fail("str")

    def __iter__(self):  # noqa: ANN204
        self._fail("iter")

    def __hash__(self) -> int:
        self._fail("hash")

    def __eq__(self, _other: object) -> bool:
        self._fail("eq")

    @property
    def payload(self) -> str:
        self._fail("property")


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


def _invalid_type_tag(value: object) -> str:
    """Return the fixed production category without invoking value protocols."""
    if value is None:
        return "none"
    return {
        bool: "bool",
        int: "int",
        float: "float",
        bytes: "bytes",
        list: "list",
        dict: "dict",
    }.get(type(value), "object")


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
