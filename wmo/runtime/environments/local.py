"""A bounded JSONL local-process implementation of the executable environment seam.

This adapter gives every episode a new temporary workspace and a long-lived process. It is an
execution boundary, not a security boundary: a customer process still has the permissions of the
user who starts WMO. Customers that need an isolated VM can provide an E2B or other
``EnvironmentRuntime`` implementation instead.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import queue
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import ToolCall
from wmo.common.tasks import TaskCase
from wmo.runtime.environments.interface import EnvironmentSession, Observation

_WORKSPACE_ENVIRONMENT_VARIABLE = "WMO_SANDBOX_WORKSPACE"
_END_OF_STREAM = object()
_ALLOWED_CHILD_ENVIRONMENT_KEYS = frozenset({"LANG", "LC_ALL", "LC_CTYPE", "PATH", "TZ"})
_DEFAULT_CHILD_ENVIRONMENT = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": os.defpath,
    "TZ": "UTC",
}


@dataclass(frozen=True)
class _ProcessIdentity:
    """PID plus kernel start time, which prevents signaling a reused PID."""

    pid: int
    started: tuple[int, int]


@dataclass(frozen=True)
class _ProcessRecord:
    """One process-tree snapshot row."""

    identity: _ProcessIdentity
    parent_pid: int
    zombie: bool


class _KqueueEvent(Protocol):
    fflags: int


class _Kqueue(Protocol):
    """The Darwin queue operations owned by one descendant tracker."""

    def close(self) -> None: ...

    def control(self, changes: object, max_events: int, timeout: int) -> list[_KqueueEvent]: ...


class _KqueueFactory(Protocol):
    def __call__(self) -> _Kqueue: ...


class _KeventFactory(Protocol):
    """Create one Darwin process-notification registration event."""

    def __call__(self, ident: int, *, filter: int, flags: int, fflags: int) -> object: ...


@dataclass(frozen=True)
class _DarwinKqueueBindings:
    """Dynamically loaded Darwin-only ``select`` bindings for containment proof."""

    kqueue: _KqueueFactory
    kevent: _KeventFactory
    filter_process: int
    event_add: int
    event_clear: int
    note_fork: int
    note_exit: int


class _DarwinProcBsdInfo(ctypes.Structure):
    """ABI layout returned by Darwin ``PROC_PIDTBSDINFO``."""

    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("xstatus", ctypes.c_uint32),
        ("pid", ctypes.c_uint32),
        ("ppid", ctypes.c_uint32),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        ("ruid", ctypes.c_uint32),
        ("rgid", ctypes.c_uint32),
        ("svuid", ctypes.c_uint32),
        ("svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("comm", ctypes.c_char * 16),
        ("name", ctypes.c_char * 32),
        ("nfiles", ctypes.c_uint32),
        ("pgid", ctypes.c_uint32),
        ("pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("nice", ctypes.c_int32),
        ("start_seconds", ctypes.c_uint64),
        ("start_microseconds", ctypes.c_uint64),
    ]


class _DescendantTracker:
    """Track exact descendants and kernel fork evidence without trusting numeric groups."""

    def __init__(
        self,
        root_identity: _ProcessIdentity,
        bindings: _DarwinKqueueBindings,
    ) -> None:
        snapshot = _process_snapshot()
        root = snapshot.get(root_identity.pid)
        if root is None or root.identity != root_identity:
            raise LocalProcessCleanupError("could not identify local environment process")
        self._known = {root.identity}
        self._root = root.identity
        self._fork_observed = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._failure: Exception | None = None
        self._bindings = bindings
        self._kqueue = bindings.kqueue()
        try:
            event = bindings.kevent(
                root_identity.pid,
                filter=bindings.filter_process,
                flags=bindings.event_add | bindings.event_clear,
                fflags=bindings.note_fork | bindings.note_exit,
            )
            self._kqueue.control((event,), 0, 0)
            self._thread = threading.Thread(
                target=self._monitor,
                name=f"wmo-local-descendants-{root_identity.pid}",
                daemon=True,
            )
            self._thread.start()
        except BaseException:  # constructor must close its partially-owned FD
            self._kqueue.close()
            raise

    def live_identities(self) -> tuple[_ProcessIdentity, ...]:
        """Refresh and return only identities still owned by the tracked tree."""
        self._refresh()
        if self._failure is not None:
            raise LocalProcessCleanupError("local descendant monitor failed") from self._failure
        snapshot = _process_snapshot()
        with self._lock:
            return tuple(
                identity
                for identity in self._known
                if (record := snapshot.get(identity.pid)) is not None
                and record.identity == identity
                and not record.zombie
            )

    def stop(self, timeout_seconds: float) -> str | None:
        """Stop the monitor and return any immutable fork-containment failure evidence."""
        self._stop.set()
        self._thread.join(timeout_seconds)
        if self._thread.is_alive():
            raise LocalProcessCleanupError("local descendant monitor did not stop")
        try:
            if self._failure is not None:
                raise LocalProcessCleanupError("local descendant monitor failed") from self._failure
            return self.containment_failure()
        finally:
            self._kqueue.close()

    def containment_failure(self) -> str | None:
        """Return immutable root PID evidence when any fork prevents containment proof."""
        self._read_kernel_events()
        with self._lock:
            if not self._fork_observed:
                return None
            return (
                f"fork observed from root pid={self._root.pid} "
                f"start={self._root.started[0]}.{self._root.started[1]}"
            )

    def _monitor(self) -> None:
        """Poll the kernel tree while the root is alive and through teardown."""
        try:
            while not self._stop.wait(0.005):
                self._read_kernel_events()
                self._refresh()
        except Exception as exc:  # noqa: BLE001 - surfaced synchronously during cleanup
            self._failure = exc
            self._stop.set()

    def _read_kernel_events(self) -> None:
        """Latch kernel fork evidence even when no numeric child PID remains observable."""
        for event in self._kqueue.control(None, 8, 0):
            if event.fflags & self._bindings.note_fork:
                with self._lock:
                    self._fork_observed = True

    def _refresh(self) -> None:
        """Add recursively reachable children without forgetting detached identities."""
        snapshot = _process_snapshot()
        with self._lock:
            live_parent_pids = {
                identity.pid
                for identity in self._known
                if (record := snapshot.get(identity.pid)) is not None
                and record.identity == identity
            }
            changed = True
            while changed:
                changed = False
                for record in snapshot.values():
                    if record.parent_pid not in live_parent_pids:
                        continue
                    if record.identity not in self._known:
                        self._known.add(record.identity)
                        changed = True
                    live_parent_pids.add(record.identity.pid)


def _process_snapshot() -> dict[int, _ProcessRecord]:
    """Read an identity-bearing process table without invoking a shell or ``ps``."""
    if sys.platform == "darwin":
        return _darwin_process_snapshot()
    if sys.platform.startswith("linux"):
        return _linux_process_snapshot()
    raise LocalProcessCleanupError("local descendant tracking requires Darwin or Linux")


def _require_containment_support() -> _DarwinKqueueBindings:
    """Load required Darwin queue bindings before allocating process resources."""
    if sys.platform != "darwin":
        raise LocalProcessCleanupError(
            "local process containment requires Darwin kernel fork notifications"
        )

    def binding(name: str) -> object:
        return getattr(select, name)

    try:
        return _DarwinKqueueBindings(
            kqueue=cast(_KqueueFactory, binding("kqueue")),
            kevent=cast(_KeventFactory, binding("kevent")),
            filter_process=cast(int, binding("KQ_FILTER_PROC")),
            event_add=cast(int, binding("KQ_EV_ADD")),
            event_clear=cast(int, binding("KQ_EV_CLEAR")),
            note_fork=cast(int, binding("KQ_NOTE_FORK")),
            note_exit=cast(int, binding("KQ_NOTE_EXIT")),
        )
    except AttributeError as error:
        raise LocalProcessCleanupError(
            "local process containment requires Darwin kernel fork notifications"
        ) from error


def _read_process_identity(pid: int) -> _ProcessIdentity:
    """Read one gated child identity independently from descendant-table initialization."""
    if sys.platform != "darwin":  # pragma: no cover - support preflight rejects other platforms
        raise LocalProcessCleanupError("local process identity requires Darwin libproc")
    record = _darwin_process_record(pid)
    if record is None:
        raise LocalProcessCleanupError("could not read gated local process identity")
    return record.identity


def _darwin_process_snapshot() -> dict[int, _ProcessRecord]:
    """Read Darwin's libproc table with PID start timestamps."""
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    count = library.proc_listallpids(None, 0)
    if count <= 0:
        raise OSError(ctypes.get_errno(), "proc_listallpids failed")
    pids = (ctypes.c_int * (count + 64))()
    returned = library.proc_listallpids(pids, ctypes.sizeof(pids))
    records: dict[int, _ProcessRecord] = {}
    for pid in pids[: max(0, returned)]:
        if pid <= 0:
            continue
        if record := _darwin_process_record(pid, library=library):
            records[pid] = record
    return records


def _darwin_process_record(
    pid: int,
    *,
    library: ctypes.CDLL | None = None,
) -> _ProcessRecord | None:
    """Read one Darwin process record with its kernel start timestamp."""
    proc = library or ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    info = _DarwinProcBsdInfo()
    size = proc.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
    if size != ctypes.sizeof(info):
        return None
    identity = _ProcessIdentity(pid=pid, started=(info.start_seconds, info.start_microseconds))
    return _ProcessRecord(
        identity=identity,
        parent_pid=info.ppid,
        zombie=info.status == 5,
    )


def _linux_process_snapshot() -> dict[int, _ProcessRecord]:
    """Read Linux procfs process relationships and kernel start ticks."""
    records: dict[int, _ProcessRecord] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2 :].split()
            pid = int(entry.name)
            identity = _ProcessIdentity(pid=pid, started=(int(fields[19]), 0))
            records[pid] = _ProcessRecord(
                identity=identity,
                parent_pid=int(fields[1]),
                zombie=fields[0] == "Z",
            )
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return records


def _signal_exact_identity(identity: _ProcessIdentity, signal_number: int) -> None:
    """Signal a PID only while its kernel start identity still matches the tracked process."""
    record = _process_snapshot().get(identity.pid)
    if record is None or record.identity != identity or record.zombie:
        return
    try:
        os.kill(identity.pid, signal_number)
    except ProcessLookupError:
        return


def _close_fd(file_descriptor: int) -> None:
    """Close one owned descriptor while treating an already-closed descriptor as complete."""
    try:
        os.close(file_descriptor)
    except OSError:
        return


class LocalProcessProtocolError(RuntimeError):
    """The local executable did not follow the JSONL environment protocol."""


class LocalProcessCrashError(RuntimeError):
    """The local executable exited before it returned the required response."""


class LocalProcessCleanupError(RuntimeError):
    """The local executable or its temporary workspace could not be cleaned up."""


@dataclass(frozen=True)
class LocalProcessLimits:
    """Finite resource limits applied to each local executable environment session.

    Args:
        request_timeout_seconds: Wall-clock limit for one protocol request and response.
        session_timeout_seconds: Wall-clock limit for the complete local process lifetime.
        cleanup_timeout_seconds: Grace period before a live process is force-killed.
        maximum_output_bytes: Largest accepted JSON response line from the environment.
        maximum_stderr_bytes: Bounded tail retained only to drain the child error pipe.
    """

    request_timeout_seconds: float = 30.0
    session_timeout_seconds: float = 300.0
    cleanup_timeout_seconds: float = 5.0
    maximum_output_bytes: int = 1_000_000
    maximum_stderr_bytes: int = 32_768

    def __post_init__(self) -> None:
        """Validate all resource bounds before a local child process can start."""
        for name, value in (
            ("request_timeout_seconds", self.request_timeout_seconds),
            ("session_timeout_seconds", self.session_timeout_seconds),
            ("cleanup_timeout_seconds", self.cleanup_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.request_timeout_seconds > self.session_timeout_seconds:
            raise ValueError("request_timeout_seconds cannot exceed session_timeout_seconds")
        for name, value in (
            ("maximum_output_bytes", self.maximum_output_bytes),
            ("maximum_stderr_bytes", self.maximum_stderr_bytes),
        ):
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class LocalProcessEnvironmentRuntime:
    """Creates one fresh JSONL process and temporary workspace per executable episode.

    The child receives one line for ``open``, one per ``execute`` request, and one best-effort
    ``close`` line. It must respond to ``open`` with ``{"ready": true}``, then respond to every
    execution request with a canonical :class:`Observation` JSON object. The command is supplied
    by the customer and is never persisted in a simulation artifact.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        limits: LocalProcessLimits | None = None,
        workspace_parent: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        """Configure the local executable without launching it yet.

        Args:
            command: Direct executable argv. Shell parsing is intentionally unsupported.
            limits: Per-session lifetime, response, and cleanup bounds.
            workspace_parent: Optional parent for generated per-episode workspaces.
            environment: Optional overrides for the documented locale, path, and timezone
                allowlist. Parent credentials and arbitrary variables are never inherited.
        """
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("local environment command must be a nonempty sequence of strings")
        if environment is not None and any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ValueError("local environment variables must be string-to-string mappings")
        unsupported = sorted(set(environment or {}).difference(_ALLOWED_CHILD_ENVIRONMENT_KEYS))
        if unsupported:
            raise ValueError(
                "local environment variables are outside the safe allowlist: "
                + ", ".join(unsupported)
            )
        self._command = tuple(command)
        self._limits = limits or LocalProcessLimits()
        self._workspace_parent = workspace_parent
        self._environment = dict(environment or {})

    def open(self, task: TaskCase) -> AbstractContextManager[EnvironmentSession]:
        """Return a simulator-owned context that opens a fresh child only when entered.

        Args:
            task: Task whose canonical JSON is supplied in the JSONL ``open`` request.

        Returns:
            A context manager that always terminates the child and removes its workspace.
        """
        return _LocalProcessContext(
            self._command,
            self._limits,
            self._workspace_parent,
            self._environment,
            task,
        )


class _LocalProcessContext(AbstractContextManager[EnvironmentSession]):
    """One context manager that starts and later proves cleanup of one local child process."""

    def __init__(
        self,
        command: tuple[str, ...],
        limits: LocalProcessLimits,
        workspace_parent: Path | None,
        environment: Mapping[str, str],
        task: TaskCase,
    ) -> None:
        self._command = command
        self._limits = limits
        self._workspace_parent = workspace_parent
        self._environment = environment
        self._task = task
        self._session: _LocalProcessSession | None = None

    def __enter__(self) -> EnvironmentSession:
        """Start the child and complete its bounded protocol handshake."""
        session = _LocalProcessSession(
            self._command,
            self._limits,
            self._workspace_parent,
            self._environment,
        )
        self._session = session
        try:
            session.start(self._task)
        except BaseException:
            with suppress(LocalProcessCleanupError):
                session.close()
            raise
        return session

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Terminate the child and remove its private workspace on every exit path."""
        del exception_type, exception, traceback
        if self._session is not None:
            self._session.close()
        return False


class _LocalProcessSession:
    """One stateful JSONL protocol session over a bounded child process."""

    def __init__(
        self,
        command: tuple[str, ...],
        limits: LocalProcessLimits,
        workspace_parent: Path | None,
        environment: Mapping[str, str],
    ) -> None:
        self._command = command
        self._limits = limits
        self._workspace_parent = workspace_parent
        self._environment = dict(environment)
        self._process: subprocess.Popen[bytes] | None = None
        self._descendants: _DescendantTracker | None = None
        self._workspace: Path | None = None
        self._stdout: queue.Queue[bytes | object] = queue.Queue(maxsize=1)
        self._stderr_tail = bytearray()
        self._stderr_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._io_stop = threading.Event()
        self._io_threads: list[threading.Thread] = []
        self._started_at: float | None = None
        self._closed = False
        self._termination_complete = False

    def start(self, task: TaskCase) -> None:
        """Create workspace, launch the child, and verify the mandatory ready response."""
        bindings = _require_containment_support()
        workspace_parent = self._workspace_parent
        if workspace_parent is not None:
            workspace_parent.mkdir(parents=True, exist_ok=True)
            if (
                workspace_parent.is_symlink()
                or not workspace_parent.is_dir()
                or workspace_parent.resolve() != workspace_parent.absolute()
            ):
                raise LocalProcessProtocolError(
                    "local environment workspace parent must be a real directory"
                )
        self._workspace = Path(tempfile.mkdtemp(prefix="wmo-sandbox-", dir=workspace_parent))
        child_environment = dict(_DEFAULT_CHILD_ENVIRONMENT)
        child_environment.update(self._environment)
        child_environment[_WORKSPACE_ENVIRONMENT_VARIABLE] = str(self._workspace)
        child_environment["HOME"] = str(self._workspace)
        child_environment["TMPDIR"] = str(self._workspace)
        gate_read, gate_write = os.pipe()
        gate_program = Path(__file__).with_name("local_exec_gate.py")
        try:
            self._process = subprocess.Popen(
                (sys.executable, str(gate_program), str(gate_read), *self._command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._workspace,
                env=child_environment,
                start_new_session=True,
                pass_fds=(gate_read,),
            )
        except OSError as error:
            _close_fd(gate_read)
            _close_fd(gate_write)
            launch_error = LocalProcessCrashError(
                f"local environment process could not start: {type(error).__name__}"
            )
            try:
                self._remove_workspace()
            except OSError as cleanup_error:
                launch_error.add_note(
                    "local startup cleanup evidence: workspace cleanup failed with "
                    f"{type(cleanup_error).__name__}"
                )
            raise launch_error from error
        _close_fd(gate_read)
        root_identity: _ProcessIdentity | None = None
        try:
            root_identity = _read_process_identity(self._process.pid)
            self._descendants = _DescendantTracker(root_identity, bindings)
            os.write(gate_write, b"1")
            _close_fd(gate_write)
        except BaseException as error:  # preserve startup errors after no-fail cleanup
            evidence = self._abort_gated_start(gate_write, root_identity)
            if evidence:
                error.add_note("local startup cleanup evidence: " + "; ".join(evidence))
            raise
        self._started_at = time.monotonic()
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        assert self._process.stdin is not None
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            os.set_blocking(stream.fileno(), False)
        stdout_thread = threading.Thread(
            target=self._drain_stdout,
            args=(self._process.stdout.fileno(),),
            name="wmo-local-environment-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process.stderr.fileno(),),
            name="wmo-local-environment-stderr",
            daemon=True,
        )
        self._io_threads = [stdout_thread, stderr_thread]
        for thread in self._io_threads:
            thread.start()
        response = self._request({"kind": "open", "task": task.model_dump(mode="json")})
        if response != {"ready": True}:
            raise LocalProcessProtocolError(
                "local environment open response must be {'ready': true}"
            )

    def _abort_gated_start(
        self,
        gate_write: int,
        root_identity: _ProcessIdentity | None,
    ) -> tuple[str, ...]:
        """Close a failed launch without masking its original exception or leaking resources."""
        evidence: list[str] = []
        deadline = time.monotonic() + self._limits.cleanup_timeout_seconds
        _close_fd(gate_write)
        process = self._process
        if process is not None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                if root_identity is None:
                    evidence.append("gated root identity unavailable; signal withheld")
                else:
                    try:
                        _signal_exact_identity(root_identity, signal.SIGKILL)
                    except Exception as error:  # noqa: BLE001 - retained as cleanup evidence
                        evidence.append(f"identity signal failed with {type(error).__name__}")
                try:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    evidence.append("gated root did not exit before cleanup deadline")
                except BaseException as error:  # noqa: BLE001 - cleanup evidence must not mask
                    evidence.append(f"bounded root wait failed with {type(error).__name__}")
            except BaseException as error:  # noqa: BLE001 - cleanup evidence must not mask
                evidence.append(f"initial root wait failed with {type(error).__name__}")
            tracker = self._descendants
            if tracker is not None:
                try:
                    containment = tracker.stop(max(0.0, deadline - time.monotonic()))
                    if containment:
                        evidence.append(containment)
                except BaseException as error:  # noqa: BLE001 - cleanup evidence must not mask
                    evidence.append(f"tracker cleanup failed with {type(error).__name__}")
            try:
                self._close_unstarted_pipes(process, evidence)
            except BaseException as error:  # noqa: BLE001 - cleanup evidence must not mask
                evidence.append(f"pipe cleanup failed with {type(error).__name__}")
        try:
            self._remove_workspace()
        except BaseException as error:  # noqa: BLE001 - cleanup evidence must not mask
            evidence.append(f"workspace cleanup failed with {type(error).__name__}")
        self._descendants = None
        self._closed = True
        try:
            self._termination_complete = process is None or process.poll() is not None
        except BaseException as error:  # noqa: BLE001 - cleanup evidence must not mask
            evidence.append(f"root status check failed with {type(error).__name__}")
            self._termination_complete = False
        return tuple(evidence)

    @staticmethod
    def _close_unstarted_pipes(
        process: subprocess.Popen[bytes],
        evidence: list[str],
    ) -> None:
        """Close raw pipes created before any reader worker can exist."""
        for attribute in ("stdin", "stdout", "stderr"):
            stream = getattr(process, attribute)
            if stream is None:
                continue
            try:
                stream.raw.close()
            except OSError as error:
                evidence.append(f"{attribute} close failed with {type(error).__name__}")
            setattr(process, attribute, None)

    def execute(self, action: ToolCall) -> Observation:
        """Send one canonical tool call to the child and parse its bounded observation response."""
        response = self._request({"kind": "execute", "action": action.model_dump(mode="json")})
        try:
            return Observation.model_validate(response)
        except ValueError as error:
            raise LocalProcessProtocolError(
                "local environment execute response must be a canonical Observation"
            ) from error

    def close(self) -> None:
        """Best-effort protocol close followed by mandatory process and workspace cleanup."""
        if self._closed:
            return
        self._closed = True
        cleanup_errors: list[str] = []
        try:
            self._send_close()
        except (OSError, ValueError):
            cleanup_errors.append("close protocol request failed")
        try:
            self._terminate_process()
        except (
            LocalProcessCleanupError,
            OSError,
            ValueError,
            subprocess.TimeoutExpired,
        ) as error:
            cleanup_errors.append(f"child process termination failed: {error}")
        try:
            self._remove_workspace()
        except OSError:
            cleanup_errors.append("temporary workspace removal failed")
        if cleanup_errors:
            rendered = ", ".join(cleanup_errors)
            raise LocalProcessCleanupError(f"local environment cleanup failed: {rendered}")

    def _request(self, payload: JsonObject) -> JsonObject:
        """Write one JSONL request and return the next bounded object response."""
        if self._closed:
            raise LocalProcessProtocolError("local environment session is already closed")
        self._require_within_session_limit()
        process = self._require_process()
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        with self._request_lock:
            if process.poll() is not None:
                raise self._crash_error(process)
            stdin = process.stdin
            if stdin is None:
                raise LocalProcessCrashError("local environment process has no stdin pipe")
            try:
                self._write_pipe(stdin.fileno(), encoded, self._remaining_timeout())
            except (OSError, TimeoutError) as error:
                raise LocalProcessCrashError(
                    f"local environment request pipe failed: {type(error).__name__}"
                ) from error
            return self._read_response(process)

    def _read_response(self, process: subprocess.Popen[bytes]) -> JsonObject:
        """Wait for one output line without allowing a child to block an episode indefinitely."""
        timeout = self._remaining_timeout()
        try:
            value = self._stdout.get(timeout=timeout)
        except queue.Empty as error:
            self._terminate_process()
            raise TimeoutError(
                "local environment response exceeded its configured timeout"
            ) from error
        if value is _END_OF_STREAM:
            raise self._crash_error(process)
        line = cast(bytes, value)
        if len(line) > self._limits.maximum_output_bytes:
            self._terminate_process()
            raise LocalProcessProtocolError(
                "local environment response exceeded maximum_output_bytes"
            )
        if not line.endswith(b"\n"):
            raise LocalProcessProtocolError(
                "local environment response must be one newline-delimited JSON object"
            )
        try:
            decoded = json.loads(line)
        except (TypeError, ValueError) as error:
            raise LocalProcessProtocolError(
                "local environment response is not valid JSON"
            ) from error
        if not isinstance(decoded, dict):
            raise LocalProcessProtocolError("local environment response must be a JSON object")
        return cast(JsonObject, decoded)

    def _send_close(self) -> None:
        """Write a non-blocking close hint while preserving cleanup authority in this process."""
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            return
        try:
            os.write(process.stdin.fileno(), b'{"kind":"close"}\n')
        except (BlockingIOError, OSError):
            return

    def _terminate_process(self) -> None:
        """Terminate every identity-tracked descendant, including detached process groups."""
        process = self._process
        if process is None or self._termination_complete:
            return
        tracker = self._descendants
        cleanup_error: Exception | None = None
        started = time.monotonic()
        deadline = started + self._limits.cleanup_timeout_seconds
        try:
            if tracker is None:
                raise LocalProcessCleanupError("local descendant tracker is unavailable")
            term_deadline = started + self._limits.cleanup_timeout_seconds * 0.25
            if not self._signal_until_empty(process, tracker, signal.SIGTERM, term_deadline):
                kill_deadline = started + self._limits.cleanup_timeout_seconds * 0.75
                if not self._signal_until_empty(process, tracker, signal.SIGKILL, kill_deadline):
                    raise subprocess.TimeoutExpired(
                        self._command,
                        self._limits.cleanup_timeout_seconds,
                    )
        except Exception as exc:  # noqa: BLE001 - re-raised after the monitor is joined
            cleanup_error = exc
        finally:
            if tracker is not None:
                try:
                    evidence = tracker.stop(max(0.0, deadline - time.monotonic()))
                    if evidence:
                        containment_error = LocalProcessCleanupError(
                            f"local process containment is unproven: {evidence}"
                        )
                        cleanup_error = cleanup_error or containment_error
                except Exception as exc:  # noqa: BLE001 - combined with termination evidence
                    cleanup_error = cleanup_error or exc
            self._descendants = None
            try:
                self._close_pipe_workers(process, deadline)
            except Exception as exc:  # noqa: BLE001 - combined with termination evidence
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            raise cleanup_error
        self._termination_complete = True

    def _signal_until_empty(
        self,
        process: subprocess.Popen[bytes],
        tracker: _DescendantTracker,
        signal_number: int,
        deadline: float,
    ) -> bool:
        """Re-scan and signal exact live identities until none remain or grace expires."""
        while True:
            identities = tracker.live_identities()
            for identity in identities:
                _signal_exact_identity(identity, signal_number)
            leader_exited = process.poll() is not None
            if leader_exited and not tracker.live_identities():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if not leader_exited:
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=min(0.01, remaining))
            else:
                time.sleep(min(0.005, remaining))

    def _close_pipe_workers(
        self,
        process: subprocess.Popen[bytes],
        deadline: float,
    ) -> None:
        """Stop nonblocking pipe readers and close raw FDs within one cleanup grace."""
        self._io_stop.set()
        worker_failed = False
        for thread in self._io_threads:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                worker_failed = True
        for attribute in ("stdin", "stdout", "stderr"):
            stream = getattr(process, attribute)
            if stream is None:
                continue
            stream.raw.close()
            setattr(process, attribute, None)
        if worker_failed:
            raise LocalProcessCleanupError("local pipe worker did not stop before cleanup deadline")

    @staticmethod
    def _write_pipe(file_descriptor: int, payload: bytes, timeout_seconds: float) -> None:
        """Write a complete request through a nonblocking pipe within a finite wait."""
        deadline = time.monotonic() + timeout_seconds
        offset = 0
        while offset < len(payload):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("local environment request pipe write timed out")
            _, writable, _ = select.select((), (file_descriptor,), (), remaining)
            if not writable:
                raise TimeoutError("local environment request pipe write timed out")
            offset += os.write(file_descriptor, payload[offset:])

    def _remove_workspace(self) -> None:
        """Delete only the uniquely-created per-session workspace after process termination."""
        if self._workspace is None:
            return
        shutil.rmtree(self._workspace)
        self._workspace = None

    def _require_process(self) -> subprocess.Popen[bytes]:
        """Return the active process or fail before attempting an invalid pipe operation."""
        if self._process is None:
            raise LocalProcessProtocolError("local environment session has not started")
        return self._process

    def _require_within_session_limit(self) -> None:
        """Fail and terminate a child whose complete episode budget has expired."""
        if self._started_at is None:
            raise LocalProcessProtocolError("local environment session has not started")
        if time.monotonic() - self._started_at <= self._limits.session_timeout_seconds:
            return
        self._terminate_process()
        raise TimeoutError("local environment session exceeded its configured timeout")

    def _remaining_timeout(self) -> float:
        """Return the smaller request and complete-session wait budget."""
        if self._started_at is None:
            raise LocalProcessProtocolError("local environment session has not started")
        elapsed = time.monotonic() - self._started_at
        remaining_session = self._limits.session_timeout_seconds - elapsed
        if remaining_session <= 0:
            self._terminate_process()
            raise TimeoutError("local environment session exceeded its configured timeout")
        return min(self._limits.request_timeout_seconds, remaining_session)

    def _crash_error(self, process: subprocess.Popen[bytes]) -> LocalProcessCrashError:
        """Build a non-secret crash error without embedding potentially sensitive child stderr."""
        return LocalProcessCrashError(
            f"local environment process exited before its response (return code {process.poll()})"
        )

    def _drain_stdout(self, file_descriptor: int) -> None:
        """Move framed output through a nonblocking reader that cleanup can always join."""
        buffered = bytearray()
        try:
            while not self._io_stop.is_set():
                readable, _, _ = select.select((file_descriptor,), (), (), 0.01)
                if not readable:
                    continue
                chunk = os.read(file_descriptor, min(65_536, self._limits.maximum_output_bytes + 1))
                if not chunk:
                    return
                buffered.extend(chunk)
                newline = buffered.find(b"\n")
                if newline >= 0:
                    self._queue_stdout(bytes(buffered[: newline + 1]))
                    del buffered[: newline + 1]
                elif len(buffered) > self._limits.maximum_output_bytes:
                    self._queue_stdout(bytes(buffered))
                    buffered.clear()
        except (OSError, ValueError):
            return
        finally:
            self._queue_stdout(_END_OF_STREAM)

    def _drain_stderr(self, file_descriptor: int) -> None:
        """Drain stderr nonblockingly while retaining only a bounded tail."""
        try:
            while not self._io_stop.is_set():
                readable, _, _ = select.select((file_descriptor,), (), (), 0.01)
                if not readable:
                    continue
                chunk = os.read(file_descriptor, 4_096)
                if not chunk:
                    return
                with self._stderr_lock:
                    self._stderr_tail.extend(chunk)
                    excess = len(self._stderr_tail) - self._limits.maximum_stderr_bytes
                    if excess > 0:
                        del self._stderr_tail[:excess]
        except (OSError, ValueError):
            return

    def _queue_stdout(self, value: bytes | object) -> None:
        """Publish one bounded protocol item without letting a malicious writer block cleanup."""
        try:
            self._stdout.put_nowait(value)
        except queue.Full:
            return
