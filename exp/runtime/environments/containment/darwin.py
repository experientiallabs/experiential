"""Darwin process containment proven with kernel fork notifications and libproc identities.

Darwin offers no unprivileged process namespace, so containment is proven instead of enforced:
a gate keeps the customer command blocked until a kqueue fork monitor is attached, cleanup
signals only exact PID plus start-time identities, and any observed fork fails cleanup closed
because a forked descendant cannot be proven reachable.
"""

from __future__ import annotations

import ctypes
import os
import select
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from exp.common.rollouts.containment import (
    ContainmentGuarantee,
    ContainmentReport,
    containment_report,
)
from exp.runtime.environments.containment.contract import (
    LocalProcessCleanupError,
    LocalProcessCrashError,
    ProcessLaunch,
    close_fd,
)


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
    """One kernel queue event; only its filter flags are read.

    Attributes:
        fflags: Process-notification flags such as fork or exit.
    """

    fflags: int


class _Kqueue(Protocol):
    """The Darwin queue operations owned by one descendant tracker."""

    def close(self) -> None:
        """Close the queue descriptor."""
        ...

    def control(self, changes: object, max_events: int, timeout: int) -> list[_KqueueEvent]:
        """Apply registrations and return up to ``max_events`` ready events."""
        ...


class _KqueueFactory(Protocol):
    """Create one Darwin kernel queue."""

    def __call__(self) -> _Kqueue:
        """Return a new kernel queue."""
        ...


class _KeventFactory(Protocol):
    """Create one Darwin process-notification registration event."""

    def __call__(self, ident: int, *, filter: int, flags: int, fflags: int) -> object:
        """Return one registration for process ``ident``."""
        ...


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
        """Verify the root identity, register fork and exit notifications, and start polling."""
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
                name=f"exp-local-descendants-{root_identity.pid}",
                daemon=True,
            )
            self._thread.start()
        except BaseException:  # noqa: BLE001 - constructor must close its partially-owned FD
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
    raise LocalProcessCleanupError("local descendant tracking requires Darwin libproc")


def _require_containment_support() -> _DarwinKqueueBindings:
    """Load required Darwin queue bindings before allocating process resources."""
    if sys.platform != "darwin":
        raise LocalProcessCleanupError(
            "local process containment requires Darwin kernel fork notifications"
        )

    def binding(name: str) -> object:
        """Read one required Darwin-only ``select`` attribute."""
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


def _signal_exact_identity(identity: _ProcessIdentity, signal_number: int) -> None:
    """Signal a PID only while its kernel start identity still matches the tracked process."""
    record = _process_snapshot().get(identity.pid)
    if record is None or record.identity != identity or record.zombie:
        return
    try:
        os.kill(identity.pid, signal_number)
    except ProcessLookupError:
        return


DARWIN_CONTAINMENT_PROTOCOL = "darwin-kqueue-v1"
_DARWIN_UNAVAILABLE = {
    ContainmentGuarantee.DESCENDANT_CONTAINMENT: (
        "Darwin has no unprivileged process namespace; any fork fails the session closed"
    ),
    ContainmentGuarantee.CPU_TIME_LIMIT: "Darwin local execution applies no resource limits",
    ContainmentGuarantee.MEMORY_LIMIT: "Darwin local execution applies no resource limits",
    ContainmentGuarantee.FILE_SIZE_LIMIT: "Darwin local execution applies no resource limits",
    ContainmentGuarantee.PROCESS_COUNT_LIMIT: "Darwin local execution applies no resource limits",
    ContainmentGuarantee.NO_NEW_PRIVILEGES: "Darwin has no no-new-privileges process control",
    ContainmentGuarantee.HOST_PROCESS_ISOLATION: (
        "Darwin local processes keep the user's visibility of other processes"
    ),
    ContainmentGuarantee.FILESYSTEM_WRITE_CONFINEMENT: (
        "Darwin local execution does not confine filesystem writes"
    ),
    ContainmentGuarantee.NETWORK_DENIED: "Darwin local execution does not restrict network access",
}


@dataclass(frozen=True)
class DarwinContainment:
    """The detected Darwin backend: its report and the kernel bindings every launch uses.

    Attributes:
        report: Startup detection report.
        bindings: Loaded kqueue bindings, or None when the report refuses local execution.
    """

    report: ContainmentReport
    bindings: _DarwinKqueueBindings | None

    def launch(self, launch: ProcessLaunch) -> DarwinContainedProcess:
        """Start the gated command with fork monitoring attached.

        Args:
            launch: Command, private workspace, environment, and startup cleanup bound.

        Returns:
            The running contained process.

        Raises:
            LocalProcessCleanupError: The host refused containment.
        """
        if self.bindings is None:
            raise LocalProcessCleanupError(self.report.refusal or "Darwin containment refused")
        return launch_darwin(launch, self.bindings)


def detect_darwin_containment() -> DarwinContainment:
    """Load the Darwin kernel bindings that containment proof requires and report the result.

    Returns:
        The Darwin backend; its report carries a refusal when the bindings are missing.
    """
    try:
        bindings = _require_containment_support()
    except LocalProcessCleanupError as error:
        refused = containment_report(
            platform=sys.platform,
            protocol=DARWIN_CONTAINMENT_PROTOCOL,
            enforced=(),
            reasons={},
            mechanisms=(),
            refusal=str(error),
        )
        return DarwinContainment(report=refused, bindings=None)
    report = containment_report(
        platform=sys.platform,
        protocol=DARWIN_CONTAINMENT_PROTOCOL,
        enforced=set(ContainmentGuarantee).difference(_DARWIN_UNAVAILABLE),
        reasons=_DARWIN_UNAVAILABLE,
        mechanisms=("exec_gate", "kqueue_fork_notifications", "libproc_identities", "new_session"),
    )
    return DarwinContainment(report=report, bindings=bindings)


def launch_darwin(
    launch: ProcessLaunch,
    bindings: _DarwinKqueueBindings,
) -> DarwinContainedProcess:
    """Start the command behind the exec gate and attach fork monitoring before it runs.

    Args:
        launch: Command, private workspace, environment, and startup cleanup bound.
        bindings: Darwin kqueue bindings loaded by :func:`detect_darwin_containment`.

    Returns:
        The running contained process with its descendant tracker attached.

    Raises:
        LocalProcessCrashError: The gate process could not be started.
    """
    gate_read, gate_write = os.pipe()
    gate_program = Path(__file__).with_name("darwin_gate.py")
    try:
        process = subprocess.Popen(
            (sys.executable, str(gate_program), str(gate_read), *launch.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=launch.workspace,
            env=dict(launch.environment),
            start_new_session=True,
            pass_fds=(gate_read,),
        )
    except OSError as error:
        close_fd(gate_read)
        close_fd(gate_write)
        raise LocalProcessCrashError(
            f"local environment process could not start: {type(error).__name__}"
        ) from error
    close_fd(gate_read)
    root_identity: _ProcessIdentity | None = None
    tracker: _DescendantTracker | None = None
    try:
        root_identity = _read_process_identity(process.pid)
        tracker = _DescendantTracker(root_identity, bindings)
        os.write(gate_write, b"1")
        close_fd(gate_write)
    except BaseException as error:  # noqa: BLE001 - preserve startup errors after no-fail cleanup
        evidence = _abort_gated_start(
            process,
            tracker,
            gate_write,
            root_identity,
            launch.cleanup_timeout_seconds,
        )
        if evidence:
            error.add_note("local startup cleanup evidence: " + "; ".join(evidence))
        raise
    return DarwinContainedProcess(process, tracker, launch.command)


class DarwinContainedProcess:
    """One gated Darwin process whose exact descendants are tracked until cleanup proof."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        tracker: _DescendantTracker,
        command: tuple[str, ...],
    ) -> None:
        """Hold the gated root, its attached tracker, and the argv used in timeout errors."""
        self._process = process
        self._tracker: _DescendantTracker | None = tracker
        self._command = command

    @property
    def process(self) -> subprocess.Popen[bytes]:
        """Return the gated root process that carries the JSONL protocol streams."""
        return self._process

    def terminate(self, timeout_seconds: float) -> None:
        """Terminate every identity-tracked descendant, including detached process groups.

        Args:
            timeout_seconds: Complete grace; SIGTERM gets a quarter, SIGKILL up to three quarters.

        Raises:
            LocalProcessCleanupError: The tracker failed or observed an unprovable fork.
            subprocess.TimeoutExpired: Tracked processes survived the forced-kill bound.
        """
        process = self._process
        tracker = self._tracker
        cleanup_error: Exception | None = None
        started = time.monotonic()
        deadline = started + timeout_seconds
        try:
            if tracker is None:
                raise LocalProcessCleanupError("local descendant tracker is unavailable")
            term_deadline = started + timeout_seconds * 0.25
            if not _signal_until_empty(process, tracker, signal.SIGTERM, term_deadline):
                kill_deadline = started + timeout_seconds * 0.75
                if not _signal_until_empty(process, tracker, signal.SIGKILL, kill_deadline):
                    raise subprocess.TimeoutExpired(self._command, timeout_seconds)
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
            self._tracker = None
        if cleanup_error is not None:
            raise cleanup_error


def _signal_until_empty(
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


def _abort_gated_start(
    process: subprocess.Popen[bytes],
    tracker: _DescendantTracker | None,
    gate_write: int,
    root_identity: _ProcessIdentity | None,
    cleanup_timeout_seconds: float,
) -> tuple[str, ...]:
    """Close a failed launch without masking its original exception or leaking resources."""
    evidence: list[str] = []
    deadline = time.monotonic() + cleanup_timeout_seconds
    close_fd(gate_write)
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
    if tracker is not None:
        try:
            containment = tracker.stop(max(0.0, deadline - time.monotonic()))
            if containment:
                evidence.append(containment)
        except BaseException as error:  # noqa: BLE001 - cleanup evidence must not mask
            evidence.append(f"tracker cleanup failed with {type(error).__name__}")
    try:
        _close_unstarted_pipes(process, evidence)
    except BaseException as error:  # noqa: BLE001 - cleanup evidence must not mask
        evidence.append(f"pipe cleanup failed with {type(error).__name__}")
    return tuple(evidence)


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
