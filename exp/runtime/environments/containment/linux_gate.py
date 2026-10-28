"""Linux namespace supervisor that contains one customer command and proves its teardown.

The parent backend runs ``python -I -S linux_gate.py PLAN_JSON COMMAND...``; ``--probe`` instead
exercises every mechanism once and prints the observed facts as JSON. This file imports only the
standard library so it starts without the EXP package and cannot be shadowed by its directory.

Three processes form one session:

- the supervisor (this process, a child of EXP) unshares a user, PID, and optionally network
  namespace, then waits for a termination request on its lifeline descriptor;
- the init process is PID 1 of the new PID namespace, dies with the supervisor, reaps orphans,
  and forwards termination to everything inside;
- the command process starts its own session, applies resource limits, no-new-privileges and
  Landlock, then executes the customer argv.

The kernel kills every process in a PID namespace when its init exits, and the supervisor can
reap init only after the namespace is empty. The supervisor reports that reap on its status
descriptor, which is the teardown proof the parent requires.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import resource
import selectors
import shutil
import signal
import socket
import sys
import tempfile
from collections.abc import Callable

_PR_SET_PDEATHSIG = 1
_PR_SET_NO_NEW_PRIVS = 38
_LANDLOCK_CREATE_RULESET = 444
_LANDLOCK_ADD_RULE = 445
_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_ACCESS_FS_WRITE_FILE = 1 << 1
_ACCESS_FS_DIRECTORY_WRITES = (
    (1 << 4)  # REMOVE_DIR
    | (1 << 5)  # REMOVE_FILE
    | (1 << 6)  # MAKE_CHAR
    | (1 << 7)  # MAKE_DIR
    | (1 << 8)  # MAKE_REG
    | (1 << 9)  # MAKE_SOCK
    | (1 << 10)  # MAKE_FIFO
    | (1 << 11)  # MAKE_BLOCK
    | (1 << 12)  # MAKE_SYM
)
_ACCESS_FS_REFER = 1 << 13
_ACCESS_FS_TRUNCATE = 1 << 14
_SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0
_SCOPE_SIGNAL = 1 << 1
_WRITABLE_DEVICES = ("/dev/null", "/dev/zero", "/dev/full")
_MS_NOSUID = 2
_MS_NODEV = 4
_MS_NOEXEC = 8
_MS_REC = 16384
_MS_PRIVATE = 1 << 18
MINIMUM_LANDLOCK_ABI = 3
_SETUP_FAILURE = 70
_EXEC_FAILURE = 127
_LIBC = ctypes.CDLL(None, use_errno=True)


class _RulesetAttr(ctypes.Structure):
    """ABI layout of ``struct landlock_ruleset_attr``; its accepted size grows with the ABI."""

    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    """Packed ABI layout of ``struct landlock_path_beneath_attr``."""

    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def landlock_abi() -> int:
    """Return the kernel Landlock ABI version, or zero when Landlock is unavailable."""
    version = _LIBC.syscall(
        ctypes.c_long(_LANDLOCK_CREATE_RULESET),
        None,
        ctypes.c_size_t(0),
        ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    return max(0, int(version))


def _prctl(option: int, value: int) -> None:
    """Call ``prctl`` with one argument and raise the kernel error on failure."""
    if _LIBC.prctl(ctypes.c_int(option), ctypes.c_ulong(value), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _landlock_call(number: int, *arguments: object) -> int:
    """Invoke one Landlock syscall and raise its kernel error on failure."""
    result = int(_LIBC.syscall(ctypes.c_long(number), *arguments))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, f"landlock syscall {number}: {os.strerror(error)}")
    return result


def restrict_filesystem_writes(workspace: str, abi: int) -> None:
    """Confine writes to ``workspace`` and scope signals and abstract sockets to the domain.

    Reads and execution stay unrestricted. Any Landlock domain also denies tracing, and thus
    reading the environment or memory of, every process outside the domain.

    Args:
        workspace: The only directory tree in which files may be written, created, or removed.
        abi: Kernel Landlock ABI version detected at startup.

    Raises:
        OSError: The ruleset could not be created, populated, or enforced.
    """
    file_rights = _ACCESS_FS_WRITE_FILE | (_ACCESS_FS_TRUNCATE if abi >= 3 else 0)
    handled = file_rights | _ACCESS_FS_DIRECTORY_WRITES | (_ACCESS_FS_REFER if abi >= 2 else 0)
    scoped = (_SCOPE_ABSTRACT_UNIX_SOCKET | _SCOPE_SIGNAL) if abi >= 6 else 0
    attributes = _RulesetAttr(handled_access_fs=handled, handled_access_net=0, scoped=scoped)
    size = 24 if abi >= 6 else 8
    ruleset = _landlock_call(
        _LANDLOCK_CREATE_RULESET,
        ctypes.byref(attributes),
        ctypes.c_size_t(size),
        ctypes.c_uint32(0),
    )
    try:
        for path, rights in ((workspace, handled),) + tuple(
            (device, file_rights) for device in _WRITABLE_DEVICES if os.path.exists(device)
        ):
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = _PathBeneathAttr(allowed_access=rights, parent_fd=parent)
                _landlock_call(
                    _LANDLOCK_ADD_RULE,
                    ctypes.c_int(ruleset),
                    ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
                    ctypes.byref(rule),
                    ctypes.c_uint32(0),
                )
            finally:
                os.close(parent)
        _landlock_call(_LANDLOCK_RESTRICT_SELF, ctypes.c_int(ruleset), ctypes.c_uint32(0))
    finally:
        os.close(ruleset)


def contain_self(plan: dict[str, object]) -> None:
    """Apply the command-process restrictions in the order the kernel requires.

    Args:
        plan: Decoded supervisor plan with ``rlimits``, ``no_new_privs``, ``landlock_abi``,
            ``workspace`` and the optional ``private_procfs`` entries.
    """
    if plan.get("private_procfs"):
        mount_private_procfs()
    os.setsid()
    for signal_number in (signal.SIGPIPE, signal.SIGXFSZ, signal.SIGTERM):
        signal.signal(signal_number, signal.SIG_DFL)
    limits = plan["rlimits"]
    assert isinstance(limits, dict)
    for name, value in limits.items():
        assert isinstance(name, str) and isinstance(value, int)
        resource.setrlimit(getattr(resource, name), (value, value))
    if plan["no_new_privs"]:
        _prctl(_PR_SET_NO_NEW_PRIVS, 1)
    abi = plan["landlock_abi"]
    workspace = plan["workspace"]
    assert isinstance(abi, int) and isinstance(workspace, str)
    if abi:
        restrict_filesystem_writes(workspace, abi)


def mount_private_procfs() -> None:
    """Mount procfs for this PID namespace without propagating any mount to the host.

    Raises:
        OSError: The host refused private propagation or the namespace's procfs mount.
    """
    operations = (
        (None, b"/", None, _MS_PRIVATE | _MS_REC),
        (b"proc", b"/proc", b"proc", _MS_NOSUID | _MS_NODEV | _MS_NOEXEC),
    )
    for source, target, filesystem, flags in operations:
        if (
            _LIBC.mount(
                ctypes.c_char_p(source),
                ctypes.c_char_p(target),
                ctypes.c_char_p(filesystem),
                ctypes.c_ulong(flags),
                None,
            )
            != 0
        ):
            error = ctypes.get_errno()
            raise OSError(error, "private procfs: " + os.strerror(error))


def enter_namespaces(deny_network: bool, *, private_procfs: bool = False) -> None:
    """Unshare user/PID namespaces, plus a mount namespace for procfs and optional network."""
    flags = os.CLONE_NEWUSER | os.CLONE_NEWPID
    if private_procfs:
        flags |= os.CLONE_NEWNS
    if deny_network:
        flags |= os.CLONE_NEWNET
    os.unshare(flags)


def _exit_code(status: int) -> int:
    """Map a wait status to a shell-style exit code that fits a process return code."""
    code = os.waitstatus_to_exitcode(status)
    return 128 - code if code < 0 else code


def _detach_stdio() -> None:
    """Point standard streams at the null device so only the command holds EXP's pipes."""
    null = os.open(os.devnull, os.O_RDWR)
    for descriptor in (0, 1, 2):
        os.dup2(null, descriptor)
    os.close(null)


def _forward_to_namespace(signal_number: int) -> None:
    """Send one signal to every process in this PID namespace except init itself."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(-1, signal_number)


class _SetupError(RuntimeError):
    """A containment step failed inside the command process before exec."""


def _run_init(channel: socket.socket, child: Callable[[], None]) -> None:
    """Act as PID 1: start the command process, reap every orphan, and exit with its status.

    Args:
        channel: Supervisor socket for the death-signal handshake and the readiness verdict.
        child: Code the command process runs after fork; it must exec or raise.
    """
    _prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    channel.sendall(b"a")
    if channel.recv(1) != b"g":
        os._exit(_SETUP_FAILURE)
    signal.signal(signal.SIGTERM, lambda number, _frame: _forward_to_namespace(number))
    failure_read, failure_write = os.pipe()
    command_pid = os.fork()
    if command_pid == 0:
        os.close(failure_read)
        channel.close()
        try:
            child()
        except BaseException as error:  # noqa: BLE001 - reported to init, then exit
            reason = str(error) if isinstance(error, _SetupError) else type(error).__name__
            os.write(failure_write, reason.encode()[:512])
        os._exit(_EXEC_FAILURE)
    os.close(failure_write)
    failure = b""
    while chunk := os.read(failure_read, 512):
        failure += chunk
    os.close(failure_read)
    channel.sendall(b"ready\n" if not failure else b"error " + failure + b"\n")
    channel.close()
    _detach_stdio()
    code = _SETUP_FAILURE
    while True:
        pid, status = os.wait()
        if pid == command_pid:
            code = _exit_code(status)
            break
    _forward_to_namespace(signal.SIGKILL)
    with contextlib.suppress(ChildProcessError):
        while True:
            os.wait()
    os._exit(code)


def _start_init(child: Callable[[], None], closing: tuple[int, ...]) -> tuple[int, str]:
    """Fork the namespace init, complete the handshake, and return its PID and verdict.

    Args:
        child: Code for the command process forked by init.
        closing: Supervisor-only descriptors init must not keep open.

    Returns:
        The init PID in the supervisor's namespace and ``ready`` or an ``error`` verdict.
    """
    supervisor_end, init_end = socket.socketpair()
    init_pid = os.fork()
    if init_pid == 0:
        try:
            supervisor_end.close()
            for descriptor in closing:
                os.close(descriptor)
            _run_init(init_end, child)
        finally:
            os._exit(_SETUP_FAILURE)
    init_end.close()
    verdict = "error init exited before arming its death signal"
    if supervisor_end.recv(1) == b"a":
        supervisor_end.sendall(b"g")
        received = b""
        while not received.endswith(b"\n") and (chunk := supervisor_end.recv(1024)):
            received += chunk
        verdict = received.decode(errors="replace").strip() or "error init exited early"
    supervisor_end.close()
    return init_pid, verdict


def _emit(status_fd: int, event: dict[str, object]) -> None:
    """Write one newline-delimited status event, ignoring a parent that already left."""
    with contextlib.suppress(OSError):
        os.write(status_fd, json.dumps(event, separators=(",", ":")).encode() + b"\n")


def _wait_for_exit(pidfd: int, timeout_seconds: float) -> bool:
    """Return whether the process behind ``pidfd`` exits within the timeout."""
    with selectors.DefaultSelector() as selector:
        selector.register(pidfd, selectors.EVENT_READ)
        return bool(selector.select(timeout_seconds))


def _supervise(plan: dict[str, object], command: list[str]) -> int:
    """Run one contained command until it exits or the lifeline requests termination."""
    status_fd = plan["status_fd"]
    lifeline_fd = plan["lifeline_fd"]
    grace = plan["termination_grace_seconds"]
    assert isinstance(status_fd, int) and isinstance(lifeline_fd, int)
    assert isinstance(grace, float)
    for descriptor in (status_fd, lifeline_fd):
        os.set_inheritable(descriptor, False)
    try:
        enter_namespaces(bool(plan["deny_network"]), private_procfs=bool(plan["private_procfs"]))
    except OSError as error:
        _emit(status_fd, {"event": "error", "reason": f"namespace setup: {error.strerror}"})
        _emit(status_fd, {"event": "reaped", "returncode": _SETUP_FAILURE})
        return _SETUP_FAILURE

    def command_process() -> None:
        """Contain the forked command process, then replace it with the customer argv."""
        try:
            contain_self(plan)
        except OSError as error:
            raise _SetupError(f"containment setup: {error.strerror or error}") from error
        os.execvpe(command[0], command, os.environ)

    init_pid, verdict = _start_init(command_process, (status_fd, lifeline_fd))
    if verdict != "ready":
        reason = verdict.removeprefix("error ").strip()
        _emit(status_fd, {"event": "error", "reason": reason})
        os.waitpid(init_pid, 0)
        _emit(status_fd, {"event": "reaped", "returncode": _SETUP_FAILURE})
        return _SETUP_FAILURE
    _emit(status_fd, {"event": "ready", "init_pid": init_pid})
    _detach_stdio()
    pidfd = os.pidfd_open(init_pid)
    with selectors.DefaultSelector() as selector:
        selector.register(pidfd, selectors.EVENT_READ)
        selector.register(lifeline_fd, selectors.EVENT_READ)
        events = selector.select()
    if not any(key.fd == pidfd for key, _ in events):
        with contextlib.suppress(ProcessLookupError):
            os.kill(init_pid, signal.SIGTERM)
        if not _wait_for_exit(pidfd, grace):
            with contextlib.suppress(ProcessLookupError):
                os.kill(init_pid, signal.SIGKILL)
    _, status = os.waitpid(init_pid, 0)
    os.close(pidfd)
    code = _exit_code(status)
    _emit(status_fd, {"event": "reaped", "returncode": code})
    return code


def _probe() -> int:
    """Exercise every containment mechanism once and print the observed facts as JSON."""
    facts: dict[str, object] = {"landlock_abi": landlock_abi()}
    root = tempfile.mkdtemp(prefix="exp-containment-probe-")
    workspace = os.path.join(root, "workspace")
    outside = os.path.join(root, "outside")
    os.mkdir(workspace)
    os.mkdir(outside)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    supervisor_pid = os.getpid()
    try:
        try:
            enter_namespaces(deny_network=True)
            facts["network_namespace"] = True
        except OSError as network_error:
            facts["network_namespace"] = False
            facts["network_namespace_error"] = network_error.strerror
            try:
                enter_namespaces(deny_network=False)
            except OSError as error:
                facts["namespace_error"] = error.strerror
                return _report(facts)
        report_read, report_write = os.pipe()
        abi = facts["landlock_abi"]
        assert isinstance(abi, int)
        plan: dict[str, object] = {
            "rlimits": {"RLIMIT_CPU": 30, "RLIMIT_NPROC": 64, "RLIMIT_CORE": 0},
            "no_new_privs": True,
            "landlock_abi": abi if abi >= MINIMUM_LANDLOCK_ABI else 0,
            "workspace": workspace,
        }

        def probe_process() -> None:
            """Apply the command restrictions, observe their effect, and report it."""
            observed: dict[str, object] = {"pid": os.getpid(), "parent": os.getppid()}
            try:
                os.unshare(os.CLONE_NEWNS)
                mount_private_procfs()
            except OSError as error:
                observed["procfs_error"] = error.strerror or str(error)
            observed["procfs_private"] = os.readlink("/proc/self") == str(os.getpid()) and {
                int(pid) for pid in os.listdir("/proc") if pid.isdigit()
            } == {1, 2}
            try:
                contain_self(plan)
                observed["contained"] = True
            except OSError as error:
                observed["contain_error"] = error.strerror or str(error)
            with open("/proc/self/status", encoding="utf-8") as status:
                observed["no_new_privs"] = "NoNewPrivs:\t1" in status.read()
            observed["network_reachable"] = _reachable(port)
            observed["writes_confined"] = _writes_confined(workspace, outside)
            observed["host_environment_readable"] = _readable(f"/proc/{supervisor_pid}/environ")
            observed["host_signalable"] = _signalable(supervisor_pid)
            observed["process_limit_enforced"] = _process_limit_enforced()
            os.write(report_write, json.dumps(observed).encode())
            os._exit(0)

        init_pid, verdict = _start_init(probe_process, ())
        try:
            os.close(os.pidfd_open(init_pid))
            facts["pidfd"] = True
        except OSError:
            facts["pidfd"] = False
        os.close(report_write)
        payload = b""
        while chunk := os.read(report_read, 4096):
            payload += chunk
        os.close(report_read)
        os.waitpid(init_pid, 0)
        facts["init_verdict"] = verdict
        facts["command"] = json.loads(payload) if payload else None
        return _report(facts)
    finally:
        listener.close()
        shutil.rmtree(root, ignore_errors=True)


def _reachable(port: int) -> bool:
    """Return whether the probe can connect to a listener in the supervisor's network."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def _writes_confined(workspace: str, outside: str) -> bool:
    """Return whether writes succeed in the workspace and fail beside it."""
    try:
        with open(os.path.join(workspace, "inside"), "w", encoding="utf-8") as handle:
            handle.write("x")
    except OSError:
        return False
    try:
        with open(os.path.join(outside, "escape"), "w", encoding="utf-8") as handle:
            handle.write("x")
    except PermissionError:
        return True
    return False


def _readable(path: str) -> bool:
    """Return whether one byte of a host-process file can be read."""
    try:
        with open(path, "rb") as handle:
            handle.read(1)
    except OSError:
        return False
    return True


def _signalable(pid: int) -> bool:
    """Return whether a null signal reaches a host process through its procfs descriptor."""
    try:
        directory = os.open(f"/proc/{pid}", os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return False
    try:
        signal.pidfd_send_signal(directory, 0)
    except OSError:
        return False
    finally:
        os.close(directory)
    return True


def _process_limit_enforced() -> bool:
    """Return whether RLIMIT_NPROC refuses a fork here; the kernel exempts real root from it."""
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    try:
        child = os.fork()
    except BlockingIOError:
        return True
    if child == 0:
        os._exit(0)
    os.waitpid(child, 0)
    return False


def _report(facts: dict[str, object]) -> int:
    """Print probe facts as one JSON document for the parent detector."""
    sys.stdout.write(json.dumps(facts, sort_keys=True) + "\n")
    sys.stdout.flush()
    return 0


def main() -> int:
    """Dispatch to probe or supervisor mode from the command line.

    Returns:
        The process exit code: the command's own code, or a setup or usage failure code.
    """
    if sys.argv[1:] == ["--probe"]:
        return _probe()
    if len(sys.argv) < 3:
        return 64
    plan = json.loads(sys.argv[1])
    assert isinstance(plan, dict)
    return _supervise(plan, sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
