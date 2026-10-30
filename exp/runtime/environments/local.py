"""A bounded JSONL local-process implementation of the executable environment seam.

This adapter gives every episode a new temporary workspace and a long-lived process. The host
containment backend, detected once at construction, decides which guarantees every session
enforces (see :attr:`LocalProcessEnvironmentRuntime.containment`); a session that cannot apply
them is refused. The customer process still runs as the user who starts EXP, so anything the
report lists as unavailable is not contained. Customers that need an isolated VM can provide an
E2B or other ``EnvironmentRuntime`` implementation instead.
"""

from __future__ import annotations

import json
import math
import os
import queue
import selectors
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.common.rollouts.containment import ContainmentReport
from exp.common.tasks import TaskCase
from exp.runtime.environments.containment.contract import (
    ContainedProcess,
    ContainmentBackend,
    LocalProcessCleanupError,
    LocalProcessCrashError,
    ProcessLaunch,
    ResourceLimits,
)
from exp.runtime.environments.containment.host import host_containment
from exp.runtime.environments.interface import EnvironmentSession, Observation

_WORKSPACE_ENVIRONMENT_VARIABLE = "EXP_SANDBOX_WORKSPACE"
_END_OF_STREAM = object()
_ALLOWED_CHILD_ENVIRONMENT_KEYS = frozenset({"LANG", "LC_ALL", "LC_CTYPE", "PATH", "TZ"})
_DEFAULT_CHILD_ENVIRONMENT = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": os.defpath,
    "TZ": "UTC",
}


class LocalProcessProtocolError(RuntimeError):
    """The local executable did not follow the JSONL environment protocol."""


@dataclass(frozen=True)
class LocalProcessLimits:
    """Finite resource limits applied to each local executable environment session.

    Attributes:
        request_timeout_seconds: Wall-clock limit for one protocol request and response.
        session_timeout_seconds: Wall-clock limit for the complete local process lifetime.
        cleanup_timeout_seconds: Grace period before a live process is force-killed.
        maximum_output_bytes: Largest accepted JSON response line from the environment.
        maximum_stderr_bytes: Bounded tail retained only to drain the child error pipe.
        resources: Kernel CPU, memory, file-size, and process-count limits, enforced where the
            containment report lists them and reported unavailable elsewhere.
    """

    request_timeout_seconds: float = 30.0
    session_timeout_seconds: float = 300.0
    cleanup_timeout_seconds: float = 5.0
    maximum_output_bytes: int = 1_000_000
    maximum_stderr_bytes: int = 32_768
    resources: ResourceLimits = ResourceLimits()

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
        if not isinstance(self.resources, ResourceLimits):
            raise ValueError("resources must be a ResourceLimits value")


class LocalProcessEnvironmentRuntime:
    """Creates one fresh JSONL process and temporary workspace per executable episode.

    The child receives one line for ``open``, one per ``execute`` request, and one best-effort
    ``close`` line. It must respond to ``open`` with ``{"ready": true}``, then respond to every
    execution request with a canonical :class:`Observation` JSON object. The command is supplied
    by the customer and is never persisted in a simulation artifact.

    Construction detects the host containment once (cached per process) and exposes it as
    :attr:`containment`; sandbox simulation records its level in every cell binding.
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
        self._backend = host_containment()

    @property
    def containment(self) -> ContainmentReport:
        """Return the host containment every session from this runtime enforces or refuses."""
        return self._backend.report

    def open(self, task: TaskCase) -> AbstractContextManager[EnvironmentSession]:
        """Return a simulator-owned context that opens a fresh child only when entered.

        Args:
            task: Task whose canonical JSON is supplied in the JSONL ``open`` request.

        Returns:
            A context manager that always terminates the child and removes its workspace.
        """
        return _LocalProcessContext(
            self._backend,
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
        backend: ContainmentBackend,
        command: tuple[str, ...],
        limits: LocalProcessLimits,
        workspace_parent: Path | None,
        environment: Mapping[str, str],
        task: TaskCase,
    ) -> None:
        """Bind one task to the runtime configuration without starting any process."""
        self._backend = backend
        self._command = command
        self._limits = limits
        self._workspace_parent = workspace_parent
        self._environment = environment
        self._task = task
        self._session: _LocalProcessSession | None = None

    def __enter__(self) -> EnvironmentSession:
        """Start the child and complete its bounded protocol handshake."""
        session = _LocalProcessSession(
            self._backend,
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
        backend: ContainmentBackend,
        command: tuple[str, ...],
        limits: LocalProcessLimits,
        workspace_parent: Path | None,
        environment: Mapping[str, str],
    ) -> None:
        """Hold the session configuration; :meth:`start` allocates every resource."""
        self._backend = backend
        self._command = command
        self._limits = limits
        self._workspace_parent = workspace_parent
        self._environment = dict(environment)
        self._contained: ContainedProcess | None = None
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
        refusal = self._backend.report.refusal
        if refusal is not None:
            raise LocalProcessCleanupError(refusal)
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
        self._workspace = Path(tempfile.mkdtemp(prefix="exp-sandbox-", dir=workspace_parent))
        child_environment = dict(_DEFAULT_CHILD_ENVIRONMENT)
        child_environment.update(self._environment)
        child_environment[_WORKSPACE_ENVIRONMENT_VARIABLE] = str(self._workspace)
        child_environment["HOME"] = str(self._workspace)
        child_environment["TMPDIR"] = str(self._workspace)
        launch = ProcessLaunch(
            command=self._command,
            workspace=self._workspace,
            environment=child_environment,
            cleanup_timeout_seconds=self._limits.cleanup_timeout_seconds,
            startup_timeout_seconds=self._limits.request_timeout_seconds,
            resources=self._limits.resources,
        )
        try:
            self._contained = self._backend.launch(launch)
        except BaseException as error:  # noqa: BLE001 - preserve startup errors after cleanup
            self._closed = True
            try:
                self._remove_workspace()
            except BaseException as cleanup_error:  # noqa: BLE001 - evidence must not mask
                error.add_note(
                    "local startup cleanup evidence: workspace cleanup failed with "
                    f"{type(cleanup_error).__name__}"
                )
            raise
        process = self._contained.process
        self._started_at = time.monotonic()
        assert process.stdout is not None
        assert process.stderr is not None
        assert process.stdin is not None
        for stream in (process.stdin, process.stdout, process.stderr):
            os.set_blocking(stream.fileno(), False)
        stdout_thread = threading.Thread(
            target=self._drain_stdout,
            args=(process.stdout.fileno(),),
            name="exp-local-environment-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(process.stderr.fileno(),),
            name="exp-local-environment-stderr",
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
        contained = self._contained
        if contained is None:
            return
        process = contained.process
        if process.poll() is not None or process.stdin is None:
            return
        try:
            os.write(process.stdin.fileno(), b'{"kind":"close"}\n')
        except (BlockingIOError, OSError):
            return

    def _terminate_process(self) -> None:
        """Terminate the whole contained process tree, then stop and close the pipe workers."""
        contained = self._contained
        if contained is None or self._termination_complete:
            return
        cleanup_error: Exception | None = None
        deadline = time.monotonic() + self._limits.cleanup_timeout_seconds
        try:
            contained.terminate(self._limits.cleanup_timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - re-raised after the pipe workers are joined
            cleanup_error = exc
        finally:
            try:
                self._close_pipe_workers(contained.process, deadline)
            except Exception as exc:  # noqa: BLE001 - combined with termination evidence
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            raise cleanup_error
        self._termination_complete = True

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
        """Write within a finite wait using selectors without select's descriptor ceiling."""
        deadline = time.monotonic() + timeout_seconds
        offset = 0
        with selectors.DefaultSelector() as selector:
            selector.register(file_descriptor, selectors.EVENT_WRITE)
            while offset < len(payload):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
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
        if self._contained is None:
            raise LocalProcessProtocolError("local environment session has not started")
        return self._contained.process

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
        """Build a non-secret crash error without embedding potentially sensitive child stderr.

        End of output can precede the root's exit by the supervisor's reap, so the return code
        is read after a short bounded wait.
        """
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=min(1.0, self._limits.cleanup_timeout_seconds))
        return LocalProcessCrashError(
            f"local environment process exited before its response (return code {process.poll()})"
        )

    def _drain_stdout(self, file_descriptor: int) -> None:
        """Move framed output through a nonblocking reader that cleanup can always join."""
        buffered = bytearray()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(file_descriptor, selectors.EVENT_READ)
                while not self._io_stop.is_set():
                    if not selector.select(0.01):
                        continue
                    chunk = os.read(
                        file_descriptor, min(65_536, self._limits.maximum_output_bytes + 1)
                    )
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
            with selectors.DefaultSelector() as selector:
                selector.register(file_descriptor, selectors.EVENT_READ)
                while not self._io_stop.is_set():
                    if not selector.select(0.01):
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
