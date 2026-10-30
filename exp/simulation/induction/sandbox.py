"""Run induced transition programs in a bounded, resource-limited worker process.

Programs are model-written code, so they never execute in this process. A
:class:`ProgramSandbox` owns one long-lived worker (:mod:`exp.simulation.induction.runner`)
started without site packages, without the script directory on the import path, in a private
temporary directory, with an environment built from scratch. The worker installs its own CPU,
memory, file-size and file-descriptor limits before it reads any program, and the static
language check runs both here and in the worker. Each request has a wall-clock deadline; a
worker that misses it, crashes or breaks protocol is killed with its process group and a fresh
one starts on the next request. Every such event is a ``failed`` outcome, never an exception
that a caller could mistake for a reply. A worker that an inherited hard CPU limit leaves too
little CPU for a full request is replaced before it runs one.

The sandbox is stateless across requests: state goes in with each call and comes back in the
outcome, so one worker safely serves every backend instance, fork and validation replay.
"""

from __future__ import annotations

import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import TracebackType

from exp.common.core.artifacts import JsonObject
from exp.simulation.induction import runner
from exp.simulation.induction.contracts import ProgramOutcome, SandboxLimits, TransitionProgram

_RUNNER = Path(runner.__file__)
_READY_SECONDS = 10.0
_EXIT_REASONS = {
    64: "worker started with malformed arguments",
    71: "this platform cannot limit worker memory",
    72: "worker hashing is randomized",
    73: "worker could not hold every descriptor slot below its limit",
    74: "the inherited hard CPU limit leaves less than one request's CPU budget",
}


class SandboxUnavailableError(RuntimeError):
    """The worker cannot start with its limits on this platform; nothing executed."""


class ProgramSandbox:
    """One bounded worker that executes checked programs, one request at a time.

    Use as a context manager, or call :meth:`close`, so the worker and its directory are
    always released. Requests are serialized; the instance may be shared across threads.
    """

    def __init__(self, limits: SandboxLimits | None = None) -> None:
        """Bind limits without starting the worker; it starts on first use."""
        if sys.platform not in ("linux", "darwin"):
            raise SandboxUnavailableError(
                "induced programs need POSIX resource limits; run on Linux or macOS"
            )
        self.limits = limits or SandboxLimits()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._directory: Path | None = None
        self._buffer = b""
        self._closed = False

    def __enter__(self) -> ProgramSandbox:
        """Return this sandbox; the worker starts lazily."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the worker and remove its directory on every exit path."""
        del exception_type, exception, traceback
        self.close()

    def run(
        self,
        program: TransitionProgram,
        state: JsonObject,
        arguments: JsonObject,
        facts: JsonObject,
    ) -> ProgramOutcome:
        """Execute one program call and return its outcome.

        Args:
            program: Program to run.
            state: Visible state domains, edited by the program on a private copy.
            arguments: Call arguments.
            facts: Starting instance facts, read-only.

        Returns:
            The outcome. Refused programs never reach the worker.

        Raises:
            SandboxUnavailableError: The worker cannot start with its limits.
            RuntimeError: The sandbox was closed.
        """
        try:
            runner.check_source(program.source)
        except runner.ProgramRefused as error:
            return ProgramOutcome(status="refused", detail=str(error))
        request = json.dumps(
            {
                "source": program.source,
                "state": state,
                "args": arguments,
                "facts": facts,
                "maximum_steps": self.limits.maximum_steps,
            },
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        with self._lock:
            if self._closed:
                raise RuntimeError("program sandbox is closed; open a new one")
            return self._dispatch(request + b"\n")

    def _dispatch(self, payload: bytes) -> ProgramOutcome:
        """Run one request, moving it once to a fresh worker if the old one never started it.

        A worker whose inherited hard CPU limit leaves less than one full request either asks
        to be replaced after its reply or answers ``unstarted`` without running the request,
        so no request ever runs on less than its full CPU budget and none runs twice.
        """
        for _ in range(2):
            process = self._process or self._start()
            try:
                envelope = _envelope(self._exchange(process, payload))
            except _WorkerLost as error:
                self._stop()
                return ProgramOutcome(status="failed", detail=str(error))
            if envelope is None:
                self._stop()
                continue
            outcome, rotate = envelope
            if rotate:
                self._stop()
            return outcome
        raise SandboxUnavailableError("a fresh induced-program worker did not start the request")

    def close(self) -> None:
        """Stop the worker and refuse later requests."""
        with self._lock:
            self._closed = True
            self._stop()

    def _start(self) -> subprocess.Popen[bytes]:
        """Start one worker and wait for its readiness line after it installed limits."""
        self._directory = Path(tempfile.mkdtemp(prefix="exp-induced-"))
        process = subprocess.Popen(  # noqa: S603 - fixed interpreter and runner path
            (
                sys.executable,
                "-S",
                "-s",
                "-P",
                str(_RUNNER),
                str(self.limits.memory_bytes),
                str(self.limits.cpu_seconds),
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=self._directory,
            env={"PYTHONHASHSEED": "0", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC"},
            start_new_session=True,
        )
        self._process = process
        self._buffer = b""
        assert process.stdout is not None and process.stdin is not None
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stdin.fileno(), False)
        try:
            ready = json.loads(self._read_line(process, time.monotonic() + _READY_SECONDS))
        except (_WorkerLost, ValueError, RecursionError) as error:
            code = process.poll()
            self._stop()
            reason = _EXIT_REASONS.get(code or 0, str(error))
            raise SandboxUnavailableError(
                f"induced-program worker did not start: {reason}"
            ) from None
        if ready != {"ready": True, "protocol": runner.RUNNER_PROTOCOL}:
            self._stop()
            raise SandboxUnavailableError("induced-program worker speaks another protocol")
        return process

    def _exchange(self, process: subprocess.Popen[bytes], payload: bytes) -> bytes:
        """Write one request and read one response line before the request deadline."""
        deadline = time.monotonic() + self.limits.request_timeout_seconds
        assert process.stdin is not None
        view = memoryview(payload)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdin, selectors.EVENT_WRITE)
            while view:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise _WorkerLost("program exceeded its wall-clock limit")
                try:
                    written = os.write(process.stdin.fileno(), view)
                except BlockingIOError:
                    continue
                except OSError:
                    raise _WorkerLost("worker stopped reading; it was restarted") from None
                view = view[written:]
        return self._read_line(process, deadline)

    def _read_line(self, process: subprocess.Popen[bytes], deadline: float) -> bytes:
        """Read one bounded newline-terminated response."""
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                if len(self._buffer) > self.limits.maximum_output_bytes:
                    raise _WorkerLost("program outcome exceeds the output limit")
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise _WorkerLost("program exceeded its wall-clock limit")
                try:
                    chunk = os.read(process.stdout.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise _WorkerLost("worker exited; a CPU or memory limit may have been hit")
                self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        if len(line) > self.limits.maximum_output_bytes:
            raise _WorkerLost("program outcome exceeds the output limit")
        return line

    def _stop(self) -> None:
        """Kill the worker's process group, reap it and remove its directory."""
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=10)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        directory, self._directory = self._directory, None
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)


def _envelope(line: bytes) -> tuple[ProgramOutcome, bool] | None:
    """Decode one worker reply: the outcome and whether to rotate, or None for ``unstarted``.

    Raises:
        _WorkerLost: The line is not exactly one of the two envelopes the worker speaks.
    """
    try:
        message = json.loads(line)
    except (ValueError, RecursionError):
        # Deeply nested JSON exhausts the decoder's recursion before it can be rejected.
        raise _WorkerLost("worker returned a malformed outcome; it was restarted") from None
    if isinstance(message, dict) and set(message) == {"unstarted"}:
        if message["unstarted"] is True:
            return None
    elif (
        isinstance(message, dict)
        and set(message) == {"outcome", "rotate"}
        and isinstance(message["rotate"], bool)
    ):
        try:
            return ProgramOutcome.model_validate(message["outcome"]), message["rotate"]
        except (ValueError, RecursionError):
            pass
    raise _WorkerLost("worker returned a malformed outcome; it was restarted")


class _WorkerLost(RuntimeError):
    """The worker missed a deadline, exited or broke protocol; it is restarted."""
