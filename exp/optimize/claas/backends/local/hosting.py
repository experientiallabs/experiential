"""Enforce local lifetime limits around an entire resident learner process group."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal

from filelock import FileLock
from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.optimize.claas.backends.local.processes import OwnedSession
from exp.optimize.claas.service.configuration import RunReport
from exp.optimize.claas.service.files import (
    MAXIMUM_CONFIGURATION_BYTES,
    read_regular_file,
    write_contract,
)
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration


class HostReport(ContractModel):
    """Record supervisor termination without claiming an interrupted update was committed."""

    state: Literal["closed", "failed"]
    recorded_at: datetime
    returncode: int | None
    learner_pid: int | None = Field(default=None, strict=True, gt=0)
    session_id: int | None = Field(default=None, strict=True, gt=0)
    failure_type: str | None = None


@contextmanager
def _interrupts() -> Iterator[None]:
    """Convert termination into owned cleanup, then restore the caller's signal handlers."""
    previous = {item: signal.getsignal(item) for item in (signal.SIGINT, signal.SIGTERM)}

    def interrupt(_number: int, _frame: object) -> None:
        """Leave normal execution through the cleanup boundary on either terminal signal."""
        raise KeyboardInterrupt

    try:
        for item in previous:
            signal.signal(item, interrupt)
        yield
    finally:
        for item, handler in previous.items():
            signal.signal(item, handler)


def _finish(
    process: subprocess.Popen[bytes], grace_seconds: float, session: OwnedSession | None = None
) -> None:
    """Terminate all live session members, including Ray workers in separate process groups.

    Zombie grandchildren have stopped executing and are reaped by their OS parent.
    Operator code deliberately creating a separate session is unsupported.
    """
    previous = {item: signal.getsignal(item) for item in (signal.SIGINT, signal.SIGTERM)}
    deadline = time.monotonic() + grace_seconds + 30
    try:
        for item in previous:
            signal.signal(item, signal.SIG_IGN)
        session = session or OwnedSession(process)
        session.finish(process, grace_seconds)
    except BaseException:
        # Never infer group ownership from a PID. A lost session identity must
        # fail before fallback signaling; enumeration failure with a verified
        # leader may terminate only that child and still records uncertainty.
        if session is not None:
            session.kill_leader()
        else:
            # Capture failed before stdin delivery or any host reaping. Popen
            # still exclusively owns this direct child, so PID reuse is impossible.
            process.kill()
        process.wait(timeout=max(0, deadline - time.monotonic()))
        raise
    finally:
        if process.stdin is not None:
            process.stdin.close()
        for item, handler in previous.items():
            signal.signal(item, handler)


def run_local(configuration: RunLaunchConfiguration) -> RunReport:
    """Supervise one learner with a hard startup/run bound and bounded forced cleanup.

    Logs and environment credentials are inherited; stdin contains only bounded
    launch configuration. This synchronous adapter requires a POSIX main thread.
    It does not launch or manage an independent model server.
    """
    configuration = RunLaunchConfiguration.model_validate(configuration.model_dump())
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        raise ValueError("local learner hosting requires the main thread of a POSIX process")
    if configuration.persistence != "local":
        raise ValueError("local learner hosting requires local persistence")
    payload = configuration.model_dump_json().encode()
    if len(payload) > MAXIMUM_CONFIGURATION_BYTES:
        raise ValueError("launch configuration exceeds 128 KiB")
    directory = configuration.directory
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(directory / "host.lock", timeout=0, mode=0o600), _interrupts():
        return _run(configuration, payload)


def _run(configuration: RunLaunchConfiguration, payload: bytes) -> RunReport:
    """Retain state files on every failure and require a newly published successful receipt."""
    report_path = configuration.directory / "run-report.json"
    previous = report_path.stat() if report_path.exists() else None
    process: subprocess.Popen[bytes] | None = None
    session: OwnedSession | None = None
    failure: BaseException | None = None
    try:
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "exp.optimize.claas.service.launcher", "--config-stdin"],
                stdin=subprocess.PIPE,
                start_new_session=True,
            )
            session = OwnedSession(process)
            session.live()  # Require visibility before stdin allows model initialization.
            session.run(
                process,
                payload,
                configuration.runtime.startup_timeout_seconds
                + configuration.run.maximum_run_seconds,
            )
        finally:
            if process is not None:
                _finish(process, configuration.run.cleanup_timeout_seconds, session)
        if process.returncode != 0:
            raise RuntimeError(f"local learner exited with code {process.returncode}")
        current = report_path.stat()
        if previous is not None and (current.st_ino, current.st_mtime_ns) == (
            previous.st_ino,
            previous.st_mtime_ns,
        ):
            raise RuntimeError("local learner did not publish a new run receipt")
        report = RunReport.model_validate_json(read_regular_file(report_path, 8_388_608))
        if report.status.state != "closed":
            raise RuntimeError("local learner reported an unsuccessful run; inspect retained state")
        return report
    except BaseException as error:
        failure = error
        raise
    finally:
        write_contract(
            configuration.directory / "host-report.json",
            HostReport(
                state="failed" if failure else "closed",
                recorded_at=datetime.now(UTC),
                returncode=process.returncode if process is not None else None,
                learner_pid=process.pid if process is not None else None,
                session_id=session.session_id if session is not None else None,
                failure_type=type(failure).__name__ if failure else None,
            ),
        )
