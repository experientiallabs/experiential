"""Real child-process ownership and retained evidence without CUDA or provider calls."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from exp.optimize.claas.backends.local import hosting
from exp.optimize.claas.service.execution_test import configuration


def test_empty_burst_uses_real_launcher_and_publishes_report(tmp_path: Path) -> None:
    """The exact public command can finish an empty durable queue without opening a GPU."""
    report = hosting.run_local(configuration(tmp_path))
    assert report.status.state == "closed"
    assert report.status.updates == 0
    host = hosting.HostReport.model_validate_json((tmp_path / "host-report.json").read_bytes())
    assert host.state == "closed" and host.returncode == 0
    assert host.learner_pid is not None and host.learner_pid == host.session_id


@pytest.mark.parametrize("separate_group", [False, True])
@pytest.mark.parametrize("leader_exits", [False, True])
def test_forced_group_cleanup_stops_grandchild(
    tmp_path: Path, separate_group: bool, leader_exits: bool
) -> None:
    """A TERM-resistant inherited grandchild stops writing after the exact group is killed."""
    heartbeat = tmp_path / "heartbeat"
    descendant_pid = tmp_path / "descendant.pid"
    grandchild = (
        "import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        + ("os.setpgid(0,0); " if separate_group else "")
        + f"pathlib.Path({str(descendant_pid)!r}).write_text(str(os.getpid())); "
        + f"p=pathlib.Path({str(heartbeat)!r}); "
        "exec('while True:\\n p.write_text(str(time.monotonic_ns()))\\n time.sleep(.02)')"
    )
    child = (
        "import signal,subprocess,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "sys.stdin.buffer.read(); "
        f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
        + ("sys.exit(0)" if leader_exits else "time.sleep(60)")
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child], stdin=subprocess.PIPE, start_new_session=True
    )
    try:
        owned = hosting.OwnedSession(process)
        assert process.stdin is not None
        process.stdin.close()
        deadline = time.monotonic() + 10
        while not heartbeat.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert heartbeat.exists()
        hosting._finish(process, 0.05, owned)
        assert process.returncode == (0 if leader_exits else -signal.SIGKILL)
        time.sleep(0.05)
        saved = heartbeat.read_bytes()
        time.sleep(0.1)
        assert heartbeat.read_bytes() == saved
    finally:
        process.kill()
        if descendant_pid.exists():
            try:
                os.kill(int(descendant_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)


@pytest.mark.parametrize("failure", ["timeout", "interrupt", "nonzero", "cleanup"])
def test_failure_retains_state_and_reaps_owned_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Timeout, Ctrl-C and a failed child preserve evidence and publish a failed host receipt."""
    evidence = tmp_path / "retained.jsonl"
    evidence.write_bytes(b"retained evidence\n")
    original = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def spawn(command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        """Replace only the child fixture while checking the production launch boundary."""
        assert command == [
            sys.executable,
            "-m",
            "exp.optimize.claas.service.launcher",
            "--config-stdin",
        ]
        assert kwargs == {"stdin": subprocess.PIPE, "start_new_session": True}
        script = "import sys,time; sys.stdin.buffer.read(); "
        script += "sys.exit(7)" if failure == "nonzero" else "time.sleep(60)"
        process = original(
            [sys.executable, "-c", script], stdin=subprocess.PIPE, start_new_session=True
        )
        processes.append(process)
        return process

    monkeypatch.setattr(hosting.subprocess, "Popen", spawn)
    if failure == "interrupt":

        def interrupt(*_args: object) -> None:
            """Simulate a terminal interrupt while the real child remains owned."""
            raise KeyboardInterrupt

        monkeypatch.setattr(hosting.OwnedSession, "run", interrupt)
    if failure == "cleanup":

        def uncertain(
            _session: hosting.OwnedSession,
            _process: subprocess.Popen[bytes],
            _grace: float,
        ) -> None:
            """Keep session cleanup uncertain while the fallback reaps the known direct child."""
            raise TimeoutError("session cleanup remains uncertain")

        monkeypatch.setattr(hosting.OwnedSession, "finish", uncertain)
    config = configuration(tmp_path)
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(update={"startup_timeout_seconds": 0.1}),
            "run": config.run.model_copy(
                update={
                    "maximum_run_seconds": 0.1,
                    "cleanup_timeout_seconds": 0.1,
                }
            ),
        }
    )
    expected = {
        "timeout": subprocess.TimeoutExpired,
        "interrupt": KeyboardInterrupt,
        "nonzero": RuntimeError,
        "cleanup": TimeoutError,
    }[failure]
    with pytest.raises(expected):
        hosting.run_local(config)
    assert processes[0].returncode is not None
    assert evidence.read_bytes() == b"retained evidence\n"
    receipt = hosting.HostReport.model_validate_json((tmp_path / "host-report.json").read_bytes())
    assert receipt.state == "failed"
    assert receipt.failure_type == expected.__name__
    assert receipt.learner_pid == processes[0].pid
    assert receipt.session_id == processes[0].pid
    with pytest.raises(ProcessLookupError):
        os.kill(processes[0].pid, 0)


def test_spawn_failure_records_no_allocated_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exec failure produces a truthful failed host receipt and retains prior evidence."""
    evidence = tmp_path / "retained.jsonl"
    evidence.write_text("previous evidence")

    def fail(*_args: object, **_kwargs: object) -> None:
        """Fail before a process handle exists, as a missing interpreter would."""
        raise OSError("fixture exec failure")

    monkeypatch.setattr(hosting.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="exec failure"):
        hosting.run_local(configuration(tmp_path))
    report = hosting.HostReport.model_validate_json((tmp_path / "host-report.json").read_bytes())
    assert report.state == "failed" and report.returncode is None
    assert report.learner_pid is None and report.session_id is None
    assert report.failure_type == "OSError"
    assert evidence.read_text() == "previous evidence"
