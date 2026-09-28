"""Session cleanup respects leader identity and unrelated process ownership."""

import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from exp.optimize.claas.backends.local import hosting
from exp.optimize.claas.backends.local.processes import OwnedSession


def test_reused_leader_identity_rejects_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different process under the old leader PID must never authorize any signals."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        owned = OwnedSession(process)
        replacement = SimpleNamespace(create_time=lambda: owned.created_at + 1)
        with monkeypatch.context() as patch:
            patch.setattr("psutil.Process", Mock(return_value=replacement))
            with pytest.raises(RuntimeError, match="PID was reused"):
                owned.signal(signal.SIGKILL)
    finally:
        process.kill()
        process.wait(timeout=5)


def test_cleanup_does_not_signal_an_unrelated_session() -> None:
    """Process iteration may see other work, but only the allocated learner session is owned."""
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    child = subprocess.Popen(command, start_new_session=True)
    unrelated = subprocess.Popen(command, start_new_session=True)
    try:
        owned = OwnedSession(child)
        owned.finish(child, 0.1)
        assert child.returncode is not None
        assert unrelated.poll() is None
    finally:
        for process in (child, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_run_keeps_exited_leader_unreaped_until_session_cleanup() -> None:
    """An exited leader remains a zombie reserving its PID until descendants are drained."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; assert sys.stdin.buffer.read() == b'payload'"],
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        owned = OwnedSession(process)
        owned.run(process, b"payload", 10)
        assert process.returncode is None
        assert owned.identity.status() == psutil.STATUS_ZOMBIE
        assert owned.live() == []
        owned.finish(process, 0.1)
        assert process.returncode == 0
    finally:
        process.kill()
        process.wait(timeout=5)


def test_missing_leader_never_authorizes_session_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External reaping removes the ownership anchor even if another session reuses its SID."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    owned = OwnedSession(process)
    assert process.stdin is not None
    process.stdin.close()
    process.wait(timeout=5)
    candidates = Mock()
    monkeypatch.setattr("psutil.process_iter", candidates)
    with pytest.raises(RuntimeError, match="leader disappeared"):
        owned.signal(signal.SIGKILL)
    candidates.assert_not_called()


def test_uncertain_cleanup_never_signals_a_reused_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reaped leader cannot authorize the fallback to signal its stale numeric group ID."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    owned = OwnedSession(process)
    assert process.stdin is not None
    process.stdin.close()
    process.wait(timeout=5)
    replacement = SimpleNamespace(create_time=lambda: owned.created_at + 1)
    kill_group = Mock()
    kill_process = Mock()
    kill_direct = Mock()
    with monkeypatch.context() as patch:
        patch.setattr(process, "kill", kill_direct)
        patch.setattr("psutil.Process", Mock(return_value=replacement))
        patch.setattr("os.killpg", kill_group)
        patch.setattr("os.kill", kill_process)
        with pytest.raises(RuntimeError, match="PID was reused"):
            hosting._finish(process, 0.1, owned)
    kill_group.assert_not_called()
    kill_process.assert_not_called()
    kill_direct.assert_not_called()
