"""Tests for the stdlib-only Linux supervisor program and its probe mode."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from exp.runtime.environments.containment import linux_gate

_GATE = Path(linux_gate.__file__)


def test_wait_status_maps_signals_to_shell_style_codes() -> None:
    """A signaled command reports 128 plus its signal, a normal exit its own code."""
    pid = os.fork()
    if pid == 0:
        os._exit(7)
    _, status = os.waitpid(pid, 0)
    assert linux_gate._exit_code(status) == 7
    pid = os.fork()
    if pid == 0:
        os.kill(os.getpid(), 9)
    _, status = os.waitpid(pid, 0)
    assert linux_gate._exit_code(status) == 137


def test_private_procfs_mount_failure_remains_a_startup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused mount raises, so an admitted private-procfs launch cannot silently weaken."""

    def denied_mount(*_arguments: object) -> int:
        """Simulate a host security policy refusing the mount."""
        ctypes.set_errno(errno.EACCES)
        return -1

    monkeypatch.setattr(linux_gate._LIBC, "mount", denied_mount)

    with pytest.raises(OSError, match="private procfs") as error:
        linux_gate.mount_private_procfs()

    assert error.value.errno == errno.EACCES


def test_gate_refuses_a_missing_command_with_a_usage_code() -> None:
    """The supervisor never runs without a plan and a command argv."""
    completed = subprocess.run(
        (sys.executable, "-I", "-S", str(_GATE), "{}"),
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 64


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the probe is Linux-only")
def test_probe_prints_one_fact_document_and_leaves_no_probe_directory(tmp_path: Path) -> None:
    """Probe mode observes mechanisms from inside and removes its scratch directories."""
    completed = subprocess.run(
        (sys.executable, "-I", "-S", str(_GATE), "--probe"),
        capture_output=True,
        check=False,
        timeout=15,
        env={"PATH": os.defpath, "TMPDIR": str(tmp_path)},
    )

    assert completed.returncode == 0
    facts = json.loads(completed.stdout)
    assert isinstance(facts["landlock_abi"], int)
    assert "namespace_error" in facts or facts["init_verdict"] == "ready"
    assert list(tmp_path.iterdir()) == []
