"""Tests for Darwin identity-proven process containment that run on every host."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.runtime.environments.containment import darwin as darwin_module
from exp.runtime.environments.containment.contract import (
    LocalProcessCleanupError,
    ProcessLaunch,
    ResourceLimits,
)
from exp.runtime.environments.containment.darwin import (
    _ProcessIdentity,
    _ProcessRecord,
    _signal_exact_identity,
)


def test_reused_pid_identity_is_never_signaled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale PID or former group member cannot authorize a signal after PID reuse."""
    stale = _ProcessIdentity(pid=12345, started=(100, 1))
    replacement = _ProcessRecord(
        identity=_ProcessIdentity(pid=12345, started=(200, 2)),
        parent_pid=1,
        zombie=False,
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(darwin_module, "_process_snapshot", lambda: {12345: replacement})
    monkeypatch.setattr(darwin_module.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    _signal_exact_identity(stale, 9)

    assert signals == []


def test_darwin_without_kqueue_binding_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Darwin build without the kernel-notification binding cannot open local execution."""
    monkeypatch.setattr(darwin_module.sys, "platform", "darwin")
    monkeypatch.delattr(darwin_module.select, "kqueue", raising=False)

    backend = darwin_module.detect_darwin_containment()

    assert backend.bindings is None
    assert backend.report.level.guarantees == ()
    assert backend.report.refusal is not None and "requires Darwin" in backend.report.refusal
    with pytest.raises(LocalProcessCleanupError, match="requires Darwin"):
        backend.launch(_launch())


def _launch() -> ProcessLaunch:
    """Return a launch request that a refusing backend must reject before any allocation."""
    return ProcessLaunch(
        command=("true",),
        workspace=Path("/nonexistent-exp-workspace"),
        environment={},
        cleanup_timeout_seconds=1.0,
        startup_timeout_seconds=1.0,
        resources=ResourceLimits(),
    )
