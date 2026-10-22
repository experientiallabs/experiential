"""Tests for host containment backend selection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from exp.common.rollouts.containment import ContainmentGuarantee
from exp.runtime.environments.containment import host as host_module
from exp.runtime.environments.containment.contract import (
    LocalProcessCleanupError,
    ProcessLaunch,
    ResourceLimits,
)
from exp.runtime.environments.containment.host import (
    UNSUPPORTED_CONTAINMENT_PROTOCOL,
    host_containment,
)


def test_unsupported_platform_refuses_with_a_complete_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A platform without a backend enforces nothing and explains every guarantee."""
    monkeypatch.setattr(host_module.sys, "platform", "plan9")

    backend = host_containment()

    report = backend.report
    assert report.platform == "plan9"
    assert report.level.protocol == UNSUPPORTED_CONTAINMENT_PROTOCOL
    assert report.level.guarantees == ()
    assert report.refusal is not None and "'plan9' provides neither" in report.refusal
    assert {item.guarantee for item in report.unavailable} == set(ContainmentGuarantee)
    launch = ProcessLaunch(
        command=("true",),
        workspace=Path("/nonexistent-exp-workspace"),
        environment={},
        cleanup_timeout_seconds=1.0,
        startup_timeout_seconds=1.0,
        resources=ResourceLimits(),
    )
    with pytest.raises(LocalProcessCleanupError, match="requires Darwin"):
        backend.launch(launch)


def test_running_host_reports_its_own_platform() -> None:
    """The selected backend describes the platform it runs on."""
    assert host_containment().report.platform == sys.platform
