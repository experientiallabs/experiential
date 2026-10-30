"""Selects the containment backend for the running host at local-runtime startup."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from exp.common.rollouts.containment import ContainmentReport, containment_report
from exp.runtime.environments.containment.contract import (
    ContainedProcess,
    ContainmentBackend,
    LocalProcessCleanupError,
    ProcessLaunch,
)
from exp.runtime.environments.containment.darwin import detect_darwin_containment
from exp.runtime.environments.containment.linux import detect_linux_containment

UNSUPPORTED_CONTAINMENT_PROTOCOL = "unsupported-host-v1"


@dataclass(frozen=True)
class UnsupportedContainment:
    """A host with no containment backend; every launch is refused before any allocation.

    Attributes:
        report: Refusal report naming the unsupported platform.
    """

    report: ContainmentReport

    def launch(self, launch: ProcessLaunch) -> ContainedProcess:
        """Refuse the launch.

        Args:
            launch: The refused launch request.

        Raises:
            LocalProcessCleanupError: Always; the host cannot contain local processes.
        """
        del launch
        raise LocalProcessCleanupError(self.report.refusal or "local containment refused")


def host_containment() -> ContainmentBackend:
    """Detect and return the containment backend for ``sys.platform``.

    Returns:
        The Darwin or Linux backend, or a refusing backend for any other platform.
    """
    platform = sys.platform
    if platform == "darwin":
        return detect_darwin_containment()
    if platform.startswith("linux"):
        return detect_linux_containment()
    report = containment_report(
        platform=platform,
        protocol=UNSUPPORTED_CONTAINMENT_PROTOCOL,
        enforced=(),
        reasons={},
        mechanisms=(),
        refusal=(
            "local process containment requires Darwin kernel fork notifications or Linux "
            f"unprivileged user and PID namespaces; platform {platform!r} provides neither"
        ),
    )
    return UnsupportedContainment(report=report)
