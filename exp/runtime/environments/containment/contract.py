"""The launch contract every host containment backend implements for local process sessions."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from exp.common.rollouts.containment import ContainmentReport


class LocalProcessCrashError(RuntimeError):
    """The local executable exited before it returned the required response."""


class LocalProcessCleanupError(RuntimeError):
    """The local executable or its temporary workspace could not be cleaned up."""


@dataclass(frozen=True)
class ResourceLimits:
    """Kernel resource limits applied to the customer command and inherited by descendants.

    Hosts that cannot enforce a limit report it unavailable instead of applying it partially.

    Attributes:
        cpu_time_seconds: CPU seconds after which each process is killed.
        memory_bytes: Largest address space of each process.
        file_size_bytes: Largest file any process may write.
        process_count: Most processes the session may hold at once, the command included.
    """

    cpu_time_seconds: int = 300
    memory_bytes: int = 4 * 1024**3
    file_size_bytes: int = 1024**3
    process_count: int = 64

    def __post_init__(self) -> None:
        """Reject a limit that is not a positive integer before any process starts."""
        for name, value in (
            ("cpu_time_seconds", self.cpu_time_seconds),
            ("memory_bytes", self.memory_bytes),
            ("file_size_bytes", self.file_size_bytes),
            ("process_count", self.process_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ProcessLaunch:
    """One contained launch of a customer command inside a fresh private workspace.

    Attributes:
        command: Direct executable argv; no shell parsing.
        workspace: Uniquely created private directory used as the working directory.
        environment: Complete allowlisted child environment; nothing else is inherited.
        cleanup_timeout_seconds: Termination bound, also used to reap a failed launch.
        startup_timeout_seconds: Bound for the backend to report containment applied.
        resources: Kernel resource limits for the command and its descendants.
    """

    command: tuple[str, ...]
    workspace: Path
    environment: Mapping[str, str]
    cleanup_timeout_seconds: float
    startup_timeout_seconds: float
    resources: ResourceLimits


class ContainedProcess(Protocol):
    """A launched command whose whole process tree the backend can terminate and account for."""

    @property
    def process(self) -> subprocess.Popen[bytes]:
        """Return the root process whose standard streams carry the JSONL protocol."""
        ...

    def terminate(self, timeout_seconds: float) -> None:
        """Terminate every contained process within the bound, or raise without proof.

        Args:
            timeout_seconds: Complete grace for graceful and forced termination.

        Raises:
            LocalProcessCleanupError: Containment could not prove that no process remains.
            subprocess.TimeoutExpired: Processes remained after the termination bound.
        """
        ...


class ContainmentBackend(Protocol):
    """One host's detected containment and the launcher that enforces exactly that report."""

    @property
    def report(self) -> ContainmentReport:
        """Return the startup detection report every launch enforces or refuses."""
        ...

    def launch(self, launch: ProcessLaunch) -> ContainedProcess:
        """Start the command with every reported guarantee applied, or raise.

        Args:
            launch: Command, private workspace, environment, and bounds.

        Returns:
            The running contained process.

        Raises:
            LocalProcessCleanupError: Containment could not be applied; nothing keeps running.
            LocalProcessCrashError: The command could not be started.
        """
        ...


def close_fd(file_descriptor: int) -> None:
    """Close one owned descriptor while treating an already-closed descriptor as complete."""
    try:
        os.close(file_descriptor)
    except OSError:
        return
