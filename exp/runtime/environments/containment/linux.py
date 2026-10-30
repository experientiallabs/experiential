"""Linux namespace containment: startup capability detection and per-session supervision.

Detection runs the stdlib-only gate in probe mode once per EXP process. The probe applies every
mechanism a session would use and observes its effect from inside, so the report names only
guarantees that were enforced on this host. Each session then launches the same gate as its
supervisor, which either applies exactly the reported mechanisms or refuses the session.
"""

from __future__ import annotations

import functools
import json
import os
import selectors
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.common.rollouts.containment import (
    ContainmentGuarantee,
    ContainmentReport,
    containment_report,
)
from exp.runtime.environments.containment.contract import (
    LocalProcessCleanupError,
    LocalProcessCrashError,
    ProcessLaunch,
    close_fd,
)
from exp.runtime.environments.containment.linux_gate import MINIMUM_LANDLOCK_ABI

LINUX_CONTAINMENT_PROTOCOL = "linux-namespaces-v3"
_GATE_PROGRAM = Path(__file__).with_name("linux_gate.py")
_GATE_ARGV = (sys.executable, "-I", "-S", str(_GATE_PROGRAM))
_PROBE_TIMEOUT_SECONDS = 15.0
_SUPERVISOR_PROCESSES = 2
_REFUSAL_ADVICE = (
    "enable unprivileged user namespaces (for example kernel.unprivileged_userns_clone=1, "
    "a nonzero user.max_user_namespaces, or an AppArmor policy that permits them) or run the "
    "environment through a remote EnvironmentRuntime"
)
_ALWAYS_ENFORCED = frozenset(
    {
        ContainmentGuarantee.SESSION_ISOLATION,
        ContainmentGuarantee.DESCENDANT_CONTAINMENT,
        ContainmentGuarantee.VERIFIED_CLEANUP,
        ContainmentGuarantee.PRIVATE_WORKSPACE,
        ContainmentGuarantee.ENVIRONMENT_ALLOWLIST,
        ContainmentGuarantee.CPU_TIME_LIMIT,
        ContainmentGuarantee.MEMORY_LIMIT,
        ContainmentGuarantee.FILE_SIZE_LIMIT,
        ContainmentGuarantee.NO_NEW_PRIVILEGES,
    }
)
_BASE_MECHANISMS = (
    "new_session",
    "no_new_privs",
    "pid_namespace",
    "pidfd_supervisor",
    "rlimits",
    "user_namespace",
)


@functools.cache
def detect_linux_containment() -> LinuxContainment:
    """Probe this host once and return the backend that enforces what the probe observed.

    Returns:
        The Linux backend; its report carries a refusal when namespaces are unavailable.
    """
    try:
        completed = subprocess.run(
            (*_GATE_ARGV, "--probe"),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            env={"PATH": os.defpath, "LANG": "C.UTF-8"},
            start_new_session=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return linux_containment_from_probe({"probe_error": type(error).__name__})
    try:
        facts = json.loads(completed.stdout)
    except ValueError:
        facts = None
    if completed.returncode != 0 or not isinstance(facts, dict):
        return linux_containment_from_probe(
            {"probe_error": f"probe exited with return code {completed.returncode}"}
        )
    return linux_containment_from_probe(cast(JsonObject, facts))


def linux_containment_from_probe(facts: JsonObject) -> LinuxContainment:
    """Translate observed probe facts into the backend and its complete report.

    Args:
        facts: The JSON document printed by ``linux_gate.py --probe``.

    Returns:
        A backend whose report names only guarantees the probe saw enforced.
    """
    refusal = _probe_refusal(facts)
    platform = sys.platform
    if refusal is not None:
        report = containment_report(
            platform=platform,
            protocol=LINUX_CONTAINMENT_PROTOCOL,
            enforced=(),
            reasons={},
            mechanisms=(),
            refusal=refusal,
        )
        return LinuxContainment(
            report=report, deny_network=False, landlock_abi=0, private_procfs=False
        )
    observed = cast(JsonObject, facts["command"])
    enforced = set(_ALWAYS_ENFORCED)
    reasons: dict[ContainmentGuarantee, str] = {}
    mechanisms = list(_BASE_MECHANISMS)
    deny_network = facts.get("network_namespace") is True and observed["network_reachable"] is False
    if deny_network:
        enforced.add(ContainmentGuarantee.NETWORK_DENIED)
        mechanisms.append("network_namespace")
    else:
        detail = facts.get("network_namespace_error") or "the probe still reached the host network"
        reasons[ContainmentGuarantee.NETWORK_DENIED] = (
            f"network namespaces are unavailable on this host ({detail}); network-requiring "
            "guarantees cannot be enforced"
        )
    abi = facts.get("landlock_abi")
    landlock_abi = abi if isinstance(abi, int) and abi >= MINIMUM_LANDLOCK_ABI else 0
    if landlock_abi:
        mechanisms.append(f"landlock_abi_{landlock_abi}")
    landlock_reason = (
        f"requires Landlock ABI {MINIMUM_LANDLOCK_ABI} or newer; this kernel reports ABI {abi}"
    )
    if landlock_abi and observed["writes_confined"] is True:
        enforced.add(ContainmentGuarantee.FILESYSTEM_WRITE_CONFINEMENT)
    else:
        reasons[ContainmentGuarantee.FILESYSTEM_WRITE_CONFINEMENT] = (
            landlock_reason if not landlock_abi else "the probe could write outside its workspace"
        )
    if observed.get("process_limit_enforced") is True:
        enforced.add(ContainmentGuarantee.PROCESS_COUNT_LIMIT)
    else:
        reasons[ContainmentGuarantee.PROCESS_COUNT_LIMIT] = (
            "the probe forked past RLIMIT_NPROC; the kernel exempts real root from it, so run "
            "the environment as an unprivileged user or through a remote EnvironmentRuntime"
        )
    private_procfs = observed.get("procfs_private") is True
    if private_procfs:
        mechanisms.extend(("mount_namespace", "private_procfs"))
    if (
        landlock_abi
        and private_procfs
        and observed["host_environment_readable"] is False
        and observed["host_signalable"] is False
    ):
        enforced.add(ContainmentGuarantee.HOST_PROCESS_ISOLATION)
    else:
        reasons[ContainmentGuarantee.HOST_PROCESS_ISOLATION] = (
            landlock_reason
            if not landlock_abi
            else (
                "requires a private procfs mount; this host refused or did not isolate it "
                f"({observed.get('procfs_error', 'host PIDs remained visible')}); allow private "
                "procfs mounts in the host security policy or use a remote EnvironmentRuntime"
                if not private_procfs
                else "the probe could read or signal a host process"
            )
        )
    report = containment_report(
        platform=platform,
        protocol=LINUX_CONTAINMENT_PROTOCOL,
        enforced=enforced,
        reasons=reasons,
        mechanisms=mechanisms,
    )
    return LinuxContainment(
        report=report,
        deny_network=deny_network,
        landlock_abi=landlock_abi,
        private_procfs=private_procfs,
    )


def _probe_refusal(facts: JsonObject) -> str | None:
    """Return why local execution must be refused, or None when the baseline held."""
    prefix = "Linux local process containment requires unprivileged user and PID namespaces"
    if (error := facts.get("probe_error")) is not None:
        return f"{prefix}; the capability probe failed ({error}); {_REFUSAL_ADVICE}"
    if (error := facts.get("namespace_error")) is not None:
        return f"{prefix}; this host refused them ({error}); {_REFUSAL_ADVICE}"
    observed = facts.get("command")
    if (
        facts.get("init_verdict") != "ready"
        or not isinstance(observed, dict)
        or observed.get("contained") is not True
        or observed.get("no_new_privs") is not True
        or observed.get("pid") != 2
        or observed.get("parent") != 1
    ):
        detail = observed.get("contain_error") if isinstance(observed, dict) else None
        return (
            f"{prefix}; the probe could not apply process containment "
            f"({detail or facts.get('init_verdict')}); {_REFUSAL_ADVICE}"
        )
    if facts.get("pidfd") is not True:
        return f"{prefix} and pidfd_open (Linux 5.3 or newer)"
    return None


@dataclass(frozen=True)
class LinuxContainment:
    """The detected Linux backend: one report and the plan every session enforces.

    Attributes:
        report: Startup detection report.
        deny_network: Whether sessions run in a fresh network namespace.
        landlock_abi: Landlock ABI applied to sessions, or zero when Landlock is not used.
        private_procfs: Whether sessions mount a procfs that exposes only their PID namespace.
    """

    report: ContainmentReport
    deny_network: bool
    landlock_abi: int
    private_procfs: bool

    def launch(self, launch: ProcessLaunch) -> LinuxContainedProcess:
        """Start the supervisor and wait until it reports the reported containment applied.

        Args:
            launch: Command, private workspace, environment, limits, and bounds.

        Returns:
            The running contained process.

        Raises:
            LocalProcessCleanupError: The host refused containment or the supervisor failed.
            LocalProcessCrashError: The supervisor or the command could not be started.
        """
        if self.report.refusal is not None:
            raise LocalProcessCleanupError(self.report.refusal)
        status_read, status_write = os.pipe()
        lifeline_read, lifeline_write = os.pipe()
        resources = launch.resources
        plan = {
            "status_fd": status_write,
            "lifeline_fd": lifeline_read,
            "deny_network": self.deny_network,
            "landlock_abi": self.landlock_abi,
            "private_procfs": self.private_procfs,
            "no_new_privs": True,
            "workspace": str(launch.workspace),
            "termination_grace_seconds": float(launch.cleanup_timeout_seconds * 0.25),
            "rlimits": {
                "RLIMIT_CPU": resources.cpu_time_seconds,
                "RLIMIT_AS": resources.memory_bytes,
                "RLIMIT_FSIZE": resources.file_size_bytes,
                "RLIMIT_NPROC": resources.process_count + _SUPERVISOR_PROCESSES,
                "RLIMIT_CORE": 0,
            },
        }
        try:
            process = subprocess.Popen(
                (*_GATE_ARGV, json.dumps(plan, separators=(",", ":")), *launch.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=launch.workspace,
                env=dict(launch.environment),
                start_new_session=True,
                pass_fds=(status_write, lifeline_read),
            )
        except OSError as error:
            for descriptor in (status_read, status_write, lifeline_read, lifeline_write):
                close_fd(descriptor)
            raise LocalProcessCrashError(
                f"local environment process could not start: {type(error).__name__}"
            ) from error
        close_fd(status_write)
        close_fd(lifeline_read)
        contained = LinuxContainedProcess(process, status_read, lifeline_write)
        try:
            contained.await_ready(launch.startup_timeout_seconds)
        except BaseException as error:  # noqa: BLE001 - preserve startup errors after cleanup
            evidence = contained.abort(launch.cleanup_timeout_seconds)
            if evidence:
                error.add_note("local startup cleanup evidence: " + "; ".join(evidence))
            raise
        return contained


class LinuxContainedProcess:
    """One supervised namespace session whose teardown is proven by the supervisor's reap."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        status_fd: int,
        lifeline_fd: int,
    ) -> None:
        """Own the supervisor, its status reader end, and its lifeline writer end."""
        self._process = process
        self._status_fd: int | None = status_fd
        self._status_buffer = bytearray()
        self._lifeline_fd: int | None = lifeline_fd
        self._init_pid: int | None = None
        self._teardown_proven = False

    @property
    def process(self) -> subprocess.Popen[bytes]:
        """Return the supervisor process whose standard streams reach the command."""
        return self._process

    def await_ready(self, timeout_seconds: float) -> None:
        """Wait for the supervisor to report that every planned mechanism is applied.

        Args:
            timeout_seconds: Bound on namespace setup and the command's exec.

        Raises:
            LocalProcessCleanupError: Containment could not be applied in time or at all.
            LocalProcessCrashError: The supervisor died or the command could not execute.
        """
        event = self._next_event(time.monotonic() + timeout_seconds)
        if event is None and self._process.poll() is None:
            raise LocalProcessCleanupError(
                "local process containment could not be applied: the supervisor did not report "
                f"readiness within {timeout_seconds:g}s"
            )
        if event is None:
            raise LocalProcessCrashError(
                "local containment supervisor exited before reporting readiness "
                f"(return code {self._process.poll()})"
            )
        if event.get("event") == "ready":
            init_pid = event.get("init_pid")
            if not isinstance(init_pid, int) or isinstance(init_pid, bool):
                raise LocalProcessCleanupError(
                    "local process containment could not be applied: malformed readiness report"
                )
            self._init_pid = init_pid
            return
        reason = str(event.get("reason", "unknown supervisor failure"))
        if reason.startswith(("namespace setup:", "containment setup:", "init ")):
            raise LocalProcessCleanupError(
                f"local process containment could not be applied: {reason}"
            )
        raise LocalProcessCrashError(f"local environment process could not start: {reason}")

    def terminate(self, timeout_seconds: float) -> None:
        """Ask the supervisor to tear the namespace down and require its reap evidence.

        Args:
            timeout_seconds: Complete grace; the supervisor uses a quarter for SIGTERM.

        Raises:
            LocalProcessCleanupError: The supervisor did not prove the namespace empty.
        """
        if self._teardown_proven:
            return
        deadline = time.monotonic() + timeout_seconds
        self._close_lifeline()
        try:
            self._process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            self._process.kill()
            with suppress(subprocess.TimeoutExpired):
                self._process.wait(timeout=1.0)
            raise LocalProcessCleanupError(
                "local process containment is unproven: the supervisor did not report namespace "
                f"teardown within {timeout_seconds:g}s and was killed"
            ) from None
        if not self._read_reaped(max(deadline, time.monotonic() + 0.1)):
            raise LocalProcessCleanupError(
                "local process containment is unproven: the supervisor exited "
                f"(return code {self._process.returncode}) without namespace teardown evidence"
            )
        self._teardown_proven = True
        self._close_status()

    def abort(self, timeout_seconds: float) -> tuple[str, ...]:
        """Tear down a session that failed before the protocol started, returning evidence.

        Args:
            timeout_seconds: Bound for the supervisor to exit after its lifeline closes.

        Returns:
            Cleanup evidence; empty when teardown was proven and every descriptor closed.
        """
        evidence: list[str] = []
        try:
            self.terminate(timeout_seconds)
        except LocalProcessCleanupError as error:
            evidence.append(str(error))
        for attribute in ("stdin", "stdout", "stderr"):
            stream = getattr(self._process, attribute)
            if stream is None:
                continue
            try:
                stream.raw.close()
            except OSError as error:
                evidence.append(f"{attribute} close failed with {type(error).__name__}")
            setattr(self._process, attribute, None)
        self._close_status()
        return tuple(evidence)

    def _read_reaped(self, deadline: float) -> bool:
        """Consume remaining status events and return whether the namespace reap was reported."""
        while (event := self._next_event(deadline)) is not None:
            if event.get("event") == "reaped":
                return True
        return False

    def _next_event(self, deadline: float) -> JsonObject | None:
        """Return the next status event, or None at end of stream or deadline."""
        status_fd = self._status_fd
        if status_fd is None:
            return None
        with selectors.DefaultSelector() as selector:
            selector.register(status_fd, selectors.EVENT_READ)
            while (newline := self._status_buffer.find(b"\n")) < 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    return None
                chunk = os.read(status_fd, 4096)
                if not chunk:
                    return None
                self._status_buffer.extend(chunk)
        line = bytes(self._status_buffer[:newline])
        del self._status_buffer[: newline + 1]
        try:
            event = json.loads(line)
        except ValueError:
            return None
        return cast(JsonObject, event) if isinstance(event, dict) else None

    def _close_lifeline(self) -> None:
        """Close the lifeline once; its end of stream is the supervisor's termination request."""
        if self._lifeline_fd is not None:
            close_fd(self._lifeline_fd)
            self._lifeline_fd = None

    def _close_status(self) -> None:
        """Close the status descriptor once after teardown evidence is consumed."""
        if self._status_fd is not None:
            close_fd(self._status_fd)
            self._status_fd = None
