"""DNS monitoring is quiet, bounded, and cannot leave a stuck resolver child behind."""

import asyncio
import os
import sys
from collections.abc import Callable
from unittest.mock import AsyncMock, Mock

import pytest

from exp.runtime.capture import health
from exp.runtime.capture.health import CaptureHealth, CaptureHealthFailure


@pytest.mark.parametrize(
    "domains",
    [(), ("localhost",), ("https://chatgpt.com/",), ("chatgpt.com;exit",), ("a.com",) * 33],
)
def test_monitor_requires_valid_bounded_provider_hosts(domains: tuple[str, ...]) -> None:
    """Invalid targets are rejected before any child can be launched."""
    with pytest.raises(ValueError):
        CaptureHealth(domains)


@pytest.mark.parametrize("returncode", [0, 1, -9])
@pytest.mark.parametrize("verbose", [False, True])
def test_lookup_uses_isolated_system_resolver_without_credentials_or_output(
    monkeypatch: pytest.MonkeyPatch, returncode: int, verbose: bool
) -> None:
    """Success depends on resolver completion without opening an HTTP connection."""
    process = Mock(spec=asyncio.subprocess.Process)
    process.returncode = returncode
    process.wait = AsyncMock(return_value=returncode)
    create = AsyncMock(return_value=process)
    monkeypatch.setattr(health.asyncio, "create_subprocess_exec", create)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-must-not-inherit")
    diagnostics: list[str] = []
    result = asyncio.run(
        CaptureHealth(
            ("chatgpt.com",), on_diagnostic=diagnostics.append if verbose else None
        ).check()
    )
    assert result == (() if returncode == 0 else (CaptureHealthFailure("chatgpt.com", "dns"),))
    create.assert_awaited_once_with(
        sys.executable,
        "-I",
        "-c",
        "import socket, sys; socket.getaddrinfo(sys.argv[1], 443, type=socket.SOCK_STREAM)",
        "chatgpt.com",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env={"PATH": "/usr/bin:/bin"},
    )
    process.kill.assert_not_called()
    assert diagnostics == (
        [f"dns_check_failed: resolver exited with code {returncode} · chatgpt.com"]
        if verbose and returncode != 0
        else []
    )


def test_launch_failure_reports_monitor_unavailability(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable executable is distinct from a failed DNS lookup and leaks no exception."""
    monkeypatch.setattr(
        health.asyncio,
        "create_subprocess_exec",
        AsyncMock(side_effect=OSError("synthetic private launch detail")),
    )
    diagnostics: list[str] = []
    assert asyncio.run(
        CaptureHealth(("chatgpt.com",), on_diagnostic=diagnostics.append).check()
    ) == (CaptureHealthFailure("chatgpt.com", "monitor"),)
    assert diagnostics == ["dns_check_failed: could not start resolver check · chatgpt.com"]


@pytest.mark.parametrize("launch_error", [False, True])
def test_broken_diagnostics_preserve_lookup_failure(
    monkeypatch: pytest.MonkeyPatch, launch_error: bool
) -> None:
    """Terminal write errors cannot replace the monitor's DNS or launch failure result."""
    process = Mock(spec=asyncio.subprocess.Process)
    process.returncode = 1
    process.wait = AsyncMock(return_value=1)
    create = AsyncMock(side_effect=OSError() if launch_error else None, return_value=process)
    monkeypatch.setattr(health.asyncio, "create_subprocess_exec", create)
    diagnostic = Mock(side_effect=OSError("synthetic closed terminal"))
    assert asyncio.run(CaptureHealth(("chatgpt.com",), on_diagnostic=diagnostic).check()) == (
        CaptureHealthFailure("chatgpt.com", "monitor" if launch_error else "dns"),
    )
    diagnostic.assert_called_once()


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("host_count", [1, 3])
@pytest.mark.parametrize("broken_diagnostics", [False, True])
def test_stuck_resolver_process_is_killed_and_reaped(
    monkeypatch: pytest.MonkeyPatch, cancelled: bool, host_count: int, broken_diagnostics: bool
) -> None:
    """A real owned child cannot survive the probe's deadline or cancellation."""
    original_create = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []
    domains = tuple(f"provider{index}.example.com" for index in range(host_count))
    diagnostics: list[str] = []
    monkeypatch.setattr(health, "_LOOKUP_TIMEOUT", 0.1 if not cancelled else 30)

    def diagnostic(message: str) -> None:
        """Represent a closed terminal without preventing cleanup of real resolver children."""
        diagnostics.append(message)
        if broken_diagnostics:
            raise OSError("synthetic closed terminal")

    async def run() -> None:
        """Substitute a sleeping child so the test never uses DNS or system interception."""
        started = asyncio.Event()

        async def create(
            *command: str, stdin: int, stdout: int, stderr: int, env: dict[str, str]
        ) -> asyncio.subprocess.Process:
            """Record the real child while retaining the production isolation settings."""
            process = await original_create(
                sys.executable,
                "-I",
                "-c",
                "import time; time.sleep(60)",
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                env=env,
            )
            processes.append(process)
            if len(processes) == host_count:
                started.set()
            return process

        monkeypatch.setattr(health.asyncio, "create_subprocess_exec", create)
        baseline = asyncio.all_tasks()
        task = asyncio.create_task(CaptureHealth(domains, on_diagnostic=diagnostic).check())
        await asyncio.wait_for(started.wait(), 3)
        if cancelled:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
        else:
            assert await asyncio.wait_for(task, 3) == tuple(
                CaptureHealthFailure(host, "dns") for host in domains
            )
        assert asyncio.all_tasks() == baseline
        assert sorted(diagnostics) == (
            []
            if cancelled
            else sorted(f"dns_check_failed: timed out after 0.1s · {host}" for host in domains)
        )

    try:
        asyncio.run(run())
        assert len(processes) == host_count
        for process in processes:
            assert process.returncode is not None
            with pytest.raises(ChildProcessError):
                os.waitpid(process.pid, os.WNOHANG)
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()


def test_check_runs_each_unique_host_concurrently_within_domain_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One slow domain cannot serialize the batch, and duplicate targets create no extra work."""
    domains = tuple(f"provider{index}.example.com" for index in range(31))

    async def run() -> None:
        """Hold every synthetic lookup until the complete bounded batch has started."""
        started: set[str] = set()
        all_started = asyncio.Event()

        async def lookup(
            host: str, on_diagnostic: Callable[[str], None] | None
        ) -> CaptureHealthFailure | None:
            """Require all 31 unique targets to start without using an external resolver."""
            assert host not in started
            started.add(host)
            if len(started) == len(domains):
                all_started.set()
            await all_started.wait()
            return CaptureHealthFailure(host, "dns") if host == domains[3] else None

        monkeypatch.setattr(health, "_lookup", lookup)
        assert await asyncio.wait_for(CaptureHealth((*domains, domains[0])).check(), 1) == (
            CaptureHealthFailure(domains[3], "dns"),
        )
        assert started == set(domains)

    asyncio.run(run())


def test_watch_requires_consecutive_failure_for_same_host_and_resets_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alternating providers and transient failures do not stop capture prematurely."""
    first = CaptureHealthFailure("api.openai.com", "dns")
    second = CaptureHealthFailure("chatgpt.com", "dns")
    checks = AsyncMock(side_effect=[(first,), (second,), (), (first,), (first,)])
    monkeypatch.setattr(CaptureHealth, "check", checks)
    monkeypatch.setattr(health, "_CHECK_INTERVAL", 0)
    assert asyncio.run(CaptureHealth((first.host, second.host)).watch()) == first
    assert checks.await_count == 5


def test_watch_waits_quietly_between_checks_and_never_overlaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow lookup batch finishes before the next interval or batch can begin."""
    failure = CaptureHealthFailure("chatgpt.com", "monitor")
    sequence: list[str] = []
    original_sleep = asyncio.sleep

    async def pause(delay: float) -> None:
        """Record only the monitor interval while yielding the event loop."""
        assert delay == 5
        sequence.append("pause")
        await original_sleep(0)

    async def check() -> tuple[CaptureHealthFailure, ...]:
        """Ensure each check owns the entire interval between recorded pause events."""
        sequence.append("start")
        await original_sleep(0)
        assert sequence[-1] == "start"
        sequence.append("end")
        return (failure,)

    checks = AsyncMock(side_effect=check)
    monkeypatch.setattr(CaptureHealth, "check", checks)
    monkeypatch.setattr(health.asyncio, "sleep", pause)
    assert asyncio.run(CaptureHealth((failure.host,)).watch()) == failure
    assert sequence == ["pause", "start", "end", "pause", "start", "end"]


def test_watch_can_stop_during_quiet_interval_without_launching_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping Capture promptly cancels a sleeping monitor without creating a child."""
    checks = AsyncMock(side_effect=AssertionError("unexpected lookup"))
    monkeypatch.setattr(CaptureHealth, "check", checks)

    async def run() -> None:
        """Cancel after the monitor has started its initial quiet interval."""
        baseline = asyncio.all_tasks()
        task = asyncio.create_task(CaptureHealth(("chatgpt.com",)).watch())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert asyncio.all_tasks() == baseline

    asyncio.run(run())
    checks.assert_not_awaited()
