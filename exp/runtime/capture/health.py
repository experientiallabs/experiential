"""Check system DNS quietly with bounded, independently cancellable resolver children."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

from exp.runtime.capture.policy import validate_domains

_CHECK_INTERVAL = 5.0
_LOOKUP_TIMEOUT = 3.0
_CONSECUTIVE_FAILURES = 2
_LOOKUP_SCRIPT = "import socket, sys; socket.getaddrinfo(sys.argv[1], 443, type=socket.SOCK_STREAM)"


@dataclass(frozen=True)
class CaptureHealthFailure:
    """Describe an unavailable system lookup without retaining resolver output."""

    host: str
    kind: Literal["dns", "monitor"]


class CaptureHealth:
    """Observe selected providers through the system resolver without sending HTTP requests."""

    def __init__(
        self, domains: tuple[str, ...], *, on_diagnostic: Callable[[str], None] | None = None
    ) -> None:
        """Keep the same validated, bounded host selection as Capture's traffic policy."""
        self._domains = validate_domains(domains)
        self._on_diagnostic = on_diagnostic

    async def check(self) -> tuple[CaptureHealthFailure, ...]:
        """Return failed lookups from one concurrent, bounded check of all selected hosts."""
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(_lookup(host, self._on_diagnostic)) for host in self._domains
            ]
        return tuple(failure for task in tasks if (failure := task.result()) is not None)

    async def watch(self) -> CaptureHealthFailure:
        """Wait quietly until one host fails consecutive checks, resetting on recovery.

        Checks never overlap. Each completed check is followed by a quiet interval;
        the initial interval lets callers establish their own startup baseline.
        """
        consecutive: dict[str, int] = {}
        while True:
            await asyncio.sleep(_CHECK_INTERVAL)
            failures = await self.check()
            consecutive = {
                failure.host: consecutive.get(failure.host, 0) + 1 for failure in failures
            }
            for failure in failures:
                if consecutive[failure.host] >= _CONSECUTIVE_FAILURES:
                    return failure


async def _lookup(
    host: str, on_diagnostic: Callable[[str], None] | None
) -> CaptureHealthFailure | None:
    """Bound native resolution in an owned process and reap it on timeout or cancellation.

    Python's native resolver can outlive cancellation in a worker thread. A small
    isolated child preserves system DNS behavior while making that work stoppable.
    No user credentials, Python startup configuration, or diagnostic output pass
    through this probe.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            _LOOKUP_SCRIPT,
            host,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin"},
        )
    except OSError:
        _diagnose(on_diagnostic, f"dns_check_failed: could not start resolver check · {host}")
        return CaptureHealthFailure(host, "monitor")
    try:
        await asyncio.wait_for(process.wait(), timeout=_LOOKUP_TIMEOUT)
    except TimeoutError:
        _diagnose(on_diagnostic, f"dns_check_failed: timed out after {_LOOKUP_TIMEOUT:g}s · {host}")
        return CaptureHealthFailure(host, "dns")
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
    if process.returncode == 0:
        return None
    _diagnose(
        on_diagnostic, f"dns_check_failed: resolver exited with code {process.returncode} · {host}"
    )
    return CaptureHealthFailure(host, "dns")


def _diagnose(callback: Callable[[str], None] | None, message: str) -> None:
    """Keep optional terminal output from changing health results or resolver cleanup."""
    if callback is not None:
        with suppress(Exception):
            callback(message)
