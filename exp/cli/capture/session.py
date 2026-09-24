"""Own foreground network interception, cloud heartbeats, and bounded shutdown."""

from __future__ import annotations

import asyncio
import signal
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

from exp.runtime.capture.control import CaptureCloudError, CaptureRun, CaptureRunClient
from exp.runtime.capture.health import CaptureHealth, CaptureHealthFailure
from exp.runtime.capture.proxy import CaptureBypassReason, CaptureProxy
from exp.runtime.capture.upload import CaptureUploader, UploadStats

_SHUTDOWN_TIMEOUT = 5.0
_STARTUP_TIMEOUT = 180.0
_WAITING_NOTICE_DELAY = 3.0


async def run_session(
    *,
    domains: tuple[str, ...],
    ca_directory: Path,
    uploader: CaptureUploader,
    control: CaptureRunClient,
    run: CaptureRun,
    on_started: Callable[[], None],
    on_progress: Callable[[UploadStats], None],
    on_warning: Callable[[str], None],
    on_waiting: Callable[[], None] | None = None,
    on_bypass: Callable[[str, str, CaptureBypassReason], None] | None = None,
    on_diagnostic: Callable[[str], None] | None = None,
) -> UploadStats:
    """Serve provider traffic until interrupted, always releasing interception.

    Args:
        domains: Exact provider hostnames chosen for this capture run.
        ca_directory: Private directory holding the trusted capture CA.
        uploader: Nonblocking bounded trace queue and background uploader.
        control: Normal authenticated Platform capture-run client.
        run: Organization-bound run acknowledged before interception.
        on_started: Terminal callback once network interception is ready.
        on_progress: Terminal callback receiving content-free upload counters.
        on_warning: Terminal callback for recoverable upload failures.
        on_waiting: Optional callback when network startup remains pending.
        on_bypass: Optional callback naming an app and host excluded after trust rejection.
        on_diagnostic: Optional content-free transport events for verbose output.

    Returns:
        Final upload counters after bounded flushing.

    Raises:
        RuntimeError: The local network redirector fails to start or stops unexpectedly.
    """
    stop = asyncio.Event()
    ready = asyncio.Event()
    loop = asyncio.get_running_loop()
    proxy = CaptureProxy(
        sink=uploader.submit, domains=domains, on_bypass=on_bypass, on_diagnostic=on_diagnostic
    )
    health = CaptureHealth(domains, on_diagnostic=on_diagnostic)
    original_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in original_signals:
        loop.add_signal_handler(sig, stop.set)
    proxy_task: asyncio.Task[None] | None = None
    heartbeat_task: asyncio.Task[None] | None = None
    health_task: asyncio.Task[CaptureHealthFailure] | None = None
    network_failure: CaptureHealthFailure | None = None
    remaining_failures: tuple[CaptureHealthFailure, ...] = ()
    uploader_started = False
    try:
        baseline = await _check_before_capture(health, stop)
        if baseline is None:
            return uploader.stats
        if baseline:
            raise RuntimeError(
                f"Cannot resolve {baseline[0].host}. Check your connection and try again."
                if baseline[0].kind == "dns"
                else "Capture could not check network health. Try again."
            )
        proxy_task = asyncio.create_task(proxy.serve(ca_directory=ca_directory, ready=ready.set))
        if not await _wait_for_proxy(proxy_task, ready, stop, on_waiting=on_waiting):
            return uploader.stats
        uploader.start()
        uploader_started = True
        health_task = asyncio.create_task(health.watch())
        if not stop.is_set():
            on_started()
        next_heartbeat = time.monotonic() + 15.0
        while not stop.is_set():
            if proxy_task.done():
                await proxy_task
                raise RuntimeError(
                    "Capture proxy stopped unexpectedly. Local interception is stopping."
                )
            if health_task.done():
                network_failure = await health_task
                break
            stats = uploader.stats
            stats = replace(
                stats, dropped_exchanges=stats.dropped_exchanges + proxy.dropped_exchanges
            )
            on_progress(stats)
            if heartbeat_task is not None and heartbeat_task.done():
                try:
                    await heartbeat_task
                except CaptureCloudError:
                    on_warning("Platform is temporarily unavailable; capture uploads will retry.")
                heartbeat_task = None
            if heartbeat_task is None and time.monotonic() >= next_heartbeat:
                heartbeat_task = asyncio.create_task(
                    control.heartbeat(
                        run,
                        pending_batches=uploader.pending_current_run,
                        upload_errors=stats.upload_errors,
                    )
                )
                next_heartbeat = time.monotonic() + 15.0
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except TimeoutError:
                continue
    finally:
        if health_task is not None:
            health_task.cancel()
            await asyncio.gather(health_task, return_exceptions=True)
        proxy.shutdown()
        try:
            if proxy_task is not None:
                if not ready.is_set() and not proxy_task.done():
                    proxy_task.cancel()
                try:
                    await _finish_proxy(proxy_task)
                except (RuntimeError, OSError):
                    # A diagnostic must not replace an unconfirmed shutdown or backend error.
                    if ready.is_set():
                        with suppress(Exception):
                            failures = await health.check()
                            if failures:
                                on_warning(
                                    f"DNS for {failures[0].host} is also unavailable; "
                                    "check your connection before retrying Capture."
                                    if failures[0].kind == "dns"
                                    else "Capture could not verify network health."
                                )
                    raise
                if ready.is_set():
                    remaining_failures = await health.check()
        finally:
            try:
                if heartbeat_task is not None:
                    heartbeat_task.cancel()
                    await asyncio.gather(heartbeat_task, return_exceptions=True)
                if uploader_started:
                    await asyncio.to_thread(uploader.close, timeout=5.0)
            finally:
                for sig, handler in original_signals.items():
                    loop.remove_signal_handler(sig)
                    signal.signal(sig, handler)
    if network_failure is not None:
        for failure in remaining_failures:
            if failure.host != network_failure.host:
                on_warning(
                    f"DNS for {failure.host} is unavailable after Capture stopped; "
                    "check your connection before retrying."
                    if failure.kind == "dns"
                    else f"Capture could not verify network health for {failure.host}."
                )
        remaining_failures = tuple(
            failure for failure in remaining_failures if failure.host == network_failure.host
        )
    if remaining_failures:
        failure = remaining_failures[0]
        continued = (
            "still " if network_failure is not None and network_failure.kind == "dns" else ""
        )
        raise RuntimeError(
            f"Capture stopped, but DNS for {failure.host} is {continued}unavailable. "
            "Check your connection before retrying."
            if failure.kind == "dns"
            else "Capture stopped. Network health could not be verified; check your connection."
        )
    if network_failure is not None:
        raise RuntimeError(
            f"Capture stopped after repeated DNS failures for {network_failure.host}. "
            "DNS is responding again."
            if network_failure.kind == "dns"
            else "Capture stopped because its network health check became unavailable."
        )
    stats = uploader.stats
    return replace(stats, dropped_exchanges=stats.dropped_exchanges + proxy.dropped_exchanges)


async def _check_before_capture(
    health: CaptureHealth, stop: asyncio.Event
) -> tuple[CaptureHealthFailure, ...] | None:
    """Cancel and reap the initial DNS check promptly if startup is interrupted."""
    checked = asyncio.create_task(health.check())
    interrupted = asyncio.create_task(stop.wait())
    try:
        await asyncio.wait((checked, interrupted), return_when=asyncio.FIRST_COMPLETED)
        return None if stop.is_set() else checked.result()
    finally:
        checked.cancel()
        interrupted.cancel()
        await asyncio.gather(checked, interrupted, return_exceptions=True)


async def _wait_for_proxy(
    task: asyncio.Task[None],
    ready: asyncio.Event,
    stop: asyncio.Event,
    on_waiting: Callable[[], None] | None = None,
) -> bool:
    """Report pending startup once without extending approval or cancellation bounds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _STARTUP_TIMEOUT
    readiness = asyncio.create_task(ready.wait())
    interrupted = asyncio.create_task(stop.wait())
    notice = asyncio.create_task(asyncio.sleep(_WAITING_NOTICE_DELAY))
    waiting = {task, readiness, interrupted, notice}
    try:
        while True:
            done, _ = await asyncio.wait(
                waiting,
                timeout=max(0.0, deadline - loop.time()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if task.done():
                await task
                raise RuntimeError("Capture proxy stopped before it was ready.")
            if stop.is_set():
                return False
            if ready.is_set():
                return True
            if not done:
                raise RuntimeError(
                    "Capture could not start the macOS network extension. Enable Mitmproxy "
                    "Redirector in System Settings > General > Login Items & Extensions > "
                    "Network Extensions, then run exp capture again."
                )
            if notice in done:
                waiting.remove(notice)
                if on_waiting is not None:
                    on_waiting()
    finally:
        readiness.cancel()
        interrupted.cancel()
        notice.cancel()
        await asyncio.gather(readiness, interrupted, notice, return_exceptions=True)


async def _finish_proxy(task: asyncio.Task[None]) -> None:
    """Require confirmed backend shutdown before reporting interception disabled."""
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=_SHUTDOWN_TIMEOUT)
    except TimeoutError as exc:
        task.cancel()
        raise RuntimeError(
            "Network extension shutdown could not be confirmed. Disable Mitmproxy Redirector "
            "in macOS System Settings before retrying Capture."
        ) from exc
    except asyncio.CancelledError:
        if not task.cancelled():
            raise
