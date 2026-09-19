"""Compose a foreground capture lifetime with independent network recovery."""

from __future__ import annotations

import asyncio
import signal
import socket
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from exp.runtime.capture.control import CaptureCloudError, CaptureRun, CaptureRunClient
from exp.runtime.capture.proxy import CaptureProxy
from exp.runtime.capture.resolver import UpstreamResolver
from exp.runtime.capture.system import CaptureSystemSession
from exp.runtime.capture.upload import CaptureUploader, UploadStats


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
) -> UploadStats:
    """Serve provider traffic until interrupted, always restoring owned networking.

    Args:
        domains: Exact provider hostnames chosen for this capture run.
        ca_directory: Private directory holding the trusted capture CA.
        uploader: Nonblocking bounded trace queue and background uploader.
        control: Normal authenticated Platform capture-run client.
        run: Organization-bound run acknowledged before host changes.
        on_started: Terminal callback once the proxy and hosts helper are ready.
        on_progress: Terminal callback receiving content-free upload counters.
        on_warning: Terminal callback for recoverable upload failures.

    Returns:
        Final upload counters after bounded flushing.

    Raises:
        RuntimeError: Proxy startup or the privileged networking helper fails.
    """
    stop = asyncio.Event()
    ready = asyncio.Event()
    loop = asyncio.get_running_loop()
    resolver = UpstreamResolver()
    proxy = CaptureProxy(sink=uploader.submit, domains=domains, resolver=resolver)
    original_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in original_signals:
        loop.add_signal_handler(sig, stop.set)
    helper: CaptureSystemSession | None = None
    proxy_task: asyncio.Task[None] | None = None
    heartbeat_task: asyncio.Task[None] | None = None
    uploader_started = False
    try:
        await resolver.prime(domains)
        if stop.is_set():
            return uploader.stats
        port = _available_port()
        proxy_task = asyncio.create_task(
            proxy.serve(port=port, ca_directory=ca_directory, ready=ready.set)
        )
        await _wait_for_proxy(proxy_task, ready)
        if stop.is_set():
            return uploader.stats
        helper = await asyncio.to_thread(
            CaptureSystemSession.start, upstream_port=port, domains=domains
        )
        uploader.start()
        uploader_started = True
        if not stop.is_set():
            on_started()
        next_heartbeat = time.monotonic() + 15.0
        while not stop.is_set():
            if proxy_task.done():
                await proxy_task
                raise RuntimeError(
                    "Capture proxy stopped unexpectedly. Networking is being restored."
                )
            await asyncio.to_thread(helper.heartbeat)
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
        try:
            if helper is not None:
                await asyncio.to_thread(helper.close)
        finally:
            try:
                proxy.shutdown()
                if proxy_task is not None:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(proxy_task, return_exceptions=True), timeout=5.0
                        )
                    except TimeoutError:
                        on_warning(
                            "Proxy shutdown timed out. Run exp capture reset if requests fail."
                        )
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
    stats = uploader.stats
    return replace(stats, dropped_exchanges=stats.dropped_exchanges + proxy.dropped_exchanges)


def _available_port() -> int:
    """Choose a loopback port, with proxy startup detecting any intervening bind."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


async def _wait_for_proxy(task: asyncio.Task[None], ready: asyncio.Event) -> None:
    """Require real proxy readiness before permitting privileged host changes."""
    readiness = asyncio.create_task(ready.wait())
    try:
        done, _ = await asyncio.wait(
            {task, readiness}, timeout=15.0, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            await task
            raise RuntimeError("Capture proxy stopped before it was ready.")
        if readiness not in done:
            raise RuntimeError(
                "Capture proxy did not become ready; no hosts overrides were installed."
            )
    finally:
        readiness.cancel()
        await asyncio.gather(readiness, return_exceptions=True)
