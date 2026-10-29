"""Close Capture's mitmproxy servers, including an interrupted local-mode startup."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable

from mitmproxy.addons.proxyserver import Proxyserver
from mitmproxy.proxy.mode_servers import LocalRedirectorInstance

from exp.runtime.capture.watchdog import CaptureWatchdog

_CONNECTION_SHUTDOWN_TIMEOUT = 2.0


async def start_capture_watchdog(proxyserver: Proxyserver) -> CaptureWatchdog | None:
    """Transfer control to an independent watchdog while interception is still disabled."""
    for server in proxyserver.servers:
        if isinstance(server, LocalRedirectorInstance):
            if sys.platform != "darwin":
                raise RuntimeError("Capture's network watchdog requires macOS.")
            native = type(server)._server
            if type(server)._instance is server and native is not None:
                control = await native.take_control_socket()
                return await CaptureWatchdog.start(control)
    return None


def capture_server_closed(proxyserver: Proxyserver) -> Awaitable[None] | None:
    """Observe closure of the currently owned local backend without stopping it.

    The native method clones its shutdown watch receiver for every waiter. This
    observer can run alongside cleanup's wait_closed call, and cancelling it
    does not close the backend or consume another waiter's notification.

    Args:
        proxyserver: The connection manager owned by this Capture process.

    Returns:
        The owned native backend's closure notification, or None when no local
        backend is fully initialized and owned by this manager.
    """
    for server in proxyserver.servers:
        if not isinstance(server, LocalRedirectorInstance):
            continue
        cls = type(server)
        native = cls._server
        if cls._instance is server and native is not None:
            return native.wait_closed()
    return None


async def stop_capture_servers(
    proxyserver: Proxyserver, watchdog: CaptureWatchdog | None = None
) -> None:
    """Stop each owned server and report cleanup failures after attempting all of them.

    Args:
        proxyserver: The server manager belonging to this foreground Capture process.
        watchdog: The independent owner of this manager's control connection, when active.

    Raises:
        RuntimeError: A server could not confirm shutdown.
    """
    errors: list[Exception] = []
    for server in tuple(proxyserver.servers):
        try:
            if isinstance(server, LocalRedirectorInstance):
                await _stop_local_redirector(server, proxyserver, watchdog)
            elif server.is_running:
                await server.stop()
        except Exception as exc:  # noqa: BLE001 - Attempt all owned cleanup before reporting failure.
            errors.append(exc)
    if errors:
        raise RuntimeError(
            "Capture could not confirm network interception shutdown. Disable Mitmproxy "
            "Redirector in macOS System Settings if connections fail."
        ) from errors[0]


async def _stop_local_redirector(
    server: LocalRedirectorInstance,
    proxyserver: Proxyserver,
    watchdog: CaptureWatchdog | None = None,
) -> None:
    """Release only this instance's native redirector, including partial initialization.

    Mitmproxy 12's local mode stores its native handle and owner in class-level
    fields. Startup cancellation can leave an owner without a native handle, and
    ordinary stop deliberately retains the native daemon. This module owns the
    boundary that accesses those internals so foreground Capture can fully close
    its redirector without changing an unrelated owner's state.
    """
    cls = type(server)
    if cls._instance is not server:
        return
    native = cls._server
    if native is None:
        cls._instance = None
        return
    # The macOS selector parser requires at least one action. Include then exclude
    # the same PID to disable every flow, including PID 0, without an empty list.
    # LocalRedirectorInstance.stop() sends an empty selector, so this owned boundary
    # releases its singleton directly before draining and closing the native handle.
    cls._instance = None
    try:
        if watchdog is None:
            native.set_intercept("0,!0")
        else:
            await watchdog.disable()
    finally:
        # Cleanup must never confer authority over a replacement owner.
        if cls._server is native and (cls._instance is None or cls._instance is server):
            try:
                await _quiesce_connections(server, proxyserver)
            finally:
                if cls._server is native and (cls._instance is None or cls._instance is server):
                    native.close()
                    cls._instance = None
                    cls._server = None
                    # Clear ownership before awaiting closure so a new capture can own a
                    # distinct handle without this cleanup clearing it afterward.
                    await native.wait_closed()


async def _quiesce_connections(server: LocalRedirectorInstance, proxyserver: Proxyserver) -> None:
    """Close owned writers and drain transport tasks while their native channels still exist.

    Native streams can report an open writer after their command channel closes.
    Marking each writer closed before releasing the backend prevents pending TLS
    and UDP events from attempting writes through a terminated native channel.
    """
    errors: list[Exception] = []
    tasks: set[asyncio.Task[None]] = set()
    for handler in tuple(proxyserver.connections.values()):
        if handler.client.proxy_mode != server.mode:
            continue
        for transport in tuple(handler.transports.values()):
            if transport.writer is not None:
                try:
                    transport.writer.close()
                except OSError:
                    # A native stream marks itself closed before reporting a lost channel.
                    pass
                except Exception as exc:  # noqa: BLE001 - Close every owned writer first.
                    errors.append(exc)
            if transport.handler is not None and not transport.handler.done():
                tasks.add(transport.handler)
        tasks.update(task for task in handler.wakeup_timer if not task.done())
    for task in tasks:
        task.cancel("Capture is stopping")
    if tasks:
        done, pending = await asyncio.wait(tasks, timeout=_CONNECTION_SHUTDOWN_TIMEOUT)
        for task in done:
            try:
                task.result()
            except (asyncio.CancelledError, OSError):
                pass
            except Exception as exc:  # noqa: BLE001 - Native closure must still be attempted.
                errors.append(exc)
        if pending:
            raise RuntimeError("Capture could not finish closing active network connections.")
    if errors:
        raise RuntimeError("Capture could not close every active network connection.") from errors[
            0
        ]
