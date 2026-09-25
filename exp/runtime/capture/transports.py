"""Contain native stream write failures within the affected Capture connection."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast

import mitmproxy_rs
from mitmproxy import connection
from mitmproxy.addons.proxyserver import Proxyserver
from mitmproxy.proxy.server import ConnectionIO


def guard_native_writer(
    proxyserver: Proxyserver,
    client: connection.Client,
    target: connection.Connection,
    on_failure: Callable[[str], None],
) -> None:
    """Guard one owned native writer before its first connection-processing event.

    Args:
        proxyserver: This Capture process's connection registry.
        client: Client identifying the owning proxy connection.
        target: Client or upstream connection whose writer is now initialized.
        on_failure: Content-free callback receiving the failed client connection ID.

    The client_connected and server_connected hooks run after their ConnectionIO
    exists. Its writer field is the single internal integration boundary here:
    mitmproxy uses write, write_eof, is_closing, drain, and close structurally, but
    annotates only its two concrete upstream writer implementations. The cast
    permits this explicit adapter without modifying global classes or factories.
    Native macOS command channels belong to individual flows, so a write failure
    does not establish that the redirector or another connection has failed.
    """
    handler = proxyserver.connections.get(client.id)
    if handler is None:
        return
    transport = handler.transports.get(target)
    if transport is None or not isinstance(transport.writer, mitmproxy_rs.Stream):
        return
    writer = _NativeWriter(transport.writer, transport, client.id, on_failure)
    transport.writer = cast(asyncio.StreamWriter, writer)


class _NativeWriter:
    """Forward a native writer while containing its documented synchronous OSError."""

    def __init__(
        self,
        stream: mitmproxy_rs.Stream,
        transport: ConnectionIO,
        client_id: str,
        on_failure: Callable[[str], None],
    ) -> None:
        """Retain the native stream and its existing per-connection cancellation target."""
        self._stream = stream
        self._transport = transport
        self._client_id = client_id
        self._on_failure = on_failure
        self._closed = False
        self._failed = False
        self._loop = asyncio.get_running_loop()

    def write(self, data: bytes) -> None:
        """Forward unchanged bytes or retire this writer after a native channel failure."""
        if self._closed:
            return
        try:
            self._stream.write(data)
        except OSError:
            self._fail()

    def write_eof(self) -> None:
        """Keep native half-close behavior and contain a dead command channel."""
        if self._closed:
            return
        try:
            self._stream.write_eof()
        except OSError:
            self._fail()

    def is_closing(self) -> bool:
        """Stop further writes immediately even if the native stream still reports open."""
        return self._closed or self._stream.is_closing()

    async def drain(self) -> None:
        """Preserve native backpressure and mitmproxy's ordinary drain-error handling."""
        if self._failed:
            raise OSError("Capture native connection is closed.")
        try:
            await self._stream.drain()
        except OSError:
            self._fail()
            raise

    def close(self) -> None:
        """Close the owned writer once, including streams whose native receiver has ended."""
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
        except OSError:
            # Native close marks its stream closed before a dead-channel send fails.
            pass

    def _fail(self) -> None:
        """Report once and cancel only the transport attached to this native stream."""
        if self._failed:
            return
        self._failed = True
        try:
            self.close()
        finally:
            # Initial client processing can precede registration of its read task.
            # Resolve the task on the next loop turn instead of retaining None.
            self._loop.call_soon(self._cancel_transport)
            self._on_failure(self._client_id)

    def _cancel_transport(self) -> None:
        """Retire the failed connection after its transport task has been registered."""
        task = self._transport.handler
        if task is not None and not task.done():
            task.cancel("Capture native connection closed")
