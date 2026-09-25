"""Exercise scoped native stream failures without activating network interception."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Generator
from typing import Literal
from unittest.mock import AsyncMock, Mock

import mitmproxy_rs
import pytest
from mitmproxy import connection, options
from mitmproxy.addons.proxyserver import Proxyserver
from mitmproxy.proxy import commands, context, events, layer, mode_specs
from mitmproxy.proxy.mode_servers import ProxyConnectionHandler
from mitmproxy.proxy.server import ConnectionIO

from exp.runtime.capture.transports import guard_native_writer


def _registered_native_writer() -> tuple[Proxyserver, connection.Client, ConnectionIO, Mock]:
    """Register one synthetic native transport behind the real connection registry."""
    client = connection.Client(peername=("127.0.0.1", 10001), sockname=("127.0.0.1", 10002))
    native = Mock(spec=mitmproxy_rs.Stream)
    native.is_closing.return_value = False
    native.drain = AsyncMock()
    transport = ConnectionIO(reader=native, writer=native)
    handler = Mock(spec=ProxyConnectionHandler)
    handler.transports = {client: transport}
    manager = Proxyserver()
    manager.connections[client.id] = handler
    return manager, client, transport, native


def test_guard_forwards_native_success_and_preserves_reader() -> None:
    """Keep original bytes, backpressure, and reader ownership for healthy native streams."""

    async def scenario() -> None:
        """Run the writer surface without changing the native transport's successful behavior."""
        manager, client, transport, native = _registered_native_writer()
        failures: list[str] = []
        guard_native_writer(manager, client, client, failures.append)
        writer = transport.writer
        assert writer is not None and writer is not native
        assert transport.reader is native
        guard_native_writer(manager, client, client, failures.append)
        assert transport.writer is writer
        writer.write(b"unchanged synthetic bytes")
        await writer.drain()
        writer.write_eof()
        assert not writer.is_closing()
        writer.close()
        writer.close()
        assert writer.is_closing()
        native.write.assert_called_once_with(b"unchanged synthetic bytes")
        native.drain.assert_awaited_once()
        native.write_eof.assert_called_once_with()
        native.close.assert_called_once_with()
        assert failures == []

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["write", "write_eof", "drain"])
def test_native_oserror_cancels_only_its_transport_after_registration(
    operation: Literal["write", "write_eof", "drain"],
) -> None:
    """A late handler registration still receives cancellation and only one failure report."""

    async def scenario() -> None:
        """Break one native channel while an independent connection task remains alive."""
        manager, client, transport, native = _registered_native_writer()
        getattr(native, operation).side_effect = OSError("Server has been shut down.")
        native.close.side_effect = OSError("Server has been shut down.")
        failures: list[str] = []
        guard_native_writer(manager, client, client, failures.append)
        writer = transport.writer
        assert writer is not None
        sibling = asyncio.create_task(asyncio.Event().wait())
        try:
            assert transport.handler is None
            if operation == "write":
                writer.write(b"reply")
            elif operation == "write_eof":
                writer.write_eof()
            else:
                with pytest.raises(OSError):
                    await writer.drain()
            assert writer.is_closing()
            assert failures == [client.id]
            transport.handler = asyncio.create_task(asyncio.Event().wait())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert transport.handler.cancelled()
            assert not sibling.done()
            writer.write(b"late reply")
            writer.write_eof()
            writer.close()
            assert failures == [client.id]
            native.close.assert_called_once_with()
        finally:
            sibling.cancel()
            await asyncio.gather(sibling, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["write", "write_eof", "drain", "close"])
def test_unexpected_native_errors_are_not_suppressed(
    operation: Literal["write", "write_eof", "drain", "close"],
) -> None:
    """Only documented OSError transport failures receive scoped containment."""

    async def scenario() -> None:
        """Keep an unexpected native programming error visible to the caller."""
        manager, client, transport, native = _registered_native_writer()
        getattr(native, operation).side_effect = ValueError("unexpected native failure")
        failures: list[str] = []
        guard_native_writer(manager, client, client, failures.append)
        writer = transport.writer
        assert writer is not None
        with pytest.raises(ValueError, match="unexpected native failure"):
            if operation == "write":
                writer.write(b"reply")
            elif operation == "write_eof":
                writer.write_eof()
            elif operation == "drain":
                await writer.drain()
            else:
                writer.close()
        assert failures == []

    asyncio.run(scenario())


def test_guard_covers_upstream_native_writer_and_leaves_other_transports_untouched() -> None:
    """Server hooks guard new native upstream streams without wrapping normal asyncio writers."""

    async def scenario() -> None:
        """Resolve only the requested connection within its owning client's registry entry."""
        manager, client, client_transport, native = _registered_native_writer()
        target = connection.Server(address=("127.0.0.1", 10003))
        server_transport = ConnectionIO(reader=native, writer=native)
        manager.connections[client.id].transports[target] = server_transport
        failures: list[str] = []
        guard_native_writer(manager, client, target, failures.append)
        assert server_transport.writer is not native
        assert server_transport.reader is native
        assert client_transport.writer is native
        ordinary = Mock(spec=asyncio.StreamWriter)
        client_transport.writer = ordinary
        guard_native_writer(manager, client, client, failures.append)
        assert client_transport.writer is ordinary
        missing = connection.Server(address=("127.0.0.1", 10004))
        guard_native_writer(manager, client, missing, failures.append)
        manager.connections.clear()
        guard_native_writer(manager, client, target, failures.append)
        assert failures == []

    asyncio.run(scenario())


def test_drain_cancellation_remains_cancellation() -> None:
    """Normal task cancellation must not report a native failure or alter sibling lifetimes."""

    async def scenario() -> None:
        """Cancel native backpressure through its ordinary asyncio exception path."""
        manager, client, transport, native = _registered_native_writer()
        native.drain.side_effect = asyncio.CancelledError()
        failures: list[str] = []
        guard_native_writer(manager, client, client, failures.append)
        writer = transport.writer
        assert writer is not None
        with pytest.raises(asyncio.CancelledError):
            await writer.drain()
        assert failures == []
        native.close.assert_not_called()

    asyncio.run(scenario())


class _ReplyLayer(layer.Layer):
    """Emit a real SendData command through mitmproxy's ordinary writer dispatch."""

    def __init__(self, ctx: context.Context) -> None:
        """Retain the client context that receives synthetic local replies."""
        super().__init__(ctx)

    def _handle_event(self, event: events.Event) -> Generator[commands.Command, None, None]:
        """Write one synthetic datagram for every dispatched event."""
        yield commands.SendData(self.context.client, b"synthetic reply")


def test_real_native_udp_receiver_loss_does_not_crash_dispatch_or_break_sibling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reproduce native is_closing=False after receiver loss and contain real SendData errors."""

    async def scenario() -> None:
        """Use two loopback UDP servers with no provider requests or system interception."""
        failed_stream: asyncio.Future[mitmproxy_rs.Stream] = asyncio.Future()
        healthy_stream: asyncio.Future[mitmproxy_rs.Stream] = asyncio.Future()
        release = asyncio.Event()

        async def receive_failed(stream: mitmproxy_rs.Stream) -> None:
            """Retain the first stream while its own native server is deliberately closed."""
            failed_stream.set_result(stream)
            await release.wait()

        async def receive_healthy(stream: mitmproxy_rs.Stream) -> None:
            """Keep an independent native stream available for normal replies."""
            healthy_stream.set_result(stream)
            await release.wait()

        failed_backend = await mitmproxy_rs.udp.start_udp_server("127.0.0.1", 0, receive_failed)
        healthy_backend = await mitmproxy_rs.udp.start_udp_server("127.0.0.1", 0, receive_healthy)
        failed_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        healthy_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        healthy_sender.setblocking(False)
        failures: list[str] = []
        tasks: list[asyncio.Task[bool]] = []
        try:
            failed_sender.sendto(b"synthetic request", failed_backend.getsockname())
            healthy_sender.sendto(b"synthetic request", healthy_backend.getsockname())
            broken = await asyncio.wait_for(failed_stream, 2)
            healthy = await asyncio.wait_for(healthy_stream, 2)
            manager = Proxyserver()
            handlers: list[ProxyConnectionHandler] = []
            for stream in (broken, healthy):
                handler = ProxyConnectionHandler(
                    Mock(), stream, stream, options.Options(), mode_specs.ProxyMode.parse("local")
                )
                handler.layer = _ReplyLayer(handler.layer.context)
                manager.connections[handler.client.id] = handler
                task = asyncio.create_task(asyncio.Event().wait())
                tasks.append(task)
                handler.transports[handler.client].handler = task
                guard_native_writer(manager, handler.client, handler.client, failures.append)
                handlers.append(handler)
            failed_backend.close()
            await asyncio.wait_for(failed_backend.wait_closed(), 2)
            assert not broken.is_closing()
            await handlers[0].server_event(events.Start())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert tasks[0].cancelled()
            assert broken.is_closing()
            assert failures == [handlers[0].client.id]
            await handlers[0].server_event(events.Start())
            assert failures == [handlers[0].client.id]
            await handlers[1].server_event(events.Start())
            received = await asyncio.wait_for(
                asyncio.get_running_loop().sock_recv(healthy_sender, 1024), 2
            )
            assert received == b"synthetic reply"
            assert not tasks[1].done()
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            failed_sender.close()
            healthy_sender.close()
            failed_backend.close()
            healthy_backend.close()
            await asyncio.wait_for(failed_backend.wait_closed(), 2)
            await asyncio.wait_for(healthy_backend.wait_closed(), 2)

    asyncio.run(scenario())
    assert "mitmproxy has crashed" not in caplog.text
    assert "Server has been shut down" not in caplog.text
