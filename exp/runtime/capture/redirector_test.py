"""Exercise native redirector lifetimes with real mode instances and fake OS handles."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from collections.abc import Awaitable, Callable
from typing import Literal
from unittest.mock import Mock

import mitmproxy_rs
import pytest
from mitmproxy import options
from mitmproxy.addons.proxyserver import Proxyserver
from mitmproxy.proxy import commands, context, events, layer
from mitmproxy.proxy.mode_servers import (
    LocalRedirectorInstance,
    ProxyConnectionHandler,
    RegularInstance,
)

from exp.runtime.capture import redirector
from exp.runtime.capture.redirector import capture_server_closed, stop_capture_servers


class FakeNative:
    """Record native operations without installing or activating any system component."""

    def __init__(self, *, fail_clear: bool = False) -> None:
        """Configure an optional interception-disable failure."""
        self.events: list[str] = []
        self.fail_clear = fail_clear

    def set_intercept(self, spec: str) -> None:
        """Record the exact specification sent by the real mitmproxy instance."""
        self.events.append(f"intercept:{spec}")
        if not spec:
            raise IndexError("the macOS redirector requires a nonempty action list")
        if spec == "0,!0" and self.fail_clear:
            raise OSError("control channel disconnected")

    def close(self) -> None:
        """Record closing the native control channel."""
        self.events.append("close")

    async def wait_closed(self) -> None:
        """Record acknowledgement of native shutdown."""
        self.events.append("closed")


@pytest.fixture(autouse=True)
def isolated_native_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test's native handle isolated from other mode-instance tests."""
    monkeypatch.setattr(LocalRedirectorInstance, "_instance", None)
    monkeypatch.setattr(LocalRedirectorInstance, "_server", None)


def _local(proxyserver: Proxyserver) -> LocalRedirectorInstance:
    """Register a real local-mode instance without starting the operating-system backend."""
    server = LocalRedirectorInstance.make("local", proxyserver)
    proxyserver.servers._instances[server.mode] = server
    return server


def _fake_start(monkeypatch: pytest.MonkeyPatch, native: FakeNative) -> None:
    """Replace only native startup; mitmproxy's real start and stop methods still execute."""

    async def start(
        tcp: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
        udp: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
    ) -> FakeNative:
        """Supply an inert native redirector handle."""
        return native

    monkeypatch.setattr(mitmproxy_rs.local, "start_local_redirector", start)


def test_cancelled_startup_releases_incomplete_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation at macOS approval leaves no singleton blocking a future capture."""

    async def scenario() -> None:
        """Cancel the real local-mode startup while its fake native operation is pending."""
        entered = asyncio.Event()

        async def pending(
            tcp: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
            udp: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
        ) -> FakeNative:
            """Model the unbounded native approval wait without activating anything."""
            entered.set()
            await asyncio.Future[None]()
            raise AssertionError("pending startup unexpectedly completed")

        monkeypatch.setattr(mitmproxy_rs.local, "start_local_redirector", pending)
        proxyserver = Proxyserver()
        server = _local(proxyserver)
        startup = asyncio.create_task(server.start())
        await entered.wait()
        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup
        assert LocalRedirectorInstance._instance is server
        assert LocalRedirectorInstance._server is None
        await stop_capture_servers(proxyserver)
        assert LocalRedirectorInstance._instance is None
        assert LocalRedirectorInstance._server is None

    asyncio.run(scenario())


def test_active_stop_disables_and_closes_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown avoids the native empty-action crash and confirms backend closure."""
    native = FakeNative()
    _fake_start(monkeypatch, native)

    async def scenario() -> None:
        """Drive the actual mitmproxy local instance through start and complete shutdown."""
        proxyserver = Proxyserver()
        server = _local(proxyserver)
        await server.start()
        await stop_capture_servers(proxyserver)
        assert LocalRedirectorInstance._instance is None
        assert LocalRedirectorInstance._server is None
        await stop_capture_servers(proxyserver)

    asyncio.run(scenario())
    assert native.events == [f"intercept:!{os.getpid()}", "intercept:0,!0", "close", "closed"]


def test_disabled_selector_is_valid_for_the_installed_native_parser() -> None:
    """The native parser preserves the explicit include-then-exclude PID pair."""
    assert (
        mitmproxy_rs.local.LocalRedirector.describe_spec("0,!0") == "Include PID 0. Exclude PID 0."
    )


def test_failed_disable_still_closes_native_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken interception-control command cannot bypass native handle cleanup."""
    native = FakeNative(fail_clear=True)
    _fake_start(monkeypatch, native)

    async def scenario() -> None:
        """Surface the stop failure only after closing and releasing the native singleton."""
        proxyserver = Proxyserver()
        server = _local(proxyserver)
        await server.start()
        with pytest.raises(RuntimeError, match="could not confirm network interception shutdown"):
            await stop_capture_servers(proxyserver)
        assert LocalRedirectorInstance._instance is None
        assert LocalRedirectorInstance._server is None

    asyncio.run(scenario())
    assert native.events[-3:] == ["intercept:0,!0", "close", "closed"]


def test_different_owner_is_not_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale manager must not disable the native handle owned by another capture."""
    native = FakeNative()
    _fake_start(monkeypatch, native)

    async def scenario() -> None:
        """Keep the active owner's singleton and specification unchanged."""
        stale_manager = Proxyserver()
        _local(stale_manager)
        active_manager = Proxyserver()
        active = _local(active_manager)
        await active.start()
        await stop_capture_servers(stale_manager)
        assert LocalRedirectorInstance._instance is active
        assert native.events == [f"intercept:!{os.getpid()}"]
        await stop_capture_servers(active_manager)

    asyncio.run(scenario())


def test_regular_server_is_stopped_only_while_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test transport follows its ordinary lifecycle without local-mode state access."""
    stopped: list[RegularInstance] = []

    async def stop(server: RegularInstance) -> None:
        """Observe regular-server shutdown without opening a socket."""
        stopped.append(server)

    async def scenario() -> None:
        """Invoke stop only for the regular instance that reports an active listener."""
        manager = Proxyserver()
        server = RegularInstance.make("regular", manager)
        manager.servers._instances[server.mode] = server
        monkeypatch.setattr(RegularInstance, "is_running", property(lambda self: False))
        monkeypatch.setattr(RegularInstance, "stop", stop)
        await stop_capture_servers(manager)
        assert stopped == []
        monkeypatch.setattr(RegularInstance, "is_running", property(lambda self: True))
        await stop_capture_servers(manager)
        assert stopped == [server]

    asyncio.run(scenario())


def test_new_owner_during_old_native_close_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Awaiting old native shutdown must not clear a later capture's singleton."""

    async def scenario() -> None:
        """Start a distinct capture while the old native close acknowledgement is pending."""
        closing = asyncio.Event()
        release = asyncio.Event()

        class WaitingNative(FakeNative):
            """Delay only closure acknowledgement, never operating-system cleanup."""

            async def wait_closed(self) -> None:
                """Allow another owner to start before this shutdown finishes."""
                closing.set()
                await release.wait()
                await super().wait_closed()

        previous = WaitingNative()
        _fake_start(monkeypatch, previous)
        previous_manager = Proxyserver()
        await _local(previous_manager).start()
        cleanup = asyncio.create_task(stop_capture_servers(previous_manager))
        await closing.wait()
        current = FakeNative()
        _fake_start(monkeypatch, current)
        current_manager = Proxyserver()
        current_server = _local(current_manager)
        await current_server.start()
        release.set()
        await cleanup
        assert LocalRedirectorInstance._instance is current_server
        assert current.events == [f"intercept:!{os.getpid()}"]
        await stop_capture_servers(current_manager)

    asyncio.run(scenario())


def test_one_failure_does_not_skip_other_owned_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    """All owned shutdown operations are attempted before reporting a local-mode failure."""
    native = FakeNative(fail_clear=True)
    _fake_start(monkeypatch, native)
    stopped: list[RegularInstance] = []

    async def stop(server: RegularInstance) -> None:
        """Record regular transport cleanup after the local transport failed."""
        stopped.append(server)

    async def scenario() -> None:
        """Put the failing local server first to exercise continued cleanup."""
        manager = Proxyserver()
        await _local(manager).start()
        regular = RegularInstance.make("regular", manager)
        manager.servers._instances[regular.mode] = regular
        monkeypatch.setattr(RegularInstance, "is_running", property(lambda self: True))
        monkeypatch.setattr(RegularInstance, "stop", stop)
        with pytest.raises(RuntimeError, match="could not confirm network interception shutdown"):
            await stop_capture_servers(manager)
        assert stopped == [regular]

    asyncio.run(scenario())


class _FinalWrite(layer.Layer):
    """Model a TLS alert or UDP reply queued while its connection is being closed."""

    def _handle_event(self, event: events.Event) -> layer.CommandGenerator[None]:
        """Exercise mitmproxy's real SendData dispatch with synthetic content only."""
        yield commands.SendData(self.context.client, b"synthetic shutdown reply")


def _connection(
    manager: Proxyserver,
    server: LocalRedirectorInstance,
    native: FakeNative,
    protocol: Literal["tcp", "udp"] = "tcp",
) -> tuple[ProxyConnectionHandler, Mock]:
    """Register a real mitmproxy handler backed by an inert native-like stream."""
    writer = Mock(spec=mitmproxy_rs.Stream)
    writer.get_extra_info.side_effect = lambda name, default=None: {
        "transport_protocol": protocol,
        "peername": ("127.0.0.1", 43123),
        "sockname": ("127.0.0.1", 443),
    }.get(name, default)
    closed = False

    def close() -> None:
        """Mark the native-like writer closed before its backend can disappear."""
        nonlocal closed
        closed = True
        native.events.append("writer-close")

    def write(data: bytes) -> None:
        """Expose exactly the native error if a late reply reaches a closed backend."""
        if "close" in native.events:
            raise OSError("Server has been shut down.")
        native.events.append("writer-write")

    writer.close.side_effect = close
    writer.write.side_effect = write
    writer.is_closing.side_effect = lambda: closed
    handler = ProxyConnectionHandler(Mock(), writer, writer, options.Options(), server.mode)
    handler.layer = _FinalWrite(context.Context(handler.client, options.Options()))
    manager.connections[handler.client.id] = handler
    return handler, writer


@pytest.mark.parametrize("protocol", ["tcp", "udp"])
@pytest.mark.parametrize("fail_clear", [False, True])
def test_shutdown_quiesces_late_writes_before_closing_native(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    protocol: Literal["tcp", "udp"],
    fail_clear: bool,
) -> None:
    """Real SendData dispatch cannot hit a closed native channel during TCP or UDP teardown."""
    native = FakeNative(fail_clear=fail_clear)
    _fake_start(monkeypatch, native)
    caplog.set_level(logging.ERROR)

    async def scenario() -> None:
        """Queue final replies while a simulated active native connection is cancelled."""
        manager = Proxyserver()
        server = _local(manager)
        await server.start()
        handler, writer = _connection(manager, server, native, protocol)
        entered = asyncio.Event()

        async def connection() -> None:
            """Dispatch several shutdown replies through the actual mitmproxy event handler."""
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                for _ in range(3):
                    await handler.server_event(events.ConnectionClosed(handler.client))
                native.events.append("connection-drained")

        task = asyncio.create_task(connection())
        handler.transports[handler.client].handler = task
        await entered.wait()
        if fail_clear:
            with pytest.raises(RuntimeError, match="could not confirm network interception"):
                await stop_capture_servers(manager)
        else:
            await stop_capture_servers(manager)
        assert task.done()
        assert writer.is_closing()
        writer.write.assert_not_called()

    asyncio.run(scenario())
    assert native.events.index("intercept:0,!0") < native.events.index("writer-close")
    assert native.events.index("writer-close") < native.events.index("connection-drained")
    assert native.events.index("connection-drained") < native.events.index("close")
    assert "mitmproxy has crashed" not in caplog.text
    assert "Server has been shut down" not in caplog.text


def test_slow_connection_drain_is_bounded_and_still_closes_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uncooperative task cannot postpone native cleanup beyond its finite drain budget."""
    monkeypatch.setattr(redirector, "_CONNECTION_SHUTDOWN_TIMEOUT", 0.02)
    native = FakeNative()
    _fake_start(monkeypatch, native)

    async def scenario() -> None:
        """Ignore one cancellation without retaining the network redirector handle."""
        entered = asyncio.Event()
        release = asyncio.Event()
        manager = Proxyserver()
        server = _local(manager)
        await server.start()
        handler, writer = _connection(manager, server, native)

        async def stubborn() -> None:
            """Delay one connection's cleanup until the bounded native stop has completed."""
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()

        task = asyncio.create_task(stubborn())
        handler.transports[handler.client].handler = task
        await entered.wait()
        try:
            with pytest.raises(RuntimeError, match="could not confirm network interception"):
                await asyncio.wait_for(stop_capture_servers(manager), timeout=0.5)
            assert native.events[-2:] == ["close", "closed"]
            assert writer.is_closing()
            assert LocalRedirectorInstance._server is None
        finally:
            release.set()
            await task

    asyncio.run(scenario())


def test_owner_replaced_during_connection_drain_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An awaited connection drain cannot close a native handle reused by a new owner."""
    native = FakeNative()
    _fake_start(monkeypatch, native)

    async def scenario() -> None:
        """Start a replacement after interception is cleared but before old tasks finish."""
        entered = asyncio.Event()
        replacement_ready = asyncio.Event()
        previous_manager = Proxyserver()
        previous = _local(previous_manager)
        await previous.start()
        handler, writer = _connection(previous_manager, previous, native)
        current_manager = Proxyserver()
        current = _local(current_manager)

        async def connection() -> None:
            """Use cancellation cleanup to simulate a separately owned replacement instance."""
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await current.start()
                replacement_ready.set()

        task = asyncio.create_task(connection())
        handler.transports[handler.client].handler = task
        await entered.wait()
        await stop_capture_servers(previous_manager)
        assert replacement_ready.is_set()
        assert writer.is_closing()
        assert LocalRedirectorInstance._instance is current
        assert "close" not in native.events
        await stop_capture_servers(current_manager)
        assert native.events[-2:] == ["close", "closed"]

    asyncio.run(scenario())


def test_real_native_udp_stream_closes_before_backend_release(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Real native stream state prevents late sends while an isolated loopback backend closes."""
    caplog.set_level(logging.ERROR)

    async def scenario() -> None:
        """Use native UDP transport without installing or activating any macOS extension."""
        received: asyncio.Future[mitmproxy_rs.Stream] = asyncio.get_running_loop().create_future()
        release_callback = asyncio.Event()

        async def receive(stream: mitmproxy_rs.Stream) -> None:
            """Retain one synthetic UDP stream until its owner performs shutdown."""
            received.set_result(stream)
            await release_callback.wait()

        backend = await mitmproxy_rs.udp.start_udp_server("127.0.0.1", 0, receive)

        class NativeUdp(FakeNative):
            """Reuse real native close acknowledgement with an inert interception selector."""

            def close(self) -> None:
                """Release the real UDP server only after its writer became closed."""
                assert received.result().is_closing()
                super().close()
                backend.close()

            async def wait_closed(self) -> None:
                """Wait for native transport closure without any operating-system redirector."""
                await asyncio.wait_for(backend.wait_closed(), timeout=1.0)
                await super().wait_closed()

        native = NativeUdp()
        _fake_start(monkeypatch, native)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(b"synthetic UDP request", backend.getsockname())
            stream = await asyncio.wait_for(received, timeout=1.0)
            manager = Proxyserver()
            server = _local(manager)
            await server.start()
            handler = ProxyConnectionHandler(Mock(), stream, stream, options.Options(), server.mode)
            handler.layer = _FinalWrite(context.Context(handler.client, options.Options()))
            manager.connections[handler.client.id] = handler
            entered = asyncio.Event()

            async def connection() -> None:
                """Dispatch a final datagram after cancellation through the real native writer."""
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    await handler.server_event(events.ConnectionClosed(handler.client))
                    native.events.append("connection-drained")

            task = asyncio.create_task(connection())
            handler.transports[handler.client].handler = task
            await entered.wait()
            await asyncio.wait_for(stop_capture_servers(manager), timeout=2.0)
            assert task.done()
            assert stream.is_closing()
            assert native.events.index("connection-drained") < native.events.index("close")
        finally:
            release_callback.set()
            sender.close()
            backend.close()
            await asyncio.wait_for(backend.wait_closed(), timeout=1.0)

    asyncio.run(scenario())
    assert "mitmproxy has crashed" not in caplog.text
    assert "Server has been shut down" not in caplog.text


def test_backend_notification_requires_initialized_local_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing, partial, regular, and foreign-owned backends never report a false shutdown."""
    native = FakeNative()
    _fake_start(monkeypatch, native)

    async def scenario() -> None:
        """Inspect only ownership metadata without constructing spurious lifetime waiters."""
        empty = Proxyserver()
        assert capture_server_closed(empty) is None
        regular_manager = Proxyserver()
        regular = RegularInstance.make("regular", regular_manager)
        regular_manager.servers._instances[regular.mode] = regular
        assert capture_server_closed(regular_manager) is None
        previous_manager = Proxyserver()
        previous = _local(previous_manager)
        LocalRedirectorInstance._instance = previous
        assert capture_server_closed(previous_manager) is None
        LocalRedirectorInstance._instance = None
        current_manager = Proxyserver()
        await _local(current_manager).start()
        assert capture_server_closed(previous_manager) is None
        assert native.events == [f"intercept:!{os.getpid()}"]
        await stop_capture_servers(current_manager)

    asyncio.run(scenario())


def test_real_native_backend_waiters_are_independent_and_cancellable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled real native watcher neither closes the backend nor steals cleanup's signal."""

    async def scenario() -> None:
        """Observe a loopback UDP backend through the local-owner seam without OS interception."""

        async def unused(stream: mitmproxy_rs.Stream) -> None:
            """Close unexpected loopback traffic; this fixture never sends a request."""
            stream.close()

        backend = await mitmproxy_rs.udp.start_udp_server("127.0.0.1", 0, unused)

        class NativeUdp(FakeNative):
            """Use the real native lifetime with an inert interception configuration."""

            def close(self) -> None:
                """Request native shutdown while concurrent observers remain attached."""
                super().close()
                backend.close()

            async def wait_closed(self) -> None:
                """Give every call its own real native shutdown receiver."""
                await backend.wait_closed()
                await super().wait_closed()

        native = NativeUdp()
        _fake_start(monkeypatch, native)
        manager = Proxyserver()
        await _local(manager).start()
        first = capture_server_closed(manager)
        second = capture_server_closed(manager)
        assert first is not None and second is not None
        cancelled = asyncio.ensure_future(first)
        observer = asyncio.ensure_future(second)
        try:
            await asyncio.sleep(0.01)
            assert not cancelled.done() and not observer.done()
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            assert not observer.done()
            assert "close" not in native.events
            await asyncio.wait_for(stop_capture_servers(manager), timeout=1.0)
            await asyncio.wait_for(observer, timeout=1.0)
            assert native.events.count("closed") == 2
            assert capture_server_closed(manager) is None
        finally:
            cancelled.cancel()
            observer.cancel()
            await asyncio.gather(cancelled, observer, return_exceptions=True)
            backend.close()
            await asyncio.wait_for(backend.wait_closed(), timeout=1.0)

    asyncio.run(scenario())
