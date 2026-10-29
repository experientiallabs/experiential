"""Synthetic TLS providers prove byte-preserving streaming and upstream verification."""

import asyncio
import json
import socket
import ssl
import zlib
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import httpx
import mitmproxy_rs
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from mitmproxy import certs, connection, http, options, tcp, tls, websocket
from mitmproxy.addons.proxyserver import Proxyserver
from mitmproxy.proxy import commands, context, events, layer, layers, server_hooks
from mitmproxy.proxy.mode_servers import ProxyConnectionHandler
from mitmproxy.proxy.mode_specs import ProxyMode
from mitmproxy.proxy.server import ConnectionIO
from mitmproxy.tools.dump import DumpMaster
from wsproto.frame_protocol import Opcode

from exp.common.core.artifacts import SourceIdentity
from exp.common.traces.ingest.otlp import normalize_otlp_payload
from exp.runtime.capture import proxy as capture_module
from exp.runtime.capture.certificates import prepare_certificate
from exp.runtime.capture.normalization import CapturedExchange
from exp.runtime.capture.proxy import CaptureProxy, _Body, _CaptureTlsConfig
from exp.runtime.capture.upload import CaptureUploader


@pytest.fixture
def regular_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a loopback CONNECT listener while retaining production host and TLS policy."""
    original = capture_module._capture_options

    def regular_options(domains: tuple[str, ...], directory: Path) -> options.Options:
        """Substitute only the transport mode; never activate the system redirector."""
        configured = original(domains, directory)
        configured.update(mode=["regular@127.0.0.1:0"])
        return configured

    monkeypatch.setattr(capture_module, "_capture_options", regular_options)


def test_native_capture_starts_inactive_until_watchdog_owns_control(tmp_path: Path) -> None:
    """No application can be intercepted before independent cleanup is armed."""
    configured = capture_module._capture_options(("api.openai.com",), tmp_path)
    assert len(configured.mode) == 1
    native_mode = ProxyMode.parse(configured.mode[0])
    assert native_mode.type_name == "local"
    assert native_mode.data == "0,!0"
    assert configured.allow_hosts == [r"^api\.openai\.com\.?:[0-9]+$"]
    assert configured.ssl_insecure is False


def test_capture_stop_monitor_without_native_owner_waits_for_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary test listener remains active until its requested shutdown."""
    monkeypatch.setattr(capture_module, "capture_server_closed", lambda manager: None)

    async def run() -> None:
        """Wait through one event-loop turn before requesting the non-native listener stop."""
        baseline = asyncio.all_tasks()
        stop = asyncio.Event()
        task = asyncio.create_task(capture_module._wait_for_capture_stop(Proxyserver(), stop))
        await asyncio.sleep(0)
        assert not task.done()
        stop.set()
        await asyncio.wait_for(task, 1)
        assert asyncio.all_tasks() == baseline

    asyncio.run(run())


@pytest.mark.parametrize("ending", ["stop", "cancel", "both"])
def test_capture_stop_monitor_reaps_native_and_stop_waiters(
    monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    """Stop, cancellation, and simultaneous closure finish without leaking either waiter."""

    async def run() -> None:
        """Use a native-style Future so cancellation and simultaneous completion are observable."""
        baseline = asyncio.all_tasks()
        closed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        monkeypatch.setattr(capture_module, "capture_server_closed", lambda manager: closed)
        stop = asyncio.Event()
        task = asyncio.create_task(capture_module._wait_for_capture_stop(Proxyserver(), stop))
        await asyncio.sleep(0)
        assert not task.done()
        if ending == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            if ending == "both":
                closed.set_result(None)
            stop.set()
            await asyncio.wait_for(task, 1)
        assert closed.done()
        assert closed.cancelled() is (ending != "both")
        assert asyncio.all_tasks() == baseline

    asyncio.run(run())


@pytest.mark.parametrize("backend_error", [False, True])
def test_capture_stop_monitor_reports_unexpected_native_end_without_raw_error(
    monkeypatch: pytest.MonkeyPatch, backend_error: bool
) -> None:
    """Native completion or failure produces one fixed message and cleans the remaining waiter."""

    async def run() -> None:
        """End an observed backend independently of Capture's stop request."""
        baseline = asyncio.all_tasks()
        closed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        monkeypatch.setattr(capture_module, "capture_server_closed", lambda manager: closed)
        stop = asyncio.Event()
        task = asyncio.create_task(capture_module._wait_for_capture_stop(Proxyserver(), stop))
        await asyncio.sleep(0)
        if backend_error:
            closed.set_exception(OSError("synthetic private backend detail"))
        else:
            closed.set_result(None)
        with pytest.raises(RuntimeError) as failure:
            await asyncio.wait_for(task, 1)
        assert str(failure.value) == (
            "Capture network backend stopped unexpectedly. Run exp capture again."
        )
        assert not stop.is_set()
        assert asyncio.all_tasks() == baseline

    asyncio.run(run())


@pytest.mark.parametrize("backend_dies", [False, True])
def test_real_proxy_stops_cleanly_after_backend_lifetime_ends(
    tmp_path: Path,
    regular_proxy: None,
    monkeypatch: pytest.MonkeyPatch,
    backend_dies: bool,
) -> None:
    """Drive actual proxy startup and cleanup with an independently controlled backend monitor."""

    async def run() -> None:
        """Close an isolated loopback listener after backend death or a normal Capture stop."""
        closed = asyncio.Event()
        monitor_finished = asyncio.Event()

        async def backend_closed() -> None:
            """Expose a fake native lifetime and prove its monitor is reaped during cleanup."""
            try:
                await closed.wait()
            finally:
                monitor_finished.set()

        monkeypatch.setattr(
            capture_module, "capture_server_closed", lambda manager: backend_closed()
        )
        proxy = CaptureProxy(sink=lambda exchange: True, domains=("api.openai.com",))
        ready = asyncio.Event()
        task = asyncio.create_task(proxy.serve(ca_directory=tmp_path / "proxy", ready=ready.set))
        try:
            await asyncio.wait_for(ready.wait(), 5)
            assert proxy._master is not None
            manager = proxy._master.addons.get("proxyserver")
            assert isinstance(manager, Proxyserver)
            port = next(iter(manager.servers)).listen_addrs[0][1]
            if backend_dies:
                closed.set()
                with pytest.raises(RuntimeError) as failure:
                    await asyncio.wait_for(task, 5)
                assert str(failure.value) == (
                    "Capture network backend stopped unexpectedly. Run exp capture again."
                )
            else:
                proxy.shutdown()
                await asyncio.wait_for(task, 5)
            assert monitor_finished.is_set()
            assert proxy._master is None
            assert not proxy._host_bypasses and not proxy._app_bypasses
            assert all(not server.is_running for server in manager.servers)
            with pytest.raises(OSError):
                await asyncio.open_connection("127.0.0.1", port)
        finally:
            if not task.done():
                proxy.shutdown()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


async def _connect_tls(
    proxy: CaptureProxy,
    upstream_port: int,
    hostname: str,
    trusted_ca: Path,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, bytes]:
    """Tunnel to the original loopback destination, then present the selected TLS SNI."""
    assert proxy._master is not None
    proxyserver = proxy._master.addons.get("proxyserver")
    assert isinstance(proxyserver, Proxyserver)
    proxy_port = next(iter(proxyserver.servers)).listen_addrs[0][1]
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    writer.write(
        f"CONNECT 127.0.0.1:{upstream_port} HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{upstream_port}\r\n\r\n".encode()
    )
    await writer.drain()
    header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
    assert header.startswith(b"HTTP/1.1 200"), header
    client_context = ssl.create_default_context(cafile=str(trusted_ca))
    await asyncio.wait_for(writer.start_tls(client_context, server_hostname=hostname), 5)
    tls = writer.get_extra_info("ssl_object")
    assert isinstance(tls, ssl.SSLObject)
    peer_certificate = tls.getpeercert(binary_form=True)
    assert isinstance(peer_certificate, bytes)
    return reader, writer, peer_certificate


def _certificate(directory: Path, host: str) -> tuple[Path, Path, Path]:
    """Create an isolated test CA and one upstream TLS certificate."""
    directory.mkdir()
    store = certs.CertStore.from_store(directory, "upstream", 2048)
    entry = store.get_cert(host, [x509.DNSName(host)])
    certificate = directory / "server.pem"
    key = directory / "server.key"
    certificate.write_bytes(entry.cert.to_pem())
    key.write_bytes(
        entry.privatekey.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return certificate, key, directory / "upstream-ca-cert.pem"


@pytest.mark.parametrize(
    "valid_hostname,truncated,encoding",
    [
        (True, False, "identity"),
        (True, True, "identity"),
        (True, False, "gzip"),
        (True, True, "gzip"),
        (False, False, "identity"),
    ],
)
def test_real_tls_sse_passes_unchanged_and_rejects_wrong_upstream_hostname(
    tmp_path: Path, valid_hostname: bool, truncated: bool, encoding: str, regular_proxy: None
) -> None:
    """Drive actual TLS streaming and verify capture, delivery, and hostname checks."""

    async def run() -> None:
        """Run the asynchronous synthetic networking scenario to completion."""
        host = "api.openai.com"
        certificate, key, upstream_ca = _certificate(
            tmp_path / "upstream", host if valid_hostname else "wrong.example"
        )
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(certificate, key)
        sni_seen: list[str | None] = []

        def sni_callback(
            socket: ssl.SSLSocket | ssl.SSLObject,
            name: str | None,
            context: ssl.SSLContext | ssl.SSLSocket,
        ) -> None:
            """Record the hostname actually presented to the synthetic upstream."""
            sni_seen.append(name)

        server_context.set_servername_callback(sni_callback)
        received: list[bytes] = []
        first = b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
        last = (
            b'data: {"type":"response.completed","response":{"model":"test","output":[], '
            b'"usage":{"input_tokens":4,"output_tokens":2}}}\n\n'
        )
        if encoding == "gzip":
            compressor = zlib.compressobj(wbits=31)
            first = compressor.compress(first) + compressor.flush(zlib.Z_SYNC_FLUSH)
            last = compressor.compress(last) + compressor.flush(
                zlib.Z_SYNC_FLUSH if truncated else zlib.Z_FINISH
            )
        release = asyncio.Event()

        async def upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Serve a delayed synthetic SSE response and record the original request."""
            try:
                header = await reader.readuntil(b"\r\n\r\n")
                length = next(
                    int(line.split(b":", 1)[1])
                    for line in header.splitlines()
                    if line.lower().startswith(b"content-length:")
                )
                received.append(header + await reader.readexactly(length))
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    + (b"Content-Encoding: gzip\r\n" if encoding == "gzip" else b"")
                    + b"Content-Length: "
                    + str(len(first + last) + (100 if truncated else 0)).encode()
                    + b"\r\nConnection: close\r\n\r\n"
                    + first
                )
                await writer.drain()
                await release.wait()
                writer.write(last)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(upstream, "127.0.0.1", 0, ssl=server_context)
        upstream_port = server.sockets[0].getsockname()[1]
        captured: list[CapturedExchange] = []
        projected: list[tuple[int, int]] = []
        diagnostics: list[str] = []
        run_id, ingest_id = str(uuid4()), str(uuid4())
        upload_prefix = (
            "/storage/v1/object/upload/sign/artifacts/orgs/organization/telemetry-traces/otlp/"
        )

        def platform(request: httpx.Request) -> httpx.Response:
            """Emulate signed upload and finalize while validating projected trace evidence."""
            if request.url.host == "storage.example":
                assert "authorization" not in request.headers
                assert b"Bearer secret" not in request.content
                payload = json.loads(request.content)
                recorded_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
                assert recorded_span["status"] == {"code": 1}
                normalized = normalize_otlp_payload(
                    payload,
                    source=SourceIdentity(kind="otlp", source_id="capture-integration"),
                )
                assert not normalized.issues
                span = normalized.traces[0].spans[0]
                assert span.usage is not None
                projected.append((span.usage.input_tokens, span.usage.output_tokens))
                return httpx.Response(200)
            assert request.headers["authorization"] == "Bearer PLATFORM-KEY"
            if request.url.path.endswith("/batches/upload"):
                assert run_id in request.url.path
                return httpx.Response(
                    200,
                    json={
                        "status": "pending",
                        "ingest_id": ingest_id,
                        "signed_url": (
                            f"https://storage.example{upload_prefix}{ingest_id}/"
                            f"{'a' * 43}?token=signed"
                        ),
                    },
                )
            assert request.url.path.endswith(f"/{ingest_id}/finalize")
            return httpx.Response(202)

        uploader = CaptureUploader(
            "https://platform.example",
            "organization",
            run_id,
            "PLATFORM-KEY",
            tmp_path / "spool" / run_id,
            upload_origin="https://storage.example",
            upload_path_prefix=upload_prefix,
            transport=httpx.MockTransport(platform),
            on_diagnostic=diagnostics.append,
        )
        if valid_hostname:
            uploader.start()

        def sink(exchange: CapturedExchange) -> bool:
            """Keep a test copy and enqueue the same observed exchange for delivery."""
            captured.append(exchange)
            return uploader.submit(exchange) if valid_hostname else True

        proxy = CaptureProxy(
            sink=sink,
            domains=(host,),
            upstream_ca_file=upstream_ca,
        )
        ready = asyncio.Event()
        prepare_certificate(tmp_path / "proxy", (host,))
        proxy_task = asyncio.create_task(
            proxy.serve(ca_directory=tmp_path / "proxy", ready=ready.set)
        )
        try:
            await asyncio.wait_for(ready.wait(), 5)
            reader, writer, peer_certificate = await _connect_tls(
                proxy, upstream_port, host, tmp_path / "proxy/mitmproxy-ca-cert.pem"
            )
            assert peer_certificate != x509.load_pem_x509_certificate(
                certificate.read_bytes()
            ).public_bytes(serialization.Encoding.DER)
            peer_names = (
                x509.load_der_x509_certificate(peer_certificate)
                .extensions.get_extension_for_class(x509.SubjectAlternativeName)
                .value
            )
            assert list(peer_names) == [x509.DNSName(host)]
            request = b'{"model":"test","input":"hello","stream":true}'
            writer.write(
                b"POST /v1/responses?beta=true HTTP/1.1\r\nHost: api.openai.com\r\n"
                b"Authorization: Bearer secret\r\nContent-Type: application/json\r\n"
                b"Content-Length: " + str(len(request)).encode() + b"\r\n\r\n" + request
            )
            await writer.drain()
            header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            if valid_hostname:
                assert header.startswith(b"HTTP/1.1 200")
                assert await asyncio.wait_for(reader.readexactly(len(first)), 2) == first
                assert not release.is_set()
                release.set()
                assert await asyncio.wait_for(reader.readexactly(len(last)), 2) == last
                assert received[0].endswith(request)
                assert b"Host: api.openai.com\r\n" in received[0]
                assert received[0].startswith(b"POST /v1/responses?beta=true ")
                deadline = asyncio.get_running_loop().time() + 3
                while not captured:
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.01)
                assert len(captured) == 1
                assert captured[0].request == request
                assert captured[0].response == first + last
                assert captured[0].host == host
                assert captured[0].failed is truncated
                deadline = asyncio.get_running_loop().time() + 3
                while uploader.stats.uploaded_batches != 1 or not any(
                    "upload_accepted" in event for event in diagnostics
                ):
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.01)
                assert projected == [(4, 2)]
                assert any(
                    "capture_saved" in event
                    and f"trace {captured[0].trace_id}" in event
                    and "completed=True" in event
                    and "interrupted=False" in event
                    and f"transport_error={truncated}" in event
                    and "4 in / 2 out tokens" in event
                    for event in diagnostics
                )
                assert any("upload_accepted" in event for event in diagnostics)
            else:
                assert header.startswith(b"HTTP/1.1 502")
                assert not received
            assert sni_seen == [host]
            writer.close()
            with suppress(ConnectionResetError):
                await writer.wait_closed()
        finally:
            release.set()
            proxy.shutdown()
            await asyncio.wait_for(proxy_task, 5)
            server.close()
            await server.wait_closed()
            uploader.close()

    asyncio.run(run())


@pytest.mark.parametrize("destination", ["192.0.2.15", "2001:db8::15"])
def test_local_mode_certificate_uses_only_selected_sni(tmp_path: Path, destination: str) -> None:
    """Constrained certificates exclude original destination IPs without rewriting routing."""
    host = "api.openai.com"
    directory = tmp_path / "proxy"
    prepare_certificate(directory, (host,))
    config = _CaptureTlsConfig(frozenset({host}))
    config.certstore = certs.CertStore.from_store(directory, "mitmproxy", 2048)
    client = connection.Client(
        peername=("127.0.0.1", 12345),
        sockname=(destination, 443),
        sni="API.OpenAI.com.",
        proxy_mode=ProxyMode.parse("local"),
    )
    ctx = context.Context(client, options.Options(mode=["local"]))
    ctx.server = connection.Server(address=(destination, 443), sni=host)
    entry = config.get_cert(ctx)
    assert entry.cert.cn == host
    assert list(entry.cert.altnames) == [x509.DNSName(host)]
    assert ctx.server.address == (destination, 443)
    assert ctx.server.sni == host
    assert client.sni == "API.OpenAI.com."


@pytest.mark.parametrize("sni", [None, "unselected.example", "sub.api.openai.com", "127.0.0.1"])
def test_certificate_issuance_refuses_unselected_sni(sni: str | None) -> None:
    """A selected destination cannot cause issuance for missing, nested, or unrelated SNI."""
    config = _CaptureTlsConfig(frozenset({"api.openai.com"}))
    client = connection.Client(peername=("127.0.0.1", 12345), sockname=("192.0.2.15", 443), sni=sni)
    ctx = context.Context(client, options.Options(mode=["local"]))
    ctx.server = connection.Server(address=("api.openai.com", 443))
    with pytest.raises(RuntimeError, match="requires a selected DNS server name"):
        config.get_cert(ctx)


def test_stream_copy_limit_never_changes_forwarded_chunks() -> None:
    """Discard oversized copies while forwarding every original byte."""
    body = _Body(3)
    assert body.tee(b"ab") == b"ab"
    assert body.tee(b"cd") == b"cd"
    assert body.tee(b"ef") == b"ef"
    assert body.overflow and not body.data


@pytest.mark.parametrize("identified", [False, True])
def test_real_client_ca_rejection_retries_with_original_tls_without_stopping_capture(
    tmp_path: Path, regular_proxy: None, monkeypatch: pytest.MonkeyPatch, identified: bool
) -> None:
    """Rejected clients retry untouched while another identified app remains captured."""

    async def run() -> None:
        """Exercise both TLS trust stores through ephemeral loopback listeners only."""
        hostname = "api.openai.com"
        certificate, key, upstream_ca = _certificate(tmp_path / "upstream", hostname)
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(certificate, key)
        received: list[bytes] = []
        captured: list[CapturedExchange] = []
        warnings: list[tuple[str, str, capture_module.CaptureBypassReason]] = []
        bypassed = asyncio.Event()
        identity = (101, "/Applications/Synthetic.app/Contents/MacOS/client")

        async def upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Serve a fixed provider-shaped response without making external requests."""
            try:
                received.append(await reader.readuntil(b"\r\n\r\n"))
                body = b'{"id":"r1","model":"test","output":[]}'
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
            finally:
                writer.close()
                with suppress(ConnectionError):
                    await writer.wait_closed()

        def warn(app: str, host: str, reason: capture_module.CaptureBypassReason) -> None:
            """Observe degraded coverage only after the actual rejection hook has run."""
            warnings.append((app, host, reason))
            bypassed.set()

        server = await asyncio.start_server(upstream, "127.0.0.1", 0, ssl=server_context)
        port = server.sockets[0].getsockname()[1]
        proxy = CaptureProxy(
            sink=lambda exchange: captured.append(exchange) is None,
            domains=(hostname,),
            upstream_ca_file=upstream_ca,
            on_bypass=warn,
        )
        if identified:
            monkeypatch.setattr(proxy, "_client_identity", lambda client: identity)
        ready = asyncio.Event()
        proxy_task = asyncio.create_task(
            proxy.serve(ca_directory=tmp_path / "proxy", ready=ready.set)
        )
        try:
            await asyncio.wait_for(ready.wait(), 5)
            assert proxy._master is not None
            manager = proxy._master.addons.get("proxyserver")
            assert isinstance(manager, Proxyserver)
            proxy_port = next(iter(manager.servers)).listen_addrs[0][1]
            client_context = ssl.create_default_context(cafile=str(upstream_ca))

            def reject_certificate() -> None:
                """Send a real fatal unknown-CA alert rather than a silent socket close."""
                with socket.create_connection(("127.0.0.1", proxy_port), timeout=5) as sock:
                    sock.sendall(
                        f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{port}\r\n\r\n".encode()
                    )
                    header = b""
                    while not header.endswith(b"\r\n\r\n"):
                        chunk = sock.recv(4096)
                        assert chunk and len(header) < 8192
                        header += chunk
                    assert header.startswith(b"HTTP/1.1 200")
                    with client_context.wrap_socket(sock, server_hostname=hostname):
                        pytest.fail("the client must reject Capture's untrusted certificate")

            with pytest.raises(ssl.SSLCertVerificationError):
                await asyncio.wait_for(asyncio.to_thread(reject_certificate), 7)
            await asyncio.wait_for(bypassed.wait(), 5)
            assert warnings == [("client" if identified else "All apps", hostname, "certificate")]
            assert not proxy_task.done()
            assert not captured and not received
            original_certificate = x509.load_pem_x509_certificate(certificate.read_bytes())
            request = (
                b"POST /v1/responses HTTP/1.1\r\nHost: api.openai.com\r\n"
                b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
            )
            reader, writer, peer = await _connect_tls(proxy, port, hostname, upstream_ca)
            assert peer == original_certificate.public_bytes(serialization.Encoding.DER)
            writer.write(request)
            await writer.drain()
            assert (await asyncio.wait_for(reader.read(), 5)).endswith(b'"output":[]}')
            writer.close()
            await writer.wait_closed()
            assert len(received) == 1 and not captured
            assert not proxy_task.done()

            if identified:
                identity = (202, "/Applications/Healthy.app/Contents/MacOS/healthy")
                reader, writer, peer = await _connect_tls(
                    proxy, port, hostname, tmp_path / "proxy/mitmproxy-ca-cert.pem"
                )
                assert peer != original_certificate.public_bytes(serialization.Encoding.DER)
                writer.write(request)
                await writer.drain()
                assert (await asyncio.wait_for(reader.read(), 5)).endswith(b'"output":[]}')
                writer.close()
                await writer.wait_closed()
                assert len(received) == 2
                assert len(captured) == 1
            assert len(warnings) == 1
        finally:
            proxy.shutdown()
            await asyncio.wait_for(proxy_task, 5)
            server.close()
            await server.wait_closed()

    asyncio.run(run())


@pytest.mark.parametrize("alert", ["unknown ca", "bad certificate", "certificate unknown"])
def test_client_certificate_alert_bypasses_once_without_exposing_error(
    monkeypatch: pytest.MonkeyPatch, alert: str
) -> None:
    """A missing identity visibly bypasses only the selected host, without a global stop."""
    warnings: list[tuple[str, str, capture_module.CaptureBypassReason]] = []
    proxy = CaptureProxy(
        sink=lambda exchange: True,
        domains=("api.openai.com",),
        on_bypass=lambda app, host, reason: warnings.append((app, host, reason)),
    )
    shutdowns: list[bool] = []
    monkeypatch.setattr(proxy, "shutdown", lambda: shutdowns.append(True))
    client = connection.Client(
        peername=("127.0.0.1", 1),
        sockname=("127.0.0.1", 2),
        sni="API.OPENAI.COM.",
        error=f"OpenSSL {alert.upper()} private-handshake-detail",
    )
    ctx = context.Context(client, options.Options())
    data = tls.TlsData(client, ctx)
    proxy.tls_failed_client(data)
    proxy.tls_failed_client(data)
    assert not shutdowns
    assert proxy._host_bypasses == {"api.openai.com"}
    assert warnings == [("All apps", "api.openai.com", "certificate")]


@pytest.mark.parametrize("error", ["unknown ca", "The client disconnected during the handshake."])
def test_client_tls_callbacks_after_requested_shutdown_cannot_change_stop_reason(
    error: str,
) -> None:
    """Late callbacks during teardown cannot turn an ordinary stop into a certificate failure."""
    proxy = CaptureProxy(sink=lambda exchange: True, domains=("api.openai.com",))
    client = connection.Client(
        peername=("127.0.0.1", 1),
        sockname=("127.0.0.1", 2),
        sni="api.openai.com",
        error=error,
    )
    ctx = context.Context(client, options.Options())
    proxy.shutdown()
    for _ in range(3):
        proxy.tls_failed_client(tls.TlsData(client, ctx))
    assert not proxy._host_bypasses and not proxy._app_bypasses


@pytest.mark.parametrize("cancel_task", [False, True])
def test_real_incomplete_handshakes_closed_during_shutdown_are_not_ca_failures(
    tmp_path: Path,
    regular_proxy: None,
    monkeypatch: pytest.MonkeyPatch,
    cancel_task: bool,
) -> None:
    """Closing three real TLS transports during requested or cancelled shutdown stays orderly."""

    async def run() -> None:
        """Pause three loopback handshakes, then close their transports at the cleanup boundary."""
        host = "api.openai.com"
        proxy = CaptureProxy(sink=lambda exchange: True, domains=(host,))
        writers: list[asyncio.StreamWriter] = []
        failures: list[str | None] = []
        original_stop = capture_module.stop_capture_servers
        original_failed = proxy.tls_failed_client

        def failed(data: tls.TlsData) -> None:
            """Record actual TLS callbacks so the regression cannot pass without disconnects."""
            failures.append(data.conn.error)
            original_failed(data)

        async def stop_and_disconnect(proxyserver: Proxyserver) -> None:
            """Model native shutdown closing its streams after the regular listener stops."""
            await original_stop(proxyserver)
            for writer in writers:
                writer.close()
            await asyncio.gather(
                *(writer.wait_closed() for writer in writers), return_exceptions=True
            )
            deadline = asyncio.get_running_loop().time() + 2
            while len(failures) < 3:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.01)

        monkeypatch.setattr(proxy, "tls_failed_client", failed)
        monkeypatch.setattr(capture_module, "stop_capture_servers", stop_and_disconnect)
        ready = asyncio.Event()
        proxy_task = asyncio.create_task(
            proxy.serve(ca_directory=tmp_path / "proxy", ready=ready.set)
        )
        try:
            await asyncio.wait_for(ready.wait(), 5)
            assert proxy._master is not None
            proxyserver = proxy._master.addons.get("proxyserver")
            assert isinstance(proxyserver, Proxyserver)
            port = next(iter(proxyserver.servers)).listen_addrs[0][1]
            for _ in range(3):
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writers.append(writer)
                writer.write(b"CONNECT 127.0.0.1:1 HTTP/1.1\r\nHost: 127.0.0.1:1\r\n\r\n")
                await writer.drain()
                header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2)
                assert header.startswith(b"HTTP/1.1 200")
                incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
                client_context = ssl.create_default_context(
                    cafile=str(tmp_path / "proxy/mitmproxy-ca-cert.pem")
                )
                client = client_context.wrap_bio(incoming, outgoing, server_hostname=host)
                with pytest.raises(ssl.SSLWantReadError):
                    client.do_handshake()
                writer.write(outgoing.read())
                await writer.drain()
                assert await asyncio.wait_for(reader.read(8192), 2)
            assert not failures
            assert not proxy._host_bypasses and not proxy._app_bypasses
            if cancel_task:
                proxy_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(proxy_task, 5)
            else:
                proxy.shutdown()
                await asyncio.wait_for(proxy_task, 5)
            assert len(failures) == 3
            assert all(
                error is not None
                and error.startswith("The client disconnected during the handshake.")
                for error in failures
            )
            assert not proxy._host_bypasses and not proxy._app_bypasses
            assert proxy._master is None
        finally:
            for writer in writers:
                writer.close()
            if not proxy_task.done():
                proxy.shutdown()
                await asyncio.gather(proxy_task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("scenario", ["unselected", "upstream", "unrelated_error"])
def test_unrelated_tls_failure_does_not_stop_capture(
    monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    """Opaque hosts, provider failures, and unrelated protocol errors retain their behavior."""
    proxy = CaptureProxy(sink=lambda exchange: True, domains=("api.openai.com",))
    shutdowns: list[bool] = []
    monkeypatch.setattr(proxy, "shutdown", lambda: shutdowns.append(True))
    client = connection.Client(
        peername=("127.0.0.1", 1),
        sockname=("127.0.0.1", 2),
        sni="unrelated.example" if scenario == "unselected" else "api.openai.com",
        error="unsupported protocol" if scenario == "unrelated_error" else "unknown ca",
    )
    ctx = context.Context(client, options.Options())
    ctx.server = connection.Server(address=("api.openai.com", 443), error="unknown ca")
    affected = ctx.server if scenario == "upstream" else client
    proxy.tls_failed_client(tls.TlsData(affected, ctx))
    assert not shutdowns
    assert not proxy._host_bypasses and not proxy._app_bypasses


@pytest.mark.parametrize("error", ["unknown ca", "The client disconnected during the handshake."])
def test_native_failure_does_not_misclassify_generic_tls_disconnect(
    monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    """Known transport losses bypass only the generic guard, never a certificate alert."""
    proxy = CaptureProxy(sink=lambda exchange: True, domains=("api.openai.com",))
    shutdowns: list[bool] = []
    monkeypatch.setattr(proxy, "shutdown", lambda: shutdowns.append(True))
    client = connection.Client(
        peername=("127.0.0.1", 1),
        sockname=("127.0.0.1", 2),
        sni="api.openai.com",
        error=error,
    )
    ctx = context.Context(client, options.Options())
    proxy._transport_failures.add(client.id)
    for _ in range(3):
        proxy.tls_failed_client(tls.TlsData(client, ctx))
    assert not shutdowns
    if error == "unknown ca":
        assert proxy._host_bypasses == {"api.openai.com"}
    else:
        assert not shutdowns
        assert not proxy._host_bypasses and not proxy._app_bypasses
    proxy.client_disconnected(client)
    assert not proxy._transport_failures


def test_native_guard_hooks_use_only_the_capture_owned_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client and upstream hooks route their exact targets and failure callback to this master."""
    proxy = CaptureProxy(sink=lambda exchange: True, domains=("api.openai.com",))
    owned = Proxyserver()
    master = Mock(spec=DumpMaster, addons=Mock())
    master.addons.get.return_value = owned
    client = connection.Client(peername=("127.0.0.1", 1), sockname=("127.0.0.1", 2))
    upstream = connection.Server(address=("api.openai.com", 443))
    data = server_hooks.ServerConnectionHookData(upstream, client)
    calls: list[tuple[Proxyserver, connection.Client, connection.Connection]] = []
    callbacks: list[Callable[[str], None]] = []

    def guard(
        manager: Proxyserver,
        owner: connection.Client,
        target: connection.Connection,
        on_failure: Callable[[str], None],
    ) -> None:
        """Observe only the existing transport integration boundary without native I/O."""
        calls.append((manager, owner, target))
        callbacks.append(on_failure)

    monkeypatch.setattr(capture_module, "guard_native_writer", guard)
    proxy.client_connected(client)
    proxy.server_connected(data)
    assert not calls

    proxy._master = master
    owned.connections[client.id] = Mock(spec=ProxyConnectionHandler, client=client)
    proxy.client_connected(client)
    proxy.server_connected(data)
    assert calls == [(owned, client, client), (owned, client, upstream)]
    for callback in callbacks:
        callback(client.id)
    assert proxy._transport_failures == {client.id}
    proxy.client_disconnected(client)
    assert not proxy._transport_failures


@pytest.mark.parametrize("upstream", [False, True])
def test_native_guard_hook_attributes_real_adapter_failures_to_the_owning_client(
    upstream: bool,
) -> None:
    """Hook-installed adapters report their failed connection without stopping its siblings."""

    async def run() -> None:
        """Exercise the actual writer adapter with one inert native stream and owned registry."""
        proxy = CaptureProxy(sink=lambda exchange: True, domains=("api.openai.com",))
        manager = Proxyserver()
        master = Mock(spec=DumpMaster, addons=Mock())
        master.addons.get.return_value = manager
        proxy._master = master
        client = connection.Client(
            peername=("127.0.0.1", 1), sockname=("127.0.0.1", 443), sni="api.openai.com"
        )
        server = connection.Server(address=("api.openai.com", 443))
        native = Mock(spec=mitmproxy_rs.Stream)
        native.write.side_effect = OSError("Server has been shut down.")
        transport = ConnectionIO(writer=native)
        target = server if upstream else client
        manager.connections[client.id] = Mock(
            spec=ProxyConnectionHandler, client=client, transports={target: transport}
        )
        if upstream:
            proxy.server_connected(server_hooks.ServerConnectionHookData(server, client))
        else:
            proxy.client_connected(client)
        assert transport.writer is not None
        assert transport.writer is not native
        transport.writer.write(b"synthetic bytes")
        await asyncio.sleep(0)
        assert proxy._transport_failures == {client.id}
        assert transport.writer.is_closing()
        native.close.assert_called_once_with()
        master.shutdown.assert_not_called()
        proxy.client_disconnected(client)
        assert not proxy._transport_failures

    asyncio.run(run())


@pytest.mark.parametrize("state", ["missing", "ended", "stopping", "no_master"])
def test_transport_failure_attribution_ignores_inactive_clients(state: str) -> None:
    """Late guard callbacks retain nothing after the client's or Capture's lifetime ends."""
    proxy = CaptureProxy(sink=lambda exchange: True, domains=("api.openai.com",))
    manager = Proxyserver()
    master = Mock(spec=DumpMaster, addons=Mock())
    master.addons.get.return_value = manager
    proxy._master = master
    client = connection.Client(peername=("127.0.0.1", 1), sockname=("127.0.0.1", 2))
    if state != "missing":
        manager.connections[client.id] = Mock(spec=ProxyConnectionHandler, client=client)
    if state == "ended":
        client.timestamp_end = 123.0
    elif state == "stopping":
        proxy.shutdown()
    elif state == "no_master":
        proxy._master = None
    proxy._record_transport_failure(client.id)
    assert not proxy._transport_failures


def test_transport_failure_history_is_bounded_by_active_client_lifetimes() -> None:
    """Repeated failures deduplicate and completed connections leave no retained identifiers."""
    proxy = CaptureProxy(sink=lambda exchange: True, domains=("api.openai.com",))
    manager = Proxyserver()
    master = Mock(spec=DumpMaster, addons=Mock())
    master.addons.get.return_value = manager
    proxy._master = master
    for port in range(1, 5):
        client = connection.Client(peername=("127.0.0.1", port), sockname=("127.0.0.1", 443))
        manager.connections[client.id] = Mock(spec=ProxyConnectionHandler, client=client)
        proxy._record_transport_failure(client.id)
        proxy._record_transport_failure(client.id)
        assert proxy._transport_failures == {client.id}
        client.timestamp_end = 123.0
        proxy.client_disconnected(client)
        proxy._record_transport_failure(client.id)
        assert not proxy._transport_failures
        del manager.connections[client.id]
        proxy._record_transport_failure(client.id)
        assert not proxy._transport_failures


def test_native_and_sibling_disconnects_do_not_disable_capture() -> None:
    """Native transport failures and ambiguous sibling disconnects stay connection-local."""
    proxy = CaptureProxy(sink=lambda exchange: True, domains=("api.openai.com",))
    manager = Proxyserver()
    master = Mock(spec=DumpMaster, addons=Mock())
    master.addons.get.return_value = manager
    proxy._master = master
    failed = connection.Client(
        peername=("127.0.0.1", 1),
        sockname=("127.0.0.1", 443),
        sni="api.openai.com",
        error="The client disconnected during the handshake.",
    )
    manager.connections[failed.id] = Mock(spec=ProxyConnectionHandler, client=failed, transports={})
    proxy._record_transport_failure(failed.id)
    proxy.tls_failed_client(tls.TlsData(failed, context.Context(failed, options.Options())))
    for attempt in range(3):
        sibling = connection.Client(
            peername=("127.0.0.1", attempt + 2),
            sockname=("127.0.0.1", 443),
            sni="api.openai.com",
            error="The client disconnected during the handshake.",
        )
        proxy.tls_failed_client(tls.TlsData(sibling, context.Context(sibling, options.Options())))
    master.shutdown.assert_not_called()
    assert not proxy._host_bypasses
    proxy.client_disconnected(failed)
    assert not proxy._transport_failures


def test_real_silent_tls_failures_do_not_exclude_trusted_clients(
    tmp_path: Path, regular_proxy: None
) -> None:
    """Real ambiguous TLS failures leave later trusted connections eligible for capture."""

    async def run() -> None:
        """Retry untrusted handshakes through an explicitly addressed loopback proxy only."""
        hostname = "api.openai.com"
        _, _, unrelated_ca = _certificate(tmp_path / "unrelated", "unrelated.example")
        diagnostics: list[str] = []
        proxy = CaptureProxy(
            sink=lambda exchange: True, domains=(hostname,), on_diagnostic=diagnostics.append
        )
        ready = asyncio.Event()
        proxy_task = asyncio.create_task(
            proxy.serve(ca_directory=tmp_path / "proxy", ready=ready.set)
        )
        try:
            await asyncio.wait_for(ready.wait(), 5)
            assert proxy._master is not None
            proxyserver = proxy._master.addons.get("proxyserver")
            assert isinstance(proxyserver, Proxyserver)
            proxy_port = next(iter(proxyserver.servers)).listen_addrs[0][1]
            for attempt in range(3):
                reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
                try:
                    # Lazy TLS inspection fails before connecting to this closed local port.
                    writer.write(b"CONNECT 127.0.0.1:1 HTTP/1.1\r\nHost: 127.0.0.1:1\r\n\r\n")
                    await writer.drain()
                    assert (await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 200")
                    client_context = ssl.create_default_context(cafile=str(unrelated_ca))
                    with pytest.raises(ssl.SSLCertVerificationError):
                        await asyncio.wait_for(
                            writer.start_tls(client_context, server_hostname=hostname), 5
                        )
                finally:
                    writer.close()
                    with suppress(OSError):
                        await writer.wait_closed()
                # Wait for the real TLS callback before checking continued availability.
                deadline = asyncio.get_running_loop().time() + 2
                while sum("tls_client_failed:" in event for event in diagnostics) < attempt + 1:
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.01)
                assert not proxy._host_bypasses and not proxy._app_bypasses
                _, trusted_writer, _ = await _connect_tls(
                    proxy, 1, hostname, tmp_path / "proxy/mitmproxy-ca-cert.pem"
                )
                trusted_writer.close()
                with suppress(ConnectionError):
                    await trusted_writer.wait_closed()
            assert not proxy_task.done()
            assert proxy._master is not None
            assert all(instance.is_running for instance in proxyserver.servers)
        finally:
            proxy.shutdown()
            with suppress(RuntimeError):
                await asyncio.wait_for(proxy_task, 5)

    asyncio.run(run())


def test_native_process_identity_survives_writer_guard() -> None:
    """Process scoping reads the native reader even after Capture wraps its writer."""

    async def run() -> None:
        """Install the writer adapter on its owning event loop."""
        proxy = CaptureProxy(sink=lambda exchange: True, domains=("api.openai.com",))
        manager = Proxyserver()
        master = Mock(spec=DumpMaster, addons=Mock())
        master.addons.get.return_value = manager
        proxy._master = master
        client = connection.Client(peername=("127.0.0.1", 1), sockname=("127.0.0.1", 443))
        native = Mock(spec=mitmproxy_rs.Stream)
        native.get_extra_info.side_effect = {"pid": 123, "process_name": "/Applications/client"}.get
        transport = ConnectionIO(reader=native, writer=native)
        manager.connections[client.id] = Mock(
            spec=ProxyConnectionHandler, client=client, transports={client: transport}
        )
        proxy.client_connected(client)
        assert transport.writer is not native and transport.reader is native
        assert proxy._client_identity(client) == (123, "/Applications/client")

    asyncio.run(run())


@pytest.mark.parametrize("pid,name", [(None, "/app"), (0, "/app"), (1, None), (1, "")])
def test_incomplete_native_process_identity_uses_explicit_host_fallback(
    pid: int | None, name: str | None
) -> None:
    """Unavailable metadata never creates an ambiguous app-level bypass key."""
    proxy = CaptureProxy(sink=lambda exchange: True, domains=("api.openai.com",))
    manager = Proxyserver()
    master = Mock(spec=DumpMaster, addons=Mock())
    master.addons.get.return_value = manager
    proxy._master = master
    client = connection.Client(
        peername=("127.0.0.1", 1),
        sockname=("127.0.0.1", 443),
        sni="api.openai.com",
        error="unknown ca",
    )
    native = Mock(spec=mitmproxy_rs.Stream)
    native.get_extra_info.side_effect = {"pid": pid, "process_name": name}.get
    manager.connections[client.id] = Mock(
        spec=ProxyConnectionHandler,
        client=client,
        transports={client: ConnectionIO(reader=native, writer=native)},
    )
    proxy.tls_failed_client(tls.TlsData(client, context.Context(client, options.Options())))
    assert proxy._host_bypasses == {"api.openai.com"}
    assert not proxy._app_bypasses
    master.shutdown.assert_not_called()


def test_bypass_is_scoped_to_process_and_host_and_sanitizes_display_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing app cannot suppress healthy apps or another selected provider."""
    warnings: list[tuple[str, str, capture_module.CaptureBypassReason]] = []
    proxy = CaptureProxy(
        sink=lambda exchange: True,
        domains=("api.openai.com", "api.anthropic.com"),
        on_bypass=lambda app, host, reason: warnings.append((app, host, reason)),
    )
    identity = (123, "/Applications/cli\n\x1b")
    monkeypatch.setattr(proxy, "_client_identity", lambda client: identity)
    client = connection.Client(
        peername=("127.0.0.1", 1),
        sockname=("127.0.0.1", 443),
        sni="api.openai.com",
        error="unknown ca",
    )
    ctx = context.Context(client, options.Options())
    proxy.tls_failed_client(tls.TlsData(client, ctx))
    proxy.tls_failed_client(tls.TlsData(client, ctx))
    assert warnings == [("cli", "api.openai.com", "certificate")]
    for pid, name, host, expected in [
        (123, "/Applications/cli\n\x1b", "API.OPENAI.COM.", True),
        (456, "/Applications/cli\n\x1b", "api.openai.com", False),
        (123, "/Applications/other", "api.openai.com", False),
        (123, "/Applications/cli\n\x1b", "api.anthropic.com", False),
        (123, "/Applications/cli\n\x1b", "unselected.example", False),
    ]:
        identity = (pid, name)
        hello = tls.ClientHelloData(ctx, Mock(spec=tls.ClientHello, sni=host))
        proxy.tls_clienthello(hello)
        assert hello.ignore_connection is expected
    assert not proxy._host_bypasses


def test_bypass_capacity_falls_back_to_selected_host_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full process table remains bounded and visibly isolates the overflow host."""
    monkeypatch.setattr(capture_module, "_MAX_TLS_BYPASSES", 2)
    identity = (1, "/Applications/client")
    warnings: list[tuple[str, str, capture_module.CaptureBypassReason]] = []
    proxy = CaptureProxy(
        sink=lambda exchange: True,
        domains=("api.openai.com", "api.anthropic.com"),
        on_bypass=lambda app, host, reason: warnings.append((app, host, reason)),
    )
    monkeypatch.setattr(proxy, "_client_identity", lambda client: identity)
    client = connection.Client(
        peername=("127.0.0.1", 1),
        sockname=("127.0.0.1", 443),
        sni="api.openai.com",
        error="unknown ca",
    )
    ctx = context.Context(client, options.Options())
    for pid in range(1, 5):
        identity = (pid, "/Applications/client")
        proxy.tls_failed_client(tls.TlsData(client, ctx))
    assert len(proxy._app_bypasses) == 2
    assert proxy._host_bypasses == {"api.openai.com"}
    assert len(warnings) == 3 and warnings[-1] == ("All apps", "api.openai.com", "certificate")


def test_generic_handshake_bursts_do_not_disable_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two apps' isolated cancellations do not combine into a false shared trust failure."""
    warnings: list[tuple[str, str, capture_module.CaptureBypassReason]] = []
    proxy = CaptureProxy(
        sink=lambda exchange: True,
        domains=("api.openai.com",),
        on_bypass=lambda app, host, reason: warnings.append((app, host, reason)),
    )
    identity = (1, "/app")
    monkeypatch.setattr(proxy, "_client_identity", lambda client: identity)
    client = connection.Client(
        peername=("127.0.0.1", 1),
        sockname=("127.0.0.1", 443),
        sni="api.openai.com",
        error="The client disconnected during the handshake.",
    )
    data = tls.TlsData(client, context.Context(client, options.Options()))
    for pid in (1, 2, 1, 2):
        identity = (pid, "/app")
        proxy.tls_failed_client(data)
    assert not warnings
    identity = (1, "/app")
    proxy.tls_failed_client(data)
    assert not warnings
    assert not proxy._app_bypasses
    assert not proxy._host_bypasses


@pytest.mark.parametrize("broken_diagnostics", [False, True])
def test_websocket_capture_survives_handshake_bursts_and_safe_diagnostics(
    broken_diagnostics: bool,
) -> None:
    """Capture completed Responses messages without retaining websocket history."""
    captured: list[CapturedExchange] = []
    diagnostics: list[str] = []

    def diagnostic(message: str) -> None:
        """Exercise both recording and a broken console without affecting inference."""
        diagnostics.append(message)
        if broken_diagnostics:
            raise OSError("synthetic console failure")

    proxy = CaptureProxy(
        sink=lambda exchange: captured.append(exchange) is None,
        domains=("api.openai.com",),
        on_diagnostic=diagnostic,
    )
    flow = http.HTTPFlow(
        connection.Client(
            peername=("127.0.0.1", 12), sockname=("127.0.0.1", 13), sni="api.openai.com"
        ),
        connection.Server(address=("api.openai.com", 443)),
    )
    flow.request = http.Request.make(
        "GET", "https://api.openai.com/v1/responses", headers={"Host": "api.openai.com"}
    )
    flow.request.headers["Authorization"] = "Bearer secret-credential"
    flow.request.path += "?secret-query=1"
    flow.client_conn.error = "The client disconnected during the handshake. secret-error"
    ctx = context.Context(flow.client_conn, options.Options())
    for _ in range(4):
        proxy.tls_failed_client(tls.TlsData(flow.client_conn, ctx))
    hello = tls.ClientHelloData(ctx, Mock(spec=tls.ClientHello, sni="api.openai.com"))
    proxy.tls_clienthello(hello)
    assert not hello.ignore_connection
    asyncio.run(proxy.requestheaders(flow))
    flow.response = http.Response.make(101)
    proxy.responseheaders(flow)
    flow.websocket = websocket.WebSocketData()
    frames = [
        (True, {"type": "response.create", "model": "test", "input": "secret-prompt"}),
        (False, {"type": "response.created", "response": {"id": "r1"}}),
        (
            False,
            {"type": "response.completed", "response": {"id": "r1", "model": "test", "output": []}},
        ),
    ]
    for from_client, event in frames:
        raw = json.dumps(event).encode()
        message = websocket.WebSocketMessage(Opcode.TEXT, from_client, raw)
        flow.websocket.messages.append(message)
        proxy.websocket_message(flow)
        assert message.content == raw
        assert not message.dropped
        assert not flow.websocket.messages
    assert len(captured) == 1
    assert json.loads(captured[0].request)["input"] == "secret-prompt"

    assert any("websocket_request_started" in event for event in diagnostics)
    assert any("websocket_request_completed" in event for event in diagnostics)
    assert not any("secret" in event or "Bearer" in event for event in diagnostics)


def test_idle_websockets_do_not_exhaust_request_capture_capacity() -> None:
    """Many persistent idle connections share one bounded active-request slot."""
    captured: list[CapturedExchange] = []
    proxy = CaptureProxy(
        sink=lambda exchange: captured.append(exchange) is None,
        domains=("chatgpt.com",),
        max_active_flows=1,
    )
    flows: list[http.HTTPFlow] = []
    for index in range(20):
        flow = http.HTTPFlow(
            connection.Client(
                peername=("127.0.0.1", index + 1),
                sockname=("127.0.0.1", 443),
                sni="chatgpt.com",
            ),
            connection.Server(address=("chatgpt.com", 443)),
        )
        flow.request = http.Request.make(
            "GET",
            "https://chatgpt.com/backend-api/codex/responses",
            headers={"Host": "chatgpt.com", "Upgrade": "websocket"},
        )
        asyncio.run(proxy.requestheaders(flow))
        proxy.request(flow)
        flow.response = http.Response.make(101)
        proxy.responseheaders(flow)
        proxy.response(flow)
        flow.websocket = websocket.WebSocketData()
        flows.append(flow)
    assert proxy.dropped_exchanges == 0
    assert not proxy._captures

    def send(flow: http.HTTPFlow, from_client: bool, event: dict[str, str]) -> None:
        """Feed exact WebSocket bytes while asserting transparent forwarding."""
        assert flow.websocket is not None
        raw = json.dumps(event).encode()
        message = websocket.WebSocketMessage(Opcode.TEXT, from_client, raw)
        flow.websocket.messages.append(message)
        proxy.websocket_message(flow)
        assert message.content == raw and not message.dropped
        assert not flow.websocket.messages

    # A genuinely concurrent request still respects the existing capacity bound.
    send(flows[0], True, {"type": "response.create", "model": "test", "input": "first"})
    assert len(proxy._captures) == 1
    send(flows[1], True, {"type": "response.create", "model": "test", "input": "overflow"})
    assert proxy.dropped_exchanges == 1
    proxy.websocket_end(flows[0])
    assert not proxy._captures
    # A skipped request must finish before this socket resumes capture, preventing
    # its response from being attached to a later request when capacity returns.
    send(flows[1], True, {"type": "response.create", "model": "test", "input": "also skipped"})
    assert proxy.dropped_exchanges == 2
    assert flows[1].websocket is not None
    for _ in range(2):
        flows[1].websocket.messages.append(
            websocket.WebSocketMessage(
                Opcode.TEXT,
                False,
                json.dumps({"type": "response.completed", "response": {"id": "skipped"}}).encode(),
            )
        )
        proxy.websocket_message(flows[1])
    for flow in flows[1:]:
        send(flow, True, {"type": "response.create", "model": "test", "input": "next"})
        assert len(proxy._captures) == 1
        assert flow.websocket is not None
        flow.websocket.messages.append(
            websocket.WebSocketMessage(
                Opcode.TEXT,
                False,
                json.dumps({"type": "response.completed", "response": {"id": "done"}}).encode(),
            )
        )
        proxy.websocket_message(flow)
        assert not proxy._captures
    assert len(captured) == 20
    assert proxy.dropped_exchanges == 2


@pytest.mark.parametrize("skipped_first", [False, True])
@pytest.mark.parametrize("error_outcome", [False, True])
def test_overflowed_websocket_group_cannot_misattribute_responses(
    skipped_first: bool, error_outcome: bool
) -> None:
    """Unknown response IDs across overflow drain without uploading mismatched pairs."""
    captured: list[CapturedExchange] = []
    first = json.dumps({"type": "response.create", "input": "first" * 16}).encode()
    second = json.dumps({"type": "response.create", "input": "second" * 16}).encode()
    proxy = CaptureProxy(
        sink=lambda exchange: captured.append(exchange) is None,
        domains=("chatgpt.com",),
        max_body_bytes=max(len(first), len(second)) + 1,
    )
    flow = http.HTTPFlow(
        connection.Client(
            peername=("127.0.0.1", 1), sockname=("127.0.0.1", 443), sni="chatgpt.com"
        ),
        connection.Server(address=("chatgpt.com", 443)),
    )
    flow.request = http.Request.make(
        "GET",
        "https://chatgpt.com/backend-api/codex/responses",
        headers={"Host": "chatgpt.com"},
    )
    asyncio.run(proxy.requestheaders(flow))
    flow.response = http.Response.make(101)
    proxy.responseheaders(flow)
    flow.websocket = websocket.WebSocketData()

    def send(from_client: bool, raw: bytes) -> None:
        """Forward unchanged synthetic frames through production capture callbacks."""
        assert flow.websocket is not None
        message = websocket.WebSocketMessage(Opcode.TEXT, from_client, raw)
        flow.websocket.messages.append(message)
        proxy.websocket_message(flow)
        assert message.content == raw and not message.dropped

    send(True, first)
    send(True, second)
    assert not proxy._captures
    assert proxy.dropped_exchanges == 2
    for response_id in ("second", "first") if skipped_first else ("first", "second"):
        if response_id == "second" and error_outcome:
            send(False, b'{"type":"error","status":400,"error":{"code":"invalid_request"}}')
            continue
        for kind in ("response.created", "response.completed"):
            send(False, json.dumps({"type": kind, "response": {"id": response_id}}).encode())
        assert not captured
    send(True, first)
    for kind in ("response.created", "response.completed"):
        send(False, json.dumps({"type": kind, "response": {"id": "fresh"}}).encode())
    assert len(captured) == 1
    assert captured[0].request == first
    assert json.loads(captured[0].response)["id"] == "fresh"
    assert not proxy._captures
    assert proxy.dropped_exchanges == 2


@pytest.mark.parametrize("from_client", [False, True])
def test_oversized_websocket_frame_never_pairs_unknown_responses(from_client: bool) -> None:
    """Unparsed oversized frames make only this socket ineligible for collection."""
    captured: list[CapturedExchange] = []
    proxy = CaptureProxy(
        sink=lambda exchange: captured.append(exchange) is None,
        domains=("chatgpt.com",),
        max_body_bytes=160,
    )
    flow = http.HTTPFlow(
        connection.Client(
            peername=("127.0.0.1", 1), sockname=("127.0.0.1", 443), sni="chatgpt.com"
        ),
        connection.Server(address=("chatgpt.com", 443)),
    )
    flow.request = http.Request.make(
        "GET", "https://chatgpt.com/responses", headers={"Host": "chatgpt.com"}
    )
    asyncio.run(proxy.requestheaders(flow))
    flow.response = http.Response.make(101)
    proxy.responseheaders(flow)
    flow.websocket = websocket.WebSocketData()
    frames = [
        (True, b'{"type":"response.create","input":"first"}'),
        (from_client, json.dumps({"type": "response.create", "input": "x" * 200}).encode()),
        (False, b'{"type":"response.created","response":{"id":"oversized"}}'),
        (False, b'{"type":"response.completed","response":{"id":"oversized"}}'),
        (False, b'{"type":"response.completed","response":{"id":"first"}}'),
        (True, b'{"type":"response.create","input":"later"}'),
        (False, b'{"type":"response.completed","response":{"id":"later"}}'),
    ]
    for from_client, raw in frames:
        message = websocket.WebSocketMessage(Opcode.TEXT, from_client, raw)
        flow.websocket.messages.append(message)
        proxy.websocket_message(flow)
        assert message.content == raw and not message.dropped
        assert not flow.websocket.messages
    assert not proxy._captures
    assert not captured


@pytest.mark.parametrize("parallel_streams", [False, True])
def test_websocket_response_ownership_survives_errors_and_parallel_lanes(
    parallel_streams: bool,
) -> None:
    """Errors and interleaved named lanes cannot steal another request's response."""
    captured: list[CapturedExchange] = []
    proxy = CaptureProxy(
        sink=lambda exchange: captured.append(exchange) is None, domains=("chatgpt.com",)
    )
    flow = http.HTTPFlow(
        connection.Client(
            peername=("127.0.0.1", 1), sockname=("127.0.0.1", 443), sni="chatgpt.com"
        ),
        connection.Server(address=("chatgpt.com", 443)),
    )
    flow.request = http.Request.make(
        "GET", "https://chatgpt.com/responses", headers={"Host": "chatgpt.com"}
    )
    asyncio.run(proxy.requestheaders(flow))
    flow.response = http.Response.make(101)
    proxy.responseheaders(flow)
    flow.websocket = websocket.WebSocketData()
    frames = [
        (True, b'{"type":"response.create","input":"failed"}'),
        (False, b'{"type":"error","status":400,"error":{"code":"invalid_request"}}'),
        (True, b'{"type":"response.create","input":"next"}'),
        (False, b'{"type":"response.created","response":{"id":"next"}}'),
        (False, b'{"type":"response.completed","response":{"id":"next"}}'),
    ]
    if parallel_streams:
        frames = [
            (True, b'{"type":"response.create","stream_id":"a","input":"first"}'),
            (True, b'{"type":"response.create","stream_id":"b","input":"second"}'),
            (False, b'{"type":"response.created","stream_id":"b","response":{"id":"second"}}'),
            (False, b'{"type":"response.completed","stream_id":"b","response":{"id":"second"}}'),
            (False, b'{"type":"response.created","stream_id":"a","response":{"id":"first"}}'),
            (False, b'{"type":"response.completed","stream_id":"a","response":{"id":"first"}}'),
        ]
    for from_client, raw in frames:
        message = websocket.WebSocketMessage(Opcode.TEXT, from_client, raw)
        flow.websocket.messages.append(message)
        proxy.websocket_message(flow)
        assert message.content == raw and not message.dropped
    assert not proxy._captures
    assert proxy.dropped_exchanges == (0 if parallel_streams else 1)
    expected = ["second", "first"] if parallel_streams else ["next"]
    assert [json.loads(exchange.request)["input"] for exchange in captured] == expected
    assert [json.loads(exchange.response)["id"] for exchange in captured] == expected


@pytest.mark.parametrize(
    "scenario",
    ["unmatched_host", "authority_mismatch", "unsupported_path", "large_request", "large_response"],
)
def test_real_tls_uncaptured_traffic_keeps_destination_and_bytes(
    tmp_path: Path, regular_proxy: None, scenario: str
) -> None:
    """Prove ignored hosts keep their certificate and uncaptured HTTP is forwarded intact."""

    async def run() -> None:
        """Run one isolated TLS exchange without DNS or system capture changes."""
        hostname = "unrelated.example" if scenario == "unmatched_host" else "api.openai.com"
        authority = "api.anthropic.com" if scenario == "authority_mismatch" else hostname
        request_path = "/unknown" if scenario == "unsupported_path" else "/v1/responses"
        request = b'{"input":"' + (b"x" * 80 if scenario == "large_request" else b"hi") + b'"}'
        response = b'{"output":"' + (b"y" * 80 if scenario == "large_response" else b"ok") + b'"}'
        certificate, key, upstream_ca = _certificate(tmp_path / "upstream", hostname)
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(certificate, key)
        sni_seen: list[str | None] = []

        def record_sni(
            socket: ssl.SSLSocket | ssl.SSLObject,
            name: str | None,
            context: ssl.SSLContext | ssl.SSLSocket,
        ) -> None:
            """Observe the untouched TLS server name at the original destination."""
            sni_seen.append(name)

        server_context.set_servername_callback(record_sni)
        received: list[bytes] = []

        async def upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Echo a finite HTTP reply after recording the forwarded request body."""
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                received.append(headers + await reader.readexactly(len(request)))
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                    + str(len(response)).encode()
                    + b'\r\nAlt-Svc: h3=":443"\r\nConnection: close\r\n\r\n'
                    + response
                )
                await writer.drain()
            finally:
                writer.close()
                with suppress(ConnectionError):
                    await writer.wait_closed()

        server = await asyncio.start_server(upstream, "127.0.0.1", 0, ssl=server_context)
        upstream_port = server.sockets[0].getsockname()[1]
        captured: list[CapturedExchange] = []
        proxy = CaptureProxy(
            sink=lambda exchange: captured.append(exchange) is None,
            domains=("api.openai.com", "api.anthropic.com"),
            max_body_bytes=64,
            upstream_ca_file=upstream_ca,
        )
        ready = asyncio.Event()
        proxy_task = asyncio.create_task(
            proxy.serve(ca_directory=tmp_path / "proxy", ready=ready.set)
        )
        writer: asyncio.StreamWriter | None = None
        try:
            await asyncio.wait_for(ready.wait(), 5)
            # An ignored host must validate against its real CA alone. Trusting the
            # proxy CA here would hide accidental interception of an unrelated host.
            trusted_ca = (
                upstream_ca
                if scenario == "unmatched_host"
                else tmp_path / "proxy/mitmproxy-ca-cert.pem"
            )
            reader, writer, peer_certificate = await _connect_tls(
                proxy, upstream_port, hostname, trusted_ca
            )
            upstream_leaf = x509.load_pem_x509_certificate(certificate.read_bytes()).public_bytes(
                serialization.Encoding.DER
            )
            assert (peer_certificate == upstream_leaf) == (scenario == "unmatched_host")
            wire_request = (
                f"POST {request_path} HTTP/1.1\r\nHost: {authority}\r\n".encode()
                + b"Content-Type: application/json\r\nContent-Length: "
                + str(len(request)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + request
            )
            writer.write(wire_request)
            await writer.drain()
            headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            assert headers.startswith(b"HTTP/1.1 200"), headers
            assert b'Alt-Svc: h3=":443"\r\n' in headers
            assert await asyncio.wait_for(reader.readexactly(len(response)), 5) == response
            assert await asyncio.wait_for(reader.read(), 5) == b""
            assert received == [wire_request]
            assert sni_seen == [hostname]
            assert not captured
            assert not proxy._captures
            assert proxy.dropped_exchanges == int(scenario in {"large_request", "large_response"})
        finally:
            if writer is not None:
                writer.close()
                with suppress(ConnectionError):
                    await writer.wait_closed()
            proxy.shutdown()
            await asyncio.wait_for(proxy_task, 5)
            server.close()
            await server.wait_closed()

    asyncio.run(run())


def test_real_unknown_tls_protocol_passes_through_without_capture(
    tmp_path: Path, regular_proxy: None
) -> None:
    """Opaque application bytes on a selected TLS host remain usable and unrecorded."""

    async def run() -> None:
        """Send an application protocol that cannot be mistaken for an HTTP request."""
        hostname = "api.openai.com"
        certificate, key, upstream_ca = _certificate(tmp_path / "upstream", hostname)
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(certificate, key)
        payloads = (b"\x00opaque-client-frame\xff\x01", b"\x00second-client-frame\xff\x03")
        replies = (b"\x00opaque-server-frame\xfe\x02", b"\x00second-server-frame\xfe\x04")
        received: list[bytes] = []
        observed_flows: list[tcp.TCPFlow] = []
        retained_sizes: list[int] = []

        class HistoryObserver:
            """Observe mitmproxy's own flow state after Capture has handled each chunk."""

            def tcp_message(self, flow: tcp.TCPFlow) -> None:
                """Keep flow references so later cleanup cannot hide retained opaque bytes."""
                observed_flows.append(flow)
                retained_sizes.append(sum(len(message.content) for message in flow.messages))

        async def upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Read and return binary frames over real TLS with no HTTP interpretation."""
            try:
                for payload, reply in zip(payloads, replies, strict=True):
                    received.append(await reader.readexactly(len(payload)))
                    writer.write(reply)
                    await writer.drain()
            finally:
                writer.close()
                with suppress(ConnectionError):
                    await writer.wait_closed()

        server = await asyncio.start_server(upstream, "127.0.0.1", 0, ssl=server_context)
        captured: list[CapturedExchange] = []
        proxy = CaptureProxy(
            sink=lambda exchange: captured.append(exchange) is None,
            domains=(hostname,),
            upstream_ca_file=upstream_ca,
        )
        ready = asyncio.Event()
        proxy_task = asyncio.create_task(
            proxy.serve(ca_directory=tmp_path / "proxy", ready=ready.set)
        )
        writer: asyncio.StreamWriter | None = None
        try:
            await asyncio.wait_for(ready.wait(), 5)
            assert proxy._master is not None
            proxy._master.addons.add(HistoryObserver())
            reader, writer, _ = await _connect_tls(
                proxy,
                server.sockets[0].getsockname()[1],
                hostname,
                tmp_path / "proxy/mitmproxy-ca-cert.pem",
            )
            for payload, reply in zip(payloads, replies, strict=True):
                writer.write(payload)
                await writer.drain()
                assert await asyncio.wait_for(reader.readexactly(len(reply)), 5) == reply
            assert await asyncio.wait_for(reader.read(), 5) == b""
            assert received == list(payloads)
            assert len(observed_flows) >= 4
            assert set(retained_sizes) == {0}
            assert all(not flow.messages for flow in observed_flows)
            assert not captured
            assert not proxy._captures
            assert proxy.dropped_exchanges == 0
        finally:
            if writer is not None:
                writer.close()
                with suppress(ConnectionError):
                    await writer.wait_closed()
            proxy.shutdown()
            await asyncio.wait_for(proxy_task, 5)
            server.close()
            await server.wait_closed()

    asyncio.run(run())


@pytest.mark.parametrize("port,payload", [(443, b"\xc0\x00\x00\x00\x01quic"), (53, b"\x01\x02dns")])
def test_udp_and_quic_bypass_decryption_and_preserve_datagrams(port: int, payload: bytes) -> None:
    """Select an unrecorded UDP layer before QUIC/DNS parsing and relay both directions."""
    captured: list[CapturedExchange] = []
    proxy = CaptureProxy(
        sink=lambda exchange: captured.append(exchange) is None, domains=("api.openai.com",)
    )
    client = connection.Client(
        peername=("127.0.0.1", 4567), sockname=("127.0.0.1", port), transport_protocol="udp"
    )
    ctx = context.Context(client, options.Options())
    ctx.server = connection.Server(
        address=("api.openai.com", port), transport_protocol="udp", timestamp_start=1.0
    )
    next_layer = layer.NextLayer(ctx)
    proxy.next_layer(next_layer)
    bypass = next_layer.layer
    assert isinstance(bypass, layers.UDPLayer)
    assert bypass.flow is None
    assert list(bypass.handle_event(events.Start())) == []
    for source, destination in [(ctx.client, ctx.server), (ctx.server, ctx.client)]:
        forwarded = list(bypass.handle_event(events.DataReceived(source, payload)))
        assert len(forwarded) == 1
        assert isinstance(forwarded[0], commands.SendData)
        assert forwarded[0].connection is destination
        assert forwarded[0].data == payload
    assert not captured
    assert not proxy._captures


@pytest.mark.parametrize("ending", ["error", "response", "shutdown"])
def test_abandoned_websocket_upgrade_is_not_a_model_request(ending: str) -> None:
    """An empty GET upgrade must not inflate capture or drop counts before response.create."""
    captured: list[CapturedExchange] = []
    proxy = CaptureProxy(
        sink=lambda exchange: captured.append(exchange) is None, domains=("chatgpt.com",)
    )
    flow = http.HTTPFlow(
        connection.Client(peername=("127.0.0.1", 1), sockname=("127.0.0.1", 2), sni="chatgpt.com"),
        connection.Server(address=("chatgpt.com", 443)),
    )
    flow.request = http.Request.make(
        "GET",
        "https://chatgpt.com/backend-api/codex/responses",
        headers={"Host": "chatgpt.com", "Upgrade": "websocket"},
    )
    asyncio.run(proxy.requestheaders(flow))
    proxy.request(flow)
    if ending == "error":
        proxy.error(flow)
    elif ending == "response":
        flow.response = http.Response.make(403)
        proxy.responseheaders(flow)
        proxy.response(flow)
    else:
        proxy._finish_pending()
    assert not captured
    assert not proxy._captures
    assert proxy.dropped_exchanges == 0


@pytest.mark.parametrize("port", [53, 443])
def test_dns_diagnostics_report_lifecycle_without_destinations_or_queries(port: int) -> None:
    """Verbose DNS diagnostics reveal forwarding progress without recording unrelated traffic."""
    diagnostics: list[str] = []
    proxy = CaptureProxy(
        sink=lambda exchange: True, domains=("chatgpt.com",), on_diagnostic=diagnostics.append
    )
    manager = Proxyserver()
    master = Mock(spec=DumpMaster, addons=Mock())
    master.addons.get.return_value = manager
    proxy._master = master
    client = connection.Client(
        peername=("127.0.0.1", 1), sockname=("127.0.0.1", 2), transport_protocol="udp"
    )
    ctx = context.Context(client, options.Options())
    ctx.server.address = ("private-resolver.example", port)
    manager.connections[client.id] = Mock(
        spec=ProxyConnectionHandler, client=client, layer=layer.NextLayer(ctx), transports={}
    )
    proxy.client_connected(client)
    proxy.client_disconnected(client)
    if port == 53:
        assert len(diagnostics) == 2
        assert all("DNS forwarding" in event for event in diagnostics)
        assert not any("private-resolver" in event for event in diagnostics)
    else:
        assert not diagnostics
