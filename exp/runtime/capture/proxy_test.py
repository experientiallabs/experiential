"""Synthetic TLS providers prove byte-preserving streaming and upstream verification."""

import asyncio
import json
import ssl
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from mitmproxy import certs, connection, http, websocket
from wsproto.frame_protocol import Opcode

from exp.common.core.artifacts import SourceIdentity
from exp.runtime.capture.normalization import CapturedExchange
from exp.runtime.capture.proxy import CaptureProxy, _Body
from exp.runtime.capture.resolver import UpstreamResolver
from exp.runtime.capture.upload import CaptureUploader
from exp.simulation.ingest.otlp import normalize_otlp_payload


class _LocalResolver(UpstreamResolver):
    """Resolve synthetic providers to loopback without altering system DNS."""

    async def resolve(self, host: str) -> str:
        """Return loopback exclusively for the synthetic TLS server."""
        return "127.0.0.1"


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


@pytest.mark.parametrize("valid_hostname", [True, False])
def test_real_tls_sse_passes_unchanged_and_rejects_wrong_upstream_hostname(
    tmp_path: Path, valid_hostname: bool
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
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
                    + str(len(first + last)).encode()
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
        run_id, ingest_id = str(uuid4()), str(uuid4())
        upload_prefix = (
            "/storage/v1/object/upload/sign/artifacts/orgs/organization/telemetry-traces/otlp/"
        )

        def platform(request: httpx.Request) -> httpx.Response:
            """Emulate signed upload and finalize while validating projected trace evidence."""
            if request.url.host == "storage.example":
                assert "authorization" not in request.headers
                assert b"Bearer secret" not in request.content
                normalized = normalize_otlp_payload(
                    json.loads(request.content),
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
            resolver=_LocalResolver(),
            upstream_port=upstream_port,
            upstream_ca_file=upstream_ca,
        )
        ready = asyncio.Event()
        proxy_task = asyncio.create_task(
            proxy.serve(port=0, ca_directory=tmp_path / "proxy", ready=ready.set)
        )
        try:
            await asyncio.wait_for(ready.wait(), 5)
            assert proxy._master is not None
            proxyserver = proxy._master.addons.get("proxyserver")
            proxy_port = next(iter(proxyserver.servers)).listen_addrs[0][1]
            client_context = ssl.create_default_context(
                cafile=str(tmp_path / "proxy/mitmproxy-ca-cert.pem")
            )
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", proxy_port, ssl=client_context, server_hostname=host
            )
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
                await asyncio.sleep(0.02)
                assert len(captured) == 1
                assert captured[0].request == request
                assert captured[0].response == first + last
                assert captured[0].host == host
                deadline = asyncio.get_running_loop().time() + 3
                while uploader.stats.uploaded_batches != 1:
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.01)
                assert projected == [(4, 2)]
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


def test_stream_copy_limit_never_changes_forwarded_chunks() -> None:
    """Discard oversized copies while forwarding every original byte."""
    body = _Body(3)
    assert body.tee(b"ab") == b"ab"
    assert body.tee(b"cd") == b"cd"
    assert body.tee(b"ef") == b"ef"
    assert body.overflow and not body.data


def test_websocket_request_completion_retains_no_flow_message_history() -> None:
    """Capture completed Responses messages without retaining websocket history."""
    captured: list[CapturedExchange] = []
    proxy = CaptureProxy(
        sink=lambda exchange: captured.append(exchange) is None,
        domains=("api.openai.com",),
        resolver=_LocalResolver(),
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
    asyncio.run(proxy.requestheaders(flow))
    flow.response = http.Response.make(101)
    proxy.responseheaders(flow)
    flow.websocket = websocket.WebSocketData()
    frames = [
        (True, {"type": "response.create", "model": "test", "input": "hello"}),
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
    assert json.loads(captured[0].request)["input"] == "hello"
