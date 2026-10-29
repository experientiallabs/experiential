"""TLS pass-through capture with bounded copies and independently queued delivery."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

import mitmproxy_rs
from cryptography import x509
from mitmproxy import certs, connection, http, options, tcp, tls
from mitmproxy.addons.errorcheck import ErrorCheck
from mitmproxy.addons.proxyserver import Proxyserver
from mitmproxy.addons.tlsconfig import TlsConfig
from mitmproxy.proxy import context, layer, layers, server_hooks
from mitmproxy.tools.dump import DumpMaster

from exp.runtime.capture.normalization import CapturedExchange, CaptureProtocol, capture_protocol
from exp.runtime.capture.policy import validate_domains
from exp.runtime.capture.redirector import (
    capture_server_closed,
    start_capture_watchdog,
    stop_capture_servers,
)
from exp.runtime.capture.transports import guard_native_writer
from exp.runtime.capture.watchdog import CaptureWatchdog

logger = logging.getLogger(__name__)
_MAX_TLS_BYPASSES = 128
_WEBSOCKET_CAPTURE = "exp_capture_websocket"
_WEBSOCKET_SKIPPED = "exp_capture_skipped_requests"
CaptureBypassReason = Literal["certificate"]


class _CaptureTlsConfig(TlsConfig):
    """Issue DNS-only leaves without changing the connection's original destination.

    Mitmproxy also includes server.address in its default leaf SANs. Local mode
    supplies the original IP there, which Capture's constrained CA prohibits.
    Override only certificate selection and retain upstream TLS verification,
    cipher negotiation, and connection handling from the standard TLS addon.
    """

    name = "tlsconfig"

    def __init__(self, domains: frozenset[str]) -> None:
        """Bind certificate issuance to Capture's already validated exact DNS names."""
        self._domains = domains

    def get_cert(self, conn_context: context.Context) -> certs.CertStoreEntry:
        """Return a leaf for one selected SNI without borrowing any upstream SANs.

        Raises:
            RuntimeError: TLS inspection reached an unselected or missing DNS SNI.
        """
        host = (conn_context.client.sni or "").lower().rstrip(".")
        if host not in self._domains:
            raise RuntimeError("Capture TLS inspection requires a selected DNS server name.")
        return self.certstore.get_cert(host, [x509.DNSName(host)])


@dataclass
class _Body:
    """A capped byte tee that always returns the caller's original chunk."""

    limit: int
    data: bytearray = field(default_factory=bytearray)
    overflow: bool = False

    def tee(self, chunk: bytes) -> bytes:
        """Copy only within the cap and forward bytes without alteration."""
        if not self.overflow:
            if len(self.data) + len(chunk) > self.limit:
                self.data.clear()
                self.overflow = True
            else:
                self.data.extend(chunk)
        return chunk


@dataclass
class _WebsocketRequest:
    """One unstitched Responses request awaiting its protocol-level completion.

    Attributes:
        body: Bounded original request frame.
        started_ns: Request observation time in Unix nanoseconds.
        response_id: Provider ID assigned by response.created, initially unknown.
        stream_id: Named ordered lane, or None for the default lane.
        trace_id: Locally generated identifier joining this request to its saved capture.
    """

    body: bytes
    started_ns: int
    response_id: str | None = None
    stream_id: str | None = None
    trace_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class _Capture:
    """Only allowlisted request metadata and bounded copied bodies."""

    protocol: CaptureProtocol
    host: str
    path: str
    started_ns: int
    request: _Body
    response: _Body
    request_encoding: str
    response_encoding: str = ""
    response_content_type: str = ""
    status: int = 0
    request_done: bool = False
    response_done: bool = False
    failed: bool = False
    websocket: bool = False
    websocket_requests: deque[_WebsocketRequest] = field(default_factory=deque)
    trace_id: str = field(default_factory=lambda: uuid4().hex)


class CaptureProxy:
    """Observe allowed model exchanges while forwarding original stream bytes."""

    def __init__(
        self,
        *,
        sink: Callable[[CapturedExchange], bool],
        domains: tuple[str, ...],
        max_body_bytes: int = 8 * 1024 * 1024,
        max_active_flows: int = 16,
        upstream_ca_file: Path | None = None,
        on_bypass: Callable[[str, str, CaptureBypassReason], None] | None = None,
        on_diagnostic: Callable[[str], None] | None = None,
    ) -> None:
        """Configure finite capture state and a nonblocking exchange sink.

        The bypass callback receives a sanitized app label, exact selected host,
        and fixed failure category. It never receives raw TLS errors or payloads.
        """
        if not domains or max_body_bytes < 1 or max_active_flows < 1:
            raise ValueError("capture needs domains and positive memory limits")
        self._domains = frozenset(validate_domains(domains))
        self._sink = sink
        self._max_body_bytes = max_body_bytes
        self._max_active_flows = max_active_flows
        self._upstream_ca_file = upstream_ca_file
        self._on_bypass = on_bypass
        self._on_diagnostic = on_diagnostic
        self._app_bypasses: set[tuple[tuple[int, str], str]] = set()
        self._host_bypasses: set[str] = set()
        self._captures: dict[str, _Capture] = {}
        self._master: DumpMaster | None = None
        self._stopping = False
        self._transport_failures: set[str] = set()
        self.dropped_exchanges = 0

    async def serve(self, *, ca_directory: Path, ready: Callable[[], None]) -> None:
        """Inspect selected hosts through the operating system's local redirector.

        The redirector preserves each connection's original destination and excludes
        this process, including provider forwarding and cloud uploads. No hostname
        resolution or system DNS changes are performed by Capture.

        Raises:
            RuntimeError: Startup, cleanup, or the native capture backend fails.
        """
        opts = _capture_options(tuple(self._domains), ca_directory)
        master = DumpMaster(opts, with_termlog=False, with_dumper=False)
        self._master = master
        # Embedded callers own startup errors and cleanup, rather than sys.exit.
        errorcheck = master.addons.get("errorcheck")
        if isinstance(errorcheck, ErrorCheck):
            errorcheck.finish()
            master.addons.remove(errorcheck)
        master.addons.remove(master.addons.get("tlsconfig"))
        master.addons.add(_CaptureTlsConfig(self._domains))
        nextlayer = master.addons.get("nextlayer")
        master.addons.remove(nextlayer)
        master.addons.add(self, nextlayer)
        opts.update(
            keep_host_header=True,
            keep_alt_svc_header=True,
            connection_strategy="lazy",
            http3=False,
            block_global=False,
        )
        if self._upstream_ca_file is not None:
            opts.update(ssl_verify_upstream_trusted_ca=str(self._upstream_ca_file))
        proxyserver = master.addons.get("proxyserver")
        assert isinstance(proxyserver, Proxyserver)
        watchdog: CaptureWatchdog | None = None
        try:
            if not await proxyserver.setup_servers():
                raise RuntimeError(
                    "Capture could not start the network extension. Approve Mitmproxy "
                    "Redirector in macOS System Settings, then run exp capture again."
                )
            watchdog = await start_capture_watchdog(proxyserver)
            if not master.should_exit.is_set():
                await master.running()
                if not master.should_exit.is_set():
                    if watchdog is not None:
                        await watchdog.activate()
                    ready()
                    await _wait_for_capture_stop(proxyserver, master.should_exit, watchdog)
        finally:
            self._stopping = True
            try:
                # Stop redirection before releasing the proxy and upload lifetime.
                # master.done() alone does not stop mitmproxy's local redirector.
                if watchdog is None:
                    await stop_capture_servers(proxyserver)
                else:
                    try:
                        await stop_capture_servers(proxyserver, watchdog)
                    finally:
                        await watchdog.close()
            finally:
                try:
                    await master.done()
                finally:
                    self._master = None
                    self._finish_pending()
                    self._transport_failures.clear()

    def _finish_pending(self) -> None:
        """Account for interrupted captures even when backend cleanup reports failure."""
        for flow_id, capture in tuple(self._captures.items()):
            if capture.websocket:
                for request in capture.websocket_requests:
                    self._submit(
                        CapturedExchange(
                            protocol="responses",
                            host=capture.host,
                            path=capture.path,
                            started_ns=request.started_ns,
                            ended_ns=time.time_ns(),
                            request=request.body,
                            response=b"",
                            status=0,
                            failed=True,
                        )
                    )
            else:
                capture.failed = True
                capture.request_done = capture.response_done = True
                self._finish(flow_id, capture)
        self._captures.clear()

    def shutdown(self) -> None:
        """Request bounded server shutdown without blocking the caller's event loop."""
        self._stopping = True
        if self._master is not None:
            self._master.shutdown()

    def client_connected(self, client: connection.Client) -> None:
        """Contain native client write failures within their own connection."""
        self._diagnostic("connection_opened", client)
        self._guard_transport(client, client)

    def server_connected(self, data: server_hooks.ServerConnectionHookData) -> None:
        """Apply the same connection lifetime to native upstream UDP writers."""
        self._diagnostic("upstream_connected", data.client)
        self._guard_transport(data.client, data.server)

    def server_connect_error(self, data: server_hooks.ServerConnectionHookData) -> None:
        """Expose forwarding failure without printing upstream error contents."""
        self._diagnostic("upstream_connection_failed", data.client)

    def _guard_transport(self, client: connection.Client, target: connection.Connection) -> None:
        """Install the native writer guard only in this Capture master's transports."""
        if self._master is None:
            return
        proxyserver = self._master.addons.get("proxyserver")
        assert isinstance(proxyserver, Proxyserver)
        guard_native_writer(proxyserver, client, target, self._record_transport_failure)

    def _record_transport_failure(self, client_id: str) -> None:
        """Attribute failures only while their client remains active."""
        if self._master is None or self._stopping:
            return
        proxyserver = self._master.addons.get("proxyserver")
        assert isinstance(proxyserver, Proxyserver)
        handler = proxyserver.connections.get(client_id)
        if handler is not None and handler.client.timestamp_end is None:
            if client_id not in self._transport_failures:
                self._diagnostic("native_transport_failed", handler.client)
            self._transport_failures.add(client_id)

    def client_disconnected(self, client: connection.Client) -> None:
        """Release failure attribution with the owning connection's lifetime."""
        self._diagnostic("connection_closed", client)
        self._transport_failures.discard(client.id)

    def _client_identity(self, client: connection.Client) -> tuple[int, str] | None:
        """Read native process identity from the owned reader without probing the OS."""
        if self._master is None:
            return None
        proxyserver = self._master.addons.get("proxyserver")
        assert isinstance(proxyserver, Proxyserver)
        handler = proxyserver.connections.get(client.id)
        transport = handler.transports.get(client) if handler is not None else None
        if transport is None or not isinstance(transport.reader, mitmproxy_rs.Stream):
            return None
        pid = transport.reader.get_extra_info("pid")
        name = transport.reader.get_extra_info("process_name")
        if not isinstance(pid, int) or pid <= 0 or not isinstance(name, str) or not name:
            return None
        return pid, name

    def tls_clienthello(self, data: tls.ClientHelloData) -> None:
        """Preserve original encrypted traffic for a client that rejected inspection."""
        host = (data.client_hello.sni or "").lower().rstrip(".")
        if host not in self._domains:
            return
        self._diagnostic("tls_client_hello", data.context.client)
        identity = self._client_identity(data.context.client)
        if host in self._host_bypasses or (
            identity is not None and (identity, host) in self._app_bypasses
        ):
            data.ignore_connection = True
            self._diagnostic("tls_passthrough", data.context.client)

    def _bypass_client(
        self,
        client: connection.Client,
        host: str,
        reason: CaptureBypassReason,
    ) -> None:
        """Remember rejected app/host pairs for this run, with bounded host fallback."""
        if host in self._host_bypasses:
            return
        identity = self._client_identity(client)
        if identity is not None and (identity, host) in self._app_bypasses:
            return
        if identity is None or len(self._app_bypasses) >= _MAX_TLS_BYPASSES:
            self._host_bypasses.add(host)
            application = "All apps"
        else:
            self._app_bypasses.add((identity, host))
            application = (
                "".join(char for char in Path(identity[1]).name if char.isprintable())[:80] or "App"
            )
        if self._on_bypass is not None:
            self._on_bypass(application, host, reason)

    def tls_failed_client(self, data: tls.TlsData) -> None:
        """Bypass explicit trust rejection, keeping ambiguous disconnects connection-local.

        The failed handshake cannot be repaired or replayed as encrypted pass-through.
        Future connections from that app to that host retain original TLS instead.
        A missing native identity or full app table uses a visible host-wide bypass.
        No certificate verification is disabled, and bypasses last only for this run.
        """
        if self._stopping or data.conn is not data.context.client:
            return
        host = (data.conn.sni or "").lower().rstrip(".")
        if host not in self._domains:
            return
        identity = self._client_identity(data.context.client)
        if host in self._host_bypasses or (
            identity is not None and (identity, host) in self._app_bypasses
        ):
            return
        error = (data.conn.error or "").lower()
        if any(
            alert in error for alert in ("unknown ca", "bad certificate", "certificate unknown")
        ):
            self._diagnostic("tls_client_failed: certificate", data.context.client)
            self._bypass_client(data.context.client, host, "certificate")
            return
        # EOF is not a trust decision. Normal cancellations and native transport
        # losses can produce the same callback, even for a working trusted app.
        self._diagnostic(
            "tls_client_failed: disconnected"
            if error.startswith("the client disconnected during the handshake.")
            else "tls_client_failed: negotiation",
            data.context.client,
        )

    def _diagnostic(self, event: str, client: connection.Client) -> None:
        """Emit only internal event labels, selected hosts, and opaque connection IDs.

        Diagnostics cannot disrupt forwarding, even when their output sink fails.
        No raw errors, URLs, headers, process arguments, or message bodies are emitted.
        """
        host = (client.sni or "").lower().rstrip(".")
        if self._on_diagnostic is None:
            return
        if host not in self._domains:
            # DNS has no TLS name. Observe only forwarding lifecycle, never queries.
            if client.transport_protocol != "udp" or self._master is None:
                return
            proxyserver = self._master.addons.get("proxyserver")
            assert isinstance(proxyserver, Proxyserver)
            handler = proxyserver.connections.get(client.id)
            address = handler.layer.context.server.address if handler is not None else None
            if address is None or address[1] != 53:
                return
            host = "DNS forwarding"
        try:
            self._on_diagnostic(f"{event} · {host} · connection {client.id[:8]}")
        except Exception:  # noqa: BLE001
            pass

    def tls_established_client(self, data: tls.TlsData) -> None:
        """Record successful client trust and TLS negotiation without certificate content."""
        self._diagnostic("tls_client_ready", data.context.client)

    def tls_established_server(self, data: tls.TlsData) -> None:
        """Record verified upstream TLS negotiation without peer certificate details."""
        self._diagnostic("tls_upstream_ready", data.context.client)

    def tls_failed_server(self, data: tls.TlsData) -> None:
        """Distinguish upstream TLS failure from client trust without copying raw errors."""
        self._diagnostic("tls_upstream_failed", data.context.client)

    def next_layer(self, nextlayer: layer.NextLayer) -> None:
        """Pass UDP, including QUIC and DNS, through without decrypting or recording it.

        This hook runs before mitmproxy's protocol selection. HTTPS over TCP is
        selected by allow_hosts before its TLS layer is created; HTTP/3 is outside
        this capture engine's supported protocols and must remain usable unchanged.
        """
        if nextlayer.context.client.transport_protocol == "udp":
            nextlayer.layer = layers.UDPLayer(nextlayer.context, ignore=True)

    def tcp_message(self, flow: tcp.TCPFlow) -> None:
        """Release opaque TCP history while mitmproxy forwards the current chunk unchanged."""
        flow.messages.clear()

    async def requestheaders(self, flow: http.HTTPFlow) -> None:
        """Tee supported HTTPS requests without rewriting their original destination."""
        flow.request.stream = True
        try:
            host = (
                (urlsplit(f"//{flow.request.host_header or ''}").hostname or "").lower().rstrip(".")
            )
        except ValueError:
            return
        if (
            host not in self._domains
            or flow.request.scheme != "https"
            or host != (flow.client_conn.sni or "").lower().rstrip(".")
        ):
            return
        protocol = capture_protocol(flow.request.method, flow.request.path)
        self._diagnostic(f"http_request: {protocol or 'unsupported endpoint'}", flow.client_conn)
        if protocol is None:
            return
        if flow.request.method.upper() == "GET":
            # An idle upgraded connection owns no request buffers or capture slot.
            flow.metadata[_WEBSOCKET_CAPTURE] = True
            return
        if len(self._captures) >= self._max_active_flows:
            self.dropped_exchanges += 1
            self._diagnostic("capture_dropped: active request limit", flow.client_conn)
            return
        capture = _Capture(
            protocol=protocol,
            host=host,
            path=flow.request.path.partition("?")[0],
            started_ns=time.time_ns(),
            request=_Body(self._max_body_bytes),
            response=_Body(self._max_body_bytes),
            request_encoding=flow.request.headers.get("content-encoding", ""),
            websocket=flow.request.method.upper() == "GET",
        )
        self._captures[flow.id] = capture
        self._diagnostic(f"capture_started · trace {capture.trace_id}", flow.client_conn)
        flow.request.stream = capture.request.tee

    def request(self, flow: http.HTTPFlow) -> None:
        """Mark request completion, including servers replying before upload finishes."""
        capture = self._captures.get(flow.id)
        if capture is not None:
            capture.request_done = True
            self._finish(flow.id, capture)

    def responseheaders(self, flow: http.HTTPFlow) -> None:
        """Stream every response and attach a bounded tee only to selected flows."""
        if flow.response is None:
            return
        flow.response.stream = True
        if flow.metadata.get(_WEBSOCKET_CAPTURE) is True:
            self._diagnostic(
                "http_response: 101"
                if flow.response.status_code == 101
                else f"websocket_upgrade_failed: {flow.response.status_code}",
                flow.client_conn,
            )
            return
        capture = self._captures.get(flow.id)
        if capture is not None:
            if capture.websocket and flow.response.status_code != 101:
                self._captures.pop(flow.id, None)
                self._diagnostic(
                    f"websocket_upgrade_failed: {flow.response.status_code}", flow.client_conn
                )
                return
            capture.status = flow.response.status_code
            capture.response_encoding = flow.response.headers.get("content-encoding", "")
            capture.response_content_type = flow.response.headers.get("content-type", "")
            capture.websocket = flow.response.status_code == 101
            self._diagnostic(f"http_response: {capture.status}", flow.client_conn)
            if not capture.websocket:
                flow.response.stream = capture.response.tee

    def response(self, flow: http.HTTPFlow) -> None:
        """Queue completed captures without waiting for normalization or cloud delivery."""
        capture = self._captures.get(flow.id)
        if capture is not None:
            capture.response_done = True
            self._finish(flow.id, capture)

    def error(self, flow: http.HTTPFlow) -> None:
        """Capture available response evidence without copying transport error secrets."""
        capture = self._captures.get(flow.id)
        if capture is not None:
            if capture.websocket and capture.status != 101:
                self._captures.pop(flow.id, None)
                self._diagnostic("websocket_upgrade_failed", flow.client_conn)
                return
            capture.failed = True
            capture.request_done = True
            capture.response_done = True
            self._diagnostic(f"http_flow_failed · trace {capture.trace_id}", flow.client_conn)
            self._finish(flow.id, capture)

    def websocket_message(self, flow: http.HTTPFlow) -> None:
        """Copy Responses request/completion messages and discard retained flow history."""
        websocket = flow.websocket
        if websocket is None or not websocket.messages:
            return
        message = websocket.messages[-1]
        websocket.messages.clear()
        if (
            flow.metadata.get(_WEBSOCKET_CAPTURE) is not True
            or flow.response is None
            or flow.response.status_code != 101
        ):
            return
        capture = self._captures.get(flow.id)
        if len(message.content) > self._max_body_bytes:
            # An unparsed frame may contain a request or terminal event. Its
            # ownership is unknowable within the body cap, so never pair later
            # responses on this socket with earlier buffered requests.
            self._captures.pop(flow.id, None)
            pending = len(capture.websocket_requests) if capture is not None else 0
            self.dropped_exchanges += max(1, pending + int(message.from_client))
            flow.metadata[_WEBSOCKET_CAPTURE] = False
            flow.metadata.pop(_WEBSOCKET_SKIPPED, None)
            self._diagnostic("capture_disabled: oversized WebSocket frame", flow.client_conn)
            return
        try:
            event = json.loads(message.content)
        except (ValueError, RecursionError):
            return
        if not isinstance(event, dict):
            return
        if not isinstance(event.get("type"), str):
            return
        skipped = int(flow.metadata.get(_WEBSOCKET_SKIPPED, 0))
        if message.from_client and event.get("type") == "response.create":
            if skipped or (capture is None and len(self._captures) >= self._max_active_flows):
                self._skip_websocket_request(flow)
                self._diagnostic("capture_dropped: active request limit", flow.client_conn)
                return
            if capture is None:
                capture = _Capture(
                    protocol="responses",
                    host=(flow.client_conn.sni or "").lower().rstrip("."),
                    path=flow.request.path.partition("?")[0],
                    started_ns=time.time_ns(),
                    request=_Body(self._max_body_bytes),
                    response=_Body(self._max_body_bytes),
                    request_encoding="",
                    websocket=True,
                    status=101,
                )
                self._captures[flow.id] = capture
            pending_bytes = sum(len(request.body) for request in capture.websocket_requests)
            if len(capture.websocket_requests) >= 8 or (
                pending_bytes + len(message.content) > self._max_body_bytes
            ):
                self._skip_websocket_request(flow)
                return
            stream_id = event.get("stream_id")
            capture.websocket_requests.append(
                _WebsocketRequest(
                    message.content,
                    time.time_ns(),
                    stream_id=stream_id if isinstance(stream_id, str) else None,
                )
            )
            self._diagnostic(
                f"websocket_request_started · trace {capture.websocket_requests[-1].trace_id}",
                flow.client_conn,
            )
        elif not message.from_client:
            if event.get("type") == "error":
                # Request errors have no response object or reliable response ID.
                # Discard ambiguous buffers and consume this terminal outcome.
                self._skip_websocket_request(flow, new_request=False)
                outstanding = int(flow.metadata.get(_WEBSOCKET_SKIPPED, 0))
                flow.metadata[_WEBSOCKET_SKIPPED] = max(0, outstanding - 1)
                return
            response = event.get("response")
            if not isinstance(response, dict):
                return
            response_id = response.get("id")
            if not isinstance(response_id, str):
                return
            terminal = event.get("type") in {
                "response.completed",
                "response.failed",
                "response.incomplete",
            }
            if capture is None:
                if terminal and skipped:
                    flow.metadata[_WEBSOCKET_SKIPPED] = skipped - 1
                return
            stream_id = event.get("stream_id")
            requests = [
                request
                for request in capture.websocket_requests
                if request.stream_id == (stream_id if isinstance(stream_id, str) else None)
            ]
            if event.get("type") == "response.created" and isinstance(response_id, str):
                for request in requests:
                    if request.response_id is None:
                        request.response_id = response_id
                        break
            if terminal:
                for request in requests:
                    if request.response_id == response_id or (
                        request.response_id is None and len(requests) == 1 and not skipped
                    ):
                        capture.websocket_requests.remove(request)
                        if not capture.websocket_requests:
                            self._captures.pop(flow.id, None)
                        self._diagnostic(
                            f"websocket_request_completed · trace {request.trace_id}"
                            f" · {event['type']}",
                            flow.client_conn,
                        )
                        self._submit(
                            CapturedExchange(
                                protocol="responses",
                                host=capture.host,
                                path=capture.path,
                                started_ns=request.started_ns,
                                ended_ns=time.time_ns(),
                                request=request.body,
                                response=json.dumps(response, separators=(",", ":")).encode(),
                                status=200,
                                failed=event.get("type") != "response.completed",
                                trace_id=request.trace_id,
                            )
                        )
                        break
                else:
                    if skipped:
                        flow.metadata[_WEBSOCKET_SKIPPED] = skipped - 1

    def _skip_websocket_request(self, flow: http.HTTPFlow, *, new_request: bool = True) -> None:
        """Drain an ambiguous response group without attributing skipped responses.

        Once a request is skipped, created events cannot safely identify earlier
        requests whose IDs have not arrived. Release the entire buffered group
        and wait for all outstanding terminal events before collecting again.

        Args:
            flow: The upgraded flow whose capture ownership became ambiguous.
            new_request: Whether this event adds one newly skipped client request.
        """
        capture = self._captures.pop(flow.id, None)
        skipped = int(new_request) + (len(capture.websocket_requests) if capture is not None else 0)
        flow.metadata[_WEBSOCKET_SKIPPED] = int(flow.metadata.get(_WEBSOCKET_SKIPPED, 0)) + skipped
        self.dropped_exchanges += skipped

    def websocket_end(self, flow: http.HTTPFlow) -> None:
        """Retain failed request evidence for calls interrupted before completion."""
        capture = self._captures.pop(flow.id, None)
        if capture is not None:
            for request in capture.websocket_requests:
                self._submit(
                    CapturedExchange(
                        protocol="responses",
                        host=capture.host,
                        path=capture.path,
                        started_ns=request.started_ns,
                        ended_ns=time.time_ns(),
                        request=request.body,
                        response=b"",
                        status=0,
                        failed=True,
                        trace_id=request.trace_id,
                    )
                )

    def _finish(self, flow_id: str, capture: _Capture) -> None:
        """Release each completed HTTP capture exactly once into the bounded sink."""
        if capture.websocket or not (capture.request_done and capture.response_done):
            return
        self._captures.pop(flow_id, None)
        if capture.request.overflow or capture.response.overflow:
            self.dropped_exchanges += 1
            return
        self._submit(
            CapturedExchange(
                protocol=capture.protocol,
                host=capture.host,
                path=capture.path,
                started_ns=capture.started_ns,
                ended_ns=time.time_ns(),
                request=bytes(capture.request.data),
                response=bytes(capture.response.data),
                status=capture.status,
                request_encoding=capture.request_encoding,
                response_encoding=capture.response_encoding,
                response_content_type=capture.response_content_type,
                failed=capture.failed,
                trace_id=capture.trace_id,
            )
        )

    def _submit(self, exchange: CapturedExchange) -> None:
        """Isolate capture sinks from the live inference path, including sink bugs."""
        try:
            self._sink(exchange)
        except Exception:  # noqa: BLE001
            self.dropped_exchanges += 1
            logger.warning("Capture queue rejected an exchange; inference is unaffected")


async def _wait_for_capture_stop(
    proxyserver: Proxyserver, stop: asyncio.Event, watchdog: CaptureWatchdog | None = None
) -> None:
    """Stop on a requested shutdown, lost backend, or failed independent watchdog."""
    closed = capture_server_closed(proxyserver)
    stop_task = asyncio.create_task(stop.wait())
    backend_task = asyncio.ensure_future(closed) if closed is not None else None
    watchdog_task = asyncio.create_task(watchdog.wait_failed()) if watchdog is not None else None
    tasks = [task for task in (stop_task, backend_task, watchdog_task) if task is not None]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if watchdog_task in done and not stop.is_set():
            assert watchdog_task is not None
            watchdog_task.result()
        if backend_task in done and not stop.is_set():
            assert backend_task is not None
            cause = None if backend_task.cancelled() else backend_task.exception()
            raise RuntimeError(
                "Capture network backend stopped unexpectedly. Run exp capture again."
            ) from cause
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _capture_options(domains: tuple[str, ...], ca_directory: Path) -> options.Options:
    """Select provider TLS names and start inactive until the independent watchdog is ready."""
    return options.Options(
        confdir=str(ca_directory),
        mode=["local:0,!0"],
        allow_hosts=[rf"^{re.escape(domain)}\.?:[0-9]+$" for domain in domains],
        show_ignored_hosts=False,
        ssl_insecure=False,
        upstream_cert=False,
    )
