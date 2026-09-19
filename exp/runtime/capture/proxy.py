"""TLS pass-through capture with bounded copies and independently queued delivery."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from mitmproxy import http, options, tls
from mitmproxy.proxy.server_hooks import ServerConnectionHookData
from mitmproxy.tools.dump import DumpMaster

from exp.runtime.capture.normalization import CapturedExchange, CaptureProtocol, capture_protocol
from exp.runtime.capture.resolver import UpstreamResolver

logger = logging.getLogger(__name__)


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
    """One unstitched Responses request awaiting its protocol-level completion."""

    body: bytes
    started_ns: int
    response_id: str | None = None


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


class CaptureProxy:
    """Observe allowed model exchanges while forwarding original stream bytes."""

    def __init__(
        self,
        *,
        sink: Callable[[CapturedExchange], bool],
        domains: tuple[str, ...],
        resolver: UpstreamResolver,
        max_body_bytes: int = 8 * 1024 * 1024,
        max_active_flows: int = 16,
        upstream_port: int = 443,
        upstream_ca_file: Path | None = None,
    ) -> None:
        """Configure finite capture state and a nonblocking exchange sink."""
        if not domains or max_body_bytes < 1 or max_active_flows < 1:
            raise ValueError("capture needs domains and positive memory limits")
        self._domains = frozenset(domain.lower().rstrip(".") for domain in domains)
        self._resolver = resolver
        self._sink = sink
        self._max_body_bytes = max_body_bytes
        self._max_active_flows = max_active_flows
        self._upstream_port = upstream_port
        self._upstream_ca_file = upstream_ca_file
        self._captures: dict[str, _Capture] = {}
        self._master: DumpMaster | None = None
        self._ready: Callable[[], None] | None = None
        self.dropped_exchanges = 0

    async def serve(self, *, port: int, ca_directory: Path, ready: Callable[[], None]) -> None:
        """Serve an unprivileged loopback TLS port until shutdown is requested.

        The caller owns privileged relaying, trust installation, and host overrides.
        Upstream verification remains enabled; neither API credentials nor flow dumps
        are logged or persisted by mitmproxy.
        """
        self._ready = ready
        opts = options.Options(
            listen_host="127.0.0.1",
            listen_port=port,
            confdir=str(ca_directory),
            mode=["reverse:https://capture.invalid:443"],
            ssl_insecure=False,
            upstream_cert=False,
        )
        master = DumpMaster(opts, with_termlog=False, with_dumper=False)
        self._master = master
        master.addons.add(self)
        opts.update(
            keep_host_header=True,
            keep_alt_svc_header=False,
            connection_strategy="lazy",
            http3=False,
            block_global=True,
        )
        if self._upstream_ca_file is not None:
            opts.update(ssl_verify_upstream_trusted_ca=str(self._upstream_ca_file))
        try:
            await master.run()
        finally:
            self._master = None
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

    def running(self) -> None:
        """Notify orchestration only after mitmproxy has initialized its listeners."""
        if self._ready is not None:
            self._ready()

    def shutdown(self) -> None:
        """Request bounded server shutdown without blocking the caller's event loop."""
        if self._master is not None:
            self._master.shutdown()

    async def tls_clienthello(self, data: tls.ClientHelloData) -> None:
        """Bind the original allowlisted SNI to a directly resolved upstream."""
        host = (data.client_hello.sni or "").lower().rstrip(".")
        if host not in self._domains:
            data.context.server.error = "capture only forwards configured TLS hosts"
            return
        try:
            address = await self._resolver.resolve(host)
        except (ValueError, OSError):
            data.context.server.error = "capture upstream DNS resolution failed"
            return
        data.context.server.address = (address, self._upstream_port)
        data.context.server.sni = host
        data.establish_server_tls_first = False

    async def server_connect(self, data: ServerConnectionHookData) -> None:
        """Avoid hosts-file recursion while retaining hostname certificate validation."""
        host = (data.client.sni or "").lower().rstrip(".")
        if host not in self._domains:
            data.server.error = "capture only forwards configured TLS hosts"
            return
        try:
            address = await self._resolver.resolve(host)
        except (ValueError, OSError):
            data.server.error = "capture upstream DNS resolution failed"
            return
        data.server.address = (address, self._upstream_port)
        data.server.sni = host

    async def requestheaders(self, flow: http.HTTPFlow) -> None:
        """Route by DNS without changing authority, then tee only inference bodies."""
        try:
            host = (
                (urlsplit(f"//{flow.request.host_header or ''}").hostname or "").lower().rstrip(".")
            )
        except ValueError:
            host = ""
        if host not in self._domains or host != (flow.client_conn.sni or "").lower().rstrip("."):
            flow.response = http.Response.make(421, b"Capture host does not match TLS SNI")
            return
        try:
            address = await self._resolver.resolve(host)
        except (ValueError, OSError):
            flow.response = http.Response.make(502, b"Capture upstream DNS resolution failed")
            return
        authority = flow.request.host_header
        flow.request.host = address
        flow.request.port = self._upstream_port
        flow.request.host_header = authority
        flow.request.stream = True
        protocol = capture_protocol(flow.request.method, flow.request.path)
        if protocol is None:
            return
        if len(self._captures) >= self._max_active_flows:
            self.dropped_exchanges += 1
            return
        capture = _Capture(
            protocol=protocol,
            host=host,
            path=flow.request.path.partition("?")[0],
            started_ns=time.time_ns(),
            request=_Body(self._max_body_bytes),
            response=_Body(self._max_body_bytes),
            request_encoding=flow.request.headers.get("content-encoding", ""),
        )
        self._captures[flow.id] = capture
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
        capture = self._captures.get(flow.id)
        if capture is not None:
            capture.status = flow.response.status_code
            capture.response_encoding = flow.response.headers.get("content-encoding", "")
            capture.response_content_type = flow.response.headers.get("content-type", "")
            capture.websocket = flow.response.status_code == 101
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
            capture.failed = True
            capture.request_done = True
            capture.response_done = True
            self._finish(flow.id, capture)

    def websocket_message(self, flow: http.HTTPFlow) -> None:
        """Copy Responses request/completion messages and discard retained flow history."""
        websocket = flow.websocket
        if websocket is None or not websocket.messages:
            return
        message = websocket.messages[-1]
        websocket.messages.clear()
        capture = self._captures.get(flow.id)
        if capture is None or len(message.content) > self._max_body_bytes:
            if capture is not None:
                self.dropped_exchanges += 1
            return
        try:
            event = json.loads(message.content)
        except (ValueError, RecursionError):
            return
        if not isinstance(event, dict):
            return
        if not isinstance(event.get("type"), str):
            return
        if message.from_client and event.get("type") == "response.create":
            pending_bytes = sum(len(request.body) for request in capture.websocket_requests)
            if len(capture.websocket_requests) >= 8 or (
                pending_bytes + len(message.content) > self._max_body_bytes
            ):
                self.dropped_exchanges += 1
                return
            capture.websocket_requests.append(_WebsocketRequest(message.content, time.time_ns()))
        elif not message.from_client:
            response = event.get("response")
            if not isinstance(response, dict):
                return
            response_id = response.get("id")
            if event.get("type") == "response.created" and isinstance(response_id, str):
                for request in capture.websocket_requests:
                    if request.response_id is None:
                        request.response_id = response_id
                        break
            if event.get("type") in {
                "response.completed",
                "response.failed",
                "response.incomplete",
            }:
                for request in capture.websocket_requests:
                    if request.response_id == response_id or (
                        request.response_id is None and len(capture.websocket_requests) == 1
                    ):
                        capture.websocket_requests.remove(request)
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
                            )
                        )
                        break

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
            )
        )

    def _submit(self, exchange: CapturedExchange) -> None:
        """Isolate capture sinks from the live inference path, including sink bugs."""
        try:
            self._sink(exchange)
        except Exception:  # noqa: BLE001
            self.dropped_exchanges += 1
            logger.warning("Capture queue rejected an exchange; inference is unaffected")
