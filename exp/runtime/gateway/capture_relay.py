"""Preserve caller-facing capture facts at an outer ASGI relay.

The native collector still owns response content, eligibility, buffering and
delivery. This observer supplies only facts visible at the caller-facing hop.
Request bytes are retained in bounded chunks and handed off after the response;
JSON parsing and destination serialization stay on the native delivery worker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import unquote_plus

from exp.common.core.artifacts import JsonObject

if TYPE_CHECKING:
    from exp_gateway_native import CaptureCollector

_LOGGER = logging.getLogger(__name__)
_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "x-auth-token",
        "password",
        "passwd",
        "credential",
        "credentials",
    }
)
_QUERY = frozenset(
    {
        "key",
        "api_key",
        "apikey",
        "api-key",
        "token",
        "access_token",
        "id_token",
        "refresh_token",
        "auth",
        "authorization",
        "signature",
        "sig",
        "secret",
        "password",
        "passwd",
        "client_secret",
    }
)
_PATHS = frozenset({"/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/systemone"})
_Receive = Callable[[], Awaitable[object]]
_Send = Callable[[object], Awaitable[None]]


class _App(Protocol):
    """Minimal ASGI application contract for a byte-transparent observer."""

    async def __call__(self, scope: dict[str, object], receive: _Receive, send: _Send) -> None:
        """Serve a scope through the supplied receive and send channels."""
        ...


def _headers(raw: object) -> tuple[tuple[bytes, bytes], ...]:
    """Freeze ASGI header pairs without decoding or normalizing their values."""
    return tuple(cast("Sequence[tuple[bytes, bytes]]", raw or ()))


def redact_headers(headers: Sequence[tuple[bytes, bytes]]) -> list[list[str]]:
    """Keep ordered header names and redact the existing credential header set."""
    result: list[list[str]] = []
    for name, value in headers:
        key = name.decode("latin-1").lower()
        protected = key in _HEADERS or key.endswith(
            (
                "-secret",
                "-token",
                "-password",
                "-passwd",
                "-credential",
                "-credentials",
                "-authorization",
                "-authentication",
                "-api-key",
                "-auth",
            )
        )
        result.append([key, "<redacted>" if protected else value.decode("latin-1")])
    return result


def redact_query(query: str) -> str:
    """Redact credential query values without rewriting other encoded pairs."""
    parts: list[str] = []
    for pair in query.split("&"):
        name, sep, _ = pair.partition("=")
        key = re.sub(r"[\s-]+", "_", unquote_plus(name).strip().lower())
        protected = key in _QUERY or key.endswith(
            ("_token", "_secret", "_key", "signature", "_sig")
        )
        parts.append(f"{name}=<redacted>" if sep and protected else pair)
    return "&".join(parts)


class _Observation:
    """Bounded caller-facing facts retained for one in-flight request."""

    def __init__(self, scope: dict[str, object], wire: bool, maximum_bytes: int) -> None:
        """Start the request clock and retain the configured wire-body limit."""
        self.scope = scope
        self.wire = wire
        self.maximum_bytes = maximum_bytes
        self.received_at = datetime.now(UTC).isoformat()
        self.started = time.perf_counter()
        self.first_at: str | None = None
        self.last_at: str | None = None
        self.first_ms: float | None = None
        self.total_ms: float | None = None
        self.chunks: list[bytes] = []
        self.retained = 0
        self.total = 0
        self.truncated = False
        self.headers: tuple[tuple[bytes, bytes], ...] = ()
        self.request_id: str | None = None
        self.completed = False
        self.disconnected = False

    def keep(self, body: bytes) -> None:
        """Count all received bytes while retaining only a bounded prefix."""
        self.total += len(body)
        if not self.truncated:
            if self.retained + len(body) > self.maximum_bytes:
                self.truncated = True
            else:
                self.chunks.append(body)
                self.retained += len(body)

    def delivered(self, body: bytes) -> None:
        """Record first and last nonempty body delivery without copying content."""
        if not body:
            return
        now = datetime.now(UTC).isoformat()
        elapsed = (time.perf_counter() - self.started) * 1000
        if self.first_at is None:
            self.first_at, self.first_ms = now, elapsed
        self.last_at, self.total_ms = now, elapsed

    def finish(self, collector: CaptureCollector) -> None:
        """Join raw chunks and redact small metadata off the serving event loop."""
        assert self.request_id is not None
        wire: JsonObject | None = None
        if self.wire:
            query = self.scope.get("query_string", b"")
            client = self.scope.get("client")
            wire = {
                "method": str(self.scope.get("method", "")),
                "path": str(self.scope.get("path", "")),
                "query": redact_query(query.decode("latin-1") if isinstance(query, bytes) else ""),
                "headers": redact_headers(_headers(self.scope.get("headers"))),
                "body_bytes": self.total,
                "truncated": self.truncated,
                "received_at": self.received_at,
            }
            if isinstance(client, (tuple, list)) and client:
                wire["client"] = str(client[0])
        metadata: JsonObject = {
            "wire_request": wire,
            "headers": redact_headers(self.headers),
            "timing": {
                "received_at": self.received_at,
                "first_byte_at": self.first_at,
                "last_byte_at": self.last_at,
                "first_byte_ms": self.first_ms,
                "total_ms": self.total_ms,
            },
            "relay_completed": self.completed,
            "client_disconnected": self.disconnected,
        }
        collector.finish_relay(
            self.request_id, json.dumps(metadata, separators=(",", ":")), b"".join(self.chunks)
        )


class CaptureRelay:
    """Observe existing hosted wire/timing capture without owning hosted policy.

    Args:
        app: Outer caller-facing ASGI app, after presentation decorators.
        collector: Collector configured with relay_metadata=True.
        wire_capture: Existing wire capture kill switch, default false.
        wire_maximum_bytes: Retained request body bound, default 4 MiB.
    """

    def __init__(
        self,
        app: object,
        collector: CaptureCollector,
        *,
        wire_capture: bool = False,
        wire_maximum_bytes: int = 4_194_304,
    ) -> None:
        """Bind the native collector and the existing wire capture settings."""
        if not 1 <= wire_maximum_bytes <= 4_194_304:
            raise ValueError("wire capture limit must be between 1 byte and 4 MiB")
        self.app = cast("_App", app)
        self.collector = collector
        self.wire_capture = wire_capture
        self.wire_maximum_bytes = wire_maximum_bytes

    async def __call__(self, scope: dict[str, object], receive: _Receive, send: _Send) -> None:
        """Forward bytes unchanged and hand metadata off only for eligible admissions."""
        if scope.get("type") != "http" or scope.get("path") not in _PATHS:
            await self.app(scope, receive, send)
            return
        observed = _Observation(scope, self.wire_capture, self.wire_maximum_bytes)

        async def receiving() -> object:
            """Observe request chunks and early disconnects without changing events."""
            event = await receive()
            if isinstance(event, dict):
                if event.get("type") == "http.disconnect" and not observed.completed:
                    observed.disconnected = True
                elif event.get("type") == "http.request" and observed.wire:
                    body = event.get("body", b"")
                    if isinstance(body, bytes):
                        observed.keep(body)
            return event

        async def sending(event: object) -> None:
            """Correlate eligible responses and time successful caller delivery."""
            if isinstance(event, dict) and event.get("type") == "http.response.start":
                observed.headers = _headers(event.get("headers"))
                request_id = next(
                    (
                        value.decode("latin-1")
                        for name, value in observed.headers
                        if name.lower() == b"x-request-id"
                    ),
                    None,
                )
                if request_id and self.collector.claim_relay(request_id):
                    observed.request_id = request_id
                else:
                    observed.chunks.clear()
            terminal = (
                isinstance(event, dict)
                and event.get("type") == "http.response.body"
                and not event.get("more_body", False)
            )
            if terminal:
                # ASGI receive may return http.disconnect as soon as send
                # completes the response, even on a healthy connection.
                observed.completed = True
            try:
                await send(event)
            except BaseException:
                observed.completed = False
                observed.disconnected = True
                raise
            if isinstance(event, dict) and event.get("type") == "http.response.body":
                body = event.get("body", b"")
                if isinstance(body, bytes):
                    observed.delivered(body)

        try:
            await self.app(scope, receiving if self.wire_capture else receive, sending)
        finally:
            if observed.request_id is not None:
                observed.disconnected |= not observed.completed
                try:
                    await asyncio.to_thread(observed.finish, self.collector)
                except Exception:  # noqa: BLE001 - observational failure must not rewrite serving.
                    _LOGGER.warning("capture.relay_failed request_id=%s", observed.request_id)
