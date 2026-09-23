"""Caller-facing metadata capture stays byte-transparent and privacy gated."""

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable
from typing import cast

import pytest
from exp_gateway_native import CaptureCollector

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.capture_relay import CaptureRelay, redact_headers, redact_query


class _Collector:
    """One-shot admission fake recording metadata and the delivery thread."""

    def __init__(self, allowed: bool) -> None:
        """Configure the initial admission verdict and an empty receipt list."""
        self.allowed = allowed
        self.records: list[tuple[str, JsonObject, bytes, int]] = []

    def claim_relay(self, request_id: str) -> bool:
        """Allow only the first eligible claim for the expected request."""
        assert request_id == "request-original"
        allowed, self.allowed = self.allowed, False
        return allowed

    def finish_relay(self, request_id: str, metadata_json: str, body: bytes) -> bool:
        """Record the exact handed-off metadata, bytes and thread identity."""
        self.records.append((request_id, json.loads(metadata_json), body, threading.get_ident()))
        return True


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize("cap", [8, 4096])
def test_relay_is_transparent_bounded_redacted_and_off_loop(
    wire: bool, allowed: bool, cap: int
) -> None:
    """Existing wire settings do not change served bytes or the consent boundary."""
    collector = _Collector(allowed)
    wire_body = b'{"messages":[{"role":"user","content":"test"}]}'
    started: dict[str, object] = {
        "type": "http.response.start",
        "status": 200,
        "headers": [(b"x-request-id", b"request-original"), (b"set-cookie", b"private")],
    }
    ended: dict[str, object] = {"type": "http.response.body", "body": b'{"ok":true}'}
    sent: list[object] = []
    scope: dict[str, object] = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "query_string": b"api%5Fkey=private&session=one%20two",
        "headers": [(b"authorization", b"Bearer private"), (b"x-session-id", b"session-1")],
        "client": ("127.0.0.1", 12345),
    }

    async def receive() -> object:
        """Return the unchanged synthetic request body."""
        return {"type": "http.request", "body": wire_body}

    async def send(event: object) -> None:
        """Retain the exact outbound event object."""
        sent.append(event)

    async def app(
        scope: dict[str, object],
        receive: Callable[[], Awaitable[object]],
        send: Callable[[object], Awaitable[None]],
    ) -> None:
        """Echo fixed response events after consuming one request body."""
        assert await receive() == {"type": "http.request", "body": wire_body}
        await send(started)
        await send(ended)

    relay = CaptureRelay(
        app, cast("CaptureCollector", collector), wire_capture=wire, wire_maximum_bytes=cap
    )
    asyncio.run(relay(scope, receive, send))
    assert sent == [started, ended]
    assert sent[0] is started and sent[1] is ended
    if not allowed:
        assert collector.records == []
        return
    request_id, metadata, body, thread_id = collector.records[0]
    assert request_id == "request-original"
    assert thread_id != threading.get_ident()
    assert metadata["headers"] == [
        ["x-request-id", "request-original"],
        ["set-cookie", "<redacted>"],
    ]
    assert metadata["relay_completed"] is True
    assert metadata["client_disconnected"] is False
    timing = metadata["timing"]
    assert isinstance(timing, dict)
    first_ms, total_ms = timing["first_byte_ms"], timing["total_ms"]
    assert isinstance(first_ms, (int, float)) and isinstance(total_ms, (int, float))
    assert 0 <= first_ms <= total_ms
    if wire:
        request = metadata["wire_request"]
        assert isinstance(request, dict)
        assert request["headers"] == [
            ["authorization", "<redacted>"],
            ["x-session-id", "session-1"],
        ]
        assert request["query"] == "api%5Fkey=<redacted>&session=one%20two"
        assert request["body_bytes"] == len(wire_body)
        assert request["truncated"] is (cap < len(wire_body))
        assert body == (wire_body if cap >= len(wire_body) else b"")
    else:
        assert metadata["wire_request"] is None and body == b""


def test_existing_credential_redaction_keeps_encoded_nonsecret_pairs() -> None:
    """Encoded credentials and token/secret suffixes preserve the old privacy rules."""
    assert redact_query("Access+Token=s&x-api-key=s&param=a%2Bb&empty&signature=s") == (
        "Access+Token=<redacted>&x-api-key=<redacted>&param=a%2Bb&empty&signature=<redacted>"
    )
    assert redact_headers([(b"X-SESSION-TOKEN", b"s"), (b"X-TEST", b"ok")]) == [
        ["x-session-token", "<redacted>"],
        ["x-test", "ok"],
    ]


def test_asgi_disconnect_after_terminal_send_is_not_a_client_abort() -> None:
    """ASGI servers close the receive channel when the response body finishes."""
    collector = _Collector(True)
    front_receive: list[Callable[[], Awaitable[object]]] = []

    async def receive() -> object:
        """Emulate the server closing its receive channel after terminal send."""
        return {"type": "http.disconnect"}

    async def send(event: object) -> None:
        """Deliver the concurrent receive-channel close during terminal send."""
        if isinstance(event, dict) and event.get("type") == "http.response.body":
            await front_receive[0]()

    async def app(
        scope: dict[str, object],
        receive: Callable[[], Awaitable[object]],
        send: Callable[[object], Awaitable[None]],
    ) -> None:
        """Complete the response without a real client disconnect."""
        front_receive.append(receive)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"x-request-id", b"request-original")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    relay = CaptureRelay(app, cast("CaptureCollector", collector), wire_capture=True)
    asyncio.run(relay({"type": "http", "path": "/v1/chat/completions"}, receive, send))
    assert collector.records[0][1]["relay_completed"] is True
    assert collector.records[0][1]["client_disconnected"] is False


@pytest.mark.parametrize(
    "header",
    [
        b"password",
        b"x-password",
        b"x-proxy-authorization",
        b"x-custom-api-key",
        b"x-credentials",
        b"x-provider-auth",
    ],
)
def test_custom_credential_headers_are_not_persisted(header: bytes) -> None:
    """Credential-bearing header variants do not enter transport records."""
    assert redact_headers([(header, b"private")]) == [[header.decode(), "<redacted>"]]
