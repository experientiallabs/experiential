"""Served-gateway e2e for the Responses-over-WebSocket transport.

Mirrors the HTTP Responses e2e over the same loopback provider and asserts
transport parity: the WebSocket frames are exactly the HTTP SSE event
payloads, request-level failures arrive as wrapped in-band error frames,
prewarm frames complete without provider work, and a bad key rejects the
upgrade itself. The wire contract matches api.openai.com's
Responses-over-WebSocket surface (verified against live captures).
"""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
from openai import OpenAI
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.tests.launch_test import (
    _configure_gateway,
    _LoopbackProvider,
    _ServedGateway,
    _unused_port,
)

if TYPE_CHECKING:
    from websockets.sync.connection import Connection

pytest.importorskip("exp_gateway_native")

_PROMPT = "websocket-prompt-canary"

# Sanitized live capture of api.openai.com's Responses-over-WebSocket
# contract; the parity assertions below are driven from it.
_CONTRACT: JsonObject = json.loads(
    (
        Path(__file__).parent.parent / "testdata" / "responses_websocket_openai_contract.json"
    ).read_text()
)


def _collapse_deltas(event_types: object) -> list[str]:
    """Collapse consecutive repeated event types to compare stream grammar.

    Delta counts vary with provider chunking, so parity with the captured
    OpenAI stream is over the ordered event vocabulary, not delta counts.
    """
    assert isinstance(event_types, list)
    collapsed: list[str] = []
    for event_type in event_types:
        assert isinstance(event_type, str)
        if not collapsed or collapsed[-1] != event_type:
            collapsed.append(event_type)
    return collapsed


def _request_frame(**overrides: object) -> str:
    """Build one response.create request frame for the served alias."""
    frame: dict[str, object] = {
        "type": "response.create",
        "model": "coding",
        "input": _PROMPT,
        "store": False,
    }
    frame.update(overrides)
    return json.dumps(frame)


def _collect_stream(socket: Connection) -> list[JsonObject]:
    """Read frames until the terminal event of one response stream."""
    events: list[JsonObject] = []
    while True:
        frame = socket.recv(timeout=10)
        assert isinstance(frame, str), "server frames must be JSON text"
        event = json.loads(frame)
        events.append(event)
        if event["type"] in ("response.completed", "response.failed", "error"):
            return events


def test_websocket_transport_mirrors_the_http_event_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One connection serves prewarm, parity streams, reuse, and wrapped errors."""
    _LoopbackProvider.calls = 0
    provider_port = _unused_port()
    gateway_port = _unused_port()
    provider = ThreadingHTTPServer(("127.0.0.1", provider_port), _LoopbackProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    _manager, raw_key = _configure_gateway(
        tmp_path,
        base_url=f"http://127.0.0.1:{provider_port}/v1",
    )
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret-canary")
    gateway = _ServedGateway(tmp_path, gateway_port)
    gateway.start()
    ws_url = f"ws://127.0.0.1:{gateway_port}/v1/responses"
    headers = {"Authorization": f"Bearer {raw_key}"}
    try:
        # The HTTP transport's event stream is the parity baseline.
        with OpenAI(api_key=raw_key, base_url=f"http://127.0.0.1:{gateway_port}/v1") as client:
            http_events = [
                event.type
                for event in client.responses.create(model="coding", input=_PROMPT, stream=True)
            ]
        assert http_events[-1] == "response.completed"

        # The upgrade may carry an Idempotency-Key (some clients send blanket
        # headers); it names ONE operation, so the transport must not key
        # every frame of the connection to it.
        with connect(
            ws_url, additional_headers={**headers, "Idempotency-Key": "connection-op"}
        ) as socket:
            # Prewarm completes an empty response without touching the provider.
            provider_calls = _LoopbackProvider.calls
            socket.send(_request_frame(generate=False, input=[]))
            prewarm = _collect_stream(socket)
            assert [event["type"] for event in prewarm] == _CONTRACT["prewarm_event_types"]
            terminal = prewarm[-1]["response"]
            assert isinstance(terminal, dict)
            assert terminal["status"] == "completed"
            assert terminal["output"] == []
            assert type(terminal["created_at"]) is int
            assert type(terminal["completed_at"]) is int
            assert _LoopbackProvider.calls == provider_calls

            # A generating request streams the HTTP transport's exact events.
            socket.send(_request_frame())
            first = _collect_stream(socket)
            assert [event["type"] for event in first] == http_events
            assert [event["sequence_number"] for event in first] == list(range(len(first)))
            assert _collapse_deltas([event["type"] for event in first]) == _collapse_deltas(
                _CONTRACT["text_response_event_types"]
            )
            completed = first[-1]["response"]
            assert isinstance(completed, dict)
            output = completed["output"]
            assert isinstance(output, list)
            message = output[0]
            assert isinstance(message, dict)
            content = message["content"]
            assert isinstance(content, list)
            part = content[0]
            assert isinstance(part, dict)
            assert part["text"] == "hello world"

            # The connection serves sequential requests without reconnecting.
            socket.send(_request_frame())
            second = _collect_stream(socket)
            assert [event["type"] for event in second] == http_events

            # An explicit stream opt-out is the api.openai.com wrapped 400.
            socket.send(_request_frame(stream=False))
            rejected = _collect_stream(socket)
            assert len(rejected) == 1
            error = rejected[0]
            captured = _CONTRACT["stream_false_error_frame"]
            assert isinstance(captured, dict)
            assert error["type"] == captured["type"]
            assert error["status"] == captured["status"]
            details = error["error"]
            assert isinstance(details, dict)
            captured_error = captured["error"]
            assert isinstance(captured_error, dict)
            assert details["message"] == captured_error["message"]
            assert details["type"] == captured_error["type"]

            # An untagged frame is a request-level error, not a closed socket.
            socket.send(json.dumps({"model": "coding", "input": _PROMPT}))
            untagged = _collect_stream(socket)
            assert untagged[0]["type"] == "error"
            assert untagged[0]["status"] == 400

            # An unknown previous_response_id answers the api.openai.com
            # error code, the one string the Codex client auto-recovers on
            # by resending the full conversation.
            socket.send(_request_frame(previous_response_id="resp_unknown"))
            not_found = _collect_stream(socket)
            assert not_found[0]["type"] == "error"
            assert not_found[0]["status"] == 400
            missing = not_found[0]["error"]
            assert isinstance(missing, dict)
            assert missing["code"] == "previous_response_not_found"

            # The socket still serves after in-band errors.
            socket.send(_request_frame())
            third = _collect_stream(socket)
            assert [event["type"] for event in third] == http_events

        usage = httpx.get(f"http://127.0.0.1:{gateway_port}/usage.json", timeout=2).json()
        assert usage["totals"]["requests"] == 4
        assert _LoopbackProvider.calls == 4
    finally:
        gateway.stop()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
    assert not provider_thread.is_alive()

    durable = b"".join(
        path.read_bytes() for path in (tmp_path / "gateway").rglob("*") if path.is_file()
    )
    for forbidden in (_PROMPT, "hello world", raw_key, "provider-secret-canary"):
        assert forbidden.encode() not in durable


def test_websocket_upgrade_authenticates_before_accepting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad key rejects the handshake with 401 and a plain GET answers 426."""
    _LoopbackProvider.calls = 0
    provider_port = _unused_port()
    gateway_port = _unused_port()
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret-canary")
    _manager, _raw_key = _configure_gateway(
        tmp_path,
        base_url=f"http://127.0.0.1:{provider_port}/v1",
    )
    gateway = _ServedGateway(tmp_path, gateway_port)
    gateway.start()
    try:
        with pytest.raises(InvalidStatus) as rejection:
            connect(
                f"ws://127.0.0.1:{gateway_port}/v1/responses",
                additional_headers={"Authorization": "Bearer xpl_unknown"},
            )
        assert rejection.value.response.status_code == 401

        with pytest.raises(InvalidStatus) as missing:
            connect(f"ws://127.0.0.1:{gateway_port}/v1/responses")
        assert missing.value.response.status_code == 401

        # A plain GET fails closed with 426, the one status the Codex client
        # maps to its HTTP fallback.
        plain = httpx.get(
            f"http://127.0.0.1:{gateway_port}/v1/responses",
            headers={"Authorization": "Bearer xpl_unknown"},
            timeout=2,
        )
        assert plain.status_code == 426
        assert plain.json()["error"]["code"] == "upgrade_required"
    finally:
        gateway.stop()
