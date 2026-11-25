"""Host response extensions traverse the served native gateway on every completion shape."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.group_commit import SyncGroupCommitLedger
from exp.runtime.gateway.response_extensions import GatewayResponseExtensions
from exp.runtime.gateway.tests.launch_test import _ServedGateway, _unused_port
from exp.runtime.gateway.tests.native_json_object_test import (
    _configure_json_gateway,
    _provider_response,
)
from exp.runtime.models import registry

_EXTENSIONS = GatewayResponseExtensions(
    usage={"cost": 0.0016, "is_byok": False},
    body={"provider": "example"},
    headers=(("x-example-notice", "settled"),),
)


def _take(self: SyncGroupCommitLedger, request_id: str) -> GatewayResponseExtensions:
    """Hand every finalized request the same host extension."""
    del self, request_id
    return _EXTENSIONS


def _price(self: SyncGroupCommitLedger, **settlement: object) -> None:
    """Accept every quote, so terminal frames flush before the durable write."""
    del self, settlement


@pytest.fixture(params=["settle", "price"])
def gateway(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[str, dict[str, str]]]:
    """Serve one alias over a loopback provider with an extension-supplying ledger.

    ``settle`` hands the extension over on the finalizing settle reply;
    ``price`` makes the ledger a pricer, so it arrives on the quote reply.
    """
    monkeypatch.setattr(SyncGroupCommitLedger, "take_response_extensions", _take, raising=False)
    if request.param == "price":
        monkeypatch.setattr(SyncGroupCommitLedger, "price_attempt", _price, raising=False)
    provider_body = _provider_response("openai-compatible")

    class Provider(BaseHTTPRequestHandler):
        """Answer every dispatch with one finite, billed SSE completion."""

        def do_POST(self) -> None:  # noqa: N802
            """Drain the request and answer with real SSE frames."""
            self.rfile.read(int(self.headers["content-length"]))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(provider_body)))
            self.end_headers()
            self.wfile.write(provider_body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep synthetic traffic out of test logs."""
            del format, args

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    base_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    factory, _ = registry._HTTP_PROVIDERS["openai-compatible"]
    monkeypatch.setattr(
        registry,
        "_HTTP_PROVIDERS",
        {**registry._HTTP_PROVIDERS, "openai-compatible": (factory, base_url)},
    )
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    raw_key = _configure_json_gateway(tmp_path, "openai-compatible", base_url)
    served = _ServedGateway(tmp_path, _unused_port())
    try:
        served.start()
        yield f"http://127.0.0.1:{served.port}", {"Authorization": f"Bearer {raw_key}"}
    finally:
        served.stop()
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)


def _payloads(text: str) -> list[JsonObject]:
    """Decode every JSON ``data:`` line of one SSE body."""
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: {")
    ]


def _assert_stamped(holder: JsonObject) -> None:
    """The host's members sit beside the engine's own, which are untouched."""
    usage = holder["usage"]
    assert isinstance(usage, dict)
    assert usage["cost"] == 0.0016
    assert usage["is_byok"] is False
    assert holder["provider"] == "example"


_MESSAGES = [{"role": "user", "content": "Give the answer."}]


def test_buffered_chat_carries_usage_members_body_members_and_headers(
    gateway: tuple[str, dict[str, str]],
) -> None:
    """A buffered Chat answer is stamped, and its token counts stay the engine's."""
    base, headers = gateway
    response = httpx.post(
        f"{base}/v1/chat/completions",
        headers=headers,
        json={"model": "coding", "messages": _MESSAGES},
        timeout=10,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    _assert_stamped(body)
    assert body["usage"]["total_tokens"] > 0
    assert response.headers["x-example-notice"] == "settled"


def test_streamed_chat_stamps_only_the_final_usage_chunk(
    gateway: tuple[str, dict[str, str]],
) -> None:
    """Only the usage chunk changes; a live stream's headers precede settlement."""
    base, headers = gateway
    response = httpx.post(
        f"{base}/v1/chat/completions",
        headers=headers,
        json={
            "model": "coding",
            "messages": _MESSAGES,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        timeout=10,
    )
    assert response.status_code == 200, response.text
    chunks = _payloads(response.text)
    stamped = [chunk for chunk in chunks if "provider" in chunk]
    assert len(stamped) == 1
    _assert_stamped(stamped[0])
    assert "x-example-notice" not in response.headers


def test_messages_stamps_the_body_and_the_message_delta(
    gateway: tuple[str, dict[str, str]],
) -> None:
    """The Anthropic body and the stream's ``message_delta`` carry the members."""
    base, headers = gateway
    request = {"model": "coding", "max_tokens": 64, "messages": _MESSAGES}
    buffered = httpx.post(f"{base}/v1/messages", headers=headers, json=request, timeout=10)
    assert buffered.status_code == 200, buffered.text
    _assert_stamped(buffered.json())
    streamed = httpx.post(
        f"{base}/v1/messages", headers=headers, json={**request, "stream": True}, timeout=10
    )
    assert streamed.status_code == 200, streamed.text
    events = _payloads(streamed.text)
    deltas = [event for event in events if event.get("type") == "message_delta"]
    assert len(deltas) == 1
    _assert_stamped(deltas[0])
    start = next(event for event in events if event.get("type") == "message_start")
    assert "provider" not in start


def test_responses_stamps_the_envelope_and_the_terminal_event(
    gateway: tuple[str, dict[str, str]],
) -> None:
    """The Responses envelope carries the members, buffered and on ``response.completed``."""
    base, headers = gateway
    request = {"model": "coding", "input": "Give the answer."}
    buffered = httpx.post(f"{base}/v1/responses", headers=headers, json=request, timeout=10)
    assert buffered.status_code == 200, buffered.text
    _assert_stamped(buffered.json())
    streamed = httpx.post(
        f"{base}/v1/responses", headers=headers, json={**request, "stream": True}, timeout=10
    )
    assert streamed.status_code == 200, streamed.text
    completed = [
        event for event in _payloads(streamed.text) if event.get("type") == "response.completed"
    ]
    assert len(completed) == 1
    envelope = completed[0]["response"]
    assert isinstance(envelope, dict)
    _assert_stamped(envelope)


def test_a_keyed_replay_returns_the_same_stamped_bytes(
    gateway: tuple[str, dict[str, str]],
) -> None:
    """Retention follows the stamp, so a replay reproduces the extended answer exactly."""
    base, headers = gateway
    keyed = {**headers, "Idempotency-Key": "extension-replay-1"}
    request = {"model": "coding", "messages": _MESSAGES}
    first = httpx.post(f"{base}/v1/chat/completions", headers=keyed, json=request, timeout=10)
    second = httpx.post(f"{base}/v1/chat/completions", headers=keyed, json=request, timeout=10)
    assert first.status_code == second.status_code == 200, first.text
    assert first.content == second.content
    _assert_stamped(first.json())
