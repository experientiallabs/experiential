"""Gemini usage trailers cross real native JSON/SSE serving and settle exactly once."""

from __future__ import annotations

import json
import socket
import sqlite3
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.catalog_authority import upsert_connection, upsert_singleton_deployment
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests.launch_test import _ServedGateway, _unused_port
from exp.runtime.gateway.tests.native_json_object_test import _frame
from exp.runtime.models import registry


def _configure(root: Path) -> tuple[GatewayManagement, str]:
    """Seed explicit cached-input pricing on a synthetic Gemini deployment."""
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider",
        connection=ConnectionConfig(provider="gemini", api_key_env="TEST_PROVIDER_KEY"),
        replace=False,
    )
    normalized, snapshot, _ = upsert_singleton_deployment(
        root,
        deployment_alias="coding",
        connection_name="provider",
        provider_model="gemini-test",
        exact_model_id="gemini-test",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True, reports_cached_input_tokens=True
        ),
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=1_000_000_000,
            cached_input_nano_usd_per_million_tokens=100_000_000,
            output_nano_usd_per_million_tokens=2_000_000_000,
            reasoning_nano_usd_per_million_tokens=2_000_000_000,
        ),
        pricing_source="synthetic-test-prices",
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="rev",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="coding")
    return manager, manager.issue_key(identity_id="default", key_id="key").raw_key


def _chunks(placement: str) -> list[bytes]:
    """Place full or partial cumulative meters around the provider-declared STOP."""
    text: JsonObject = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
    stop: JsonObject = {"candidates": [{"finishReason": "STOP"}]}
    usage: JsonObject = {
        "usageMetadata": {
            "promptTokenCount": 7,
            "candidatesTokenCount": 2,
            "cachedContentTokenCount": 3,
        }
    }
    frames: list[JsonObject]
    match placement:
        case "before":
            frames = [text, usage, stop]
        case "with":
            frames = [text, {**stop, **usage}]
        case "bad-cache" | "reconciled-cache":
            frames = [text, usage, stop, {"usageMetadata": {"cachedContentTokenCount": 10}}]
            if placement == "reconciled-cache":
                frames.extend([{"usageMetadata": {}}, {"usageMetadata": {"promptTokenCount": 12}}])
        case "pending-cache-primary" | "pending-cache-valid" | "pending-cache-sparse-valid":
            frames = [
                text,
                usage,
                stop,
                {"usageMetadata": {"cachedContentTokenCount": 1000}},
                {"usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5}},
            ]
            if placement == "pending-cache-valid":
                frames.append(
                    {"usageMetadata": {"promptTokenCount": 12, "cachedContentTokenCount": 6}}
                )
            elif placement == "pending-cache-sparse-valid":
                frames.append({"usageMetadata": {"cachedContentTokenCount": 6}})
        case "empty-cache-output" | "empty-cache-output-reconciled":
            frames = [
                text,
                stop,
                {"usageMetadata": {}},
                {"usageMetadata": {"cachedContentTokenCount": 3}},
                {"usageMetadata": {"candidatesTokenCount": 2, "thoughtsTokenCount": 4}},
                {"usageMetadata": {}},
            ]
            if placement == "empty-cache-output-reconciled":
                frames.append({"usageMetadata": {"promptTokenCount": 7}})
        case "multiple-pending-cache" | "multiple-pending-cache-reversed":
            pending = [10, 1000]
            if placement == "multiple-pending-cache-reversed":
                pending.reverse()
            frames = [text, usage, stop]
            frames.extend(
                {"usageMetadata": {"cachedContentTokenCount": count}} for count in pending
            )
            frames.append({"usageMetadata": {"promptTokenCount": 12}})
        case "cache-output-first":
            frames = [
                text,
                stop,
                {"usageMetadata": {"cachedContentTokenCount": 3}},
                {"usageMetadata": {"candidatesTokenCount": 2, "thoughtsTokenCount": 4}},
                {"usageMetadata": {}},
                {"usageMetadata": {"promptTokenCount": 7}},
            ]
        case "cache-only" | "cache-first":
            frames = [text, stop, {"usageMetadata": {"cachedContentTokenCount": 3}}]
            if placement == "cache-first":
                frames.append({"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2}})
        case "none":
            frames = [text, stop]
        case "empty":
            frames = [text, stop, {"usageMetadata": {}}]
        case "partial":
            frames = [
                text,
                {"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2}},
                stop,
                {"usageMetadata": {"cachedContentTokenCount": 3}},
                {"usageMetadata": {}},
            ]
        case _:
            frames = [text, stop, usage]
    if placement == "late-content":
        frames.extend(
            [
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "late"},
                                    {"functionCall": {"name": "late", "args": {}}},
                                ]
                            },
                            "finishReason": "MAX_TOKENS",
                        }
                    ]
                },
                {"error": {"status": "UNAVAILABLE", "message": "late error"}},
            ]
        )
    chunks = [_frame(frame) for frame in frames]
    if placement == "malformed":
        chunks.append(b'data: {"usageMetadata":')
    return [b"".join(chunks)] if placement == "same-chunk" else chunks


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "placement",
    [
        "before",
        "with",
        "same-chunk",
        "later-chunk",
        "partial",
        "bad-cache",
        "reconciled-cache",
        "pending-cache-primary",
        "pending-cache-valid",
        "pending-cache-sparse-valid",
        "empty-cache-output",
        "empty-cache-output-reconciled",
        "multiple-pending-cache",
        "multiple-pending-cache-reversed",
        "cache-output-first",
        "cache-only",
        "cache-first",
        "none",
        "empty",
        "late-content",
        "malformed",
    ],
)
def test_native_gemini_usage_reaches_response_and_one_durable_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool, placement: str
) -> None:
    """The real worker drains metadata without reopening content or making another attempt."""
    chunks = _chunks(placement)
    expected_meter = {
        "reconciled-cache": (12, 2, 10, 7_000),
        "pending-cache-primary": (12, 5, 3, 19_300),
        "pending-cache-valid": (12, 5, 6, 16_600),
        "pending-cache-sparse-valid": (12, 5, 6, 16_600),
        "empty-cache-output-reconciled": (7, 6, 3, 16_300),
        "multiple-pending-cache": (12, 2, 10, 7_000),
        "multiple-pending-cache-reversed": (12, 2, 10, 7_000),
        "cache-output-first": (7, 6, 3, 16_300),
    }.get(placement, (7, 2, 3, 8_300))
    unknown_meter = placement in {"none", "empty", "cache-only", "empty-cache-output"}
    requests: list[str] = []
    settlements: list[JsonObject] = []

    class Provider(BaseHTTPRequestHandler):
        """Serve finite SSE with independently flushed Gemini frames."""

        def do_POST(self) -> None:  # noqa: N802
            """Record the SSE endpoint used even when the downstream requests JSON."""
            self.rfile.read(int(self.headers["content-length"]))
            requests.append(self.path)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(sum(map(len, chunks))))
            self.end_headers()
            for index, chunk in enumerate(chunks):
                if index:
                    time.sleep(0.015)
                self.wfile.write(chunk)
                self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:
            """Suppress synthetic content from logs."""
            del format, args

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    factory, _ = registry._HTTP_PROVIDERS["gemini"]
    monkeypatch.setattr(
        registry,
        "_HTTP_PROVIDERS",
        {
            **registry._HTTP_PROVIDERS,
            "gemini": (factory, f"http://127.0.0.1:{upstream.server_port}/v1"),
        },
    )
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-key")
    manager, key = _configure(tmp_path)
    gateway = _ServedGateway(tmp_path, _unused_port())
    settle = gateway.control_plane.settle

    def record_settle(argument: str) -> str:
        """Observe actual native settlement callbacks without replacing durable accounting."""
        settlements.append(json.loads(argument))
        return settle(argument)

    monkeypatch.setattr(gateway.control_plane, "settle", record_settle)
    try:
        gateway.start()
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "coding",
                "max_tokens": 16,
                "stream": stream,
                "stream_options": {"include_usage": True} if stream else None,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=10,
        )
        assert response.status_code == 200, response.text
        assert len(requests) == 1
        assert ":streamGenerateContent" in requests[0] and "alt=sse" in requests[0]
        if stream:
            envelopes = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            choices = [choice for envelope in envelopes for choice in envelope.get("choices", [])]
            assert "".join(choice["delta"].get("content", "") for choice in choices) == "hi"
            assert sum(choice.get("finish_reason") is not None for choice in choices) == 1
            assert (
                next(choice["finish_reason"] for choice in choices if choice.get("finish_reason"))
                == "stop"
            )
            assert not any(choice["delta"].get("tool_calls") for choice in choices)
            assert response.text.count("data: [DONE]") == 1
            usage = next((item["usage"] for item in envelopes if item.get("usage")), None)
        else:
            result = response.json()
            assert result["choices"][0]["message"]["content"] == "hi"
            assert result["choices"][0]["finish_reason"] == "stop"
            assert not result["choices"][0]["message"].get("tool_calls")
            usage = result.get("usage")
        if not unknown_meter:
            assert usage is not None
            assert usage["prompt_tokens"] == expected_meter[0]
            assert usage["completion_tokens"] == expected_meter[1]
            assert usage["prompt_tokens_details"]["cached_tokens"] == expected_meter[2]
    finally:
        gateway.stop()
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)
    assert len(settlements) == 1
    assert settlements[0]["outcome"] == "completed"
    assert settlements[0]["failure"] is None
    with sqlite3.connect(manager.database_path) as connection:
        rows = connection.execute(
            "SELECT state, input_tokens, output_tokens, cached_input_tokens, "
            "estimated_cost_nano_usd FROM gateway_attempts"
        ).fetchall()
    # A present empty proto object normalizes to zero but the existing settlement
    # contract treats a finished all-zero meter as unknown, never free service.
    expected = (None, None, None, None) if unknown_meter else expected_meter
    assert rows == [("completed", *expected)]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ending", ["phase-timeout", "deadline", "provider-break", "cancel"])
def test_native_gemini_drain_ends_bounded_and_settles_known_usage_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool, ending: str
) -> None:
    """The serving worker closes a stalled trailer phase before one durable settlement."""
    closed = threading.Event()
    sent = threading.Event()
    settled = threading.Event()
    settlements: list[JsonObject] = []
    calls: list[str] = []
    body = b"".join(_chunks("same-chunk"))

    class Provider(BaseHTTPRequestHandler):
        """Send finished content and usage, but leave the advertised body incomplete."""

        def do_POST(self) -> None:  # noqa: N802
            """Expose socket closure independently of settlement callback timing."""
            self.rfile.read(int(self.headers["content-length"]))
            calls.append(self.path)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body) + 100))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            sent.set()
            if ending == "provider-break":
                return
            self.connection.settimeout(3)
            try:
                if not self.connection.recv(1):
                    closed.set()
            except ConnectionResetError:
                closed.set()
            except TimeoutError:
                pass

        def log_message(self, format: str, *args: object) -> None:
            """Keep test transport details out of request logs."""
            del format, args

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    factory, _ = registry._HTTP_PROVIDERS["gemini"]
    monkeypatch.setattr(
        registry,
        "_HTTP_PROVIDERS",
        {
            **registry._HTTP_PROVIDERS,
            "gemini": (factory, f"http://127.0.0.1:{upstream.server_port}/v1"),
        },
    )
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-key")
    manager, key = _configure(tmp_path)
    gateway = _ServedGateway(tmp_path, _unused_port())
    admit = gateway.control_plane.admit
    settle = gateway.control_plane.settle
    if ending == "deadline":
        monkeypatch.setattr(gateway.control_plane, "_request_timeout_seconds", 0.8)

    def bounded_admit(argument: str) -> str:
        """Give the existing connection body-read contract a short test allowance."""
        admission = json.loads(admit(argument))
        for rung in admission.get("route", []):
            rung["timeout_seconds"] = 0.2 if ending == "phase-timeout" else 2.0
        return json.dumps(admission)

    def record_settle(argument: str) -> str:
        """Record exactly-once callback evidence while retaining normal accounting."""
        settlements.append(json.loads(argument))
        result = settle(argument)
        settled.set()
        return result

    monkeypatch.setattr(gateway.control_plane, "admit", bounded_admit)
    monkeypatch.setattr(gateway.control_plane, "settle", record_settle)
    client: socket.socket | None = None
    try:
        gateway.start()
        payload = {
            "model": "coding",
            "max_tokens": 16,
            "stream": stream,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        started = time.monotonic()
        if ending == "cancel":
            encoded = json.dumps(payload).encode()
            client = socket.create_connection(("127.0.0.1", gateway.port), timeout=3)
            client.sendall(
                (
                    "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                    f"Authorization: Bearer {key}\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(encoded)}\r\n\r\n"
                ).encode()
                + encoded
            )
            assert sent.wait(2)
            # Allow the provider bytes to be decoded before cancelling the read.
            time.sleep(0.05)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            client.close()
            client = None
        else:
            response = httpx.post(
                f"http://127.0.0.1:{gateway.port}/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=3,
            )
            assert response.status_code == 200, response.text
            if ending == "deadline" and stream:
                # Delivery shares the hard deadline: it cannot send a final
                # frame after expiry, but accounting still uses declared STOP.
                assert '"content":"hi"' in response.text.replace(" ", "")
            else:
                assert '"finish_reason":"stop"' in response.text.replace(" ", "")
        assert settled.wait(1), "data-plane settlement must precede the sweep"
        assert time.monotonic() - started < 1.8
        if ending != "provider-break":
            assert closed.wait(0.5), "metadata drain must close its upstream transport"
    finally:
        if client is not None:
            client.close()
        gateway.stop()
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)
    assert len(calls) == 1
    assert len(settlements) == 1
    assert settlements[0]["outcome"] == "completed"
    with sqlite3.connect(manager.database_path) as connection:
        rows = connection.execute(
            "SELECT state, input_tokens, output_tokens, cached_input_tokens, "
            "estimated_cost_nano_usd FROM gateway_attempts"
        ).fetchall()
    assert rows == [("completed", 7, 2, 3, 8_300)]
