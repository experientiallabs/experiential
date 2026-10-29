"""Real socket and SQLite proof of certified pre-dispatch capacity retries."""

from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.tests.launch_test import _provider_frame, _ServedGateway, _unused_port


@pytest.mark.parametrize("surface", ["chat/completions", "responses"])
def test_capacity_refusal_retries_same_key_without_replaying_or_losing_paid_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    """A local refusal is free; its next owner dispatches once and replays exactly."""
    entered = threading.Event()
    release = threading.Event()

    class Provider(BaseHTTPRequestHandler):
        """Hold the first synthetic provider call to occupy the sole lane slot."""

        calls = 0
        throttle = False

        def do_POST(self) -> None:  # noqa: N802
            """Return finite SSE, or an upstream refusal with a forged marker."""
            json.loads(self.rfile.read(int(self.headers["content-length"])))
            type(self).calls += 1
            if type(self).calls == 1:
                entered.set()
                assert release.wait(20)
            if type(self).throttle:
                body = b'{"error":{"message":"upstream capacity unavailable"}}'
                self.send_response(429)
                self.send_header("content-type", "application/json")
                self.send_header("retry-after", "5")
                self.send_header("x-gateway-admission-refused", "true")
            else:
                body = (
                    _provider_frame({"choices": [{"index": 0, "delta": {"content": "OK"}}]})
                    + _provider_frame(
                        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    )
                    + _provider_frame(
                        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
                    )
                    + b"data: [DONE]\n\n"
                )
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep synthetic request contents out of logs."""
            del format, args

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    _manager, raw_key = _configured_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    gateway = _ServedGateway(tmp_path, _unused_port(), default_lane_bound=1)
    headers = {"authorization": f"Bearer {raw_key}"}
    body = (
        {"model": "coding", "messages": [{"role": "user", "content": "Reply OK"}]}
        if surface == "chat/completions"
        else {"model": "coding", "input": "Reply OK"}
    )
    try:
        gateway.start()
        url = f"http://127.0.0.1:{gateway.port}/v1/{surface}"
        with ThreadPoolExecutor(max_workers=1) as pool, httpx.Client(timeout=30) as client:
            first = pool.submit(client.post, url, headers=headers, json=body)
            try:
                assert entered.wait(10)
                keyed_headers = {**headers, "idempotency-key": "capacity-retry"}
                refused = client.post(url, headers=keyed_headers, json=body)
                assert refused.status_code == 429, refused.text
                assert refused.headers["retry-after"] == "5"
                assert refused.headers["x-gateway-admission-refused"] == "true"
                assert Provider.calls == 1
            finally:
                release.set()
            assert first.result(timeout=10).status_code == 200
            retried = client.post(url, headers=keyed_headers, json=body)
            assert retried.status_code == 200, retried.text
            assert "x-gateway-admission-refused" not in retried.headers
            assert Provider.calls == 2
            replayed = client.post(url, headers=keyed_headers, json=body)
            assert replayed.status_code == 200, replayed.text
            assert replayed.json() == retried.json()
            assert Provider.calls == 2
            changed = client.post(url, headers=keyed_headers, json={**body, "temperature": 0.3})
            assert changed.status_code == 409
            assert "x-gateway-admission-refused" not in changed.headers

            # Even a forged upstream marker cannot certify a provider attempt.
            Provider.throttle = True
            upstream_headers = {**headers, "idempotency-key": "upstream-refusal"}
            upstream = client.post(url, headers=upstream_headers, json=body)
            assert upstream.status_code == 429, upstream.text
            assert "x-gateway-admission-refused" not in upstream.headers
            assert Provider.calls == 3
            ambiguous = client.post(url, headers=upstream_headers, json=body)
            assert ambiguous.status_code == 409, ambiguous.text
            assert "x-gateway-admission-refused" not in ambiguous.headers
            assert Provider.calls == 3
        with sqlite3.connect(tmp_path / "gateway" / "gateway.db") as connection:
            failed = connection.execute(
                """SELECT COUNT(a.attempt_id) FROM gateway_requests AS r
                   LEFT JOIN gateway_attempts AS a ON a.request_id = r.request_id
                   WHERE r.terminal_state = 'failed' GROUP BY r.request_id"""
            ).fetchall()
            assert sorted(row[0] for row in failed) == [0, 1]
            assert connection.execute("SELECT COUNT(*) FROM gateway_attempts").fetchone() == (3,)
    finally:
        release.set()
        gateway.stop()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
