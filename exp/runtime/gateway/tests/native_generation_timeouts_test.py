"""Real-socket proof of host-authorized waits and the unchanged request deadline."""

from __future__ import annotations

import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.embeddings_contracts import ServingRequest
from exp.runtime.gateway.generation_timeouts import GatewayGenerationTimeouts
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.sqlite.store import SQLiteGatewayStore
from exp.runtime.gateway.tests.launch_test import _configure_gateway, _unused_port, _wait_ready
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames

native = pytest.importorskip("exp_gateway_native")


class _DelayedProvider(BaseHTTPRequestHandler):
    """Send headers, wait before the first token, then pause again before finishing."""

    def log_message(self, format: str, *args: object) -> None:
        """Keep synthetic request logs out of test output."""

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler contract.
        """Exercise both progress timers with semantic output between the gaps."""
        self.rfile.read(int(self.headers["content-length"]))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        try:
            self.wfile.write(b": keepalive\n\n")
            self.wfile.flush()
            time.sleep(0.3)
            self.wfile.write(_content_chunk("first "))
            self.wfile.flush()
            time.sleep(0.3)
            self.wfile.write(_content_chunk("last") + _terminal_frames())
            self.wfile.flush()
        except OSError:
            # A short bound or request deadline is expected to close the socket.
            pass


@pytest.mark.parametrize(
    ("first_token", "progress", "deadline", "expected"),
    [
        (None, None, 3.0, "first token"),
        (600.0, 600.0, 3.0, "success"),
        (0.1, 1.0, 3.0, "first token"),
        (1.0, 0.1, 3.0, "stopped making progress"),
        (1.0, 1.0, 0.2, "deadline"),
    ],
)
def test_scoped_waits_cross_native_http_and_never_extend_total_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_token: float | None,
    progress: float | None,
    deadline: float,
    expected: str,
) -> None:
    """Scaled waits prove both timeout paths using real dispatch and settlement."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "synthetic-provider-key")
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _DelayedProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    manager, raw_key = _configure_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    components = load_gateway_components(tmp_path)
    authorize = SQLiteGatewayStore.authorize_request

    def authorize_with_waits(
        store: SQLiteGatewayStore,
        *,
        raw_key: str,
        alias: str,
        request: ServingRequest,
        deadline_monotonic: float,
        app_referer: str | None = None,
        app_title: str | None = None,
        client_ip: str | None = None,
    ) -> AuthorizationSnapshot:
        """Model a host policy after real key authentication and alias authorization."""
        snapshot = authorize(
            store,
            raw_key=raw_key,
            alias=alias,
            request=request,
            deadline_monotonic=deadline_monotonic,
            app_referer=app_referer,
            app_title=app_title,
            client_ip=client_ip,
        )
        if first_token is None or progress is None:
            return snapshot
        return snapshot.model_copy(
            update={
                "generation_timeouts": GatewayGenerationTimeouts(
                    first_token_base_seconds=first_token, progress_seconds=progress
                )
            }
        )

    monkeypatch.setattr(SQLiteGatewayStore, "authorize_request", authorize_with_waits)
    control = NativeControlPlane(components, request_timeout_seconds=deadline)
    port = _unused_port()
    shutdown = native.shutdown_handle()
    failures: list[Exception] = []

    def serve() -> None:
        """Preserve worker errors while allowing bounded test cleanup."""
        try:
            serve_native_gateway(
                control,
                host="127.0.0.1",
                port=port,
                time_to_first_token_seconds=0.1,
                time_to_first_byte_seconds_per_million_input_tokens=0,
                shutdown=shutdown,
            )
        except Exception as error:  # noqa: BLE001 - asserted on the test thread.
            failures.append(error)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        _wait_ready(port, worker)
        response = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            headers={"authorization": f"Bearer {raw_key}"},
            json={"model": "coding", "messages": [{"role": "user", "content": "hello"}]},
            timeout=5,
        )
        if expected == "success":
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == "first last"
        else:
            assert response.status_code >= 400, response.text
            assert expected in response.text.lower(), response.text
        with sqlite3.connect(manager.database_path) as connection:
            rows = connection.execute("select state from gateway_attempts").fetchall()
        assert rows
        assert all(row[0] == ("completed" if expected == "success" else "failed") for row in rows)
    finally:
        shutdown.request_shutdown()
        worker.join(timeout=10)
        components.write_ledger.close()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
    assert not failures
    assert not worker.is_alive()
