"""Real native serving checks for deterministic and incremental guardrails."""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import exp_gateway_native
import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.guardrails.config import engine_from_document
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("streaming", [False, True])
def test_native_regex_redacts_before_delivery(
    tmp_path: Path, surface: str, streaming: bool
) -> None:
    """Every public surface redacts split secrets and streams a safe prefix early."""
    continue_output = threading.Event()
    upstream_finished = threading.Event()
    provider_inputs: list[str] = []

    class Provider(BaseHTTPRequestHandler):
        """Serve a split secret after an independently observable safe prefix."""

        def do_POST(self) -> None:  # noqa: N802 - HTTP handler protocol.
            """Capture the protected input and send one finite provider stream."""
            body = self.rfile.read(int(self.headers["Content-Length"]))
            provider_inputs.append(body.decode())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(_content_chunk("Safe prefix! "))
            self.wfile.flush()
            if streaming:
                continue_output.wait(5.0)
            for delta in ("ada@", "example.com", "; done!"):
                self.wfile.write(_content_chunk(delta))
                self.wfile.flush()
            self.wfile.write(_terminal_frames())
            self.wfile.flush()
            upstream_finished.set()

        def log_message(self, format: str, *args: object) -> None:
            """Keep synthetic provider traffic out of diagnostics."""
            del format, args

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    _, raw_key = _configured_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    engine = engine_from_document(
        {
            "adapters": [{"kind": "regex", "adapter_id": "email", "builtin_patterns": ["email"]}],
            "policies": [
                {
                    "policy_id": "redact",
                    "organization_id": "local",
                    "identity_id": "default",
                    "protected": True,
                    "checks": [
                        {
                            "check_id": stage,
                            "capability": "pii",
                            "stage": stage,
                            "action": "modify",
                            "adapter_id": "email",
                            "timeout_ms": 500,
                        }
                        for stage in ("input", "output")
                    ],
                }
            ],
        }
    )
    control = NativeControlPlane(
        load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "synthetic"}),
        guardrails=engine,
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ready = threading.Event()
    stop = exp_gateway_native.shutdown_handle()
    errors: list[Exception] = []

    def serve() -> None:
        """Run the real Rust server and retain startup failures for assertion."""
        try:
            serve_native_gateway(
                control,
                host="127.0.0.1",
                port=port,
                shutdown=stop,
                on_listening=ready.set,
                graceful_timeout_seconds=1.0,
            )
        except Exception as error:  # noqa: BLE001 - propagate worker errors below.
            errors.append(error)
            ready.set()

    gateway_thread = threading.Thread(target=serve, daemon=True)
    gateway_thread.start()
    try:
        assert ready.wait(10.0)
        assert not errors
        body: JsonObject = {"model": "coding", "stream": streaming, "max_tokens": 100}
        if surface == "responses":
            path = "/v1/responses"
            body.pop("max_tokens")
            body["input"] = "Contact bob@example.org"
        else:
            path = "/v1/messages" if surface == "messages" else "/v1/chat/completions"
            body["messages"] = [{"role": "user", "content": "Contact bob@example.org"}]
        with httpx.stream(
            "POST",
            f"http://127.0.0.1:{port}{path}",
            headers={"authorization": f"Bearer {raw_key}"},
            json=body,
            timeout=10.0,
        ) as response:
            assert response.status_code == 200
            chunks: list[str] = []
            for chunk in response.iter_text():
                chunks.append(chunk)
                if streaming and "Safe prefix!" in "".join(chunks):
                    if not continue_output.is_set():
                        assert not upstream_finished.is_set()
                        continue_output.set()
            text = "".join(chunks)
        assert "Safe prefix!" in text
        assert "[REDACTED]" in text
        assert "ada@" not in text
        assert "example.com" not in text
        assert provider_inputs and "bob@example.org" not in provider_inputs[0]
        assert "[REDACTED]" in provider_inputs[0]
        if streaming:
            assert continue_output.is_set()
        assert not errors
    finally:
        continue_output.set()
        stop.request_shutdown()
        gateway_thread.join(5.0)
        provider.shutdown()
        provider.server_close()
        provider_thread.join(5.0)
