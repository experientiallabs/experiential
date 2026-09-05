"""Prove HTTP and WebSocket transport context against the compiled engine."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest
from websockets.sync.client import connect

from exp.runtime.gateway.native_bridge import NativeBridgeError
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

native = pytest.importorskip("exp_gateway_native")


class _PolicyProbe:
    """Reject before provider work while recording each native callback context."""

    def __init__(self) -> None:
        """Initialize the content-free callback trace."""
        self.seen: list[tuple[str, str | None]] = []

    def call_with_context(self, method: str, argument: str, context: str | None) -> str:
        """Authenticate the fixture and refuse every data operation by policy."""
        self.seen.append((method, context))
        if method == "authenticate":
            return "{}"
        raise NativeBridgeError(
            OpenAIProtocolError(
                status_code=403,
                code="probe_policy",
                message="Probe policy refused the request.",
            )
        )


@contextmanager
def _server(probe: _PolicyProbe) -> Iterator[str]:
    """Serve the real native listener on loopback and always drain its thread."""
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    ready = threading.Event()
    stop = native.shutdown_handle()
    errors: list[Exception] = []

    def run() -> None:
        """Run the extension with an explicit shutdown signal, without subprocesses."""
        try:
            native.serve(
                probe,
                json.dumps(
                    {
                        "host": "127.0.0.1",
                        "port": port,
                        "callback_permits": 2,
                        "request_timeout_seconds": 10.0,
                        "graceful_timeout_seconds": 2.0,
                    }
                ),
                stop,
                ready.set,
            )
        except Exception as error:  # noqa: BLE001 - report the server thread's failure to the test
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert ready.wait(10), errors
        yield f"http://127.0.0.1:{port}"
    finally:
        stop.request_shutdown()
        thread.join(10)
        assert not thread.is_alive()
        assert not errors


def test_all_admission_surfaces_receive_transport_context() -> None:
    """Discovery, inference, batch and replay all reach the same scoped hook."""
    probe = _PolicyProbe()
    with _server(probe) as origin, httpx.Client(base_url=origin) as client:
        for method, path, body, extra in (
            ("GET", "/v1/models", None, {}),
            ("GET", "/v1/models/demo", None, {}),
            ("POST", "/v1/chat/completions", {"model": "demo", "messages": []}, {}),
            (
                "POST",
                "/v1/chat/completions",
                {"model": "demo", "messages": []},
                {"idempotency-key": "one"},
            ),
            ("POST", "/v1/responses", {"model": "demo", "input": "hello"}, {}),
            ("POST", "/v1/messages", {"model": "demo", "messages": [], "max_tokens": 1}, {}),
            ("POST", "/v1/embeddings", {"model": "demo", "input": "hello"}, {}),
            ("POST", "/v1/images/generations", {"model": "demo", "prompt": "hello"}, {}),
            (
                "POST",
                "/v1/batches",
                {
                    "input_file_id": "file-one",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
                {},
            ),
        ):
            response = client.request(
                method,
                path,
                json=body,
                headers={
                    "authorization": "Bearer fixture-key",
                    "x-exp-request-context": "BY",
                    **extra,
                },
            )
            assert response.status_code == 403, (path, response.text)
        assert all(context == "BY" for _, context in probe.seen)
        assert {method for method, _ in probe.seen} >= {
            "models",
            "model_detail",
            "admit",
            "claim_scope",
            "batch_create",
        }
        client.get("/v1/models", headers={"authorization": "Bearer fixture-key"})
        assert probe.seen[-1] == ("models", None)


def test_websocket_frames_keep_upgrade_context_and_ignore_body_spoofs() -> None:
    """The upgrade callback's spawned task does not lose geographic authority."""
    probe = _PolicyProbe()
    with (
        _server(probe) as origin,
        connect(
            origin.replace("http:", "ws:") + "/v1/responses",
            additional_headers={
                "authorization": "Bearer fixture-key",
                "x-exp-request-context": "BY",
            },
        ) as stream,
    ):
        for _ in range(2):
            stream.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "model": "demo",
                        "input": "hello",
                        "metadata": {"country": "US"},
                    }
                )
            )
            response = json.loads(stream.recv(timeout=10))
            assert response["status"] == 403
        assert sum(method == "admit" for method, _ in probe.seen) == 2
        assert all(context == "BY" for _, context in probe.seen)
