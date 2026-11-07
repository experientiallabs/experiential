"""Native capture completes on WebSockets and retains the HTTP relay contract."""

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from websockets.sync.client import connect

from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_capture import CaptureConfiguration, CaptureController
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.tests.launch_test import (
    _configure_gateway,
    _LoopbackProvider,
    _unused_port,
    _wait_ready,
)
from exp.runtime.gateway.tests.native_responses_ws_test import _collect_stream, _request_frame

native = pytest.importorskip("exp_gateway_native")


@pytest.mark.parametrize("relay_metadata", [False, True])
@pytest.mark.parametrize("transport", ["http", "websocket"])
def test_response_capture_completes_on_its_own_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relay_metadata: bool, transport: str
) -> None:
    """Persist every completed frame while HTTP alone awaits its outer relay."""
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackProvider)
    threading.Thread(target=provider.serve_forever, daemon=True).start()
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "local-diagnostic")
    _, key = _configure_gateway(tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1")
    records: list[str] = []
    response_stored = threading.Event()

    def persist(record: str) -> None:
        """Acknowledge records and signal when a complete response reaches the sink."""
        records.append(record)
        if json.loads(record).get("response") is not None:
            response_stored.set()

    collector = native.CaptureCollector(
        CaptureConfiguration(relay_metadata=relay_metadata, ttl_seconds=2).model_dump_json(),
        persist,
    )
    controller = CaptureController(collector, application_for=lambda _: "local-diagnostic")
    control = NativeControlPlane(load_gateway_components(tmp_path), capture=controller)
    original_admit, original_settle = control.admit, control.settle

    def admit(argument: str) -> str:
        """Make the local provider eligible for the hosted checkpoint lifecycle."""
        result = json.loads(original_admit(argument))
        for wire in result["route"]:
            wire["billing_customer_managed"] = False
        return json.dumps(result)

    def settle(argument: str) -> str:
        """Grant final response retention after real request settlement."""
        result = original_settle(argument)
        collector.settle(json.loads(argument)["request_id"], True, True)
        return result

    monkeypatch.setattr(control, "admit", admit)
    monkeypatch.setattr(control, "settle", settle)
    port, shutdown = _unused_port(), native.shutdown_handle()
    worker = threading.Thread(
        target=lambda: serve_native_gateway(
            control,
            host="127.0.0.1",
            port=port,
            capture=collector,
            shutdown=shutdown,
        ),
        daemon=True,
    )
    worker.start()
    try:
        _wait_ready(port, worker)
        if transport == "websocket":
            with connect(
                f"ws://127.0.0.1:{port}/v1/responses",
                additional_headers={
                    "Authorization": f"Bearer {key}",
                },
            ) as socket:
                # Reuse the connection: each frame must release its own capture.
                for _ in range(2):
                    response_stored.clear()
                    socket.send(_request_frame())
                    assert _collect_stream(socket)[-1]["type"] == "response.completed"
                    assert response_stored.wait(1), "WebSocket capture waited for an HTTP relay"
        else:
            response = httpx.post(
                f"http://127.0.0.1:{port}/v1/responses",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": "coding", "input": "capture", "store": False},
            )
            assert response.status_code == 200
            if relay_metadata:
                assert not response_stored.is_set()
                request_id = response.headers["x-request-id"]
                assert collector.claim_relay(request_id)
                assert collector.finish_relay(
                    request_id,
                    json.dumps(
                        {
                            "wire_request": None,
                            "headers": [],
                            "timing": {},
                            "relay_completed": True,
                            "client_disconnected": False,
                        }
                    ),
                    b"",
                )
            assert response_stored.wait(1)
        assert collector.close(1), "Completed responses left capture ownership pending"
        decoded = [json.loads(record) for record in records]
        responses = sum(record.get("response") is not None for record in decoded)
        assert collector.counts()[5] == 0
        assert responses == (2 if transport == "websocket" else 1)
    finally:
        shutdown.request_shutdown()
        worker.join(10)
        control.close(2)
        collector.close(2)
        provider.shutdown()
