"""Facts output across the served native gateway: every surface, buffered and streamed."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_capture import CaptureConfiguration, CaptureController
from exp.runtime.gateway.native_capture_facts import CAPTURE_FACTS_TAG, CaptureRecordFacts
from exp.runtime.gateway.native_capture_facts_test import rewrite
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.tests.launch_test import (
    _configure_gateway,
    _LoopbackProvider,
    _unused_port,
    _wait_ready,
)

native = pytest.importorskip("exp_gateway_native")

_PROTOCOLS = {
    "chat/completions": "chat_completions",
    "responses": "responses",
    "messages": "messages",
}


def test_served_requests_arrive_as_facts_beside_their_record_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Facts name each served request; the splice equals a decoded storage plan."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    _manager, raw_key = _configure_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    components = load_gateway_components(tmp_path)
    received: list[tuple[str, bytes, bytes]] = []
    lock = threading.Lock()

    def write_batch(records: tuple[tuple[str, bytes, bytes], ...]) -> list[bool]:
        """Keep every handed record and acknowledge it."""
        with lock:
            received.extend(records)
        return [True] * len(records)

    collector = native.CaptureCollector.batched(
        CaptureConfiguration(settlement_required=False).model_dump_json(),
        write_batch,
        bytes_output=True,
        facts_output=True,
    )
    control = NativeControlPlane(
        components, capture=CaptureController(collector, application_for=lambda _auth: "app")
    )
    port = _unused_port()
    shutdown = native.shutdown_handle()
    failures: list[BaseException] = []

    def run() -> None:
        """Serve the real data plane until shutdown."""
        try:
            serve_native_gateway(
                control, host="127.0.0.1", port=port, capture=collector, shutdown=shutdown
            )
        except BaseException as error:  # noqa: BLE001 - surfaced after shutdown.
            failures.append(error)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    served: dict[str, tuple[str, bool, str]] = {}
    try:
        _wait_ready(port, worker)
        for surface in _PROTOCOLS:
            for stream in (False, True):
                payload: dict[str, object] = {"model": "coding", "stream": stream}
                if surface == "responses":
                    payload["input"] = "capture task"
                else:
                    payload["messages"] = [{"role": "user", "content": "capture task"}]
                if surface == "messages":
                    payload["max_tokens"] = 128
                response = httpx.post(
                    f"http://127.0.0.1:{port}/v1/{surface}",
                    headers={"authorization": f"Bearer {raw_key}"},
                    json=payload,
                    timeout=10,
                )
                assert response.status_code == 200, response.text
                served[response.headers["x-request-id"]] = (
                    surface,
                    stream,
                    response.headers["x-gateway-deployment"],
                )
    finally:
        shutdown.request_shutdown()
        worker.join(timeout=10)
        assert control.close(2)
        components.write_ledger.close()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
    assert not failures
    assert collector.close(5)
    by_request: dict[str, tuple[CaptureRecordFacts, bytes]] = {}
    for tag, facts_json, record in received:
        assert tag == CAPTURE_FACTS_TAG
        facts = CaptureRecordFacts.model_validate_json(facts_json)
        assert facts.record_bytes == len(record)
        by_request[facts.request_id] = (facts, record)
    assert set(by_request) == set(served)
    for request_id, (surface, stream, deployment) in served.items():
        facts, record = by_request[request_id]
        document = json.loads(record)
        assert facts.protocol == _PROTOCOLS[surface] == document["request"]["protocol"]
        assert facts.scope.organization_id == document["request"]["scope"]["organization_id"]
        assert facts.deployment_id == deployment == document["deployment_id"]
        assert facts.captured_at == document["captured_at"]
        assert facts.complete and facts.response is not None
        assert facts.response.kind == document["response"]["kind"]
        assert facts.response.kind == ("sse" if stream else "json")
        assert facts.response.status == 200
        # A served record carries response-derived fields (at least its metrics).
        assert facts.spans is not None and facts.spans.response_fields
        assert b"capture task" not in json.dumps(facts.model_dump()).encode()
        for keep_prompt in (True, False):
            for keep_response in (True, False):
                planned = facts.planned(
                    record, keep_prompt=keep_prompt, keep_response=keep_response
                )
                assert json.loads(planned) == rewrite(
                    record, keep_prompt=keep_prompt, keep_response=keep_response
                )
