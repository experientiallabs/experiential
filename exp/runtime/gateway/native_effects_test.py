"""Tests proving admission prework cannot receive a free-capacity certificate."""

from __future__ import annotations

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.guardrails.classifiers import ScriptedClassifier
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _admit, _chat_body, _project_control_plane
from exp.runtime.gateway.tests.guardrails_native_bridge_test import _engine
from exp.runtime.gateway.tests.web_search_backend_fixture_test import (
    FailingWebSearchBackend,
    StaticWebSearchBackend,
)
from exp.runtime.gateway.web_search.contracts import GatewayWebSearchResult


def _start(control: NativeControlPlane, raw_key: str, request_id: str) -> JsonObject:
    """Invoke real attempt admission without making a model call."""
    return json.loads(
        control.start_attempt(
            json.dumps(
                {
                    "request_id": request_id,
                    "raw_key": raw_key,
                    "attempt_ordinal": 0,
                }
            )
        )
    )


@pytest.mark.parametrize("prework", ["search_success", "search_failure", "input_guardrail"])
def test_paid_admission_work_prevents_capacity_certificate_and_same_key_reentry(
    tmp_path: Path,
    prework: str,
) -> None:
    """Successful or uncertain search/classification stays billed conservatively."""
    manager, key = _configured_gateway(tmp_path)
    search = (
        FailingWebSearchBackend()
        if prework == "search_failure"
        else StaticWebSearchBackend(
            (GatewayWebSearchResult(url="https://example.test/result", title="Fixture"),)
        )
    )
    classifier = ScriptedClassifier()
    control = NativeControlPlane(
        load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"}),
        default_lane_bound=1,
        web_search=search,
        guardrails=_engine(classifier) if prework == "input_guardrail" else None,
    )
    occupied = _admit(control, key, _chat_body())
    assert "attempt_id" in _start(control, key, str(occupied["request_id"]))
    body = _chat_body(model="coding:online" if prework.startswith("search") else "coding")
    with patch.object(search, "search", wraps=search.search) as search_call:
        admitted = _admit(control, key, body, idempotency_key="prework-refusal")
    assert search_call.call_count == int(prework.startswith("search"))
    entry = control._accounting.entry(str(admitted["request_id"]))  # noqa: SLF001
    assert entry is not None and not entry.no_paid_prework
    refused = _start(control, key, str(admitted["request_id"]))
    assert refused["exhausted"] is True
    assert "known_unbilled" not in refused
    if prework == "input_guardrail":
        assert classifier.input_calls == 2
    with sqlite3.connect(manager.database_path) as connection:
        assert connection.execute(
            "SELECT terminal_state, failed_without_effects, "
            "(SELECT COUNT(*) FROM gateway_attempts WHERE request_id = r.request_id) "
            "FROM gateway_requests r WHERE request_id = ?",
            (admitted["request_id"],),
        ).fetchone() == ("failed", 0, 0)
    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, key, body, idempotency_key="prework-refusal")
    assert json.loads(raised.value.public_error_json)["code"] == "idempotency_replay_unavailable"


def test_paid_routing_embedding_prevents_capacity_certificate(tmp_path: Path) -> None:
    """A real project selector may embed before any attempted model dispatch."""

    class Embeddings(BaseHTTPRequestHandler):
        """Serve a deterministic vector for the genuine project route selector."""

        calls = 0

        def do_POST(self) -> None:  # noqa: N802
            """Return one synthetic vector without any external provider."""
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            type(self).calls += 1
            body = json.dumps(
                {
                    "model": payload["model"],
                    "data": [{"index": 0, "embedding": [1.0, 0.0]}],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep local fixture traffic out of logs."""
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Embeddings)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        manager, original, key = _project_control_plane(
            tmp_path,
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
        )
        control = NativeControlPlane(original._components, default_lane_bound=1)  # noqa: SLF001
        first = _admit(control, key, _chat_body())
        assert "attempt_id" in _start(control, key, str(first["request_id"]))
        before = Embeddings.calls
        admitted = _admit(control, key, _chat_body(), idempotency_key="routed-capacity")
        assert Embeddings.calls > before
        response = _start(control, key, str(admitted["request_id"]))
        assert response["exhausted"] is True and "known_unbilled" not in response
        with sqlite3.connect(manager.database_path) as connection:
            assert connection.execute(
                "SELECT failed_without_effects FROM gateway_requests WHERE request_id = ?",
                (admitted["request_id"],),
            ).fetchone() == (0,)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)
