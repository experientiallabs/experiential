"""Input approval precedes learned project routing and its real embedding dispatch."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _admit, _project_control_plane
from exp.runtime.gateway.tests.parallel_input_guardrails_test import _Classifier, _engine

_PROMPT = "project-guardrail-embedding-canary"


@contextmanager
def _embedding_server() -> Iterator[tuple[str, list[tuple[str, bytes]]]]:
    """Serve real embedding requests and retain their paths and bodies for ordering checks."""
    received: list[tuple[str, bytes]] = []

    class EmbeddingHandler(BaseHTTPRequestHandler):
        """Return the fixed two-dimensional vector used by the project activation."""

        def do_POST(self) -> None:  # noqa: N802 - HTTP protocol.
            """Record provider-visible input before returning a valid embedding response."""
            payload = self.rfile.read(int(self.headers["content-length"]))
            received.append((self.path, payload))
            body = json.dumps(
                {
                    "data": [{"embedding": [1.0, 0.0], "index": 0}],
                    "model": "embedder-model",
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress the fixture server's default stderr access log."""
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def _body(surface: Literal["chat", "responses", "messages"]) -> str:
    """Address the real project alias with the same initial user intent on each API."""
    payload: JsonObject = {"model": "coding"}
    if surface == "responses":
        payload["input"] = _PROMPT
    else:
        payload["messages"] = [{"role": "user", "content": _PROMPT}]
        payload["max_tokens"] = 128
    return json.dumps(payload)


def _ledger_counts(database_path: Path) -> tuple[int, int]:
    """Read durable request acceptance and generation attempt counts."""
    with sqlite3.connect(database_path) as connection:
        requests = connection.execute("select count(*) from gateway_requests").fetchone()[0]
        attempts = connection.execute("select count(*) from gateway_attempts").fetchone()[0]
    return requests, attempts


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("outcome", ["allow", "block", "error", "timeout"])
def test_project_embedding_waits_for_input_approval(
    tmp_path: Path,
    surface: Literal["chat", "responses", "messages"],
    outcome: Literal["allow", "block", "error", "timeout"],
) -> None:
    """A parallel policy cannot speculate through the project's customer-funded embedder."""
    classifier = _Classifier(outcome)
    with _embedding_server() as (base_url, received):
        manager, unguarded, key = _project_control_plane(tmp_path, base_url=base_url)
        control = NativeControlPlane(
            unguarded._components,  # noqa: SLF001 - share the real activated project fixture.
            guardrails=_engine(classifier, timeout_ms=100 if outcome == "timeout" else 3000),
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_admit, control, key, _body(surface), surface=surface)
            try:
                assert classifier.started.wait(5), "input classification never started"
                assert received == [], "project routing dispatched before input approval"
                assert _ledger_counts(manager.database_path) == (0, 0)
                assert not future.done()
                if outcome != "timeout":
                    classifier.release.set()

                if outcome == "allow":
                    admission = future.result(timeout=5)
                    assert "escalate" not in admission
                    assert admission["guardrail_input_pending"] is False
                    assert len(received) == 1
                    assert received[0][0] == "/v1/embeddings"
                    assert _PROMPT.encode() in received[0][1]
                    assert _ledger_counts(manager.database_path) == (1, 0)
                else:
                    with pytest.raises(NativeBridgeError):
                        future.result(timeout=5)
                    assert received == []
                    assert _ledger_counts(manager.database_path) == (0, 0)
            finally:
                classifier.release.set()
                try:
                    admission = future.result(timeout=5)
                except NativeBridgeError:
                    pass
                else:
                    control.abandon(json.dumps({"request_id": admission["request_id"]}))
