"""No terminal or gateway-generated search path can bypass host inspection."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import exp_gateway_native
import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.tests.launch_test import _unused_port
from exp.runtime.gateway.tests.native_tool_search_test import (
    _configure,
    _search_call_turn,
)
from exp.runtime.gateway.tests.native_waterfall_test import (
    _content_chunk,
    _sse_frame,
    _terminal_frames,
)
from exp.runtime.gateway.tests.runtime_guardrails_test import _Guard
from exp.runtime.gateway.tests.web_search_backend_fixture_test import StaticWebSearchBackend
from exp.runtime.gateway.web_search.contracts import GatewayWebSearchResult


@contextmanager
def _serving(control: NativeControlPlane) -> Iterator[str]:
    """Run an injected policy through the actual native HTTP server.

    Args:
        control: Authenticated control plane with a request-scoped host policy.

    Yields:
        The loopback URL serving the native extension.
    """
    port = _unused_port()
    ready = threading.Event()
    stop = exp_gateway_native.shutdown_handle()
    worker = threading.Thread(
        target=serve_native_gateway,
        args=(control,),
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "shutdown": stop,
            "on_listening": ready.set,
            "graceful_timeout_seconds": 1.0,
        },
        daemon=True,
    )
    worker.start()
    try:
        assert ready.wait(10)
        yield f"http://127.0.0.1:{port}"
    finally:
        stop.request_shutdown()
        worker.join(5)
        assert not worker.is_alive()


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("path", ["refusal", "search", "empty-search"])
def test_precommit_and_synthesized_output_is_inspected_before_settlement(
    tmp_path: Path,
    surface: str,
    stream: bool,
    path: str,
) -> None:
    """A real refusal flush or search prelude cannot release the synthetic marker."""

    class Provider(BaseHTTPRequestHandler):
        """Serve a refusal, an empty completion, or a safe answer."""

        def do_POST(self) -> None:  # noqa: N802
            """Return a finite synthetic SSE stream with actual usage."""
            self.rfile.read(int(self.headers["content-length"]))
            if path == "refusal":
                payload = (
                    _sse_frame(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "refusal": "withhold-marker",
                                    },
                                }
                            ]
                        }
                    )
                    + _sse_frame(
                        {
                            "error": {
                                "type": "invalid_request_error",
                                "message": "Synthetic provider failure",
                            }
                        }
                    )
                    + b"data: [DONE]\n\n"
                )
            elif path == "empty-search":
                payload = (
                    _sse_frame(
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 12,
                                "completion_tokens": 0,
                            },
                        }
                    )
                    + _sse_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                    + b"data: [DONE]\n\n"
                )
            else:
                payload = _content_chunk("Allowed synthetic answer.") + _terminal_frames()
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress synthetic request logs."""

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=provider.serve_forever, daemon=True)
    worker.start()
    key = _configure(tmp_path, f"http://127.0.0.1:{provider.server_port}/v1")
    manager = GatewayManagement(tmp_path)
    alias = manager.aliases()[0]
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="refusal-enabled",
        pool_id="coding",
        snapshot_ref=str(alias.snapshot_ref),
        catalog_sha256=str(alias.catalog_sha256),
        refusal_failover=True,
    )
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "synthetic"})
    policy = _Guard()
    search = StaticWebSearchBackend(
        (
            GatewayWebSearchResult(
                url="https://synthetic.invalid/result",
                title="withhold-marker",
            ),
        )
    )
    control = NativeControlPlane(components, runtime_guardrail=policy, web_search=search)
    body: JsonObject = {
        "model": "coding" if path == "refusal" else "coding:online",
        "stream": stream,
    }
    if surface == "responses":
        route = "/v1/responses"
        body["input"] = "Explain the synthetic result."
    else:
        route = "/v1/messages" if surface == "messages" else "/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": "Explain the synthetic result."}]
        body["max_tokens"] = 1000
        if surface == "messages" and path != "refusal":
            body["model"] = "coding"
            body["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
    try:
        with _serving(control) as url:
            response = httpx.post(
                url + route, headers={"authorization": f"Bearer {key}"}, json=body, timeout=10
            )
        assert response.status_code == 400, response.text
        assert "withhold-marker" not in response.text
        assert "Synthetic policy violation" in response.text
        assert "withhold-marker" in policy.sessions[0].text
        with sqlite3.connect(components.ledger.database_path) as connection:
            assert connection.execute(
                "select state, failure_class from gateway_attempts"
            ).fetchall() == [("failed", "guardrail")]
    finally:
        provider.shutdown()
        provider.server_close()
        worker.join(5)


@pytest.mark.parametrize("stream", [False, True])
def test_synthesized_responses_tool_search_schema_is_inspected(
    tmp_path: Path, stream: bool
) -> None:
    """A tool-search result schema is checked before Responses synthesizes hosted items."""

    class Provider(BaseHTTPRequestHandler):
        """Select a deferred tool, then return a safe answer."""

        def do_POST(self) -> None:  # noqa: N802
            """Choose the search call only before its result enters the conversation."""
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            payload = (
                _content_chunk("Allowed answer.") + _terminal_frames()
                if any(message.get("role") == "tool" for message in body["messages"])
                else _search_call_turn("tool_search")
            )
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress fixture request logs."""

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=provider.serve_forever, daemon=True)
    worker.start()
    key = _configure(tmp_path, f"http://127.0.0.1:{provider.server_port}/v1")
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "synthetic"})
    policy = _Guard()
    control = NativeControlPlane(components, runtime_guardrail=policy)
    response_tools: list[JsonObject] = [
        {
            "type": "function",
            "name": "get_weather",
            "defer_loading": True,
            "description": "Current weather withhold-marker",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
        {"type": "tool_search"},
    ]
    try:
        with _serving(control) as url:
            response = httpx.post(
                url + "/v1/responses",
                headers={"authorization": f"Bearer {key}"},
                json={
                    "model": "coding",
                    "stream": stream,
                    "input": "What's the weather?",
                    "tools": response_tools,
                },
                timeout=10,
            )
        assert response.status_code == 400, response.text
        assert "withhold-marker" not in response.text
        assert "Synthetic policy violation" in response.text
        assert "withhold-marker" in policy.sessions[0].text
    finally:
        provider.shutdown()
        provider.server_close()
        worker.join(5)
