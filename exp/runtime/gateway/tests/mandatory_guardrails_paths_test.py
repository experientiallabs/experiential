"""No terminal or gateway-generated search path can bypass host inspection."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import exp_gateway_native
import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.contracts import ClassifierVerdict, GuardrailCheck
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.tests.launch_test import _unused_port
from exp.runtime.gateway.tests.mandatory_guardrails_test import _Guard
from exp.runtime.gateway.tests.native_tool_search_test import (
    _configure,
    _search_call_turn,
)
from exp.runtime.gateway.tests.native_waterfall_test import (
    _content_chunk,
    _sse_frame,
    _terminal_frames,
)
from exp.runtime.gateway.tests.web_search_backend_fixture_test import StaticWebSearchBackend
from exp.runtime.gateway.tool_search.round import perform_round
from exp.runtime.gateway.web_search.contracts import GatewayWebSearchResult


class _RetrievedGuard(_Guard):
    """Refuse the marker only when it enters the conversation, including retrieved turns."""

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Inspect messages after each gateway-owned conversation expansion."""
        self.block_input = any(
            "withhold-marker" in message.model_dump_json() for message in request.messages
        )
        return await super().inspect_input(request=request, check=check)


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
@pytest.mark.parametrize(
    "path", ["refusal", "search", "empty-search", "search-input", "empty-search-input"]
)
def test_precommit_and_synthesized_output_is_inspected_before_settlement(
    tmp_path: Path,
    surface: str,
    stream: bool,
    path: str,
) -> None:
    """A real refusal flush or search prelude cannot release the synthetic marker."""
    provider_requests: list[JsonObject] = []

    class Provider(BaseHTTPRequestHandler):
        """Serve a refusal, an empty completion, or a safe answer."""

        def do_POST(self) -> None:  # noqa: N802
            """Return a finite synthetic SSE stream with actual usage."""
            provider_requests.append(
                json.loads(self.rfile.read(int(self.headers["content-length"])))
            )
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
            elif path.removesuffix("-input") == "empty-search":
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
    policy = _RetrievedGuard() if path.endswith("-input") else _Guard()
    search = StaticWebSearchBackend(
        (
            GatewayWebSearchResult(
                url="https://synthetic.invalid/result",
                title="withhold-marker",
            ),
        )
    )
    control = NativeControlPlane(components, guardrails=policy, web_search=search)
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
        assert "blocked by a gateway guardrail" in response.text
        blocked_input = path.endswith("-input")
        assert len(provider_requests) == (0 if blocked_input else 1)
        if blocked_input:
            assert any(
                "withhold-marker" in request.model_dump_json() for request in policy.requests
            )
        else:
            assert any("withhold-marker" in item.model_dump_json() for item in policy.completions)
        with sqlite3.connect(components.ledger.database_path) as connection:
            assert connection.execute(
                "select state, failure_class from gateway_attempts"
            ).fetchall() == ([] if blocked_input else [("failed", "guardrail")])
            assert connection.execute(
                "select terminal_state, terminal_at is not null from gateway_requests"
            ).fetchall() == [("failed", 1)]
            assert connection.execute(
                "select web_search_requests from gateway_requests"
            ).fetchall() == [(1 if blocked_input else 0,)]
    finally:
        provider.shutdown()
        provider.server_close()
        worker.join(5)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("phase", "surface"),
    [("render", "responses")]
    + [
        (phase, surface)
        for phase in ("query", "retrieved")
        for surface in ("chat", "responses", "messages")
    ],
)
def test_tool_search_is_inspected_before_execution_redial_and_rendering(
    tmp_path: Path, stream: bool, phase: str, surface: str
) -> None:
    """Reject unsafe generated queries before search and retrieved results before redial."""
    provider_requests: list[JsonObject] = []

    class Provider(BaseHTTPRequestHandler):
        """Select a deferred tool, then return a safe answer."""

        def do_POST(self) -> None:  # noqa: N802
            """Choose the search call only before its result enters the conversation."""
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            provider_requests.append(body)
            search_turn = _search_call_turn("tool_search")
            if phase == "query":
                search_turn = search_turn.replace(b"current weather", b"withhold-marker")
            payload = (
                _content_chunk("Allowed answer.") + _terminal_frames()
                if any(message.get("role") == "tool" for message in body["messages"])
                else search_turn
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
    guard_type = {"retrieved": _RetrievedGuard}.get(phase, _Guard)
    policy = guard_type()
    control = NativeControlPlane(components, guardrails=policy)
    description = "Current weather" + (
        " withhold-marker" if phase in {"retrieved", "render"} else ""
    )
    function: JsonObject = {
        "name": "get_weather",
        "description": description,
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    body: JsonObject = {"model": "coding", "stream": stream}
    if surface == "responses":
        route = "/v1/responses"
        body.update(
            input="What's the weather?",
            tools=[
                {"type": "function", **function, "defer_loading": True},
                {"type": "tool_search"},
            ],
        )
    else:
        route = "/v1/messages" if surface == "messages" else "/v1/chat/completions"
        body.update(messages=[{"role": "user", "content": "What's the weather?"}], max_tokens=1000)
        if surface == "messages":
            body["tools"] = [
                {
                    "name": "get_weather",
                    "description": description,
                    "input_schema": function["parameters"],
                    "defer_loading": True,
                },
                {"type": "tool_search_tool_bm25", "name": "tool_search_tool_bm25"},
            ]
        else:
            body["tools"] = [
                {"type": "function", "function": function, "defer_loading": True},
                {"type": "openrouter:tool_search"},
            ]
    try:
        with (
            patch(
                "exp.runtime.gateway.native_tool_search.perform_round", wraps=perform_round
            ) as search,
            patch.object(control._accounting, "settle", wraps=control._accounting.settle) as settle,
            _serving(control) as url,
        ):
            response = httpx.post(
                url + route,
                headers={"authorization": f"Bearer {key}"},
                json=body,
                timeout=10,
            )
        assert response.status_code == 400, response.text
        assert "withhold-marker" not in response.text
        assert "blocked by a gateway guardrail" in response.text
        assert search.call_count == (0 if phase == "query" else 1)
        assert len(provider_requests) == (2 if phase == "render" else 1)
        if phase != "render":
            metered = [
                json.loads(call.args[0]).get("tool_search_requests", 0)
                for call in settle.call_args_list
            ]
            assert metered == [0 if phase == "query" else 1]
        if phase == "retrieved":
            assert any(
                "withhold-marker" in request.messages[-1].model_dump_json()
                for request in policy.requests
            )
        else:
            assert any("withhold-marker" in item.model_dump_json() for item in policy.completions)
        with sqlite3.connect(components.ledger.database_path) as connection:
            attempts = connection.execute(
                "select state, failure_class, input_tokens, output_tokens "
                "from gateway_attempts order by rowid"
            ).fetchall()
        assert attempts[-1][:2] == (
            "failed",
            "guardrail",
        )
        assert attempts[0][2:] == (50, 8)
        assert len(attempts) == len(provider_requests)
    finally:
        provider.shutdown()
        provider.server_close()
        worker.join(5)
