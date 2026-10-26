"""OpenRouter's legacy ``include_reasoning`` traverses the served native gateway."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.tests.launch_test import _ServedGateway, _unused_port
from exp.runtime.gateway.tests.native_json_object_test import (
    _configure_json_gateway,
    _provider_response,
)
from exp.runtime.models import registry


def test_chat_include_reasoning_is_served_as_reasoning_exclude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``include_reasoning: false`` serves, disclosed, and never reaches the provider.

    The decoder used to refuse it with 400 ``unsupported_parameter`` ("The
    parameter 'include_reasoning' is not supported by this gateway profile.").
    A contradiction with ``reasoning.exclude`` is a typed 400 before dispatch.
    """
    received: list[JsonObject] = []
    provider_body = _provider_response("openai-compatible")

    class Provider(BaseHTTPRequestHandler):
        """Record the provider-bound request and return one finite, billed answer."""

        def do_POST(self) -> None:  # noqa: N802
            """Capture what the native plane sends and answer with real SSE frames."""
            received.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(provider_body)))
            self.end_headers()
            self.wfile.write(provider_body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep synthetic traffic out of test logs."""
            del format, args

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    base_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    factory, _ = registry._HTTP_PROVIDERS["openai-compatible"]
    monkeypatch.setattr(
        registry,
        "_HTTP_PROVIDERS",
        {**registry._HTTP_PROVIDERS, "openai-compatible": (factory, base_url)},
    )
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    raw_key = _configure_json_gateway(tmp_path, "openai-compatible", base_url)
    gateway = _ServedGateway(tmp_path, _unused_port())
    url = f"http://127.0.0.1:{gateway.port}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {raw_key}"}
    messages = [{"role": "user", "content": "Give the answer."}]
    try:
        gateway.start()
        response = httpx.post(
            url,
            headers=headers,
            json={"model": "coding", "messages": messages, "include_reasoning": False},
            timeout=10,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == '{"answer":"ready"}'
        assert body["x-experiential-ignored-parameters"] == [
            "include_reasoning->translated(reasoning.exclude)",
            "reasoning.exclude->dropped(not_carried)",
        ]
        assert len(received) == 1
        assert "include_reasoning" not in received[0]
        assert "reasoning" not in received[0]

        defaulted = httpx.post(
            url,
            headers=headers,
            json={"model": "coding", "messages": messages, "include_reasoning": True},
            timeout=10,
        )
        assert defaulted.status_code == 200, defaulted.text
        assert defaulted.json().get("x-experiential-ignored-parameters", []) == []
        assert len(received) == 2
        assert "include_reasoning" not in received[1]

        conflict = httpx.post(
            url,
            headers=headers,
            json={
                "model": "coding",
                "messages": messages,
                "include_reasoning": False,
                "reasoning": {"exclude": False},
            },
            timeout=10,
        )
        assert conflict.status_code == 400, conflict.text
        error = conflict.json()["error"]
        assert error["code"] == "invalid_parameter"
        assert error["param"] == "include_reasoning"
        assert "reasoning.exclude" in error["message"]
        assert len(received) == 2
    finally:
        gateway.stop()
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)
