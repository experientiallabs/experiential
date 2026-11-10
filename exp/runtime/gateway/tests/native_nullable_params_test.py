"""An explicit JSON null on an OpenAI-nullable field is served as the field's default."""

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


def test_chat_stream_null_is_served_as_a_non_streaming_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``"stream": null`` (with null sampling options beside it) returns one JSON completion.

    OpenAI documents ``stream`` as nullable with a default of false, so a null
    is served as a non-streaming completion rather than refused as a non-boolean.
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
    try:
        gateway.start()
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": "Give the answer."}],
                "stream": None,
                "stream_options": None,
                "temperature": None,
                "max_tokens": None,
                "stop": None,
                "metadata": None,
            },
            timeout=10,
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == '{"answer":"ready"}'
        assert body.get("x-experiential-ignored-parameters", []) == []
        assert len(received) == 1
        sent = received[0]
        for field in ("temperature", "max_tokens", "stop", "metadata"):
            assert field not in sent
    finally:
        gateway.stop()
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)
