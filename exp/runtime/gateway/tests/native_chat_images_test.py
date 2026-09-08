"""Copilot image references survive the served gateway and reach the provider intact."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import GatewayDeploymentCapabilities
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.tests.launch_test import _provider_frame, _ServedGateway, _unused_port

_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "url",
    ["https://example.test/attachment", f"data:image/png;base64,{_PNG_BASE64}"],
    ids=["remote", "inline"],
)
def test_copilot_image_mime_hint_serves_without_leaking_to_the_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool, url: str
) -> None:
    """The reported fourth-message image reaches a vision route without its client hint."""
    received: list[JsonObject] = []

    class Provider(BaseHTTPRequestHandler):
        """Capture the image request and return a finite completion with usage."""

        def do_POST(self) -> None:  # noqa: N802
            """Record the provider-bound payload before returning an answer."""
            received.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            frames = (
                _provider_frame(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "A screenshot."},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                )
                + _provider_frame(
                    {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
                )
                + b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(frames)))
            self.end_headers()
            self.wfile.write(frames)

        def log_message(self, format: str, *args: object) -> None:
            """Keep synthetic request traffic out of test logs."""
            del format, args

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    _manager, raw_key = _configured_gateway(
        tmp_path,
        base_url=f"http://127.0.0.1:{provider.server_port}/v1",
        provider="azure",
        api_version="2024-10-21",
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True, supports_image_input=True, supports_image_url_input=True
        ),
    )
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "stream": stream,
                "messages": [
                    {"role": "system", "content": "Help with screenshots."},
                    {"role": "user", "content": "Hello."},
                    {"role": "assistant", "content": "Send the screenshot."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": url,
                                    "detail": "high",
                                    "media_type": "image/png",
                                },
                            },
                            {"type": "text", "text": "What is this?"},
                        ],
                    },
                ],
            },
            timeout=10,
        )
        assert response.status_code == 200, response.text
        if stream:
            assert "A screenshot." in response.text
            assert response.text.endswith("data: [DONE]\n\n")
        else:
            assert response.json()["choices"][0]["message"]["content"] == "A screenshot."
        assert len(received) == 1
        messages = cast(list[JsonObject], received[0]["messages"])
        assert messages[3]["content"] == [
            {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
            {"type": "text", "text": "What is this?"},
        ]
    finally:
        gateway.stop()
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)
