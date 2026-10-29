"""Azure annotation-only SSE completes through the served native gateway."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from openai import OpenAI

from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.tests.launch_test import _provider_frame, _ServedGateway, _unused_port


@pytest.mark.parametrize("surface", ["chat", "responses"])
def test_azure_annotations_preserve_served_content_and_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    """Both public APIs consume trailing annotations before settling usage."""

    class Provider(BaseHTTPRequestHandler):
        """Serve synthetic Azure chunks without external credentials or calls."""

        def do_POST(self) -> None:  # noqa: N802
            """Return text, stop, annotation, usage, and the stream terminal."""
            self.rfile.read(int(self.headers["content-length"]))
            frames = (
                _provider_frame({"choices": [], "prompt_filter_results": []})
                + _provider_frame({"choices": [{"index": 0, "delta": {"content": "OK"}}]})
                + _provider_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                + _provider_frame(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": None,
                                "content_filter_results": {
                                    "hate": {"filtered": False, "severity": "safe"}
                                },
                                "content_filter_offsets": {
                                    "check_offset": 49,
                                    "start_offset": 47,
                                    "end_offset": 49,
                                },
                            }
                        ]
                    }
                )
                + _provider_frame(
                    {"choices": [], "usage": {"prompt_tokens": 13, "completion_tokens": 1}}
                )
                + b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(frames)))
            self.end_headers()
            self.wfile.write(frames)

        def log_message(self, format: str, *args: object) -> None:
            """Keep synthetic HTTP traffic out of test logs."""
            del format, args

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    _manager, raw_key = _configured_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        with OpenAI(
            api_key=raw_key, base_url=f"http://127.0.0.1:{gateway.port}/v1", max_retries=0
        ) as client:
            if surface == "chat":
                chat = client.chat.completions.create(
                    model="coding", messages=[{"role": "user", "content": "Reply OK"}]
                )
                assert chat.choices[0].message.content == "OK"
                assert chat.choices[0].finish_reason == "stop"
                assert chat.usage is not None
                assert (chat.usage.prompt_tokens, chat.usage.completion_tokens) == (13, 1)
            else:
                response = client.responses.create(model="coding", input="Reply OK", store=False)
                assert response.output_text == "OK"
                assert response.status == "completed"
                assert response.usage is not None
                assert (response.usage.input_tokens, response.usage.output_tokens) == (13, 1)
    finally:
        gateway.stop()
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)
