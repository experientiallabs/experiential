"""A relay's trailing metadata chunk with ``choices: null`` settles the served answer."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from openai import OpenAI

from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.tests.launch_test import _provider_frame, _ServedGateway, _unused_port

# The exact key set from the production operator line (Novita
# deepseek-v4.1-flash, 48 attempts on 2026-09-15): after the finish chunk the
# relay appends a chunk with ``choices: null``, no usage, and its own
# ``sla_metrics``. It carries nothing a decoder needs.
_FRAMES = (
    _provider_frame(
        {
            "id": "chatcmpl-novita",
            "object": "chat.completion.chunk",
            "created": 1789000000,
            "model": "deepseek/deepseek-v4.1-flash",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}}],
        }
    )
    + _provider_frame(
        {
            "id": "chatcmpl-novita",
            "object": "chat.completion.chunk",
            "created": 1789000000,
            "model": "deepseek/deepseek-v4.1-flash",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 1},
        }
    )
    + _provider_frame(
        {
            "id": "chatcmpl-novita",
            "object": "chat.completion.chunk",
            "created": 1789000000,
            "model": "deepseek/deepseek-v4.1-flash",
            "system_fingerprint": None,
            "choices": None,
            "sla_metrics": {"ttft_ms": 412, "tpot_ms": 9, "tokens_per_second": 108.3},
        }
    )
    + b"data: [DONE]\n\n"
)


@pytest.mark.parametrize("surface", ["chat", "responses"])
def test_trailing_null_choices_metadata_frame_settles_by_the_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    """Both public APIs answer the finished text with its usage, never a 502."""

    class Provider(BaseHTTPRequestHandler):
        """Serve the captured Novita frame sequence without external calls."""

        def do_POST(self) -> None:  # noqa: N802
            """Return text, finish + usage, the sla_metrics chunk, and [DONE]."""
            self.rfile.read(int(self.headers["content-length"]))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(_FRAMES)))
            self.end_headers()
            self.wfile.write(_FRAMES)

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
