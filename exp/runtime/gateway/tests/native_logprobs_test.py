"""Served Chat logprob parity through the native gateway and official SDK."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from exp.common.models import ModelCapabilities
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.tests.launch_test import _provider_frame, _ServedGateway, _unused_port


def test_served_chat_logprobs_preserve_nonstream_and_sdk_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capable compatible upstream preserves exact token probabilities end to end."""

    class Provider(BaseHTTPRequestHandler):
        """Serve a deterministic OpenAI-compatible probability stream."""

        requests: list[dict[str, object]] = []

        def do_POST(self) -> None:  # noqa: N802
            """Capture the frozen request and return bounded probability frames."""
            import json

            length = int(self.headers.get("content-length", "0"))
            payload = cast(dict[str, object], json.loads(self.rfile.read(length)))
            type(self).requests.append(payload)
            frames = (
                _provider_frame({"choices": []})
                + _provider_frame(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _provider_frame(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "é"},
                                "logprobs": {
                                    "content": [
                                        {
                                            "token": "é",
                                            "logprob": -0.25,
                                            "bytes": [195, 169],
                                            "top_logprobs": [
                                                {"token": "e", "logprob": -1.25, "bytes": [101]}
                                            ],
                                        }
                                    ],
                                    "refusal": None,
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _provider_frame(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "logprobs": {"content": [], "refusal": []},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                )
                + _provider_frame({"choices": []})
                + _provider_frame(
                    {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
                )
                + b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(frames)))
            self.end_headers()
            self.wfile.write(frames)

        def log_message(self, format: str, *args: object) -> None:
            """Keep synthetic provider traffic out of test logs."""
            del format, args

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    _manager, raw_key = _configured_gateway(
        tmp_path,
        base_url=f"http://127.0.0.1:{provider.server_port}/v1",
        capabilities=ModelCapabilities(supports_logprobs=True),
    )
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        with OpenAI(
            api_key=raw_key,
            base_url=f"http://127.0.0.1:{gateway.port}/v1",
            max_retries=0,
        ) as client:
            messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "prob"}]
            response = client.chat.completions.create(
                model="coding",
                messages=messages,
                logprobs=True,
                top_logprobs=1,
            )
            assert response.choices[0].logprobs is not None
            assert response.choices[0].logprobs.content is not None
            assert len(response.choices[0].logprobs.content) == 1
            record = response.choices[0].logprobs.content[0]
            assert record.token == "é"
            assert record.logprob == -0.25
            assert record.bytes == [195, 169]
            assert len(record.top_logprobs) == 1
            assert record.top_logprobs[0].token == "e"
            assert record.top_logprobs[0].logprob == -1.25

            chunks = list(
                client.chat.completions.create(
                    model="coding",
                    messages=messages,
                    logprobs=True,
                    top_logprobs=1,
                    stream=True,
                )
            )
            streamed = [
                chunk.choices[0].logprobs
                for chunk in chunks
                if chunk.choices and chunk.choices[0].logprobs is not None
            ]
            assert len(streamed) == 2
            assert streamed[0] is not None and streamed[0].content is not None
            assert streamed[0].content[0].token == "é"
            with client.chat.completions.stream(
                model="coding",
                messages=messages,
                logprobs=True,
                top_logprobs=1,
            ) as stream:
                final = stream.get_final_completion()
            assert final.choices[0].logprobs is not None
            assert final.choices[0].logprobs.content is not None
            assert final.choices[0].logprobs.content[0].bytes == [195, 169]
        assert all(request["logprobs"] is True for request in Provider.requests)
        assert all(request["top_logprobs"] == 1 for request in Provider.requests)
    finally:
        gateway.stop()
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)
