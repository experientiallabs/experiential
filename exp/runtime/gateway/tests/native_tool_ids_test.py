"""Signed tool identifiers round-trip through the served gateway and official SDK."""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelCapabilities
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.tests.launch_test import _provider_frame, _ServedGateway, _unused_port


@pytest.mark.parametrize("delimiter", ["~sig1:", "_sig1_"])
@pytest.mark.parametrize("stream", [False, True])
def test_sdk_replays_signed_tool_id_through_native_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delimiter: str, stream: bool
) -> None:
    """A stock SDK tool loop preserves the signature on output and upstream replay."""
    call_id = "toolu_bdrk_synthetic" + delimiter + base64.b64encode(bytes(range(256)) * 3).decode()
    received: list[JsonObject] = []

    class Provider(BaseHTTPRequestHandler):
        """Emit one signed tool call, then a text answer on the tool result."""

        def do_POST(self) -> None:  # noqa: N802
            """Capture the provider-bound history and serve finite SSE frames."""
            received.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            if len(received) == 1:
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"},
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                delta = {"role": "assistant", "content": "done"}
                finish = "stop"
            frames = (
                _provider_frame(
                    {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                )
                + _provider_frame(
                    {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}}
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
        capabilities=ModelCapabilities(supports_tools=True, maximum_output_tokens=128_000),
    )
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        with OpenAI(
            api_key=raw_key, base_url=f"http://127.0.0.1:{gateway.port}/v1", max_retries=0
        ) as client:
            messages: list[ChatCompletionMessageParam] = [
                {"role": "user", "content": "use terminal"}
            ]
            if stream:
                identifier = ""
                arguments = ""
                name = ""
                with client.chat.completions.create(
                    model="coding",
                    messages=messages,
                    stream=True,
                    tools=[
                        {
                            "type": "function",
                            "function": {"name": "terminal", "parameters": {"type": "object"}},
                        }
                    ],
                ) as chunks:
                    for chunk in chunks:
                        for choice in chunk.choices:
                            for call in choice.delta.tool_calls or []:
                                identifier += call.id or ""
                                if call.function:
                                    arguments += call.function.arguments or ""
                                    name += call.function.name or ""
                assistant: ChatCompletionMessageParam = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": identifier,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
                assert identifier == call_id
            else:
                response = client.chat.completions.create(
                    model="coding",
                    messages=messages,
                    tools=[
                        {
                            "type": "function",
                            "function": {"name": "terminal", "parameters": {"type": "object"}},
                        }
                    ],
                )
                output = response.choices[0].message
                assert output.tool_calls and output.tool_calls[0].id == call_id
                # The SDK's serialized output is exactly what stock tool loops echo.
                assistant = cast(ChatCompletionMessageParam, output.model_dump())
            messages.extend([assistant, {"role": "tool", "tool_call_id": call_id, "content": "ok"}])
            final = client.chat.completions.create(model="coding", messages=messages)
            assert final.choices[0].message.content == "done"
        assert len(received) == 2
        replay = cast(list[JsonObject], received[1]["messages"])
        calls = cast(list[JsonObject], replay[-2]["tool_calls"])
        assert calls[0]["id"] == replay[-1]["tool_call_id"] == call_id
    finally:
        gateway.stop()
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)
