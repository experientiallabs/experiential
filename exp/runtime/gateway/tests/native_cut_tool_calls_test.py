"""Relay tool-call shapes that used to 502 settle through the served native gateway."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from openai import OpenAI

from exp.common.models import ModelCapabilities
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.tests.launch_test import _provider_frame, _ServedGateway, _unused_port

_MINTED_ID_AND_PHANTOM = (
    _provider_frame(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            # Z.ai GLM relays open the call with a null id.
                            {
                                "index": 0,
                                "id": None,
                                "type": "function",
                                "function": {"name": "terminal", "arguments": "{}"},
                            },
                            # OpenRouter's GLM/Hunyuan relays add an empty placeholder.
                            {
                                "index": 1,
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    + _provider_frame({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}})
    + b"data: [DONE]\n\n"
)

# Content, then a call cut mid-value, then the provider closes the stream with
# no finish frame and no `[DONE]` (gpt-5.6-luna's post-commit shape).
_CLOSED_WITHOUT_TERMINAL = _provider_frame(
    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "par"}}]}
) + _provider_frame(
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": '{"path": "a'},
                        }
                    ]
                },
            }
        ]
    }
)


@pytest.mark.parametrize("shape", ["minted_id_and_phantom", "closed_without_terminal"])
def test_relay_tool_call_shapes_settle_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """A null-id call is served under a minted id, a phantom entry vanishes, and a
    stream closed on a cut call answers ``finish_reason: length`` with its text."""
    frames = {
        "minted_id_and_phantom": _MINTED_ID_AND_PHANTOM,
        "closed_without_terminal": _CLOSED_WITHOUT_TERMINAL,
    }[shape]

    class Provider(BaseHTTPRequestHandler):
        """Serve the synthetic relay frames without external credentials or calls."""

        def do_POST(self) -> None:  # noqa: N802
            """Return the fixed frames and close the stream."""
            self.rfile.read(int(self.headers["content-length"]))
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
            response = client.chat.completions.create(
                model="coding",
                messages=[{"role": "user", "content": "use terminal"}],
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "terminal", "parameters": {"type": "object"}},
                    }
                ],
            )
            choice = response.choices[0]
            if shape == "minted_id_and_phantom":
                assert choice.finish_reason == "tool_calls"
                assert choice.message.tool_calls is not None
                assert len(choice.message.tool_calls) == 1
                call = choice.message.tool_calls[0]
                assert call.type == "function"
                assert call.id.startswith("call_gw0_")
                assert call.function.name == "terminal"
                assert call.function.arguments == "{}"
            else:
                assert choice.finish_reason == "length"
                assert choice.message.content == "par"
                assert not choice.message.tool_calls
    finally:
        gateway.stop()
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)
