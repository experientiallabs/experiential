"""Chat Completions custom tools served end to end through the compiled data plane."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import cast

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import GatewayDeploymentCapabilities
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests import native_waterfall_test as waterfall
from exp.runtime.gateway.tests.native_waterfall_test import _ServingEngine, _sse_frame

_LARK = 'start: "print(" /[0-9+]+/ ")"'
_INPUT = 'print(2+2)\n"café"'


class _ChatWireUpstream(BaseHTTPRequestHandler):
    """An OpenAI-compatible Chat wire that knows only function tools."""

    payloads: list[JsonObject] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Answer with the translated function, its JSON wrapper split mid-escape."""
        payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).payloads.append(payload)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        arguments = json.dumps({"input": _INPUT})
        fragments = [arguments[:12], arguments[12:15], arguments[15:]]
        start = {"index": 0, "id": "call-c", "type": "function"}
        self.wfile.write(
            _sse_frame(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {**start, "function": {"name": "code_exec", "arguments": ""}}
                                ]
                            },
                        }
                    ]
                }
            )
        )
        for fragment in fragments:
            self.wfile.write(
                _sse_frame(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": fragment}}
                                    ]
                                },
                            }
                        ]
                    }
                )
            )
        self.wfile.write(
            _sse_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        )
        self.wfile.write(waterfall._terminal_frames(prompt_tokens=7, completion_tokens=3))

    def log_message(self, format: str, *args: object) -> None:
        """Keep loopback request output quiet."""


@pytest.fixture(name="engine")
def chat_wire_engine(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_ServingEngine]:
    """Serve the real control plane over a certified pool of Chat-wire deployments."""
    original = waterfall._configured_pool_gateway

    def configured(
        root: Path, *, refusal_failover: bool, base_urls: tuple[str, str]
    ) -> tuple[GatewayManagement, str]:
        """Declare the fixture provider's real streaming tool capability."""
        capabilities = GatewayDeploymentCapabilities(
            supports_streaming=True, supports_streaming_tool_arguments=True
        )
        return original(
            root,
            refusal_failover=refusal_failover,
            base_urls=base_urls,
            gateway_capabilities=(capabilities, capabilities),
        )

    _ChatWireUpstream.payloads = []
    monkeypatch.setattr(waterfall, "_configured_pool_gateway", configured)
    monkeypatch.setattr(waterfall, "_PrimaryUpstream", _ChatWireUpstream)
    monkeypatch.setattr(waterfall, "_SecondaryUpstream", _ChatWireUpstream)
    fixture = cast(
        "Callable[[pytest.TempPathFactory], Iterator[_ServingEngine]]",
        inspect.unwrap(waterfall._engine),
    )
    yield from fixture(tmp_path_factory)


def _chat_custom_request(stream: bool) -> JsonObject:
    """The catalog drift audit's Chat custom-tool repro, forcing the custom tool."""
    return {
        "model": "coding",
        "stream": stream,
        "messages": [{"role": "user", "content": "Use code_exec to print 2+2."}],
        "tools": [
            {
                "type": "custom",
                "custom": {
                    "name": "code_exec",
                    "description": "Executes python code",
                    "format": {
                        "type": "grammar",
                        "grammar": {"syntax": "lark", "definition": _LARK},
                    },
                },
            }
        ],
        "tool_choice": {"type": "custom", "custom": {"name": "code_exec"}},
    }


@pytest.mark.parametrize("stream", [False, True])
def test_chat_custom_tool_is_translated_on_a_chat_wire_and_answered_in_chat_shape(
    engine: _ServingEngine, stream: bool
) -> None:
    """The provider sees a one-``input`` function; the caller sees its own custom call."""
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=_chat_custom_request(stream),
        timeout=30,
    )

    assert response.status_code == 200, response.text
    (sent,) = _ChatWireUpstream.payloads
    (tool,) = cast(list[JsonObject], sent["tools"])
    function = cast(JsonObject, tool["function"])
    assert function["name"] == "code_exec"
    parameters = cast(JsonObject, function["parameters"])
    assert parameters["required"] == ["input"]
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "code_exec"}}
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        calls = [
            call
            for chunk in chunks
            for call in chunk["choices"][0]["delta"].get("tool_calls", [])
            if chunk["choices"]
        ]
        assert calls == [
            {
                "index": 0,
                "id": "call-c",
                "type": "custom",
                "custom": {"name": "code_exec", "input": ""},
            },
            {"index": 0, "custom": {"input": _INPUT}},
        ]
        finishes = [
            chunk["choices"][0]["finish_reason"]
            for chunk in chunks
            if chunk["choices"] and chunk["choices"][0]["finish_reason"]
        ]
        assert finishes == ["tool_calls"]
        disclosed = chunks[0]["x-experiential-ignored-parameters"]
    else:
        body = response.json()
        assert body["choices"][0]["message"]["tool_calls"] == [
            {"id": "call-c", "type": "custom", "custom": {"name": "code_exec", "input": _INPUT}}
        ]
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        disclosed = body["x-experiential-ignored-parameters"]
    assert "tools.custom->translated(custom_as_function_input)" in disclosed
    assert "tools.custom.format->translated(grammar_as_instructions)" in disclosed
