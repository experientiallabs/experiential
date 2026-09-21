"""Gateway-executed tool search traverses admission, the withheld round trip, and rendering.

A loopback OpenAI-compatible provider serves two turns: first a ``tool_search``
call (which the gateway must withhold and answer itself), then a text answer
that calls nothing. The real Rust data plane re-dials the same rung with the
extended conversation and renders the round trip per surface.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.catalog_authority import upsert_connection, upsert_singleton_deployment
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests.launch_test import _ServedGateway, _unused_port

_ANSWER = "It is 18C in Bern."


def _frame(payload: JsonObject) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _search_call_turn(tool_name: str) -> bytes:
    return (
        _frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_search_1",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": '{"query":"current weather"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        + _frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        + _frame({"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 8}})
        + b"data: [DONE]\n\n"
    )


def _text_turn() -> bytes:
    return (
        _frame({"choices": [{"index": 0, "delta": {"content": _ANSWER}, "finish_reason": "stop"}]})
        + _frame({"choices": [], "usage": {"prompt_tokens": 90, "completion_tokens": 7}})
        + b"data: [DONE]\n\n"
    )


class _Provider:
    def __init__(self) -> None:
        self.requests: list[Any] = []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                provider.requests.append(body)
                tool_names = [
                    tool.get("function", {}).get("name") or tool.get("name")
                    for tool in body.get("tools", [])
                ]
                has_result = any(message.get("role") == "tool" for message in body["messages"])
                payload = (
                    _text_turn()
                    if has_result or "tool_search" not in tool_names
                    else _search_call_turn("tool_search")
                )
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _configure(root: Path, base_url: str) -> str:
    """Configure the tool-capable fixture alias and return its authorized client key."""
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider-main",
        connection=ConnectionConfig(
            provider="openai-compatible", base_url=base_url, api_key_env="TEST_PROVIDER_KEY"
        ),
        replace=False,
    )
    normalized, snapshot, _ = upsert_singleton_deployment(
        root,
        deployment_alias="coding",
        connection_name="provider-main",
        provider_model="provider-model-exact",
        exact_model_id="model-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(supports_tools=True, maximum_output_tokens=128_000),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True, supports_streaming_tool_arguments=True
        ),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-one",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="coding")
    return manager.issue_key(identity_id="default", key_id="key-one").raw_key


_TOOLS_CHAT = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email",
            "parameters": {"type": "object"},
        },
    },
    {
        "type": "function",
        "defer_loading": True,
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "defer_loading": True,
        "function": {
            "name": "book_flight",
            "description": "Book a flight",
            "parameters": {"type": "object"},
        },
    },
    {"type": "openrouter:tool_search"},
]


@pytest.mark.parametrize("stream", [False, True])
def test_chat_tool_search_round_trip_is_served_by_the_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    provider = _Provider()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    raw_key = _configure(tmp_path, f"http://127.0.0.1:{provider.server.server_port}/v1")
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "stream": stream,
                **({"stream_options": {"include_usage": True}} if stream else {}),
                "messages": [{"role": "user", "content": "What's the weather in Bern?"}],
                "tools": _TOOLS_CHAT,
            },
            timeout=30,
        )
        assert response.status_code == 200, response.text
        assert len(provider.requests) == 2
        first, second = provider.requests
        first_names = [tool["function"]["name"] for tool in first["tools"]]
        assert first_names == ["send_email", "tool_search"]
        assert all("defer_loading" not in tool for tool in first["tools"])
        second_names = [tool["function"]["name"] for tool in second["tools"]]
        assert "get_weather" in second_names and "book_flight" not in second_names
        roles = [message["role"] for message in second["messages"]]
        assert roles == ["user", "assistant", "tool"]
        assert second["messages"][1]["tool_calls"][0]["function"]["name"] == "tool_search"
        assert "get_weather" in second["messages"][2]["content"]
        if stream:
            chunks = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            content = "".join(
                chunk["choices"][0]["delta"].get("content", "")
                for chunk in chunks
                if chunk.get("choices")
            )
            assert content == _ANSWER
            # The withheld search call never reaches the caller.
            assert not any(
                "tool_calls" in chunk["choices"][0]["delta"]
                for chunk in chunks
                if chunk.get("choices")
            )
            usage = next(chunk["usage"] for chunk in chunks if chunk.get("usage"))
        else:
            body = response.json()
            message = body["choices"][0]["message"]
            assert message["content"] == _ANSWER
            assert "tool_calls" not in message
            assert body["choices"][0]["finish_reason"] == "stop"
            usage = body["usage"]
        assert usage["server_tool_use_details"] == {"tool_search_requests": 1}
    finally:
        gateway.stop()
        provider.close()


def test_messages_tool_search_renders_the_round_trip_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _Provider()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    raw_key = _configure(tmp_path, f"http://127.0.0.1:{provider.server.server_port}/v1")
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/messages",
            headers={"x-api-key": raw_key, "anthropic-version": "2023-06-01"},
            json={
                "model": "coding",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "What's the weather in Bern?"}],
                "tools": [
                    {"name": "send_email", "input_schema": {"type": "object"}},
                    {
                        "name": "get_weather",
                        "description": "Current weather for a city",
                        "input_schema": {"type": "object"},
                        "defer_loading": True,
                    },
                    {"type": "tool_search_tool_bm25", "name": "tool_search_tool_bm25"},
                ],
            },
            timeout=30,
        )
        assert response.status_code == 200, response.text
        assert len(provider.requests) == 2
        body = response.json()
        kinds = [block["type"] for block in body["content"]]
        assert kinds == ["server_tool_use", "tool_search_tool_result", "text"]
        use, result, text = body["content"]
        assert use["name"] == "tool_search_tool_bm25"
        assert use["input"] == {"query": "current weather"}
        assert result["tool_use_id"] == use["id"]
        assert result["content"]["type"] == "tool_search_tool_search_result"
        assert [ref["tool_name"] for ref in result["content"]["tool_references"]] == ["get_weather"]
        assert text["text"] == _ANSWER
        assert body["usage"]["server_tool_use"] == {"tool_search_requests": 1}
        assert body["stop_reason"] == "end_turn"
    finally:
        gateway.stop()
        provider.close()


def test_responses_tool_search_renders_hosted_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _Provider()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    raw_key = _configure(tmp_path, f"http://127.0.0.1:{provider.server.server_port}/v1")
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/responses",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "input": "What's the weather in Bern?",
                "store": False,
                "tools": [
                    {"type": "function", "name": "send_email", "parameters": {"type": "object"}},
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Current weather for a city",
                        "parameters": {"type": "object"},
                        "defer_loading": True,
                    },
                    {"type": "tool_search"},
                ],
            },
            timeout=30,
        )
        assert response.status_code == 200, response.text
        assert len(provider.requests) == 2
        body = response.json()
        kinds = [item["type"] for item in body["output"]]
        assert kinds[:2] == ["tool_search_call", "tool_search_output"]
        assert kinds[-1] == "message"
        call, output = body["output"][0], body["output"][1]
        assert call["call_id"] == output["call_id"]
        assert [tool["name"] for tool in output["tools"]] == ["get_weather"]
        assert body["output"][-1]["content"][0]["text"] == _ANSWER
        assert body["usage"]["server_tool_use_details"] == {"tool_search_requests": 1}
    finally:
        gateway.stop()
        provider.close()
