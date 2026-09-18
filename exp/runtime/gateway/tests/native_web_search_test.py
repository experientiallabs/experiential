"""Gateway-executed web search traverses admission, injection, and every public surface.

A loopback Exa fixture answers the search, a loopback OpenAI-compatible provider
answers the model turn (citing one of the injected URLs), and the real Rust data
plane renders citations, blocks, and the per-search count on Chat, Responses, and
Messages. Without ``EXA_API_KEY`` the same request serves with a disclosed drop.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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

_URL = "https://rust-lang.org/blog/rust-1.99"
_ANSWER = f"Rust 1.99 shipped this week ([rust-lang.org]({_URL}))."


def _frame(payload: JsonObject) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _provider_body() -> bytes:
    return (
        _frame({"choices": [{"index": 0, "delta": {"content": _ANSWER}, "finish_reason": "stop"}]})
        + _frame({"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 9}})
        + b"data: [DONE]\n\n"
    )


class _Fixtures:
    """Loopback Exa and provider servers recording what the gateway sent them."""

    def __init__(self) -> None:
        self.search_requests: list[JsonObject] = []
        self.provider_requests: list[JsonObject] = []
        fixtures = self

        class Exa(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                fixtures.search_requests.append(
                    json.loads(self.rfile.read(int(self.headers["content-length"])))
                )
                assert self.headers["x-api-key"] == "exa-fixture-key"
                body = json.dumps(
                    {
                        "results": [
                            {
                                "url": _URL,
                                "title": "Announcing Rust 1.99",
                                "highlights": ["Rust 1.99 is out."],
                            },
                            {"url": "https://example.com/unused", "title": "Unused"},
                        ]
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        class Provider(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                fixtures.provider_requests.append(
                    json.loads(self.rfile.read(int(self.headers["content-length"])))
                )
                body = _provider_body()
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self.exa = ThreadingHTTPServer(("127.0.0.1", 0), Exa)
        self.provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
        self.threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in (self.exa, self.provider)
        ]
        for thread in self.threads:
            thread.start()

    def close(self) -> None:
        for server in (self.exa, self.provider):
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=5)


def _configure(root: Path, base_url: str) -> str:
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
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
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


def _injected_instruction(provider_request: JsonObject) -> str:
    messages = provider_request["messages"]
    assert isinstance(messages, list)
    instructions = [
        message["content"]
        for message in messages
        if isinstance(message, dict) and message.get("role") in {"system", "developer"}
    ]
    assert instructions, provider_request
    return "\n".join(str(text) for text in instructions)


@pytest.mark.parametrize("stream", [False, True])
def test_chat_web_search_injects_results_and_cites_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    fixtures = _Fixtures()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    monkeypatch.setenv("EXA_API_KEY", "exa-fixture-key")
    monkeypatch.setenv("EXA_SEARCH_URL", f"http://127.0.0.1:{fixtures.exa.server_port}/search")
    raw_key = _configure(tmp_path, f"http://127.0.0.1:{fixtures.provider.server_port}/v1")
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding:online",
                "stream": stream,
                **({"stream_options": {"include_usage": True}} if stream else {}),
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "What is the latest Rust release?"},
                ],
                "plugins": [{"id": "web", "max_results": 2}],
            },
            timeout=20,
        )
        assert response.status_code == 200, response.text
        assert fixtures.search_requests[0]["query"] == "What is the latest Rust release?"
        assert fixtures.search_requests[0]["numResults"] == 2
        sent = fixtures.provider_requests[0]
        assert "plugins" not in sent and "web_search_options" not in sent
        assert sent["model"] == "provider-model-exact"
        instruction = _injected_instruction(sent)
        assert "Be brief." in instruction
        assert _URL in instruction and "Announcing Rust 1.99" in instruction
        expected_annotation = {
            "type": "url_citation",
            "url_citation": {
                "url": _URL,
                "title": "Announcing Rust 1.99",
                "start_index": _ANSWER.index(_URL),
                "end_index": _ANSWER.index(_URL) + len(_URL),
            },
        }
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
            annotation_chunks = [
                chunk["choices"][0]["delta"]["annotations"]
                for chunk in chunks
                if chunk.get("choices") and "annotations" in chunk["choices"][0]["delta"]
            ]
            assert annotation_chunks == [[expected_annotation]]
            usage = next(chunk["usage"] for chunk in chunks if chunk.get("usage"))
        else:
            body = response.json()
            message = body["choices"][0]["message"]
            assert message["content"] == _ANSWER
            assert message["annotations"] == [expected_annotation]
            usage = body["usage"]
        assert usage["server_tool_use_details"] == {"web_search_requests": 1}
        assert usage["prompt_tokens"] == 120
    finally:
        gateway.stop()
        fixtures.close()


def test_responses_web_search_tool_is_served_by_the_gateway_on_a_foreign_rung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _Fixtures()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    monkeypatch.setenv("EXA_API_KEY", "exa-fixture-key")
    monkeypatch.setenv("EXA_SEARCH_URL", f"http://127.0.0.1:{fixtures.exa.server_port}/search")
    raw_key = _configure(tmp_path, f"http://127.0.0.1:{fixtures.provider.server_port}/v1")
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/responses",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "input": "What is the latest Rust release?",
                "tools": [{"type": "web_search", "search_context_size": "low"}],
                "store": False,
            },
            timeout=20,
        )
        assert response.status_code == 200, response.text
        assert fixtures.search_requests[0]["numResults"] == 3
        sent = fixtures.provider_requests[0]
        # The hosted tool never reaches an OpenAI-compatible rung; the results do.
        assert not sent.get("tools")
        assert _URL in _injected_instruction(sent)
        body = response.json()
        message = next(item for item in body["output"] if item["type"] == "message")
        part = message["content"][0]
        assert part["text"] == _ANSWER
        assert part["annotations"] == [
            {
                "type": "url_citation",
                "url": _URL,
                "title": "Announcing Rust 1.99",
                "start_index": _ANSWER.index(_URL),
                "end_index": _ANSWER.index(_URL) + len(_URL),
            }
        ]
        assert body["usage"]["server_tool_use_details"] == {"web_search_requests": 1}
    finally:
        gateway.stop()
        fixtures.close()


def test_messages_web_search_server_tool_is_served_by_the_gateway_on_a_foreign_rung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _Fixtures()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    monkeypatch.setenv("EXA_API_KEY", "exa-fixture-key")
    monkeypatch.setenv("EXA_SEARCH_URL", f"http://127.0.0.1:{fixtures.exa.server_port}/search")
    raw_key = _configure(tmp_path, f"http://127.0.0.1:{fixtures.provider.server_port}/v1")
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/messages",
            headers={"x-api-key": raw_key, "anthropic-version": "2023-06-01"},
            json={
                "model": "coding",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "What is the latest Rust release?"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            },
            timeout=20,
        )
        assert response.status_code == 200, response.text
        sent = fixtures.provider_requests[0]
        assert not sent.get("tools")
        assert _URL in _injected_instruction(sent)
        body = response.json()
        kinds = [block["type"] for block in body["content"]]
        assert kinds == ["server_tool_use", "web_search_tool_result", "text"]
        use, result, text = body["content"]
        assert use["name"] == "web_search"
        assert use["input"] == {"query": "What is the latest Rust release?"}
        assert result["tool_use_id"] == use["id"]
        assert [hit["url"] for hit in result["content"]] == [_URL, "https://example.com/unused"]
        assert result["content"][0]["type"] == "web_search_result"
        assert text["text"] == _ANSWER
        assert body["usage"]["server_tool_use"] == {"web_search_requests": 1}
        ignored = body.get("x-experiential-ignored-parameters", [])
        assert not any(entry.startswith("web_search->") for entry in ignored)
    finally:
        gateway.stop()
        fixtures.close()


def test_without_a_backend_the_ask_is_dropped_with_disclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _Fixtures()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    raw_key = _configure(tmp_path, f"http://127.0.0.1:{fixtures.provider.server_port}/v1")
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": "What is the latest Rust release?"}],
                "web_search_options": {"search_context_size": "medium"},
            },
            timeout=20,
        )
        assert response.status_code == 200, response.text
        assert fixtures.search_requests == []
        body = response.json()
        assert (
            "web_search->dropped(search_unavailable)" in body["x-experiential-ignored-parameters"]
        )
        assert "annotations" not in body["choices"][0]["message"]
        assert "server_tool_use_details" not in body["usage"]
        assert "web_search_options" not in fixtures.provider_requests[0]
    finally:
        gateway.stop()
        fixtures.close()
