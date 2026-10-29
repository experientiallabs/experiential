"""Schema-free JSON requests traverse native serving and provider-specific wires."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

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
from exp.runtime.gateway.json_object import JSON_OBJECT_SYSTEM_INSTRUCTION
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests.launch_test import _ServedGateway, _unused_port
from exp.runtime.models import registry

_ANSWER = '{"answer":"ready"}'


def _frame(payload: JsonObject) -> bytes:
    """Encode one finite provider event for the actual native stream reader."""
    return f"data: {json.dumps(payload)}\n\n".encode()


def _provider_response(provider: str) -> bytes:
    """Return a JSON answer using the selected provider's real event vocabulary."""
    frames: list[JsonObject]
    if provider == "anthropic":
        frames = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_fixture",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 100, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": _ANSWER},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]
    elif provider == "gemini":
        frames = [
            {"candidates": [{"content": {"parts": [{"text": _ANSWER}]}}]},
            {
                "candidates": [{"finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 5},
            },
        ]
    else:
        frames = [
            {"choices": [{"index": 0, "delta": {"content": _ANSWER}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 5}},
        ]
    return b"".join(_frame(frame) for frame in frames) + (
        b"data: [DONE]\n\n" if provider == "openai-compatible" else b""
    )


def _configure_json_gateway(root: Path, provider: str, base_url: str) -> str:
    """Seed an explicit gateway alias using a real provider adapter and synthetic key."""
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider-main",
        connection=ConnectionConfig(
            provider=provider,
            base_url=base_url if provider == "openai-compatible" else None,
            api_key_env="TEST_PROVIDER_KEY",
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
        capabilities=ModelCapabilities(maximum_output_tokens=128_000),
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


@pytest.mark.parametrize("provider", ["openai-compatible", "anthropic", "gemini"])
@pytest.mark.parametrize("stream", [False, True])
def test_json_object_serves_through_native_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str, stream: bool
) -> None:
    """Decode, admission, provider shaping, native streaming and JSON aggregation agree."""
    received: list[JsonObject] = []
    provider_body = _provider_response(provider)

    class Provider(BaseHTTPRequestHandler):
        """Record provider-bound requests and return one finite, billed JSON answer."""

        def do_POST(self) -> None:  # noqa: N802
            """Capture the request that the native plane actually sends on the socket."""
            received.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            if (
                provider == "openai-compatible"
                and "json" not in json.dumps(received[-1].get("messages")).lower()
            ):
                # Native JSON mode requires an explicit JSON instruction in input.
                self.send_error(400, "messages must mention JSON for json_object mode")
                return
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
    factory, _ = registry._HTTP_PROVIDERS[provider]
    monkeypatch.setattr(
        registry, "_HTTP_PROVIDERS", {**registry._HTTP_PROVIDERS, provider: (factory, base_url)}
    )
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    raw_key = _configure_json_gateway(tmp_path, provider, base_url)
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "stream": stream,
                "messages": [{"role": "user", "content": "Give the answer."}],
                "response_format": {"type": "json_object"},
            },
            timeout=10,
        )
        assert response.status_code == 200, response.text
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
            assert response.text.endswith("data: [DONE]\n\n")
        else:
            content = response.json()["choices"][0]["message"]["content"]
        assert json.loads(content) == {"answer": "ready"}
        envelopes = chunks if stream else [response.json()]
        disclosures = {
            value
            for envelope in envelopes
            for value in envelope.get("x-experiential-ignored-parameters", [])
        }
        assert ("response_format->instruction(json_object)" in disclosures) == (
            provider == "anthropic"
        )
        assert len(received) == 1
        payload = received[0]
        if provider == "openai-compatible":
            assert payload["response_format"] == {"type": "json_object"}
            messages = cast(list[JsonObject], payload["messages"])
            assert messages[0] == {"role": "system", "content": JSON_OBJECT_SYSTEM_INSTRUCTION}
        elif provider == "anthropic":
            assert payload["system"] == JSON_OBJECT_SYSTEM_INSTRUCTION
            assert "output_config" not in payload
        else:
            generation = cast(JsonObject, payload["generationConfig"])
            assert generation["responseMimeType"] == "application/json"
            assert payload["systemInstruction"] == {
                "parts": [{"text": JSON_OBJECT_SYSTEM_INSTRUCTION}]
            }
            assert "responseJsonSchema" not in generation
    finally:
        gateway.stop()
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)
