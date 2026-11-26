"""Declared vLLM sampling controls survive the actual native HTTP serving path."""

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
from exp.runtime.gateway.tests.native_json_object_test import _provider_response


def _configure(root: Path, base_url: str, declared: bool) -> str:
    """Create one explicitly qualified FIFO route with a synthetic provider credential.

    Args:
        root: Isolated test gateway workspace.
        base_url: Loopback provider endpoint that records native HTTP requests.
        declared: Whether the route explicitly supports vLLM sampling extras.

    Returns:
        The synthetic customer key granted access to the configured alias.
    """
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider",
        connection=ConnectionConfig(
            provider="openai-compatible", base_url=base_url, api_key_env="TEST_PROVIDER_KEY"
        ),
        replace=False,
    )
    normalized, snapshot, _ = upsert_singleton_deployment(
        root,
        deployment_alias="coding",
        connection_name="provider",
        provider_model="fixture-adapter",
        exact_model_id="fixture-revision",
        revision=None,
        capabilities=ModelCapabilities(
            maximum_output_tokens=32768,
            supports_temperature=True,
            supports_top_p=True,
            supports_top_k=True,
            supports_presence_penalty=True,
        ),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True, supports_vllm_sampling=declared
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


@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_vllm_controls_reach_the_wire_without_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: bool, stream: bool
) -> None:
    """Both response modes preserve qualified controls and disclose unqualified drops.

    Args:
        tmp_path: Isolated test gateway workspace.
        monkeypatch: Fixture supplying a synthetic provider credential.
        declared: Whether to qualify the route for vLLM sampling extras.
        stream: Whether the customer requests streamed response events.
    """
    received: list[JsonObject] = []
    body = _provider_response("openai-compatible")

    class Provider(BaseHTTPRequestHandler):
        """Capture provider-bound requests without any paid inference."""

        def do_POST(self) -> None:  # noqa: N802
            """Record the actual native socket payload and return one finite answer."""
            received.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep synthetic traffic out of test logs."""
            del format, args

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    gateway: _ServedGateway | None = None
    try:
        key = _configure(tmp_path, f"http://127.0.0.1:{upstream.server_port}/v1", declared)
        gateway = _ServedGateway(tmp_path, _unused_port())
        gateway.start()
        controls: JsonObject = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.1,
            "min_p": 0.05,
            "max_tokens": 32768,
        }
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": "Give an answer."}],
                "stream": stream,
                **controls,
            },
            timeout=10,
        )
        assert response.status_code == 200, response.text
        assert "ready" in response.text
        assert len(received) == 1
        assert "priority" not in received[0]
        for name, value in controls.items():
            if not declared and name in {"repetition_penalty", "min_p"}:
                assert name not in received[0]
            else:
                assert received[0][name] == value
        envelopes = (
            [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            if stream
            else [response.json()]
        )
        disclosures = {
            value
            for envelope in envelopes
            for value in envelope.get("x-experiential-ignored-parameters", [])
        }
        assert disclosures == (
            set()
            if declared
            else {
                "repetition_penalty->dropped(unsupported_by_provider)",
                "min_p->dropped(unsupported_by_provider)",
            }
        )
    finally:
        if gateway is not None:
            gateway.stop()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)
