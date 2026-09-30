# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Real native loopback Fast requests settle the served tier on both OpenAI surfaces."""

from __future__ import annotations

import json
import sqlite3
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
from exp.common.models.catalog import (
    BillingSource,
    GatewayLongContextTier,
    GatewayServiceTierPrices,
)
from exp.runtime.gateway.catalog_authority import upsert_connection, upsert_singleton_deployment
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests.launch_test import _ServedGateway, _unused_port
from exp.runtime.gateway.tests.native_json_object_test import _frame
from exp.runtime.models import registry


def _configure(root: Path, provider: str, *, configured: bool) -> tuple[GatewayManagement, str]:
    """Create one house lane with a complete priority schedule and synthetic credentials."""
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider",
        connection=ConnectionConfig(provider=provider, api_key_env="TEST_PROVIDER_KEY"),
        replace=False,
    )
    tier = GatewayServiceTierPrices(
        input_nano_usd_per_million_tokens=2_000_000,
        cached_input_nano_usd_per_million_tokens=200_000,
        cache_creation_input_nano_usd_per_million_tokens=2_500_000,
        cache_creation_1h_input_nano_usd_per_million_tokens=2_500_000,
        output_nano_usd_per_million_tokens=4_000_000,
        long_context=GatewayLongContextTier(
            input_threshold_tokens=100,
            input_nano_usd_per_million_tokens=4_000_000,
            cached_input_nano_usd_per_million_tokens=400_000,
            cache_creation_input_nano_usd_per_million_tokens=5_000_000,
            cache_creation_1h_input_nano_usd_per_million_tokens=5_000_000,
            output_nano_usd_per_million_tokens=8_000_000,
        ),
    )
    normalized, snapshot, _ = upsert_singleton_deployment(
        root,
        deployment_alias="coding",
        connection_name="provider",
        provider_model="gpt-fixture",
        exact_model_id="gpt-fixture",
        revision=None,
        capabilities=ModelCapabilities(service_tier_pricing_enabled=configured),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            reports_cached_input_tokens=True,
            reports_cache_creation_input_tokens=True,
        ),
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=1_000_000,
            cached_input_nano_usd_per_million_tokens=100_000,
            cache_creation_input_nano_usd_per_million_tokens=1_250_000,
            cache_creation_1h_input_nano_usd_per_million_tokens=1_250_000,
            output_nano_usd_per_million_tokens=2_000_000,
            priority=tier if configured else None,
        ),
        billing_source=BillingSource.HOST_MANAGED,
        pricing_source="synthetic-fixture",
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="rev",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="coding")
    return manager, manager.issue_key(identity_id="default", key_id="key").raw_key


def _body(responses: bool, served: str | None, *, writes: int = 0) -> bytes:
    """The terminal meter states the served tier, after an earlier requested-tier echo."""
    if responses:
        terminal: JsonObject = {
            "id": "resp_fixture",
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": writes},
            },
        }
        if served is not None:
            terminal["service_tier"] = served
        frames: list[JsonObject] = [
            {
                "type": "response.created",
                "response": {"id": "resp_fixture", "service_tier": "priority"},
            },
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "item_id": "msg_fixture",
                "content_index": 0,
                "delta": "Hello",
            },
            {"type": "response.completed", "response": terminal},
        ]
        return b"".join(_frame(frame) for frame in frames)
    meter: JsonObject = {
        "choices": [],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 20, "cache_write_tokens": writes},
        },
    }
    if served is not None:
        meter["service_tier"] = served
    return (
        _frame({"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": "stop"}]})
        + _frame(meter)
        + b"data: [DONE]\n\n"
    )


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("surface", ["chat", "responses"])
@pytest.mark.parametrize("writes", [0, 60])
@pytest.mark.parametrize(
    "served,expected",
    [("priority", 408), ("fast", 408), ("default", 102), (None, None), ("mystery", None)],
)
def test_real_native_tier_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    stream: bool,
    surface: str,
    writes: int,
    served: str | None,
    expected: int | None,
) -> None:
    """Actual sockets prove canonical forwarding, exact settlement and unknown holds."""
    received: list[JsonObject] = []
    body = _body(provider == "openai", served, writes=writes)
    if expected is not None:
        expected += writes // 4 if served == "default" else writes

    class Provider(BaseHTTPRequestHandler):
        """Serve a finite provider stream without credentials or paid calls."""

        def do_POST(self) -> None:  # noqa: N802
            """Capture wire semantics and return controlled observed billing evidence."""
            received.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep synthetic requests quiet."""
            del format, args

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    if provider == "openai":
        monkeypatch.setattr(registry, "OPENAI_BASE_URL", base_url)
    else:
        factory, _ = registry._HTTP_PROVIDERS[provider]
        monkeypatch.setattr(
            registry,
            "_HTTP_PROVIDERS",
            {
                **registry._HTTP_PROVIDERS,
                provider: (factory, base_url),
            },
        )
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-key")
    manager, key = _configure(tmp_path, provider, configured=True)
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        path = "responses" if surface == "responses" else "chat/completions"
        payload: JsonObject = {"model": "coding", "stream": stream, "service_tier": "fast"}
        if surface == "responses":
            payload.update({"input": "Hello", "max_output_tokens": 16})
        else:
            payload.update({"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/{path}",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=10,
        )
        assert response.status_code == 200, response.text
        assert "Hello" in response.text
        assert len(received) == 1
        assert received[0]["service_tier"] == "priority"
    finally:
        gateway.stop()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)
    with sqlite3.connect(manager.database_path) as connection:
        row = connection.execute(
            "SELECT state, estimated_cost_nano_usd, budget_settled_nano_usd, "
            "service_tier_settlement, cache_creation_input_tokens, "
            "cache_creation_1h_input_tokens FROM gateway_attempts"
        ).fetchone()
    assert row[:3] == ("completed", expected, expected)
    assert row[4:] == (writes, None)
    receipt = json.loads(row[3])
    assert receipt["resolution"] == (
        "confirmed" if expected is not None else "missing" if served is None else "unknown"
    )


@pytest.mark.parametrize("surface", ["chat/completions", "responses"])
def test_unconfigured_fast_lane_is_refused_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    """A synthetic endpoint on a closed port must never be contacted without a tier card."""
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-key")
    monkeypatch.setattr(registry, "OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    manager, key = _configure(tmp_path, "openai", configured=False)
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        payload: JsonObject = {"model": "coding", "service_tier": "fast"}
        if surface == "responses":
            payload["input"] = "Hello"
        else:
            payload["messages"] = [{"role": "user", "content": "Hello"}]
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/{surface}",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=10,
        )
        assert response.status_code == 400, response.text
        assert response.json()["error"]["param"] == "service_tier"
    finally:
        gateway.stop()
    with sqlite3.connect(manager.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone() == (0,)
