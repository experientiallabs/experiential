"""Real loopback Anthropic usage reaches durable cache-write settlement and replay."""

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
from exp.runtime.gateway.catalog_authority import upsert_connection, upsert_singleton_deployment
from exp.runtime.gateway.contracts import GatewayEvent, GatewayEventKind, GatewayUsage
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.platform import AttemptSettlementRequest
from exp.runtime.gateway.sqlite.platform import SQLiteGatewayPlatform
from exp.runtime.gateway.tests.launch_test import _ServedGateway, _unused_port
from exp.runtime.gateway.tests.native_json_object_test import _frame, _provider_response
from exp.runtime.models import registry


def _configure(root: Path, *, hour_price: int | None) -> tuple[GatewayManagement, str]:
    """Create a local Anthropic alias with explicit prices and a synthetic credential.

    Args:
        root: Isolated test gateway state directory.
        hour_price: Explicit one-hour price, absent in the unpriced regression.

    Returns:
        Management handle and the issued local virtual key.
    """
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider",
        connection=ConnectionConfig(provider="anthropic", api_key_env="TEST_PROVIDER_KEY"),
        replace=False,
    )
    normalized, snapshot, _ = upsert_singleton_deployment(
        root,
        deployment_alias="coding",
        connection_name="provider",
        provider_model="claude-test",
        exact_model_id="claude-test",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            reports_cached_input_tokens=True,
            reports_cache_creation_input_tokens=True,
        ),
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=3_000_000_000,
            cached_input_nano_usd_per_million_tokens=300_000_000,
            cache_creation_input_nano_usd_per_million_tokens=3_750_000_000,
            cache_creation_1h_input_nano_usd_per_million_tokens=hour_price,
            output_nano_usd_per_million_tokens=15_000_000_000,
        ),
        pricing_source="synthetic-published-rate-fixture",
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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "breakdown,hour_price,expected_hour,expected_cost",
    [
        (
            {"ephemeral_5m_input_tokens": 400, "ephemeral_1h_input_tokens": 200},
            6_000_000_000,
            200,
            3_780_000,
        ),
        (
            {"ephemeral_5m_input_tokens": 600, "ephemeral_1h_input_tokens": 0},
            6_000_000_000,
            0,
            3_330_000,
        ),
        (None, 6_000_000_000, None, None),
        ({"ephemeral_1h_input_tokens": 200}, 6_000_000_000, None, None),
        ({"ephemeral_5m_input_tokens": 400, "ephemeral_1h_input_tokens": 200}, None, 200, None),
    ],
)
def test_native_cache_write_settlement_uses_observed_ttl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    breakdown: JsonObject | None,
    hour_price: int | None,
    expected_hour: int | None,
    expected_cost: int | None,
) -> None:
    """HTTP, native normalization, nano-USD storage, and exact settlement replay agree."""
    frames = [
        json.loads(line[6:])
        for line in _provider_response("anthropic").splitlines()
        if line.startswith(b"data: ")
    ]
    frames[0]["message"]["usage"] = {
        "input_tokens": 50,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 300,
        "output_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200},
    }
    # Final meters supersede the smaller start-frame totals. Missing final TTL
    # evidence must invalidate the earlier known breakdown rather than reuse it.
    frames[-2]["usage"] = {
        "input_tokens": 300,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 600,
        "output_tokens": 10,
    }
    if breakdown is not None:
        frames[-2]["usage"]["cache_creation"] = breakdown
    body = b"".join(_frame(frame) for frame in frames)
    requests: list[JsonObject] = []

    class Provider(BaseHTTPRequestHandler):
        """Serve one finite Anthropic lifecycle with controlled accounting evidence."""

        def do_POST(self) -> None:  # noqa: N802
            """Capture the real provider-bound request and stream its synthetic usage."""
            requests.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep synthetic prompt content out of logs."""
            del format, args

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    factory, _ = registry._HTTP_PROVIDERS["anthropic"]
    monkeypatch.setattr(
        registry,
        "_HTTP_PROVIDERS",
        {
            **registry._HTTP_PROVIDERS,
            "anthropic": (factory, f"http://127.0.0.1:{upstream.server_port}/v1"),
        },
    )
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-key")
    manager, key = _configure(tmp_path, hour_price=hour_price)
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={
                "model": "coding",
                "max_tokens": 16,
                "stream": stream,
                "cache_control": {"type": "ephemeral", "ttl": "5m" if hour_price is None else "1h"},
                "messages": [{"role": "user", "content": "Report the observed usage."}],
            },
            timeout=10,
        )
        assert response.status_code == 200, response.text
        assert len(requests) == 1
        assert requests[0]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "5m" if hour_price is None else "1h",
        }
    finally:
        gateway.stop()
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)
    with sqlite3.connect(manager.database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM gateway_attempts").fetchone()
    assert row["state"] == "completed"
    assert (
        row["input_tokens"],
        row["cached_input_tokens"],
        row["cache_creation_input_tokens"],
        row["cache_creation_1h_input_tokens"],
        row["output_tokens"],
    ) == (1000, 100, 600, expected_hour, 10)
    assert row["estimated_cost_nano_usd"] == expected_cost
    usage = GatewayUsage(
        input_tokens=1000,
        output_tokens=10,
        cached_input_tokens=100,
        cache_creation_input_tokens=600,
        cache_creation_1h_input_tokens=expected_hour,
    )
    platform = SQLiteGatewayPlatform(manager.database_path)
    replay = AttemptSettlementRequest(
        organization_id="local",
        attempt_id=row["attempt_id"],
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED, sequence_number=1, usage=usage
        ),
    )
    assert platform.settle_attempt(replay).usage == usage
    assert platform.settle_attempt(replay).estimated_cost_nano_usd == expected_cost
    different = usage.model_copy(update={"cache_creation_1h_input_tokens": 201})
    with pytest.raises(ValueError, match="differs from durable"):
        platform.settle_attempt(
            replay.model_copy(
                update={
                    "terminal_event": GatewayEvent(
                        kind=GatewayEventKind.COMPLETED, sequence_number=1, usage=different
                    )
                }
            )
        )
    with sqlite3.connect(manager.database_path) as connection:
        after = connection.execute(
            "SELECT estimated_cost_nano_usd, budget_settled_nano_usd FROM gateway_attempts"
        ).fetchone()
    assert after == (expected_cost, row["budget_settled_nano_usd"])
