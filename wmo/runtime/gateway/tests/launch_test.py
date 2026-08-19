"""Black-box local launch certification with real sockets, SQLite, and official SDK clients."""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import openai
import pytest
import uvicorn
from fastapi.testclient import TestClient
from openai import AsyncOpenAI, OpenAI
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from wmo.runtime.gateway.catalog_authority import upsert_singleton_deployment
from wmo.runtime.gateway.lifecycle import load_local_gateway
from wmo.runtime.gateway.management import GatewayManagement


class _LoopbackProvider(BaseHTTPRequestHandler):
    """Serve a finite OpenAI-compatible SSE response over a real loopback socket."""

    calls = 0

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        """Read one provider request and return text, usage, and terminal frames."""
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        type(self).calls += 1
        if payload["model"] == "provider-model-primary":
            body = b'{"error":{"message":"primary unavailable"}}'
            self.send_response(503)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        frames = b"".join(
            (
                _provider_frame(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "hello "},
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
                _provider_frame(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "world"},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ),
                _provider_frame(
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                    }
                ),
                b"data: [DONE]\n\n",
            )
        )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(frames)))
        self.end_headers()
        self.wfile.write(frames)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logs so test output cannot retain payload context."""
        del format, args


def test_real_loopback_launch_serves_both_official_sdk_clients_and_revocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sync and async SDK traffic reaches a real upstream and content-free accounting."""
    assert openai.__version__ == "3.0.0"
    _LoopbackProvider.calls = 0
    provider_port = _unused_port()
    gateway_port = _unused_port()
    provider = ThreadingHTTPServer(("127.0.0.1", provider_port), _LoopbackProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    manager, raw_key = _configure_gateway(
        tmp_path,
        base_url=f"http://127.0.0.1:{provider_port}/v1",
    )
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret-canary")
    runtime = load_local_gateway(tmp_path, graceful_timeout_seconds=2)
    server = uvicorn.Server(
        uvicorn.Config(
            runtime.app,
            host="127.0.0.1",
            port=gateway_port,
            log_level="error",
        )
    )
    gateway_thread = threading.Thread(target=server.run, daemon=True)
    gateway_thread.start()
    _wait_ready(gateway_port, gateway_thread)
    base_url = f"http://127.0.0.1:{gateway_port}/v1"
    prompt_canary = "private-prompt-canary"
    try:
        with OpenAI(api_key=raw_key, base_url=base_url) as client:
            chat = list(
                client.chat.completions.create(
                    model="coding",
                    messages=[{"role": "user", "content": prompt_canary}],
                    stream=True,
                )
            )
            response = client.responses.create(model="coding", input=prompt_canary)
        assert (
            "".join(chunk.choices[0].delta.content or "" for chunk in chat if chunk.choices)
            == "hello world"
        )
        assert response.output_text == "hello world"
        asyncio.run(_exercise_async_sdk(base_url, raw_key, prompt_canary))

        usage = httpx.get(f"http://127.0.0.1:{gateway_port}/usage.json", timeout=2).json()
        assert usage["totals"]["requests"] == 4
        assert usage["totals"]["input_tokens"] == 8
        assert usage["totals"]["output_tokens"] == 8
        assert _LoopbackProvider.calls == 4

        manager.revoke_key(key_id="key-one")
        with OpenAI(api_key=raw_key, base_url=base_url) as revoked:
            with pytest.raises(openai.AuthenticationError):
                revoked.models.list()
    finally:
        server.should_exit = True
        gateway_thread.join(timeout=5)
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
    assert not gateway_thread.is_alive()
    assert not provider_thread.is_alive()

    durable = b"".join(
        path.read_bytes() for path in (tmp_path / "gateway").rglob("*") if path.is_file()
    )
    for forbidden in (prompt_canary, "hello world", raw_key, "provider-secret-canary"):
        assert forbidden.encode() not in durable


def test_fresh_root_cli_certifies_lifecycle_and_executes_ordered_waterfall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Noninteractive authoring launches and attributes a real provider waterfall."""
    _LoopbackProvider.calls = 0
    provider_port = _unused_port()
    provider = ThreadingHTTPServer(("127.0.0.1", provider_port), _LoopbackProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret-canary")
    runner = CliRunner()
    base_url = f"http://127.0.0.1:{provider_port}/v1"
    budget_period = datetime.now(UTC).strftime("%Y-%m")
    gateway_client: TestClient | None = None
    try:
        commands = (
            ["config", "gateway", "init", "--root", str(tmp_path), "--json"],
            [
                "config",
                "gateway",
                "provider",
                "add",
                "provider-primary",
                "--provider",
                "openai-compatible",
                "--credential-env",
                "LOOPBACK_PROVIDER_KEY",
                "--base-url",
                base_url,
                "--root",
                str(tmp_path),
                "--non-interactive",
                "--json",
            ],
            [
                "config",
                "gateway",
                "provider",
                "add",
                "provider-secondary",
                "--provider",
                "openai-compatible",
                "--credential-env",
                "LOOPBACK_PROVIDER_KEY",
                "--base-url",
                base_url,
                "--root",
                str(tmp_path),
                "--non-interactive",
                "--json",
            ],
        )
        for command in commands:
            result = runner.invoke(app, command)
            assert result.exit_code == 0, result.output
        catalog_sha256 = ""
        for deployment_alias, connection_name in (
            ("primary", "provider-primary"),
            ("secondary", "provider-secondary"),
        ):
            created = runner.invoke(
                app,
                [
                    "config",
                    "gateway",
                    "alias",
                    "create",
                    deployment_alias,
                    "--deployment",
                    f"{connection_name}:provider-model-{deployment_alias}",
                    "--exact-model",
                    "model-revision-exact",
                    "--maximum-output-tokens",
                    "16",
                    "--input-price",
                    "1000000",
                    "--output-price",
                    "2000000",
                    "--root",
                    str(tmp_path),
                    "--non-interactive",
                    "--json",
                ],
            )
            assert created.exit_code == 0, created.output
            catalog_sha256 = json.loads(created.stdout)["data"]["catalog_sha256"]
        certification = runner.invoke(
            app,
            [
                "config",
                "gateway",
                "pool",
                "certify",
                "coding",
                "--deployment-alias",
                "primary",
                "--deployment-alias",
                "secondary",
                "--exact-model",
                "model-revision-exact",
                "--certification-id",
                "certification-one",
                "--provenance",
                "loopback exact-model comparison",
                "--evidence-sha256",
                "a" * 64,
                "--certified-at",
                "2026-08-18T00:00:00Z",
                "--expected-catalog-sha256",
                catalog_sha256,
                "--revision",
                "revision-waterfall-one",
                "--root",
                str(tmp_path),
                "--non-interactive",
                "--json",
            ],
        )
        assert certification.exit_code == 0, certification.output
        for command in (
            [
                "config",
                "gateway",
                "identity",
                "create",
                "default",
                "--root",
                str(tmp_path),
                "--non-interactive",
                "--json",
            ],
            [
                "config",
                "gateway",
                "grant",
                "add",
                "default",
                "coding",
                "--root",
                str(tmp_path),
                "--non-interactive",
                "--json",
            ],
        ):
            result = runner.invoke(app, command)
            assert result.exit_code == 0, result.output
        issued = runner.invoke(
            app,
            [
                "config",
                "gateway",
                "key",
                "issue",
                "default",
                "--key-id",
                "key-one",
                "--root",
                str(tmp_path),
                "--non-interactive",
                "--json",
            ],
        )
        assert issued.exit_code == 0, issued.output
        raw_key = json.loads(issued.stdout)["data"]["raw_key"]
        for budget_arguments in (
            ["--scope", "team"],
            ["--scope", "identity", "--identity", "default"],
            ["--scope", "pool", "--alias", "coding", "--pool", "coding"],
            [
                "--scope",
                "deployment",
                "--alias",
                "coding",
                "--pool",
                "coding",
                "--deployment",
                "primary",
            ],
            [
                "--scope",
                "deployment",
                "--alias",
                "coding",
                "--pool",
                "coding",
                "--deployment",
                "secondary",
            ],
        ):
            configured = runner.invoke(
                app,
                [
                    "config",
                    "gateway",
                    "budget",
                    "set",
                    "--period",
                    budget_period,
                    *budget_arguments,
                    "--limit-micro-usd",
                    "1000000",
                    "--root",
                    str(tmp_path),
                    "--non-interactive",
                    "--json",
                ],
            )
            assert configured.exit_code == 0, configured.output

        runtime = load_local_gateway(tmp_path, graceful_timeout_seconds=2)
        gateway_client = TestClient(runtime.app)
        gateway_client.__enter__()
        first_response = gateway_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": "waterfall-canary"}],
            },
        )

        assert first_response.status_code == 200, first_response.text
        assert first_response.json()["choices"][0]["message"]["content"] == "hello world"
        assert first_response.headers["x-gateway-route-depth"] == "1"
        assert _LoopbackProvider.calls == 3
        remaining_result = runner.invoke(
            app,
            [
                "config",
                "gateway",
                "budget",
                "remaining",
                "--period",
                budget_period,
                "--root",
                str(tmp_path),
                "--json",
            ],
        )
        assert remaining_result.exit_code == 0, remaining_result.output
        budget_rows = json.loads(remaining_result.stdout)["items"]
        primary_budget = next(
            row for row in budget_rows if row["budget"]["scope"].get("deployment_id") == "primary"
        )
        assert primary_budget["charged_micro_usd"] > 0
        exhausted_primary = runner.invoke(
            app,
            [
                "config",
                "gateway",
                "budget",
                "set",
                "--period",
                budget_period,
                "--scope",
                "deployment",
                "--alias",
                "coding",
                "--pool",
                "coding",
                "--deployment",
                "primary",
                "--limit-micro-usd",
                str(primary_budget["charged_micro_usd"]),
                "--replace",
                "--root",
                str(tmp_path),
                "--non-interactive",
                "--json",
            ],
        )
        assert exhausted_primary.exit_code == 0, exhausted_primary.output
        second_response = gateway_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": "budget-fallback-canary"}],
            },
        )
        assert second_response.status_code == 200, second_response.text
        assert second_response.headers["x-gateway-route-depth"] == "1"
        assert _LoopbackProvider.calls == 4

        replay_headers = {
            "Authorization": f"Bearer {raw_key}",
            "Idempotency-Key": "budget-replay-one",
        }
        replay_payload = {
            "model": "coding",
            "messages": [{"role": "user", "content": "budget-replay-canary"}],
        }
        original = gateway_client.post(
            "/v1/chat/completions",
            headers=replay_headers,
            json=replay_payload,
        )
        replay = gateway_client.post(
            "/v1/chat/completions",
            headers=replay_headers,
            json=replay_payload,
        )
        assert original.status_code == 200, original.text
        assert replay.status_code == 200, replay.text
        assert replay.json() == original.json()
        assert _LoopbackProvider.calls == 5

        refreshed = runner.invoke(
            app,
            [
                "config",
                "gateway",
                "budget",
                "remaining",
                "--period",
                budget_period,
                "--root",
                str(tmp_path),
                "--json",
            ],
        )
        identity_budget = next(
            row
            for row in json.loads(refreshed.stdout)["items"]
            if row["budget"]["scope"]["kind"] == "identity"
        )
        exhausted_identity = runner.invoke(
            app,
            [
                "config",
                "gateway",
                "budget",
                "set",
                "--period",
                budget_period,
                "--scope",
                "identity",
                "--identity",
                "default",
                "--limit-micro-usd",
                str(identity_budget["charged_micro_usd"]),
                "--replace",
                "--root",
                str(tmp_path),
                "--non-interactive",
                "--json",
            ],
        )
        assert exhausted_identity.exit_code == 0, exhausted_identity.output
        quota_response = gateway_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={"model": "coding", "input": "quota-canary"},
        )
        assert quota_response.status_code == 429
        assert quota_response.json()["error"]["code"] == "insufficient_quota"
        assert quota_response.json()["error"]["type"] == "insufficient_quota"
        assert _LoopbackProvider.calls == 5
        connection = sqlite3.connect(tmp_path / "gateway" / "gateway.db")
        try:
            attempts = connection.execute(
                "SELECT deployment_id, route_depth, state FROM gateway_attempts "
                "ORDER BY started_at, attempt_id"
            ).fetchall()
        finally:
            connection.close()
        assert attempts == [
            ("primary", 0, "failed"),
            ("primary", 0, "failed"),
            ("secondary", 1, "completed"),
            ("secondary", 1, "completed"),
            ("secondary", 1, "completed"),
        ]
    finally:
        if gateway_client is not None:
            gateway_client.__exit__(None, None, None)
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
    assert not provider_thread.is_alive()


async def _exercise_async_sdk(base_url: str, raw_key: str, prompt: str) -> None:
    """Exercise async non-streaming Chat and streaming Responses over real HTTP."""
    async with AsyncOpenAI(api_key=raw_key, base_url=base_url) as client:
        chat = await client.chat.completions.create(
            model="coding",
            messages=[{"role": "user", "content": prompt}],
        )
        stream = await client.responses.create(model="coding", input=prompt, stream=True)
        events = [event async for event in stream]
    assert chat.choices[0].message.content == "hello world"
    assert events[-1].type == "response.completed"


def _configure_gateway(root: Path, *, base_url: str) -> tuple[GatewayManagement, str]:
    """Create explicit launch state against one real loopback provider."""
    manager = GatewayManagement(root)
    manager.initialize()
    manager.upsert_provider_connection(
        connection_id="provider-main",
        config=ConnectionConfig(
            provider="openai-compatible",
            base_url=base_url,
            api_key_env="LOOPBACK_PROVIDER_KEY",
        ),
    )
    serving_connections = {
        item.connection_id: item.config for item in manager.provider_connections()
    }
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="coding",
        connection_name="provider-main",
        provider_model="provider-model-exact",
        exact_model_id="model-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(
            input_micro_usd_per_million_tokens=1_000_000,
            output_micro_usd_per_million_tokens=2_000_000,
        ),
        pricing_source="loopback-test",
        replace=False,
        serving_connections=serving_connections,
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
    issued = manager.issue_key(identity_id="default", key_id="key-one")
    return manager, issued.raw_key


def _provider_frame(payload: dict[str, object]) -> bytes:
    """Encode one compact provider SSE data frame."""
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _unused_port() -> int:
    """Return one currently unused loopback TCP port."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_ready(port: int, thread: threading.Thread) -> None:
    """Wait until the gateway readiness endpoint answers or fail boundedly."""
    deadline = time.monotonic() + 10
    url = f"http://127.0.0.1:{port}/health/ready"
    while time.monotonic() < deadline:
        if not thread.is_alive():
            raise AssertionError("gateway server exited before readiness")
        try:
            response = httpx.get(url, timeout=0.2)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.02)
    raise AssertionError("gateway did not become ready")
