"""End-to-end native-engine observability over real sockets and a mock provider.

The native engine serves in a child process (its shutdown path is the real
SIGTERM drain), a loopback HTTP server plays the provider, and the fallback
port is deliberately dead. Plain and replay-keyed chat is served natively; an
alias on a provider without a native dialect escalates, and its relay against
the dead fallback also exercises the fallback-unavailable signal. The test
asserts the ``/metrics.json`` snapshot moved for served, escalated, and
proxied traffic and that the body stays content-free.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import httpx
import pytest

from exp.common.models import (
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.catalog_authority import (
    ConnectionConfig,
    upsert_connection,
    upsert_singleton_deployment,
)
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.management import GatewayManagement

_CHILD_SOURCE = """\
import json
import sys
from pathlib import Path

import exp_gateway_native

from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_bridge import NativeControlPlane

components = load_gateway_components(
    Path(sys.argv[1]),
    environment={
        "TEST_PROVIDER_KEY": "provider-secret-canary",
        "TEST_GEMINI_KEY": "gemini-secret-canary",
    },
)
control = NativeControlPlane(
    components,
    data_plane_metrics=exp_gateway_native.metrics_snapshot_json,
)
config = json.dumps(
    {
        "host": "127.0.0.1",
        "port": int(sys.argv[2]),
        "fallback_port": int(sys.argv[3]),
        "request_timeout_seconds": 30.0,
        "graceful_timeout_seconds": 1.0,
    }
)
exp_gateway_native.serve(control, config)
"""


class _LoopbackProvider(BaseHTTPRequestHandler):
    """Serve one finite OpenAI-compatible SSE completion per request."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        """Return text, usage, and terminal frames for any provider request."""
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        frames = b"".join(
            (
                _provider_frame(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ),
                _provider_frame(
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
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


def _provider_frame(payload: dict[str, object]) -> bytes:
    """Encode one provider SSE data frame."""
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _unused_port() -> int:
    """Allocate one currently unused loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _activate_dialectless_alias(root: Path, manager: GatewayManagement) -> None:
    """Grant one alias on a provider without a native dialect, so it escalates."""
    upsert_connection(
        root,
        name="gemini-main",
        connection=ConnectionConfig(provider="gemini", api_key_env="TEST_GEMINI_KEY"),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="gem",
        connection_name="gemini-main",
        provider_model="gemini-model-exact",
        exact_model_id="gemini-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="gem",
        alias_name="gem",
        revision_id="revision-gem",
        pool_id="gem",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="gem")


def _wait_live(port: int, child: subprocess.Popen[bytes]) -> None:
    """Poll the native liveness route until it answers or the child dies."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise AssertionError(f"native engine exited early: {child.returncode}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health/live", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.05)
    raise AssertionError("native engine did not become live in time")


def test_native_metrics_snapshot_moves_for_served_escalated_and_proxied_traffic(
    tmp_path: Path,
) -> None:
    """Served, replay-keyed, escalated, and fallback-failed traffic all land."""
    pytest.importorskip("exp_gateway_native")
    provider_port = _unused_port()
    gateway_port = _unused_port()
    dead_fallback_port = _unused_port()
    provider = ThreadingHTTPServer(("127.0.0.1", provider_port), _LoopbackProvider)
    provider_thread = Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    manager, raw_key = _configured_gateway(
        tmp_path,
        base_url=f"http://127.0.0.1:{provider_port}/v1",
    )
    _activate_dialectless_alias(tmp_path, manager)
    child = subprocess.Popen(  # noqa: S603 - fixed argv built from this test.
        [
            sys.executable,
            "-c",
            _CHILD_SOURCE,
            str(tmp_path),
            str(gateway_port),
            str(dead_fallback_port),
        ],
        env=dict(os.environ),
    )
    prompt_canary = "private-prompt-canary"
    try:
        _wait_live(gateway_port, child)
        base = f"http://127.0.0.1:{gateway_port}"
        headers = {"Authorization": f"Bearer {raw_key}"}
        body = {"model": "coding", "messages": [{"role": "user", "content": prompt_canary}]}
        for _ in range(2):
            served = httpx.post(
                f"{base}/v1/chat/completions", headers=headers, json=body, timeout=10
            )
            assert served.status_code == 200
            assert served.json()["choices"][0]["message"]["content"] == "ok"
        # A replay-keyed request is served natively through the replay store.
        keyed = httpx.post(
            f"{base}/v1/chat/completions",
            headers={**headers, "Idempotency-Key": "replay-key-one"},
            json=body,
            timeout=10,
        )
        assert keyed.status_code == 200
        # An alias without a native dialect escalates; the dead fallback makes
        # the relay fail with the only signal a dead embedded engine produces.
        escalated = httpx.post(
            f"{base}/v1/chat/completions",
            headers=headers,
            json={"model": "gem", "messages": [{"role": "user", "content": prompt_canary}]},
            timeout=10,
        )
        assert escalated.status_code == 502
        assert escalated.json()["error"]["code"] == "fallback_engine_unavailable"

        snapshot = httpx.get(f"{base}/metrics.json", timeout=10)
        assert snapshot.status_code == 200
        payload = snapshot.json()
        data_plane = payload["data_plane"]
        assert data_plane["served_requests"] == 3
        assert data_plane["requests"] == {
            "completed": 3,
            "incomplete": 0,
            "failed": 0,
            "cancelled": 0,
        }
        assert data_plane["escalated_requests"]["provider_dialect"] == 1
        assert data_plane["proxied_requests"] == 1
        assert data_plane["fallback_engine_unavailable"] == 1
        assert data_plane["active_requests"] == 0
        assert data_plane["active_proxies"] == 0
        assert data_plane["time_to_first_byte_ms"]["count"] == 3
        assert data_plane["request_duration_ms"]["count"] == 3
        assert data_plane["permit_wait_ms"]["count"] == 3
        assert data_plane["bridge_call_ms"]["count"] >= 1
        control_plane = payload["control_plane"]
        assert control_plane["inflight_attempts"] == 0
        assert control_plane["sweep_retained_settlements_replayed"] == 0
        assert control_plane["sweep_abandoned_attempts_cancelled"] == 0
        assert control_plane["accounting_healthy"] is True

        # The snapshot is content-free: no alias, key, prompt, or credential.
        rendered = snapshot.text
        for forbidden in ("coding", "gem", raw_key, prompt_canary, "provider-secret-canary"):
            assert forbidden not in rendered

        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=20) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
