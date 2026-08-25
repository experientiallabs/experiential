"""End-to-end native-engine observability over real sockets and a mock provider.

The native engine serves in a child process (its shutdown path is the real
SIGTERM drain) and a loopback HTTP server plays the provider. Plain and
replay-keyed chat is served natively; a host-policy-rejected alias escalates
and fails closed as the shared internal error. The test asserts the
``/metrics.json`` snapshot moved for served and escalated traffic, that the
same snapshot serves at ``/metrics`` in the Prometheus text format, and that
both bodies stay content-free.
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

from exp.runtime.gateway.lifecycle_test import (
    _activate_alias_for_escalation_policy,
    _configured_gateway,
)
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.native_metrics_text import METRICS_CONTENT_TYPE

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
    },
)


def native_route_eligible(route, request):
    \"\"\"Reject the escalated fixture alias by name; every other route is native.\"\"\"
    del request
    return route.snapshot.authorization.alias != "gem"


control = NativeControlPlane(
    components,
    data_plane_metrics=exp_gateway_native.metrics_snapshot_json,
    native_route_eligible=native_route_eligible,
)
config = json.dumps(
    {
        "host": "127.0.0.1",
        "port": int(sys.argv[2]),
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


def _activate_escalating_alias(root: Path, manager: GatewayManagement) -> None:
    """Grant one otherwise-native alias the child's host policy rejects."""
    _activate_alias_for_escalation_policy(root, manager, alias="gem")


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


def test_native_metrics_snapshot_moves_for_served_keyed_and_escalated_traffic(
    tmp_path: Path,
) -> None:
    """Served, replay-keyed, and escalated traffic all land in the snapshot."""
    pytest.importorskip("exp_gateway_native")
    provider_port = _unused_port()
    gateway_port = _unused_port()
    provider = ThreadingHTTPServer(("127.0.0.1", provider_port), _LoopbackProvider)
    provider_thread = Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    manager, raw_key = _configured_gateway(
        tmp_path,
        base_url=f"http://127.0.0.1:{provider_port}/v1",
    )
    _activate_escalating_alias(tmp_path, manager)
    child = subprocess.Popen(  # noqa: S603 - fixed argv built from this test.
        [
            sys.executable,
            "-c",
            _CHILD_SOURCE,
            str(tmp_path),
            str(gateway_port),
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
        # The child's host policy rejects this alias by name, so it always
        # escalates; with no python engine anywhere the escalation fails
        # closed as the shared internal error.
        escalated = httpx.post(
            f"{base}/v1/chat/completions",
            headers=headers,
            json={"model": "gem", "messages": [{"role": "user", "content": prompt_canary}]},
            timeout=10,
        )
        assert escalated.status_code == 500
        assert escalated.json()["error"]["code"] == "internal_error"

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
        assert data_plane["escalated_requests"]["host_policy"] == 1
        assert data_plane["active_requests"] == 0
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

        # The same snapshot serves at /metrics in the Prometheus text format,
        # equally content-free and carrying the same driven-traffic counts.
        exposition = httpx.get(f"{base}/metrics", timeout=10)
        assert exposition.status_code == 200
        assert exposition.headers["content-type"] == METRICS_CONTENT_TYPE
        text = exposition.text
        assert 'exp_gateway_requests_total{outcome="completed"} 3' in text
        assert "exp_gateway_served_requests_total 3" in text
        assert 'exp_gateway_escalated_requests_total{kind="host_policy"} 1' in text
        assert 'exp_gateway_time_to_first_byte_ms_bucket{le="+Inf"} 3' in text
        assert "exp_gateway_accounting_healthy 1" in text
        for forbidden in ("coding", "gem", raw_key, prompt_canary, "provider-secret-canary"):
            assert forbidden not in text

        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=20) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
