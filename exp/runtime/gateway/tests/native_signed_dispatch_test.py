"""End-to-end dispatch-time signing over the native engine's open retry.

Body-signing dialects must sign immediately before every physical open
attempt: a first attempt can consume minutes before a retryable failure, so
reusing its headers on the retry could send an expired SigV4 signature. The
native engine serves in a child process whose control plane rewrites the
admitted Bedrock URL to a loopback upstream and stamps each ``sign_dispatch``
invocation into the authorization header; the upstream kills the first
connection before any response and serves a real binary event stream on the
second, so the test proves the retry carried a freshly minted signature.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

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
from exp.runtime.gateway.tests.native_dialect_parity_test import _eventstream_message

_CHILD_SOURCE = """\
import json
import sys
from pathlib import Path

import exp_gateway_native

from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_bridge import NativeControlPlane


class _RewritingControlPlane:
    \"\"\"Delegate to the real control plane, pointing Bedrock at loopback.

    ``admit`` rewrites the admitted ConverseStream URL to the test upstream,
    and ``sign_dispatch`` counts invocations into the authorization header so
    the upstream observes exactly which signing call each attempt carried.
    \"\"\"

    def __init__(self, inner, upstream_base):
        \"\"\"Wrap one real control plane above the loopback upstream.\"\"\"
        self._inner = inner
        self._upstream_base = upstream_base
        self._signatures = 0

    def __getattr__(self, name):
        \"\"\"Delegate every other boundary callback unchanged.\"\"\"
        return getattr(self._inner, name)

    def admit(self, argument):
        \"\"\"Admit through the real control plane, then rewrite the route's URL.\"\"\"
        result = self._inner.admit(argument)
        parsed = json.loads(result)
        route = parsed.get("route")
        if isinstance(route, list):
            for wire in route:
                if isinstance(wire, dict) and wire.get("dialect") == "bedrock_converse_stream":
                    wire["url"] = self._upstream_base + "/model/test/converse-stream"
        return json.dumps(parsed, separators=(",", ":"))

    def sign_dispatch(self, argument):
        \"\"\"Stamp one counted signature so each attempt is distinguishable.\"\"\"
        json.loads(argument)
        self._signatures += 1
        return json.dumps(
            {
                "headers": {
                    "authorization": f"AWS4-HMAC-SHA256 attempt-{self._signatures}",
                    "content-type": "application/json",
                }
            },
            separators=(",", ":"),
        )


components = load_gateway_components(
    Path(sys.argv[1]),
    environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
)
control = _RewritingControlPlane(
    NativeControlPlane(components),
    f"http://127.0.0.1:{sys.argv[3]}",
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


class _FlakyBedrockUpstream(BaseHTTPRequestHandler):
    """Kill the first connection unanswered; stream real frames afterwards."""

    seen_authorizations: list[str] = []
    lock = Lock()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        """Record the signed header, then fail once and serve once."""
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        with _FlakyBedrockUpstream.lock:
            _FlakyBedrockUpstream.seen_authorizations.append(self.headers.get("authorization", ""))
            first = len(_FlakyBedrockUpstream.seen_authorizations) == 1
        if first:
            # Close before any response bytes: a retryable transport failure.
            self.close_connection = True
            self.connection.close()
            return
        body = b"".join(
            (
                _eventstream_message("messageStart", {"role": "assistant"}),
                _eventstream_message(
                    "contentBlockDelta",
                    {"contentBlockIndex": 0, "delta": {"text": "signed"}},
                ),
                _eventstream_message("messageStop", {"stopReason": "end_turn"}),
                _eventstream_message(
                    "metadata",
                    {"usage": {"inputTokens": 2, "outputTokens": 1}},
                ),
            )
        )
        self.send_response(200)
        self.send_header("content-type", "application/vnd.amazon.eventstream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logs so test output cannot retain payload context."""
        del format, args


def _unused_port() -> int:
    """Allocate one currently unused loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


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


def _activate_bedrock_alias(root: Path, manager: GatewayManagement) -> None:
    """Grant one Bedrock alias whose dialect signs its dispatch body."""
    upsert_connection(
        root,
        name="bedrock-main",
        connection=ConnectionConfig(provider="bedrock", region="us-east-1"),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="bed",
        connection_name="bedrock-main",
        provider_model="us.anthropic.claude-sonnet-4-5",
        exact_model_id="bedrock-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="bed",
        alias_name="bed",
        revision_id="revision-bed",
        pool_id="bed",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="bed")


def test_open_retry_signs_every_physical_attempt(tmp_path: Path) -> None:
    """A retried open carries a freshly minted signature, never a reused one."""
    pytest.importorskip("exp_gateway_native")
    upstream_port = _unused_port()
    gateway_port = _unused_port()
    _FlakyBedrockUpstream.seen_authorizations = []
    upstream = ThreadingHTTPServer(("127.0.0.1", upstream_port), _FlakyBedrockUpstream)
    upstream_thread = Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    manager, raw_key = _configured_gateway(tmp_path)
    _activate_bedrock_alias(tmp_path, manager)
    child = subprocess.Popen(  # noqa: S603 - fixed argv built from this test.
        [
            sys.executable,
            "-c",
            _CHILD_SOURCE,
            str(tmp_path),
            str(gateway_port),
            str(upstream_port),
        ],
        env=dict(os.environ),
    )
    try:
        _wait_live(gateway_port, child)
        served = httpx.post(
            f"http://127.0.0.1:{gateway_port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={"model": "bed", "messages": [{"role": "user", "content": "hi"}]},
            timeout=20,
        )
        assert served.status_code == 200
        assert served.json()["choices"][0]["message"]["content"] == "signed"
        # One signing per physical attempt: the failed first open and its
        # retry each carried their own freshly minted headers.
        assert _FlakyBedrockUpstream.seen_authorizations == [
            "AWS4-HMAC-SHA256 attempt-1",
            "AWS4-HMAC-SHA256 attempt-2",
        ]
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=20) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
