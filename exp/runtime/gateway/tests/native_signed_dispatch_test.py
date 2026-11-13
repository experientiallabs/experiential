"""End-to-end dispatch-time signing over the native engine's open retry.

Each retry signs its frozen body in Rust using a synthetic credential and a
loopback endpoint frozen at admission. The upstream verifies both signatures
against the exact received bytes and forces a new timestamp before the retry.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import signal
import socket
import subprocess
import sys
import time
import zlib
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import quote

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


def _eventstream_message(
    name: str, payload: dict[str, object], *, exception: bool = False
) -> bytes:
    """Encode one AWS event-stream message the way Bedrock frames its stream.

    Args:
        name: Event-type (or exception-type) header value.
        payload: JSON payload object.
        exception: Whether to frame the message as a service exception.

    Returns:
        One complete binary event-stream message with valid checksums.
    """
    headers = [
        (":message-type", "exception" if exception else "event"),
        (":exception-type" if exception else ":event-type", name),
    ]
    block = b""
    for header_name, value in headers:
        block += bytes([len(header_name)]) + header_name.encode()
        block += bytes([7]) + len(value).to_bytes(2, "big") + value.encode()
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    total = 12 + len(block) + len(body) + 4
    prelude = total.to_bytes(4, "big") + len(block).to_bytes(4, "big")
    message = prelude + zlib.crc32(prelude).to_bytes(4, "big") + block + body
    return message + zlib.crc32(message).to_bytes(4, "big")


_CHILD_SOURCE = """\
import json
import sys
from pathlib import Path

import exp_gateway_native

from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_bridge import NativeControlPlane


from exp.runtime.models.providers.bedrock import BedrockClient
from urllib.parse import quote, urlsplit

_original_converse = BedrockClient.converse_stream_url
_original_invoke = BedrockClient.invoke_url

def _loopback(original, client):
    \"\"\"Retain the real model path on the private synthetic provider.\"\"\"
    url = original(client)
    return f"http://127.0.0.1:{sys.argv[3]}" + url[url.index("/model/"):]

BedrockClient.converse_stream_url = lambda self: _loopback(_original_converse, self)
BedrockClient.invoke_url = lambda self: _loopback(_original_invoke, self)

def _authority(self, *, url):
    \"\"\"Export synthetic signing material directly to the native owner.\"\"\"
    endpoint = urlsplit(url)
    return {"kind": "sigv4", "default_chain": False, "profile": None,
            "access_key": "fixture-access", "secret_key": "fixture-secret",
            "session_token": None, "region": "us-east-1", "service": "bedrock",
            "host": endpoint.netloc, "path": quote(endpoint.path, safe="/~"),
            "query": "", "accept": "application/json" if url.endswith("/invoke")
            else "application/vnd.amazon.eventstream"}

BedrockClient.native_gateway_signing = _authority

components = load_gateway_components(
    Path(sys.argv[1]),
    environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
)
control = NativeControlPlane(components)
config = json.dumps(
    {
        "host": "127.0.0.1",
        "port": int(sys.argv[2]),
        "request_timeout_seconds": 30.0,
        "graceful_timeout_seconds": 1.0,
    }
)
exp_gateway_native.serve(control, config, native_callbacks=control._accounting.native.callbacks())
"""


def assert_native_signature(headers: Mapping[str, str], path: str, body: bytes) -> None:
    """Independently verify the native SigV4 result over the received HTTP bytes."""
    values = {name.lower(): value for name, value in headers.items()}
    header = values["authorization"]
    fields = dict(
        item.split("=", 1) for item in header.removeprefix("AWS4-HMAC-SHA256 ").split(", ")
    )
    credential, scope = fields["Credential"].split("/", 1)
    assert credential == "fixture-access"
    signed = fields["SignedHeaders"]
    canonical_headers = "".join(
        f"{name}:{' '.join(values[name].split())}\n" for name in signed.split(";")
    )
    canonical = (
        f"POST\n{quote(path, safe='/~')}\n\n{canonical_headers}\n"
        f"{signed}\n{hashlib.sha256(body).hexdigest()}"
    )
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n{values['x-amz-date']}\n{scope}\n"
        f"{hashlib.sha256(canonical.encode()).hexdigest()}"
    )
    key = b"AWS4fixture-secret"
    for part in scope.split("/"):
        key = hmac.digest(key, part.encode(), "sha256")
    assert hmac.compare_digest(
        fields["Signature"], hmac.digest(key, string_to_sign.encode(), "sha256").hex()
    )


class _FlakyBedrockUpstream(BaseHTTPRequestHandler):
    """Kill the first connection unanswered; stream real frames afterwards."""

    seen_authorizations: list[str] = []
    lock = Lock()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        """Record the signed header, then fail once and serve once."""
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        assert_native_signature(dict(self.headers.items()), self.path, body)
        with _FlakyBedrockUpstream.lock:
            _FlakyBedrockUpstream.seen_authorizations.append(self.headers.get("authorization", ""))
            first = len(_FlakyBedrockUpstream.seen_authorizations) == 1
        if first:
            time.sleep(1.1)  # Cross a signing timestamp boundary before retry.
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
        capabilities=ModelCapabilities(maximum_output_tokens=128_000),
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
        signatures = _FlakyBedrockUpstream.seen_authorizations
        assert len(signatures) == 2 and signatures[0] != signatures[1]
        assert all(
            value.startswith("AWS4-HMAC-SHA256 Credential=fixture-access/") for value in signatures
        )
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=20) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
