"""End-to-end embeddings tests against the served native engine.

One shared native serving subprocess (the driver from ``native_messages_test``)
serves a seeded root with two aliases on one OpenAI-compatible loopback
connection: ``coding`` (a chat model) and ``embedder`` (a catalog-declared
embeddings model). The tests drive ``POST /v1/embeddings`` through the real
Rust data plane and shared python control plane with the official OpenAI
client and raw HTTP, and read the settled accounting back from the live usage
report.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import openai
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.catalog_authority import upsert_singleton_deployment
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.tests.native_messages_test import (
    _DRIVER_SOURCE,
    _HOST,
    _REQUEST_TIMEOUT_SECONDS,
    _ServingEngine,
)

pytest.importorskip("exp_gateway_native")

_VECTOR_WIDTH = 3


def _vector(position: int) -> list[float]:
    """Return the deterministic loopback vector for one input position."""
    return [float(position + 1), 0.5, -0.25]


class _EmbeddingsUpstream(BaseHTTPRequestHandler):
    """OpenAI-compatible ``/embeddings`` mock whose shape is selected by the first input."""

    payloads: list[JsonObject] = []
    payloads_lock = threading.Lock()

    def _answer(self, status: int, body: JsonObject) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Answer one canned embeddings response selected by the first input."""
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        with self.payloads_lock:
            self.payloads.append(payload)
        if not self.path.endswith("/embeddings"):
            self._answer(404, {"error": {"message": "unknown route", "type": "invalid_request"}})
            return
        raw_inputs = payload["input"]
        inputs = [raw_inputs] if isinstance(raw_inputs, str) else list(raw_inputs)
        selector = inputs[0]
        if selector == "reject-param":
            self._answer(
                400,
                {
                    "error": {
                        "message": "'$.input' is too long. Maximum length is 2048 items.",
                        "type": "invalid_request_error",
                        "param": "input",
                        "code": "invalid_value",
                    }
                },
            )
            return
        if selector == "server-error":
            self._answer(500, {"error": {"message": "boom", "type": "server_error"}})
            return
        encoding = payload.get("encoding_format", "float")
        data: list[JsonObject] = []
        # Answer in reverse order so the gateway's index-based reordering is exercised.
        for position in reversed(range(len(inputs))):
            vector = _vector(position)
            embedding: object = (
                base64.b64encode(struct.pack(f"<{_VECTOR_WIDTH}f", *vector)).decode()
                if encoding == "base64"
                else vector
            )
            data.append({"object": "embedding", "index": position, "embedding": embedding})
        if selector == "short-count":
            data.pop()
        prompt_tokens = sum(len(text.split()) for text in inputs)
        body: JsonObject = {"object": "list", "data": data, "model": payload["model"]}
        if selector != "unbilled":
            body["usage"] = {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}
        self._answer(200, body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name.
        """Keep the test output quiet."""
        del format, args


@pytest.fixture(scope="module", name="engine")
def _engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ServingEngine]:
    """Serve one shared native engine over a root with chat and embeddings aliases.

    Yields:
        The live serving facts as a :class:`_ServingEngine`.
    """
    root = tmp_path_factory.mktemp("native-embeddings-root")
    with _EmbeddingsUpstream.payloads_lock:
        _EmbeddingsUpstream.payloads.clear()
    upstream = ThreadingHTTPServer((_HOST, 0), _EmbeddingsUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    manager, raw_key = _configured_gateway(
        root, base_url=f"http://{_HOST}:{upstream.server_address[1]}/v1"
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="embedder",
        connection_name="provider-main",
        provider_model="embedder-model-exact",
        exact_model_id="embedder-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(supports_embeddings=True),
        gateway_capabilities=GatewayDeploymentCapabilities(),
        prices=GatewayTokenPrices(input_micro_usd_per_million_tokens=20_000),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="embedder",
        alias_name="embedder",
        revision_id="revision-embedder",
        pool_id="embedder",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="embedder")
    driver = root / "native_embeddings_driver.py"
    driver.write_text(_DRIVER_SOURCE + "\n")
    config = json.dumps({"root": str(root), "request_timeout_seconds": _REQUEST_TIMEOUT_SECONDS})
    stderr_log = root / "driver-stderr.log"
    environment = dict(os.environ)
    environment["TEST_PROVIDER_KEY"] = "provider-secret-canary"
    stderr_sink = stderr_log.open("wb")
    process = subprocess.Popen(  # noqa: S603 - the interpreter runs our generated driver.
        [sys.executable, str(driver), config],
        stdout=subprocess.PIPE,
        stderr=stderr_sink,
        env=environment,
        text=True,
    )
    try:
        announced_ports: list[int] = []

        def _collect_announcements() -> None:
            """Record every port announcement the driver prints on stdout."""
            assert process.stdout is not None
            for line in process.stdout:
                announced_ports.append(int(json.loads(line)["port"]))

        reader = threading.Thread(target=_collect_announcements, daemon=True)
        reader.start()
        live_deadline = time.monotonic() + 30
        port = 0
        while True:
            if announced_ports:
                port = announced_ports[-1]
                try:
                    models = httpx.get(
                        f"http://{_HOST}:{port}/v1/models",
                        headers={"authorization": f"Bearer {raw_key}"},
                        timeout=2.0,
                    )
                    if models.status_code == 200 and sorted(
                        item["id"] for item in models.json()["data"]
                    ) == ["coding", "embedder"]:
                        break
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    pass
            assert process.poll() is None, f"driver died: {stderr_log.read_text()}"
            assert time.monotonic() < live_deadline, "native engine never became live"
            time.sleep(0.05)
        yield _ServingEngine(port=port, raw_key=raw_key, root=root)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        exit_code = process.wait(timeout=20)
        stderr_sink.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
        assert exit_code == 0, f"driver exited {exit_code}: {stderr_log.read_text()}"


def _terminal_attempts(base: str, state: str) -> int:
    """Read one terminal-state attempt total from the live usage report."""
    report = httpx.get(f"{base}/usage.json", timeout=5.0).json()
    for count in report["totals"]["terminal_counts"]:
        if count["state"] == state:
            return int(count["attempts"])
    return 0


def _total(base: str, key: str) -> int:
    """Read one integer total (``requests``, ``input_tokens``, ...) from the live usage report."""
    return int(httpx.get(f"{base}/usage.json", timeout=5.0).json()["totals"][key])


def _last_upstream_payload() -> JsonObject:
    """Return the most recent payload the loopback provider received."""
    with _EmbeddingsUpstream.payloads_lock:
        return _EmbeddingsUpstream.payloads[-1]


def _post(engine: _ServingEngine, body: JsonObject, **headers: str) -> httpx.Response:
    """POST one raw embeddings body with the seeded key."""
    return httpx.post(
        f"{engine.base}/v1/embeddings",
        headers={"authorization": f"Bearer {engine.raw_key}", **headers},
        json=body,
        timeout=30.0,
    )


def test_official_client_round_trips_base64_vectors_and_settles(engine: _ServingEngine) -> None:
    """The OpenAI SDK's default base64 request decodes to the provider's exact floats."""
    completed_before = _terminal_attempts(engine.base, "completed")
    requests_before = _total(engine.base, "requests")
    input_before = _total(engine.base, "input_tokens")
    output_before = _total(engine.base, "output_tokens")
    client = openai.OpenAI(base_url=f"{engine.base}/v1", api_key=engine.raw_key)
    raw = client.embeddings.with_raw_response.create(
        model="embedder", input=["alpha beta", "gamma"]
    )
    response = raw.parse()
    assert raw.headers["x-gateway-alias"] == "embedder"
    assert raw.headers["x-gateway-provider"] == "openai-compatible"
    assert raw.headers["x-gateway-deployment"]
    assert raw.headers["x-gateway-route-depth"] == "0"
    assert raw.headers["x-request-id"]
    assert response.model == "embedder"
    assert response.usage.prompt_tokens == 3
    assert response.usage.total_tokens == 3
    assert [item.index for item in response.data] == [0, 1]
    assert [item.embedding for item in response.data] == [_vector(0), _vector(1)]
    upstream = _last_upstream_payload()
    assert upstream["model"] == "embedder-model-exact"
    assert upstream["input"] == ["alpha beta", "gamma"]
    assert upstream["encoding_format"] == "base64"
    assert _terminal_attempts(engine.base, "completed") == completed_before + 1
    assert _total(engine.base, "requests") == requests_before + 1
    assert _total(engine.base, "input_tokens") == input_before + 3
    assert _total(engine.base, "output_tokens") == output_before


def test_raw_float_request_forwards_every_field_and_ignores_idempotency_key(
    engine: _ServingEngine,
) -> None:
    """A string input with dimensions and float encoding crosses verbatim; user stays gateway-side."""
    response = _post(
        engine,
        {
            "model": "embedder",
            "input": "one two three",
            "dimensions": 3,
            "encoding_format": "float",
            "user": "tenant-7",
        },
        **{"Idempotency-Key": "ignored-on-embeddings", "x-client-request-id": "corr-1"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-client-request-id"] == "corr-1"
    assert response.json() == {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": _vector(0)}],
        "model": "embedder",
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }
    # `user` stays gateway-side (attribution label), never on the provider wire.
    assert _last_upstream_payload() == {
        "model": "embedder-model-exact",
        "input": ["one two three"],
        "dimensions": 3,
        "encoding_format": "float",
    }


def test_chat_alias_is_refused_on_the_model_field_and_accounted(engine: _ServingEngine) -> None:
    """An alias with no embeddings rung fails closed before any provider dispatch."""
    payloads_before = len(_EmbeddingsUpstream.payloads)
    requests_before = _total(engine.base, "requests")
    response = _post(engine, {"model": "coding", "input": "hello"})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unsupported_capability"
    assert error["param"] == "model"
    assert "'coding' does not serve embeddings" in error["message"]
    assert len(_EmbeddingsUpstream.payloads) == payloads_before
    assert _total(engine.base, "requests") == requests_before + 1


def test_protocol_and_key_failures_are_openai_shaped(engine: _ServingEngine) -> None:
    """Decode failures name the field; an unknown key is a uniform 401."""
    missing_input = _post(engine, {"model": "embedder"})
    assert missing_input.status_code == 400
    assert missing_input.json()["error"]["param"] == "input"
    empty_item = _post(engine, {"model": "embedder", "input": ["ok", ""]})
    assert empty_item.status_code == 400
    unknown = httpx.post(
        f"{engine.base}/v1/embeddings",
        headers={"authorization": "Bearer not-a-key"},
        json={"model": "embedder", "input": "hello"},
        timeout=10.0,
    )
    assert unknown.status_code == 401
    assert unknown.json()["error"]["code"] == "invalid_key"


def test_provider_client_error_relays_the_parameter_and_detail(engine: _ServingEngine) -> None:
    """A provider 400 reaches the caller as a 400 naming the rejected field."""
    failed_before = _terminal_attempts(engine.base, "failed")
    response = _post(engine, {"model": "embedder", "input": ["reject-param"]})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "input"
    assert "Maximum length is 2048 items" in error["message"]
    assert _terminal_attempts(engine.base, "failed") == failed_before + 1


@pytest.mark.parametrize(
    ("selector", "expected_fragment"),
    (
        ("short-count", "malformed response"),
        ("unbilled", "malformed response"),
        ("server-error", "provider service failed"),
    ),
)
def test_unbillable_or_failing_provider_answers_fail_closed(
    engine: _ServingEngine, selector: str, expected_fragment: str
) -> None:
    """A vector-count mismatch, a missing prompt_tokens, or a 5xx never reaches the caller."""
    failed_before = _terminal_attempts(engine.base, "failed")
    response = _post(engine, {"model": "embedder", "input": [selector, "second"]})
    assert response.status_code == 502, response.text
    error = response.json()["error"]
    assert error["code"] == "all_routes_failed"
    assert expected_fragment in error["message"]
    assert _terminal_attempts(engine.base, "failed") == failed_before + 1
