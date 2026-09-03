"""End-to-end image-generation tests against the served native engine.

One shared native serving subprocess (the driver from ``native_messages_test``)
serves a seeded root with a chat alias (``coding``) and an image alias
(``painter``) on one OpenAI-compatible loopback connection whose
``/images/generations`` route answers base64 images with token usage. The
tests drive ``POST /v1/images/generations`` through the real Rust data plane
and shared python control plane with the official OpenAI client and raw HTTP.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import openai
import pytest
from openai.types import ImagesResponse

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

_PIXEL = base64.b64encode(b"\x89PNG\r\n\x1a\nfake-pixels").decode()


class _ImagesUpstream(BaseHTTPRequestHandler):
    """OpenAI-compatible ``/images/generations`` mock; the prompt selects the shape."""

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
        """Answer one canned images response selected by the prompt."""
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        with self.payloads_lock:
            self.payloads.append(payload)
        if not self.path.endswith("/images/generations"):
            self._answer(404, {"error": {"message": "unknown route", "type": "invalid_request"}})
            return
        prompt = str(payload["prompt"])
        if prompt == "reject-param":
            self._answer(
                400,
                {
                    "error": {
                        "message": "Invalid value: 'huge'. Supported values are: '1024x1024'.",
                        "type": "invalid_request_error",
                        "param": "size",
                        "code": "invalid_value",
                    }
                },
            )
            return
        if prompt == "server-error":
            self._answer(500, {"error": {"message": "boom", "type": "server_error"}})
            return
        count = int(payload.get("n", 1))
        if prompt == "short-count":
            count -= 1
        data: list[JsonObject] = [
            {"b64_json": _PIXEL, "revised_prompt": f"{prompt} #{index}"} for index in range(count)
        ]
        body: JsonObject = {
            "created": 1_700_000_000,
            "data": data,
            "quality": payload.get("quality", "auto"),
            "size": payload.get("size", "1024x1024"),
        }
        if prompt != "unbilled":
            body["usage"] = {
                "input_tokens": len(prompt.split()),
                "output_tokens": 272 * count,
                "total_tokens": len(prompt.split()) + 272 * count,
                "input_tokens_details": {"text_tokens": len(prompt.split()), "image_tokens": 0},
            }
        self._answer(200, body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name.
        """Keep the test output quiet."""
        del format, args


@pytest.fixture(scope="module", name="engine")
def _engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ServingEngine]:
    """Serve one shared native engine over a root with chat and image aliases."""
    root = tmp_path_factory.mktemp("native-images-root")
    with _ImagesUpstream.payloads_lock:
        _ImagesUpstream.payloads.clear()
    upstream = ThreadingHTTPServer((_HOST, 0), _ImagesUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    manager, raw_key = _configured_gateway(
        root, base_url=f"http://{_HOST}:{upstream.server_address[1]}/v1"
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="painter",
        connection_name="provider-main",
        provider_model="painter-model-exact",
        exact_model_id="painter-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(supports_image_generation=True),
        gateway_capabilities=GatewayDeploymentCapabilities(),
        prices=GatewayTokenPrices(
            input_micro_usd_per_million_tokens=5_000_000,
            output_micro_usd_per_million_tokens=40_000_000,
        ),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="painter",
        alias_name="painter",
        revision_id="revision-painter",
        pool_id="painter",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="painter")
    driver = root / "native_images_driver.py"
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
            assert process.stdout is not None
            for line in process.stdout:
                announced_ports.append(int(json.loads(line)["port"]))

        threading.Thread(target=_collect_announcements, daemon=True).start()
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
                    ) == ["coding", "painter"]:
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


def _total(base: str, key: str) -> int:
    return int(httpx.get(f"{base}/usage.json", timeout=5.0).json()["totals"][key])


def _terminal_attempts(base: str, state: str) -> int:
    report = httpx.get(f"{base}/usage.json", timeout=5.0).json()
    for count in report["totals"]["terminal_counts"]:
        if count["state"] == state:
            return int(count["attempts"])
    return 0


def _post(engine: _ServingEngine, body: JsonObject, **headers: str) -> httpx.Response:
    return httpx.post(
        f"{engine.base}/v1/images/generations",
        headers={"authorization": f"Bearer {engine.raw_key}", **headers},
        json=body,
        timeout=30.0,
    )


def test_official_client_generates_images_and_settles_both_token_legs(
    engine: _ServingEngine,
) -> None:
    """The OpenAI SDK's generate call returns the provider's images and bills its usage."""
    completed_before = _terminal_attempts(engine.base, "completed")
    input_before = _total(engine.base, "input_tokens")
    output_before = _total(engine.base, "output_tokens")
    client = openai.OpenAI(base_url=f"{engine.base}/v1", api_key=engine.raw_key)
    raw = client.images.with_raw_response.generate(
        model="painter", prompt="a calico cat", n=2, size="1024x1024", quality="low"
    )
    response = raw.parse()
    assert isinstance(response, ImagesResponse)
    assert raw.headers["x-gateway-alias"] == "painter"
    assert raw.headers["x-gateway-provider"] == "openai-compatible"
    assert response.data is not None and len(response.data) == 2
    assert response.data[0].b64_json == _PIXEL
    assert response.data[1].revised_prompt == "a calico cat #1"
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens) == (3, 544)
    with _ImagesUpstream.payloads_lock:
        upstream = _ImagesUpstream.payloads[-1]
    assert upstream == {
        "model": "painter-model-exact",
        "prompt": "a calico cat",
        "n": 2,
        "size": "1024x1024",
        "quality": "low",
    }
    assert _terminal_attempts(engine.base, "completed") == completed_before + 1
    assert _total(engine.base, "input_tokens") == input_before + 3
    assert _total(engine.base, "output_tokens") == output_before + 544


def test_raw_request_defaults_n_and_ignores_idempotency_key(engine: _ServingEngine) -> None:
    """A bare prompt is one image; the caller's Idempotency-Key never keys this surface."""
    response = _post(
        engine,
        {"model": "painter", "prompt": "one cat", "user": "tenant-7"},
        **{"Idempotency-Key": "ignored", "x-client-request-id": "corr-1"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-client-request-id"] == "corr-1"
    body = response.json()
    assert len(body["data"]) == 1
    assert body["usage"]["output_tokens"] == 272
    with _ImagesUpstream.payloads_lock:
        upstream = _ImagesUpstream.payloads[-1]
    assert upstream == {"model": "painter-model-exact", "prompt": "one cat", "n": 1}


def test_chat_alias_and_streaming_are_refused_with_field_errors(engine: _ServingEngine) -> None:
    """A chat alias fails on model; a streaming request fails on stream; both are OpenAI-shaped."""
    chat = _post(engine, {"model": "coding", "prompt": "a cat"})
    assert chat.status_code == 400
    assert chat.json()["error"]["param"] == "model"
    assert "does not generate images" in chat.json()["error"]["message"]
    streaming = _post(engine, {"model": "painter", "prompt": "a cat", "stream": True})
    assert streaming.status_code == 400
    assert streaming.json()["error"]["param"] == "stream"
    unknown = httpx.post(
        f"{engine.base}/v1/images/generations",
        headers={"authorization": "Bearer not-a-key"},
        json={"model": "painter", "prompt": "a cat"},
        timeout=10.0,
    )
    assert unknown.status_code == 401


def test_provider_client_error_relays_the_parameter(engine: _ServingEngine) -> None:
    """A provider 400 reaches the caller as a 400 naming the rejected field."""
    response = _post(engine, {"model": "painter", "prompt": "reject-param"})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "size"
    assert "Supported values are" in error["message"]


@pytest.mark.parametrize(
    ("prompt", "expected_fragment"),
    [
        ("unbilled", "malformed response"),
        ("short-count", "malformed response"),
        ("server-error", "provider service failed"),
    ],
)
def test_unbillable_or_failing_provider_answers_fail_closed(
    engine: _ServingEngine, prompt: str, expected_fragment: str
) -> None:
    """No usage, a missing image, or a 5xx never hands the caller an unaccounted image."""
    failed_before = _terminal_attempts(engine.base, "failed")
    response = _post(engine, {"model": "painter", "prompt": prompt, "n": 2})
    assert response.status_code == 502, response.text
    assert response.json()["error"]["code"] == "all_routes_failed"
    assert expected_fragment in response.json()["error"]["message"]
    assert _terminal_attempts(engine.base, "failed") == failed_before + 1
