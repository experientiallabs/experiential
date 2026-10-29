"""End-to-end Azure image-generation tests against the served native engine.

One shared native serving subprocess serves four image aliases on one Azure
connection (``provider="azure"``, dated API version) whose loopback upstream
answers the three Azure image wires: the OpenAI Image API on the deployments
path (token-billed ``gpt-image``), the MAI image API, and the BFL FLUX.2
provider API (both per-image priced, answering ``data[0].b64_json`` without
usage). The tests drive ``POST /v1/images/generations`` through the real Rust
data plane and read the upstream request and the ledger back.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    BilledUnitKind,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    GatewayUnitPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.catalog_authority import upsert_singleton_deployment
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests.native_messages_test import (
    _DRIVER_SOURCE,
    _HOST,
    _REQUEST_TIMEOUT_SECONDS,
    _ServingEngine,
)

pytest.importorskip("exp_gateway_native")

_PIXEL = base64.b64encode(b"\x89PNG\r\n\x1a\nazure-pixels").decode()
_MAI_RATE = 40_000_000  # ~$0.04 per MAI-Image-2.6 image.
_FLUX_RATE = 30_000_000
_ALIASES = ["azure-gpt-image", "coding", "flux2", "mai", "mai-uncarded"]


class _AzureUpstream(BaseHTTPRequestHandler):
    """Loopback Azure resource answering the Image API, MAI, and BFL routes."""

    requests: list[tuple[str, dict[str, str], JsonObject]] = []
    lock = threading.Lock()

    def _answer(self, status: int, body: JsonObject) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Record the request and answer the route's canned shape."""
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        headers = {key.lower(): value for key, value in self.headers.items()}
        with self.lock:
            self.requests.append((self.path, headers, payload))
        if self.path.startswith("/openai/deployments/") and "/images/generations" in self.path:
            if headers.get("api-key") != "provider-secret-canary":
                self._answer(401, {"error": {"message": "bad key"}})
                return
            count = int(payload.get("n", 1))
            self._answer(
                200,
                {
                    "created": 1_700_000_000,
                    "data": [{"b64_json": _PIXEL} for _ in range(count)],
                    "usage": {"input_tokens": 4, "output_tokens": 272 * count},
                },
            )
            return
        if self.path == "/mai/v1/images/generations":
            if headers.get("api-key") != "provider-secret-canary":
                self._answer(401, {"error": {"message": "bad key"}})
                return
            self._answer(200, {"data": [{"b64_json": _PIXEL}]})
            return
        if self.path.startswith("/providers/blackforestlabs/v1/"):
            if headers.get("authorization") != "Bearer provider-secret-canary":
                self._answer(401, {"error": {"message": "bad bearer"}})
                return
            self._answer(200, {"data": [{"b64_json": _PIXEL}]})
            return
        self._answer(404, {"error": {"message": f"unknown route {self.path}"}})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name.
        """Keep the test output quiet."""
        del format, args


def _add_alias(
    root: Path,
    manager: GatewayManagement,
    *,
    alias: str,
    provider_model: str,
    exact_model_id: str,
    prices: GatewayTokenPrices,
) -> None:
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias=alias,
        connection_name="provider-main",
        provider_model=provider_model,
        exact_model_id=exact_model_id,
        revision=None,
        capabilities=ModelCapabilities(supports_image_generation=True),
        gateway_capabilities=GatewayDeploymentCapabilities(),
        prices=prices,
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id=alias,
        alias_name=alias,
        revision_id=f"revision-{alias}",
        pool_id=alias,
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id=alias)


def _per_image(rate: int) -> GatewayTokenPrices:
    return GatewayTokenPrices(units=GatewayUnitPrices(kind=BilledUnitKind.IMAGE, rates={"": rate}))


@pytest.fixture(scope="module", name="engine")
def _engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ServingEngine]:
    """Serve one native engine over an Azure connection with four image aliases."""
    root = tmp_path_factory.mktemp("native-azure-images-root")
    with _AzureUpstream.lock:
        _AzureUpstream.requests.clear()
    upstream = ThreadingHTTPServer((_HOST, 0), _AzureUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    manager, raw_key = _configured_gateway(
        root,
        base_url=f"http://{_HOST}:{upstream.server_address[1]}",
        provider="azure",
        api_version="2025-04-01-preview",
    )
    _add_alias(
        root,
        manager,
        alias="azure-gpt-image",
        provider_model="gpt-image-deploy",
        exact_model_id="gpt-image-1.5",
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=5_000_000,
            output_nano_usd_per_million_tokens=40_000_000,
        ),
    )
    _add_alias(
        root,
        manager,
        alias="mai",
        provider_model="my-mai-deployment",
        exact_model_id="mai-image-2.6",
        prices=_per_image(_MAI_RATE),
    )
    _add_alias(
        root,
        manager,
        alias="mai-uncarded",
        provider_model="mai-uncarded-deployment",
        exact_model_id="mai-image-2.5",
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=5_000_000,
            output_nano_usd_per_million_tokens=33_000_000,
        ),
    )
    _add_alias(
        root,
        manager,
        alias="flux2",
        provider_model="flux2-pro-deployment",
        exact_model_id="flux.2-pro",
        prices=_per_image(_FLUX_RATE),
    )
    driver = root / "native_azure_images_driver.py"
    driver.write_text(_DRIVER_SOURCE + "\n")
    config = json.dumps({"root": str(root), "request_timeout_seconds": _REQUEST_TIMEOUT_SECONDS})
    stderr_log = root / "driver-stderr.log"
    environment = dict(os.environ)
    environment["TEST_PROVIDER_KEY"] = "provider-secret-canary"
    environment.pop("AZURE_OPENAI_ENDPOINT", None)
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
                    if (
                        models.status_code == 200
                        and sorted(item["id"] for item in models.json()["data"]) == _ALIASES
                    ):
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


def _post(engine: _ServingEngine, body: JsonObject) -> httpx.Response:
    return httpx.post(
        f"{engine.base}/v1/images/generations",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=body,
        timeout=30.0,
    )


def _last_upstream() -> tuple[str, dict[str, str], JsonObject]:
    with _AzureUpstream.lock:
        return _AzureUpstream.requests[-1]


def _upstream_count() -> int:
    with _AzureUpstream.lock:
        return len(_AzureUpstream.requests)


def _last_attempt(engine: _ServingEngine) -> tuple[object, ...]:
    with sqlite3.connect(engine.root / "gateway" / "gateway.db") as connection:
        row = connection.execute(
            """
            SELECT state, estimated_cost_nano_usd, billed_unit_kind, billed_quantity_milli,
                   output_tokens
            FROM gateway_attempts ORDER BY rowid DESC LIMIT 1
            """
        ).fetchone()
    return tuple(row)


def test_gpt_image_uses_the_deployments_image_api_and_bills_tokens(
    engine: _ServingEngine,
) -> None:
    response = _post(engine, {"model": "azure-gpt-image", "prompt": "a red fox", "n": 2})
    assert response.status_code == 200, response.text
    assert [item["b64_json"] for item in response.json()["data"]] == [_PIXEL, _PIXEL]
    path, headers, payload = _last_upstream()
    assert path == (
        "/openai/deployments/gpt-image-deploy/images/generations?api-version=2025-04-01-preview"
    )
    assert headers["api-key"] == "provider-secret-canary"
    assert payload == {"model": "gpt-image-deploy", "prompt": "a red fox", "n": 2}
    state, cost, kind, _quantity, output_tokens = _last_attempt(engine)
    assert (state, kind, output_tokens) == ("completed", None, 544)
    assert cost == (4 * 5_000_000 + 544 * 40_000_000 + 999_999) // 1_000_000


def test_mai_dispatches_width_and_height_and_bills_one_image(engine: _ServingEngine) -> None:
    response = _post(engine, {"model": "mai", "prompt": "a red fox", "size": "1536x1024"})
    assert response.status_code == 200, response.text
    assert response.json()["data"] == [{"b64_json": _PIXEL}]
    path, headers, payload = _last_upstream()
    assert path == "/mai/v1/images/generations"
    assert headers["api-key"] == "provider-secret-canary"
    assert "authorization" not in headers
    assert payload == {
        "model": "my-mai-deployment",
        "prompt": "a red fox",
        "width": 1536,
        "height": 1024,
    }
    assert _last_attempt(engine) == ("completed", _MAI_RATE, "image", 1_000, None)


def test_flux2_dispatches_the_bfl_provider_route_with_a_bearer_key(
    engine: _ServingEngine,
) -> None:
    response = _post(
        engine,
        {"model": "flux2", "prompt": "a red fox", "size": "1024x1024", "output_format": "jpeg"},
    )
    assert response.status_code == 200, response.text
    path, headers, payload = _last_upstream()
    assert path == "/providers/blackforestlabs/v1/flux-2-pro?api-version=preview"
    assert headers["authorization"] == "Bearer provider-secret-canary"
    assert "api-key" not in headers
    assert payload == {
        "model": "flux2-pro-deployment",
        "prompt": "a red fox",
        "n": 1,
        "width": 1024,
        "height": 1024,
        "output_format": "jpeg",
    }
    assert _last_attempt(engine) == ("completed", _FLUX_RATE, "image", 1_000, None)


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"model": "mai", "prompt": "x", "n": 2}, "n"),
        ({"model": "mai", "prompt": "x", "size": "512x512"}, "size"),
        ({"model": "flux2", "prompt": "x", "quality": "hd"}, "quality"),
    ],
)
def test_unrepresentable_requests_are_refused_before_dispatch(
    engine: _ServingEngine, body: JsonObject, param: str
) -> None:
    before = _upstream_count()
    response = _post(engine, body)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == param
    assert _upstream_count() == before


def test_a_usage_free_wire_without_a_per_image_card_is_refused(engine: _ServingEngine) -> None:
    before = _upstream_count()
    response = _post(engine, {"model": "mai-uncarded", "prompt": "a red fox"})
    assert response.status_code == 400, response.text
    assert "per-image price" in response.json()["error"]["message"]
    assert _upstream_count() == before
